"""
PoC: Visual Prompting + Dual-Teacher Anomaly Detection
Paper: arXiv:2606.09670 - "Visual Prompting Meets Feature Reconstruction-Based AD"
PRJ-017 / 2026-06-09

Key contributions implemented:
  1. Dual-Teacher Student architecture (Strong frozen teacher + Weak unfrozen teacher + Student)
     Loss: L_total = L(Student, WeakTeacher) + λ * L(WeakTeacher, StrongTeacher)
  2. VP Mask post-processing  (GrabCut as SAM proxy; drop-in replaceable with SAM)
  3. Anomaly scoremap from feature cosine distance

Backbone: WideResNet-50 pretrained on ImageNet (same spirit as paper's MMR encoder)
Dataset:  MVTec-style folder layout, or runs a smoke test with random tensors

Usage:
  # Quick smoke test (no data needed):
  python poc.py --mode smoke

  # Train on MVTec bottle category:
  python poc.py --mode train --data_root /path/to/mvtec --category bottle

  # Evaluate (outputs scoremaps in ./output/):
  python poc.py --mode eval --data_root /path/to/mvtec --category bottle --ckpt ./output/model_last.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights

# ─────────────────────────────────────────────
# 1.  Encoder  (shared backbone logic)
# ─────────────────────────────────────────────

class MultiScaleEncoder(nn.Module):
    """WideResNet-50 that returns intermediate feature maps at 3 scales."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.wide_resnet50_2(weights=weights)
        # Layers we tap into (matching MMR-style multiscale)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride-4  → 64×64 for 256-input
        self.layer2 = base.layer2   # stride-8  → 32×32
        self.layer3 = base.layer3   # stride-16 → 16×16

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]          # list of feature maps


# ─────────────────────────────────────────────
# 2.  Student  (lightweight decoder head)
# ─────────────────────────────────────────────

class StudentDecoder(nn.Module):
    """
    Mirrors the teacher's channel dims at each scale with Conv+BN+ReLU blocks.
    Accepts the concatenation of weak-teacher features and tries to reconstruct them.
    """

    def __init__(self, channels=(256, 512, 1024)):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 1),
            )
            for c in channels
        ])

    def forward(self, teacher_features):
        return [h(f) for h, f in zip(self.heads, teacher_features)]


# ─────────────────────────────────────────────
# 3.  Dual-Teacher Model
# ─────────────────────────────────────────────

class DualTeacherAD(nn.Module):
    """
    Strong Teacher  – frozen, provides regularisation signal to Weak Teacher
    Weak Teacher    – unfrozen, adapts to target domain
    Student         – learns to reconstruct Weak Teacher's features

    L_total = L(Student, WeakTeacher) + λ * L(WeakTeacher, StrongTeacher)
    """

    def __init__(self, lam: float = 1.5):
        super().__init__()
        self.lam = lam

        # Both teachers start from the same pretrained weights
        self.strong_teacher = MultiScaleEncoder(pretrained=True)
        self.weak_teacher   = MultiScaleEncoder(pretrained=True)
        self.student        = StudentDecoder()

        # Strong teacher is always frozen
        for p in self.strong_teacher.parameters():
            p.requires_grad = False

    # ---- Loss helpers -------------------------------------------------------

    @staticmethod
    def cosine_loss(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
        """Mean (1 – cosine_similarity) across all scales and spatial positions."""
        loss = 0.0
        for fa, fb in zip(a, b):
            # fa, fb: (B, C, H, W)  → flatten spatial → (B*H*W, C)
            B, C, H, W = fa.shape
            fa_flat = fa.permute(0, 2, 3, 1).reshape(-1, C)
            fb_flat = fb.permute(0, 2, 3, 1).reshape(-1, C)
            loss = loss + (1.0 - F.cosine_similarity(fa_flat, fb_flat, dim=1)).mean()
        return loss / len(a)

    # ---- Forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            st_feats = self.strong_teacher(x)     # always no-grad

        wt_feats = self.weak_teacher(x)            # grad flows here
        s_feats  = self.student(wt_feats)          # grad flows here

        loss_s_wt  = self.cosine_loss(s_feats, wt_feats)
        loss_wt_st = self.cosine_loss(wt_feats, st_feats)
        loss       = loss_s_wt + self.lam * loss_wt_st

        return loss, loss_s_wt.item(), loss_wt_st.item()

    # ---- Inference ----------------------------------------------------------

    @torch.no_grad()
    def anomaly_score_map(self, x: torch.Tensor, out_size: tuple = (256, 256)):
        """
        Returns pixel-level anomaly scoremap (B, 1, H, W) using the
        cosine distance between student and weak-teacher features.
        Strong teacher is disabled at inference (as per the paper).
        """
        wt_feats = self.weak_teacher(x)
        s_feats  = self.student(wt_feats)

        maps = []
        for s, t in zip(s_feats, wt_feats):
            B, C, H, W = s.shape
            dist = 1.0 - F.cosine_similarity(s, t, dim=1, eps=1e-8)  # (B,H,W)
            dist = dist.unsqueeze(1)                                   # (B,1,H,W)
            dist = F.interpolate(dist, size=out_size, mode="bilinear", align_corners=False)
            maps.append(dist)

        scoremap = torch.stack(maps, dim=0).mean(dim=0)    # (B,1,H,W)
        return scoremap


# ─────────────────────────────────────────────
# 4.  Visual Prompting Mask (GrabCut proxy)
# ─────────────────────────────────────────────

def grabcut_fg_mask(img_np: np.ndarray, dilation: int = 15) -> np.ndarray:
    import cv2
    """
    Lightweight FG/BG mask via GrabCut (SAM drop-in proxy for the PoC).
    img_np: uint8 H×W×3 (BGR)
    Returns: binary mask H×W, 1 = foreground
    """
    h, w = img_np.shape[:2]
    # Simple center-crop rect as initial foreground hint
    rect = (w // 8, h // 8, w * 6 // 8, h * 6 // 8)
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_np, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        fg_mask = cv2.dilate(fg_mask, kernel)

    return fg_mask


def apply_vp_mask(scoremap: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Mixing strategy from paper: dilate FG scoremap and clip to peak of raw scoremap.
    scoremap: H×W float32
    fg_mask:  H×W uint8 {0,1}
    """
    # Zero out background
    masked = scoremap * fg_mask.astype(np.float32)
    # Clip to raw peak (retains sharp discrimination)
    peak = scoremap.max()
    masked = np.clip(masked, 0, peak)
    return masked


# ─────────────────────────────────────────────
# 5.  Dataset  (MVTec-style)
# ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    """
    Minimal MVTec loader.
    Requires: opencv-python
    train split: data_root/<category>/train/good/*.png
    test split:  data_root/<category>/test/**/*.png
                 masks:       data_root/<category>/ground_truth/**/*.png
    """

    def __init__(self, root: str, category: str, split: str = "train", img_size: int = 256):
        self.split    = split
        self.img_size = img_size
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        base = Path(root) / category
        if split == "train":
            self.img_paths  = sorted((base / "train" / "good").glob("*.png"))
            self.img_paths += sorted((base / "train" / "good").glob("*.jpg"))
            self.labels     = [0] * len(self.img_paths)
        else:
            self.img_paths = []
            self.labels    = []
            test_base = base / "test"
            for cls_dir in sorted(test_base.iterdir()):
                label = 0 if cls_dir.name == "good" else 1
                for p in sorted(cls_dir.glob("*.png")):
                    self.img_paths.append(p)
                    self.labels.append(label)
                for p in sorted(cls_dir.glob("*.jpg")):
                    self.img_paths.append(p)
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(str(self.img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.tf(img)
        label  = self.labels[idx]
        return tensor, label, str(self.img_paths[idx])


# ─────────────────────────────────────────────
# 6.  Training loop
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = MVTecDataset(args.data_root, args.category, split="train", img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = DualTeacherAD(lam=args.lam).to(device)
    # Only weak_teacher + student are trainable
    opt = torch.optim.Adam(
        list(model.weak_teacher.parameters()) + list(model.student.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, _, _ in dl:
            imgs = imgs.to(device)
            opt.zero_grad()
            loss, ls_wt, lwt_st = model(imgs)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(dl)
        print(f"Epoch [{epoch:03d}/{args.epochs}]  loss={avg:.4f}  "
              f"(L_S_WT={ls_wt:.4f}, L_WT_ST={lwt_st:.4f})")

    ckpt_path = out_dir / "model_last.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


# ─────────────────────────────────────────────
# 7.  Evaluation loop
# ─────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualTeacherAD(lam=args.lam).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = MVTecDataset(args.data_root, args.category, split="test", img_size=args.img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    out_dir = Path(args.output_dir) / "scoremaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores, all_labels = [], []

    for imgs, labels, paths in dl:
        imgs = imgs.to(device)
        scoremap = model.anomaly_score_map(imgs, out_size=(args.img_size, args.img_size))
        scoremap_np = scoremap[0, 0].cpu().numpy()

        # VP mask post-processing
        orig_bgr = cv2.imread(paths[0])
        orig_bgr = cv2.resize(orig_bgr, (args.img_size, args.img_size))
        fg_mask  = grabcut_fg_mask(orig_bgr, dilation=args.mask_dilation)
        scoremap_masked = apply_vp_mask(scoremap_np, fg_mask)

        img_score = float(scoremap_masked.max())
        all_scores.append(img_score)
        all_labels.append(int(labels[0]))

        # Save heatmap overlay
        import cv2
        fname = Path(paths[0]).stem
        heat = (scoremap_masked / (scoremap_masked.max() + 1e-8) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_bgr, 0.5, heat_color, 0.5, 0)
        cv2.imwrite(str(out_dir / f"{fname}_overlay.png"), overlay)

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(all_labels, all_scores)
        print(f"\nImage-level AUROC: {auroc:.4f}")
    except ImportError:
        print("Install scikit-learn for AUROC. Raw scores saved.")

    scores_path = Path(args.output_dir) / "scores.npy"
    np.save(str(scores_path), {"scores": all_scores, "labels": all_labels})
    print(f"Scoremaps saved to {out_dir}")


# ─────────────────────────────────────────────
# 8.  Smoke test (no real data needed)
# ─────────────────────────────────────────────

def smoke_test():
    print("=== Smoke Test: Dual-Teacher AD ===")
    device = torch.device("cpu")
    model  = DualTeacherAD(lam=1.5).to(device)

    B = 2
    x = torch.randn(B, 3, 256, 256)

    # -- Training forward
    model.train()
    loss, ls_wt, lwt_st = model(x)
    print(f"  Train forward OK  loss={loss.item():.4f}  "
          f"L_S_WT={ls_wt:.4f}  L_WT_ST={lwt_st:.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # -- Backward
    loss.backward()
    grad_wt = sum(p.grad.abs().sum().item()
                  for p in model.weak_teacher.parameters() if p.grad is not None)
    grad_st = sum(p.grad.abs().sum().item()
                  for p in model.strong_teacher.parameters() if p.grad is not None)
    print(f"  Gradients  weak_teacher={grad_wt:.2f}  strong_teacher={grad_st:.2f}")
    assert grad_wt > 0,  "Weak teacher must have gradients"
    assert grad_st == 0, "Strong teacher must be frozen (no gradients)"

    # -- Inference scoremap
    model.eval()
    scoremap = model.anomaly_score_map(x, out_size=(256, 256))
    print(f"  Scoremap shape: {tuple(scoremap.shape)}  "
          f"range=[{scoremap.min():.4f}, {scoremap.max():.4f}]")
    assert scoremap.shape == (B, 1, 256, 256)

    # -- VP mask (requires opencv-python; skipped if not installed)
    try:
        import cv2  # noqa: F401
        dummy_bgr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        fg_mask   = grabcut_fg_mask(dummy_bgr, dilation=15)
        masked    = apply_vp_mask(scoremap[0, 0].numpy(), fg_mask)
        print(f"  VP mask FG ratio: {fg_mask.mean():.2f}  "
              f"masked scoremap range=[{masked.min():.4f}, {masked.max():.4f}]")
    except ImportError:
        print("  VP mask: SKIPPED (pip install opencv-python to enable)")

    print("\n=== All checks passed ✓ ===")


# ─────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dual-Teacher Visual Prompting AD PoC")
    p.add_argument("--mode",       choices=["smoke", "train", "eval"], default="smoke")
    p.add_argument("--data_root",  default="./data/mvtec")
    p.add_argument("--category",   default="bottle")
    p.add_argument("--ckpt",       default="./output/model_last.pth")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lam",        type=float, default=1.5,
                   help="λ: regularisation strength of strong teacher (paper: 1.5 for AeBAD-S)")
    p.add_argument("--mask_dilation", type=int, default=15,
                   help="VP mask dilation (paper: 40px AeBAD-S, 15px MVTec)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
