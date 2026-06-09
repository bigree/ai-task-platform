"""
cranio_diff/losses.py

Loss functions for Cranio-Diff (Section 4.5).

Total objective (Eq. 11):
    L = L_diff + λ1·L_LPIPS + λ2·L_id
    λ1 = λ2 = 0.20  (empirically chosen in paper)

Components:
    L_LPIPS  : perceptual loss (Eq. 9)
    L_id     : ArcFace identity cosine loss (Eq. 10)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LPIPS Perceptual Loss (Eq. 9) ─────────────────────────────────────────────

class LPIPSLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    L_LPIPS = Σ_l ||φ_l(x̂_0) - φ_l(x_0)||²

    Uses the `lpips` library (pip install lpips).
    Falls back to a simple VGG-based implementation if not available.
    """

    def __init__(self, net: str = "vgg", device: str = "cuda"):
        super().__init__()
        self.device = device
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            self.lpips_fn.requires_grad_(False)
            self._use_lpips = True
        except ImportError:
            print("Warning: lpips not installed. Using VGG feature loss fallback.")
            self._use_lpips = False
            self.vgg = _VGGFeatureExtractor().to(device)
            self.vgg.requires_grad_(False)

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) in [-1, 1]
            real:      (B, 3, H, W) in [-1, 1]
        Returns:
            Scalar perceptual loss.
        """
        generated = generated.to(self.device)
        real = real.to(self.device)

        if self._use_lpips:
            return self.lpips_fn(generated, real).mean()
        else:
            gen_feats = self.vgg(generated)
            real_feats = self.vgg(real)
            return sum(F.mse_loss(g, r) for g, r in zip(gen_feats, real_feats))


class _VGGFeatureExtractor(nn.Module):
    """Lightweight VGG feature extractor for perceptual loss fallback."""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        layers = list(vgg.features.children())
        self.slice1 = nn.Sequential(*layers[:4])    # relu1_2
        self.slice2 = nn.Sequential(*layers[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*layers[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # [-1,1] → [0,1] → VGG normalize
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]


# ── ArcFace Identity Loss (Eq. 10) ────────────────────────────────────────────

class ArcFaceIdentityLoss(nn.Module):
    """
    Identity-preserving cosine similarity loss (Eq. 10):

        L_id = 1 - (z_gen^T · z_gt) / (||z_gen|| · ||z_gt||)

    where z_gen = F_arc(x̂_0), z_gt = F_arc(x_0)
    and F_arc is a pretrained ArcFace face recognition model.

    Uses insightface or facenet-pytorch as the backbone.
    """

    def __init__(self, backbone: str = "facenet", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.backbone_name = backbone

        if backbone == "facenet":
            self._init_facenet()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'facenet'.")

    def _init_facenet(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.face_encoder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.face_encoder.requires_grad_(False)
            self._available = True
        except ImportError:
            print("Warning: facenet-pytorch not installed. Identity loss will use cosine on raw pixels.")
            self._available = False

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode face images to identity embeddings."""
        if self._available:
            # facenet expects (B, 3, 160, 160) normalized to [-1, 1]
            imgs = F.interpolate(images, size=(160, 160), mode="bilinear", align_corners=False)
            return self.face_encoder(imgs.to(self.device))
        else:
            # Fallback: global average pool as crude embedding
            return F.adaptive_avg_pool2d(images, (1, 1)).flatten(1)

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) in [-1, 1]
            real:      (B, 3, H, W) in [-1, 1]
        Returns:
            Scalar identity loss ∈ [0, 2].
        """
        z_gen = self._encode(generated)
        z_gt  = self._encode(real)
        cosine_sim = F.cosine_similarity(z_gen, z_gt, dim=1)
        return (1.0 - cosine_sim).mean()


# ── Combined Loss (Eq. 11) ────────────────────────────────────────────────────

class CranioDiffLoss(nn.Module):
    """
    Total Cranio-Diff training objective (Eq. 11):

        L = L_diff + λ1·L_LPIPS + λ2·L_id
        λ1 = λ2 = 0.20

    Usage:
        criterion = CranioDiffLoss(device=device)
        loss, breakdown = criterion(l_diff, generated_faces, real_faces)
    """

    def __init__(
        self,
        lambda_lpips: float = 0.20,
        lambda_id: float = 0.20,
        device: str = "cuda",
    ):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.lambda_id = lambda_id
        self.lpips = LPIPSLoss(device=device)
        self.identity = ArcFaceIdentityLoss(device=device)

    def forward(
        self,
        l_diff: torch.Tensor,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            l_diff:    diffusion MSE loss (from model.forward)
            generated: decoded face images (B, 3, H, W) in [-1, 1]
            real:      ground truth faces  (B, 3, H, W) in [-1, 1]

        Returns:
            (total_loss, breakdown_dict)
        """
        l_lpips = self.lpips(generated, real)
        l_id    = self.identity(generated, real)

        total = l_diff + self.lambda_lpips * l_lpips + self.lambda_id * l_id

        breakdown = {
            "loss_diff":  l_diff.item(),
            "loss_lpips": l_lpips.item(),
            "loss_id":    l_id.item(),
            "loss_total": total.item(),
        }
        return total, breakdown
