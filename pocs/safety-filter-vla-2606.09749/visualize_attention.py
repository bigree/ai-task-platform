"""
visualize_attention.py
OpenVLA Attention Map Visualizer — PoC for Safety Filter (arXiv:2606.09749)

Usage:
    # Synthetic mode (no GPU / no model weights needed)
    python visualize_attention.py --synthetic

    # Real mode (requires ~14GB VRAM + model download)
    python visualize_attention.py \
        --image path/to/scene.jpg \
        --instruction "pick up the red cube" \
        --model openvla/openvla-7b \
        --device cuda

Output:
    out/attention_layer_XX.png   — per-layer attention heatmap overlaid on image
    out/attention_aggregate.png  — mean across all layers
    out/attention_report.json    — numerical summary (collision risk score per bbox)

Architecture note:
    OpenVLA = SigLIP vision encoder + Llama-2-7B LLM (PrismaticVLM)
    Image tokens: 256 patches (16x16 grid, 224px input) or 729 patches (27x27, 384px)
    Layout: [BOS] [img_0..img_N] [sys_prompt] [instruction] [action_tokens]
    We hook LlamaAttention.forward() to capture attention rows for action tokens.
    Action token → image patch attention = "where the model looks when deciding actions".
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────

def resize_and_normalize(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    return np.array(img)


def overlay_heatmap(
    base_img: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "hot",
) -> np.ndarray:
    """Overlay attention heatmap on base image."""
    h, w = base_img.shape[:2]
    attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    heatmap = cm.get_cmap(colormap)(attn_norm)[..., :3]  # drop alpha
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = np.array(Image.fromarray(heatmap_u8).resize((w, h)))
    blended = (base_img * (1 - alpha) + heatmap_resized * alpha).astype(np.uint8)
    return blended


def attention_to_grid(attn_weights: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Reshape flat patch attention to 2D grid."""
    assert attn_weights.shape[0] == grid_h * grid_w, (
        f"Expected {grid_h * grid_w} patches, got {attn_weights.shape[0]}"
    )
    return attn_weights.reshape(grid_h, grid_w)


def compute_collision_risk(
    attn_grid: np.ndarray,
    task_bbox: tuple | None,
    image_size: int,
    patch_size: int,
) -> dict:
    """
    Compute attention mass inside vs outside task bounding box.

    Args:
        attn_grid:  (grid_h, grid_w) attention values
        task_bbox:  (x0, y0, x1, y1) in pixel coords. None = use center 50% as task region.
        image_size: input image size in pixels
        patch_size: size of each vision patch in pixels

    Returns:
        dict with task_mass, non_task_mass, risk_score in [0, 1]
    """
    grid_h, grid_w = attn_grid.shape

    if task_bbox is None:
        # Default: treat center region as task-relevant
        cx, cy = grid_w // 2, grid_h // 2
        r = max(1, min(grid_h, grid_w) // 4)
        mask = np.zeros_like(attn_grid, dtype=bool)
        mask[cy - r : cy + r, cx - r : cx + r] = True
    else:
        x0, y0, x1, y1 = task_bbox
        scale_x = grid_w / image_size
        scale_y = grid_h / image_size
        gx0 = int(x0 * scale_x)
        gy0 = int(y0 * scale_y)
        gx1 = int(x1 * scale_x)
        gy1 = int(y1 * scale_y)
        mask = np.zeros_like(attn_grid, dtype=bool)
        mask[gy0:gy1, gx0:gx1] = True

    total = attn_grid.sum() + 1e-8
    task_mass = attn_grid[mask].sum() / total
    non_task_mass = attn_grid[~mask].sum() / total
    risk_score = float(non_task_mass)  # high = model attending to irrelevant regions

    return {
        "task_mass": float(task_mass),
        "non_task_mass": float(non_task_mass),
        "risk_score": risk_score,
        "safe": risk_score < 0.5,
    }


# ──────────────────────────────────────────────
#  Synthetic mode (no model required)
# ──────────────────────────────────────────────

def generate_synthetic_data(
    n_layers: int = 32,
    n_heads: int = 32,
    grid_h: int = 16,
    grid_w: int = 16,
    n_action_tokens: int = 7,
    seed: int = 42,
) -> dict:
    """
    Generate realistic-looking synthetic attention data.
    Simulates: most attention on a task object (center-left),
    with moderate spillover to task-irrelevant regions.
    """
    rng = np.random.default_rng(seed)
    n_patches = grid_h * grid_w

    layers = {}
    for layer_idx in range(n_layers):
        # Create spatially structured attention (Gaussian peaks)
        x = np.linspace(0, 1, grid_w)
        y = np.linspace(0, 1, grid_h)
        xx, yy = np.meshgrid(x, y)

        # Primary task object: center-left
        cx1, cy1 = 0.35, 0.50
        sigma1 = 0.12 + 0.05 * rng.random()
        attn_task = np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * sigma1 ** 2))

        # Task-irrelevant distractor: upper-right (e.g. background object)
        cx2, cy2 = 0.75, 0.25
        sigma2 = 0.08 + 0.04 * rng.random()
        distractor_weight = 0.15 + 0.20 * (layer_idx / n_layers)  # grows in later layers
        attn_distractor = np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * sigma2 ** 2))

        # Combine + noise
        attn_base = attn_task + distractor_weight * attn_distractor
        noise = rng.exponential(0.02, size=(grid_h, grid_w))
        attn_combined = attn_base + noise

        # Per-head variation
        head_attns = []
        for _ in range(n_heads):
            jitter = rng.normal(0, 0.03, size=(grid_h, grid_w))
            h_attn = np.clip(attn_combined + jitter, 0, None)
            h_attn /= h_attn.sum() + 1e-8
            head_attns.append(h_attn)

        # Mean-head attention for action tokens
        mean_attn = np.mean(head_attns, axis=0)
        layers[layer_idx] = {
            "mean_head": mean_attn,
            "per_head": np.stack(head_attns),
        }

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "layers": layers,
    }


def make_synthetic_image(size: int = 224) -> np.ndarray:
    """Create a simple synthetic scene image (robot arm + two objects)."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # grey background

    # Table surface
    img[size // 2 :, :] = [150, 120, 90]

    # Task object: red cube (center-left)
    cx, cy = int(size * 0.35), int(size * 0.55)
    s = size // 10
    img[cy - s : cy + s, cx - s : cx + s] = [220, 50, 50]

    # Distractor: blue cylinder (upper-right)
    dx, dy = int(size * 0.72), int(size * 0.30)
    r = size // 14
    Y, X = np.ogrid[:size, :size]
    mask = (X - dx) ** 2 + (Y - dy) ** 2 <= r ** 2
    img[mask] = [50, 100, 220]

    # Robot arm (simplified: dark grey bar from top)
    img[: int(size * 0.45), int(size * 0.30) : int(size * 0.42)] = [60, 60, 60]

    return img


# ──────────────────────────────────────────────
#  Real OpenVLA mode
# ──────────────────────────────────────────────

class AttentionExtractor:
    """
    Hooks into LlamaAttention layers to capture attention weights.
    Works with HuggingFace transformers LlamaModel.
    """

    def __init__(self, model, n_image_tokens: int):
        self.n_image_tokens = n_image_tokens
        self.hooks = []
        self.captured: dict[int, np.ndarray] = {}
        self._register_hooks(model)

    def _register_hooks(self, model):
        import torch

        for layer_idx, layer in enumerate(model.language_model.model.layers):
            # LlamaDecoderLayer → self_attn
            self_attn = layer.self_attn

            def make_hook(idx):
                def hook(module, inputs, output):
                    # output: (hidden_states, attn_weights, past_kv)
                    # attn_weights: (batch, n_heads, seq_len, seq_len) if output_attentions=True
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_w = output[1]  # (B, H, T, T)
                        with torch.no_grad():
                            # Action token rows: last few tokens
                            # Image token columns: positions 1 to n_image_tokens+1
                            action_rows = attn_w[0, :, -7:, :]  # (H, 7, T)
                            img_cols = action_rows[
                                :, :, 1 : self.n_image_tokens + 1
                            ]  # (H, 7, N_img)
                            # Mean over action tokens and aggregate
                            mean_over_actions = img_cols.mean(dim=1)  # (H, N_img)
                            mean_over_heads = mean_over_actions.mean(dim=0)  # (N_img,)
                            self.captured[idx] = mean_over_heads.cpu().float().numpy()

                return hook

            h = self_attn.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_real_inference(args) -> dict:
    """Load OpenVLA and run forward pass with attention output."""
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        print("[ERROR] torch / transformers not installed.")
        print("Run: pip install torch transformers accelerate")
        sys.exit(1)

    print(f"Loading model: {args.model} …")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    # Determine image token count from model config
    vision_cfg = model.config.vision_config
    patch_size = getattr(vision_cfg, "patch_size", 14)
    image_size = getattr(vision_cfg, "image_size", 224)
    grid = image_size // patch_size
    n_image_tokens = grid * grid
    print(f"Vision grid: {grid}x{grid} = {n_image_tokens} image tokens")

    image = Image.open(args.image).convert("RGB")
    inputs = processor(args.instruction, image).to(args.device, dtype=torch.bfloat16)

    extractor = AttentionExtractor(model, n_image_tokens)

    with torch.no_grad():
        _ = model(
            **inputs,
            output_attentions=True,
        )

    extractor.remove_hooks()

    layers = {}
    for layer_idx, flat_attn in extractor.captured.items():
        grid_attn = flat_attn.reshape(grid, grid)
        layers[layer_idx] = {
            "mean_head": grid_attn,
            "per_head": None,
        }

    return {
        "n_layers": len(layers),
        "grid_h": grid,
        "grid_w": grid,
        "layers": layers,
        "image": np.array(image),
    }


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def visualize(
    data: dict,
    base_img: np.ndarray,
    task_bbox: tuple | None,
    theta: float,
    out_dir: Path,
):
    n_layers = data["n_layers"]
    grid_h = data["grid_h"]
    grid_w = data["grid_w"]
    layers = data["layers"]

    print(f"Visualizing {n_layers} layers …")

    all_risks = []
    per_layer_results = []

    # ── Per-layer plots ──
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig_all, axes_all = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    axes_all = axes_all.flatten()

    for layer_idx in range(n_layers):
        attn_grid = layers[layer_idx]["mean_head"]

        risk = compute_collision_risk(
            attn_grid,
            task_bbox,
            image_size=base_img.shape[0],
            patch_size=base_img.shape[0] // grid_h,
        )
        all_risks.append(risk["risk_score"])
        per_layer_results.append({"layer": layer_idx, **risk})

        blended = overlay_heatmap(base_img, attn_grid)
        axes_all[layer_idx].imshow(blended)
        color = "red" if not risk["safe"] else "green"
        axes_all[layer_idx].set_title(
            f"L{layer_idx:02d}  risk={risk['risk_score']:.2f}",
            fontsize=7,
            color=color,
        )
        axes_all[layer_idx].axis("off")

        # Save individual layer image
        img_out = out_dir / f"attention_layer_{layer_idx:02d}.png"
        fig_single, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(blended)
        ax.set_title(
            f"Layer {layer_idx} | risk={risk['risk_score']:.3f} | "
            + ("⚠ UNSAFE" if not risk["safe"] else "✓ SAFE"),
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")
        fig_single.tight_layout()
        fig_single.savefig(img_out, dpi=120, bbox_inches="tight")
        plt.close(fig_single)

    # Hide unused subplots
    for i in range(n_layers, len(axes_all)):
        axes_all[i].axis("off")

    fig_all.suptitle("OpenVLA Action Token → Image Patch Attention (all layers)", fontsize=11)
    fig_all.tight_layout()
    fig_all.savefig(out_dir / "attention_all_layers.png", dpi=100, bbox_inches="tight")
    plt.close(fig_all)

    # ── Aggregate (mean across layers) ──
    agg_attn = np.mean(
        [layers[i]["mean_head"] for i in range(n_layers)], axis=0
    )
    agg_risk = compute_collision_risk(
        agg_attn,
        task_bbox,
        image_size=base_img.shape[0],
        patch_size=base_img.shape[0] // grid_h,
    )

    fig_agg, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Original image
    axes[0].imshow(base_img)
    axes[0].set_title("Original Scene", fontsize=11)
    axes[0].axis("off")

    # Aggregate attention overlay
    blended_agg = overlay_heatmap(base_img, agg_attn)
    axes[1].imshow(blended_agg)
    axes[1].set_title("Aggregate Attention (mean layers)", fontsize=11)
    axes[1].axis("off")

    # Risk score over layers
    ax = axes[2]
    layer_ids = list(range(n_layers))
    colors = ["red" if r >= theta else "green" for r in all_risks]
    ax.bar(layer_ids, all_risks, color=colors, width=0.8)
    ax.axhline(theta, color="orange", linestyle="--", linewidth=1.5, label=f"θ={theta}")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Collision Risk Score", fontsize=10)
    ax.set_title("Risk Score per Layer", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    aggregate_label = (
        f"Aggregate risk: {agg_risk['risk_score']:.3f}  →  "
        + ("⚠ UNSAFE — action should be BLOCKED" if not agg_risk["safe"] else "✓ SAFE — action OK")
    )
    fig_agg.suptitle(aggregate_label, fontsize=11, fontweight="bold",
                     color="red" if not agg_risk["safe"] else "darkgreen")
    fig_agg.tight_layout()
    fig_agg.savefig(out_dir / "attention_aggregate.png", dpi=130, bbox_inches="tight")
    plt.close(fig_agg)
    print(f"Saved: {out_dir}/attention_aggregate.png")

    # ── Risk Report ──
    report = {
        "aggregate": agg_risk,
        "theta": theta,
        "verdict": "UNSAFE" if not agg_risk["safe"] else "SAFE",
        "n_unsafe_layers": sum(1 for r in all_risks if r >= theta),
        "n_layers": n_layers,
        "per_layer": per_layer_results,
    }
    report_path = out_dir / "attention_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    return report


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def parse_bbox(s: str | None):
    if s is None:
        return None
    parts = [float(x) for x in s.split(",")]
    assert len(parts) == 4, "bbox format: x0,y0,x1,y1"
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA Attention Visualizer (Safety Filter PoC)"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no model/GPU needed)")
    parser.add_argument("--model", default="openvla/openvla-7b",
                        help="HuggingFace model ID")
    parser.add_argument("--image", default=None,
                        help="Path to input image (real mode)")
    parser.add_argument("--instruction", default="pick up the red cube",
                        help="Language instruction for the robot")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda / cpu / mps")
    parser.add_argument("--theta", type=float, default=0.50,
                        help="Collision risk threshold (default 0.50)")
    parser.add_argument("--task-bbox", default=None,
                        help="Task object bbox in pixels: x0,y0,x1,y1")
    parser.add_argument("--out-dir", default="out",
                        help="Output directory")
    parser.add_argument("--n-layers", type=int, default=32,
                        help="[Synthetic] Number of layers to simulate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    task_bbox = parse_bbox(args.task_bbox)

    if args.synthetic:
        print("=== Synthetic mode ===")
        print(f"Instruction: '{args.instruction}'")
        data = generate_synthetic_data(n_layers=args.n_layers)
        base_img = make_synthetic_image(size=224)
        # Save synthetic scene for reference
        Image.fromarray(base_img).save(out_dir / "synthetic_scene.png")
        print(f"Scene saved: {out_dir}/synthetic_scene.png")
    else:
        if args.image is None:
            print("[ERROR] --image required in real mode (or use --synthetic)")
            sys.exit(1)
        data = run_real_inference(args)
        base_img = data.pop("image", None)
        if base_img is None:
            base_img = make_synthetic_image()

    report = visualize(data, base_img, task_bbox, args.theta, out_dir)

    # ── Console summary ──
    print("\n" + "=" * 50)
    print(f"VERDICT : {report['verdict']}")
    print(f"Risk    : {report['aggregate']['risk_score']:.3f}  (θ={args.theta})")
    print(f"Task mass   : {report['aggregate']['task_mass']:.3f}")
    print(f"Non-task mass: {report['aggregate']['non_task_mass']:.3f}")
    print(f"Unsafe layers: {report['n_unsafe_layers']} / {report['n_layers']}")
    print("=" * 50)
    print(f"\nOutputs saved to: {out_dir}/")
    print("  attention_aggregate.png  — main result")
    print("  attention_all_layers.png — grid of all layers")
    print("  attention_layer_XX.png   — per-layer images")
    print("  attention_report.json    — numerical report")


if __name__ == "__main__":
    main()
