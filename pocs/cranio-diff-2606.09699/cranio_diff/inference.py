"""
cranio_diff/inference.py

Inference script: skull X-ray → reconstructed face image.

Usage:
    python -m cranio_diff.inference \
        --skull path/to/skull.jpg \
        --age 45 --gender male --view frontal --bmi 0 \
        --controlnet-path ./checkpoints/epoch_0350/controlnet \
        --unet-path ./checkpoints/epoch_0350/unet \
        --output result.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import CranioDiff, build_biometric_prompt


def load_skull_image(path: str, size: int = 512) -> torch.Tensor:
    """Load and preprocess skull X-ray image."""
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(img).unsqueeze(0)  # (1, 3, H, W)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (1, 3, H, W) tensor in [-1,1] → PIL Image."""
    img = (tensor.squeeze(0).float().clamp(-1, 1) + 1.0) / 2.0
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))


def run_inference(
    skull_path: str,
    age: int = 45,
    gender: str = "male",
    view: str = "frontal",
    bmi_delta: int = 0,
    controlnet_path: str | None = None,
    unet_path: str | None = None,
    output_path: str = "result.png",
    device: str = "cuda",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """
    Full inference pipeline.

    Args:
        skull_path:       Path to skull X-ray image.
        age:              25 / 45 / 65
        gender:           "male" / "female"
        view:             "frontal" / "lateral"
        bmi_delta:        -10 / 0 / +10
        controlnet_path:  Path to fine-tuned ControlNet weights.
        unet_path:        Path to fine-tuned UNet weights.
        output_path:      Where to save the generated face.
        device:           "cuda" or "cpu"
        num_steps:        DDPM inference steps (paper default: 50).
        guidance_scale:   CFG scale (paper default: 7.5).
        controlnet_scale: ControlNet conditioning strength.
        seed:             Random seed for reproducibility.

    Returns:
        PIL Image of the reconstructed face.
    """
    # 1. Build biometric text prompt
    prompt = build_biometric_prompt(
        age=age, gender=gender, orientation=view, bmi_delta=bmi_delta
    )
    print(f"Prompt: {prompt}")

    # 2. Load model
    print("Loading model...")
    model = CranioDiff(device=device, dtype=torch.float16)

    if controlnet_path:
        from diffusers import ControlNetModel
        model.controlnet = ControlNetModel.from_pretrained(controlnet_path).to(device)
    if unet_path:
        from diffusers import UNet2DConditionModel
        model.unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)

    model.eval()

    # 3. Load skull image
    skull_tensor = load_skull_image(skull_path).to(device)

    # 4. Generate face
    print(f"Generating face: {num_steps} steps, guidance={guidance_scale}...")
    result_pil = model.generate(
        skull_image=skull_tensor,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        seed=seed,
    )

    # 5. Save
    if isinstance(result_pil, torch.Tensor):
        result_pil = tensor_to_pil(result_pil)

    result_pil.save(output_path)
    print(f"Saved → {output_path}")
    return result_pil


# ── Multi-variation generation (age × BMI grid) ───────────────────────────────

def generate_variation_grid(
    skull_path: str,
    gender: str = "male",
    view: str = "frontal",
    output_dir: str = "./results",
    **kwargs,
) -> None:
    """
    Generate all 9 age × BMI variations for a given skull (Figure 5 in paper).
    """
    ages       = [25, 45, 65]
    bmi_deltas = [-10, 0, 10]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for age in ages:
        for bmi in bmi_deltas:
            fname = f"{gender}_{view}_age{age}_bmi{bmi:+d}.png"
            run_inference(
                skull_path=skull_path,
                age=age, gender=gender, view=view, bmi_delta=bmi,
                output_path=str(out / fname),
                **kwargs,
            )
            print(f"Generated: {fname}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cranio-Diff Inference")
    parser.add_argument("--skull",            required=True, help="Path to skull X-ray image")
    parser.add_argument("--age",              type=int, default=45, choices=[25, 45, 65])
    parser.add_argument("--gender",           default="male", choices=["male", "female"])
    parser.add_argument("--view",             default="frontal", choices=["frontal", "lateral"])
    parser.add_argument("--bmi",              type=int, default=0, choices=[-10, 0, 10])
    parser.add_argument("--controlnet-path",  default=None)
    parser.add_argument("--unet-path",        default=None)
    parser.add_argument("--output",           default="result.png")
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--steps",            type=int, default=50)
    parser.add_argument("--guidance-scale",   type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--grid",             action="store_true",
                        help="Generate all age×BMI variations (Figure 5)")
    args = parser.parse_args()

    if args.grid:
        generate_variation_grid(
            skull_path=args.skull,
            gender=args.gender,
            view=args.view,
            output_dir=str(Path(args.output).parent),
            device=args.device,
            controlnet_path=args.controlnet_path,
            unet_path=args.unet_path,
        )
    else:
        run_inference(
            skull_path=args.skull,
            age=args.age,
            gender=args.gender,
            view=args.view,
            bmi_delta=args.bmi,
            controlnet_path=args.controlnet_path,
            unet_path=args.unet_path,
            output_path=args.output,
            device=args.device,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed,
        )
