# PoC: Cranio-Diff

**Unofficial PyTorch implementation of:**
> Cranio-Diff: Diffusion-based Cross-domain Craniofacial Reconstruction with 2D X-ray Skull Guidance and Structural Identity Constraints  
> Prasad et al., arXiv:2606.09699 (2026)

---

## What this does

Reconstructs a photorealistic face image from a 2D skull X-ray, conditioned on biometric attributes (age / gender / BMI).

```
Skull X-ray + "45-year-old male, frontal, average build"
        ↓
  Cranio-Diff (ControlNet + Stable Diffusion v1.5)
        ↓
  Reconstructed face image
```

## Architecture

```
Frozen VAE encoder  →  z_0  →  forward diffusion  →  z_t
                                                         ↓
Skull X-ray  →  ControlNet (trainable)  →  C_s  →  Denoising UNet (trainable)
                                                         ↑
Biometric text  →  Frozen CLIP encoder  →  τ  ──────────┘
                                                         ↓
                                              VAE decoder  →  x̂_0
```

**Loss (Eq. 11):**
```
L = L_diff + 0.20 × L_LPIPS + 0.20 × L_id
```

## Quickstart

```bash
pip install -r requirements.txt
```

### Synthetic test (no GPU / data needed)
```bash
python -m cranio_diff.train --synthetic --epochs 2 --batch-size 2 --device cpu
```

### Training with real data
```bash
python -m cranio_diff.train \
    --data-root ./data/s2f \
    --epochs 350 \
    --batch-size 14 \
    --lr 1e-4 \
    --device cuda
```

### Inference
```bash
python -m cranio_diff.inference \
    --skull ./data/test_skull.jpg \
    --age 45 --gender male --view frontal --bmi 0 \
    --controlnet-path ./checkpoints/epoch_0350/controlnet \
    --output result.png
```

### Generate all age × BMI variations (Figure 5)
```bash
python -m cranio_diff.inference \
    --skull ./data/test_skull.jpg \
    --gender male --view frontal \
    --grid --output ./results/
```

## Key hyperparameters (Section 5.1)

| Parameter       | Value  |
|----------------|--------|
| Image size      | 512×512 |
| Epochs          | 350    |
| Batch size      | 14     |
| Learning rate   | 1e-4   |
| Weight decay    | 1e-5   |
| λ_LPIPS         | 0.20   |
| λ_id (ArcFace)  | 0.20   |
| Optimizer       | AdamW  |

## File structure

```
cranio_diff/
├── __init__.py
├── model.py      # CranioDiff model + biometric prompt builder
├── losses.py     # LPIPS + ArcFace identity loss + combined CranioDiffLoss
├── dataset.py    # S2FDataset + SyntheticS2FDataset
├── train.py      # Training loop
└── inference.py  # Inference pipeline + variation grid generator
```

## Notes

- This is a **PoC implementation** based on the paper description. The original S2F dataset (120 subjects, Indian individuals) is **not publicly available**.
- Backbone: `runwayml/stable-diffusion-v1-5` (paper uses fine-tuned `Realistic_Vision_V5.1` — swap `MODEL_ID` in `model.py` for closer reproduction).
- Training requires ~40 GB VRAM for batch=14 at 512×512 (paper used H200 141GB). Reduce batch size for smaller GPUs.

## Citation

```bibtex
@article{prasad2026craniodiff,
  title   = {Cranio-Diff: Diffusion-based Cross-domain Craniofacial Reconstruction
             with 2D X-ray Skull Guidance and Structural Identity Constraints},
  author  = {Prasad, Ravi Shankar and Gurjar, Naresh and Baghel, Shashank
             and Chirag and Singh, Dinesh},
  journal = {arXiv preprint arXiv:2606.09699},
  year    = {2026}
}
```

---
*This PoC was generated as part of the [AI Task Platform](https://github.com/bigree/ai-task-platform) project.*
