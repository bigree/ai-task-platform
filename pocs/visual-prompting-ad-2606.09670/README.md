# Visual Prompting + Dual-Teacher Anomaly Detection

PyTorch PoC for **arXiv:2606.09670**  
"Visual Prompting Meets Feature Reconstruction-Based Anomaly Detection with Dual-Teacher Supervision"  
IBM Research Europe, Jun 2026

---

## What this implements

| Paper contribution | Status |
|---|---|
| Dual-Teacher (Strong frozen + Weak unfrozen + Student) | ✅ |
| Dual loss: `L = L(S, WT) + λ·L(WT, ST)` (cosine similarity) | ✅ |
| VP mask post-processing (FG/BG isolation + mixing) | ✅ (GrabCut proxy) |
| Synthetic training data via diffusion | ⬜ (out of scope for PoC) |

SAM can replace GrabCut in `grabcut_fg_mask()` for production use.

---

## Architecture

```
Input Image
    │
    ├─→ Strong Teacher (WideResNet-50, frozen) ──────────────┐
    │                                                         │  λ · L_WT_ST
    ├─→ Weak Teacher   (WideResNet-50, unfrozen) ────────────┤
    │        │                                                │
    │        └─→ Student (Conv heads) ── L_S_WT ────────────┘
    │
    └─→ [Inference] Weak Teacher + Student → cosine distance → scoremap
                                                    │
                                              VP Mask (GrabCut)
                                                    │
                                            masked scoremap
```

---

## Quickstart

```bash
pip install torch torchvision opencv-python scikit-learn

# Smoke test (no data needed)
python poc.py --mode smoke

# Train on MVTec
python poc.py --mode train \
  --data_root /path/to/mvtec \
  --category bottle \
  --epochs 100 \
  --lam 1.5

# Evaluate → output/scoremaps/
python poc.py --mode eval \
  --data_root /path/to/mvtec \
  --category bottle \
  --ckpt output/model_last.pth
```

### Colab

```python
!git clone https://github.com/bigree/ai-task-platform.git
%cd ai-task-platform/pocs/visual-prompting-ad-2606.09670
!pip install torch torchvision opencv-python scikit-learn
!python poc.py --mode smoke
```

---

## Key hyperparameters

| Param | Default | Paper recommendation |
|---|---|---|
| `--lam` | 1.5 | 1.5 (AeBAD-S) / 0.5 (MVTec) |
| `--mask_dilation` | 15 | 40px (AeBAD-S) / 15px (MVTec) |
| `--epochs` | 100 | — |

---

## Results (paper, MMR+++ on AeBAD-S)

| Metric | Baseline MMR | MMR+++ (paper) |
|---|---|---|
| I-AUROC | 84.7 | **88.2** (+3.5%) |
| AUPRO | 88.7 | **90.2** (+1.5%) |

---

## Dataset layout (MVTec-style)

```
data/mvtec/
└── bottle/
    ├── train/good/*.png
    └── test/
        ├── good/*.png
        └── broken_large/*.png
```

---

## Files

```
poc.py           # Full implementation (single file)
requirements.txt # Dependencies
```
