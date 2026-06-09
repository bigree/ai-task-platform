"""
test_pipeline.py
SDモデルをロードせず、データセット・損失・プロンプトだけを検証する軽量テスト。
数秒で完了する。
"""
import torch
from cranio_diff.dataset import SyntheticS2FDataset
from cranio_diff.losses import LPIPSLoss, ArcFaceIdentityLoss, CranioDiffLoss
from cranio_diff.model import build_biometric_prompt

print("=== 1. Biometric prompt builder ===")
for age in [25, 45, 65]:
    print(" ", build_biometric_prompt(age=age, gender="male", orientation="frontal", bmi_delta=0))

print("\n=== 2. SyntheticDataset ===")
ds = SyntheticS2FDataset(n_samples=4, image_size=64)  # 64px で軽量化
batch = ds[0]
print(f"  skull: {batch['skull'].shape}, face: {batch['face'].shape}")
print(f"  prompt: {batch['prompt']}")

print("\n=== 3. LPIPS loss (VGG fallback) ===")
lpips = LPIPSLoss(device="cpu")
gen  = torch.randn(2, 3, 64, 64)
real = torch.randn(2, 3, 64, 64)
loss_val = lpips(gen, real)
print(f"  LPIPS loss: {loss_val.item():.4f}")

print("\n=== 4. Identity loss (pixel fallback) ===")
id_loss = ArcFaceIdentityLoss(backbone="facenet", device="cpu")
loss_id = id_loss(gen, real)
print(f"  Identity loss: {loss_id.item():.4f}")

print("\n=== 5. CranioDiffLoss (combined) ===")
criterion = CranioDiffLoss(device="cpu")
l_diff = torch.tensor(0.05)
total, breakdown = criterion(l_diff, gen, real)
print(f"  {breakdown}")

print("\n✅ All components OK. SD model loading skipped (use GPU for full training).")
