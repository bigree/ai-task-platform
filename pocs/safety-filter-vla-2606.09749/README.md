# PoC: Attention-Guided Safety Filter for VLA

> **Paper:** Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models  
> arXiv:[2606.09749](https://arxiv.org/abs/2606.09749v1) (2026)

**Current status: Synthetic PoC — θ未キャリブレーション / 実画像未検証**  
See [open issues](#open-issues) for next steps.

---

## What this does

VLAモデルのaction tokenのAttentionを解析し、task-irrelevantな物体への注視を検出して行動をブロックするSafety Filter。追加学習不要でフロントアタッチできる。

```
Language instruction + Camera image
        ↓
  VLA Model (frozen)
        ↓
  Action token attention extraction  ← per-layer hooks
        ↓
  Collision risk score = non-task attention mass
        ↓
  score < θ → SAFE (execute)
  score ≥ θ → UNSAFE (halt + replan)
```

**外科AI応用:** 術野内の非ターゲット組織（血管・神経）への接触回避に直接転用可能。

---

## Quickstart

### Synthetic mode（GPU不要・即実行）

```bash
pip install -r requirements.txt

# CLI
python visualize_attention.py --synthetic

# Colab CLI
colab new -s vla
colab exec -s vla -f openvla_attention_visualizer.ipynb
colab stop -s vla
```

### Real mode（OpenVLA 7B / T4 GPU以上）

```bash
# requirements.txtのtorch/transformers行をアンコメント
pip install -r requirements.txt

python visualize_attention.py \
  --image path/to/scene.jpg \
  --instruction "pick up the red cube" \
  --model openvla/openvla-7b \
  --device cuda \
  --theta 0.45
```

---

## Results (synthetic, 2026-06-13)

| Metric | Value |
|---|---|
| Mode | synthetic (16×16 grid, 32 layers) |
| Task mass | 0.650 |
| Non-task mass | 0.350 |
| Risk score | 0.350 |
| Verdict (θ=0.5) | ✓ SAFE |
| Unsafe layers | 0 / 32 |

**θ感度:** θ=0.35以下に下げると大半の層がUNSAFE判定 → 実データでのキャリブレーションが必要。

---

## Architecture

```
Input image → SigLIP encoder → image tokens (256 or 729)
Language instruction → Llama-2-7B tokenizer

[BOS] [img_0..img_N] [instruction tokens] [action tokens]
                              ↓
              LlamaAttention forward hooks (all 32 layers)
                              ↓
         action_token rows × image_patch columns → attn_grid (16×16)
                              ↓
              compute_risk(): task_bbox内外でattention質量を分割
                              ↓
              risk_score = non_task_mass / total_mass
```

- VLA backbone: OpenVLA-7B (SigLIP + Llama-2-7B) — frozen
- Hook点: `model.language_model.model.layers[i].self_attn`
- Action tokens: シーケンス末尾7トークン

---

## Open Issues

- [ ] **θキャリブレーション** — 実ロボットデータ or 腹腔鏡フレームで最適θを決定
- [ ] **実画像トライアル** — `MODE='real'` でOpenVLA本体を使った検証
- [ ] **task_bbox自動推定** — Grounding DINOまたはSAM2と連携してBBoxを自動取得
- [ ] **シミュレータ評価** — PyBullet/IsaacGymで衝突回避率を定量評価
- [ ] **π0対応** — OpenVLA以外のVLAバックボーンへの移植確認

---

## Files

| File | Description |
|---|---|
| `visualize_attention.py` | CLIスクリプト（synthetic / real両対応） |
| `openvla_attention_visualizer.ipynb` | Colabノートブック（colab-cli対応済み） |
| `requirements.txt` | 依存パッケージ |

---

## Citation

```bibtex
@article{arxiv260609749,
  title  = {Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models},
  year   = {2026},
  note   = {arXiv preprint arXiv:2606.09749}
}
```

---
*PRJ-017 AI Task Platform — Priority 1/3*
