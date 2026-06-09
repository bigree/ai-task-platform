"""
cranio_diff/dataset.py

S2F (Skull-to-Face) dataset loader.
Paper Section 3: 120 subjects × frontal/lateral × age(25,45,65) × BMI(±10%) = 4320 samples.

Expected directory structure:
    data/
      s2f/
        train/
          subject_001/
            skull_frontal.jpg
            skull_lateral.jpg
            face_frontal_age25_bmi-10.jpg
            face_frontal_age25_bmi0.jpg
            face_frontal_age25_bmi+10.jpg
            face_frontal_age45_bmi-10.jpg
            ...
        test/
          subject_109/ ...
"""

from __future__ import annotations
import json
from pathlib import Path
from itertools import product

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .model import build_biometric_prompt

# ── Constants ────────────────────────────────────────────────────────────────

AGES       = [25, 45, 65]
BMI_DELTAS = [-10, 0, 10]
VIEWS      = ["frontal", "lateral"]

# Data augmentation from Section 5.1
_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.15, contrast=0.20),
    transforms.RandomRotation(degrees=5),
    transforms.RandomErasing(p=0.5, scale=(0.0, 0.05)),
])


def _to_tensor_norm(img: Image.Image, size: int = 512) -> torch.Tensor:
    """PIL → (3, size, size) tensor in [-1, 1]."""
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(img)


# ── Dataset ───────────────────────────────────────────────────────────────────

class S2FDataset(Dataset):
    """
    Skull-to-Face dataset.

    Each sample:
        skull_image  : (3, 512, 512)  X-ray skull image
        face_image   : (3, 512, 512)  ground truth face
        prompt       : str            biometric text prompt
        meta         : dict           {subject, age, bmi_delta, view, gender}
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = True,
        image_size: int = 512,
    ):
        self.root = Path(root) / split
        self.augment = augment and (split == "train")
        self.image_size = image_size
        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        """
        Scan data directory and build a list of (skull, face, meta) triplets.

        Naming convention assumed:
            skull_{view}.jpg
            face_{view}_age{age}_bmi{delta:+d}.jpg
            meta.json  (optional, for gender info)
        """
        samples = []
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue

            # Load optional metadata
            meta_file = subject_dir / "meta.json"
            gender = "male"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    gender = meta.get("gender", "male")

            for view, age, bmi_delta in product(VIEWS, AGES, BMI_DELTAS):
                skull_path = subject_dir / f"skull_{view}.jpg"
                face_path  = subject_dir / f"face_{view}_age{age}_bmi{bmi_delta:+d}.jpg"

                if not skull_path.exists() or not face_path.exists():
                    continue

                samples.append({
                    "skull_path": skull_path,
                    "face_path":  face_path,
                    "subject":    subject_dir.name,
                    "view":       view,
                    "age":        age,
                    "bmi_delta":  bmi_delta,
                    "gender":     gender,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        skull_img = Image.open(s["skull_path"]).convert("RGB")
        face_img  = Image.open(s["face_path"]).convert("RGB")

        skull = _to_tensor_norm(skull_img, self.image_size)
        face  = _to_tensor_norm(face_img,  self.image_size)

        if self.augment:
            # Apply same spatial transform to both (consistency)
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            skull = _AUGMENT(skull)
            torch.manual_seed(seed)
            face  = _AUGMENT(face)

        prompt = build_biometric_prompt(
            age=s["age"],
            gender=s["gender"],
            orientation=s["view"],
            bmi_delta=s["bmi_delta"],
        )

        return {
            "skull":   skull,
            "face":    face,
            "prompt":  prompt,
            "subject": s["subject"],
            "meta":    {k: s[k] for k in ("age", "bmi_delta", "view", "gender")},
        }


# ── Synthetic demo dataset (no real data needed for PoC testing) ──────────────

class SyntheticS2FDataset(Dataset):
    """
    Generates random noise tensors to verify the training loop runs
    without requiring real skull/face data.

    Usage:
        dataset = SyntheticS2FDataset(n_samples=32)
    """

    def __init__(self, n_samples: int = 64, image_size: int = 512):
        self.n = n_samples
        self.size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        skull = torch.randn(3, self.size, self.size)
        face  = torch.randn(3, self.size, self.size)
        age   = AGES[idx % len(AGES)]
        bmi   = BMI_DELTAS[idx % len(BMI_DELTAS)]
        view  = VIEWS[idx % len(VIEWS)]
        prompt = build_biometric_prompt(age=age, bmi_delta=bmi, orientation=view)
        return {
            "skull":   skull,
            "face":    face,
            "prompt":  prompt,
            "subject": f"synthetic_{idx:04d}",
            "meta":    {"age": age, "bmi_delta": bmi, "view": view, "gender": "male"},
        }
