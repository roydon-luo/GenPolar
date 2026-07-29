"""Dataset loader for angle images or precomputed RGB Stokes components."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ANGLE_FILES = ("0.png", "45.png", "90.png", "135.png")
STOKES_FILES = ("s0.png", "s1.png", "s2.png")


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def aligned_crop(arrays: list[np.ndarray], size: int) -> list[np.ndarray]:
    height, width = arrays[0].shape[:2]
    if any(array.shape[:2] != (height, width) for array in arrays):
        raise ValueError("All images in a scene must have the same resolution")
    if min(height, width) < size:
        raise ValueError(
            f"Scene resolution {width}x{height} is smaller than crop size {size}"
        )
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return [array[top : top + size, left : left + size] for array in arrays]


def legal_stokes_scale(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = (
        0.5 * (s0 + s1),
        0.5 * (s0 + s2),
        0.5 * (s0 - s1),
        0.5 * (s0 - s2),
    )
    maximum = max(float(np.nanmax(angle)) for angle in angles)
    if np.isfinite(maximum) and maximum > 1.0:
        return s0 / maximum, s1 / maximum, s2 / maximum
    return s0, s1, s2


def to_chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()


class CPDMImageDataset(Dataset):
    """Return paper-normalized `polarization`, `rgb`, and CLIP `input_ids`."""

    def __init__(self, options, tokenizer):
        self.crop_size = int(options.get("gt_size", 512))
        self.stokes_scale = float(options.get("stokes_scale", 1.0))
        self.tokenizer = tokenizer
        self.samples: list[tuple[str, Path]] = []

        sources = options.get("data_source", {})
        for source in sources.values():
            root = Path(source["dataroot_gt"]).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"Dataset source does not exist: {root}")
            self.samples.extend(self._scan(root))
        if not self.samples:
            raise RuntimeError("No valid polarization scenes found")

        self.input_ids = tokenizer(
            "denoised polarized images",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    @staticmethod
    def _scan(root: Path) -> list[tuple[str, Path]]:
        samples: list[tuple[str, Path]] = []
        candidates = [root] + sorted(path for path in root.iterdir() if path.is_dir())
        for scene in candidates:
            if all((scene / name).is_file() for name in ANGLE_FILES):
                samples.append(("angles", scene))
            elif all((scene / name).is_file() for name in STOKES_FILES):
                samples.append(("stokes", scene))
        samples.extend(("npy", path) for path in sorted(root.glob("*.npy")))
        return samples

    def _load(self, kind: str, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if kind == "angles":
            i0, i45, i90, i135 = aligned_crop(
                [read_rgb(path / name) for name in ANGLE_FILES], self.crop_size
            )
            return (
                (i0 + i45 + i90 + i135) / 2.0,
                i0 - i90,
                i45 - i135,
            )

        if kind == "stokes":
            s0, s1, s2 = aligned_crop(
                [read_rgb(path / name) for name in STOKES_FILES], self.crop_size
            )
            return 2.0 * s0, 2.0 * s1 - 1.0, 2.0 * s2 - 1.0

        array = np.load(path, mmap_mode="r")
        if array.ndim == 4 and array.shape[2] >= 3:
            s0, s1, s2 = (np.asarray(array[:, :, index, :3]) for index in range(3))
        elif array.ndim == 3 and array.shape[-1] >= 9:
            s0, s1, s2 = (
                np.asarray(array[:, :, start : start + 3]) for start in (0, 3, 6)
            )
        else:
            raise ValueError(f"Unsupported Stokes array shape {array.shape}: {path}")
        s0, s1, s2 = aligned_crop(
            [item.astype(np.float32) for item in (s0, s1, s2)], self.crop_size
        )
        return legal_stokes_scale(s0, s1, s2)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s0, s1, s2 = self._load(*self.samples[index])
        target = torch.cat((to_chw(s1), to_chw(s2))).clamp(-1.0, 1.0)
        condition = to_chw(s0).clamp(0.0, 2.0) - 1.0
        return {
            "polarization": target * self.stokes_scale,
            "rgb": condition,
            "input_ids": self.input_ids.clone(),
        }

    def __len__(self) -> int:
        return len(self.samples)
