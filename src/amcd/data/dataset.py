"""PyTorch Dataset over pre-normalized energy tensors from the preprocess stage."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EnergyDataset(Dataset):
    """
    Loads pre-saved normalized (low, high) energy tensor pairs for one split.
    Tensors are stored as (C, n_bands, n_frames) float32.
    Normalization was applied during preprocess using train-split stats only.
    """

    def __init__(self, preprocessed_dir: Path, split: str) -> None:
        split_dir = preprocessed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Exclude macOS resource-fork files (._*) that appear on external volumes
        low_paths = sorted(p for p in split_dir.glob("*_low.pt") if not p.name.startswith("._"))
        high_paths = sorted(p for p in split_dir.glob("*_high.pt") if not p.name.startswith("._"))

        if len(low_paths) != len(high_paths):
            raise RuntimeError(
                f"Mismatched tensors in {split_dir}: "
                f"{len(low_paths)} low vs {len(high_paths)} high"
            )
        if not low_paths:
            raise RuntimeError(f"No tensors found in {split_dir}")

        self.low_paths = low_paths
        self.high_paths = high_paths
        self.scene_ids = [p.stem.replace("_low", "") for p in low_paths]

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.load(self.low_paths[idx], weights_only=False)
        high = torch.load(self.high_paths[idx], weights_only=False)
        return low, high
