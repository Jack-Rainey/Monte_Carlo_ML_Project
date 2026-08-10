"""PyTorch Dataset over pre-normalized energy tensors from the preprocess stage."""
from __future__ import annotations

import json
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

        # Membership comes from the MANIFEST, never from whatever files happen to
        # be on disk (F-25). Globbing made the loader trust directory contents over
        # `splits.json`, so a scene that changed splits between preprocess runs was
        # left behind in its old directory and silently trained on while being
        # scored as held-out — with splits.json still reporting it correctly, so no
        # artifact revealed the leak.
        manifest_path = preprocessed_dir / "splits.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}. Re-run the preprocess "
                f"stage — split membership is defined by that file, not by the "
                f"contents of {split_dir}."
            )
        manifest: dict[str, str] = json.loads(manifest_path.read_text())
        scene_ids = sorted(sid for sid, sp in manifest.items() if sp == split)
        if not scene_ids:
            raise RuntimeError(f"Split {split!r} has no scenes in {manifest_path}")

        low_paths = [split_dir / f"{sid}_low.pt" for sid in scene_ids]
        high_paths = [split_dir / f"{sid}_high.pt" for sid in scene_ids]
        missing = [p.name for p in (*low_paths, *high_paths) if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Split {split!r} is missing {len(missing)} tensor(s) the manifest "
                f"declares, e.g. {missing[:3]}. The preprocessed directory is stale "
                f"relative to {manifest_path}; re-run preprocess."
            )

        # Any tensor present but NOT in the manifest means a previous run wrote a
        # different assignment into this directory. Refuse rather than ignore: this
        # is the residue that caused the leak, and silently skipping it would hide
        # that the run_dir holds two datasets. (Resource-fork files that appear on
        # exFAT volumes are not residue.)
        # Both suffixes, not just `_low` — the check claims "any tensor not in the
        # manifest", and an orphan `_high` with no `_low` partner would otherwise
        # be invisible to the backstop guarding the F-25 blocker (F-39).
        on_disk = {p.stem.rsplit("_", 1)[0] for p in split_dir.glob("*.pt")
                   if not p.name.startswith("._")}
        orphans = sorted(on_disk - set(scene_ids))
        if orphans:
            raise RuntimeError(
                f"Split {split!r} contains {len(orphans)} tensor(s) absent from the "
                f"manifest, e.g. {orphans[:3]}. They are left over from a previous "
                f"preprocess run under a different split assignment — training on "
                f"them would score trained-on scenes as held-out. Re-run preprocess "
                f"(it now clears split directories) or use a fresh --run-dir."
            )

        self.low_paths = low_paths
        self.high_paths = high_paths
        self.scene_ids = scene_ids

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.load(self.low_paths[idx], weights_only=False)
        high = torch.load(self.high_paths[idx], weights_only=False)
        return low, high
