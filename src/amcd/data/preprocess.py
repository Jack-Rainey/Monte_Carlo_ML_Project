"""
preprocess stage:
  renders/ → energy tensors (encoded, normalized, split-assigned) → preprocessed/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..representations import build_representation
from ..runtime import Verbosity, emit
from ..simulators.base import SceneSpec
from .normalization import compute_stats, normalize
from .splits import assign_split


def run_preprocess(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    out_dir = run_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous split assignment before writing this one (F-25). Split
    # membership can change between runs (a different split_assignment seed, a
    # different sizing mode), and a scene that MOVES leaves its tensor behind in
    # the old directory. That residue was then trained on while being scored as
    # held-out — with splits.json still reporting the correct assignment, so
    # nothing in the artifacts revealed it. Rewriting the directory set is the
    # producer half of the fix; EnergyDataset refusing unlisted tensors is the
    # consumer half, and neither is sufficient alone.
    for stale in out_dir.iterdir():
        if stale.is_dir() and stale.name != "carrier":
            shutil.rmtree(stale)
    # Create a directory for EVERY declared split, including ones that end up
    # empty (F-40). Otherwise a split that receives no scenes has no directory and
    # EnergyDataset raises "Split directory not found", pointing at the filesystem
    # instead of at the empty split that `split_counts` now reports as 0.
    for split_name in config.splits:
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Load scene specs
    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scene specs found in {scenes_dir}. Run gen-scenes first.")

    scenes = [SceneSpec.from_json(p) for p in scene_paths]

    # Assign splits deterministically from scene spec hashes (config-declared set)
    splits: dict[str, str] = {
        s.scene_id: assign_split(s.to_dict(), config)
        for s in scenes
    }

    # The training split (role: train) is the sole source of normalization stats.
    # Its uniqueness is enforced at config load (REQUIRED_ROLE_COUNTS) rather than
    # here, so the failure lands before gen-scenes and render rather than after
    # them (F-44).
    train_split = config.the_split_with_role("train")

    # Build representation (rep-agnostic: params validated by the rep's own schema)
    rep = build_representation(
        config.representation.name, config.representation.params,
        sample_rate=config.sample_rate,
    )

    # Encode all IRs
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    for scene in scenes:
        render_dir = renders_dir / scene.scene_id
        low_ir = np.load(render_dir / "low.npy")    # (C, T) float32
        high_ir = np.load(render_dir / "high.npy")  # (C, T) float32

        # Shape invariant: storage is (C, T)
        assert low_ir.shape == (config.n_channels, config.n_samples), (
            f"Unexpected low IR shape {low_ir.shape} for {scene.scene_id}"
        )

        low_energy = rep.encode(low_ir)    # (C, n_bands, n_frames)
        high_energy = rep.encode(high_ir)

        encoded[scene.scene_id] = (low_energy, high_energy, low_ir)

    # Compute norm stats from TRAINING split only (invariant #3)
    train_ids = [sid for sid, sp in splits.items() if sp == train_split]
    if not train_ids:
        raise RuntimeError(
            f"Training split {train_split!r} is empty — increase scenes.n_id or adjust fracs."
        )

    train_lows = [encoded[sid][0] for sid in train_ids]
    train_highs = [encoded[sid][1] for sid in train_ids]
    norm_stats = compute_stats(train_lows, train_highs)

    # Derive shape metadata from first tensor
    sample_low = encoded[scenes[0].scene_id][0]
    n_bands = sample_low.shape[1]
    n_frames = sample_low.shape[2]

    # Save normalized tensors per split
    for scene in scenes:
        split = splits[scene.scene_id]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        low_energy, high_energy, low_ir = encoded[scene.scene_id]

        # Both inputs and targets normalized with high stats so the residual skip
        # connection (pred = low + model(low)) lives in the same space as the target.
        norm_low = normalize(low_energy, norm_stats["high_mean"], norm_stats["high_std"])
        norm_high = normalize(high_energy, norm_stats["high_mean"], norm_stats["high_std"])

        torch.save(norm_low, split_dir / f"{scene.scene_id}_low.pt")
        torch.save(norm_high, split_dir / f"{scene.scene_id}_high.pt")

        # Save carrier (raw low-ray IR for D3 reconstruction)
        carrier_dir = out_dir / "carrier"
        carrier_dir.mkdir(parents=True, exist_ok=True)
        np.save(carrier_dir / f"{scene.scene_id}.npy", low_ir)

    # Save metadata. Counts are keyed on the CONFIG-DECLARED split set, defaulting
    # to 0, not on the splits that happen to have received a scene (F-30): keying
    # on observed values made the empty-split warning below unreachable, so a
    # declared split that received nothing simply vanished from the record instead
    # of being reported as 0.
    all_split_names = list(config.splits)
    meta = {
        "n_channels": config.n_channels,
        "n_bands": n_bands,
        "n_frames": n_frames,
        "center_freqs": rep.center_freqs,
        # Declared domain of the saved tensors ("db" | "amplitude"); dB-assuming
        # eval consumers key on this stamp, never on the rep class (F-19).
        "value_domain": rep.value_domain,
        "norm_stats": norm_stats,
        "split_counts": {
            sp: sum(1 for s in splits.values() if s == sp)
            for sp in all_split_names
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    counts = meta["split_counts"]
    counts_str = ", ".join(f"{sp}={n}" for sp, n in counts.items())
    emit(verbosity, "progress", f"  Preprocessed {len(scenes)} scenes ({counts_str}) → {out_dir}")
    emit(
        verbosity, "metrics",
        f"  Energy tensors: (C={config.n_channels}, bands={n_bands}, frames={n_frames})",
    )

    for sp, count in counts.items():
        if count == 0:
            emit(
                verbosity, "warning",
                f"  WARNING: {sp} split is empty. Increase scenes.n_id or adjust fracs.",
            )
