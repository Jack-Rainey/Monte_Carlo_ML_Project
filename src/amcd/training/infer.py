"""infer stage: apply trained model to all test scenes, save predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import EnergyDataset
from ..data.normalization import denormalize
from ..representations import build_representation
from ..models.cnn import build_model  # noqa: F401 — import also triggers registration
from ..runtime import Verbosity, emit


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_infer(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    preprocessed_dir = run_dir / "preprocessed"
    checkpoint_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous predictions before writing this model's (F-37). Predictions
    # are keyed by scene_id, not by model or split assignment, so a re-run that
    # moves a scene out of a test split — or simply trains a different model —
    # leaves the OLD model's prediction on disk under a name the eval stage globs.
    # Same residue pattern as F-25, one stage downstream.
    for stale in (*predictions_dir.glob("*_pred.pt"),
                  *predictions_dir.glob("*_decoded_ir.npy")):
        stale.unlink()

    best_ckpt = checkpoint_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_ckpt}. Run train first.")

    with open(preprocessed_dir / "meta.json") as f:
        meta = json.load(f)
    with open(preprocessed_dir / "splits.json") as f:
        split_map: dict[str, str] = json.load(f)

    n_channels = meta["n_channels"]
    norm_stats = meta["norm_stats"]
    device = _select_device()

    # Instantiate representation for D3 decode (required per §6, §3-D3)
    rep = build_representation(
        config.representation.name, config.representation.params,
        sample_rate=config.sample_rate,
    )
    carrier_dir = preprocessed_dir / "carrier"

    model = build_model(config.model.name, n_channels, config.model.params).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    # Run on all populated test splits (held-out final eval — never used for selection).
    present_test_splits = [
        sp for sp in config.test_split_names
        if any(s == sp for s in split_map.values())
    ]
    # A declared test split with no scenes is dropped here silently otherwise (F-45);
    # eval/stats/report now report it as unscored, so the reason must be visible at
    # the stage that first drops it.
    for sp in config.test_split_names:
        if sp not in present_test_splits:
            emit(
                verbosity, "warning",
                f"  WARNING: declared test split {sp!r} has no scenes — no predictions "
                f"will be written for it; it will be reported as unscored.",
            )

    if not present_test_splits:
        raise RuntimeError(
            "No test scenes found in any test split. "
            "Increase scenes.n_id or check split assignment."
        )

    total = 0
    with torch.no_grad():
        for split_name in present_test_splits:
            test_ds = EnergyDataset(preprocessed_dir, split_name)
            loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
            for i, (low, _high) in enumerate(loader):
                low = low.to(device)
                pred = low + model(low)           # (1, C, n_bands, n_frames)
                pred_cpu = pred.squeeze(0).cpu()  # (C, n_bands, n_frames)
                scene_id = test_ds.scene_ids[i]
                torch.save(pred_cpu, predictions_dir / f"{scene_id}_pred.pt")

                # D3: decode predicted energy onto low-ray carrier (§3-D3, §6)
                pred_db = denormalize(pred_cpu, norm_stats["high_mean"], norm_stats["high_std"])
                carrier = np.load(carrier_dir / f"{scene_id}.npy")  # (C, T)
                decoded_ir = rep.decode(pred_db, carrier)            # (C, T) float32
                np.save(predictions_dir / f"{scene_id}_decoded_ir.npy", decoded_ir)

                total += 1

    emit(verbosity, "progress", f"  Saved {total} predictions → {predictions_dir}")
