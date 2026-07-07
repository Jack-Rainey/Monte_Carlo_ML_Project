"""Normalization stats — computed from training split only (invariant #3)."""
from __future__ import annotations

import torch


def compute_stats(
    train_lows: list[torch.Tensor],
    train_highs: list[torch.Tensor],
) -> dict[str, float]:
    """
    Compute scalar mean/std for low inputs and high targets separately, from the
    TRAINING split only (invariant #3: stats never see valid/test).

    NOTE on which stats are applied: the residual framing (pred = low + model(low))
    requires input and target to live in ONE affine frame, so preprocess.py
    normalizes BOTH low and high with the HIGH stats. `low_mean`/`low_std` are
    therefore computed and stamped for provenance/diagnostics but are NOT applied
    to any tensor. See preprocess.py and ledger F-02.
    """
    if not train_lows or not train_highs:
        raise ValueError("Cannot compute stats: training split is empty.")

    low_all = torch.stack(train_lows)   # (N, C, bands, frames)
    high_all = torch.stack(train_highs)

    return {
        "low_mean": float(low_all.mean()),
        "low_std": float(low_all.std()),
        "high_mean": float(high_all.mean()),
        "high_std": float(high_all.std()),
    }


def normalize(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (tensor - mean) / (std + 1e-8)


def denormalize(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * (std + 1e-8) + mean
