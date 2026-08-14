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
    therefore computed and stamped but are NOT applied to any tensor. See
    preprocess.py.

    WHY THEY ARE COMPUTED ANYWAY, since "emits output, contributes nothing" is a
    fair reading of unapplied output: they are the only on-disk record of how the
    LOW-RAY leg's distribution differs from the high leg — the axis the roadmap's
    ray-count sweep (paper §6) varies, and the quantity that says whether the
    single affine frame above is a reasonable framing or a lossy one at a given
    ray budget.

    They are recorded under `low_mean`/`low_std`, which say nothing themselves —
    `tests/test_invariants.py` asserts those four key names verbatim, so renaming
    them is a coordinated change. The disclosure is a SIBLING key in the same
    `meta.json`, `norm_stats_applied`, naming which of the four were applied.
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
