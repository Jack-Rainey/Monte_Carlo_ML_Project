"""
Loss construction for the train stage.

This is the single seam where the training loss is built. Today it is one Huber
term on log-band-energy; design_spec §3 names further terms that will attach here
(a differentiable metric-consistency term computed from the predicted EDR, tail
weighting, and the stubbed inter-channel spatial term). Keeping construction in
one place means adding those terms never touches the trainer loop.

Why the δ scaling matters (design_spec §2 H2, §3, invariant #7 "δ vs signal scale").
The reframe to log-band-energy exists so the Huber knee δ is O(1)-meaningful in dB.
But preprocess z-scores both operands by the training high-target std (high_std ≈
tens of dB), so the loss runs on dimensionless, unit-variance operands. A δ quoted
in dB is therefore NOT δ in the operand domain: it must be divided by high_std, or
the knee sits tens of dB out and Huber silently degenerates to MSE (the H2 pathology
the design claims to resolve). `config.huber_delta` is δ in dB; `delta_db_to_norm`
maps it into the normalized domain the criterion actually sees.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import Config


def delta_db_to_norm(delta_db: float, high_std: float) -> float:
    """Convert a Huber δ expressed in dB into the z-scored (÷ high_std) loss domain.

    The loss compares operands normalized by the training high-target std, so a
    residual of `delta_db` dB corresponds to `delta_db / high_std` in that domain.
    Setting HuberLoss(delta=...) to this value places the quadratic→linear knee at
    exactly `delta_db` dB, restoring the O(1)-in-dB meaning (design_spec §2 H2 / §3).
    The 1e-8 mirrors normalization.normalize's guard against a degenerate zero std.
    """
    return delta_db / (high_std + 1e-8)


def build_criterion(config: Config, norm_stats: dict) -> nn.Module:
    """Build the training criterion from the config δ (in dB) and the run's norm stats.

    `norm_stats` is the preprocess-stamped dict (meta["norm_stats"]) carrying the
    training high-target std used to normalize both operands. δ stays a config value
    in dB (a §7 tuned-capable op-point); the per-run high_std is data-derived, so the
    conversion happens here, at construction, not in the config.
    """
    delta_norm = delta_db_to_norm(config.huber_delta, norm_stats["high_std"])
    return nn.HuberLoss(delta=delta_norm)
