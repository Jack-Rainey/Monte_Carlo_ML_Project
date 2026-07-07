"""Energy-domain signal metrics (per scene)."""
from __future__ import annotations

import numpy as np
import torch

from .metric_row import MetricTriple


def compute_signal_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
) -> dict[str, MetricTriple]:
    """
    All tensors: (C, n_bands, n_frames) normalized or denormalized log energy.

    Returns per-scene metric triples (low, pred, high) — see metric_row. The eval
    stage derives `improved`/`baseline_rel_ratio` uniformly from each triple.
    """
    # Energy MSE vs the high reference. The reference metric value is 0 (high vs
    # itself), so improved ⟺ pred_mse < baseline_mse and baseline_rel_ratio ⟺ the
    # old improvement_ratio (baseline_mse / pred_mse).
    pred_mse = float((high_ref - pred_energy).pow(2).mean())
    baseline_mse = float((high_ref - low_ref).pow(2).mean())

    # Energy-domain SNR on the W channel (channel 0, omni/ISO-3382) in linear power.
    # Use mean(P²)/mean((P_high-P_pred)²) so both terms have the same units (P²).
    # Diagnostic-only (a pred-vs-high quantity, no baseline comparison): low=high=NaN
    # so improvement is reported as undefined rather than misattributed.
    high_linear = 10.0 ** (high_ref / 10.0)      # (C, n_bands, n_frames)
    pred_linear = 10.0 ** (pred_energy / 10.0)
    high_w = high_linear[0]                        # W channel: (n_bands, n_frames)
    pred_w = pred_linear[0]
    signal_power_sq = float(high_w.pow(2).mean())
    error_power_sq = float((high_w - pred_w).pow(2).mean())
    energy_snr_db = 10.0 * np.log10(signal_power_sq / (error_power_sq + 1e-10))

    return {
        "energy_mse": MetricTriple(low=baseline_mse, pred=pred_mse, high=0.0),
        "energy_snr_db": MetricTriple(
            low=float("nan"), pred=float(energy_snr_db), high=float("nan")
        ),
    }
