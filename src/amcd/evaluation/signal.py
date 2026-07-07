"""Energy-domain signal metrics (per scene)."""
from __future__ import annotations

import numpy as np
import torch

from .metric_row import MetricTriple


def compute_signal_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
    *,
    value_domain: str,
) -> dict[str, MetricTriple]:
    """
    All tensors: (C, n_bands, n_frames) in the rep's declared `value_domain` —
    "db" (log band-energy; the banded-rep contract) or "amplitude" (raw samples;
    the waveform identity rep). The caller passes the preprocess-stamped domain,
    never infers it from a rep class (F-19).

    Returns per-scene metric triples (low, pred, high) — see metric_row. The eval
    stage derives `improved`/`baseline_rel_ratio` uniformly from each triple.
    `energy_mse` is an operand-domain MSE, defined in either domain. `energy_snr_db`
    undoes a dB encoding (10**(x/10)); on an amplitude-domain rep that undo is
    meaningless, so the metric is reported as the documented all-NaN triple
    (undefined) instead of silently emitting garbage numbers.
    """
    if value_domain not in ("db", "amplitude"):
        raise ValueError(
            f"Unknown representation value_domain {value_domain!r}; expected 'db' or 'amplitude'."
        )

    # Energy MSE vs the high reference. The reference metric value is 0 (high vs
    # itself), so improved ⟺ pred_mse < baseline_mse and baseline_rel_ratio ⟺ the
    # old improvement_ratio (baseline_mse / pred_mse).
    pred_mse = float((high_ref - pred_energy).pow(2).mean())
    baseline_mse = float((high_ref - low_ref).pow(2).mean())

    if value_domain == "db":
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
        energy_snr_db = float(10.0 * np.log10(signal_power_sq / (error_power_sq + 1e-10)))
    else:
        energy_snr_db = float("nan")

    return {
        "energy_mse": MetricTriple(low=baseline_mse, pred=pred_mse, high=0.0),
        "energy_snr_db": MetricTriple(
            low=float("nan"), pred=energy_snr_db, high=float("nan")
        ),
    }
