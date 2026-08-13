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
) -> tuple[dict[str, MetricTriple], dict[tuple[str, str], str]]:
    """
    All tensors: (C, n_bands, n_frames) in the rep's declared `value_domain` —
    "db" (log band-energy; the banded-rep contract) or "amplitude" (raw samples;
    the waveform identity rep). The caller passes the preprocess-stamped domain,
    never infers it from a rep class (F-19).

    Returns (triples, nan_reasons). Each metric declares its improvement `kind`
    on its triple (F-20); nan_reasons maps (metric, leg) → why that leg is NaN,
    for every NaN consumed leg, so the evaluator can log the drop (F-21).

    `energy_mse` is an operand-domain MSE vs the high reference, defined in
    either domain; kind = match_reference (high vs itself is genuinely 0, a
    finite reachable reference). `energy_snr_db` undoes a dB encoding
    (10**(x/10)); kind = maximize — its low leg is the SNR of the LOW-RAY
    baseline against the high reference, so improvement = SNR(pred) − SNR(low)
    with no reference leg (SNR of the reference against itself is +∞, which is
    why match-reference framing silently unscored it — F-20). On an
    amplitude-domain rep the dB undo is meaningless, so both SNR legs are NaN
    (undefined) with the reason logged instead of silently emitting garbage.
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

    nan_reasons: dict[tuple[str, str], str] = {}
    if value_domain == "db":
        # Energy-domain SNR on the W channel (channel 0, omni/ISO-3382) in linear power.
        # Use mean(P²)/mean((P_high-P_x)²) so both terms have the same units (P²).
        high_w = (10.0 ** (high_ref / 10.0))[0]        # W channel: (n_bands, n_frames)
        signal_power_sq = float(high_w.pow(2).mean())

        def _snr_db(x: torch.Tensor) -> float:
            x_w = (10.0 ** (x / 10.0))[0]
            error_power_sq = float((high_w - x_w).pow(2).mean())
            return float(10.0 * np.log10(signal_power_sq / (error_power_sq + 1e-10)))

        pred_snr_db = _snr_db(pred_energy)
        low_snr_db = _snr_db(low_ref)
    else:
        pred_snr_db = low_snr_db = float("nan")
        reason = "energy_snr_db undefined for amplitude value_domain (no dB encoding to undo)"
        nan_reasons[("energy_snr_db", "low")] = reason
        nan_reasons[("energy_snr_db", "pred")] = reason

    triples = {
        "energy_mse": MetricTriple(
            low=baseline_mse, pred=pred_mse, high=0.0, kind="match_reference",
            # Operand-domain squared: whatever the representation encodes in. The
            # reporting layer resolves it from the stamped `value_domain`.
            unit="operand_domain_squared",
        ),
        "energy_snr_db": MetricTriple(
            low=low_snr_db, pred=pred_snr_db, high=float("nan"), kind="maximize", unit="dB"
        ),
    }
    return triples, nan_reasons
