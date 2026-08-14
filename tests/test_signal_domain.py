"""Signal-metric value-domain guard.

`energy_snr_db` undoes a dB encoding (10**(x/10)). The waveform rep encodes raw
amplitude, so that undo is meaningless there — pre-guard, the E1 waveform run
would have silently emitted garbage SNR numbers. These pin: the guard keys on
the rep's DECLARED `value_domain` (never isinstance of a concrete rep — the
scaffold/seam rule); amplitude-domain reps get the documented all-NaN triple;
dB-domain reps keep a finite SNR; unknown domains fail loud.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from amcd.evaluation.signal import compute_signal_metrics
from amcd.representations import build_representation

from tests.conftest import EVAL_FREQS, tiny_config


def _fake_tensors(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    shape = (4, 1, 200)  # (C, bands, frames/samples)
    pred, high, low = (torch.from_numpy(rng.normal(0, 1, shape)).float() for _ in range(3))
    return pred, high, low


def test_energy_snr_undefined_for_amplitude_domain_rep() -> None:
    """On the waveform (amplitude) path, energy_snr_db must be the documented
    all-NaN triple — never a finite-but-garbage 10**(amplitude/10) number. The
    domain comes from the rep's declaration, exactly what preprocess stamps."""
    rep = build_representation("waveform", {}, sample_rate=8000, eval_freqs_hz=EVAL_FREQS)
    assert rep.value_domain == "amplitude"

    rng = np.random.default_rng(7)
    ir = rng.normal(0, 0.1, (4, 200)).astype(np.float32)
    pred = rep.encode(ir)
    high = rep.encode(ir * 0.9)
    low = rep.encode(ir * 0.5)

    out, nan_reasons = compute_signal_metrics(pred, high, low, value_domain=rep.value_domain)
    snr = out["energy_snr_db"]
    assert math.isnan(snr.pred) and math.isnan(snr.low) and math.isnan(snr.high)
    # No silent exclusion: both consumed legs carry an explicit reason.
    assert "amplitude" in nan_reasons[("energy_snr_db", "low")]
    assert "amplitude" in nan_reasons[("energy_snr_db", "pred")]
    # The operand-domain MSE stays defined for amplitude reps.
    assert math.isfinite(out["energy_mse"].pred) and math.isfinite(out["energy_mse"].low)


def test_energy_snr_finite_for_db_domain_rep() -> None:
    """The dB path (spectrogram/EDR declare "db") scores SNR as a `maximize`
    metric: finite pred AND low legs (improvement = SNR(pred) − SNR(low)),
    high leg structurally absent — and nothing to log as dropped."""
    # Params come from the config layer, never from literals here: the rep's own
    # schema forbids extras and requires every field, so a hardcoded dict silently
    # goes stale the moment the rep declares a new one (CLAUDE.md: no
    # experiment-governing values in test fixtures).
    cfg = tiny_config()
    rep = build_representation(
        cfg.representation.name, cfg.representation.params,
        sample_rate=cfg.sample_rate, eval_freqs_hz=EVAL_FREQS,
    )
    assert rep.value_domain == "db"

    pred, high, low = _fake_tensors(11)
    out, nan_reasons = compute_signal_metrics(pred, high, low, value_domain="db")
    snr = out["energy_snr_db"]
    assert snr.kind == "maximize"
    assert math.isfinite(snr.pred) and math.isfinite(snr.low)
    assert math.isnan(snr.high)  # structurally absent, not a drop
    assert nan_reasons == {}


def test_unknown_value_domain_fails_loud() -> None:
    """No hidden default: an undeclared/unknown domain is an error, not a guess."""
    pred, high, low = _fake_tensors(13)
    with pytest.raises(ValueError, match="value_domain"):
        compute_signal_metrics(pred, high, low, value_domain="linear")
