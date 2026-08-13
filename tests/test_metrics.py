"""Room-acoustic metric known-answer probes (ISO 3382; ledger AC-02, AC-04).

These are the load-bearing evidence for the two metric fixes — the dry_run pipeline's
zero-delay, low-noise synthetic IRs do not exercise either path, so correctness is
proved here with synthetic IRs of known structure, not by the pipeline run.
"""
from pathlib import Path

import numpy as np
import torch

from amcd.config import Config

from amcd.evaluation.room_acoustic import channel_band_avg_metrics
from tests.conftest import EVAL_FREQS

_SR = 48000
_ISO = [500.0, 1000.0]
_ONSET_DB = -20.0
# A margin of 0 disables the resolvability floor entirely. Not an experiment value:
# these probes assert what the ESTIMATOR does, so a known-answer test must see the
# raw fitted value, including one too short to be reported in a real run.
# `metric_band_resolvability_margin` is config-declared for the pipeline
# (configs/base.yaml); tests that exercise the floor itself set it explicitly.
_NO_FLOOR = 0.0

#: Disables the ISO 3382-1 SNR admissibility bound (AC-176) for probes whose
#: subject is the ESTIMATOR itself. Same rationale as `_NO_FLOOR`: a known-answer
#: test must see the raw fitted value, including one the pipeline would refuse.
#: Tests whose subject IS the bound pass the standard's values explicitly.
_NO_SNR_BOUND = {"T30": 0.0, "EDT": 0.0}
#: ISO 3382-1's own requirement: >= 45 dB of usable decay for T30, >= 20 for EDT.
_ISO_SNR_BOUND = {"T30": 45.0, "EDT": 20.0}

#: The shipped octave-filter order, read from the config that governs it (F-143).
#: NOT a literal 4: `metric_octave_filter.order` is an experiment parameter, and a
#: probe holding its own copy would keep asserting against the old filter while
#: claiming to describe the shipped one — the `_base_scalar` precedent below.
_ORDER = Config.load(Path("configs/base.yaml")).metric_octave_filter.order

#: Realized out-of-band leakage bounds, likewise read rather than pinned (RD-186).
#: Keyed by octaves outside the band.
_STOPBAND_DB = Config.load(
    Path("configs/base.yaml")
).metric_octave_filter.stopband_rejection_db



def _decaying_noise_ir(rt60: float, duration_s: float, seed: int) -> np.ndarray:
    """A synthetic reverb IR: white noise under a -60 dB/RT60 energy-decay envelope,
    direct sound at t=0. (T,) float64."""
    rng = np.random.default_rng(seed)
    n = int(_SR * duration_s)
    t = np.arange(n) / _SR
    envelope = 10.0 ** (-3.0 * t / rt60)          # amplitude → energy -60 dB at RT60
    return rng.standard_normal(n) * envelope


def _add_noise_floor(ir: np.ndarray, floor_db: float, seed: int) -> np.ndarray:
    """Add a stationary white-noise floor `floor_db` below the IR's peak amplitude."""
    rng = np.random.default_rng(seed)
    peak = float(np.abs(ir).max())
    floor_rms = peak * 10.0 ** (floor_db / 20.0)
    return ir + rng.standard_normal(ir.shape) * floor_rms


def _metrics(ir_w: np.ndarray) -> dict:
    values, _reasons = channel_band_avg_metrics(
        ir_w, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    return values


def test_c50_invariant_to_noise_floor() -> None:
    """AC-04: with the late window truncated at the Lundeby index, reported C50 is
    ~invariant to the noise-floor level. (Pre-fix, integrating the full tail, a -40 dB
    floor vs -55 dB floor moved C50 by many dB.)"""
    clean = _decaying_noise_ir(rt60=0.5, duration_s=1.5, seed=1)
    ir_hi_floor = _add_noise_floor(clean, floor_db=-40.0, seed=2)
    ir_lo_floor = _add_noise_floor(clean, floor_db=-55.0, seed=2)

    c50_hi = _metrics(ir_hi_floor)["C50"]
    c50_lo = _metrics(ir_lo_floor)["C50"]

    assert np.isfinite(c50_hi) and np.isfinite(c50_lo)
    # Tolerance 0.1 dB is a genuine kill probe (ledger AC-06): on this IR the pre-fix
    # code (late window integrated to end-of-record) diverges 0.56 dB and FAILS, while
    # the fix's truncated late window gives 0.001 dB. A loose tol would pass pre-fix.
    assert abs(c50_hi - c50_lo) < 0.1, (
        f"C50 not noise-floor invariant: {c50_hi:.3f} vs {c50_lo:.3f} dB — "
        f"late window may still include the noise tail (AC-04)."
    )


def test_metrics_invariant_to_leading_silence() -> None:
    """AC-02: onset alignment makes T30/EDT/C50 invariant to leading propagation-delay
    silence — prepending zeros must not change any metric."""
    ir = _add_noise_floor(_decaying_noise_ir(rt60=0.6, duration_s=1.5, seed=3),
                          floor_db=-50.0, seed=4)
    pad = np.zeros(int(0.02 * _SR))              # 20 ms of leading silence (≈ 6.8 m delay)
    ir_delayed = np.concatenate([pad, ir])

    m0 = _metrics(ir)
    m1 = _metrics(ir_delayed)

    for key in ("T30", "EDT", "C50"):
        assert np.isfinite(m0[key]) and np.isfinite(m1[key])
        assert abs(m0[key] - m1[key]) < 1e-6 * max(1.0, abs(m0[key])), (
            f"{key} changed under leading silence: {m0[key]} vs {m1[key]} — onset "
            f"alignment not applied (AC-02)."
        )


def test_c50_nan_carries_lundeby_truncation_reason() -> None:
    """F-21: when Lundeby truncation lands before the 50 ms split, C50 is NaN — a
    physics-legitimate absence (a sub-50 ms IR has no honest C50) — but the drop
    must carry a reason, not vanish silently. This is the exact
    test_geometry_shift N=2-not-3 pathology from the first full dry run."""
    # Very fast decay (RT60 = 30 ms) over a -25 dB noise floor: band energy falls
    # below the Lundeby threshold (~floor + 10 dB) around 0.25·RT60 ≈ 7.5 ms, so
    # the truncation index sits far before the 50 ms split.
    ir = _add_noise_floor(_decaying_noise_ir(rt60=0.03, duration_s=0.5, seed=5),
                          floor_db=-25.0, seed=6)
    values, reasons = channel_band_avg_metrics(
        ir, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    assert np.isnan(values["C50"]), "fixture must actually produce an unscored C50"
    assert "C50" in reasons
    assert "50 ms split" in reasons["C50"], (
        f"C50 drop reason missing/wrong: {reasons.get('C50')!r}"
    )


def test_paired_metrics_share_a_band_set_across_legs() -> None:
    """AC-08 kill test: a pred-only degenerate band must never produce a scored
    cross-band-set comparison. compute_room_acoustic_metrics intersects the
    surviving-band set across all three legs, so every leg is averaged over the
    SAME bands (ISO-3382 band averages are only comparable over a common set).
    Pre-fix, each leg averaged its own surviving bands and the scene scored."""
    import pytest
    from scipy.signal import butter, sosfiltfilt

    from amcd.evaluation.room_acoustic import (
        _shared_truncation_per_band,
        channel_per_band_metrics,
        compute_room_acoustic_metrics,
    )

    high = _add_noise_floor(_decaying_noise_ir(rt60=0.5, duration_s=1.5, seed=7),
                            floor_db=-50.0, seed=8)
    pred = _add_noise_floor(_decaying_noise_ir(rt60=0.55, duration_s=1.5, seed=12),
                            floor_db=-48.0, seed=13)
    # `low` carries healthy decaying content ONLY in the 500 Hz octave; its 1000 Hz
    # eval band sees just a stationary floor → Lundeby-degenerate there (C50 NaN),
    # while pred/high stay finite in both bands.
    #
    # NOTE (AC-17): this fixture used to make PRED the degenerate leg, relying on
    # pred's own collapsed Lundeby index. Since the integration window is now derived
    # from the physical legs only (RD-43), a degenerate pred is integrated over the
    # healthy window and gets SCORED rather than dropped — which is the more honest
    # outcome (a model must not escape scoring by emitting garbage), but it no longer
    # produces a band drop. The band-intersection code under test is unchanged and
    # leg-agnostic; the drop is induced here from a leg that can still cause one.
    # That pred cannot influence the window at all is asserted separately by
    # test_prediction_cannot_set_the_integration_window.
    low_base = _add_noise_floor(_decaying_noise_ir(rt60=0.6, duration_s=1.5, seed=9),
                                floor_db=-45.0, seed=10)
    sos = butter(8, [500.0 / 2**0.5, 500.0 * 2**0.5], btype="bandpass",
                 fs=_SR, output="sos")
    rng = np.random.default_rng(11)
    stationary = rng.standard_normal(low_base.shape) * (
        np.abs(low_base).max() * 10.0 ** (-35.0 / 20.0)
    )
    low = sosfiltfilt(sos, low_base) + stationary

    # Fixture must bite: low's 1000 Hz band C50 is NaN, its 500 Hz band finite.
    low_bands = channel_per_band_metrics(
        low, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR, min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER)
    assert np.isfinite(low_bands[0][0]["C50"]) and np.isnan(low_bands[1][0]["C50"])

    triples, reasons, _window, _acct = compute_room_acoustic_metrics(
        pred[None, :], high[None, :], low[None, :],
        sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    c50 = triples["C50"]
    # The drop is attributed to the causing leg and marked as affecting all legs.
    # `low` is a PHYSICAL leg, so it still sets the band set (AC-25 removed only
    # pred's vote); the wording moved to "the physical legs" with it.
    assert ("C50", "low") in reasons
    assert "excluded from every leg's average" in reasons[("C50", "low")]
    assert "physical legs" in reasons[("C50", "low")]

    # Every leg is averaged over the intersected set ({500 Hz} only): the high
    # leg must equal its own single-band 500 Hz value, and that must genuinely
    # differ from its own-two-band average (else the fixture proves nothing).
    #
    # The single-band reference must be computed over the SAME shared window the
    # paired call used (AC-17) — comparing a shared-window value against a
    # standalone-window one would fail for a reason that has nothing to do with
    # band composition, which is what this test is about.
    shared_500 = _shared_truncation_per_band(
        {"low": low, "high": high},
        sample_rate=_SR, iso_eval_freqs=[500.0], onset_rel_db=_ONSET_DB,
        octave_filter_order=_ORDER,
    )
    high_500 = channel_band_avg_metrics(
        high, sample_rate=_SR, iso_eval_freqs=[500.0], onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR, min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
        trunc_idx_per_band=shared_500)[0]["C50"]
    high_both = channel_band_avg_metrics(
        high, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR, min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER)[0]["C50"]
    assert c50.high == pytest.approx(high_500)
    assert high_500 != pytest.approx(high_both)
    assert np.isfinite(c50.pred) and np.isfinite(c50.low)  # still scored, same band set


# ---------------------------------------------------------------------------
# AC-17 / RD-43: the cross-leg shared Schroeder integration window
# ---------------------------------------------------------------------------

def test_t30_invariant_to_leg_noise_floor() -> None:
    """AC-17 kill test: the Monte-Carlo noise floor is this study's INDEPENDENT
    VARIABLE, so a metric difference that comes only from the floor is a
    manufactured result.

    Identical decay and identical noise realization in both legs; only the floor
    amplitude differs, by sqrt(40) — the 5,000:200,000 ray ratio. Pre-fix, each leg
    was truncated at its own Lundeby index and the noisier leg read T30 short by
    -15.7 / -27.8 / -50.2 % at floors of -50 / -40 / -30 dB (measured), against the
    project's own declared T30 JND of 0.05 (`d0b_t30_jnd_frac`). With one shared
    window the two legs must agree.
    """
    from amcd.evaluation.room_acoustic import (
        _shared_truncation_per_band,
        channel_band_avg_metrics,
    )

    n = int(3.0 * _SR)
    t = np.arange(n) / _SR
    # AC-34: a SEED SWEEP asserting on the WORST realization, not one fixed pair.
    #
    # The shipped test used a single seed pair and passed comfortably. Measured
    # independently over 24 seeds with independent noise per leg, the MEAN residual
    # is 0.05-2.70 % — the fix works — but the per-scene WORST reaches +9.27 % at a
    # -50→-40 dB floor and -10.32 % at -30 dB, up to 2x the declared
    # `d0b_t30_jnd_frac` of 0.05. That matters because D0b's JND test is a PER-SCENE
    # decision: a residual that averages down over a split does not average down
    # inside one scene's verdict. Asserting on the worst case is what keeps the
    # residual from growing unnoticed; the real closure is RD-55's Lundeby
    # extrapolated-tail compensation, which removes it rather than bounding it.
    seeds = range(8)
    worst: dict[tuple[float, float], float] = {}
    for t60 in (0.5, 1.0, 2.0):
        for floor_db in (-80.0, -60.0, -50.0, -40.0, -30.0):
            for seed in seeds:
                decay = (np.random.default_rng(100 + seed).standard_normal(n)
                         * np.exp(-6.9077 * t / t60))
                # Independent noise per leg — the realistic case. Sharing one
                # realization and scaling it, as the original fixture did,
                # understates the residual by construction.
                amp = 10.0 ** (floor_db / 20.0)
                legs = {
                    "high": (decay + amp * np.random.default_rng(200 + seed)
                             .standard_normal(n)).astype(np.float32),
                    "low": (decay + amp * np.sqrt(40.0) * np.random.default_rng(300 + seed)
                            .standard_normal(n)).astype(np.float32),
                }
                shared = _shared_truncation_per_band(
                    legs, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
                    octave_filter_order=_ORDER,
                )
                vals = {
                    leg: channel_band_avg_metrics(
                        ir, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
                        band_resolvability_margin=_NO_FLOOR, trunc_idx_per_band=shared,
                        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
                    )[0]["T30"]
                    for leg, ir in legs.items()
                }
                assert np.isfinite(vals["high"]) and np.isfinite(vals["low"])
                rel = abs(vals["low"] - vals["high"]) / t60
                key = (t60, floor_db)
                worst[key] = max(worst.get(key, 0.0), rel)

    overall = max(worst.values())
    # Pinned at the measured worst case, not at the JND: the residual is REAL
    # in-window noise, and a bound set at the JND would hide it growing up to that
    # point. Tightening this number is progress; loosening it is a regression that
    # has to be argued for. Pre-fix, per-leg truncation reached 66.8 % on this grid.
    assert overall < 0.12, (
        f"worst-case cross-leg T30 residual is {100 * overall:.2f}% with IDENTICAL "
        f"decay (per (T60, floor): "
        f"{ {k: round(100 * v, 2) for k, v in sorted(worst.items())} }) — the "
        f"integration window may be noise-floor dependent again (AC-17/AC-34)."
    )


def test_prediction_cannot_set_the_integration_window() -> None:
    """RD-43 kill test: a model output must never influence the window used to
    measure its own ground truth.

    Deriving the shared index as a min over ALL legs (including `pred`) would let a
    degenerate prediction shorten the window for `high` and `low`, compressing the
    legs together and shrinking |pred - high| — a worse model scoring better by
    corrupting the measurement of its target. The physical legs' values must be
    identical whether `pred` is healthy or degenerate.
    """
    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    high = _add_noise_floor(_decaying_noise_ir(rt60=0.8, duration_s=2.0, seed=21),
                            floor_db=-60.0, seed=22)
    low = _add_noise_floor(_decaying_noise_ir(rt60=0.8, duration_s=2.0, seed=21),
                           floor_db=-45.0, seed=23)
    healthy = _add_noise_floor(_decaying_noise_ir(rt60=0.8, duration_s=2.0, seed=21),
                               floor_db=-50.0, seed=24)
    # Degenerate pred: a near-stationary floor, whose own Lundeby index collapses
    # to the 10 ms minimum and would truncate the physical legs if it were consulted.
    degenerate = (np.random.default_rng(25).standard_normal(high.shape)
                  * np.abs(high).max() * 10.0 ** (-3.0 / 20.0))

    out = {}
    for label, pred in (("healthy", healthy), ("degenerate", degenerate)):
        triples, _reasons, window, _acct = compute_room_acoustic_metrics(
            pred[None, :], high[None, :], low[None, :],
            sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
            band_resolvability_margin=_NO_FLOOR,
            min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
        )
        out[label] = (triples, window)

    # The window itself, and both physical legs, are untouched by pred's condition.
    assert out["healthy"][1] == out["degenerate"][1], (
        f"pred changed the shared integration window: {out['healthy'][1]} vs "
        f"{out['degenerate'][1]} — the window must derive from low/high only (RD-43)."
    )
    for metric in ("T30", "EDT", "C50"):
        for leg in ("high", "low"):
            a = getattr(out["healthy"][0][metric], leg)
            b = getattr(out["degenerate"][0][metric], leg)
            assert (np.isnan(a) and np.isnan(b)) or a == b, (
                f"{metric}.{leg} moved from {a} to {b} when only pred changed — a "
                f"model output is setting the measurement of its own target (RD-43)."
            )
    # And the window is recorded for every scored scene, not just dropped ones (RD-44).
    assert set(out["healthy"][1]) == {"500", "1000"}
    for band in out["healthy"][1].values():
        idx, src = band
        assert isinstance(idx, int) and src in ("low", "high")


def test_prediction_cannot_set_the_band_set() -> None:
    """AC-25 kill test — the analogue of the window test above, for RD-43's SECOND
    channel.

    The resolvability floor was applied to each leg's own measured value INCLUDING
    `pred`, and the band intersection then dropped that band from EVERY leg's
    average. So a prediction that was unmeasurable in one band changed the reported
    value of its own ground truth.

    REPRODUCED before the fix on exactly this fixture: high and low both carry
    identical 0.30 s decays, safely above the floor in both bands; replacing only
    PRED's 500 Hz octave with a 0.015 s decay moved HIGH's reported EDT from
    0.2926 s to 0.2498 s — a 14.6 % change with no change to high's waveform. Note
    |pred - high| GREW there, so the bias is not even conservatively directed; that
    is why it is closed rather than argued about.
    """
    import pytest
    from scipy.signal import butter, sosfiltfilt

    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    margin = 2.0  # the shipped value: the floor must actually bite for pred
    high = _add_noise_floor(_decaying_noise_ir(rt60=0.30, duration_s=1.0, seed=31),
                            floor_db=-60.0, seed=32)
    low = _add_noise_floor(_decaying_noise_ir(rt60=0.30, duration_s=1.0, seed=31),
                           floor_db=-50.0, seed=33)
    healthy = _add_noise_floor(_decaying_noise_ir(rt60=0.30, duration_s=1.0, seed=31),
                               floor_db=-55.0, seed=34)

    def _octave(fc: float, ir: np.ndarray) -> np.ndarray:
        sos = butter(8, [fc / 2**0.5, fc * 2**0.5], btype="bandpass",
                     fs=_SR, output="sos")
        return sosfiltfilt(sos, ir)

    # Sub-resolvable pred in the 500 Hz octave ONLY: a 12 ms decay there, the
    # healthy decay at 1000 Hz. Built band-by-band rather than by substitution into
    # a broadband IR — leakage from the surrounding content otherwise dominates the
    # 500 Hz fit and the fixture measures nothing (its EDT read 0.0797 s, well above
    # the floor). Constructed this way the 500 Hz EDT saturates at 0.0144 s, i.e.
    # the filter's own response, which is exactly the condition being tested.
    fast = _decaying_noise_ir(rt60=0.012, duration_s=1.0, seed=35)
    fast_500 = _octave(500.0, fast)
    degenerate = (
        fast_500 / max(float(np.abs(fast_500).max()), 1e-30) * float(np.abs(healthy).max())
        + _octave(1000.0, healthy)
    )

    out = {}
    for label, pred in (("healthy", healthy), ("degenerate", degenerate)):
        triples, reasons, _window, acct = compute_room_acoustic_metrics(
            pred[None, :], high[None, :], low[None, :],
            sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
            band_resolvability_margin=margin,
            min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
        )
        out[label] = (triples, reasons, acct)

    # The fixture must bite: pred is genuinely unresolvable in 500 Hz.
    assert out["degenerate"][2]["EDT"]["pred_unresolved_hz"] == [500.0], (
        "fixture did not produce a sub-resolvable pred band, so this proves nothing"
    )

    # THE ASSERTION: bit-identical physical legs.
    for metric in ("T30", "EDT", "C50"):
        for leg in ("high", "low"):
            a = getattr(out["healthy"][0][metric], leg)
            b = getattr(out["degenerate"][0][metric], leg)
            assert (np.isnan(a) and np.isnan(b)) or a == b, (
                f"{metric}.{leg} moved from {a} to {b} when only pred changed — a "
                f"model output is setting the band set used to measure its own "
                f"target (AC-25)."
            )
        # The band set itself is pred-independent.
        assert (out["healthy"][2][metric]["kept_hz"]
                == out["degenerate"][2][metric]["kept_hz"])

    # And pred is UNSCORED rather than silently averaged over a different set.
    assert np.isnan(out["degenerate"][0]["EDT"].pred)
    assert "AC-25" in out["degenerate"][1][("EDT", "pred")]
    assert np.isfinite(out["healthy"][0]["EDT"].pred)


# ---------------------------------------------------------------------------
# AC-36 / AC-40: the octave filter's treatment of an ONSET-ALIGNED impulse
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def _config_iso_freqs() -> list[float]:
    """`configs/base.yaml`'s declared evaluation bands (F-106).

    AC-40's known-answer test was parametrized over the module constant `_ISO`.
    They are equal today, so the test had teeth — but a change to the DECLARED band
    set would not have reached it, which is exactly the config/code coupling AC-40
    exists to prevent. Read at collection time so the parametrize ids show the real
    bands.
    """
    from pathlib import Path

    import yaml

    return [float(f) for f in
            yaml.safe_load(Path("configs/base.yaml").read_text())["iso_eval_freqs"]]


@pytest.mark.parametrize("fc", _config_iso_freqs())
def test_an_onset_impulse_keeps_the_band_energy_an_interior_one_gets(fc: float) -> None:
    """AC-36/AC-40 — the one change that moves every reported ISO metric.

    Two defects met at the first sample of an onset-aligned IR, and neither had a
    test until now:

      * scipy's default `padtype="odd"` REFLECTS the signal about sample 0, so the
        direct arrival became a step of twice the peak that never existed.
        MEASURED at 500 Hz: 6.171 units of in-band energy for 1 unit in, against
        the 0.013228 the band's own bandwidth allows — 420x, and ~23 dB on C50.
      * zero-padding fixed that but then DISCARDED the acausal pre-ringing when the
        guard was stripped, throwing away 50.9 % of the arrival's in-band energy
        (0.006728 vs 0.013228). C50's numerator was not the ISO integral, and the
        deficit is a monotone function of DRR — so it was common-mode across LEGS
        but not across SCENES.

    The known answer is the analytic bandwidth integral, which an impulse far from
    either edge also realizes. A refactor that drops the guard or the fold restores
    one of the two failures silently.
    """
    from amcd.evaluation.room_acoustic import _band_energy

    n = 2 * _SR
    at_onset = np.zeros(n, dtype=np.float32)
    at_onset[0] = 1.0
    interior = np.zeros(n, dtype=np.float32)
    interior[n // 2] = 1.0

    e_onset = float(_band_energy(at_onset, fc, _SR, _ORDER).sum())
    e_interior = float(_band_energy(interior, fc, _SR, _ORDER).sum())

    assert e_onset == pytest.approx(e_interior, rel=1e-3), (
        f"an impulse AT the onset index yields {e_onset:.6f} of in-band energy at "
        f"{fc:g} Hz but an interior one yields {e_interior:.6f} — the guard, the "
        f"energy fold, or both have regressed (AC-36)"
    )
    # The absolute bound catches the padtype="odd" reflection even if both paths
    # regressed together: 6.171 at 500 Hz would sail past a ratio test.
    bandwidth_fraction = (fc * 2 ** 0.5 - fc / 2 ** 0.5) / (_SR / 2)
    assert e_onset < 2.0 * bandwidth_fraction, (
        f"in-band energy {e_onset:.4f} exceeds twice the {fc:g} Hz band's own "
        f"bandwidth fraction ({bandwidth_fraction:.4f}) — a filter with |H| <= 1 "
        f"cannot do that, so the padding is manufacturing energy again"
    )


def test_the_placement_axis_moves_c50_through_the_iso_path() -> None:
    """AC-28 — the scaffold's direct arrival must make C50 live on distance.

    Before AC-28 the "direct sound" was `direct_gain * exp(-t/0.02)`, a one-pole
    envelope with a 7.96 Hz corner, so only 6.06e-7 of its energy reached the
    500 Hz octave. MEASURED CONSEQUENCE: C50 read **1.91 dB at d = 0.5, 1, 2, 4 and
    8 m — identical to three significant figures** — while the closed-form DRR the
    scene report publishes swung +7.55 to -16.53 dB. The declared placement axis
    was inert in every reported ISO-3382 metric, so `test_placement_shift` carried
    no acoustic difference at the metric level.

    THE ROW'S OWN ACCEPTANCE CRITERION IS NOT USED, BECAUSE IT IS NOT TRUE OF C50.
    AC-28 asks that "C50 through the ISO path must fall ~6 dB per doubling of d and
    cross 0 dB near d = r_c". That is the behaviour of DRR, not of C50, and the two
    are different quantities: C50's early window is the first 50 ms, which holds the
    direct arrival PLUS 50 ms of reverberant tail, so C50 exceeds DRR everywhere and
    tends to a tail-only asymptote instead of falling without bound. MEASURED at
    10x8x3.5 m, alpha 0.2 (r_c = 1.193 m):

        d      d/r_c   C50 ISO   DRR closed-form   dC50 per doubling
        0.5 m   0.42   +11.602      +7.551
        1.0 m   0.84    +6.721      +1.530             -4.88 dB
        2.0 m   1.68    +3.495      -4.490             -3.23 dB
        4.0 m   3.35    +2.114     -10.511             -1.38 dB
        8.0 m   6.71    +1.708     -16.531             -0.41 dB

    So C50 never crosses 0 dB and its slope flattens rather than holding 6 dB. That
    is CORRECT physics for C50, and the row's criterion would fail a correct
    implementation. Raised as AC-102.

    What is asserted instead is the property AC-28 actually establishes — the axis
    is LIVE and by a margin that dwarfs the project's own JND.
    """
    from amcd.acoustics import critical_distance
    from amcd.evaluation.room_acoustic import channel_band_avg_metrics
    from amcd.simulators.base import SceneSpec, build_simulator

    from pathlib import Path

    from amcd.config import Config

    cfg = Config.load(
        Path("configs/base.yaml"), Path("configs/overlays/simulator_dry_run.yaml")
    )
    iso = [float(f) for f in cfg.iso_eval_freqs]
    sim = build_simulator(
        cfg.simulator.name, cfg.simulator.params, n_channels=1,
        n_samples=int(cfg.sample_rate * cfg.ir_duration), sample_rate=cfg.sample_rate,
    )
    dims, alpha = (10.0, 8.0, 3.5), 0.2
    lx, ly, lz = dims
    surface = 2.0 * (lx * ly + ly * lz + lx * lz)
    r_c = critical_distance(surface, alpha)

    c50 = []
    for d in (0.5, 1.0, 2.0, 4.0, 8.0):
        scene = SceneSpec(
            scene_id=f"placement-{d}", seed=7, geometry_family="shoebox",
            dims=dims, material_absorption=alpha,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(1.0 + d, 1.0, 1.5),
            sim_params={}, split_regime="test_placement_shift",
            regime_axes={"placement": "near_corner"},
        )
        vals, _ = channel_band_avg_metrics(
            sim.render(scene, cfg.high_ray_budget).ir[0],
            sample_rate=cfg.sample_rate, iso_eval_freqs=iso,
            onset_rel_db=cfg.metric_onset_rel_db,
            band_resolvability_margin=cfg.metric_band_resolvability_margin,
            min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
        )
        c50.append(vals["C50"])

    assert all(b < a for a, b in zip(c50, c50[1:])), (
        f"C50 is not monotone decreasing over a 16x distance range: {c50}. The "
        f"direct arrival is no longer carrying the placement axis into the reported "
        f"ISO metrics (AC-28), and r_c here is {r_c:.3f} m"
    )
    # Measured swing is 9.89 dB; the bound is 5x the JND rather than the measured
    # value so a scaffold tweak has room to move it without a spurious failure,
    # while still separating decisively from the pre-AC-28 0.00 dB.
    swing = c50[0] - c50[-1]
    assert swing > 5.0 * cfg.d0b_c50_jnd_db, (
        f"C50 swings only {swing:.2f} dB over a 16x distance range, against this "
        f"project's own d0b_c50_jnd_db of {cfg.d0b_c50_jnd_db:g} dB (measured 9.89 dB "
        f"when this test was written). The pre-AC-28 scaffold gave 0.00 dB here; a "
        f"small-but-nonzero swing means the direct arrival has been partly "
        f"re-band-limited"
    )

    # THIS is the assertion that actually discriminates AC-28 (F-144). The two above
    # do NOT: reverting the direct arrival to its pre-AC-28 one-pole envelope still
    # gives a monotone C50 with a 23.41 dB swing, because the swing is delivered by
    # the room-constant TAIL SCALING (RD-75), not by the arrival being broadband.
    # Running that revert as a negative control is what exposed it.
    #
    # AC-28's own second consequence is the discriminating one: with a one-pole
    # envelope the global peak sits 300-550 samples INTO the diffuse tail, which
    # violates `_find_onset`'s documented AC-07 assumption that the direct sound is
    # the loudest arrival. A broadband impulse is the first and largest sample by
    # construction.
    scene = SceneSpec(
        scene_id="placement-peak", seed=7, geometry_family="shoebox",
        dims=dims, material_absorption=alpha,
        source_pos=(1.0, 1.0, 1.5), receiver_pos=(3.0, 1.0, 1.5),
        sim_params={}, split_regime="test_placement_shift",
        regime_axes={"placement": "near_corner"},
    )
    ir = sim.render(scene, cfg.high_ray_budget).ir[0]
    onset = int(np.argmax(np.abs(ir) > 0))          # first non-silent sample = d/c
    peak = int(np.argmax(ir.astype(np.float64) ** 2))
    assert peak == onset, (
        f"the loudest sample is at index {peak}, {peak - onset} samples INTO the "
        f"diffuse tail rather than at the direct arrival ({onset}). A one-pole "
        f"envelope puts it there; a broadband impulse cannot. This violates "
        f"`_find_onset`'s AC-07 assumption that the direct sound is the loudest "
        f"arrival, and it is the property that distinguishes AC-28's fix from the "
        f"defect it replaced (the C50 swing above does NOT — it survives the revert)"
    )


def test_the_headroom_guard_reads_exactly_the_reported_metric_bands() -> None:
    """AC-37-R4 / RD-187 — the guard's operand IS `iso_eval_freqs`, not a copy of it.

    It used to be a copy: `min_db_headroom_octave_centres_hz` in
    configs/representations/spectrogram.yaml declared the evaluation band set a
    SECOND time, because `build_representation` took `sample_rate` as its only
    cross-cutting argument and so a representation could not read the master config.
    That is the AC-24 divergence shape, and it was admissible only while a test
    asserted the two were equal.

    The band set is now handed down, so there is nothing left to drift — and this
    test changes accordingly: instead of comparing two declarations, it asserts the
    guard's operand tracks `iso_eval_freqs` when that MOVES. A representation that
    reintroduced its own copy would pass an equality check against the shipped
    config and fail this one.

    EQUALITY, NOT CONTAINMENT, still. The defect being fixed is a minimum taken over
    too wide a band set — the old guard minimised over all 27 ladder bands and was
    decided by a low single-FFT-bin band, 4.91 dB away from the bands the metrics
    actually use. Containment would permit arbitrary over-coverage.
    """
    from pathlib import Path

    import yaml

    from amcd.representations.base import build_representation

    cfg = _base_config()
    params = yaml.safe_load(
        Path("configs/representations/spectrogram.yaml").read_text()
    )
    assert "min_db_headroom_octave_centres_hz" not in params, (
        "the representation config declares the evaluation band set again. It is "
        "handed down by `build_representation` now; a second declaration is the "
        "AC-24 shape and nothing keeps the two in step any more (RD-187)."
    )

    shipped = [float(f) for f in cfg.iso_eval_freqs]
    rep = build_representation(
        cfg.representation.name, cfg.representation.params,
        sample_rate=cfg.sample_rate, eval_freqs_hz=shipped,
    )
    assert sorted(rep.eval_freqs_hz) == sorted(shipped)

    # And it MOVES with the band set — which a re-introduced literal would not.
    moved = [250.0, 2000.0]
    rep_moved = build_representation(
        cfg.representation.name, cfg.representation.params,
        sample_rate=cfg.sample_rate, eval_freqs_hz=moved,
    )
    assert sorted(rep_moved.eval_freqs_hz) == sorted(moved)
    assert rep_moved.headroom_band_indices != rep.headroom_band_indices, (
        "the guard selects the same ladder bands for two different evaluation band "
        "sets, so its operand is not actually derived from them — the AC-37-R4 "
        "defect, where the guard is calibrated on one band set and enforced on "
        "another"
    )


def test_the_headroom_guard_names_the_offending_channel_not_just_a_band() -> None:
    """AC-69 — a 4-channel field where ONLY the W channel is on the floor.

    THE PROPERTY HAD ZERO REGRESSION COVERAGE. Every existing test encodes a
    single-channel IR, where `amax(dim=2)` and the pre-fix `amax(dim=(0,2))` are
    identical — the acoustics-reviewer monkeypatched the guard back to the old
    channel-maxing semantics and the whole suite still passed.

    Why the channel matters: the reported ISO-3382 path reads `ir[0]`, the W
    channel, exclusively (`compute_room_acoustic_metrics`). A W channel sitting on
    the absolute `min_db` floor while higher-order channels stay loud corrupts every
    reported metric, and a channel-max accepts it. Not hypothetical under the real
    backend — `simulators/base.py` declares `acn_n3d`, and N3D scales degree l by
    sqrt(2l+1), so higher-order channels are systematically LOUDER.
    """
    cfg, rep, sim, scene = _ac37_setup()

    mono = sim.render(scene, cfg.high_ray_budget).ir           # (1, T)
    field = np.repeat(mono, 4, axis=0).copy()                  # (4, T), all native
    field[0] *= 10.0 ** (-70.0 / 20.0)                         # W alone, 70 dB down

    with pytest.raises(ValueError, match="min_db headroom guard") as exc:
        rep.encode(field.astype(np.float32))

    msg = str(exc.value)
    assert "channel 0" in msg, (
        f"the guard fired but did not name channel 0 — with channels 1-3 at their "
        f"native level, a channel-MAX would not have fired at all, so the message "
        f"naming the channel is what distinguishes the two semantics (AC-69). "
        f"Got: {msg[:200]}"
    )


def test_the_headroom_guard_ignores_a_spectral_slope_outside_the_metric_bands() -> None:
    """F-M3 — a lowpassed IR must NOT be rejected with a message blaming level.

    The old guard minimised across ALL bands, making it a spectral-flatness test:
    the dry-run tail is white so nothing tripped it, but a 2nd-order 4 kHz lowpass
    — far gentler than air absorption over a 4.25 s IR — dropped the top band and
    `encode` raised, telling the operator to "fix the level (source_power /
    normalize_ir ...)" when the level was fine. Any spectrally sloped render (air
    absorption, frequency-dependent alpha — both roadmap items) would trip it.

    With the operand restricted to the reported ISO span, a slope OUTSIDE that span
    is no longer the guard's business.
    """
    from scipy.signal import butter, sosfilt

    cfg, rep, sim, scene = _ac37_setup()
    ir = sim.render(scene, cfg.high_ray_budget).ir

    sos = butter(2, 4000.0, btype="lowpass", fs=cfg.sample_rate, output="sos")
    sloped = sosfilt(sos, ir.astype(np.float64), axis=-1).astype(np.float32)

    # PIN THE STIMULUS (F-141). Without this the test is vacuous the moment the
    # cutoff or the render level moves: it would then assert only that a healthy IR
    # encodes, which every other test already covers. The slope must be steep enough
    # that the OLD all-band operand would have rejected it.
    import torch

    bands = []
    for c in range(sloped.shape[0]):
        st = torch.stft(
            torch.from_numpy(sloped[c]), n_fft=rep.n_fft, hop_length=rep.hop_length,
            window=rep._window, return_complex=True, center=True,
        )
        bands.append(torch.einsum("bf,fn->bn", rep._filter_bank, st.abs().pow(2)))
    peak_db = torch.amax(10.0 * torch.log10(torch.stack(bands).clamp(min=1e-10)), dim=2)
    headroom = peak_db - rep.min_db
    all_band_min = float(headroom.min())
    iso_min = float(headroom[:, rep.headroom_band_indices].min())

    assert all_band_min < rep.min_db_headroom_db, (
        f"the lowpass no longer drives ANY ladder band below the guard threshold "
        f"({all_band_min:.2f} dB vs {rep.min_db_headroom_db:g}), so this test would "
        f"pass even with the old all-band operand and proves nothing about F-M3"
    )
    assert iso_min >= rep.min_db_headroom_db, (
        f"the lowpass drove a REPORTED-band down to {iso_min:.2f} dB; the stimulus "
        f"is no longer a slope outside the metric bands"
    )

    rep.encode(sloped)  # must NOT raise


@pytest.mark.parametrize("octaves", [1, 2])
@pytest.mark.parametrize("fc", _ISO)
def test_the_octave_filter_meets_its_declared_stopband_rejection(
    fc: float, octaves: int
) -> None:
    """AC-68 — a tone one and two octaves out of band must land below the declared
    rejection, on BOTH sides of the band.

    The module calls itself "the standard ISO-3382 path", and ISO 3382-1 specifies
    IEC 61260 class 1 octave filters. This one is not class 1: measured rejection
    is -37.43 / -38.49 dB one octave out at 500 Hz and -46.59 / -47.33 dB two
    octaves out, where class 1 wants 60 dB+. That gap is DECLARED in
    `_butter_octave_filter` rather than closed, because the filter order also sets
    the ringing `_band_resolvable_decay_s` measures — a steeper filter buys
    selectivity with a longer unresolvable floor, which is a research trade.

    What this test is for: the declaration must not rot the way the resolvability
    floors did (AC-65). It is benign today — one scalar absorption across all
    simulated bands means leakage carries no wrong decay — and becomes live under
    AC-63's per-band absorption, where a loud band's decay could leak into a quiet
    band's T30 at only ~38 dB down.
    """
    from amcd.evaluation.room_acoustic import _band_energy

    n = _SR
    t = np.arange(n) / _SR
    bound = _STOPBAND_DB[octaves]

    for direction, f_tone in (("below", fc / 2 ** octaves), ("above", fc * 2 ** octaves)):
        if f_tone >= _SR / 2 * 0.9 or f_tone < 20.0:
            continue
        tone = np.sin(2.0 * np.pi * f_tone * t).astype(np.float32)
        total = float((tone.astype(np.float64) ** 2).sum())
        in_band = float(_band_energy(tone, fc, _SR, _ORDER).astype(np.float64).sum())
        rejection_db = 10.0 * np.log10(in_band / total)

        assert rejection_db <= bound, (
            f"a {f_tone:g} Hz tone ({octaves} octave(s) {direction} the {fc:g} Hz "
            f"band) leaks {rejection_db:.2f} dB into it, against the declared "
            f"{bound:g} dB. The octave filter's selectivity has changed — update "
            f"`_butter_octave_filter`'s measured table AND "
            f"`metric_octave_filter.stopband_rejection_db` in configs/base.yaml "
            f"together, and re-check the AC-63 per-band-absorption case (AC-68)."
        )


def test_the_octave_filter_edges_are_minus_six_db_and_bands_are_NOT_complementary(
) -> None:
    """AC-68 / AC-104 — the band edges, and the property they do NOT give.

    `sosfiltfilt` applies the section forwards and backwards, so |H|^2 is squared
    and the nominal -3 dB band edges present as -6 dB. That much is correct.

    IT DOES NOT FOLLOW that adjacent bands are power-complementary, and an earlier
    version of this test asserted that in its NAME and docstring while measuring
    only the edges — a false property pinned by a test that could never have caught
    it (AC-104). The squaring is exactly what breaks complementarity: at every
    crossover the single-pass bank sums |H|^2 = 1.00000, while the shipped
    zero-phase bank sums |H|^4 = 0.50000, i.e. -3.010 dB.

    Both halves are asserted here so neither can be quietly restored: the edges must
    stay at -6 dB, AND the crossover sum must stay at 0.5 rather than drifting toward
    1.0, which would mean the zero-phase convention had changed. Nil consequence
    today because bands are AVERAGED and never summed; live under AC-63's per-band
    absorption.
    """
    from scipy.signal import butter, sosfreqz

    from amcd.evaluation.room_acoustic import _band_energy

    n = _SR
    t = np.arange(n) / _SR
    for fc in _ISO:
        for edge in (fc / 2 ** 0.5, fc * 2 ** 0.5):
            tone = np.sin(2.0 * np.pi * edge * t).astype(np.float32)
            total = float((tone.astype(np.float64) ** 2).sum())
            in_band = float(_band_energy(tone, fc, _SR, _ORDER).astype(np.float64).sum())
            db = 10.0 * np.log10(in_band / total)
            assert db == pytest.approx(-6.0, abs=0.3), (
                f"the {fc:g} Hz band's {edge:.1f} Hz edge reads {db:.2f} dB, not "
                f"-6 dB. sosfiltfilt squares |H|^2, so -3 dB edges must present as "
                f"-6 dB; a departure means the band edges or the zero-phase "
                f"convention changed (AC-68)"
            )

    # The crossover between the two eval octaves: 500*sqrt2 == 1000/sqrt2.
    crossover = _ISO[0] * 2 ** 0.5
    total_pow4 = 0.0
    for fc in _ISO:
        sos = butter(4, [fc / 2 ** 0.5, fc * 2 ** 0.5], btype="bandpass",
                     fs=_SR, output="sos")
        _w, h = sosfreqz(sos, worN=[crossover], fs=_SR)
        total_pow4 += float(abs(h[0]) ** 2) ** 2

    assert total_pow4 == pytest.approx(0.5, abs=0.02), (
        f"adjacent octave bands sum to {total_pow4:.5f} of the input power at their "
        f"{crossover:.1f} Hz crossover. The zero-phase bank squares |H|^2, so the "
        f"correct value is 0.5 (-3.010 dB) and the bank is NOT power-complementary; "
        f"a value near 1.0 would mean the filtering became single-pass, which would "
        f"reintroduce a group delay into EDT (AC-104)"
    )


def test_a_leg_that_both_excludes_a_band_and_is_floor_limited_keeps_both_reasons(
    monkeypatch,
) -> None:
    """F-M9 — the AC-38 disclosure must not overwrite a band-EXCLUSION reason.

    `compute_room_acoustic_metrics` writes `nan_reasons[(metric, leg)]` twice for
    the same key: once when that leg's NaN band is excluded from every leg's
    average, and again when that leg is floor-limited in a band that WAS kept.
    `evaluator.py` forwards exactly one reason per (metric, leg) to `drops.csv`, so
    an unconditional second assignment silently deletes the exclusion — the harder
    of the two facts, and the one that explains a changed band average.

    The input is CONSTRUCTED rather than rendered. F-M9 records that no instance is
    reachable inside base.yaml's declared support — a Schroeder EDR is monotone, so
    "non-decaying EDR" is near-unreachable, and Lundeby's floor makes "<2 samples"
    unreachable — so the collision is injected at the per-band layer, which is
    precisely where the aggregation under test consumes it. That keeps the guard
    honest about being a guard on an unexercised path rather than a fix for a live
    one.
    """
    import amcd.evaluation.room_acoustic as ra

    nan = float("nan")

    def fake_per_band(ir_w, **kw):
        leg = getattr(ir_w, "_leg", "?")
        if leg == "high":
            return [
                # band 0 — NaN for a HARD reason, so the band is excluded for all legs
                ({"T30": nan, "EDT": nan, "C50": nan},
                 {"T30": "non-decaying EDR (slope >= 0) in the [-5, -35] dB window"},
                 {}),
                # band 1 — finite but BELOW the floor, so AC-38 discloses it
                ({"T30": 0.011, "EDT": 0.006, "C50": 3.0}, {},
                 {"T30": "T30 0.0110 s is below the 0.0407 s the 500 Hz octave band can resolve"}),
            ]
        return [
            ({"T30": 0.30, "EDT": 0.28, "C50": 2.0}, {}, {}),
            ({"T30": 0.30, "EDT": 0.28, "C50": 2.0}, {}, {}),
        ]

    monkeypatch.setattr(ra, "channel_per_band_metrics", fake_per_band)
    monkeypatch.setattr(
        ra, "_shared_truncation_per_band", lambda *a, **k: [(48000, "high"), (48000, "high")]
    )

    class Tagged(np.ndarray):
        """Carries the leg identity through `ir[0]`, which is how the function
        under test hands each leg to `channel_per_band_metrics`."""

        def __array_finalize__(self, obj):
            self._leg = getattr(obj, "_leg", "?")

    def tag(leg):
        arr = np.zeros((1, 96000), dtype=np.float32).view(Tagged)
        arr._leg = leg
        return arr

    _triples, reasons, _window, _acct = ra.compute_room_acoustic_metrics(
        tag("pred"), tag("high"), tag("low"),
        sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=2.0,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )

    high_t30 = reasons[("T30", "high")]
    assert "non-decaying EDR" in high_t30, (
        "the band-EXCLUSION reason was overwritten by the AC-38 resolvability "
        f"disclosure — drops.csv would carry only the caveat, not the cause of the "
        f"changed band average (F-M9). Got: {high_t30!r}"
    )
    assert "resolvability-limited but REPORTED" in high_t30, (
        f"the AC-38 disclosure was lost instead. Got: {high_t30!r}"
    )


#: The resolvability floors `_band_resolvable_decay_s` measures at 48 kHz, in
#: SECONDS, exactly as its docstring declares them (RR-39 names that docstring as
#: the one place they are written down; this pins it). AC-65: the previous four
#: values drifted through the AC-36/F-67 energy fold unnoticed because the suite
#: called the function without ever asserting its result.
_DECLARED_FLOORS_48K = {
    500.0: {"T30": 0.020360, "EDT": 0.009556},
    1000.0: {"T30": 0.010162, "EDT": 0.004802},
}


@pytest.mark.parametrize("fc", _ISO)
def test_the_band_resolvability_floors_are_the_declared_values(fc: float) -> None:
    """AC-65 — the floors must equal what `_band_resolvable_decay_s`'s docstring says.

    Those four numbers are not decoration. Multiplied by
    `metric_band_resolvability_margin` they decide which bands carry the AC-38
    resolvability caveat into `metrics.parquet`, and the margin's own calibration
    is stated against them. When the AC-36 energy fold moved the filter's ringing,
    the declared values did not follow and nothing caught it: the docstring claimed
    500 Hz -> T30 17.881 ms / EDT 11.765 ms and 1000 Hz -> 8.924 / 5.771 ms, while
    the function returned 20.360 / 9.556 and 10.162 / 4.802 (T30 +13.9 %, EDT
    -18.8 % / -16.8 %).

    Corroborated independently from inside the ledger: AC-27's resolution quotes
    f*T30 = 9.85-10.18 across 125-4000 Hz, which reproduces exactly here
    (measured 9.88-10.18), so the ledger and the docstring had already disagreed.

    Tolerance is 0.5 %, tight enough to catch that 13.9 % drift and loose enough to
    survive a scipy point release.
    """
    from amcd.evaluation.room_acoustic import _band_resolvable_decay_s

    # The constant's name promises 48 kHz; nothing else enforces it. Without this,
    # a change to `_SR` would silently compare 48 kHz floors against another rate
    # and the name would lie (RR-122).
    assert _SR == 48000, (
        f"_DECLARED_FLOORS_48K holds floors measured at 48 kHz but _SR is {_SR}. "
        f"The floors scale as 1/f and are sample-rate dependent — re-measure and "
        f"rename the constant, or key it by sample rate."
    )
    measured = _band_resolvable_decay_s(fc, _SR, _ORDER)
    declared = _DECLARED_FLOORS_48K[fc]
    for metric, want in declared.items():
        assert measured[metric] == pytest.approx(want, rel=5e-3), (
            f"the {fc:g} Hz {metric} resolvability floor measures "
            f"{measured[metric] * 1000:.3f} ms against the "
            f"{want * 1000:.3f} ms declared in `_band_resolvable_decay_s`'s "
            f"docstring — the filter path has moved and the ONE place these values "
            f"are written down did not follow (AC-65). Update BOTH, and re-state "
            f"the margin calibration in configs/base.yaml, which is quoted against "
            f"them."
        )


def test_the_resolvability_floors_scale_as_one_over_f() -> None:
    """AC-65 companion: the floors are the FILTER's own decay, so f * T30 is a
    constant of the filter design rather than of any band. Holding it pins the
    1/f scaling the docstring claims, which a per-band literal table would not."""
    from amcd.evaluation.room_acoustic import _band_resolvable_decay_s

    products = [
        fc * _band_resolvable_decay_s(fc, _SR, _ORDER)["T30"]
        for fc in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
    ]
    assert max(products) - min(products) < 0.4, (
        f"f * T30 spans {min(products):.2f}-{max(products):.2f} across 125-4000 Hz; "
        f"the floors no longer scale as 1/f, so the docstring's claim that they do "
        f"is false (AC-65)"
    )


@pytest.mark.parametrize("fc", _ISO)
@pytest.mark.parametrize(
    "n_record",
    # Spans the short-record branch and the normal one. 32 is _MIN_FILTER_SAMPLES,
    # the shortest record `_iso3382_band_metrics` admits; 4608 is the 500 Hz guard.
    [32, 40, 64, 128, 512, 2304, 4608, 9216],
)
def test_the_energy_fold_conserves_energy_at_every_record_length(
    n_record: int, fc: float
) -> None:
    """F-68-R3 — the fold must be exactly energy-conserving, at ANY record length.

    `_band_energy` filters a zero-padded record and folds the acausal ringing in
    the guard back onto its mirrored support. The property that makes that
    legitimate rather than arbitrary is CONSERVATION: the energy inside the
    returned record must equal the energy of the full filtered signal, guard
    included. Two defects broke it, in opposite directions:

      * the outermost pad sample was folded TWICE (the reversed slice already
        covered it), leaving folded/full = 1.000000001490 at 500 Hz and
        1.000000004612 at 1000 Hz rather than exactly 1.0. Physically ~1.7e-28 and
        irrelevant on its own;
      * the same two lines were the ONLY handling of the short-record branch, and
        there the fold mirrored just `min(guard, n_record - 1)` samples per end and
        DROPPED the rest. MEASURED at 500 Hz, folded/full by record length:

            32 samples -> 0.7029    64 -> 0.9545    512 -> 0.99999
            40 samples -> 0.8558   128 -> 0.9872

        i.e. a record admitted at the declared 32-sample minimum silently lost
        29.7 % of its band energy. A very-late onset trim (AC-07) is exactly what
        produces such a record, and nothing logged the loss.

    Both are fixed by folding every pad sample with the mirror index CLAMPED into
    the record. This test is parametrized across the branch boundary on purpose: at
    n_record > guard no clamping happens, below it every sample does.

    TOLERANCE. The fold is exact: carried end to end in float64 it gives
    folded/full = 1.000000000000000 at both bands and every length above. But
    `_band_energy` RETURNS float32, so summing its output costs ~1e-8 of relative
    cast noise, varying in sign. 1e-6 is therefore the tightest bound this test can
    assert through the public return, and it is four orders of magnitude tighter
    than the 0.70 the short-record defect produced.
    """
    from amcd.evaluation.room_acoustic import _band_energy, _butter_octave_filter

    ir = np.zeros(n_record, dtype=np.float32)
    ir[n_record // 2] = 1.0

    folded = float(_band_energy(ir, fc, _SR, _ORDER).astype(np.float64).sum())
    filtered, _guard = _butter_octave_filter(ir, fc, _SR, _ORDER)
    full = float((filtered.astype(np.float64) ** 2).sum())

    assert folded == pytest.approx(full, rel=1e-6), (
        f"the fold is not energy-conserving at {fc:g} Hz with n_record={n_record}: "
        f"folded/full = {folded / full:.9f}. Either a pad sample is being folded "
        f"twice, or — if the ratio is below 1 — pad samples beyond one record "
        f"length are being discarded again (F-68-R3)"
    )


@pytest.mark.parametrize("fc", _ISO)
def test_a_record_shorter_than_one_guard_width_is_unmeasurable_not_approximated(
    fc: float,
) -> None:
    """AC-100 / AC-106 — below one guard width, NaN with a reason, never a number.

    `_band_energy` folds the filter's acausal ringing onto its mirrored support.
    Below one guard width that mirror falls outside the record, so the energy is
    clamped to the record edge — conserving it, but depositing it at the far end
    from the arrival it belongs to. MEASURED at n_record = 32: the last sample holds
    30.24 % of the band's energy, 24x its neighbour, and T30 reads 0.00706 s against
    0.00336 s for the same signal in a long record. The predecessor behaviour was
    worse — it silently DISCARDED up to 29.7 %.

    Neither is a number this project may report, so the third option is taken: the
    metric is unmeasurable and says so, which is what the drop log exists for.

    The fold itself is still exercised and still conserves energy at those lengths
    (`test_the_energy_fold_conserves_energy_at_every_record_length`) — this bounds
    what may be REPORTED, not what `_band_energy` computes.
    """
    from amcd.evaluation.room_acoustic import (
        _filter_guard_samples, _iso3382_band_metrics,
    )

    guard = _filter_guard_samples(fc, _SR, _ORDER)

    short = np.zeros(guard - 1, dtype=np.float32)
    short[len(short) // 2] = 1.0
    values, reasons, _res = _iso3382_band_metrics(
        short, fc, _SR, band_resolvability_margin=2.0
    , min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER)
    for metric in ("T30", "EDT", "C50"):
        assert np.isnan(values[metric]), (
            f"{metric} returned {values[metric]} for a {len(short)}-sample record at "
            f"{fc:g} Hz, below the {guard}-sample guard width — the clamped fold "
            f"deposits energy at the record edge there, so this is an approximation "
            f"reported as a measurement (AC-100/AC-106)"
        )
        assert "guard" in reasons[metric], (
            f"{metric} is NaN but its reason does not name the guard width: "
            f"{reasons[metric]!r}. Nothing may leave a result without a reason."
        )

    # ...and one sample above the bound it is measurable again, so the guard is a
    # boundary and not a blanket refusal of short records.
    ok = np.zeros(max(guard, _MIN_FILTER_SAMPLES_FOR_TEST), dtype=np.float32)
    ok[len(ok) // 2] = 1.0
    values_ok, _r, _res2 = _iso3382_band_metrics(
        ok, fc, _SR, band_resolvability_margin=2.0
    , min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER)
    assert not np.isnan(values_ok["T30"]), (
        f"a {len(ok)}-sample record at {fc:g} Hz is at or above the {guard}-sample "
        f"guard width and must still be measurable"
    )


#: Mirrors `_MIN_FILTER_SAMPLES`; the guard-width bound is `max()` of the two.
_MIN_FILTER_SAMPLES_FOR_TEST = 32


def test_a_record_ending_at_full_scale_is_not_given_a_dc_tail() -> None:
    """AC-36 secondary: `padtype="constant"` replicates the EDGE SAMPLE, so a
    record whose last sample is non-zero got a DC tail — measured -25.5 % band
    energy. Padding with explicit zeros at BOTH ends makes "constant" replicate
    0.0, which is what an impulse response's surroundings actually are."""
    from amcd.evaluation.room_acoustic import _band_energy

    n = _SR // 2
    ends_high = np.zeros(n, dtype=np.float32)
    ends_high[0] = 1.0
    ends_high[-1] = 1.0
    padded_tail = np.concatenate([ends_high, np.zeros(_SR // 4, dtype=np.float32)])

    e_abrupt = float(_band_energy(ends_high, 500.0, _SR, _ORDER).sum())
    e_padded = float(_band_energy(padded_tail, 500.0, _SR, _ORDER).sum())
    assert e_abrupt == pytest.approx(e_padded, rel=1e-3), (
        f"truncating the record changed its band energy ({e_abrupt:.6f} vs "
        f"{e_padded:.6f}) — the trailing pad is replicating the last sample again"
    )


# ---------------------------------------------------------------------------
# AC-37: `min_db` is an ABSOLUTE floor, so it can be reached by LEVEL alone
# ---------------------------------------------------------------------------
#
# Everything below runs at PRODUCTION framing, loaded from configs/base.yaml.
# That is not incidental: AC-37 is a property of a particular (sample_rate,
# n_fft, min_db) triple, and the tiny overlay's 8 kHz / n_fft 256 framing builds
# a different filter ladder with a different floor-to-signal relationship. Every
# threshold below is pulled from the same config rather than written here
# (CLAUDE.md: no experiment-governing values in test fixtures).


def _base_config():
    """`configs/base.yaml` — the declared production values, not literals here.

    The module constants above (`_SR`, `_ISO`, `_ONSET_DB`) predate this and match
    base.yaml today; tests added since pull from config so a change to the declared
    band set or margin reaches them instead of silently disagreeing.
    """
    from pathlib import Path

    from amcd.config import Config

    return Config.load(Path("configs/base.yaml"))


def _ac37_setup():
    """Production framing + the scaffold backend, both from config.

    `base.yaml` alone declares `gsound_sir`, which cannot render here; the
    `simulator_dry_run` overlay swaps ONLY the backend, leaving 48 kHz / n_fft
    2048 / `min_db` -80 intact. The scaffold is reached through
    `build_simulator`, never constructed directly, so this test sits on the seam
    rather than on the scaffold.

    The backend matters because AC-37 is a defect of ABSOLUTE LEVEL. A
    unit-variance synthetic IR carries ~70 dB more headroom above `min_db` than
    the level convention a render actually produces (`direct = 1/d` against a
    room-constant tail), and at that headroom the defect is invisible — measured,
    the oracle's T30 error stays <= 1.6 % at every gain down to -60 dB. Building
    this probe out of normalized noise would have produced a green test over a
    live defect.
    """
    from pathlib import Path

    from amcd.config import Config
    from amcd.representations.base import build_representation
    from amcd.simulators.base import SceneSpec, build_simulator

    cfg = Config.load(
        Path("configs/base.yaml"), Path("configs/overlays/simulator_dry_run.yaml")
    )
    rep = build_representation(
        cfg.representation.name, cfg.representation.params,
        sample_rate=cfg.sample_rate, eval_freqs_hz=EVAL_FREQS,
    )
    sim = build_simulator(
        cfg.simulator.name, cfg.simulator.params,
        n_channels=1, n_samples=int(cfg.sample_rate * cfg.ir_duration),
        sample_rate=cfg.sample_rate,
    )
    scene = SceneSpec(
        scene_id="ac37", seed=7, geometry_family="shoebox",
        dims=(10.0, 8.0, 3.5), material_absorption=0.16,
        source_pos=(2.0, 2.0, 1.5), receiver_pos=(5.0, 5.0, 1.5),
        sim_params={}, split_regime="id", regime_axes={},
    )
    return cfg, rep, sim, scene


def _oracle_t30_error_frac(cfg, rep, sim, scene, gain_db: float):
    """T30 of a DEFINITIONALLY PERFECT prediction against T30 of its own target.

    The oracle is D0b's: `decode(encode(high), low)` imposes high's TRUE band
    energies onto the low-ray carrier, so a decode that preserved them would
    reproduce high's T30 exactly, at any level.

    Measured through `compute_room_acoustic_metrics`, i.e. the REPORTED path, not
    the standalone one. That is load-bearing: the Schroeder window is shared and
    set by the PHYSICAL legs (AC-17/RD-43), so `pred` never gets its own Lundeby
    cut. Given its own cut the oracle truncates the injected floor away and the
    defect hides — measured, the same scene reads 0.02 % standalone against
    126.8 % through the shared window at -30 dB.

    Returns (t30_high, t30_pred, relative error, worst-band headroom above min_db).
    """
    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    g = 10.0 ** (gain_db / 20.0)
    high = (sim.render(scene, cfg.high_ray_budget).ir * g).astype(np.float32)
    low = (sim.render(scene, cfg.low_ray_budget).ir * g).astype(np.float32)

    env = rep.encode(high)
    # Per (channel, band), and only over the bands the guard reads — the SAME
    # operand `_check_min_db_headroom` applies. This line previously used
    # `dim=(0, 2)`, maxing over CHANNELS, which is the semantics AC-37-R5 removed
    # from the guard; leaving it here put the wrong definition next to the right one
    # as a template for the next test to copy (AC-69).
    peak = torch.amax(env, dim=2) - rep.min_db          # (C, n_bands)
    headroom = float(peak[:, rep.headroom_band_indices].min())
    oracle = rep.decode(env, low)

    triples, _reasons, _window, _acct = compute_room_acoustic_metrics(
        oracle, high, low,
        sample_rate=cfg.sample_rate,
        iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    t = triples["T30"]
    return t.high, t.pred, abs(t.pred - t.high) / t.high, headroom


@pytest.mark.parametrize("gain_db", [0.0, -20.0])
def test_the_decoded_oracle_reproduces_its_targets_t30_at_any_level(gain_db: float) -> None:
    """AC-37 — the known-answer test, written BEFORE the remedy because it measures
    the defect rather than the fix.

    `min_db` is an ABSOLUTE dB floor on the encoded band energy (AC-33), not a
    level below the per-scene peak. `decode` then rescales the carrier's band
    power to `10**(env/10)`, so wherever the true power sits below the floor the
    decode BOOSTS the carrier UP to it — injecting a non-decaying energy floor
    into the prediction, inside the shared Schroeder window the ISO metrics are
    integrated over.

    Scaling the SAME scene by a gain is therefore not a no-op: it slides the
    scene down onto a fixed floor. The oracle is definitionally perfect, so any
    error here belongs to the representation — the model, carrier, filter and
    Schroeder path are all bypassed. AC-37 measured +0.00 % at 0 dB, +1.48 % at
    -25, +24.4 % at -30, +411 % at -35 and +935 % at -40, with a `min_db: -200`
    control clean to <= 0.38 % everywhere, which is what isolates it to `min_db`.

    The tolerance is this project's own T30 JND, not a number chosen here.
    INERT at a scene's NATIVE level (worst |dT30| 0.25 % over every declared
    corner), so without this test the defect first appears on the emulated gsound
    render RD-33a gates — the most expensive place to find it, and looking like a
    model failure rather than a representation constant.
    """
    cfg, rep, sim, scene = _ac37_setup()
    t30_high, t30_pred, err, headroom = _oracle_t30_error_frac(
        cfg, rep, sim, scene, gain_db
    )

    assert err <= cfg.d0b_t30_jnd_frac, (
        f"at {gain_db:g} dB of level the decoded oracle reads T30 {t30_pred:.4f} s "
        f"against its target's {t30_high:.4f} s — {err * 100:.1f} % error, past this "
        f"project's own d0b_t30_jnd_frac of {cfg.d0b_t30_jnd_frac * 100:g} %. The "
        f"prediction is definitionally perfect, so this is min_db "
        f"({rep.min_db:g} dB, ABSOLUTE) injecting an energy floor into the decode. "
        f"Worst-band headroom above the floor here is {headroom:.1f} dB (AC-37)."
    )


@pytest.mark.parametrize("gain_db", [-30.0, -40.0])
def test_a_scene_that_would_breach_the_t30_jnd_is_refused_by_encode(gain_db: float) -> None:
    """AC-37, the other half: below the declared headroom, `encode` REFUSES.

    These are the two gains the test above cannot assert a T30 for, because the
    remedy the user chose is a GUARD, not a correction of the decode (RD-86 (a);
    (b), a per-scene-relative decode floor, was rejected because it changes the
    decode contract). Measured on this scene BEFORE the guard existed, through
    the reported path:

        gain      headroom    T30 high    T30 pred     error
         0 dB      72.6 dB     0.9681 s    0.9682 s     0.01 %
       -20 dB      52.6 dB     0.9681 s    0.9804 s     1.27 %
       -30 dB      42.6 dB     0.9681 s    2.1959 s   126.82 %
       -40 dB      32.6 dB     0.9681 s   10.9364 s  1029.65 %

    So the two halves together pin the whole contract: where encoding is
    permitted the oracle's T30 is within `d0b_t30_jnd_frac`, and where it is not
    permitted the refusal is loud. Neither half alone would catch a regression
    that widened the guard until the breaching levels were admitted again.
    """
    cfg, rep, sim, scene = _ac37_setup()
    g = 10.0 ** (gain_db / 20.0)
    high = (sim.render(scene, cfg.high_ray_budget).ir * g).astype(np.float32)

    with pytest.raises(ValueError, match="min_db headroom guard") as exc:
        rep.encode(high)
    # The message has to name the operand, or an operator cannot act on it.
    assert "min_db_headroom_db" in str(exc.value)
    assert "Hz band peaks at" in str(exc.value)


# ---------------------------------------------------------------------------
# AC-38: the resolvability floor DISCLOSES, it no longer censors
# ---------------------------------------------------------------------------


def _fitted_t30(t60: float, fc: float, seed: int, duration_s: float = 0.5) -> float:
    """One realization's fitted T30 through the real ISO path, floor NOT applied."""
    from amcd.evaluation.room_acoustic import (
        _band_energy,
        _decay_times_from_energy,
        _lundeby_truncate,
    )

    rng = np.random.default_rng(seed)
    n = int(_SR * duration_s)
    t = np.arange(n) / _SR
    ir = (rng.standard_normal(n) * 10.0 ** (-3.0 * t / t60)).astype(np.float32)
    energy = _band_energy(ir, fc, _SR, _ORDER)
    energy = energy[: _lundeby_truncate(energy, _SR)]
    if len(energy) < 2:
        return float("nan")
    return _decay_times_from_energy(energy, _SR, min_decay_range_db=_NO_SNR_BOUND)[0]


@pytest.mark.parametrize("true_t60", [0.02, 0.03, 0.04])
def test_disclosing_the_floor_tracks_the_true_t60_better_than_suppressing(
    true_t60: float,
) -> None:
    """AC-38: censoring an estimator on its OWN value biases the surviving mean up.

    The floor's threshold is a per-(metric, band) constant and independent of the
    datum — AC-26 got that right, and the margin of 2.0 is separately calibrated.
    What stayed wrong was the DECISION, `fitted_value < floor`: it removes exactly
    the low realizations, so whatever survives reads high.

    Measured at 200 realizations/point through this same path (the suite runs a
    smaller sweep for time; these are the full figures):

        true T60   suppressed   disclose err   suppress err
          0.02 s    279/400        +8.3 %        +13.4 %
          0.03 s    194/400        +5.4 %         +7.0 %
          0.04 s    100/400        +3.1 %         +7.1 %
          0.05 s     12/400        +2.4 %         +3.3 %
          0.06 s      0/400        +2.6 %         +2.6 %

    Two properties, and the second matters as much as the first: disclosing is
    strictly closer to truth wherever the floor bites, AND the two coincide exactly
    where it does not — so this is not a change that quietly moves every number.

    The residual +2 to +8 % is the ESTIMATOR's own bias at these decays, which no
    threshold can remove; it is disclosed via `metric_edt_variance_limited_s`
    (RD-78), not suppressed.
    """
    from amcd.evaluation.room_acoustic import _band_resolvable_decay_s

    cfg = _base_config()
    margin = cfg.metric_band_resolvability_margin
    fitted, survivors = [], []
    for i in range(30):
        for fc in [float(f) for f in cfg.iso_eval_freqs]:
            t30 = _fitted_t30(true_t60, fc, seed=1000 * i + int(fc))
            if np.isnan(t30):
                continue
            fitted.append(t30)
            if t30 >= margin * _band_resolvable_decay_s(fc, _SR, _ORDER)["T30"]:
                survivors.append(t30)

    assert len(fitted) > len(survivors), (
        f"the floor suppressed nothing at true T60 = {true_t60} s, so this probe "
        f"asserts nothing — pick a T60 nearer the floor"
    )
    disclose_err = abs(float(np.mean(fitted)) / true_t60 - 1.0)
    suppress_err = abs(float(np.mean(survivors)) / true_t60 - 1.0)
    assert disclose_err < suppress_err, (
        f"at true T60 = {true_t60} s the disclosed mean is {disclose_err * 100:.1f} % "
        f"from truth and the suppressed mean {suppress_err * 100:.1f} % — suppressing "
        f"is supposed to be the WORSE estimator; if this flips, the AC-38 argument "
        f"does not hold at this decay and the row must be re-opened, not the test "
        f"re-tuned"
    )


def test_a_floor_limited_band_does_not_cost_a_scene_from_the_paired_comparison() -> None:
    """RD-93: the regression that count-and-disclose creates if pred is suppressed
    everywhere it falls below the floor.

    Mechanism. Before AC-38 a floor-limited band left `kept` for EVERY leg, so the
    scene was still scored on the remaining band. Suppress pred in that same band
    while DISCLOSING it for the physical legs and the band now survives, pred is
    NaN inside it, and the whole scene leaves `paired_improvement` — a selection on
    the dependent variable, in the optimistic direction, and strictly larger than
    the one F-70 already records.

    It is not hypothetical: implemented that way, the canonical dry run moved
    `test_material_shift` EDT to n_scored 3 → 2, deleted its MDES (0.0281 → N/A)
    and RAISED pred_mean 0.0649 → 0.0858 s, because the scene it dropped was the
    low one. The fix is AC-25's own qualifier — pred is suppressed only in a band
    THE PHYSICS RESOLVES — after which the same run reads n_scored 3, MDES 0.0306,
    pred_mean 0.0634 s (down, as removing an upward bias should be).

    So: a decay below the floor in ALL legs must leave the scene scored.
    """
    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    cfg = _base_config()
    iso = [float(f) for f in cfg.iso_eval_freqs]
    # Well below the 500 Hz T30 floor (2 x 0.0204 s) in every leg: nobody can
    # resolve this decay, so nobody may be singled out for failing to.
    legs = {
        name: _decaying_noise_ir(0.012, 0.5, seed=s).astype(np.float32)[None, :]
        for name, s in (("pred", 3), ("high", 1), ("low", 2))
    }
    triples, _reasons, _window, acct = compute_room_acoustic_metrics(
        legs["pred"], legs["high"], legs["low"],
        sample_rate=cfg.sample_rate, iso_eval_freqs=iso,
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    assert acct["T30"]["resolvability_limited_hz"], (
        "fixture does not bite: no band was floor-limited, so this asserts nothing"
    )
    assert not np.isnan(triples["T30"].pred), (
        "pred was unscored in a band the PHYSICAL legs cannot resolve either — that "
        "drops the scene from paired_improvement and enlarges F-70's selection "
        "(RD-93). Suppression is only legitimate where the physics resolves the band."
    )
    assert acct["T30"]["pred_unresolved_in_floor_limited_hz"] == [], (
        "a floor-limited band is again costing pred its score (RD-93)"
    )


# ---------------------------------------------------------------------------
# AC-42: C50 gets the guard and the accounting entry its siblings already had
# ---------------------------------------------------------------------------


def test_a_degenerate_pred_leaves_c50_unscored_not_at_plus_200_db() -> None:
    """AC-42: C50 was guarded only against `late == 0`, never against `late` at the
    numeric floor.

    MEASURED before the guard, with pred replaced by a 5 ms decay: T30.pred and
    EDT.pred are correctly unscored with AC-25 reasons while `C50.pred` reports
    +148 dB at 500 Hz and +266 dB at 1000 Hz — finite, therefore scored, therefore
    pooled into the split's `pred_mean` and its CI. The paired `improved` flag was
    always safe (a +200 dB pred simply fails to improve), so this is an ABSOLUTES
    defect, which is exactly the kind that reaches an E1 table unchallenged.
    """
    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    cfg = _base_config()
    high = _decaying_noise_ir(0.30, 1.0, seed=1).astype(np.float32)[None, :]
    low = _decaying_noise_ir(0.30, 1.0, seed=2).astype(np.float32)[None, :]
    pred = _decaying_noise_ir(0.005, 1.0, seed=3).astype(np.float32)[None, :]

    triples, reasons, _window, acct = compute_room_acoustic_metrics(
        pred, high, low,
        sample_rate=cfg.sample_rate,
        iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )

    assert np.isnan(triples["C50"].pred), (
        f"a 5 ms decay reported C50 = {triples['C50'].pred:+.1f} dB as a SCORED "
        f"absolute; it must be unscored with a reason (AC-42)"
    )
    # The sibling accounting entry C50 was missing.
    assert acct["C50"]["pred_unresolved_hz"], (
        "C50.pred is NaN but no `pred_unresolved_hz` entry was recorded, so the "
        "split's counts cannot see it (AC-42)"
    )
    assert "AC-42" in reasons[("C50", "pred")]
    # The physical legs are untouched — a model output never changes its own
    # ground truth (AC-25).
    assert np.isfinite(triples["C50"].high) and np.isfinite(triples["C50"].low)


def test_a_large_but_real_c50_is_not_unscored_by_the_guard() -> None:
    """AC-42's guard must not eat AC-39's finding.

    A strongly direct-dominated scene has a genuinely large C50 — at the corner
    base.yaml's declared support admits (3x3x2.4 m, alpha 0.98, d 2 m) the rendered
    high leg measures +48.8 dB at 500 Hz and +50.7 dB at 1000 Hz. That is the
    DIRECT ARRIVAL, an ISO-3382-real quantity, and AC-39 exists to stop it being
    reported as a filter artifact.

    This is why C50 inherits T30's verdict and ONLY T30's (RD-98): EDT at that same
    corner IS below its floor, so a C50 coupled to EDT would unscore a perfectly
    measurable +50 dB, while T30 there reads 0.0678-0.0800 s — comfortably resolved.

    A float32 residue bound was also tried here and REMOVED. It bounded a sum over
    the late window using the EARLY peak, which is never an operand of that sum, so
    it reduced to a C50 ceiling set by the direct arrival's crest factor (57.6-62.2
    dB, varying with arrival shape alone) and censored the PHYSICAL legs at true
    T60 <= 0.04 s — with a firing probability monotone in absorption, i.e.
    confounded with `test_material_shift`'s own axis. Raised independently by the
    acoustics-reviewer and the falsifier. `test_a_physical_leg_is_never_censored_
    for_its_own_c50` below is the regression guard.
    """
    from amcd.evaluation.room_acoustic import channel_band_avg_metrics

    cfg = _base_config()
    n = int(cfg.sample_rate * 1.0)
    direct_dominated = _decaying_noise_ir(0.30, 1.0, seed=4).astype(np.float32) * 1e-3
    direct_dominated[0] = 1.0

    values, reasons = channel_band_avg_metrics(
        direct_dominated, sample_rate=cfg.sample_rate,
        iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    assert np.isfinite(values["C50"]), (
        f"a direct-dominated scene's C50 was unscored ({reasons.get('C50')}) — the "
        f"AC-42 guard is eating real high-clarity values, which is precisely what "
        f"coupling it to the decay verdict would have done (RD-98)"
    )
    assert values["C50"] > 20.0, (
        f"fixture does not bite: C50 = {values['C50']:.1f} dB is not high-clarity, "
        f"so this asserts nothing about the guard"
    )


@pytest.mark.parametrize("true_t60", [0.0179, 0.02, 0.03, 0.04, 0.05])
def test_a_physical_leg_is_never_censored_for_its_own_c50(true_t60: float) -> None:
    """The regression guard for a blocker this lane shipped and then removed.

    AC-42's first implementation guarded C50 with a "float32 accumulation residue",
    `sqrt(n_late) * eps * max(energy over the integrated region)`. The derivation
    was wrong: `late` is a sum over the LATE window and `max(...)` spans the EARLY
    one, so the rule bounded nothing about that sum. Algebraically it was a C50
    ceiling set by the direct arrival's crest factor — measured 57.6-62.2 dB
    varying with arrival SHAPE at a fixed tail level, and moving the wrong way, a
    sharper arrival lowering the ceiling.

    The consequence is what makes this worth a permanent test rather than a note:
    it censored the PHYSICAL legs, firing at true T60 <= 0.04 s, with a firing
    probability monotone in absorption — i.e. correlated with `test_material_shift`'s
    own independent variable. That is exactly the censoring-on-the-datum defect
    AC-38 removes elsewhere in this same file, reintroduced beside it.

    `room_acoustic.py:420` states the declared support admits Eyring T60 = 0.0179 s,
    so the sweep starts there. Whatever guards C50 in future, a low/high leg must
    keep its value: only `pred` may ever be suppressed, and only in a band the
    physical legs themselves resolve.
    """
    from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

    cfg = _base_config()
    ir = _decaying_noise_ir(true_t60, 1.0, seed=5).astype(np.float32)[None, :]
    triples, _reasons, _window, _acct = compute_room_acoustic_metrics(
        ir.copy(), ir.copy(), ir.copy(),
        sample_rate=cfg.sample_rate,
        iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
    )
    for leg in ("low", "high"):
        assert np.isfinite(getattr(triples["C50"], leg)), (
            f"the {leg} leg's C50 was unscored at true T60 = {true_t60} s, a decay "
            f"base.yaml's declared support admits. A physical leg must never be "
            f"censored on its own value — that is AC-38's whole thesis, and a C50 "
            f"guard that fires more often as absorption rises is confounded with "
            f"the test_material_shift axis."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cluster C5 / C6 — the truncation window and the record-length gate.
# Added at the cycle-5 integration for AC-58, AC-64 and F-186, each of which
# asks for exactly one of these as its acceptance test. All three are
# RENDER-FREE by construction: they are the half of ITEM 0 / ITEM 0b that a
# known answer can settle, so the emulated renders are spent only on what a
# known answer cannot.
# ─────────────────────────────────────────────────────────────────────────────

def _base_scalar(name: str) -> float:
    """One experiment-governing value, read from `configs/base.yaml` (F-219).

    These were module literals with a `# configs/base.yaml` comment standing in for
    a read — `ir_duration: 4.25` and `d0b_t30_jnd_frac: 0.05`, both duplicated from
    the config that governs them. Either could move and every test built on them
    would keep asserting against the old value while claiming to describe the
    shipped one.
    """
    return float(getattr(Config.load(Path("configs/base.yaml")), name))


def _backend_native_support_s() -> float:
    """The longest native record the ACTIVE backend can produce, from its config.

    gsound compiles `maxIRLength = 3.0 s` and does not expose it, so no render
    exceeds it whatever `ir_duration` says — which is why this is read from
    `max_ir_length_s` rather than derived from the decay (AC-178).
    """
    params = Config.load(Path("configs/base.yaml")).simulator.params
    return float(params["max_ir_length_s"])


#: Realizations per cell where a test averages over draws. The estimator is
#: stochastic — the decay is noise under an envelope — so a single draw measures a
#: draw, not the estimator. 30 is the smallest count at which the spread of a
#: 2-band mean is stable enough to assert on; it is a TEST parameter, not an
#: experiment value, and nothing in the pipeline reads it.
_SEEDS = 30


def _padded_decay(rt60: float, seed: int, floor_rms: float = 0.0) -> np.ndarray:
    """A decay written into the backend's REALIZED native support, then zero-padded
    to the full `ir_duration` record — the production record shape.

    Both halves matter and neither is free (AC-178):

    * The native part is bounded by what the backend can actually produce
      (`max_ir_length_s`), NOT by the decay. The earlier version used `2 * T60`, a
      free parameter with no backend meaning, so above T60 ~ 2.1 s it wrote MORE
      decay than any render can contain — and the tests that rest on it are
      precisely the ones about a room too reverberant for its record.
    * `_fit_to_window` then pads to the configured length, so the last stretch of
      every real record is exactly zero unless synthesis leaves a numerical floor.
      That padding is what the truncation-window tests are about.
    """
    n_record = int(_base_scalar("ir_duration") * _SR)
    n_native = min(int(_backend_native_support_s() * _SR), n_record)
    ir = np.zeros(n_record)
    ir[:n_native] = _decaying_noise_ir(rt60, n_native / _SR, seed)[:n_native]
    if floor_rms:
        ir = ir + np.random.default_rng(seed + 1).standard_normal(n_record) * floor_rms
    return ir


class TestTheTruncationIndexUnderAZeroPad:
    """AC-58: a pure gain must move neither the reported metric nor the window.

    ISO 3382's T30/EDT/C50 are RATIOS, so scaling a recording cannot change them.
    The Schroeder limit is chosen by `_lundeby_truncate`, and on a zero-padded
    record — which is every gsound record, since they are padded from their native
    length out to `ir_duration` — the last 10 % is EXACTLY zero. The old code
    estimated a noise floor there anyway, clamped the result to 1e-30, and so
    turned a relative "10 dB above the floor" test into an ABSOLUTE 1e-29 that the
    signal crosses at a gain-dependent place.

    Measured through the production path on a retained real render
    (`experiments/ac175_probe/ac175_mid`, 500 Hz) over 10^-8 … 10^+8:

        before   index 20948 -> 40672   swing 19724 samples = 411 ms
        after    index unchanged        swing 0

    The fix truncates at the point where band energy falls to `peak * eps` instead,
    which is invariant because both terms scale together. A bare "last non-zero
    sample" is not: the octave filter rings into the pad, and how far that ringing
    survives before underflowing depends on absolute magnitude — measured at 126.7
    ms of residual swing, better than 411 and still not a room-acoustic quantity.
    """

    def test_the_metric_is_gain_invariant_even_though_the_index_is_not(self) -> None:
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        ir = _padded_decay(0.6, seed=0)
        for fc in _ISO:
            values = [
                _iso3382_band_metrics(
                    ir * 10.0**e, fc, _SR, band_resolvability_margin=_NO_FLOOR
                , min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER)[0]
                for e in (-6, -3, 0, 3)
            ]
            for metric in ("T30", "EDT", "C50"):
                got = [v[metric] for v in values]
                scale = max(abs(min(got)), 1e-12)
                # Relative, and tied to the JND rather than to bit-exactness: the
                # residual is float32 noise in the filter output (~5.6e-5, i.e.
                # 0.1 % of d0b_t30_jnd_frac), not the index moving the answer.
                assert (max(got) - min(got)) / scale < _base_scalar("d0b_t30_jnd_frac") / 50.0, (
                    f"{metric} at {fc} Hz moved across 1e-6..1e+3 of pure gain: "
                    f"{got}. ISO 3382 metrics are ratios; a level change must not "
                    "move them (AC-58)."
                )

    def test_the_truncation_index_is_gain_invariant(self) -> None:
        """The window itself, not only the metric it produces.

        Pinning the metric alone is not enough: `source_power` is a config knob, so
        a window that tracks absolute level couples the integration limit to a
        setting that has nothing to do with the room. This asserts the index is
        EXACTLY unchanged — not approximately — because the fix is a comparison
        against a scaled threshold, and anything less than exact would mean it had
        regressed to a level test.
        """
        from amcd.evaluation.room_acoustic import _lundeby_truncate, _band_energy

        ir = _padded_decay(0.6, seed=0)
        for fc in _ISO:
            idx = {
                _lundeby_truncate(_band_energy(ir * 10.0**e, fc, _SR, _ORDER), _SR)
                for e in (-8, -6, -3, 0, 3, 8)
            }
            assert len(idx) == 1, (
                f"the truncation index at {fc} Hz moved across 1e-8..1e+8 of pure "
                f"gain: {sorted(idx)}. ISO 3382 metrics are ratios, and so is the "
                "window they are integrated over (AC-58)."
            )


#: Realizations per (T60, band) cell. The estimator is stochastic — the decay is
#: noise under an envelope — so a single draw measures a draw, not the estimator.
#: Not an experiment-governing value: it sets the resolution of a test's own
#: claim, and it is stated here because the claim quotes it.


class TestT30RecoversAKnownDecayInAPaddedRecord:
    """AC-64's own no-render acceptance test, run as the row specifies.

    The row predicts a ~600 ms window that would under-read T30 by -43 % at
    T60 = 2.0 s. That does NOT happen: at every T60 the index lands at the
    native/pad boundary, and the mean error runs 0.9-2.6 %.

    But the estimator is NOT uniformly inside the JND, and the first version of
    this test said it was — because it drew ONE realization per cell and got a
    lucky one. Over `_SEEDS` draws (240 cells):

        T60 0.5 s   500 Hz   4/30 breach   max err 0.0720   mean 0.0259
        T60 0.5 s  1000 Hz   1/30 breach   max err 0.0571   mean 0.0193
        T60 1.0 s   500 Hz   3/30 breach   max err 0.0579   mean 0.0216
        T60 1.0 s  1000 Hz   0/30          max err 0.0360   mean 0.0122
        T60 2.0 s   500 Hz   0/30          max err 0.0437   mean 0.0149
        T60 2.0 s  1000 Hz   0/30          max err 0.0409   mean 0.0147
        T60 3.0 s   500 Hz   0/30          max err 0.0355   mean 0.0126
        T60 3.0 s  1000 Hz   0/30          max err 0.0255   mean 0.0092

    Eight breaches in 240, all at the SHORT end, none worse than 1.44x the JND.
    So the honest claim is about the MEAN and the tail, not about every draw, and
    the short-T60 tail is real estimator variance that a reported number inherits.

    This still does not discharge AC-64. Its 600 ms index was measured on a real
    gsound IR whose artifact was destroyed by an in-memory probe, and only a
    retained-artifact render can say whether that IR's tail is genuinely ~280 dB
    down from there. What this pins is the ESTIMATOR, so that render measures the
    backend rather than both at once.
    """

    def test_the_mean_error_is_inside_the_jnd_at_every_declared_decay(self) -> None:
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        bad = []
        for rt60 in (0.5, 1.0, 2.0, 3.0):
            for fc in _ISO:
                errs = []
                for seed in range(_SEEDS):
                    values, nan_reasons, _ = _iso3382_band_metrics(
                        _padded_decay(rt60, seed=seed), fc, _SR,
                        band_resolvability_margin=_NO_FLOOR,
                        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
                    )
                    t30 = values["T30"]
                    assert np.isfinite(t30), (
                        f"T30 is NaN at T60={rt60}s, {fc} Hz, seed {seed}: "
                        f"{nan_reasons.get('T30')}"
                    )
                    errs.append(abs(t30 - rt60) / rt60)
                mean_err = float(np.mean(errs))
                if mean_err > _base_scalar("d0b_t30_jnd_frac"):
                    bad.append((rt60, fc, mean_err))
        assert not bad, (
            f"MEAN T30 error breaches d0b_t30_jnd_frac at {bad}. "
            "The estimator must recover a known decay through the shipped ISO "
            "path before a render can be read as measuring the backend (AC-64)."
        )

    def test_the_breach_rate_is_bounded_and_confined_to_the_short_end(self) -> None:
        """The tail the mean hides — pinned so it cannot grow unnoticed.

        Asserting only the mean would let the short-T60 variance widen silently,
        and that variance is what a reported T30 inherits. Asserting zero breaches
        is what the first version of this test did, and it was false.
        """
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        breaches: dict[tuple[float, float], int] = {}
        for rt60 in (0.5, 1.0, 2.0, 3.0):
            for fc in _ISO:
                n = 0
                for seed in range(_SEEDS):
                    values, _, _ = _iso3382_band_metrics(
                        _padded_decay(rt60, seed=seed), fc, _SR,
                        band_resolvability_margin=_NO_FLOOR,
                        min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
                    )
                    if abs(values["T30"] - rt60) / rt60 > _base_scalar("d0b_t30_jnd_frac"):
                        n += 1
                if n:
                    breaches[(rt60, fc)] = n

        total = sum(breaches.values())
        assert total <= 16, (  # measured 8 of 240; 2x headroom for filter drift
            f"T30 breaches the JND on {total} of {4 * len(_ISO) * _SEEDS} draws "
            f"({breaches}), against 8 measured at the cycle-5 integration. The "
            "estimator's short-decay variance has grown."
        )
        assert all(rt60 <= 1.0 for rt60, _ in breaches), (
            f"a breach appeared at a LONG decay: {breaches}. Short-end variance "
            "is expected and bounded above; a long-decay breach is the truncation "
            "failure AC-64 predicts, and means the window moved."
        )


class TestARoomTooReverberantForItsRecord:
    """F-186 / AC-176: a decay longer than its record is UNSCORED, not quietly wrong.

    base.yaml's largest declared shoebox at nominal alpha 0.05 is T60 = 4.200 s and
    fits the 4.25 s record; under the alpha_eff convention ITEM 0 is deciding
    (alpha_eff = 1 - sqrt(1 - alpha)) the same room is T60 = 8.294 s and does not.

    Before the AC-176 bound the estimator returned 6.3868 s at 500 Hz (-22.99 %)
    and 5.5690 s at 1000 Hz (-32.86 %) with `nan_reason` AND `resolvability` both
    None. The sign is the point: T30 UNDER-reads, so a room whose decay outruns its
    record is reported as LESS reverberant than it is — the direction that makes
    the truncation look like a shorter room rather than a broken measurement.

    ISO 3382-1 does not permit a T30 here at all: only 30.7 dB of decay is
    available (60 x 4.25 / 8.294) against the >= 45 dB the standard requires. So
    the correct output is unscored-with-a-reason, and that is now what it is.
    """

    def test_a_decay_that_fits_its_record_is_scored(self) -> None:
        """The control. Without it the refusal below could be refusing everything.

        T60 = 2.0 s, not the 4.2 s this used: under the backend's realized support
        (`max_ir_length_s` = 3.0 s, compiled and not settable) a 4.2 s decay gets
        60 x 3.0 / 4.2 = 42.9 dB of range and is correctly REFUSED. It only looked
        like a control while the helper wrote `2 * T60` of decay into the full
        4.25 s record and ignored the backend's cap (AC-178) — i.e. while the
        "control" described a record no render can produce. At 2.0 s the range is
        90 dB, well inside the standard's 45.
        """
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        for fc in _ISO:
            values, reasons, _ = _iso3382_band_metrics(
                _padded_decay(2.000, seed=0), fc, _SR,
                band_resolvability_margin=_NO_FLOOR,
                min_decay_range_db=_ISO_SNR_BOUND,
            octave_filter_order=_ORDER,
            )
            assert reasons.get("T30") is None, (
                f"a decay that FITS its record was refused at {fc} Hz: "
                f"{reasons['T30']}. The bound must refuse truncation, not length."
            )
            err = abs(values["T30"] - 2.000) / 2.000
            assert err <= _base_scalar("d0b_t30_jnd_frac"), f"control T30 off by {err:.2%} at {fc} Hz"

    def test_a_decay_twice_its_record_is_unscored_with_a_reason(self) -> None:
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        for fc in _ISO:
            values, reasons, _ = _iso3382_band_metrics(
                _padded_decay(8.294, seed=0), fc, _SR,
                band_resolvability_margin=_NO_FLOOR,
                min_decay_range_db=_ISO_SNR_BOUND,
            octave_filter_order=_ORDER,
            )
            assert np.isnan(values["T30"]), (
                f"T30 = {values['T30']} at {fc} Hz for a room whose decay is twice "
                "its record. A number here is the silent under-read F-186 is about."
            )
            reason = reasons.get("T30")
            assert reason and "decay range" in reason and "45" in reason, (
                f"T30 is NaN at {fc} Hz but the reason does not name the decay "
                f"range and the ISO bound: {reason!r}. Nothing may leave a result "
                "silently (CLAUDE.md)."
            )

    def test_the_bound_is_config_declared_not_hardcoded(self) -> None:
        """The same record is scored or refused purely by what the config declares.

        This is what makes the scaffold's lower value legitimate rather than a
        quiet weakening: the threshold is an experiment-governing value, so it
        lives in config and a run's provenance records which one it used.
        """
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        ir = _padded_decay(8.294, seed=0)
        strict, _, _ = _iso3382_band_metrics(
            ir, _ISO[0], _SR, band_resolvability_margin=_NO_FLOOR,
            min_decay_range_db=_ISO_SNR_BOUND,
            octave_filter_order=_ORDER)
        lax, _, _ = _iso3382_band_metrics(
            ir, _ISO[0], _SR, band_resolvability_margin=_NO_FLOOR,
            min_decay_range_db={"T30": 15.0, "EDT": 10.0},
            octave_filter_order=_ORDER)
        assert np.isnan(strict["T30"]) and np.isfinite(lax["T30"]), (
            "the declared bound did not change the outcome on an identical "
            "record, so it is not the thing deciding admissibility"
        )

    def test_the_metric_floor_must_be_declared_for_every_reported_metric(self) -> None:
        """No hidden default: an undeclared metric raises rather than passing."""
        from amcd.evaluation.room_acoustic import _iso3382_band_metrics

        with pytest.raises(KeyError, match="declares no floor"):
            _iso3382_band_metrics(
                _padded_decay(1.0, seed=0), _ISO[0], _SR,
                band_resolvability_margin=_NO_FLOOR,
                min_decay_range_db={"EDT": 20.0},   # T30 missing
                octave_filter_order=_ORDER,
            )


class TestTheHeadroomFloorIsPinnedAgainstBandLeakage:
    """AC-105's outstanding residual: the leakage sweep, as a regression test.

    The AC-37-R4 guard's operand is the REPORTED ISO span, which is right. But the
    ladder bands it EXCLUDES are not out of band for the octave filter that
    computes the metric — 315.0 Hz carries -17.40 dB of the 500 Hz octave's weight
    and 1587.4 Hz carries -17.38 dB of the 1000 Hz octave's — so an excluded band
    pinned at `min_db` leaks into a reported T30.

    That is what makes a declared threshold of 35 dB unsafe even though T30's own
    regression window is 35 dB wide: 35 is necessary and NOT sufficient, and the
    floor for any declared value is ~52 dB = 35 + 17.4.

    Nothing pinned this. AC-105's docstring correction landed and its regression
    test did not, so the next person to tune `min_db_headroom_db` downward would
    meet no resistance until a reported T30 moved.
    """

    #: From the config's own derivation: 35 dB of T30 regression window plus the
    #: 17.4 dB an adjacent excluded ladder band leaks into the octave metric.
    _DERIVED_FLOOR_DB = 52.0

    def test_the_shipped_threshold_is_at_or_above_the_leakage_derived_floor(self) -> None:
        from pathlib import Path

        import yaml

        params = yaml.safe_load(
            (Path(__file__).resolve().parent.parent
             / "configs" / "representations" / "spectrogram.yaml").read_text()
        )
        shipped = float(params["min_db_headroom_db"])
        assert shipped >= self._DERIVED_FLOOR_DB, (
            f"min_db_headroom_db is {shipped} dB, below the {self._DERIVED_FLOOR_DB} dB "
            "floor derived as 35 (T30's regression window) + 17.4 (the weight an "
            "EXCLUDED ladder band still carries in the reported octave metric). "
            "Below this an excluded band on the floor moves a reported T30: "
            "measured +1.1 % at 40 dB, +4.4 % at 35, +91.7 % at 30 against a 5 % "
            "JND. If the operand or the ladder changed, re-derive the floor and "
            "move the constant — do not lower the threshold to meet it (AC-105)."
        )

    def test_the_excluded_neighbour_really_does_leak_into_the_metric(self) -> None:
        """The measurement the floor rests on, so the constant is not folklore.

        If this stops holding, the 17.4 dB term is wrong and the floor above is
        wrong with it — which is exactly the drift AC-24's duplicate-declaration
        shape produces when a derived constant outlives its derivation.
        """
        from amcd.evaluation.room_acoustic import _band_energy

        sr = _SR
        n = int(0.5 * sr)
        t = np.arange(n) / sr
        # A pure tone at the excluded ladder band adjacent to the 1000 Hz octave.
        excluded_hz = 1587.4
        tone = np.sin(2 * np.pi * excluded_hz * t).astype(np.float64)
        in_band = np.sin(2 * np.pi * 1000.0 * t).astype(np.float64)

        def band_power(x):
            e = _band_energy(x, 1000.0, sr, _ORDER)
            e = e[0] if isinstance(e, tuple) else e
            return float(np.sum(e))

        leak_db = 10.0 * np.log10(band_power(tone) / band_power(in_band))
        assert -25.0 < leak_db < -10.0, (
            f"the excluded {excluded_hz} Hz band leaks {leak_db:.2f} dB into the "
            "1000 Hz octave metric, against the ~-17.4 dB the headroom floor is "
            "derived from. Re-derive _DERIVED_FLOOR_DB before trusting it (AC-105)."
        )


class TestGeometryAdjudicatesTheOnset:
    """AC-181 — `_find_onset` thresholds below the GLOBAL peak, so it assumes the
    direct sound is the loudest sample, and under a noise carrier it is not.

    Upstream synthesizes every path onto one carrier
    (`result[c, t] = normalized_sh[c] * carrier[t]`, binding.cpp:423), so the direct
    arrival's realized amplitude is its weight times a standard-normal draw at its
    own bin. A small draw puts it under a threshold its own peak set, and t=0 lands
    on a reflection — moving the C50 50 ms split, the EDT anchor and the Schroeder
    start together.

    The rate is 0.7-1.2 % at the DRR base.yaml's placement support admits; the
    ERROR when it happens is the room's whole mixing time. These are known-answer
    tests over a constructed miss rather than a rate measurement, because the rate
    is not what makes it a defect.
    """

    _SR = 48000
    _REL_DB = -20.0
    _DELAY = 480          # 10 ms — the geometric direct arrival
    _REFLECTION = 720     # 15 ms — a 5 ms early-reflection gap after it
    _TOL = 48             # 1 ms, base.yaml's metric_onset_tolerance_ms

    def _ir_whose_direct_is_quiet(self) -> np.ndarray:
        """Direct arrival at `_DELAY`, 30 dB BELOW a reflection at `_REFLECTION`.

        Silence between them, which is what a real render's early-reflection gap
        looks like and what makes a missed onset land far away rather than one
        sample late.
        """
        ir = np.zeros(self._SR // 4, dtype=np.float32)
        ir[self._DELAY] = 0.03           # -30.5 dB re the reflection
        ir[self._REFLECTION] = 1.0
        ir[self._REFLECTION + 1:] = 0.01 * np.random.default_rng(7).standard_normal(
            len(ir) - self._REFLECTION - 1
        )
        return ir

    def test_without_geometry_the_detector_lands_on_the_reflection(self) -> None:
        """The defect itself, reproduced. -30.5 dB is below the -20 dB threshold,
        so the direct arrival does not cross a bar the reflection set."""
        from amcd.evaluation.room_acoustic import _find_onset

        onset, reason = _find_onset(self._ir_whose_direct_is_quiet(), self._REL_DB)
        assert onset == self._REFLECTION, onset
        assert reason is None, "no geometry was supplied, so there is nothing to check"

    def test_with_geometry_the_onset_is_the_direct_arrival(self) -> None:
        from amcd.evaluation.room_acoustic import _find_onset

        onset, reason = _find_onset(
            self._ir_whose_direct_is_quiet(), self._REL_DB,
            expected_sample=self._DELAY, tolerance_samples=self._TOL,
        )
        assert onset == self._DELAY, (
            f"onset landed at {onset}, {1000 * (onset - self._DELAY) / self._SR:.2f} ms "
            f"after the direct arrival — on the reflection (AC-181)"
        )
        assert reason is not None and "AC-181" in reason, reason

    def test_a_correctly_detected_onset_is_left_alone_and_says_nothing(self) -> None:
        """The guard must not fire on the normal case, or every scene would carry a
        caveat and the disclosure would mean nothing."""
        from amcd.evaluation.room_acoustic import _find_onset

        ir = np.zeros(self._SR // 4, dtype=np.float32)
        ir[self._DELAY] = 1.0
        ir[self._REFLECTION] = 0.3
        onset, reason = _find_onset(
            ir, self._REL_DB, expected_sample=self._DELAY, tolerance_samples=self._TOL
        )
        assert onset == self._DELAY and reason is None

    def test_the_detector_still_places_it_inside_the_window(self) -> None:
        """Geometry adjudicates; it does not override. Filter smearing and
        sub-sample timing move the true crossing by a few samples, and within the
        tolerance the DETECTOR's answer is kept — `expected` is an integer floor of
        a real-valued delay, so pinning to it exactly would be its own small error.
        """
        from amcd.evaluation.room_acoustic import _find_onset

        ir = np.zeros(self._SR // 4, dtype=np.float32)
        ir[self._DELAY + 3] = 1.0        # 3 samples inside the 48-sample window
        onset, reason = _find_onset(
            ir, self._REL_DB, expected_sample=self._DELAY, tolerance_samples=self._TOL
        )
        assert onset == self._DELAY + 3 and reason is None

    def test_the_reason_reaches_every_metric_of_the_leg_it_belongs_to(self) -> None:
        """A misplaced t=0 moves T30's Schroeder start, EDT's anchor and C50's split
        at once, so it is a caveat on all three — not on whichever one happens to
        drop.
        """
        from amcd.evaluation.room_acoustic import compute_room_acoustic_metrics

        quiet = self._ir_whose_direct_is_quiet()
        clean = np.zeros_like(quiet)
        clean[self._DELAY] = 1.0
        clean[self._DELAY + 1:] = 0.05 * np.random.default_rng(9).standard_normal(
            len(clean) - self._DELAY - 1
        )

        def stack(w):
            return np.stack([w] + [np.zeros_like(w)] * 15)

        _triples, reasons, _window, _acct = compute_room_acoustic_metrics(
            stack(quiet), stack(clean), stack(clean),
            sample_rate=self._SR, iso_eval_freqs=_ISO, onset_rel_db=self._REL_DB,
            band_resolvability_margin=_NO_FLOOR, min_decay_range_db=_NO_SNR_BOUND,
            octave_filter_order=_ORDER,
            expected_onset_samples=self._DELAY, onset_tolerance_samples=self._TOL,
        )
        for metric in ("T30", "EDT", "C50"):
            assert "AC-181" in reasons.get((metric, "pred"), ""), (
                f"{metric} carries no onset caveat for the leg whose t=0 moved: "
                f"{reasons.get((metric, 'pred'))!r}"
            )
            assert "AC-181" not in reasons.get((metric, "high"), ""), (
                f"{metric} caveats the high leg, whose onset was detected correctly"
            )
