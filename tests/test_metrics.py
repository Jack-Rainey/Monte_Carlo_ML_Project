"""Room-acoustic metric known-answer probes (ISO 3382; ledger AC-02, AC-04).

These are the load-bearing evidence for the two metric fixes — the dry_run pipeline's
zero-delay, low-noise synthetic IRs do not exercise either path, so correctness is
proved here with synthetic IRs of known structure, not by the pipeline run.
"""
import numpy as np
import torch

from amcd.evaluation.room_acoustic import channel_band_avg_metrics

_SR = 48000
_ISO = [500.0, 1000.0]
_ONSET_DB = -20.0
# A margin of 0 disables the resolvability floor entirely. Not an experiment value:
# these probes assert what the ESTIMATOR does, so a known-answer test must see the
# raw fitted value, including one too short to be reported in a real run.
# `metric_band_resolvability_margin` is config-declared for the pipeline
# (configs/base.yaml); tests that exercise the floor itself set it explicitly.
_NO_FLOOR = 0.0


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
        band_resolvability_margin=_NO_FLOOR)
    assert np.isfinite(low_bands[0][0]["C50"]) and np.isnan(low_bands[1][0]["C50"])

    triples, reasons, _window, _acct = compute_room_acoustic_metrics(
        pred[None, :], high[None, :], low[None, :],
        sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR,
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
    )
    high_500 = channel_band_avg_metrics(
        high, sample_rate=_SR, iso_eval_freqs=[500.0], onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR, trunc_idx_per_band=shared_500)[0]["C50"]
    high_both = channel_band_avg_metrics(
        high, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        band_resolvability_margin=_NO_FLOOR)[0]["C50"]
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
                )
                vals = {
                    leg: channel_band_avg_metrics(
                        ir, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
                        band_resolvability_margin=_NO_FLOOR, trunc_idx_per_band=shared,
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


@pytest.mark.parametrize("fc", _ISO)
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

    e_onset = float(_band_energy(at_onset, fc, _SR).sum())
    e_interior = float(_band_energy(interior, fc, _SR).sum())

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

    e_abrupt = float(_band_energy(ends_high, 500.0, _SR).sum())
    e_padded = float(_band_energy(padded_tail, 500.0, _SR).sum())
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
        cfg.representation.name, cfg.representation.params, sample_rate=cfg.sample_rate
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
    headroom = float((torch.amax(env, dim=(0, 2)) - rep.min_db).min())
    oracle = rep.decode(env, low)

    triples, _reasons, _window, _acct = compute_room_acoustic_metrics(
        oracle, high, low,
        sample_rate=cfg.sample_rate,
        iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
        onset_rel_db=cfg.metric_onset_rel_db,
        band_resolvability_margin=cfg.metric_band_resolvability_margin,
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
    energy = _band_energy(ir, fc, _SR)
    energy = energy[: _lundeby_truncate(energy, _SR)]
    if len(energy) < 2:
        return float("nan")
    return _decay_times_from_energy(energy, _SR)[0]


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
            if t30 >= margin * _band_resolvable_decay_s(fc, _SR)["T30"]:
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
    )
    for leg in ("low", "high"):
        assert np.isfinite(getattr(triples["C50"], leg)), (
            f"the {leg} leg's C50 was unscored at true T60 = {true_t60} s, a decay "
            f"base.yaml's declared support admits. A physical leg must never be "
            f"censored on its own value — that is AC-38's whole thesis, and a C50 "
            f"guard that fires more often as absorption rises is confounded with "
            f"the test_material_shift axis."
        )
