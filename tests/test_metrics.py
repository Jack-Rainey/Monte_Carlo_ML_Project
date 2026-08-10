"""Room-acoustic metric known-answer probes (ISO 3382; ledger AC-02, AC-04).

These are the load-bearing evidence for the two metric fixes — the dry_run pipeline's
zero-delay, low-noise synthetic IRs do not exercise either path, so correctness is
proved here with synthetic IRs of known structure, not by the pipeline run.
"""
import numpy as np

from amcd.evaluation.room_acoustic import channel_band_avg_metrics

_SR = 48000
_ISO = [500.0, 1000.0]
_ONSET_DB = -20.0
# Properties of the synthetic fixtures below, not experiment-governing values: these
# probes assert what the ESTIMATOR does, so the AC-23 measurability floor is disabled
# (0.0) — a known-answer test must see the raw value, including one too short to be
# reported in a real run. `metric_min_measurable_t60_s` is config-declared for the
# pipeline (configs/base.yaml); tests that exercise the floor itself set it explicitly.
_MIN_T60 = 0.0


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
        min_measurable_t60_s=_MIN_T60,
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
        min_measurable_t60_s=_MIN_T60,
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
        min_measurable_t60_s=_MIN_T60)
    assert np.isfinite(low_bands[0][0]["C50"]) and np.isnan(low_bands[1][0]["C50"])

    triples, reasons, _window = compute_room_acoustic_metrics(
        pred[None, :], high[None, :], low[None, :],
        sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        min_measurable_t60_s=_MIN_T60,
    )
    c50 = triples["C50"]
    # The drop is attributed to the causing leg and marked as affecting all legs.
    assert ("C50", "low") in reasons
    assert "EVERY leg" in reasons[("C50", "low")]

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
        min_measurable_t60_s=_MIN_T60, trunc_idx_per_band=shared_500)[0]["C50"]
    high_both = channel_band_avg_metrics(
        high, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
        min_measurable_t60_s=_MIN_T60)[0]["C50"]
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
    for t60 in (0.5, 1.0, 2.0):
        for floor_db in (-80.0, -60.0, -50.0, -40.0, -30.0):
            decay = np.random.default_rng(0).standard_normal(n) * np.exp(-6.9077 * t / t60)
            # Same noise realization in both legs; only its LEVEL differs.
            noise = np.random.default_rng(1).standard_normal(n)
            amp = 10.0 ** (floor_db / 20.0)
            legs = {
                "high": (decay + amp * noise).astype(np.float32),
                "low": (decay + amp * np.sqrt(40.0) * noise).astype(np.float32),
            }
            shared = _shared_truncation_per_band(
                legs, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
            )
            vals = {
                leg: channel_band_avg_metrics(
                    ir, sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
                    min_measurable_t60_s=_MIN_T60, trunc_idx_per_band=shared,
                )[0]["T30"]
                for leg, ir in legs.items()
            }
            assert np.isfinite(vals["high"]) and np.isfinite(vals["low"])
            rel = abs(vals["low"] - vals["high"]) / t60
            # The bound is the project's declared T30 JND (`d0b_t30_jnd_frac` = 0.05),
            # not a slack tolerance. MEASURED over this exact grid
            # (scratchpad/p_residual.py), shared window vs per-leg truncation:
            #
            #   floor   -80    -60    -50     -40     -30
            #   shared  0.09   1.20   1.41   +2.11   -4.92   (worst 4.92 %)
            #   per-leg 0.12   5.03  17.96   27.69   66.77   (worst 66.8 %)
            #
            # What remains after the fix is not a truncation artifact: the noisier leg
            # genuinely carries more energy INSIDE the common window, which is the real
            # noise the denoiser exists to remove. The tell is the SIGN — pre-fix the
            # residual was systematically negative (the noisy leg always read short,
            # the signature of a shortened window); post-fix it scatters either side of
            # zero. The -30 dB row is an extreme case (the low leg's floor sits at
            # -14 dB after the sqrt(40) scaling), included so the bound is tested where
            # it is tightest.
            assert rel < 0.05, (
                f"T30 differs by {100 * rel:.1f}% between legs at T60={t60}s, "
                f"floor={floor_db}dB with IDENTICAL decay — the integration window "
                f"is still noise-floor dependent (AC-17)."
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
        triples, _reasons, window = compute_room_acoustic_metrics(
            pred[None, :], high[None, :], low[None, :],
            sample_rate=_SR, iso_eval_freqs=_ISO, onset_rel_db=_ONSET_DB,
            min_measurable_t60_s=_MIN_T60,
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
