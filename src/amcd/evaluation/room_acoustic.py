"""
Room-acoustic metrics T30, C50, EDT (ISO 3382).

Metric source of truth (design_spec §3): reported metrics are computed from the
**decoded waveform** via the standard ISO-3382 path — IIR octave-band filter,
Lundeby noise-floor truncation, Schroeder backward integration. Never from the
STFT energy grid directly.

Public API
----------
compute_room_acoustic_metrics(pred_ir, high_ref_ir, low_ref_ir, *, sample_rate,
                               iso_eval_freqs, onset_rel_db, band_resolvability_margin)
    Standard ISO-3382 waveform path for the eval stage (pred/high/low in one
    call) → (metric triples, NaN reasons, shared integration window, band
    accounting); legs band-averaged over the band set the PHYSICAL legs resolve
    (AC-08/AC-25) and integrated over the cross-leg-shared Schroeder window
    (AC-17). Neither the band set nor the window is ever set by `pred` (RD-43).
channel_band_avg_metrics(ir_w, *, sample_rate, iso_eval_freqs, onset_rel_db,
                         band_resolvability_margin, trunc_idx_per_band=None)
    Single-IR onset-aligned band-averaged metrics → (values, NaN reasons); the
    D0b oracle probe's unit. Pass `trunc_idx_per_band` to join a paired
    comparison's shared window.
channel_per_band_metrics(ir_w, *, sample_rate, iso_eval_freqs, onset_rel_db,
                         band_resolvability_margin, trunc_idx_per_band=None)
    Per-band (values, reasons) — the shared unit under both of the above
    (identical alignment + truncation).
_shared_truncation_per_band(reference_legs, *, sample_rate, iso_eval_freqs,
                            onset_rel_db)
    Per band, the one Schroeder integration limit every leg of a paired
    comparison must use (AC-17). Reference legs are PHYSICAL legs only — never a
    model output (RD-43).

Energy-domain helpers (private, kept for training metric-consistency loss — §3-D4)
-----------------------------------------------------------------------------------
_schroeder_edr, _t30, _edt, _c50, _metrics_from_energy, _band_avg_metrics,
_find_iso_bands
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .metric_row import MetricTriple


# Minimum record length for the zero-phase 4th-order octave filter. sosfiltfilt on a
# 4-section SOS has padlen=27, so it needs > 27 samples; 32 gives headroom. Numerical
# guard, not an experiment parameter.
_MIN_FILTER_SAMPLES = 32


# ---------------------------------------------------------------------------
# Standard ISO-3382 waveform path (the reported-metric source)
# ---------------------------------------------------------------------------

def _lundeby_truncate(energy_samples: np.ndarray, sample_rate: int) -> int:
    """
    Simplified Lundeby-style noise-floor truncation.
    Returns the sample index at which to truncate before Schroeder integration.
    Estimates noise floor from the last 10% of the record; finds the last sample
    where a short-time smoothed energy exceeds noise_power × 10 dB.
    Falls back to a 10 ms minimum if no samples exceed the threshold (degenerate IR).
    """
    n = len(energy_samples)
    # 10 ms minimum: enough samples for a valid regression, never n//2 (that forces
    # the noise tail back in for any IR that decays in under half the record).
    min_samples = max(2, int(0.010 * sample_rate))

    noise_region = energy_samples[int(0.9 * n):]
    noise_power = float(np.mean(noise_region)) if len(noise_region) > 0 else 1e-30
    noise_power = max(noise_power, 1e-30)
    threshold = noise_power * 10.0  # 10 dB above noise floor

    # Smooth energy with a 10 ms window to reduce sample-level jitter
    win = max(1, int(0.010 * sample_rate))
    kernel = np.ones(win) / win
    smoothed = np.convolve(energy_samples, kernel, mode="same")

    # Find the last sample that exceeds the threshold
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return min_samples
    truncate_idx = int(above[-1]) + 1
    return max(truncate_idx, min_samples)


def _butter_octave_filter(ir_w: np.ndarray, fc: float, sample_rate: int) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth octave-band filter centered at fc Hz.
    Passband: [fc / sqrt(2), fc * sqrt(2)].
    Uses sosfiltfilt for zero-phase response (no group-delay offset in EDT).

    LEADING SILENCE IS EXPLICIT, NOT LEFT TO scipy's DEFAULT PADDING. `sosfiltfilt`
    pads with `padtype="odd"`, which reflects the signal about its first sample: for
    an onset-aligned IR — whose first sample IS the direct arrival, the largest in
    the record — that manufactures a step of twice the peak immediately before t=0,
    an arrival that never existed. MEASURED at 48 kHz with a unit impulse at index
    0: the 500 Hz octave returns 6.171 units of energy for 1 unit in, against the
    ~0.0147 its own bandwidth allows — a 420x inflation, entirely from the padding.
    Downstream it inflated C50 by ~23 dB.

    The defect was INERT until AC-28 made the scaffold's direct arrival a genuine
    broadband impulse; before that no IR had an impulsive first sample, so nothing
    exercised the reflection. It is a property of the metric path, not of that fix.

    An impulse response is silent before its direct arrival, so the physically
    correct context is ZEROS. The guard is scaled as 1/fc because the filter's
    ringing is (measured 500 Hz T30 21.97 ms, 1000 Hz 10.96 ms, exactly 1/f), and it
    is stripped afterwards so integration still starts at the direct arrival, as
    ISO 3382 requires.

    KNOWN RESIDUAL: `filtfilt` is non-causal, so an impulse at the arrival index
    produces a symmetric response and stripping the guard discards its pre-ringing
    — measured, about half the impulse's in-band energy (0.0067 vs the 0.0132 an
    interior impulse yields). That is a property of zero-phase filtering, it is
    common-mode across legs, and paired improvements are unaffected; it does bias
    ABSOLUTE C50 low for strongly direct-dominated scenes.
    """
    nyq = sample_rate / 2.0
    f_lo = fc / 2.0 ** 0.5
    f_hi = fc * 2.0 ** 0.5
    f_lo = max(f_lo, 10.0)      # stay well above DC
    f_hi = min(f_hi, nyq * 0.99)  # stay below Nyquist
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    # ~4x the filter's own T30 in this band; zeros are cheap and the bound only has
    # to exceed the ringing, not match it.
    guard = int(np.ceil(48.0 / fc * sample_rate))
    padded = np.concatenate([np.zeros(guard, dtype=np.float64), np.asarray(ir_w, dtype=np.float64)])
    filtered = sosfiltfilt(sos, padded, padtype="constant")
    return filtered[guard:].astype(np.float32)


def _find_onset(ir_w: np.ndarray, rel_db: float) -> int:
    """Index of the direct-sound arrival: the first sample whose energy rises above
    `rel_db` dB (negative) below the peak energy.

    ISO 3382 integration starts at the direct sound. Real geometric-acoustic renders
    carry a propagation delay (dist/c) of leading near-silence before the direct
    arrival, which would otherwise mis-reference the C50 50 ms split and the Schroeder
    start. Detected on the broadband W channel (propagation delay is frequency-
    independent). Returns 0 for a degenerate/zero IR or if nothing crosses (AC-02).

    ASSUMPTION (AC-07): the threshold is relative to the GLOBAL peak, so the direct
    sound must be the loudest arrival — true for normal IRs (the direct path is
    shortest / least attenuated). A pathological IR whose direct sound sits > |rel_db|
    below a later reflection would land onset on the reflection; revisit if real
    renders exhibit occluded-direct geometries."""
    energy = ir_w.astype(np.float64) ** 2
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0.0:
        return 0
    threshold = peak * 10.0 ** (rel_db / 10.0)
    above = np.where(energy >= threshold)[0]
    return int(above[0]) if above.size else 0


def _band_energy(ir_w: np.ndarray, fc: float, sample_rate: int) -> np.ndarray:
    """Octave-band energy envelope (squared band-filtered samples) — the input both
    to Lundeby truncation and to Schroeder integration. Factored out so a truncation
    index can be derived WITHOUT computing metrics (AC-17)."""
    return _butter_octave_filter(ir_w, fc, sample_rate) ** 2


def _decay_times_from_energy(
    energy_trunc: np.ndarray, sample_rate: int
) -> tuple[float, str | None, float, str | None]:
    """T30 and EDT from an already-truncated band energy envelope.

    Factored out so the RESOLVABILITY FLOOR is measured through the very path it
    governs (AC-26/AC-27) rather than through a second implementation that could
    drift from it — the AC-24 precedent. Returns (t30, t30_reason, edt, edt_reason).
    """
    edr = np.cumsum(energy_trunc[::-1])[::-1].copy()
    edr = np.maximum(edr, 1e-30)
    edr_db = 10.0 * np.log10(edr / edr[0])
    t_s = np.arange(len(edr_db)) / sample_rate

    def _slope_to_rt(lo_db: float, hi_db: float) -> tuple[float, str | None]:
        mask = (edr_db >= hi_db) & (edr_db <= lo_db)
        if mask.sum() < 2:
            return float("nan"), f"<2 EDR points in the [{lo_db:g}, {hi_db:g}] dB regression window"
        coeffs = np.polyfit(t_s[mask], edr_db[mask], 1)
        slope = float(coeffs[0])
        if slope >= 0.0:
            return float("nan"), f"non-decaying EDR (slope ≥ 0) in the [{lo_db:g}, {hi_db:g}] dB window"
        return float(-60.0 / slope), None

    t30, t30_reason = _slope_to_rt(-5.0, -35.0)
    edt, edt_reason = _slope_to_rt(0.0, -10.0)
    return t30, t30_reason, edt, edt_reason


@lru_cache(maxsize=None)
def _band_resolvable_decay_s(fc: float, sample_rate: int) -> dict[str, float]:
    """The shortest decay the `fc` octave band can resolve: the filter's OWN.

    A unit impulse carries no decay at all, so whatever T30/EDT this returns is
    manufactured entirely by the filter's ringing. A room decaying faster than
    this is not being measured — the filter is.

    MEASURED here rather than asserted, and the numbers scale exactly as 1/f
    (48 kHz): 500 Hz → T30 20.309 ms, EDT 9.551 ms; 1000 Hz → 10.037 / 4.793 ms.
    `sosfiltfilt` runs the 4th-order section forwards and backwards, which doubles
    the effective order — the reason an earlier estimate of "~3 ms of ringing" from
    the nominal 353 Hz bandwidth understated it by 3-7x (AC-27).

    T30 and EDT get SEPARATE floors because they are separate questions: EDT fits
    the first 10 dB, T30 the 5-35 dB span, and the filter's own response supports
    the first for twice as long. A single scalar governing both was AC-27's open
    question; this is the answer.
    """
    # Long enough that the filter's ringing decays fully inside the record at any
    # band, and never shorter than the filter's own minimum.
    n = max(int(0.5 * sample_rate), _MIN_FILTER_SAMPLES * 4)
    impulse = np.zeros(n, dtype=np.float32)
    impulse[0] = 1.0
    energy = _band_energy(impulse, fc, sample_rate)
    t30, _, edt, _ = _decay_times_from_energy(
        energy[: _lundeby_truncate(energy, sample_rate)], sample_rate
    )
    return {"T30": t30, "EDT": edt}


def _shared_truncation_per_band(
    reference_legs: dict[str, np.ndarray],  # leg name → (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],
    onset_rel_db: float,
) -> list[tuple[int, str]]:
    """Per eval band, the truncation index SHARED by every leg of a paired
    comparison, plus the name of the leg that set it (AC-17).

    Why this exists
    ---------------
    `_lundeby_truncate` is noise-floor dependent, and the noise floor IS this
    study's independent variable (ray budget). Truncating each leg at its own index
    integrates the legs over DIFFERENT limits, manufacturing a metric difference
    with no acoustic cause: with identical decay and only the floor scaled by
    sqrt(40) (the 5,000:200,000 ray ratio), the noisier leg read T30 12-16 % short
    at a -50 dB floor and ~51 % short at -30 dB — 3-10x the project's own declared
    T30 JND (`d0b_t30_jnd_frac` = 0.05). All legs are the same room, so ISO-3382
    band metrics are only comparable over a common integration limit.

    The index is the MINIMUM over the reference legs, so no leg is ever integrated
    into its own noise floor. Where the shared window is too short to support a
    metric, that metric is NaN with a reason (F-21) rather than a biased number.

    `reference_legs` is passed EXPLICITLY and must contain only PHYSICAL legs
    (`low`/`high`) — never a model output (RD-43). Deriving the window from `pred`
    would let a degenerate prediction shorten the window used to measure its own
    ground truth, compressing the legs together and shrinking |pred - high|: a
    worse model scoring better by corrupting the measurement of its target.

    Each leg is onset-aligned first (AC-02), so the returned indices are counted
    from each leg's own direct arrival and are directly comparable.
    """
    if not reference_legs:
        raise ValueError(
            "_shared_truncation_per_band requires at least one reference leg; "
            "an empty leg set would leave the integration limit undefined."
        )
    aligned = {
        leg: ir[_find_onset(ir, onset_rel_db):] for leg, ir in reference_legs.items()
    }
    out: list[tuple[int, str]] = []
    for fc in iso_eval_freqs:
        per_leg = {
            leg: _lundeby_truncate(_band_energy(ir, float(fc), sample_rate), sample_rate)
            if ir.shape[0] >= _MIN_FILTER_SAMPLES else 0
            for leg, ir in aligned.items()
        }
        limiting = min(per_leg, key=lambda leg: per_leg[leg])
        out.append((per_leg[limiting], limiting))
    return out


def _iso3382_band_metrics(
    ir_w: np.ndarray,   # (T,) W-channel waveform
    fc: float,
    sample_rate: int,
    *,
    band_resolvability_margin: float,
    trunc_idx: int | None = None,
    trunc_source: str | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    T30, EDT, C50 for a single octave band centered at fc Hz.

    Returns (values, nan_reasons): a metric whose window is degenerate is NaN,
    and nan_reasons carries WHY for every NaN metric — surfaced up through
    `channel_band_avg_metrics` to the eval drop log, so no band leaves a result
    silently (F-21).
    """
    all_metrics = ("T30", "EDT", "C50")
    # Guard: the zero-phase 4th-order octave filter (sosfiltfilt, padlen=27) rejects
    # records shorter than ~28 samples. A very-late onset trim (AC-07) can leave too
    # few samples; a sub-millisecond record has no valid room metric anyway → nan.
    if ir_w.shape[0] < _MIN_FILTER_SAMPLES:
        reason = f"record shorter than the {_MIN_FILTER_SAMPLES}-sample octave-filter minimum"
        return {m: float("nan") for m in all_metrics}, {m: reason for m in all_metrics}
    energy = _band_energy(ir_w, fc, sample_rate)  # (T,)

    # Lundeby truncation before Schroeder. `trunc_idx` is supplied by a paired
    # caller so every leg integrates over the SAME limit (AC-17); a single-IR caller
    # passes None and gets this IR's own index.
    if trunc_idx is None:
        trunc_idx = _lundeby_truncate(energy, sample_rate)
        trunc_source = "self"
    trunc_idx = min(trunc_idx, len(energy))
    window_note = (
        "" if trunc_source in (None, "self")
        else f" [shared integration window, set by the {trunc_source} leg]"
    )
    energy_trunc = energy[:trunc_idx]

    if len(energy_trunc) < 2:
        reason = f"Lundeby truncation leaves <2 samples{window_note}"
        return {m: float("nan") for m in all_metrics}, {m: reason for m in all_metrics}

    # Schroeder backward integration on truncated portion
    t30, t30_reason, edt, edt_reason = _decay_times_from_energy(energy_trunc, sample_rate)

    # ── Resolvability floor (AC-26/AC-27, replacing AC-23's scalar) ────────────
    #
    # A decay the band physically cannot resolve is unscored, not small. The floor
    # is the FILTER'S OWN decay in this band times a declared margin — a per-(metric,
    # band) constant, measured through this same path, and INDEPENDENT of the value
    # being tested.
    #
    # The independence is the whole correction. The previous floor was one scalar
    # (0.05 s) compared against the FITTED value, which censors the low tail of the
    # estimator and biases the surviving mean upward. MEASURED, 200 realizations at
    # 500+1000 Hz, noise-free, true T60 = 0.06 s — the Eyring median of the split the
    # threshold was written for:
    #
    #     floor 0.05 s  : 53.0 % of EDT realizations suppressed,
    #                     survivor mean 0.0639 s vs 0.0487 s → +31.2 % bias
    #     filter floor  :  1.0 % suppressed, 0.0491 s → +0.8 %
    #
    # It also refuted base.yaml's own claim that 0.05 s was "a degenerate-case guard,
    # not an active suppression threshold": for that split it suppressed the majority.
    #
    # What the floor does NOT fix, and must not pretend to: EDT below ~0.15 s is
    # VARIANCE-limited, not ringing-limited (sd 24-31 % of T60 vs 6-10 % for T30, and
    # biased ~19 % low at 0.06 s). That is an estimator property to DISCLOSE — see
    # `metric_edt_variance_limited_s` and the per-split counts in the eval output —
    # not something a threshold can remove.
    floors = _band_resolvable_decay_s(float(fc), sample_rate)
    for _name, _val in (("T30", t30), ("EDT", edt)):
        floor = band_resolvability_margin * floors[_name]
        if not np.isnan(_val) and _val < floor:
            reason = (
                f"{_name} {_val:.4f} s is below the {floor:.4f} s the {fc:g} Hz "
                f"octave band can resolve ({band_resolvability_margin:g} x the "
                f"filter's own {_name} of {floors[_name]:.4f} s) — at this decay the "
                f"fitted slope measures the filter, not the room{window_note}"
            )
            if _name == "T30":
                t30, t30_reason = float("nan"), reason
            else:
                edt, edt_reason = float("nan"), reason

    # C50: early/late split at 50 ms. The late window integrates only to the Lundeby
    # truncation index (as T30/EDT do), NOT the full record — otherwise the noise-floor
    # tail inflates `late` and, since the low-ray carrier is noisier than the high-ray
    # reference, biases the baseline-relative C50 comparison (ISO 3382-1). (AC-04)
    split = int(np.ceil(0.050 * sample_rate))
    c50_reason: str | None = None
    if split >= trunc_idx:
        c50 = float("nan")
        c50_reason = (
            f"50 ms split (sample {split}) ≥ Lundeby truncation index ({trunc_idx}) — "
            f"no late window; a sub-50 ms IR has no honest C50{window_note}"
        )
    else:
        early = float(energy[:split].sum())
        late = float(energy[split:trunc_idx].sum())
        if early > 0 and late > 0:
            c50 = float(10.0 * np.log10(early / late))
        else:
            c50 = float("nan")
            c50_reason = "zero energy in the early or late C50 window"

    values = {"T30": t30, "EDT": edt, "C50": c50}
    reasons = {
        m: r
        for m, r in (("T30", t30_reason), ("EDT", edt_reason), ("C50", c50_reason))
        if r is not None
    }
    return values, reasons


def channel_band_avg_metrics(
    ir_w: np.ndarray,             # (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],  # config.iso_eval_freqs (§7)
    onset_rel_db: float,          # config.metric_onset_rel_db (§3 metric path)
    band_resolvability_margin: float,  # config.metric_band_resolvability_margin (§3, AC-23)
    trunc_idx_per_band: list[tuple[int, str]] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Onset-align a W-channel IR to its direct arrival, then average ISO-3382 band
    metrics (T30/EDT/C50) over the evaluation bands.

    `trunc_idx_per_band` (from `_shared_truncation_per_band`) makes this leg
    integrate over a window shared with the other legs of a paired comparison
    (AC-17). Omit it for a genuinely standalone IR, which then uses its own
    Lundeby index.

    Single-IR unit (the D0b oracle probe and known-answer tests): averages THIS
    IR's surviving bands. The eval stage's paired triples do NOT use this
    average — they band-intersect across legs first (see
    compute_room_acoustic_metrics, AC-08) — but both are built on the same
    `channel_per_band_metrics`, so onset alignment (AC-02) and Lundeby
    late-window truncation (AC-04) have a single source of truth.

    Returns (values, nan_reasons). A metric is NaN iff every eval band is NaN;
    nan_reasons then aggregates the per-band reasons. A PARTIAL band drop (some
    bands NaN, average still defined) also gets a reason, prefixed "partial:", so
    the changed composition of the band average is visible, not silent (F-21).
    """
    per_band = channel_per_band_metrics(
        ir_w, sample_rate=sample_rate,
        iso_eval_freqs=iso_eval_freqs, onset_rel_db=onset_rel_db,
        band_resolvability_margin=band_resolvability_margin,
        trunc_idx_per_band=trunc_idx_per_band,
    )
    out: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for metric in ("T30", "EDT", "C50"):
        vals = [v[metric] for v, _ in per_band if not np.isnan(v[metric])]
        out[metric] = float(np.mean(vals)) if vals else float("nan")
        n_nan = len(per_band) - len(vals)
        if n_nan > 0:
            band_reasons = "; ".join(
                f"{fc:g} Hz: {r[metric]}"
                for fc, (v, r) in zip(iso_eval_freqs, per_band)
                if metric in r
            )
            prefix = "" if not vals else f"partial: {len(vals)}/{len(per_band)} bands kept — "
            reasons[metric] = f"{prefix}{n_nan}/{len(per_band)} eval bands NaN ({band_reasons})"
    return out, reasons


def channel_per_band_metrics(
    ir_w: np.ndarray,             # (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],  # config.iso_eval_freqs (§7)
    onset_rel_db: float,          # config.metric_onset_rel_db (§3 metric path)
    band_resolvability_margin: float,  # config.metric_band_resolvability_margin (§3, AC-23)
    trunc_idx_per_band: list[tuple[int, str]] | None = None,
) -> list[tuple[dict[str, float], dict[str, str]]]:
    """Onset-align a W-channel IR (AC-02), then compute per-eval-band ISO-3382
    metrics — one (values, nan_reasons) pair per band of `iso_eval_freqs`.

    The shared unit behind both consumers: `channel_band_avg_metrics` (single-IR
    band average — probe/tests) and `compute_room_acoustic_metrics` (cross-leg
    band intersection — eval, AC-08). `trunc_idx_per_band` carries the shared
    integration window of a paired comparison (AC-17); None = this IR's own."""
    if trunc_idx_per_band is not None and len(trunc_idx_per_band) != len(iso_eval_freqs):
        raise ValueError(
            f"trunc_idx_per_band has {len(trunc_idx_per_band)} entries but there are "
            f"{len(iso_eval_freqs)} eval bands — the shared integration window must be "
            f"declared per band."
        )
    onset = _find_onset(ir_w, onset_rel_db)  # align t=0 to the direct arrival
    ir_w = ir_w[onset:]
    return [
        _iso3382_band_metrics(
            ir_w, float(fc), sample_rate,
            band_resolvability_margin=band_resolvability_margin,
            trunc_idx=None if trunc_idx_per_band is None else trunc_idx_per_band[b][0],
            trunc_source=None if trunc_idx_per_band is None else trunc_idx_per_band[b][1],
        )
        for b, fc in enumerate(iso_eval_freqs)
    ]


def compute_room_acoustic_metrics(
    pred_ir: np.ndarray,        # (C, T) float32 decoded IR
    high_ref_ir: np.ndarray,    # (C, T) raw high-ray IR
    low_ref_ir: np.ndarray,     # (C, T) raw low-ray carrier
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],  # from config.iso_eval_freqs (§7)
    onset_rel_db: float,          # from config.metric_onset_rel_db (§3 metric path)
    band_resolvability_margin: float,  # from config.metric_band_resolvability_margin (§3, AC-23)
) -> tuple[
    dict[str, MetricTriple],
    dict[tuple[str, str], str],
    dict[str, tuple[int, str]],
    dict[str, dict],
]:
    """
    Standard ISO-3382 room-acoustic metrics (T30, C50, EDT) from decoded waveforms.

    Uses W-channel (ch 0), onset-aligned per IR (AC-02) so metrics are invariant to
    leading propagation-delay silence. Returns (triples, nan_reasons): one triple
    per metric, kind = match_reference (pred should match the high reference's
    value), plus (metric, leg) → reason for everything dropped (F-21).

    Every leg is band-averaged over the SAME band set, because ISO-3382
    band-averaged metrics are only comparable over a common one (AC-08: averaging
    each leg over its own surviving bands let a pred-only band drop produce a
    scored comparison mixing acoustic difference with band-composition difference).

    That band set is derived from the PHYSICAL LEGS ONLY — `low` and `high` — for
    the same reason the integration window is (AC-25, closing the second of RD-43's
    two channels). Intersecting all three legs meant a `pred` band failing the
    resolvability floor was dropped from EVERY leg's average, so a model output
    changed the reported value of its own ground truth. REPRODUCED: with high and
    low both carrying identical 0.30 s decays safely above the floor in both bands,
    replacing only PRED's 500 Hz octave with a 0.015 s decay moved HIGH's reported
    EDT from 0.2926 s to 0.2498 s — a 14.6 % change with no change to high's
    waveform. The direction is not guaranteed either way (there |pred - high| GREW),
    which is exactly why it is closed rather than argued about.

    A `pred` that is NaN in a physically-kept band is itself unscored, with a
    reason — the honest reading is "the model produced nothing measurable in a band
    the physics resolves", not "this band does not count". `low` and `high` stay
    scored over the physical band set regardless.

    All legs also share ONE Schroeder integration window per band (AC-17), likewise
    derived from the physical legs (RD-43). Returned alongside the triples so every
    scored row can record it (RD-44), not just the dropped ones.
    """
    physical_legs = ("low", "high")
    shared_trunc = _shared_truncation_per_band(
        {"low": low_ref_ir[0], "high": high_ref_ir[0]},
        sample_rate=sample_rate,
        iso_eval_freqs=iso_eval_freqs,
        onset_rel_db=onset_rel_db,
    )
    per_leg = {
        leg: channel_per_band_metrics(
            ir[0], sample_rate=sample_rate,
            iso_eval_freqs=iso_eval_freqs, onset_rel_db=onset_rel_db,
            band_resolvability_margin=band_resolvability_margin,
            trunc_idx_per_band=shared_trunc,
        )
        for leg, ir in [("pred", pred_ir), ("high", high_ref_ir), ("low", low_ref_ir)]
    }
    n_bands = len(iso_eval_freqs)
    triples: dict[str, MetricTriple] = {}
    nan_reasons: dict[tuple[str, str], str] = {}
    band_accounting: dict[str, dict] = {}
    for metric in ("T30", "EDT", "C50"):
        finite = {
            leg: [not np.isnan(bands[b][0][metric]) for b in range(n_bands)]
            for leg, bands in per_leg.items()
        }
        # The band set is the physical legs' — pred never votes (AC-25).
        kept = [b for b in range(n_bands) if all(finite[leg][b] for leg in physical_legs)]
        excluded = [b for b in range(n_bands) if b not in kept]

        leg_vals: dict[str, float] = {}
        for leg in per_leg:
            if not kept or not all(finite[leg][b] for b in kept):
                # pred reaches this branch when it is unmeasurable in a band the
                # physics resolves; low/high only when their own bands failed.
                leg_vals[leg] = float("nan")
            else:
                leg_vals[leg] = float(np.mean([per_leg[leg][b][0][metric] for b in kept]))

        band_accounting[metric] = {
            "n_bands": n_bands,
            "n_bands_kept": len(kept),
            "kept_hz": [float(iso_eval_freqs[b]) for b in kept],
            # Recorded as a DISCLOSURE, never as a band exclusion (AC-25).
            "pred_unresolved_hz": [
                float(iso_eval_freqs[b]) for b in kept if not finite["pred"][b]
            ],
        }

        if excluded:
            scope = (
                f"partial: {len(kept)}/{n_bands} bands kept from the physical legs — "
                if kept else "no eval band finite in both physical legs — "
            )
            for leg in physical_legs:
                causes = [b for b in excluded if not finite[leg][b]]
                if causes:
                    cause_str = "; ".join(
                        f"{iso_eval_freqs[b]:g} Hz: {per_leg[leg][b][1][metric]}"
                        for b in causes
                    )
                    nan_reasons[(metric, leg)] = (
                        scope + f"this leg's NaN bands are excluded from every "
                        f"leg's average ({cause_str})"
                    )
                elif not kept:
                    culprits = ", ".join(
                        l for l in physical_legs if any(not finite[l][b] for b in excluded)
                    )
                    nan_reasons[(metric, leg)] = (
                        scope + f"this leg's bands were finite; excluded by the "
                        f"other physical leg's failures ({culprits})"
                    )
        if not kept:
            nan_reasons[(metric, "pred")] = (
                "no eval band finite in both physical legs — nothing to compare against"
            )
        elif np.isnan(leg_vals["pred"]):
            unresolved = "; ".join(
                f"{iso_eval_freqs[b]:g} Hz: {per_leg['pred'][b][1][metric]}"
                for b in kept if not finite["pred"][b]
            )
            nan_reasons[(metric, "pred")] = (
                f"pred is unmeasurable in {len(band_accounting[metric]['pred_unresolved_hz'])}"
                f"/{len(kept)} of the bands the physical legs resolve, so pred is "
                f"unscored — the physical legs keep their own values, since a model "
                f"output must not change the reported value of its ground truth "
                f"(AC-25) ({unresolved})"
            )
        triples[metric] = MetricTriple(
            low=leg_vals["low"], pred=leg_vals["pred"], high=leg_vals["high"],
            kind="match_reference",
        )
    window = {
        f"{fc:g}": (idx, src)
        for fc, (idx, src) in zip(iso_eval_freqs, shared_trunc)
    }
    return triples, nan_reasons, window, band_accounting


# ---------------------------------------------------------------------------
# Energy-domain helpers — training metric-consistency proxy (§3-D4, future E2/E3)
# NOT used by the eval stage or D0b; kept here for the loss term.
# ---------------------------------------------------------------------------

def _find_iso_bands(center_freqs: list[float], iso_eval_freqs: list[float]) -> list[int]:
    """Return band indices closest to the ISO 3382 evaluation frequencies.

    `iso_eval_freqs` comes from `config.iso_eval_freqs` (§7) — the single source of
    truth for the evaluation band set; this helper never hardcodes it."""
    cf = np.array(center_freqs)
    return [int(np.argmin(np.abs(cf - float(t)))) for t in iso_eval_freqs]


def _schroeder_edr(energy_linear_1d: np.ndarray) -> np.ndarray:
    """Schroeder reverse-cumulative-sum → EDR in dB relative to t=0."""
    edr = np.cumsum(energy_linear_1d[::-1])[::-1].copy()
    edr = np.maximum(edr, 1e-30)
    return 10.0 * np.log10(edr / edr[0])


def _fit_line_in_window(edr_db: np.ndarray, t: np.ndarray, lo: float, hi: float) -> float | None:
    mask = (edr_db >= hi) & (edr_db <= lo)
    if mask.sum() < 2:
        return None
    coeffs = np.polyfit(t[mask], edr_db[mask], 1)
    return float(coeffs[0])


def _t30(edr_db: np.ndarray, frame_duration: float) -> float:
    t = np.arange(len(edr_db)) * frame_duration
    slope = _fit_line_in_window(edr_db, t, lo=-5.0, hi=-35.0)
    if slope is None or slope >= 0.0:
        return float("nan")
    return float(-60.0 / slope)


def _edt(edr_db: np.ndarray, frame_duration: float) -> float:
    t = np.arange(len(edr_db)) * frame_duration
    slope = _fit_line_in_window(edr_db, t, lo=0.0, hi=-10.0)
    if slope is None or slope >= 0.0:
        return float("nan")
    return float(-60.0 / slope)


def _c50(energy_linear_1d: np.ndarray, frame_duration: float) -> float:
    # Frame-quantized 50 ms split: ceil(0.050/frame_duration) rounds UP to the next
    # STFT frame (~53 ms at hop 512 / 48 kHz), a ~3 ms offset from the exact 50 ms
    # used by the reported waveform path. Acceptable for this training proxy; the
    # proxy-vs-standard validation (§3) accounts for the offset. See ledger AC-05.
    # RD-06: this proxy assumes t=0 = the direct arrival (no onset alignment here,
    # unlike the reported path's channel_band_avg_metrics). Valid for the energy grid,
    # whose frames already start at the IR's t=0; revisit if the proxy ever consumes
    # real renders carrying propagation delay.
    split_frame = int(np.ceil(0.050 / frame_duration))
    if split_frame >= len(energy_linear_1d):
        return float("nan")
    early = float(energy_linear_1d[:split_frame].sum())
    late = float(energy_linear_1d[split_frame:].sum())
    if early <= 0.0 or late <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(early / late))


def _metrics_from_energy(energy_db: np.ndarray, frame_duration: float) -> dict[str, float]:
    energy_linear = 10.0 ** (energy_db / 10.0)
    edr_db = _schroeder_edr(energy_linear)
    return {
        "T30": _t30(edr_db, frame_duration),
        "EDT": _edt(edr_db, frame_duration),
        "C50": _c50(energy_linear, frame_duration),
    }


def _band_avg_metrics(
    energy_ch: torch.Tensor,   # (n_bands, n_frames) dB
    iso_band_indices: list[int],
    frame_duration: float,
) -> dict[str, float]:
    per_band: list[dict] = []
    for b in iso_band_indices:
        band_db = energy_ch[b].numpy()
        per_band.append(_metrics_from_energy(band_db, frame_duration))
    result: dict[str, float] = {}
    for key in ("T30", "EDT", "C50"):
        vals = [d[key] for d in per_band if not np.isnan(d[key])]
        result[key] = float(np.mean(vals)) if vals else float("nan")
    return result
