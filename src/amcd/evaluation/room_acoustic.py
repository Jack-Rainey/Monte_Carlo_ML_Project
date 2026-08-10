"""
Room-acoustic metrics T30, C50, EDT (ISO 3382).

Metric source of truth (design_spec §3): reported metrics are computed from the
**decoded waveform** via the standard ISO-3382 path — IIR octave-band filter,
Lundeby noise-floor truncation, Schroeder backward integration. Never from the
STFT energy grid directly.

Public API
----------
compute_room_acoustic_metrics(pred_ir, high_ref_ir, low_ref_ir, *, sample_rate,
                               iso_eval_freqs, onset_rel_db, min_measurable_t60_s)
    Standard ISO-3382 waveform path for the eval stage (pred/high/low in one
    call) → (metric triples, NaN reasons, shared integration window); legs
    band-averaged over the cross-leg-shared band set (AC-08) and integrated over
    the cross-leg-shared Schroeder window (AC-17).
channel_band_avg_metrics(ir_w, *, sample_rate, iso_eval_freqs, onset_rel_db,
                         min_measurable_t60_s, trunc_idx_per_band=None)
    Single-IR onset-aligned band-averaged metrics → (values, NaN reasons); the
    D0b oracle probe's unit. Pass `trunc_idx_per_band` to join a paired
    comparison's shared window.
channel_per_band_metrics(ir_w, *, sample_rate, iso_eval_freqs, onset_rel_db,
                         min_measurable_t60_s, trunc_idx_per_band=None)
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
    """
    nyq = sample_rate / 2.0
    f_lo = fc / 2.0 ** 0.5
    f_hi = fc * 2.0 ** 0.5
    f_lo = max(f_lo, 10.0)      # stay well above DC
    f_hi = min(f_hi, nyq * 0.99)  # stay below Nyquist
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, ir_w).astype(np.float32)


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
    min_measurable_t60_s: float,
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

    # Measurability floor (AC-23). Below a few tens of ms the zero-phase 4th-order
    # octave filter's own ringing is comparable to the decay itself, so the fitted
    # slope measures the FILTER, not the room: at a true T60 of 0.06 s the 500/1000 Hz
    # bands read EDT +89 %, and +21 % at 0.10 s. `test_material_shift` is exactly this
    # regime (Eyring T60 median 0.059 s), so an unflagged number there would be an
    # estimator artifact reported as a physical quantity. Suppress rather than report
    # — a decay time the band cannot resolve is unscored, not small.
    for _name, _val, _set in (("T30", t30, "t30"), ("EDT", edt, "edt")):
        if not np.isnan(_val) and _val < min_measurable_t60_s:
            reason = (
                f"{_name} {_val:.4f} s is below the {min_measurable_t60_s:g} s "
                f"minimum measurable decay for the {fc:g} Hz octave band — at this "
                f"decay the filter's own ringing dominates the fitted slope"
                f"{window_note}"
            )
            if _set == "t30":
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
    min_measurable_t60_s: float,  # config.metric_min_measurable_t60_s (§3, AC-23)
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
        min_measurable_t60_s=min_measurable_t60_s,
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
    min_measurable_t60_s: float,  # config.metric_min_measurable_t60_s (§3, AC-23)
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
            min_measurable_t60_s=min_measurable_t60_s,
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
    min_measurable_t60_s: float,  # from config.metric_min_measurable_t60_s (§3, AC-23)
) -> tuple[dict[str, MetricTriple], dict[tuple[str, str], str], dict[str, tuple[int, str]]]:
    """
    Standard ISO-3382 room-acoustic metrics (T30, C50, EDT) from decoded waveforms.

    Uses W-channel (ch 0), onset-aligned per IR (AC-02) so metrics are invariant to
    leading propagation-delay silence. Returns (triples, nan_reasons): one triple
    per metric, kind = match_reference (pred should match the high reference's
    value), plus (metric, leg) → reason for everything dropped (F-21).

    Each leg is band-averaged over the SAME band set — the bands finite in all
    three legs — because ISO-3382 band-averaged metrics are only comparable over
    a common band set (AC-08: averaging each leg over its own surviving bands
    let a pred-only band drop produce a scored comparison that mixed acoustic
    difference with band-composition difference). No shared band → all legs NaN
    (unscored), with the excluded bands and their causes in nan_reasons.

    All three legs also share ONE Schroeder integration window per band (AC-17),
    derived from the PHYSICAL legs only — `low` and `high`, the two whose noise
    floor is the study's independent variable. `pred` consumes that window but never
    sets it (RD-43): a degenerate prediction must not be able to shorten the window
    used to measure its own ground truth. Returns the per-band window alongside the
    triples so every scored row can record it (RD-44), not just the dropped ones.
    """
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
            min_measurable_t60_s=min_measurable_t60_s,
            trunc_idx_per_band=shared_trunc,
        )
        for leg, ir in [("pred", pred_ir), ("high", high_ref_ir), ("low", low_ref_ir)]
    }
    n_bands = len(iso_eval_freqs)
    triples: dict[str, MetricTriple] = {}
    nan_reasons: dict[tuple[str, str], str] = {}
    for metric in ("T30", "EDT", "C50"):
        finite = {
            leg: [not np.isnan(bands[b][0][metric]) for b in range(n_bands)]
            for leg, bands in per_leg.items()
        }
        kept = [b for b in range(n_bands) if all(finite[leg][b] for leg in per_leg)]
        leg_vals = {
            leg: float(np.mean([per_leg[leg][b][0][metric] for b in kept])) if kept
            else float("nan")
            for leg in per_leg
        }
        excluded = [b for b in range(n_bands) if b not in kept]
        if excluded:
            scope = (
                f"partial: {len(kept)}/{n_bands} bands kept across all legs — " if kept
                else f"no eval band finite in all legs — "
            )
            for leg in per_leg:
                causes = [b for b in excluded if not finite[leg][b]]
                if causes:
                    cause_str = "; ".join(
                        f"{iso_eval_freqs[b]:g} Hz: {per_leg[leg][b][1][metric]}"
                        for b in causes
                    )
                    nan_reasons[(metric, leg)] = (
                        scope + f"this leg's NaN bands are excluded from EVERY "
                        f"leg's average ({cause_str})"
                    )
                elif not kept:
                    # Leg is NaN purely because other legs' bands failed — it
                    # still needs a logged reason (its consumed value is NaN).
                    culprits = ", ".join(sorted(
                        l for l in per_leg if any(not finite[l][b] for b in excluded)
                    ))
                    nan_reasons[(metric, leg)] = (
                        scope + f"this leg's bands were finite; excluded by the "
                        f"other legs' failures ({culprits})"
                    )
        triples[metric] = MetricTriple(
            low=leg_vals["low"], pred=leg_vals["pred"], high=leg_vals["high"],
            kind="match_reference",
        )
    window = {
        f"{fc:g}": (idx, src)
        for fc, (idx, src) in zip(iso_eval_freqs, shared_trunc)
    }
    return triples, nan_reasons, window


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
