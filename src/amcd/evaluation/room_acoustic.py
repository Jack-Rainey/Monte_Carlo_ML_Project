"""
Room-acoustic metrics T30, C50, EDT (ISO 3382).

Metric source of truth (design_spec §3): reported metrics are computed from the
**decoded waveform** via the standard ISO-3382 path — IIR octave-band filter,
Lundeby noise-floor truncation, Schroeder backward integration. Never from the
STFT energy grid directly.

Implements the objective room-acoustic metrics of `docs/research_I_paper.md`
§4.6 (background §3.1.7).

TWO DECLARED DEPARTURES FROM THE STANDARD, so "the standard ISO-3382 path" is read
as the procedure and not as a conformance claim:

* the octave filter is **not IEC 61260 class 1** — realized out-of-band rejection
  is ~-37 to -41 dB one octave out, against the 60 dB+ class 1 asks. Measured
  figures and why the order is not simply raised: `_butter_octave_filter` (AC-68);
* the band-resolvability floor is a **project-defined** criterion (the filter's own
  measured decay times a declared margin), NOT ISO 3382-2's BT > 16, and is more
  permissive than it at 500 Hz. See `configs/base.yaml`
  `metric_band_resolvability_margin` (AC-26-R6).

Both are disclosures, not defects to fix silently; each names the ledger row that
decided it.

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

Internal, but load-bearing to the paired path (private — do not import)
----------------------------------------------------------------------
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


def _filter_guard_samples(fc: float, sample_rate: int) -> int:
    """Zero-pad width for the `fc` octave band, in samples.

    ~4x the filter's own T30 in this band, so it exceeds the ringing rather than
    matching it; scales as 1/fc because the ringing does. ONE definition, shared by
    `_butter_octave_filter` (which applies it) and `_iso3382_band_metrics` (which
    refuses records shorter than it) — two expressions of one constant is the AC-24
    shape, and this is exactly the kind of value that drifts.
    """
    return int(np.ceil(48.0 / fc * sample_rate))


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


def _butter_octave_filter(
    ir_w: np.ndarray, fc: float, sample_rate: int
) -> tuple[np.ndarray, int]:
    """
    Zero-phase 4th-order Butterworth octave-band filter centered at fc Hz.
    Passband: [fc / sqrt(2), fc * sqrt(2)]. `sosfiltfilt` gives the zero-phase
    response (no group-delay offset in EDT). Returns (filtered, guard), the guard
    still attached — `_band_energy` owns what happens to it.

    Implements the octave-band decomposition of `docs/research_I_paper.md` §4.6
    (background §3.1.7).

    THE PAD IS EXPLICIT ZEROS AT BOTH ENDS, NOT scipy's DEFAULT (AC-36). An impulse
    response is silent before its direct arrival and its surroundings are 0.0 at both
    ends, so that is what the filter must see. `padtype="odd"` reflects about the
    first sample (420x the 500 Hz band's in-band energy, ~23 dB on C50) and
    `padtype="constant"` replicates the EDGE sample (-25.5 % band energy on a record
    ending at full scale); padding with explicit zeros makes "constant" replicate 0.0.

    The guard scales as 1/fc because the filter's ringing does;
    `_band_resolvable_decay_s` measures that ringing and, with
    `_DECLARED_FLOORS_48K` in tests/test_metrics.py, is one of the only two sites
    those values are written down (RR-39) — change both together.

    REALIZED SELECTIVITY — DECLARED, AND IT MEETS NO IEC 61260 CLASS (AC-68/AC-107).
    ISO 3382-1 specifies IEC 61260 class 1. MEASURED through `_band_energy` with a
    pure tone, dB re the tone's total energy:

        fc        -2 oct   -1 oct   lo edge   centre   hi edge   +1 oct   +2 oct
        500 Hz    -46.59   -37.43    -6.00     -0.00    -6.01    -38.49   -47.33
        1000 Hz   -49.59   -40.29    -6.01     -0.00    -6.01    -41.36   -50.48

    Class 1 wants ~70 dB in the far stopband and class 2 ~60 dB, so at one octave out
    this filter meets NEITHER. The order is not raised because it also sets the
    ringing `_band_resolvable_decay_s` measures — steeper skirts buy selectivity with
    a longer unresolvable floor, a research trade rather than a cleanup (F-143).

    Two further realized properties, neither what a reader would assume: the -6 dB
    band EDGES are correct (`sosfiltfilt` squares |H|², so -3 dB presents as -6 dB),
    but that squaring BREAKS power-complementarity rather than preserving it — at
    every crossover the single-pass bank sums |H|² = 1.00000 while this one sums
    |H|⁴ = 0.50000, i.e. -3.010 dB (AC-104). The realized band is ~0.9 octave: ENBW
    317.46 Hz against a nominal 353.55 (AC-108). Nil consequence today (bands are
    AVERAGED, never summed, and one scalar absorption means leakage carries no wrong
    decay); live under AC-63's per-band absorption, where a loud band's decay could
    leak into a quiet band's T30 at ~38 dB down. All pinned by
    `test_the_octave_filter_meets_its_declared_stopband_rejection`.

    THE FOLD IS ACCURATE — THERE IS NO CONVENTION BIAS ON ABSOLUTE C50 (AC-103,
    correcting F-68/F-68-R2). `filtfilt` is non-causal, so a sample at index i
    produces a response SYMMETRIC about i; the half outside the record is real
    in-band energy and `_band_energy` folds it back. Onset alignment puts the direct
    sound ON the boundary, so its FULL band energy lands post-onset — what causality
    requires. Measured against a KNOWN ANSWER (direct impulse + white exponential
    tail, for which the ideal band C50 equals the broadband C50; n=200, 500 Hz,
    DRR 20 dB):

        true T60        0.300    0.100   0.0758    0.050    0.030   s
        folded - true   +0.066   +0.214   +0.276   +0.311   -0.363  dB
        causal - true   -0.506   -1.531   -2.056   -3.310   -6.855  dB

    A previously declared "+0.0122 to +2.2294 dB magnitude bias" is RETRACTED: it was
    measured against a causal `sosfilt` comparator whose own group delay (1.85 ms at
    500 Hz) pushes energy past the 50 ms split, so it reported the comparator's bias.
    The residual is placement within a guard width, not magnitude.
    """
    nyq = sample_rate / 2.0
    f_lo = fc / 2.0 ** 0.5
    f_hi = fc * 2.0 ** 0.5
    f_lo = max(f_lo, 10.0)      # stay well above DC
    f_hi = min(f_hi, nyq * 0.99)  # stay below Nyquist
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    # ~4x the filter's own T30 in this band; zeros are cheap and the bound only has
    # to exceed the ringing, not match it. Both ends, for the reason in the docstring.
    guard = _filter_guard_samples(fc, sample_rate)
    zeros = np.zeros(guard, dtype=np.float64)
    padded = np.concatenate([zeros, np.asarray(ir_w, dtype=np.float64), zeros])
    return sosfiltfilt(sos, padded, padtype="constant").astype(np.float32), guard


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
    filtered, guard = _butter_octave_filter(ir_w, fc, sample_rate)
    energy = filtered.astype(np.float64) ** 2
    n_record = len(energy) - 2 * guard

    # FOLD THE ACAUSAL RINGING BACK IN, ONTO ITS MIRRORED SUPPORT (AC-36, F-67).
    #
    # `filtfilt` is zero-phase, so a sample at index i produces a response
    # SYMMETRIC about i. The half lying outside the record belongs to that arrival,
    # and simply trimming it discarded 50.9 % of a direct arrival's in-band energy —
    # measured 0.006728 at 500 Hz against the 0.013228 that both an interior impulse
    # and the analytic bandwidth integral give — so C50's numerator was not the ISO
    # integral. The deficit is a monotone function of DRR (-3.71 dB at d = 0.5 m,
    # -0.00 at 8 m), i.e. common-mode across LEGS but NOT across SCENES.
    #
    # WHERE the energy goes matters as much as that it is kept (F-67): it must land
    # on its mirrored support, not on sample 0. Lumping it there put it on the EDR's
    # normalization anchor and manufactured a step at t=0 of up to -2.95 dB, inside
    # the 0-to-(-10) dB window EDT is fitted over.
    #
    # Reflecting each guard sample onto its mirror image about the boundary is where
    # the energy actually came from, so it conserves energy AND leaves the EDR
    # smooth. Both ends are folded: the trailing one is not symmetry for its own
    # sake — a record truncated mid-decay (AC-22's subject) has real signal at its
    # last sample whose response extends past the end, and that energy lands in the
    # C50 late window and the Lundeby estimate.
    #
    # EVERY pad sample is folded, and the mirror index is CLAMPED into the record
    # (F-68-R3). The earlier `min(guard, n_record - 1)` form discarded every pad
    # sample beyond one record length — up to 29.7 % of band energy at n_record = 32
    # (500 Hz), with nothing logged — and re-added the outermost sample twice.
    # Conservation is now exact end to end in float64 (folded/full =
    # 1.000000000000000); the ~1e-08 through the float32 return is cast noise.
    # Pinned by `test_the_energy_fold_conserves_energy_at_every_record_length`.
    #
    # Clamping trades an energy LOSS for a PLACEMENT error rather than eliminating
    # both: below one guard width the mirror falls outside the record and the energy
    # lands on the record edge — the far end from the arrival it belongs to, since
    # leading-pad energy clamps to the LAST sample and trailing to the FIRST. At
    # n_record = 32 that puts 30.24 % of band energy on one sample. Which is why
    # `_iso3382_band_metrics` REFUSES to report a metric below one guard width
    # (AC-100/AC-106) rather than letting the approximation reach a number.
    #
    # In the normal case (n_record > guard) no clamping occurs and this is the plain
    # mirror.
    if guard > 0 and n_record > 0:
        k = np.arange(1, guard + 1)
        last = guard + n_record - 1
        # Leading pad: padded index `guard - k` is the mirror of `guard + k`.
        np.add.at(energy, np.minimum(guard + k, last), energy[guard - k])
        # Trailing pad: padded index `last + k` is the mirror of `last - k`.
        np.add.at(energy, np.maximum(last - k, guard), energy[last + k])
    return energy[guard:guard + n_record].astype(np.float32)


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
    (48 kHz): **500 Hz → T30 20.360 ms, EDT 9.556 ms; 1000 Hz → T30 10.162 ms,
    EDT 4.802 ms**.
    THIS DOCSTRING IS THE ONE PLACE THOSE VALUES ARE WRITTEN DOWN (RR-39) — they
    have moved twice as the filter path was corrected (AC-36's energy fold last),
    and every restatement elsewhere became a contradiction. Cite this function.

    WRITTEN DOWN IN EXACTLY TWO PLACES — here and `_DECLARED_FLOORS_48K` in
    tests/test_metrics.py, which pins them (AC-65/RR-119). Change both together.
    The values drifted once already through the AC-36/F-67 energy fold (T30 +13.9 %,
    EDT -18.8 % / -16.8 %) precisely because nothing asserted them; the ledger
    noticed before the code did, since AC-27's resolution quoted f·T30 = 9.85-10.18,
    which reproduces here (measured 9.88-10.18 across 125-4000 Hz).
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
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """
    T30, EDT, C50 for a single octave band centered at fc Hz.

    Returns (values, nan_reasons, resolvability):

    * `values` — a metric whose window is degenerate is NaN.
    * `nan_reasons` — WHY, for every NaN metric, surfaced up to the eval drop log
      so no band leaves a result silently (F-21).
    * `resolvability` — metric → reason, for every metric whose FITTED VALUE falls
      below what this band can resolve. **Reported, never applied here (AC-38).**

    The third return is the AC-38 correction. The floor's THRESHOLD is a
    per-(metric, band) constant and is independent of the datum, which was right;
    the DECISION `fitted_value < floor` was not, because suppressing on the value
    censors the low tail of a noisy estimator and biases the surviving mean up —
    measured, 18.8 % of T30 suppressed with +7.5 % survivor bias at true
    T60 = 0.04 s, a corner base.yaml's own declared support admits.

    Censoring is now a decision only a caller that knows the leg's ROLE may take:
    `compute_room_acoustic_metrics` discloses for the physical legs and suppresses
    for `pred` (see there for why the asymmetry is necessary rather than stylistic),
    while `channel_band_avg_metrics` — the single-IR probe unit — suppresses, as it
    always has.
    """
    all_metrics = ("T30", "EDT", "C50")
    # ── The record must be at least one GUARD WIDTH long (AC-100/AC-106) ─────────
    #
    # Two admission bounds, both numerical guards rather than experiment parameters.
    # `_MIN_FILTER_SAMPLES` is what `sosfiltfilt` needs at all (padlen=27). The
    # band-dependent one is what the ENERGY FOLD needs to be meaningful, and it was
    # missing.
    #
    # `_band_energy` folds the filter's acausal ringing back onto its mirrored
    # support. Below one guard width the mirror lands outside the record and the
    # energy is CLAMPED to the nearest in-record index — which conserves energy but
    # deposits it at the far end from the arrival it belongs to: measured at
    # n_record = 32, the last sample ends up holding 30.24 % of the band's energy,
    # 24x its neighbour, and T30 reads 0.00706 s against 0.00336 s for the same
    # signal in a long record. The alternative (discarding it, which is what this
    # code did before F-68-R3) silently lost up to 29.7 %.
    #
    # Neither belongs in a reported number, and this project does not let a result
    # leave silently. So a record too short for its own band is NaN WITH A REASON —
    # the honest third option, and the one the AC-38 machinery is built to carry.
    #
    # INERT on both declared configs: the shortest post-onset W record over the
    # canonical dry run's 58 scene-legs is 10981 samples against guards of 4608
    # (500 Hz) and 2304 (1000 Hz), and base.yaml's ir_duration 4.25 s is 17x larger
    # again. It bounds a path that was reachable in principle, not one that fires.
    guard_samples = _filter_guard_samples(fc, sample_rate)
    if ir_w.shape[0] < max(_MIN_FILTER_SAMPLES, guard_samples):
        reason = (
            f"record is {ir_w.shape[0]} samples, shorter than the {guard_samples}-sample "
            f"zero-pad guard the {fc:g} Hz octave filter needs "
            f"({1000.0 * guard_samples / sample_rate:.0f} ms). Below one guard width the "
            f"acausal ringing folds to a mirror outside the record and is clamped to the "
            f"record edge, which conserves energy but deposits it at the far end from its "
            f"arrival — measured 30.2 % of band energy on the final sample at 32 samples. "
            f"Unmeasurable rather than approximated (AC-100/AC-106)"
        )
        return (
            {m: float("nan") for m in all_metrics},
            {m: reason for m in all_metrics},
            {},
        )
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
        return (
            {m: float("nan") for m in all_metrics},
            {m: reason for m in all_metrics},
            {},
        )

    # Schroeder backward integration on truncated portion
    t30, t30_reason, edt, edt_reason = _decay_times_from_energy(energy_trunc, sample_rate)

    # ── Resolvability floor: MEASURED AND REPORTED, NOT APPLIED (AC-26/27/38) ──
    #
    # A decay below what the band can resolve is a real caveat, and the floor is the
    # right instrument for it: the FILTER'S OWN decay in this band times a declared
    # margin — a per-(metric, band) constant, measured through this same path, and
    # independent of the value being tested. AC-26 fixed the THRESHOLD that way and
    # the margin of 2.0 is independently calibrated (see base.yaml).
    #
    # AC-38 is what the threshold's independence did NOT buy: the DECISION was still
    # `fitted_value < floor`, and suppressing a value because of its own magnitude
    # censors the low tail of a noisy estimator, biasing the surviving mean upward.
    # MEASURED, 200 realizations x 2 bands (500 Hz shown; re-derived this cycle,
    # margin 2.0, seed 20260811 — see configs/base.yaml for the full table and both
    # bands). Columns: what suppression discards, and what it does to what survives.
    #
    #     true T60   T30 suppressed   error of the SUPPRESSED mean   of the DISCLOSED
    #      0.02 s       200/200          no value at all                  +13.4 %
    #      0.03 s       192/200             +38.9 %                        +6.4 %
    #      0.04 s        79/200             +14.8 %                        +5.7 %
    #      0.05 s        12/200              +5.4 %                        +3.7 %
    #      0.06 s         0/200              +3.7 %                        +3.7 %
    #
    # and base.yaml's declared support admits Eyring T60 = 0.0179 s, below the 500 Hz
    # T30 floor. So the verdict is RETURNED here and applied by whoever knows the
    # leg's role; this function no longer decides. The same reasoning already
    # governs `metric_edt_variance_limited_s` (RD-78) and drove RD-46's reversal:
    # censoring an estimator on its own value is a distortion, not a disclosure.
    #
    # What no threshold can fix, and this must not pretend to: EDT below ~0.15 s is
    # VARIANCE-limited, not ringing-limited (sd 24-31 % of T60 vs 6-10 % for T30,
    # biased ~19 % low at 0.06 s) — disclosed via `metric_edt_variance_limited_s`.
    # C50 is computed BEFORE the resolvability verdict because the verdict cites it
    # (AC-39): the band's own clarity is what distinguishes the two mechanisms that
    # can make a decay read short, and the reader is owed which one applies.
    #
    # Early/late split at 50 ms. The late window integrates only to the Lundeby
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

    # ── The verdict's WORDING names both mechanisms, not one (AC-39) ───────────
    #
    # The old text said "at this decay the fitted slope measures the filter, not the
    # room". That is one cause, and at a corner base.yaml's declared support admits
    # (3x3x2.4 m, alpha 0.98, d 2 m, rendered T60 0.0758 s) it is the WRONG one: the
    # high leg reads EDT 0.0121 s at 500 Hz and 0.0065 s at 1000 Hz, but T30 reads
    # 0.0852 / 0.0757 s — the room decay measures fine. What makes the first 10 dB
    # short there is the DIRECT ARRIVAL dominating (C50 = 45.6 / 51.1 dB), which is
    # an ISO-3382-real quantity, not a filter artifact. Both bands then dropped under
    # a reason that told the E1 reader the wrong cause.
    #
    # So EDT names both possibilities and carries the band's C50, which is what lets
    # a reader tell them apart: a high C50 says direct-dominated, a low one says the
    # decay really is at the filter's own scale. T30 regresses -5 to -35 dB, well
    # past the direct arrival's influence, so its wording stays filter-oriented.
    c50_note = "C50 unscored in this band" if np.isnan(c50) else f"the band's C50 is {c50:+.1f} dB"
    _mechanism = {
        "T30": "at this decay the fitted slope measures the filter's own ringing "
               "rather than the room",
        "EDT": "the first 10 dB is set by the direct arrival and/or the filter's own "
               "ringing at this decay — a large C50 indicates the former, which is an "
               "ISO-3382-real quantity and not a filter artifact",
    }
    resolvability: dict[str, str] = {}
    floors = _band_resolvable_decay_s(float(fc), sample_rate)
    for _name, _val in (("T30", t30), ("EDT", edt)):
        floor = band_resolvability_margin * floors[_name]
        if not np.isnan(_val) and _val < floor:
            resolvability[_name] = (
                f"{_name} {_val:.4f} s is below the {floor:.4f} s the {fc:g} Hz "
                f"octave band can resolve ({band_resolvability_margin:g} x the "
                f"filter's own {_name} of {floors[_name]:.4f} s) — "
                f"{_mechanism[_name]} ({c50_note}){window_note}"
            )

    # ── C50 inherits T30's verdict, and ONLY T30's (AC-42) ────────────────────
    #
    # C50 had no resolvability entry at all: guarded only against `late == 0`, a
    # degenerate 5 ms-decay pred reported +148 dB at 500 Hz and +266 dB at 1000 Hz
    # as SCORED absolutes, pooled into the split's `pred_mean` and its CI.
    #
    # The instrument is T30's verdict because it is the one that says the LATE
    # WINDOW carries no measurable room decay — and the late window is C50's
    # denominator. Where T30 is unresolvable the energy after 50 ms is the filter's
    # own ringing rather than reverberation, so the ratio is not the ISO-3382
    # quantity C50 is defined to be.
    #
    # NOT EDT's verdict, and not a numeric-precision bound. Both were tried and
    # both are wrong here:
    #   * EDT — at the corner AC-39 names (alpha 0.98, d 1-2 m) EDT is below its
    #     floor while C50 measures +48.8 to +55.3 dB and T30 measures fine. That
    #     C50 is the DIRECT ARRIVAL, ISO-3382-real, and AC-39 exists to affirm it.
    #   * a float32 residue bound on `late` was tried and rejected — it reduces to a
    #     C50 ceiling set by the direct arrival's crest factor (57.6-62.2 dB, moving
    #     the WRONG way with arrival shape), and it fired on the PHYSICAL legs with a
    #     probability monotone in absorption, i.e. confounded with
    #     `test_material_shift`'s own axis (AC-42).
    #
    # Physical legs are never censored by this: `compute_room_acoustic_metrics`
    # suppresses only `pred`, and only in bands the physical legs themselves
    # resolve. A band where NO leg resolves the decay stays scored for everyone.
    if "T30" in resolvability and not np.isnan(c50):
        resolvability["C50"] = (
            f"C50 {c50:+.1f} dB is unscored because the {fc:g} Hz band cannot "
            f"resolve this decay: T30 is below the band's floor, so the energy "
            f"after the 50 ms split — C50's denominator — is the filter's own "
            f"ringing rather than room reverberation (AC-42){window_note}"
        )

    values = {"T30": t30, "EDT": edt, "C50": c50}
    reasons = {
        m: r
        for m, r in (("T30", t30_reason), ("EDT", edt_reason), ("C50", c50_reason))
        if r is not None
    }
    # A metric already NaN for a HARDER reason (degenerate window, non-decaying EDR)
    # is not additionally "resolvability-limited" — that would double-count it in the
    # accounting and give the drop log two reasons for one drop.
    resolvability = {m: r for m, r in resolvability.items() if not np.isnan(values[m])}
    return values, reasons, resolvability


def channel_band_avg_metrics(
    ir_w: np.ndarray,             # (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],  # config.iso_eval_freqs (§7)
    onset_rel_db: float,          # config.metric_onset_rel_db (§3 metric path)
    band_resolvability_margin: float,  # config.metric_band_resolvability_margin (§3, AC-26/AC-27)
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

    THIS UNIT STILL SUPPRESSES BELOW THE RESOLVABILITY FLOOR (AC-38). Its callers
    are single-IR probes — the D0b oracle in `diagnostics/probe.py` and known-answer
    tests — where there is no leg role to reason about and no paired population
    whose mean could be biased, so the AC-38 argument (censoring an estimator on its
    own value distorts a reported split mean) does not apply. `signature and
    behaviour both frozen` is deliberate rather than inherited: the reported path is
    `compute_room_acoustic_metrics`, which discloses instead, and the divergence
    between the two is written up in the lane-M inbox as RD-190, against
    `probe.py:256,262`
    so the D0b half is assigned to someone rather than silently left behind.
    """
    per_band = channel_per_band_metrics(
        ir_w, sample_rate=sample_rate,
        iso_eval_freqs=iso_eval_freqs, onset_rel_db=onset_rel_db,
        band_resolvability_margin=band_resolvability_margin,
        trunc_idx_per_band=trunc_idx_per_band,
    )
    # Apply the floor here, so this unit's contract is exactly what it always was.
    suppressed = [
        ({**v, **{m: float("nan") for m in res}}, {**r, **res})
        for v, r, res in per_band
    ]
    out: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for metric in ("T30", "EDT", "C50"):
        vals = [v[metric] for v, _ in suppressed if not np.isnan(v[metric])]
        out[metric] = float(np.mean(vals)) if vals else float("nan")
        n_nan = len(suppressed) - len(vals)
        if n_nan > 0:
            band_reasons = "; ".join(
                f"{fc:g} Hz: {r[metric]}"
                for fc, (v, r) in zip(iso_eval_freqs, suppressed)
                if metric in r
            )
            prefix = "" if not vals else f"partial: {len(vals)}/{len(suppressed)} bands kept — "
            reasons[metric] = f"{prefix}{n_nan}/{len(suppressed)} eval bands NaN ({band_reasons})"
    return out, reasons


def channel_per_band_metrics(
    ir_w: np.ndarray,             # (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],  # config.iso_eval_freqs (§7)
    onset_rel_db: float,          # config.metric_onset_rel_db (§3 metric path)
    band_resolvability_margin: float,  # config.metric_band_resolvability_margin (§3, AC-26/AC-27)
    trunc_idx_per_band: list[tuple[int, str]] | None = None,
) -> list[tuple[dict[str, float], dict[str, str], dict[str, str]]]:
    """Onset-align a W-channel IR (AC-02), then compute per-eval-band ISO-3382
    metrics — one (values, nan_reasons, resolvability) triple per band of
    `iso_eval_freqs`.

    The shared unit behind both consumers: `channel_band_avg_metrics` (single-IR
    band average — probe/tests) and `compute_room_acoustic_metrics` (cross-leg
    band intersection — eval, AC-08). `trunc_idx_per_band` carries the shared
    integration window of a paired comparison (AC-17); None = this IR's own.

    It REPORTS the resolvability verdict and applies nothing (AC-38) — the two
    consumers above answer differently, so the decision cannot live here."""
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
    band_resolvability_margin: float,  # from config.metric_band_resolvability_margin (§3, AC-26/AC-27)
) -> tuple[
    dict[str, MetricTriple],
    dict[tuple[str, str], str],
    dict[str, tuple[int, str]],
    dict[str, dict],
]:
    """
    Standard ISO-3382 room-acoustic metrics (T30, C50, EDT) from decoded waveforms.

    Uses W-channel (ch 0), onset-aligned per IR (AC-02) so metrics are invariant to
    leading propagation-delay silence.

    Returns a 4-TUPLE (RR-85 — this said "(triples, nan_reasons)" while returning
    four values, and never named `band_accounting` at all, which is the structure
    carrying the caveat columns into `ci_table.csv`):

    1. `triples` — {metric: MetricTriple}, one per metric in (T30, EDT, C50), with
       `kind="match_reference"` (pred should match the HIGH reference's value).
       T30/EDT in SECONDS, C50 in dB. NaN where unscored, with a reason in (2).
    2. `nan_reasons` — {(metric, leg): reason}. Covers two cases, not one: a leg
       that is UNSCORED, and a leg that is SCORED but carries the AC-38
       resolvability caveat. `evaluator.py`'s drop sweep handles both (a reason on
       a finite leg is logged as a partial intra-leg drop). Legs are
       "pred"/"high"/"low".
    3. `window` — {band: (sample_index, source_leg)}, the shared Schroeder
       integration limit per band (AC-17). Keyed by `f"{fc:g}"` — a STRING, e.g.
       "500" — and valued by the truncation index in SAMPLES from that leg's own
       onset, plus the name of the physical leg that set it (the minimum). Returned
       so every scored row can record its window, not only the dropped ones (RD-44).
    4. `band_accounting` — {metric: {...}}: `n_bands`, `n_bands_kept`, `kept_hz`,
       `pred_unresolved_hz`, `resolvability_limited_hz` and
       `pred_unresolved_in_floor_limited_hz` (RD-93's pred-side selection detector).
       Frequencies in Hz as floats. These are what reach `metrics.parquet` and, via
       lane P, `ci_table.csv`'s caveat columns — so this is the load-bearing return,
       not an extra.

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
        # ── The resolvability floor, applied PER LEG ROLE (AC-38) ──────────────
        #
        # `channel_per_band_metrics` reports the verdict and applies nothing, so
        # this is where the two roles diverge:
        #
        #   physical legs (low/high) — DISCLOSED. The value is reported and the
        #     band counted. Suppressing here censored the estimator on its own
        #     magnitude and biased the reported split mean upward (AC-38: 18.8 %
        #     of T30 suppressed, +7.5 % survivor bias at true T60 = 0.04 s).
        #
        #   pred — SUPPRESSED, but ONLY IN A BAND THE PHYSICAL LEGS RESOLVE. That
        #     qualifier is AC-25's own wording ("the model produced nothing
        #     measurable in a band THE PHYSICS RESOLVES") and it is load-bearing,
        #     not decoration. Suppressing pred wherever it fell below the floor —
        #     including in bands the physical legs cannot resolve either — MOVED the
        #     bias instead of removing it, MEASURED on the canonical dry run:
        #     test_material_shift EDT went n_scored 3 → 2, improvement_mdes 0.0281 →
        #     N/A, and pred_mean ROSE 0.0649 → 0.0858 s because the scene it dropped
        #     was the low one. Precisely the pred-side selection F-70 records,
        #     enlarged by the very change meant to remove a bias (RD-93).
        #
        #     Where NO leg can resolve the band, the paired comparison is still
        #     like-for-like — both legs are measured by the same instrument with the
        #     same limitation — so the honest act is to disclose all three and keep
        #     the datum. Where the physics DOES resolve the band and only pred
        #     failed, that is a model failure and stays suppressed: this is what
        #     keeps AC-25's guarantee and gives AC-42 its guard against a degenerate
        #     prediction reporting +203 dB C50 as a scored absolute.
        def _why(leg: str, b: int, _m: str = metric) -> str:
            """Why `leg` has no value for this metric in band `b`.

            Two dicts can hold it now: a hard NaN reason from the estimator, or
            the AC-38 resolvability verdict (which only `pred` acts on). Looking
            in one alone would KeyError exactly when the floor is what suppressed
            pred — the case AC-42's degenerate prediction produces.
            """
            values, reasons, resolvability = per_leg[leg][b]
            return reasons.get(_m) or resolvability.get(_m) or "reason not recorded"

        physical_floor_limited = [
            any(metric in per_leg[leg][b][2] for leg in physical_legs)
            for b in range(n_bands)
        ]
        finite = {
            leg: [
                not np.isnan(bands[b][0][metric])
                and not (
                    leg == "pred"
                    and metric in bands[b][2]
                    and not physical_floor_limited[b]
                )
                for b in range(n_bands)
            ]
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
            # AC-38: bands the PHYSICAL legs report despite being below the floor.
            # Reported, not suppressed — but a reader must be able to see which
            # numbers carry the caveat, or the disclosure is only a code comment.
            "resolvability_limited_hz": [
                float(iso_eval_freqs[b]) for b in kept if physical_floor_limited[b]
            ],
            # RD-93's detector. The floor alone can no longer put a band here —
            # pred is only floor-suppressed where the physical legs DO resolve the
            # band — so a non-zero count means pred failed for a HARD reason (a
            # non-decaying EDR, a degenerate window) in a band that is itself
            # floor-limited. Those scenes do leave `paired_improvement`, so this is
            # the residual pred-side selection F-70 has to bound, now separated
            # from the far larger one AC-38 would otherwise have created.
            "pred_unresolved_in_floor_limited_hz": [
                float(iso_eval_freqs[b])
                for b in kept
                if not finite["pred"][b] and physical_floor_limited[b]
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
                        f"{iso_eval_freqs[b]:g} Hz: {_why(leg, b)}" for b in causes
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
        # AC-38: a physical leg that REPORTS a floor-limited value still owes the
        # reader a reason. It is a caveat on a scored number, not a drop, so it is
        # attached to a finite leg — which `evaluator.py`'s drop sweep already
        # handles ("A reason on a FINITE leg is a partial intra-leg drop ... logged
        # too"). Without this the disclosure would exist only as a count.
        disclosed = [b for b in kept if physical_floor_limited[b]]
        if disclosed:
            # All three legs, not just the physical ones: where the band is
            # floor-limited pred is disclosed too, and a reader comparing a pred
            # absolute against a high absolute must see the caveat on both.
            for leg in per_leg:
                bands = [b for b in disclosed if metric in per_leg[leg][b][2]]
                if not bands:
                    continue
                detail = "; ".join(
                    f"{iso_eval_freqs[b]:g} Hz: {per_leg[leg][b][2][metric]}"
                    for b in bands
                )
                disclosure = (
                    f"resolvability-limited but REPORTED, not suppressed (AC-38): "
                    f"{len(bands)}/{len(kept)} kept bands sit below what the band "
                    f"resolves. Suppressing them censored the estimator on its own "
                    f"value and biased this split's mean upward, so the value is "
                    f"reported with this caveat instead ({detail})"
                )
                # APPEND, never overwrite (F-M9). The same (metric, leg) key may
                # already carry a band-EXCLUSION reason written above, and
                # `evaluator.py`'s drop sweep forwards exactly one reason per
                # (metric, leg) to `drops.csv` — so assigning here would silently
                # delete the exclusion, which is the harder fact of the two. No
                # instance is constructible inside base.yaml's declared support
                # (a Schroeder EDR is monotone, and Lundeby's floor makes the
                # "<2 samples" branch unreachable), so this is a guard against a
                # path that exists rather than a fix for one that fires.
                prior = nan_reasons.get((metric, leg))
                nan_reasons[(metric, leg)] = (
                    disclosure if prior is None else f"{prior} | ALSO: {disclosure}"
                )

        if not kept:
            nan_reasons[(metric, "pred")] = (
                "no eval band finite in both physical legs — nothing to compare against"
            )
        elif np.isnan(leg_vals["pred"]):
            unresolved = "; ".join(
                f"{iso_eval_freqs[b]:g} Hz: {_why('pred', b)}"
                for b in kept if not finite["pred"][b]
            )
            # RD-93: when the band pred failed in is ITSELF floor-limited, this
            # scene leaves the paired comparison only because AC-38 kept that band.
            # Said in the drop log, so the selection is visible per row and not
            # only in an aggregate nobody computes yet (F-70).
            overlap = band_accounting[metric]["pred_unresolved_in_floor_limited_hz"]
            overlap_note = (
                ""
                if not overlap
                else (
                    f" NOTE: {len(overlap)} of these ({', '.join(f'{f:g}' for f in overlap)}"
                    f" Hz) are bands the physical legs report only because the "
                    f"resolvability floor now discloses instead of suppressing "
                    f"(AC-38) — before that change this band left every leg's average "
                    f"and this scene stayed in the paired comparison. It is now "
                    f"excluded from it, which enlarges the pred-side selection F-70 "
                    f"records."
                )
            )
            nan_reasons[(metric, "pred")] = (
                f"pred is unmeasurable in {len(band_accounting[metric]['pred_unresolved_hz'])}"
                f"/{len(kept)} of the bands the physical legs resolve, so pred is "
                f"unscored — the physical legs keep their own values, since a model "
                f"output must not change the reported value of its ground truth "
                f"(AC-25) ({unresolved}).{overlap_note}"
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
    # RD-06: this proxy assumes t=0 is the direct arrival — no onset alignment, unlike
    # the reported waveform path `compute_room_acoustic_metrics`, which `evaluator.py`
    # calls. (`channel_band_avg_metrics` is the single-IR D0b probe unit, not the
    # reported path — AC-70.) Valid for the energy grid, whose frames already start at
    # the IR's t=0; revisit if the proxy ever consumes renders carrying propagation
    # delay.
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
