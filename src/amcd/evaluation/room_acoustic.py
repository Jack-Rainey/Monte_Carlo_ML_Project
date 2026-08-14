"""
Room-acoustic metrics T30, C50, EDT (ISO 3382).

Metric source of truth (design_spec §3): reported metrics are computed from the
**decoded waveform** via the standard ISO-3382 path — IIR octave-band filter,
Lundeby noise-floor truncation, Schroeder backward integration. Never from the
STFT energy grid directly.

Implements the objective room-acoustic metrics of `docs/research_I_paper.md`
§4.6 (background §3.1.7).

TWO DECLARED DEPARTURES FROM THE STANDARD, so "the standard ISO-3382 path" reads
as the procedure and not as a conformance claim:

* the octave filter is **not IEC 61260 class 1** — see `_butter_octave_filter`
  and `metric_octave_filter.stopband_rejection_db` in `configs/base.yaml`;
* the band-resolvability floor is a **project-defined** criterion (the filter's
  own measured decay times a declared margin), not ISO 3382-2's BT > 16. See
  `configs/base.yaml` `metric_band_resolvability_margin`.

Public API
----------
compute_room_acoustic_metrics(...)
    The eval stage's entry point: pred/high/low in one call → (metric triples,
    NaN reasons, shared integration window, band accounting). Legs are
    band-averaged over the band set the PHYSICAL legs resolve and integrated
    over one cross-leg-shared Schroeder window; neither the band set nor the
    window is ever set by `pred`.
channel_band_avg_metrics(...)
    Single-IR onset-aligned band-averaged metrics → (values, NaN reasons); the
    D0b oracle probe's unit. Pass `trunc_idx_per_band` to join a paired
    comparison's shared window.
channel_per_band_metrics(...)
    Per-band (values, reasons) — the shared unit under both of the above.
find_onset(ir_w, rel_db, *, expected_sample=None, tolerance_samples=0)
    (Sample index of the direct arrival, why it is not where the detector said).
    Broadband W channel only; `rel_db` is relative to the GLOBAL peak; returns 0
    for a degenerate IR. Reached cross-stage as well as from here — the render
    stage's onset-mismatch QC criterion (`simulators/qc.py`) uses this detector
    so admission and measurement cannot disagree about where t=0 is.

The `_schroeder_edr`, `_t30`, `_edt`, `_c50`, `_metrics_from_energy`,
`_band_avg_metrics` and `_find_iso_bands` helpers work in the energy domain and
are kept for the training metric-consistency loss (design_spec §3-D4), not for
reported metrics.
"""
from __future__ import annotations

from collections import namedtuple
from functools import lru_cache

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .metric_row import MetricTriple


# Minimum record length for the zero-phase 4th-order octave filter. sosfiltfilt on a
# 4-section SOS has padlen=27, so it needs > 27 samples; 32 gives headroom. Numerical
# guard, not an experiment parameter.
_MIN_FILTER_SAMPLES = 32


#: The filter order the guard width below was calibrated against. The guard scales
#: off it rather than hardcoding it, so a config that changes `order` moves the
#: guard with it instead of silently under-padding.
_GUARD_CALIBRATED_ORDER = 4


def _filter_guard_samples(fc: float, sample_rate: int, order: int) -> int:
    """Zero-pad width for the `fc` octave band, in samples.

    ~4x the filter's own T30 in this band, so it exceeds the ringing rather than
    matching it; scales as 1/fc because the ringing does, and linearly in `order`
    because a Butterworth section's decay time does too.

    One definition, shared by `_butter_octave_filter` (which applies it) and
    `_iso3382_band_metrics` (which refuses records shorter than it).
    """
    return int(np.ceil(48.0 / fc * sample_rate * order / _GUARD_CALIBRATED_ORDER))


# ---------------------------------------------------------------------------
# Standard ISO-3382 waveform path (the reported-metric source)
# ---------------------------------------------------------------------------

def _lundeby_truncate(energy_samples: np.ndarray, sample_rate: int) -> int:
    """Sample index at which to truncate before Schroeder integration.

    Two regimes, and PRODUCTION IS THE SECOND ONE:

    1. **A measured noise floor.** Estimated from the last 10 % of the record;
       truncate at the last sample whose 10 ms-smoothed energy is 10 dB above it.
       The Lundeby-style path, taken by a record with real additive noise.
    2. **A record that ends in exact silence** — every gsound record, which is
       zero-padded from its native length out to `ir_duration`. There is no noise
       floor to measure, and estimating one anyway breaks the gain-invariance
       ISO 3382 metrics require (T30/EDT/C50 are ratios): `mean(noise_region)` is
       0.0, so the clamped threshold becomes an ABSOLUTE level and the index then
       depends on the recording's gain.

    In regime 2, truncate where band energy falls to what float64 can still
    represent RELATIVE TO ITS OWN PEAK. Below `peak * eps` the samples are the
    filter's ringing decaying into rounding error, and because peak and sample
    scale together the index is exactly gain-invariant. A bare "last non-zero
    sample" is not: how far the ringing survives before underflowing depends on
    absolute magnitude.
    """
    n = len(energy_samples)
    # 10 ms minimum: enough samples for a valid regression, never n//2 (that forces
    # the noise tail back in for any IR that decays in under half the record).
    min_samples = max(2, int(0.010 * sample_rate))

    noise_region = energy_samples[int(0.9 * n):]
    if len(noise_region) == 0 or not np.any(noise_region):
        peak = float(np.max(energy_samples)) if n else 0.0
        if peak <= 0.0:
            return min_samples
        above = np.flatnonzero(energy_samples > peak * np.finfo(np.float64).eps)
        if len(above) == 0:
            return min_samples
        return max(int(above[-1]) + 1, min_samples)

    noise_power = max(float(np.mean(noise_region)), 1e-30)
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
    ir_w: np.ndarray, fc: float, sample_rate: int, order: int
) -> tuple[np.ndarray, int]:
    """
    Zero-phase Butterworth octave-band filter centered at fc Hz, `order` per section.
    Passband: [fc / sqrt(2), fc * sqrt(2)]. `sosfiltfilt` gives the zero-phase
    response (no group-delay offset in EDT). Returns (filtered, guard), the guard
    still attached — `_band_energy` owns what happens to it.

    Implements the octave-band decomposition of `docs/research_I_paper.md` §4.6
    (background §3.1.7).

    THE PAD IS EXPLICIT ZEROS AT BOTH ENDS, NOT scipy's DEFAULT. An impulse
    response is silent before its direct arrival and its surroundings are 0.0 at
    both ends, so that is what the filter must see. `padtype="odd"` would reflect
    about the first sample and `padtype="constant"` replicates the EDGE sample;
    padding with explicit zeros makes "constant" replicate 0.0.

    SELECTIVITY MEETS NO IEC 61260 CLASS, and ISO 3382-1 specifies class 1. The
    realized rejection is ~-37 to -41 dB one octave out, against class 1's ~70 dB
    and class 2's ~60 dB. The order is NOT raised to close the gap, because it
    also sets the ringing `_band_resolvable_decay_s` measures: steeper skirts buy
    selectivity with a longer unresolvable floor. It is therefore a declared
    experiment parameter, `configs/base.yaml` `metric_octave_filter.order`, with
    the measured rejection declared beside it as `stopband_rejection_db`.

    Two realized properties a reader would not assume. The -6 dB band edges are
    correct (`sosfiltfilt` squares |H|², so -3 dB presents as -6 dB), but that
    squaring breaks power-complementarity: at a crossover this bank sums |H|⁴, i.e.
    -3.01 dB rather than 0. And the realized band is ~0.9 octave (ENBW 317 Hz
    against a nominal 354). Neither bites while one scalar absorption means
    leakage carries no wrong decay and bands are averaged rather than summed; both
    become live under per-band absorption.
    """
    nyq = sample_rate / 2.0
    f_lo = fc / 2.0 ** 0.5
    f_hi = fc * 2.0 ** 0.5
    f_lo = max(f_lo, 10.0)      # stay well above DC
    f_hi = min(f_hi, nyq * 0.99)  # stay below Nyquist
    sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=sample_rate, output="sos")
    # ~4x the filter's own T30 in this band; zeros are cheap and the bound only has
    # to exceed the ringing, not match it. Both ends, for the reason in the docstring.
    guard = _filter_guard_samples(fc, sample_rate, order)
    zeros = np.zeros(guard, dtype=np.float64)
    padded = np.concatenate([zeros, np.asarray(ir_w, dtype=np.float64), zeros])
    return sosfiltfilt(sos, padded, padtype="constant").astype(np.float32), guard


def find_onset(
    ir_w: np.ndarray,
    rel_db: float,
    *,
    expected_sample: int | None = None,
    tolerance_samples: int = 0,
) -> tuple[int, str | None]:
    """(Index of the direct-sound arrival, why it is not where the detector said).

    ISO 3382 integration starts at the direct sound, and a geometric-acoustic
    render carries dist/c of leading near-silence before it — which would
    otherwise mis-reference the C50 50 ms split and the Schroeder start. Detected
    on the broadband W channel (propagation delay is frequency-independent).
    Returns 0 for a degenerate/zero IR or if nothing crosses.

    THE DETECTOR ASSUMES THE DIRECT SOUND IS THE LOUDEST SAMPLE, AND UNDER A NOISE
    CARRIER IT IS NOT ALWAYS. The threshold sits `rel_db` below the GLOBAL peak,
    while the direct arrival's realized amplitude is its weight times a
    standard-normal draw at its own bin; a small draw puts it under a bar its own
    peak set. The miss rate is ~1 % at the DRR this study admits, but a miss lands
    on the far side of the early-reflection gap — on the first strong reflection,
    10 % of the C50 window away, moving both the EDT anchor and the Schroeder start.

    SO GEOMETRY ADJUDICATES. `expected_sample` is `floor(|src - rcv| / c * fs)`
    from the scene and the backend's declared speed: the arrival's position is
    known, not estimated. The detector still places it within `tolerance_samples`,
    because it is what knows about filter smearing and sub-sample timing; it is
    overruled only outside that window, and then the reason is returned rather
    than swallowed.

    `expected_sample=None` disables the adjudication and returns no reason — for
    probes and known-answer tests that synthesize an IR with no geometry behind
    it, never for the reported path.

    NOT COVERED: an OCCLUDED direct path, where geometry says the arrival is at
    `expected_sample` and no sound reaches there. The roadmap's non-shoebox
    families (paper §6) can produce it and a scene would have to declare the
    occlusion; every family declared today is a convex box with line of sight.
    """
    energy = ir_w.astype(np.float64) ** 2
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0.0:
        return 0, None
    threshold = peak * 10.0 ** (rel_db / 10.0)
    above = np.flatnonzero(energy >= threshold)
    detected = int(above[0]) if above.size else 0
    if expected_sample is None:
        return detected, None

    expected = int(np.clip(expected_sample, 0, max(len(energy) - 1, 0)))
    if abs(detected - expected) <= tolerance_samples:
        return detected, None

    # Re-search INSIDE the window rather than jumping straight to `expected`: the
    # arrival is usually still there and above threshold, just not the first
    # sample in the whole record that is.
    lo = max(expected - tolerance_samples, 0)
    hi = min(expected + tolerance_samples + 1, len(energy))
    local = np.flatnonzero(energy[lo:hi] >= threshold)
    chosen = lo + int(local[0]) if local.size else expected
    where = "within the window" if local.size else "nowhere in the window"
    return chosen, (
        f"onset detector landed at sample {detected} but geometry "
        f"puts the direct arrival at {expected}, a disagreement of "
        f"{abs(detected - expected)} samples against a {tolerance_samples}-sample "
        f"tolerance. The threshold is {rel_db:g} dB below the GLOBAL peak, so a "
        f"direct arrival whose carrier draw is small falls under a bar its own "
        f"peak set, and t=0 lands on a reflection. Using sample {chosen} "
        f"instead — the first crossing {where}."
    )


def _band_energy(ir_w: np.ndarray, fc: float, sample_rate: int, order: int) -> np.ndarray:
    """Octave-band energy envelope (squared band-filtered samples) — the input both
    to Lundeby truncation and to Schroeder integration. Factored out so a shared
    truncation index can be derived without computing metrics."""
    filtered, guard = _butter_octave_filter(ir_w, fc, sample_rate, order)
    energy = filtered.astype(np.float64) ** 2
    n_record = len(energy) - 2 * guard

    # FOLD THE ACAUSAL RINGING BACK IN, ONTO ITS MIRRORED SUPPORT.
    #
    # `filtfilt` is zero-phase, so a sample at index i produces a response
    # SYMMETRIC about i. The half lying outside the record is real in-band energy
    # belonging to that arrival — trimming it away costs up to half a direct
    # arrival's band energy, so C50's numerator would not be the ISO integral.
    #
    # WHERE it goes matters as much as that it is kept: onto the mirror image
    # about the boundary, which is where it came from, so the fold conserves
    # energy AND leaves the EDR smooth. Lumping it on sample 0 instead would put
    # it on the EDR's normalization anchor and manufacture a step at t=0, inside
    # the 0-to-(-10) dB window EDT is fitted over. Both ends fold: a record
    # truncated mid-decay has real signal at its last sample whose response
    # extends past the end, and that energy lands in the C50 late window.
    #
    # The mirror index is clamped into the record, which below one guard width
    # trades an energy loss for a PLACEMENT error — the energy lands on the far
    # end of the record from the arrival it belongs to. Which is why
    # `_iso3382_band_metrics` refuses to report a metric below one guard width
    # rather than letting the approximation reach a number. Above it (the normal
    # case) no clamping occurs and this is the plain mirror.
    if guard > 0 and n_record > 0:
        k = np.arange(1, guard + 1)
        last = guard + n_record - 1
        # Leading pad: padded index `guard - k` is the mirror of `guard + k`.
        np.add.at(energy, np.minimum(guard + k, last), energy[guard - k])
        # Trailing pad: padded index `last + k` is the mirror of `last - k`.
        np.add.at(energy, np.maximum(last - k, guard), energy[last + k])
    return energy[guard:guard + n_record].astype(np.float32)


#: Shape of the config block `_available_decay_range_db` reads, so the estimator
#: names its own inputs rather than a caller assembling a dict. `configs/base.yaml`
#: `metric_decay_range_fit:` holds the values and the measurements behind them.
DecayRangeFit = namedtuple(
    "DecayRangeFit", "window smoothing_cycles min_smoothing_s max_kernel_frac"
)

#: Placeholder for the one call path where the decay range is computed and then
#: never read — `_band_resolvable_decay_s`, which disables the range gate with
#: 0.0 floors. Named rather than passed as the shipped config so that path cannot
#: quietly acquire a dependence on values it must not depend on.
_RANGE_UNUSED = DecayRangeFit(
    window=(0.05, 0.40), smoothing_cycles=20.0, min_smoothing_s=0.010,
    max_kernel_frac=0.10,
)


def _available_decay_range_db(
    energy_trunc: np.ndarray, sample_rate: int, band_centre_hz: float,
    filter_order: int, fit: DecayRangeFit,
) -> float:
    """How many dB of decay the record actually holds, from the ENERGY ENVELOPE.

    Measured on the smoothed band-energy envelope rather than on the Schroeder
    curve, because a backward integral always terminates at -inf on its last
    sample: any range estimated from it tracks the record LENGTH rather than the
    room's decay, and saturates instead of falling as truncation worsens.

    The envelope has no such terminal artifact — it simply decays at 60/T60 dB
    per second — so fitting its slope recovers the decay rate, and the range the
    window holds is `60 * window_s / T60`.

    ONE FIT OVER AN EARLY WINDOW, and the window's END is the load-bearing part.
    A gsound record is not a single exponential: it decays at the room's rate and
    then falls off a terminal synthesis taper an order of magnitude steeper, so a
    fit that reaches the knee reports a decay far faster than the room's and a
    range far larger than the record holds. Ending the fit early keeps it inside
    the room decay for any knee AT OR BEYOND the window's end, which a
    knee-position-dependent trim does not: over synthetic decays with the knee
    swept from 50 % to 100 % of the record, a fit running to 70 % over-reads by up
    to 204 % when the knee arrives at 50 %. Over the reported bands at shippable
    window lengths this window scores every case, at 9.1 % mean error and a worst
    over-read of +12.5 %.

    THE CLAIM STOPS THERE, and the boundary is measured rather than assumed. A
    knee INSIDE the fit window is not covered and is not safe: measured worst
    over-reads are +531 % at knee 0.35, +683 % at 0.25 and +901 % at 0.10, all
    admitted against the 45 dB floor. Production does not reach that regime — gsound's ~10x taper and
    Lundeby truncation put the knee near 86 % of the truncated record — so this is
    a bound on what the construction guarantees, not a live defect.

    `scripts/decay_range_probe.py` measures all of it, including the knees inside
    the window, and against the shallowest-of-K construction this replaced.

    KNOWN LIMIT, not yet closed: on records short relative to the band the
    estimate is biased low, and because the smoothing kernel scales as 1/fc the
    bias is larger in the low bands — at a 0.10 s window, -40.6 % at 500 Hz
    against -8.5 % at 1000 Hz. Records shorten with absorption, which is
    `test_material_shift`'s own independent variable, so the residual is a
    selection effect on that split rather than a random error. It does not bite at
    E1's declared operating points, which sit far above the floor. The decisive
    measurement is an admit-rate sweep over matched populations whose true range is
    fixed at the ISO threshold; it has not been run (ledger AC-188).

    A non-decaying envelope returns NaN, which `_admissible` treats as
    unmeasurable rather than as a pass.
    """
    energy = np.asarray(energy_trunc, dtype=np.float64)
    n = len(energy)
    smoothing_s = max(fit.min_smoothing_s, fit.smoothing_cycles / band_centre_hz)
    # THE KERNEL IS AN ABSOLUTE DURATION AND THE RECORD IS NOT, so on a short
    # record the two collide: 20 cycles is 160 ms at 125 Hz, and D0b already
    # scores 10 ms windows. Two consequences, both handled here.
    #
    # `np.convolve(..., "same")` returns the longer of its arguments, so a kernel
    # longer than the record silently returns an array of the KERNEL's length and
    # every index derived from `n` below addresses a different array than the one
    # it was computed for.
    #
    # And a kernel that is a large fraction of the record leaves no span to fit a
    # slope over. So the kernel is capped at a fraction of the record as well: on
    # a long record it is the full 20 cycles, and on a short one it shortens —
    # noisier, but a noisy estimate beats refusing a scene, and refusals here are
    # not random. Records shorten with absorption, which is `test_material_shift`'s
    # own independent variable, so a length-dependent refusal selects on the very
    # axis that split exists to measure.
    win = max(1, min(int(smoothing_s * sample_rate),
                     int(fit.max_kernel_frac * n)))

    # EDGE-NORMALISED, dividing by how many samples each output actually averaged
    # rather than by the kernel width. A plain `ones(win)/win` tapers the first and
    # last half-kernel toward zero, which reads as a steep decay exactly where the
    # fit starts — biasing the estimate low, worse the shorter the record and
    # worse the lower the band, since the kernel grows as 1/fc. Records shorten
    # with absorption, so that bias selected on `test_material_shift`'s own axis.
    kernel = np.ones(win)
    counts = np.convolve(np.ones(n), kernel, mode="same")
    smoothed = np.convolve(energy, kernel, mode="same") / counts

    peak = float(smoothed.max()) if smoothed.size else 0.0
    if peak <= 0.0:
        return float("nan")
    envelope_db = 10.0 * np.log10(np.maximum(smoothed, peak * 1e-300) / peak)
    t_s = np.arange(n) / sample_rate

    lo_frac, hi_frac = fit.window
    # The fit also starts no earlier than a half kernel in, where the local mean
    # stops being asymmetric, and spans enough kernel widths to be a slope rather
    # than one smoothed value repeated.
    lo, hi = max(int(lo_frac * n), win // 2), int(hi_frac * n)
    if hi - lo < 2:
        return float("nan")
    fit_t, fit_db = t_s[lo:hi], envelope_db[lo:hi]
    finite = np.isfinite(fit_db)
    if finite.sum() < 2:
        return float("nan")
    slope = float(np.polyfit(fit_t[finite], fit_db[finite], 1)[0])
    if slope >= 0.0:
        return float("nan")
    return 60.0 * (n / sample_rate) / (-60.0 / slope)


def _decay_times_from_energy(
    energy_trunc: np.ndarray,
    sample_rate: int,
    *,
    min_decay_range_db: dict[str, float],
    band_centre_hz: float,
    filter_order: int,
    decay_range_fit: DecayRangeFit,
) -> tuple[float, str | None, float, str | None]:
    """T30 and EDT from an already-truncated band energy envelope.

    Factored out so the RESOLVABILITY FLOOR is measured through the very path it
    governs, rather than through a second implementation that could drift from it.
    Returns (t30, t30_reason, edt, edt_reason).
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

    # ISO 3382-1 SNR ADMISSIBILITY — the bound for decays too SLOW for their
    # record, mirroring `_band_resolvable_decay_s` for decays too fast.
    #
    # Everything above returns a number for ANY record: the Schroeder backward
    # integral terminates at -inf dB on its last sample, so the regression mask is
    # never empty however little genuine decay the record holds, and the terminal
    # plunge steepens the least-squares fit. The failure is silent and signed —
    # T30 UNDER-reads, so a decay that outruns its record reads as a shorter room.
    #
    # Range from `_available_decay_range_db`; see its docstring for why the
    # envelope and never the Schroeder curve.
    window_s = len(energy_trunc) / sample_rate
    available_db = _available_decay_range_db(
        energy_trunc, sample_rate, band_centre_hz, filter_order, decay_range_fit
    )

    def _admissible(value, reason, metric):
        floor = min_decay_range_db.get(metric)
        if floor is None:
            raise KeyError(
                f"metric_min_decay_range_db declares no floor for {metric!r}. "
                "Every reported decay metric must declare the SNR range ISO "
                "3382-1 requires of it — there is no default."
            )
        if reason is not None or not np.isfinite(value):
            return value, reason
        # A floor of 0.0 declares no requirement, so it must not refuse — not even
        # on an unmeasurable range. `_band_resolvable_decay_s` measures the
        # filter's own ringing from a unit impulse, which has no decay to have a
        # range at all, and depends on that.
        if floor <= 0.0:
            return value, reason
        if not np.isfinite(available_db) or available_db < floor:
            got = "unmeasurable" if not np.isfinite(available_db) else f"{available_db:.1f} dB"
            return float("nan"), (
                f"decay range {got} over the {window_s * 1000:.0f} ms integration "
                f"window is below the {floor:g} dB ISO 3382-1 requires for {metric}"
                " — the record truncates the decay, so this is not a room-acoustic"
                " quantity"
            )
        return value, reason

    t30, t30_reason = _admissible(t30, t30_reason, "T30")
    edt, edt_reason = _admissible(edt, edt_reason, "EDT")
    return t30, t30_reason, edt, edt_reason


@lru_cache(maxsize=None)
def _band_resolvable_decay_s(fc: float, sample_rate: int, order: int) -> dict[str, float]:
    """The shortest decay the `fc` octave band can resolve: the filter's OWN.

    A unit impulse carries no decay at all, so whatever T30/EDT this returns is
    manufactured entirely by the filter's ringing. A room decaying faster than
    this is not being measured — the filter is.

    MEASURED here rather than asserted, and a function of `order` as well as `fc`,
    so both are part of the cache key. The realized values are written down in
    exactly one other place — `_DECLARED_FLOORS_48K` in tests/test_metrics.py,
    which pins them — and change together with this function.

    T30 and EDT get SEPARATE floors because they are separate questions: EDT fits
    the first 10 dB, T30 the 5-35 dB span, and the filter's own response supports
    the first for about twice as long.
    """
    # Long enough that the filter's ringing decays fully inside the record at any
    # band, and never shorter than the filter's own minimum.
    n = max(int(0.5 * sample_rate), _MIN_FILTER_SAMPLES * 4)
    impulse = np.zeros(n, dtype=np.float32)
    impulse[0] = 1.0
    energy = _band_energy(impulse, fc, sample_rate, order)
    # Floors of 0.0 on purpose: this measures the FILTER's own ringing from a unit
    # impulse, which carries no room decay, so ISO 3382-1's decay-range
    # admissibility must not apply — it would make the floor itself unmeasurable.
    t30, _, edt, _ = _decay_times_from_energy(
        energy[: _lundeby_truncate(energy, sample_rate)],
        sample_rate,
        min_decay_range_db={"T30": 0.0, "EDT": 0.0},
        band_centre_hz=fc,
        filter_order=order,
        # Any valid fit: the floors above are 0.0, so `_admissible` short-circuits
        # and the decay range is never consulted. This function measures the
        # FILTER's own ringing from a unit impulse, which has no room decay to
        # estimate a range for — so it must not depend on how that estimate is
        # configured, or the resolvability floor would move when the estimator's
        # window did.
        decay_range_fit=_RANGE_UNUSED,
    )
    return {"T30": t30, "EDT": edt}


def _shared_truncation_per_band(
    reference_legs: dict[str, np.ndarray],  # leg name → (T,) W-channel waveform
    *,
    sample_rate: int,
    iso_eval_freqs: list[float],
    onset_rel_db: float,
    octave_filter_order: int,
    expected_onset_samples: int | None = None,
    onset_tolerance_samples: int = 0,
) -> list[tuple[int, str]]:
    """Per eval band, the truncation index SHARED by every leg of a paired
    comparison, plus the name of the leg that set it.

    Why this exists
    ---------------
    Each leg's own truncation index is a function of the ray budget, so truncating
    legs at their own indices integrates them over DIFFERENT limits and
    manufactures a metric difference with no acoustic cause. All legs are the same
    room, so ISO-3382 band metrics are only comparable over a common limit.

    On a record with a real noise floor, that floor IS this study's independent
    variable. On a gsound record there is no such floor — the record is zero-padded
    from its native length, so `_lundeby_truncate` takes its silent-tail branch and
    the index lands where the backend stopped writing — but that native length is
    itself a function of the ray budget, so the index moves with the budget either
    way. On real renders the per-leg indices differ by tens to hundreds of ms.

    WHAT SHARING DOES NOT FIX. It removes the part of a paired difference the
    window causes, which on this backend is ~1 % of it: T30 is fitted over the -5
    to -35 dB span of the EDR and an index beyond that span barely moves the fit.
    Fitting both legs over an identical span, so record length cannot enter at all,
    still leaves the fitted late slopes differing by ~30 % on average. The low-ray
    leg does not hold a shorter version of the same decay; it holds a DIFFERENT
    decay, because 5,000 rays do not sample the late field densely enough. That is
    also why ISO 3382-1 extrapolated-tail compensation is not shipped: measured on
    real renders it makes the paired difference worse, not better, because no
    treatment of where the integral ends can reconcile two different decays.

    The consequence for reporting: a LOW-leg (and therefore a `pred`) ABSOLUTE is
    not a budget-independent room-acoustic quantity at any window. The HIGH leg is
    the one an absolute may be quoted from. `reporting/tables.py` carries both
    caveats.

    The index is the MINIMUM over the reference legs, so no leg is ever integrated
    into its own noise floor. Where the shared window is too short to support a
    metric, that metric is NaN with a reason rather than a biased number.

    `reference_legs` must contain only PHYSICAL legs (`low`/`high`) — never a model
    output. Deriving the window from `pred` would let a degenerate prediction
    shorten the window used to measure its own ground truth, compressing the legs
    together and shrinking |pred - high|: a worse model scoring better by
    corrupting the measurement of its target.

    Each leg is onset-aligned first, so the returned indices are counted from each
    leg's own direct arrival and are directly comparable.
    """
    if not reference_legs:
        raise ValueError(
            "_shared_truncation_per_band requires at least one reference leg; "
            "an empty leg set would leave the integration limit undefined."
        )
    aligned = {
        leg: ir[find_onset(
            ir, onset_rel_db,
            expected_sample=expected_onset_samples,
            tolerance_samples=onset_tolerance_samples,
        )[0]:]
        for leg, ir in reference_legs.items()
    }
    out: list[tuple[int, str]] = []
    for fc in iso_eval_freqs:
        per_leg = {
            leg: _lundeby_truncate(
                _band_energy(ir, float(fc), sample_rate, octave_filter_order), sample_rate
            )
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
    min_decay_range_db: dict[str, float],
    decay_range_fit: DecayRangeFit,
    octave_filter_order: int,
    trunc_idx: int | None = None,
    trunc_source: str | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """
    T30, EDT, C50 for a single octave band centered at fc Hz.

    Returns (values, nan_reasons, resolvability):

    * `values` — a metric whose window is degenerate is NaN.
    * `nan_reasons` — WHY, for every NaN metric, surfaced up to the eval drop log
      so no band leaves a result silently.
    * `resolvability` — metric → reason, for every metric whose FITTED VALUE falls
      below what this band can resolve. **Reported here, never applied here.**

    Suppressing a value because of its own magnitude censors the low tail of a
    noisy estimator and biases the surviving mean upward, so it is a decision only
    a caller that knows the leg's ROLE may take: `compute_room_acoustic_metrics`
    discloses for the physical legs and suppresses for `pred`, while
    `channel_band_avg_metrics` (the single-IR probe unit) suppresses.
    """
    all_metrics = ("T30", "EDT", "C50")
    # Two admission bounds, both numerical guards rather than experiment
    # parameters: `_MIN_FILTER_SAMPLES` is what `sosfiltfilt` needs at all, and the
    # band-dependent guard width is what the energy fold needs to be meaningful.
    # Below one guard width the fold's mirror lands outside the record and clamps
    # to the record edge, depositing the energy at the far end from its arrival.
    # Unmeasurable with a reason rather than approximated.
    guard_samples = _filter_guard_samples(fc, sample_rate, octave_filter_order)
    if ir_w.shape[0] < max(_MIN_FILTER_SAMPLES, guard_samples):
        reason = (
            f"record is {ir_w.shape[0]} samples, shorter than the {guard_samples}-sample "
            f"zero-pad guard the {fc:g} Hz octave filter needs "
            f"({1000.0 * guard_samples / sample_rate:.0f} ms). Below one guard width the "
            f"acausal ringing folds to a mirror outside the record and is clamped to the "
            f"record edge, which conserves energy but deposits it at the far end from its "
            f"arrival. Unmeasurable rather than approximated"
        )
        return (
            {m: float("nan") for m in all_metrics},
            {m: reason for m in all_metrics},
            {},
        )
    energy = _band_energy(ir_w, fc, sample_rate, octave_filter_order)  # (T,)

    # Lundeby truncation before Schroeder. `trunc_idx` is supplied by a paired
    # caller so every leg integrates over the SAME limit; a single-IR caller passes
    # None and gets this IR's own index.
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
    t30, t30_reason, edt, edt_reason = _decay_times_from_energy(
        energy_trunc, sample_rate, min_decay_range_db=min_decay_range_db,
        band_centre_hz=fc, filter_order=octave_filter_order,
        decay_range_fit=decay_range_fit,
    )

    # C50 is computed BEFORE the resolvability verdict because the verdict cites
    # it: the band's own clarity is what distinguishes the two mechanisms that can
    # make a decay read short, and the reader is owed which one applies.
    #
    # Early/late split at 50 ms. The late window integrates only to the Lundeby
    # truncation index (as T30/EDT do), never the full record — otherwise the
    # noise-floor tail inflates `late`, and since the low-ray carrier is noisier
    # than the high-ray reference that biases the paired C50 comparison.
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

    # A short EDT has two possible causes and the reader is owed which one, so the
    # verdict names both and carries the band's C50 to tell them apart: a high C50
    # says the first 10 dB is direct-dominated (an ISO-3382-real quantity), a low
    # one says the decay really is at the filter's own scale. T30 regresses -5 to
    # -35 dB, past the direct arrival's influence, so its wording stays
    # filter-oriented.
    c50_note = "C50 unscored in this band" if np.isnan(c50) else f"the band's C50 is {c50:+.1f} dB"
    _mechanism = {
        "T30": "at this decay the fitted slope measures the filter's own ringing "
               "rather than the room",
        "EDT": "the first 10 dB is set by the direct arrival and/or the filter's own "
               "ringing at this decay — a large C50 indicates the former, which is an "
               "ISO-3382-real quantity and not a filter artifact",
    }
    resolvability: dict[str, str] = {}
    floors = _band_resolvable_decay_s(float(fc), sample_rate, octave_filter_order)
    for _name, _val in (("T30", t30), ("EDT", edt)):
        floor = band_resolvability_margin * floors[_name]
        if not np.isnan(_val) and _val < floor:
            resolvability[_name] = (
                f"{_name} {_val:.4f} s is below the {floor:.4f} s the {fc:g} Hz "
                f"octave band can resolve ({band_resolvability_margin:g} x the "
                f"filter's own {_name} of {floors[_name]:.4f} s) — "
                f"{_mechanism[_name]} ({c50_note}){window_note}"
            )

    # C50 inherits T30's verdict, and only T30's. T30's is the verdict that says
    # the LATE WINDOW carries no measurable room decay, and the late window is
    # C50's denominator: where T30 is unresolvable the energy after 50 ms is the
    # filter's own ringing rather than reverberation, so the ratio is not the
    # ISO-3382 quantity C50 is defined to be. Not EDT's — a short EDT can mean a
    # dominant direct arrival, which is exactly when C50 is large and real.
    if "T30" in resolvability and not np.isnan(c50):
        resolvability["C50"] = (
            f"C50 {c50:+.1f} dB is unscored because the {fc:g} Hz band cannot "
            f"resolve this decay: T30 is below the band's floor, so the energy "
            f"after the 50 ms split — C50's denominator — is the filter's own "
            f"ringing rather than room reverberation{window_note}"
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
    iso_eval_freqs: list[float],  # config.iso_eval_freqs
    onset_rel_db: float,          # config.metric_onset_rel_db
    band_resolvability_margin: float,      # config.metric_band_resolvability_margin
    min_decay_range_db: dict[str, float],  # config.metric_min_decay_range_db
    decay_range_fit: DecayRangeFit,       # config.metric_decay_range_fit
    octave_filter_order: int,              # config.metric_octave_filter.order
    #: `floor(|src - rcv| / c * fs)` — where GEOMETRY says the direct arrival is.
    #: None = no geometry available (standalone probes), leaving the detector alone.
    expected_onset_samples: int | None = None,
    #: How far the energy detector may disagree with it before geometry adjudicates
    #: (config.metric_onset_tolerance_ms, converted to samples).
    onset_tolerance_samples: int = 0,
    trunc_idx_per_band: list[tuple[int, str]] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Onset-align a W-channel IR to its direct arrival, then average ISO-3382 band
    metrics (T30/EDT/C50) over the evaluation bands.

    `trunc_idx_per_band` (from `_shared_truncation_per_band`) makes this leg
    integrate over a window shared with the other legs of a paired comparison.
    Omit it for a genuinely standalone IR, which then uses its own Lundeby index.

    The single-IR unit (the D0b oracle probe and known-answer tests): averages THIS
    IR's surviving bands. The eval stage's paired triples do NOT use this average —
    they band-intersect across legs first — but both are built on
    `channel_per_band_metrics`, so onset alignment and late-window truncation have
    one source of truth.

    Returns (values, nan_reasons). A metric is NaN iff every eval band is NaN;
    nan_reasons then aggregates the per-band reasons. A PARTIAL band drop (some
    bands NaN, average still defined) also gets a reason, prefixed "partial:", so
    the changed composition of the band average is visible rather than silent.

    THIS UNIT SUPPRESSES BELOW THE RESOLVABILITY FLOOR, where the reported path
    (`compute_room_acoustic_metrics`) discloses instead. Its callers are single-IR
    probes: there is no leg role to reason about and no paired population whose
    mean censoring could bias.
    """
    per_band = channel_per_band_metrics(
        ir_w, sample_rate=sample_rate,
        iso_eval_freqs=iso_eval_freqs, onset_rel_db=onset_rel_db,
        band_resolvability_margin=band_resolvability_margin,
        min_decay_range_db=min_decay_range_db,
        decay_range_fit=decay_range_fit,
        octave_filter_order=octave_filter_order,
        expected_onset_samples=expected_onset_samples,
        onset_tolerance_samples=onset_tolerance_samples,
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
    iso_eval_freqs: list[float],  # config.iso_eval_freqs
    onset_rel_db: float,          # config.metric_onset_rel_db
    band_resolvability_margin: float,      # config.metric_band_resolvability_margin
    min_decay_range_db: dict[str, float],  # config.metric_min_decay_range_db
    decay_range_fit: DecayRangeFit,       # config.metric_decay_range_fit
    octave_filter_order: int,              # config.metric_octave_filter.order
    #: `floor(|src - rcv| / c * fs)` — where GEOMETRY says the direct arrival is.
    #: None = no geometry available (standalone probes), leaving the detector alone.
    expected_onset_samples: int | None = None,
    #: How far the energy detector may disagree with it before geometry adjudicates
    #: (config.metric_onset_tolerance_ms, converted to samples).
    onset_tolerance_samples: int = 0,
    trunc_idx_per_band: list[tuple[int, str]] | None = None,
) -> list[tuple[dict[str, float], dict[str, str], dict[str, str]]]:
    """Onset-align a W-channel IR, then compute per-eval-band ISO-3382 metrics —
    one (values, nan_reasons, resolvability) triple per band of `iso_eval_freqs`.

    The shared unit behind both consumers: `channel_band_avg_metrics` (single-IR
    band average — probe/tests) and `compute_room_acoustic_metrics` (cross-leg band
    intersection — eval). `trunc_idx_per_band` carries the shared integration
    window of a paired comparison; None = this IR's own.

    It REPORTS the resolvability verdict and applies nothing — the two consumers
    answer differently, so the decision cannot live here."""
    if trunc_idx_per_band is not None and len(trunc_idx_per_band) != len(iso_eval_freqs):
        raise ValueError(
            f"trunc_idx_per_band has {len(trunc_idx_per_band)} entries but there are "
            f"{len(iso_eval_freqs)} eval bands — the shared integration window must be "
            f"declared per band."
        )
    # Align t=0 to the direct arrival, adjudicated by geometry where the scene
    # supplies one. The disclosure half is asked for separately by
    # `compute_room_acoustic_metrics`, which knows the leg's name and can attribute
    # the reason; this returns one triple per BAND and the onset is a property of
    # the leg, so there is no per-band slot for it. `find_onset` is pure, so asking
    # twice cannot diverge.
    ir_w = ir_w[find_onset(
        ir_w, onset_rel_db,
        expected_sample=expected_onset_samples,
        tolerance_samples=onset_tolerance_samples,
    )[0]:]
    return [
        _iso3382_band_metrics(
            ir_w, float(fc), sample_rate,
            band_resolvability_margin=band_resolvability_margin,
            min_decay_range_db=min_decay_range_db,
            decay_range_fit=decay_range_fit,
            octave_filter_order=octave_filter_order,
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
    iso_eval_freqs: list[float],  # config.iso_eval_freqs
    onset_rel_db: float,          # config.metric_onset_rel_db
    band_resolvability_margin: float,      # config.metric_band_resolvability_margin
    min_decay_range_db: dict[str, float],  # config.metric_min_decay_range_db
    decay_range_fit: DecayRangeFit,       # config.metric_decay_range_fit
    octave_filter_order: int,              # config.metric_octave_filter.order
    expected_onset_samples: int | None = None,  # geometry's direct arrival
    onset_tolerance_samples: int = 0,           # config.metric_onset_tolerance_ms
) -> tuple[
    dict[str, MetricTriple],
    dict[tuple[str, str], str],
    dict[str, tuple[int, str]],
    dict[str, dict],
]:
    """
    Standard ISO-3382 room-acoustic metrics (T30, C50, EDT) from decoded waveforms.

    Uses W-channel (ch 0), onset-aligned per IR so metrics are invariant to leading
    propagation-delay silence.

    Returns a 4-tuple:

    1. `triples` — {metric: MetricTriple}, one per metric in (T30, EDT, C50), with
       `kind="match_reference"` (pred should match the HIGH reference's value).
       T30/EDT in SECONDS, C50 in dB. NaN where unscored, with a reason in (2).
    2. `nan_reasons` — {(metric, leg): reason}, legs "pred"/"high"/"low". Covers
       two cases: a leg that is UNSCORED, and a leg that is SCORED but carries the
       resolvability caveat. `evaluator.py`'s drop sweep handles both.
    3. `window` — {band: (sample_index, source_leg)}, the shared Schroeder
       integration limit per band. Keyed by `f"{fc:g}"` — a STRING, e.g. "500" —
       valued by the truncation index in SAMPLES from that leg's own onset, plus
       the physical leg that set it. Returned so every scored row can record its
       window, not only the dropped ones.
    4. `band_accounting` — {metric: {...}}: `n_bands`, `n_bands_kept`, `kept_hz`,
       `pred_unresolved_hz`, `resolvability_limited_hz` and
       `pred_unresolved_in_floor_limited_hz`, frequencies in Hz as floats. These
       reach `metrics.parquet` and `ci_table.csv`'s caveat columns.

    EVERY LEG IS BAND-AVERAGED OVER THE SAME BAND SET, because ISO-3382
    band-averaged metrics are only comparable over a common one — averaging each
    leg over its own surviving bands mixes acoustic difference with
    band-composition difference.

    THAT BAND SET IS DERIVED FROM THE PHYSICAL LEGS ONLY, for the same reason the
    integration window is. Intersecting all three legs meant a `pred` band failing
    the resolvability floor was dropped from EVERY leg's average, so a model output
    changed the reported value of its own ground truth — a >10 % swing in the high
    leg's reported EDT with no change to the high waveform, in either direction.

    A `pred` that is NaN in a physically-kept band is itself unscored, with a
    reason: "the model produced nothing measurable in a band the physics resolves",
    not "this band does not count". `low` and `high` stay scored over the physical
    band set regardless.
    """
    physical_legs = ("low", "high")
    # Where t=0 went, per leg, before any metric is computed. A misplaced onset
    # moves the C50 50 ms split, the EDT anchor and the Schroeder start at once, so
    # it is a caveat on every metric of that leg rather than on one band of it.
    onset_reasons = {
        leg: find_onset(
            ir[0], onset_rel_db,
            expected_sample=expected_onset_samples,
            tolerance_samples=onset_tolerance_samples,
        )[1]
        for leg, ir in [("pred", pred_ir), ("high", high_ref_ir), ("low", low_ref_ir)]
    }
    shared_trunc = _shared_truncation_per_band(
        {"low": low_ref_ir[0], "high": high_ref_ir[0]},
        sample_rate=sample_rate,
        iso_eval_freqs=iso_eval_freqs,
        onset_rel_db=onset_rel_db,
        octave_filter_order=octave_filter_order,
        expected_onset_samples=expected_onset_samples,
        onset_tolerance_samples=onset_tolerance_samples,
    )
    per_leg = {
        leg: channel_per_band_metrics(
            ir[0], sample_rate=sample_rate,
            iso_eval_freqs=iso_eval_freqs, onset_rel_db=onset_rel_db,
            band_resolvability_margin=band_resolvability_margin,
            min_decay_range_db=min_decay_range_db,
            decay_range_fit=decay_range_fit,
            octave_filter_order=octave_filter_order,
            expected_onset_samples=expected_onset_samples,
            onset_tolerance_samples=onset_tolerance_samples,
            trunc_idx_per_band=shared_trunc,
        )
        for leg, ir in [("pred", pred_ir), ("high", high_ref_ir), ("low", low_ref_ir)]
    }
    n_bands = len(iso_eval_freqs)
    triples: dict[str, MetricTriple] = {}
    nan_reasons: dict[tuple[str, str], str] = {}
    band_accounting: dict[str, dict] = {}
    for metric in ("T30", "EDT", "C50"):
        # THE RESOLVABILITY FLOOR IS APPLIED PER LEG ROLE. `channel_per_band_
        # metrics` reports the verdict and applies nothing; this is where the two
        # roles diverge.
        #
        #   physical legs (low/high) — DISCLOSED. The value is reported and the
        #     band counted. Suppressing here would censor the estimator on its own
        #     magnitude and bias the reported split mean upward.
        #
        #   pred — SUPPRESSED, but ONLY IN A BAND THE PHYSICAL LEGS RESOLVE. That
        #     qualifier is load-bearing: suppressing pred wherever it fell below the
        #     floor MOVES the bias rather than removing it, because the scenes it
        #     drops are the low ones. Where NO leg resolves the band the paired
        #     comparison is still like-for-like — one instrument, one limitation —
        #     so all three are disclosed and the datum kept. Where the physics DOES
        #     resolve the band and only pred failed, that is a model failure.
        def _why(leg: str, b: int, _m: str = metric) -> str:
            """Why `leg` has no value for this metric in band `b`.

            Either dict can hold it: a hard NaN reason from the estimator, or the
            resolvability verdict (which only `pred` acts on). Looking in one alone
            would KeyError exactly when the floor is what suppressed pred.
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
                    # THE EXEMPTION DOES NOT EXTEND TO C50. For T30 and EDT,
                    # keeping pred in a band no leg resolves is right — the error
                    # is bounded by the decay itself. C50's is not: its denominator
                    # is the late window, and where the physics is floor-limited
                    # that window holds filter ringing rather than reverberation, so
                    # a degenerate pred can score hundreds of dB. Only PRED is
                    # suppressed; censoring a physical leg on a verdict that
                    # correlates with absorption would confound
                    # `test_material_shift` with its own independent variable.
                    and not (physical_floor_limited[b] and metric != "C50")
                )
                for b in range(n_bands)
            ]
            for leg, bands in per_leg.items()
        }
        # The band set is the physical legs' — pred never votes.
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
            # A disclosure, never a band exclusion.
            "pred_unresolved_hz": [
                float(iso_eval_freqs[b]) for b in kept if not finite["pred"][b]
            ],
            # Bands the PHYSICAL legs report despite being below the floor. Which
            # numbers carry the caveat has to be visible, or the disclosure is only
            # a code comment.
            "resolvability_limited_hz": [
                float(iso_eval_freqs[b]) for b in kept if physical_floor_limited[b]
            ],
            # The residual pred-side selection this design still has to bound: pred
            # is only floor-suppressed where the physical legs DO resolve the band,
            # so a non-zero count here means pred failed for a HARD reason (a
            # non-decaying EDR, a degenerate window) in a band that is itself
            # floor-limited. Those scenes do leave `paired_improvement`.
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
        # A physical leg that REPORTS a floor-limited value still owes the reader a
        # reason. It is a caveat on a scored number rather than a drop, so it is
        # attached to a finite leg; `evaluator.py`'s drop sweep logs those too.
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
                    f"resolvability-limited but REPORTED, not suppressed: "
                    f"{len(bands)}/{len(kept)} kept bands sit below what the band "
                    f"resolves. Suppressing them would censor the estimator on its "
                    f"own value and bias this split's mean upward, so the value is "
                    f"reported with this caveat instead ({detail})"
                )
                # APPEND, never overwrite: the same (metric, leg) key may already
                # carry a band-EXCLUSION reason written above, and `evaluator.py`
                # forwards exactly one reason per (metric, leg) to `drops.csv`, so
                # assigning would silently delete the harder fact of the two.
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
            # When the band pred failed in is ITSELF floor-limited, this scene
            # leaves the paired comparison only because the floor discloses rather
            # than suppresses. Said per row, so the pred-side selection is visible
            # in the drop log and not only in an aggregate.
            overlap = band_accounting[metric]["pred_unresolved_in_floor_limited_hz"]
            overlap_note = (
                ""
                if not overlap
                else (
                    f" NOTE: {len(overlap)} of these ({', '.join(f'{f:g}' for f in overlap)}"
                    f" Hz) are bands the physical legs report only because the "
                    f"resolvability floor discloses instead of suppressing, so this "
                    f"scene's exclusion is itself pred-side selection."
                )
            )
            nan_reasons[(metric, "pred")] = (
                f"pred is unmeasurable in {len(band_accounting[metric]['pred_unresolved_hz'])}"
                f"/{len(kept)} of the bands the physical legs resolve, so pred is "
                f"unscored — the physical legs keep their own values, since a model "
                f"output must not change the reported value of its ground truth "
                f"({unresolved}).{overlap_note}"
            )
        triples[metric] = MetricTriple(
            low=leg_vals["low"], pred=leg_vals["pred"], high=leg_vals["high"],
            kind="match_reference",
            # Declared beside the values it describes: T30/EDT are the fitted decay
            # TIMES, C50 an early/late energy RATIO in dB.
            unit="dB" if metric == "C50" else "s",
        )
    # Attach the onset disclosure to every metric of the leg it belongs to,
    # appending for the reason the resolvability disclosure does.
    for leg, reason in onset_reasons.items():
        if reason is None:
            continue
        for metric in ("T30", "EDT", "C50"):
            prior = nan_reasons.get((metric, leg))
            nan_reasons[(metric, leg)] = (
                reason if prior is None else f"{prior} | ALSO: {reason}"
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
    # Frame-quantized 50 ms split: `ceil` rounds UP to the next STFT frame (~53 ms
    # at hop 512 / 48 kHz), a ~3 ms offset from the exact 50 ms of the reported
    # waveform path, which the proxy-vs-standard validation (design_spec §3)
    # accounts for. This proxy also assumes t=0 IS the direct arrival — it does no
    # onset alignment — which holds for the energy grid but not for a record
    # carrying propagation delay.
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
