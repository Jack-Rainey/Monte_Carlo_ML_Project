"""Measure the ISO decay-range estimator against a KNOWN available range.

`_available_decay_range_db` decides whether a band's T30 is REPORTED or comes back
unscored, so an error in it is an error in which numbers reach a table — and one
that cannot be seen from inside a run, because no real record carries a known
answer to check against.

A synthetic one does. A band-filtered exponential with a set T60, over a window of
set length, holds exactly `60 * window_s / T60` dB of decay. Two shapes are
generated, because a gsound record is not a single exponential:

  clean  — one exponential across the whole window
  taper  — the room decay to a knee, then a synthesis taper 10x steeper, which is
           what upstream's adaptive energy trim leaves behind

THE KNEE POSITION IS SWEPT, and that is the point of the probe rather than an
extra. Any estimator that fits up to a fixed fraction of the record is exact when
that fraction happens to sit before the knee and catastrophic when it does not, so
a single knee position cannot distinguish a robust estimator from a lucky one.

Two estimators are compared: the shipped one, and the `shallowest of K sub-fits`
construction it replaced. The numbers in `_available_decay_range_db`'s docstring
are this script's output.

Usage:  python scripts/decay_range_probe.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from amcd.evaluation.room_acoustic import (  # noqa: E402
    _available_decay_range_db,
    _butter_octave_filter,
    _filter_guard_samples,
)

SAMPLE_RATE = 48000
FILTER_ORDER = 4
#: The reported ISO band centres plus the octaves either side, so the low bands —
#: where the envelope carries fewest independent samples and every estimator is
#: noisiest — are represented rather than assumed benign.
BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
#: (T60 seconds, window seconds). Chosen so the true range spans 48-90 dB, i.e.
#: straddles ISO 3382-1's 45 dB T30 requirement in both directions.
#:
#: THE SHORT WINDOWS ARE THE POINT OF THE LAST FOUR. The smoothing kernel is an
#: absolute duration (20 cycles of the band centre) and the fit window is a
#: FRACTION of the record, so the two collide as the record shortens — and records
#: shorten with absorption, which is `test_material_shift`'s own axis. A probe
#: that only ever sees half-second windows cannot see that, and did not: it passed
#: while the estimator returned NaN for every record shorter than its own kernel.
#: `trunc_s` reaches 0.010 s in the shipped D0b artifact, so these are not
#: hypothetical geometries.
DECAYS = (
    (0.6, 0.5), (0.4, 0.6), (1.0, 0.8), (2.0, 0.9),
    (0.40, 0.17), (0.40, 0.08), (0.10, 0.04), (0.10, 0.02),
)
#: 1.0 means no taper at all.
KNEE_FRACTIONS = (0.5, 0.6, 0.7, 0.85, 1.0)
TAPER_STEEPNESS = 10.0

_SHALLOWEST_SEGMENTS = 4


def _synthetic_band_energy(
    t60_s: float, window_s: float, band_centre_hz: float, knee_frac: float
) -> np.ndarray:
    """Band-filtered noise decaying at `t60_s`, tapered after `knee_frac`.

    Seeded off the band centre so each band is an independent realization and the
    comparison is not one draw's luck, while the whole probe stays reproducible.
    """
    n = int(window_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    envelope = 10.0 ** (-3.0 * t / t60_s)
    if knee_frac < 1.0:
        knee = int(knee_frac * n)
        t_after = np.arange(n - knee) / SAMPLE_RATE
        envelope[knee:] = envelope[knee] * 10.0 ** (
            -3.0 * t_after / (t60_s / TAPER_STEEPNESS)
        )
    rng = np.random.default_rng(int(band_centre_hz))
    ir = (rng.standard_normal(n) * envelope).astype(np.float32)

    filtered, guard = _butter_octave_filter(
        ir, band_centre_hz, SAMPLE_RATE, FILTER_ORDER
    )
    energy = filtered.astype(np.float64) ** 2
    n_record = len(energy) - 2 * guard
    return energy[guard:guard + n_record]


def _shallowest_of_k(energy, sample_rate: int, _fc: float, _order: int) -> float:
    """The REPLACED estimator, kept here so the claim that it was worse is checkable.

    Takes the shallowest of K sub-fits on a fixed-10 ms-smoothed envelope, on the
    reasoning that the taper is always steeper than the room, so the shallowest
    segment must be the room. It is a minimum over noisy slope estimates, and a
    minimum statistic is biased.
    """
    win = max(1, int(0.010 * sample_rate))
    smoothed = np.convolve(energy, np.ones(win) / win, mode="same")
    peak = float(smoothed.max()) if smoothed.size else 0.0
    if peak <= 0.0:
        return float("nan")
    envelope_db = 10.0 * np.log10(np.maximum(smoothed, peak * 1e-300) / peak)
    t_s = np.arange(len(envelope_db)) / sample_rate
    lo, hi = int(0.05 * len(envelope_db)), int(0.95 * len(envelope_db))
    seg_t, seg_db = t_s[lo:hi], envelope_db[lo:hi]

    shallowest = None
    edges = np.linspace(0, len(seg_db), _SHALLOWEST_SEGMENTS + 1, dtype=int)
    for a, b in zip(edges, edges[1:]):
        finite = np.isfinite(seg_db[a:b])
        if finite.sum() < 2:
            continue
        slope = float(np.polyfit(seg_t[a:b][finite], seg_db[a:b][finite], 1)[0])
        if slope >= 0.0:
            continue
        shallowest = slope if shallowest is None else max(shallowest, slope)
    if shallowest is None:
        return float("nan")
    return 60.0 * (len(energy) / sample_rate) / (-60.0 / shallowest)


ESTIMATORS = {
    "shipped (early-window fit)": _available_decay_range_db,
    "replaced (shallowest of 4)": _shallowest_of_k,
}


def main() -> int:
    errors: dict[str, list[tuple[float, float, float]]] = {k: [] for k in ESTIMATORS}
    for knee_frac in KNEE_FRACTIONS:
        for t60_s, window_s in DECAYS:
            true_range_db = 60.0 * window_s / t60_s
            for fc in BANDS:
                energy = _synthetic_band_energy(t60_s, window_s, fc, knee_frac)
                for name, estimator in ESTIMATORS.items():
                    got = estimator(energy, SAMPLE_RATE, fc, FILTER_ORDER)
                    errors[name].append((
                        100.0 * (got - true_range_db) / true_range_db, fc, knee_frac
                    ))

    print(f"{len(KNEE_FRACTIONS) * len(DECAYS) * len(BANDS)} cases: "
          f"{len(BANDS)} bands x {len(DECAYS)} decays x "
          f"{len(KNEE_FRACTIONS)} knee positions\n")
    print(f"{'estimator':>28} {'mean|err|':>10} {'worst over':>11} "
          f"{'worst under':>12} {'refused':>9}")
    for name, rows in errors.items():
        values = [e for e, _, _ in rows if e == e]  # NaN error == a refusal
        refused = len(rows) - len(values)
        print(f"{name:>28} {statistics.mean(abs(v) for v in values):>9.1f}% "
              f"{max(values):>+10.1f}% {min(values):>+11.1f}% "
              f"{refused:>6}/{len(rows)}")

    # PER BAND, because the kernel scales as 1/fc: a mean over all bands hides an
    # estimator that refuses the low bands and admits the high ones on the SAME
    # decay, which is a selection effect rather than an accuracy one.
    print("\nShipped estimator by band — a spread here is band-dependent "
          "admission on identical decays:")
    print(f"{'band':>7} {'guard ms':>9} {'mean|err|':>10} {'refused':>9}")
    for fc in BANDS:
        rows = [e for e, b, _ in errors["shipped (early-window fit)"] if b == fc]
        ok = [e for e in rows if e == e]
        guard_ms = 1000.0 * _filter_guard_samples(fc, SAMPLE_RATE, FILTER_ORDER) / SAMPLE_RATE
        mean = f"{statistics.mean(abs(v) for v in ok):.1f}%" if ok else "all refused"
        print(f"{fc:>7.0f} {guard_ms:>9.0f} {mean:>10} "
              f"{len(rows) - len(ok):>6}/{len(rows)}")
    print("  A refusal here is the filter's own ringing outlasting the record, so "
          "the band\n  carries no room decay to measure. It scales as 1/fc, which "
          "is why the low bands\n  refuse first — physics, not an estimator "
          "artifact, and the reported bands are\n  500/1000 Hz.")

    # THE REGIME THAT ACTUALLY SHIPS. Reported bands only, and windows no shorter
    # than the real dataset's — its median truncated window is ~0.17 s. The all-
    # cells summary above deliberately includes records too short to measure, so
    # it understates the estimator on the population E1 will score.
    print("\nReported bands (500/1000 Hz) at shippable window lengths:")
    shipped = []
    for knee_frac in KNEE_FRACTIONS:
        for t60_s, window_s in DECAYS:
            if window_s < 0.15:
                continue
            true_range_db = 60.0 * window_s / t60_s
            for fc in (500.0, 1000.0):
                got = _available_decay_range_db(
                    _synthetic_band_energy(t60_s, window_s, fc, knee_frac),
                    SAMPLE_RATE, fc, FILTER_ORDER,
                )
                shipped.append(100.0 * (got - true_range_db) / true_range_db)
    ok = [v for v in shipped if v == v]
    print(f"  {len(ok)}/{len(shipped)} scored, mean|err| "
          f"{statistics.mean(abs(v) for v in ok):.1f}%, "
          f"worst over {max(ok):+.1f}%, worst under {min(ok):+.1f}%")

    print("\nOver-reading is the UNSAFE direction: it admits a T30 the record "
          "cannot support.\nWorst over-read per knee position, shipped estimator:")
    for knee_frac in KNEE_FRACTIONS:
        at_knee = [e for e, _, k in errors["shipped (early-window fit)"]
                   if k == knee_frac and e == e]
        label = "no taper" if knee_frac == 1.0 else f"knee at {knee_frac:.0%}"
        print(f"  {label:>14}: {max(at_knee):+6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
