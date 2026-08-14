"""The shared Schroeder window exercised on REAL renders, against per-leg windows.

The eval path shares one integration limit across the legs of a paired
comparison, because
each leg's own Lundeby index is a function of the ray budget — so truncating legs at
their own indices integrates them over DIFFERENT limits and manufactures a metric
difference with no acoustic cause.

Until now that fix was exercised only by a synthetic known-answer test, because the
defect is inert under `dry_run` by construction: the scaffold fills its whole window,
so both legs stop at the same sample and per-leg and shared indices coincide. Its
first exercise on real data would otherwise have been the expensive emulated render
it was written to protect.

The nine retained renders in `experiments/support_law/irs/` — each present at 5,000
and 200,000 rays — make that evidence free. For every (scene, band) this reports:

  * each leg's OWN truncation index, and the shared one (the minimum);
  * T30 and C50 computed BOTH ways, and the difference sharing removes.

What to read it for: the `shared` columns are what the pipeline reports, and the
`own` columns are what it would report without sharing. The gap between them is the
artifact the shared window exists to delete — and it is not small on real records.

    python scripts/shared_window_probe.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from amcd.evaluation.room_acoustic import (  # noqa: E402
    _band_energy,
    find_onset,
    _iso3382_band_metrics,
    _lundeby_truncate,
    _shared_truncation_per_band,
)

SR = 48000
ORDER = 4
ONSET_DB = -20.0
BANDS = [500.0, 1000.0]
#: Floors of 0.0: the subject is the WINDOW, so a metric must not additionally be
#: refused by the ISO SNR bound — that would hide the very comparison being made.
NO_FLOOR = 0.0
NO_SNR = {"T30": 0.0, "EDT": 0.0}
IRS = Path("experiments/support_law/irs")


def _metrics(ir_w, fc, trunc_idx, trunc_source):
    """T30/C50 for one leg at a given truncation index."""
    values, _reasons, _res = _iso3382_band_metrics(
        ir_w, fc, SR,
        band_resolvability_margin=NO_FLOOR,
        min_decay_range_db=NO_SNR,
        octave_filter_order=ORDER,
        trunc_idx=trunc_idx,
        trunc_source=trunc_source,
    )
    return values["T30"], values["C50"]


def main() -> None:
    scenes = sorted(
        {p.name[: p.name.rindex("_b")] for p in IRS.glob("*_b*.npy")
         if not p.name.startswith("._")}
    )
    if not scenes:
        raise SystemExit(
            f"no retained renders in {IRS}. This probe reads the artifacts "
            f"scripts/support_law_probe.py leaves behind; run that first."
        )

    print(f"{'scene':<22} {'band':>5} {'own lo':>7} {'own hi':>7} {'shared':>7} "
          f"{'set by':>7} | {'dT30 own':>9} {'dT30 shared':>12} "
          f"{'dC50 own':>9} {'dC50 shared':>12}")
    print("-" * 122)

    t30_own, t30_shared, c50_own, c50_shared, index_gap = [], [], [], [], []
    for name in scenes:
        low = np.load(IRS / f"{name}_b5000.npy")
        high = np.load(IRS / f"{name}_b200000.npy")
        lw = low[0][find_onset(low[0], ONSET_DB)[0]:]
        hw = high[0][find_onset(high[0], ONSET_DB)[0]:]

        shared = _shared_truncation_per_band(
            {"low": lw, "high": hw},
            sample_rate=SR, iso_eval_freqs=BANDS,
            onset_rel_db=ONSET_DB, octave_filter_order=ORDER,
        )
        for b, fc in enumerate(BANDS):
            n_lo = _lundeby_truncate(_band_energy(lw, fc, SR, ORDER), SR)
            n_hi = _lundeby_truncate(_band_energy(hw, fc, SR, ORDER), SR)
            idx, source = shared[b]

            # PER-LEG: each leg truncated at its own index — what the reported
            # numbers would be without a shared window.
            t_lo_own, c_lo_own = _metrics(lw, fc, n_lo, "self")
            t_hi_own, c_hi_own = _metrics(hw, fc, n_hi, "self")
            # SHARED: both legs at the minimum, which is what ships.
            t_lo_sh, c_lo_sh = _metrics(lw, fc, idx, source)
            t_hi_sh, c_hi_sh = _metrics(hw, fc, idx, source)

            d_t30_own = 100.0 * (t_lo_own - t_hi_own) / t_hi_own
            d_t30_sh = 100.0 * (t_lo_sh - t_hi_sh) / t_hi_sh
            d_c50_own = c_lo_own - c_hi_own
            d_c50_sh = c_lo_sh - c_hi_sh

            index_gap.append(abs(n_lo - n_hi))
            t30_own.append(abs(d_t30_own))
            t30_shared.append(abs(d_t30_sh))
            c50_own.append(abs(d_c50_own))
            c50_shared.append(abs(d_c50_sh))

            print(f"{name:<22} {fc:>5.0f} {n_lo:>7d} {n_hi:>7d} {idx:>7d} "
                  f"{source:>7} | {d_t30_own:>+8.2f}% {d_t30_sh:>+11.2f}% "
                  f"{d_c50_own:>+8.2f}dB {d_c50_sh:>+11.2f}dB")

    def stat(xs):
        a = np.asarray([x for x in xs if np.isfinite(x)])
        return a.mean(), a.max(), len(a)

    gap = np.asarray(index_gap)
    print(f"\nPer-leg truncation indices differ by {gap.mean():.0f} samples on "
          f"average ({1000 * gap.mean() / SR:.1f} ms), worst {gap.max()} "
          f"({1000 * gap.max() / SR:.1f} ms).")
    print("Under dry_run this gap is 0 by construction, which is why the fix could "
          "not be exercised on scaffold data.\n")

    for label, own, shared_, unit in (
        ("T30", t30_own, t30_shared, "%"),
        ("C50", c50_own, c50_shared, "dB"),
    ):
        m_o, w_o, n = stat(own)
        m_s, w_s, _ = stat(shared_)
        print(f"|low - high| {label}:  per-leg window  mean {m_o:6.2f}{unit}  "
              f"worst {w_o:6.2f}{unit}")
        print(f"{'':<18}shared window   mean {m_s:6.2f}{unit}  "
              f"worst {w_s:6.2f}{unit}   (n={n})")


if __name__ == "__main__":
    main()
