"""F-89/RD-55: does extrapolated-tail compensation fix budget-dependent absolutes?

IT DOES NOT, AND THIS IS THE MEASUREMENT THAT SETTLED IT. RD-55 proposed ISO
3382-1's modelled-tail treatment on the reasoning that a reported absolute depends
on where the record stopped, and that on gsound the record's native length is itself
budget-dependent. The reasoning is sound and the remedy is the standard one; it is
simply not what is wrong here.

Run against the nine retained real renders in `experiments/support_law/irs/`, each
present at 5,000 and 200,000 rays:

    |T30 low - T30 high| / T30 high    mean      worst    over the 5 % JND
    no compensation                   20.22 %   62.27 %       14/18
    compensated                       23.01 %   62.27 %       15/18

THE IMPLEMENTATION IS CORRECT, and `_control()` is the known answer that proves it:
on a synthetic exponential decay truncated to 45 dB of range it cuts the truncation
bias from -0.83 % to -0.01 %, which is exactly what the treatment is for. So the
negative result above is about the DATA, not about the code under test.

THE MECHANISM IS NOT THE INTEGRATION LIMIT, and `_slopes()` shows why. Fitting both
legs over an IDENTICAL span — the shorter record's own length, so record length
cannot enter — their late slopes still differ by 29.59 % on average and 88.62 % at
worst. The low-ray leg does not hold a SHORTER version of the same decay; it holds a
DIFFERENT decay, because 5,000 rays do not sample the late field densely enough to
reproduce it. Two different decays cannot be reconciled by changing where an
integral ends.

Kept as a script rather than deleted with the branch: the conclusion is NEGATIVE,
and a negative result nobody can re-run is an assertion. Shipping the compensation
would move every reported number for a benefit this project's own data says it does
not deliver, so it lives here and not in `evaluation/room_acoustic.py`.

    python scripts/tail_compensation_probe.py            # the three tables
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from amcd.evaluation.room_acoustic import _band_energy, _find_onset, _lundeby_truncate

SR = 48000
ORDER = 4
ONSET_DB = -20.0
BANDS = [500.0, 1000.0]
FIT_LO_DB, FIT_HI_DB = -5.0, -25.0
IRS = Path("experiments/support_law/irs")


def tail_constant(energy_trunc, sample_rate, fit_lo_db=FIT_LO_DB, fit_hi_db=FIT_HI_DB):
    """Modelled energy beyond the record's end, in the same units as the sum."""
    win = max(1, int(0.010 * sample_rate))
    sm = np.convolve(energy_trunc, np.ones(win) / win, mode="same")
    peak = float(sm.max())
    if peak <= 0.0:
        return 0.0, "no energy"
    db = 10.0 * np.log10(np.maximum(sm, peak * 1e-300) / peak)
    t = np.arange(len(db)) / sample_rate
    mask = (db <= fit_lo_db) & (db >= fit_hi_db)
    if mask.sum() < 2:
        return 0.0, f"<2 samples in the [{fit_lo_db:g}, {fit_hi_db:g}] dB fit window"
    m, b = np.polyfit(t[mask], db[mask], 1)
    if m >= 0.0:
        return 0.0, "non-decaying late slope"
    t_end = len(energy_trunc) / sample_rate
    e_end = peak * 10.0 ** ((m * t_end + b) / 10.0)
    r = 10.0 ** (m / (10.0 * sample_rate))
    if not 0.0 < r < 1.0:
        return 0.0, "degenerate per-sample decay ratio"
    return float(e_end * r / (1.0 - r)), None


def t30(energy_trunc, sample_rate, compensate: bool):
    edr = np.cumsum(energy_trunc[::-1])[::-1].astype(np.float64)
    if compensate:
        c, _ = tail_constant(energy_trunc, sample_rate)
        edr = edr + c
    edr = np.maximum(edr, 1e-300)
    db = 10.0 * np.log10(edr / edr[0])
    t = np.arange(len(db)) / sample_rate
    mask = (db >= -35.0) & (db <= -5.0)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(t[mask], db[mask], 1)[0]
    return float(-60.0 / slope) if slope < 0 else float("nan")


def _scenes():
    return sorted(
    {p.name[: p.name.rindex("_b")] for p in IRS.glob("*_b*.npy")
     if not p.name.startswith("._")}
)


def _late_slope_db_per_s(energy, sample_rate):
    """Slope of the smoothed band energy over its [-5, -25] dB span, dB/s."""
    win = max(1, int(0.010 * sample_rate))
    sm = np.convolve(energy, np.ones(win) / win, mode="same")
    peak = float(sm.max())
    if peak <= 0:
        return float("nan")
    db = 10.0 * np.log10(np.maximum(sm, peak * 1e-300) / peak)
    t = np.arange(len(db)) / sample_rate
    mask = (db <= FIT_LO_DB) & (db >= FIT_HI_DB)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(t[mask], db[mask], 1)[0])


def _legs(name, fc):
    """(low, high) truncated band energies for one scene and band."""
    low = np.load(IRS / f"{name}_b5000.npy")
    high = np.load(IRS / f"{name}_b200000.npy")
    lw = low[0][_find_onset(low[0], ONSET_DB)[0]:]
    hw = high[0][_find_onset(high[0], ONSET_DB)[0]:]
    el, eh = _band_energy(lw, fc, SR, ORDER), _band_energy(hw, fc, SR, ORDER)
    return el[: _lundeby_truncate(el, SR)], eh[: _lundeby_truncate(eh, SR)]


def _budget_dependence():
    """Does compensation reduce the gap between the two budgets' absolutes?"""
    print("=== 1. T30 absolutes, 5,000 vs 200,000 rays ===\n")
    print(f"{'scene':<22} {'band':>6}  {'raw delta %':>12}  {'compensated %':>14}")
    raw, comp = [], []
    for name in _scenes():
        for fc in BANDS:
            el, eh = _legs(name, fc)
            out = []
            for flag in (False, True):
                a, b = t30(el, SR, flag), t30(eh, SR, flag)
                out.append(100.0 * (a - b) / b if b > 0 else float("nan"))
            raw.append(abs(out[0]))
            comp.append(abs(out[1]))
            print(f"{name:<22} {fc:>6.0f}  {out[0]:>+12.2f}  {out[1]:>+14.2f}")
    raw, comp = np.array(raw), np.array(comp)
    ok = np.isfinite(raw) & np.isfinite(comp)
    print(f"\n|delta| mean   raw {raw[ok].mean():6.2f} %   "
          f"compensated {comp[ok].mean():6.2f} %")
    print(f"|delta| worst  raw {raw[ok].max():6.2f} %   "
          f"compensated {comp[ok].max():6.2f} %")
    print(f"over the 5 % JND: raw {(raw[ok] > 5).sum()}/{ok.sum()}   "
          f"compensated {(comp[ok] > 5).sum()}/{ok.sum()}")


def _control():
    """The known answer: compensation MUST remove a pure truncation bias."""
    print("\n\n=== 2. Control — one synthetic decay, truncated ===\n")
    true_t60 = 0.8
    rng = np.random.default_rng(11)
    n_full = int(4.0 * SR)
    t = np.arange(n_full) / SR
    energy = (rng.standard_normal(n_full) ** 2) * 10.0 ** (-6.0 * t / true_t60)
    ref = t30(energy, SR, False)
    print(f"true T60 {true_t60} s; the full 4.0 s record reads T30 = {ref:.4f} s\n")
    print(f"{'record (s)':>11} {'dB held':>8} {'raw T30':>9} {'raw err %':>10} "
          f"{'comp T30':>9} {'comp err %':>11}")
    for secs in (3.0, 2.0, 1.5, 1.0, 0.8, 0.6):
        e = energy[: int(secs * SR)]
        raw, comp = t30(e, SR, False), t30(e, SR, True)
        print(f"{secs:>11.2f} {60.0 * secs / true_t60:>8.1f} {raw:>9.4f} "
              f"{100 * (raw - ref) / ref:>+10.2f} {comp:>9.4f} "
              f"{100 * (comp - ref) / ref:>+11.2f}")


def _slopes():
    """Why compensation cannot help: the two legs hold DIFFERENT decays."""
    print("\n\n=== 3. Late slope, both legs fitted over an IDENTICAL span ===\n")
    print(f"{'scene':<22} {'band':>5} {'span (s)':>9} {'slope low':>10} "
          f"{'slope high':>11} {'diff %':>8}")
    diffs = []
    for name in _scenes():
        for fc in BANDS:
            el, eh = _legs(name, fc)
            n = min(len(el), len(eh))
            sl = _late_slope_db_per_s(el[:n], SR)
            sh = _late_slope_db_per_s(eh[:n], SR)
            d = 100.0 * (sl - sh) / sh if sh == sh and sh != 0 else float("nan")
            if np.isfinite(d):
                diffs.append(abs(d))
            print(f"{name:<22} {fc:>5.0f} {n / SR:>9.3f} {sl:>10.2f} "
                  f"{sh:>11.2f} {d:>+8.2f}")
    d = np.array(diffs)
    print(f"\nSlope difference over an IDENTICAL span: mean {d.mean():.2f} %, "
          f"worst {d.max():.2f} %, n={len(d)}")
    print("Record length is held equal here, so no integration-limit remedy can "
          "touch this.")


if __name__ == "__main__":
    _budget_dependence()
    _control()
    _slopes()
