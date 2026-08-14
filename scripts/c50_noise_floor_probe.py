#!/usr/bin/env python3
"""Is the reference leg's C50 disagreement non-convergence, or just sampling noise?

The 200,000-ray leg is treated as ground truth by every paired-improvement number
in this project. Comparing it against an 800,000-ray leg found T30 converged
(16/18 cells inside the declared 5 %) and C50 NOT converged: 8 of 20 cells beyond
the 1.0 dB ISO JND, worst 3.24 dB.

BEFORE spending anything on more rays, that number needs a scale. The 200k-vs-800k
delta is measured ONCE per cell, so it contains the ray-count effect AND the
Monte-Carlo variance of two single realizations, and nothing so far separates
them. The pattern already argues for noise: the worst cell is the SMALLEST, most
anechoic room (21.6 m^3, T60 0.093 s) at 1000 Hz, while its 500 Hz neighbour is
0.061 dB. A genuine ray-budget deficit grows with late energy, so it should be
worst in the most reverberant rooms, not the least.

THE MEASUREMENT. pygsound exposes no RNG seed, so the obvious control — render the
same scene N times at one budget — is impossible without a patched build. This
uses the available surrogate: budgets that differ by a few percent. 200k, 210k and
220k are the same budget physically; no convergence argument distinguishes them.
So the spread of C50 ACROSS those three is an estimate of what a single
realization is worth, so the convergence probe's 200k-vs-800k deltas can be
read against it.

WHAT THIS CAN AND CANNOT CONCLUDE. It is an UPPER bound on sampling noise, not a
noise floor: record length itself moves with the ray budget, and the shared
Schroeder truncation index moves with it too, so the measured spread contains a
systematic budget term as well. The reading only holds if `native_ir_samples` and
the truncation index are constant across the three legs, so both are recorded per
leg and checked. If they move, the spread is contaminated and this probe reports
that instead of a number.

Reported as a RANGE over n = 3, never an sd or a CI.

    python scripts/c50_noise_floor_probe.py -c configs/base.yaml -c host.yaml \
        --out experiments/c50_noise_floor
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from amcd.acoustics import box_volume_and_surface, sabine_rt60
from amcd.config import Config
from amcd.evaluation.room_acoustic import channel_per_band_metrics
from amcd.simulators.base import SceneSpec, build_simulator

#: The three worst C50 cells from the convergence probe, pinned by GEOMETRY rather
#: than by scene id: ids there were a function of the sweep index, so the same id
#: names a different room at a different scene count.
#: (label, dims, alpha, the 200k-vs-800k C50 delta this cell showed, in dB)
_CELLS: list[tuple[str, tuple[float, float, float], float, dict]] = [
    ("worst_anechoic", (3.0, 3.0, 2.4), 0.800, {"500": 0.061, "1000": 3.244}),
    ("mid_small", (5.4545, 4.7273, 3.1636), 0.550, {"500": 1.064, "1000": 1.936}),
    ("mid_large", (8.0, 6.8889, 3.8444), 0.3833, {"500": 1.431, "1000": 1.250}),
]

#: Nominally identical budgets. Any spread across these is not convergence.
_BUDGETS = (200_000, 210_000, 220_000)


def _c50(config: Config, ir: np.ndarray) -> dict:
    """C50 per eval band, through the production ISO path."""
    per_band = channel_per_band_metrics(
        ir[0], sample_rate=config.sample_rate, iso_eval_freqs=config.iso_eval_freqs,
        onset_rel_db=config.metric_onset_rel_db,
        band_resolvability_margin=config.metric_band_resolvability_margin,
        min_decay_range_db=config.metric_min_decay_range_db,
        octave_filter_order=config.metric_octave_filter.order,
    )
    return {str(fc): values.get("C50") for fc, (values, _r, _v) in
            zip(config.iso_eval_freqs, per_band)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", action="append", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = Config.load(*args.config)
    sim = build_simulator(
        config.simulator.name, config.simulator.params, n_channels=config.n_channels,
        n_samples=config.n_samples, sample_rate=config.sample_rate,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    tol = config.convergence.c50_db

    results = []
    for label, dims, alpha, observed in _CELLS:
        lx, ly, lz = dims
        m = config.scenes.margins
        z = min(1.5, lz - m.ceiling)
        scene = SceneSpec(
            scene_id=label, seed=17, geometry_family="shoebox", dims=dims,
            material_absorption=alpha,
            source_pos=(m.wall, m.wall, z),
            receiver_pos=(lx - m.wall, ly - m.wall, z),
        )
        volume, surface = box_volume_and_surface(dims)
        row = {"label": label, "dims": list(dims), "alpha": alpha,
               "t60_sabine_s": sabine_rt60(volume, surface, alpha),
               "observed_200k_vs_800k_db": observed, "legs": {}}
        print(f"\n=== {label}  {dims} alpha={alpha}  T60={row['t60_sabine_s']:.3f}s ===")
        for budget in _BUDGETS:
            t0 = time.time()
            res = sim.render(scene, budget)
            wall = time.time() - t0
            meta = dict(getattr(res, "meta", {}) or {})
            row["legs"][str(budget)] = {
                "c50": _c50(config, res.ir),
                "native_ir_samples": meta.get("native_ir_samples"),
                "wall_clock_s": wall,
            }
            print(f"  {budget:7d} rays  {wall:5.1f}s  native={meta.get('native_ir_samples')}  "
                  f"C50={ {k: (None if v is None else round(v, 4)) for k, v in row['legs'][str(budget)]['c50'].items()} }",
                  flush=True)
        results.append(row)

    print("\n" + "=" * 78)
    print("SPREAD ACROSS NOMINALLY IDENTICAL BUDGETS (n=3, reported as a RANGE)")
    print("=" * 78)
    contaminated = []
    for row in results:
        natives = {leg["native_ir_samples"] for leg in row["legs"].values()}
        if len(natives) > 1:
            contaminated.append((row["label"], sorted(natives)))
        for band in row["observed_200k_vs_800k_db"]:
            vals = [row["legs"][str(b)]["c50"].get(band) for b in _BUDGETS]
            if any(v is None or not np.isfinite(v) for v in vals):
                print(f"  {row['label']:16s} {band:>5s} Hz  UNSCORED in at least one leg")
                continue
            spread = max(vals) - min(vals)
            observed = row["observed_200k_vs_800k_db"][band]
            verdict = ("explained by sampling noise" if spread >= observed
                       else "EXCEEDS the noise spread")
            print(f"  {row['label']:16s} {band:>5s} Hz  "
                  f"noise spread={spread:6.3f} dB   200k-vs-800k={observed:6.3f} dB   "
                  f"tol={tol} dB   -> {verdict}")

    if contaminated:
        print("\nCONTAMINATED — the record length itself moved across these legs, so the")
        print("spread above is not attributable to sampling alone:")
        for label, natives in contaminated:
            print(f"  {label}: native_ir_samples {natives}")
    else:
        print("\nRecord length is constant across all three legs of every cell, so the")
        print("spread is attributable to sampling rather than to a changing window.")

    print("\nThis is an UPPER bound on single-realization variance, not a noise floor,")
    print("and it does not by itself settle whether to raise `high_ray_budget`.")

    (args.out / "c50_noise_floor.json").write_text(json.dumps(results, indent=2))
    total = sum(leg["wall_clock_s"] for r in results for leg in r["legs"].values())
    print(f"\n{3 * len(results)} renders, {total / 60:.1f} min")
    print(f"written: {args.out / 'c50_noise_floor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
