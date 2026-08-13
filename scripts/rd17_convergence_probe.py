#!/usr/bin/env python3
"""RD-17: is the high ray-budget leg actually a converged reference?

Every paired-improvement metric in this project, D0a's headroom and D0b's carrier
test all treat `high_ray_budget` as ground truth. Nothing had ever checked that.
This renders the same scenes at both budgets and reports whether the reported ISO
quantities agree inside the config-declared `convergence:` tolerances.

WHAT THIS IS AND IS NOT (RD-263). A TOLERANCE CHECK over a handful of scenes. It is
NOT a CI-backed convergence claim: there is one realization per (scene, budget)
because pygsound exposes no RNG seed (RD-23), so a disagreement cannot be separated
from Monte-Carlo variance without repeats. Read it as engineering feasibility
evidence, never as an E4 result.

It also collects the per-render falsification of the backend's declared record
support (AC-184), because those points are free here and the shipped power law is
fitted on only three renders.

FORWARD-LOOKING, and the reason the artifact is shaped the way it is: the eventual
deliverable is a CONVERGENCE CURVE — reported quantity against ray count, per band,
so a reader can see where the reference leg stops moving. `rd17_results.json`
therefore stores one record per (scene, leg, band) with the budget on each leg,
rather than only the pass/fail verdict this script prints. Re-running with several
`--reference-multiple` values accumulates the points that curve needs. Plotting it
is out of scope here (paper section 6 future work); not foreclosing it is not.

    python scripts/rd17_convergence_probe.py -c configs/base.yaml -c host.yaml \
        --scenes 12 --out experiments/rd17_probe
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


def _probe_scenes(config: Config, n: int) -> list[SceneSpec]:
    """Scenes spanning the declared support, ordered by Sabine T60.

    Spread across the geometry x absorption box rather than sampled, because the
    question is whether convergence holds ACROSS the population's range — a random
    draw would concentrate in the middle, which is where it is least in doubt.
    """
    families = config.scenes.geometry_families
    shoebox = families["shoebox"].dims
    scenes: list[SceneSpec] = []
    for i in range(n):
        f = i / max(n - 1, 1)
        # Low alpha with a large room gives the long decays; the reverse gives short.
        dims = tuple(float(lo + (1.0 - f) * (hi - lo)) for lo, hi in shoebox)
        alpha = float(0.05 + f * (0.80 - 0.05))
        lx, ly, lz = dims
        m = config.scenes.margins
        src = (m.wall, m.wall, min(1.5, lz - m.ceiling))
        rcv = (lx - m.wall, ly - m.wall, min(1.5, lz - m.ceiling))
        scenes.append(SceneSpec(
            scene_id=f"rd17_{i:02d}", seed=1000 + i, geometry_family="shoebox",
            dims=dims, material_absorption=alpha, source_pos=src, receiver_pos=rcv,
        ))
    return sorted(scenes, key=lambda s: -sabine_rt60(*box_volume_and_surface(s.dims),
                                                     s.material_absorption))


def _iso(config: Config, ir_multichannel: np.ndarray) -> dict:
    """Reported ISO quantities, THROUGH THE PRODUCTION PATH (RD-17's own condition).

    Re-deriving them here would test this script rather than the pipeline.
    """
    # W (channel 0) — the omnidirectional component ISO 3382-1 is defined on. The
    # metric functions take ONE channel; handing them the (16, N) array reads the
    # channel axis as time and every band refuses with "record is 16 samples".
    per_band = channel_per_band_metrics(
        ir_multichannel[0], sample_rate=config.sample_rate, iso_eval_freqs=config.iso_eval_freqs,
        onset_rel_db=config.metric_onset_rel_db,
        band_resolvability_margin=config.metric_band_resolvability_margin,
        min_decay_range_db=config.metric_min_decay_range_db,
    )
    # Returns a LIST, one triple per entry of `iso_eval_freqs`, in that order.
    out = {}
    for fc, (values, reasons, _resolv) in zip(config.iso_eval_freqs, per_band):
        out[str(fc)] = {
            "T30": values.get("T30"), "T30_reason": reasons.get("T30"),
            "C50": values.get("C50"), "C50_reason": reasons.get("C50"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", action="append", required=True, type=Path)
    ap.add_argument("--scenes", type=int, required=True,
                    help="how many scenes to span the declared support with")
    ap.add_argument("--reference-multiple", type=float, required=True,
                    help="the CHECK leg, as a multiple of high_ray_budget. RD-17 asks "
                         "whether the high leg has converged, which can only be "
                         "answered against MORE rays than it uses -- comparing it to "
                         "the LOW leg measures the denoising gap the study exists to "
                         "close, not convergence.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = Config.load(*args.config)
    sim = build_simulator(
        config.simulator.name, config.simulator.params,
        n_channels=config.n_channels, n_samples=config.n_samples,
        sample_rate=config.sample_rate,
    )
    tol = config.convergence
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in _probe_scenes(config, args.scenes):
        volume, surface = box_volume_and_surface(scene.dims)
        t60 = sabine_rt60(volume, surface, scene.material_absorption)
        row = {"scene_id": scene.scene_id, "dims": list(scene.dims),
               "alpha": scene.material_absorption, "t60_sabine_s": t60, "legs": {}}
        for leg, budget in (("high", config.high_ray_budget),
                            ("reference", int(args.reference_multiple * config.high_ray_budget))):
            t0 = time.time()
            res = sim.render(scene, budget)
            wall = time.time() - t0
            meta = dict(getattr(res, "meta", {}) or {})
            row["legs"][leg] = {
                "ray_budget": budget,
                "wall_clock_s": wall,
                "native_ir_samples": meta.get("native_ir_samples"),
                "realized_support_s": meta.get("realized_support_s"),
                "predicted_support_s": meta.get("predicted_support_s"),
                "support_realized_over_predicted": meta.get("support_realized_over_predicted"),
                "iso": _iso(config, res.ir),
            }
            print(f"  {scene.scene_id} {leg:4} T60={t60:6.3f}s  {wall:6.1f}s  "
                  f"support={meta.get('realized_support_s'):.4f}s "
                  f"ratio={meta.get('support_realized_over_predicted'):.3f}", flush=True)

        # The verdict: does the HIGH leg reproduce a leg with MORE rays, inside
        # tolerance? That is what "converged reference" means. Comparing high
        # against LOW would measure the low-high gap -- D0a headroom, the thing the
        # model is trained to close -- and would call the reference unconverged
        # precisely when the denoising problem is hardest.
        verdict = {}
        for fc in row["legs"]["reference"]["iso"]:
            hi, lo = row["legs"]["reference"]["iso"][fc], row["legs"]["high"]["iso"][fc]
            band = {}
            for metric, thresh, rel in (("T30", tol.t30_frac, True),
                                        ("C50", tol.c50_db, False)):
                h, l = hi.get(metric), lo.get(metric)
                if h is None or l is None or not np.isfinite(h) or not np.isfinite(l):
                    band[metric] = {"delta": None,
                                    "reason": hi.get(f"{metric}_reason") or lo.get(f"{metric}_reason")}
                    continue
                d = abs(l - h) / abs(h) if rel else abs(l - h)
                band[metric] = {"delta": d, "tolerance": thresh, "within": bool(d <= thresh)}
            verdict[fc] = band
        row["verdict"] = verdict
        results.append(row)

    (args.out / "rd17_results.json").write_text(json.dumps(results, indent=2))

    # Summary — scored vs attempted per metric, never a bare pass/fail.
    print("\n=== RD-17 tolerance check (NOT a CI-backed convergence claim) ===")
    print(f"tolerances: T30 {tol.t30_frac:.0%} rel, C50 {tol.c50_db} dB abs")
    for metric in ("T30", "C50"):
        cells = [(r["scene_id"], fc, b[metric])
                 for r in results for fc, b in r["verdict"].items()]
        scored = [c for c in cells if c[2].get("delta") is not None]
        within = [c for c in scored if c[2]["within"]]
        print(f"\n{metric}: {len(within)}/{len(scored)} cells within tolerance "
              f"({len(cells) - len(scored)} unscored)")
        for sid, fc, b in sorted(scored, key=lambda c: -c[2]["delta"])[:5]:
            mark = "ok " if b["within"] else "OVER"
            unit = "" if metric == "C50" else ""
            print(f"    {mark} {sid} {fc:>7} Hz  delta={b['delta']:.4f}{unit}")
    total = sum(r["legs"][l]["wall_clock_s"] for r in results for l in ("high", "reference"))
    print(f"\n{2 * len(results)} renders, {total / 60:.1f} min wall-clock "
          f"({total / (2 * len(results)):.1f} s/render)")
    print(f"written: {args.out / 'rd17_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
