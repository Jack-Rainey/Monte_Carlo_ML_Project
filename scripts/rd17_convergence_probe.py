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
import zlib
from pathlib import Path

import numpy as np

from amcd.acoustics import box_volume_and_surface, sabine_rt60
from amcd.config import Config
from amcd.evaluation.room_acoustic import (
    _band_energy,
    _find_onset,
    channel_per_band_metrics,
)
from amcd.simulators.base import SceneSpec, build_simulator


def _room_key(dims: tuple[float, float, float], alpha: float) -> str:
    """Stable digest of the ROOM, so an id and a seed name a geometry rather than a
    position in some sweep.

    `zlib.crc32` over a fixed-precision rendering, not `hash()`: the builtin is
    salted per process, so seeds would differ between runs of the same probe.
    """
    return f"{zlib.crc32(f'{dims[0]:.6f},{dims[1]:.6f},{dims[2]:.6f},{alpha:.6f}'.encode()):08x}"


def _scene_id(n: int, i: int, dims: tuple[float, float, float], alpha: float) -> str:
    """Scene id carrying the population it was drawn from AND the room itself.

    The bare `rd17_{i:02d}` this replaced was a function of the sweep index alone,
    so the same id named a different room at a different `--scenes` count: `rd17_04`
    is 8.000 x 6.889 x 3.844 m in a 10-scene run and 8.727 x 7.455 x 4.055 m in a
    12-scene run, and they shared seed 1004. A later probe asked to re-render "the
    scene that failed" by id silently rendered a different room, and any fit that
    pooled two artifacts on `scene_id` merged two geometries.
    """
    return f"rd17_n{n:02d}_{i:02d}_{_room_key(dims, alpha)}"


def _scene_seed(dims: tuple[float, float, float], alpha: float) -> int:
    """Seed derived from the room, so two different rooms cannot share one."""
    return int(_room_key(dims, alpha), 16)


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
            scene_id=_scene_id(n, i, dims, alpha), seed=_scene_seed(dims, alpha),
            geometry_family="shoebox",
            dims=dims, material_absorption=alpha, source_pos=src, receiver_pos=rcv,
        ))
    return sorted(scenes, key=lambda s: -sabine_rt60(*box_volume_and_surface(s.dims),
                                                     s.material_absorption))


def _band_energy_totals(config: Config, ir_multichannel: np.ndarray, n_limit: int) -> dict:
    """Total in-band energy per eval band, over the first `n_limit` post-onset
    samples — RD-33a condition (ii)'s THIRD declared quantity.

    `convergence.band_energy_frac` has been declared in `configs/base.yaml` since
    the tolerance block was written and nothing had ever evaluated it, so the
    probe answered two of the three quantities the gate condition names. It is the
    quantity that matters most downstream: the model's output domain is per-band
    log power, so band energy is what it is ultimately scored against.

    WHY A COMMON SAMPLE LIMIT rather than each leg's own integration window: the
    record length itself moves with the ray budget (AC-185), and the shared
    Lundeby window tracks it too (F-89) — so integrating each leg to its own end
    would fold a record-length difference into what is meant to be an energy
    comparison. `n_limit` is the SHORTER of the two legs' native records, which
    asks the convergence question and only that one.

    Uses the metric path's own onset alignment and octave filter rather than a
    second implementation, for the AC-24 reason the T60 formulas are shared.
    """
    ir_w = ir_multichannel[0]
    onset = _find_onset(ir_w, config.metric_onset_rel_db)
    aligned = ir_w[onset:onset + n_limit]
    return {
        str(fc): float(_band_energy(aligned, float(fc), config.sample_rate).sum())
        for fc in config.iso_eval_freqs
    }


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
        # Geometry recorded alongside the measurement, because the candidate models
        # for realized record support disagree about which of these is the
        # independent variable and the existing artifacts cannot tell them apart:
        # every probe room so far was produced by scaling all three dims by one
        # factor, which makes the rooms geometrically SIMILAR and
        # corr(log V, log 4V/S) exactly 1.0000. Storing surface, mean free path and
        # the depth bound means the next refit needs no re-render.
        row = {"scene_id": scene.scene_id, "dims": list(scene.dims),
               "alpha": scene.material_absorption, "t60_sabine_s": t60,
               "volume_m3": volume, "surface_area_m2": surface,
               "mean_free_path_m": 4.0 * volume / surface,
               "diffuse_depth": config.simulator.params.get("diffuse_depth"),
               "legs": {}}
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
                # Kept so band energy can be integrated over a window common to both
                # legs once the pair is complete (see `_band_energy_totals`).
                "_ir": res.ir,
            }
            print(f"  {scene.scene_id} {leg:4} T60={t60:6.3f}s  {wall:6.1f}s  "
                  f"support={meta.get('realized_support_s'):.4f}s "
                  f"ratio={meta.get('support_realized_over_predicted'):.3f}", flush=True)

        # The verdict: does the HIGH leg reproduce a leg with MORE rays, inside
        # tolerance? That is what "converged reference" means. Comparing high
        # against LOW would measure the low-high gap -- D0a headroom, the thing the
        # model is trained to close -- and would call the reference unconverged
        # precisely when the denoising problem is hardest.
        # Band energy over a window BOTH legs cover, so the comparison is about
        # energy and not about the record-length difference the budget also causes.
        n_common = min(int(row["legs"][leg]["native_ir_samples"] or 0)
                       for leg in ("high", "reference"))
        for leg in ("high", "reference"):
            row["legs"][leg]["band_energy"] = _band_energy_totals(
                config, row["legs"][leg].pop("_ir"), n_common)
        row["band_energy_window_samples"] = n_common

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
            e_ref = row["legs"]["reference"]["band_energy"][fc]
            e_hi = row["legs"]["high"]["band_energy"][fc]
            if e_ref > 0.0:
                d = abs(e_hi - e_ref) / e_ref
                band["band_energy"] = {"delta": d, "tolerance": tol.band_energy_frac,
                                       "within": bool(d <= tol.band_energy_frac)}
            else:
                band["band_energy"] = {
                    "delta": None,
                    "reason": "reference leg carries zero energy in this band, so a "
                              "relative difference is undefined",
                }
            verdict[fc] = band
        row["verdict"] = verdict
        results.append(row)

    (args.out / "rd17_results.json").write_text(json.dumps(results, indent=2))

    # Summary — scored vs attempted per metric, never a bare pass/fail.
    print("\n=== RD-17 tolerance check (NOT a CI-backed convergence claim) ===")
    print(f"tolerances: T30 {tol.t30_frac:.0%} rel, C50 {tol.c50_db} dB abs, "
          f"band energy {tol.band_energy_frac:.0%} rel")
    for metric in ("T30", "C50", "band_energy"):
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
