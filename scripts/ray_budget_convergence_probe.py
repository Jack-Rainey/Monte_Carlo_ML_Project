#!/usr/bin/env python3
"""Is the high ray-budget leg actually a converged reference?

Every paired-improvement metric in this project, D0a's headroom and D0b's carrier
test all treat `high_ray_budget` as ground truth. Nothing had ever checked that.
This renders the same scenes at both budgets and reports whether the reported ISO
quantities agree inside the config-declared `convergence:` tolerances.

WHAT THIS IS AND IS NOT. A TOLERANCE CHECK over a handful of scenes. It is
NOT a CI-backed convergence claim: there is one realization per (scene, budget)
because pygsound exposes no RNG seed, so a disagreement cannot be separated
from Monte-Carlo variance without repeats. Read it as engineering feasibility
evidence, never as an E4 result.

It also collects the per-render falsification of the backend's declared record
support, because those points are free here and the shipped power law is
fitted on only three renders.

FORWARD-LOOKING, and the reason the artifact is shaped the way it is: the eventual
deliverable is a CONVERGENCE CURVE — reported quantity against ray count, per band,
so a reader can see where the reference leg stops moving. `convergence_results.json`
therefore stores one record per (scene, leg, band) with the budget on each leg,
rather than only the pass/fail verdict this script prints. Re-running with several
`--reference-multiple` values accumulates the points that curve needs. Plotting it
is out of scope here (paper section 6 future work); not foreclosing it is not.

    python scripts/ray_budget_convergence_probe.py -c configs/base.yaml -c host.yaml \
        --scenes 12 --out experiments/ray_budget_convergence
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
    find_onset,
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

    An id derived from the sweep index alone would name a different room at a
    different `--scenes` count: index 4`
    is 8.000 x 6.889 x 3.844 m in a 10-scene run and 8.727 x 7.455 x 4.055 m in a
    12-scene run, and they shared seed 1004. A later probe asked to re-render "the
    scene that failed" by id silently rendered a different room, and any fit that
    pooled two artifacts on `scene_id` merged two geometries.
    """
    return f"conv_n{n:02d}_{i:02d}_{_room_key(dims, alpha)}"


def _scene_seed(dims: tuple[float, float, float], alpha: float) -> int:
    """Seed derived from the room, so two different rooms cannot share one."""
    return int(_room_key(dims, alpha), 16)


def _absorption_support(config: Config) -> tuple[float, float]:
    """The union of every DECLARED material regime's absorption range.

    Read from config, never a literal: the population the reference leg has to be
    converged over is whichever one the active config declares. `research_i.yaml`
    declares `ceiling_absorptive` up to 0.98 — the regime `test_material_shift` is
    made of — and a probe that stopped at `mixed`'s 0.80 would report convergence
    over five of six splits and extrapolate the sixth.
    """
    ranges = [r.absorption for r in config.scenes.material_regimes.values()]
    if not ranges:
        raise ValueError(
            "the config declares no material regimes, so there is no absorption "
            "support to sweep. The probe measures convergence over the declared "
            "population; it cannot invent one."
        )
    return min(lo for lo, _ in ranges), max(hi for _, hi in ranges)


def _probe_scenes(config: Config, n: int) -> list[SceneSpec]:
    """A CROSSED (family x room size x absorption) design, ordered by Sabine T60.

    Crossed, not a diagonal. Walking one parameter `f` from (large room, low α) to
    (small room, high α) makes size and absorption perfectly anti-correlated, so
    no result from it can distinguish "the deficit grows with room size" from "the
    deficit grows as absorption falls" — and it never renders the two off-diagonal
    corners. That is the same confound the support law was already burned by,
    where rooms that were all scalings of one shoebox made room size and decay
    time inseparable until a crossed probe refuted the fitted law outright.

    Every axis is read from config: the families the config declares, each one's
    declared dims, and the union of the declared absorption regimes. `n` selects
    how many absorption levels the cross uses; the family and size axes are taken
    whole, because dropping a level from either is what produced the diagonal.
    """
    families = config.scenes.geometry_families
    alpha_lo, alpha_hi = _absorption_support(config)
    n_alpha = max(2, n // (2 * len(families)))
    alphas = [alpha_lo + i * (alpha_hi - alpha_lo) / (n_alpha - 1)
              for i in range(n_alpha)]

    scenes: list[SceneSpec] = []
    for family, spec in sorted(families.items()):
        for corner, pick in (("min", 0), ("max", 1)):
            dims = tuple(float(bounds[pick]) for bounds in spec.dims)
            for alpha in alphas:
                lx, ly, lz = dims
                m = config.scenes.margins
                src = (m.wall, m.wall, min(1.5, lz - m.ceiling))
                rcv = (lx - m.wall, ly - m.wall, min(1.5, lz - m.ceiling))
                scenes.append(SceneSpec(
                    scene_id=_scene_id(len(alphas), len(scenes), dims, alpha),
                    seed=_scene_seed(dims, alpha),
                    geometry_family=family,
                    dims=dims, material_absorption=float(alpha),
                    source_pos=src, receiver_pos=rcv,
                ))
    return sorted(scenes, key=lambda s: -sabine_rt60(*box_volume_and_surface(s.dims),
                                                     s.material_absorption))


def _band_energy_totals(config: Config, ir_multichannel: np.ndarray, n_limit: int) -> dict:
    """Total in-band energy per eval band, over the first `n_limit` post-onset
    samples — the third quantity the dataset-render gate declares
    (docs/design_spec.md §11.1 condition ii).

    `convergence.band_energy_frac` has been declared in `configs/base.yaml` since
    the tolerance block was written and nothing had ever evaluated it, so the
    probe answered two of the three quantities the gate condition names. It is the
    quantity that matters most downstream: the model's output domain is per-band
    log power, so band energy is what it is ultimately scored against.

    WHY A COMMON SAMPLE LIMIT rather than each leg's own integration window: the
    record length itself moves with the ray budget, and the shared
    Lundeby window tracks it too — so integrating each leg to its own end
    would fold a record-length difference into what is meant to be an energy
    comparison. `n_limit` is the SHORTER of the two legs' native records, which
    asks the convergence question and only that one.

    Uses the metric path's own onset alignment and octave filter rather than a
    second implementation, for the reason the T60 formulas are shared.
    """
    ir_w = ir_multichannel[0]
    onset = find_onset(ir_w, config.metric_onset_rel_db)
    aligned = ir_w[onset:onset + n_limit]
    return {
        str(fc): float(_band_energy(aligned, float(fc), config.sample_rate).sum())
        for fc in config.iso_eval_freqs
    }


def _iso(config: Config, ir_multichannel: np.ndarray) -> dict:
    """Reported ISO quantities, THROUGH THE PRODUCTION PATH.

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
        octave_filter_order=config.metric_octave_filter.order,
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
                    help="the CHECK leg, as a multiple of high_ray_budget. The question is "
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
        #
        # A missing `native_ir_samples` REFUSES rather than defaulting to 0: a 0
        # here would integrate a zero-length slice, total every band to 0.0, and
        # be recorded as "the reference leg carries zero energy in this band" — a
        # claim about the render, caused by a missing metadata key. The quantity
        # this whole probe exists to measure would come back as a physical fact
        # that nothing measured.
        onsets = {}
        for leg in ("high", "reference"):
            native = row["legs"][leg].get("native_ir_samples")
            if not native or int(native) <= 0:
                raise ValueError(
                    f"scene {row['scene_id']!r} leg {leg!r}: backend "
                    f"{config.simulator.name!r} declared no positive "
                    f"`native_ir_samples`, so the common band-energy window "
                    f"cannot be established. A probe that cannot establish its "
                    f"own window must refuse, not report zeros."
                )
            onsets[leg] = find_onset(
                row["legs"][leg]["_ir"][0], config.metric_onset_rel_db)[0]
        # POST-ONSET duration, per leg, so both integrate the same physical span
        # of their OWN record. A common absolute sample count would let the leg
        # with the longer native record contribute real decay where the shorter
        # contributes only zero padding — a bias with a fixed sign, against a 5 %
        # tolerance.
        n_common = min(int(row["legs"][leg]["native_ir_samples"]) - onsets[leg]
                       for leg in ("high", "reference"))
        for leg in ("high", "reference"):
            row["legs"][leg]["band_energy"] = _band_energy_totals(
                config, row["legs"][leg].pop("_ir"), n_common)
        row["band_energy_window_samples"] = n_common
        row["band_energy_onset_samples"] = onsets

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

    (args.out / "convergence_results.json").write_text(json.dumps(results, indent=2))

    # Summary — scored vs attempted per metric, never a bare pass/fail.
    print("\n=== ray-budget tolerance check (NOT a CI-backed convergence claim) ===")
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
    print(f"written: {args.out / 'convergence_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
