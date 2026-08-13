#!/usr/bin/env python3
"""What actually sets the length of the record GSound-SIR returns?

The record-length gate in `gen-scenes` refuses a scene whose decay its backend
cannot capture, so it needs to predict realized support BEFORE any render exists.
That prediction is a config-declared law, and the shipped one
(`support_s = c * T60**k`) is refuted: over the 44 retained renders it leaves
11.7 % rms residual against 2.6 % for a volume form, and in a joint fit T60's
exponent is -0.001 +/- 0.011 — indistinguishable from zero.

THE PROBLEM THIS PROBE EXISTS FOR: those 44 renders cannot say what should
replace it. `rd17_convergence_probe._probe_scenes` scales all three dimensions by
one interpolation factor, so every room it has ever produced is geometrically
SIMILAR, and for similar boxes the mean free path 4V/S is proportional to
V**(1/3) exactly. Measured over those renders, corr(log V, log 4V/S) = 1.0000.
Volume and mean free path are the same regressor there, and a third candidate
fits better than either with ONE free parameter instead of two:

    support ~ diffuse_depth * (4V/S) / c     rms 1.07 %, one parameter
    support ~ c * V**a                       rms 1.74 %, two parameters
    support ~ c * V**a * T60**k              rms 1.57 %, three parameters

The third is a MECHANISM, not a curve fit — `diffuse_depth` reflections, each
travelling one mean free path, at the speed of sound — and it is what the open
row against `diffuse_depth` says: that knob is physically a time bound.

The three disagree where it matters. On the declared corridor family
(`test_geometry_shift`, 30 scenes), 30 x 1.5 x 2.4 m has V = 108 m^3 but
4V/S = 1.79 m, a shape no shoebox in the existing probe set reaches:

    volume law  -> 1.136 s support -> 47.3 dB of decay range -> ADMITTED
    depth law   -> 0.661 s support -> 27.5 dB               -> REFUSED

against `scenes.iso_t30_decay_range_db: 45.0`. Opposite gate verdicts on a
reported test split, in the direction that admits a scene whose T30 then silently
under-reads by 23-33 %.

So this probe breaks the confound by construction, in three groups:

  1. DEPTH SWEEP (3 renders, one room). `diffuse_depth` 50 / 100 / 200 with the
     geometry held fixed. Nothing else can separate the depth mechanism from a
     power law in size, because every geometric quantity is constant across these
     three. If support scales with depth, no power law in V survives.
  2. CORRIDORS (3 scenes). Volume matched to shoebox cells but 4V/S far lower —
     the family the two laws disagree about, and the one that governs a split.
  3. SHOEBOX CROSS (6 scenes). 3 volumes x 2 absorptions, fully crossed, so V and
     T60 vary independently rather than along one diagonal.

IRs ARE RETAINED to disk. Every probe before this discarded them, so each new
physical question cost another emulated render; the retained artifacts let the
Schroeder/Lundeby truncation rows be re-derived offline as often as needed.

    python scripts/support_law_probe.py -c configs/base.yaml -c host.yaml \
        --out experiments/support_law
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
from amcd.simulators.base import SceneSpec, build_simulator

#: Rooms are declared here rather than sampled: the probe's whole purpose is to
#: put cells where the candidate laws DISAGREE, and a random draw concentrates in
#: the middle, which is where they agree.
#:
#: (group, label, dims, alpha). Corridor alpha is held at 0.30 across all three so
#: shape varies without absorption varying with it — the mistake this probe exists
#: to undo. The shoebox cross uses the extremes of `material_regimes.mixed`.
#: The three corridors were CHOSEN, by search over the declared corridor range, to
#: minimise |corr(log V, log 4V/S)| across the whole cell set: 1.0000 in every
#: previous probe, 0.7420 here. They sit at the long-thin extreme because that is
#: where a corridor's volume and its mean free path pull hardest in opposite
#: directions — 30 x 1.5 x 3.0 m has 6.3x the volume of the small shoebox and
#: essentially the same mean free path (1.935 vs 1.846 m).
_CELLS: list[tuple[str, str, tuple[float, float, float], float]] = [
    ("corridor", "corr_thin_low", (30.0, 1.5, 3.0), 0.30),
    ("corridor", "corr_thin_tall", (30.0, 1.5, 3.5), 0.30),
    ("corridor", "corr_wide", (30.0, 2.0, 3.0), 0.30),
    ("shoebox", "box_small_absorptive", (3.0, 3.0, 2.4), 0.80),
    ("shoebox", "box_small_live", (3.0, 3.0, 2.4), 0.05),
    ("shoebox", "box_mid_absorptive", (7.5, 5.0, 4.0), 0.80),
    ("shoebox", "box_mid_live", (7.5, 5.0, 4.0), 0.05),
    ("shoebox", "box_large_absorptive", (12.0, 10.0, 5.0), 0.80),
    ("shoebox", "box_large_live", (12.0, 10.0, 5.0), 0.05),
]

#: The depth sweep holds this room fixed and varies only `diffuse_depth`.
_DEPTH_ROOM: tuple[float, float, float] = (7.5, 5.0, 4.0)
_DEPTH_ALPHA = 0.30
_DEPTH_VALUES = (50, 100, 200)


def _scene(label: str, dims: tuple[float, float, float], alpha: float,
           config: Config, family: str) -> SceneSpec:
    """Source and receiver placed by the config's own margins, as `gen-scenes`
    does, so the probe measures the declared population's geometry rather than a
    placement of its own invention."""
    lx, ly, lz = dims
    m = config.scenes.margins
    z = min(1.5, lz - m.ceiling)
    # crc32, not hash(): the builtin is salted per process, so seeds would differ
    # between runs of the same probe and the artifact would not be reproducible.
    return SceneSpec(
        scene_id=label, seed=zlib.crc32(label.encode()), geometry_family=family,
        dims=dims, material_absorption=alpha,
        source_pos=(m.wall, m.wall, z),
        receiver_pos=(lx - m.wall, ly - m.wall, z),
    )


def _render_one(sim, scene: SceneSpec, budget: int, out_dir: Path,
                tag: str, diffuse_depth: int | None) -> dict:
    """One render, with the IR kept. Returns the row; the array goes to disk.

    `diffuse_depth` is passed in rather than read back off the simulator: params
    reach a backend as constructor kwargs, so only backends that choose to keep a
    `params` dict expose one, and asking for it would couple this probe to that
    choice. The caller set the value, so the caller states it.
    """
    t0 = time.time()
    res = sim.render(scene, budget)
    wall = time.time() - t0
    meta = dict(getattr(res, "meta", {}) or {})

    ir_path = out_dir / "irs" / f"{tag}.npy"
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ir_path, res.ir)

    volume, surface = box_volume_and_surface(scene.dims)
    return {
        "tag": tag,
        "scene_id": scene.scene_id,
        "dims": list(scene.dims),
        "alpha": scene.material_absorption,
        "volume_m3": volume,
        "surface_area_m2": surface,
        "mean_free_path_m": 4.0 * volume / surface,
        "t60_sabine_s": sabine_rt60(volume, surface, scene.material_absorption),
        "ray_budget": budget,
        "diffuse_depth": diffuse_depth,
        "wall_clock_s": wall,
        "native_ir_samples": meta.get("native_ir_samples"),
        "realized_support_s": meta.get("realized_support_s"),
        "predicted_support_s": meta.get("predicted_support_s"),
        "support_realized_over_predicted": meta.get("support_realized_over_predicted"),
        "ir_path": str(ir_path.relative_to(out_dir)),
    }


def _fit(rows: list[dict], regressors: list[str]) -> dict:
    """Least squares in log space. `regressors` are row keys; the constant is
    implicit. Returns coefficients and BOTH residual summaries — rms is what a
    fit is usually judged on, max is what a GATE is judged on, because the gate
    fails on its worst scene, not its average one."""
    y = np.log([r["realized_support_s"] for r in rows])
    columns = [np.ones(len(rows))]
    columns += [np.log([r[key] for r in rows]) for key in regressors]
    X = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = np.exp(y - X @ beta) - 1.0
    return {
        "regressors": regressors,
        "coefficient": float(np.exp(beta[0])),
        "exponents": {k: float(b) for k, b in zip(regressors, beta[1:])},
        "rms_pct": float(100.0 * np.sqrt((resid**2).mean())),
        "max_abs_pct": float(100.0 * np.abs(resid).max()),
    }


def _depth_law(rows: list[dict], speed_of_sound: float) -> dict:
    """The one-parameter mechanism: support = k * depth * (4V/S) / c."""
    pred = np.array([r["diffuse_depth"] * r["mean_free_path_m"] / speed_of_sound
                     for r in rows])
    realized = np.array([r["realized_support_s"] for r in rows])
    k = float((realized / pred).mean())
    resid = realized / (k * pred) - 1.0
    return {
        "form": "k * diffuse_depth * (4V/S) / c",
        "k": k,
        "rms_pct": float(100.0 * np.sqrt((resid**2).mean())),
        "max_abs_pct": float(100.0 * np.abs(resid).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", action="append", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="ray budgets for the corridor/shoebox cells. Defaults to "
                         "the config's low and high budgets. The law is FITTED at "
                         "the lowest of them: realized support rises with budget, "
                         "and a gate that must not over-predict belongs at the "
                         "conservative end.")
    args = ap.parse_args()

    config = Config.load(*args.config)
    budgets = args.budgets or [config.low_ray_budget, config.high_ray_budget]
    args.out.mkdir(parents=True, exist_ok=True)

    def _sim(depth: int | None = None):
        params = dict(config.simulator.params)
        if depth is not None:
            params["diffuse_depth"] = depth
        return build_simulator(
            config.simulator.name, params, n_channels=config.n_channels,
            n_samples=config.n_samples, sample_rate=config.sample_rate,
        )

    rows: list[dict] = []

    # The depth sweep asks a question only a backend with a reflection-order bound
    # can answer. The scaffold fills its whole window by construction, so running
    # it there would produce three identical rows and read as evidence of
    # depth-independence — the opposite of the truth.
    has_depth = "diffuse_depth" in config.simulator.params
    if not has_depth:
        print(f"=== depth sweep SKIPPED: backend {config.simulator.name!r} declares "
              f"no `diffuse_depth`, so there is no reflection-order bound to vary ===")
    print(f"=== depth sweep: {_DEPTH_ROOM} m, alpha {_DEPTH_ALPHA}, "
          f"budget {budgets[0]} ===" if has_depth else "")
    for depth in (_DEPTH_VALUES if has_depth else ()):
        scene = _scene(f"depth_{depth}", _DEPTH_ROOM, _DEPTH_ALPHA, config, "shoebox")
        row = _render_one(_sim(depth), scene, budgets[0], args.out,
                          f"depth_{depth}", depth)
        row["group"] = "depth"
        rows.append(row)
        print(f"  depth={depth:4d}  support={row['realized_support_s']:.4f}s  "
              f"{row['wall_clock_s']:.1f}s", flush=True)

    for group, label, dims, alpha in _CELLS:
        for budget in budgets:
            scene = _scene(label, dims, alpha, config, group)
            row = _render_one(_sim(), scene, budget, args.out, f"{label}_b{budget}",
                              config.simulator.params.get("diffuse_depth"))
            row["group"] = group
            rows.append(row)
            support = row["realized_support_s"]
            print(f"  {label:22s} b={budget:7d}  V={row['volume_m3']:6.1f}  "
                  f"4V/S={row['mean_free_path_m']:5.3f}  "
                  f"support={f'{support:.4f}s' if support else 'UNREPORTED'}  "
                  f"{row['wall_clock_s']:.1f}s", flush=True)

    # ── The comparison ────────────────────────────────────────────────────────
    fit_rows = [r for r in rows if r["group"] != "depth" and r["ray_budget"] == min(budgets)]
    corridors = [r for r in fit_rows if r["group"] == "corridor"]
    speed = float(config.simulator.params["speed_of_sound_m_s"])

    # A backend that reports no realized support cannot answer this probe's
    # question, and fitting a law to the rows that DID report would silently fit a
    # subset. Refuse, naming the count — never render an unscored quantity as a
    # number.
    unreported = [r["tag"] for r in fit_rows if not r["realized_support_s"]]
    if unreported:
        (args.out / "support_law_results.json").write_text(json.dumps(
            {"rows": rows, "candidates": None,
             "refused": f"{len(unreported)} of {len(fit_rows)} renders reported no "
                        f"realized_support_s"}, indent=2))
        print(f"\nREFUSED: {len(unreported)} of {len(fit_rows)} renders carry no "
              f"`realized_support_s` in their meta, so there is no realized record "
              f"length to fit a law to. Backend {config.simulator.name!r} does not "
              f"report one — this probe needs a backend whose record length is an "
              f"emergent property, which is the whole reason the law exists.")
        print(f"written: {args.out / 'support_law_results.json'}")
        return 1

    candidates = {
        "volume": _fit(fit_rows, ["volume_m3"]),
        "volume_t60": _fit(fit_rows, ["volume_m3", "t60_sabine_s"]),
        "mean_free_path": _fit(fit_rows, ["mean_free_path_m"]),
        "t60_only": _fit(fit_rows, ["t60_sabine_s"]),
    }
    if has_depth:
        candidates["depth_mechanism"] = _depth_law(fit_rows, speed)

    print(f"\n=== candidate laws, fitted at budget {min(budgets)} (n={len(fit_rows)}) ===")
    for name, f in candidates.items():
        print(f"  {name:18s} rms={f['rms_pct']:6.2f}%  max={f['max_abs_pct']:6.2f}%")

    # The gate fails on the corridor family or it does not, so that is where the
    # laws are scored — an average over a set dominated by shoeboxes would hide
    # exactly the disagreement this probe was built to resolve.
    print("\n=== corridor cells only — the family the laws disagree about ===")
    for r in corridors:
        depth_pred = candidates["depth_mechanism"]["k"] * r["diffuse_depth"] * \
            r["mean_free_path_m"] / speed
        vol = candidates["volume"]
        vol_pred = vol["coefficient"] * r["volume_m3"] ** vol["exponents"]["volume_m3"]
        print(f"  {r['scene_id']:12s} realized={r['realized_support_s']:.4f}s  "
              f"volume-law={vol_pred:.4f}s ({100*(vol_pred/r['realized_support_s']-1):+.1f}%)  "
              f"depth-law={depth_pred:.4f}s "
              f"({100*(depth_pred/r['realized_support_s']-1):+.1f}%)")

    print("\n=== depth sweep — does support track diffuse_depth? ===")
    depth_rows = [r for r in rows if r["group"] == "depth"]
    base = next(r for r in depth_rows if r["diffuse_depth"] == 100)
    for r in sorted(depth_rows, key=lambda r: r["diffuse_depth"]):
        print(f"  depth={r['diffuse_depth']:4d}  support={r['realized_support_s']:.4f}s  "
              f"ratio to depth=100: {r['realized_support_s'] / base['realized_support_s']:.3f}  "
              f"(proportional would be {r['diffuse_depth'] / 100:.3f})")

    (args.out / "support_law_results.json").write_text(json.dumps(
        {"rows": rows, "candidates": candidates,
         "fitted_at_ray_budget": min(budgets),
         "speed_of_sound_m_s": speed}, indent=2))
    total = sum(r["wall_clock_s"] for r in rows)
    print(f"\n{len(rows)} renders, {total / 60:.1f} min wall-clock; IRs retained under "
          f"{args.out / 'irs'}")
    print(f"written: {args.out / 'support_law_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
