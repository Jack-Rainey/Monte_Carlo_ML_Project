"""gen-scenes stage: deterministic procedural scene specs from config + seed.

Regime-aware and fully config-driven (design_spec §6.1): the id regime is the
controlled baseline; each shift split (config.splits with an `axes` override)
perturbs exactly ONE axis (geometry / placement / material). The generator reads
every sampling range from `config.scenes` — it hardcodes nothing and branches on
the axis *value*, never on a split name, so a new shift axis or geometry family
is a config edit, not a code change.

A generated scene carries its target split name in `split_regime` ("id" for the
id pool, else the shift split name) so data/splits.py can route it with no
name mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..acoustics import eyring_rt60, sabine_rt60
from ..config import Config, Margins, PlacementRegime
from ..runtime import Verbosity, emit
from ..simulators.base import SceneSpec, simulator_min_separation


def _sample_dims(
    geometry: str,
    rng: np.random.Generator,
    geometry_families: dict[str, dict],
) -> tuple[float, float, float]:
    ranges = geometry_families[geometry].dims  # [[lo,hi], [lo,hi], [lo,hi]]
    return tuple(float(rng.uniform(lo, hi)) for lo, hi in ranges)


def _placement_bounds(
    dims: tuple[float, float, float],
    margins: "Margins",
    height_range: list[float] | None,
) -> tuple[list[float], list[float]]:
    """Admissible (lo, hi) box for a source or receiver, per axis.

    Raises rather than clamping when a declared `height_range` does not fit
    between floor and ceiling margins: silently narrowing it would generate a
    height distribution nobody declared.
    """
    lx, ly, lz = dims
    lo = [margins.wall, margins.wall, margins.floor]
    hi = [lx - margins.wall, ly - margins.wall, lz - margins.ceiling]
    if height_range is not None:
        z_lo, z_hi = height_range
        if z_lo < lo[2] or z_hi > hi[2]:
            raise ValueError(
                f"placement height_range {height_range} does not fit inside the "
                f"admissible height band [{lo[2]}, {hi[2]}] for a room of height "
                f"{lz} m (floor margin {margins.floor}, ceiling margin "
                f"{margins.ceiling}). Widen the room range or narrow the heights."
            )
        lo[2], hi[2] = z_lo, z_hi
    for axis, (a, b) in enumerate(zip(lo, hi)):
        if a >= b:
            raise ValueError(
                f"margins leave no room on axis {axis} for dims {dims}: "
                f"admissible range [{a}, {b}] is empty."
            )
    return lo, hi


def _sample_positions(
    placement: str,
    dims: tuple[float, float, float],
    rng: np.random.Generator,
    placement_regimes: dict[str, "PlacementRegime"],
    margins: "Margins",
    max_attempts: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict]:
    """Sample a source/receiver pair; returns (src, rcv, stats).

    `stats` carries the rejection accounting: attempts, and how many draws fell
    BELOW the minimum separation versus ABOVE the maximum. Those two have
    opposite acoustic sign — rejecting below strips high-DRR (close) pairs,
    rejecting above strips low-DRR (distant) ones — and which dominates depends
    on the room family, so a single acceptance rate conflates them (AC-14).

    When the regime declares a `distance_range`, the pair is resampled JOINTLY
    until it satisfies the constraint. Resampling only the receiver would leave
    the source uniform but the receiver conditioned, which is not the same
    distribution — and source-receiver distance sets the direct-to-reverberant
    ratio, hence C50/D50/EDT, so the difference is scientific, not cosmetic.

    With `distance_range: null` the loop body runs exactly once and issues the
    same two 3-vector `rng.uniform` calls in the same order as before this
    constraint existed, so unconstrained configs keep their exact RNG stream and
    reproduce their existing datasets bit-for-bit.
    """
    regime = placement_regimes[placement]
    lo, hi = _placement_bounds(dims, margins, regime.height_range)

    if regime.type == "corner":
        # Receiver biased toward the (min, min) corner in the HORIZONTAL plane only
        # (AC-10). Applying the fraction to z as well collapsed the receiver height
        # band — with height_range [1.2, 1.8] and corner_frac 0.2, receivers were
        # confined to [1.2, 1.32]. A declared height range is an ergonomic band, not
        # a room boundary: 1.2 m is 1.2 m off the floor either way, so biasing z buys
        # no boundary proximity while silently narrowing a reported robustness split.
        rcv_hi = [lo[i] + regime.corner_frac * (hi[i] - lo[i]) for i in (0, 1)] + [hi[2]]
    else:  # interior: uniform anywhere in the admissible box
        rcv_hi = hi

    # Greatest separation this box admits at all — the 10 m cap is inert in a small
    # shoebox (max ~3.7 m) but does the rejecting in a long corridor (AC-14).
    max_reachable = float(np.linalg.norm(np.subtract(hi, lo)))
    stats = {"attempts": 0, "below_min": 0, "above_max": 0,
             "max_reachable_m": max_reachable}

    for attempt in range(1, max_attempts + 1):
        stats["attempts"] = attempt
        src = tuple(float(v) for v in rng.uniform(lo, hi))
        rcv = tuple(float(v) for v in rng.uniform(lo, rcv_hi))
        if regime.distance_range is None:
            return src, rcv, stats
        # Either bound may be null (RD-48): a backend imposes a minimum with no
        # matching maximum, so each side is tested only when it is declared.
        d_lo, d_hi = regime.distance_range
        d = float(np.linalg.norm(np.subtract(src, rcv)))
        if d_lo is not None and d < d_lo:
            stats["below_min"] += 1
        elif d_hi is not None and d > d_hi:
            stats["above_max"] += 1
        else:
            return src, rcv, stats

    raise RuntimeError(
        f"placement regime {placement!r}: could not satisfy distance_range "
        f"{regime.distance_range} in {max_attempts} attempts for a room of dims "
        f"{dims}. The constraint may be geometrically unreachable here (max "
        f"separation in this box is {max_reachable:.2f} m; "
        f"{stats['below_min']} draws fell below the minimum, "
        f"{stats['above_max']} above the maximum) — widen distance_range, adjust "
        f"the geometry range, or raise scenes.max_placement_attempts."
    )


def _check_regimes_clear_backend_floor(config: Config) -> float:
    """Reject any placement regime that could emit a scene the backend cannot render.

    The declared-config half of AC-13/F-48, checked before a single scene exists.
    Without it, base.yaml's unconstrained regimes emitted separations below every
    shipped backend's floor (measured P(d < 0.3 m) = 0.186 %/scene → ~67 % chance a
    600-scene run aborts), and the only guard fired INSIDE render — mid-batch,
    hours into an emulated render, with the stage sentinel never written.

    EVERY declared regime is checked, not just the one the id baseline names
    (RD-45): `near_corner` is the regime behind `test_placement_shift`, and a
    regime that is unused today is a trap for the config that selects it tomorrow.

    The backend floor is a LOWER LIMIT on the researcher's choice, never its
    source (RD-57) — the message therefore says to raise the config value, and
    the scientifically motivated minimum stays in the config where it belongs.
    """
    floor = simulator_min_separation(config)
    if floor <= 0.0:
        return floor
    offenders = []
    for name, regime in config.scenes.placement_regimes.items():
        rng = regime.distance_range
        declared_min = None if rng is None else rng[0]
        if declared_min is None or declared_min < floor:
            offenders.append((name, declared_min))
    if offenders:
        lines = "\n".join(
            f"    {name}: distance_range lower bound = "
            + ("null (no minimum declared)" if lo is None else f"{lo} m")
            for name, lo in offenders
        )
        raise ValueError(
            f"simulator {config.simulator.name!r} cannot render a source-receiver "
            f"separation below {floor} m, but these placement regimes admit closer "
            f"pairs:\n{lines}\n"
            f"Declare a `distance_range` lower bound of at least {floor} m on each. "
            f"That bound is a research choice about the scene distribution — the "
            f"backend floor is only a lower limit on it, not a recommended value."
        )
    return floor


def _sample_material(
    material: str,
    rng: np.random.Generator,
    material_regimes: dict[str, dict],
) -> float:
    lo, hi = material_regimes[material].absorption
    return float(rng.uniform(lo, hi))


def _generation_plan(config: Config) -> list[tuple[str, int, dict[str, str], int | None]]:
    """Ordered (split_regime, count, axis_overrides, seed) entries to generate.

    In frac mode the id-pool is ONE entry tagged "id" (scenes are hash-bucketed
    into train/valid/test_id later, by data/splits.py). In count mode each
    id-pool split is its own entry tagged with its split name, so it is routed
    directly and gets exactly the count it declared.

    Shift splits are always their own entry either way. Config validation
    guarantees the single-axis shift and the sizing-mode consistency.
    """
    scenes_cfg = config.scenes
    plan: list[tuple[str, int, dict[str, str], int | None]] = []
    if config.id_pool_is_counted:
        for name, spec in config.id_pool_splits.items():
            plan.append((name, spec.count, {}, spec.seed))
    else:
        plan.append(("id", scenes_cfg.n_id, {}, None))
    for split_name, spec in config.shift_splits.items():
        plan.append((split_name, spec.count, dict(spec.axes), spec.seed))
    return plan


def _room_acoustics(
    dims: tuple[float, float, float],
    absorption: float,
    distance: float,
    *,
    alpha_limit: float,
    ir_duration_s: float,
) -> dict:
    """Closed-form acoustic descriptors of one scene, from geometry alone.

    These are the quantities the RD-29 disclosure is actually about
    (acoustics-reviewer AC-09). Source-receiver DISTANCE alone is nearly blind to
    the thing that matters: direct-to-reverberant ratio depends on d/r_c, and
    absorption moves r_c independently of d — two splits can share a median
    distance and differ by ~11 dB in median DRR.

    Estimates, not measurements: diffuse-field formulae over a shoebox, reported
    so the E1 write-up can characterize the dataset it generated. The rendered
    IRs remain the source of truth for every reported metric.

    VALIDITY (AC-21). The estimates above are reported with no indication of when
    their own premise has failed — and for `test_material_shift`, the very split
    this artifact was built to characterize, it has failed for 100 % of scenes
    (alpha median 0.894, Sabine/Eyring ratio median 2.51, 23 % with r_c larger than
    the room's longest dimension). So each scene now also carries three flags. No
    formula changes and nothing is dropped: a +4.9 dB DRR from a formula outside
    its domain is still reported, but it is reported AS an extrapolation.
    """
    if not distance > 0.0:
        # DRR divides by d²; a coincident pair would silently report +inf into a
        # canonical artifact. Geometrically degenerate anyway — guarded rather
        # than clamped, so it cannot be reported as if it were a real scene
        # (F-42; same family as the AC-13 minimum-separation gap).
        raise ValueError(
            f"source-receiver distance must be > 0 to characterize a scene; got "
            f"{distance}. Declare a placement `distance_range` with a positive "
            f"lower bound."
        )
    lx, ly, lz = dims
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + ly * lz + lx * lz)
    alpha = float(np.clip(absorption, 1e-6, 1.0 - 1e-6))

    # Shared declarations (amcd.acoustics) — the scaffold renders from the same
    # constant and the same formula, so the described room and the rendered room
    # cannot drift apart (AC-24).
    t60_sabine = sabine_rt60(volume, surface, alpha)
    # Eyring is the better estimate at high absorption, where Sabine overpredicts
    # — and ceiling_absorptive reaches α = 0.98.
    t60_eyring = eyring_rt60(volume, surface, alpha)
    # Room constant R = Sα/(1-α); critical distance r_c = sqrt(R/16π).
    room_constant = surface * alpha / (1.0 - alpha)
    r_c = float(np.sqrt(room_constant / (16.0 * np.pi)))
    return {
        "volume_m3": volume,
        "surface_m2": surface,
        "absorption": alpha,
        "t60_sabine_s": float(t60_sabine),
        "t60_eyring_s": float(t60_eyring),
        "critical_distance_m": r_c,
        "d_over_rc": distance / r_c if r_c > 0 else float("inf"),
        # Diffuse-field DRR: direct 1/(4πd²) against the reverberant field 4/R.
        "drr_db": float(10.0 * np.log10(room_constant / (16.0 * np.pi * distance**2))),
        # ── Validity indicators (AC-21) ──────────────────────────────────────
        # Sabine and Eyring agree only for small α; the ratio is a direct,
        # assumption-free readout of how far the diffuse-field model is being
        # stretched (median 2.51 on test_material_shift, max 3.90).
        "sabine_eyring_ratio": float(t60_sabine / t60_eyring),
        "alpha_above_diffuse_limit": bool(alpha > alpha_limit),
        # A critical distance larger than the room means the "reverberant field"
        # the DRR formula divides by does not exist inside this room at all.
        "rc_exceeds_max_dim": bool(r_c > max(dims)),
        # AC-22's realized gate: this scene's decay against the record length.
        # Sabine (the longer estimate) so the flag errs toward declaring a scene
        # unsupported rather than silently truncating it.
        "t60_exceeds_ir_duration": bool(t60_sabine > ir_duration_s),
    }


def _disclose_and_gate_record_length(config: Config, report: dict, verbosity) -> None:
    """Disclose the declared-support corner; gate on the REALIZED over-limit rate.

    AC-22 in the shape RD-56 settled. The two halves are deliberately different in
    kind:

      * The CORNER (`Config.worst_case_t60`) is the product of two independent
        extremes — largest room, lowest absorption — and has near-zero probability
        of being drawn, so gating on it would reject configs whose realized scenes
        are all fine. It is printed and stamped, never used as a threshold.
      * The GATE is `scenes.max_t60_over_ir_duration_frac` applied to the scenes
        that actually exist. That is the population the metrics are computed over.

    The gate is the OVERALL fraction, while the disclosure is per split. Gating
    per split would let the smallest split set the threshold for every other one:
    a single over-limit scene in a 30-scene shift split is 3.3 %, and a tolerance
    declared to permit that would silently also permit 16 scenes in a 500-scene
    train split. The per-split counts still appear in the report and in this
    error, because the shift splits are exactly where the decay distribution
    departs from the id baseline.
    """
    corner = config.worst_case_t60()
    emit(
        verbosity, "progress",
        f"  Declared-support corner: Sabine T60 {corner['t60_sabine_s']:.2f} s "
        f"({corner['geometry_family']} {corner['dims_m']} m at alpha "
        f"{corner['absorption']}) vs ir_duration {corner['ir_duration_s']:.2f} s"
        + ("" if corner["covered_by_record"] else "  — NOT covered by the record"),
    )

    limit = config.scenes.max_t60_over_ir_duration_frac
    per_split = {
        name: (entry["t60_over_ir_duration"]["t60_exceeds_ir_duration"]["count"],
               entry["n_scenes"])
        for name, entry in report.items()
    }
    over = sum(count for count, _ in per_split.values())
    total = sum(n for _, n in per_split.values())
    if total and (over / total) > limit:
        lines = "\n".join(
            f"    {name}: {count}/{n} scenes"
            for name, (count, n) in per_split.items() if count
        )
        raise ValueError(
            f"ir_duration is {config.ir_duration} s, but {over} of {total} scenes "
            f"({over / total:.3%}) exceed it — more than "
            f"scenes.max_t60_over_ir_duration_frac ({limit}) allows:\n{lines}\n"
            f"A T30/EDT fitted over a truncated record measures the truncation, not "
            f"the room. Lengthen ir_duration, narrow the geometry/absorption ranges, "
            f"or raise the declared tolerance and say why. For reference the declared "
            f"support reaches Sabine T60 {corner['t60_sabine_s']:.2f} s "
            f"({corner['geometry_family']} {corner['dims_m']} m at alpha "
            f"{corner['absorption']})."
        )


def _flag_counts(room_stats: list[dict], flags: tuple[str, ...], **context) -> dict:
    """Count and fraction for each named per-scene boolean, plus its context.

    Reported as counts rather than a bare boolean so a reader sees how much of a
    split is affected: "100 % of test_material_shift is outside the diffuse-field
    domain" and "one scene is" are very different disclosures, and the flag alone
    cannot tell them apart (AC-21/AC-22).
    """
    n = len(room_stats)
    out: dict = {"n_scenes": n, **context}
    for flag in flags:
        count = sum(1 for r in room_stats if r[flag])
        out[flag] = {
            "count": count,
            "fraction": (count / n) if n else None,
        }
    return out


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def run_gen_scenes(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    out_dir = run_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove any previously generated specs before writing this set (F-27).
    # Scene ids are positional (`scene_0000`…), so regenerating with FEWER scenes
    # leaves high-numbered orphans behind; render and preprocess glob the
    # directory and would pull them into the dataset, under a different config,
    # while placement_report.json declared the smaller set.
    for stale in out_dir.glob("scene_*.json"):
        stale.unlink()

    # Config-level pre-flight: no scene is generated under a placement regime the
    # active backend could not render (AC-13/F-48/RD-45).
    _check_regimes_clear_backend_floor(config)

    scenes_cfg = config.scenes
    id_axes = dict(scenes_cfg.id_regime)  # {geometry, placement, material}
    plan = _generation_plan(config)

    # Shared stream for entries that declare no seed of their own (frac mode).
    shared_rng = np.random.default_rng(config.seed("scene_generation"))

    scene_idx = 0
    report: dict[str, dict] = {}
    for split_regime, count, overrides, split_seed in plan:
        axes = {**id_axes, **overrides}  # controlled: exactly one axis differs from id
        # A split with its own seed is generated independently and reproducibly;
        # otherwise it draws from the shared scene_generation stream.
        rng = np.random.default_rng(split_seed) if split_seed is not None else shared_rng

        regime = scenes_cfg.placement_regimes[axes["placement"]]
        attempts_total = 0
        distances: list[float] = []
        src_heights: list[float] = []
        rcv_heights: list[float] = []
        rejected = {"below_min": 0, "above_max": 0}
        unreachable_max = 0
        room_stats: list[dict] = []

        for _ in range(count):
            scene_seed = int(rng.integers(0, 2**31))
            scene_rng = np.random.default_rng(scene_seed)

            dims = _sample_dims(axes["geometry"], scene_rng, scenes_cfg.geometry_families)
            src, rcv, stats = _sample_positions(
                axes["placement"], dims, scene_rng,
                scenes_cfg.placement_regimes, scenes_cfg.margins,
                scenes_cfg.max_placement_attempts,
            )
            absorption = _sample_material(axes["material"], scene_rng, scenes_cfg.material_regimes)

            attempts_total += stats["attempts"]
            rejected["below_min"] += stats["below_min"]
            rejected["above_max"] += stats["above_max"]
            declared_max = None if regime.distance_range is None else regime.distance_range[1]
            if declared_max is not None and stats["max_reachable_m"] < declared_max:
                unreachable_max += 1
            distance = float(np.linalg.norm(np.subtract(src, rcv)))
            distances.append(distance)
            src_heights.append(src[2])
            rcv_heights.append(rcv[2])
            room_stats.append(_room_acoustics(
                dims, absorption, distance,
                alpha_limit=scenes_cfg.diffuse_field_alpha_limit,
                ir_duration_s=config.ir_duration,
            ))

            spec = SceneSpec(
                scene_id=f"scene_{scene_idx:04d}",
                seed=scene_seed,
                geometry_family=axes["geometry"],
                dims=dims,
                material_absorption=absorption,
                source_pos=src,
                receiver_pos=rcv,
                sim_params={},
                split_regime=split_regime,
                regime_axes=dict(axes),
            )

            (out_dir / f"{spec.scene_id}.json").write_text(
                json.dumps(spec.to_dict(), indent=2)
            )
            scene_idx += 1

        report[split_regime] = {
            "n_scenes": count,
            "seed": split_seed,
            "placement_regime": axes["placement"],
            "height_range_declared": regime.height_range,
            "distance_range_declared": regime.distance_range,
            "placement_attempts": attempts_total,
            "acceptance_rate": (count / attempts_total) if attempts_total else None,
            # Split by SIDE (AC-14): rejecting below the minimum strips close,
            # high-DRR pairs; rejecting above the maximum strips distant, low-DRR
            # ones. Shoeboxes reject mostly below and corridors mostly above, so
            # one pooled rate hides which tail of the DRR distribution was cut.
            "rejected_below_min": rejected["below_min"],
            "rejected_above_max": rejected["above_max"],
            "rooms_that_cannot_reach_max_distance": unreachable_max,
            "source_receiver_distance_m": _summarize(distances) if distances else None,
            # Source and receiver kept SEPARATE (AC-10): pooling them hid a
            # corner-bias bug that collapsed only the receiver height band.
            "source_height_m": _summarize(src_heights) if src_heights else None,
            "receiver_height_m": _summarize(rcv_heights) if rcv_heights else None,
            # The DRR-relevant descriptors (AC-09) — see _room_acoustics. Guarded
            # like the three siblings above (F-46): `_summarize` reduces over the
            # list, so an empty split reached numpy's bare "zero-size array to
            # reduction operation minimum" instead of a named, readable summary.
            **{
                f"{key}": (
                    _summarize([r[key] for r in room_stats]) if room_stats else None
                )
                for key in ("volume_m3", "t60_sabine_s", "t60_eyring_s",
                            "critical_distance_m", "d_over_rc", "drr_db",
                            "sabine_eyring_ratio")
            },
            # Validity of the estimates directly above (AC-21) and of the record
            # length against them (AC-22). Counts, not just a flag, so the reader
            # sees HOW MUCH of a split is outside the diffuse-field domain rather
            # than only that some of it is.
            "diffuse_field_validity": _flag_counts(
                room_stats,
                ("alpha_above_diffuse_limit", "rc_exceeds_max_dim"),
                alpha_limit=scenes_cfg.diffuse_field_alpha_limit,
            ),
            "t60_over_ir_duration": _flag_counts(
                room_stats, ("t60_exceeds_ir_duration",),
                ir_duration_s=config.ir_duration,
            ),
        }

    # Canonical, not verbosity-gated. Two jobs: it is the rejection-sampling
    # accounting ("nothing leaves a result silently" — how many draws were
    # discarded to satisfy distance_range, RD-37), and it QUANTIFIES the realized
    # source-receiver distance distribution per split. The latter is what lets the
    # E1 report state exactly which distance distribution stood in for Research
    # I's unspecified mid_pair/far_pair sub-ranges (RD-29).
    (out_dir / "placement_report.json").write_text(json.dumps(report, indent=2))

    _disclose_and_gate_record_length(config, report, verbosity)

    n_shift = sum(sp.count for sp in config.shift_splits.values())
    emit(
        verbosity, "progress",
        f"  Generated {scene_idx} scene specs "
        f"({scene_idx - n_shift} id + {n_shift} shift) → {out_dir}",
    )
