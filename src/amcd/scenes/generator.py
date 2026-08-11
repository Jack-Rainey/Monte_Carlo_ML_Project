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
import math
from pathlib import Path

import numpy as np

from ..acoustics import critical_distance, diffuse_field_drr_db, eyring_rt60, sabine_rt60
from ..config import Config, Margins, PlacementRegime
from ..runtime import Verbosity, emit
from ..simulators.base import SceneSpec, simulator_min_separation

#: c·SABINE_K, which is 24·ln10 by SABINE_K's own definition (`amcd.acoustics`).
#: The ISO 3382-1 §5.3 minimum measurement distance d_min = 2·sqrt(V/(c·T60)) is
#: therefore free of the speed of sound in BOTH its Sabine and its Eyring form —
#: substituting either T60 leaves c only inside this product, and the volume
#: cancels with it. No speed-of-sound value appears in this module (AC-30).
_C_TIMES_SABINE_K = 24.0 * math.log(10.0)


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

    The declared-config half of AC-13/F-48, checked before a single scene exists —
    the only other guard fires inside render, mid-batch, with the stage sentinel
    never written.

    EVERY declared regime is checked, not just the one the id baseline names
    (RD-45): a regime unused today is a trap for the config that selects it
    tomorrow.

    The backend floor is a LOWER LIMIT on the researcher's choice, never its
    source (RD-57), so the message says to raise the config value and the
    scientifically motivated minimum stays in the config where it belongs.
    """
    floor = simulator_min_separation(config)

    # A DECLARED minimum is required unconditionally, independently of the backend
    # (F-61). The whole check used to sit behind `if floor <= 0.0: return`, so a
    # backend legitimately declaring a zero floor — a valid value under the
    # `Simulator` contract, and one no shipped backend exercises — made both this
    # check and render's pre-flight early-return. At that point `distance_range:
    # null` is legal again and the pre-F-48 near-field population returns with no
    # error at all: `_room_acoustics` rejects only d == 0 exactly, so d = 0.001 m
    # would report a ~+60 dB DRR into placement_report.json as if it were measured.
    #
    # The researcher's minimum is a SCIENTIFIC choice about the scene distribution
    # (base.yaml argues 1.0 m from the critical distance and ISO 3382-1 §5.3); the
    # backend floor is only a lower limit on that choice (RD-57). So the two are
    # now checked separately, in that order.
    missing = [
        name for name, regime in config.scenes.placement_regimes.items()
        if regime.distance_range is None or regime.distance_range[0] is None
    ]
    if missing:
        raise ValueError(
            f"these placement regimes declare no minimum source-receiver "
            f"separation: {', '.join(sorted(missing))}.\n"
            f"Every regime must declare a `distance_range` lower bound. It is a "
            f"research choice about the scene distribution — at very short range "
            f"the direct term dominates, C50/D50 saturate and the diffuse tail this "
            f"study is about is buried — so it is required whatever the active "
            f"backend's own floor happens to be, including zero."
        )
    if floor <= 0.0:
        return floor
    offenders = [
        (name, regime.distance_range[0])
        for name, regime in config.scenes.placement_regimes.items()
        if regime.distance_range[0] < floor
    ]
    if offenders:
        lines = "\n".join(
            f"    {name}: distance_range lower bound = {lo} m" for name, lo in offenders
        )
        raise ValueError(
            f"simulator {config.simulator.name!r} cannot render a source-receiver "
            f"separation below {floor} m, but these placement regimes admit closer "
            f"pairs:\n{lines}\n"
            f"Raise each `distance_range` lower bound to at least {floor} m. "
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
    characterization: str,
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
    the room's longest dimension). So each scene now also carries four flags. No
    formula changes and nothing is dropped: a +4.9 dB DRR from a formula outside
    its domain is still reported, but it is reported AS an extrapolation.

    NOT EVERY GEOMETRY IS AN ENCLOSURE (RD-64). Every quantity here — Sabine/Eyring
    T60, room constant, critical distance, DRR — is derived from `dims` on the
    assumption of a closed box. That holds for shoebox and corridor and fails for
    the roadmap's outdoor and partially-open scenes (paper §6), which design_spec
    §6 says the architecture must not preclude. So the geometry family DECLARES its
    `characterization`, and a family declaring "none" gets a recorded reason
    instead of a number — per "nothing leaves a result silently" — rather than
    meaningless closed-box values in the canonical placement_report.json.
    """
    if characterization == "none":
        # An unmodelled geometry is UNCHARACTERIZED, not zero and not NaN: every
        # numeric key is absent so no consumer can average it, and the reason
        # travels with the scene.
        return {
            "characterization": "none",
            "uncharacterized_reason": (
                "geometry family declares characterization: none — it is not a "
                "closed enclosure, so Sabine/Eyring T60, room constant, critical "
                "distance and diffuse-field DRR are undefined for it"
            ),
            # `t60_exceeds_ir_duration` is OMITTED, not False (F-71). A False here
            # reads as "measured, and within the record", so the scene entered the
            # record-length gate's denominator as passing and N uncharacterized
            # scenes shrank the overall over-limit fraction by N/(N+M) — a dataset
            # whose enclosed scenes breach the limit could pass by adding
            # non-enclosures. Omission is what `_scene_is_characterized` reads to
            # exclude the scene from both the fraction and the gate.
        }
    if characterization != "sabine":
        raise ValueError(
            f"unknown geometry characterization {characterization!r}; expected "
            f"'sabine' or 'none'."
        )
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
    # Room constant, critical distance and diffuse-field DRR come from
    # `amcd.acoustics` for the AC-24 reason the T60s already did (RD-75): the
    # scaffold now scales its reverberant tail from the SAME formulas, so the DRR
    # this report publishes and the DRR the render realizes cannot diverge.
    r_c = critical_distance(surface, alpha)

    # ISO 3382-1 §5.3 minimum measurement distance (AC-30): d_min = 2·sqrt(V/(c·T60)),
    # which reduces to 2·sqrt(αS/(c·K)) for Sabine and the same with −ln(1−α) in place
    # of α for Eyring — volume-independent, and free of c (see _C_TIMES_SABINE_K).
    # Reported and counted rather than enforced: the criterion is PER SCENE, varying
    # with each scene's own absorption and surface, while the config declares ONE
    # global placement floor, so the floor cannot satisfy it everywhere. The
    # per-scene criterion stays deferred; how far the floor falls short does not.
    d_min_sabine = 2.0 * math.sqrt(alpha * surface / _C_TIMES_SABINE_K)
    d_min_eyring = 2.0 * math.sqrt(-math.log1p(-alpha) * surface / _C_TIMES_SABINE_K)

    return {
        "characterization": "sabine",
        "volume_m3": volume,
        "surface_m2": surface,
        "absorption": alpha,
        "t60_sabine_s": float(t60_sabine),
        "t60_eyring_s": float(t60_eyring),
        "critical_distance_m": r_c,
        "d_over_rc": distance / r_c if r_c > 0 else float("inf"),
        "iso_min_distance_sabine_m": d_min_sabine,
        "iso_min_distance_eyring_m": d_min_eyring,
        # Diffuse-field DRR: direct 1/(4πd²) against the reverberant field 4/R.
        "drr_db": diffuse_field_drr_db(surface, alpha, distance),
        # ── Validity indicators (AC-21) ──────────────────────────────────────
        # Sabine and Eyring agree only for small α; the ratio is a direct,
        # assumption-free readout of how far the diffuse-field model is being
        # stretched (median 2.51 on test_material_shift, max 3.90).
        "sabine_eyring_ratio": float(t60_sabine / t60_eyring),
        "alpha_above_diffuse_limit": bool(alpha > alpha_limit),
        # A critical distance larger than the room means the "reverberant field"
        # the DRR formula divides by does not exist inside this room at all.
        "rc_exceeds_max_dim": bool(r_c > max(dims)),
        # The SHARPEST per-scene condition, and the one the shipped flags were
        # missing (AC-29). Inside the critical distance the receiver sits in the
        # DIRECT field, so the diffuse-field DRR being reported has no reverberant
        # field to divide by. `d_over_rc` was already computed and summarized as a
        # value but never flagged or counted, so the per-split validity summary
        # omitted the strictest indicator it already had in hand. MEASURED over the
        # realized base.yaml set: this fires for 92.5 % of test_material_shift and
        # 17.2 % of id, against `rc_exceeds_max_dim`'s 35.0 % and 0.0 % — an
        # under-report of ~2.6x on the split the flags were built for.
        "receiver_inside_critical_distance": bool(distance < r_c),
        # AC-30's realized disclosure: this scene's own ISO 3382-1 §5.3 floor against
        # the separation actually drawn. Both estimates are carried because they
        # disagree substantially at high α, and the reader needs to see by how much.
        "below_iso_min_distance_sabine": bool(distance < d_min_sabine),
        "below_iso_min_distance_eyring": bool(distance < d_min_eyring),
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

    An overall fraction can nonetheless hide a split far over on its own, and the
    per-shift breakdown IS the research result — so every split above the declared
    limit is WARNED about unconditionally, whether or not the overall gate trips
    (RD-65). Warning, not gating, for the reason in the paragraph above.

    Both the gate and the warning score only CHARACTERIZED scenes (F-71/RD-94):
    an uncharacterized scene has no closed-form T60, so it can be counted neither
    for nor against the record length, and a gate that scored none of its scenes
    is UNSCORED, never passed.
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
    # (over-limit count, scenes SCORED, scenes ATTEMPTED) per split. An
    # uncharacterized scene (RD-64) carries no `t60_exceeds_ir_duration`, so it
    # leaves the denominator here exactly as `_flag_counts` already drops it from
    # the reported fraction (F-71). `n_uncharacterized` is emitted only when
    # nonzero, so its absence means every scene in the split was scored.
    per_split = {}
    for name, entry in report.items():
        block = entry["t60_over_ir_duration"]
        attempted = entry["n_scenes"]
        scored = attempted - block.get("n_uncharacterized", 0)
        per_split[name] = (block["t60_exceeds_ir_duration"]["count"], scored, attempted)

    # RD-65. Emitted before the overall gate can raise, so a failing run still names
    # the splits responsible.
    for name, (count, scored, attempted) in per_split.items():
        if not scored:
            if attempted:
                emit(verbosity, "warning",
                     f"  WARNING: split {name!r}: 0 of {attempted} scenes are "
                     f"characterized, so its over-limit fraction is UNDEFINED — "
                     f"reported as null, never as 0.0 (RD-64/F-71).")
            continue
        frac = count / scored
        if frac > limit:
            emit(verbosity, "warning",
                 f"  WARNING: split {name!r}: {count}/{scored} scenes ({frac:.3%}) "
                 f"exceed ir_duration {config.ir_duration} s — above this config's "
                 f"own scenes.max_t60_over_ir_duration_frac ({limit}). The gate is "
                 f"the OVERALL fraction and may still pass; a shift split far over on "
                 f"its own is a fact about that split's decay distribution (RD-65).")

    over = sum(count for count, _, _ in per_split.values())
    total = sum(scored for _, scored, _ in per_split.values())
    attempted_total = sum(attempted for _, _, attempted in per_split.values())
    if not total:
        # Scenes exist but none is characterized: the gate has nothing to measure,
        # and falling through would pass silently — F-71's defect one level up, at
        # the outdoor/partially-open config the RD-64 seam exists to enable (RD-94).
        if attempted_total:
            emit(verbosity, "warning",
                 f"  WARNING: the record-length gate scored 0 of {attempted_total} "
                 f"scenes — every geometry family in this config declares "
                 f"characterization: none, so no closed-form T60 exists to compare "
                 f"against ir_duration {config.ir_duration} s. The gate is UNSCORED, "
                 f"not passed (RD-94).")
        return
    if (over / total) > limit:
        lines = "\n".join(
            f"    {name}: {count}/{scored} scenes"
            for name, (count, scored, _) in per_split.items() if count
        )
        excluded = attempted_total - total
        exclusion = (
            f"\n{excluded} of {attempted_total} scenes are excluded from this "
            f"fraction as uncharacterized (RD-64)." if excluded else ""
        )
        raise ValueError(
            f"ir_duration is {config.ir_duration} s, but {over} of {total} scenes "
            f"({over / total:.3%}) exceed it — more than "
            f"scenes.max_t60_over_ir_duration_frac ({limit}) allows:\n{lines}{exclusion}\n"
            f"A T30/EDT fitted over a truncated record measures the truncation, not "
            f"the room. Lengthen ir_duration, narrow the geometry/absorption ranges, "
            f"or raise the declared tolerance and say why. For reference the declared "
            f"support reaches Sabine T60 {corner['t60_sabine_s']:.2f} s "
            f"({corner['geometry_family']} {corner['dims_m']} m at alpha "
            f"{corner['absorption']})."
        )


def _scene_is_characterized(room: dict, flags: tuple[str, ...]) -> bool:
    """True for a `characterization: sabine` scene, False for a `none` one (RD-64):
    a non-enclosure carries a reason instead of the closed-form quantities `flags`
    describe, so it can be counted neither for nor against them."""
    return all(flag in room for flag in flags)


def _flag_counts(room_stats: list[dict], flags: tuple[str, ...], **context) -> dict:
    """Count and fraction for each named per-scene boolean, plus its context.

    Reported as counts rather than a bare boolean so a reader sees how much of a
    split is affected: "100 % of test_material_shift is outside the diffuse-field
    domain" and "one scene is" are very different disclosures, and the flag alone
    cannot tell them apart (AC-21/AC-22).
    """
    # An uncharacterized scene (RD-64) has no closed-form quantities, so it cannot
    # be counted for or against a diffuse-field flag. Excluded from BOTH numerator
    # and denominator, and the exclusion is itself reported — a fraction whose
    # denominator silently shrank is exactly the silent drop the project forbids.
    #
    # `n_uncharacterized` is emitted ONLY when nonzero, so its absence in a report
    # entry means every scene in that split was characterized. That is the contract
    # `_disclose_and_gate_record_length` reads to derive the gate's denominator.
    modelled = [r for r in room_stats if _scene_is_characterized(r, flags)]
    n_uncharacterized = len(room_stats) - len(modelled)
    n = len(modelled)
    out: dict = {"n_scenes": n, **context}
    if n_uncharacterized:
        out["n_uncharacterized"] = n_uncharacterized
        out["uncharacterized_note"] = (
            "scenes whose geometry family declares characterization: none are "
            "excluded from these fractions — the closed-form model they measure "
            "does not apply to a non-enclosure (RD-64)"
        )
    for flag in flags:
        count = sum(1 for r in modelled if r[flag])
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
                characterization=(
                    scenes_cfg.geometry_families[axes["geometry"]].characterization
                ),
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
                    _summarize([r[key] for r in room_stats if key in r])
                    if any(key in r for r in room_stats) else None
                )
                for key in ("volume_m3", "t60_sabine_s", "t60_eyring_s",
                            "critical_distance_m", "d_over_rc", "drr_db",
                            "sabine_eyring_ratio", "iso_min_distance_sabine_m",
                            "iso_min_distance_eyring_m")
            },
            # Validity of the estimates directly above (AC-21) and of the record
            # length against them (AC-22). Counts, not just a flag, so the reader
            # sees HOW MUCH of a split is outside the diffuse-field domain rather
            # than only that some of it is.
            "diffuse_field_validity": _flag_counts(
                room_stats,
                ("alpha_above_diffuse_limit", "rc_exceeds_max_dim",
                 "receiver_inside_critical_distance"),
                alpha_limit=scenes_cfg.diffuse_field_alpha_limit,
            ),
            "t60_over_ir_duration": _flag_counts(
                room_stats, ("t60_exceeds_ir_duration",),
                ir_duration_s=config.ir_duration,
            ),
            # AC-30: the REALIZED shortfall of the single global placement floor
            # against the per-scene ISO 3382-1 §5.3 minimum, so the E1 report
            # discloses it as measured rather than asserting compliance. The
            # declared floor lives in the config; this is what it bought.
            "below_iso_min_distance": _flag_counts(
                room_stats,
                ("below_iso_min_distance_sabine", "below_iso_min_distance_eyring"),
                declared_distance_min_m=(
                    None if regime.distance_range is None else regime.distance_range[0]
                ),
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
