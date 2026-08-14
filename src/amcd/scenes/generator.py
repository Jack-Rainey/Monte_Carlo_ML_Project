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

The stage's second canonical output is `placement_report.json`: per-split
placement accounting plus three validity blocks — diffuse-field, record-length
and ISO 3382-1 §5.3 distance — and it enforces the record-length gate, which can
abort the run. A geometry family declaring `characterization: none` is not an
enclosure, so it is excluded from all three blocks and carries a reason instead
of a number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..acoustics import (
    box_volume_and_surface,
    clip_absorption,
    critical_distance,
    min_measurement_distance,
    diffuse_field_drr_db,
    eyring_rt60,
    sabine_rt60,
)
from ..config import Config, Margins, PlacementRegime
from ..runtime import RunContext, emit
from ..simulators.base import (
    SceneSpec,
    simulator_max_eval_freq_hz,
    simulator_min_separation,
    simulator_realized_absorption,
    simulator_realized_support_s,
)


#: Why excluding a non-enclosure from the record-length fractions is not the same
#: as it being fine. Emitted twice — per scene in `uncharacterized_reason` and per
#: split in `record_decay_range`'s `uncharacterized_note` — so it is declared once
#: here rather than drifting between the two.
_RECORD_LENGTH_UNCHECKED = (
    "its record-length adequacy is therefore UNCHECKED, not merely excluded: it is "
    "still rendered into a record of fixed ir_duration, and a non-enclosure has a "
    "finite decay too — it merely is not Sabine's, so no closed form here can "
    "compare it against the record"
)


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
    on the room family, so a single acceptance rate conflates them.

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
        # Horizontal plane only. A declared height range is an ergonomic band,
        # not a room boundary — 1.2 m is 1.2 m off the floor either way — so
        # biasing z buys no boundary proximity while narrowing a reported split.
        rcv_hi = [lo[i] + regime.corner_frac * (hi[i] - lo[i]) for i in (0, 1)] + [hi[2]]
    else:  # interior: uniform anywhere in the admissible box
        rcv_hi = hi

    # Greatest separation this box admits at all: which side of `distance_range`
    # does the rejecting depends on the family, and the accounting has to say.
    max_reachable = float(np.linalg.norm(np.subtract(hi, lo)))
    stats = {"attempts": 0, "below_min": 0, "above_max": 0,
             "max_reachable_m": max_reachable}

    for attempt in range(1, max_attempts + 1):
        stats["attempts"] = attempt
        src = tuple(float(v) for v in rng.uniform(lo, hi))
        rcv = tuple(float(v) for v in rng.uniform(lo, rcv_hi))
        if regime.distance_range is None:
            return src, rcv, stats
        # Either bound may be null — a backend imposes a minimum with no matching
        # maximum — so each side is tested only when it is declared.
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


def _check_eval_bands_against_the_backend(config: Config) -> None:
    """Refuse an eval band the active backend cannot render faithfully.

    A backend may realize a physical effect wrongly in a way it cannot fix, and
    declare the band above which that error exceeds a stated tolerance — gsound
    attenuates air absorption at a quarter of the ISO value, compiled in and not
    exposed for pre-compensation, which is inert at 500/1000 Hz and reaches ~19 %
    of T60 at 8 kHz.

    Checked here rather than in `Config` because `config.py` is in `_CORE_SOURCES`,
    so importing the simulator seam from it would put `simulators/base.py` into
    every stage's cache key. gen-scenes is the first stage, so failing here costs
    nothing. A backend declaring no ceiling is simply unconstrained.
    """
    limit = simulator_max_eval_freq_hz(config)
    if limit is None:
        return
    over = [f for f in config.iso_eval_freqs if float(f) > limit]
    if over:
        raise ValueError(
            f"iso_eval_freqs {over} exceed the {limit:g} Hz ceiling the "
            f"{config.simulator.name!r} backend declares it can render faithfully "
            f"(air_absorption_max_eval_freq_hz). Above it the backend's realized air "
            f"absorption puts T60 outside its own declared tolerance, so a metric "
            f"reported from those bands would describe the renderer's error rather "
            f"than the room. Raise the ceiling only with a measurement that "
            f"justifies it, or drop the bands."
        )


def _check_regimes_clear_backend_floor(config: Config) -> float:
    """Reject any placement regime that could emit a scene the backend cannot render.

    The declared-config check, run before a single scene exists; the realized-scene
    backstop is `render._preflight_separations`. EVERY declared regime is checked,
    not just the one the id baseline names — a regime unused today is a trap for
    the config that selects it tomorrow.

    The backend floor is a LOWER LIMIT on the researcher's choice, never its
    source, so the message says to raise the config value and the scientifically
    motivated minimum stays in the config where it belongs.
    """
    floor = simulator_min_separation(config)

    # A DECLARED minimum is required unconditionally, including from a backend
    # declaring zero, which the `Simulator` contract allows. The researcher's
    # minimum is a scientific choice about the scene distribution; the backend
    # floor is only a lower limit on it. Hence two checks, in this order.
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
    realized_support_s,
    realized_absorption,
    iso_t30_decay_range_db: float,
    characterization: str,
) -> dict:
    """Closed-form acoustic descriptors of one scene, from geometry alone.

    `dims` and `distance` in metres; `absorption` is the nominal Sabine α
    (dimensionless) — see the caveat at the α clip below. Every returned key
    carries its own unit in its name.

    These are what characterizes a split, not source-receiver distance: DRR
    depends on d/r_c, and absorption moves r_c independently of d, so two splits
    can share a median distance and differ by ~11 dB in median DRR.

    Estimates, not measurements: diffuse-field formulae over a shoebox, reported
    so the E1 write-up can characterize the dataset it generated. The rendered
    IRs remain the source of truth for every reported metric.

    VALIDITY. A high-absorption split falls outside the diffuse-field premise for
    most or all of its scenes, so each scene carries validity flags beside the
    estimates. No formula changes and nothing is dropped: a DRR from a formula
    outside its domain is still reported, but reported AS an extrapolation.

    NOT EVERY GEOMETRY IS AN ENCLOSURE. Every quantity here — Sabine/Eyring T60,
    room constant, critical distance, DRR — assumes a closed box. That holds for
    shoebox and corridor and fails for the roadmap's outdoor and partially-open
    scenes (paper §6), so the geometry family DECLARES its `characterization` and
    a family declaring "none" gets a recorded reason instead of meaningless
    closed-box values in the canonical placement_report.json.
    """
    if characterization == "none":
        # UNCHARACTERIZED, not zero and not NaN: every numeric key is absent, so
        # no consumer can average it. The reason below is not serialized per scene
        # — what reaches the artifact is the per-split `uncharacterized_note` and
        # the `n_uncharacterized` count.
        return {
            "characterization": "none",
            "uncharacterized_reason": (
                "geometry family declares characterization: none — it is not a "
                "closed enclosure, so Sabine/Eyring T60, room constant, critical "
                f"distance and diffuse-field DRR are undefined for it, and "
                f"{_RECORD_LENGTH_UNCHECKED}"
            ),
            # `decay_range_below_iso_t30` is OMITTED, not False. A False reads as
            # "measured, and within the record", which would put the scene into the
            # record-length gate's denominator as passing — so a dataset whose
            # enclosed scenes breach the limit could pass by adding non-enclosures.
            # Omission is what `_scene_is_characterized` reads to exclude it.
        }
    if characterization != "sabine":
        raise ValueError(
            f"unknown geometry characterization {characterization!r}; expected "
            f"'sabine' or 'none'."
        )
    if not distance > 0.0:
        # DRR divides by d²; a coincident pair would report +inf into a canonical
        # artifact. Guarded rather than clamped, so it cannot be reported as if it
        # were a real scene.
        raise ValueError(
            f"source-receiver distance must be > 0 to characterize a scene; got "
            f"{distance}. Declare a placement `distance_range` with a positive "
            f"lower bound."
        )
    volume, surface = box_volume_and_surface(dims)
    # NOMINAL α, as configured, governing EVERY quantity below — t60_*,
    # critical_distance_m, drr_db and the record-length flag all scale with it.
    #
    # WHETHER THAT IS ALSO THE RENDERED ROOM IS THE BACKEND'S ANSWER. This module
    # names no backend, so it asks through the simulator seam and records the
    # reply as `absorption_realized`, with `absorption_is_as_declared` beside it.
    # A reader checking a d_min or a DRR against the wrong one of the two is
    # checking a room that was never rendered.
    alpha, _alpha_clipped = clip_absorption(absorption)
    alpha_realized = realized_absorption(alpha)

    # From `amcd.acoustics`, which the scaffold also renders from, so the
    # described room and the rendered room cannot drift apart.
    t60_sabine = sabine_rt60(volume, surface, alpha)
    # Eyring is the better estimate at high absorption, where Sabine overpredicts
    # — and ceiling_absorptive reaches α = 0.98.
    t60_eyring = eyring_rt60(volume, surface, alpha)
    # How much record the active backend will produce for THIS scene's decay, and
    # the decay range that record therefore holds. Sabine, not Eyring: the longer
    # T60 gives the smaller decay range, which errs toward refusing.
    support_s = float(realized_support_s(t60_sabine, volume, surface))
    available_decay_db = 60.0 * support_s / t60_sabine
    r_c = critical_distance(surface, alpha)

    # ISO 3382-1 §5.3 minimum measurement distance. Reported and counted rather
    # than enforced: the criterion is PER SCENE, varying with each scene's own
    # absorption and surface, while the config declares ONE global placement
    # floor, so no floor can satisfy it everywhere.
    #
    # ABSORPTION-AREA CONVENTION — the two reverberant-field radii in this record
    # descend from DIFFERENT conventions, and the record has to say so:
    #   * `critical_distance_m` uses the Hopkins–Stryker room constant R = Sα/(1−α);
    #   * `iso_min_distance_*_m` descends from ISO's Sabine absorption area A = Sα.
    # Against ISO's OWN radius sqrt(Sα/16π) the ratio is 1.907 at every α. Against
    # the radius THIS record publishes it is 1.907·sqrt(1−α), which falls below 1
    # at α ≈ 0.725 — so `receiver_inside_critical_distance` and
    # `below_iso_min_distance_sabine` swap which is stricter there. Neither number
    # is wrong under its own definition, which is why both definitions are stated.
    d_min_sabine = min_measurement_distance(alpha, surface, "sabine")
    d_min_eyring = min_measurement_distance(alpha, surface, "eyring")

    return {
        "characterization": "sabine",
        "volume_m3": volume,
        "surface_m2": surface,
        "absorption": alpha,
        # What the ACTIVE BACKEND builds from the nominal α above. Where the two
        # differ, every closed form in this record describes the declared room and
        # this is the one that was rendered.
        "absorption_realized": float(alpha_realized),
        "absorption_is_as_declared": bool(
            abs(alpha_realized - alpha) <= 1e-12 * max(1.0, alpha)
        ),
        "t60_sabine_s": float(t60_sabine),
        "t60_eyring_s": float(t60_eyring),
        # Hopkins–Stryker: r_c = sqrt(R/16π) with R = Sα/(1−α). See the
        # absorption-area note above for why this and `iso_min_distance_*_m` are
        # not on one convention.
        "critical_distance_m": r_c,
        "d_over_rc": distance / r_c if r_c > 0 else float("inf"),
        # ISO's Sabine absorption area A = Sα — NOT the room constant above.
        "iso_min_distance_sabine_m": d_min_sabine,
        "iso_min_distance_eyring_m": d_min_eyring,
        # Diffuse-field DRR: direct 1/(4πd²) against the reverberant field 4/R.
        "drr_db": diffuse_field_drr_db(surface, alpha, distance),
        # ── Validity indicators ──────────────────────────────────────────────
        # Sabine and Eyring agree only for small α; the ratio is a direct,
        # assumption-free readout of how far the diffuse-field model is being
        # stretched.
        "sabine_eyring_ratio": float(t60_sabine / t60_eyring),
        "alpha_above_diffuse_limit": bool(alpha > alpha_limit),
        # r_c larger than the room's longest dimension: MOST of the room is inside
        # the direct field. (Not "no reverberant field at all" — that needs r_c above
        # the diagonal, up to sqrt(3)× larger. The exact per-scene condition is the
        # next flag.)
        "rc_exceeds_max_dim": bool(r_c > max(dims)),
        # The sharpest per-scene condition: inside r_c the receiver sits in the
        # DIRECT field, so the diffuse-field DRR being reported has no reverberant
        # field to divide by.
        "receiver_inside_critical_distance": bool(distance < r_c),
        # This scene's own ISO 3382-1 §5.3 floor against the separation actually
        # drawn. Both estimates are carried because they disagree substantially at
        # high α, and the reader needs to see by how much.
        "below_iso_min_distance_sabine": bool(distance < d_min_sabine),
        "below_iso_min_distance_eyring": bool(distance < d_min_eyring),
        # The design-time record-length flag, founded on the BACKEND'S OWN
        # REALIZED SUPPORT rather than on `ir_duration`. `ir_duration` is the
        # window the pipeline ALLOCATES, not what a backend fills, and on gsound
        # the two differ enormously: the adaptive energy trim closes the record as
        # a sub-linear power of T60, so the captured fraction collapses as rooms
        # get more reverberant.
        #
        # The criterion is DECAY RANGE, not seconds, because that is what ISO
        # 3382-1 states: a record holds 60·support/T60 dB of decay, and T30 needs
        # 45 dB of it.
        #
        # STILL NOT A MEASUREMENT — Sabine from geometry × a scalar α, erring long
        # so the flag declares a scene unsupported rather than silently truncating
        # it. Sabine assumes a 4V/S mean free path and under-predicts decay in a
        # disproportionate enclosure, such as the corridor family.
        "record_decay_range_db": available_decay_db,
        "record_support_s": support_s,
        "decay_range_below_iso_t30": bool(available_decay_db < iso_t30_decay_range_db),
    }


#: Top-level keys of `placement_report.json` that are NOT a split record, and are
#: therefore skipped by `_disclose_and_gate_record_length` rather than scored.
#:
#: Empty today: every top-level key is currently a split. It is a DECLARED set
#: rather than an assumption because the roadmap keeps pushing metadata into this
#: artifact, and a gate that hardcoded "every top-level key is a split" would
#: foreclose that. Adding a metadata key means adding it here, in the same commit.
#:
#: The set is checked DISJOINT from the run's generation-plan regimes before it is
#: applied. Skipping is silent by construction, so an entry here that names a real
#: split would drop that split out of the warning loop AND out of the gate's
#: `over`/`total` sums with no error — the very "a split goes missing from the gate
#: unnoticed" failure this constant exists to prevent, in the other direction.
_NON_SPLIT_REPORT_KEYS: frozenset[str] = frozenset()


def _regime_label(config: Config, name: str) -> str:
    """How to name one `placement_report.json` key in an operator-facing message.

    The report is keyed by generation-plan regime, and in frac mode the `id` key is
    a POOL three declared splits wide, so calling it a "split" in a warning
    invites the reader to hear `train`.
    """
    if not config.id_pool_is_counted and name == "id":
        return f"regime 'id' (pools {'/'.join(config.id_pool_splits)})"
    return f"split {name!r}"


def _disclose_declared_support_corner(config: Config, verbosity) -> dict:
    """Print the declared-support T60 corner; return it for the gate's error text.

    Disclosure only — never a threshold. The corner is the product of two
    independent extremes (largest room, lowest absorption) and has near-zero
    probability of being drawn, so gating on it would reject configs whose
    realized scenes are all fine.

    `Config.worst_case_t60` returns a reasoned `None` when no family declares
    `characterization: sabine`, which must be formatted as such rather than
    assumed numeric.
    """
    corner = config.worst_case_t60()
    if corner["t60_sabine_s"] is None:
        emit(
            verbosity, "progress",
            f"  Declared-support corner: UNSCORED — {corner['uncharacterized_reason']} "
            f"(families skipped: {', '.join(corner['skipped_families'])})",
        )
    else:
        emit(
            verbosity, "progress",
            f"  Declared-support corner: Sabine T60 {corner['t60_sabine_s']:.2f} s "
            f"({corner['geometry_family']} {corner['dims_m']} m at alpha "
            f"{corner['absorption']}) vs ir_duration {corner['ir_duration_s']:.2f} s"
            + ("" if corner["covered_by_record"] else "  — NOT covered by the record"),
        )
    return corner


def _warn_regimes_over_limit(
    config: Config,
    per_split: dict[str, tuple[int, int, int]],
    limit: float,
    verbosity,
) -> None:
    """Name every regime whose OWN over-limit fraction exceeds the declared limit.

    `per_split` maps regime -> (over-limit count, scored, attempted). Three states
    are distinguished rather than collapsed, because they are different facts: a
    regime that generated nothing, one whose scenes are all uncharacterized so its
    fraction is UNDEFINED, and one genuinely over its limit.

    Warnings, not gates — the gate is the overall fraction — and emitted BEFORE it
    can raise, so a failing run still names the regimes responsible.
    """
    for name, (count, scored, attempted) in per_split.items():
        label = _regime_label(config, name)
        if not attempted:
            emit(verbosity, "warning",
                 f"  WARNING: {label} generated 0 scenes, so the record-length gate "
                 f"has nothing to score for it. A declared split with no scenes is a "
                 f"fact about this run — reported, not skipped.")
            continue
        if not scored:
            emit(verbosity, "warning",
                 f"  WARNING: {label}: 0 of {attempted} scenes are "
                 f"characterized, so its over-limit fraction is UNDEFINED — "
                 f"reported as null, never as 0.0.")
            continue
        frac = count / scored
        if frac > limit:
            emit(verbosity, "warning",
                 f"  WARNING: {label}: {count}/{scored} scenes ({frac:.3%}) "
                 f"exceed ir_duration {config.ir_duration} s — above this config's "
                 f"own scenes.max_frac_below_iso_t30_decay_range ({limit}). The gate is "
                 f"the OVERALL fraction and may still pass; a shift split far over on "
                 f"its own is a fact about that split's decay distribution.")


def _disclose_and_gate_record_length(config: Config, report: dict, verbosity) -> None:
    """Disclose the declared-support corner; gate on the over-limit rate of the
    closed-form ESTIMATE.

    Not on a measurement: `decay_range_below_iso_t30` is Sabine from geometry, and
    nothing in gen-scenes compares a RENDERED decay against `ir_duration`. The
    instrument's limitations are named at the flag's own definition in
    `_room_acoustics`. It is still worth having as the only check that runs BEFORE
    a batch is rendered.

    The two halves differ in kind. The CORNER (`Config.worst_case_t60`) is the
    product of two independent extremes and has near-zero probability of being
    drawn, so it is printed and stamped, never used as a threshold. The GATE is
    `scenes.max_frac_below_iso_t30_decay_range` applied to the scenes that actually
    exist, which is the population the metrics are computed over.

    The gate is the OVERALL fraction and the disclosure is per split: a per-split
    gate would let the smallest split set the tolerance for every other one. Every
    split over the limit is warned about unconditionally, whether or not the
    overall gate trips, because the per-shift breakdown IS the research result.

    `report` is keyed by generation-plan REGIME, not declared split — in frac mode
    `train`/`valid`/`test_id` are pooled as one `id` entry and separated later by
    `data/splits.py`; in count mode each is its own entry. The messages below say
    which they mean.

    Scores only CHARACTERIZED scenes (rule at `_scene_is_characterized`); a gate
    that scored none is UNSCORED, never passed.
    """
    corner = _disclose_declared_support_corner(config, verbosity)
    limit = config.scenes.max_frac_below_iso_t30_decay_range

    # The over-declared direction: skipping is silent by construction, so a real
    # regime named in `_NON_SPLIT_REPORT_KEYS` would leave both the warning loop
    # and the gate sums with no error at all.
    regimes = {entry[0] for entry in _generation_plan(config)}
    masked = regimes & _NON_SPLIT_REPORT_KEYS
    if masked:
        raise ValueError(
            f"_NON_SPLIT_REPORT_KEYS names {sorted(masked)}, which this config "
            f"generates as scene regimes. Those scenes would leave the record-length "
            f"gate's numerator AND denominator with nothing reported — the silent "
            f"drop the constant exists to prevent. It may only name report metadata."
        )

    # (over-limit count, scenes SCORED, scenes ATTEMPTED) per regime.
    per_split = {}
    for name, entry in report.items():
        if name in _NON_SPLIT_REPORT_KEYS:
            continue
        # Diagnosed, not a bare KeyError from the next line.
        if not isinstance(entry, dict) or "record_decay_range" not in entry:
            raise ValueError(
                f"placement_report key {name!r} is neither a split record nor a "
                f"declared non-split key: the record-length gate scores every "
                f"top-level key of the report, and this one carries no "
                f"'record_decay_range' block. If it is metadata rather than a "
                f"split, add it to `_NON_SPLIT_REPORT_KEYS` in the same commit that "
                f"writes it — a report key nothing scores and nothing declares is "
                f"how a split would go missing from the gate unnoticed."
            )
        block = entry["record_decay_range"]
        attempted = entry["n_scenes"]
        # `_flag_counts` PUBLISHES the scored count as the block's own `n_scenes`.
        # Read it, and cross-check against the derived count rather than deriving
        # alone — a change to `_flag_counts` must break the gate loudly.
        derived = attempted - block.get("n_uncharacterized", 0)
        scored = block.get("n_scenes", derived)
        if scored != derived:
            raise ValueError(
                f"split {name!r}: the record-length block publishes n_scenes="
                f"{scored} but its own counts imply {derived} "
                f"(n_scenes {attempted} − n_uncharacterized "
                f"{block.get('n_uncharacterized', 0)}). Two expressions for the "
                f"scored denominator have diverged; the gate refuses "
                f"to pick one."
            )
        per_split[name] = (block["decay_range_below_iso_t30"]["count"], scored, attempted)

    _warn_regimes_over_limit(config, per_split, limit, verbosity)

    over = sum(count for count, _, _ in per_split.values())
    total = sum(scored for _, scored, _ in per_split.values())
    attempted_total = sum(attempted for _, _, attempted in per_split.values())
    if not total:
        # Falling through here would be a silent pass over a gate that measured
        # nothing.
        if attempted_total:
            # The cause is what this run GENERATED, not what the config declares: a
            # config can declare a `sabine` family and still generate no scene from
            # it, because the regime axes select the family. Saying "every geometry
            # family declares none" sends the operator to `geometry_families` when
            # the lever is `id_regime` / a split's `axes`.
            unused = sorted(
                fam for fam, spec in config.scenes.geometry_families.items()
                if spec.characterization == "sabine"
            )
            lever = (
                f" Families declaring characterization: sabine exist but no scene "
                f"was generated from one ({', '.join(unused)}) — the lever is "
                f"scenes.id_regime / a split's axes, not geometry_families."
                if unused else
                " No geometry family in this config declares characterization: sabine."
            )
            emit(verbosity, "warning",
                 f"  WARNING: the record-length gate scored 0 of {attempted_total} "
                 f"scenes — no scene in this run came from a family declaring "
                 f"characterization: sabine, so no closed-form T60 exists to compare "
                 f"against ir_duration {config.ir_duration} s.{lever} The gate is "
                 f"UNSCORED, not passed.")
        return

    if total != attempted_total:
        # The gate discloses its own coverage on a PASS too, not only inside the
        # failure message. The silent path needs a MIXED characterized/uncharacterized
        # split — unreachable from any shipped config today, which is exactly why it
        # would go unnoticed when the roadmap's outdoor families land.
        emit(verbosity, "warning",
             f"  WARNING: the record-length gate scored {total} of "
             f"{attempted_total} scenes; {attempted_total - total} are excluded as "
             f"uncharacterized and their record-length adequacy is UNCHECKED. "
             f"The verdict below covers the scored scenes only.")
    if (over / total) > limit:
        lines = "\n".join(
            f"    {name}: {count}/{scored} scenes"
            for name, (count, scored, _) in per_split.items() if count
        )
        excluded = attempted_total - total
        exclusion = (
            f"\n{excluded} of {attempted_total} scenes are excluded from this "
            f"fraction as uncharacterized." if excluded else ""
        )
        raise ValueError(
            f"{over} of {total} scenes ({over / total:.3%}) get less than "
            f"{config.scenes.iso_t30_decay_range_db:g} dB of decay range from the "
            f"record the {config.simulator.name!r} backend will actually produce — "
            f"more than scenes.max_frac_below_iso_t30_decay_range ({limit}) allows:\n"
            f"{lines}{exclusion}\n"
            f"This is measured against the backend's DECLARED REALIZED SUPPORT, not "
            f"against ir_duration ({config.ir_duration} s), which is only the window "
            f"allocated. A backend whose record is shorter than the "
            f"window will fail here while appearing to 'fit'.\n"
            f"A T30/EDT fitted over a truncated record measures the truncation, not "
            f"the room. Narrow the geometry/absorption ranges, "
            f"or raise the declared tolerance and say why. For reference the declared "
            f"support reaches Sabine T60 {corner['t60_sabine_s']:.2f} s "
            f"({corner['geometry_family']} {corner['dims_m']} m at alpha "
            f"{corner['absorption']})."
        )


def _scene_is_characterized(room: dict, flags: tuple[str, ...]) -> bool:
    """True for a `characterization: sabine` scene, False for a `none` one:
    a non-enclosure carries a reason instead of the closed-form quantities `flags`
    describe, so it can be counted neither for nor against them."""
    return all(flag in room for flag in flags)


def _flag_counts(
    room_stats: list[dict],
    flags: tuple[str, ...],
    *,
    uncharacterized_consequence: str,
    **context,
) -> dict:
    """Count and fraction for each named per-scene boolean, plus its context.

    Reported as counts rather than a bare boolean so a reader sees how much of a
    split is affected: "all of a split is outside the diffuse-field domain" and
    "one scene is" are very different disclosures, and the flag alone cannot tell
    them apart.

    `uncharacterized_consequence` is REQUIRED, not defaulted: what the
    exclusion COSTS differs per block, so a default would hand a new block a
    sentence that does not describe it — the silent exclusion this project forbids.
    Keyword-only and outside `**context`, so it shapes the note rather than landing
    in the artifact as a context key of its own.
    """
    # Uncharacterized scenes leave BOTH numerator and denominator — see
    # `_scene_is_characterized` — and the exclusion is itself reported, because a
    # fraction whose denominator silently shrank is the drop this project forbids.
    # `n_uncharacterized` is emitted ONLY when nonzero; that is the contract
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
            f"does not apply to a non-enclosure. {uncharacterized_consequence}"
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


def run_gen_scenes(config: Config, run_dir: Path, ctx: RunContext) -> None:
    """Generate the config's scene specs; write them and `placement_report.json`.

    Clears stale `scene_*.json` first, then writes one spec per scene plus the
    canonical per-split report into `run_dir/scenes/`.

    Raises `ValueError` from the record-length gate if the config's realized draws
    breach `scenes.max_frac_below_iso_t30_decay_range`. No stage sentinel is written in
    that case, so render will not proceed past it.
    """
    verbosity = ctx.verbosity
    out_dir = run_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene ids are positional (`scene_0000`…), so regenerating with FEWER scenes
    # leaves high-numbered orphans that render and preprocess would glob into the
    # dataset under a different config, while placement_report.json declared the
    # smaller set.
    for stale in out_dir.glob("scene_*.json"):
        stale.unlink()

    # Config-level pre-flight: no scene is generated under a placement regime the
    # active backend could not render.
    _check_regimes_clear_backend_floor(config)
    _check_eval_bands_against_the_backend(config)

    # The record-length gate's denominator. Resolved once here and passed down as
    # a plain callable, so `_room_acoustics` never touches the registry and this
    # module names no backend. The declaration is validated on the first call,
    # before any scene is written.
    def support_of(t60_s: float, volume_m3: float, surface_m2: float) -> float:
        return simulator_realized_support_s(
            config, t60_s, volume_m3, surface_m2, config.ir_duration)

    # The other backend fact this module records without naming a backend: what
    # the renderer's room actually absorbs, given a nominal alpha.
    def realized_alpha_of(alpha_nominal: float) -> float:
        return simulator_realized_absorption(config, alpha_nominal)

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
        # (scene, reason) for every scene this stage could not characterize,
        # mirroring the eval stage's drops.csv. `room_stats` reaches the report
        # only through `_summarize` (numeric) and `_flag_counts` (boolean), so
        # without this the reason would survive only as an aggregate count.
        uncharacterized: list[dict[str, str]] = []

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
            scene_id = f"scene_{scene_idx:04d}"
            room = _room_acoustics(
                dims, absorption, distance,
                alpha_limit=scenes_cfg.diffuse_field_alpha_limit,
                realized_support_s=support_of,
                realized_absorption=realized_alpha_of,
                iso_t30_decay_range_db=config.scenes.iso_t30_decay_range_db,
                characterization=(
                    scenes_cfg.geometry_families[axes["geometry"]].characterization
                ),
            )
            room_stats.append(room)
            if "uncharacterized_reason" in room:
                uncharacterized.append(
                    {"scene": scene_id, "reason": room["uncharacterized_reason"]}
                )

            spec = SceneSpec(
                scene_id=scene_id,
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
            # The per-scene (unit, reason) pairs, emitted only when non-empty so
            # a fully characterized split carries no empty list. The aggregate is
            # each flag block's `n_uncharacterized`; this is the record under it.
            **({"uncharacterized": uncharacterized} if uncharacterized else {}),
            "placement_regime": axes["placement"],
            "height_range_declared": regime.height_range,
            "distance_range_declared": regime.distance_range,
            "placement_attempts": attempts_total,
            "acceptance_rate": (count / attempts_total) if attempts_total else None,
            # Split by SIDE: rejecting below the minimum strips close, high-DRR
            # pairs; rejecting above the maximum strips distant, low-DRR ones.
            # Shoeboxes reject mostly below and corridors mostly above, so one
            # pooled rate hides which tail of the DRR distribution was cut.
            "rejected_below_min": rejected["below_min"],
            "rejected_above_max": rejected["above_max"],
            "rooms_that_cannot_reach_max_distance": unreachable_max,
            "source_receiver_distance_m": _summarize(distances) if distances else None,
            # Source and receiver kept SEPARATE: a placement regime may bias one
            # and not the other, and pooling would hide it.
            "source_height_m": _summarize(src_heights) if src_heights else None,
            "receiver_height_m": _summarize(rcv_heights) if rcv_heights else None,
            # The DRR-relevant descriptors — see `_room_acoustics`. Guarded like
            # the siblings above: `_summarize` reduces over the list, so an empty
            # split would raise numpy's bare zero-size-reduction error.
            **{
                f"{key}": (
                    _summarize([r[key] for r in room_stats if key in r])
                    if any(key in r for r in room_stats) else None
                )
                for key in ("volume_m3", "absorption", "absorption_realized",
                            "t60_sabine_s", "t60_eyring_s",
                            "critical_distance_m", "d_over_rc", "drr_db",
                            "sabine_eyring_ratio", "iso_min_distance_sabine_m",
                            "iso_min_distance_eyring_m")
            },
            # WHICH ROOM THE BLOCK ABOVE DESCRIBES. Every quantity in it — T60,
            # r_c, DRR, both d_min variants — is a closed form in the NOMINAL
            # absorption, and whether the renderer builds that room is the
            # backend's convention. Counted rather than asserted, so a split partly
            # rendered under a different absorption cannot read as fully faithful.
            "absorption_as_declared": _flag_counts(
                room_stats, ("absorption_is_as_declared",),
                uncharacterized_consequence=(
                    "A non-enclosure carries no closed-form estimate for the "
                    "convention to invalidate, so nothing is lost here — but the "
                    "backend still renders it under whatever convention it declares."
                ),
            ),
            # Validity of the estimates directly above, and of the record length
            # against them. Counts rather than a flag, so the reader sees HOW MUCH
            # of a split is outside the diffuse-field domain.
            "diffuse_field_validity": _flag_counts(
                room_stats,
                ("alpha_above_diffuse_limit", "rc_exceeds_max_dim",
                 "receiver_inside_critical_distance"),
                uncharacterized_consequence=(
                    "These flags ask how far the CLOSED-BOX diffuse-field model is "
                    "being stretched, and it was never applied to these scenes. For "
                    "a free-field family nothing is lost; for a partially-open one "
                    "a quasi-diffuse field does exist and how far it departs from "
                    "the model is unmeasured here, not absent."
                ),
                alpha_limit=scenes_cfg.diffuse_field_alpha_limit,
            ),
            "record_decay_range": _flag_counts(
                room_stats, ("decay_range_below_iso_t30",),
                uncharacterized_consequence=(
                    f"For each of them {_RECORD_LENGTH_UNCHECKED}. The "
                    f"n_uncharacterized count above is the number of scenes nothing "
                    f"checked."
                ),
                iso_t30_decay_range_db=config.scenes.iso_t30_decay_range_db,
            ),
            # The REALIZED shortfall of the single global placement floor against
            # the per-scene ISO 3382-1 §5.3 minimum, so the E1 report can disclose
            # it as measured rather than assert compliance.
            "below_iso_min_distance": _flag_counts(
                room_stats,
                ("below_iso_min_distance_sabine", "below_iso_min_distance_eyring"),
                uncharacterized_consequence=(
                    "ISO 3382-1 §5.3's d_min is a functional of an enclosure's "
                    "diffuse-field T60, so the CLOSED-BOX formula has no V/(cT) to "
                    "evaluate here. For a free-field family the criterion is vacuous "
                    "— it exists to place the receiver in a reverberant field there "
                    "is none of. For a partially-open one the substantive criterion "
                    "still applies and is simply unmeasured by this block."
                ),
                declared_distance_min_m=(
                    None if regime.distance_range is None else regime.distance_range[0]
                ),
            ),
        }

    # Canonical, not verbosity-gated. Two jobs: the rejection-sampling accounting
    # (how many draws were discarded to satisfy distance_range), and the realized
    # per-split source-receiver distance distribution — which is what lets the E1
    # report state exactly what stood in for Research I's unspecified mid_pair /
    # far_pair sub-ranges.
    #
    # Every top-level key here is a SPLIT RECORD. A metadata key must be declared
    # in `_NON_SPLIT_REPORT_KEYS` in the same commit that writes it, or the gate
    # below refuses the report.
    (out_dir / "placement_report.json").write_text(json.dumps(report, indent=2))

    _disclose_and_gate_record_length(config, report, verbosity)

    n_shift = sum(sp.count for sp in config.shift_splits.values())
    emit(
        verbosity, "progress",
        f"  Generated {scene_idx} scene specs "
        f"({scene_idx - n_shift} id + {n_shift} shift) → {out_dir}",
    )
