"""
Typed, composable configuration — the single source of truth for a run.

Config is loaded by merging YAML layers (base + experiment overrides) and then
resolving three parameter *roles* (design_spec §7) into one concrete run:

  - fixed   — a scalar, used as-is.
  - tuned   — `{tune: {space, scale}, value: <current op-point>}`; the search
              engine (E3) selects one value on the validation split. Until then
              the declared `value` is the concrete operating point.
  - swept   — `{sweep: [v0, v1, ...]}`; a research axis. Every value is a sibling
              run reported on the held-out test splits. `expand_sweeps()` produces
              the sibling concrete configs; a single run resolves to one selected
              index (default 0).

Resolution happens once at load time: any tune/sweep leaf is collapsed to a
concrete scalar and its role metadata is recorded separately (for stamping and,
later, the search engine). Every stage therefore reads plain scalars — the role
grammar never leaks past `Config.load`.

No value has a behavioral default in this module: everything comes from a YAML
layer or raises. `configs/base.yaml` holds the defaults.
"""
from __future__ import annotations

import datetime
import importlib.metadata
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from numpy.random import SeedSequence
from pydantic import BaseModel, PrivateAttr, model_validator

from . import provenance
from .acoustics import sabine_rt60


#: Where `configs/` may live, most-preferred first.
#:
#: `configs/` used to be resolved three levels up from this module and nowhere
#: else, which silently assumed a source checkout — so a wheel installed into
#: site-packages could not find `base.yaml` and failed with a bare
#: `FileNotFoundError` deep inside `_merge_yaml`, naming a path three parents above
#: a site-packages directory (F-73). That is the same class of defect as a
#: platform-keyed branch: a host-layout assumption baked into package code.
#:
#: The packaged location is listed FIRST and is not a source checkout's layout, so
#: shipping `configs/` as package data later needs no change here.
_CONFIG_ROOT_CANDIDATES = (
    Path(__file__).parent / "configs",            # packaged alongside the package
    Path(__file__).parent.parent.parent / "configs",  # source checkout: repo/configs
)


def _resolve_configs_dir() -> Path:
    """The first candidate that actually holds `base.yaml`.

    Falls back to the source-checkout candidate when none does, so importing
    `amcd.config` never fails on layout alone and `_CONFIGS_DIR` is always a real
    Path. A layout that resolves to nothing is reported by `_require_configs`, at
    the point a caller asks to LOAD a config — with a message naming every
    location tried, rather than a bare FileNotFoundError from `_merge_yaml`.
    """
    for candidate in _CONFIG_ROOT_CANDIDATES:
        if (candidate / "base.yaml").is_file():
            return candidate
    return _CONFIG_ROOT_CANDIDATES[-1]


def _require_configs() -> None:
    """Raise an actionable error if `configs/base.yaml` is not where we look.

    Called before any load. Every value that governs a run comes from a config
    layer, so there is no default to degrade to — the only useful response is to
    say where we looked and what would fix it.
    """
    if not _BASE_YAML.is_file():
        tried = "\n".join(f"    {c / 'base.yaml'}" for c in _CONFIG_ROOT_CANDIDATES)
        raise FileNotFoundError(
            "amcd cannot find `configs/base.yaml`, which holds every default a run "
            f"is built from.\n  Tried:\n{tried}\n"
            "  Run from a source checkout (where `configs/` sits beside `src/`), or "
            "install a build that ships `configs/` as package data. There is no "
            "built-in fallback: a value that governs an experiment never has a "
            "default in Python."
        )
    # The plugin params directories too, not just base.yaml (F-80). A root holding
    # base.yaml but none of these — precisely the half-finished "ship configs/ as
    # package data" the message above recommends — used to win the candidate race
    # and then load a VALIDATED Config with `representation.params={}` and
    # `simulator.params={}`, because `_attach_params_block` cannot distinguish a
    # missing DIRECTORY from "this registered plugin needs no params file". The run
    # died later on a pydantic missing-field error naming neither, which is the
    # confusion F-73 exists to end.
    missing = [d.name for d in (_MODELS_DIR, _REPS_DIR, _SIMS_DIR) if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"amcd found `base.yaml` in {_CONFIGS_DIR} but not the plugin parameter "
            f"director{'y' if len(missing) == 1 else 'ies'} {missing}. Each plugin's "
            f"concrete parameters live beside it there, and a missing directory is "
            f"indistinguishable from a parameter-free plugin, so the run would load "
            f"with empty params and fail later without naming this. Install a build "
            f"that ships the whole `configs/` tree, or run from a source checkout."
        )


_CONFIGS_DIR = _resolve_configs_dir()
_BASE_YAML = _CONFIGS_DIR / "base.yaml"
_MODELS_DIR = _CONFIGS_DIR / "models"
_REPS_DIR = _CONFIGS_DIR / "representations"
_SIMS_DIR = _CONFIGS_DIR / "simulators"

# Named seeds, one per stochastic pipeline aspect. Order is stable-for-life:
# it fixes how per-aspect seeds are derived from `seeds.master`, so appending a
# new name never perturbs existing derivations. The split seed in particular is
# pinned — changing it after a run reshuffles train/test membership (leakage).
# LOCKSTEP: every name here must have a matching `int | None` field on `Seeds`
# below — `Seeds.resolved()` does `getattr(self, name) for name in SEED_NAMES`.
# Append to both, or `resolved()` raises AttributeError.
SEED_NAMES = (
    "scene_generation",
    "split_assignment",
    "weight_init",
    "data_shuffle",
    "bootstrap",
)

# Marker keys that turn a YAML leaf into a role node rather than a scalar value.
_ROLE_KEYS = ("tune", "sweep")

#: Tag the generator puts on frac-mode id-pool scenes, meaning "not yet assigned —
#: hash-bucket me". It lives in the same namespace as split names in
#: `SceneSpec.split_regime`, so a split actually NAMED "id" collides with it and
#: silently re-routes scenes (a held-out shift split named `id` was observed
#: landing 18 scenes in TRAIN). Reserved in `Config._check` rather than renamed,
#: because the tag also appears in existing on-disk scene specs.
ID_POOL_TAG = "id"

#: Names inside `preprocessed/` that are NOT split directories. A split declared
#: with one of these names would collide with that directory — `carrier` in
#: particular is exempted from the stale-split sweep, so a split named `carrier`
#: would be permanently exempted from clearing and reinstate the F-25 leak for
#: itself. Reserved for the same reason as ID_POOL_TAG (F-38).
RESERVED_SPLIT_NAMES = (ID_POOL_TAG, "carrier")

#: The roles the pipeline spine understands. `role` is what routes a split: `train`
#: and `valid` drive the trainer, `test` splits are the held-out sets that infer /
#: eval / stats / report consume. A split whose role is outside this set is
#: understood by NO stage, so it is generated, RENDERED and preprocessed and then
#: appears in no result at all — reproduced with a one-character typo: nine stages
#: `[done]`, exit 0, the split present in preprocessed/meta.json with scenes on disk
#: and absent from ci_table.csv and summary.txt (F-44).
#:
#: Declared as a tuple + explicit membership check rather than a typing.Literal, to
#: match the house pattern (RESERVED_SPLIT_NAMES above, METRIC_KINDS in
#: evaluation/metric_row.py, REQUIRED_PROVENANCE_KEYS in simulators/base.py): the
#: vocabulary is data the error message can name, and `typing.Literal` is not used
#: anywhere in this package.
SPLIT_ROLES = ("train", "valid", "test")

#: How many splits each role must have. Single-holdout validation is what the
#: pipeline implements today: one training set, one model-selection set, any number
#: of held-out test sets (design_spec §6.1 invariant #9 — test splits are never
#: pooled). Declared as data rather than inline `if`s so that k-fold or repeated
#: holdout — a plausible instantiation of the roadmap's deeper hyperparameter
#: search (research_I_paper.md §6) — relaxes `valid` to a range here instead of
#: needing the validation rewritten (RD-53).
REQUIRED_ROLE_COUNTS = {"train": 1, "valid": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Role grammar (fixed / tuned / swept)
# ─────────────────────────────────────────────────────────────────────────────

def _is_role_node(node: Any) -> bool:
    return isinstance(node, dict) and any(k in node for k in _ROLE_KEYS)


def _resolve_role_node(node: dict, path: str, selection: dict[str, int]) -> tuple[Any, dict]:
    """Collapse one tune/sweep node to a concrete scalar; return (value, role_meta)."""
    if "tune" in node and "sweep" in node:
        raise ValueError(f"{path}: a parameter cannot be both `tune` and `sweep`")

    if "tune" in node:
        _reject_unknown_role_keys(node, path, allowed={"tune", "value"})
        spec = node["tune"]
        if not isinstance(spec, dict) or "space" not in spec:
            raise ValueError(f"{path}: `tune` requires a `space` (e.g. [lo, hi])")
        if "value" not in node:
            raise ValueError(
                f"{path}: a tuned parameter needs an explicit `value` (current "
                f"operating point) so a single run has a concrete point before search"
            )
        scale = spec.get("scale", "linear")
        _check_value_in_space(node["value"], spec["space"], scale, path)
        meta = {"role": "tuned", "space": spec["space"], "scale": scale, "value": node["value"]}
        return node["value"], meta

    # sweep
    _reject_unknown_role_keys(node, path, allowed={"sweep"})
    values = node["sweep"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: `sweep` must be a non-empty list of values")
    idx = selection.get(path, 0)
    if not (0 <= idx < len(values)):
        raise ValueError(f"{path}: sweep selection index {idx} out of range")
    meta = {"role": "swept", "values": values, "selected_index": idx}
    return values[idx], meta


def _reject_unknown_role_keys(node: dict, path: str, allowed: set[str]) -> None:
    """Reject stray keys on a role node (e.g. a `value` on a `sweep`, silently
    ignored otherwise) so a malformed role node fails loudly, not silently."""
    unknown = set(node) - allowed
    if unknown:
        raise ValueError(
            f"{path}: unexpected key(s) {sorted(unknown)} on a role node; "
            f"allowed here: {sorted(allowed)}"
        )


def _check_value_in_space(value: Any, space: Any, scale: str, path: str) -> None:
    """A tuned `value` (the current operating point) must lie inside its declared
    `space`, else the single-run point sits outside the region search will explore."""
    if not (isinstance(space, (list, tuple)) and len(space) == 2):
        raise ValueError(f"{path}: `tune.space` must be [lo, hi]; got {space!r}")
    lo, hi = space
    # bool is an int subclass; a tuned `value: true` would silently coerce to 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: tuned `value` must be numeric; got {value!r}")
    if scale == "log" and (lo <= 0 or hi <= 0 or value <= 0):
        raise ValueError(f"{path}: log-scale space and value must be positive; got space={space}, value={value}")
    if not (min(lo, hi) <= value <= max(lo, hi)):
        raise ValueError(
            f"{path}: tuned value {value} lies outside its declared space [{lo}, {hi}]"
        )


def _resolve_roles(node: Any, selection: dict[str, int], path: str = "") -> tuple[Any, dict[str, dict]]:
    """
    Walk a merged-config tree, collapsing every tune/sweep leaf to a scalar.
    Returns (concrete_tree, {dotted_path: role_meta}).
    """
    roles: dict[str, dict] = {}
    if _is_role_node(node):
        value, meta = _resolve_role_node(node, path or "<root>", selection)
        roles[path] = meta
        return value, roles
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            out[key], child_roles = _resolve_roles(child, selection, child_path)
            roles.update(child_roles)
        return out, roles
    return node, roles


def _sweep_axes(node: Any, path: str = "") -> dict[str, int]:
    """Map each swept-parameter path → its number of values (for expansion)."""
    axes: dict[str, int] = {}
    if _is_role_node(node):
        if "sweep" in node:
            axes[path] = len(node["sweep"])
        return axes
    if isinstance(node, dict):
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            axes.update(_sweep_axes(child, child_path))
    return axes


# ─────────────────────────────────────────────────────────────────────────────
# Nested sub-models
# ─────────────────────────────────────────────────────────────────────────────

class Seeds(BaseModel):
    """One master seed plus optional per-aspect overrides (design_spec §10 "Invariants",
    inv #5; the per-aspect derivation itself is §7 "Config system").

    Any aspect left unset is derived independently from `master` via
    SeedSequence.spawn — single-knob reproducibility with per-aspect independence
    (so e.g. split assignment shares no entropy with bootstrap resampling)."""

    model_config = {"extra": "forbid"}

    master: int
    scene_generation: int | None = None
    split_assignment: int | None = None
    weight_init: int | None = None
    data_shuffle: int | None = None
    bootstrap: int | None = None

    def resolved(self) -> dict[str, int]:
        children = SeedSequence(self.master).spawn(len(SEED_NAMES))
        out: dict[str, int] = {}
        for name, child in zip(SEED_NAMES, children):
            explicit = getattr(self, name)
            out[name] = int(explicit) if explicit is not None else int(child.generate_state(1)[0])
        # inv #5: each aspect draws from its OWN entropy. Spawned children are
        # distinct by construction, but explicit overrides could collide (e.g.
        # split_assignment == weight_init) and silently couple two aspects.
        if len(set(out.values())) != len(out):
            dupes = {n: s for n, s in out.items() if list(out.values()).count(s) > 1}
            raise ValueError(
                f"per-aspect seeds must be pairwise distinct (inv #5); "
                f"colliding overrides: {dupes}"
            )
        return out

    @model_validator(mode="after")
    def _no_seed_collision(self) -> "Seeds":
        self.resolved()  # fail fast at load if explicit overrides collide
        return self


class SplitSpec(BaseModel):
    """One config-declared evaluation split (design_spec §6.1, inv #9/#10).

    Shift splits always declare a `count` and exactly one axis override in `axes`
    — a controlled, single-axis distribution shift held out for robustness.

    id-pool splits (empty `axes`) can be sized two ways, and a config must pick
    exactly one for all of them (`Config._check`):

      frac mode  — `scenes.n_id` scenes are generated as one pool and
                   hash-bucketed by `frac`; the split with no `frac` is the
                   residual. Proportional sizing from a single pool size.
      count mode — each id-pool split declares its own `count` and its own
                   `seed`, and its scenes are generated directly into it. This is
                   how Research I specifies its dataset (Figure 6: explicit
                   counts 500/60/60 and per-split seeds 1001-1006), which frac
                   mode cannot express — fracs cannot hit exact counts, and one
                   shared seed cannot be six.

    Both modes are live (the dry-run/test configs use frac; `research_i.yaml`
    uses count), so neither is dead weight.
    """

    model_config = {"extra": "forbid"}

    role: str            # "train" | "valid" | "test"
    frac: float | None = None    # frac-mode id-pool sizing; residual has none
    count: int | None = None     # shift-split sizing, and count-mode id-pool sizing
    axes: dict[str, str] = {}    # axis overrides vs id baseline; empty ⇒ id pool
    #: This split's own scene-generation seed. `None` ⇒ draw from the shared
    #: `scene_generation` stream (frac mode). Required in count mode, where the
    #: whole point is that each split is generated independently and reproducibly.
    seed: int | None = None

    @property
    def is_id_pool(self) -> bool:
        return not self.axes


class Margins(BaseModel):
    """Per-axis clearances (m) keeping sources/receivers off the surfaces.

    Separate wall/floor/ceiling values because Research I specifies them
    separately (Figure 5: wall 0.5, floor 0.5, ceiling 0.3) — a single scalar
    cannot express that.
    """

    model_config = {"extra": "forbid"}

    wall: float      # horizontal (x, y) clearance
    floor: float     # clearance above z = 0
    ceiling: float   # clearance below z = height

    @model_validator(mode="after")
    def _non_negative(self) -> "Margins":
        # A negative margin inverts the admissible box outward: `_placement_bounds`
        # would compute lo < 0 and hi > room extent, its emptiness check would pass,
        # and sources would be placed OUTSIDE the room with no error at all.
        for name in ("wall", "floor", "ceiling"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"margins.{name} must be >= 0; got {getattr(self, name)} "
                    f"(a negative margin places sources outside the room)"
                )
        return self


class GeometryFamily(BaseModel):
    """One named room-shape family: per-axis (lo, hi) metre sampling ranges.

    Typed (rather than a bare dict) for the same reason as PlacementRegime: an
    unrecognised key here is a HIDDEN GEOMETRY PARAMETER. Untyped, `dims` typo'd
    to `dimz` loaded fine and failed later as a bare KeyError inside the
    generator, and a stray `shape:` was accepted and silently ignored.
    """

    model_config = {"extra": "forbid"}

    dims: list[list[float]]   # [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]]

    #: Which closed-form acoustic characterization applies to this family. NO
    #: DEFAULT — each family states it, because the alternative is a uniform spine
    #: assuming a property that only happens to hold today (RD-64, the same class
    #: as the metric `kind` contract).
    #:
    #:   "sabine" — a closed enclosure. Sabine/Eyring T60, room constant, critical
    #:              distance and diffuse-field DRR are all derived from `dims`, and
    #:              the AC-22 record-length gate compares against that T60.
    #:   "none"   — not an enclosure. `_room_acoustics` records a (split, reason)
    #:              instead of a number, and `worst_case_t60` skips the family.
    #:
    #: "none" exists for the roadmap's OUTDOOR and PARTIALLY-OPEN scenes (paper
    #: §6), which design_spec §6 says the architecture must not preclude. Without
    #: it, admitting one means either emitting meaningless closed-box numbers into
    #: the canonical placement_report.json or rewriting the gate — so the seam is
    #: declared now, while the only cost is one field.
    characterization: str

    @model_validator(mode="after")
    def _check(self) -> "GeometryFamily":
        if self.characterization not in ("sabine", "none"):
            raise ValueError(
                f"characterization must be 'sabine' (a closed enclosure) or 'none' "
                f"(not an enclosure); got {self.characterization!r}. It has no "
                f"default: a family that does not state it would silently receive "
                f"closed-box Sabine numbers (RD-64)."
            )
        if len(self.dims) != 3:
            raise ValueError(f"dims must have 3 axis ranges; got {len(self.dims)}")
        for axis, rng in enumerate(self.dims):
            if len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(f"dims[{axis}] must be [lo, hi] with lo <= hi; got {rng}")
            if rng[0] <= 0:
                raise ValueError(f"dims[{axis}] must be positive; got {rng}")
        return self


class MaterialRegime(BaseModel):
    """One named material distribution: an absorption coefficient (lo, hi) range."""

    model_config = {"extra": "forbid"}

    absorption: list[float]   # [lo, hi], each in (0, 1)

    @model_validator(mode="after")
    def _check(self) -> "MaterialRegime":
        if len(self.absorption) != 2 or self.absorption[0] > self.absorption[1]:
            raise ValueError(
                f"absorption must be [lo, hi] with lo <= hi; got {self.absorption}"
            )
        if not (0.0 < self.absorption[0] and self.absorption[1] < 1.0):
            raise ValueError(
                f"absorption coefficients must lie in (0, 1); got {self.absorption}"
            )
        return self


class PlacementRegime(BaseModel):
    """One named source/receiver placement policy (design_spec §6.1).

    `height_range` and `distance_range` are REQUIRED KEYS whose value may be
    `null`. Presence is mandatory so no constraint is ever silently defaulted;
    `null` explicitly declares *no constraint*, which is a different statement
    from "nobody thought about it" and takes the unconstrained sampling path.
    """

    model_config = {"extra": "forbid"}

    #: "interior" — receiver uniform in the admissible box.
    #: "corner"   — receiver biased into a corner sub-box of size `corner_frac`.
    type: str
    corner_frac: float | None

    #: (lo, hi) metres for BOTH source and receiver z, or null for the full
    #: admissible height. Research I pins 1.2-1.8 m (seated/standing ear height).
    height_range: list[float] | None

    #: Metres for the source-receiver separation, enforced by rejection sampling.
    #: Research I pins 1.0-10.0 m. Unlike `height_range`, an ELEMENT may be null,
    #: because a backend imposes a MINIMUM separation with no corresponding maximum
    #: (AC-13/F-48) and `[lo, hi]`-only could not express that. Legal spellings:
    #:
    #:   null          no constraint at all
    #:   [lo, null]    minimum only
    #:   [null, hi]    maximum only
    #:   [lo, hi]      both, lo < hi
    #:
    #: `[null, null]` is REJECTED: a nullable element is a new sub-convention and
    #: must not become a second way to spell the existing whole-range null (RD-48).
    distance_range: list[float | None] | None

    @model_validator(mode="after")
    def _check(self) -> "PlacementRegime":
        if self.type not in ("interior", "corner"):
            raise ValueError(f"placement type must be interior|corner; got {self.type!r}")
        if self.type == "corner":
            if self.corner_frac is None:
                raise ValueError("placement type 'corner' requires a `corner_frac`")
            if not (0.0 < self.corner_frac <= 1.0):
                raise ValueError(f"corner_frac must be in (0, 1]; got {self.corner_frac}")
        elif self.corner_frac is not None:
            raise ValueError(
                f"corner_frac is meaningless for placement type {self.type!r} "
                f"(set it null) — a value here would be silently ignored"
            )
        # height_range takes both bounds or nothing — no backend constrains one
        # side of it, so it keeps the simpler contract.
        if self.height_range is not None:
            rng = self.height_range
            if len(rng) != 2 or rng[0] >= rng[1]:
                raise ValueError(f"height_range must be [lo, hi] with lo < hi; got {rng}")
            if rng[0] < 0:
                raise ValueError(f"height_range must be non-negative; got {rng}")

        if self.distance_range is not None:
            rng = self.distance_range
            if len(rng) != 2:
                raise ValueError(
                    f"distance_range must be a two-element [lo, hi]; got {rng}. "
                    f"Either bound may be null (see the field comment for the four "
                    f"legal spellings)."
                )
            lo, hi = rng
            if lo is None and hi is None:
                raise ValueError(
                    "distance_range [null, null] is not a legal spelling — write "
                    "`distance_range: null` to declare no constraint. A nullable "
                    "bound exists so a MINIMUM can be declared without a maximum, "
                    "not to give 'no constraint' a second spelling."
                )
            for bound, value in (("lower", lo), ("upper", hi)):
                if value is not None and value < 0:
                    raise ValueError(
                        f"distance_range {bound} bound must be non-negative; got {value}"
                    )
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(
                    f"distance_range must have lo < hi when both bounds are given; got {rng}"
                )
        return self


class Scenes(BaseModel):
    """Procedural scene-generation config (design_spec §6.1). All sampling ranges
    live here — the generator hardcodes nothing."""

    model_config = {"extra": "forbid"}

    #: frac-mode id-pool size (hash-bucketed by split fracs). Explicitly `null` in
    #: count mode, where each id-pool split declares its own `count` instead.
    n_id: int | None
    id_regime: dict[str, str]                       # baseline {geometry, placement, material}
    geometry_families: dict[str, GeometryFamily]
    placement_regimes: dict[str, PlacementRegime]
    material_regimes: dict[str, MaterialRegime]
    margins: Margins
    #: Rejection-sampling bound for `distance_range`. Consumed only when a regime
    #: declares one; exceeding it raises rather than silently emitting a scene
    #: that violates the constraint.
    max_placement_attempts: int

    #: Largest fraction of generated scenes whose Sabine T60 may exceed
    #: `ir_duration` (AC-22). Nothing previously checked that the declared record
    #: length could support the decay the declared geometry x absorption ranges
    #: admit, and a T30/EDT fitted over a truncated record measures the truncation,
    #: not the room. The gate binds on the REALIZED set rather than on the support
    #: corner, because the corner has near-zero draw probability while the realized
    #: rate is what actually reaches the metrics; the corner is disclosed
    #: separately (`Config.worst_case_t60`), never used as a threshold.
    #: Sabine, not Eyring: it is the longer of the two, so the check errs toward
    #: declaring a scene unsupported rather than silently truncating it.
    max_t60_over_ir_duration_frac: float

    #: Absorption above which the diffuse-field assumptions behind Sabine/Eyring,
    #: the critical distance and DRR stop holding well (AC-21). Textbook guidance
    #: puts the practical limit near alpha = 0.3; beyond it those closed forms are
    #: extrapolations, so scenes are FLAGGED (never dropped, never recomputed) and
    #: the flag is summarized per split in scenes/placement_report.json.
    diffuse_field_alpha_limit: float


class ModelSpec(BaseModel):
    """Model selection: a registry name plus its own parameter block, loaded from
    configs/models/<name>.yaml (design_spec §7/§8). The master config never bakes
    in architecture-specific fields."""

    model_config = {"extra": "forbid"}

    name: str
    params: dict[str, Any] = {}


class SimulatorSpec(BaseModel):
    """Simulator (render backend) selection: a registry name plus its own parameter
    block, loaded from configs/simulators/<name>.yaml (design_spec §5/§8, mirrors
    ModelSpec/RepresentationSpec). The master config never bakes in backend-specific
    fields — GSound-SIR's pinned commit SHA, filterbank band edges, specular count
    and retained-path policy live beside the backend, so swapping simulators is a
    config edit.

    NOT here (named non-goal, RD-40): `low_ray_budget` / `high_ray_budget`. They are
    the swept research axis (design_spec §7 "Config system — three parameter roles",
    the swept-role note) and stay TOP-LEVEL Config fields.
    Inside a plugin block they would sit under `_merge_layer`'s F-11 name-change
    scoping, which drops a block's `params` when the name changes — silently
    discarding the sweep the moment a second raytracer is selected.
    """

    model_config = {"extra": "forbid"}

    name: str
    params: dict[str, Any] = {}


class RepresentationSpec(BaseModel):
    """Representation (output-domain) selection: a registry name plus its own
    parameter block, loaded from configs/representations/<name>.yaml (design_spec
    §8, mirrors ModelSpec). The master config never bakes in rep-specific fields
    (e.g. spectrogram STFT framing) — swapping the output domain is drop-in."""

    model_config = {"extra": "forbid"}

    name: str
    params: dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level config
# ─────────────────────────────────────────────────────────────────────────────

class Config(BaseModel):
    """Resolved, concrete run configuration. Every field is populated from a YAML
    layer; there are no behavioral defaults here (edit configs/base.yaml instead)."""

    model_config = {"extra": "forbid"}

    # Experiment identity: the experiment-ledger label (design_spec §11 — D0a, E1,
    # E2, …) a real experiment run sets so its artifacts are traceable to the
    # ledger row. Stamped into each run's config.yaml; "" for unlabeled dev runs.
    # No pipeline consumer yet — real (gsound_sir) experiment runs will set it.
    run_id: str = ""

    # Randomness (per-aspect seeds)
    seeds: Seeds

    # Simulator / representation / model selection
    simulator: SimulatorSpec
    representation: RepresentationSpec
    model: ModelSpec

    # Audio
    sample_rate: int
    ir_duration: float
    ambisonics_order: int

    # Simulation ray budgets
    low_ray_budget: int
    high_ray_budget: int

    # Scene generation + evaluation splits (config-declared set)
    scenes: Scenes
    splits: dict[str, SplitSpec]

    # Training
    n_epochs: int
    batch_size: int
    learning_rate: float
    huber_delta: float
    early_stopping_patience: int

    # Reporting
    report_format: str

    # QC thresholds
    max_onset_ms: float
    min_energy_db: float

    # D0a headroom-probe verdict thresholds (design_spec §4)
    d0a_gap_large_db: float
    d0a_gap_small_db: float

    # D0b carrier-ceiling JND tolerances (design_spec §4.2)
    d0b_t30_jnd_frac: float
    d0b_edt_jnd_frac: float
    d0b_c50_jnd_db: float

    # ISO 3382 evaluation bands (Hz) — design_spec §7 / §3
    iso_eval_freqs: list[int]

    # Direct-arrival onset threshold (dB below peak) for metric integration start (§3)
    metric_onset_rel_db: float

    # How many times the octave filter's OWN decay a room's decay must exceed to be
    # resolvable in that band (dimensionless). The floor itself is measured per
    # (metric, band) from the filter's impulse response, so this declares the SAFETY
    # MARGIN, not the threshold — replacing a single absolute scalar that was
    # compared against the fitted value and therefore censored its own estimator
    # (AC-26/AC-27). Below the floor, T30/EDT is unscored-with-reason, never a small
    # number.
    metric_band_resolvability_margin: float

    # ISO 3382-1 SNR ADMISSIBILITY (AC-176), per metric, in dB of usable decay.
    # The standard requires the decay to be measurable over >= 45 dB before a T30
    # is permitted (>= 35 dB for T20, >= 20 dB for EDT); below that the fit is not
    # a room-acoustic quantity. Nothing enforced it, and nothing could: a
    # Schroeder backward integral terminates at -inf dB by construction, so the
    # [-5, -35] dB regression window is NEVER empty however little genuine decay
    # the record holds, and the terminal plunge is silently included in the fit.
    # Measured through the shipped path: -0.98 % T30 error at 85 dB of available
    # decay, -5.41 % at 42.5, -22.99 % at 30.7, -55.88 % at 18.2 -- with
    # `nan_reason` None and `resolvability` empty at every one of those points.
    #
    # DECLARED, not hardcoded, because the value is not universal: the dry-run
    # SCAFFOLD's 0.25 s records carry only 19-30 dB (measured over all 29 canonical
    # legs), so an ISO-conformant bound would refuse every scaffold scene. The
    # scaffold overlay declares its own lower value with that reason attached; the
    # real-render configs declare the standard's.
    metric_min_decay_range_db: dict[str, float]

    # Decay time (s) below which the EDT ESTIMATOR is variance-limited rather than
    # filter-limited: measured sd 24-31 % of T60 below ~0.15 s, against 6-10 % for
    # T30. Not a suppression threshold — no threshold can remove estimator variance
    # — but a DISCLOSURE bound: eval counts, per split, how many scenes' EDT falls
    # below it so a high-uncertainty population is never read as a point estimate
    # (AC-27/RD-78).
    metric_edt_variance_limited_s: float

    # Stats — bootstrap CI (design_spec §9)
    bootstrap_n_resamples: int
    bootstrap_alpha: float
    bootstrap_power: float

    # Recorded role metadata (populated by load(); not a YAML field).
    resolved_roles: dict[str, dict] = {}

    # Pre-resolution declared tree (with tune/sweep nodes intact), kept so
    # expand_sweeps() can re-resolve at other selections without re-reading files.
    _declared: dict = PrivateAttr(default_factory=dict)

    # ── Derived quantities ────────────────────────────────────────────────────
    @property
    def n_channels(self) -> int:
        return (self.ambisonics_order + 1) ** 2

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.ir_duration)

    def worst_case_t60(self) -> dict:
        """The longest Sabine T60 this config's DECLARED scene support admits (AC-22).

        A disclosure, not a threshold. Nothing previously stated the relationship
        between `ir_duration` and the decay the declared ranges allow, and the two
        can disagree badly: base's largest shoebox at its lowest absorption gives
        4.20 s, and Research I's declared support gives 4.09 s against an RI-pinned
        3.0 s record. A T30 fitted over a truncated record measures the truncation.

        This is deliberately NOT used as a gate. The corner is the product of two
        independent extremes and has near-zero probability of being drawn, so
        gating on it would reject configs whose realized scenes are all fine. The
        gate is `scenes.max_t60_over_ir_duration_frac`, applied to the REALIZED set
        at gen-scenes (RD-56). The E1 write-up needs this number either way, so it
        is computed here and stamped into resolved.yaml.

        Returns the corner and the geometry/material that produced it, so the
        number is never reported without saying which corner it came from.
        """
        alpha_min = min(m.absorption[0] for m in self.scenes.material_regimes.values())
        alpha_regime = min(
            self.scenes.material_regimes.items(), key=lambda kv: kv[1].absorption[0]
        )[0]
        worst = None
        skipped: list[str] = []
        for family, spec in self.scenes.geometry_families.items():
            # A family that is not an enclosure has no Sabine T60 to sweep, and
            # forcing the closed-box formula onto one would put a fabricated number
            # in `resolved.yaml` (RD-64). Skipped and NAMED, never silently omitted.
            if spec.characterization == "none":
                skipped.append(family)
                continue
            lx, ly, lz = (axis[1] for axis in spec.dims)  # upper bound of each axis
            volume = lx * ly * lz
            surface = 2.0 * (lx * ly + ly * lz + lx * lz)
            t60 = sabine_rt60(volume, surface, alpha_min)
            if worst is None or t60 > worst["t60_sabine_s"]:
                worst = {
                    "t60_sabine_s": float(t60),
                    "geometry_family": family,
                    "dims_m": [float(lx), float(ly), float(lz)],
                    "material_regime": alpha_regime,
                    "absorption": float(alpha_min),
                }
        if worst is None:
            return {
                "t60_sabine_s": None,
                "uncharacterized_reason": (
                    "no geometry family declares characterization: sabine, so the "
                    "declared support admits no closed-form decay corner"
                ),
                "skipped_families": skipped,
                "ir_duration_s": float(self.ir_duration),
                "covered_by_record": None,
            }
        if skipped:
            worst["skipped_families"] = skipped
        worst["ir_duration_s"] = float(self.ir_duration)
        worst["covered_by_record"] = bool(self.ir_duration >= worst["t60_sabine_s"])
        return worst

    def seed(self, name: str) -> int:
        """Concrete seed for a named pipeline aspect (see SEED_NAMES)."""
        if name not in SEED_NAMES:
            raise KeyError(f"Unknown seed name {name!r}. Known: {SEED_NAMES}")
        return self.seeds.resolved()[name]

    # ── Split helpers ─────────────────────────────────────────────────────────
    @property
    def id_pool_splits(self) -> dict[str, SplitSpec]:
        return {name: sp for name, sp in self.splits.items() if sp.is_id_pool}

    @property
    def shift_splits(self) -> dict[str, SplitSpec]:
        return {name: sp for name, sp in self.splits.items() if not sp.is_id_pool}

    @property
    def test_split_names(self) -> tuple[str, ...]:
        """All held-out test splits, in declaration order (never pooled — inv #9)."""
        return self.split_names_with_role("test")

    def split_names_with_role(self, role: str) -> tuple[str, ...]:
        """Declared splits carrying `role`, in declaration order.

        The single lookup every stage goes through, so role routing lives in one
        place: adding a role later touches SPLIT_ROLES and its consumers, never the
        enumeration logic in trainer.py / preprocess.py again (RD-53). Raises on a
        role outside the vocabulary, so a typo cannot silently return an empty
        tuple at a call site (F-44)."""
        if role not in SPLIT_ROLES:
            raise ValueError(
                f"unknown split role {role!r}; expected one of {list(SPLIT_ROLES)} "
                f"(amcd.config.SPLIT_ROLES)."
            )
        return tuple(name for name, sp in self.splits.items() if sp.role == role)

    def the_split_with_role(self, role: str) -> str:
        """The single split carrying `role` — for roles `REQUIRED_ROLE_COUNTS` pins
        to exactly one. `Config._check` has already guaranteed the count, so this
        never has to guess or silently take the first of several (F-44)."""
        names = self.split_names_with_role(role)
        if len(names) != 1:
            raise ValueError(
                f"expected exactly one split with role {role!r}, got {list(names)} — "
                f"Config._check should have rejected this at load."
            )
        return names[0]

    # ── Validation ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _check(self) -> "Config":
        """Every cross-field validation, as a list of named checks."""
        self._check_scalar_domains()
        self._check_reserved_split_names()
        # Role vocabulary + cardinality. Must run BEFORE the shift-split check,
        # which already assumes `role == "test"` is meaningful (F-44/RD-53).
        self._check_split_roles()
        self._check_id_pool_sizing()
        self._check_split_seeds()
        self._check_inert_split_fields()
        self._check_split_counts_positive()
        self._check_shift_splits()
        return self

    def _check_scalar_domains(self) -> None:
        """Ranges a §7 tuned sweep (or a typo) could otherwise drive out of bounds.

        Caught at load rather than deep in torch/stats (F-07, F-08). F-08 in
        particular: `bootstrap_n_resamples` ≤ 0 gives degenerate CIs, and CIs are
        the load-bearing evidence for every headline claim; the D0a/D0b thresholds
        are gate multipliers a negative value would silently invert.
        """
        positive_fields = (
            "huber_delta", "learning_rate",
            "bootstrap_n_resamples",
            "d0a_gap_large_db", "d0a_gap_small_db",
            "d0b_t30_jnd_frac", "d0b_edt_jnd_frac", "d0b_c50_jnd_db",
        )
        for field in positive_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be > 0; got {getattr(self, field)}")
        # 0 attempts makes every scene raise "could not satisfy distance_range None
        # in 0 attempts" — loud, but misdiagnosed as a constraint problem for a
        # config that declares no constraint at all (F-32).
        if self.scenes.max_placement_attempts <= 0:
            raise ValueError(
                f"scenes.max_placement_attempts must be > 0; "
                f"got {self.scenes.max_placement_attempts}"
            )
        # Fractions in (0, 1).
        for field in ("bootstrap_alpha", "bootstrap_power"):
            if not (0.0 < getattr(self, field) < 1.0):
                raise ValueError(f"{field} must be in (0, 1); got {getattr(self, field)}")
        # Power ≤ alpha is nonsensical (a zero effect already "achieves" power = alpha
        # two-sided), and mdes() would silently return ≈0 instead of erroring (F-17).
        if self.bootstrap_power <= self.bootstrap_alpha:
            raise ValueError(
                f"bootstrap_power must exceed bootstrap_alpha; "
                f"got power={self.bootstrap_power}, alpha={self.bootstrap_alpha}"
            )
        # metric_onset_rel_db is a threshold BELOW the peak, so it must be negative.
        if self.metric_onset_rel_db >= 0:
            raise ValueError(
                f"metric_onset_rel_db must be < 0 (dB below peak); got {self.metric_onset_rel_db}"
            )

    def _check_reserved_split_names(self) -> None:
        """`id` is the generator's "hash-bucket me" tag (a split of that name
        silently captures or loses scenes instead of being routed), and `carrier` is
        a non-split directory inside preprocessed/ that the stale-split sweep
        deliberately skips (F-38)."""
        clashes = sorted(set(self.splits) & set(RESERVED_SPLIT_NAMES))
        if clashes:
            raise ValueError(
                f"split name(s) {clashes} are reserved (they collide with pipeline "
                f"sentinels or non-split directories, and would silently misroute or "
                f"retain scenes). Reserved: {list(RESERVED_SPLIT_NAMES)}."
            )

    def _check_shift_splits(self) -> None:
        """Each shift split is a CONTROLLED single-axis perturbation of the id
        baseline, and the baseline itself is a declared regime.

        A shift split that matched id on its named axis, or named a regime that
        does not exist, would still generate and render scenes — and then be
        reported as a distribution shift that was never applied.
        """
        id_regime = self.scenes.id_regime
        axis_registries = {
            "geometry": self.scenes.geometry_families,
            "placement": self.scenes.placement_regimes,
            "material": self.scenes.material_regimes,
        }
        for name, sp in self.shift_splits.items():
            if sp.role != "test":
                raise ValueError(f"shift split {name!r} must have role 'test'")
            if sp.count is None:
                raise ValueError(f"shift split {name!r} needs an explicit `count`")
            if len(sp.axes) != 1:
                raise ValueError(
                    f"shift split {name!r} must perturb exactly one axis; got {sp.axes}"
                )
            (axis, value), = sp.axes.items()
            if axis not in axis_registries:
                raise ValueError(f"shift split {name!r}: unknown axis {axis!r}")
            if value not in axis_registries[axis]:
                raise ValueError(
                    f"shift split {name!r}: {axis}={value!r} not in scenes.{axis} regimes"
                )
            if value == id_regime.get(axis):
                raise ValueError(
                    f"shift split {name!r}: {axis}={value!r} equals the id baseline "
                    f"(no shift) — a shift split must differ from id on its named axis"
                )

        # id baseline axis values must themselves be declared regimes.
        for axis, reg in axis_registries.items():
            if id_regime.get(axis) not in reg:
                raise ValueError(
                    f"scenes.id_regime.{axis}={id_regime.get(axis)!r} not in scenes.{axis} regimes"
                )

    # ── id-pool sizing modes ──────────────────────────────────────────────────
    @property
    def id_pool_is_counted(self) -> bool:
        """True when id-pool splits are sized by explicit `count` + `seed`.

        The two modes are mutually exclusive and validated in
        `_check_id_pool_sizing`, so this single predicate is safe to branch on
        downstream (the generator and split assignment both do)."""
        return any(sp.count is not None for sp in self.id_pool_splits.values())

    def _check_split_counts_positive(self) -> None:
        """A declared `count` must be a real number of scenes (F-46).

        Count validation was asymmetric: id-pool counts had to be `> 0`, while a
        shift split's count only had to be non-None — so `count: 0` was legal
        under the declared schema and gen-scenes then died inside `_summarize`
        with a bare `zero-size array to reduction operation minimum which has no
        identity`, naming neither the split nor the count. A value the schema
        admits must not be a value the pipeline mishandles.
        """
        for name, sp in self.splits.items():
            if sp.count is not None and sp.count <= 0:
                raise ValueError(
                    f"split {name!r}: count must be > 0; got {sp.count}. A split "
                    f"that should not be generated is removed from `splits`, not "
                    f"declared with zero scenes."
                )

    def _check_split_roles(self) -> None:
        """Every split declares a role the spine understands, and the roles the
        pipeline can only have one of have exactly one (F-44).

        Without this, a role outside SPLIT_ROLES is understood by no stage: the
        split is generated, rendered and preprocessed, then silently absent from
        every inferential artifact with exit code 0 — under research_i.yaml that is
        60 emulated renders producing nothing beneath a report that looks complete.
        Two siblings close here too: two `valid` splits (the trainer took the first
        of them without comment) and zero `valid` splits (a bare StopIteration with
        an empty message, raised only AFTER render and preprocess).

        Checked at config load — the cheapest possible point — rather than at
        preprocess, which is where the old train-only check lived."""
        unknown = {
            name: sp.role for name, sp in self.splits.items() if sp.role not in SPLIT_ROLES
        }
        if unknown:
            raise ValueError(
                f"split(s) declare an unknown role: "
                f"{ {n: r for n, r in sorted(unknown.items())} }. "
                f"Expected one of {list(SPLIT_ROLES)} (amcd.config.SPLIT_ROLES). A "
                f"role no stage recognises would still be generated, rendered and "
                f"preprocessed, then appear in no result at all."
            )
        for role, required in REQUIRED_ROLE_COUNTS.items():
            names = self.split_names_with_role(role)
            if len(names) != required:
                raise ValueError(
                    f"exactly {required} split must have role {role!r}; got "
                    f"{len(names)}: {list(names)}. "
                    f"(amcd.config.REQUIRED_ROLE_COUNTS)"
                )

    def _check_id_pool_sizing(self) -> None:
        """id-pool splits are sized ALL by `frac` or ALL by `count`, never mixed.

        A mixed declaration has no coherent meaning — some splits proportional to
        a pool that the others do not draw from — and would quietly produce a
        dataset of the wrong size, so it is rejected rather than interpreted.
        """
        id_pool = self.id_pool_splits
        if not id_pool:
            raise ValueError("At least one id-pool split (empty `axes`) is required.")

        fracced = {n for n, sp in id_pool.items() if sp.frac is not None}
        counted = {n for n, sp in id_pool.items() if sp.count is not None}
        if fracced & counted:
            raise ValueError(
                f"id-pool split(s) {sorted(fracced & counted)} declare BOTH `frac` "
                f"and `count`; pick one sizing mode per config."
            )
        if counted and len(counted) != len(id_pool):
            raise ValueError(
                f"id-pool sizing must be all-`count` or all-`frac`, not mixed. "
                f"Counted: {sorted(counted)}; not counted: {sorted(set(id_pool) - counted)}. "
                f"In count mode every id-pool split needs an explicit `count` "
                f"(and `frac: null`)."
            )

        if counted:
            # ── count mode (Research I) ──
            if self.scenes.n_id is not None:
                raise ValueError(
                    f"scenes.n_id must be null in count mode: each id-pool split "
                    f"declares its own `count`, so a pool size would be unused and "
                    f"misleading (got n_id={self.scenes.n_id})."
                )
            missing_seed = sorted(n for n, sp in id_pool.items() if sp.seed is None)
            if missing_seed:
                raise ValueError(
                    f"count mode requires an explicit per-split `seed`; missing on "
                    f"{missing_seed}. Count mode exists to make each split's "
                    f"generation independent and reproducible (inv #5), which a "
                    f"shared stream cannot provide."
                )
            for name, sp in id_pool.items():
                if sp.count <= 0:
                    raise ValueError(f"split {name!r}: count must be > 0; got {sp.count}")
        else:
            # ── frac mode ──
            if self.scenes.n_id is None:
                raise ValueError(
                    "scenes.n_id is required in frac mode (it is the pool the fracs "
                    "divide up); set explicit per-split `count`s to use count mode."
                )
            frac_sum = 0.0
            residuals = []
            for name, sp in id_pool.items():
                if sp.frac is None:
                    residuals.append(name)
                else:
                    if not (0.0 < sp.frac < 1.0):
                        raise ValueError(f"split {name!r}: frac must be in (0, 1)")
                    frac_sum += sp.frac
            if len(residuals) != 1:
                raise ValueError(
                    f"Exactly one id-pool split must be the residual (no `frac`); "
                    f"got {residuals or 'none'}"
                )
            if frac_sum >= 1.0:
                raise ValueError(f"id-pool fracs sum to {frac_sum} (must be < 1.0)")

    def _check_inert_split_fields(self) -> None:
        """Reject split fields that would be silently ignored.

        Same principle as the `seeds.split_assignment` guard: a config value that
        does nothing is worse than an error, because it reads as controlling
        something it does not. `seed` in frac mode is the dangerous one — the
        researcher believes the split is independently seeded, and it is not.
        """
        for name, sp in self.splits.items():
            if not sp.is_id_pool and sp.frac is not None:
                raise ValueError(
                    f"split {name!r} is a shift split (it declares `axes`), which is "
                    f"sized by `count`; its `frac` would be ignored. Remove it."
                )
        if not self.id_pool_is_counted:
            seeded = sorted(
                n for n, sp in self.id_pool_splits.items() if sp.seed is not None
            )
            if seeded:
                raise ValueError(
                    f"id-pool split(s) {seeded} declare a `seed`, but in frac mode "
                    f"id-pool scenes are generated as ONE pool from the shared "
                    f"`scene_generation` stream and then hash-bucketed — the seed "
                    f"would be ignored, so the split is not independently seeded. "
                    f"Switch to count-mode sizing, or remove the seed."
                )
        else:
            # F-35: count mode's contract is that every split is generated
            # independently, so a shift split without its own seed silently falls
            # back to the shared stream and breaks that promise asymmetrically.
            unseeded = sorted(
                n for n, sp in self.shift_splits.items() if sp.seed is None
            )
            if unseeded:
                raise ValueError(
                    f"count mode requires an explicit `seed` on EVERY split; missing "
                    f"on shift split(s) {unseeded}. Count mode exists so each split "
                    f"is generated independently and reproducibly (inv #5)."
                )

    def _check_split_seeds(self) -> None:
        """Per-split seeds must be pairwise distinct and distinct from the named
        per-aspect seeds (inv #5: each stochastic aspect draws its own entropy).

        Two splits sharing a seed would generate the *same scenes* — an
        overlap that reads as a legitimate dataset and is invisible downstream.
        """
        declared = {n: sp.seed for n, sp in self.splits.items() if sp.seed is not None}
        if not declared:
            return
        seen: dict[int, str] = {}
        for name, seed in declared.items():
            if seed in seen:
                raise ValueError(
                    f"splits {seen[seed]!r} and {name!r} share seed {seed}; per-split "
                    f"seeds must be pairwise distinct or the splits generate "
                    f"identical scenes (inv #5)."
                )
            seen[seed] = name
        collisions = {n: s for n, s in declared.items() if s in set(self.seeds.resolved().values())}
        if collisions:
            raise ValueError(
                f"per-split seed(s) {collisions} collide with a named per-aspect "
                f"seed; each stochastic aspect must draw its own entropy (inv #5)."
            )
        # A split_assignment override is INERT in count mode (no hash-bucketing
        # happens), and a config value that silently does nothing is worse than an
        # error — it reads as controlling something it does not.
        if self.id_pool_is_counted and self.seeds.split_assignment is not None:
            raise ValueError(
                "seeds.split_assignment is set but count-mode id-pool splits are not "
                "hash-bucketed, so it would have no effect. Remove the override (or "
                "switch to frac-mode sizing)."
            )

    # ── Loading ───────────────────────────────────────────────────────────────
    @staticmethod
    def _merge_yaml(paths: list[Path]) -> dict:
        # A missing config ROOT is a layout problem, not a missing-file problem,
        # and the two need different messages (F-73). Checked here so every entry
        # point — load, with_overrides, expand_sweeps — is covered by one guard.
        _require_configs()
        merged: dict = {}
        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _merge_layer(merged, data)
        return merged

    @classmethod
    def _from_merged(cls, merged: dict, selection: dict[str, int] | None) -> "Config":
        # Attach each plugin's parameter block from its configs/<kind>/<name>.yaml,
        # so the master config never bakes in model- or rep-specific fields (§7/§8).
        # RD-13: `simulator` is attached HERE, not merely listed in _PLUGIN_BLOCKS.
        # _PLUGIN_BLOCKS membership only scopes params across a name change (F-11);
        # attachment is what puts the params file's contents into the tree BEFORE
        # _resolve_roles runs, which is what lets a `sweep:` inside simulator params
        # (e.g. the roadmap's retained-path-count axis) expand into sibling runs.
        merged = {
            **merged,
            "model": _attach_params_block(merged.get("model"), "model", _MODELS_DIR),
            "representation": _attach_params_block(
                merged.get("representation"), "representation", _REPS_DIR
            ),
            "simulator": _attach_params_block(merged.get("simulator"), "simulator", _SIMS_DIR),
        }

        concrete, roles = _resolve_roles(merged, selection or {})
        concrete["resolved_roles"] = roles
        obj = cls(**concrete)
        obj._declared = merged
        return obj

    @classmethod
    def load(cls, *config_paths: Path, selection: dict[str, int] | None = None) -> "Config":
        """Merge YAMLs left-to-right on top of base.yaml, resolve roles, validate."""
        paths = list(config_paths)
        if not any(Path(p).resolve() == _BASE_YAML.resolve() for p in paths):
            paths = [_BASE_YAML] + paths
        return cls._from_merged(cls._merge_yaml(paths), selection)

    @classmethod
    def with_overrides(cls, **kwargs) -> "Config":
        """Load base.yaml then apply keyword overrides (used in tests)."""
        merged = cls._merge_yaml([_BASE_YAML])
        _merge_layer(merged, kwargs)
        return cls._from_merged(merged, None)

    def expand_sweeps(self) -> list["Config"]:
        """Expand every `{sweep: [...]}` into the cartesian product of sibling runs.

        Each sibling is a fully concrete Config selecting one value per swept axis
        (design_spec §7 provenance: a swept param → N sibling runs). Used by E4; a
        run with no sweeps returns just this config.
        """
        axes = _sweep_axes(self._declared)
        if not axes:
            return [self]
        names = list(axes)
        siblings: list[Config] = []
        for combo in itertools.product(*(range(axes[n]) for n in names)):
            selection = dict(zip(names, combo))
            siblings.append(self._from_merged(self._declared, selection))
        return siblings

    # ── Provenance ────────────────────────────────────────────────────────────
    def stamp(self, run_dir: Path) -> None:
        """Write the resolved concrete config, the flattened resolved point + seeds,
        and package versions.

        Note: config.yaml holds the RESOLVED run (tune/sweep leaves already collapsed
        to scalars), not the pre-resolution declared tree — the declared spaces are
        recorded compactly under `roles` in resolved.yaml. The declared tree itself
        (`self._declared`) is kept in memory for expand_sweeps() but is not stamped."""
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = self.model_dump()
        # config.yaml = the concrete run as validated (incl. resolved seeds + roles).
        (run_dir / "config.yaml").write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))

        # resolved.yaml = the flattened concrete point + concrete seeds, distinct
        # from the declared role spaces (E3/E4 provenance, reproducibility supplement).
        resolved = {
            "seeds": self.seeds.resolved(),
            "roles": self.resolved_roles,
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "test_split_names": list(self.test_split_names),
            # The longest decay the DECLARED ranges admit, against the record
            # length. Disclosure, never a gate (AC-22/RD-56) — the E1 write-up
            # needs it, and a config whose record does not cover its own support
            # should say so in its own provenance rather than in a comment.
            "worst_case_t60": self.worst_case_t60(),
        }
        (run_dir / "resolved.yaml").write_text(yaml.dump(resolved, default_flow_style=False, sort_keys=False))

        packages = ["torch", "numpy", "scipy", "pydantic", "pandas", "pyarrow"]
        versions: dict[str, str] = {
            "python": sys.version,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "not-installed"
        # git sha/dirty are HUMAN provenance resolved from the PACKAGE, never the
        # run_dir — see amcd.provenance (F-56).
        versions["git_sha"] = provenance.git_sha()
        versions["git_dirty"] = provenance.git_is_dirty()
        # Whole-package and human-facing, never a cache key — see amcd.provenance,
        # which owns that distinction (ALL_SOURCES).
        #
        # What is NOT said there, because it is about this stamp: it describes THIS
        # INVOCATION, and stamp() runs before any stage does, so it is not a claim
        # that this code produced the run_dir's artifacts. It read as one — an
        # all-cached re-run re-stamps the current hash, so a run_dir whose renders
        # predate a backend change carried a stamp asserting the new code made them
        # (F-75). Per-stage truth lives in `stages/<stage>.done`
        # (`code_version_unscoped`). Said in the FILE below, not only here, because
        # a comment does not reach whoever reads versions.json.
        versions["code_version"] = provenance.code_version(provenance.ALL_SOURCES)
        versions["code_version_describes"] = (
            "this invocation, not necessarily the artifacts in this run_dir; "
            "cached stages may predate it — see stages/<stage>.done"
        )
        # WHICH MACHINE, beside which code — the same config and code_version give
        # different weights on MPS and on CUDA/CPU. Neither is a cache key; see
        # amcd.provenance (F-74).
        versions["device"] = str(provenance.select_device())
        versions["platform_machine"] = provenance.host_platform()
        (run_dir / "versions.json").write_text(json.dumps(versions, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Plugin blocks + layer merge
# ─────────────────────────────────────────────────────────────────────────────

# kind → (package to import so its plugins self-register, registry attribute name).
_PLUGIN_REGISTRY = {
    "model": ("amcd.models", "model_registry"),
    "representation": ("amcd.representations", "representation_registry"),
    "simulator": ("amcd.simulators", "simulator_registry"),
}


def _require_registered(kind: str, name: str) -> None:
    """Fail loud at load if `name` is not a registered plugin of `kind`.

    Used only on the no-params-file path: the registry — not the presence of a
    `<name>.yaml` — is the source of truth for valid names, so a parameter-free or
    stub plugin (e.g. `edr`) needs no file, while a typo'd name is still caught
    here (with the available-names list) instead of proceeding silently (F-12)."""
    import importlib

    pkg_name, registry_attr = _PLUGIN_REGISTRY[kind]
    importlib.import_module(pkg_name)  # trigger the package's registrations
    from . import registry as _registry_mod

    registry = getattr(_registry_mod, registry_attr)
    if name not in registry:
        raise ValueError(f"Unknown {kind}: {name!r}. Available: {registry.keys()}")


def _attach_params_block(block: Any, kind: str, config_dir: Path) -> dict:
    """Resolve a `{name, params}` plugin block by loading its per-name params file
    from `config_dir/<name>.yaml` and layering the inline `params` override on top.

    Shared by `model` and `representation` (design_spec §7/§8): the master config
    carries only the plugin NAME plus optional overrides; the concrete default
    parameter set lives beside the plugin so the master config stays plugin-agnostic.

    A missing `<name>.yaml` is allowed only for a REGISTERED plugin (F-12): a
    parameter-free or not-yet-implemented stub (e.g. `edr`) needs no file and gets
    empty params, so it reaches its own NotImplementedError rather than a
    file-not-found; an unknown name fails loud here.
    """
    if not isinstance(block, dict) or "name" not in block:
        raise ValueError(f"config `{kind}` must be a mapping with a `name` key")
    params_file = config_dir / f"{block['name']}.yaml"
    if params_file.exists():
        with open(params_file) as f:
            file_params = yaml.safe_load(f) or {}
    else:
        _require_registered(kind, block["name"])
        file_params = {}
    merged_params = {**file_params, **block.get("params", {})}  # inline overrides file
    return {"name": block["name"], "params": merged_params}


#: Config blocks of the form `{name, params}` whose params are scoped to `name`.
_PLUGIN_BLOCKS = ("model", "representation", "simulator")


def _merge_layer(base: dict, incoming: dict) -> None:
    """Deep-merge one config layer, with plugin-block params SCOPED TO THE NAME.

    A `{name, params}` block's params belong to that specific plugin. When an
    incoming layer switches `model`/`representation` name, the prior name's params
    must not bleed into the new plugin — they are for a different schema (F-11).
    So drop the accumulated block's `params` on a name change, then deep-merge.
    This is the single merge primitive; use it wherever config layers combine."""
    for key in _PLUGIN_BLOCKS:
        inc, cur = incoming.get(key), base.get(key)
        if (
            isinstance(inc, dict) and isinstance(cur, dict)
            and "name" in inc and inc["name"] != cur.get("name")
        ):
            cur.pop("params", None)
    _deep_update(base, incoming)


def _deep_update(base: dict, overrides: dict) -> None:
    """Recursively merge `overrides` into `base` (nested dicts merged, not replaced).

    Note: this is name-agnostic; layered merges of `{name, params}` plugin blocks
    must go through `_merge_layer` so params cannot bleed across a name switch."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
