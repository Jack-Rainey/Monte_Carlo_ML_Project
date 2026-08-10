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
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from numpy.random import SeedSequence
from pydantic import BaseModel, PrivateAttr, model_validator


_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
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
    """One master seed plus optional per-aspect overrides (design_spec §5, inv #5).

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

    @model_validator(mode="after")
    def _check(self) -> "GeometryFamily":
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

    #: (lo, hi) metres for the source-receiver separation, enforced by rejection
    #: sampling, or null for no constraint. Research I pins 1.0-10.0 m.
    distance_range: list[float] | None

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
        for name in ("height_range", "distance_range"):
            rng = getattr(self, name)
            if rng is None:
                continue
            if len(rng) != 2 or rng[0] >= rng[1]:
                raise ValueError(f"{name} must be [lo, hi] with lo < hi; got {rng}")
            if rng[0] < 0:
                raise ValueError(f"{name} must be non-negative; got {rng}")
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
    the swept research axis (design_spec §7 l.219) and stay TOP-LEVEL Config fields.
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
        return tuple(name for name, sp in self.splits.items() if sp.role == "test")

    # ── Validation ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _check(self) -> "Config":
        # Positivity guards for scalars a §7 tuned sweep (or a typo) could otherwise
        # drive ≤ 0, caught here at load rather than deep in torch/stats. (F-07, F-08)
        # F-08 in particular: bootstrap_n_resamples ≤ 0 → degenerate/empty CIs, and CIs
        # are the load-bearing evidence for every headline claim; the D0a/D0b thresholds
        # are gate multipliers a negative value would silently invert.
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

        # Reserved names: `id` is the generator's "hash-bucket me" tag (a split of
        # that name silently captures or loses scenes instead of being routed), and
        # `carrier` is a non-split directory inside preprocessed/ that the stale-
        # split sweep deliberately skips.
        clashes = sorted(set(self.splits) & set(RESERVED_SPLIT_NAMES))
        if clashes:
            raise ValueError(
                f"split name(s) {clashes} are reserved (they collide with pipeline "
                f"sentinels or non-split directories, and would silently misroute or "
                f"retain scenes). Reserved: {list(RESERVED_SPLIT_NAMES)}."
            )
        self._check_id_pool_sizing()
        self._check_split_seeds()
        self._check_inert_split_fields()

        # Shift splits: each needs a count and exactly one axis, over a known regime
        # value that differs from the id baseline (controlled single-axis shift).
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
        return self

    # ── id-pool sizing modes ──────────────────────────────────────────────────
    @property
    def id_pool_is_counted(self) -> bool:
        """True when id-pool splits are sized by explicit `count` + `seed`.

        The two modes are mutually exclusive and validated in
        `_check_id_pool_sizing`, so this single predicate is safe to branch on
        downstream (the generator and split assignment both do)."""
        return any(sp.count is not None for sp in self.id_pool_splits.values())

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
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=run_dir.parent, stderr=subprocess.DEVNULL
            ).decode().strip()
            versions["git_sha"] = sha
        except Exception:
            versions["git_sha"] = "unavailable"
        (run_dir / "versions.json").write_text(json.dumps(versions, indent=2))


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
