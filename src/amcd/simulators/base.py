from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class SceneSpec:
    """One scene as SPECIFIED, not as rendered (RR-84).

    A config-sampled description of a room and a source/receiver placement, passed
    unchanged from gen-scenes through the render stage to whichever backend is
    selected. It carries no acoustics: every derived quantity (Sabine T60, critical
    distance, DRR) is computed elsewhere from these fields, and a backend may
    realize different acoustics from the same spec — see the absorption note below.

    COORDINATE FRAME: right-handed, metres, origin at one corner of the box, axes
    along `dims`. Positions are absolute in that frame, never fractions of a
    dimension.
    """

    scene_id: str
    #: Drawn from the named `scene_generation` seed aspect, never shared with the
    #: split, model-init or shuffling aspects (per-aspect seeds).
    seed: int
    #: Which room shape the generator sampled, from the `scenes.geometry_families`
    #: config keys — "shoebox" is the only one today. Config names it, not code.
    geometry_family: str
    #: (width, length, height) in METRES.
    dims: tuple[float, float, float]
    #: Absorption coefficient in [0, 1]: ONE scalar applied to ALL SIX surfaces and
    #: FREQUENCY-INDEPENDENT. It is the NOMINAL value — what a backend realizes from
    #: it is that backend's own convention (AC-54/RD-144), so a closed form derived
    #: from this number describes the declared room, not necessarily the rendered one.
    material_absorption: float
    #: `source_pos` / `receiver_pos`: METRES, in the frame above.
    source_pos: tuple[float, float, float]
    receiver_pos: tuple[float, float, float]
    #: Per-scene backend overrides. RESERVED AND UNUSED: `scenes/generator.py`
    #: writes `{}` for every scene and no simulator reads it — it only round-trips
    #: through `to_dict`/`from_dict`. The field exists so a future regime can vary a
    #: backend parameter per scene without a second scene type; wiring it up means
    #: giving it a consumer, not just a value.
    sim_params: dict = field(default_factory=dict)
    #: Which distribution-shift regime this scene was generated for.
    #: "id" → in-distribution (train/valid/test_id); shift names → locked to that
    #: test split.
    split_regime: str = "id"
    #: Labels for each distribution-shift axis (geometry/placement/material). A shift
    #: scene differs from the id baseline in exactly one of these — the
    #: controlled-shift integrity check (invariant #10) asserts on these labels, not
    #: on the `split_regime` tag.
    regime_axes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "geometry_family": self.geometry_family,
            "dims": list(self.dims),
            "material_absorption": self.material_absorption,
            "source_pos": list(self.source_pos),
            "receiver_pos": list(self.receiver_pos),
            "sim_params": self.sim_params,
            "split_regime": self.split_regime,
            "regime_axes": self.regime_axes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        return cls(
            scene_id=d["scene_id"],
            seed=d["seed"],
            geometry_family=d["geometry_family"],
            dims=tuple(d["dims"]),
            material_absorption=d["material_absorption"],
            source_pos=tuple(d["source_pos"]),
            receiver_pos=tuple(d["receiver_pos"]),
            sim_params=d.get("sim_params", {}),
            split_regime=d.get("split_regime", "id"),
            regime_axes=d.get("regime_axes", {}),
        )

    @classmethod
    def from_json(cls, path) -> "SceneSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))


#: Per-path arrays every `PathData` carries, with the dtype each is stored and
#: reloaded at. Pinned to the ACTUAL keys of upstream `getPathData()["path_data"][i]`,
#: verified by introspection against the pinned SHA (docs/gsound_sir_setup.md §4) —
#: not copied from upstream's docs, which disagree with the built module in at least
#: one place (`generate_ambisonic_ir` has no `path_types` argument).
#:
#: `intensities` is the (N, num_bands) one; every other array is (N,) or (N, 3).
PATH_ARRAY_DTYPES: dict[str, str] = {
    "distances": "float32",
    "intensities": "float32",
    "listener_directions": "float32",
    "source_directions": "float32",
    "path_types": "uint32",
    "speeds_of_sound": "float32",
    "relative_speeds": "float32",
    "source_indices": "uint64",
}

#: Scalars upstream reports alongside the arrays, with their dtypes. Names here are
#: RESERVED: `describe()` merges them into the descriptor block, so a descriptor key
#: of the same name is stripped on read (RR-74).
PATH_SCALARS: dict[str, type] = {
    "num_paths": int,
    "num_bands": int,
    "total_energy": float,
    "kept_energy_percentage": float,
}

#: Scalars that may legitimately be None because the quantity is UNDEFINED rather
#: than zero — `kept_energy_percentage` when the leg carries no energy at all, where
#: a 0.0 would read as "we retained almost nothing" for a subset that in fact
#: retained everything (F-85). Coerced only when present.
PATH_SCALARS_NULLABLE = ("kept_energy_percentage",)


@dataclass
class PathData:
    """The retained propagation paths for one rendered leg (design_spec §8).

    THE FILE MUST BE SELF-DESCRIBING (RD-24). `intensities` is (N, num_bands) and
    its band meaning — which frequency each column is — lives nowhere in the array
    itself. Left implicit it would live only in the simulator config that produced
    it, so a path file from a SECOND raytracer (the roadmap wants several) would be
    uninterpretable the moment it was separated from that config. `describe()`
    therefore travels INSIDE the parquet, in the file's own key/value metadata.

    The descriptor also carries `ray_budget`, `leg` and `realization_index`
    (RD-117/RD-23): the current `paths_{low,high}.parquet` filename convention
    encodes exactly two legs and one realization, and RD-23's requirement ON THIS
    GATE is that the artifact layout must not foreclose a realization index. Naming
    is not the identifier — the file's own metadata is — so adding budgets (the E4
    ray-count sweep) or realizations later needs no migration of files already
    written.
    """

    #: (N,) metres, per path.
    distances: np.ndarray
    #: (N, num_bands) per-band energy. Bands are named by `band_edges_hz` /
    #: `band_centres_hz` in the descriptor — never positionally by convention.
    intensities: np.ndarray
    #: (N, 3) unit vectors in the WORLD frame, one per path (AC-81).
    #:
    #: SENSE, which the ambisonic encoding depends on and which was undeclared:
    #: `listener_directions` is the direction of ARRIVAL AT THE LISTENER — it
    #: points from the listener toward where the energy comes from, matching
    #: upstream's `directionFromListener`. `source_directions` is the direction of
    #: EMISSION AT THE SOURCE, pointing from the source along the path's departure.
    #: They are NOT negatives of each other except on the direct path in free
    #: field: a reflected path leaves the source and arrives at the listener along
    #: different bearings, which is the whole reason both are stored.
    #:
    #: A sign error here rotates the soundfield rather than failing, so it would
    #: survive every scalar metric in this project — T30, EDT and C50 are all
    #: computed from the W channel alone and are blind to it.
    listener_directions: np.ndarray
    source_directions: np.ndarray
    #: (N,) upstream path-type bitmask. The bit→meaning mapping is upstream's
    #: `gs::PathFlags` and is NOT captured by the descriptor: a reader can group
    #: paths by identical masks but cannot name what a bit means without upstream
    #: at `descriptor["commit_sha"]` (RR-75).
    path_types: np.ndarray
    #: (N,) m/s, per path. Cross-checked against the backend's DECLARED
    #: `speed_of_sound_m_s` at render time (RD-19): gsound's 344 m/s lives in C++
    #: and can only be declared, so this array is the only way that declaration is
    #: falsifiable.
    speeds_of_sound: np.ndarray
    #: (N,) m/s Doppler RADIAL VELOCITY of the path — source/listener closing speed,
    #: which upstream applies as `shift = 1 + relative_speed / speed_of_sound`. Zero
    #: for these static scenes. NOT a propagation speed: `speeds_of_sound` above is
    #: the one to divide a distance by (AC-62).
    relative_speeds: np.ndarray
    #: (N,) index of the source each path came from.
    source_indices: np.ndarray

    #: Paths in THIS FILE, i.e. after retention. `descriptor["synthesis_num_paths"]`
    #: is how many the IR was synthesized from, and the two differ by orders of
    #: magnitude under `top_k` (RR-70).
    num_paths: int
    #: Width of the `intensities` band axis; named by the descriptor's
    #: `band_centres_hz` / `band_edges_hz`, which must agree with it (F-88).
    num_bands: int
    #: Summed per-band intensity over ALL simulated paths, retained or not — the
    #: denominator `kept_energy_percentage` is a share of.
    total_energy: float
    #: Share of `total_energy` in the retained subset, in percent. None when
    #: `total_energy` is zero, where the share is undefined (F-85).
    kept_energy_percentage: float | None

    #: Everything needed to interpret the arrays without the producing config.
    #: Written verbatim into the parquet's key/value metadata; see `describe()`.
    descriptor: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.num_paths)
        for name, dtype in PATH_ARRAY_DTYPES.items():
            # The declared dtype is enforced at construction, not only on read
            # (F-95), so written == declared == read-back. A backend needing more
            # precision widens PATH_ARRAY_DTYPES rather than passing a wider array;
            # a lossy cast here is currently silent (F-110/F-122).
            arr = np.asarray(getattr(self, name), dtype=dtype)
            if arr.shape[0] != n:
                raise ValueError(
                    f"PathData.{name} has {arr.shape[0]} rows but num_paths is {n}; "
                    f"every per-path array must agree with the path count."
                )
            setattr(self, name, arr)
        if self.intensities.ndim != 2 or self.intensities.shape[1] != int(self.num_bands):
            raise ValueError(
                f"PathData.intensities must be (num_paths, num_bands) = "
                f"({n}, {self.num_bands}); got {self.intensities.shape}. The band axis "
                f"is what `descriptor['band_edges_hz']` names — a mismatch means the "
                f"file would describe bands it does not contain."
            )

    def to_parquet(self, path: Path) -> None:
        """Write the paths plus their descriptor into one self-describing file.

        `intensities` becomes a fixed-size list column so the band axis survives the
        round trip as a shape rather than as 8 positionally-named columns whose
        meaning a reader would have to reconstruct.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {
            name: pa.array(np.ascontiguousarray(getattr(self, name)).reshape(-1))
            if getattr(self, name).ndim == 1
            else pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(getattr(self, name)).reshape(-1)),
                getattr(self, name).shape[1],
            )
            for name in PATH_ARRAY_DTYPES
        }
        table = pa.table(columns)
        # Parquet key/value metadata is bytes-only, so the descriptor travels as one
        # JSON blob under a single key rather than as N stringified entries.
        table = table.replace_schema_metadata(
            {b"amcd_path_data": json.dumps(self.describe()).encode("utf-8")}
        )
        pq.write_table(table, Path(path))

    @classmethod
    def from_parquet(cls, path: Path) -> "PathData":
        """Reconstruct from a file written by `to_parquet`, descriptor included."""
        import pyarrow.parquet as pq

        table = pq.read_table(Path(path))
        raw = (table.schema.metadata or {}).get(b"amcd_path_data")
        if raw is None:
            raise ValueError(
                f"{path} carries no `amcd_path_data` metadata, so its band axis, "
                f"producing simulator and commit sha are unknown. A path file "
                f"without its descriptor is uninterpretable by design (RD-24) — it "
                f"was not written by PathData.to_parquet."
            )
        record = json.loads(raw.decode("utf-8"))
        # `describe()` writes the descriptor and the scalars into one block; split
        # them back apart so a round trip returns the SAME `descriptor` it was given
        # rather than one that has absorbed the scalar fields.
        descriptor = {k: v for k, v in record.items() if k not in PATH_SCALARS}

        import pyarrow as pa

        arrays = {}
        for name, dtype in PATH_ARRAY_DTYPES.items():
            col = table.column(name).combine_chunks()
            if pa.types.is_fixed_size_list(col.type):
                width = col.type.list_size
                arrays[name] = np.asarray(col.flatten(), dtype=dtype).reshape(-1, width)
            else:
                arrays[name] = np.asarray(col, dtype=dtype)

        scalars = {
            name: None
            if record[name] is None and name in PATH_SCALARS_NULLABLE
            else kind(record[name])
            for name, kind in PATH_SCALARS.items()
        }
        return cls(**arrays, **scalars, descriptor=descriptor)

    def describe(self) -> dict:
        """The self-describing block: the descriptor plus the scalars it must agree with.

        Kept as one method so the written file and any in-memory reader see the same
        record, and so a missing key fails in one place rather than per call site.
        `PATH_SCALARS` names win: a descriptor key of the same name is overwritten
        here and stripped again by `from_parquet`, so those names are reserved
        (RR-74).
        """
        return {**self.descriptor, **{name: getattr(self, name) for name in PATH_SCALARS}}


#: Descriptor keys a `PathData` must carry to be interpretable on its own (RD-24).
#: `band_edges_hz` + `band_centres_hz` name the `intensities` columns; `simulator` +
#: `commit_sha` say what produced them; `ray_budget` + `leg` + `realization_index`
#: identify WHICH render this is without relying on the filename (RD-117/RD-23).
#:
#: A FLOOR, not the full set: a backend may add its own keys, and gsound_sir does
#: (`synthesis_num_paths`). Only absence is an error (RR-70).
REQUIRED_PATH_DESCRIPTOR_KEYS = (
    "simulator",
    "commit_sha",
    "band_edges_hz",
    "band_centres_hz",
    "sample_rate",
    "speed_of_sound_m_s",
    "path_retention",
    "ray_budget",
    "leg",
    "realization_index",
)


def validate_path_descriptor(paths: "PathData", *, simulator_name: str, scene_id: str) -> None:
    """Raise unless `paths.descriptor` carries every required key (RD-24).

    The `validate_provenance` of the path artifact: without a declared set, a second
    raytracer silently omits the facts that make an expensive path file readable and
    the artifact degrades with no error.
    """
    missing = [k for k in REQUIRED_PATH_DESCRIPTOR_KEYS if k not in paths.descriptor]
    if missing:
        raise ValueError(
            f"simulator {simulator_name!r} returned PathData whose descriptor is "
            f"missing {missing} for scene {scene_id!r}. A path file must be readable "
            f"without the config that produced it "
            f"(amcd.simulators.base.REQUIRED_PATH_DESCRIPTOR_KEYS)."
        )

    # Presence is not interpretability (F-88). `__post_init__` only compares
    # num_bands against `intensities`, both from the same producer and so
    # self-consistent by construction; nothing checked that the descriptor NAMES the
    # right number of bands. A file naming 8 centres for 9 intensity columns is
    # exactly the uninterpretable path file RD-24 exists to prevent.
    n_bands = int(paths.num_bands)
    centres, edges = paths.descriptor["band_centres_hz"], paths.descriptor["band_edges_hz"]
    if len(centres) != n_bands or len(edges) != n_bands - 1:
        raise ValueError(
            f"simulator {simulator_name!r} returned PathData for scene {scene_id!r} "
            f"whose descriptor names {len(centres)} band centres and {len(edges)} "
            f"band edges for {n_bands} intensity columns; a filterbank of {n_bands} "
            f"bands has exactly {n_bands} centres and {n_bands - 1} crossovers. The "
            f"band axis would be misnamed, which is the failure "
            f"REQUIRED_PATH_DESCRIPTOR_KEYS exists to prevent (RD-24)."
        )


@dataclass
class IRResult:
    """Result of a single render call.

    `meta` is the simulator's own provenance record for this leg. It is written
    into the render stage's CANONICAL meta.json at every save level, so it must
    carry at least `REQUIRED_PROVENANCE_KEYS` — see below.

    `paths` is the retained propagation paths for this leg, the producer half of
    the path-conditioned-variant seam design_spec §8 shows (RD-08). It is `None`
    for any backend that does not export paths — the scaffold does not — and the
    render stage keys on the FIELD, never on the simulator's type, so a backend
    without paths needs no downstream edit (the scaffolding rule).
    """
    ir: np.ndarray  # (C, T) float32, channel-first
    meta: dict = field(default_factory=dict)
    paths: PathData | None = None


#: Provenance every simulator MUST report per rendered leg, validated by the render
#: stage (RD-31). Without a declared set, "each simulator describes itself" lets a
#: second raytracer silently omit the facts that make an expensive dataset
#: interpretable, and the canonical record degrades with no error. Same contract
#: shape as design_spec §6's required metric `kind`: no default, and the spine
#: never assumes one.
#:
#: Keys are deliberately simulator-AGNOSTIC — they name physical//numerical facts
#: any geometric-acoustic backend can state, not GSound concepts:
#:   simulator            registry name that produced this leg
#:   ray_budget           the budget this leg was rendered at
#:   speed_of_sound_m_s   the speed the backend actually used (RD-19: gsound's
#:                        344 m/s lives in C++, so it is DECLARED, not configured)
#:   ambisonic_convention channel ordering + normalization, as a string the backend
#:                        defines and documents at its own constant (gsound_sir's is
#:                        `_AMBISONIC_CONVENTION`, measured rather than read off a
#:                        source comment — AC-57). Named here, never valued: a
#:                        second backend's convention is its own fact.
#:   rng_seeded           whether the render is reproducible from a seed. A backend
#:                        may refine it with its own keys where one boolean hides a
#:                        distinction that matters (gsound_sir splits the ray tracer
#:                        from the synthesis carrier — AC-59).
REQUIRED_PROVENANCE_KEYS = (
    "simulator",
    "ray_budget",
    "speed_of_sound_m_s",
    "ambisonic_convention",
    "rng_seeded",
)


def validate_provenance(meta: dict, *, simulator_name: str, scene_id: str, leg: str) -> None:
    """Raise unless `meta` carries every required provenance key (RD-31)."""
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"simulator {simulator_name!r} returned IRResult.meta missing required "
            f"provenance key(s) {missing} for scene {scene_id!r} leg {leg!r}. "
            f"Every simulator must declare {list(REQUIRED_PROVENANCE_KEYS)} "
            f"(amcd.simulators.base.REQUIRED_PROVENANCE_KEYS)."
        )


@runtime_checkable
class Simulator(Protocol):
    """Render backend (design_spec §8 "Plugin interfaces (signatures)").

    Implementations additionally carry a nested pydantic `Params` schema
    (`extra="forbid"`) describing their own config block — see `build_simulator`
    — and must populate `REQUIRED_PROVENANCE_KEYS` in each `IRResult.meta`.
    Neither is expressible in a `runtime_checkable` Protocol, so both are
    enforced at construction and at render time respectively, not by isinstance.

    A third required member, `min_source_receiver_distance_m`, is declared below.

    A backend may ALSO declare `host_scoped_params() -> tuple[str, ...]`, naming
    the params of its own that are host facts rather than dataset facts (F-86).
    It is deliberately NOT a member of this Protocol: Protocol membership is
    structural, so declaring it here would make every backend that legitimately
    has nothing to redact — the scaffold included — fail `issubclass` (F-121).
    `simulator_host_scoped_params` below is the accessor, and it treats absence as
    "nothing to redact".
    Implementation constraint that goes with it: **no simulator `__init__` may
    require the render environment.** The floor is consulted at gen-scenes, which
    runs in the native pipeline env, while the render backend may live in a
    separate x86 env (docs/gsound_sir_setup.md); an `__init__` that imported a
    native dependency would make scene generation unavailable off the render host.
    """

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult: ...

    @classmethod
    def min_source_receiver_distance_m(cls, params: dict) -> float:
        """Smallest source-receiver separation this backend can render, in metres.

        A CLASSMETHOD over validated params, not an instance attribute, because it
        is needed BEFORE any render — gen-scenes must reject a placement regime
        that would emit unrenderable scenes, and doing that by constructing the
        backend would couple scene generation to the render environment (RD-60).

        This is deliberately NOT a `REQUIRED_PROVENANCE_KEYS` member: those
        validate an `IRResult`, i.e. after a render has already happened, which is
        far too late for a floor whose whole job is to prevent one (RD-49). Left
        optional, a second raytracer would omit it and the pre-flight would
        silently degrade to a 0.0 floor — the silent-contract failure RD-31 closed
        on the post-render side.

        Derive it where it is already implied (gsound_sir: source_radius +
        listener_radius) rather than declaring a second number that can disagree
        with the geometry it describes.
        """
        ...


def build_simulator(
    name: str,
    params: dict,
    *,
    n_channels: int,
    n_samples: int,
    sample_rate: int,
):
    """Instantiate a registered simulator, validating `params` against its own schema.

    Mirrors `build_model` (models/cnn.py): registry lookup → nested
    `Params(extra="forbid")` validation → construct. Keeps the render stage
    backend-agnostic — adding a simulator needs no change here beyond registering
    it with its own `Params` schema (design_spec §5).
    """
    from ..registry import simulator_registry

    SimClass = simulator_registry.get(name)
    validated = SimClass.Params(**params).model_dump()
    # Fail here, not at render: a backend missing the pre-render half of the
    # contract must be caught at construction, the same way validate_provenance
    # catches the post-render half (RD-49).
    _validate_min_separation_declared(SimClass, name, validated)
    return SimClass(
        n_channels=n_channels,
        n_samples=n_samples,
        sample_rate=sample_rate,
        **validated,
    )


def _validate_min_separation_declared(SimClass, name: str, validated: dict) -> float:
    """Raise unless `SimClass` declares a usable `min_source_receiver_distance_m`."""
    getter = getattr(SimClass, "min_source_receiver_distance_m", None)
    if not callable(getter):
        raise TypeError(
            f"simulator {name!r} does not declare the required classmethod "
            f"`min_source_receiver_distance_m(params) -> float`. Every backend must "
            f"state the smallest source-receiver separation it can render, so scene "
            f"generation can reject unrenderable placements BEFORE a render "
            f"(amcd.simulators.base.Simulator)."
        )
    floor = getter(validated)
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or floor < 0:
        raise ValueError(
            f"simulator {name!r} declared min_source_receiver_distance_m="
            f"{floor!r}; expected a non-negative number of metres."
        )
    return float(floor)


def simulator_realized_support_s(config, t60_s: float, volume_m3: float,
                                 surface_m2: float, window_s: float) -> float:
    """Seconds of usable record the active backend will produce for this room.

    The record-length gate needs this BEFORE any render exists, and it is a backend
    fact, not a scene fact: gsound's adaptive energy trim closes the record when it
    runs out of paths to trace, while the scaffold fills its whole window. Geometry
    AND decay are both passed because different backends bind on different things —
    gsound measurably does not use `t60_s` (AC-186), and one that did must be able
    to say so without this signature changing. `ir_duration` is neither — it is the window the pipeline allocates, and
    gating against it asks whether the decay fits a buffer rather than whether the
    backend will fill that buffer.

    Same shape as `simulator_min_separation` and for the same reasons: registry
    lookup plus the backend's own `Params` validation, deliberately WITHOUT
    instantiating, so scene generation stays runnable on a host with no render
    environment (RD-60), and so no stage outside `simulators/` names a backend.
    """
    from ..registry import simulator_registry

    name = config.simulator.name
    SimClass = simulator_registry.get(name)
    validated = SimClass.Params(**config.simulator.params).model_dump()
    return _validate_realized_support_declared(
        SimClass, name, validated, t60_s, volume_m3, surface_m2, window_s)


def _validate_realized_support_declared(
    SimClass, name: str, validated: dict, t60_s: float, volume_m3: float,
    surface_m2: float, window_s: float
) -> float:
    """Raise unless `SimClass` declares a usable `realized_support_s`."""
    getter = getattr(SimClass, "realized_support_s", None)
    if not callable(getter):
        raise TypeError(
            f"simulator {name!r} does not declare the required classmethod "
            f"`realized_support_s(params, t60_s, volume_m3, surface_m2, window_s) "
            f"-> float`. Every backend "
            f"must state "
            f"how much record it will actually produce for a given decay, so scene "
            f"generation can refuse an unmeasurable scene BEFORE a render. A backend "
            f"that fills its whole window says so by returning that window "
            f"(amcd.simulators.base.Simulator)."
        )
    support = getter(validated, t60_s, volume_m3, surface_m2, window_s)
    if not isinstance(support, (int, float)) or isinstance(support, bool) or support <= 0:
        raise ValueError(
            f"simulator {name!r} declared realized_support_s={support!r} for "
            f"t60_s={t60_s!r}; expected a positive number of seconds."
        )
    return float(support)


def simulator_host_scoped_params(config) -> tuple[str, ...]:
    """The active backend's declared host-scoped param names (F-86).

    Registry lookup only — no instantiation and no `Params` validation, because the
    answer is a property of the CLASS and the caller (`render._canonical_meta`) may
    run before or after validation. A backend that declares nothing returns `()`:
    the safe direction, since the failure mode of a missing declaration is a full
    provenance echo, not a silent redaction.
    """
    from ..registry import simulator_registry

    SimClass = simulator_registry.get(config.simulator.name)
    declared = getattr(SimClass, "host_scoped_params", None)
    if not callable(declared):
        return ()
    return tuple(declared())


def simulator_min_separation(config) -> float:
    """The active backend's minimum source-receiver separation, in metres.

    Registry lookup → the backend's own `Params` validation → the classmethod.
    Deliberately does NOT instantiate the simulator: gen-scenes calls this, and
    scene generation must stay runnable on a host with no render environment
    (RD-60). One helper, so no stage outside `simulators/` names a backend or
    touches `build_simulator` itself.
    """
    from ..registry import simulator_registry

    name = config.simulator.name
    SimClass = simulator_registry.get(name)
    validated = SimClass.Params(**config.simulator.params).model_dump()
    return _validate_min_separation_declared(SimClass, name, validated)
