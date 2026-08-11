from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class SceneSpec:
    scene_id: str
    seed: int
    geometry_family: str
    dims: tuple[float, float, float]
    material_absorption: float
    source_pos: tuple[float, float, float]
    receiver_pos: tuple[float, float, float]
    sim_params: dict = field(default_factory=dict)
    # Which distribution-shift regime this scene was generated for.
    # "id" → in-distribution (train/valid/test_id); shift names → locked to that test split.
    split_regime: str = "id"
    # Labels for each distribution-shift axis (geometry/placement/material). A shift
    # scene differs from the id baseline in exactly one of these — the controlled-shift
    # integrity check (invariant #10) asserts on these labels, not the split_regime tag.
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

#: Scalars upstream reports alongside the arrays, with their dtypes.
PATH_SCALARS: dict[str, type] = {
    "num_paths": int,
    "num_bands": int,
    "total_energy": float,
    "kept_energy_percentage": float,
}


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
    (RD-96/RD-23): the current `paths_{low,high}.parquet` filename convention
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
    #: (N, 3) unit vectors.
    listener_directions: np.ndarray
    source_directions: np.ndarray
    #: (N,) upstream path-type bitmask.
    path_types: np.ndarray
    #: (N,) m/s, per path. Cross-checked against the backend's DECLARED
    #: `speed_of_sound_m_s` at render time (RD-19): gsound's 344 m/s lives in C++
    #: and can only be declared, so this array is the only way that declaration is
    #: falsifiable.
    speeds_of_sound: np.ndarray
    relative_speeds: np.ndarray
    #: (N,) index of the source each path came from.
    source_indices: np.ndarray

    num_paths: int
    num_bands: int
    total_energy: float
    kept_energy_percentage: float

    #: Everything needed to interpret the arrays without the producing config.
    #: Written verbatim into the parquet's key/value metadata; see `describe()`.
    descriptor: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.num_paths)
        for name in PATH_ARRAY_DTYPES:
            arr = np.asarray(getattr(self, name))
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

        scalars = {name: kind(record[name]) for name, kind in PATH_SCALARS.items()}
        return cls(**arrays, **scalars, descriptor=descriptor)

    def describe(self) -> dict:
        """The self-describing block: the descriptor plus the scalars it must agree with.

        Kept as one method so the written file and any in-memory reader see the same
        record, and so a missing key fails in one place rather than per call site.
        """
        return {**self.descriptor, **{name: getattr(self, name) for name in PATH_SCALARS}}


#: Descriptor keys a `PathData` must carry to be interpretable on its own (RD-24).
#: `band_edges_hz` + `band_centres_hz` name the `intensities` columns; `simulator` +
#: `commit_sha` say what produced them; `ray_budget` + `leg` + `realization_index`
#: identify WHICH render this is without relying on the filename (RD-96/RD-23).
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
#:   ambisonic_convention channel ordering + normalization. For GSound-SIR this is
#:                        **"acn_n3d"**, not SN3D — verified in the auralizer
#:                        binding (binding.cpp:18 "normalization constant K(l,m)
#:                        for N3D", :43 "N3D/ACN ordering"). Getting this wrong is
#:                        a per-degree sqrt(2l+1) error; it is invisible today
#:                        because every live scalar metric uses channel 0, where
#:                        N3D and SN3D agree exactly, and becomes load-bearing the
#:                        moment evaluation/spatial.py is filled in (AC-15/RD-25).
#:   rng_seeded           whether the render is reproducible from a seed (RD-23:
#:                        pygsound exposes none, so reproducibility rests on the
#:                        cached artifacts, not on re-render bit-identity)
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
