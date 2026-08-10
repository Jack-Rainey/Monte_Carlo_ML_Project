from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class IRResult:
    """Result of a single render call.

    `meta` is the simulator's own provenance record for this leg. It is written
    into the render stage's CANONICAL meta.json at every save level, so it must
    carry at least `REQUIRED_PROVENANCE_KEYS` — see below.
    """
    ir: np.ndarray  # (C, T) float32, channel-first
    meta: dict = field(default_factory=dict)


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
