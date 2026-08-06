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
    """Render backend (design_spec §8 l.243).

    Implementations additionally carry a nested pydantic `Params` schema
    (`extra="forbid"`) describing their own config block — see `build_simulator`
    — and must populate `REQUIRED_PROVENANCE_KEYS` in each `IRResult.meta`.
    Neither is expressible in a `runtime_checkable` Protocol, so both are
    enforced at construction and at render time respectively, not by isinstance.
    """

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult: ...


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
    return SimClass(
        n_channels=n_channels,
        n_samples=n_samples,
        sample_rate=sample_rate,
        **validated,
    )
