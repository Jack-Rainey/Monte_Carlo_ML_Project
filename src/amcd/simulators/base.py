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
    """Result of a single render call."""
    ir: np.ndarray  # (C, T) float32, channel-first
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Simulator(Protocol):
    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult: ...
