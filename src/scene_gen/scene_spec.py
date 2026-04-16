from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Point3D") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


@dataclass(frozen=True)
class MaterialProfile:
    regime: str
    wall_north: float
    wall_east: float
    wall_south: float
    wall_west: float
    floor: float
    ceiling: float
    descriptors: dict[str, float]


@dataclass(frozen=True)
class GeometrySpec:
    family: str
    height_m: float
    footprint_vertices_xy: list[tuple[float, float]]
    parameters: dict[str, float]
    area_m2: float
    volume_m3: float


@dataclass(frozen=True)
class PlacementSpec:
    regime: str
    source: Point3D
    receiver: Point3D
    source_receiver_distance_m: float
    source_wall_margin_m: float
    receiver_wall_margin_m: float


@dataclass(frozen=True)
class SimulationSpec:
    sample_rate_hz: int
    ir_duration_s: float
    hoa_order: int
    low_ray_count: int
    high_ray_count: int
    retained_path_policy: str
    retained_path_value: int
    dtype: str = "float32"

    @property
    def expected_num_channels(self) -> int:
        return (self.hoa_order + 1) ** 2

    @property
    def expected_num_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.ir_duration_s))


@dataclass(frozen=True)
class SplitMetadata:
    split: str
    subset: str
    split_seed: int
    scene_index_within_subset: int


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    global_seed: int
    geometry: GeometrySpec
    materials: MaterialProfile
    placement: PlacementSpec
    simulation: SimulationSpec
    split_metadata: SplitMetadata
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: Path) -> "SceneSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            scene_id=raw["scene_id"],
            global_seed=raw["global_seed"],
            geometry=GeometrySpec(**raw["geometry"]),
            materials=MaterialProfile(**raw["materials"]),
            placement=PlacementSpec(
                regime=raw["placement"]["regime"],
                source=Point3D(**raw["placement"]["source"]),
                receiver=Point3D(**raw["placement"]["receiver"]),
                source_receiver_distance_m=raw["placement"]["source_receiver_distance_m"],
                source_wall_margin_m=raw["placement"]["source_wall_margin_m"],
                receiver_wall_margin_m=raw["placement"]["receiver_wall_margin_m"],
            ),
            simulation=SimulationSpec(**raw["simulation"]),
            split_metadata=SplitMetadata(**raw["split_metadata"]),
            provenance=raw.get("provenance", {}),
        )
