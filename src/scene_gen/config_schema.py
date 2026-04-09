from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class NumericRange:
    min: float
    max: float

    def validate(self, name: str) -> None:
        if self.max < self.min:
            raise ValueError(f"Invalid range for {name}: max < min")


@dataclass(frozen=True)
class FamilyWeight:
    name: str
    weight: float


@dataclass(frozen=True)
class PlacementRegimeWeight:
    name: str
    weight: float


@dataclass(frozen=True)
class MaterialRegimeWeight:
    name: str
    weight: float


@dataclass(frozen=True)
class PathsConfig:
    scene_specs_dir: str
    scene_manifest_path: str
    metadata_root_dir: str
    raw_data_root_dir: str
    split_root_dir: str
    qc_root_dir: str
    listening_test_root_dir: str


@dataclass(frozen=True)
class GeometrySamplingConfig:
    shoebox_length_m: NumericRange
    shoebox_width_m: NumericRange
    shoebox_height_m: NumericRange
    corridor_length_m: NumericRange
    corridor_width_m: NumericRange
    corridor_height_m: NumericRange
    l_room_outer_width_m: NumericRange
    l_room_outer_length_m: NumericRange
    l_room_arm_width_m: NumericRange
    l_room_arm_length_m: NumericRange
    l_room_height_m: NumericRange
    alcove_main_width_m: NumericRange
    alcove_main_length_m: NumericRange
    alcove_depth_m: NumericRange
    alcove_width_m: NumericRange
    alcove_height_m: NumericRange


@dataclass(frozen=True)
class PlacementSamplingConfig:
    source_height_m: NumericRange
    receiver_height_m: NumericRange
    minimum_wall_margin_m: float
    minimum_floor_margin_m: float
    minimum_ceiling_margin_m: float
    source_receiver_distance_m: NumericRange


@dataclass(frozen=True)
class SimulationConfig:
    sample_rate_hz: int
    ir_duration_s: float
    hoa_order: int
    low_ray_count: int
    high_ray_count: int
    retained_path_policy: str
    retained_path_value: int
    dtype: str


@dataclass(frozen=True)
class QcConfig:
    max_onset_error_ms: float
    min_total_energy: float
    require_non_empty_paths: bool
    max_retained_paths_file_size_mb: float


@dataclass(frozen=True)
class SplitSubsetConfig:
    count: int
    seed: int
    families: list[FamilyWeight]
    placement_regimes: list[PlacementRegimeWeight]
    material_regimes: list[MaterialRegimeWeight]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    base_seed: int
    paths: PathsConfig
    geometry_sampling: GeometrySamplingConfig
    placement_sampling: PlacementSamplingConfig
    simulation: SimulationConfig
    qc: QcConfig
    splits: dict[str, SplitSubsetConfig]

    def validate(self) -> None:
        if self.simulation.hoa_order < 0:
            raise ValueError("hoa_order must be non-negative")
        if self.simulation.low_ray_count <= 0 or self.simulation.high_ray_count <= 0:
            raise ValueError("Ray counts must be positive")
        if self.simulation.high_ray_count <= self.simulation.low_ray_count:
            raise ValueError("high_ray_count must exceed low_ray_count")
        if self.simulation.retained_path_value <= 0:
            raise ValueError("retained_path_value must be positive")
        if self.qc.max_onset_error_ms <= 0:
            raise ValueError("max_onset_error_ms must be positive")
        if self.qc.max_retained_paths_file_size_mb <= 0:
            raise ValueError("max_retained_paths_file_size_mb must be positive")
        for split_name, split_cfg in self.splits.items():
            if split_cfg.count <= 0:
                raise ValueError(f"Split {split_name} must contain at least one example")
            for label, items in {
                "families": split_cfg.families,
                "placement_regimes": split_cfg.placement_regimes,
                "material_regimes": split_cfg.material_regimes,
            }.items():
                if not items:
                    raise ValueError(f"Split {split_name} is missing {label}")
                if sum(item.weight for item in items) <= 0:
                    raise ValueError(f"Split {split_name} has non-positive total weight for {label}")


def _range_from_mapping(raw: dict[str, Any], name: str) -> NumericRange:
    value = NumericRange(min=float(raw["min"]), max=float(raw["max"]))
    value.validate(name)
    return value


def _weighted_list(raw: list[dict[str, Any]], cls: type[FamilyWeight | PlacementRegimeWeight | MaterialRegimeWeight]):
    return [cls(name=item["name"], weight=float(item["weight"])) for item in raw]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    geometry = raw["geometry_sampling"]
    placement = raw["placement_sampling"]
    simulation = raw["simulation"]
    qc = raw["qc"]

    cfg = DatasetConfig(
        dataset_name=raw["dataset_name"],
        base_seed=int(raw["base_seed"]),
        paths=PathsConfig(**raw["paths"]),
        geometry_sampling=GeometrySamplingConfig(
            shoebox_length_m=_range_from_mapping(geometry["shoebox_length_m"], "shoebox_length_m"),
            shoebox_width_m=_range_from_mapping(geometry["shoebox_width_m"], "shoebox_width_m"),
            shoebox_height_m=_range_from_mapping(geometry["shoebox_height_m"], "shoebox_height_m"),
            corridor_length_m=_range_from_mapping(geometry["corridor_length_m"], "corridor_length_m"),
            corridor_width_m=_range_from_mapping(geometry["corridor_width_m"], "corridor_width_m"),
            corridor_height_m=_range_from_mapping(geometry["corridor_height_m"], "corridor_height_m"),
            l_room_outer_width_m=_range_from_mapping(geometry["l_room_outer_width_m"], "l_room_outer_width_m"),
            l_room_outer_length_m=_range_from_mapping(geometry["l_room_outer_length_m"], "l_room_outer_length_m"),
            l_room_arm_width_m=_range_from_mapping(geometry["l_room_arm_width_m"], "l_room_arm_width_m"),
            l_room_arm_length_m=_range_from_mapping(geometry["l_room_arm_length_m"], "l_room_arm_length_m"),
            l_room_height_m=_range_from_mapping(geometry["l_room_height_m"], "l_room_height_m"),
            alcove_main_width_m=_range_from_mapping(geometry["alcove_main_width_m"], "alcove_main_width_m"),
            alcove_main_length_m=_range_from_mapping(geometry["alcove_main_length_m"], "alcove_main_length_m"),
            alcove_depth_m=_range_from_mapping(geometry["alcove_depth_m"], "alcove_depth_m"),
            alcove_width_m=_range_from_mapping(geometry["alcove_width_m"], "alcove_width_m"),
            alcove_height_m=_range_from_mapping(geometry["alcove_height_m"], "alcove_height_m"),
        ),
        placement_sampling=PlacementSamplingConfig(
            source_height_m=_range_from_mapping(placement["source_height_m"], "source_height_m"),
            receiver_height_m=_range_from_mapping(placement["receiver_height_m"], "receiver_height_m"),
            minimum_wall_margin_m=float(placement["minimum_wall_margin_m"]),
            minimum_floor_margin_m=float(placement["minimum_floor_margin_m"]),
            minimum_ceiling_margin_m=float(placement["minimum_ceiling_margin_m"]),
            source_receiver_distance_m=_range_from_mapping(
                placement["source_receiver_distance_m"],
                "source_receiver_distance_m",
            ),
        ),
        simulation=SimulationConfig(
            sample_rate_hz=int(simulation["sample_rate_hz"]),
            ir_duration_s=float(simulation["ir_duration_s"]),
            hoa_order=int(simulation["hoa_order"]),
            low_ray_count=int(simulation["low_ray_count"]),
            high_ray_count=int(simulation["high_ray_count"]),
            retained_path_policy=str(simulation["retained_path_policy"]),
            retained_path_value=int(simulation["retained_path_value"]),
            dtype=str(simulation.get("dtype", "float32")),
        ),
        qc=QcConfig(
            max_onset_error_ms=float(qc["max_onset_error_ms"]),
            min_total_energy=float(qc["min_total_energy"]),
            require_non_empty_paths=bool(qc["require_non_empty_paths"]),
            max_retained_paths_file_size_mb=float(qc["max_retained_paths_file_size_mb"]),
        ),
        splits={
            split_name: SplitSubsetConfig(
                count=int(split_raw["count"]),
                seed=int(split_raw["seed"]),
                families=_weighted_list(split_raw["families"], FamilyWeight),
                placement_regimes=_weighted_list(split_raw["placement_regimes"], PlacementRegimeWeight),
                material_regimes=_weighted_list(split_raw["material_regimes"], MaterialRegimeWeight),
            )
            for split_name, split_raw in raw["splits"].items()
        },
    )
    cfg.validate()
    return cfg
