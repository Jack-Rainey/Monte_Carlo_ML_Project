from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random

from .config_schema import NumericRange, PlacementSamplingConfig
from .geometry import RoomGeometry
from .scene_spec import PlacementSpec, Point3D


@dataclass(frozen=True)
class PlacementResult:
    source: Point3D
    receiver: Point3D
    source_wall_margin_m: float
    receiver_wall_margin_m: float


def _sample_height(rng: Random, range_cfg: NumericRange, floor_margin: float, ceiling_margin: float, room_height: float) -> float:
    z = rng.uniform(range_cfg.min, range_cfg.max)
    if z < floor_margin or z > room_height - ceiling_margin:
        raise RuntimeError(
            "Height sampling configuration is incompatible with floor/ceiling constraints."
        )
    return z


def _sample_point_for_regime(room: RoomGeometry, regime: str, margin: float, rng: Random) -> tuple[float, float]:
    vertices = room.footprint_vertices_xy
    max_attempts = 5000
    for _ in range(max_attempts):
        x, y = room.sample_valid_xy(rng, minimum_wall_margin_m=margin)
        wall_margin = room.wall_margin(x, y)
        if regime == "interior_random" and wall_margin >= margin + 0.15:
            return x, y
        if regime == "near_wall" and margin <= wall_margin <= margin + 0.40:
            return x, y
        if regime == "near_corner":
            nearest_vertex_distance = min(((x - vx) ** 2 + (y - vy) ** 2) ** 0.5 for vx, vy in vertices)
            if nearest_vertex_distance <= margin + 1.00:
                return x, y
    raise RuntimeError(f"Failed to sample regime={regime} for room family={room.family}")


def _distance_ok(distance_m: float, distance_range: NumericRange, pair_regime: str) -> bool:
    if not isfinite(distance_m):
        return False
    if distance_m < distance_range.min or distance_m > distance_range.max:
        return False
    if pair_regime == "close_pair":
        return 1.0 <= distance_m <= 2.0
    if pair_regime == "mid_pair":
        return 2.0 <= distance_m <= 5.0
    if pair_regime == "far_pair":
        return 5.0 <= distance_m <= distance_range.max
    return True


def sample_placement(room: RoomGeometry, regime: str, cfg: PlacementSamplingConfig, rng: Random) -> PlacementSpec:
    if "+" in regime:
        wall_regime, pair_regime = regime.split("+", maxsplit=1)
    else:
        wall_regime, pair_regime = regime, "mid_pair"

    source_z = _sample_height(
        rng,
        cfg.source_height_m,
        cfg.minimum_floor_margin_m,
        cfg.minimum_ceiling_margin_m,
        room.height_m,
    )
    receiver_z = _sample_height(
        rng,
        cfg.receiver_height_m,
        cfg.minimum_floor_margin_m,
        cfg.minimum_ceiling_margin_m,
        room.height_m,
    )

    max_attempts = 5000
    for _ in range(max_attempts):
        source_x, source_y = _sample_point_for_regime(room, wall_regime, cfg.minimum_wall_margin_m, rng)
        receiver_x, receiver_y = _sample_point_for_regime(room, wall_regime, cfg.minimum_wall_margin_m, rng)
        source = Point3D(x=source_x, y=source_y, z=source_z)
        receiver = Point3D(x=receiver_x, y=receiver_y, z=receiver_z)
        distance_m = source.distance_to(receiver)
        if _distance_ok(distance_m, cfg.source_receiver_distance_m, pair_regime):
            source_margin = room.wall_margin(source_x, source_y)
            receiver_margin = room.wall_margin(receiver_x, receiver_y)
            return PlacementSpec(
                regime=regime,
                source=source,
                receiver=receiver,
                source_receiver_distance_m=distance_m,
                source_wall_margin_m=source_margin,
                receiver_wall_margin_m=receiver_margin,
            )
    raise RuntimeError(f"Failed to sample placement for regime={regime}")
