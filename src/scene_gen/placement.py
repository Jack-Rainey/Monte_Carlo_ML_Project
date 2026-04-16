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


def _sample_height(
    rng: Random,
    range_cfg: NumericRange,
    floor_margin: float,
    ceiling_margin: float,
    room_height: float,
) -> float:
    z = rng.uniform(range_cfg.min, range_cfg.max)
    if z < floor_margin or z > room_height - ceiling_margin:
        raise RuntimeError(
            "Height sampling configuration is incompatible with floor/ceiling constraints."
        )
    return z


def _nearest_corner_info(room: RoomGeometry, x: float, y: float) -> tuple[int, float]:
    vertices = room.footprint_vertices_xy
    best_idx = -1
    best_dist = float("inf")
    for idx, (vx, vy) in enumerate(vertices):
        dist = ((x - vx) ** 2 + (y - vy) ** 2) ** 0.5
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx, best_dist


def _corner_radius_for_pair(pair_regime: str, margin: float) -> float:
    base_radius = margin + 1.00
    if pair_regime == "close_pair":
        return base_radius
    if pair_regime == "mid_pair":
        return max(base_radius, margin + 1.75)
    if pair_regime == "far_pair":
        return base_radius
    return base_radius


def _sample_point_near_specific_corner(
    room: RoomGeometry,
    corner_idx: int,
    margin: float,
    corner_radius: float,
    rng: Random,
) -> tuple[float, float]:
    max_attempts = 5000

    for _ in range(max_attempts):
        x, y = room.sample_valid_xy(rng, minimum_wall_margin_m=margin)
        nearest_idx, nearest_dist = _nearest_corner_info(room, x, y)
        if nearest_idx == corner_idx and nearest_dist <= corner_radius:
            return x, y

    raise RuntimeError(
        f"Failed to sample point near specific corner idx={corner_idx} "
        f"for room family={room.family}"
    )


def _sample_point_for_regime(
    room: RoomGeometry,
    regime: str,
    margin: float,
    rng: Random,
    *,
    corner_radius: float | None = None,
) -> tuple[float, float]:
    max_attempts = 5000
    for _ in range(max_attempts):
        x, y = room.sample_valid_xy(rng, minimum_wall_margin_m=margin)
        wall_margin = room.wall_margin(x, y)

        if regime == "interior_random" and wall_margin >= margin + 0.15:
            return x, y

        if regime == "near_wall" and margin <= wall_margin <= margin + 0.40:
            return x, y

        if regime == "near_corner":
            _, nearest_vertex_distance = _nearest_corner_info(room, x, y)
            radius = corner_radius if corner_radius is not None else (margin + 1.00)
            if nearest_vertex_distance <= radius:
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


def _corner_distance(vertices: list[tuple[float, float]], i: int, j: int) -> float:
    x1, y1 = vertices[i]
    x2, y2 = vertices[j]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _ordered_corner_pairs_for_pair_regime(
    room: RoomGeometry,
    pair_regime: str,
) -> list[tuple[int, int]]:
    vertices = room.footprint_vertices_xy
    pairs = [(i, j) for i in range(len(vertices)) for j in range(len(vertices))]

    target = {
        "close_pair": 1.5,
        "mid_pair": 3.5,
        "far_pair": 6.0,
    }.get(pair_regime, 3.5)

    pairs.sort(key=lambda ij: abs(_corner_distance(vertices, ij[0], ij[1]) - target))
    return pairs


def sample_placement(
    room: RoomGeometry,
    regime: str,
    cfg: PlacementSamplingConfig,
    rng: Random,
) -> PlacementSpec:
    if "+" in regime:
        wall_regime, pair_regime = regime.split("+", maxsplit=1)
    else:
        wall_regime, pair_regime = regime, "mid_pair"

    max_source_attempts = 250
    max_receiver_attempts_per_source = 250

    for _ in range(max_source_attempts):
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

        corner_radius = _corner_radius_for_pair(pair_regime, cfg.minimum_wall_margin_m)

        if wall_regime == "near_corner":
            corner_pairs = _ordered_corner_pairs_for_pair_regime(room, pair_regime)

            for source_corner_idx, receiver_corner_idx in corner_pairs:
                source_x, source_y = _sample_point_near_specific_corner(
                    room,
                    source_corner_idx,
                    cfg.minimum_wall_margin_m,
                    corner_radius,
                    rng,
                )
                source = Point3D(x=source_x, y=source_y, z=source_z)

                for _ in range(max_receiver_attempts_per_source):
                    receiver_x, receiver_y = _sample_point_near_specific_corner(
                        room,
                        receiver_corner_idx,
                        cfg.minimum_wall_margin_m,
                        corner_radius,
                        rng,
                    )
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
        else:
            source_x, source_y = _sample_point_for_regime(
                room,
                wall_regime,
                cfg.minimum_wall_margin_m,
                rng,
            )
            source = Point3D(x=source_x, y=source_y, z=source_z)

            for _ in range(max_receiver_attempts_per_source):
                receiver_x, receiver_y = _sample_point_for_regime(
                    room,
                    wall_regime,
                    cfg.minimum_wall_margin_m,
                    rng,
                )
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

    raise RuntimeError(
        f"Failed to sample placement for regime={regime}, "
        f"room_family={room.family}, room_height_m={room.height_m}"
    )