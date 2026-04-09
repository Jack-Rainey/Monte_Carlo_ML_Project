from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from random import Random

from .config_schema import GeometrySamplingConfig
from .scene_spec import GeometrySpec


@dataclass(frozen=True)
class RoomGeometry:
    family: str
    height_m: float
    footprint_vertices_xy: list[tuple[float, float]]
    parameters: dict[str, float]

    @property
    def area_m2(self) -> float:
        area_twice = 0.0
        vertices = self.footprint_vertices_xy
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            area_twice += x1 * y2 - x2 * y1
        return abs(area_twice) * 0.5

    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.height_m

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [v[0] for v in self.footprint_vertices_xy]
        ys = [v[1] for v in self.footprint_vertices_xy]
        return min(xs), min(ys), max(xs), max(ys)

    def contains_xy(self, x: float, y: float) -> bool:
        inside = False
        vertices = self.footprint_vertices_xy
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def wall_margin(self, x: float, y: float) -> float:
        if not self.contains_xy(x, y):
            return -1.0
        best = float("inf")
        vertices = self.footprint_vertices_xy
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            best = min(best, _point_to_segment_distance(x, y, x1, y1, x2, y2))
        return best

    def sample_valid_xy(self, rng: Random, minimum_wall_margin_m: float, max_attempts: int = 5000) -> tuple[float, float]:
        x_min, y_min, x_max, y_max = self.bounds
        for _ in range(max_attempts):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if self.contains_xy(x, y) and self.wall_margin(x, y) >= minimum_wall_margin_m:
                return x, y
        raise RuntimeError(
            f"Failed to sample a valid position in {self.family} after {max_attempts} attempts."
        )

    def to_geometry_spec(self) -> GeometrySpec:
        return GeometrySpec(
            family=self.family,
            height_m=self.height_m,
            footprint_vertices_xy=self.footprint_vertices_xy,
            parameters=self.parameters,
            area_m2=self.area_m2,
            volume_m3=self.volume_m3,
        )


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return hypot(px - cx, py - cy)


def _uniform(rng: Random, value_range) -> float:
    return rng.uniform(value_range.min, value_range.max)


def sample_room_geometry(family: str, cfg: GeometrySamplingConfig, rng: Random) -> RoomGeometry:
    if family == "shoebox":
        width = _uniform(rng, cfg.shoebox_width_m)
        length = max(width, _uniform(rng, cfg.shoebox_length_m))
        height = _uniform(rng, cfg.shoebox_height_m)
        vertices = [(0.0, 0.0), (width, 0.0), (width, length), (0.0, length)]
        parameters = {"width_m": width, "length_m": length}
        return RoomGeometry(family=family, height_m=height, footprint_vertices_xy=vertices, parameters=parameters)

    if family == "corridor":
        width = _uniform(rng, cfg.corridor_width_m)
        length = max(3.0 * width, _uniform(rng, cfg.corridor_length_m))
        height = _uniform(rng, cfg.corridor_height_m)
        vertices = [(0.0, 0.0), (width, 0.0), (width, length), (0.0, length)]
        parameters = {"width_m": width, "length_m": length}
        return RoomGeometry(family=family, height_m=height, footprint_vertices_xy=vertices, parameters=parameters)

    if family == "l_room":
        outer_width = _uniform(rng, cfg.l_room_outer_width_m)
        outer_length = _uniform(rng, cfg.l_room_outer_length_m)
        arm_width = min(_uniform(rng, cfg.l_room_arm_width_m), outer_width - 0.5)
        arm_length = min(_uniform(rng, cfg.l_room_arm_length_m), outer_length - 0.5)
        height = _uniform(rng, cfg.l_room_height_m)
        vertices = [
            (0.0, 0.0),
            (outer_width, 0.0),
            (outer_width, arm_length),
            (arm_width, arm_length),
            (arm_width, outer_length),
            (0.0, outer_length),
        ]
        parameters = {
            "outer_width_m": outer_width,
            "outer_length_m": outer_length,
            "arm_width_m": arm_width,
            "arm_length_m": arm_length,
        }
        return RoomGeometry(family=family, height_m=height, footprint_vertices_xy=vertices, parameters=parameters)

    if family == "alcove":
        main_width = _uniform(rng, cfg.alcove_main_width_m)
        main_length = _uniform(rng, cfg.alcove_main_length_m)
        alcove_depth = _uniform(rng, cfg.alcove_depth_m)
        alcove_width = min(_uniform(rng, cfg.alcove_width_m), main_length - 1.0)
        height = _uniform(rng, cfg.alcove_height_m)
        y0 = rng.uniform(0.5, max(0.6, main_length - alcove_width - 0.5))
        y1 = y0 + alcove_width
        vertices = [
            (0.0, 0.0),
            (main_width, 0.0),
            (main_width, y0),
            (main_width + alcove_depth, y0),
            (main_width + alcove_depth, y1),
            (main_width, y1),
            (main_width, main_length),
            (0.0, main_length),
        ]
        parameters = {
            "main_width_m": main_width,
            "main_length_m": main_length,
            "alcove_depth_m": alcove_depth,
            "alcove_width_m": alcove_width,
            "alcove_y0_m": y0,
            "alcove_y1_m": y1,
        }
        return RoomGeometry(family=family, height_m=height, footprint_vertices_xy=vertices, parameters=parameters)

    raise ValueError(f"Unsupported family: {family}")
