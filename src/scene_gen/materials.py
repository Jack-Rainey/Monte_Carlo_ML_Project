from __future__ import annotations

from random import Random
from statistics import mean, pstdev

from .scene_spec import MaterialProfile


_REGIME_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "live": {
        "wall_north": (0.05, 0.20),
        "wall_east": (0.05, 0.20),
        "wall_south": (0.05, 0.20),
        "wall_west": (0.05, 0.20),
        "floor": (0.02, 0.15),
        "ceiling": (0.05, 0.20),
    },
    "balanced": {
        "wall_north": (0.20, 0.45),
        "wall_east": (0.20, 0.45),
        "wall_south": (0.20, 0.45),
        "wall_west": (0.20, 0.45),
        "floor": (0.15, 0.50),
        "ceiling": (0.20, 0.50),
    },
    "dry": {
        "wall_north": (0.45, 0.75),
        "wall_east": (0.45, 0.75),
        "wall_south": (0.45, 0.75),
        "wall_west": (0.45, 0.75),
        "floor": (0.40, 0.90),
        "ceiling": (0.45, 0.90),
    },
    "reflective_floor": {
        "wall_north": (0.20, 0.55),
        "wall_east": (0.20, 0.55),
        "wall_south": (0.20, 0.55),
        "wall_west": (0.20, 0.55),
        "floor": (0.02, 0.12),
        "ceiling": (0.30, 0.70),
    },
    "ceiling_absorptive": {
        "wall_north": (0.15, 0.45),
        "wall_east": (0.15, 0.45),
        "wall_south": (0.15, 0.45),
        "wall_west": (0.15, 0.45),
        "floor": (0.10, 0.30),
        "ceiling": (0.50, 0.90),
    },
    "asymmetric_walls": {
        "wall_north": (0.05, 0.20),
        "wall_east": (0.45, 0.75),
        "wall_south": (0.10, 0.35),
        "wall_west": (0.40, 0.70),
        "floor": (0.15, 0.50),
        "ceiling": (0.20, 0.50),
    },
}


def _u(rng: Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def sample_material_profile(regime: str, rng: Random) -> MaterialProfile:
    if regime not in _REGIME_RANGES:
        raise ValueError(f"Unknown material regime: {regime}")
    ranges = _REGIME_RANGES[regime]
    values = {name: _u(rng, lo, hi) for name, (lo, hi) in ranges.items()}
    walls = [values[name] for name in ("wall_north", "wall_east", "wall_south", "wall_west")]
    descriptors = {
        "mean_absorption": mean(values.values()),
        "absorption_std": pstdev(values.values()),
        "wall_mean_absorption": mean(walls),
        "floor_ceiling_contrast": abs(values["floor"] - values["ceiling"]),
        "wall_asymmetry_score": max(walls) - min(walls),
    }
    return MaterialProfile(regime=regime, descriptors=descriptors, **values)
