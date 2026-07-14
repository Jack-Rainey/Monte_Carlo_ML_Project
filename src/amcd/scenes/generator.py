"""gen-scenes stage: deterministic procedural scene specs from config + seed.

Regime-aware and fully config-driven (design_spec §6.1): the id regime is the
controlled baseline; each shift split (config.splits with an `axes` override)
perturbs exactly ONE axis (geometry / placement / material). The generator reads
every sampling range from `config.scenes` — it hardcodes nothing and branches on
the axis *value*, never on a split name, so a new shift axis or geometry family
is a config edit, not a code change.

A generated scene carries its target split name in `split_regime` ("id" for the
id pool, else the shift split name) so data/splits.py can route it with no
name mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..runtime import Verbosity, emit
from ..simulators.base import SceneSpec


def _sample_dims(
    geometry: str,
    rng: np.random.Generator,
    geometry_families: dict[str, dict],
) -> tuple[float, float, float]:
    ranges = geometry_families[geometry]["dims"]  # [[lo,hi], [lo,hi], [lo,hi]]
    return tuple(float(rng.uniform(lo, hi)) for lo, hi in ranges)


def _sample_positions(
    placement: str,
    dims: tuple[float, float, float],
    rng: np.random.Generator,
    placement_regimes: dict[str, dict],
    margin: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lx, ly, lz = dims
    lo = [margin, margin, margin]
    hi = [lx - margin, ly - margin, lz - margin]

    src = tuple(float(v) for v in rng.uniform(lo, hi))

    regime = placement_regimes[placement]
    if regime["type"] == "corner":
        # Receiver biased toward the (min,min,min) corner sub-box.
        corner_frac = regime["corner_frac"]
        corner_hi = [lo[i] + corner_frac * (hi[i] - lo[i]) for i in range(3)]
        rcv = tuple(float(v) for v in rng.uniform(lo, corner_hi))
    else:  # interior: uniform anywhere in the interior
        rcv = tuple(float(v) for v in rng.uniform(lo, hi))
    return src, rcv


def _sample_material(
    material: str,
    rng: np.random.Generator,
    material_regimes: dict[str, dict],
) -> float:
    lo, hi = material_regimes[material]["absorption"]
    return float(rng.uniform(lo, hi))


def run_gen_scenes(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    out_dir = run_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_cfg = config.scenes
    id_axes = dict(scenes_cfg.id_regime)  # {geometry, placement, material}

    # Ordered (split_regime, count, axis_overrides): id pool first, then each
    # shift split. Config validation already guarantees single-axis shifts.
    plan: list[tuple[str, int, dict[str, str]]] = [("id", scenes_cfg.n_id, {})]
    for split_name, spec in config.shift_splits.items():
        plan.append((split_name, spec.count, dict(spec.axes)))

    rng = np.random.default_rng(config.seed("scene_generation"))

    scene_idx = 0
    for split_regime, count, overrides in plan:
        axes = {**id_axes, **overrides}  # controlled: exactly one axis differs from id
        for _ in range(count):
            scene_seed = int(rng.integers(0, 2**31))
            scene_rng = np.random.default_rng(scene_seed)

            dims = _sample_dims(axes["geometry"], scene_rng, scenes_cfg.geometry_families)
            src, rcv = _sample_positions(
                axes["placement"], dims, scene_rng,
                scenes_cfg.placement_regimes, scenes_cfg.margin,
            )
            absorption = _sample_material(axes["material"], scene_rng, scenes_cfg.material_regimes)

            spec = SceneSpec(
                scene_id=f"scene_{scene_idx:04d}",
                seed=scene_seed,
                geometry_family=axes["geometry"],
                dims=dims,
                material_absorption=absorption,
                source_pos=src,
                receiver_pos=rcv,
                sim_params={},
                split_regime=split_regime,
                regime_axes=dict(axes),
            )

            (out_dir / f"{spec.scene_id}.json").write_text(
                json.dumps(spec.to_dict(), indent=2)
            )
            scene_idx += 1

    n_shift = scene_idx - scenes_cfg.n_id
    emit(
        verbosity, "progress",
        f"  Generated {scene_idx} scene specs "
        f"({scenes_cfg.n_id} id + {n_shift} shift) → {out_dir}",
    )
