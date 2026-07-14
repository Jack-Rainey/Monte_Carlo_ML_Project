"""render stage: call simulator for each scene, save paired low+high IRs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..registry import simulator_registry
from ..runtime import Verbosity, emit
from .base import SceneSpec


def run_render(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Simulator registrations are triggered by importing the simulators package
    # (amcd/simulators/__init__.py). Selection is purely registry-driven — no
    # stage is aware of any specific simulator.
    SimClass = simulator_registry.get(config.simulator)
    sim = SimClass(
        n_channels=config.n_channels,
        n_samples=config.n_samples,
        sample_rate=config.sample_rate,
    )

    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scene specs found in {scenes_dir}. Run gen-scenes first.")

    for scene_path in scene_paths:
        scene = SceneSpec.from_json(scene_path)
        out_dir = renders_dir / scene.scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        low_result = sim.render(scene, config.low_ray_budget)
        high_result = sim.render(scene, config.high_ray_budget)

        assert low_result.ir.shape == (config.n_channels, config.n_samples), (
            f"Expected IR shape ({config.n_channels}, {config.n_samples}), "
            f"got {low_result.ir.shape}"
        )

        np.save(out_dir / "low.npy", low_result.ir)
        np.save(out_dir / "high.npy", high_result.ir)

        # Per-render QC record (design_spec §6): diagnostic only — nothing
        # downstream reads it, so it may sit behind the save gate (F-23).
        if verbosity.saves("diagnostics"):
            meta = {
                "scene_id": scene.scene_id,
                "low": low_result.meta,
                "high": high_result.meta,
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    emit(verbosity, "progress", f"  Rendered {len(scene_paths)} scenes → {renders_dir}")
