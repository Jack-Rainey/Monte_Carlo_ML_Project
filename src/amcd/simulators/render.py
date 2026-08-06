"""render stage: call simulator for each scene, save paired low+high IRs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..runtime import Verbosity, emit
from .base import IRResult, SceneSpec, build_simulator, validate_provenance


def _canonical_meta(
    config: Config,
    scene: SceneSpec,
    low: IRResult,
    high: IRResult,
) -> dict:
    """The provenance record for one scene's rendered pair.

    Written at EVERY save level (RD-16). This is the only record of how an
    expensive dataset was made — under emulation a re-render can cost hours, so
    gating it behind `diagnostics` meant the default `save=1` run produced a
    dataset nobody could later characterize.

    Simulator-agnostic by construction: the stage contributes what IT knows (the
    resolved config context), and each leg's backend specifics come from the
    simulator's own `IRResult.meta`, validated against REQUIRED_PROVENANCE_KEYS.
    No branch here knows what a gsound is.
    """
    return {
        "scene_id": scene.scene_id,
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "n_channels": config.n_channels,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
        "low": low.meta,
        "high": high.meta,
    }


def run_render(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Selection is purely registry-driven via build_simulator — no stage is aware
    # of any specific simulator, and each backend validates its own params block.
    sim = build_simulator(
        config.simulator.name,
        config.simulator.params,
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

        for leg, result in (("low", low_result), ("high", high_result)):
            validate_provenance(
                result.meta,
                simulator_name=config.simulator.name,
                scene_id=scene.scene_id,
                leg=leg,
            )

        assert low_result.ir.shape == (config.n_channels, config.n_samples), (
            f"Expected IR shape ({config.n_channels}, {config.n_samples}), "
            f"got {low_result.ir.shape}"
        )

        np.save(out_dir / "low.npy", low_result.ir)
        np.save(out_dir / "high.npy", high_result.ir)

        # Canonical provenance — never verbosity-gated (RD-16). Diagnostic extras
        # (Step 4's per-criterion QC record) attach behind `saves("diagnostics")`
        # when they exist; there are none yet. See docs/verbosity.md.
        (out_dir / "meta.json").write_text(
            json.dumps(_canonical_meta(config, scene, low_result, high_result), indent=2)
        )

    emit(verbosity, "progress", f"  Rendered {len(scene_paths)} scenes → {renders_dir}")
