"""render stage: call simulator for each scene, save paired low+high IRs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..config import Config
from ..runtime import Verbosity, emit
from .base import (
    IRResult,
    SceneSpec,
    build_simulator,
    simulator_min_separation,
    validate_path_descriptor,
    validate_provenance,
)

#: Simulator params that are HOST facts, not dataset facts, and so are redacted from
#: the canonical provenance echo below (RD-114). `render_python` is an absolute path to
#: a machine-local interpreter: stamping it would make the same render carry different
#: provenance on the Apple-Silicon and the native-x86_64 host the project must both
#: support, and would leak a user home path into every scene's meta.json. The value
#: still governs nothing about the result — it only says which interpreter ran it —
#: and it moves to `RunContext.host` when RD-20 lands.
_HOST_SCOPED_PARAMS = ("render_python",)


def _preflight_separations(config: Config, scenes: list[SceneSpec]) -> None:
    """Reject the whole batch before rendering any of it, listing every offender.

    The realized-scene backstop to the generator's declared-config check
    (AC-13/F-48/RD-45): scenes on disk may have been generated under a different
    backend, or under an older config, so the floor is re-checked against the
    actual separations here. Failing per-scene mid-loop would abort an emulated
    batch hours in with the sentinel unwritten — the cost this whole check exists
    to avoid — so all offenders are collected and reported at once.
    """
    floor = simulator_min_separation(config)
    if floor <= 0.0:
        return
    offenders = []
    for scene in scenes:
        d = float(
            np.linalg.norm(
                np.asarray(scene.source_pos, dtype=np.float64)
                - np.asarray(scene.receiver_pos, dtype=np.float64)
            )
        )
        if d < floor:
            offenders.append((scene.scene_id, d))
    if offenders:
        lines = "\n".join(f"    {sid}: {d:.4f} m" for sid, d in offenders)
        raise ValueError(
            f"{len(offenders)} of {len(scenes)} scenes have a source-receiver "
            f"separation below simulator {config.simulator.name!r}'s floor of "
            f"{floor} m, so none were rendered:\n{lines}\n"
            f"Raise the `distance_range` lower bound on the offending placement "
            f"regime(s) and regenerate scenes."
        )


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
    params = {
        k: v for k, v in config.simulator.params.items() if k not in _HOST_SCOPED_PARAMS
    }
    return {
        "scene_id": scene.scene_id,
        "simulator": {"name": config.simulator.name, "params": params},
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

    scenes = [SceneSpec.from_json(p) for p in scene_paths]

    # Validate the whole batch before rendering any of it (AC-13/F-48/RD-45).
    _preflight_separations(config, scenes)

    # Drop renders belonging to a previous, larger scene set before adding to this
    # one (F-47, widening F-38). Scene ids are POSITIONAL, so regenerating with
    # fewer scenes leaves high-numbered orphans that a later config silently
    # reuses under a different geometry. gen-scenes gives `scene_*.json` the same
    # treatment; renders are the expensive artifact, so leaving them unpruned was
    # the more costly half of the gap.
    current_ids = {scene.scene_id for scene in scenes}
    pruned = 0
    for stale in renders_dir.iterdir():
        if stale.is_dir() and stale.name not in current_ids:
            shutil.rmtree(stale)
            pruned += 1
    if pruned:
        emit(verbosity, "progress", f"  Pruned {pruned} orphan render dir(s) from {renders_dir}")

    for scene in scenes:
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

        # The retained-path artifact, for backends that export paths (RD-08). Keyed
        # on the FIELD, never on the simulator's type: a backend without paths — the
        # scaffold — writes none and needs no downstream edit, which is the whole
        # point of the scaffolding rule. Written at every save level for the same
        # reason meta.json is (RD-16): under emulation a re-render costs hours, so an
        # artifact this expensive to reproduce is canonical, not observability.
        for leg, result in (("low", low_result), ("high", high_result)):
            if result.paths is None:
                continue
            # The producer knows its ray budget; the STAGE owns the leg's label.
            result.paths.descriptor["leg"] = leg
            validate_path_descriptor(
                result.paths, simulator_name=config.simulator.name, scene_id=scene.scene_id
            )
            result.paths.to_parquet(out_dir / f"paths_{leg}.parquet")

        # Canonical provenance — never verbosity-gated (RD-16). Diagnostic extras
        # (Step 4's per-criterion QC record) attach behind `saves("diagnostics")`
        # when they exist; there are none yet. See docs/verbosity.md.
        (out_dir / "meta.json").write_text(
            json.dumps(_canonical_meta(config, scene, low_result, high_result), indent=2)
        )

    emit(verbosity, "progress", f"  Rendered {len(scene_paths)} scenes → {renders_dir}")
