"""render stage: the paired low/high render of every scene spec.

Writes per scene, under `renders/<scene_id>/`:
  low.npy, high.npy     (n_channels, n_samples) float32, the two ray budgets
  paths_{low,high}.parquet  retained propagation paths, only for backends that
                        export them (RD-08); the scaffold writes none
  meta.json             canonical provenance at EVERY save level (RD-16), incl.
                        `artifact_sha256` over the files above (F-90)

The whole batch is validated before any of it is rendered — separations against
the backend's declared floor — and orphan render dirs from a larger previous
scene set are pruned first. Backend refusals are collected and reported together
rather than aborting mid-batch (F-125).
"""
from __future__ import annotations

import hashlib
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
    simulator_host_scoped_params,
    simulator_min_separation,
    validate_path_descriptor,
    validate_provenance,
)


def _sha256(path: Path) -> str:
    """Digest one written artifact, streamed — an IR pair is ~26 MB per scene."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    artifact_sha256: dict[str, str] | None = None,
) -> dict:
    """The provenance record for one scene's rendered pair.

    Written at EVERY save level (RD-16). This is the only record of how an
    expensive dataset was made — under emulation a re-render can cost hours, so
    gating it behind `diagnostics` meant the default `save=1` run produced a
    dataset nobody could later characterize.

    Simulator-agnostic by construction: the stage contributes what IT knows (the
    resolved config context), and each leg's backend specifics come from the
    simulator's own `IRResult.meta`, validated against REQUIRED_PROVENANCE_KEYS.
    No branch here knows what a gsound is — including which params are host-scoped,
    which the BACKEND declares and this asks for (F-86).

    `artifact_sha256` is the integrity record for the files written beside this one
    (F-90). `rng_seeded: false` puts reproducibility on the cached artifacts rather
    than on re-render bit-identity, so without digests two physically different
    datasets carry byte-identical provenance and a truncated write is undetectable.
    """
    host_scoped = simulator_host_scoped_params(config)
    params = {k: v for k, v in config.simulator.params.items() if k not in host_scoped}
    return {
        "scene_id": scene.scene_id,
        "simulator": {"name": config.simulator.name, "params": params},
        "artifact_sha256": artifact_sha256 or {},
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

    # (scene_id, reason) for every scene the backend refused. Collected rather than
    # raised in place (F-125), for the reason `_preflight_separations` states above:
    # a backend-side refusal at scene 500 of 720 would abort an emulated batch hours
    # in with the sentinel unwritten, and redoing it costs those hours again. Here
    # the whole batch is attempted and every offender is reported at once — and
    # nothing is silently dropped, because a non-empty list is fatal below.
    refused: list[tuple[str, str]] = []

    for scene in scenes:
        out_dir = renders_dir / scene.scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            low_result = sim.render(scene, config.low_ray_budget)
            high_result = sim.render(scene, config.high_ray_budget)
        except ValueError as exc:
            # The backend's own contract failures (a silent leg, a band-count
            # disagreement) — never an unexpected error class, which still aborts.
            refused.append((scene.scene_id, str(exc)))
            emit(verbosity, "progress", f"  REFUSED {scene.scene_id}: {exc}")
            continue

        for leg, result in (("low", low_result), ("high", high_result)):
            validate_provenance(
                result.meta,
                simulator_name=config.simulator.name,
                scene_id=scene.scene_id,
                leg=leg,
            )

        # Both legs, and `raise` not `assert` — `python -O` strips asserts (F-98).
        expected_shape = (config.n_channels, config.n_samples)
        for leg, result in (("low", low_result), ("high", high_result)):
            if result.ir.shape != expected_shape:
                raise ValueError(
                    f"scene {scene.scene_id!r} leg {leg!r}: simulator "
                    f"{config.simulator.name!r} returned an IR of shape "
                    f"{result.ir.shape}, expected {expected_shape} "
                    f"(n_channels, n_samples) from the resolved config."
                )

        # Names this stage WROTE, accumulated as it writes them. Not a listing of
        # out_dir: a directory scan picks up whatever the host put there — on macOS
        # over a non-native filesystem that means AppleDouble `._low.npy` sidecars —
        # which would put a host fact into canonical provenance and make the same
        # render's meta.json differ between the two supported hosts (F-90/RD-114).
        written: list[str] = []

        np.save(out_dir / "low.npy", low_result.ir)
        np.save(out_dir / "high.npy", high_result.ir)
        written += ["low.npy", "high.npy"]

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
            written.append(f"paths_{leg}.parquet")

        # Digest what this scene produced, AFTER the last write and BEFORE meta.json,
        # which is the file that carries them (F-90).
        digests = {name: _sha256(out_dir / name) for name in sorted(written)}

        # Canonical provenance — never verbosity-gated (RD-16). Diagnostic extras
        # (Step 4's per-criterion QC record) attach behind `saves("diagnostics")`
        # when they exist; there are none yet. See docs/verbosity.md.
        (out_dir / "meta.json").write_text(
            json.dumps(
                _canonical_meta(config, scene, low_result, high_result, digests),
                indent=2,
            )
        )

    # Scored vs attempted, always — a refusal must never leave the run looking whole.
    rendered = len(scenes) - len(refused)
    emit(
        verbosity,
        "progress",
        f"  Rendered {rendered} of {len(scenes)} scenes → {renders_dir}",
    )
    if refused:
        lines = "\n".join(f"    {sid}: {reason}" for sid, reason in refused)
        raise ValueError(
            f"the render backend refused {len(refused)} of {len(scenes)} scenes, so "
            f"the dataset is incomplete and the stage is not done:\n{lines}\n"
            f"Every other scene WAS rendered and its artifacts are on disk, so a "
            f"re-run after fixing these costs only the refused scenes."
        )
