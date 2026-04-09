from __future__ import annotations

from pathlib import Path

from scene_gen.backend import DryRunBackend, GSoundSIRBackend, SimulationBackend
from scene_gen.config_schema import load_dataset_config
from scene_gen.dataset_builder import DatasetBuilder
from scene_gen.pilot import PilotSweepRunner


def load_backend(name: str) -> SimulationBackend:
    normalized = name.strip().lower()
    if normalized == "dry_run":
        return DryRunBackend()
    if normalized == "gsoundsir":
        return GSoundSIRBackend()
    raise ValueError(f"Unknown backend: {name}")


def build_dataset(config_path: str | Path, project_root: str | Path, backend_name: str, subset: str | None = None) -> None:
    config = load_dataset_config(config_path)
    backend = load_backend(backend_name)
    builder = DatasetBuilder(config=config, project_root=project_root, backend=backend)
    builder.generate_scene_specs(overwrite=True)
    if subset is None:
        for subset_name in config.splits:
            builder.render_subset(subset_name=subset_name, strict_qc=False)
    else:
        builder.render_subset(subset_name=subset, strict_qc=False)


def run_pilot(config_path: str | Path, project_root: str | Path, backend_name: str, low_candidates: list[int], high_candidates: list[int]) -> dict:
    config = load_dataset_config(config_path)
    backend = load_backend(backend_name)
    runner = PilotSweepRunner(config=config, project_root=project_root, backend=backend)
    return runner.run(low_ray_candidates=low_candidates, high_ray_candidates=high_candidates)
