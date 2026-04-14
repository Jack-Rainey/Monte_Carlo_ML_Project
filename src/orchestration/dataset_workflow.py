from __future__ import annotations

from pathlib import Path
import time

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
        subset_names = list(config.splits.keys())
    else:
        subset_names = [subset]

    total_scenes = sum(config.splits[name].count for name in subset_names)
    rendered_scenes = 0
    start_time = time.perf_counter()

    print()
    print(f"STARTING DATASET BUILD OF: {total_scenes} TOTAL SCENES...")
    print()

    for subset_name in subset_names:
        print()
        print(f"STARTING SUBSET: {subset_name}")

        records = builder.render_subset(subset_name=subset_name, strict_qc=False)
        rendered_scenes += len(records)

        elapsed_s = time.perf_counter() - start_time
        elapsed_str = f"{elapsed_s:.1f}s" if elapsed_s < 60 else f"{elapsed_s / 60:.1f}m"

        print(f"Rendered {rendered_scenes}/{total_scenes} scenes in {elapsed_str} so far...")

    total_elapsed_s = time.perf_counter() - start_time
    total_elapsed_str = (
        f"{total_elapsed_s:.1f}s" if total_elapsed_s < 60 else f"{total_elapsed_s / 60:.1f}m"
    )

    print()
    print(f"Dataset build complete: {rendered_scenes}/{total_scenes} scenes in {total_elapsed_str}")


def run_pilot(config_path: str | Path, project_root: str | Path, backend_name: str, low_candidates: list[int], high_candidates: list[int]) -> dict:
    config = load_dataset_config(config_path)
    backend = load_backend(backend_name)
    runner = PilotSweepRunner(config=config, project_root=project_root, backend=backend)
    return runner.run(low_ray_candidates=low_candidates, high_ray_candidates=high_candidates)