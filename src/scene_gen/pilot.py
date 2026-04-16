from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean
import json

from .backend import SimulationBackend
from .config_schema import DatasetConfig
from .dataset_builder import DatasetBuilder


class PilotSweepRunner:
    def __init__(self, config: DatasetConfig, project_root: str | Path, backend: SimulationBackend) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.backend = backend

    def run(self, low_ray_candidates: list[int], high_ray_candidates: list[int], subset_name: str = "train") -> dict:
        if subset_name not in self.config.splits:
            raise KeyError(f"Unknown subset: {subset_name}")

        sweep_results: dict[str, dict] = {}
        for low_rays in low_ray_candidates:
            for high_rays in high_ray_candidates:
                variant_key = f"low_{low_rays}_high_{high_rays}"
                if high_rays <= low_rays:
                    sweep_results[variant_key] = {"skipped": True, "reason": "high_rays <= low_rays"}
                    continue
                cfg_variant = replace(
                    self.config,
                    simulation=replace(
                        self.config.simulation,
                        low_ray_count=low_rays,
                        high_ray_count=high_rays,
                    ),
                    paths=replace(
                        self.config.paths,
                        scene_manifest_path=str(Path(self.config.paths.scene_manifest_path).with_name(f"pilot_{variant_key}.jsonl")),
                        raw_data_root_dir=str(Path(self.config.paths.raw_data_root_dir) / variant_key),
                        metadata_root_dir=str(Path(self.config.paths.metadata_root_dir) / variant_key),
                        qc_root_dir=str(Path(self.config.paths.qc_root_dir) / variant_key),
                        listening_test_root_dir=str(Path(self.config.paths.listening_test_root_dir) / variant_key),
                    ),
                )
                builder = DatasetBuilder(cfg_variant, self.project_root, self.backend)
                builder.generate_scene_specs(overwrite=True)
                records = builder.render_subset(subset_name, strict_qc=False)
                passed = sum(1 for record in records if record["qc"]["passed"])
                low_total_energy = [record["qc"]["metrics"]["low_total_energy"] for record in records]
                high_total_energy = [record["qc"]["metrics"]["high_total_energy"] for record in records]
                sweep_results[variant_key] = {
                    "skipped": False,
                    "record_count": len(records),
                    "passed_count": passed,
                    "failed_count": len(records) - passed,
                    "mean_low_total_energy": mean(low_total_energy) if low_total_energy else 0.0,
                    "mean_high_total_energy": mean(high_total_energy) if high_total_energy else 0.0,
                }

        out_path = self.project_root / self.config.paths.qc_root_dir / "pilot_sweep_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sweep_results, indent=2, sort_keys=True), encoding="utf-8")
        return sweep_results
