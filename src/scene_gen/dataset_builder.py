from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random
import json

import numpy as np

from .backend import RenderArtifacts, SimulationBackend
from .config_schema import DatasetConfig
from .geometry import sample_room_geometry
from .manifest import SceneManifestWriter
from .materials import sample_material_profile
from .placement import sample_placement
from .qc import run_qc
from .scene_spec import SceneSpec, SimulationSpec
from .splits import choose_scene_factors, iter_split_metadata


class DatasetBuilder:
    def __init__(self, config: DatasetConfig, project_root: str | Path, backend: SimulationBackend) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.backend = backend

        self.scene_specs_root = self.project_root / self.config.paths.scene_specs_dir
        self.scene_manifest_path = self.project_root / self.config.paths.scene_manifest_path
        self.metadata_root = self.project_root / self.config.paths.metadata_root_dir
        self.raw_root = self.project_root / self.config.paths.raw_data_root_dir
        self.split_root = self.project_root / self.config.paths.split_root_dir
        self.qc_root = self.project_root / self.config.paths.qc_root_dir
        self.listening_root = self.project_root / self.config.paths.listening_test_root_dir

        for path in (
            self.scene_specs_root,
            self.metadata_root,
            self.raw_root,
            self.split_root,
            self.qc_root,
            self.listening_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.manifest_writer = SceneManifestWriter(self.scene_manifest_path)

    def generate_scene_specs(self, overwrite: bool = False) -> list[SceneSpec]:
        all_specs: list[SceneSpec] = []
        if overwrite and self.scene_manifest_path.exists():
            self.scene_manifest_path.unlink()
        for subset_name, split_cfg in self.config.splits.items():
            subset_dir = self.scene_specs_root / subset_name
            subset_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for old_file in subset_dir.glob("*.json"):
                    old_file.unlink()
            split_index_path = self.split_root / f"{subset_name}.txt"
            scene_ids: list[str] = []

            for meta in iter_split_metadata(self.config, subset_name):
                rng = Random(meta.split_seed)
                family, placement_regime, material_regime = choose_scene_factors(split_cfg, rng)
                geometry = sample_room_geometry(family, self.config.geometry_sampling, rng)
                materials = sample_material_profile(material_regime, rng)
                placement = sample_placement(geometry, placement_regime, self.config.placement_sampling, rng)
                scene_id = f"{subset_name}_{meta.scene_index_within_subset:05d}"
                spec = SceneSpec(
                    scene_id=scene_id,
                    global_seed=meta.split_seed,
                    geometry=geometry.to_geometry_spec(),
                    materials=materials,
                    placement=placement,
                    simulation=SimulationSpec(**asdict(self.config.simulation)),
                    split_metadata=meta,
                    provenance={
                        "dataset_name": self.config.dataset_name,
                        "config_path": str(self.config.paths.scene_manifest_path),
                    },
                )
                scene_spec_path = subset_dir / f"{scene_id}.json"
                if scene_spec_path.exists() and not overwrite:
                    raise FileExistsError(f"Scene spec already exists: {scene_spec_path}")
                spec.write_json(scene_spec_path)
                self.manifest_writer.append(spec, scene_spec_path)
                scene_ids.append(scene_id)
                all_specs.append(spec)

            split_index_path.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
        return all_specs

    def render_subset(self, subset_name: str, strict_qc: bool = True) -> list[dict]:
        subset_dir = self.scene_specs_root / subset_name
        records: list[dict] = []
        for spec_path in sorted(subset_dir.glob("*.json")):
            scene = SceneSpec.from_json(spec_path)
            raw_out_dir = self.raw_root / subset_name / scene.scene_id
            metadata_out_dir = self.metadata_root / subset_name / scene.scene_id
            qc_out_dir = self.qc_root / subset_name
            listening_out_dir = self.listening_root / subset_name
            metadata_out_dir.mkdir(parents=True, exist_ok=True)
            qc_out_dir.mkdir(parents=True, exist_ok=True)
            listening_out_dir.mkdir(parents=True, exist_ok=True)

            artifacts = self.backend.render(scene, raw_out_dir)
            low = np.load(artifacts.low_hoa_path)
            high = np.load(artifacts.high_hoa_path)
            qc_result = run_qc(scene, low, high, artifacts.paths_path, self.config.qc)
            qc_result.write_json(qc_out_dir / f"{scene.scene_id}_qc.json")

            render_record = {
                "scene_id": scene.scene_id,
                "subset": scene.split_metadata.subset,
                "artifacts": {
                    "low_hoa_path": str(artifacts.low_hoa_path.relative_to(self.project_root)),
                    "high_hoa_path": str(artifacts.high_hoa_path.relative_to(self.project_root)),
                    "paths_path": str(artifacts.paths_path.relative_to(self.project_root)),
                    "preview_wav_path": str(artifacts.preview_wav_path.relative_to(self.project_root)),
                },
                "qc": {
                    "passed": qc_result.passed,
                    "issues": qc_result.issues,
                    "metrics": qc_result.metrics,
                },
            }
            with (metadata_out_dir / "render_record.json").open("w", encoding="utf-8") as handle:
                json.dump(render_record, handle, indent=2, sort_keys=True)

            preview_target = listening_out_dir / f"{scene.scene_id}_preview.wav"
            if preview_target.resolve() != artifacts.preview_wav_path.resolve():
                preview_target.write_bytes(artifacts.preview_wav_path.read_bytes())

            if strict_qc and not qc_result.passed:
                raise RuntimeError(
                    f"QC failed for {scene.scene_id}: {'; '.join(qc_result.issues)}"
                )
            records.append(render_record)
        return records
