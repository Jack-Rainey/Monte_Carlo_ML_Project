from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from .scene_spec import SceneSpec


class SceneManifestWriter:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, scene_spec: SceneSpec, scene_spec_path: str | Path) -> None:
        row = {
            "scene_id": scene_spec.scene_id,
            "subset": scene_spec.split_metadata.subset,
            "split": scene_spec.split_metadata.split,
            "family": scene_spec.geometry.family,
            "material_regime": scene_spec.materials.regime,
            "placement_regime": scene_spec.placement.regime,
            "scene_spec_path": str(scene_spec_path),
            "simulation": asdict(scene_spec.simulation),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
