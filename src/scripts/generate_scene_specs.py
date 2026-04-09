from __future__ import annotations

import argparse
from pathlib import Path

from scene_gen.backend import DryRunBackend
from scene_gen.config_schema import load_dataset_config
from scene_gen.dataset_builder import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic scene specifications.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/scenes/procedural_rir_dataset_v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_dataset_config(Path(args.project_root) / args.config)
    builder = DatasetBuilder(config=config, project_root=args.project_root, backend=DryRunBackend())
    builder.generate_scene_specs(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
