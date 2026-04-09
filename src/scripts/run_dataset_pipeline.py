from __future__ import annotations

import argparse

from orchestration.dataset_workflow import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the procedural HOA-RIR dataset.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/scenes/procedural_rir_dataset_v1.json")
    parser.add_argument("--backend", default="dry_run", choices=["dry_run", "gsoundsir"])
    parser.add_argument("--subset", default=None)
    args = parser.parse_args()

    build_dataset(
        config_path=args.config,
        project_root=args.project_root,
        backend_name=args.backend,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
