from __future__ import annotations

import argparse

from orchestration.dataset_workflow import run_pilot


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pilot ray-budget sweep.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/scenes/procedural_rir_dataset_v1.json")
    parser.add_argument("--backend", default="dry_run", choices=["dry_run", "gsoundsir"])
    parser.add_argument("--low-rays", default="2500,5000,10000,20000")
    parser.add_argument("--high-rays", default="100000,200000")
    args = parser.parse_args()

    results = run_pilot(
        config_path=args.config,
        project_root=args.project_root,
        backend_name=args.backend,
        low_candidates=_parse_csv_ints(args.low_rays),
        high_candidates=_parse_csv_ints(args.high_rays),
    )
    print(results)


if __name__ == "__main__":
    main()
