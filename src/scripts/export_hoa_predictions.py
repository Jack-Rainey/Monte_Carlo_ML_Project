from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from training.hoa_dataset import discover_records, load_dataset_spec, load_stats, validate_record_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export denormalized HOA predictions for a trained model.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--dataset-spec",
        default="configs/scenes/procedural_rir_dataset_real_backend_full_v1.json",
    )
    parser.add_argument(
        "--dataset-config",
        dest="dataset_spec_legacy",
        default=None,
        help="Deprecated alias for --dataset-spec.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test_id", "test_material_shift", "test_placement_shift", "test_geometry_shift"],
    )
    parser.add_argument(
        "--max-scenes-per-split",
        type=int,
        default=None,
        help="Optional cap to keep export small while debugging/listening.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to data/processed/<experiment_name>/<run_name> under the project root.",
    )
    return parser.parse_args()


def resolve_dataset_spec_arg(args: argparse.Namespace) -> str:
    if args.dataset_spec_legacy is not None:
        print(
            "[deprecated] --dataset-config now means dataset-generation spec. Use --dataset-spec instead.",
            flush=True,
        )
        return args.dataset_spec_legacy
    return args.dataset_spec


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / resolve_dataset_spec_arg(args)).resolve()
    run_dir = Path(args.run_dir).resolve()

    best_model_path = run_dir / "best_model.keras"
    last_model_path = run_dir / "final_model.keras"

    if best_model_path.exists():
        model_path = best_model_path
    elif last_model_path.exists():
        model_path = last_model_path
    else:
        raise FileNotFoundError(f"Could not find saved model in {run_dir}")

    stats_path = run_dir / "channel_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find normalization stats: {stats_path}")

    input_stats, target_stats = load_stats(stats_path)
    dataset_spec = load_dataset_spec(project_root, dataset_spec_path)
    model = tf.keras.models.load_model(model_path)

    if args.output_root is None:
        experiment_name = run_dir.parent.name
        run_name = run_dir.name
        output_root = project_root / "data" / "processed" / experiment_name / run_name
    else:
        output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    export_manifest: list[dict] = []

    for split_name in args.splits:
        records = discover_records(project_root, dataset_spec_path, split_name, limit=args.max_scenes_per_split)
        validate_record_shapes(records, dataset_spec)
        split_output_dir = output_root / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            low = np.load(record.low_path).astype(np.float32, copy=False)
            low_norm = input_stats.normalize(low)
            model_input = low_norm.T[None, ...]
            pred_norm_t_c = model.predict(model_input, verbose=0)[0]
            pred_norm_c_t = pred_norm_t_c.T
            pred_high = target_stats.denormalize(pred_norm_c_t).astype(np.float32, copy=False)

            scene_output_dir = split_output_dir / record.scene_id
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            pred_path = scene_output_dir / "pred_high_hoa.npy"
            np.save(pred_path, pred_high)

            export_manifest.append(
                {
                    "scene_id": record.scene_id,
                    "split": split_name,
                    "prediction_path": str(pred_path.relative_to(project_root)),
                    "source_low_path": str(record.low_path.relative_to(project_root)),
                    "source_high_path": str(record.high_path.relative_to(project_root)),
                }
            )

    with (output_root / "prediction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(export_manifest, handle, indent=2)

    print({"prediction_count": len(export_manifest), "output_root": str(output_root)}, flush=True)


if __name__ == "__main__":
    main()