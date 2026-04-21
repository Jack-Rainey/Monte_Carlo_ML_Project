from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from training.hoa_dataset import load_dataset_spec, load_stats
from training.hoa_paths_dataset import discover_records_with_paths
from training.path_features import PathFeatureStats, load_path_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export denormalized HOA predictions for the path-conditioned model.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-spec", default="configs/scenes/procedural_rir_dataset_real_backend_full_v1.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test_id"])
    parser.add_argument("--max-scenes-per-split", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_path_stats_npz(stats_path: Path) -> PathFeatureStats:
    with np.load(stats_path) as stats:
        return PathFeatureStats(mean=stats["path_mean"], std=stats["path_std"])


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / args.dataset_spec).resolve()
    run_dir = Path(args.run_dir).resolve()

    model_path = run_dir / "best_model.keras"
    if not model_path.exists():
        model_path = run_dir / "final_model.keras"
    stats_path = run_dir / "channel_stats.npz"
    input_stats, target_stats = load_stats(stats_path)
    path_stats = load_path_stats_npz(stats_path)

    with (run_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    path_top_k = int(run_metadata["path_top_k"])

    model = tf.keras.models.load_model(model_path, compile=False)
    dataset_spec = load_dataset_spec(project_root, dataset_spec_path)

    if args.output_root is None:
        output_root = project_root / "data" / "processed" / run_dir.parent.name / run_dir.name
    else:
        output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    export_manifest = []
    for split_name in args.splits:
        records = discover_records_with_paths(project_root, dataset_spec_path, split_name, limit=args.max_scenes_per_split)
        split_output_dir = output_root / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            low = np.load(record.low_path).astype(np.float32, copy=False)
            paths = load_path_feature_matrix(record.paths_csv_path, top_k=path_top_k).astype(np.float32, copy=False)
            pred_norm = model.predict({"low_hoa_input": input_stats.normalize(low).T[None, ...], "path_features_input": path_stats.normalize(paths)[None, ...]}, verbose=0)[0]
            pred_high = target_stats.denormalize(pred_norm.T).astype(np.float32, copy=False)
            scene_output_dir = split_output_dir / record.scene_id
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            pred_path = scene_output_dir / "pred_high_hoa.npy"
            np.save(pred_path, pred_high)
            export_manifest.append({"scene_id": record.scene_id, "split": split_name, "prediction_path": str(pred_path.relative_to(project_root)), "source_low_path": str(record.low_path.relative_to(project_root)), "source_high_path": str(record.high_path.relative_to(project_root)), "source_paths_csv_path": str(record.paths_csv_path.relative_to(project_root))})

    with (output_root / "prediction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(export_manifest, handle, indent=2)
    print({"prediction_count": len(export_manifest), "output_root": str(output_root)}, flush=True)


if __name__ == "__main__":
    main()
