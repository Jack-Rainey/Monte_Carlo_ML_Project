from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.acoustic_metrics import compute_scalar_acoustic_metrics, metric_errors
from evaluation.signal_metrics import compute_signal_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate low / predicted / high HOA impulse-response triplets."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--prediction-root",
        required=True,
        help="Directory containing prediction_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation CSVs. Defaults to prediction_root/evaluation.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split filter, e.g. valid test_id.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=48000,
        help="Sample rate used for acoustic metric computation.",
    )
    parser.add_argument(
        "--metric-channel-index",
        type=int,
        default=0,
        help="HOA channel used for first-pass scalar acoustic metrics.",
    )
    parser.add_argument(
        "--max-scenes-per-split",
        type=int,
        default=None,
        help="Optional limit for smoke-testing evaluation on each split.",
    )
    return parser.parse_args()


def load_manifest(prediction_root: Path) -> list[dict]:
    manifest_path = prediction_root / "prediction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, list):
        raise ValueError(f"Expected manifest list in {manifest_path}")

    return manifest


def filter_manifest(
    manifest: list[dict],
    *,
    splits: list[str] | None,
    max_scenes_per_split: int | None,
) -> list[dict]:
    if splits is not None:
        split_set = set(splits)
        manifest = [item for item in manifest if item["split"] in split_set]

    if max_scenes_per_split is None:
        return manifest

    counts: dict[str, int] = {}
    filtered: list[dict] = []

    for item in manifest:
        split = item["split"]
        current_count = counts.get(split, 0)

        if current_count < max_scenes_per_split:
            filtered.append(item)
            counts[split] = current_count + 1

    return filtered


def load_hoa_triplet(project_root: Path, item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_path = project_root / item["source_low_path"]
    pred_path = project_root / item["prediction_path"]
    high_path = project_root / item["source_high_path"]

    low = np.load(low_path).astype(np.float32, copy=False)
    pred = np.load(pred_path).astype(np.float32, copy=False)
    high = np.load(high_path).astype(np.float32, copy=False)

    if low.shape != pred.shape or pred.shape != high.shape:
        raise ValueError(
            f"Shape mismatch for scene {item['scene_id']}: "
            f"low={low.shape}, pred={pred.shape}, high={high.shape}"
        )

    return low, pred, high


def summarize_numeric_columns(df: pd.DataFrame, group_col: str = "split") -> pd.DataFrame:
    numeric_cols = [
        col for col in df.columns
        if col != group_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    summary = (
        df.groupby(group_col)[numeric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join([part for part in col if part])
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def add_fraction_improved(summary_df: pd.DataFrame, per_example_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    improvement_cols = [
        col for col in per_example_df.columns
        if col.endswith("_improvement")
        or col.endswith("_improvement_db")
        or col.endswith("_improvement_ratio")
    ]

    for split, group in per_example_df.groupby("split"):
        row = {"split": split}
        for col in improvement_cols:
            if col.endswith("_improvement_ratio"):
                row[f"{col}_fraction_improved"] = float((group[col] > 1.0).mean())
            else:
                row[f"{col}_fraction_improved"] = float((group[col] > 0.0).mean())
        rows.append(row)

    fraction_df = pd.DataFrame(rows)
    return summary_df.merge(fraction_df, on="split", how="left")


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    prediction_root = (project_root / args.prediction_root).resolve()
    output_dir = (
        (prediction_root / "evaluation")
        if args.output_dir is None
        else (project_root / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(prediction_root)
    manifest = filter_manifest(
        manifest,
        splits=args.splits,
        max_scenes_per_split=args.max_scenes_per_split,
    )

    if not manifest:
        raise ValueError("No manifest entries matched the requested filters.")

    signal_rows: list[dict] = []
    acoustic_rows: list[dict] = []

    for item in manifest:
        scene_id = item["scene_id"]
        split = item["split"]

        low, pred, high = load_hoa_triplet(project_root, item)

        signal_row = {
            "scene_id": scene_id,
            "split": split,
            "shape": str(tuple(low.shape)),
            **compute_signal_metrics(low=low, pred=pred, high=high),
        }
        signal_rows.append(signal_row)

        low_acoustic = compute_scalar_acoustic_metrics(
            low,
            sample_rate_hz=args.sample_rate_hz,
            channel_index=args.metric_channel_index,
        )
        pred_acoustic = compute_scalar_acoustic_metrics(
            pred,
            sample_rate_hz=args.sample_rate_hz,
            channel_index=args.metric_channel_index,
        )
        high_acoustic = compute_scalar_acoustic_metrics(
            high,
            sample_rate_hz=args.sample_rate_hz,
            channel_index=args.metric_channel_index,
        )

        acoustic_row = {
            "scene_id": scene_id,
            "split": split,
            "metric_channel_index": args.metric_channel_index,
            **metric_errors(
                low_metrics=low_acoustic,
                pred_metrics=pred_acoustic,
                high_metrics=high_acoustic,
            ),
        }
        acoustic_rows.append(acoustic_row)

    signal_df = pd.DataFrame(signal_rows)
    acoustic_df = pd.DataFrame(acoustic_rows)

    signal_summary_df = add_fraction_improved(
        summarize_numeric_columns(signal_df),
        signal_df,
    )
    acoustic_summary_df = add_fraction_improved(
        summarize_numeric_columns(acoustic_df),
        acoustic_df,
    )

    signal_df.to_csv(output_dir / "signal_metrics_per_example.csv", index=False)
    signal_summary_df.to_csv(output_dir / "signal_metrics_summary.csv", index=False)

    acoustic_df.to_csv(output_dir / "acoustic_metrics_per_example.csv", index=False)
    acoustic_summary_df.to_csv(output_dir / "acoustic_metrics_summary.csv", index=False)

    evaluation_manifest = {
        "prediction_root": str(prediction_root.relative_to(project_root)),
        "output_dir": str(output_dir.relative_to(project_root)),
        "scene_count": len(manifest),
        "splits": sorted({item["split"] for item in manifest}),
        "sample_rate_hz": args.sample_rate_hz,
        "metric_channel_index": args.metric_channel_index,
        "outputs": {
            "signal_metrics_per_example": str((output_dir / "signal_metrics_per_example.csv").relative_to(project_root)),
            "signal_metrics_summary": str((output_dir / "signal_metrics_summary.csv").relative_to(project_root)),
            "acoustic_metrics_per_example": str((output_dir / "acoustic_metrics_per_example.csv").relative_to(project_root)),
            "acoustic_metrics_summary": str((output_dir / "acoustic_metrics_summary.csv").relative_to(project_root)),
        },
    }

    with (output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation_manifest, handle, indent=2)

    print(
        {
            "scene_count": len(manifest),
            "output_dir": str(output_dir),
            "signal_metrics": str(output_dir / "signal_metrics_summary.csv"),
            "acoustic_metrics": str(output_dir / "acoustic_metrics_summary.csv"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()