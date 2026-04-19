from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose energy distribution of low / predicted / high HOA IRs."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--prediction-root",
        required=True,
        help="Directory containing prediction_manifest.json from export_hoa_predictions.py",
    )
    parser.add_argument(
        "--early-ms",
        type=float,
        default=50.0,
        help="Early window length in milliseconds, starting at detected onset.",
    )
    parser.add_argument(
        "--late-ms",
        type=float,
        default=750.0,
        help="Late tail window length in milliseconds, measured from the end.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=48000,
        help="Sample rate used by the dataset.",
    )
    parser.add_argument(
        "--onset-channel-index",
        type=int,
        default=0,
        help="Channel used for onset detection.",
    )
    return parser.parse_args()


def detect_onset_index(x_ch_t: np.ndarray, channel_index: int = 0, threshold_ratio: float = 0.1) -> int:
    channel = np.abs(x_ch_t[channel_index])
    peak = float(np.max(channel))
    if peak <= 0.0:
        return 0
    threshold = threshold_ratio * peak
    hits = np.flatnonzero(channel >= threshold)
    if hits.size == 0:
        return 0
    return int(hits[0])


def summarize_array(
    x_ch_t: np.ndarray,
    *,
    onset_idx: int,
    early_n: int,
    late_n: int,
) -> dict[str, float]:
    abs_x = np.abs(x_ch_t, dtype=np.float32)

    early_stop = min(onset_idx + early_n, x_ch_t.shape[1])
    late_start = max(0, x_ch_t.shape[1] - late_n)

    return {
        "peak_abs": float(np.max(abs_x)),
        "rms": float(np.sqrt(np.mean(np.square(x_ch_t), dtype=np.float64))),
        "l1": float(np.sum(abs_x, dtype=np.float64)),
        "l2_sq": float(np.sum(np.square(x_ch_t), dtype=np.float64)),
        "early_l2_sq": float(np.sum(np.square(x_ch_t[:, onset_idx:early_stop]), dtype=np.float64)),
        "late_l2_sq": float(np.sum(np.square(x_ch_t[:, late_start:]), dtype=np.float64)),
    }


def ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-20 else float("nan")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    prediction_root = Path(args.prediction_root).resolve()

    manifest_path = prediction_root / "prediction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    early_n = int(round(args.early_ms * 1e-3 * args.sample_rate_hz))
    late_n = int(round(args.late_ms * 1e-3 * args.sample_rate_hz))

    per_scene = []
    aggregate = {
        "scene_count": 0,
        "avg_pred_over_high_peak_abs": 0.0,
        "avg_low_over_high_peak_abs": 0.0,
        "avg_pred_over_high_rms": 0.0,
        "avg_low_over_high_rms": 0.0,
        "avg_pred_over_high_early_l2_sq": 0.0,
        "avg_low_over_high_early_l2_sq": 0.0,
        "avg_pred_over_high_late_l2_sq": 0.0,
        "avg_low_over_high_late_l2_sq": 0.0,
    }

    for item in manifest:
        scene_id = item["scene_id"]
        split_name = item["split"]

        low = np.load(project_root / item["source_low_path"]).astype(np.float32, copy=False)
        high = np.load(project_root / item["source_high_path"]).astype(np.float32, copy=False)
        pred = np.load(project_root / item["prediction_path"]).astype(np.float32, copy=False)

        onset_idx = detect_onset_index(high, channel_index=args.onset_channel_index)
        low_stats = summarize_array(low, onset_idx=onset_idx, early_n=early_n, late_n=late_n)
        pred_stats = summarize_array(pred, onset_idx=onset_idx, early_n=early_n, late_n=late_n)
        high_stats = summarize_array(high, onset_idx=onset_idx, early_n=early_n, late_n=late_n)

        scene_summary = {
            "scene_id": scene_id,
            "split": split_name,
            "onset_idx": onset_idx,
            "low": low_stats,
            "predicted": pred_stats,
            "high": high_stats,
            "ratios": {
                "pred_over_high_peak_abs": ratio(pred_stats["peak_abs"], high_stats["peak_abs"]),
                "low_over_high_peak_abs": ratio(low_stats["peak_abs"], high_stats["peak_abs"]),
                "pred_over_high_rms": ratio(pred_stats["rms"], high_stats["rms"]),
                "low_over_high_rms": ratio(low_stats["rms"], high_stats["rms"]),
                "pred_over_high_early_l2_sq": ratio(pred_stats["early_l2_sq"], high_stats["early_l2_sq"]),
                "low_over_high_early_l2_sq": ratio(low_stats["early_l2_sq"], high_stats["early_l2_sq"]),
                "pred_over_high_late_l2_sq": ratio(pred_stats["late_l2_sq"], high_stats["late_l2_sq"]),
                "low_over_high_late_l2_sq": ratio(low_stats["late_l2_sq"], high_stats["late_l2_sq"]),
            },
        }

        per_scene.append(scene_summary)
        aggregate["scene_count"] += 1
        for key in (
            "pred_over_high_peak_abs",
            "low_over_high_peak_abs",
            "pred_over_high_rms",
            "low_over_high_rms",
            "pred_over_high_early_l2_sq",
            "low_over_high_early_l2_sq",
            "pred_over_high_late_l2_sq",
            "low_over_high_late_l2_sq",
        ):
            aggregate[f"avg_{key}"] += scene_summary["ratios"][key]

    if aggregate["scene_count"] > 0:
        n = aggregate["scene_count"]
        for key in list(aggregate.keys()):
            if key.startswith("avg_"):
                aggregate[key] /= n

    out = {
        "prediction_root": str(prediction_root),
        "sample_rate_hz": args.sample_rate_hz,
        "early_ms": args.early_ms,
        "late_ms": args.late_ms,
        "aggregate": aggregate,
        "per_scene": per_scene,
    }

    out_path = prediction_root / "prediction_energy_diagnostics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(json.dumps({"diagnostics_path": str(out_path), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()