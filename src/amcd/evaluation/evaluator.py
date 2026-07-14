"""eval stage: compute per-scene metrics → metrics/metrics.parquet + drops.csv"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.normalization import denormalize
from ..runtime import Verbosity, emit
from .metric_row import KIND_LEGS, MetricDrop, MetricTriple, metric_improvement
from .signal import compute_signal_metrics
from .room_acoustic import compute_room_acoustic_metrics
from .spatial import compute_spatial_metrics
from .perceptual import compute_perceptual_metrics


def run_eval(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    preprocessed_dir = run_dir / "preprocessed"
    predictions_dir = run_dir / "predictions"
    renders_dir = run_dir / "renders"
    carrier_dir = preprocessed_dir / "carrier"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(preprocessed_dir / "meta.json") as f:
        meta = json.load(f)
    with open(preprocessed_dir / "splits.json") as f:
        splits: dict[str, str] = json.load(f)

    norm_stats = meta["norm_stats"]
    # Preprocess-stamped domain of the saved tensors ("db" | "amplitude"); keyed
    # from the stamp, not the rep class, so eval stays rep-agnostic (F-19). A
    # missing key means a pre-stamp preprocess run — fail loud, re-run preprocess.
    value_domain = meta["value_domain"]

    pred_paths = sorted(p for p in predictions_dir.glob("*_pred.pt") if not p.name.startswith("._"))
    if not pred_paths:
        raise RuntimeError(f"No predictions found in {predictions_dir}. Run infer first.")

    rows: list[dict] = []
    # Drop log (F-21): one row per (scene, split, metric, leg) that is NaN in a
    # leg its kind consumes (unscored) or was only partially computed — written
    # to metrics/drops.csv so nothing leaves a result silently.
    drop_rows: list[dict] = []

    for pred_path in pred_paths:
        scene_id = pred_path.stem.replace("_pred", "")
        if scene_id not in splits:
            # splits.json is the source of truth for split membership; a prediction
            # with no split entry is a pipeline inconsistency, not a "test_id" scene.
            raise KeyError(
                f"scene {scene_id!r} has a prediction but no entry in splits.json; "
                f"refusing to guess its split (config-as-source-of-truth)."
            )
        split = splits[scene_id]

        # Energy tensors for signal metrics
        pred_norm = torch.load(pred_path, weights_only=False)  # (C, n_bands, n_frames)
        split_dir = preprocessed_dir / split
        low_norm = torch.load(split_dir / f"{scene_id}_low.pt", weights_only=False)
        high_norm = torch.load(split_dir / f"{scene_id}_high.pt", weights_only=False)

        pred_db = denormalize(pred_norm, norm_stats["high_mean"], norm_stats["high_std"])
        low_db = denormalize(low_norm, norm_stats["high_mean"], norm_stats["high_std"])
        high_db = denormalize(high_norm, norm_stats["high_mean"], norm_stats["high_std"])

        all_metrics: dict[str, MetricTriple] = {}
        # Producer-supplied NaN reasons, keyed (metric, leg) — merged across
        # producers, consumed by the drop sweep below (F-21).
        nan_reasons: dict[tuple[str, str], str] = {}

        # Signal metrics — operand domain, with dB-only SNR keyed on the
        # stamped value_domain (F-19)
        signal_triples, signal_reasons = compute_signal_metrics(
            pred_db, high_db, low_db, value_domain=value_domain
        )
        all_metrics.update(signal_triples)
        nan_reasons.update(signal_reasons)

        # Room-acoustic metrics — standard ISO-3382 path on decoded waveforms (§3, §6)
        decoded_ir_path = predictions_dir / f"{scene_id}_decoded_ir.npy"
        high_ir_path = renders_dir / scene_id / "high.npy"
        low_ir_path = carrier_dir / f"{scene_id}.npy"

        if decoded_ir_path.exists() and high_ir_path.exists() and low_ir_path.exists():
            decoded_ir = np.load(decoded_ir_path)          # (C, T)
            high_ref_ir = np.load(high_ir_path)            # (C, T)
            low_ref_ir = np.load(low_ir_path)              # (C, T)
            room_triples, room_reasons = compute_room_acoustic_metrics(
                decoded_ir, high_ref_ir, low_ref_ir,
                sample_rate=config.sample_rate,
                iso_eval_freqs=[float(f) for f in config.iso_eval_freqs],
                onset_rel_db=config.metric_onset_rel_db,
            )
            all_metrics.update(room_triples)
            nan_reasons.update(room_reasons)
        else:
            # Room-acoustic artifacts (decoded IR / reference waveforms) absent for
            # this scene — record an all-NaN triple for each ISO-3382 metric rather
            # than dropping the row. improvement is then undefined (None), never a
            # borrowed flag. Simulator-agnostic: no stage special-cases a simulator.
            nan_triple = MetricTriple(
                float("nan"), float("nan"), float("nan"), kind="match_reference"
            )
            reason = "decoded/reference IR artifacts absent for this scene"
            for key in ("T30", "C50", "EDT"):
                all_metrics[key] = nan_triple
                for leg in KIND_LEGS[nan_triple.kind]:
                    nan_reasons[(key, leg)] = reason

        for producer in (compute_spatial_metrics, compute_perceptual_metrics):
            triples, reasons = producer(pred_db, high_db, low_db)
            all_metrics.update(triples)
            nan_reasons.update(reasons)

        # One row per (scene, metric). `improved`/`baseline_rel_ratio` are derived
        # per-metric from that metric's own triple and declared kind — no metric
        # borrows another's flag (F-07), no implicit match-reference assumption
        # (F-20). improved is None where undefined (F-08).
        for metric_name, triple in all_metrics.items():
            improved, baseline_rel_ratio = metric_improvement(triple)
            rows.append({
                "scene_id": scene_id,
                "split": split,
                "metric": metric_name,
                "kind": triple.kind,
                "low_val": float(triple.low),
                "pred_val": float(triple.pred),
                "high_ref": float(triple.high),
                "baseline_rel_ratio": baseline_rel_ratio,
                "improved": improved,
            })
            # Drop sweep (F-21): every consumed-leg NaN must carry a reason; a
            # missing one is still logged, visibly attributed to the producer.
            # A reason on a FINITE leg is a partial intra-leg drop (e.g. some
            # eval bands NaN) — logged too, so the count change is visible.
            for leg in KIND_LEGS[triple.kind]:
                reason = nan_reasons.get((metric_name, leg))
                if math.isnan(getattr(triple, leg)):
                    drop = MetricDrop(
                        metric_name, leg, reason or "reason not supplied by producer"
                    )
                elif reason is not None:
                    drop = MetricDrop(metric_name, leg, reason)
                else:
                    continue
                drop_rows.append({"scene_id": scene_id, "split": split, **drop._asdict()})

    if not rows:
        raise RuntimeError("No metric rows produced — check predictions and test split.")

    df = pd.DataFrame(rows)
    df.to_parquet(metrics_dir / "metrics.parquet", index=False)

    # Always written, even when empty (header only): "no drops" is then an
    # explicit statement, distinguishable from "log never produced" (F-21).
    drops_df = pd.DataFrame(
        drop_rows, columns=["scene_id", "split", "metric", "leg", "reason"]
    )
    drops_df.to_csv(metrics_dir / "drops.csv", index=False)

    n_scenes = df["scene_id"].nunique()
    # Headline count: scenes whose energy MSE improved over the low-ray baseline.
    energy_rows = df[df["metric"] == "energy_mse"]
    n_improved = int((energy_rows["improved"] == True).sum())  # noqa: E712
    emit(
        verbosity, "metrics",
        f"  Evaluated {n_scenes} scenes, "
        f"{n_improved}/{n_scenes} improved (energy MSE) over baseline → {metrics_dir / 'metrics.parquet'}",
    )
    emit(
        verbosity, "metrics",
        f"  {len(drop_rows)} dropped/partial metric legs logged → {metrics_dir / 'drops.csv'}",
    )
