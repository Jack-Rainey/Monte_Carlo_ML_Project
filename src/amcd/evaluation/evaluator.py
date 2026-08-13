"""eval stage: compute per-scene metrics → metrics/metrics.parquet + drops.csv
+ iso_integration_windows.json (the shared Schroeder window per scene/band, AC-17)."""
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
    # Shared ISO-3382 Schroeder integration window per (scene, band) (AC-17/RD-44).
    # Recorded for every scene the room-acoustic path scores, so a reported absolute
    # T30/EDT/C50 can always be traced to the window — and the leg — that set it.
    iso_windows: dict[str, dict[str, dict[str, object]]] = {}

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
        # infer only ever writes predictions for TEST splits, so a prediction whose
        # scene now sits in train/valid is residue from an earlier run under a
        # different split assignment — a different model's output, which would
        # otherwise be scored and reported as if it belonged here (F-37).
        if split not in config.test_split_names:
            raise RuntimeError(
                f"scene {scene_id!r} has a prediction but splits.json assigns it to "
                f"{split!r}, which is not a test split. Predictions are only produced "
                f"for test splits, so this file is left over from a previous run "
                f"under a different split assignment. Re-run infer (it now clears "
                f"stale predictions) or use a fresh --run-dir."
            )

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
            (
                room_triples, room_reasons, room_window, band_accounting
            ) = compute_room_acoustic_metrics(
                decoded_ir, high_ref_ir, low_ref_ir,
                sample_rate=config.sample_rate,
                iso_eval_freqs=[float(f) for f in config.iso_eval_freqs],
                onset_rel_db=config.metric_onset_rel_db,
                band_resolvability_margin=config.metric_band_resolvability_margin,
                min_decay_range_db=config.metric_min_decay_range_db,
            )
            all_metrics.update(room_triples)
            nan_reasons.update(room_reasons)
            # The shared Schroeder window is recorded for EVERY scored scene, not
            # only for dropped ones (RD-44): reported ISO absolutes are windowed by
            # the noisier physical leg, so a reader must be able to see the window
            # that produced them.
            # Units declared IN the artifact (AC-132): the index is in SAMPLES and
            # the band key in Hz, and neither was stated — nor the sample_rate a
            # reader needs to convert one to seconds.
            iso_windows[scene_id] = {
                band: {
                    "trunc_idx_samples": idx,
                    "trunc_s": idx / config.sample_rate,
                    "band_hz": float(band),
                    "sample_rate": config.sample_rate,
                    "set_by_leg": src,
                }
                for band, (idx, src) in room_window.items()
            }
        else:
            # Room-acoustic artifacts (decoded IR / reference waveforms) absent for
            # this scene — record an all-NaN triple for each ISO-3382 metric rather
            # than dropping the row. improvement is then undefined (None), never a
            # borrowed flag. Simulator-agnostic: no stage special-cases a simulator.
            def _nan_triple(metric: str) -> MetricTriple:
                # The unit survives an unscored row: a reader must be able to see
                # what the column WOULD have held.
                return MetricTriple(
                    float("nan"), float("nan"), float("nan"),
                    kind="match_reference", unit="dB" if metric == "C50" else "s",
                )

            nan_triple = _nan_triple("T30")
            reason = "decoded/reference IR artifacts absent for this scene"
            band_accounting = {}
            for key in ("T30", "C50", "EDT"):
                all_metrics[key] = _nan_triple(key)
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
            # Band accounting travels WITH the row (F-62). "N sc/att" could not tell
            # a fully-scored scene from a partially-scored one: on the RI smoke run
            # EDT/test_id read 4/4 while one scene's EDT was a ONE-BAND average and
            # the other three were two-band averages, visible only in drops.csv. The
            # split's CI then pools per-scene improvements computed over different
            # band sets under a headline count that reads as complete.
            acct = band_accounting.get(metric_name)
            # EDT below this bound is variance-limited, not filter-limited (AC-27):
            # measured sd 24-31 % of T60 against 6-10 % for T30. No threshold can
            # remove that, so it is disclosed per scene and counted per split
            # (RD-78) rather than suppressed.
            edt_uncertain = (
                metric_name == "EDT"
                and not math.isnan(triple.high)
                and triple.high < config.metric_edt_variance_limited_s
            )
            rows.append({
                "scene_id": scene_id,
                "split": split,
                "metric": metric_name,
                "kind": triple.kind,
                # Carried from the producer so the reported Unit column reads a
                # DECLARATION rather than a second map that has to agree with it.
                "unit": triple.unit,
                "low_val": float(triple.low),
                "pred_val": float(triple.pred),
                "high_ref": float(triple.high),
                "baseline_rel_ratio": baseline_rel_ratio,
                "improved": improved,
                "n_bands_kept": None if acct is None else acct["n_bands_kept"],
                "n_bands_total": None if acct is None else acct["n_bands"],
                "n_bands_pred_unresolved": (
                    None if acct is None else len(acct["pred_unresolved_hz"])
                ),
                # AC-38: bands whose value is REPORTED despite sitting below what
                # the band can resolve. Suppressing them censored the estimator on
                # its own magnitude and biased the split mean up (+7.5 % at true
                # T60 = 0.04 s), so the number is disclosed with a caveat instead —
                # the shape `metric_edt_variance_limited_s` already uses (RD-78).
                "n_bands_resolvability_limited": (
                    None if acct is None else len(acct["resolvability_limited_hz"])
                ),
                # RD-93: the OVERLAP that makes AC-38 cost something. A band that is
                # floor-limited AND unresolved in pred is a band that, before AC-38,
                # left every leg's average — leaving the scene IN the paired
                # comparison. It now stays, pred is NaN in it, and the scene leaves
                # `paired_improvement` instead. That is F-70's selection on the
                # dependent variable, enlarged and optimistic. F-70's bound lives in
                # stats/aggregate.py (integrator queue), so this column is what makes
                # the enlargement measurable rather than invisible.
                "n_bands_pred_unresolved_in_floor_limited": (
                    None if acct is None
                    else len(acct["pred_unresolved_in_floor_limited_hz"])
                ),
                "estimator_variance_limited": edt_uncertain,
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

    # Canonical, not verbosity-gated: without it a reported ISO absolute cannot be
    # interpreted, because the window is set by the noisier physical leg (RD-44).
    (metrics_dir / "iso_integration_windows.json").write_text(
        json.dumps(iso_windows, indent=2, sort_keys=True)
    )

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
