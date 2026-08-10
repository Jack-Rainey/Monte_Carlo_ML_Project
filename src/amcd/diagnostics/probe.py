"""
D0 pre-training diagnostics (design_spec §4 / §11).

D0a — headroom probe: how far apart are low-ray and high-ray energy spectra?
       Large gap → signal to learn. Small gap → band energy may have converged
       at this ray budget and the bottleneck lies elsewhere.

D0b — oracle upper bound / carrier ceiling (design_spec §4.2):
       Decode the *true* high-ray energy envelope onto the low-ray carrier (D3)
       and measure residual error against the raw high-ray reference using the
       **standard ISO-3382 metric function** — the same one used by the eval stage.
       Tests whether the carrier (low-ray IR fine structure) limits metric recovery.
       REQUIRES real renders — dry_run noise IRs have no coherent room acoustics
       and will produce CARRIER BOTTLENECK (metrics run but are meaningless on
       stochastic noise; this is not the same as INDETERMINATE).

Run on ALL splits (train, valid, and every test split), reported separately.
Never pool splits — per-split headroom genuinely differs (Invariant 9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.normalization import denormalize
from ..evaluation.room_acoustic import (
    _shared_truncation_per_band,
    channel_band_avg_metrics,
)
from ..representations import build_representation
from ..runtime import Verbosity, emit

# Per-metric JND tolerances (ISO 3382 difference limens) are config-declared —
# see config.d0b_t30_jnd_frac / d0b_edt_jnd_frac / d0b_c50_jnd_db.


def run_diagnostics(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    preprocessed_dir = run_dir / "preprocessed"
    renders_dir = run_dir / "renders"
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    with open(preprocessed_dir / "meta.json") as f:
        meta = json.load(f)
    with open(preprocessed_dir / "splits.json") as f:
        splits: dict[str, str] = json.load(f)

    norm_stats = meta["norm_stats"]

    # Discover all splits present in this run
    all_splits = sorted(set(splits.values()))

    # ─── D0a ─────────────────────────────────────────────────────────────────
    per_split: dict[str, dict] = {}
    per_scene: dict[str, float] = {}

    for split_name in all_splits:
        split_dir = preprocessed_dir / split_name
        scene_ids = [sid for sid, sp in splits.items() if sp == split_name]
        if not scene_ids:
            continue

        scene_gaps: list[float] = []
        per_band_gaps: list[list[float]] = []

        for sid in scene_ids:
            low_pt = split_dir / f"{sid}_low.pt"
            high_pt = split_dir / f"{sid}_high.pt"
            if not low_pt.exists() or not high_pt.exists():
                continue

            low_norm = torch.load(low_pt, weights_only=True)
            high_norm = torch.load(high_pt, weights_only=True)

            low_db = denormalize(low_norm, norm_stats["high_mean"], norm_stats["high_std"])
            high_db = denormalize(high_norm, norm_stats["high_mean"], norm_stats["high_std"])

            gap = (low_db - high_db).abs()  # (C, n_bands, n_frames)
            per_band = gap.mean(dim=(0, 2)).tolist()  # (n_bands,)
            scene_mean_gap = float(gap.mean())

            scene_gaps.append(scene_mean_gap)
            per_band_gaps.append(per_band)
            per_scene[sid] = scene_mean_gap

        if not scene_gaps:
            continue

        gap_mean = float(np.mean(scene_gaps))
        gap_std = float(np.std(scene_gaps))
        band_mean = np.mean(per_band_gaps, axis=0).tolist() if per_band_gaps else []

        if gap_mean >= config.d0a_gap_large_db:
            verdict = f"substantial gap ({gap_mean:.1f} dB ≥ {config.d0a_gap_large_db} dB) — signal to learn at this ray budget"
        elif gap_mean >= config.d0a_gap_small_db:
            verdict = f"moderate gap ({gap_mean:.1f} dB) — some signal; watch per-band results"
        else:
            verdict = f"small gap ({gap_mean:.1f} dB < {config.d0a_gap_small_db} dB) — band energy may have converged; denoising unlikely to help"

        per_split[split_name] = {
            "n_scenes": len(scene_gaps),
            "mean_gap_db": gap_mean,
            "std_gap_db": gap_std,
            "verdict": verdict,
            "per_band_mean_gap_db": band_mean,
        }

    result_d0a = {
        "per_scene_gap_db": per_scene,
        "per_split": per_split,
    }
    (diag_dir / "d0a_gap.json").write_text(json.dumps(result_d0a, indent=2))

    emit(verbosity, "metrics", "\n  D0a — Headroom probe (low-ray vs high-ray energy gap per split)")
    emit(verbosity, "metrics", f"  {'Split':<28} {'n':>4}  {'mean gap':>9}  {'std':>6}  Verdict")
    emit(verbosity, "metrics", "  " + "-" * 80)
    for sp, info in per_split.items():
        emit(
            verbosity, "metrics",
            f"  {sp:<28} {info['n_scenes']:>4}  {info['mean_gap_db']:>8.2f} dB"
            f"  {info['std_gap_db']:>5.2f}  {info['verdict']}",
        )

    # ─── D0b ─────────────────────────────────────────────────────────────────
    _run_d0b(
        config=config,
        preprocessed_dir=preprocessed_dir,
        renders_dir=renders_dir,
        diag_dir=diag_dir,
        splits=splits,
        norm_stats=norm_stats,
        verbosity=verbosity,
    )


# ---------------------------------------------------------------------------
# D0b — carrier ceiling test
# ---------------------------------------------------------------------------

def _run_d0b(
    config: Config,
    preprocessed_dir: Path,
    renders_dir: Path,
    diag_dir: Path,
    splits: dict[str, str],
    norm_stats: dict[str, float],
    verbosity: Verbosity,
) -> None:
    """
    Carrier ceiling test (design_spec §4.2):
    - Oracle IR = decode(true high-ray energy, low-ray carrier)
    - Reference = raw high-ray IR
    - Both scored with the standard ISO-3382 waveform path
    - Residual = |oracle_metric - reference_metric|
    - Gate: residual within JND → carrier sufficient; else carrier is the bottleneck

    Per-split reporting (Invariant 9 — never pool splits).
    """
    all_splits = sorted(set(splits.values()))
    iso_eval_freqs = [float(f) for f in config.iso_eval_freqs]

    # Instantiate representation for D3 decode
    rep = build_representation(
        config.representation.name, config.representation.params,
        sample_rate=config.sample_rate,
    )
    carrier_dir = preprocessed_dir / "carrier"

    per_split_residuals: dict[str, dict] = {}
    per_scene_residuals: dict[str, dict] = {}

    for split_name in all_splits:
        split_dir = preprocessed_dir / split_name
        scene_ids = [sid for sid, sp in splits.items() if sp == split_name]
        if not scene_ids:
            continue

        scene_results: list[dict[str, float]] = []

        for sid in scene_ids:
            high_pt = split_dir / f"{sid}_high.pt"
            carrier_path = carrier_dir / f"{sid}.npy"
            ref_waveform_path = renders_dir / sid / "high.npy"

            if not high_pt.exists() or not carrier_path.exists():
                continue

            # Oracle: decode true high-ray energy onto low-ray carrier
            high_norm = torch.load(high_pt, weights_only=True)
            high_db = denormalize(high_norm, norm_stats["high_mean"], norm_stats["high_std"])
            carrier = np.load(carrier_path)           # (C, T)
            oracle_ir = rep.decode(high_db, carrier)  # (C, T)

            # Reference: raw high-ray waveform
            if not ref_waveform_path.exists():
                continue
            high_ref_ir = np.load(ref_waveform_path)  # (C, T)

            # ISO-3382 metrics for oracle and reference (W-channel, onset-aligned, all
            # eval bands) — same shared path as the eval stage (AC-02/AC-04).
            #
            # D0b is a PAIRED comparison (`residual = |oracle - ref|`), so it carries
            # the AC-17 defect exactly as the eval stage did: truncating each leg at
            # its own noise-dependent Lundeby index integrates them over different
            # limits and manufactures a residual with no acoustic cause. Both legs
            # therefore share one window per band. The reference set is the two
            # PHYSICAL legs available here — the raw high-ray reference and the
            # carrier-decoded oracle, whose floor tracks the low-ray carrier it was
            # decoded onto. No model output is involved in D0b at all (RD-43).
            shared_trunc = _shared_truncation_per_band(
                {"reference": high_ref_ir[0], "oracle": oracle_ir[0]},
                sample_rate=config.sample_rate,
                iso_eval_freqs=iso_eval_freqs,
                onset_rel_db=config.metric_onset_rel_db,
            )
            oracle_metrics, oracle_nan_reasons = channel_band_avg_metrics(
                oracle_ir[0], sample_rate=config.sample_rate,
                iso_eval_freqs=iso_eval_freqs, onset_rel_db=config.metric_onset_rel_db,
                min_measurable_t60_s=config.metric_min_measurable_t60_s,
                trunc_idx_per_band=shared_trunc,
            )
            ref_metrics, ref_nan_reasons = channel_band_avg_metrics(
                high_ref_ir[0], sample_rate=config.sample_rate,
                iso_eval_freqs=iso_eval_freqs, onset_rel_db=config.metric_onset_rel_db,
                min_measurable_t60_s=config.metric_min_measurable_t60_s,
                trunc_idx_per_band=shared_trunc,
            )

            residuals: dict[str, float] = {}
            for key in ("T30", "EDT", "C50"):
                o_val, r_val = oracle_metrics[key], ref_metrics[key]
                residuals[key] = abs(o_val - r_val) if not (np.isnan(o_val) or np.isnan(r_val)) else float("nan")

            per_scene_residuals[sid] = {
                "oracle": oracle_metrics,
                "reference": ref_metrics,
                "residual": residuals,
                # The window both legs were integrated over, and which leg set it
                # (AC-17/RD-44) — a residual is only interpretable alongside it.
                "iso_integration_window": {
                    f"{fc:g}": {"trunc_idx": idx, "set_by_leg": src}
                    for fc, (idx, src) in zip(iso_eval_freqs, shared_trunc)
                },
            }
            # No silent exclusion (F-21): a NaN residual (leg dropped by the shared
            # metric unit) carries its reasons into the probe record.
            if oracle_nan_reasons or ref_nan_reasons:
                per_scene_residuals[sid]["nan_reasons"] = {
                    "oracle": oracle_nan_reasons, "reference": ref_nan_reasons,
                }
            scene_results.append(residuals)

        if not scene_results:
            continue

        split_summary: dict[str, dict] = {}
        for key in ("T30", "EDT", "C50"):
            vals = [r[key] for r in scene_results if not np.isnan(r[key])]
            split_summary[key] = {
                "mean_residual": float(np.mean(vals)) if vals else float("nan"),
                "std_residual": float(np.std(vals)) if vals else float("nan"),
                "n": len(vals),
            }
        per_split_residuals[split_name] = split_summary

    result_d0b = {
        "per_scene": per_scene_residuals,
        "per_split": per_split_residuals,
    }
    (diag_dir / "d0b_oracle.json").write_text(json.dumps(result_d0b, indent=2))

    # ─── Gate verdict ────────────────────────────────────────────────────────
    emit(verbosity, "metrics", "\n  D0b — Carrier ceiling test (oracle IR vs high-ray reference, ISO-3382 path)")
    emit(verbosity, "metrics", f"  {'Split':<28} {'T30 resid':>12}  {'EDT resid':>12}  {'C50 resid':>12}  Verdict")
    emit(verbosity, "metrics", "  " + "-" * 90)

    all_clear = True
    any_indeterminate = False

    for split_name, summary in per_split_residuals.items():
        t30_r = summary["T30"]["mean_residual"]
        edt_r = summary["EDT"]["mean_residual"]
        c50_r = summary["C50"]["mean_residual"]
        t30_n = summary["T30"]["n"]
        edt_n = summary["EDT"]["n"]
        c50_n = summary["C50"]["n"]

        # JND thresholds scale from the independent reference metric (not self-referential)
        scene_ids_sp = [sid for sid, sp in splits.items() if sp == split_name]
        ref_t30 = float(np.nanmean([
            per_scene_residuals[sid]["reference"]["T30"]
            for sid in scene_ids_sp if sid in per_scene_residuals
        ]))
        ref_edt = float(np.nanmean([
            per_scene_residuals[sid]["reference"]["EDT"]
            for sid in scene_ids_sp if sid in per_scene_residuals
        ]))

        t30_thresh = config.d0b_t30_jnd_frac * ref_t30 if not np.isnan(ref_t30) else float("nan")
        edt_thresh = config.d0b_edt_jnd_frac * ref_edt if not np.isnan(ref_edt) else float("nan")
        c50_thresh = config.d0b_c50_jnd_db

        def _verdict(r: float, thresh: float, n: int) -> str:
            if n == 0 or np.isnan(r) or np.isnan(thresh):
                return "N/A"
            return "PASS" if r <= thresh else "FAIL"

        t30_v = _verdict(t30_r, t30_thresh, t30_n)
        edt_v = _verdict(edt_r, edt_thresh, edt_n)
        c50_v = _verdict(c50_r, c50_thresh, c50_n)

        if "FAIL" in (t30_v, edt_v, c50_v):
            all_clear = False
        if "N/A" in (t30_v, edt_v, c50_v):
            # Any missing metric is insufficient coverage — cannot declare CEILING CLEARS
            any_indeterminate = True

        t30_s = f"{t30_r:.4f}s" if not np.isnan(t30_r) else "   N/A"
        edt_s = f"{edt_r:.4f}s" if not np.isnan(edt_r) else "   N/A"
        c50_s = f"{c50_r:.4f}dB" if not np.isnan(c50_r) else "   N/A"

        emit(
            verbosity, "metrics",
            f"  {split_name:<28} "
            f"{t30_s:>12}({t30_v})  "
            f"{edt_s:>12}({edt_v})  "
            f"{c50_s:>12}({c50_v})",
        )

    emit(verbosity, "metrics", "")
    if any_indeterminate:
        emit(verbosity, "metrics", "  D0b verdict: INDETERMINATE — one or more ISO-3382 metrics unavailable in at least one split")
        emit(verbosity, "metrics", "  Re-run on real GSound-SIR renders (ir_duration ≥ 1 s) before treating this gate as informative")
    elif all_clear:
        emit(verbosity, "metrics", "  D0b verdict: CARRIER CEILING CLEARS — oracle IR recovers reference metrics within JND")
        emit(verbosity, "metrics", "  Upper bound: carrier fine structure is not the bottleneck given perfect energy envelope.")
        emit(verbosity, "metrics", "  This does not guarantee a trained model reaches the same ceiling (oracle uses ground-truth energy).")
        emit(verbosity, "metrics", "  Proceed to E1.")
    else:
        emit(verbosity, "metrics", "  D0b verdict: CARRIER BOTTLENECK — oracle misses reference metrics beyond JND")
        emit(verbosity, "metrics", "  Early-reflection sparsity in the low-ray carrier limits metric recovery")
        emit(verbosity, "metrics", "  Investigate carrier quality before training")

    emit(verbosity, "metrics", f"  → d0b_oracle.json written to {diag_dir}")
