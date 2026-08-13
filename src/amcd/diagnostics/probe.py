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

PER-SPLIT RECORD SCHEMA (RR-64). Both probes write one entry per split into their
artifact's `per_split` map, and both artifacts are read outside this module (see
`tests/test_dataset_integrity.py`), so the shape is declared here once rather than
inferred from the six sites that construct it:

    n_scenes        int  — scenes SCORED. The same quantity `stats/aggregate.py`
                           and `reporting/tables.py` publish as `n_scored`; the
                           probes keep `n_scenes` because existing consumers index
                           it, and adding a second name here is how AC-24's pair
                           drifted apart.
    n_attempted     int  — scenes the split contained, i.e. the denominator.
    dropped         list — [{"scene": str, "reason": str}], one per unscored
                           scene, mirroring the eval stage's drops.csv (F-21).
    unscored_reason str  — present IF AND ONLY IF `n_scenes == 0`.

The emit-iff invariant is what both consumers rely on, so both index
`unscored_reason` directly; a `.get` default would say the contract is optional
while the writers treat it as guaranteed. D0b entries additionally carry the
per-metric residual blocks `T30`/`EDT`/`C50` — present iff `n_scenes > 0`, which
is the condition its verdict loop branches on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.normalization import denormalize
from ..evaluation.room_acoustic import (
    channel_per_band_metrics,
    _shared_truncation_per_band,
    channel_band_avg_metrics,
)
from ..representations import build_representation
from ..simulators.base import simulator_models_early_reflections
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

    # Enumerate the CONFIG-DECLARED splits, in declaration order, not the splits
    # that happen to have received a scene (F-45) — a declared split with no scenes
    # is a fact about this run and must be reported as 0, not omitted. Any split
    # present in the data but not declared would be a routing bug, so it is appended
    # rather than silently ignored.
    declared_splits = list(config.splits)
    all_splits = declared_splits + sorted(set(splits.values()) - set(declared_splits))

    # ─── D0a ─────────────────────────────────────────────────────────────────
    per_split: dict[str, dict] = {}
    per_scene: dict[str, float] = {}

    for split_name in all_splits:
        split_dir = preprocessed_dir / split_name
        scene_ids = [sid for sid, sp in splits.items() if sp == split_name]
        if not scene_ids:
            # Live branch now that enumeration is config-declared (F-45): record the
            # empty split rather than skipping it, so the probe's split list matches
            # the config's declared set.
            per_split[split_name] = {
                "n_scenes": 0,
                "n_attempted": 0,
                "dropped": [],
                "unscored_reason": "declared in config but received no scenes",
            }
            emit(verbosity, "warning",
                 f"  WARNING: declared split {split_name!r} has no scenes — "
                 f"recorded as 0, not omitted (F-45).")
            continue

        scene_gaps: list[float] = []
        per_band_gaps: list[list[float]] = []
        # (scene, reason) for every scene this probe could not score, mirroring the
        # eval stage's drops.csv (F-21). F-45 fixed the SPLIT enumeration; these
        # per-scene skips stayed silent, so a split where 5 of 20 scenes lacked
        # tensors reported headroom over the survivors and read as complete (F-72).
        dropped: list[dict[str, str]] = []

        for sid in scene_ids:
            low_pt = split_dir / f"{sid}_low.pt"
            high_pt = split_dir / f"{sid}_high.pt"
            missing = [p.name for p in (low_pt, high_pt) if not p.exists()]
            if missing:
                dropped.append({
                    "scene": sid,
                    "reason": f"preprocessed tensor missing: {', '.join(missing)}",
                })
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

        if dropped:
            emit(verbosity, "warning",
                 f"  WARNING: D0a scored {len(scene_gaps)} of {len(scene_ids)} scenes "
                 f"in split {split_name!r} — {len(dropped)} dropped, per-scene reasons "
                 f"in d0a_gap.json (F-72).")

        if not scene_gaps:
            # Every scene failed. Skipping here made the split vanish from the
            # artifact — indistinguishable from a split that was never declared, the
            # state F-45 closed on the other axis. Recorded with its reason instead,
            # and the print branch below renders it as unscored, never as a number.
            per_split[split_name] = {
                "n_scenes": 0,
                "n_attempted": len(scene_ids),
                "dropped": dropped,
                "unscored_reason": (
                    f"all {len(scene_ids)} scenes failed to load — see `dropped` for "
                    f"the per-scene reasons"
                ),
            }
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
            # Schema: module docstring.
            "n_scenes": len(scene_gaps),
            "n_attempted": len(scene_ids),
            "dropped": dropped,
            "mean_gap_db": gap_mean,
            "std_gap_db": gap_std,
            "verdict": verdict,
            "per_band_mean_gap_db": band_mean,
        }

    result_d0a = {
        "backend_limitations": _backend_limitations(config),
        "per_scene_gap_db": per_scene,
        "per_split": per_split,
    }
    (diag_dir / "d0a_gap.json").write_text(json.dumps(result_d0a, indent=2))

    emit(verbosity, "metrics", "\n  D0a — Headroom probe (low-ray vs high-ray energy gap per split)")
    emit(verbosity, "metrics",
         f"  {'Split':<28} {'scored/att':>10}  {'mean gap':>9}  {'std':>6}  Verdict")
    emit(verbosity, "metrics", "  " + "-" * 80)
    for sp, info in per_split.items():
        if info["n_scenes"] == 0:
            # Declared but empty (F-45): named with its reason, never rendered as a
            # number — a 0.00 dB gap would read as a measured result.
            counts = f"0/{info['n_attempted']}"
            emit(
                verbosity, "metrics",
                f"  {sp:<28} {counts:>10}  unscored — {info['unscored_reason']}",
            )
            continue
        counts = f"{info['n_scenes']}/{info['n_attempted']}"
        emit(
            verbosity, "metrics",
            f"  {sp:<28} {counts:>10}  {info['mean_gap_db']:>8.2f} dB"
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

#: The backend-limitation note both D0 artifacts carry (AC-43).
#:
#: EDT fits the FIRST 10 dB, which in a real room IS the early-reflection span — the
#: reason EDT moves systematically with source-receiver distance. A backend whose
#: diffuse tail begins at the direct arrival has no structure there, so its EDT is
#: nearly inert on the placement axis while C50 stays live (AC-28 gave the scaffold a
#: real 1/d direct term against a room-constant tail, which is what C50 integrates).
#:
#: In BOTH artifacts because both publish per-split EDT: D0a's headroom gap and D0b's
#: oracle residual are each read as "how much EDT signal is there", and under such a
#: backend the placement-axis answer is a property of the renderer, not of the room.
_NO_EARLY_REFLECTIONS_NOTE = (
    "The active backend renders no early-reflection cluster, so its diffuse tail "
    "begins at the direct arrival and the first 10 dB — exactly what EDT fits — "
    "carries no reflection structure. EDT is therefore nearly inert on the PLACEMENT "
    "axis: measured 0.5517 / 0.7888 / 0.7994 / 0.7848 / 0.7853 s at "
    "d = 0.5/1/2/4/8 m (10x8x3.5 m, alpha 0.2), non-monotone and flat to within 2 % "
    "from 1 m out, against C50's monotone 9.90 dB swing over the same 16x range. Any "
    "EDT figure here for a placement split is a plumbing result, not an acoustic one. "
    "T30, C50 and the material/geometry axes are unaffected. Adding the cluster is "
    "the real simulator's job, not a scaffold fix (AC-28/AC-43)."
)


def _backend_limitations(config) -> dict:
    """Limitations of the ACTIVE backend that change how this artifact reads.

    Emitted as a declared block rather than omitted when empty: an artifact with no
    `backend_limitations` key is indistinguishable from one written before the block
    existed, and "we checked and there are none" is a different fact from "nobody
    checked".
    """
    limitations = {}
    if not simulator_models_early_reflections(config):
        limitations["edt_placement_axis"] = _NO_EARLY_REFLECTIONS_NOTE
    return {"simulator": config.simulator.name, "limitations": limitations}


def _band_intersected_pair(
    oracle_w: np.ndarray, reference_w: np.ndarray, *, config,
    iso_eval_freqs: list[float], shared_trunc,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Band-average two legs over the bands BOTH resolve, plus the drop reasons.

    `channel_band_avg_metrics` averages one IR over its own surviving bands, which
    is right for a standalone IR and wrong for a comparison: two legs of the same
    scene can then be averaged over different band sets, and the difference between
    those sets shows up as a residual with no acoustic cause. The eval stage
    already intersects (AC-08); this is the same rule for D0b (F-101).

    A band excluded here is recorded with the leg and the reason, so a residual
    computed over fewer bands is visible rather than silent.
    """
    per_leg = {
        leg: channel_per_band_metrics(
            ir, sample_rate=config.sample_rate, iso_eval_freqs=iso_eval_freqs,
            onset_rel_db=config.metric_onset_rel_db,
            band_resolvability_margin=config.metric_band_resolvability_margin,
            min_decay_range_db=config.metric_min_decay_range_db,
            octave_filter_order=config.metric_octave_filter.order,
            trunc_idx_per_band=shared_trunc,
        )
        for leg, ir in (("oracle", oracle_w), ("reference", reference_w))
    }
    oracle_vals: dict[str, float] = {}
    ref_vals: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for metric in ("T30", "EDT", "C50"):
        kept, dropped = [], []
        for b, fc in enumerate(iso_eval_freqs):
            finite = {
                leg: not np.isnan(bands[b][0][metric]) and metric not in bands[b][2]
                for leg, bands in per_leg.items()
            }
            if all(finite.values()):
                kept.append(b)
            else:
                lost = [leg for leg, ok in finite.items() if not ok]
                dropped.append(f"{fc:g} Hz ({'/'.join(lost)})")
        if not kept:
            oracle_vals[metric] = float("nan")
            ref_vals[metric] = float("nan")
            reasons[metric] = (
                f"no eval band is resolvable in BOTH legs: {', '.join(dropped)}"
            )
            continue
        oracle_vals[metric] = float(np.mean(
            [per_leg["oracle"][b][0][metric] for b in kept]))
        ref_vals[metric] = float(np.mean(
            [per_leg["reference"][b][0][metric] for b in kept]))
        if dropped:
            reasons[metric] = (
                f"partial: averaged over {len(kept)} of {len(iso_eval_freqs)} bands; "
                f"excluded from BOTH legs to keep the residual comparable: "
                f"{', '.join(dropped)}"
            )
    return oracle_vals, ref_vals, reasons


def _d0b_level_sweep(
    *, rep, high_db, carrier, high_ref_ir, config, iso_eval_freqs, gains_db,
) -> dict[str, dict]:
    """AC-37 (c): the oracle's T30 error as a function of the scene's ABSOLUTE LEVEL.

    `min_db` is an absolute floor on the encoded band energy, not a level below the
    scene's own peak, and `decode` rescales the carrier's band-frame power to
    `10**(env/10)`. So wherever true power sits below the floor, the decode BOOSTS
    the carrier up to it and injects a non-decaying energy floor into the prediction
    — inside the shared Schroeder window the ISO metrics are integrated over. Quiet
    scenes are the ones at risk, and how quiet is a property of the render's level
    convention rather than of the model.

    `encode`'s headroom guard refuses a scene that would breach the JND (AC-37 (a)),
    and the known-answer test pins where that boundary is. This is the third half the
    user asked for: the SHIPPED artifact says how much margin the real dataset has,
    per split, rather than the margin existing only in a test.

    NO RE-RENDER, AND THE ARITHMETIC IS EXACT. Scaling a waveform by `g` scales band
    ENERGY by `g**2`, so the encoded dB envelope shifts by exactly the gain in dB —
    the sweep re-clamps the shifted envelope at `min_db` and decodes that. The
    carrier is not scaled because `decode` rescales its band power to the envelope's
    target regardless, and the reference T30 is not recomputed because the ISO
    estimator is gain-invariant by construction (the Lundeby limit is taken relative
    to the record's own peak).

    Returns {gain_db: {t30_rel_error, headroom_db, breaches_jnd}} for one scene.
    """
    import torch

    out: dict[str, dict] = {}
    ref_t30 = channel_band_avg_metrics(
        high_ref_ir[0], sample_rate=config.sample_rate,
        iso_eval_freqs=iso_eval_freqs, onset_rel_db=config.metric_onset_rel_db,
        band_resolvability_margin=config.metric_band_resolvability_margin,
        min_decay_range_db=config.metric_min_decay_range_db,
        octave_filter_order=config.metric_octave_filter.order,
    )[0]["T30"]
    for gain_db in gains_db:
        shifted = torch.clamp(high_db + float(gain_db), min=rep.min_db)
        # The guard's own operand, so the number reported here is the number
        # `encode` would have judged (AC-69: the wrong reduction next to the right
        # one is how the definition drifts).
        peak = torch.amax(shifted, dim=2) - rep.min_db
        headroom = float(peak[:, rep.headroom_band_indices].min())
        oracle_ir = rep.decode(shifted, carrier)
        shared = _shared_truncation_per_band(
            {"reference": high_ref_ir[0], "oracle": oracle_ir[0]},
            sample_rate=config.sample_rate, iso_eval_freqs=iso_eval_freqs,
            onset_rel_db=config.metric_onset_rel_db,
            octave_filter_order=config.metric_octave_filter.order,
        )
        oracle_m, ref_m, _reasons = _band_intersected_pair(
            oracle_ir[0], high_ref_ir[0], config=config,
            iso_eval_freqs=iso_eval_freqs, shared_trunc=shared,
        )
        o_t30, r_t30 = oracle_m["T30"], ref_m["T30"]
        rel = (
            abs(o_t30 - r_t30) / r_t30
            if not (np.isnan(o_t30) or np.isnan(r_t30) or r_t30 <= 0.0)
            else float("nan")
        )
        out[f"{gain_db:g}"] = {
            "t30_rel_error": rel,
            "headroom_db": headroom,
            # NaN is not a pass. A gain whose oracle could not be scored says so,
            # rather than counting toward "no breach".
            "breaches_jnd": bool(rel > config.d0b_t30_jnd_frac) if rel == rel else None,
        }
    out["reference_t30_s"] = {"value": float(ref_t30)}
    return out


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
    # Config-declared enumeration, in declaration order — the same rule as D0a ~130
    # lines above (F-45). Enumerating the splits that RECEIVED a scene instead let a
    # declared-but-empty split vanish from d0b_oracle.json entirely.
    declared_splits = list(config.splits)
    all_splits = declared_splits + sorted(set(splits.values()) - set(declared_splits))
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
            per_split_residuals[split_name] = {
                "n_scenes": 0,
                "n_attempted": 0,
                "dropped": [],
                "unscored_reason": "declared in config but received no scenes",
            }
            continue

        scene_results: list[dict[str, float]] = []
        # Same (scene, reason) accounting as D0a (F-72).
        dropped: list[dict[str, str]] = []

        for sid in scene_ids:
            high_pt = split_dir / f"{sid}_high.pt"
            carrier_path = carrier_dir / f"{sid}.npy"
            ref_waveform_path = renders_dir / sid / "high.npy"

            missing = [
                label for label, path in (
                    (high_pt.name, high_pt),
                    (f"carrier/{carrier_path.name}", carrier_path),
                    (f"renders/{sid}/{ref_waveform_path.name}", ref_waveform_path),
                ) if not path.exists()
            ]
            if missing:
                dropped.append({
                    "scene": sid,
                    "reason": f"input missing: {', '.join(missing)}",
                })
                continue

            # Oracle: decode true high-ray energy onto low-ray carrier
            high_norm = torch.load(high_pt, weights_only=True)
            high_db = denormalize(high_norm, norm_stats["high_mean"], norm_stats["high_std"])
            carrier = np.load(carrier_path)           # (C, T)
            oracle_ir = rep.decode(high_db, carrier)  # (C, T)

            # Reference: raw high-ray waveform (existence checked with the other two
            # inputs above, so a missing render leaves a logged reason).
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
                octave_filter_order=config.metric_octave_filter.order,
            )
            # BAND-INTERSECTED ACROSS LEGS, as eval does (AC-08 / F-101).
            #
            # Averaging each leg over ITS OWN surviving bands lets the residual be
            # decided by band COMPOSITION rather than by acoustics: measured at an
            # identical true T60 = 0.045 s with only different noise realizations,
            # the oracle averaged [1000] while the reference averaged [500, 1000],
            # giving a residual of 0.00244 from composition alone. The asymmetry is
            # DIRECTIONAL, because the oracle sits on the noisier low-ray carrier
            # and so loses bands more often — so this inflates the very residual
            # D0b compares against a JND.
            oracle_metrics, ref_metrics, band_reasons = _band_intersected_pair(
                oracle_ir[0], high_ref_ir[0], config=config,
                iso_eval_freqs=iso_eval_freqs, shared_trunc=shared_trunc,
            )
            oracle_nan_reasons = band_reasons
            ref_nan_reasons = band_reasons

            residuals: dict[str, float] = {}
            for key in ("T30", "EDT", "C50"):
                o_val, r_val = oracle_metrics[key], ref_metrics[key]
                residuals[key] = abs(o_val - r_val) if not (np.isnan(o_val) or np.isnan(r_val)) else float("nan")

            per_scene_residuals[sid] = {
                "oracle": oracle_metrics,
                "reference": ref_metrics,
                "residual": residuals,
                # AC-37 (c): how much absolute-level margin this scene has before
                # `min_db` starts injecting an energy floor into the decode. The
                # residual above is measured at the scene's NATIVE level, where the
                # defect is inert; this is what says how far from inert it is.
                "level_sweep": _d0b_level_sweep(
                    rep=rep, high_db=high_db, carrier=carrier,
                    high_ref_ir=high_ref_ir, config=config,
                    iso_eval_freqs=iso_eval_freqs,
                    gains_db=config.d0b_level_sweep_db,
                ),
                # The window both legs were integrated over, and which leg set it
                # (AC-17/RD-44) — a residual is only interpretable alongside it.
                "iso_integration_window": {
                    f"{fc:g}": {
                        "trunc_idx_samples": idx,
                        "trunc_s": idx / config.sample_rate,
                        "band_hz": float(fc),
                        "sample_rate": config.sample_rate,
                        "set_by_leg": src,
                    }
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

        if dropped:
            emit(verbosity, "warning",
                 f"  WARNING: D0b scored {len(scene_results)} of {len(scene_ids)} "
                 f"scenes in split {split_name!r} — {len(dropped)} dropped, per-scene "
                 f"reasons in d0b_oracle.json (F-72).")

        if not scene_results:
            # Carries no per-metric residuals, which is the condition the verdict
            # loop below reads as INDETERMINATE (F-72).
            per_split_residuals[split_name] = {
                "n_scenes": 0,
                "n_attempted": len(scene_ids),
                "dropped": dropped,
                "unscored_reason": (
                    f"all {len(scene_ids)} scenes lacked a required input — see "
                    f"`dropped` for the per-scene reasons"
                ),
            }
            continue

        split_summary: dict[str, dict] = {}
        for key in ("T30", "EDT", "C50"):
            vals = [r[key] for r in scene_results if not np.isnan(r[key])]
            split_summary[key] = {
                "mean_residual": float(np.mean(vals)) if vals else float("nan"),
                "std_residual": float(np.std(vals)) if vals else float("nan"),
                "n": len(vals),
            }
        # Scored-vs-attempted for the split as a whole. The per-metric `n` above is a
        # further, narrower count: a scene can be scored here and still yield a NaN
        # residual for one metric, which `nan_reasons` records per scene.
        split_summary["n_scenes"] = len(scene_results)
        split_summary["n_attempted"] = len(scene_ids)
        split_summary["dropped"] = dropped
        per_split_residuals[split_name] = split_summary

    result_d0b = {
        "backend_limitations": _backend_limitations(config),
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
        # THE `all_clear` RULE, stated once, here, at the loop that consumes it
        # (F-45/F-72): a verdict may never clear a split it did not measure. A split
        # with no scenes, or whose scenes all failed to load, carries only its
        # unscored reason — so it is INDETERMINATE, never a pass. Omitting such a
        # split (the pre-fix behaviour) left `all_clear` True over it silently.
        #
        # Branches on the DECLARED condition (`n_scenes == 0`), the same one D0a's
        # print loop uses. Branching on `"T30" not in summary` agreed with it only
        # because `channel_band_avg_metrics` happens to return all three keys — an
        # assumption the schema does not state.
        if summary["n_scenes"] == 0:
            any_indeterminate = True
            emit(
                verbosity, "metrics",
                f"  {split_name:<28} {'N/A':>12}  {'N/A':>12}  {'N/A':>12}  "
                f"N/A — {summary['unscored_reason']}",
            )
            continue
        t30_r = summary["T30"]["mean_residual"]
        edt_r = summary["EDT"]["mean_residual"]
        c50_r = summary["C50"]["mean_residual"]
        t30_n = summary["T30"]["n"]
        edt_n = summary["EDT"]["n"]
        c50_n = summary["C50"]["n"]

        # JND thresholds scale from the independent reference metric (not self-referential)
        # The threshold's reference must be averaged over the SAME scenes the
        # residual is (S-F1). `nanmean` here silently spanned every scene in the
        # split while the residual spans only the scored ones, so a split with
        # attrition compared a residual from one population against a JND scaled by
        # another — and the two populations differ by exactly the scenes whose
        # decay the estimator could not resolve.
        def _reference(metric: str) -> float:
            vals = [
                per_scene_residuals[sid]["reference"][metric]
                for sid, sp in splits.items()
                if sp == split_name and sid in per_scene_residuals
                and not np.isnan(per_scene_residuals[sid]["residual"][metric])
            ]
            return float(np.mean(vals)) if vals else float("nan")

        ref_t30 = _reference("T30")
        ref_edt = _reference("EDT")

        t30_thresh = config.d0b_t30_jnd_frac * ref_t30 if not np.isnan(ref_t30) else float("nan")
        edt_thresh = config.d0b_edt_jnd_frac * ref_edt if not np.isnan(ref_edt) else float("nan")
        c50_thresh = config.d0b_c50_jnd_db

        # COVERAGE IS PART OF THE VERDICT (S-F1). This degraded to N/A only at
        # n == 0, so a split that lost most of its scenes to unresolvable bands or
        # failed loads could still print PASS on whatever survived. The survivors
        # are not a random subset: a scene drops when the estimator cannot resolve
        # its decay, which correlates with absorption — an axis the shift splits
        # vary on purpose. Clearing a split on its most-measurable scenes is
        # exactly the false clearance D0b exists to prevent.
        n_declared = summary["n_scenes"]
        min_scored = config.d0b_min_scored_frac * n_declared

        def _verdict(r: float, thresh: float, n: int) -> str:
            if n == 0 or np.isnan(r) or np.isnan(thresh):
                return "N/A"
            if n < min_scored:
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

        # A residual averaged over a subset of the split is only interpretable
        # alongside how much of the split it covers (F-72). TWO axes of attrition,
        # both annotated: scenes dropped before scoring, and scored scenes whose
        # residual came back NaN for one metric — the latter thins a per-metric mean
        # without touching `n_scenes`, so keying the annotation on drops alone
        # printed a PASS over a subset with nothing said.
        n_dropped = len(summary["dropped"])
        thinned = {
            key: summary[key]["n"] for key in ("T30", "EDT", "C50")
            if summary[key]["n"] < summary["n_scenes"]
        }
        parts = []
        if n_dropped:
            parts.append(f"{summary['n_scenes']}/{summary['n_attempted']} scored, "
                         f"{n_dropped} dropped")
        if thinned:
            parts.append("per-metric n: " + ", ".join(
                f"{k} {v}/{summary['n_scenes']}" for k, v in thinned.items()))
        coverage = f"  [{'; '.join(parts)}]" if parts else ""
        emit(
            verbosity, "metrics",
            f"  {split_name:<28} "
            f"{t30_s:>12}({t30_v})  "
            f"{edt_s:>12}({edt_v})  "
            f"{c50_s:>12}({c50_v}){coverage}",
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
