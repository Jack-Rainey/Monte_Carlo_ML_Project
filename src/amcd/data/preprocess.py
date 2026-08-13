"""
preprocess stage:
  renders/ → energy tensors (encoded, normalized, split-assigned) → preprocessed/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..representations import build_representation
from ..runtime import RunContext, emit
from ..simulators.base import SceneSpec

#: The `compute_stats` keys actually applied to the saved tensors, as (mean, std).
#: ONE declaration, read both by the `normalize()` calls and by what `meta.json`
#: reports as applied, so the artifact cannot assert a normalization the tensors
#: did not get (F-165). Both legs take the HIGH stats — see `normalization.py`.
_APPLIED_STAT_KEYS = ("high_mean", "high_std")
from .normalization import compute_stats, normalize
from .splits import assign_split


def _spectral_slope_db_per_decade(
    energy_db: "torch.Tensor", center_freqs: list[float]
) -> float:
    """Least-squares slope of per-band peak level against log10(frequency), dB/decade.

    THE DISCLOSURE THE NARROWED HEADROOM GUARD DROPPED (RD-188). AC-37's guard first
    took a minimum across every band, which is a spectral-FLATNESS constraint: it
    rejected a steeply sloped render even when the two REPORTED metric bands had
    ample headroom. Narrowing its operand to those bands removed the false rejection
    — and removed the only signal that the slope existed, so a render whose top
    octaves are 40 dB down now passes without comment.

    Flat is 0. Negative is the usual direction (energy falling with frequency, which
    real air absorption and most sources produce). A strongly negative value is not
    an error and is deliberately NOT gated: it is a property of the render, and the
    reader of `preprocessed/meta.json` is owed it.

    Per-band PEAK over time, not mean: the guard's own operand is the peak
    (`amax(dim=2)`), and a slope computed on a different reduction than the guard
    applies would describe a different quantity than the one that was checked.
    Reduced over channels by taking the W channel's slope — index 0, the omni
    channel every reported ISO metric reads.

    Returns NaN for a representation with fewer than two bands, where a slope is
    undefined rather than zero.
    """
    import numpy as _np

    if len(center_freqs) < 2:
        return float("nan")
    peak = energy_db[0].amax(dim=1).detach().cpu().numpy()  # (n_bands,)
    logf = _np.log10(_np.asarray(center_freqs, dtype=float))
    finite = _np.isfinite(peak) & _np.isfinite(logf)
    if finite.sum() < 2:
        return float("nan")
    return float(_np.polyfit(logf[finite], peak[finite], 1)[0])


def run_preprocess(config: Config, run_dir: Path, ctx: RunContext) -> None:
    # RD-20: the runtime context, not a bare verbosity — see `amcd.runtime.RunContext`.
    verbosity = ctx.verbosity
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    out_dir = run_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous split assignment before writing this one (F-25). Split
    # membership can change between runs (a different split_assignment seed, a
    # different sizing mode), and a scene that MOVES leaves its tensor behind in
    # the old directory. That residue was then trained on while being scored as
    # held-out — with splits.json still reporting the correct assignment, so
    # nothing in the artifacts revealed it. Rewriting the directory set is the
    # producer half of the fix; EnergyDataset refusing unlisted tensors is the
    # consumer half, and neither is sufficient alone.
    for stale in out_dir.iterdir():
        if stale.is_dir() and stale.name != "carrier":
            shutil.rmtree(stale)
    # Create a directory for EVERY declared split, including ones that end up
    # empty (F-40). Otherwise a split that receives no scenes has no directory and
    # EnergyDataset raises "Split directory not found", pointing at the filesystem
    # instead of at the empty split that `split_counts` now reports as 0.
    for split_name in config.splits:
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Load scene specs
    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scene specs found in {scenes_dir}. Run gen-scenes first.")

    scenes = [SceneSpec.from_json(p) for p in scene_paths]

    # `carrier/` is excluded from the rmtree above because it is keyed by scene id
    # rather than by split, so it needs the same pruning gen-scenes gives
    # `scene_*.json` and render now gives `renders/` (F-47, widening F-38).
    # Measured before this: shrinking a run_dir from 29 scenes to 14 left 15 orphan
    # .npy here. Inert only because infer/eval look carriers up BY scene_id — an
    # invariant nothing stated or tested — and scene ids are POSITIONAL, so the
    # orphans occupy ids a later config reuses under different geometry.
    carrier_dir = out_dir / "carrier"
    carrier_dir.mkdir(parents=True, exist_ok=True)
    current_ids = {s.scene_id for s in scenes}
    pruned = 0
    for stale in carrier_dir.glob("*.npy"):
        if stale.stem not in current_ids:
            stale.unlink()
            pruned += 1
    if pruned:
        emit(verbosity, "progress", f"  Pruned {pruned} orphan carrier file(s) from {carrier_dir}")

    # Assign splits deterministically from scene spec hashes (config-declared set)
    splits: dict[str, str] = {
        s.scene_id: assign_split(s.to_dict(), config)
        for s in scenes
    }

    # The training split (role: train) is the sole source of normalization stats.
    # Its uniqueness is enforced at config load (REQUIRED_ROLE_COUNTS) rather than
    # here, so the failure lands before gen-scenes and render rather than after
    # them (F-44).
    train_split = config.the_split_with_role("train")

    # Build representation (rep-agnostic: params validated by the rep's own schema)
    rep = build_representation(
        config.representation.name, config.representation.params,
        sample_rate=config.sample_rate,
        eval_freqs_hz=[float(x) for x in config.iso_eval_freqs],
    )

    # Encode all IRs
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    # Per-scene, per-leg spectral slope (RD-188) — see the note at the encode call.
    slopes: dict[str, dict[str, float]] = {}
    for scene in scenes:
        render_dir = renders_dir / scene.scene_id
        low_ir = np.load(render_dir / "low.npy")    # (C, T) float32
        high_ir = np.load(render_dir / "high.npy")  # (C, T) float32

        # Shape invariant: storage is (C, T)
        assert low_ir.shape == (config.n_channels, config.n_samples), (
            f"Unexpected low IR shape {low_ir.shape} for {scene.scene_id}"
        )

        # NOT wrapped in try/except, and that is load-bearing (RD-188). The AC-37
        # headroom guard errs toward REJECTING a scene, which is only safe from
        # selection effects because a rejection aborts the whole run: swallowing it
        # per scene would silently drop scenes whose spectra sit near the floor, and
        # that population correlates with absorption — `test_material_shift`'s own
        # declared axis. `test_preprocess_does_not_swallow_an_encode_refusal` pins
        # the absence.
        low_energy = rep.encode(low_ir)    # (C, n_bands, n_frames)
        high_energy = rep.encode(high_ir)

        # WHAT THE NARROWED GUARD NO LONGER SEES (RD-188). AC-37's guard originally
        # took a minimum across ALL bands, which is a spectral-FLATNESS constraint:
        # a steeply sloped render fails it even when every REPORTED band has ample
        # headroom, so the operand was narrowed to the reported metric bands (F-M3).
        # That removed the false rejection and removed the disclosure with it — a
        # 2nd-order 4 kHz lowpass now passes SILENTLY where it previously failed
        # loudly, and the slope is a real property of the render.
        #
        # So the slope is RECORDED rather than gated: nothing about it should reject
        # a scene, and nothing should hide it either.
        slopes[scene.scene_id] = {
            leg: _spectral_slope_db_per_decade(energy, rep.center_freqs)
            for leg, energy in (("low", low_energy), ("high", high_energy))
        }

        encoded[scene.scene_id] = (low_energy, high_energy, low_ir)

    # Compute norm stats from TRAINING split only (invariant #3)
    train_ids = [sid for sid, sp in splits.items() if sp == train_split]
    if not train_ids:
        raise RuntimeError(
            f"Training split {train_split!r} is empty — increase scenes.n_id or adjust fracs."
        )

    train_lows = [encoded[sid][0] for sid in train_ids]
    train_highs = [encoded[sid][1] for sid in train_ids]
    norm_stats = compute_stats(train_lows, train_highs)

    # Derive shape metadata from first tensor
    sample_low = encoded[scenes[0].scene_id][0]
    n_bands = sample_low.shape[1]
    n_frames = sample_low.shape[2]

    # Save normalized tensors per split
    for scene in scenes:
        split = splits[scene.scene_id]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        low_energy, high_energy, low_ir = encoded[scene.scene_id]

        # Both inputs and targets normalized with high stats so the residual skip
        # connection (pred = low + model(low)) lives in the same space as the target.
        # The stat KEYS come from _APPLIED_STAT_KEYS, which is also what meta.json
        # reports as applied — so the artifact cannot claim one thing while the
        # tensors were normalized with another (F-165).
        mean_key, std_key = _APPLIED_STAT_KEYS
        norm_low = normalize(low_energy, norm_stats[mean_key], norm_stats[std_key])
        norm_high = normalize(high_energy, norm_stats[mean_key], norm_stats[std_key])

        torch.save(norm_low, split_dir / f"{scene.scene_id}_low.pt")
        torch.save(norm_high, split_dir / f"{scene.scene_id}_high.pt")

        # Save carrier (raw low-ray IR for D3 reconstruction). The directory is
        # created and pruned once, above.
        np.save(carrier_dir / f"{scene.scene_id}.npy", low_ir)

    # Save metadata. Counts are keyed on the CONFIG-DECLARED split set, defaulting
    # to 0, not on the splits that happen to have received a scene (F-30): keying
    # on observed values made the empty-split warning below unreachable, so a
    # declared split that received nothing simply vanished from the record instead
    # of being reported as 0.
    all_split_names = list(config.splits)
    meta = {
        "n_channels": config.n_channels,
        "n_bands": n_bands,
        "n_frames": n_frames,
        "center_freqs": rep.center_freqs,
        # What a "band" actually IS for this run — edges, bin counts, the covered
        # range, the power that reaches no band, and (for banded reps that can
        # measure it) the per-band in-band energy fraction (AC-19). Recorded
        # rather than implied: `center_freqs` alone is an IRREGULAR series once
        # under-resolved bands are dropped, and reads as a third-octave ladder
        # when it is not one. Reps that expose no band structure omit the key.
        **(
            {"band_description": rep.describe_bands()}
            if hasattr(rep, "describe_bands") else {}
        ),
        # Declared domain of the saved tensors ("db" | "amplitude"); dB-assuming
        # eval consumers key on this stamp, never on the rep class (F-19).
        "value_domain": rep.value_domain,
        # RD-188: per-scene, per-leg spectral slope in dB/decade. Recorded, never
        # gated — see the note at the encode call for what the narrowed AC-37 guard
        # stopped seeing.
        "spectral_slope_db_per_decade": slopes,
        "norm_stats": norm_stats,
        # WHICH of those four were applied, in the file rather than in a comment no
        # reader of meta.json sees (F-M11). Derived from the keys actually used
        # above, never asserted independently of them — see `normalization.py` for
        # why the unapplied pair is kept.
        "norm_stats_applied": {
            "applied_to_both_legs": list(_APPLIED_STAT_KEYS),
            "recorded_only": [
                k for k in sorted(norm_stats) if k not in _APPLIED_STAT_KEYS
            ],
            "note": (
                "The residual framing (pred = low + model(low)) needs input and "
                "target in ONE affine frame, so both legs use the high stats "
                "(F-02). The recorded_only stats are applied to nothing; they are "
                "the record of how the low-ray leg's distribution differs from the "
                "high leg (F-M11)."
            ),
        },
        "split_counts": {
            sp: sum(1 for s in splits.values() if s == sp)
            for sp in all_split_names
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    counts = meta["split_counts"]
    counts_str = ", ".join(f"{sp}={n}" for sp, n in counts.items())
    emit(verbosity, "progress", f"  Preprocessed {len(scenes)} scenes ({counts_str}) → {out_dir}")
    emit(
        verbosity, "metrics",
        f"  Energy tensors: (C={config.n_channels}, bands={n_bands}, frames={n_frames})",
    )

    for sp, count in counts.items():
        if count == 0:
            emit(
                verbosity, "warning",
                f"  WARNING: {sp} split is empty. Increase scenes.n_id or adjust fracs.",
            )
