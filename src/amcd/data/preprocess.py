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
from ..simulators.base import SceneSpec, admitted_digest

#: The `compute_stats` keys actually applied to the saved tensors, as (mean, std).
#: ONE declaration, read both by the `normalize()` calls and by what `meta.json`
#: reports as applied, so the artifact cannot assert a normalization the tensors
#: did not get. Both legs take the HIGH stats — see `normalization.py`.
_APPLIED_STAT_KEYS = ("high_mean", "high_std")
from .normalization import compute_stats, normalize
from .splits import assign_split


def _spectral_slope_db_per_decade(
    energy_db: "torch.Tensor", center_freqs: list[float]
) -> float:
    """Least-squares slope of per-band peak level against log10(frequency), dB/decade.

    WHAT THE HEADROOM GUARD DOES NOT SEE. The guard (`min_db` in
    `configs/representations/spectrogram.yaml`, applied in `rep.encode`) has the
    REPORTED metric bands as its operand, so a render whose top octaves are 40 dB
    down clears it: spectral tilt outside those bands is unchecked. This records
    the tilt so it is not also unreported.

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


def _admitted_scenes(
    generated: list[SceneSpec], renders_dir: Path
) -> tuple[list[SceneSpec], dict[str, dict], str]:
    """Split the generated scenes into the admitted ones and the excluded ones.

    Returns `(admitted, excluded, admitted_sha256)` — the digest identifies the
    MEMBERSHIP this run's tensors were built from, and is stamped into
    `preprocessed/meta.json` so a later reader can tell which dataset a split
    describes rather than assuming the manifest on disk is still the one used.

    A MISSING MANIFEST IS FATAL, never "assume everything was admitted". That
    fallback would silently reinstate the whole batch on a run whose render stage
    predates exclusion, which is exactly the case where the difference matters.
    """
    manifest_path = renders_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No render manifest at {manifest_path}. It records which scenes were "
            f"ADMITTED to the dataset, so without it there is no way to tell an "
            f"excluded render from an admitted one — both have artifacts on disk. "
            f"Re-run the render stage (already-rendered scenes are reused)."
        )
    manifest = json.loads(manifest_path.read_text())

    # The digest is CHECKED, not just carried. `render` writes it beside the id
    # list it describes, so the two disagreeing means the file was edited after the
    # stage wrote it — by hand, by a partial write, or by a tool. Verifying it here
    # is what makes it a integrity record rather than decoration.
    recorded_digest = manifest.get("admitted_sha256")
    actual_digest = admitted_digest(manifest["admitted"])
    if recorded_digest != actual_digest:
        raise RuntimeError(
            f"{manifest_path} records admitted_sha256 {recorded_digest} but its own "
            f"admitted list digests to {actual_digest}. The file has been modified "
            f"since the render stage wrote it, so it no longer describes the "
            f"renders on disk. Re-run the render stage (rendered scenes are reused)."
        )

    excluded = {e["scene_id"]: e for e in manifest["excluded"]}
    admitted = [s for s in generated if s.scene_id not in excluded]

    listed = set(manifest["admitted"])
    if {s.scene_id for s in admitted} != listed:
        raise RuntimeError(
            f"{manifest_path} lists {len(listed)} admitted scenes, but the scene "
            f"specs in the run directory resolve to "
            f"{len({s.scene_id for s in admitted})}. The manifest describes a "
            f"different batch than gen-scenes produced — re-run both stages rather "
            f"than preprocessing a set neither stage agrees on."
        )
    return admitted, excluded, actual_digest


def _split_attrition(
    generated: list[SceneSpec], excluded: dict[str, dict], config: Config
) -> dict[str, dict]:
    """Per split: how many scenes were generated, admitted, and lost to what.

    Counts EXCLUDED scenes in the design_spec §11.1a sense — never in the dataset.
    A scene that is admitted but carries an unscored metric is not here; it is in
    `metrics/drops.csv` and in the split's scored/attempted count, and folding the
    two into one denominator states something false about each.

    Computed over the GENERATED scenes, not the admitted ones, because the
    denominator is the whole point — "112 admitted" is not a disclosure, "112 of
    120" is. Split assignment is per-scene (a hash of the spec, or direct routing
    by `split_regime`), never positional, so an excluded scene leaves the survivors
    where they were and this denominator is the one the split would have had.
    """
    counts = {
        name: {"generated": 0, "admitted": 0, "excluded": 0, "refused": 0, "qc_failed": 0}
        for name in config.splits
    }
    for scene in generated:
        row = counts[assign_split(scene.to_dict(), config)]
        row["generated"] += 1
        entry = excluded.get(scene.scene_id)
        if entry is None:
            row["admitted"] += 1
        else:
            row["excluded"] += 1
            row[entry["category"]] += 1
    return counts


def _enforce_per_split_bound(config: Config, attrition: dict[str, dict]) -> None:
    """Refuse a batch whose attrition CONCENTRATES on one split.

    The bound that actually binds. QC's energy floor bites hardest where
    absorption is highest, which is `test_material_shift`'s `ceiling_absorptive`
    regime by construction, so exclusions are expected to correlate with the very
    axes the shift splits vary. 5 lost scenes is 0.7 % of 720 and 4.2 % of a
    120-scene split: a global bound is nearly blind to the case where attrition IS
    the confound.

    Checked here rather than in `render` because split membership is assigned here.
    Putting it in the render stage would fold split logic into that stage's
    fingerprint, so changing how splits are assigned would re-render every scene to
    learn something that costs seconds downstream.
    """
    bound = config.max_excluded_frac_per_split
    breaches = [
        f"  {name}: {row['excluded']} of {row['generated']} excluded "
        f"({row['excluded'] / row['generated']:.1%}), over the declared {bound:.1%} "
        f"({row['qc_failed']} failed QC, {row['refused']} refused by the backend)"
        for name, row in attrition.items()
        if row["generated"] and row["excluded"] / row["generated"] > bound
    ]
    if breaches:
        raise ValueError(
            "attrition is concentrated on one or more splits, so the surviving "
            "scenes are a selected subset rather than a sample of the split they "
            "are named after:\n"
            + "\n".join(breaches)
            + "\nEach split is reported against the others, so a per-split "
            "difference here would be partly an admission effect and partly the "
            "shift under study, with no way to separate them. Every excluded scene "
            "and its reason is in renders/manifest.json. Nothing is lost by "
            "stopping: the renders are on disk and preprocess re-runs in seconds."
        )


def run_preprocess(config: Config, run_dir: Path, ctx: RunContext) -> None:
    verbosity = ctx.verbosity
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    out_dir = run_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous split assignment before writing this one. Split
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
    # empty. Otherwise a split that receives no scenes has no directory and
    # EnergyDataset raises "Split directory not found", pointing at the filesystem
    # instead of at the empty split that `split_counts` now reports as 0.
    for split_name in config.splits:
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Load scene specs
    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scene specs found in {scenes_dir}. Run gen-scenes first.")

    generated = [SceneSpec.from_json(p) for p in scene_paths]
    # THE MANIFEST IS THE DATASET, not the set of scenes that were generated and
    # not the set of directories under renders/. A scene excluded by QC still has
    # its artifacts on disk — that is what makes the exclusion re-derivable — so
    # anything that discovered scenes by listing would train on renders the render
    # stage refused to admit.
    scenes, excluded, membership_sha = _admitted_scenes(generated, renders_dir)

    # `carrier/` is excluded from the rmtree above because it is keyed by scene id
    # rather than by split, so it needs the same pruning gen-scenes gives
    # `scene_*.json` and render gives `renders/`. Shrinking a run_dir leaves
    # orphan .npy here otherwise, and scene ids are POSITIONAL, so those orphans
    # occupy ids a later config reuses under different geometry.
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

    attrition = _split_attrition(generated, excluded, config)
    _enforce_per_split_bound(config, attrition)
    if excluded:
        emit(verbosity, "progress",
             f"  {len(excluded)} of {len(generated)} scenes were excluded at render; "
             f"per-split attrition is in preprocessed/meta.json and the report footer")

    # The training split (role: train) is the sole source of normalization stats.
    # Its uniqueness is enforced at config load (REQUIRED_ROLE_COUNTS) rather than
    # here, so the failure lands before gen-scenes and render rather than after
    # them.
    train_split = config.the_split_with_role("train")

    # Build representation (rep-agnostic: params validated by the rep's own schema)
    rep = build_representation(
        config.representation.name, config.representation.params,
        sample_rate=config.sample_rate,
        eval_freqs_hz=[float(x) for x in config.iso_eval_freqs],
    )

    # Encode all IRs
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    # Per-scene, per-leg spectral slope — see the note at the encode call.
    slopes: dict[str, dict[str, float]] = {}
    for scene in scenes:
        render_dir = renders_dir / scene.scene_id
        low_ir = np.load(render_dir / "low.npy")    # (C, T) float32
        high_ir = np.load(render_dir / "high.npy")  # (C, T) float32

        # Shape invariant: storage is (C, T)
        assert low_ir.shape == (config.n_channels, config.n_samples), (
            f"Unexpected low IR shape {low_ir.shape} for {scene.scene_id}"
        )

        # NOT wrapped in try/except, and that is load-bearing. The headroom
        # guard errs toward REJECTING a scene, which is only safe from selection
        # effects because a rejection aborts the whole run: swallowing it
        # per scene would silently drop scenes whose spectra sit near the floor, and
        # that population correlates with absorption — `test_material_shift`'s own
        # declared axis. `test_preprocess_does_not_swallow_an_encode_refusal` pins
        # the absence.
        low_energy = rep.encode(low_ir)    # (C, n_bands, n_frames)
        high_energy = rep.encode(high_ir)

        # Recorded, never gated — see `_spectral_slope_db_per_decade` for what the
        # headroom guard's operand leaves unchecked.
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
        # tensors were normalized with another.
        mean_key, std_key = _APPLIED_STAT_KEYS
        norm_low = normalize(low_energy, norm_stats[mean_key], norm_stats[std_key])
        norm_high = normalize(high_energy, norm_stats[mean_key], norm_stats[std_key])

        torch.save(norm_low, split_dir / f"{scene.scene_id}_low.pt")
        torch.save(norm_high, split_dir / f"{scene.scene_id}_high.pt")

        # Save carrier (raw low-ray IR for D3 reconstruction). The directory is
        # created and pruned once, above.
        np.save(carrier_dir / f"{scene.scene_id}.npy", low_ir)

    # Save metadata. Counts are keyed on the CONFIG-DECLARED split set, defaulting
    # to 0, not on the splits that happen to have received a scene: keying
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
        # measure it) the per-band in-band energy fraction. Recorded
        # rather than implied: `center_freqs` alone is an IRREGULAR series once
        # under-resolved bands are dropped, and reads as a third-octave ladder
        # when it is not one. Reps that expose no band structure omit the key.
        **(
            {"band_description": rep.describe_bands()}
            if hasattr(rep, "describe_bands") else {}
        ),
        # Declared domain of the saved tensors ("db" | "amplitude"); dB-assuming
        # eval consumers key on this stamp, never on the rep class.
        "value_domain": rep.value_domain,
        # Per-scene, per-leg spectral slope in dB/decade. Recorded, never gated —
        # see `_spectral_slope_db_per_decade`.
        "spectral_slope_db_per_decade": slopes,
        "norm_stats": norm_stats,
        # WHICH of those four were applied, in the file rather than in a comment no
        # reader of meta.json sees. Derived from the keys actually used
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
                ". The recorded_only stats are applied to nothing; they are "
                "the record of how the low-ray leg's distribution differs from the "
                "high leg."
            ),
        },
        "split_counts": {
            sp: sum(1 for s in splits.values() if s == sp)
            for sp in all_split_names
        },
        # Per split: generated, admitted, and what the difference was lost to.
        # Recorded beside `split_counts` because that count alone cannot say
        # whether a small split was SPECIFIED small or ARRIVED small — and the
        # scenes QC excludes are not a random subset of the split, since the energy
        # floor bites hardest at high absorption, which is a shift axis.
        "split_attrition": attrition,
        # WHICH membership these tensors were built from. Stamped so a
        # reader can tell whether a split describes the manifest now on
        # disk, rather than assuming it does.
        "admitted_sha256": membership_sha,
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
