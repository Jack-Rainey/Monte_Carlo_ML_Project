"""render stage: the paired low/high render of every scene spec.

Writes per scene, under `renders/<scene_id>/`:
  low.npy, high.npy         (n_channels, n_samples) float32, the two ray budgets
  paths_{low,high}.parquet  retained propagation paths, only for backends that
                            export them
  meta.json                 canonical provenance incl. `artifact_sha256`
                            and the fingerprint the pair was rendered under

and, for the batch, `renders/manifest.json` and `renders/qc_failures.csv`
(both canonical) plus `renders/qc_record.csv` at save level 4.

Nothing already rendered is rendered twice: the batch is validated up front, and
a scene whose artifacts carry the current fingerprint and scene digest and still
verify is reused rather than re-rendered.

A BAD SCENE IS EXCLUDED; A BAD BACKEND ABORTS. Exclusion here is EXCLUDED in the
sense design_spec §11.1a defines — never in the dataset, as distinct from a scene
that is admitted and carries one unscored metric. Research I §B.4 excludes examples
that fail the admission criteria rather than discarding the dataset, and this
stage does the same: a scene the backend REFUSED (`SceneRefused`) or that failed a
gating QC criterion is left out of `manifest.json`, which is what downstream reads
as the dataset. Everything else — a missing provenance key, an IR of the wrong
shape, a scene that wrote no artifacts, a wrong channel count — means the backend
is misconfigured rather than the scene being bad, so every remaining scene would
fail identically and the run aborts on the first one.

Exclusion is bounded, because it has no natural floor: `max_excluded_frac` and
`max_refused_frac` stop a broken backend from quietly producing a small dataset
instead of an error. The per-split bound lives in `preprocess`, where splits are
assigned.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..runtime import RunContext, emit
from . import qc
from .base import (
    IRResult,
    _sha256,
    verify_render_artifacts,  # noqa: F401 — re-exported for existing importers
    SceneRefused,
    SceneSpec,
    admitted_digest,
    build_simulator,
    simulator_host_scoped_params,
    simulator_min_separation,
    validate_path_descriptor,
    validate_provenance,
)


def _qc_coverage(criteria: list[qc.QCRecord]) -> dict[str, dict[str, int]]:
    """Per gating criterion: how many were SCORED, out of how many attempted.

    The `n scored / attempted` discipline the metric path already applies, applied
    to admission. Without it a criterion can go unscored on the whole batch and
    read as clean: `manifest.json` shows every scene admitted, no exclusion is
    recorded, and the attrition bounds see nothing, because a skip is not a
    failure. RI's paired onset check is the live case — geometry overrules the
    detector on both legs and the record becomes unscored — so the criterion the
    reproduction is named for can be silently absent from the dataset it admits.
    """
    coverage: dict[str, dict[str, int]] = {}
    for record in criteria:
        row = coverage.setdefault(record.criterion, {"scored": 0, "attempted": 0})
        row["attempted"] += 1
        row["scored"] += int(record.scored)
    return coverage


def _write_manifest(
    renders_dir: Path,
    scenes: list[SceneSpec],
    refused: list[tuple[str, str]],
    failures: list[qc.QCRecord],
    criteria: list[qc.QCRecord],
) -> dict:
    """Write `renders/manifest.json` — WHICH SCENES THE DATASET IS MADE OF.

    The authority on membership, and the reason exclusion is safe. Downstream
    stages read this rather than listing `renders/`, so a scene that failed QC
    cannot reach training by virtue of its directory still being on disk, and the
    set that WAS used is recorded rather than reconstructed.

    Excluded entries carry their CATEGORY, because the two are not the same fact.
    A `qc_failed` scene has artifacts on disk and its exclusion re-derives from
    them, so a re-run reproduces the membership exactly. A `refused` scene has no
    artifacts, so a re-run re-renders it and — the backend exposing no RNG seed —
    may admit it. `admitted_sha256` exists for that: it digests the admitted id
    set, so membership drifting between runs is detectable instead of silent.

    Canonical at every save level: it is what a result means, not observability.
    """
    excluded_by_qc: dict[str, list[dict]] = {}
    for r in failures:
        excluded_by_qc.setdefault(r.scene_id, []).append({
            "leg": r.leg, "criterion": r.criterion,
            "measured": r.measured, "threshold": r.threshold,
        })
    excluded = (
        [{"scene_id": sid, "category": "refused", "reason": reason}
         for sid, reason in refused]
        + [{"scene_id": sid, "category": "qc_failed", "criteria": crit}
           for sid, crit in sorted(excluded_by_qc.items())]
    )
    dropped = {e["scene_id"] for e in excluded}
    admitted = [s.scene_id for s in scenes if s.scene_id not in dropped]

    manifest = {
        "generated": len(scenes),
        "admitted": admitted,
        "admitted_sha256": admitted_digest(admitted),
        "excluded": sorted(excluded, key=lambda e: e["scene_id"]),
        # Which admission criteria actually RAN, per criterion. An admitted set is
        # only as meaningful as the checks that admitted it.
        "qc_coverage": _qc_coverage(criteria),
    }
    (renders_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _enforce_attrition_bounds(config: Config, manifest: dict) -> None:
    """Refuse a batch that lost more than the config allows.

    Per-example exclusion has no natural floor: without this, a backend
    misconfigured in a way that survives the contract checks excludes every scene
    individually and produces a small dataset that still trains and still reports.

    Only the GLOBAL bounds are checked here. The per-split bound — the one that
    actually binds, since attrition concentrating on a shift axis is a selection
    effect on that axis — is enforced in `preprocess`, which is where split
    membership is assigned. Checking it here would put split assignment into the
    render stage's fingerprint, so a change to split logic would re-render 720
    scenes to discover a fact that costs seconds downstream.
    """
    generated = manifest["generated"]
    if not generated:
        return
    n_refused = sum(1 for e in manifest["excluded"] if e["category"] == "refused")
    breaches = []
    for label, count, bound in (
        ("excluded", len(manifest["excluded"]), config.max_excluded_frac),
        ("refused by the backend", n_refused, config.max_refused_frac),
    ):
        if count / generated > bound:
            breaches.append(
                f"  {count} of {generated} scenes {label} "
                f"({count / generated:.1%}), over the declared {bound:.1%}"
            )
    # A criterion that never RAN is not a criterion that passed. Bounded per
    # criterion rather than over the pooled total: pooling lets three healthy
    # criteria mask a fourth that scored nothing, and it is the fourth — RI's
    # paired onset check, say — whose absence changes what "admitted" means.
    for criterion, row in sorted(manifest["qc_coverage"].items()):
        if not row["attempted"]:
            continue
        unscored = 1.0 - row["scored"] / row["attempted"]
        if unscored > config.max_unscored_gating_frac:
            breaches.append(
                f"  QC criterion {criterion!r} scored only {row['scored']} of "
                f"{row['attempted']} attempts ({unscored:.1%} unscored), over the "
                f"declared {config.max_unscored_gating_frac:.1%}"
            )

    if breaches:
        raise ValueError(
            "the render stage lost more of the batch than the dataset can carry:\n"
            + "\n".join(breaches)
            + f"\nEvery excluded scene and its reason is in "
            f"{'renders/manifest.json'}, and every failed criterion with its "
            f"measurement is in renders/qc_failures.csv. Attrition at this scale is "
            f"a statement about the backend or the thresholds, not about the scenes: "
            f"diagnose it before raising the bound. Nothing is lost by stopping — "
            f"every rendered scene verifies on disk, so a re-run reuses it and "
            f"re-scoring costs seconds."
        )






def _preflight_separations(config: Config, scenes: list[SceneSpec]) -> None:
    """Reject the whole batch before rendering any of it, listing every offender.

    Scenes on disk may have been generated under a different backend or an older
    config, so the backend's separation floor is re-checked against realized
    separations here rather than trusted from generation time.
    """
    floor = simulator_min_separation(config)
    if floor <= 0.0:
        return
    offenders = []
    for scene in scenes:
        d = float(
            np.linalg.norm(
                np.asarray(scene.source_pos, dtype=np.float64)
                - np.asarray(scene.receiver_pos, dtype=np.float64)
            )
        )
        if d < floor:
            offenders.append((scene.scene_id, d))
    if offenders:
        lines = "\n".join(f"    {sid}: {d:.4f} m" for sid, d in offenders)
        raise ValueError(
            f"{len(offenders)} of {len(scenes)} scenes have a source-receiver "
            f"separation below simulator {config.simulator.name!r}'s floor of "
            f"{floor} m, so none were rendered:\n{lines}\n"
            f"Raise the `distance_range` lower bound on the offending placement "
            f"regime(s) and regenerate scenes."
        )


def _canonical_meta(
    config: Config,
    scene: SceneSpec,
    low: IRResult,
    high: IRResult,
    artifact_sha256: dict[str, str],
    artifact_fingerprint: str | None,
    wall_clock_s: dict[str, float],
) -> dict:
    """The provenance record for one scene's rendered pair.

    Simulator-agnostic: the stage contributes the resolved config context, and
    each leg's backend specifics come from the simulator's own `IRResult.meta`.
    """
    host_scoped = simulator_host_scoped_params(config)
    params = {k: v for k, v in config.simulator.params.items() if k not in host_scoped}
    return {
        "scene_id": scene.scene_id,
        # WHICH ROOM these bytes are of. Scene ids are positional, so `scene_0000`
        # names a different room after a regeneration, and nothing in the artifact
        # fingerprint mentions the scene — without this a resumed run would serve
        # the previous scene set's IRs under the new set's ids.
        "scene_sha256": _scene_sha256(scene),
        "simulator": {"name": config.simulator.name, "params": params},
        "artifact_sha256": artifact_sha256,
        # What these BYTES were rendered under, so a resumed run can tell a
        # reusable scene from one belonging to a different config. Deliberately
        # not the whole render fingerprint: the QC thresholds are in that and do
        # not change the IR, so re-scoring them must not force a re-render.
        "artifact_fingerprint": artifact_fingerprint,
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "n_channels": config.n_channels,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
        # One realization per (scene, budget) today. Recorded rather than implied
        # so E4's repeated realizations are a value here, not a file rename.
        "realization_index": 0,
        # Per-leg render wall-clock, in seconds. The measured price of this
        # dataset: design_spec §11.1 requires a render-count request above the
        # standing threshold to quote a MEASURED estimate, and E4 prices its ray
        # budgets off the same numbers. Only the run that renders a scene can
        # collect this, so it is canonical rather than observability.
        "wall_clock_s": wall_clock_s,
        "low": low.meta,
        "high": high.meta,
    }


def _scene_sha256(scene: SceneSpec) -> str:
    """Digest of the scene SPEC — which room these bytes are of."""
    return hashlib.sha256(
        json.dumps(scene.to_dict(), sort_keys=True).encode()
    ).hexdigest()


def _reusable(out_dir: Path, artifact_fingerprint: str | None, scene: SceneSpec) -> bool:
    """Whether `out_dir` already holds THIS scene's render under this config, intact.

    Three arms, and all three are load-bearing:

    * the recorded artifact fingerprint matches — same config inputs produced it;
    * the recorded scene digest matches — it is the same ROOM, which the
      fingerprint cannot say because scene ids are positional;
    * every recorded artifact digest still verifies — the bytes are whole, which
      neither of the above can say for a directory a killed run half-wrote.

    An absent `scene_sha256` is a mismatch, not a pass: a run_dir written before
    the key existed re-renders rather than silently reusing.

    Returns False rather than raising on a missing or unreadable meta.json — an
    unusable directory is simply re-rendered.
    """
    if artifact_fingerprint is None:
        return False
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        recorded = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    if recorded.get("artifact_fingerprint") != artifact_fingerprint:
        return False
    if recorded.get("scene_sha256") != _scene_sha256(scene):
        return False
    digests = recorded.get("artifact_sha256") or {}
    if not digests:
        return False
    return all(
        (out_dir / name).exists() and _sha256(out_dir / name) == digest
        for name, digest in digests.items()
    )


def _path_row_count(path: Path) -> int | None:
    """Rows in a retained-path parquet, from the footer alone (no data read)."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _expected_onset_samples(meta: dict, sample_rate: int) -> dict[str, int | None]:
    """Per leg, `floor(|src - rcv| / c * fs)` from that leg's own provenance.

    Read from what the RENDERER recorded rather than recomputed from the scene
    spec, so the backend's own speed of sound is used — gsound compiles 344.0 m/s
    while `amcd.acoustics` declares 343.0. `evaluation/evaluator.py` derives the
    same quantity the same way for the reported metric path.

    PER LEG, because the QC criterion's job is to catch a pair whose halves do not
    describe the same geometry; reading one leg and applying it to both would
    define that defect away. None for a leg whose backend declares neither, which
    the criterion records as a skip.
    """
    out: dict[str, int | None] = {}
    for leg in ("low", "high"):
        record = meta.get(leg, {})
        distance_m, speed = record.get("distance_m"), record.get("speed_of_sound_m_s")
        out[leg] = (
            None if distance_m is None or not speed
            else int(float(distance_m) / float(speed) * sample_rate)
        )
    return out


def _score(config: Config, scene_id: str, out_dir: Path) -> list[qc.QCRecord]:
    """Score one scene's PERSISTED artifacts against the QC criteria.

    Reads from disk on both paths — a freshly rendered scene and a reused one —
    so QC judges what the dataset actually contains, and so a re-scoring run needs
    nothing but the render directory.
    """
    meta = json.loads((out_dir / "meta.json").read_text())
    legs = {leg: np.load(out_dir / f"{leg}.npy") for leg in ("low", "high")}

    # Path files come from the DIGEST RECORD, not from a directory listing: an
    # unrecorded `paths_{leg}.parquet` left by an earlier run under a different
    # `path_retention` is outside provenance. `run_render` reports such a file
    # rather than deleting it (see the WARNING it emits), so it can still be sitting
    # in the directory; scoring an admission criterion against it is what this
    # avoids.
    recorded = set(meta.get("artifact_sha256") or {})
    path_files = {
        leg: (out_dir / f"paths_{leg}.parquet"
              if f"paths_{leg}.parquet" in recorded else None)
        for leg in ("low", "high")
    }
    return qc.score_scene(
        scene_id,
        legs=legs,
        ray_budgets={"low": config.low_ray_budget, "high": config.high_ray_budget},
        path_files=path_files,
        path_row_counts={
            leg: None if p is None else _path_row_count(p)
            for leg, p in path_files.items()
        },
        sample_rate=config.sample_rate,
        onset_rel_db=config.metric_onset_rel_db,
        expected_onset_samples=_expected_onset_samples(meta, config.sample_rate),
        onset_tolerance_samples=int(round(
            config.metric_onset_tolerance_ms / 1000.0 * config.sample_rate
        )),
        onset_mismatch_tolerance_ms=config.onset_mismatch_tolerance_ms,
        min_energy_db=config.min_energy_db,
        min_energy_reference=config.min_energy_reference,
        max_path_file_mb=config.max_path_file_mb,
        require_non_empty_path_file=config.require_non_empty_path_file,
    )


def run_render(config: Config, run_dir: Path, ctx: RunContext) -> None:
    verbosity = ctx.verbosity
    scenes_dir = run_dir / "scenes"
    renders_dir = run_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    sim = build_simulator(
        config.simulator.name,
        config.simulator.params,
        n_channels=config.n_channels,
        n_samples=config.n_samples,
        sample_rate=config.sample_rate,
    )

    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scene specs found in {scenes_dir}. Run gen-scenes first.")

    scenes = [SceneSpec.from_json(p) for p in scene_paths]

    _preflight_separations(config, scenes)

    # Scene ids are positional, so regenerating with fewer scenes leaves
    # high-numbered orphans that a later config would reuse under a different
    # geometry.
    current_ids = {scene.scene_id for scene in scenes}
    pruned = 0
    for stale in renders_dir.iterdir():
        if stale.is_dir() and stale.name not in current_ids:
            shutil.rmtree(stale)
            pruned += 1
    if pruned:
        emit(verbosity, "progress", f"  Pruned {pruned} orphan render dir(s) from {renders_dir}")

    # Reported, not deleted: an unrecorded file is evidence of whatever wrote it.
    for kept in sorted(renders_dir.iterdir()):
        if not kept.is_dir() or kept.name not in current_ids:
            continue
        meta_path = kept / "meta.json"
        if not meta_path.exists():
            continue
        recorded = set(json.loads(meta_path.read_text()).get("artifact_sha256", {}))
        unexpected = sorted(
            f.name for f in kept.iterdir()
            if f.is_file() and f.name != "meta.json" and f.name not in recorded
            and not f.name.startswith("._")
        )
        if unexpected:
            emit(verbosity, "warning",
                 f"  WARNING: {kept} holds {unexpected}, which this run did not "
                 f"write and meta.json does not record. A file no digest covers is "
                 f"outside provenance — most likely left by an earlier run under a "
                 f"different `path_retention`.")

    refused: list[tuple[str, str]] = []
    qc_records: list[qc.QCRecord] = []
    elapsed: list[float] = []
    reused = 0

    for scene in scenes:
        out_dir = renders_dir / scene.scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if not ctx.force and _reusable(out_dir, ctx.artifact_fingerprint_sha, scene):
            reused += 1
            qc_records += _score(config, scene.scene_id, out_dir)
            continue

        try:
            t0 = time.monotonic()
            low_result = sim.render(scene, config.low_ray_budget)
            t1 = time.monotonic()
            high_result = sim.render(scene, config.high_ray_budget)
            t2 = time.monotonic()
        except SceneRefused as exc:
            # ONLY SceneRefused is per-scene. Every other error class says the
            # BACKEND is wrong, not the scene, so it propagates and aborts rather
            # than excluding all 720 scenes one at a time.
            refused.append((scene.scene_id, str(exc)))
            emit(verbosity, "progress", f"  REFUSED {scene.scene_id}: {exc}")
            continue

        for leg, result in (("low", low_result), ("high", high_result)):
            validate_provenance(
                result.meta,
                simulator_name=config.simulator.name,
                scene_id=scene.scene_id,
                leg=leg,
            )

        # `raise`, not `assert` — `python -O` strips asserts.
        expected_shape = (config.n_channels, config.n_samples)
        for leg, result in (("low", low_result), ("high", high_result)):
            if result.ir.shape != expected_shape:
                raise ValueError(
                    f"scene {scene.scene_id!r} leg {leg!r}: simulator "
                    f"{config.simulator.name!r} returned an IR of shape "
                    f"{result.ir.shape}, expected {expected_shape} "
                    f"(n_channels, n_samples) from the resolved config."
                )

        # Accumulated as written, never a listing of out_dir: a scan would sweep
        # up host artifacts (macOS AppleDouble `._low.npy` sidecars) and put a
        # host fact into provenance.
        written: list[str] = []

        np.save(out_dir / "low.npy", low_result.ir)
        np.save(out_dir / "high.npy", high_result.ir)
        written += ["low.npy", "high.npy"]

        # Keyed on the field, never on the simulator's type: a backend that
        # exports no paths writes none and needs no downstream edit.
        for leg, result in (("low", low_result), ("high", high_result)):
            if result.paths is None:
                continue
            # The producer knows its ray budget; the stage owns the leg's label.
            result.paths.descriptor["leg"] = leg
            validate_path_descriptor(
                result.paths, simulator_name=config.simulator.name, scene_id=scene.scene_id
            )
            result.paths.to_parquet(out_dir / f"paths_{leg}.parquet")
            written.append(f"paths_{leg}.parquet")

        # After the last write, before the meta.json that carries them.
        digests = {name: _sha256(out_dir / name) for name in sorted(written)}
        if not digests:
            raise ValueError(
                f"scene {scene.scene_id!r} wrote no artifacts, so its meta.json "
                f"would carry an empty `artifact_sha256` — indistinguishable from "
                f"a render whose integrity record was simply never populated."
            )

        # Canonical provenance, never verbosity-gated (docs/verbosity.md).
        (out_dir / "meta.json").write_text(
            json.dumps(
                _canonical_meta(
                    config, scene, low_result, high_result, digests,
                    ctx.artifact_fingerprint_sha,
                    {"low": t1 - t0, "high": t2 - t1},
                ),
                indent=2,
            )
        )

        qc_records += _score(config, scene.scene_id, out_dir)
        # Numbered against the WHOLE batch, not against a rendered-only count:
        # `reused` is still accumulating, so a denominator built from it would
        # shrink as the run proceeds and the line would not read as progress.
        elapsed.append(t2 - t0)
        emit(verbosity, "progress",
             f"  [{len(elapsed) + reused}/{len(scenes)}] {scene.scene_id} "
             f"{t2 - t0:.1f}s (mean {sum(elapsed) / len(elapsed):.1f}s)")

    rendered = len(scenes) - len(refused) - reused
    emit(
        verbosity,
        "progress",
        f"  Rendered {rendered} of {len(scenes)} scenes "
        f"({reused} reused, {len(refused)} refused) → {renders_dir}",
    )

    # Written for the whole batch before the raise below: it is the evidence for
    # that raise, and an empty file is the record of a clean batch. Skips are in
    # it too — what was NOT scored bears on an admission decision as much as what
    # failed, and only this file is canonical.
    # Partitioned on GATING first: a non-gating disclosure is not a criterion, so
    # folding it into "skipped" would inflate that count the way counting skips as
    # passes once inflated the pass count.
    criteria = [r for r in qc_records if r.gating]
    disclosures = [r for r in qc_records if not r.gating]
    failures = [r for r in criteria if r.failed]
    unscored = [r for r in criteria if not r.scored]
    scored = [r for r in criteria if r.scored]
    qc.write_qc_csv(renders_dir / "qc_failures.csv",
                    failures + unscored + disclosures)
    emit(verbosity, "metrics",
         f"  QC: {len(scored) - len(failures)} of {len(scored)} scored criteria "
         f"passed, {len(unscored)} skipped, {len(disclosures)} disclosures "
         f"recorded, across {len(scenes) - len(refused)} scenes")

    # Per-scene measured values for every criterion, pass or fail — observability,
    # unlike the failure table above.
    if verbosity.saves("diagnostics"):
        qc.write_qc_csv(renders_dir / "qc_record.csv", qc_records)

    manifest = _write_manifest(renders_dir, scenes, refused, failures, criteria)
    emit(verbosity, "metrics",
         f"  Admitted {len(manifest['admitted'])} of {len(scenes)} scenes "
         f"({len(manifest['excluded'])} excluded) → {renders_dir / 'manifest.json'}")

    _enforce_attrition_bounds(config, manifest)
