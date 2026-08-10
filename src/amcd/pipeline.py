"""Stage runner: dispatch, timing, and fingerprinted stage caching.

Caching is what makes a long run resumable, and — because renders under x86
emulation are the most expensive artifact this pipeline produces — it is also the
easiest way to end up with a silently mixed dataset. A bare "this stage ran"
sentinel cannot tell whether it ran under the CURRENT config, so changing a
simulator parameter or a scene range and re-running the same run_dir would reuse
stale artifacts and produce a dataset that is part old, part new, with nothing on
disk recording the split (RD-16).

So each stage may declare the config inputs it depends on (`STAGE_FINGERPRINT`).
The sentinel stores that fingerprint, and a mismatch is a LOUD FAILURE, never a
silent re-use and never a silent re-run: only the operator can decide whether the
existing artifacts are salvageable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from . import provenance
from .config import Config
from .runtime import Verbosity, emit

STAGES = ["gen-scenes", "render", "preprocess", "diagnostics", "train", "infer", "eval", "stats", "report"]


def _sentinel(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / f"{stage.replace('-', '_')}.done"


# ─────────────────────────────────────────────────────────────────────────────
# Stage cache fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _gen_scenes_fingerprint(config: Config) -> dict:
    """Config inputs that determine the generated scene set — and whether it is
    ADMITTED.

    Covers the sampling ranges, the split set (shift splits carry their own
    counts, and their scenes are generated here) and every seed that feeds scene
    sampling. Change any of them and the scenes on disk are for a different
    experiment.

    Only the GENERATION-relevant split fields are included (F-50). A `SplitSpec`
    also carries `frac` and `role`, which are consumed at preprocess and cannot
    change a single generated scene — dumping the whole spec meant editing
    `train.frac` refused the cached `render` and forced a complete re-render of a
    dataset that provably had not changed. Failing safe, but the cost it imposed is
    precisely the cost RD-16/RD-30 built this machinery to avoid. `frac`/`role`
    still reach `_preprocess_fingerprint`, which carries the full dump.

    `ir_duration` and the SIMULATOR are here for a different reason (F-57): they
    do not change which scenes are sampled, they change whether gen-scenes SUCCEEDS
    and what it discloses. `ir_duration` is what AC-22's record-length gate compares
    against, and the simulator declares the minimum source-receiver separation
    AC-13's pre-flight check enforces. Without them the gate was bypassable through
    the cache — REPRODUCED: generate at 4.25 s, change ONLY `ir_duration` to 0.25 s,
    and gen-scenes printed `[skip] (cached)` while `--force` on the same config
    raised "16 of 29 scenes (55.172%) exceed it". `placement_report.json` also stayed
    stamped at 4.25 s while `config.yaml` said 0.25 — the disclosure artifact
    contradicting its own run.
    """
    return {
        "scenes": config.scenes.model_dump(),
        "splits": {
            name: {"count": sp.count, "seed": sp.seed, "axes": sp.axes}
            for name, sp in config.splits.items()
        },
        "seed_scene_generation": config.seed("scene_generation"),
        "ir_duration": config.ir_duration,
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
    }


def _render_fingerprint(config: Config) -> dict:
    """Config inputs that determine the rendered IR pair for a given scene.

    `simulator.params` carries the pinned upstream `commit_sha`, so a GSound-SIR
    version change invalidates the cache with no extra wiring.
    """
    return {
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
    }


def _preprocess_fingerprint(config: Config) -> dict:
    """Config inputs that determine the split assignment and the encoded tensors.

    `split_assignment` belongs here specifically (F-29): it is the most
    leakage-critical value in the project, it is consumed HERE rather than at
    gen-scenes, and without it in a fingerprint, repinning the split seed on an
    existing run_dir was a complete no-op — splits.json kept the old assignment
    while config.yaml stamped the new seed, a provenance lie.
    """
    return {
        "splits": {name: sp.model_dump() for name, sp in config.splits.items()},
        "seed_split_assignment": config.seed("split_assignment"),
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
    }


def _train_fingerprint(config: Config) -> dict:
    """Config inputs that determine the trained weights.

    RD-41 deferred this on COST grounds — "the expensive artifact is the render" —
    and that rationale does not survive contact with what the gap actually did
    (F-53). REPRODUCED end to end: on a complete run_dir, changing
    `model.params.hidden_channels` 8→32, `n_layers` 2→4, `learning_rate`
    0.001→0.05, `huber_delta` 1.0→3.0 and `n_epochs` 2→7, then re-running
    `amcd all`, printed `[skip]` for ALL NINE stages and exited 0. `config.yaml`
    was re-stamped with the new values while `checkpoints/best.pt` still held the
    OLD model, so `report/summary.txt` and `stats/ci_table.csv` were the old
    model's numbers under the new model's provenance stamp. A cache that is cheap
    to rebuild is not a reason to leave the reported result unprotected.

    Both training seeds are named individually (inv #5): weight init and data
    shuffle are separate aspects and a run that differs in either is a different
    run.
    """
    return {
        "model": {"name": config.model.name, "params": config.model.params},
        "n_epochs": config.n_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "huber_delta": config.huber_delta,
        "early_stopping_patience": config.early_stopping_patience,
        "seed_weight_init": config.seed("weight_init"),
        "seed_data_shuffle": config.seed("data_shuffle"),
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "code_version": _code_version("train"),
    }


def _infer_fingerprint(config: Config) -> dict:
    """Config inputs that determine the predictions.

    Inference re-instantiates the model from `model.name`/`params` to load the
    checkpoint, and decodes through the representation, so both are inputs even
    though neither is re-trained here. Everything else it depends on arrives
    through the chain from `train` (F-53).
    """
    return {
        "model": {"name": config.model.name, "params": config.model.params},
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "code_version": _code_version("infer"),
    }


def _eval_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the reported metrics.

    RD-54 promoted this out of RD-41's DEFERRED set because the metric path is
    actively in flux. RD-59 then established that config keys ALONE would close
    RD-54 with a mechanism that cannot see its own trigger case: the AC-17 shared
    Schroeder window (`f3c3543`) was a CODE change in
    `evaluation/room_acoustic.py`, and no config key moved with it. So the commit
    sha is part of this fingerprint — a cached `metrics.parquet` written under a
    different revision of the metric code is refused, not silently reported.

    Coarse on purpose: any commit invalidates eval, including commits that touch
    nothing relevant. That is the correct direction to err for the one stage whose
    output IS the research claim, and eval is cheap next to render.
    """
    return {
        "iso_eval_freqs": list(config.iso_eval_freqs),
        "metric_onset_rel_db": config.metric_onset_rel_db,
        "metric_band_resolvability_margin": config.metric_band_resolvability_margin,
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "sample_rate": config.sample_rate,
        "code_version": _code_version("eval"),
    }


def _stats_fingerprint(config: Config) -> dict:
    """Config inputs that determine the CIs and MDES.

    Its dependence on eval is expressed by `STAGE_UPSTREAM["stats"] = "eval"`, NOT
    by a key here. An earlier version carried
    `"upstream_eval": _fingerprint_sha(_eval_fingerprint(config))` — a chain
    RECOMPUTED FROM THE CURRENT CONFIG, which is exactly what the STAGE_UPSTREAM
    docstring below forbids, in the fix that closed the row about it (F-54). Both
    consequences reproduced: `stats --force` after a `metric_onset_rel_db` change
    stamped the NEW config's eval sha while `eval.done` still recorded the old one,
    so the provenance chain claimed artifacts that did not exist; and the
    non-forced failure named no leaf, losing the recursive diff F-49 bought.
    """
    return {
        "bootstrap_n_resamples": config.bootstrap_n_resamples,
        "bootstrap_alpha": config.bootstrap_alpha,
        "bootstrap_power": config.bootstrap_power,
        "seed_bootstrap": config.seed("bootstrap"),
    }


#: Per stage, the package sources its OUTPUT is a function of — the code half of a
#: fingerprint (F-55). `provenance.code_version` adds the core modules (config,
#: seeds, registry, shared acoustics) to every scope, so these list only what is
#: specific to the stage.
#:
#: Declared rather than inferred, and asserted in tests against what the stage
#: actually imports, because the failure mode of a wrong scope is silent: a stage
#: whose real dependency is missing here goes on serving a cached artifact under
#: changed code, which is the defect this exists to close.
STAGE_CODE_SCOPE: dict[str, tuple[str, ...]] = {
    # Trains weights: the model, the loss, the trainer, the dataset it reads, and
    # the representation whose domain the loss is expressed in.
    "train": ("training", "models", "data", "representations"),
    # Loads the checkpoint and decodes predictions through the representation.
    "infer": ("training", "models", "representations"),
    # The stage whose output IS the research claim. `representations` is in scope
    # because eval decodes before measuring.
    "eval": ("evaluation", "representations"),
}


def _code_version(stage: str) -> str:
    """The content hash of `stage`'s declared sources — see
    `amcd.provenance.code_version`.

    Shared with `Config.stamp` through that module so the cache key and
    `versions.json` cannot describe different code (F-56).
    """
    return provenance.code_version(STAGE_CODE_SCOPE[stage])


#: Per-stage declaration of the config inputs a cached artifact depends on.
#:
#: `None` means "no fingerprint declared yet" — the stage caches on the bare
#: sentinel, exactly as before. Every stage is listed explicitly, including the
#: `None`s, so an unwired stage is DECLARED rather than silently absent (the same
#: rule docs/verbosity.md applies to verbosity wiring).
#:
#: SCOPE, stated because the mechanism looks stronger than it is: a fingerprint
#: built from config keys sees CONFIG changes only. A code change to a stage —
#: a redefined metric, a different sampling rule — moves nothing in these payloads
#: and a cached artifact is served under the new code (RD-59). The three stages
#: that produce or shape the REPORTED RESULT (`train`, `infer`, `eval`) therefore
#: carry `code_version()` as well; the others do not, so the remedy for a code
#: change upstream of them is a fresh run_dir or `--force`.
#:
#: The remaining `None`s are `diagnostics` and `report` (ledger RD-41). Both are
#: terminal — nothing chains to them — so an unwired fingerprint there cannot
#: corrupt another stage's provenance, only its own re-use.
STAGE_FINGERPRINT: dict[str, Callable[[Config], dict] | None] = {
    "gen-scenes": _gen_scenes_fingerprint,
    "render": _render_fingerprint,
    "preprocess": _preprocess_fingerprint,
    "diagnostics": None,
    "train": _train_fingerprint,
    "infer": _infer_fingerprint,
    "eval": _eval_fingerprint,
    "stats": _stats_fingerprint,
    "report": None,
}

#: Which stage's artifacts each stage consumes. The chain is what makes a
#: fingerprint transitive: renders are renders OF scenes, so a render is stale
#: when the SCENES changed even if every simulator parameter is untouched.
#:
#: Chaining reads the upstream SENTINEL's recorded fingerprint, never a
#: recomputation from the current config (F-26). Recomputing made `--force` a
#: laundering step: forcing a downstream stage stamped the CURRENT config's chain
#: value even though the stage had consumed the OLD upstream artifacts, leaving a
#: run_dir whose chain validated while the artifacts disagreed — precisely the
#: silent mixed dataset this machinery exists to prevent.
#: The chain runs unbroken from the scenes to the reported CIs:
#:
#:   gen-scenes → render → preprocess → train → infer → eval → stats
#:
#: Before F-53 it stopped at `preprocess`. `eval` and `stats` carried fingerprints,
#: which READ as "the reported metrics are cache-protected", while neither was
#: chained to its actual inputs and `train`/`infer` carried none at all — so a
#: changed model was reported under a stale checkpoint's numbers. Wiring the chain
#: is what makes a fingerprint transitive: `stats` is stale when the SCENES changed,
#: through six links, without `_stats_fingerprint` naming a single scene key.
#:
#: `diagnostics` and `report` are terminal and stay unchained (RD-41).
STAGE_UPSTREAM: dict[str, str | None] = {
    "gen-scenes": None,
    "render": "gen-scenes",
    "preprocess": "render",
    "diagnostics": None,
    "train": "preprocess",
    "infer": "train",
    "eval": "infer",
    "stats": "eval",
    "report": None,
}


def _normalize(payload: dict) -> dict:
    """Round-trip a fingerprint through JSON so it compares equal to a stored one.

    Not cosmetic: `model_dump()` yields tuples, and JSON has no tuple type, so a
    freshly computed fingerprint would never equal the one read back from a
    sentinel — every cached stage would raise a spurious "config changed". The
    stored form is the canonical form, so compare in it.
    """
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _fingerprint_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(_normalize(payload), sort_keys=True).encode()
    ).hexdigest()


def _diff_fingerprints(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Leaf-level differences, so the error says WHAT changed (RD-35).

    A bare sha mismatch forces the operator to decide blind whether an expensive
    renders/ directory is salvageable; naming the changed field is the difference
    between a five-second judgement and a re-render under emulation.

    RECURSES into nested dicts and reports dotted paths (F-49). A top-level-only
    comparison degenerated on exactly the two fields most likely to change —
    `scenes` and `splits`, the only nested payloads — printing the whole ~700
    character dict twice with the single changed value buried inside, which fails
    the stated purpose above in the one case it was written for.
    """
    lines = []
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key, _ABSENT), new.get(key, _ABSENT)
        if before == after:
            continue
        path = f"{prefix}{key}"
        if isinstance(before, dict) and isinstance(after, dict):
            lines.extend(_diff_fingerprints(before, after, prefix=f"{path}."))
        else:
            lines.append(
                f"    {path}: {_render_leaf(before)} → {_render_leaf(after)}"
            )
    return lines


#: Distinguishes "key absent" from a key whose value happens to be the string
#: "<absent>", so a diff cannot misreport which side a key is missing from.
_ABSENT = object()


def _render_leaf(value) -> str:
    return "<absent>" if value is _ABSENT else repr(value)


def _dispatch(stage: str) -> Callable[[Config, Path, Verbosity], None]:
    if stage == "gen-scenes":
        from .scenes.generator import run_gen_scenes
        return run_gen_scenes
    if stage == "render":
        from .simulators.render import run_render
        return run_render
    if stage == "preprocess":
        from .data.preprocess import run_preprocess
        return run_preprocess
    if stage == "diagnostics":
        from .diagnostics.probe import run_diagnostics
        return run_diagnostics
    if stage == "train":
        from .training.trainer import run_train
        return run_train
    if stage == "infer":
        from .training.infer import run_infer
        return run_infer
    if stage == "eval":
        from .evaluation.evaluator import run_eval
        return run_eval
    if stage == "stats":
        from .stats.aggregate import run_stats
        return run_stats
    if stage == "report":
        from .reporting.tables import run_report
        return run_report
    raise ValueError(f"Unknown stage: {stage!r}")


class Pipeline:
    def __init__(self, config: Config, run_dir: Path, verbosity: Verbosity, force: bool = False) -> None:
        self.config = config
        self.run_dir = run_dir
        self.verbosity = verbosity
        self.force = force

    def _recorded_fingerprint(self, stage: str) -> dict | None:
        """The fingerprint stored in `stage`'s sentinel, or None if it has not run."""
        sentinel = _sentinel(self.run_dir, stage)
        if not sentinel.exists():
            return None
        try:
            return json.loads(sentinel.read_text()).get("fingerprint")
        except (json.JSONDecodeError, AttributeError):
            return None

    def _effective_fingerprint(self, stage: str) -> dict:
        """This stage's own config inputs, plus the upstream sentinel's RECORDED
        fingerprint (F-26).

        Taking the upstream value from disk rather than recomputing it from the
        current config is the whole point: it describes the artifacts this stage
        actually consumed, so a `--force` cannot launder a stale upstream into a
        chain that validates.
        """
        own = _normalize(STAGE_FINGERPRINT[stage](self.config))
        upstream = STAGE_UPSTREAM.get(stage)
        if upstream is None:
            return own
        # An upstream that declares no fingerprint cannot anchor a chain: its
        # sentinel records `null`, which is indistinguishable from "never ran".
        # No STAGE_UPSTREAM value points at such a stage today, but RD-41 exists to
        # wire the remaining seven, and the first chain to cross one would
        # otherwise report "has not completed" for a stage that did (F-41).
        if STAGE_FINGERPRINT[upstream] is None:
            raise RuntimeError(
                f"Stage {stage!r} declares {upstream!r} as its upstream, but "
                f"{upstream!r} has no fingerprint in STAGE_FINGERPRINT, so its "
                f"sentinel cannot say WHICH config its artifacts belong to. Give "
                f"{upstream!r} a fingerprint before chaining to it."
            )
        recorded = self._recorded_fingerprint(upstream)
        if recorded is None:
            raise RuntimeError(
                f"Stage {stage!r} depends on {upstream!r}, which has not completed "
                f"in {self.run_dir} (no readable fingerprinted sentinel). Run "
                f"{upstream!r} first — running {stage!r} now would record a "
                f"provenance chain for artifacts that do not exist."
            )
        # The upstream artifacts must ALSO be current for this config. Recursing
        # here (rather than only comparing our own inputs) is what catches a stale
        # ancestor when a downstream stage is run on its own: `amcd render` after
        # editing a room dimension would otherwise reuse renders that match the
        # OLD scene specs still sitting on disk.
        upstream_expected = self._effective_fingerprint(upstream)
        if recorded != upstream_expected:
            diff = _diff_fingerprints(recorded, upstream_expected) or [
                "    (nested value changed)"
            ]
            raise RuntimeError(
                f"Stage {stage!r} cannot run: its upstream stage {upstream!r} holds "
                f"artifacts built under a DIFFERENT config.\n"
                f"  Changed inputs to {upstream!r}:\n" + "\n".join(diff) + "\n"
                f"  Re-run {upstream!r} (with --force) before {stage!r}, or use a "
                f"fresh --run-dir."
            )
        return {"upstream": {upstream: _fingerprint_sha(recorded)}, **own}

    def _is_done(self, stage: str) -> bool:
        """Whether `stage`'s cached artifacts are valid for the CURRENT config.

        Raises rather than returning False on a fingerprint mismatch: silently
        re-running would overwrite artifacts the operator may still need, and
        silently skipping would mix two experiments in one run_dir. Both failure
        modes are invisible after the fact, so the decision belongs to a human.
        """
        sentinel = _sentinel(self.run_dir, stage)
        if self.force or not sentinel.exists():
            return False

        if STAGE_FINGERPRINT[stage] is None:
            return True  # stage declares no config dependency; bare sentinel

        expected = self._effective_fingerprint(stage)
        try:
            recorded = json.loads(sentinel.read_text())
            found = recorded["fingerprint"]
        except (json.JSONDecodeError, TypeError, KeyError):
            raise RuntimeError(
                f"Stage {stage!r} has a cached sentinel with no fingerprint "
                f"({sentinel}). It predates fingerprinted caching, so whether its "
                f"artifacts match the current config cannot be established. "
                f"Re-run with --force to rebuild, or use a fresh --run-dir."
            ) from None

        if found != expected:
            diff = _diff_fingerprints(found, expected) or ["    (nested value changed)"]
            raise RuntimeError(
                f"Stage {stage!r} was cached under a DIFFERENT config; reusing it "
                f"would silently mix two experiments in one run_dir.\n"
                f"  Changed inputs:\n" + "\n".join(diff) + "\n"
                f"  Re-run with --force to rebuild {stage!r} (this discards its "
                f"existing artifacts), or use a fresh --run-dir to keep both."
            )
        return True

    def _mark_done(self, stage: str) -> None:
        # Sentinels are functional (caching input), never verbosity-gated (F-23).
        # The recorded fingerprint chains the UPSTREAM SENTINEL, so it describes the
        # artifacts this run actually consumed — including under --force (F-26).
        payload = {
            "completed_at": time.time(),
            # Full payload, not just the sha: the mismatch error must be able to
            # name the field that changed (RD-35).
            "fingerprint": (
                self._effective_fingerprint(stage)
                if STAGE_FINGERPRINT[stage] else None
            ),
        }
        s = _sentinel(self.run_dir, stage)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps(payload, indent=2, default=str))

    def _record_timing(self, stage: str, seconds: float) -> None:
        # Per-stage wall time is provenance (RD-09): written from save level 1.
        if not self.verbosity.saves("provenance"):
            return
        path = self.run_dir / "timings.json"
        timings = json.loads(path.read_text()) if path.exists() else {}
        timings[stage] = seconds
        path.write_text(json.dumps(timings, indent=2))

    def run_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Valid: {STAGES}")

        t0 = time.time()
        try:
            # Validate the provenance chain BEFORE doing any work, so a stale
            # upstream is refused up front rather than after an expensive render,
            # and so --force cannot run the stage and only then fail to record its
            # chain. Inside the try so the failure still reaches stderr (F-24).
            if STAGE_FINGERPRINT[stage] is not None:
                self._effective_fingerprint(stage)

            if self._is_done(stage):
                emit(self.verbosity, "progress", f"[skip] {stage} (cached)")
                return

            # INVALIDATE BEFORE MUTATING (F-58). The stage is about to overwrite
            # its artifacts, so from this moment the old sentinel describes a
            # directory that no longer exists. Leaving it in place until success
            # meant a stage that failed or was killed PART-WAY THROUGH WRITING left
            # the PREVIOUS run's success sentinel standing over half-new artifacts,
            # and the next invocation printed `[skip] (cached)` against it —
            # reproduced with gen-scenes aborting after writing all 29 scene JSONs.
            # The costly instance is render: the orphan prune runs first, so a
            # --force render killed at scene 300/600 leaves scenes 300-599 holding
            # the PREVIOUS config's .npy under a sentinel that validates. That is
            # the silently mixed dataset RD-16 exists to prevent, and under
            # emulation it is the multi-hour case.
            _sentinel(self.run_dir, stage).unlink(missing_ok=True)

            emit(self.verbosity, "progress", f"\n[run ] {stage}")
            fn = _dispatch(stage)
            fn(self.config, self.run_dir, self.verbosity)
        except Exception as exc:
            emit(self.verbosity, "error", f"[FAIL] {stage}: {exc}")
            raise

        self._mark_done(stage)
        elapsed = time.time() - t0
        self._record_timing(stage, elapsed)
        emit(self.verbosity, "timing", f"[done] {stage} ({elapsed:.1f}s)")

    def run_all(self) -> None:
        for stage in STAGES:
            self.run_stage(stage)
