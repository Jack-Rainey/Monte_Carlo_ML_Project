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

from .config import Config
from .runtime import Verbosity, emit

STAGES = ["gen-scenes", "render", "preprocess", "diagnostics", "train", "infer", "eval", "stats", "report"]


def _sentinel(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / f"{stage.replace('-', '_')}.done"


# ─────────────────────────────────────────────────────────────────────────────
# Stage cache fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _gen_scenes_fingerprint(config: Config) -> dict:
    """Config inputs that determine the generated scene set.

    Covers the sampling ranges, the split set (shift splits carry their own
    counts, and their scenes are generated here) and every seed that feeds scene
    sampling. Change any of them and the scenes on disk are for a different
    experiment.
    """
    return {
        "scenes": config.scenes.model_dump(),
        "splits": {name: sp.model_dump() for name, sp in config.splits.items()},
        "seed_scene_generation": config.seed("scene_generation"),
    }


def _render_fingerprint(config: Config) -> dict:
    """Config inputs that determine the rendered IR pair for a given scene.

    Chains the upstream gen-scenes fingerprint (RD-30): renders are per-scene, so
    a render is stale if the SCENES changed, not only if a simulator parameter
    did. Without the chain, editing a room dimension and re-running would keep
    renders of the old geometry against the new scene specs.

    `simulator.params` carries the pinned upstream `commit_sha`, so a GSound-SIR
    version change invalidates the cache with no extra wiring.
    """
    return {
        "upstream_gen_scenes": _fingerprint_sha(_gen_scenes_fingerprint(config)),
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
    }


#: Per-stage declaration of the config inputs a cached artifact depends on.
#:
#: `None` means "no fingerprint declared yet" — the stage caches on the bare
#: sentinel, exactly as before. Every stage is listed explicitly, including the
#: `None`s, so an unwired stage is DECLARED rather than silently absent (the same
#: rule docs/verbosity.md applies to verbosity wiring).
#:
#: The `None`s are a live gap, not a design: preprocess/train/eval reuse is the
#: same hazard one stage further down. Tracked in docs/review_ledger.md so they
#: cannot quietly become permanent.
STAGE_FINGERPRINT: dict[str, Callable[[Config], dict] | None] = {
    "gen-scenes": _gen_scenes_fingerprint,
    "render": _render_fingerprint,
    "preprocess": None,
    "diagnostics": None,
    "train": None,
    "infer": None,
    "eval": None,
    "stats": None,
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


def _diff_fingerprints(old: dict, new: dict) -> list[str]:
    """Field-level differences, so the error says WHAT changed (RD-35).

    A bare sha mismatch forces the operator to decide blind whether an expensive
    renders/ directory is salvageable; naming the changed field is the difference
    between a five-second judgement and a re-render under emulation.
    """
    lines = []
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key, "<absent>"), new.get(key, "<absent>")
        if before != after:
            lines.append(f"    {key}: {before!r} → {after!r}")
    return lines


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

        fingerprint_fn = STAGE_FINGERPRINT[stage]
        if fingerprint_fn is None:
            return True  # stage declares no config dependency; bare sentinel

        expected = _normalize(fingerprint_fn(self.config))
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
        fingerprint_fn = STAGE_FINGERPRINT[stage]
        payload = {
            "completed_at": time.time(),
            # Full payload, not just the sha: the mismatch error must be able to
            # name the field that changed (RD-35).
            "fingerprint": fingerprint_fn(self.config) if fingerprint_fn else None,
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

        if self._is_done(stage):
            emit(self.verbosity, "progress", f"[skip] {stage} (cached)")
            return

        emit(self.verbosity, "progress", f"\n[run ] {stage}")
        t0 = time.time()
        try:
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
