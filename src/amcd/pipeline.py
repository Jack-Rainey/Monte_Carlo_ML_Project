"""Stage runner: caching (sentinel files), dispatch, timing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .config import Config
from .runtime import Verbosity, emit

STAGES = ["gen-scenes", "render", "preprocess", "diagnostics", "train", "infer", "eval", "stats", "report"]


def _sentinel(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / f"{stage.replace('-', '_')}.done"


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
        return not self.force and _sentinel(self.run_dir, stage).exists()

    def _mark_done(self, stage: str) -> None:
        # Sentinels are functional (caching input), never verbosity-gated (F-23).
        s = _sentinel(self.run_dir, stage)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(str(time.time()))

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
