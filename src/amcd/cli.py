"""CLI entry point: amcd <stage> --config configs/base.yaml [--config configs/override.yaml]"""
from __future__ import annotations

import datetime
from pathlib import Path

import click

from .pipeline import STAGES, Pipeline
from .runtime import RunContext, Verbosity, emit


def _make_run_dir(label: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / f"{label}_{ts}"


@click.group()
def main() -> None:
    """Acoustic Monte Carlo Denoising pipeline."""


def common_options(fn):
    """The option set shared by every stage command and `all`.

    The two verbosity defaults are the sanctioned exception to
    no-hidden-defaults (CLAUDE.md "Output verbosity is not an experiment
    value"): runtime output levels only, quarantined to this CLI layer —
    `Config` never carries them. Default 1 on both axes is the
    provenance/timing rung, so a bare invocation is non-blocking yet never
    lacks reproducibility metadata (config snapshot, seeds, git SHA,
    timings); save=0 deliberately omits provenance and is never a default.
    Ladder and per-stage wiring table: docs/verbosity.md.
    """
    options = [
        click.option(
            "--config", "-c",
            multiple=True,
            type=click.Path(exists=True, path_type=Path),
            required=True,
            help="Config YAML file(s). Multiple files are merged left-to-right.",
        ),
        click.option(
            "--run-dir", "-r",
            type=click.Path(path_type=Path),
            default=None,
            help="Run directory (default: experiments/<stage>_<timestamp>).",
        ),
        click.option(
            "--force", is_flag=True, default=False,
            help="Re-run even if cached, rebuilding every artifact from scratch. "
                 "Use when you do not trust the artifacts.",
        ),
        click.option(
            "--revalidate", is_flag=True, default=False,
            help="Re-run past a fingerprint mismatch while KEEPING every artifact "
                 "whose own fingerprint still matches. Use when the artifacts are "
                 "fine and the rule that judges them changed — a QC threshold "
                 "costs a re-score, not a re-render.",
        ),
        click.option(
            "--save-verbosity",
            type=click.IntRange(0, 5),
            default=1,
            show_default=True,
            help="How much a run writes to disk: 0=canonical results only, "
                 "1=+provenance, 2=+progress, 3=+intermediate metrics, "
                 "4=+full diagnostics, 5=+visual. Results are identical at "
                 "every level (docs/verbosity.md).",
        ),
        click.option(
            "--show-verbosity",
            type=click.IntRange(0, 5),
            default=1,
            show_default=True,
            help="How much a run prints live: 0=warnings/errors only, "
                 "1=+run id and timings, 2=+progress, 3=+intermediate metrics, "
                 "4=+full diagnostics, 5=+visual preview (TTY only; "
                 "docs/verbosity.md).",
        ),
    ]
    for opt in reversed(options):
        fn = opt(fn)
    return fn


def _invoke(
    stage: str | None,
    config: tuple[Path, ...],
    run_dir: Path | None,
    force: bool,
    revalidate: bool,
    save_verbosity: int,
    show_verbosity: int,
) -> None:
    """Shared command body; `stage=None` runs all stages."""
    from .config import Config
    from .device import select_device
    if force and revalidate:
        raise click.UsageError(
            "--force and --revalidate contradict each other: the first discards "
            "every artifact, the second keeps the ones that still match. Pick the "
            "one that describes why you are re-running."
        )
    cfg = Config.load(*config)
    verbosity = Verbosity(save=save_verbosity, show=show_verbosity)
    if run_dir is None:
        run_dir = _make_run_dir(stage if stage is not None else "all")
    run_dir = Path(run_dir)
    if verbosity.saves("provenance"):
        cfg.stamp(run_dir, device=str(select_device()))
    emit(verbosity, "timing", f"Run dir: {run_dir.resolve()}")
    # The CLI is where the runtime context is ASSEMBLED — it is the layer that
    # knows the machine and the invocation, which is exactly what RunContext
    # carries and Config must not.
    pipeline = Pipeline(
        cfg, run_dir, RunContext(verbosity), force=force, revalidate=revalidate
    )
    if stage is None:
        pipeline.run_all()
    else:
        pipeline.run_stage(stage)


def _add_stage_command(stage: str) -> None:
    @main.command(name=stage)
    @common_options
    def _cmd(
        config: tuple[Path, ...],
        run_dir: Path | None,
        force: bool,
        revalidate: bool,
        save_verbosity: int,
        show_verbosity: int,
    ) -> None:
        _invoke(stage, config, run_dir, force, revalidate,
                save_verbosity, show_verbosity)

    _cmd.__name__ = stage.replace("-", "_")


for _s in STAGES:
    _add_stage_command(_s)


@main.command("all")
@common_options
def run_all(
    config: tuple[Path, ...],
    run_dir: Path | None,
    force: bool,
    revalidate: bool,
    save_verbosity: int,
    show_verbosity: int,
) -> None:
    """Run all stages: gen-scenes → render → preprocess → diagnostics → train → infer → eval → stats → report."""
    _invoke(None, config, run_dir, force, revalidate,
            save_verbosity, show_verbosity)
