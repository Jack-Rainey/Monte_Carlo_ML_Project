"""CLI entry point: amcd <stage> --config configs/base.yaml [--config configs/override.yaml]"""
from __future__ import annotations

import datetime
from pathlib import Path

import click

from .pipeline import STAGES, Pipeline


def _make_run_dir(label: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / f"{label}_{ts}"


@click.group()
def main() -> None:
    """Acoustic Monte Carlo Denoising pipeline."""


def _add_stage_command(stage: str) -> None:
    fn_name = stage.replace("-", "_")

    @main.command(name=stage)
    @click.option(
        "--config", "-c",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Config YAML file(s). Multiple files are merged left-to-right.",
    )
    @click.option(
        "--run-dir", "-r",
        type=click.Path(path_type=Path),
        default=None,
        help="Run directory (default: experiments/<stage>_<timestamp>).",
    )
    @click.option("--force", is_flag=True, default=False, help="Re-run even if cached.")
    def _cmd(config: tuple[Path, ...], run_dir: Path | None, force: bool) -> None:
        from .config import Config
        cfg = Config.load(*config)
        if run_dir is None:
            run_dir = _make_run_dir(stage)
        run_dir = Path(run_dir)
        cfg.stamp(run_dir)
        print(f"Run dir: {run_dir}")
        pipeline = Pipeline(cfg, run_dir, force=force)
        pipeline.run_stage(stage)

    _cmd.__name__ = fn_name


for _s in STAGES:
    _add_stage_command(_s)


@main.command("all")
@click.option(
    "--config", "-c",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Config YAML file(s). Multiple files are merged left-to-right.",
)
@click.option(
    "--run-dir", "-r",
    type=click.Path(path_type=Path),
    default=None,
    help="Run directory (default: experiments/all_<timestamp>).",
)
@click.option("--force", is_flag=True, default=False, help="Re-run all stages.")
def run_all(config: tuple[Path, ...], run_dir: Path | None, force: bool) -> None:
    """Run all stages: gen-scenes → render → preprocess → diagnostics → train → infer → eval → stats → report."""
    from .config import Config
    cfg = Config.load(*config)
    if run_dir is None:
        run_dir = _make_run_dir("all")
    run_dir = Path(run_dir)
    cfg.stamp(run_dir)
    print(f"Run dir: {run_dir.resolve()}")
    pipeline = Pipeline(cfg, run_dir, force=force)
    pipeline.run_all()
