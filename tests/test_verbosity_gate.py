"""End-to-end tests for the verbosity gate.

Each test isolates one guarantee independently: verbosity is side-effect-free
with respect to results.

The two module-scoped pipeline runs use the tiny test config through the real
CLI (`amcd all`), so the full cli → Pipeline → stage threading is what
is exercised, not direct stage calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from amcd.cli import main

from tests.conftest import tiny_cli_args

# Artifacts that must exist at EVERY save level: canonical results,
# inter-stage inputs, and stage sentinels. Globs are relative to the run dir;
# each must match at least once.
CANONICAL_GLOBS = [
    "scenes/scene_*.json",
    "renders/*/low.npy",
    "renders/*/high.npy",
    # Canonical render provenance: the only record of how an expensive dataset
    # was made, so it is written at every save level.
    "renders/*/meta.json",
    # The evidence behind a QC refusal — a raise whose evidence file is
    # suppressed at the default save level is not a reportable result.
    "renders/qc_failures.csv",
    "preprocessed/meta.json",
    "preprocessed/splits.json",
    "preprocessed/carrier/*.npy",
    "preprocessed/*/*_low.pt",
    "preprocessed/*/*_high.pt",
    "checkpoints/best.pt",
    "predictions/*_pred.pt",
    "predictions/*_decoded_ir.npy",
    "metrics/metrics.parquet",
    "metrics/drops.csv",
    "diagnostics/d0a_gap.json",
    "diagnostics/d0b_oracle.json",
    "stats/ci_table.csv",
    "stats/summary.json",
    "report/summary.txt",
    "report/metrics_table.csv",
    "stages/*.done",
]

# Observability-only artifacts gated by --save-verbosity (docs/verbosity.md):
# absent at save=0, present at save=4.
GATED_PATHS = [
    "config.yaml",           # Config.stamp trio: provenance, save >= 1
    "resolved.yaml",
    "versions.json",
    "timings.json",          # provenance, save >= 1
    "checkpoints/train_log.csv",  # intermediate metrics, save >= 3
    "report/config.yaml",    # bundle copies: provenance, save >= 1
    "report/versions.json",
]
# No gated globs today: render meta.json became canonical, and Step 4's
# per-criterion QC record is the next artifact that will legitimately sit here.
GATED_GLOBS: list[str] = []


def _run_all(run_dir: Path, *flags: str) -> "click.testing.Result":
    result = CliRunner().invoke(
        main,
        ["all", *tiny_cli_args(), "-r", str(run_dir), *flags],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(scope="module")
def quiet_run(tmp_path_factory: pytest.TempPathFactory):
    """Full pipeline at the floor of both axes: save=0, show=0."""
    run_dir = tmp_path_factory.mktemp("verbosity_quiet")
    result = _run_all(run_dir, "--save-verbosity", "0", "--show-verbosity", "0")
    return result, run_dir


@pytest.fixture(scope="module")
def loud_run(tmp_path_factory: pytest.TempPathFactory):
    """Full pipeline at save=4 (all disk categories that exist today), show=3."""
    run_dir = tmp_path_factory.mktemp("verbosity_loud")
    result = _run_all(run_dir, "--save-verbosity", "4", "--show-verbosity", "3")
    return result, run_dir


def test_show_levels_gate_console_output(quiet_run, loud_run) -> None:
    """The flags are threaded, not inert — show=0 is silent on stdout,
    show=3 emits run identity, progress, and metrics lines."""
    quiet_result, _ = quiet_run
    loud_result, _ = loud_run
    assert quiet_result.stdout == ""
    for expected in ("Run dir:", "[run ]", "[done]", "Stats for"):
        assert expected in loud_result.stdout


def test_canonical_set_complete_at_save_zero(quiet_run) -> None:
    """No functional artifact sits behind the save gate — the entire
    canonical set exists at save=0, and only observability artifacts don't."""
    _, run_dir = quiet_run
    for pattern in CANONICAL_GLOBS:
        assert list(run_dir.glob(pattern)), f"canonical artifact missing at save=0: {pattern}"
    for rel in GATED_PATHS:
        assert not (run_dir / rel).exists(), f"gated artifact leaked at save=0: {rel}"
    for pattern in GATED_GLOBS:
        assert not list(run_dir.glob(pattern)), f"gated artifact leaked at save=0: {pattern}"


def test_gated_artifacts_present_at_save_four(loud_run) -> None:
    """The same gated set exists once the save level admits it (so the save=0
    absences above prove gating, not a stage that never wrote them)."""
    _, run_dir = loud_run
    for rel in GATED_PATHS:
        assert (run_dir / rel).exists(), f"gated artifact missing at save=4: {rel}"
    for pattern in GATED_GLOBS:
        assert list(run_dir.glob(pattern)), f"gated artifact missing at save=4: {pattern}"


def test_results_identical_across_verbosity_levels(quiet_run, loud_run) -> None:
    """Load-bearing claim: verbosity is side-effect-free w.r.t. results.
    Two independent full runs at (save=0, show=0) and (save=4, show=3) with
    identical config/seeds produce exactly equal canonical results."""
    _, dir_a = quiet_run
    _, dir_b = loud_run
    df_a = pd.read_parquet(dir_a / "metrics" / "metrics.parquet")
    df_b = pd.read_parquet(dir_b / "metrics" / "metrics.parquet")
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)
    summary_a = (dir_a / "stats" / "summary.json").read_text()
    summary_b = (dir_b / "stats" / "summary.json").read_text()
    assert summary_a == summary_b
    assert (dir_a / "report" / "summary.txt").read_text().replace(dir_a.name, "") \
        == (dir_b / "report" / "summary.txt").read_text().replace(dir_b.name, "")


def test_out_of_range_levels_rejected() -> None:
    """IntRange(0,5) — out-of-range is a usage error, never a clamp."""
    for flag in ("--save-verbosity", "--show-verbosity"):
        result = CliRunner().invoke(
            main, ["all", *tiny_cli_args(), flag, "6"]
        )
        assert result.exit_code != 0
        assert "is not in the range" in result.output or "Invalid value" in result.output


def test_failures_reach_stderr_at_show_zero(tmp_path: Path) -> None:
    """Fatal errors always emit, to stderr, regardless of show level.
    `render` on a run dir with no scenes fails; show=0 must not swallow it."""
    result = CliRunner().invoke(
        main,
        ["render", *tiny_cli_args(), "-r", str(tmp_path),
         "--save-verbosity", "0", "--show-verbosity", "0"],
    )
    assert result.exit_code != 0
    assert "[FAIL] render" in result.stderr
    assert result.stdout == ""


def test_default_run_writes_provenance(tmp_path: Path) -> None:
    """The CLI defaults (save=1) anchor to the provenance rung — a bare
    invocation records config snapshot, seeds, git SHA, and timings; an
    explicit save=0 omits exactly that provenance."""
    default_dir = tmp_path / "default"
    result = CliRunner().invoke(
        main,
        ["gen-scenes", *tiny_cli_args(), "-r", str(default_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    for fname in ("config.yaml", "resolved.yaml", "versions.json", "timings.json"):
        assert (default_dir / fname).exists(), f"default run lacks provenance: {fname}"
    # Default show=1 prints run identity but not progress.
    assert "Run dir:" in result.stdout
    assert "[run ]" not in result.stdout

    bare_dir = tmp_path / "save0"
    result = CliRunner().invoke(
        main,
        ["gen-scenes", *tiny_cli_args(), "-r", str(bare_dir),
         "--save-verbosity", "0"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    for fname in ("config.yaml", "resolved.yaml", "versions.json", "timings.json"):
        assert not (bare_dir / fname).exists(), f"save=0 wrote provenance: {fname}"
    # ... while the stage's canonical output is untouched by the level.
    assert list(bare_dir.glob("scenes/scene_*.json"))
