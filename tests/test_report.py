"""Report rendering guards (ledger RR-14, F-21).

The results table must never present an unscored quantity as a result: an
n_scored == 0 row renders as `unscored` with NO descriptive mean in any results
column, and every row shows scored/attempted so drops are visible, not inferred.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from amcd.reporting.tables import run_report

from tests.conftest import tiny_config


def _summary_row(metric: str, **overrides) -> dict:
    row = {
        "split": "test_id", "metric": metric, "kind": "match_reference",
        "n_attempted": 3, "n_pred": 3,
        "pred_mean": 0.4321, "pred_ci_lower": 0.30, "pred_ci_upper": 0.55,
        "pred_std": 0.10,
        "n_scored": 3,
        "improvement_mean": 0.1234, "improvement_ci_lower": 0.05,
        "improvement_ci_upper": 0.20, "improvement_std": 0.07,
        "improvement_mdes": 0.30, "n_improved": 2, "pct_improved": 66.7,
    }
    row.update(overrides)
    return row


def _render(rows: list[dict]) -> str:
    cfg = tiny_config()
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "stats").mkdir()
        (run_dir / "stats" / "summary.json").write_text(json.dumps(rows))
        run_report(cfg, run_dir)
        return (run_dir / "report" / "summary.txt").read_text()


def test_unscored_row_renders_unscored_with_no_numbers() -> None:
    """RR-14: an n_scored == 0 row must say `unscored` and must NOT show its
    descriptive pred_mean (pre-fix, energy_snr_db printed −1.6…−2.7 dB "Pred
    mean" beside all-N/A inferential cells — an unscored quantity reading as a
    result)."""
    unscored = _summary_row(
        "energy_snr_db", kind="maximize", n_scored=0, n_pred=0,
        pred_mean=-2.71, improvement_mean=float("nan"),
        improvement_ci_lower=float("nan"), improvement_ci_upper=float("nan"),
        improvement_std=float("nan"), improvement_mdes=float("nan"),
        n_improved=0, pct_improved=float("nan"),
    )
    txt = _render([_summary_row("energy_mse"), unscored])

    snr_line = next(l for l in txt.splitlines() if l.startswith("energy_snr_db"))
    assert "unscored" in snr_line
    assert "-2.71" not in snr_line and "nan" not in snr_line.lower()
    # The scored row keeps its numbers.
    mse_line = next(l for l in txt.splitlines() if l.startswith("energy_mse"))
    assert "0.4321" in mse_line and "unscored" not in mse_line


def test_rows_show_scored_over_attempted() -> None:
    """F-21: the N column is scored/attempted, so a dropped scene is a visible
    gap (2/3), never a silently smaller population."""
    txt = _render([_summary_row("C50", n_scored=2, n_attempted=3, n_improved=1,
                                pct_improved=50.0)])
    c50_line = next(l for l in txt.splitlines() if l.startswith("C50"))
    assert "2/3" in c50_line
