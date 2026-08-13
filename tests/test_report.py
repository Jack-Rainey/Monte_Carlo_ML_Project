"""Report rendering guards (ledger RR-14, F-21, AC-48).

The results table must never present an unscored quantity as a result: an
n_scored == 0 row renders as `unscored` with NO descriptive mean in any results
column, and every row shows scored/attempted so drops are visible, not inferred.

It must also never present a physical quantity without its unit: every reported
improvement column carries one, declared per metric and REFUSED when undeclared
(AC-48).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from amcd.reporting.tables import run_report

from tests.conftest import QUIET, tiny_config


#: What each metric's PRODUCER declares on its `MetricTriple` and carries through
#: `metrics.parquet` — mirrored here only so a fixture row looks like a real one.
#: The reporting layer no longer holds a map of its own (RD-201/AC-127).
_PRODUCER_UNITS = {
    "T30": "s", "EDT": "s", "C50": "dB", "energy_snr_db": "dB",
    "energy_mse": "operand_domain_squared",
}


def _summary_row(metric: str, **overrides) -> dict:
    row = {
        "split": "test_id", "metric": metric, "kind": "match_reference",
        "unit": _PRODUCER_UNITS.get(metric, ""),
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


def _render(rows: list[dict], value_domain: str = "db") -> str:
    cfg = tiny_config()
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "stats").mkdir()
        (run_dir / "stats" / "summary.json").write_text(json.dumps(rows))
        # The operand domain is read from preprocess's own stamp, never inferred
        # from a rep class (F-19), so the report needs it on disk (AC-48).
        (run_dir / "preprocessed").mkdir()
        (run_dir / "preprocessed" / "meta.json").write_text(
            json.dumps({"value_domain": value_domain})
        )
        run_report(cfg, run_dir, QUIET)
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


class TestReportedImprovementsCarryTheirUnit:
    """AC-48: `Imp mean`, the CI and MDES are `paired_improvement`'s output in the
    METRIC's own units, and one table mixes seconds (T30, EDT) with decibels
    (C50) — so a bare number is not interpretable.

    The unit CANNOT be derived from `kind` (RD-201): `kind` is
    match_reference|maximize|minimize, and T30 and C50 share `match_reference`
    while differing in unit. It is declared by the PRODUCER on its own
    `MetricTriple` and carried through `metrics.parquet`, so the reported column
    reads a declaration rather than a second map this layer keeps in step by hand.
    An undeclared metric is REFUSED rather than rendered blank — a blank unit
    beside a physical quantity is the silent exclusion the drop log exists to
    prevent.
    """

    def test_seconds_and_decibels_are_distinguished(self) -> None:
        txt = _render([_summary_row("T30"), _summary_row("C50")])
        t30 = next(l for l in txt.splitlines() if l.startswith("T30"))
        c50 = next(l for l in txt.splitlines() if l.startswith("C50"))
        assert " s " in t30, t30
        assert " dB " in c50, c50

    def test_an_operand_domain_metric_takes_the_stamped_domains_unit(self) -> None:
        """`energy_mse` is an operand-domain MSE (evaluation/signal.py), so its
        unit is the representation's — squared — not a fixed string."""
        db = _render([_summary_row("energy_mse")], value_domain="db")
        amp = _render([_summary_row("energy_mse")], value_domain="amplitude")
        assert "dB²" in next(l for l in db.splitlines() if l.startswith("energy_mse"))
        # `a.u.²`, not `amp²` — the amplitude domain is raw samples in arbitrary
        # units, and `amp` reads as the ampere (AC-125).
        assert "a.u.²" in next(l for l in amp.splitlines() if l.startswith("energy_mse"))

    def test_a_metric_with_no_declared_unit_is_refused_by_name(self) -> None:
        """The guard against the NEXT metric, not just today's five: a new metric
        must declare its unit rather than inherit a blank."""
        with pytest.raises(ValueError, match="c80_undeclared"):
            _render([_summary_row("c80_undeclared", unit="")])

    def test_an_undeclared_metric_is_refused_even_when_it_is_UNSCORED(self) -> None:
        """F-163/AC-130: the refusal must be over the metric SET, not the scored
        subset.

        `_metric_row` returns the `unscored` line before it needs a unit, so a
        lazy per-row lookup made this contract depend on the DATA: a metric added
        and validated on a run where it happened to be all-NaN passed, then
        crashed the report on the later run that scored it — the failure moving
        away in time from the change that caused it.
        """
        with pytest.raises(ValueError, match="c80_undeclared"):
            _render([_summary_row("c80_undeclared", n_scored=0, n_pred=0)])

    def test_an_unknown_operand_domain_is_refused(self) -> None:
        with pytest.raises(ValueError, match="value_domain"):
            _render([_summary_row("energy_mse")], value_domain="quefrency")
