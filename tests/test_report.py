"""Report rendering guards.

The results table must never present an unscored quantity as a result: an
n_scored == 0 row renders as `unscored` with NO descriptive mean in any results
column, and every row shows scored/attempted so drops are visible, not inferred.

It must also never present a physical quantity without its unit: every reported
improvement column carries one, declared per metric and REFUSED when undeclared.
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
#: The reporting layer no longer holds a map of its own.
_PRODUCER_UNITS = {
    "T30": "s", "EDT": "s", "C50": "dB", "energy_snr_db": "dB",
    "energy_mse": "operand_domain_squared",
}


def _summary_row(metric: str, **overrides) -> dict:
    row = {
        # attempted-population bound. Defaults match the scored columns with
        # nothing imputed, which is what a fully-scored split produces.
        "n_attempted_scorable": 3, "n_pred_unscored_imputed": 0,
        "improvement_mean_attempted": 0.1234,
        "improvement_ci_lower_attempted": 0.05,
        "improvement_ci_upper_attempted": 0.20,
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


def _render(rows: list[dict], value_domain: str = "db", config=None) -> str:
    cfg = tiny_config() if config is None else config
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "stats").mkdir()
        (run_dir / "stats" / "summary.json").write_text(json.dumps(rows))
        # The operand domain is read from preprocess's own stamp, never inferred
        # from a rep class, so the report needs it on disk.
        (run_dir / "preprocessed").mkdir()
        (run_dir / "preprocessed" / "meta.json").write_text(
            json.dumps({"value_domain": value_domain})
        )
        run_report(cfg, run_dir, QUIET)
        return (run_dir / "report" / "summary.txt").read_text()


def test_unscored_row_renders_unscored_with_no_numbers() -> None:
    """An n_scored == 0 row must say `unscored` and must NOT show its
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
    """The N column is scored/attempted, so a dropped scene is a visible
    gap (2/3), never a silently smaller population."""
    txt = _render([_summary_row("C50", n_scored=2, n_attempted=3, n_improved=1,
                                pct_improved=50.0)])
    c50_line = next(l for l in txt.splitlines() if l.startswith("C50"))
    assert "2/3" in c50_line


class TestReportedImprovementsCarryTheirUnit:
    """`Imp mean`, the CI and MDES are `paired_improvement`'s output in the
    METRIC's own units, and one table mixes seconds (T30, EDT) with decibels
    (C50) — so a bare number is not interpretable.

    The unit CANNOT be derived from `kind`: `kind` is
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
        unit is the representation's — squared — not a fixed string.

        End-to-end through the DECLARED domain only. The amplitude case is asserted
        against `_unit_for` below rather than by stamping a domain the configured
        representation does not declare: the report cross-checks the
        stamp against that declaration and refuses when they disagree, which is
        the point — a run_dir stamped `amplitude` under a `db` representation is
        exactly the mislabelling that check exists to catch.
        """
        db = _render([_summary_row("energy_mse")], value_domain="db")
        assert "dB²" in next(l for l in db.splitlines() if l.startswith("energy_mse"))

    def test_each_operand_domain_has_its_own_rendered_unit(self) -> None:
        """`a.u.²`, not `amp²` — the amplitude domain is raw samples in arbitrary
        units, and `amp` reads as the ampere."""
        from amcd.reporting.tables import _unit_for

        token = _PRODUCER_UNITS["energy_mse"]
        assert _unit_for("energy_mse", token, "db") == "dB²"
        assert _unit_for("energy_mse", token, "amplitude") == "a.u.²"

    def test_a_stamp_that_contradicts_the_configured_representation_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The stamp decides the unit, and it is an artifact of a stage that
        may have run under a different config. Trusting it alone lets the table
        label numbers produced in one domain with the unit of another."""
        with pytest.raises(ValueError, match="value_domain"):
            _render([_summary_row("energy_mse")], value_domain="amplitude")

    def test_a_metric_with_no_declared_unit_is_refused_by_name(self) -> None:
        """The guard against the NEXT metric, not just today's five: a new metric
        must declare its unit rather than inherit a blank."""
        with pytest.raises(ValueError, match="c80_undeclared"):
            _render([_summary_row("c80_undeclared", unit="")])

    def test_an_undeclared_metric_is_refused_even_when_it_is_UNSCORED(self) -> None:
        """The refusal must be over the metric SET, not the scored
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


class TestTheUnconvergedReferenceReachesTheReader:
    """The ray-budget probe measured that the high leg is NOT a converged
    reference for C50 — worst deviation 3.24 dB against the 1.0 dB ISO JND.

    Every paired improvement in this table is computed against that leg, so this is
    a caveat on the GROUND TRUTH rather than on the scored population, and the
    existing caveat machinery could not express it: a C50 row can be 12/12 scored,
    fully band-resolved and low-variance, and still rest on a premise measured
    false. It therefore appears on every row of the metric, in every split.
    """

    def test_every_row_of_an_unconverged_metric_carries_the_caveat(self) -> None:
        """C50 and T30 are both declared unconverged; EDT is not, and the caveat
        must discriminate — a label on every row would carry no information."""
        txt = _render([_summary_row("C50"), _summary_row("T30"), _summary_row("EDT")])
        rows = {m: next(l for l in txt.splitlines() if l.startswith(m))
                for m in ("C50", "T30", "EDT")}
        assert "reference unconverged" in rows["C50"], rows["C50"]
        assert "reference unconverged" in rows["T30"], rows["T30"]
        assert "reference unconverged" not in rows["EDT"], rows["EDT"]

    def test_a_fully_scored_row_still_carries_it(self) -> None:
        """The point of the row: nothing about the scored population reveals this."""
        txt = _render([_summary_row("C50", n_scored=3, n_attempted=3)])
        c50 = next(l for l in txt.splitlines() if l.startswith("C50"))
        assert "3/3" in c50 and "reference unconverged" in c50, c50

    def test_the_footer_states_the_measurement_and_the_tolerance(self) -> None:
        """"Unconverged" alone is not actionable — 3.24 dB against a 1.0 dB JND is
        a different fact from 1.1 dB."""
        txt = _render([_summary_row("C50")])
        assert "REFERENCE CONVERGENCE" in txt
        assert "3.24 dB" in txt, txt
        assert "12/20" in txt, txt
        assert "declared 1 dB" in txt, txt

    def test_the_csv_carries_it_too(self) -> None:
        """Rule: the machine-readable artifact is the one a downstream
        analysis opens, so it must not ship the improvement without the caveat."""
        import csv
        import tempfile

        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            (run_dir / "stats").mkdir()
            (run_dir / "stats" / "summary.json").write_text(
                json.dumps([_summary_row("C50"), _summary_row("EDT")])
            )
            (run_dir / "preprocessed").mkdir()
            (run_dir / "preprocessed" / "meta.json").write_text(
                json.dumps({"value_domain": "db"})
            )
            run_report(cfg, run_dir, QUIET)
            rows = list(csv.DictReader(
                (run_dir / "report" / "metrics_table.csv").open()
            ))
        by_metric = {r["metric"]: r["reference_convergence"] for r in rows}
        assert by_metric["C50"] == "unconverged", by_metric
        assert by_metric["EDT"] == "converged", by_metric

    def test_the_machine_readable_column_carries_all_three_states(self) -> None:
        """A boolean has to render "nobody measured this" as True, which is the
        reading `reference_unmeasured` exists to prevent. summary.txt separates
        the two; the artifact a downstream analysis actually opens must too."""
        import csv

        from amcd.config import UnmeasuredMetric

        cfg = tiny_config()
        # base.yaml declares T30 unconverged too; this test needs one metric in
        # each of the three states, so T30 is the converged one here.
        cfg.convergence.reference_unconverged = {
            "C50": cfg.convergence.reference_unconverged["C50"]
        }
        cfg.convergence.reference_unmeasured = {
            "EDT": UnmeasuredMetric(reason="never probed", gate="E4")
        }
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            (run_dir / "stats").mkdir()
            (run_dir / "stats" / "summary.json").write_text(
                json.dumps([_summary_row("C50"), _summary_row("EDT"),
                            _summary_row("T30")])
            )
            (run_dir / "preprocessed").mkdir()
            (run_dir / "preprocessed" / "meta.json").write_text(
                json.dumps({"value_domain": "db"})
            )
            run_report(cfg, run_dir, QUIET)
            rows = list(csv.DictReader(
                (run_dir / "report" / "metrics_table.csv").open()
            ))
        by_metric = {r["metric"]: r["reference_convergence"] for r in rows}
        assert by_metric["C50"] == "unconverged", by_metric
        assert by_metric["EDT"] == "unmeasured", by_metric
        assert by_metric["T30"] == "converged", by_metric

    def test_a_declaration_this_run_cannot_apply_is_LOGGED_not_silent(self) -> None:
        """A caveat that matches no reported metric is a SKIP, and this project logs
        every skip with its reason.

        Not refused: a run may legitimately report a subset — the energy metrics
        without the ISO ones — and raising there would fail a correct config on a
        correct run. But a misspelt metric name would otherwise vanish entirely,
        taking the finding with it, so the footer names it either way.
        """
        cfg = tiny_config()
        cfg.convergence.reference_unconverged = {
            "C80": cfg.convergence.reference_unconverged["C50"]
        }
        txt = _render([_summary_row("C50")])
        assert "C80" not in txt, "the caveat was applied to a metric it does not name"

        txt = _render([_summary_row("C50")], config=cfg)
        assert "C80: declared unconverged, but SKIPPED" in txt, txt
        c50 = next(l for l in txt.splitlines() if l.startswith("C50 "))
        assert "reference unconverged" not in c50, (
            "a declaration for C80 must not caveat C50's rows"
        )

    def _render_to_file(self, cfg) -> str:
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            (run_dir / "stats").mkdir()
            (run_dir / "stats" / "summary.json").write_text(
                json.dumps([_summary_row("C50")])
            )
            (run_dir / "preprocessed").mkdir()
            (run_dir / "preprocessed" / "meta.json").write_text(
                json.dumps({"value_domain": "db"})
            )
            run_report(cfg, run_dir, QUIET)
            return (run_dir / "report" / "summary.txt").read_text()

    def test_nothing_is_said_when_the_reference_is_converged(self) -> None:
        """The footer must not carry a permanent paragraph about a resolved
        concern: if a later probe clears C50, emptying the maps clears the text."""
        cfg = tiny_config()
        cfg.convergence.reference_unconverged = {}
        cfg.convergence.reference_unmeasured = {}
        txt = self._render_to_file(cfg)
        assert "REFERENCE CONVERGENCE" not in txt
        assert "reference unconverged" not in txt

    def test_a_quantity_nobody_measured_is_named_in_the_footer(self) -> None:
        """The third state has to REACH the reader, or it is the two-state contract
        again: an unmeasured quantity that renders nothing is indistinguishable from
        a converged one, which is exactly what the map exists to prevent."""
        cfg = tiny_config()
        cfg.convergence.reference_unconverged = {}
        txt = self._render_to_file(cfg)
        assert "NEVER MEASURED" in txt
        assert "band_energy" in txt
        assert "E4 ray-count scoping" in txt

    def test_an_unmeasured_quantity_is_not_reported_as_a_measurement(self) -> None:
        """It carries no deviation and no cell counts by construction, so it must
        not render in the measured-and-failed section."""
        cfg = tiny_config()
        cfg.convergence.reference_unconverged = {}
        txt = self._render_to_file(cfg)
        measured = txt.split("NEVER MEASURED")[0]
        assert "band_energy" not in measured
        assert "MEASURED AND FAILED" not in measured

    def test_a_quantity_cannot_be_both_unconverged_and_unmeasured(self) -> None:
        from amcd.config import ConvergenceTolerance

        with pytest.raises(Exception, match="BOTH"):
            ConvergenceTolerance(
                t30_frac=0.05, c50_db=1.0, band_energy_frac=0.05,
                reference_unconverged={"C50": {
                    "worst_deviation": 3.24, "tolerance": 1.0, "unit": "dB",
                    "n_within_tolerance": 12, "n_cells": 20,
                }},
                reference_unmeasured={"C50": {"reason": "r", "gate": "g"}},
            )


class TestTheModelsOwnFailuresAreBoundedNotDropped:
    """A scene where the model produced nothing measurable leaves the scored
    population, so the CI beside it is conditioned on the model having WORKED.

    That is optimistic, and the loss is not uniform: the censoring probability
    correlates with room absorption, which is `test_material_shift`'s own declared
    axis. Deriving the band set from the physical legs trades a ground-truth
    contamination for this selection, and only the
    first was discussed. The bound is the same bootstrap with those scenes imputed
    at ZERO improvement — the honest floor for a failure, since the model did not
    make the metric worse than the baseline, it produced nothing.
    """

    def test_a_run_with_a_model_failure_shows_BOTH_populations(self) -> None:
        txt = _render([_summary_row(
            "T30", n_scored=3, n_attempted=4,
            n_attempted_scorable=4, n_pred_unscored_imputed=1,
            improvement_mean_attempted=0.0926,
            improvement_ci_lower_attempted=0.0100,
            improvement_ci_upper_attempted=0.1700,
        )])
        t30 = next(l for l in txt.splitlines() if l.startswith("T30"))
        assert "0.1234" in t30 and "3/4" in t30, t30
        assert "1 model-failed" in t30, t30

        bound = next(l for l in txt.splitlines() if "attempted-population bound" in l)
        assert "0.0926" in bound, bound
        assert "[0.0100, 0.1700]" in bound, bound
        assert "1 imputed at 0" in bound, bound

    def test_a_fully_scored_row_gets_no_second_line(self) -> None:
        """The bound is a second ESTIMATE of the same quantity over a different
        population. Where the populations coincide, printing it twice would invite
        reading one as a correction of the other."""
        txt = _render([_summary_row("T30")])
        assert "attempted-population bound" not in txt
        # The footer LEGEND explains `model-failed` unconditionally; what must be
        # absent is the caveat on the row.
        t30 = next(l for l in txt.splitlines() if l.startswith("T30"))
        assert "model-failed" not in t30, t30

    def test_the_footer_says_which_scenes_are_in_neither_population(self) -> None:
        """A scene whose PHYSICAL legs failed has no ground truth to have improved
        on, so imputing zero for it would invent a datum rather than bound one."""
        txt = _render([_summary_row(
            "T30", n_pred_unscored_imputed=1, n_attempted_scorable=4
        )])
        assert "imputed at ZERO improvement" in txt
        assert "PHYSICAL legs failed are in neither population" in txt


class TestTheAttemptedBoundIsComputedOverTheRightScenes:
    """The aggregation half of the pred-side selection bound, asserted through
    `run_stats` rather than
    through a fixture: which scenes are imputed is the load-bearing decision."""

    def _metrics_frame(self, rows: list[dict]):
        import pandas as pd

        return pd.DataFrame(rows)

    def _row(self, scene: str, low, pred, high, improved=None):
        return {
            "scene_id": scene, "split": "test_id", "metric": "T30",
            "kind": "match_reference", "unit": "s",
            "low_val": low, "pred_val": pred, "high_ref": high,
            "improved": improved,
        }

    def _summary(self, tmp_path: Path, rows: list[dict]) -> dict:
        from amcd.stats.aggregate import run_stats

        cfg = tiny_config()
        (tmp_path / "metrics").mkdir(parents=True, exist_ok=True)
        self._metrics_frame(rows).to_parquet(tmp_path / "metrics" / "metrics.parquet")
        run_stats(cfg, tmp_path, QUIET)
        summary = json.loads((tmp_path / "stats" / "summary.json").read_text())
        return next(r for r in summary if r["split"] == "test_id" and r["metric"] == "T30")

    def test_a_pred_failure_is_imputed_and_a_physical_failure_is_not(
        self, tmp_path: Path
    ) -> None:
        nan = float("nan")
        row = self._summary(tmp_path, [
            # Three scored scenes, all improving.
            self._row("s0", 1.0, 0.55, 0.5, True),
            self._row("s1", 1.0, 0.55, 0.5, True),
            self._row("s2", 1.0, 0.55, 0.5, True),
            # pred failed, physics fine → imputed at zero improvement.
            self._row("s3", 1.0, nan, 0.5, None),
            # physics failed → in NEITHER population; nothing to have improved on.
            self._row("s4", 1.0, 0.55, nan, None),
        ])
        assert row["n_scored"] == 3, row
        assert row["n_attempted"] == 5, row
        assert row["n_pred_unscored_imputed"] == 1, row
        assert row["n_attempted_scorable"] == 4, (
            "the physically-unscorable scene must not enter the bound's population"
        )

    def test_the_bound_is_lower_than_the_conditioned_estimate(
        self, tmp_path: Path
    ) -> None:
        """The whole point: dropping model failures inflates the reported help."""
        nan = float("nan")
        row = self._summary(tmp_path, [
            self._row("s0", 1.0, 0.55, 0.5, True),
            self._row("s1", 1.0, 0.55, 0.5, True),
            self._row("s2", 1.0, 0.55, 0.5, True),
            self._row("s3", 1.0, nan, 0.5, None),
        ])
        assert row["improvement_mean_attempted"] < row["improvement_mean"], row
        assert row["improvement_mean_attempted"] == pytest.approx(
            row["improvement_mean"] * 3 / 4
        ), "a zero imputed into n=3 must move the mean by exactly 3/4"

    def test_with_no_failures_the_two_populations_agree_exactly(
        self, tmp_path: Path
    ) -> None:
        row = self._summary(tmp_path, [
            self._row("s0", 1.0, 0.55, 0.5, True),
            self._row("s1", 1.0, 0.60, 0.5, True),
            self._row("s2", 1.0, 0.52, 0.5, True),
        ])
        assert row["n_pred_unscored_imputed"] == 0
        assert row["improvement_mean_attempted"] == pytest.approx(
            row["improvement_mean"]
        )


class TestEachSplitSectionCarriesItsOwnRecordLengthCount:
    """The record-length gate is the OVERALL over-limit fraction across
    regimes, which is the one aggregation invariant #9 forbids for results.

    That is the right GATE: a per-split gate would let the smallest split set the
    tolerance for train. But research_i's 0.01 over 720 scenes permits 7 over-limit
    scenes that could ALL sit in the 30-scene `test_geometry_shift` — 23 % of it —
    and still pass, and the per-shift breakdown IS the research result. It reached
    only scenes/placement_report.json, surfacing in the console solely when the gate
    tripped.
    """

    def _run_dir(self, tmp_path: Path, placement: dict | None) -> Path:
        (tmp_path / "stats").mkdir()
        (tmp_path / "stats" / "summary.json").write_text(json.dumps([
            _summary_row("T30", split="test_id"),
            _summary_row("T30", split="test_geometry_shift"),
        ]))
        (tmp_path / "preprocessed").mkdir()
        (tmp_path / "preprocessed" / "meta.json").write_text(
            json.dumps({"value_domain": "db"})
        )
        if placement is not None:
            (tmp_path / "scenes").mkdir()
            (tmp_path / "scenes" / "placement_report.json").write_text(
                json.dumps(placement)
            )
        return tmp_path

    @staticmethod
    def _entry(over: int, scored: int) -> dict:
        return {"record_decay_range": {
            "n_scenes": scored,
            "decay_range_below_iso_t30": {"count": over, "fraction": over / scored},
        }}

    def test_each_split_reports_its_OWN_count(self, tmp_path: Path) -> None:
        cfg = tiny_config()
        run_dir = self._run_dir(tmp_path, {
            "id": self._entry(0, 20),
            "test_geometry_shift": self._entry(7, 30),
        })
        run_report(cfg, run_dir, QUIET)
        txt = (run_dir / "report" / "summary.txt").read_text()

        lines = txt.splitlines()
        geo_at = next(i for i, l in enumerate(lines) if "test_geometry_shift" in l)
        note = next(l for l in lines[geo_at:] if l.startswith("Record length"))
        assert "7/30" in note and "23.3%" in note, note
        # And the clean split says so rather than inheriting the shift split's count.
        id_at = next(i for i, l in enumerate(lines) if "(test_id," in l)
        id_note = next(l for l in lines[id_at:] if l.startswith("Record length"))
        assert "0/20" in id_note, id_note

    def test_the_pooled_id_regime_is_named_as_pooled(self, tmp_path: Path) -> None:
        """In frac mode `train`/`valid`/`test_id` share one `id` entry, so a
        count printed beside `test_id` covers all three. Saying `split 'test_id'`
        there would invite the reader to size it against test_id alone."""
        cfg = tiny_config()
        run_dir = self._run_dir(tmp_path, {
            "id": self._entry(3, 20),
            "test_geometry_shift": self._entry(0, 30),
        })
        run_report(cfg, run_dir, QUIET)
        txt = (run_dir / "report" / "summary.txt").read_text()
        note = next(
            l for l in txt.splitlines()
            if l.startswith("Record length") and "3/20" in l
        )
        assert "regime 'id'" in note and "pooling" in note, note

    def test_an_unscored_regime_is_not_reported_as_zero(self, tmp_path: Path) -> None:
        """A fraction whose denominator is zero is UNDEFINED, and 0.0
        would read as 'measured, and within the record'."""
        cfg = tiny_config()
        run_dir = self._run_dir(tmp_path, {
            "id": self._entry(0, 20),
            "test_geometry_shift": {"record_decay_range": {
                "n_scenes": 0, "decay_range_below_iso_t30": {"count": 0},
            }},
        })
        run_report(cfg, run_dir, QUIET)
        txt = (run_dir / "report" / "summary.txt").read_text()
        assert "UNSCORED" in txt and "undefined, not 0" in txt

    def test_a_run_dir_with_no_placement_report_still_renders(
        self, tmp_path: Path
    ) -> None:
        """The disclosure is about scene generation; its absence is not a reason to
        refuse a metrics table."""
        cfg = tiny_config()
        run_dir = self._run_dir(tmp_path, None)
        run_report(cfg, run_dir, QUIET)
        txt = (run_dir / "report" / "summary.txt").read_text()
        assert "Record length" not in txt
        assert "T30" in txt
