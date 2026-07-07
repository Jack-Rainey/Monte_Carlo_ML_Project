"""Per-metric improvement kill tests (ledger F-07, F-08).

KT-1 proves the improvement flag is computed PER METRIC from that metric's own
(low, pred, high) triple — a metric can never inherit another metric's flag
(F-07). KT-2 proves the stats improvement pct is bounded [0, 100] even with a
NaN-bearing group, the exact `test_geometry_shift/C50 = 150%` pathology (F-08).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from amcd.evaluation.metric_row import MetricTriple, metric_improvement
from amcd.stats.aggregate import run_stats

from tests.conftest import tiny_config


def test_improvement_is_per_metric_not_borrowed() -> None:
    """KT-1 (F-07): a scene where energy MSE improves but T30 worsens must report
    improved=True for energy and improved=False for T30 — no shared flag."""
    # energy_mse: high_ref = 0, pred closer to 0 than baseline → improved.
    energy = MetricTriple(low=4.0, pred=1.0, high=0.0)
    # T30: prediction lands FARTHER from the high reference than the baseline does.
    t30 = MetricTriple(low=0.50, pred=0.90, high=0.55)

    energy_improved, energy_ratio = metric_improvement(energy)
    t30_improved, t30_ratio = metric_improvement(t30)

    assert energy_improved is True, "energy MSE should improve on this triple"
    assert t30_improved is False, (
        "T30 worsened but reported improved — the flag was borrowed from energy (F-07)"
    )
    assert energy_ratio > 1.0 and t30_ratio < 1.0


def test_undefined_improvement_is_none_not_false() -> None:
    """A NaN in any leg of the triple → improvement undefined (None), never False —
    so `stats` excludes it from BOTH numerator and denominator (F-08 precondition)."""
    for triple in (
        MetricTriple(np.nan, np.nan, np.nan),
        MetricTriple(np.nan, 12.3, np.nan),   # diagnostic-only: pred, no baseline
        MetricTriple(0.5, np.nan, 0.4),       # missing prediction
    ):
        improved, ratio = metric_improvement(triple)
        assert improved is None and np.isnan(ratio)


def test_pct_improved_bounded_with_nan_group() -> None:
    """KT-2 (F-08): a (split, metric) group of 3 scenes with one NaN row must yield
    pct_improved ∈ [0, 100]. Pre-fix, n=2 (NaN dropped) while n_improved counted the
    full group → 150%."""
    cfg = tiny_config()
    rows = []
    # Metric with 3 scenes in one split; scene 3 has NaN pred_val and undefined
    # improvement (the contract the evaluator now guarantees).
    for i, (pred, imp) in enumerate([(1.0, True), (2.0, True), (np.nan, None)]):
        rows.append({
            "scene_id": f"scene_{i}", "split": "test_geometry_shift",
            "metric": "C50", "low_val": 3.0, "pred_val": pred,
            "high_ref": 2.0, "baseline_rel_ratio": 1.0, "improved": imp,
        })
    # A hostile row set: improved=True even where pred_val is NaN. stats must STILL
    # never exceed 100 (defends the bound independent of upstream correctness).
    for i, (pred, imp) in enumerate([(1.0, True), (2.0, True), (np.nan, True)]):
        rows.append({
            "scene_id": f"h_{i}", "split": "test_geometry_shift",
            "metric": "T30", "low_val": 3.0, "pred_val": pred,
            "high_ref": 2.0, "baseline_rel_ratio": 1.0, "improved": imp,
        })

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        pd.DataFrame(rows).to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir)

        ci = pd.read_csv(run_dir / "stats" / "ci_table.csv")

    for _, r in ci.iterrows():
        if r["n_scored"] > 0:
            assert 0.0 <= r["pct_improved"] <= 100.0, (
                f"{r['metric']}: pct_improved={r['pct_improved']} out of [0,100] "
                f"(n_improved={r['n_improved']}, n_scored={r['n_scored']}) — F-08"
            )


def test_pct_bounded_when_improved_column_is_pure_bool() -> None:
    """Guard the currently-implicit dtype invariant: the F-08 bound must hold even
    when `improved` round-trips as pure `bool` (no None anywhere — the case when
    every metric is defined for every scene and no diagnostic-only metric injects
    None). Pins `improved.notna()` + `== True` against a silent dtype regression."""
    cfg = tiny_config()
    rows = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "T30",
         "low_val": 3.0, "pred_val": float(i + 1), "high_ref": 2.0,
         "baseline_rel_ratio": 1.0, "improved": imp}
        for i, imp in enumerate([True, False, True])
    ]
    df = pd.DataFrame(rows)
    assert df["improved"].dtype == bool, "fixture must exercise the pure-bool path"

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        df.to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir)
        ci = pd.read_csv(run_dir / "stats" / "ci_table.csv")

    row = ci[ci["metric"] == "T30"].iloc[0]
    assert row["n_scored"] == 3 and row["n_improved"] == 2
    assert 0.0 <= row["pct_improved"] <= 100.0
