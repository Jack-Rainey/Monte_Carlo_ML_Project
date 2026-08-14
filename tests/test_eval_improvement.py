"""Per-metric improvement kill tests.

The improvement flag is computed PER METRIC from that metric's own
(low, pred, high) triple — a metric can never inherit another metric's flag —
and the stats improvement pct stays bounded [0, 100] even with a NaN-bearing
group, the `test_geometry_shift/C50 = 150%` pathology.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amcd.evaluation.metric_row import MetricTriple, metric_improvement, paired_improvement
from amcd.stats.aggregate import run_stats

from tests.conftest import QUIET, tiny_config


def test_improvement_is_per_metric_not_borrowed() -> None:
    """A scene where energy MSE improves but T30 worsens must report
    improved=True for energy and improved=False for T30 — no shared flag."""
    # energy_mse: high_ref = 0, pred closer to 0 than baseline → improved.
    energy = MetricTriple(low=4.0, pred=1.0, high=0.0, kind="match_reference", unit="s")
    # T30: prediction lands FARTHER from the high reference than the baseline does.
    t30 = MetricTriple(low=0.50, pred=0.90, high=0.55, kind="match_reference", unit="s")

    energy_improved, energy_ratio = metric_improvement(energy)
    t30_improved, t30_ratio = metric_improvement(t30)

    assert energy_improved is True, "energy MSE should improve on this triple"
    assert t30_improved is False, (
        "T30 worsened but reported improved — the flag was borrowed from energy"
    )
    assert energy_ratio > 1.0 and t30_ratio < 1.0


def test_undefined_improvement_is_none_not_false() -> None:
    """A NaN in any CONSUMED leg of the triple → improvement undefined (None), never
    False — so `stats` excludes it from BOTH numerator and denominator (precondition)."""
    for triple in (
        MetricTriple(np.nan, np.nan, np.nan, kind="match_reference", unit="s"),
        MetricTriple(np.nan, 12.3, np.nan, kind="match_reference", unit="s"),  # missing baseline+ref
        MetricTriple(0.5, np.nan, 0.4, kind="match_reference", unit="s"),      # missing prediction
        MetricTriple(np.nan, 12.3, np.nan, kind="maximize", unit="s"),         # missing baseline leg
        MetricTriple(3.0, np.nan, np.nan, kind="minimize", unit="s"),          # missing prediction
    ):
        improved, ratio = metric_improvement(triple)
        assert improved is None and np.isnan(ratio)


def test_maximize_minimize_improvement_semantics() -> None:
    """A maximize/minimize metric scores from (low, pred) alone — the high
    reference leg is structurally absent (NaN) and must NOT unscore it (that NaN
    is exactly what unscored energy_snr_db pre-fix). Pre-fix behaviour returns
    None here; the kind taxonomy returns a real flag."""
    up_good = MetricTriple(low=10.0, pred=14.0, high=np.nan, kind="maximize", unit="s")
    up_bad = MetricTriple(low=10.0, pred=7.0, high=np.nan, kind="maximize", unit="s")
    down_good = MetricTriple(low=5.0, pred=2.0, high=np.nan, kind="minimize", unit="s")
    down_bad = MetricTriple(low=5.0, pred=8.0, high=np.nan, kind="minimize", unit="s")

    assert metric_improvement(up_good)[0] is True
    assert metric_improvement(up_bad)[0] is False
    assert metric_improvement(down_good)[0] is True
    assert metric_improvement(down_bad)[0] is False
    # baseline_rel_ratio is a match-reference-only diagnostic → NaN here.
    assert np.isnan(metric_improvement(up_good)[1])


def test_unknown_kind_fails_loud() -> None:
    """No hidden default: an undeclared/unknown kind is an error, never a
    silent fall-back to match-reference."""
    bad = MetricTriple(1.0, 2.0, 3.0, kind="best_effort", unit="s")
    with pytest.raises(ValueError, match="kind"):
        metric_improvement(bad)
    with pytest.raises(ValueError, match="kind"):
        paired_improvement(bad)


def test_paired_improvement_sign_consistent_for_maximize_minimize() -> None:
    """paired > 0 ⟺ improved, for the non-match-reference kinds too — the stats
    quantity and the eval flag stay the same comparison in different forms."""
    rng = np.random.default_rng(99)
    for kind in ("maximize", "minimize"):
        for _ in range(100):
            low, pred = rng.normal(0.0, 2.0, 2)
            t = MetricTriple(low, pred, np.nan, kind=kind, unit="s")
            improved, _ = metric_improvement(t)
            assert (paired_improvement(t) > 0) == improved


def test_pct_improved_bounded_with_nan_group() -> None:
    """A (split, metric) group of 3 scenes with one NaN row must yield
    pct_improved ∈ [0, 100]. Pre-fix, n=2 (NaN dropped) while n_improved counted the
    full group → 150%."""
    cfg = tiny_config()
    rows = []
    # Metric with 3 scenes in one split; scene 3 has NaN pred_val and undefined
    # improvement (the contract the evaluator now guarantees).
    for i, (pred, imp) in enumerate([(1.0, True), (2.0, True), (np.nan, None)]):
        rows.append({
            "scene_id": f"scene_{i}", "split": "test_geometry_shift",
            "metric": "C50", "kind": "match_reference", "low_val": 3.0,
            "pred_val": pred, "high_ref": 2.0, "baseline_rel_ratio": 1.0,
            "improved": imp,
        })
    # A hostile row set: improved=True even where pred_val is NaN. stats must STILL
    # never exceed 100 (defends the bound independent of upstream correctness).
    for i, (pred, imp) in enumerate([(1.0, True), (2.0, True), (np.nan, True)]):
        rows.append({
            "scene_id": f"h_{i}", "split": "test_geometry_shift",
            "metric": "T30", "kind": "match_reference", "low_val": 3.0,
            "pred_val": pred, "high_ref": 2.0, "baseline_rel_ratio": 1.0,
            "improved": imp,
        })

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        pd.DataFrame(rows).to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir, QUIET)

        ci = pd.read_csv(run_dir / "stats" / "ci_table.csv")

    for _, r in ci.iterrows():
        if r["n_scored"] > 0:
            assert 0.0 <= r["pct_improved"] <= 100.0, (
                f"{r['metric']}: pct_improved={r['pct_improved']} out of [0,100] "
                f"(n_improved={r['n_improved']}, n_scored={r['n_scored']})"
            )


def test_pct_bounded_when_improved_column_is_pure_bool() -> None:
    """Guard the currently-implicit dtype invariant: the bound must hold even
    when `improved` round-trips as pure `bool` (no None anywhere — the case when
    every metric is defined for every scene and no diagnostic-only metric injects
    None). Pins `improved.notna()` + `== True` against a silent dtype regression."""
    cfg = tiny_config()
    rows = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "T30",
         "kind": "match_reference", "low_val": 3.0, "pred_val": float(i + 1),
         "high_ref": 2.0, "baseline_rel_ratio": 1.0, "improved": imp}
        for i, imp in enumerate([True, False, True])
    ]
    df = pd.DataFrame(rows)
    assert df["improved"].dtype == bool, "fixture must exercise the pure-bool path"

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        df.to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir, QUIET)
        ci = pd.read_csv(run_dir / "stats" / "ci_table.csv")

    row = ci[ci["metric"] == "T30"].iloc[0]
    assert row["n_scored"] == 3 and row["n_improved"] == 2
    assert 0.0 <= row["pct_improved"] <= 100.0
