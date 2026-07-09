"""Stats-stage numerics (ledger F-09, F-13/F-14/F-16, F-15, F-18).

MDES (minimum detectable effect size) is the load-bearing "could we even have seen
an effect this small?" number at this design's small per-split n. These pin:
the unbiased sample std (ddof=1, F-09); the exact noncentral-t solver
(F-13/F-14/F-16); per-group bootstrap substreams (F-15); and — F-18 — that the
inferential σ is the std of the per-scene PAIRED improvement |low−high| − |pred−high|
(design_spec §9), never the std of the absolute pred value, which diverges from it
by up to ~2.4× in the dry run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.stats

from amcd.stats.aggregate import bootstrap_ci, mdes, run_stats

from tests.conftest import tiny_config


def _achieved_power(ncp: float, df: int, t_crit: float) -> float:
    """Independent oracle: two-sided noncentral-t power at `ncp`, NOT reusing the
    module's bisection. Prefer scipy's `nct` where it is finite (a genuinely
    independent implementation); fall back to the chi-square-mixture integral in the
    small-df hole zone where `nct.cdf` returns NaN (this is exactly the F-16 regime
    the raw scipy recompute could not check)."""
    p = scipy.stats.nct.sf(t_crit, df, ncp) + scipy.stats.nct.cdf(-t_crit, df, ncp)
    if np.isfinite(p):
        return float(p)
    integrand = lambda v: (  # noqa: E731
        scipy.stats.norm.cdf(ncp - t_crit * np.sqrt(v / df))
        + scipy.stats.norm.cdf(-t_crit * np.sqrt(v / df) - ncp)
    ) * scipy.stats.chi2.pdf(v, df)
    return float(scipy.integrate.quad(integrand, 0.0, np.inf)[0])


def test_bootstrap_ci_reports_sample_std_not_population() -> None:
    vals = np.array([1.0, 2.0, 4.0])
    ci = bootstrap_ci(vals, n_resamples=200, alpha=0.05, rng=np.random.default_rng(0))
    assert ci["std"] == pytest.approx(float(np.std(vals, ddof=1)))
    # The pre-fix population std (ddof=0) is smaller — guard against its return.
    assert ci["std"] != pytest.approx(float(np.std(vals, ddof=0)))


@pytest.mark.parametrize(
    "n, power, alpha",
    [
        (3, 0.8, 0.05),   # smallest shipped split, base config
        (4, 0.8, 0.05),
        (5, 0.8, 0.05),
        (8, 0.8, 0.05),
        # F-16 hole zone: at low alpha / high power the root sits where scipy's
        # nct.cdf is NaN, so the old nudge-then-bisect solver truncated the bracket
        # below the root and understated MDES. Plausible as a multiple-comparison
        # correction across the 6 splits. The hole-free evaluator must nail it.
        (3, 0.95, 0.01),
        (3, 0.99, 0.001),
    ],
)
def test_mdes_achieves_target_power_via_noncentral_t(n: int, power: float, alpha: float) -> None:
    """F-13/F-16: for n≥3 (df≥2) MDES is the EXACT small-sample effect — plugging its
    noncentrality back into the two-sided noncentral-t power reproduces the target
    power. Non-tautological: `_achieved_power` recomputes power from the returned
    effect independently of the bisection that found it (scipy `nct` where finite, a
    separate integral in the hole zone)."""
    std_sample = 1.5
    d = mdes(std_sample, n, power, alpha)
    assert np.isfinite(d) and d > 0, f"mdes non-finite/≤0 at n={n}, power={power}, alpha={alpha}"

    df = n - 1
    ncp = d * np.sqrt(n) / std_sample
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df)
    achieved = _achieved_power(ncp, df, t_crit)
    assert achieved == pytest.approx(power, abs=1e-4), (
        f"n={n}, power={power}, alpha={alpha}: MDES effect achieves power {achieved:.5f} ≠ target"
    )


def test_mdes_returns_nan_for_n2_df1() -> None:
    """F-14: at n=2 (df=1) the earlier clamp-to-1.0 solver returned a finite,
    ~2×-too-small MDES (overstating detectability). MDES must instead be N/A there —
    an unpowered 1-df test — never a finite value that misses target power."""
    assert np.isnan(mdes(1.5, 2, 0.8, 0.05))
    assert np.isnan(mdes(1.5, 1, 0.8, 0.05))  # n≤1 unchanged guard


def test_mdes_exceeds_large_sample_z_form_at_small_n() -> None:
    """The old large-sample z-form understated MDES ~2× at n=3; the noncentral-t
    fix must exceed it clearly, so the approximation is no longer what we report."""
    std_sample, n, power, alpha = 1.5, 3, 0.8, 0.05
    d = mdes(std_sample, n, power, alpha)
    z_form = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(power)) * std_sample / np.sqrt(n)
    assert d > 1.9 * z_form


def test_mdes_converges_to_normal_form_at_large_n() -> None:
    """Sanity: as n grows the noncentral-t MDES approaches the large-sample z-form
    (the approximation is only wrong at small n)."""
    std_sample, n, power, alpha = 1.0, 500, 0.8, 0.05
    d = mdes(std_sample, n, power, alpha)
    z_form = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(power)) * std_sample / np.sqrt(n)
    assert d == pytest.approx(z_form, rel=0.02)


def _run_stats_on_rows(cfg, rows: list[dict]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        pd.DataFrame(rows).to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir)
        return pd.read_csv(run_dir / "stats" / "ci_table.csv")


def test_run_stats_mdes_uses_paired_improvement_sigma() -> None:
    """F-18: MDES (and improvement_std) come from the per-scene PAIRED improvement
    |low−high| − |pred−high| (design_spec §9), never from the absolute pred values.
    Fixture is built so std(pred_val) ≠ std(paired) — following the wrong σ fails.
    Also pins F-09 (ddof=1 sample std) on the paired quantity, and that the
    descriptive pred_std column still carries the absolute-value sample std."""
    cfg = tiny_config()
    # (low, pred, high): paired = |low−high| − |pred−high| = [0.30, 0.05, −0.10];
    # pred vals = [0.20, 0.35, 0.55]. std(pred)≈0.176 vs std(paired)≈0.202 — distinct.
    triples = [(0.90, 0.20, 0.40), (0.50, 0.35, 0.40), (0.45, 0.55, 0.40)]
    rows = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "T30",
         "kind": "match_reference", "low_val": lo, "pred_val": pr, "high_ref": hi,
         "baseline_rel_ratio": 1.0,
         "improved": abs(pr - hi) < abs(lo - hi)}
        for i, (lo, pr, hi) in enumerate(triples)
    ]
    ci = _run_stats_on_rows(cfg, rows)
    row = ci[ci["metric"] == "T30"].iloc[0]

    paired = [abs(lo - hi) - abs(pr - hi) for lo, pr, hi in triples]
    paired_std = float(np.std(paired, ddof=1))
    pred_std = float(np.std([pr for _, pr, _ in triples], ddof=1))
    assert paired_std != pytest.approx(pred_std)  # fixture really separates the two σ

    assert row["improvement_mean"] == pytest.approx(float(np.mean(paired)))
    assert row["improvement_std"] == pytest.approx(paired_std)
    assert row["improvement_mdes"] == pytest.approx(
        mdes(paired_std, len(paired), cfg.bootstrap_power, cfg.bootstrap_alpha)
    )
    assert row["improvement_mdes"] != pytest.approx(
        mdes(pred_std, len(paired), cfg.bootstrap_power, cfg.bootstrap_alpha)
    )
    # Descriptive absolute-value columns remain, clearly separated.
    assert row["pred_std"] == pytest.approx(pred_std)
    assert row["n_pred"] == 3 and row["n_scored"] == 3


def test_paired_improvement_sign_matches_improved_flag() -> None:
    """F-18 consistency: paired > 0 ⟺ improved == True, row by row, against
    metric_improvement itself — the stats-stage quantity and the eval-stage flag
    are the same comparison in different forms."""
    from amcd.evaluation.metric_row import MetricTriple, metric_improvement

    rng = np.random.default_rng(1234)
    for _ in range(200):
        low, pred, high = rng.normal(0.0, 2.0, 3)
        improved, _ = metric_improvement(MetricTriple(low, pred, high, kind="match_reference"))
        paired = abs(low - high) - abs(pred - high)
        assert (paired > 0) == improved, f"sign mismatch at {(low, pred, high)}"


def test_bootstrap_substream_per_group_is_order_and_set_invariant() -> None:
    """F-15: a group's CI bounds must not depend on which OTHER groups exist —
    each (split, metric, quantity) gets its own seeded substream. Pre-fix, one
    shared RNG stream meant adding metric B perturbed metric A's bounds."""
    cfg = tiny_config()

    def t30_rows() -> list[dict]:
        return [
            {"scene_id": f"s_{i}", "split": "test_id", "metric": "T30",
             "kind": "match_reference", "low_val": 0.9, "pred_val": p, "high_ref": 0.4,
             "baseline_rel_ratio": 1.0, "improved": True}
            for i, p in enumerate([0.20, 0.35, 0.55, 0.42, 0.61])
        ]

    extra = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "C50",
         "kind": "match_reference", "low_val": 3.0, "pred_val": p, "high_ref": 2.0,
         "baseline_rel_ratio": 1.0, "improved": True}
        for i, p in enumerate([1.0, 2.5, 2.2, 1.8])
    ]

    alone = _run_stats_on_rows(cfg, t30_rows())
    with_extra = _run_stats_on_rows(cfg, extra + t30_rows())

    a = alone[alone["metric"] == "T30"].iloc[0]
    b = with_extra[with_extra["metric"] == "T30"].iloc[0]
    for col in ("improvement_ci_lower", "improvement_ci_upper",
                "pred_ci_lower", "pred_ci_upper", "improvement_mdes"):
        assert a[col] == b[col], f"{col} of T30 changed when C50 was added (F-15)"


def test_run_stats_maximize_kind_uses_pred_minus_low() -> None:
    """F-20: a `maximize` metric's inferential quantity is pred − low — computed
    from (low, pred) alone, with the high leg structurally absent (NaN). Pre-fix,
    the spine's hardcoded |low−high| − |pred−high| saw the NaN high leg and
    silently unscored the whole group (energy_snr_db: n_scored 0 in every split)."""
    cfg = tiny_config()
    pairs = [(10.0, 14.0), (11.0, 9.5), (8.0, 12.0)]  # (low_snr, pred_snr)
    rows = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "energy_snr_db",
         "kind": "maximize", "low_val": lo, "pred_val": pr, "high_ref": np.nan,
         "baseline_rel_ratio": np.nan, "improved": pr > lo}
        for i, (lo, pr) in enumerate(pairs)
    ]
    ci = _run_stats_on_rows(cfg, rows)
    row = ci[ci["metric"] == "energy_snr_db"].iloc[0]

    paired = [pr - lo for lo, pr in pairs]
    assert row["n_scored"] == 3 and row["n_attempted"] == 3
    assert row["improvement_mean"] == pytest.approx(float(np.mean(paired)))
    assert row["improvement_std"] == pytest.approx(float(np.std(paired, ddof=1)))
    assert np.isfinite(row["improvement_mdes"])
    assert row["pct_improved"] == pytest.approx(200.0 / 3.0)


def test_run_stats_rejects_mixed_or_missing_kind() -> None:
    """F-20 fail-loud contract: one metric declares exactly one kind, and a
    pre-taxonomy parquet (no kind column) is an error, never a silent guess."""
    cfg = tiny_config()
    base = {"split": "test_id", "metric": "T30", "low_val": 3.0, "pred_val": 1.0,
            "high_ref": 2.0, "baseline_rel_ratio": 1.0, "improved": True}
    mixed = [
        {**base, "scene_id": "s_0", "kind": "match_reference"},
        {**base, "scene_id": "s_1", "kind": "maximize"},
    ]
    with pytest.raises(ValueError, match="kind"):
        _run_stats_on_rows(cfg, mixed)

    no_kind_col = [{**base, "scene_id": "s_0"}]
    with pytest.raises(KeyError, match="kind"):
        _run_stats_on_rows(cfg, no_kind_col)


def test_run_stats_reports_attempted_vs_scored() -> None:
    """F-21: n_attempted counts every per-scene row in the (split, metric) group;
    a NaN-legged (unscored) scene shows up as the scored/attempted gap instead of
    silently shrinking the population."""
    cfg = tiny_config()
    rows = [
        {"scene_id": f"s_{i}", "split": "test_geometry_shift", "metric": "C50",
         "kind": "match_reference", "low_val": 3.0, "pred_val": pred,
         "high_ref": 2.0, "baseline_rel_ratio": 1.0, "improved": imp}
        for i, (pred, imp) in enumerate([(1.0, True), (2.5, False), (np.nan, None)])
    ]
    ci = _run_stats_on_rows(cfg, rows)
    row = ci[ci["metric"] == "C50"].iloc[0]
    assert row["n_attempted"] == 3 and row["n_scored"] == 2
