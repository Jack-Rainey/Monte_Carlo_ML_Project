"""stats stage: bootstrap percentile CI + MDES over per-scene metrics.

The inferential quantities (CI tested against 0, MDES) are computed on the
per-scene PAIRED improvement — the baseline-vs-denoised difference design_spec
§9 requires — never on the absolute per-scene metric value, whose dispersion is
a different (up to ~2.4× divergent) σ (ledger F-18). The paired quantity is
keyed on each metric's declared `kind` via `paired_improvement` (F-20): the
spine never assumes match-reference. The absolute pred-value mean/CI are still
reported, as descriptive columns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats

from ..config import Config
from ..evaluation.metric_row import MetricTriple, paired_improvement
from ..runtime import Verbosity, emit


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap percentile CI (robust to non-normal metric distributions).

    All parameters are explicit — no defaults — so the caller (run_stats) always
    supplies config-declared values and a seeded RNG.
    """
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "std": float("nan")}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std": 0.0}

    boot_means = np.array([
        values[rng.integers(0, n, n)].mean()
        for _ in range(n_resamples)
    ])
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        # Sample std (ddof=1): this feeds mdes() as the population-σ estimate, and the
        # unbiased estimator is required at this design's small per-split n — ddof=0
        # biases σ down by √((n−1)/n) (≈ 0.82 at n=3), overstating detectability (F-09).
        "std": float(values.std(ddof=1)),
    }


def _two_sided_power(ncp: float, t_crit: float, df: int) -> float:
    """Two-sided one-sample t-test power at noncentrality `ncp` — computed
    HOLE-FREE so it is safe to bisect over.

    The test statistic under an effect is noncentral-t: T = (Z + ncp)/√(V/df) with
    Z~N(0,1) ⟂ V~χ²_df. scipy's `nct.cdf` returns ISOLATED NaN holes near the root
    at small df, and those holes are dense enough at df≤4 that a fixed nudge cannot
    reliably escape them — a NaN reaching the bisection reads as `nan < power == False`
    and truncates the bracket below the true root, understating MDES (ledger F-14/F-16).

    So instead of `nct.cdf`, integrate the exact rejection probability over the
    chi-square scale mixture: conditional on V=v the statistic is normal, so
    P(reject | v) = Φ(ncp − t_crit·s) + Φ(−t_crit·s − ncp) with s = √(v/df) (both
    tails; the far tail is negligible for ncp>0 but included for exactness). The
    integrand is smooth in v and the result is smooth and monotone increasing in
    ncp (from alpha at 0 to 1), with no NaN — no nudging or clamping needed.
    """
    def integrand(v: float) -> float:
        s = np.sqrt(v / df)
        return (
            scipy.stats.norm.cdf(ncp - t_crit * s)
            + scipy.stats.norm.cdf(-t_crit * s - ncp)
        ) * scipy.stats.chi2.pdf(v, df)

    # Integrate over the chi-square's actual support, not [0, ∞): at large df the
    # density is a narrow spike around v=df that quad's infinite-interval transform
    # samples right past (→ NaN). Bounding by extreme quantiles guarantees quad sees
    # the mass; the truncated tails carry <2e-12 of the probability, negligible here.
    lo_v = float(scipy.stats.chi2.ppf(1e-12, df))
    hi_v = float(scipy.stats.chi2.ppf(1 - 1e-12, df))
    value, _ = scipy.integrate.quad(integrand, lo_v, hi_v)
    return float(value)


def mdes(std: float, n: int, power: float, alpha: float) -> float:
    """Minimum detectable effect size for a two-sided one-sample t-test.

    The MDES is the mean effect δ whose test achieves `power` at significance
    `alpha` with n samples (ν = n−1 df). Solved EXACTLY via the noncentral-t: the
    t-statistic under an effect δ is noncentral-t with noncentrality λ = δ√n/σ, so
    we find the λ whose two-sided power equals `power` and return δ = λσ/√n. Power
    is evaluated hole-free by `_two_sided_power` (see there re ledger F-14/F-16).

    Why not the large-sample normal form `(z_{1-α/2}+z_power)·σ/√n`: at this design's
    per-split n (test splits are as small as 3) the normal critical value understates
    the small-sample one ~2.2× at n=3, making MDES ~2.2× too small — overstating
    detectability (ledger F-13). The noncentral-t is correct at every n and converges
    to the normal form as n grows.

    Returns NaN for n ≤ 2 (df ≤ 1): a one-sample t-test with a single degree of
    freedom is essentially unpowered (MDES ≈ 11σ), so MDES is reported N/A there
    rather than as a misleadingly precise huge number (ledger F-14)."""
    if n <= 2 or std == 0:
        return float("nan")
    df = n - 1
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df)

    # Grow the bracket until power ≥ target (monotone ⇒ this brackets the root).
    hi = 8.0
    while _two_sided_power(hi, t_crit, df) < power:
        hi *= 2.0
        if hi > 1e6:
            return float("nan")  # target power unreachable (degenerate n/σ or power≤alpha)
    # Bisection over the smooth monotone power (no NaN to defend against now).
    lo = 0.0
    for _ in range(200):
        if hi - lo < 1e-9:
            break
        mid = 0.5 * (lo + hi)
        if _two_sided_power(mid, t_crit, df) < power:
            lo = mid
        else:
            hi = mid
    ncp = 0.5 * (lo + hi)
    return float(ncp * std / np.sqrt(n))


def _substream_rng(bootstrap_seed: int, *key_parts: str) -> np.random.Generator:
    """Independent bootstrap substream per (split, metric, quantity) (ledger F-15).

    A single RNG stream shared across groups makes each group's CI bounds depend
    on how much entropy preceding groups consumed — adding or removing a metric
    or split perturbs unchanged groups' bounds by Monte-Carlo noise. Keying each
    stream on (bootstrap seed, stable key digest) makes its resampling invariant
    to group order and to which other groups exist; the per-quantity key part
    likewise decouples a group's improvement CI from its pred CI. sha256, not
    hash(): the builtin is salted per process (PYTHONHASHSEED) and would break
    run-to-run reproducibility.
    """
    digest = hashlib.sha256("/".join(key_parts).encode()).digest()
    stream_key = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([bootstrap_seed, stream_key]))


def _count_true(mask) -> int:
    """Count of True in a boolean column, and 0 when the column is absent.

    Absent rather than zero is the case that matters: the band-accounting columns
    exist only for the ISO-3382 metrics, so a spatial or perceptual metric has no
    band composition to report and must read 0, never NaN — a NaN here would
    propagate into a count a reader takes as a scene tally.
    """
    if mask is None:
        return 0
    return int(mask.fillna(False).astype(bool).sum())


def run_stats(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    metrics_path = run_dir / "metrics" / "metrics.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found at {metrics_path}. Run eval first.")

    stats_dir = run_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(metrics_path)
    if "kind" not in df.columns:
        raise KeyError(
            "metrics.parquet has no 'kind' column — produced by a pre-taxonomy "
            "eval stage; re-run eval (F-20: every metric declares its improvement kind)."
        )
    bootstrap_seed = config.seed("bootstrap")

    # Compute per-(split, metric) stats — never pool test splits (invariant #9).
    summary: list[dict] = []
    for (split_name, metric_name), group in df.groupby(["split", "metric"]):
        # One metric, one declared kind (F-20): the paired improvement below is
        # only meaningful under a single kind, so mixed declarations fail loud.
        kinds = group["kind"].unique()
        if len(kinds) != 1:
            raise ValueError(
                f"{split_name}/{metric_name}: inconsistent metric kinds {sorted(kinds)} "
                f"across scenes — a metric declares exactly one improvement kind (F-20)."
            )
        kind = str(kinds[0])
        # Descriptive: distribution of the metric's absolute per-scene pred value.
        # Reported for context only — its σ is NOT the §9 detectability σ (F-18).
        pred_values = group["pred_val"].dropna().values.astype(float)
        pred_ci = bootstrap_ci(
            pred_values,
            n_resamples=config.bootstrap_n_resamples,
            alpha=config.bootstrap_alpha,
            rng=_substream_rng(bootstrap_seed, split_name, metric_name, "pred"),
        )

        # Inferential (design_spec §9): the per-scene PAIRED improvement for
        # the group's declared kind — see `paired_improvement`. Recomputed from
        # the legs, never read from the `improved` column, so a hostile or
        # inconsistent column cannot skew CI/MDES (F-18).
        paired_all = np.array([
            paired_improvement(MetricTriple(
                float(r.low_val), float(r.pred_val), float(r.high_ref), kind
            ))
            for r in group.itertuples()
        ])
        paired = paired_all[np.isfinite(paired_all)]
        imp_ci = bootstrap_ci(
            paired,
            n_resamples=config.bootstrap_n_resamples,
            alpha=config.bootstrap_alpha,
            rng=_substream_rng(bootstrap_seed, split_name, metric_name, "improvement"),
        )
        mdes_val = mdes(
            imp_ci["std"], len(paired),
            power=config.bootstrap_power, alpha=config.bootstrap_alpha,
        )

        # Improvement pct: numerator and denominator over the SAME population —
        # scenes where `improved` is defined (not None/NaN). Counting n_improved over
        # rows dropped from the denominator produced pct > 100 (F-08). Under the
        # metric_row contract (`improved` is None ⟺ a consumed leg is NaN) this is
        # the same population as `paired`; the paired stats above still recompute
        # their own finite-legs mask so a hostile/inconsistent `improved` column can
        # skew only the pct, never the CI/MDES.
        improved = group["improved"]
        scored = improved.notna()
        n_scored = int(scored.sum())
        n_improved = int((improved[scored] == True).sum())  # noqa: E712

        row = {
            "split": split_name,
            "metric": metric_name,
            "kind": kind,
            # Scored-vs-attempted (F-21): n_attempted = per-scene rows in this
            # (split, metric) group (invariant #6 — rows are never collapsed
            # before stats); the report shows scored/attempted so a drop is
            # visible, never inferred. Per-leg reasons: eval's metrics/drops.csv.
            "n_attempted": len(group),
            # Descriptive absolute-value columns (context only).
            "n_pred": len(pred_values),
            "pred_mean": pred_ci["mean"],
            "pred_ci_lower": pred_ci["ci_lower"],
            "pred_ci_upper": pred_ci["ci_upper"],
            "pred_std": pred_ci["std"],
            # Inferential paired-improvement columns (§9). Power/alpha behind the
            # MDES come from config and are stamped in the run's config.yaml — the
            # column name must not hardcode them (RR-11); the improvement_ prefix
            # says which σ it derives from without reading this code (RR-12).
            "n_scored": n_scored,
            "improvement_mean": imp_ci["mean"],
            "improvement_ci_lower": imp_ci["ci_lower"],
            "improvement_ci_upper": imp_ci["ci_upper"],
            "improvement_std": imp_ci["std"],
            "improvement_mdes": mdes_val,
            "n_improved": n_improved,
            "pct_improved": float(n_improved) / n_scored * 100 if n_scored > 0 else float("nan"),
            # ── Composition of the scored population (F-62 / AC-25 / RD-78) ──
            # "N sc/att" cannot distinguish a fully-scored scene from a partially
            # scored one, and this split's CI pools per-scene improvements computed
            # over DIFFERENT band sets while `pred_mean` averages absolutes over
            # different bands. These three columns make that visible in the artifact
            # a reader consults, not only in drops.csv.
            "n_partial_band": _count_true(
                group.get("n_bands_kept") < group.get("n_bands_total")
                if "n_bands_kept" in group else None
            ),
            "n_pred_band_unresolved": _count_true(
                group.get("n_bands_pred_unresolved") > 0
                if "n_bands_pred_unresolved" in group else None
            ),
            "n_estimator_variance_limited": _count_true(
                group.get("estimator_variance_limited")
            ),
            # F-M2: this reached metrics.parquet and stopped there. A caveat the
            # reported table never renders is a code comment, not a disclosure —
            # and this is the one that marks a value the PHYSICAL legs reported
            # from a band their own octave filter cannot resolve, i.e. exactly the
            # scenes whose absolute is least trustworthy.
            "n_resolvability_limited": _count_true(
                group.get("n_bands_resolvability_limited") > 0
                if "n_bands_resolvability_limited" in group else None
            ),
        }
        summary.append(row)

    # Declared-but-unscored test splits (F-45). Everything above is keyed on the
    # splits PRESENT in metrics.parquet, so a declared test split that received no
    # scored scene would simply not appear — and an absent split is indistinguishable
    # from one that was never declared. F-30 fixed this at preprocess only; the same
    # discipline has to reach the artifacts a reader actually consults.
    scored_splits = {str(s) for s in df["split"].unique()}
    metric_names = sorted(str(m) for m in df["metric"].unique())
    for split_name in config.test_split_names:
        if split_name in scored_splits:
            continue
        for metric_name in metric_names:
            summary.append({
                "split": split_name,
                "metric": metric_name,
                # No scene reached eval, so no metric declared a kind here. Empty
                # rather than borrowed: inventing one would assert an improvement
                # direction nothing measured (F-20).
                "kind": "",
                "n_attempted": 0,
                "n_pred": 0,
                "pred_mean": float("nan"),
                "pred_ci_lower": float("nan"),
                "pred_ci_upper": float("nan"),
                "pred_std": float("nan"),
                "n_scored": 0,
                "improvement_mean": float("nan"),
                "improvement_ci_lower": float("nan"),
                "improvement_ci_upper": float("nan"),
                "improvement_std": float("nan"),
                "improvement_mdes": float("nan"),
                "n_improved": 0,
                "pct_improved": float("nan"),
                "n_partial_band": 0,
                "n_pred_band_unresolved": 0,
                "n_estimator_variance_limited": 0,
                "n_resolvability_limited": 0,
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(stats_dir / "ci_table.csv", index=False)

    # JSON-serialisable summary
    (stats_dir / "summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2)
    )

    # Counts are stated against the DECLARED test-split set, so "3 of 4" is visible
    # rather than being reported as a complete "3" (F-45).
    n_declared = len(config.test_split_names)
    n_scored_splits = len(scored_splits & set(config.test_split_names))
    n_metrics = summary_df["metric"].nunique() if not summary_df.empty else 0
    emit(
        verbosity, "metrics",
        f"  Stats for {n_metrics} metrics × {n_scored_splits} of {n_declared} "
        f"declared test splits → {stats_dir}",
    )
    for split_name in config.test_split_names:
        if split_name not in scored_splits:
            emit(
                verbosity, "warning",
                f"  WARNING: declared test split {split_name!r} has no scored scenes — "
                f"reported as unscored, not omitted.",
            )
