"""report stage: format summary table + supplementary bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import Config
from ..runtime import Verbosity, emit


def run_report(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    stats_dir = run_dir / "stats"
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = stats_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No stats found at {summary_path}. Run stats first.")

    with open(summary_path) as f:
        summary = json.load(f)

    df = pd.DataFrame(summary)

    # "N sc/att" = scored/attempted per (split, metric) (F-21): the scored count
    # is the paired-improvement population; a gap vs attempted means legs were
    # dropped — per-leg reasons in the run's metrics/drops.csv.
    col_w = {"metric": 22, "n": 8, "pred": 10, "imp": 10, "ci": 22, "mdes": 10, "improved": 14}

    def _metric_row(row: dict) -> str:
        # Inferential columns (imp mean / CI / MDES) are the §9 paired improvement
        # for the metric's declared kind; "Pred mean" is the descriptive absolute
        # value. n_scored == 0 → NOTHING here is a result: render the row as
        # `unscored`, never a number a reader could mistake for an outcome — a
        # descriptive mean included (RR-14).
        n_str = f"{row['n_scored']}/{row['n_attempted']}"
        if row["n_scored"] == 0:
            return (
                f"{row['metric']:<{col_w['metric']}} "
                f"{n_str:>{col_w['n']}} "
                f"unscored — no scene has finite legs (reasons: metrics/drops.csv)"
            )
        imp_mean_str = f"{row['improvement_mean']:.4f}"
        ci_str = f"[{row['improvement_ci_lower']:.4f}, {row['improvement_ci_upper']:.4f}]"
        improved_str = f"{row['pct_improved']:.1f}% ({row['n_improved']}/{row['n_scored']})"
        mdes_val = row["improvement_mdes"]
        mdes_str = f"{mdes_val:.4f}" if mdes_val == mdes_val else "N/A"
        return (
            f"{row['metric']:<{col_w['metric']}} "
            f"{n_str:>{col_w['n']}} "
            f"{row['pred_mean']:>{col_w['pred']}.4f} "
            f"{imp_mean_str:>{col_w['imp']}} "
            f"{ci_str:<{col_w['ci']}} "
            f"{mdes_str:>{col_w['mdes']}} "
            f"{improved_str:<{col_w['improved']}}"
        )

    # CI level from config, not hardcoded in the label (RR-17, same rule as RR-11).
    ci_label = f"Imp {100 * (1 - config.bootstrap_alpha):g}% CI"
    hdr = (
        f"{'Metric':<{col_w['metric']}} "
        f"{'N sc/att':>{col_w['n']}} "
        f"{'Pred mean':>{col_w['pred']}} "
        f"{'Imp mean':>{col_w['imp']}} "
        f"{ci_label:<{col_w['ci']}} "
        f"{'MDES':>{col_w['mdes']}} "
        f"{'% Improved':<{col_w['improved']}}"
    )

    # One section per split — never pool test splits (invariant #9).
    #
    # Sections are enumerated from the CONFIG-DECLARED test splits in declaration
    # order, not from the splits present in the data (F-45). A declared split that
    # received no scored scene previously vanished from this file entirely, and an
    # absent split is indistinguishable from one that was never declared — the same
    # silent-exclusion class the drop log exists to prevent. Ordering is therefore
    # declaration order rather than the previous alphabetical sort.
    lines = ["=" * 70, f"Run: {run_dir.name}", "=" * 70]
    present_splits = set(df["split"].unique()) if not df.empty else set()
    declared = list(config.test_split_names)
    # Anything scored but not declared would be a routing bug; surface it rather
    # than dropping it off the end of the report.
    undeclared = sorted(present_splits - set(declared))
    for split_name in declared + undeclared:
        split_rows = [r for r in summary if r["split"] == split_name]
        scored_rows = [r for r in split_rows if r.get("n_attempted", 0) > 0]
        suffix = "" if split_name in declared else "  [NOT DECLARED IN CONFIG]"
        lines += [
            "",
            f"Metric results ({split_name}, paired improvement, bootstrap CI):{suffix}",
            "",
        ]
        if not scored_rows:
            # Mirrors _metric_row's n_scored == 0 rule at the split level: nothing
            # here is a result, so render no numbers at all (RR-14).
            lines.append(
                "0 scenes — unscored: this split is declared in config but no scene "
                "reached eval (see preprocessed/meta.json split_counts)."
            )
            continue
        lines += [hdr, "-" * len(hdr)]
        for row in scored_rows:
            lines.append(_metric_row(row))

    lines += [
        "",
        "N sc/att = scenes scored / attempted; per-leg drop reasons: metrics/drops.csv",
        "=" * 70,
    ]
    summary_txt = "\n".join(lines)

    (report_dir / "summary.txt").write_text(summary_txt)
    df.to_csv(report_dir / "metrics_table.csv", index=False)

    # Supplementary bundle: copy config stamp + versions. Provenance, same gate
    # as its source (`Config.stamp` runs at save ≥ 1), so a save=0 run — the
    # sanctioned provenance-free level (RD-09) — is self-consistent rather than
    # silently missing a copy.
    if verbosity.saves("provenance"):
        for fname in ["config.yaml", "versions.json"]:
            src = run_dir / fname
            if src.exists():
                shutil.copy(src, report_dir / fname)

    emit(verbosity, "metrics", summary_txt)
    emit(verbosity, "metrics", f"\n  Report written → {report_dir}")
