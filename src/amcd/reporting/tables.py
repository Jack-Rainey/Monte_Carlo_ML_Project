"""report stage: format summary table + supplementary bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import Config


def run_report(config: Config, run_dir: Path) -> None:
    stats_dir = run_dir / "stats"
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = stats_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No stats found at {summary_path}. Run stats first.")

    with open(summary_path) as f:
        summary = json.load(f)

    df = pd.DataFrame(summary)

    col_w = {"metric": 22, "n": 5, "mean": 9, "ci": 22, "mdes": 10, "improved": 14}

    def _metric_row(row: dict) -> str:
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        # n_scored == 0 → improvement undefined for this metric (e.g. a diagnostic-only
        # metric with no baseline comparison); show N/A rather than "nan%".
        if row["n_scored"] > 0:
            improved_str = f"{row['pct_improved']:.1f}% ({row['n_improved']}/{row['n_scored']})"
        else:
            improved_str = "N/A"
        mdes_val = row["mdes_80pct"]
        mdes_str = f"{mdes_val:.4f}" if mdes_val == mdes_val else "N/A"
        return (
            f"{row['metric']:<{col_w['metric']}} "
            f"{row['n']:>{col_w['n']}} "
            f"{row['mean']:>{col_w['mean']}.4f} "
            f"{ci_str:<{col_w['ci']}} "
            f"{mdes_str:>{col_w['mdes']}} "
            f"{improved_str:<{col_w['improved']}}"
        )

    hdr = (
        f"{'Metric':<{col_w['metric']}} "
        f"{'N':>{col_w['n']}} "
        f"{'Mean':>{col_w['mean']}} "
        f"{'95% CI':<{col_w['ci']}} "
        f"{'MDES':>{col_w['mdes']}} "
        f"{'% Improved':<{col_w['improved']}}"
    )

    # One section per split — never pool test splits (invariant #9).
    lines = ["=" * 70, f"Run: {run_dir.name}", "=" * 70]
    present_splits = sorted(df["split"].unique()) if not df.empty else []
    for split_name in present_splits:
        split_rows = [r for r in summary if r["split"] == split_name]
        lines += [
            "",
            f"Metric results ({split_name}, bootstrap 95% CI):",
            "",
            hdr,
            "-" * len(hdr),
        ]
        for row in split_rows:
            lines.append(_metric_row(row))

    lines += ["", "=" * 70]
    summary_txt = "\n".join(lines)

    (report_dir / "summary.txt").write_text(summary_txt)
    df.to_csv(report_dir / "metrics_table.csv", index=False)

    # Supplementary bundle: copy config stamp + versions
    for fname in ["config.yaml", "versions.json"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, report_dir / fname)

    print(summary_txt)
    print(f"\n  Report written → {report_dir}")
