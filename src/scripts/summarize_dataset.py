from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated QC records.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--qc-root", default="results/tables/procedural_rir_dataset_v1")
    args = parser.parse_args()

    qc_root = Path(args.project_root) / args.qc_root
    summaries = []
    for qc_path in sorted(qc_root.rglob("*_qc.json")):
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
        summaries.append((qc_path.name, payload["passed"], payload["issues"]))

    passed = sum(1 for _, ok, _ in summaries if ok)
    print({
        "qc_root": str(qc_root),
        "record_count": len(summaries),
        "passed_count": passed,
        "failed_count": len(summaries) - passed,
    })


if __name__ == "__main__":
    main()
