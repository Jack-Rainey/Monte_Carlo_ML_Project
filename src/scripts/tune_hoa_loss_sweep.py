from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential hyperparameter sweep for HOA tail-aware loss.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-spec", required=True)
    parser.add_argument("--base-training-config", required=True)
    parser.add_argument("--sweep-spec", required=True)
    parser.add_argument("--train-script", default="src/scripts/train_hoa_cnn.py")
    parser.add_argument("--export-script", default="src/scripts/export_hoa_predictions.py")
    parser.add_argument("--energy-script", default="src/scripts/diagnose_prediction_energy.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-valid-scenes", type=int, default=None)
    parser.add_argument("--export-splits", nargs="+", default=["valid", "test_id"])
    parser.add_argument("--max-export-scenes-per-split", type=int, default=3)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-energy-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def get_nested_value(payload: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor = payload
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def expand_trials(base_config: dict[str, Any], sweep_spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "trials" in sweep_spec:
        trials = []
        for idx, trial in enumerate(sweep_spec["trials"], start=1):
            config = json.loads(json.dumps(base_config))
            updates = trial.get("updates", {})
            for dotted_key, value in updates.items():
                set_nested_value(config, dotted_key, value)
            trials.append(
                {
                    "trial_index": idx,
                    "trial_name": trial.get("name", f"trial_{idx:03d}"),
                    "config": config,
                    "updates": updates,
                }
            )
        return trials

    if "grid" in sweep_spec:
        grid = sweep_spec["grid"]
        keys = list(grid.keys())
        value_lists = [grid[key] for key in keys]
        trials = []
        for idx, combo in enumerate(itertools.product(*value_lists), start=1):
            config = json.loads(json.dumps(base_config))
            updates = dict(zip(keys, combo))
            for dotted_key, value in updates.items():
                set_nested_value(config, dotted_key, value)
            name_parts = []
            for key, value in updates.items():
                short_key = key.split(".")[-1]
                safe_val = str(value).replace(".", "p").replace("-", "m")
                name_parts.append(f"{short_key}_{safe_val}")
            trials.append(
                {
                    "trial_index": idx,
                    "trial_name": sweep_spec.get("trial_name_prefix", "grid") + "__" + "__".join(name_parts),
                    "config": config,
                    "updates": updates,
                }
            )
        return trials

    raise ValueError("Sweep spec must define either 'trials' or 'grid'.")


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print({"running": cmd}, flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def collect_summary(
    *,
    run_dir: Path,
    training_config_path: Path,
    trial_name: str,
    updates: dict[str, Any],
    prediction_root: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial_name": trial_name,
        "training_config_path": str(training_config_path),
        "run_dir": str(run_dir),
        "updates": updates,
        "prediction_root": str(prediction_root) if prediction_root is not None else None,
    }

    evaluation_path = run_dir / "evaluation_summary.json"
    metadata_path = run_dir / "run_metadata.json"

    if metadata_path.exists():
        payload["run_metadata"] = load_json(metadata_path)
    if evaluation_path.exists():
        eval_summary = load_json(evaluation_path)
        payload["evaluation_summary"] = eval_summary

        valid = eval_summary.get("valid", {})
        payload["valid_loss"] = valid.get("loss")
        payload["valid_mae"] = valid.get("mae")
        payload["valid_model_raw_mae"] = valid.get("model_raw_mae")
        payload["valid_identity_raw_mae"] = valid.get("identity_raw_mae")
        payload["valid_relative_mae_improvement_vs_identity"] = valid.get("relative_mae_improvement_vs_identity")
        payload["valid_model_late_raw_mae"] = valid.get("model_late_raw_mae")
        payload["valid_identity_late_raw_mae"] = valid.get("identity_late_raw_mae")
        payload["valid_relative_late_mae_improvement_vs_identity"] = valid.get("relative_late_mae_improvement_vs_identity")
        payload["valid_late_edc_mae_db"] = valid.get("late_edc_mae_db")

    if prediction_root is not None:
        diag_path = prediction_root / "prediction_energy_diagnostics.json"
        if diag_path.exists():
            diag = load_json(diag_path)
            payload["prediction_energy_diagnostics"] = diag
            agg = diag.get("aggregate", {})
            payload["avg_pred_over_high_peak_abs"] = agg.get("avg_pred_over_high_peak_abs")
            payload["avg_pred_over_high_rms"] = agg.get("avg_pred_over_high_rms")
            payload["avg_pred_over_high_early_l2_sq"] = agg.get("avg_pred_over_high_early_l2_sq")
            payload["avg_pred_over_high_late_l2_sq"] = agg.get("avg_pred_over_high_late_l2_sq")

    score = 0.0
    penalties = []
    rms_ratio = payload.get("avg_pred_over_high_rms")
    early_ratio = payload.get("avg_pred_over_high_early_l2_sq")
    late_edc = payload.get("valid_late_edc_mae_db")
    valid_loss = payload.get("valid_loss")

    if isinstance(valid_loss, (int, float)):
        score += float(valid_loss)
    if isinstance(late_edc, (int, float)):
        score += 0.05 * float(late_edc)
    if isinstance(rms_ratio, (int, float)):
        penalties.append(abs(float(rms_ratio) - 1.0))
    if isinstance(early_ratio, (int, float)):
        penalties.append(abs(float(early_ratio) - 1.0))
    if penalties:
        score += 0.1 * sum(penalties)

    payload["composite_rank_score"] = score
    return payload


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / args.dataset_spec).resolve()
    base_training_config_path = (project_root / args.base_training_config).resolve()
    sweep_spec_path = (project_root / args.sweep_spec).resolve()
    train_script_path = (project_root / args.train_script).resolve()
    export_script_path = (project_root / args.export_script).resolve()
    energy_script_path = (project_root / args.energy_script).resolve()

    base_config = load_json(base_training_config_path)
    sweep_spec = load_json(sweep_spec_path)
    trials = expand_trials(base_config, sweep_spec)

    sweep_name = sweep_spec.get("sweep_name", datetime.now().strftime("sweep_%Y%m%d_%H%M%S"))
    sweep_root = project_root / "results" / "hyperparameter_sweeps" / sweep_name
    configs_root = sweep_root / "configs"
    summaries_root = sweep_root / "summaries"
    configs_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"

    manifest = {
        "sweep_name": sweep_name,
        "project_root": str(project_root),
        "dataset_spec_path": str(dataset_spec_path),
        "base_training_config_path": str(base_training_config_path),
        "sweep_spec_path": str(sweep_spec_path),
        "trial_count": len(trials),
        "trials": [],
    }

    for trial in trials:
        trial_name = trial["trial_name"]
        trial_index = trial["trial_index"]
        trial_config = trial["config"]
        updates = trial["updates"]

        run_name = f"{sweep_name}__{trial_index:03d}__{trial_name}"
        trial_config_path = configs_root / f"{run_name}.json"
        save_json(trial_config_path, trial_config)

        trial_entry = {
            "trial_index": trial_index,
            "trial_name": trial_name,
            "run_name": run_name,
            "config_path": str(trial_config_path),
            "updates": updates,
        }

        if args.dry_run:
            manifest["trials"].append(trial_entry)
            continue

        train_cmd = [
            args.python_executable,
            str(train_script_path),
            "--project-root",
            str(project_root),
            "--dataset-spec",
            str(dataset_spec_path.relative_to(project_root)),
            "--training-config",
            str(trial_config_path.relative_to(project_root)),
            "--run-name",
            run_name,
        ]
        if args.max_train_scenes is not None:
            train_cmd += ["--max-train-scenes", str(args.max_train_scenes)]
        if args.max_valid_scenes is not None:
            train_cmd += ["--max-valid-scenes", str(args.max_valid_scenes)]

        run_command(train_cmd, cwd=project_root, env=env)

        experiment_name = get_nested_value(trial_config, "experiment_name", "hoa_cnn_v1_full_ir")
        run_dir = project_root / "results" / "checkpoints" / experiment_name / run_name

        prediction_root = None
        if not args.skip_export:
            prediction_root = project_root / "data" / "processed" / experiment_name / run_name
            if prediction_root.exists():
                shutil.rmtree(prediction_root)

            export_cmd = [
                args.python_executable,
                str(export_script_path),
                "--project-root",
                str(project_root),
                "--dataset-spec",
                str(dataset_spec_path.relative_to(project_root)),
                "--run-dir",
                str(run_dir),
                "--splits",
                *args.export_splits,
                "--max-scenes-per-split",
                str(args.max_export_scenes_per_split),
            ]
            run_command(export_cmd, cwd=project_root, env=env)

            if not args.skip_energy_diagnostics:
                energy_cmd = [
                    args.python_executable,
                    str(energy_script_path),
                    "--project-root",
                    str(project_root),
                    "--prediction-root",
                    str(prediction_root.relative_to(project_root)),
                ]
                run_command(energy_cmd, cwd=project_root, env=env)

        summary = collect_summary(
            run_dir=run_dir,
            training_config_path=trial_config_path,
            trial_name=trial_name,
            updates=updates,
            prediction_root=prediction_root,
        )
        summary_path = summaries_root / f"{run_name}.summary.json"
        save_json(summary_path, summary)

        trial_entry["run_dir"] = str(run_dir)
        trial_entry["summary_path"] = str(summary_path)
        trial_entry["composite_rank_score"] = summary.get("composite_rank_score")
        trial_entry["valid_loss"] = summary.get("valid_loss")
        trial_entry["valid_late_edc_mae_db"] = summary.get("valid_late_edc_mae_db")
        trial_entry["avg_pred_over_high_rms"] = summary.get("avg_pred_over_high_rms")
        trial_entry["avg_pred_over_high_early_l2_sq"] = summary.get("avg_pred_over_high_early_l2_sq")

        manifest["trials"].append(trial_entry)
        save_json(sweep_root / "sweep_manifest.json", manifest)

    ranked_trials = sorted(
        manifest["trials"],
        key=lambda item: float(item.get("composite_rank_score", float("inf"))),
    )
    manifest["ranked_trials"] = ranked_trials
    save_json(sweep_root / "sweep_manifest.json", manifest)

    print({"sweep_root": str(sweep_root), "trial_count": len(manifest["trials"])}, flush=True)


if __name__ == "__main__":
    main()
