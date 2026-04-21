from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.hoa_cnn_with_paths import build_hoa_cnn_with_paths_v1
from training.hoa_dataset import compute_channel_stats, load_dataset_spec, load_stats, save_stats
from training.hoa_paths_dataset import HOAWithPathsSequence, compute_path_stats_from_records, discover_records_with_paths, validate_record_shapes_with_paths
from training.losses import TailAwareRIRLoss
from training.path_features import PATH_FEATURE_COLUMNS, PathFeatureStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOA CNN with retained path features.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-spec", default="configs/scenes/procedural_rir_dataset_real_backend_full_v1.json")
    parser.add_argument("--training-config", default="configs/training/hoa_cnn_with_paths_v1_full_ir.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-valid-scenes", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--reuse-stats", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime(use_mixed_precision: bool) -> None:
    if use_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def save_path_stats(stats_path: Path, path_stats: PathFeatureStats) -> None:
    with np.load(stats_path) as stats:
        payload = {key: stats[key] for key in stats.files}
    payload["path_mean"] = path_stats.mean
    payload["path_std"] = path_stats.std
    np.savez(stats_path, **payload)


def load_path_stats(stats_path: Path) -> PathFeatureStats:
    with np.load(stats_path) as stats:
        return PathFeatureStats(mean=stats["path_mean"], std=stats["path_std"])


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / args.dataset_spec).resolve()
    training_config_path = (project_root / args.training_config).resolve()

    training_config = load_json(training_config_path)
    training_section = training_config["training"]
    model_section = training_config["model"]
    path_section = training_config["paths"]
    loss_section = training_section.get("loss", {})

    seed = int(training_section.get("seed", 42))
    set_global_seed(seed)
    configure_runtime(bool(args.mixed_precision or training_section.get("mixed_precision", False)))

    dataset_spec = load_dataset_spec(project_root, dataset_spec_path)
    path_top_k = int(path_section.get("top_k", 128))
    path_num_features = len(PATH_FEATURE_COLUMNS)

    train_records = discover_records_with_paths(project_root, dataset_spec_path, "train", limit=args.max_train_scenes)
    valid_records = discover_records_with_paths(project_root, dataset_spec_path, "valid", limit=args.max_valid_scenes)
    validate_record_shapes_with_paths(train_records, dataset_spec, path_top_k=path_top_k)
    validate_record_shapes_with_paths(valid_records, dataset_spec, path_top_k=path_top_k)

    experiment_name = training_config.get("experiment_name", "hoa_cnn_with_paths_v1_full_ir")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "results" / "checkpoints" / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stats_path = run_dir / training_config.get("stats_filename", "channel_stats.npz")
    if args.reuse_stats and stats_path.exists():
        input_stats, target_stats = load_stats(stats_path)
        path_stats = load_path_stats(stats_path)
    else:
        input_stats = compute_channel_stats(train_records, array_selector="low")
        target_stats = compute_channel_stats(train_records, array_selector="high")
        path_stats = compute_path_stats_from_records(train_records, top_k=path_top_k)
        save_stats(stats_path, input_stats=input_stats, target_stats=target_stats)
        save_path_stats(stats_path, path_stats)

    batch_size = int(training_section.get("batch_size", 1))
    train_sequence = HOAWithPathsSequence(train_records, input_stats=input_stats, target_stats=target_stats, path_stats=path_stats, expected_num_channels=dataset_spec.expected_num_channels, expected_num_samples=dataset_spec.expected_num_samples, path_top_k=path_top_k, path_num_features=path_num_features, batch_size=batch_size, shuffle=True, seed=seed)
    valid_sequence = HOAWithPathsSequence(valid_records, input_stats=input_stats, target_stats=target_stats, path_stats=path_stats, expected_num_channels=dataset_spec.expected_num_channels, expected_num_samples=dataset_spec.expected_num_samples, path_top_k=path_top_k, path_num_features=path_num_features, batch_size=batch_size, shuffle=False, seed=seed)

    model = build_hoa_cnn_with_paths_v1(sequence_length=dataset_spec.expected_num_samples, num_channels=dataset_spec.expected_num_channels, path_top_k=path_top_k, path_num_features=path_num_features, base_width=int(model_section.get("base_width", 32)), kernel_size=int(model_section.get("kernel_size", 9)), dilation_schedule=tuple(model_section.get("dilation_schedule", [1,1,2,2,4,4])), width_schedule=tuple(model_section.get("width_schedule", [32,32,64,64,32,32])), dropout_rate=float(model_section.get("dropout_rate", 0.0)), residual_prediction=bool(model_section.get("residual_prediction", True)), path_branch_width=int(model_section.get("path_branch_width", 64)), path_embedding_width=int(model_section.get("path_embedding_width", 32)))

    loss_obj = TailAwareRIRLoss(target_mean=target_stats.mean, target_std=target_stats.std, sample_rate_hz=dataset_spec.sample_rate_hz, huber_delta=float(loss_section.get("huber_delta", training_section.get("huber_delta", 1.0))), late_start_ms=float(loss_section.get("late_start_ms", 80.0)), early_ms=float(loss_section.get("early_ms", 50.0)), mid_ms=float(loss_section.get("mid_ms", 200.0)), early_weight=float(loss_section.get("early_weight", 1.0)), mid_weight=float(loss_section.get("mid_weight", 2.0)), late_weight=float(loss_section.get("late_weight", 4.0)), wave_weight=float(loss_section.get("wave_weight", 1.0)), late_wave_weight=float(loss_section.get("late_wave_weight", 0.25)), mrstft_weight=float(loss_section.get("mrstft_weight", 0.02)), late_mrstft_weight=float(loss_section.get("late_mrstft_weight", 0.05)), edc_weight=float(loss_section.get("edc_weight", 0.01)), late_edc_weight=float(loss_section.get("late_edc_weight", 0.02)), edc_floor_db=float(loss_section.get("edc_floor_db", -60.0)))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(training_section.get("learning_rate", 1e-3))), loss=loss_obj, metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

    callbacks = [
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "best.weights.h5"), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=int(training_section.get("early_stopping_patience", 12)), restore_best_weights=True, min_delta=float(training_section.get("early_stopping_min_delta", 1e-5)), verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=float(training_section.get("reduce_lr_factor", 0.5)), patience=int(training_section.get("reduce_lr_patience", 6)), min_lr=float(training_section.get("reduce_lr_min_lr", 1e-6)), verbose=1),
        keras.callbacks.TerminateOnNaN(),
    ]

    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"path_top_k": path_top_k, "path_num_features": path_num_features, "path_feature_columns": PATH_FEATURE_COLUMNS}, handle, indent=2)

    history = model.fit(train_sequence, validation_data=valid_sequence, epochs=int(training_section.get("epochs", 100)), callbacks=callbacks, verbose=1)
    model.save(run_dir / "final_model.keras")
    if (run_dir / "best.weights.h5").exists():
        model.load_weights(run_dir / "best.weights.h5")
    model.save(run_dir / "best_model.keras")
    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, handle, indent=2)
    print({"run_dir": str(run_dir), "history_keys": list(history.history.keys())}, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
