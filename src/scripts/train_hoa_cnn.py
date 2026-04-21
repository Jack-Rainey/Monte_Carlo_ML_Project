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

from models.hoa_cnn import build_hoa_cnn_v1
from training.hoa_dataset import (
    HOAFullIRSequence,
    compute_channel_stats,
    discover_records,
    load_dataset_spec,
    load_stats,
    save_stats,
    validate_record_shapes,
)
from training.losses import TailAwareRIRLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HOA CNN on full-length HOA IRs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--dataset-spec",
        default="configs/scenes/procedural_rir_dataset_real_backend_full_v1.json",
        help="Dataset-generation JSON with paths/simulation/splits.",
    )
    parser.add_argument(
        "--dataset-config",
        dest="dataset_spec_legacy",
        default=None,
        help="Deprecated alias for --dataset-spec.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training/hoa_cnn_v1_full_ir.json",
        help="Training hyperparameter JSON.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-valid-scenes", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--reuse-stats", action="store_true")
    parser.add_argument(
        "--skip-prediction-eval",
        action="store_true",
        help="Skip raw-space model-vs-identity evaluation if you only want Keras normalized metrics.",
    )
    return parser.parse_args()


def resolve_dataset_spec_arg(args: argparse.Namespace) -> str:
    if args.dataset_spec_legacy is not None:
        print(
            "[deprecated] --dataset-config now means dataset-generation spec. Use --dataset-spec instead.",
            flush=True,
        )
        return args.dataset_spec_legacy
    return args.dataset_spec


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

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _schroeder_edc_db_np(array_ch_t: np.ndarray, floor_db: float = -60.0, eps: float = 1e-12) -> np.ndarray:
    energy = np.square(array_ch_t, dtype=np.float64)
    energy = np.flip(np.cumsum(np.flip(energy, axis=1), axis=1), axis=1)
    energy = energy / np.maximum(energy[:, :1], eps)
    edc_db = 10.0 * np.log10(np.maximum(energy, eps))
    return np.maximum(edc_db, floor_db).astype(np.float32, copy=False)


def _prediction_eval_for_split(
    *,
    model: keras.Model,
    records,
    input_stats,
    target_stats,
    sample_rate_hz: int,
    late_start_ms: float,
    edc_floor_db: float,
) -> dict[str, float]:
    model_raw_mae_total = 0.0
    identity_raw_mae_total = 0.0
    model_late_raw_mae_total = 0.0
    identity_late_raw_mae_total = 0.0
    target_energy_total = 0.0
    total_values = 0
    late_total_values = 0

    late_start_idx = int(round(sample_rate_hz * late_start_ms * 1e-3))
    edc_mae_db_total = 0.0
    edc_mae_db_count = 0

    for record in records:
        low = np.load(record.low_path).astype(np.float32, copy=False)
        high = np.load(record.high_path).astype(np.float32, copy=False)

        low_norm = input_stats.normalize(low)
        pred_norm_t_c = model.predict(low_norm.T[None, ...], verbose=0)[0]
        pred = target_stats.denormalize(pred_norm_t_c.T).astype(np.float32, copy=False)

        abs_model_err = np.abs(pred - high, dtype=np.float32)
        abs_identity_err = np.abs(low - high, dtype=np.float32)

        model_raw_mae_total += float(abs_model_err.sum(dtype=np.float64))
        identity_raw_mae_total += float(abs_identity_err.sum(dtype=np.float64))
        target_energy_total += float(np.abs(high, dtype=np.float32).sum(dtype=np.float64))
        total_values += int(high.size)

        if late_start_idx < high.shape[1]:
            model_late_raw_mae_total += float(abs_model_err[:, late_start_idx:].sum(dtype=np.float64))
            identity_late_raw_mae_total += float(abs_identity_err[:, late_start_idx:].sum(dtype=np.float64))
            late_total_values += int(abs_model_err[:, late_start_idx:].size)

            pred_edc = _schroeder_edc_db_np(pred, floor_db=edc_floor_db)
            high_edc = _schroeder_edc_db_np(high, floor_db=edc_floor_db)
            valid = high_edc[:, late_start_idx:] > (edc_floor_db + 1e-6)
            if np.any(valid):
                edc_mae_db_total += float(
                    np.abs(pred_edc[:, late_start_idx:] - high_edc[:, late_start_idx:])[valid].sum(dtype=np.float64)
                )
                edc_mae_db_count += int(valid.sum())

    model_raw_mae = model_raw_mae_total / max(total_values, 1)
    identity_raw_mae = identity_raw_mae_total / max(total_values, 1)
    relative_gain = 1.0 - (model_raw_mae / max(identity_raw_mae, 1e-12))
    normalized_error_ratio = model_raw_mae_total / max(target_energy_total, 1e-12)

    result = {
        "model_raw_mae": model_raw_mae,
        "identity_raw_mae": identity_raw_mae,
        "relative_mae_improvement_vs_identity": relative_gain,
        "model_l1_over_target_l1": normalized_error_ratio,
    }

    if late_total_values > 0:
        model_late_raw_mae = model_late_raw_mae_total / late_total_values
        identity_late_raw_mae = identity_late_raw_mae_total / late_total_values
        result.update(
            {
                "model_late_raw_mae": model_late_raw_mae,
                "identity_late_raw_mae": identity_late_raw_mae,
                "relative_late_mae_improvement_vs_identity": 1.0 - (
                    model_late_raw_mae / max(identity_late_raw_mae, 1e-12)
                ),
            }
        )

    if edc_mae_db_count > 0:
        result["late_edc_mae_db"] = edc_mae_db_total / edc_mae_db_count

    return result


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / resolve_dataset_spec_arg(args)).resolve()
    training_config_path = (project_root / args.training_config).resolve()

    training_config = load_json(training_config_path)
    training_section = training_config["training"]
    model_section = training_config["model"]
    loss_section = training_section.get("loss", {})
    eval_splits = training_config.get(
        "evaluation_splits",
        ["valid", "test_id", "test_material_shift", "test_placement_shift", "test_geometry_shift"],
    )

    seed = int(training_section.get("seed", 42))
    set_global_seed(seed)
    configure_runtime(bool(args.mixed_precision or training_section.get("mixed_precision", False)))

    dataset_spec = load_dataset_spec(project_root, dataset_spec_path)
    train_records = discover_records(project_root, dataset_spec_path, "train", limit=args.max_train_scenes)
    valid_records = discover_records(project_root, dataset_spec_path, "valid", limit=args.max_valid_scenes)

    validate_record_shapes(train_records, dataset_spec)
    validate_record_shapes(valid_records, dataset_spec)

    experiment_name = training_config.get("experiment_name", "hoa_cnn_v1_full_ir")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "results" / "checkpoints" / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stats_path = run_dir / training_config.get("stats_filename", "channel_stats.npz")
    if args.reuse_stats and stats_path.exists():
        input_stats, target_stats = load_stats(stats_path)
    else:
        print("Computing train-only normalization statistics...", flush=True)
        input_stats = compute_channel_stats(train_records, array_selector="low")
        target_stats = compute_channel_stats(train_records, array_selector="high")
        save_stats(stats_path, input_stats=input_stats, target_stats=target_stats)

    batch_size = int(training_section.get("batch_size", 1))
    train_sequence = HOAFullIRSequence(
        train_records,
        input_stats=input_stats,
        target_stats=target_stats,
        expected_num_channels=dataset_spec.expected_num_channels,
        expected_num_samples=dataset_spec.expected_num_samples,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    valid_sequence = HOAFullIRSequence(
        valid_records,
        input_stats=input_stats,
        target_stats=target_stats,
        expected_num_channels=dataset_spec.expected_num_channels,
        expected_num_samples=dataset_spec.expected_num_samples,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = build_hoa_cnn_v1(
        sequence_length=dataset_spec.expected_num_samples,
        num_channels=dataset_spec.expected_num_channels,
        base_width=int(model_section.get("base_width", 32)),
        kernel_size=int(model_section.get("kernel_size", 9)),
        dilation_schedule=tuple(model_section.get("dilation_schedule", [1, 1, 2, 2, 4, 4])),
        width_schedule=tuple(model_section.get("width_schedule", [32, 32, 64, 64, 32, 32])),
        dropout_rate=float(model_section.get("dropout_rate", 0.0)),
        residual_prediction=bool(model_section.get("residual_prediction", True)),
    )

    optimizer = keras.optimizers.Adam(learning_rate=float(training_section.get("learning_rate", 1e-3)))
    loss_obj = TailAwareRIRLoss(
        target_mean=target_stats.mean,
        target_std=target_stats.std,
        sample_rate_hz=dataset_spec.sample_rate_hz,
        huber_delta=float(loss_section.get("huber_delta", training_section.get("huber_delta", 1.0))),
        late_start_ms=float(loss_section.get("late_start_ms", 80.0)),
        early_ms=float(loss_section.get("early_ms", 50.0)),
        mid_ms=float(loss_section.get("mid_ms", 200.0)),
        early_weight=float(loss_section.get("early_weight", 1.0)),
        mid_weight=float(loss_section.get("mid_weight", 3.0)),
        late_weight=float(loss_section.get("late_weight", 6.0)),
        wave_weight=float(loss_section.get("wave_weight", 1.0)),
        late_wave_weight=float(loss_section.get("late_wave_weight", 0.5)),
        mrstft_weight=float(loss_section.get("mrstft_weight", 0.0)),
        late_mrstft_weight=float(loss_section.get("late_mrstft_weight", 0.0)),
        edc_weight=float(loss_section.get("edc_weight", 0.05)),
        late_edc_weight=float(loss_section.get("late_edc_weight", 0.05)),
        stft_resolutions=tuple(tuple(v) for v in loss_section.get(
            "stft_resolutions",
            [[512, 128, 512], [1024, 256, 1024], [2048, 512, 2048]],
        )),
        edc_floor_db=float(loss_section.get("edc_floor_db", -60.0)),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(f"{line}\n"))

    monitor_name = "val_loss"
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.weights.h5"),
            monitor=monitor_name,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_name,
            mode="min",
            patience=int(training_section.get("early_stopping_patience", 12)),
            restore_best_weights=True,
            min_delta=float(training_section.get("early_stopping_min_delta", 1e-5)),
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_name,
            mode="min",
            factor=float(training_section.get("reduce_lr_factor", 0.5)),
            patience=int(training_section.get("reduce_lr_patience", 6)),
            min_lr=float(training_section.get("reduce_lr_min_lr", 1e-6)),
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    metadata = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "project_root": str(project_root),
        "dataset_spec_path": str(dataset_spec_path),
        "training_config_path": str(training_config_path),
        "dataset_name": dataset_spec.dataset_name,
        "expected_num_channels": dataset_spec.expected_num_channels,
        "expected_num_samples": dataset_spec.expected_num_samples,
        "train_scene_count": len(train_records),
        "valid_scene_count": len(valid_records),
        "visible_gpus": tf.config.list_physical_devices("GPU"),
        "residual_prediction": bool(model_section.get("residual_prediction", True)),
        "loss_name": "TailAwareRIRLoss",
        "loss_config": loss_obj.get_config(),
        "checkpoint_monitor": monitor_name,
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print(
        {
            "train_scenes": len(train_records),
            "valid_scenes": len(valid_records),
            "expected_shape_ch_t": [dataset_spec.expected_num_channels, dataset_spec.expected_num_samples],
            "run_dir": str(run_dir),
        },
        flush=True,
    )

    history = model.fit(
        train_sequence,
        validation_data=valid_sequence,
        epochs=int(training_section.get("epochs", 100)),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(run_dir / "final_model.keras")

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)

    if (run_dir / "best.weights.h5").exists():
        model.load_weights(run_dir / "best.weights.h5")

    model.save(run_dir / "best_model.keras")

    eval_results: dict[str, dict[str, float]] = {}
    for split_name in eval_splits:
        split_records = discover_records(project_root, dataset_spec_path, split_name)
        validate_record_shapes(split_records, dataset_spec)
        sequence = HOAFullIRSequence(
            split_records,
            input_stats=input_stats,
            target_stats=target_stats,
            expected_num_channels=dataset_spec.expected_num_channels,
            expected_num_samples=dataset_spec.expected_num_samples,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        metrics = model.evaluate(sequence, verbose=1, return_dict=True)
        eval_results[split_name] = {name: float(value) for name, value in metrics.items()}

        if not args.skip_prediction_eval:
            eval_results[split_name].update(
                _prediction_eval_for_split(
                    model=model,
                    records=split_records,
                    input_stats=input_stats,
                    target_stats=target_stats,
                    sample_rate_hz=dataset_spec.sample_rate_hz,
                    late_start_ms=float(loss_section.get("late_start_ms", 80.0)),
                    edc_floor_db=float(loss_section.get("edc_floor_db", -60.0)),
                )
            )

    with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_results, handle, indent=2)

    print({"evaluation": eval_results, "run_dir": str(run_dir)}, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()