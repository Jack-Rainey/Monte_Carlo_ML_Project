# HOA CNN v1 patch

This patch adds a first-pass full-IR training path for the low-ray -> high-ray HOA denoising baseline.

## Files

- `src/training/hoa_dataset_v1.py`
  - discovers scene pairs from `data/metadata/.../<split>/<scene_id>/render_record.json`
  - validates full-array shapes
  - computes train-only per-channel normalization statistics
  - exposes a Keras `Sequence` for full-length `(time, channel)` batches
- `src/models/hoa_cnn_v1.py`
  - defines a small residual 1D CNN for direct full-target prediction
- `src/scripts/train_hoa_cnn_v1.py`
  - trains with Huber loss and MAE monitoring
  - saves checkpoints, history, stats, and split evaluation summary
- `src/scripts/export_hoa_cnn_v1_predictions.py`
  - exports denormalized predicted HOA tensors for selected splits
- `configs/training/hoa_cnn_v1_full_ir.json`
  - initial training/model hyperparameters

## Expected run style

From the project root:

```bash
PYTHONPATH=src python3 src/scripts/train_hoa_cnn_v1.py \
  --project-root . \
  --dataset-config configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --training-config configs/training/hoa_cnn_v1_full_ir.json \
  --max-train-scenes 20 \
  --max-valid-scenes 10
```

Then a full run:

```bash
PYTHONPATH=src python3 src/scripts/train_hoa_cnn_v1.py \
  --project-root . \
  --dataset-config configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --training-config configs/training/hoa_cnn_v1_full_ir.json
```

To export predictions after training:

```bash
PYTHONPATH=src python3 src/scripts/export_hoa_cnn_v1_predictions.py \
  --project-root . \
  --dataset-config configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --run-dir results/checkpoints/hoa_cnn_v1_full_ir/<run_name>
```

## Notes

- The training script evaluates normalized loss/MAE on `valid`, `test_id`, `test_material_shift`, `test_placement_shift`, and `test_geometry_shift` after fitting.
- The export script writes `pred_high_hoa.npy` under `data/processed/<experiment>/<run_name>/<split>/<scene_id>/`.
- This patch assumes the saved HOA arrays are channel-first on disk, matching the current pipeline, and transposes them to time-major for `Conv1D`.
- The code uses `LayerNormalization` rather than batch normalization because batch size 1 is likely for full-length 48 kHz IRs.