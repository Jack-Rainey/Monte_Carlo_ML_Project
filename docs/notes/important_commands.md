PYTHONPATH=src python3 src/scripts/train_hoa_cnn.py \
  --project-root . \
  --dataset-spec configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --training-config configs/training/hoa_cnn_v1_full_ir.json \
  --max-train-scenes 20 \
  --max-valid-scenes 10


PYTHONPATH=src python3 src/scripts/export_hoa_predictions.py \
  --project-root . \
  --dataset-spec configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --run-dir results/checkpoints/hoa_cnn_v1_full_ir/<GENERATED_DIR> \
  --splits valid test_id \
  --max-scenes-per-split 3


PYTHONPATH=src python3 src/scripts/diagnose_prediction_energy.py \
  --project-root . \
  --prediction-root data/processed/hoa_cnn_v1_full_ir/<GENERATED_DIR>


PYTHONPATH=src python3 src/scripts/export_listening_previews.py \
  --project-root . \
  --dataset-spec configs/scenes/procedural_rir_dataset_real_backend_full_v1.json \
  --prediction-root data/processed/hoa_cnn_v1_full_ir/<GENERATED_DIR> \
  --write-multichannel-hoawav

deactivate 2>/dev/null || true
hash -r
source /media/jrainey/T7/venvs/tf-t7/bin/activate
source /media/jrainey/T7/venvs/tf-t7/bin/activate