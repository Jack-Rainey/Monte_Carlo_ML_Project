# Monte Carlo ML Project Scaffold

This scaffold adds a methodology-aware procedural scene generation pipeline that matches the repo layout shown in the conversation.

## New code paths

- `src/scene_gen/`: scene generation, sampling, manifests, QC, dry-run backend
- `src/orchestration/dataset_workflow.py`: high-level entry points
- `src/scripts/`: command-line scripts
- `configs/scenes/procedural_rir_dataset_v1.json`: dataset specification
- `configs/simulation/ray_budget_pilot.json`: pilot sweep candidates

## Quick start

Generate scene specs only:

```bash
PYTHONPATH=src python src/scripts/generate_scene_specs.py --project-root . --overwrite
```

Run the full pipeline with deterministic dry-run renders:

```bash
PYTHONPATH=src python src/scripts/run_dataset_pipeline.py --project-root . --backend dry_run
```

Run the pilot sweep:

```bash
PYTHONPATH=src python src/scripts/run_pilot_sweep.py --project-root . --backend dry_run
```

Summarize QC:

```bash
PYTHONPATH=src python src/scripts/summarize_dataset.py --project-root .
```

## Wiring in GSound-SIR

The `GSoundSIRBackend` class in `src/scene_gen/backend.py` is the only required integration seam. You can adapt it to call your existing scripts under `src/gsound_tests/` or a new dedicated render script under `src/scripts/`.
