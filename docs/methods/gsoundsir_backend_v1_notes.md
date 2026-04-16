# GSound-SIR backend v1 notes

This patch adds a real `GSoundSIRBackend` for the procedural dataset scaffold.

Current scope:
- real path-data extraction with `scene.getPathData(...)`
- real HOA synthesis with `spherical_harmonics_rt.generate_ambisonic_ir(...)`
- real retained-path export
- real preview WAV generation
- supported room families: `shoebox`, `corridor`

Current limitations:
- `l_room` and `alcove` are intentionally unsupported until a custom mesh path is wired.
- Materials are collapsed to a single global box absorption coefficient for `pygsound.createbox(...)`.
- The preview WAV is a duplicated omnidirectional HOA channel, not a true binaural decode.

Recommended first real test:
```bash
PYTHONPATH=src python3 src/scripts/run_dataset_pipeline.py   --project-root .   --config configs/scenes/procedural_rir_dataset_real_backend_smoke.json   --backend gsoundsir
```
