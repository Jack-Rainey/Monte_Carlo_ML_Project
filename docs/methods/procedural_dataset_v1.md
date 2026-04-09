# Procedural Dataset v1

This repository scaffold implements a methodology-driven dataset generator for paired low-ray and high-ray HOA room impulse responses.

## Core design choices

- Indoor room-impulse-response scope only.
- Broad training support over `shoebox`, `corridor`, and `l_room` families.
- Separate held-out subsets for in-distribution testing, material shift, placement shift, and geometry shift.
- Continuous geometry sampling within bounded families.
- Hybrid material sampling: latent regime selection followed by continuous per-surface coefficients.
- HOA order fixed at 3 for v1.
- Dry-run backend provided so the full dataset pipeline can be exercised before simulator integration.

## Important implementation note

The dry-run backend is for pipeline validation only. Replace `GSoundSIRBackend.render()` with the real simulator call before generating the final research dataset.
