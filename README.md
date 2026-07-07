# amcd — Acoustic Monte Carlo Denoising

`amcd` studies whether a learned denoiser can recover reference-quality room
acoustics from **low-ray-budget** geometric-acoustic simulations — turning a cheap,
noisy Monte Carlo render into one that matches an expensive high-ray reference on
ISO-3382 metrics (T30, EDT, C50).

- **Methodology / hypotheses:** [`docs/research_I_paper.md`](docs/research_I_paper.md)
- **Build plan, invariants, and the D0→E4 experiment ledger:** [`docs/design_spec.md`](docs/design_spec.md)
- **Agent operating rules (plan/review/ledger discipline):** [`CLAUDE.md`](CLAUDE.md)
- **Open review findings:** [`docs/review_ledger.md`](docs/review_ledger.md)

## Install

The project runs in the conda env `amcd`:

```bash
conda activate amcd        # or use the interpreter at $CONDA/envs/amcd/bin/python
pip install -e .           # editable install of the amcd package
pytest tests/ -q           # 40+ invariant/config/shape tests should pass
```

> **Simulator note (arch-sensitive):** the `dry_run` simulator is pure Python and
> runs anywhere. The real `gsound_sir` render backend is **x86-only** (GSound-SIR);
> on Apple silicon the `render` stage raises `NotImplementedError` with setup
> instructions until that backend is integrated. Everything else (preprocess →
> train → eval → stats) is architecture-independent.

## Run

The pipeline is a linear DAG of cached stages. Run the whole thing, or one stage:

```bash
# full pipeline on the synthetic dry_run backend (no real renderer needed)
amcd all --config configs/base.yaml --config configs/dry_run.yaml

# a single stage against an existing run directory
amcd <stage> --config configs/base.yaml --config configs/dry_run.yaml --run-dir experiments/<run_id>
```

**Stage DAG** (`src/amcd/pipeline.py`):

```
gen-scenes → render → preprocess → diagnostics → train → infer → eval → stats → report
```

- Each stage writes a sentinel under `experiments/<run_id>/stages/`; a cached stage
  is skipped. Pass `--force` to rerun.
- Config layers merge left-to-right on top of `configs/base.yaml`; later `--config`
  files override earlier ones.
- Every run stamps `config.yaml` (resolved concrete config), `resolved.yaml`
  (concrete per-aspect seeds + role metadata + derived shapes), and `versions.json`
  into its run directory for provenance.

## Configuration is the source of truth

No behavioral value is hardcoded in Python — it is a CLI argument, a config value,
or the code raises. Config lives in `configs/`:

- `base.yaml` — the Research-I reference instantiation (all defaults).
- `dry_run.yaml` — small synthetic overlay for plumbing/CI.
- `test_tiny.yaml` — the tiny config the test suite loads.
- `models/<name>.yaml` — per-model parameter blocks; `model.name` selects one.

**Parameter roles** (`docs/design_spec.md` §7). Any config leaf may be:

| role  | grammar | meaning |
|-------|---------|---------|
| fixed | `x: 0.5` | a scalar, used as-is |
| tuned | `x: {tune: {space: [lo, hi], scale: log}, value: 0.5}` | search selects on `valid` (E3); `value` is the current point |
| swept | `x: {sweep: [a, b, c]}` | a research axis → one sibling run per value (E4); a single run picks index 0 |

Roles are parsed and validated now; `Config.expand_sweeps()` produces the sibling
runs. The search **engine** (grid/factorial/evolutionary) is stubbed
(`src/amcd/search.py`) until E3 — `fixed` is fully supported today.

**Evaluation splits** are a config-declared set (`splits:` in `base.yaml`), not
hardcoded names. Conventions: `train` / `valid` / `test_id` are the in-distribution
train/validation/test sets (hash-bucketed from the id pool by fraction); every other
split is a controlled single-axis distribution shift (`test_material_shift`,
`test_placement_shift`, `test_geometry_shift`), evaluated per-split and never pooled.
Adding a split is a one-line config entry — no code change.

**Seeds:** one `seeds.master` derives an independent seed per stochastic aspect
(scene generation, split assignment, weight init, data shuffle, bootstrap) via
`SeedSequence.spawn`; any aspect can be pinned explicitly. The split-assignment seed
is stable-for-life — changing it reshuffles train/test membership.

## Repository map

```
src/amcd/
  config.py         # typed config: role grammar, seed derivation, run stamping
  pipeline.py       # stage DAG + caching
  cli.py            # `amcd` entry point
  registry.py       # simulator / representation / model / search registries
  search.py         # HPO strategy stubs (grid/factorial/evolutionary → E3)
  scenes/           # procedural scene generation (config-driven ranges)
  simulators/       # dry_run (synthetic) + gsound_sir (x86 real renderer)
  data/             # split assignment, preprocessing, normalization
  representations/  # third-octave spectrogram / waveform / EDR encoders
  models/           # vanilla_cnn baseline (owns its Params schema)
  training/         # trainer + infer
  evaluation/       # per-scene metrics (ISO-3382 room acoustics, signal, spatial)
  stats/            # bootstrap CIs, aggregation
  diagnostics/      # D0a headroom + D0b oracle probes
  reporting/        # result tables
configs/            # base / dry_run / test_tiny / models/*
docs/               # research_I_paper.md, design_spec.md, review_ledger.md
tests/              # invariant / config / shape tests
experiments/        # run outputs (git-ignored)
```

## Experiment ledger (`docs/design_spec.md` §11)

`D0a` headroom probe → `D0b` oracle upper bound → `E1` reproduce the waveform null →
`E2` energy-residual hypothesis (lead) → `E3` tune E2 → `E4` ray-budget sweep.
