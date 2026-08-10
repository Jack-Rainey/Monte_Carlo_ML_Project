# amcd — Acoustic Monte Carlo Denoising

`amcd` studies whether a learned denoiser can recover reference-quality room
acoustics from **low-ray-budget** geometric-acoustic simulations — turning a cheap,
noisy Monte Carlo render into one that matches an expensive high-ray reference on
ISO-3382 metrics (T30, EDT, C50).

- **Methodology / hypotheses:** [`docs/research_I_paper.md`](docs/research_I_paper.md)
- **Build plan, invariants, and the D0→E4 experiment ledger:** [`docs/design_spec.md`](docs/design_spec.md)
- **Agent operating rules (plan/review/ledger discipline):** [`CLAUDE.md`](CLAUDE.md)
- **Open review findings:** [`docs/review_ledger.md`](docs/review_ledger.md)
- **Render-backend environment + upstream API reference:** [`docs/gsound_sir_setup.md`](docs/gsound_sir_setup.md)

Code and tests cite review-ledger row ids (`F-49`, `AC-17`, `RD-45`) as their
traceability device. A resolved row is **deleted** from the ledger — git history is
the audit trail — so a cited id that is not in the ledger is a row that was closed,
not a mistake. Recover it with:

```bash
git log -S 'RD-45' -- docs/review_ledger.md
```

## Install

The project runs in the conda env `amcd`:

```bash
conda activate amcd        # or use the interpreter at $CONDA/envs/amcd/bin/python
pip install -e .           # editable install of the amcd package
pytest tests/ -q           # 40+ invariant/config/shape tests should pass
```

> **Simulator note (cross-platform):** the `dry_run` simulator is pure Python and
> runs anywhere. The real `gsound_sir` render backend wraps GSound-SIR, whose
> C++ core is **x86-only**: on an x86_64 machine (Ubuntu or Windows desktop) it
> builds and runs natively; on Apple Silicon it runs under Rosetta 2 emulation in
> a dedicated `osx-64` conda env. Cross-platform operation is a project
> requirement — the emulation boundary lives in environment setup, never in
> package code, and every other stage (preprocess → train → eval → stats) is
> architecture-independent.

## Run

The pipeline is a linear DAG of cached stages. Run the whole thing, or one stage:

```bash
# THE canonical dry run: full pipeline on the synthetic backend, no real renderer.
# Three layers — the root config, the backend switch, the sizing overlay.
amcd all -c configs/base.yaml \
         -c configs/overlays/simulator_dry_run.yaml \
         -c configs/overlays/dry_run.yaml

# a single stage against an existing run directory
amcd <stage> -c configs/base.yaml \
             -c configs/overlays/simulator_dry_run.yaml \
             -c configs/overlays/dry_run.yaml \
             --run-dir experiments/<run_id>

# the Research I instantiation, scaled down to prove the RI methodology end-to-end
amcd all -c configs/base.yaml -c configs/research_i.yaml \
         -c configs/overlays/simulator_dry_run.yaml \
         -c configs/overlays/research_i_smoke.yaml
```

**Stage DAG** (`src/amcd/pipeline.py`):

```
gen-scenes → render → preprocess → diagnostics → train → infer → eval → stats → report
```

- Each stage writes a sentinel under `experiments/<run_id>/stages/`; a cached stage
  is skipped. Pass `--force` to rerun.
- Config layers merge left-to-right on top of `configs/base.yaml`; later `--config`
  files override earlier ones.
- Runs stamp `config.yaml` (resolved concrete config), `resolved.yaml`
  (concrete per-aspect seeds + role metadata + derived shapes), and `versions.json`
  into the run directory for provenance from the default save level; see
  `docs/verbosity.md` for the `--save-verbosity`/`--show-verbosity` output ladder
  (only an explicit `--save-verbosity 0` omits provenance, and no level alters
  what a run produces).

## Configuration is the source of truth

No behavioral value is hardcoded in Python — it is a CLI argument, a config value,
or the code raises. `configs/` has **three kinds of file**, and which kind a file
is determines whether you ever pass it to `-c`:

**1. Root configs** — a complete experiment definition. Always the first `-c`.

- `base.yaml` — the reference instantiation; the merge floor for everything else.
- `research_i.yaml` — the Research I reproduction, pinning Figure 5/6 verbatim.

**2. `overlays/`** — composable partial configs, always passed with `-c` *on top of*
a root config. Each does one job, so they stack without arguing:

- `simulator_dry_run.yaml` — **only** the backend switch. The single place in the
  repo that declares "use the scaffold", so no two invocations can disagree about
  which backend produced a run.
- `dry_run.yaml` — sizing/speed for `base.yaml`'s frac-mode split structure.
- `research_i_smoke.yaml` — sizing/speed for `research_i.yaml`'s count mode,
  preserving its per-split seeds, regimes and single-axis shift structure.
- `test_tiny.yaml` — the tiny sizing the test suite composes (`tests/conftest.py`).

**3. Plugin parameter blocks** — `models/`, `representations/`, `simulators/`.
Selected **by name** from a root config's `model` / `representation` / `simulator`
block, and **never** passed to `-c`. `simulator: {name: dry_run}` is what pulls in
`simulators/dry_run.yaml`; changing the name swaps the whole params block.

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
docs/               # research_I_paper.md, design_spec.md, review_ledger.md, verbosity.md
tests/              # invariant / config / shape tests
experiments/        # run outputs (git-ignored)
```

## Experiment ledger (`docs/design_spec.md` §11)

`D0a` headroom probe → `D0b` oracle upper bound → `E1` reproduce the waveform null →
`E2` energy-residual hypothesis (lead) → `E3` tune E2 → `E4` ray-budget sweep.
