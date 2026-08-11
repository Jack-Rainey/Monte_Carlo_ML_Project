# Lane R inbox — cycle4

Branch `lane/R-cycle4`. Written by lane R, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/R-cycle4", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

---

## Step 0 — `GsoundSirSimulator.render` no longer raises

`src/amcd/simulators/gsound_sir.py:124` raised `NotImplementedError`. It now
drives a subprocess worker and returns a real `IRResult`. Evidence is the
1-scene smoke render below.

The worker lives as a source STRING (`_WORKER_SRC`) inside `gsound_sir.py`
rather than as its own module, because `src/` ownership is exact-file and this
lane may not create `src/amcd/simulators/_gsound_worker.py` (`.claude/lane.json`
+ `scripts/lane_guard.py`). It imports nothing from `amcd` — the render env has
only numpy, pygsound and spherical_harmonics_rt — and communicates by files. See
RD-100 below: it should be promoted to its own module post-merge.

## Rows closed (proposed — the integrator deletes the rows)

Status changes are recorded here, not applied: lanes do not edit
`docs/review_ledger.md` (rule 3).

| row | what landed |
|---|---|
| **RD-24** | `PathData` in `simulators/base.py` pinned to upstream's exact keys/dtypes (`PATH_ARRAY_DTYPES`), with `to_parquet`/`from_parquet` carrying a descriptor **inside the parquet's own key/value metadata**: band edges AND centres, `num_bands`, simulator, `commit_sha`, `sample_rate`, `speed_of_sound_m_s`, retention policy, `synthesis_num_paths`, `ray_budget`, `leg`, `realization_index`. `REQUIRED_PATH_DESCRIPTOR_KEYS` + `validate_path_descriptor` are the RD-31-shaped contract for it. A file lacking the block is refused, not loaded headless. |
| **RD-08** | `IRResult.paths: PathData \| None`, added WITH its producer as the row requires. `None` on the scaffold leg; `simulators/render.py` writes `renders/<scene>/paths_{low,high}.parquet` keyed on the FIELD, so a backend without paths needs no downstream edit and there is no `isinstance`. |
| **RD-67** | gsound leg's `meta` now carries: `installed_commit_sha` **verified equal** to the pinned `commit_sha` (hard error otherwise, checked BEFORE any simulation); `diffuse_count` + `specular_count`; `ambisonic_convention: "acn_n3d"`; RD-21's truncation block; and a `PathData.speeds_of_sound` cross-check against the declared `speed_of_sound_m_s` (hard error on disagreement). |
| **RD-21** | `native_ir_samples`, `truncated`, `discarded_tail_db` (None — never 0.0 or -inf — when nothing was discarded) and `truncation_qc_flag` against the new config key `max_discarded_tail_db: -60.0`. |
| **AC-15** | The VALUE half is now stampable and stamped. Re-verified at source before stamping, against the pinned SHA: `auralizer/src/cpp/binding.cpp:18` "normalization constant K(l, m) for **N3D**", `:43` "**N3D/ACN** ordering". Row was DEFERRED at Step 3 pending exactly this. |

## New findings

Ledger was at RD-92; ids are provisional.

```
RD-93 | research-director | major | OPEN | configs/simulators/gsound_sir.yaml (render_python); src/amcd/simulators/render.py (_canonical_meta) | host interpreter path would be stamped into canonical per-scene provenance, making the same render differ between the Apple-Silicon and native-x86_64 hosts and leaking a user home path into 720 artifacts | FIX APPLIED, awaiting re-review: `_HOST_SCOPED_PARAMS` redacts it from the params echo; yaml marks it host-scoped; committed as null. Migrates to RunContext.host at RD-20.
RD-94 | research-director | major | OPEN | docs/lanes/cycle4-R.md render permission | a runtime "measure the low leg then decide about the high leg" judgement would have been a mid-lane widening of RD-89b, and wall clock at 200k rays is RD-17's measurement | FIX APPLIED, awaiting re-review: legs and budgets pre-declared in the scratchpad -c layer before the run (pasted below); the 200k leg was NOT run.
RD-95 | research-director | major | OPEN | src/amcd/simulators/gsound_sir.py (_WORKER_SRC); tests/test_simulator_seam.py | the cycle's headline deliverable would have had zero regression surface off the render host, so every future edit would need render permission to verify | FIX APPLIED, awaiting re-review: TestRenderWorkerContract compiles the worker, AST-checks its imports, and executes it under a venv with stub pygsound/spherical_harmonics_rt — including a full build_simulator().render() through the stubs.
RD-96 | research-director | major | OPEN | src/amcd/simulators/render.py (paths_{low,high}.parquet); ledger RD-23 | the filename convention encodes two legs and one realization, against RD-23's requirement ON THIS GATE that the artifact layout not foreclose a realization index | FIX APPLIED, awaiting re-review: ray_budget + leg + realization_index live in the parquet's own metadata, so a file is identifiable independently of its name. The FILENAME convention is still two-leg — see RD-101.
RD-97 | research-director | major | OPEN | src/amcd/simulators/gsound_sir.py (_RECEIPT_NAME); scripts/setup_gsound_sir.py:78,300 | package code cannot import scripts/ (no __init__.py, not installed, not on the lane's PYTHONPATH), so an imported receipt read would have failed at the first real render | FIX APPLIED, awaiting re-review: the receipt name + key are inlined with a comment naming both definitions. INTEGRATOR: consolidate into a shared module this lane could not create.
RD-98 | research-director | minor | OPEN | scripts/setup_gsound_sir.py:700; src/amcd/pipeline.py (F-65 exempt list); docs/verbosity.md; docs/gsound_sir_setup.md | routing anchors outside lane R's files | see "Not mine" below.
RD-99 | research-director | minor | OPEN | ledger RD-18, RD-14 | path-file size unmeasured; truncation_qc_flag consumed by nothing | FIX APPLIED, awaiting re-review: size measured (below); the meta docstring states Step 4 / RD-14 is the consumer.
```

### RD-102 — **the smoke render caught a real defect in the first worker** (fixed)

`src/amcd/simulators/gsound_sir.py` (`_WORKER_SRC`). The first worker passed
`path_retention` straight into `getPathData` and then synthesized the IR from
whatever came back. `configs/simulators/gsound_sir.yaml` has always said
retention applies **only** to the saved artifact — "IR synthesis always uses the
FULL path set, or the ray-budget axis under study would be confounded" — and
nothing enforced it.

MEASURED on the smoke scene: the IR was being built from **5,000 of 501,492**
paths, i.e. 43.1% of path energy, and `native_ir_samples` read **9,502**
(0.198 s) against a Sabine T60 of ~0.45 s for that room. After the fix the same
scene gives **46,333** samples (0.965 s), from all 501,492 paths.

Fix: one `getPathData(energy_percentage=100.0, max_rays=0)` call feeds synthesis,
and retention is applied to the ARTIFACT afterwards, reproducing upstream's own
algorithm (`ray_generator/src/pygsound/src/Scene.cpp:193-224` — sum intensities
per path, sort descending, cumulative-energy cut, then cap at `max_rays`).
Reproduced rather than requested because asking upstream to filter would need a
SECOND propagation run purely to obtain the unfiltered set the IR needs.

Regression guard:
`TestRenderWorkerContract::test_retention_trims_the_artifact_but_never_the_synthesis`
asserts upstream is always called with `(100.0, 0)`, that synthesis saw every
path, and that the saved file holds only the retained subset.

**This makes `docs/gsound_sir_setup.md` §4 stale** — it says "Path retention is
native upstream … with no custom trimming". That is no longer true and the doc is
not lane R's file. See RD-103.

### RD-100 — promote the worker to its own module (integrator)

`_WORKER_SRC` is a source string only because exact-file ownership barred a new
file. Post-merge it should become `src/amcd/simulators/_gsound_worker.py`, run by
path rather than materialized to a temp file. The `compile()` and stub-execution
tests move with it unchanged.

### RD-101 — the `paths_{low,high}.parquet` FILENAME still encodes two legs

RD-96's metadata fix means a file is self-identifying, so no already-written file
needs migrating. The naming convention itself is still two-leg/one-realization,
same limit `low.npy`/`high.npy` already carries. Belongs with RD-23 at E4, not
here — flagged so it is a decision rather than an accident.

### RD-103 — `docs/gsound_sir_setup.md` §4 retention claim is now wrong

"Path retention is native upstream, so `path_retention {…}` maps directly onto
`energy_percentage`/`max_rays` with no custom trimming." True of the API, false of
this backend as of RD-102: upstream's retention cannot be used without a second
propagation run, so it is reproduced in the worker. Not lane R's file.

### RD-104 — `render_python` documented as `--sim-python` in two places

`docs/gsound_sir_setup.md:6,16` and `scripts/setup_gsound_sir.py:700` describe a
`--sim-python` CLI flag. There is none, and lane R cannot add one (`cli.py` is not
this lane's file, and RD-20 already owns that dispatch signature). The value is a
simulator config key. Both sites need the spelling corrected. Neither is mine.

## Evidence

All commands prefixed with this worktree's `src`, per `LANE.md`.

**1. Full suite — 393 passed** (was 391 before this lane's tests; 59 in
`test_simulator_seam.py`):

```
$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-R/src \
  /Users/nortonrainey/miniconda3/envs/amcd/bin/pytest -q
393 passed in 52.37s
```

**2. Canonical dry run, end to end** — the scaffold is unaffected by the shared
`simulators/base.py` change:

```
$ … amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml \
             -c configs/overlays/dry_run.yaml
[done] gen-scenes (0.1s)   [done] render (0.5s)   [done] preprocess (1.3s)
[done] diagnostics (1.9s)  [done] train (1.0s)    [done] infer (0.4s)
[done] eval (0.4s)         [done] stats (6.8s)    [done] report (0.0s)
```

**3. `git merge v3-rebuild`** → `Already up to date.` Nothing came in, so the
evidence above and below is all measured on the tree being handed over.

### The 1-scene smoke render (RD-89b sub-grant)

Bounds were **pre-declared in a config layer written before the run** (RD-94),
kept in the scratchpad so no host path enters the repo:

```yaml
# smoke_host.yaml — exactly ONE scene, exactly ONE leg (low, 5000 diffuse rays);
# the 200,000-ray leg is NOT run this cycle; throwaway run_dir; via
# build_simulator; no reported number; does NOT lift RD-33a.
simulator:
  params:
    render_python: /Users/nortonrainey/miniconda3/envs/amcd-render-x86-verify/bin/python
scenes:
  n_id: 1
```

Everything else — sample_rate, ambisonics_order, ir_duration, ray budgets, all
gsound params — came from `configs/base.yaml` unchanged, so what ran is the real
configured backend, not a miniature of it.

```
rendering ONE leg of scene_0000
  dims=(3.159, 7.803, 2.549) absorption=0.213
  simulator=GsoundSirSimulator via build_simulator; diffuse budget=5000 (low leg only)

=== render() returned an IRResult in 0.9 s ===
ir: shape=(16, 204000) dtype=float32 finite=True peak=0.0347901

meta (validate_provenance passed):
{
  "simulator": "gsound_sir",
  "ray_budget": 5000,
  "speed_of_sound_m_s": 344.0,
  "ambisonic_convention": "acn_n3d",
  "rng_seeded": false,
  "commit_sha":           "608ea30f6dc4cda149c18947f9cae48bd379fa27",
  "installed_commit_sha": "608ea30f6dc4cda149c18947f9cae48bd379fa27",
  "diffuse_count": 5000,
  "specular_count": 2000,
  "num_paths": 5000,
  "synthesis_num_paths": 501492,
  "num_bands": 8,
  "kept_energy_percentage": 43.09951161907484,
  "native_ir_samples": 46333,
  "truncated": false,
  "discarded_tail_db": null,
  "max_discarded_tail_db": -60.0,
  "truncation_qc_flag": false
}

paths: num_paths=5000 num_bands=8 kept_energy_pct=43.0995
parquet round-trip OK; size on disk = 0.401 MB
```

Accounting, stated plainly:

- **ONE scene of RD-17's ≤4 was spent.** ≤3 remain for the Step-6 probe.
- It was **invoked three times on that same scene** — the first run exposed
  RD-102, the second confirmed the fix, the third re-measured on the shipped tree
  after `synthesis_num_paths` was added. Same scene, same leg, ~1 s each; no
  additional scene was rendered. Recording the count rather than the impression.
- **The 200,000-ray high leg has NEVER executed**, here or anywhere. Cycle 5's
  RD-17 planner should assume it is unmeasured.
- **0.9 s** for the low leg is a bare observation, **not a feasibility finding** —
  RD-17 owns that measurement, and one 5,000-ray scene says nothing about 200,000.
- **No reported number.** Nothing from this run reaches `ci_table.csv` or an
  E1/E4 claim. Throwaway run_dir under the session scratchpad.
- **RD-33a HAS NOT LIFTED.** Cycle 4 removes the blocker on RD-33a(ii); the probe
  is cycle 5's (RD-89c).
- `paths_low.parquet` = **0.401 MB** at `top_k: 5000` — the number RD-18's missing
  `max_path_file_mb` bound needs. At 720 scenes × 2 legs that is ~577 MB.

## Not mine — recorded, not done

- **RD-20** (`RunContext`, spans `pipeline.py` / `runtime.py` / `cli.py`) and
  **RR-27** (stale spec citations in `base.py` / `config.py`): integrator's serial
  queue. `runtime.py` is untouched by this lane.
- **`docs/verbosity.md`** needs a row for `renders/<scene>/paths_*.parquet`. It is
  written at EVERY save level, for RD-16's reason: under emulation a re-render
  costs hours, so an artifact this expensive is canonical, not observability. The
  table currently says the render stage's only canonical artifacts are the IR pair
  and `meta.json`.
- **F-65's guard** (`src/amcd/pipeline.py`): four new `Config` fields
  (`band_centres_hz`, `speed_of_sound_m_s`, `max_discarded_tail_db`,
  `render_python`) need to be in a stage fingerprint or the explicit exempt list.
  The first three are experiment-governing and belong in the render fingerprint;
  `render_python` is host-scoped and belongs in the exempt list.
- **`docs/gsound_sir_setup.md`** — RD-103 (retention claim) and RD-104
  (`--sim-python`).
- **`src/amcd/simulators/dry_run.py`** — lane M's; not touched.

## Self-check status

Reviewers were NOT run on this branch. Per rule 5 a lane-branch review is a
self-check and never a clean pass; the four-reviewer pass belongs on the
integrated tree. `research-director` DID review this lane's PLAN before
implementation (7 findings, all folded in — RD-93…RD-99 above), which is the
plan-stage review CLAUDE.md requires, not a code review.
