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

---

## Falsifier self-check on `lane/R-cycle4` (commit 99cc033) — NOT a clean pass

Adversarial re-derivation of the CURRENT state of the six changed files, not the
diff. Full suite re-run in this worktree: `393 passed in 52.49s`. Rows below are
NEW findings raised BY the falsifier; ids provisional, continuing from RD-104.

```
RD-105 | falsifier | blocker | OPEN | configs/simulators/gsound_sir.yaml (render_python); src/amcd/pipeline.py:79-92 (_render_fingerprint); src/amcd/simulators/render.py:21-28 | `render_python` is inside `config.simulator.params`, which `_render_fingerprint` hashes WHOLE, so the render cache identity is host-dependent. PROBE: base.yaml -> 6999713b8247…, base.yaml + a host layer -> 51033e7d57c0…, diff `simulator.params.render_python: None -> '/Users/…/bin/python'`. Same experiment, two supported hosts, two cache identities: a 720-scene dataset rendered on the Mac fails loudly on the x86_64 Linux host demanding a byte-identical re-render. render.py:28 already declares this value not a property of the dataset while pipeline.py:86 uses it as part of the dataset's identity. | CROSS-LANE (pipeline.py is not lane R's) -> integrator serial queue. Fix: exclude host-scoped params from `_render_fingerprint`, sourced from the same declaration render.py uses. Test: `_render_fingerprint` invariant to `render_python`.
RD-106 | falsifier | major | OPEN | configs/simulators/gsound_sir.yaml (max_discarded_tail_db); src/amcd/pipeline.py:79-92 | Same mechanism: `max_discarded_tail_db` is a pure DISCLOSURE threshold that cannot change any IR, but it sits in the render fingerprint. PROBE: -60.0 -> -80.0 invalidates the render cache (`simulator.params.max_discarded_tail_db: -60.0 -> -80.0`), i.e. re-tightening a QC threshold costs a full emulated re-render. | CROSS-LANE -> integrator. Fix with RD-105: fingerprint the params that determine the IR, not the whole block.
RD-107 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:481-492; configs/simulators/gsound_sir.yaml (max_discarded_tail_db: -60.0); configs/base.yaml:88 | `truncation_qc_flag` is structurally unable to fire under the canonical config. PROBE (_fit_to_window on exponential decays, n_samples 204000): flag False at Sabine T60 = 4.20 (-60.71 dB) and 4.25 (-60.01 dB), True from 4.30 (-59.29 dB) up. The flag fires iff T60 > ir_duration, and `scenes.max_frac_below_iso_t30_decay_range: 0.0` hard-gates exactly those scenes out at gen-scenes. So every base.yaml scene prints `truncation_qc_flag: false` by construction while nothing consumes it (RD-99) — a quantity that emits output and contributes nothing. It IS live under configs/research_i.yaml (ir_duration 3.0, frac 0.01). | Fix: derive the threshold from a criterion independent of ir_duration, or state it as a backstop for over-limit-tolerant configs. Test: exercise the firing case through a CONFIG, not a hand-built array.
RD-108 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:461-492, :494-517; src/amcd/simulators/render.py:146-160 | A zero-energy render is indistinguishable from a healthy one. PROBE: `_fit_to_window(np.zeros((16,300000)))` -> `truncated=True, discarded_tail_db=None, truncation_qc_flag=False, native_ir_samples=300000` — the same all-clear a good leg gets, because `total_energy == 0` is folded into "nothing was discarded". Nothing in the render path asserts the leg carries energy: `_check_declared_speed` only requires paths to EXIST, `validate_provenance` only checks key presence. A silent leg reaches low.npy/high.npy and first surfaces as a NaN in a metric. | Fix: stamp the leg's total IR energy in meta and raise (or record a logged (scene, leg, reason) drop that shows in output counts) when it is zero.
RD-109 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:121, :130 (_retain in _WORKER_SRC) | `kept_energy_percentage` fabricates 0.0 for an undefined ratio, and `_retain` diverges from the upstream algorithm it claims to reproduce. PROBE vs a transcription of Scene.cpp:191-224 — all-zero-energy paths, top_percent 50: lane keeps 4/4 and reports 0.0 %; upstream keeps 1 and reports NaN. The `total > 0.0` guard at :121 changes the selection, and 0.0 % reads as "we retained almost nothing" when the code retained everything. Violates "never render an unscored quantity as a number". | Fix: `kept_energy_percentage = None` + a logged reason when total_energy == 0; match upstream's keep=1 or document the divergence in the docstring at :104-113.
RD-110 | falsifier | major | OPEN | src/amcd/simulators/render.py:21-28, :78-85; tests/test_simulator_seam.py:770-780 | (a) `_HOST_SCOPED_PARAMS = ("render_python",)` puts backend-specific knowledge inside the stage whose own docstring (render.py:78-81) says "No branch here knows what a gsound is". A second backend's host-scoped param leaks into canonical provenance; deleting gsound_sir leaves a dead constant. (b) Its regression test is vacuous: PROBE shows `tiny_config(...)` resolves to `simulator.name = 'dry_run'` with params `['min_source_receiver_distance_m', 'speed_of_sound_m_s']`, so `assert "render_python" not in recorded["simulator"]["params"]` is true whether or not the redaction exists. Removing `_HOST_SCOPED_PARAMS` entirely leaves the suite green. | Fix: the BACKEND declares its host-scoped keys (classmethod / pydantic field marker) and the stage asks; run the redaction test through a gsound config with a non-null render_python.
RD-111 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:398-412 (_retention_args), :624-632 (PathRetention); tests/test_simulator_seam.py:744-746 | The lane's own file still asserts the pre-RD-102 contract. `_retention_args` says "Retention is native upstream, so there is no custom trimming here" and "`getPathData` is a SEPARATE CALL from the synthesis input" — both false as of this lane's own fix: there is exactly ONE getPathData call (:180-183), always `(100.0, 0)`, and the trimming is a custom reimplementation at :103-131. `PathRetention` repeats "maps directly onto upstream getPathData". The test docstring repeats it a third time. RD-103 flagged this staleness in docs/gsound_sir_setup.md and missed the same claim in the file the lane owns — and a stale contract comment is exactly what produced RD-102. | Lane R's own file: fixable before merge.
RD-112 | falsifier | major | OPEN | src/amcd/simulators/base.py:139-155, :245-259; src/amcd/simulators/gsound_sir.py:566-593 | RD-24's self-describing claim is unenforced where it matters: the descriptor's band NAMING is never checked against the actual band count. PROBE: `PathData(num_bands=9, intensities=(4,9), descriptor={band_centres_hz: 8 values, band_edges_hz: 7 values})` constructs, passes `validate_path_descriptor`, writes and reads back — a file naming 8 bands for 9 columns, which is the exact "uninterpretable path file" RD-24 exists to prevent. `__post_init__` only compares two numbers from the SAME producer, self-consistent by construction. `render()` never asserts `result["num_bands"] == len(band_centres_hz) == len(frequency_points)+1`. | Fix: assert the identity in `render()` and add the length check to `validate_path_descriptor`.
RD-113 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:475-479 (the zero pad); src/amcd/evaluation/room_acoustic.py:59-87 (_lundeby_truncate), :278-329 (_shared_truncation_per_band) | The smoke render is the first evidence that a real leg's tail is EXACT zeros (native 46333 of 204000 samples: 77.3 % of every canonical IR is fabricated silence), and that changes the Lundeby regime the whole ISO-3382 spine rests on. PROBE at 48 kHz: with a floor spanning the record (the dry-run scaffold's regime) `_lundeby_truncate` returns 25276 — a real noise-floor crossing. With a zero-padded real-render leg it returns native+240 for BOTH a 46333- and a 60000-sample native record, i.e. exactly the native support end plus the 10 ms smoothing half-window: `noise_power` clamps to the 1e-30 floor, the threshold degenerates to 1e-29 and NO noise rejection happens. AC-17's rationale ("the index is noise-floor dependent, so take the min over legs") no longer describes the mechanism: the index now tracks native IR LENGTH, which is itself ray-budget dependent, and both legs are integrated past where the low leg's decay has reached its own Monte-Carlo floor. | CROSS-LANE (room_acoustic.py is the metric lane's) -> integrator. CONFIRMING PROBE: render one scene at 5,000 and at 200,000 diffuse rays, record `native_ir_samples` per leg and the per-band `_shared_truncation_per_band` index; if the index tracks the budget, AC-17 must be restated before any E1/E4 number.
RD-114 | falsifier | major | OPEN | src/amcd/simulators/render.py:159-183; src/amcd/simulators/gsound_sir.py:603-605 | `rng_seeded: false` declares that reproducibility rests on the cached artifacts, and the cached artifacts have no integrity check: meta.json carries no digest of low.npy/high.npy/paths_*.parquet. Two physically DIFFERENT datasets therefore carry byte-identical provenance, and a truncated or partially-written IR is undetectable. The lane rendered the same scene three times and did not report whether the outputs were bit-identical — the free empirical check on the declaration it stamps into every artifact. | Fix: sha256 each written artifact into meta.json. Test: render one scene twice, hash both, diff meta.json.
RD-115 | falsifier | minor | OPEN | tests/test_simulator_seam.py:1064-1073; src/amcd/simulators/gsound_sir.py:433-443 | `test_the_parent_surfaces_a_worker_failure_with_its_stderr` passes for the wrong reason. PROBE: `render()` with a nonexistent `render_python` raises `FileNotFoundError: [Errno 2] No such file or directory: '/definitely/no_such_python'` from `subprocess.run`, never reaching the RuntimeError; the `match="no_such_python|worker failed"` alternation absorbs it. The RuntimeError is the SOLE diagnostic for a failed emulated render and its `proc.stderr`/`proc.stdout` interpolation has zero coverage. | Fix: point `render_python` at a real interpreter that exits non-zero and assert the worker's stderr text appears in the message.
RD-116 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:635-650 (PathRetention), :125-126, :412 | `PathRetention.value` accepts any float with no range or integrality check, so out-of-range experiment values are silently reinterpreted instead of raising. PROBE: `top_k: 0` and `top_k: -3` both mean "keep everything" (`0 < max_rays` fails); `top_k: 5000.7` silently truncates to 5000; `top_percent: 150` silently means "all"; `top_percent: 0` keeps 1 path. | Fix: mode-conditional validation in `model_post_init` (top_k: integral and >= 1; top_percent: 0 < v <= 100).
RD-117 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:50-58, tests/test_simulator_seam.py:711-718 | `test_the_ambisonic_convention_is_n3d_not_sn3d` asserts `_AMBISONIC_CONVENTION == "acn_n3d"` — the constant against itself. `ambisonic_convention` is the one upstream-compiled fact here stamped from a source COMMENT with no falsification, while its sibling `speed_of_sound_m_s` was deliberately config-declared AND cross-checked against the paths (RD-19). A per-degree sqrt(2l+1) error is invisible today (channel 0 only) and load-bearing at RD-25. | Fix: falsify it the way the speed is — synthesize one path at a known direction through the worker and compare the l=1 channel magnitudes against the N3D vs SN3D prediction (ratio sqrt(3)); skip when the render env is absent.
RD-118 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:562, :494-517 | `_check_declared_speed` validates only the RETAINED paths: `render()` passes `arrays["speeds_of_sound"]`, i.e. the top-5,000-by-energy subset (1 % of the 501,492 the IR was synthesized from). The falsifiability claim at base.py:121-124 and gsound_sir.py:495-500 is stated over "the paths", not over 1 % of them. The worker holds the full array at :180-183 and can check it for free before retention. | Fix: move the cross-check into the worker, over the unfiltered set.
RD-119 | falsifier | minor | OPEN | src/amcd/simulators/base.py:157-216 | The parquet round trip is not dtype-faithful: `to_parquet` writes the IN-MEMORY dtype while `from_parquet` casts to `PATH_ARRAY_DTYPES`. PROBE: float64 `distances`/`intensities` in -> float32 out, silently, no logged (unit, reason). The declared dtype contract is enforced only on read, so a second raytracer's float64 arrays lose precision on write with no error. | Fix: cast (or raise) in `to_parquet`/`__post_init__` so the declared dtype is the written dtype.
RD-120 | falsifier | minor | OPEN | tests/test_simulator_seam.py:940 | `venv / "bin" / "python"` hardcodes the POSIX venv layout, so the new worker-contract suite cannot run on Windows, which docs/gsound_sir_setup.md and CLAUDE.md declare a supported host. | Fix: `venv/"Scripts"/"python.exe"` fallback, or `sysconfig.get_path("scripts", vars=...)`.
RD-121 | falsifier | minor | OPEN | docs/ledger_inbox/R.md (evidence section) | Two claims the smoke render does not support as stated. (a) RD-21's truncation accounting NEVER RAN on real data: native 46333 < 204000 took the PADDING branch, so `truncated=false` is the only branch the one real render exercised; the row's evidence is unit tests on hand-built arrays. (b) The ~577 MB projection is n = 1, low-leg only, no variance — defensible ONLY because `top_k` fixes the row count, and that reasoning is unstated; under `top_percent`/`all` the number is unrelated to the bound RD-18 needs. | Fix: state both limits where the numbers are quoted.
RD-122 | falsifier | minor | OPEN | src/amcd/simulators/render.py:154-157 | The shape assertion covers only the LOW leg, and is a bare `assert` (stripped under `python -O`). Both legs now come from `_fit_to_window`, so the high leg's shape is checked by nothing. | Fix: check both legs, and raise rather than assert.
```

### Not raised (checked, and could not break)

- **Platform coupling** — no `platform`/arch/OS branch, no host path, no MPS
  assumption anywhere in `src/amcd/` (grep over the package). The `render_python`
  config value is the right seam; only its FINGERPRINT treatment is broken (RD-105).
- **Scaffold coupling** — no `isinstance(..., DryRunSimulator)` and no dry-run-keyed
  branch outside `simulators/dry_run.py`; `render.py:169` keys on `result.paths is
  None`, so a backend without paths needs no downstream edit. The one backend-shaped
  branch that DOES exist is `_HOST_SCOPED_PARAMS` (RD-110a).
- **`_retain` vs upstream** — transcribed Scene.cpp:191-224 and probed the cut
  index across all-zero, tie-at-the-cut, exact-boundary, `top_percent 0`,
  `top_percent 100` and `top_k 0`. Selection matches upstream in every case except
  the zero-total one (RD-109). `searchsorted(..., side="left")` correctly reproduces
  upstream's `accumulated >= target` inclusive boundary; `kind="stable"` is a
  deviation from `std::sort` only under exact ties and is the safer choice.
- **Truncation dB arithmetic and flag direction** — known-answer probe: equal energy
  in and out of the window gives -3.0103 dB, and `discarded_db > max_discarded_tail_db`
  is the correct direction (more discarded tail -> less negative dB -> flagged).
- **`_fit_to_window` padding correctness** — pads with zeros, preserves the native
  head exactly, returns C-contiguous float32 at `(n_channels, n_samples)`.
- **Provenance seam** — `REQUIRED_PROVENANCE_KEYS` / `validate_provenance` and the
  `PathData` parquet round trip (arrays, band axis as a shape, descriptor identity)
  all hold; zero-path `PathData` round-trips without raising.
- No leakage, normalization-stat or split surface is touched by this lane.

---

# acoustics-reviewer — domain-physics pass on lane/R-cycle4 @ 99cc033

SELF-CHECK on an unintegrated branch, NOT a clean pass (`docs/parallel_protocol.md`
rule 5). Every claim below was recomputed against the pinned upstream source
(`/Volumes/T7/Monte_Carlo_Research/v3/external/GSound-SIR`, SHA
608ea30f6dc4cda149c18947f9cae48bd379fa27) or measured on a synthetic signal with a
known answer; none was taken from `docs/gsound_sir_setup.md` §4.

## What is CORRECT (verified, not assumed)

- **Band definition.** `frequency_points` are exactly the geometric means of
  pygsound's compiled centres (`gsFrequencyBands.cpp:83-88`, `Context.cpp:8`);
  measured rel. err <= 3.7e-8. AC-12's 88.7412-not-88.4 is right. The two ISO eval
  bands coincide EXACTLY with simulated bands 3 and 4. `Params.model_post_init`'s
  geometric-mean constraint is the correct falsifiable relation.
- **Schroeder direction.** `np.cumsum(e[::-1])[::-1]` — backward, not forward
  (`room_acoustic.py:224`, `:696-700`). ISO windows correct: T30 [-5,-35], EDT
  [0,-10], C50 split at `ceil(0.050*fs)` with the late window ending at the
  truncation index.
- **Zero-padding is acoustically neutral.** Measured: metrics and Lundeby index
  identical to 4+ decimals with and without padding to 204000. Adding exact zeros
  contributes nothing to a backward integral. `_fit_to_window`'s pad branch is fine.
- **Channel count / decode path.** (order+1)^2 = 16, derived once; scalar metrics
  read channel 0; reported metrics come from decoded waveforms via the ISO path
  (`evaluator.py:102-114`), and `decode` imposes the predicted envelope on the
  LOW-RAY CARRIER (`representations/spectrogram.py:329-395`), so all three legs
  share one carrier.
- **Retention.** `_retain` reproduces `Scene.cpp:193-224` exactly; summing
  intensities across bands IS the right broadband energy ordering; synthesis
  genuinely uses the full path set (`gsound_sir.py:182`). RD-102's fix is correct.
- **Speed of sound.** 344 m/s is right — gsound computes
  `getAirSpeedOfSound(20 C, 101.325 kPa, 50 % RH)` (`gsSoundMedium.cpp:79`). The
  cross-check against the paths' own speeds is the right way to keep a compiled-in
  constant honest. `relative_speeds` is correctly NOT used for delay.

## Findings

```
AC-40 | source: acoustics-reviewer | blocker | status: OPEN | src/amcd/simulators/gsound_sir.py:161-163,531 ; configs/base.yaml:138 ; src/amcd/scenes/generator.py:274,314 | Effective wall absorption is 1-sqrt(1-alpha), not alpha. Upstream forms path energy as `energy = getDistanceAttenuation(d) * PROD_bounces(reflectivity)` (gsSoundPropagator.cpp:1410 specular, :1578 diffuse). getDistanceAttenuation (:4518) = 1/(4pi(1+d^2)) is ENERGY-domain; reflectivity (SoundMesh.cpp:223) = sqrt(1-alpha) is AMPLITUDE-domain. Mixing them means ENERGY is multiplied by sqrt(1-alpha) per bounce, so alpha_eff = 1-sqrt(1-alpha) and realized T60 is 1.14x-1.98x the configured value (alpha 0.05 -> x1.975; 0.80 -> x1.447). amcd derives Sabine T60, critical distance, DRR, the ir_duration choice and the max_frac_below_iso_t30_decay_range gate from the NOMINAL alpha, so the dataset's declared acoustics describe a room that was not rendered. Smoke-scene cross-check supports it: 3.159x7.803x2.549 m at alpha=0.213, max path delay 0.9226 s = 2.05xT60_nominal (-123 dB) but 1.09xT60_eff (-65 dB); an IR_THRESHOLD 'threshold of hearing' trim lands at -65 dB, not -123 dB. | REMEDY: confirm with the one-scene test below, then either pre-compensate at the createbox call site (pass 1-(1-alpha_target)^2) or re-derive every closed form in scenes/generator.py from alpha_eff. Declare which in config; do not leave the two definitions coexisting.
AC-41 | source: acoustics-reviewer | major | status: OPEN | configs/simulators/gsound_sir.yaml:25-26 | `diffuse_depth: 100` / `specular_depth: 50` are declared as ray-budget knobs but are physically a TIME bound on the decay. With mean free path 4V/S, the decay reached at depth D is -60*4*D*alpha/55.25 = -4.344*D*alpha dB, INDEPENDENT of room size. ISO 3382 T30 needs the EDR down to -35 dB, so depth 100 covers T30 only for alpha >= 0.0806 (>= 0.1547 if AC-40 holds). material_regimes.mixed is U[0.05,0.80] (configs/base.yaml:138), so 4.1% of mixed scenes -- 14.0% if AC-40 holds -- have a diffuse field that dies before the T30 regression window closes. At the declared worst corner (12x10x5 m, alpha 0.05) depth 100 stops at 1.517 s = -21.7 dB, before the T30 window opens fully. Nothing checks this. | REMEDY: declare the coverage criterion in config and gate on it at gen-scenes, exactly as scenes.max_frac_below_iso_t30_decay_range already gates the record length.
AC-42 | source: acoustics-reviewer | major | status: OPEN | src/amcd/simulators/gsound_sir.py:446-492 ; configs/simulators/gsound_sir.yaml:87 ; configs/base.yaml:45 | RD-21's truncation QC can never fire on this backend, and reports "nothing discarded" when upstream discarded the tail. pygsound compiles maxIRLength = 3.0 s (Context.cpp:33) and does NOT expose it (module.cpp:62-69 is the complete settable set); the propagator enforces it as `maxDistance = maxIRLength * c` and skips any path beyond it (gsSoundPropagator.cpp:1457, :1571-1573). So the native IR is at most 3.0*48000 + 2048 = 146048 samples < n_samples = 204000, `_fit_to_window` ALWAYS takes the pad branch, and the disclosure is always `truncated: false, discarded_tail_db: null`. `max_discarded_tail_db` and `truncation_qc_flag` are therefore structurally dead, and `ir_duration: 4.25` (justified at configs/base.yaml:34-45 by a 4.20 s corner) cannot be filled: 29.4% of every record is guaranteed zeros (60000 samples x 16 ch x 4 B = 3.8 MB/leg; ~5.5 GB over 720 scenes x 2 legs). | REMEDY: measure the discarded tail in the PATH domain inside the worker -- the fraction of `total_energy` whose delay exceeds ir_duration, which is exact, unit-correct and free -- and declare gsound's 3.0 s cap in config so ir_duration can be validated against it rather than assumed to govern.
AC-43 | source: acoustics-reviewer | major | status: OPEN | src/amcd/simulators/gsound_sir.py:50-58,602 ; src/amcd/simulators/base.py:294-301 ; tests/test_simulator_seam.py:711-718 | The `acn_n3d` stamp is wrong in two ways and its test is a tautology. Measured by porting binding.cpp:27-130 to numpy and comparing against textbook ACN/N3D and ACN/SN3D: upstream = (1/sqrt(4pi)) * (-1)^|m| * Y_N3D_ACN, exactly, for all 16 channels. (a) The global 1/sqrt(4pi) makes it the ORTHONORMAL convention, not N3D -- N3D has Y_00 = 1, here Y_00 = 0.28209. Uniform gain, so harmless for ratios, but wrong as a label. (b) The recurrence `p[m][m] = (1-2m)*p[m-1][m-1]` (binding.cpp:75) carries the Condon-Shortley phase, which ACN/N3D (ambiX) excludes. Multiplying every channel by (-1)^m is EXACTLY a 180-degree yaw: verified, max|upstream(az) - N3D(az+180)/sqrt(4pi)| < 7e-16 at four directions. Any decoder or DOA metric reading these as acn_n3d gets the sound field rotated 180 degrees in azimuth. Nil impact today (channel 0 only); load-bearing the moment evaluation/spatial.py (RD-25) lands -- the exact risk the docstring claims to guard. The docstring's stated failure mode ("a per-degree sqrt(2l+1) error") is not the actual one. tests/test_simulator_seam.py:711 asserts only that the string equals itself. | REMEDY: stamp the convention actually produced (e.g. `acn_n3d_cs_ortho`) or normalize at the seam; replace the tautological test with the known-answer SH comparison against an independent N3D/SN3D reference.
AC-44 | source: acoustics-reviewer | major | status: OPEN | src/amcd/evaluation/room_acoustic.py:59-87 (esp. :72-75) ; src/amcd/simulators/gsound_sir.py:475-479 | The Schroeder integration limit is not gain-invariant on gsound IRs. Because `_fit_to_window` always zero-pads (AC-42), the last 10% of every record is EXACTLY zero, so `noise_power` = 0, clamps to 1e-30, and the Lundeby threshold degenerates to a fixed ABSOLUTE 1e-29 instead of a noise-floor estimate. Measured on one waveform scaled over 10 decades (1e-7 .. 1e+3): trunc_idx moves 49115 -> 52668, a 74 ms swing, from gain alone. ISO 3382 T30/EDT/C50 are ratios and must be invariant to a pure gain. Inert today (peak band energy ~1e-3 >> 1e-29) but `source_power` is a config knob (configs/simulators/gsound_sir.yaml:31) and nothing guards the coupling. It also means AC-17's shared-window machinery is not doing what its docstring says -- there is no additive noise floor to find, only a hard zero end whose position is set by the ray budget. | REMEDY: when the noise region is silent, truncate at the last non-zero sample and record that reason explicitly rather than falling through to an absolute level; add a gain-invariance known-answer test over several decades.
AC-45 | source: acoustics-reviewer | major | status: OPEN | src/amcd/simulators/gsound_sir.py:186-195,605 | The noise-carrier synthesis injects metric error at the JND scale, and the one property that makes it harmless is neither recorded nor asserted. Measured over 12 carrier realizations of an identical known decay (true T60 = 0.600 s, via a faithful port of binding.cpp:372-426): T30 sd 2.5% of true, EDT sd 8.6-10.7%, C50 sd 1.03 dB with range 0.69-5.57 dB against a true 3.349 dB. C50 sd ~ 1 dB is about one JND; T30 sd is half the project's declared d0b_t30_jnd_frac = 0.05. This is harmless ONLY because the carrier is common-mode: upstream constructs `NoiseGenerator` with a hardcoded seed 42 INSIDE the function (binding.cpp:141,329), so both legs get an identical carrier prefix, and decode imposes pred's envelope on the low-ray carrier. Nothing in amcd states or checks that dependency, and `meta["rng_seeded"] = False` flattens two different RNGs -- the unseeded ray tracer and the seeded synthesis carrier -- into one boolean. If upstream ever seeded from entropy, every C50 comparison silently acquires an independent ~1 dB error with no error raised. | REMEDY: split provenance into ray_rng_seeded / synthesis_carrier_seed, and assert cross-leg carrier identity (render two legs of one scene, compare the leading samples).
AC-46 | source: acoustics-reviewer | minor | status: OPEN | configs/base.yaml:113-118,131-132 | Upstream's geometric spreading is 1/(4pi(1+d^2)), not inverse-square 1/(4pi d^2) (gsSoundPropagator.cpp:4518-4523). Direct-path energy error: -3.01 dB at d = 1.0 m, -1.60 at 1.5 m, -0.97 at 2 m, -0.17 at 5 m, -0.04 at 10 m. The `distance_range: [1.0, null]` floor sits exactly at the worst point, and base.yaml:113-118 justifies that floor with 20*log10(r_c/d) -- a textbook inverse-square DRR that upstream does not implement. Biases absolute C50/D50/DRR low for close pairs; common-mode across legs but NOT across scenes or placement regimes, so it perturbs the near_corner vs interior_random comparison. | REMEDY: note the realized spreading law where the DRR reasoning is written, or raise the distance floor to where the deviation is below the reporting resolution (~3 m, -0.46 dB).
AC-47 | source: acoustics-reviewer | minor | status: OPEN | src/amcd/simulators/gsound_sir.py:185-195 ; src/amcd/evaluation/room_acoustic.py:129 | Band energy is not confined to its band. Measured with unit energy placed in ONE simulated band and the rendered IR re-analysed: only 48.2% of the 500 Hz band's energy lands in the 500 Hz analysis octave (48.5-50.4% for all interior bands). Upstream's intermediate crossover bands are a plain 4th-order HP+LP cascade with -3 dB points AT the crossovers (binding.cpp:190-200), whose skirts do not match amcd's butter(4)+sosfiltfilt octave. Separately, the noise carrier is not unit-variance per band, so rendered energy per unit simulated band energy spans 23.6 dB across the 8 bands, tracking octave bandwidth at +3 dB/oct. Per-band metrics are immune (both are constant per-band gains); any BROADBAND waveform quantity is not. Benign today ONLY because createbox applies one scalar alpha to all 8 bands (SoundMesh.cpp:222-223), so every band decays alike. | REMEDY: needs a guard or a known-answer test before per-band absorption is used (the createbox vector-absorption overload, module.cpp:79, is a roadmap path); record that broadband waveform energy is not a calibrated quantity.
AC-48 | source: acoustics-reviewer | minor | status: OPEN | src/amcd/simulators/base.py:126 | `relative_speeds` is the only PathData field with no declared unit or meaning, in an artifact whose entire purpose (RD-24) is to be readable without its config -- and it sits directly beside `speeds_of_sound`. It is a Doppler RADIAL VELOCITY in m/s (gsSoundPropagator.cpp:1630: `shift = 1 + relativeSpeed/speedOfSound`), zero for these static scenes, and must never be used as a propagation speed. A path-conditioned model consuming the parquet could trivially misread it as one. | REMEDY: give it the same unit/meaning comment its neighbours carry, stating that it is a Doppler radial velocity and not a propagation speed.
AC-49 | source: acoustics-reviewer | minor | status: OPEN | src/amcd/simulators/gsound_sir.py:178-179 | The worker docstring says getPathData returns `{"path_data": [<per source-listener pair>, ...]}`. Upstream builds `py::list pathDataList(n_lis)` indexed per LISTENER (Scene.cpp:169,171), aggregating all sources into one entry distinguished by `source_indices`. Correct for one source today; wrong the moment a second source is added, and the comment is what a reader would rely on. | REMEDY: correct to "per listener; sources distinguished by source_indices".
```

## Most consequential risk and the test that kills it

**AC-40.** Confirming test, one scene, no new machinery: render a single shoebox
(e.g. 5 x 4 x 3 m, alpha = 0.30, V = 60 m3, S = 94 m2) and measure T30 on the W
channel through `channel_per_band_metrics`. Sabine at nominal alpha predicts
**0.342 s**; at alpha_eff = 1-sqrt(1-0.30) = 0.1633 it predicts **0.628 s**. The
two differ by 1.84x -- far outside every measurement error quantified above
(carrier sd 2.5%, band leakage, the 3.0 s cap). Whichever the render returns
settles it. If it returns ~0.63 s, every Sabine/Eyring/DRR/critical-distance
number in `scenes/generator.py`, the `ir_duration` choice, and the
`max_frac_below_iso_t30_decay_range: 0.0` gate are all computed for the wrong room, and
AC-41's under-run fraction triples.

---

# Self-check round: falsifier + acoustics-reviewer + readability-reviewer

Run by lane R on request, over commit 99cc033. **This is a SELF-CHECK on an
unintegrated branch — NOT a clean pass** (rule 5). The falsifier wrote RD-105…
RD-122 and the acoustics-reviewer wrote AC-40…AC-49 above; the
readability-reviewer deliberately did not write, so its rows are below.

Three of its findings were verified by the lane rather than taken on trust; one
of those **corrects the reviewer**, and two reviewers **contradict each other**.
Both are flagged below so the integrator does not have to rediscover it.

## Lane verification of contested claims

### AC-43 is PARTLY WRONG — `acn_n3d` stands; the defect is a missing phase

The acoustics-reviewer reported "`acn_n3d` is wrong twice". I measured it
directly — one synthetic path through `generate_ambisonic_ir` at order 1, no
propagation, no scene, no render:

```
dir +x: channels = [-0.298384  0.  0.  0.516816]   ratios to W = [1,  0,  0, -1.7321]
dir +y: channels = [-0.298384  0.516816  0.  0.]   ratios to W = [1, -1.7321,  0,  0]
dir +z: channels = [-0.298384  0. -0.516816  0.]   ratios to W = [1,  0, +1.7321,  0]
```

- **sqrt(3) = 1.7321 confirms N3D.** SN3D would give 1.0. AC-15's normalization
  half is CORRECT and should not be reopened.
- **The position of the non-zero entry confirms ACN ordering** (W, Y, Z, X).
  AC-15's ordering half is CORRECT.
- **What is real:** X and Y are negated relative to W while Z is not — the
  (-1)^|m| Condon-Shortley phase. Negating X and Y together is a **180 degree
  yaw**. So the stamp was INCOMPLETE, not wrong.

FIX APPLIED, awaiting re-review: `_SH_CONDON_SHORTLEY_PHASE = True` in
`gsound_sir.py`, stamped into meta as `sh_condon_shortley_phase`, with the
measured ratios recorded at the constant. Kept as a separate key rather than
folded into the convention string, because the string names ordering +
normalization and both of those genuinely are `acn_n3d`. Inert today (every live
scalar metric reads channel 0); load-bearing the moment `evaluation/spatial.py`
estimates a direction, where it would otherwise read 180 degrees out in azimuth
and look like an estimator bug (RD-25).

### RD-113 vs the acoustics-reviewer — an unresolved contradiction, not a finding

- **falsifier RD-113 (major):** the zero pad "silently disables the Lundeby
  noise-floor truncation"; the returned index tracks native IR LENGTH, which is
  itself ray-budget dependent, so the E4 sweep would move the measurement window
  along with the metric.
- **acoustics-reviewer:** "Zero-padding is acoustically neutral — measured
  identical metrics and identical Lundeby index with and without padding."

They agree on the MECHANISM (a record ending in exact zeros degenerates
Lundeby's noise floor to the 1e-30 clamp — the acoustics-reviewer says the same
thing in AC-44) and disagree on whether it MOVES A REPORTED NUMBER. Both ran
probes; I have not adjudicated, because the deciding experiment is a two-budget
render (5,000 and 200,000 diffuse rays on one scene, comparing per-band shared
truncation indices) and **that is outside this lane's one-scene grant.**

**INTEGRATOR: do not close either row on the other's evidence.** The anchor,
`src/amcd/evaluation/room_acoustic.py`, is not lane R's file in any case.

### RD-110 confirmed — the redaction test WAS vacuous

The falsifier was right: `tiny_config(...)` resolves to `simulator.name =
'dry_run'`, whose params never contain `render_python`, so the assertion held
whether or not the redaction existed. FIX APPLIED, awaiting re-review: the test
now calls `_canonical_meta` against a gsound params block that DOES carry the
key, with a distinctive value. Mutation-checked — with `_HOST_SCOPED_PARAMS` set
to `()`, the host path leaks and the assertion fails:

```
with redaction DELETED, host path leaks into canonical meta: True
=> the test is non-vacuous
```

## Fixed in this round (in-lane, verified)

| row | fix |
|---|---|
| **AC-43** | `sh_condon_shortley_phase` stamped, measured evidence recorded at the constant (above). |
| **RD-110** | Redaction test rewritten and mutation-checked (above). |
| **RR-28** | The three in-repo copies of the stale "retention is native upstream / no custom trimming / getPathData is a separate call" claim corrected — `_retention_args`, `PathRetention`, and the test docstring. The lane had flagged the DOC copy (RD-103) and missed the code copies. |

`393 passed` after these changes.

## Readability rows (RR-28…RR-38)

RR-28 is fixed above. The rest are open; none is a blocker.

```
RR-29 | readability-reviewer | major | OPEN | src/amcd/simulators/base.py:130-133,227-242 | PathData's four scalar fields carry no units/semantics though describe() writes them into the parquet; num_paths (RETAINED) vs descriptor["synthesis_num_paths"] (SIMULATED) is documented only at the producer — 5,000 vs 501,492 on the smoke scene; REQUIRED_PATH_DESCRIPTOR_KEYS does not say it is a floor, not the full set | four `#:` lines + one line on the floor-not-full-set point
RR-30 | readability-reviewer | major | OPEN | src/amcd/simulators/gsound_sir.py (_WORKER_SRC main()) | the cross-process request/response contract is stated nowhere; main() is the only function in the string without a docstring, and its producer sits ~400 lines away with no stated invariant that the two halves move together | docstring listing request keys by group and the three outputs with shapes/dtypes; note native_ir_shape is written but unread
RR-31 | readability-reviewer | major | OPEN | tests/test_simulator_seam.py:1-10 | the header row-index omits every cycle-4 row (RD-24/08/67/21/93/95/102) and its scope line no longer describes the file's second half; 1188 lines, 15 classes, no contents map | extend the covered-rows list, add a 6-line contents map
RR-32 | readability-reviewer | major | OPEN | docs/ledger_inbox/R.md (RD-100..RD-104) | those five are prose sections with no severity/status/row form and three lack repo-relative anchors; RD-102 is filed under New findings but is CLOSED | move into the pipe block with anchors; mark RD-102 closed
RR-33 | readability-reviewer | minor | OPEN | src/amcd/simulators/base.py:137,218-224 | describe() vs descriptor is a near-name trap, and the PATH_SCALARS-are-reserved round-trip invariant (from_parquet strips colliding keys) is invisible to callers | one line on the field, one clause in describe()
RR-34 | readability-reviewer | minor | OPEN | src/amcd/simulators/base.py:119-120 | path_types bitmask has no pointer to where its bits are defined, and the descriptor does not capture them either | name the upstream enum, or state plainly the mapping is not captured
RR-35 | readability-reviewer | minor | OPEN | src/amcd/simulators/gsound_sir.py (_fit_to_window docstring) | claims it fits (n_channels, n_samples) but never touches the channel axis, and omits its (ir, disclosure) return contract | restate as "along TIME"; name the tuple
RR-36 | readability-reviewer | minor | OPEN | src/amcd/simulators/gsound_sir.py (_WORKER_SRC header) | unstated: the Python-version floor the worker must stay compatible with under the RENDER interpreter, and why the literal is raw | two lines
RR-37 | readability-reviewer | minor | OPEN | tests/test_simulator_seam.py | __class__/__dict__ construction obscures the invariant under test; `assert sysconfig` no-op props an unused import; magic 16/777 unlinked to the stub constants; unused ChannelLayoutType stub | as described
RR-38 | readability-reviewer | minor | OPEN | README.md:139-169 | no run-directory artifact map, so this cycle's canonical renders/<scene>/paths_*.parquet is undocumented for a newcomer | integrator routing — not lane R's file
```

## What all three reviewers agreed was sound

Worth recording so it is not re-litigated: no `platform`/arch branch, host path or
MPS assumption anywhere in `src/amcd/`; no `isinstance(..., DryRunSimulator)` or
dry-run-keyed branch — `render.py` correctly keys on `result.paths is None`; the
band definition is right to 3.7e-8 against `gsFrequencyBands.cpp` (AC-12's
88.7412 confirmed) and the two ISO eval bands coincide exactly with simulated
bands 3 and 4; Schroeder runs backward with correct ISO windows; `_retain`
reproduces `Scene.cpp:193-224` and summing intensities across bands IS the correct
broadband ordering, so RD-102's fix is sound; 344 m/s is right
(`getAirSpeedOfSound(20 C, 101.325 kPa, 50% RH)`); the truncation dB arithmetic is
right (-3.0103 dB known answer) and the flag direction is right; the parquet round
trip preserves arrays, the band axis as a shape, and descriptor identity; no
leakage, normalization-stat or split surface is touched by this lane.

## STOPPED — needs a render beyond this lane's grant

Per `docs/lanes/cycle4-R.md` ("If you want more than one scene, or any render
beyond this smoke test: STOP and write it in your inbox"), the lane did NOT run
these. Both are the deciding experiments for major findings:

1. **AC-40** (blocker, effective absorption `1-sqrt(1-alpha)`): one shoebox at
   alpha = 0.30, measure T30 on W. Sabine-nominal predicts 0.342 s, alpha_eff
   predicts 0.628 s — a 1.84x gap outside all quantified error. If alpha_eff
   wins, every Sabine/Eyring/DRR number in `scenes/generator.py`, the
   `ir_duration: 4.25` choice and the `max_frac_below_iso_t30_decay_range: 0.0` gate
   are computed for a room that was never rendered.
2. **RD-113 vs AC-44** (the contradiction above): one scene at BOTH 5,000 and
   200,000 diffuse rays, comparing `native_ir_samples` and the per-band shared
   truncation index across legs. This is also the first execution of the 200k
   leg, which has never run.

Both are ≥1 additional scene from RD-17's remaining ≤3, and (2) overlaps RD-17's
own convergence content. **Lane R is not spending them.** Recommend they be
folded into the cycle-5 RD-17 probe rather than granted separately — same scenes,
same artifact, one obligation.
