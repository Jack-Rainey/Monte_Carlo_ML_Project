# Review ledger

Findings from the review agents (`research-director`, `falsifier`,
`acoustics-reviewer`, `readability-reviewer`). Holds ONLY unresolved findings —
working memory for the loop, not an audit log. One row per finding:
`ID | agent | severity | status | anchor | finding | resolution`.

Status is exactly one of two values: **OPEN** (not yet resolved) or **DEFERRED**
(intentionally out of scope for the current gate, with a one-line reason and the
gate it belongs to). There is no ADDRESSED/RESOLVED status — the moment a
finding is fixed and re-review-confirmed clean, its row is deleted (git history
of this file is the audit trail).

## OPEN findings

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-12 | research-director | minor | OPEN | gsound_sir plan Step 1 (diffuse/specular mapping) | Swept variable becomes the DIFFUSE ray budget with specular fixed; the paper's "reduce ray count" could be read as total budget → E4 x-axis ambiguous/overstated. | Fix designed into plan: label swept axis as diffuse ray budget in configs/simulators/gsound_sir.yaml + report labels, stamp both counts into render meta. Awaiting implementation + re-review. |
| RD-13 | research-director | minor | OPEN | gsound_sir plan Step 1 (SimulatorSpec/_PLUGIN_BLOCKS) | Simulator params only become sweep/tune-capable if attached in `_from_merged` (symmetric with model/representation, config.py:480-486); `_PLUGIN_BLOCKS` membership alone is merge-scoping only — without the attach step the retained-path-count sweep is foreclosed. | Fix designed into plan: wire simulator attachment into `_from_merged` + config test proving a `sweep:` inside simulator params expands. Awaiting implementation + re-review. |
| RD-14 | research-director | minor | OPEN | gsound_sir plan Step 4 (render QC gate) | Per-scene QC that raises on first failure aborts a full batch — costly under Rosetta emulation. | Fix designed into plan: persist ALL renders first, write renders/qc_failures.csv, fail loudly after the batch. Awaiting implementation + re-review. |

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. IN PROGRESS: gsound_sir build plan (2026-07-13) Step 2 implements this; delete row when re-review-confirmed. |
| RD-15 | research-director | minor | DEFERRED | scripts/setup_gsound_sir.py (version selection) | Version-management conveniences — auto-fetch/auto-switch of GSound-SIR versions at render time, side-by-side multi-version envs — are wanted eventually (user 2026-07-13) but not needed for this gate. | The ref-addressable installer (any sha/branch/latest → concrete SHA) + config `commit_sha` pin + runtime installed==pinned verification already let the researcher choose any specific upstream version; automation belongs to a later tooling pass. |

### Resume here

**2026-07-28 (Opus): gsound_sir real-render build — Step 0 CLOSED, Steps 0b–6
NOT started.** Plan at `~/.claude/plans/synthetic-jumping-pancake.md` is still
the step list, but **Steps 1 and 3 of it are now partly WRONG** — see "Upstream
API corrections" below before implementing.

- **Pinned upstream SHA:** `608ea30f6dc4cda149c18947f9cae48bd379fa27`
  (yongyizang/GSound-SIR main HEAD). Clone lives at `external/GSound-SIR`
  (gitignored — build artifact, not vendored source), verified at that SHA.
- **Step 0 PASSED (2026-07-28).** Render env `amcd-render-x86`
  (osx-64, python 3.10.18, `platform.machine() == x86_64`) has `pygsound 0.3` +
  `spherical_harmonics 0.1.0` installed and the smoke test
  (createbox → getPathData → generate_ambisonic_ir) runs end-to-end under
  Rosetta. Evidence: 1,001,014 paths / 8 bands from a 5×4×3 m box
  (diffuse 5000, specular 2000); IR shape **(16, 92859)** float32 = order 3 →
  16 channels; onset 6.50 ms vs 6.52 ms predicted from the 2.236 m direct path
  (sim uses **344 m/s**, not 343); nonzero energy 1.89e-02.

**Two build defects had to be fixed to get there (both belong in Step 0b's
installer + `docs/gsound_sir_setup.md`; neither modifies upstream source):**

1. **`import pygsound` segfaulted (SIGSEGV in `PyGILState_Ensure`).** Cause:
   `pybind11_add_module(pygsound SHARED …)` (ray_generator/src/pygsound/
   CMakeLists.txt:25) under pybind11 v3.0.2's *new* FindPython mode
   (`PYBIND11_FINDPYTHON=COMPAT`) links `Python::Python`, i.e. the full
   `libpython3.10.dylib`. conda's macOS python is **statically linked**
   (`Py_ENABLE_SHARED=0`, both defaults and conda-forge — verified), so that
   dylib is a *second, uninitialised* CPython in the process. **Fix:** configure
   with **`-DPYBIND11_FINDPYTHON=OFF`** (classic mode → `pybind11::module` →
   `-undefined dynamic_lookup`, no libpython on the link line). Verified via
   `otool -L`. Not macOS-only in principle, but only bites where python is
   static.
2. **`import spherical_harmonics` failed** (`PyInit_spherical_harmonics` not
   defined). Upstream's auralizer CMake target is `spherical_harmonics` but
   `binding.cpp:486` declares `PYBIND11_MODULE(spherical_harmonics_rt, …)`, and
   its `__init__.py` is empty — so upstream's own `test.py` cannot work as
   written. **Fix:** install/import the extension under its true name,
   **`spherical_harmonics_rt`**.

**Upstream API corrections (plan Steps 1/3 assumed otherwise — verified by
introspection, not docs):**

- `sh.generate_ambisonic_ir(order, listener_directions, intensities, distances,
  speeds, frequency_points, sample_rate, precise_early_reflections=False,
  normalize=True, early_reflection_threshold=0.01)` — **there is no
  `path_types` argument** (plan Step 3 and upstream `test.py` both pass one).
- `frequency_points` must be **`n_bands - 1` CROSSOVER points, not band
  centres** (hard runtime check: "Number of frequency points must be number of
  bands - 1"). 8 intensity bands → 7 crossovers. Plan Step 1's "band centers …
  must match ray_generator's 8 intensity bands" is wrong. **The actual crossover
  values are still UNVERIFIED** — `[125,250,500,1000,2000,4000,8000]` is
  upstream `test.py`'s choice, not confirmed against GSound's internal band
  edges. Confirm against GSound source and route to acoustics-reviewer.
- Path retention is **native upstream**: `scene.getPathData(..., 
  energy_percentage=100.0, max_rays=0, use_gpu=False)`. The plan's
  `path_retention {mode: all|top_percent|top_k, value}` maps directly onto
  `energy_percentage` / `max_rays` — no custom trimming needed.
- `ps.Context` exposes `diffuse_count`, `specular_count`, `diffuse_depth`,
  `specular_depth`, `threads_count`, `channel_type`, `sample_rate`, `normalize`
  — confirms the RD-12 diffuse/specular split.
- **PathData schema is now pinned** by `path_data[i]`'s actual keys: arrays
  `distances` (N,) f32, `intensities` (N,8) f32, `listener_directions` (N,3)
  f32, `source_directions` (N,3) f32, `path_types` (N,) uint32,
  `speeds_of_sound` (N,) f32, `relative_speeds` (N,) f32, `source_indices` (N,)
  uint64; scalars `num_paths` i64, `num_bands` i64, `total_energy` f64,
  `kept_energy_percentage` f64. (Step 2 / RD-08.)
- `createbox(width, length, height, absorp, scatter)` accepts absorption as a
  scalar **or a per-band sequence** — relevant to SceneSpec.material_absorption.

**No `src/amcd/` code changed yet.** Resume at: re-plan Steps 1–5 against the
corrections above (Plan Mode + research-director), then Step 0b installer, then
Steps 1–6. OPEN findings to clear before done: RD-12, RD-13, RD-14 (+ delete
RD-08 when Step 2 re-review-confirmed).

---

**2026-07-13 (Fable): verbosity gate CLOSED — zero OPEN rows.** F-22 (threading:
frozen `Verbosity(save, show)` cli → Pipeline → all nine stages via the widened
dispatch signature; zero bare `print` outside `runtime.emit`), F-23 (save axis
gates only observability artifacts; falsifier independently verified save=0 vs
save=5 full runs bit-identical across metrics.parquet, stats, diagnostics JSONs,
and best.pt weights), F-24 (IntRange(0,5), always-stderr warnings/errors, visual
TTY guard), RD-09 (defaults 1/1 quarantined to the CLI layer, provenance rung),
RD-10 (level-5 §6 Blender seam reserved, recorded in research-director.md's
forward-looking list), RR-19 (ladder + total per-stage wiring table in
docs/verbosity.md, single `emit` helper, shared `common_options` decorator) all
implemented and confirmed clean; the pass's own findings (RD-11, RR-20..23 —
doc wording, all minor) fixed and re-review-confirmed; rows deleted. Clean-pass
evidence: falsifier zero findings, acoustics zero findings (print→emit faithful,
no physics change), research-director + readability re-confirmed their fixes.
Suite 99 passed; dry run `experiments/all_20260713_194209` (default save=1/show=1
writes provenance quartet incl. git SHA, omits train_log.csv and renders/meta.json
as specced). The dry_run plumbing gate is complete.

**Next concrete build (roadmap, per RD-06):** the
real `gsound_sir` render backend (x86) is the single hard dependency and the
next concrete build. Off it, in order: real D0a/D0b (the dry_run D0b "CARRIER
BOTTLENECK" verdict is a plumbing artifact, scientifically OPEN — RD-07) →
E1 reproduce-the-old-null (waveform) → pre-E1/post-E1 falsifier pass incl.
RD-04 split-seed guard → only then take E2 energy-residual results. Building
the §3 loss seam (metric-consistency + tail-weighting + spatial, loss.py,
F-06 unify) in parallel is fine; taking E2 results before E1 clears is not.
δ=1.0 remains PROVISIONAL, tuned at E3 (§7).
