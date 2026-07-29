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
| RD-16 | research-director | major | OPEN | plan Step 1 (canonical meta) + src/amcd/simulators/render.py:52, src/amcd/pipeline.py:15-16,57-58 | Render meta.json is verbosity-gated at `diagnostics` (level 4), so at default save=1 the ONLY record of how an expensive dataset was made (installed SHA, diffuse+specular counts, frequency_points, normalize flags, retention mode, speed of sound, rng_seeded) is never written. Separately, stage caching is a bare sentinel with no config fingerprint: re-running a run_dir after changing simulator params silently reuses stale renders → mixed dataset, invisible and very expensive to undo under emulation. | Split meta into a CANONICAL provenance record written at EVERY save level (diagnostic extras stay gated); fingerprint (simulator name+params+commit_sha, sample_rate, n_samples, ambisonics_order, both budgets) and fail loudly on mismatch instead of skipping. Designed into plan Step 1. |
| RD-17 | research-director | major | OPEN | plan Step 6 (ray-budget probe) | Probe was scoped as a cost curve, but its load-bearing job is validating that the HIGH leg is a CONVERGED REFERENCE — D0a headroom, every paired-improvement metric, D0b's carrier test and the E4 claim all treat it as ground truth, and `high_ray_budget: 200000` is a design_spec §14 fixed parameter that has never been validated. One scene + broadband energy-SNR cannot establish that. | Reframed in plan Step 6: ≥2 scenes spanning volume/absorption extremes; convergence measured on reported quantities (per-band energy, T30, C50) via the existing ISO-3382 path; driven through build_simulator/the worker, never pygsound directly; artifact labelled engineering-feasibility, NOT an E4 result. Run right after Step 3. |
| RD-18 | research-director | major | OPEN | plan Step 4 (QC) vs docs/research_I_paper.md:480,503; docs/design_spec.md:217 | v3 implements only 2 of Research I's 4 QC criteria, and `base.yaml:101` min_energy_db −60.0 is ~40 dB stricter than RI's 1e-10 (~−100 dB re 1.0). Missing: non-empty retained-path file, max retained-path file 128 MB (declared fixed at design_spec:217 but ABSENT from config = hidden default). QC governs dataset admission and E1 is "reproduce the old null", so a silently different admission rule confounds the reproduction. NOTE: the onset check is a PAIRED low-vs-high mismatch (RI l.480), not an absolute bound and not a geometric-expectation residual. | Implement the full RI criterion set; add max_path_file_mb and a declared energy reference (dB re 1.0 FS); match −100 dB or record the deviation and why; rename max_onset_ms → onset_mismatch_tolerance_ms; update design_spec l.217 + §6 QC-record row, not just the base.yaml comment. Designed into plan Step 4. |
| RD-19 | research-director | minor | OPEN | plan Step 4 (speed_of_sound); src/amcd/simulators/dry_run.py:10 | gsound's 344 m/s lives in C++ and cannot be governed by config, so a config-only `speed_of_sound` would DESCRIBE rather than control — the exact silent-disagreement failure it is meant to prevent (DryRunSimulator hardcodes 343.0). | Simulator DECLARES its effective speed as part of the Simulator interface and stamps it into canonical meta; render stage validates config against the declared value and hard-errors on mismatch; DryRunSimulator consumes the config value. Free empirical cross-check: PathData.speeds_of_sound. Designed into plan Step 4. |
| RD-20 | research-director | minor | OPEN | plan Step 3 (--sim-python threading); src/amcd/pipeline.py:19,87 | Widening stage dispatch to a 4th positional arg is the SECOND nine-stage-plus-tests touch for a runtime-only value; the roadmap (multiple raytracers, multiple simulation paradigms, §6 Blender front-end) makes a third foreseeable, each forcing another full sweep. | Introduce one frozen `RunContext(verbosity, host)` in runtime.py; dispatch as (config, run_dir, ctx). Same one-time mechanical cost, absorbs future runtime-only values without touching stages again. Keep frozen + documented "runtime, never experiment". Designed into plan Step 3. |
| RD-21 | research-director | minor | OPEN | plan Step 3 ("trims/pads to n_samples") | Trimming gsound's natural IR to ir_duration=3.0 s discards tail energy and can silently invalidate T30/EDT for the most reverberant scenes (Step 0's small 5×4×3 box already gave 92,859 of 144,000 samples — the EASY case). Padding is harmless but indistinguishable from truncation in the artifact. Violates "nothing leaves a result silently". | Record per (scene, leg) in canonical meta: native length, `truncated` bool, discarded tail energy in dB; QC flags above a config-declared threshold. Threshold value routed to acoustics-reviewer. Designed into plan Step 3. |
| RD-22 | research-director | minor | OPEN | plan Step 1 (RD-12 labeling half) | RD-12's fix text demands "report/stats axis labels", but no ray-budget axis exists in stats/report today (`ray_budget` appears only in config.py:296-297, render.py:39-40, simulators/). As written RD-12 cannot be confirmed closed in this gate and will linger. | Scope RD-12 closure to: yaml comment + BOTH counts in canonical render meta (RD-16) + design_spec l.219 wording. DEFER the report/stats axis label to E4, when the axis is first reported. |
| RD-24 | research-director | minor | OPEN | plan Step 2 (PathData schema) | Schema pinned to gsound's exact keys, incl. `intensities (N,8)` whose band meaning lives only in configs/simulators/gsound_sir.yaml. Roadmap has multiple raytracers; a path file from a second one would be uninterpretable without its config. | Make the parquet self-describing: store band edges/centres, num_bands, simulator name and commit SHA in the file's own metadata. Designed into plan Step 2. |
| RD-25 | research-director | minor | OPEN | plan Step 3/7 (ambisonic convention) | ACN/SN3D vs FuMa is unverified. Today's blast radius is small (evaluation/spatial.py stubbed; every live metric uses channel 0, where a FuMa 1/√2 W scaling cancels in paired quantities) — the real risk is a dataset rendered under an UNRECORDED convention that becomes load-bearing when spatial metrics land. | Verify at Step 3 by reading the auralizer binding (cheap) and record the convention as a declared field in canonical render meta; acoustics-reviewer confirms at Step 7. |
| RD-26 | research-director (via user constraint) | major | OPEN | configs/base.yaml:41-43,53-54 vs docs/research_I_paper.md Figure 5/6 (l.488-514) | E1 is "reproduce the old null", but v3's base.yaml is NOT an RI reproduction config. Ray budgets/sample rate/duration/order/seed DO match RI; scene generation and splits do NOT — shoebox dims (v3 [[3,12],[3,10],[2.4,5.0]] vs RI L4-14/W3-10/H2.4-4.5), corridor dims (v3 [[15,30],[1.5,3],[2.4,3.5]] vs RI 8-24/1.8-4.0/2.4-4.0), split counts (v3 n_id 500→300/100/100 vs RI 500/60/60 + shifts 40/30/30) and per-split seeds 1001-1006. base.yaml:41-43 calls the discrepancy "expected", but nothing pins the RI values anywhere, so E1 cannot currently be run. | Add `configs/research_i.yaml` pinning Figure 5 + Figure 6 verbatim (budgets, geometry ranges, placement constraints incl. 1.0-10.0 m source-receiver distance and per-axis margins, split counts + seeds 1001-1006, retention top_k 5000, all four QC thresholds). E1 runs base.yaml + research_i.yaml; the extension runs its own overlay. RI ray pair 5,000/200,000 is FROZEN — the Step 6 probe informs the extension ladder only. Designed into plan Step 1. |

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. IN PROGRESS: gsound_sir build plan (2026-07-13) Step 2 implements this; delete row when re-review-confirmed. |
| RD-23 | research-director | minor | DEFERRED | gsound_sir plan Step 3 (determinism caveat) | pygsound exposes no RNG seed, so ONE render per (scene, budget) conflates Monte-Carlo realization variance with the budget effect — and MC variance at low budgets IS the phenomenon under study. E4's "metric vs ray count" claim needs ≥N realizations per (scene, budget) or an explicit argument that between-scene variance dominates. | Gate: E4. Not built now. Requirement on THIS gate: the render artifact layout (renders/<scene_id>/{low,high}.npy) must not foreclose adding a realization index. |
| RD-15 | research-director | minor | DEFERRED | scripts/setup_gsound_sir.py (version selection) | Version-management conveniences — auto-fetch/auto-switch of GSound-SIR versions at render time, side-by-side multi-version envs — are wanted eventually (user 2026-07-13) but not needed for this gate. | The ref-addressable installer (any sha/branch/latest → concrete SHA) + config `commit_sha` pin + runtime installed==pinned verification already let the researcher choose any specific upstream version; automation belongs to a later tooling pass. |

### Resume here

**2026-07-28 (Opus): gsound_sir real-render build — Step 0 CLOSED, Steps 0b–7
NOT started. START HERE:** the authoritative step list is now
**`~/.claude/plans/peaceful-enchanting-clock.md`** (approved 2026-07-28). It
supersedes `~/.claude/plans/synthetic-jumping-pancake.md`, whose Steps 1 and 3
are WRONG against the real upstream API. Read the new plan first; the API facts
below are its evidence base.

The new plan already folds in the research-director plan review (rows RD-16..RD-26
above, RD-23 DEFERRED) and adds a hard constraint: **Research I's render config is
frozen for E1** (ray pair 5,000/200,000 must not be changed; see RD-26).

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
- `frequency_points` must be **`n_bands - 1` CROSSOVER points (filterbank band
  EDGES), not band centres** — hard runtime check ("Number of frequency points
  must be number of bands - 1") and they are consumed as
  `CrossoverFilter crossover(sample_rate, freq_points)`
  (auralizer/src/cpp/binding.cpp:304,334). Plan Step 1's "band centers … must
  match ray_generator's 8 intensity bands" is wrong.
  **Correct values, traced end-to-end:** `pygsound::Context()`
  (ray_generator/src/pygsound/src/Context.cpp:8) overrides GSound's log-spaced
  defaults with octave band CENTRES `{63,125,250,500,1000,2000,4000,8000}` Hz;
  `gs::FrequencyBands` derives crossovers as the geometric mean of adjacent
  centres (gsFrequencyBands.cpp:83-88). So `frequency_points` must be
  **`[88.4, 176.8, 353.6, 707.1, 1414.2, 2828.4, 5656.9]` Hz**.
  ⚠️ **Upstream `auralizer/test.py` is WRONG here** — it passes
  `[125,250,500,1000,2000,4000,8000]`, i.e. the band centres used as if they
  were edges, shifting every band edge ~½ octave high and misassigning the
  simulated per-band energies in the SH synthesis. Do not copy it. (The Step 0
  smoke test copied it; harmless there, it only had to prove plumbing.)
  Route to acoustics-reviewer at Step 6 for confirmation.
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

**No `src/amcd/` code changed yet.** Resume at **Step 0b** of
`peaceful-enchanting-clock.md` (the `scripts/setup_gsound_sir.py` installer),
then Steps 1–7 in order — noting the plan moves the Step 6 probe to run right
after Step 3 (RD-17). OPEN findings to clear before done: RD-12, RD-13, RD-14,
RD-16..RD-22, RD-24, RD-25, RD-26 (+ delete RD-08 when Step 2 is
re-review-confirmed).

**Render env state (working, verified 2026-07-28):** `amcd-render-x86` has a
correctly-built `pygsound` (rebuilt with `-DPYBIND11_FINDPYTHON=OFF`; `otool -L`
shows no libpython) and the auralizer importable as `spherical_harmonics_rt`.
Step 0b's job is to make that reproducible from scratch, not to fix it again.

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
