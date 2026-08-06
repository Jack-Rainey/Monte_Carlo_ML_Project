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
| RD-13 | research-director | minor | OPEN | gsound_sir plan Step 1 (SimulatorSpec/_PLUGIN_BLOCKS) | Simulator params only become sweep/tune-capable if attached in `_from_merged` (symmetric with model/representation, config.py:480-486); `_PLUGIN_BLOCKS` membership alone is merge-scoping only — without the attach step the retained-path-count sweep is foreclosed. | FIX APPLIED (Step 1A), awaiting re-review: `_from_merged` attaches `simulator` via `_attach_params_block`; `_PLUGIN_REGISTRY`/`_PLUGIN_BLOCKS` extended. Verified live — a `sweep:` on `simulator.params.path_retention.value` expands to 3 sibling runs with role `swept`. Tests: TestSimulatorSweep. |
| RD-14 | research-director | minor | OPEN | gsound_sir plan Step 4 (render QC gate) | Per-scene QC that raises on first failure aborts a full batch — costly under Rosetta emulation. | Fix designed into plan: persist ALL renders first, write renders/qc_failures.csv, fail loudly after the batch. Awaiting implementation + re-review. |
| RD-16 | research-director | major | OPEN | plan Step 1 (canonical meta) + src/amcd/simulators/render.py:52, src/amcd/pipeline.py:15-16,57-58 | Render meta.json is verbosity-gated at `diagnostics` (level 4), so at default save=1 the ONLY record of how an expensive dataset was made (installed SHA, diffuse+specular counts, frequency_points, normalize flags, retention mode, speed of sound, rng_seeded) is never written. Separately, stage caching is a bare sentinel with no config fingerprint: re-running a run_dir after changing simulator params silently reuses stale renders → mixed dataset, invisible and very expensive to undo under emulation. | Split meta into a CANONICAL provenance record written at EVERY save level (diagnostic extras stay gated); fingerprint (simulator name+params+commit_sha, sample_rate, n_samples, ambisonics_order, both budgets) and fail loudly on mismatch instead of skipping. FIX APPLIED (Step 1A), awaiting re-review: canonical `renders/<id>/meta.json` now written at EVERY save level (verified at `--save-verbosity 0`); `STAGE_FINGERPRINT` in pipeline.py with a field-level-diff RuntimeError on mismatch (verified live: `low_ray_budget: 5000 → 4000`). Remaining for Step 3: the gsound-specific provenance values (installed SHA, both gsound ray counts, convention, truncation) fill the RD-31 slots. |
| RD-17 | research-director | major | OPEN | plan Step 6 (ray-budget probe) | Probe was scoped as a cost curve, but its load-bearing job is validating that the HIGH leg is a CONVERGED REFERENCE — D0a headroom, every paired-improvement metric, D0b's carrier test and the E4 claim all treat it as ground truth, and `high_ray_budget: 200000` is a design_spec §14 fixed parameter that has never been validated. One scene + broadband energy-SNR cannot establish that. | Reframed in plan Step 6: ≥2 scenes spanning volume/absorption extremes; convergence measured on reported quantities (per-band energy, T30, C50) via the existing ISO-3382 path; driven through build_simulator/the worker, never pygsound directly; artifact labelled engineering-feasibility, NOT an E4 result. Run right after Step 3. |
| RD-18 | research-director | major | OPEN | plan Step 4 (QC) vs docs/research_I_paper.md:480,503; docs/design_spec.md:217 | v3 implements only 2 of Research I's 4 QC criteria, and `base.yaml:101` min_energy_db −60.0 is ~40 dB stricter than RI's 1e-10 (~−100 dB re 1.0). Missing: non-empty retained-path file, max retained-path file 128 MB (declared fixed at design_spec:217 but ABSENT from config = hidden default). QC governs dataset admission and E1 is "reproduce the old null", so a silently different admission rule confounds the reproduction. NOTE: the onset check is a PAIRED low-vs-high mismatch (RI l.480), not an absolute bound and not a geometric-expectation residual. | Implement the full RI criterion set; add max_path_file_mb and a declared energy reference (dB re 1.0 FS); match −100 dB or record the deviation and why; rename max_onset_ms → onset_mismatch_tolerance_ms; update design_spec l.217 + §6 QC-record row, not just the base.yaml comment. Designed into plan Step 4. |
| RD-19 | research-director | minor | OPEN | plan Step 4 (speed_of_sound); src/amcd/simulators/dry_run.py:10 | gsound's 344 m/s lives in C++ and cannot be governed by config, so a config-only `speed_of_sound` would DESCRIBE rather than control — the exact silent-disagreement failure it is meant to prevent (DryRunSimulator hardcodes 343.0). | Simulator DECLARES its effective speed as part of the Simulator interface and stamps it into canonical meta; render stage validates config against the declared value and hard-errors on mismatch; DryRunSimulator consumes the config value. Free empirical cross-check: PathData.speeds_of_sound. Designed into plan Step 4. |
| RD-20 | research-director | minor | OPEN | plan Step 3 (--sim-python threading); src/amcd/pipeline.py:19,87 | Widening stage dispatch to a 4th positional arg is the SECOND nine-stage-plus-tests touch for a runtime-only value; the roadmap (multiple raytracers, multiple simulation paradigms, §6 Blender front-end) makes a third foreseeable, each forcing another full sweep. | Introduce one frozen `RunContext(verbosity, host)` in runtime.py; dispatch as (config, run_dir, ctx). Same one-time mechanical cost, absorbs future runtime-only values without touching stages again. Keep frozen + documented "runtime, never experiment". Designed into plan Step 3. |
| RD-21 | research-director | minor | OPEN | plan Step 3 ("trims/pads to n_samples") | Trimming gsound's natural IR to ir_duration=3.0 s discards tail energy and can silently invalidate T30/EDT for the most reverberant scenes (Step 0's small 5×4×3 box already gave 92,859 of 144,000 samples — the EASY case). Padding is harmless but indistinguishable from truncation in the artifact. Violates "nothing leaves a result silently". | Record per (scene, leg) in canonical meta: native length, `truncated` bool, discarded tail energy in dB; QC flags above a config-declared threshold. Threshold value routed to acoustics-reviewer. Designed into plan Step 3. |
| RD-22 | research-director | minor | OPEN | plan Step 1 (RD-12 labeling half) | RD-12's fix text demands "report/stats axis labels", but no ray-budget axis exists in stats/report today (`ray_budget` appears only in config.py:296-297, render.py:39-40, simulators/). As written RD-12 cannot be confirmed closed in this gate and will linger. | Scope RD-12 closure to: yaml comment + BOTH counts in canonical render meta (RD-16) + design_spec l.219 wording. DEFER the report/stats axis label to E4, when the axis is first reported. |
| RD-24 | research-director | minor | OPEN | plan Step 2 (PathData schema) | Schema pinned to gsound's exact keys, incl. `intensities (N,8)` whose band meaning lives only in configs/simulators/gsound_sir.yaml. Roadmap has multiple raytracers; a path file from a second one would be uninterpretable without its config. | Make the parquet self-describing: store band edges/centres, num_bands, simulator name and commit SHA in the file's own metadata. Designed into plan Step 2. |
| RD-25 | research-director | minor | OPEN | plan Step 3/7 (ambisonic convention) | ACN/SN3D vs FuMa is unverified. Today's blast radius is small (evaluation/spatial.py stubbed; every live metric uses channel 0, where a FuMa 1/√2 W scaling cancels in paired quantities) — the real risk is a dataset rendered under an UNRECORDED convention that becomes load-bearing when spatial metrics land. | Verify at Step 3 by reading the auralizer binding (cheap) and record the convention as a declared field in canonical render meta; acoustics-reviewer confirms at Step 7. |
| RD-26 | research-director (via user constraint) | major | OPEN | configs/base.yaml:41-43,53-54 vs docs/research_I_paper.md Figure 5/6 (l.488-514) | E1 is "reproduce the old null", but v3's base.yaml is NOT an RI reproduction config. Ray budgets/sample rate/duration/order/seed DO match RI; scene generation and splits do NOT — shoebox dims (v3 [[3,12],[3,10],[2.4,5.0]] vs RI L4-14/W3-10/H2.4-4.5), corridor dims (v3 [[15,30],[1.5,3],[2.4,3.5]] vs RI 8-24/1.8-4.0/2.4-4.0), split counts (v3 n_id 500→300/100/100 vs RI 500/60/60 + shifts 40/30/30) and per-split seeds 1001-1006. base.yaml:41-43 calls the discrepancy "expected", but nothing pins the RI values anywhere, so E1 cannot currently be run. | Add `configs/research_i.yaml` pinning Figure 5 + Figure 6 verbatim (budgets, geometry ranges, placement constraints incl. 1.0-10.0 m source-receiver distance and per-axis margins, split counts + seeds 1001-1006, retention top_k 5000, all four QC thresholds). E1 runs base.yaml + research_i.yaml; the extension runs its own overlay. RI ray pair 5,000/200,000 is FROZEN — the Step 6 probe informs the extension ladder only. Designed into Step 1.5 (see RD-27..RD-29, RD-32 for the four RI items that are NOT attainable and the merge mechanics). |
| RD-27 | research-director | major | OPEN | plan Step 1.5 (configs/research_i.yaml) vs docs/research_I_paper.md Figure 6 l.514; docs/design_spec.md §6.1 (inv #10) | RI's `test_geometry_shift` is DUAL-axis (corridor AND `near_wall+far_pair`), which violates v3 invariant #10 (a shift split perturbs exactly one axis). Reproducing it verbatim would import RI's confound; deviating makes v3's `test_geometry_shift` NOT comparable to RI's 0.61 baseline rel-L2. Gate: E1. | ARBITRATED (research-director, this session): keep invariant #10 — pin corridor + the id placement regime. E1 is a methodological, not bit-exact, reproduction (no RNG seed in pygsound — RD-23). A config comment is NOT sufficient: the E1 report must state the non-comparability per split and relabel this split ("v3-corrected single-axis geometry shift"). Disclosure obligation, not just documentation. |
| RD-28 | research-director | major | DEFERRED (implementation) / OPEN (disclosure) | plan Step 1.5; src/amcd/simulators/base.py:16 (`material_absorption: float`) vs research_I_paper.md l.512 | RI's material shift is "ceiling_absorptive AND asymmetric_walls"; `asymmetric_walls` needs per-surface absorption, which a scalar `material_absorption` (and `createbox(absorp)`, scalar-or-per-band) cannot express. Dropping half the shift changes its MAGNITUDE, so `test_material_shift` is not comparable to RI's number — same defect class as RD-26/RD-27, hence major not minor. | Implementation DEFERRED to a per-surface-materials gate. Disclosure is NOT deferrable: OPEN at E1 — the E1 report states that the material shift is half of RI's and the split is not comparable. |
| RD-29 | research-director | major | OPEN | plan Step 1.5 (placement_regimes distance_range) vs research_I_paper.md l.502,509-514 | RI never gives `mid_pair`/`far_pair` source-receiver distance sub-ranges numerically — only the global 1.0-10.0 m. Substituting the global range for all regimes changes the distance distribution of FIVE OF SIX splits (RI's id regime is `interior_random+mid_pair`), and distance sets DRR → C50/D50/EDT. Largest un-quantified deviation in the reproduction. Gate: E1. | Do not invent sub-ranges. Quantify the substitution: gen-scenes records the realized per-split source-receiver distance distribution so the E1 report states exactly what was substituted. Route with RD-37 to acoustics-reviewer at the Step 1.5 pass. |
| RD-30 | research-director | major | OPEN | plan Step 1A (pipeline.STAGE_FINGERPRINT); src/amcd/pipeline.py:15-16,57-58 | Fingerprinting only `render` leaves the RD-16 failure mode half-fixed in the very step that makes scenes/splits a live switching surface (base ↔ research_i). Changing a margin, per-split seed or split count in an existing run_dir silently serves cached gen-scenes, and render then renders the OLD geometry. | Wire `_gen_scenes_fingerprint` over {scenes, splits, scene_generation seed, per-split seeds}; render's fingerprint CHAINS the upstream gen-scenes fingerprint. Add a DEFERRED row for preprocess/train fingerprints so the remaining `None`s cannot become permanent. FIX APPLIED (Step 1A), awaiting re-review: both fingerprints wired, render chains gen-scenes via `upstream_gen_scenes`. Verified live — changing only `scenes.margin` refuses the cached render. Remaining `None`s tracked as RD-41. |
| RD-41 | research-director (follow-on to RD-30) | minor | DEFERRED | src/amcd/pipeline.py STAGE_FINGERPRINT (gate: post-E1 caching hardening) | Seven of nine stages still declare `None` (preprocess, diagnostics, train, infer, eval, stats, report) — stale-reuse is the same hazard one stage further down (e.g. changing representation params reuses cached preprocessed tensors). Declared-unwired, not silently absent, and asserted by test. | Not this gate: the expensive, irreversible artifact is the render, and 1A closes that. Wire the rest when preprocess/train inputs stop moving (post-E1). |
| RD-31 | research-director | major | OPEN | plan Step 1A (canonical meta contract); src/amcd/simulators/base.py:63-67 | "Each simulator declares its own provenance in `IRResult.meta`" with no declared required-key set lets a second raytracer (roadmap: multiple raytracers/paradigms) silently omit ambisonic convention, speed of sound or installed SHA, degrading the canonical record with no error. Mirrors design_spec §6's required metric `kind` ("no default; the spine never assumes one"). | Declare a required provenance key set on the `Simulator` interface; the render stage raises on a missing key. RD-19/RD-21/RD-25 then FILL declared slots at Steps 3/4 instead of inventing them. FIX APPLIED (Step 1A), awaiting re-review: `REQUIRED_PROVENANCE_KEYS = (simulator, ray_budget, speed_of_sound_m_s, ambisonic_convention, rng_seeded)` + `validate_provenance()` called per leg by the render stage; dry_run populates all five. |
| RD-32 | research-director | major | OPEN | plan Step 1.5 (configs/research_i.yaml overlay); src/amcd/config.py:645-654 (`_deep_update`) | YAML deep-merge cannot DELETE a key, so base.yaml's `train.frac: 0.6`, `valid.frac: 0.2` and `scenes.n_id: 500` survive into the RI overlay and would trip the all-count validator — inviting a relaxation of the very mixing guard that protects split sizing. The plan also said `n_id` "must be absent in count mode", contradicting its own required-with-explicit-null convention. | research_i.yaml sets `frac: null` / `n_id: null` explicitly; the all-frac-XOR-all-count validator reads explicit null as "no constraint" (same convention as height_range/distance_range). Add a merge test proving base+research_i resolves to pure count mode. |
| RD-33 | research-director | major | OPEN | plan sequencing (Step 1B/1C originally bundled into Step 1) | Scene/split methodology touches the leakage-critical split-assignment path and the placement distribution that sets DRR/C50, yet all review was deferred to Step 7 — AFTER the Step 6 probe and any real Rosetta render. A defect would surface only once an expensive emulated dataset existed: the most costly failure mode in this gate, and free to avoid. | Step 1A commits alone behind its bit-identity proof; 1B+1C become Step 1.5 with a falsifier + acoustics-reviewer pass at its end. HARD RULE: no real gsound render — including the Step 6 probe — until that pass is clean. Verification must include the downstream per-split MEMBERSHIP table (not just gen-scenes counts) and proof that `seeds.split_assignment` is rejected in count mode. |
| RD-34 | research-director | minor | OPEN | plan Step 1.5 (PlacementRegime `wall` type) | `wall` is selected by no config (moot under the RD-27 arbitration) and RI never gives `near_wall` numerics, so shipping it means inventing placement semantics in an untested path a later config can silently select. Premature not because it is general but because nothing defines it. | Dropped from Step 1.5; the `type` field stays extensible. Add `wall` only when a deferred RI dual-axis geometry sensitivity check (explicitly-labelled diagnostic split outside the invariant-#10 primary set; gate: E1 failure-case analysis) defines its numerics. |
| RD-35 | research-director | minor | OPEN | plan Step 1A (pipeline._mark_done / _is_done) | A sentinel storing only a sha256 cannot tell the operator WHICH field changed — under Rosetta emulation they must then decide blind whether an expensive renders/ dir is salvageable. | Store the canonical fingerprint payload (small dict) alongside the sha; the RuntimeError prints a field-level diff. Document that pre-existing legacy plain-text sentinels now raise, with --force / a fresh run_dir as the remedy. FIX APPLIED (Step 1A), awaiting re-review: sentinels are JSON `{completed_at, fingerprint}`; `_diff_fingerprints` names the changed field; legacy plain-text sentinels raise "no fingerprint" with the remedy. |
| RD-36 | research-director | minor | OPEN | plan Step 1.5 (bit-identity of base/dry_run/test_tiny) | The bit-identity proof rests on two constraints the plan never stated: base.yaml margins must stay 0.5/0.5/0.5 (RI's 0.3 m ceiling margin belongs ONLY in research_i.yaml — putting it in base silently re-datasets every dry run), and `_sample_positions` must preserve its exact RNG call sequence when height_range/distance_range are null (switching a 3-vector `uniform` to three scalar draws changes the stream). | State both as named implementation constraints; re-assert bit-identity after 1.5 against the 1A baseline. |
| RD-37 | research-director | minor | OPEN | plan Step 1.5 (_sample_positions distance rejection loop) | Unspecified whether a rejected pair resamples source AND receiver jointly or only the receiver; only joint resampling yields the uniform-conditional-on-constraint distribution, and the choice moves DRR/C50/D50 for every split. The constraint also binds asymmetrically: max achievable separation in the smallest RI shoebox is ~3.65 m (10 m cap inert) while an 8-24 m corridor IS capped. | Resample the pair jointly; record per-split acceptance rate in the gen-scenes output (no silent exclusion); route the distance-distribution substitution with RD-29 to acoustics-reviewer. |
| RD-38 | research-director | minor | DEFERRED | src/amcd/config.py:597-620,477-492 (gate: cross-model comparison / multiple raytracers) | The role grammar cannot express a sweep over a plugin NAME (`model.name`, `simulator.name`): `_attach_params_block` runs before `_resolve_roles` and requires a concrete name. The roadmap explicitly wants cross-model comparison and multiple raytracers/simulation paradigms — neither is expressible as a sweep today. | Not Step 1 work. Recorded so it is discovered now rather than at E4. |
| RD-39 | research-director | minor | OPEN | plan Step 1A (RD-12/RD-22 closure timing) | RD-12's scoped closure requires BOTH ray counts stamped in canonical render meta, which arrives at Step 3 — so RD-12 cannot close at Step 1 as the plan implied. | Keep RD-12 OPEN through Step 3. Add `configs/base.yaml:34` ("swept-capable research axis") to the diffuse-wording edit list alongside docs/design_spec.md:219. PARTIALLY APPLIED (Step 1A): both wording edits done (base.yaml ray-budget block + design_spec §7 note) and the gsound_sir.yaml diffuse/specular comment is in place; the canonical-meta half of RD-12 needs Step 3, so BOTH rows stay OPEN until then. |
| RD-40 | research-director | minor | OPEN | plan Step 1A (SimulatorSpec scope — named non-goal) | Nothing stated that `low_ray_budget`/`high_ray_budget` stay TOP-LEVEL Config fields. Migrating them into `simulator.params` during the seam work would put the swept research axis inside a plugin block, where `_merge_layer`'s F-11 name-change scoping silently drops `params` when a second raytracer is selected — foreclosing the roadmap's multiple-raytracer comparison and breaking the ray-count sweep. | Named non-goal in the plan; config test asserts the budgets survive a simulator name change. FIX APPLIED (Step 1A), awaiting re-review: non-goal documented in `SimulatorSpec`'s docstring, configs/simulators/gsound_sir.yaml, configs/base.yaml and design_spec §7; tests TestRayBudgetsStayTopLevel assert the budgets survive the switch and are absent from simulator params. |

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. IN PROGRESS: gsound_sir build plan (2026-07-13) Step 2 implements this; delete row when re-review-confirmed. |
| RD-23 | research-director | minor | DEFERRED | gsound_sir plan Step 3 (determinism caveat) | pygsound exposes no RNG seed, so ONE render per (scene, budget) conflates Monte-Carlo realization variance with the budget effect — and MC variance at low budgets IS the phenomenon under study. E4's "metric vs ray count" claim needs ≥N realizations per (scene, budget) or an explicit argument that between-scene variance dominates. | Gate: E4. Not built now. Requirement on THIS gate: the render artifact layout (renders/<scene_id>/{low,high}.npy) must not foreclose adding a realization index. |
| RD-15 | research-director | minor | DEFERRED | scripts/setup_gsound_sir.py (version selection) | Version-management conveniences — auto-fetch/auto-switch of GSound-SIR versions at render time, side-by-side multi-version envs — are wanted eventually (user 2026-07-13) but not needed for this gate. | The ref-addressable installer (any sha/branch/latest → concrete SHA) + config `commit_sha` pin + runtime installed==pinned verification already let the researcher choose any specific upstream version; automation belongs to a later tooling pass. |

### Resume here

**2026-08-05 (Opus): gsound_sir Step 1A CLOSED (implemented + self-verified, NOT
yet reviewer-confirmed). NEXT: Step 1.5.** The authoritative step list is now
**`~/.claude/plans/binary-mapping-elephant.md`** (approved 2026-08-05, after a
research-director plan review that returned four majors — RD-30..RD-33 — and
raised RD-28/RD-29 to major). It refines Step 1 of
`~/.claude/plans/peaceful-enchanting-clock.md`, which remains the authority for
Steps 2–7 and for all the upstream API facts.

**What Step 1A did** (`simulator` is now a `{name, params}` plugin block):
`SimulatorSpec` + `_from_merged` attach + `_PLUGIN_REGISTRY`/`_PLUGIN_BLOCKS`
(RD-13); `build_simulator` mirroring `build_model`; `REQUIRED_PROVENANCE_KEYS` +
`validate_provenance` on the `Simulator` interface (RD-31); canonical
`renders/<id>/meta.json` at every save level (RD-16); `STAGE_FINGERPRINT` for
gen-scenes + render with render CHAINING gen-scenes and a field-level-diff
RuntimeError (RD-16/RD-30/RD-35); `configs/simulators/{dry_run,gsound_sir}.yaml`
with the corrected band-EDGE `frequency_points`; diffuse-budget wording (RD-39);
budgets kept top-level (RD-40).

**Evidence (all run, all shown in-session):** suite **158 passed** (was 135; +23
in `tests/test_simulator_seam.py`). Drop-in proof: `amcd all -c base -c dry_run`
before vs after 1A — **209 of 210 artifacts byte-identical** (renders,
preprocessed tensors, `best.pt`, `metrics.parquet`, `drops.csv`, diagnostics
JSONs, `stats/`, `report/metrics_table.csv`); the only difference is
`report/summary.txt`'s echoed run-dir name, and the only new files are the 29
canonical `renders/*/meta.json` — which IS the RD-16 fix. Fingerprint verified
live: unchanged → `[skip] render (cached)`; `low_ray_budget` 5000→4000 → loud
refusal naming the field; `scenes.margin` change → refusal via
`upstream_gen_scenes`; `--force` rebuilds. Provenance verified at
`--save-verbosity 0`. RD-13 verified live (3 sibling runs from a sweep on
`simulator.params.path_retention.value`).

**Step 1A is NOT reviewer-confirmed.** No reviewer has run on it. Its rows
(RD-13, RD-16, RD-30, RD-31, RD-35, RD-39, RD-40) stay OPEN and say "fix applied,
awaiting re-review".

**NEXT — Step 1.5** (RI-faithful scenes/splits + `configs/research_i.yaml`):
`Scenes.margins{wall,floor,ceiling}`, `max_placement_attempts`, typed
`PlacementRegime` with required-but-nullable `height_range`/`distance_range` (NO
`wall` type — RD-34), `SplitSpec.count`+`seed` with all-frac-XOR-all-count
sizing, joint-resample rejection loop with recorded acceptance rates (RD-37),
explicit-null `frac`/`n_id` in the RI overlay (RD-32). Two named constraints the
bit-identity proof depends on (RD-36): base.yaml margins stay 0.5/0.5/0.5, and
`_sample_positions` must preserve its exact RNG call sequence when the new ranges
are null. **HARD RULE (RD-33): Step 1.5 ends with a falsifier +
acoustics-reviewer pass, and no real gsound render — including the Step 6 probe —
happens before that pass is clean.**

---

**2026-08-01 (Opus): gsound_sir real-render build — Steps 0 and 0b CLOSED,
Steps 1–7 NOT started. START HERE:** the authoritative step list is
**`~/.claude/plans/peaceful-enchanting-clock.md`** (approved 2026-07-28). It
supersedes `~/.claude/plans/synthetic-jumping-pancake.md`, whose Steps 1 and 3
are WRONG against the real upstream API. Read it first; the API facts below are
its evidence base. **Resume at Step 1** (config seam: `simulator` → `{name, params}`).

The plan folds in the research-director plan review (rows RD-16..RD-26 above,
RD-23 DEFERRED) and adds a hard constraint: **Research I's render config is
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
- **Step 0b PASSED (2026-08-01).** `scripts/setup_gsound_sir.py` +
  `docs/gsound_sir_setup.md` + `tests/test_setup_gsound_sir.py` (36 tests, no
  network/build; suite 135 green). Proven by building a **fresh** env
  `amcd-render-x86-verify` from nothing: resolved SHA == pinned, `x86_64`, both
  imports, `otool -L` clean of libpython, receipt written; smoke test
  1,000,950 paths, IR **(16, 96970)**, onset **4.35 ms observed vs 4.36 ms
  predicted** at 344 m/s. `amcd-render-x86` untouched and still passes
  `--verify-only` (it has no receipt — hand-built in Step 0 — so it reports
  "installed SHA unknown"; rebuild it with the installer before it is used for
  a real dataset, since Step 3 compares installed vs pinned SHA).

**Build defects the installer encodes** (none modifies upstream source; full
narrative in the `setup_gsound_sir.py` module docstring + the setup doc's
troubleshooting section). Steps 1–7 need only know they are handled:
`PYBIND11_FINDPYTHON=OFF` (segfault from a second libpython against a statically
linked conda python); auralizer exposed as **`spherical_harmonics_rt`** (its
CMake target name disagrees with its `PYBIND11_MODULE` name, so upstream's own
`test.py` cannot work); explicit FFTW include/lib paths (upstream links FFTW by
bare name with no working search path — the Step 0 build only linked because
`conda activate` had exported `LDFLAGS`); AppleDouble sidecar removal plus
building from a local staging copy (T7 is **exFAT**, so macOS sidecars poison
upstream's `file(GLOB om/*/*.cpp)` and setuptools' wheel step); and pinning
`Python_EXECUTABLE` for the auralizer, whose `find_package(Python …)` otherwise
takes the newest interpreter on the system (observed: a `cpython-313` extension
installed into a 3.10 env).

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
- ⚠️ **`getPathData` returns a DICT WRAPPER, not a list:**
  `{"path_data": [<per-source-listener-pair dict>, …]}`. The per-pair dict is
  `result["path_data"][i]` — the `path_data[i]` shorthand below means that, not
  subscripting the return value directly (`result[0]` raises `KeyError: 0`).
  Verified 2026-08-01.
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

**No `src/amcd/` code changed yet.** Resume at **Step 1** of
`peaceful-enchanting-clock.md`, then Steps 2–7 in order — noting the plan moves
the Step 6 probe to run right after Step 3 (RD-17). OPEN findings to clear
before done: RD-12, RD-13, RD-14, RD-16..RD-22, RD-24, RD-25, RD-26 (+ delete
RD-08 when Step 2 is re-review-confirmed).

**Reviewers have NOT run on the Step 0b code.** The gate plan puts the review
loop at Step 7; `scripts/` and `docs/` are in its scope (falsifier: no platform
coupling / config discipline; readability-reviewer: the installer's docstrings).

**Render envs (both verified 2026-08-01):** `amcd-render-x86` (Step 0, hand-built,
no receipt) and `amcd-render-x86-verify` (built from scratch by the installer,
receipt pinned to the SHA). Either works for Steps 1–7; prefer the latter, and
note `external/GSound-SIR-verify/` is a disposable second clone (~235 MB,
gitignored) that can be deleted at any time.

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
