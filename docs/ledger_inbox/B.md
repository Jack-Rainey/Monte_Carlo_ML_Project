# Lane B inbox — cycle5

Branch `lane/B-cycle5`. Written by lane B, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/B-cycle5", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

**Two things this file's format has to get right, both learned in cycle 4:**

- **Give every finding a FILE ANCHOR**, as `path` or `path:line`. The integrator's
  fold copies it into the ledger's anchor column, and that column is what assigns
  the row to a lane next cycle AND what the RD-33a gate counts. Cycle 4 shipped
  116 rows anchored "see inbox" and made the gate's own lift condition
  uncomputable.
- **Number new findings from YOUR id block only** (it is in your `LANE.md`). Every
  lane runs at once, so numbering from the ledger's maximum guarantees collisions.

**This file is PERMANENT, not a scratch pad.** The integrator's fold keeps compact
rows that point back here for the measurements, so it is the primary record for
its findings and is never truncated while an OPEN row cites it — see
`docs/ledger_inbox/README.md`. Write it for a reader in a later cycle who has only
this file and the ledger row that names it.

---

## SCOPE — read this before reading the results (RD-159)

**Lane B did NOT answer the question its own title asks.** "Is the declared
population renderable?" is cluster **C6** — `AC-54` `AC-55` `AC-56` `AC-66`
`AC-67` `RD-144` — and all six of those rows are in `cycle5.yaml`'s
`serial_queue`, not in this lane. That is the RIGHT partition (three of the six
span `scenes/generator.py` and `configs/base.yaml`, and the cluster must close as
one), but it means a clean lane-B report must not be read as a "yes."

What lane B actually did: **backend CORRECTNESS AND PROVENANCE.** It hardens the
instrument the RD-33a(ii) probe will read — a silent leg becomes detectable,
artifacts get integrity digests, the `acn_n3d` stamp stops asserting itself, the
declared speed of sound stops being checked over 1 % of the paths.

**RD-33a: this lane lifts NEITHER condition.** Zero of its 40 rows are anchored on
condition (i)'s explicit path list (`scenes/**`, `evaluation/**`, `config.py`
split handling, `configs/*.yaml` split declarations), so under RD-128's severity
scoping it clears 0 of the 20 blocker/major rows there. Condition (ii) lifts only
when the RD-17 probe RUNS, and cycle 5 schedules that probe nowhere — see RD-151.

## Pass condition — pre-declared BEFORE any edit (RD-91, RD-149)

**Declared expected effect on `ci_table.csv`: NONE.** Declared before the first
edit, per the RD-94 precedent, so the integrator's interference detector can
discriminate.

**Result: byte-identical.**

```
$ shasum -a 256 <baseline captured at 92939e3, pristine tree>
74651cd26663fcc911979d9a7b9ddd8d97433ef4376fd829d35e6277d6f23052  ci_table_BASELINE.csv

$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-B/src \
  /Users/nortonrainey/miniconda3/envs/amcd/bin/amcd all \
    -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml \
    -c configs/overlays/dry_run.yaml
[done] gen-scenes (0.1s)  [done] render (0.6s)  [done] preprocess (0.5s)
[done] diagnostics (1.9s) [done] train (0.8s)   [done] infer (0.4s)
[done] eval (0.4s)        [done] stats (6.6s)   [done] report (0.1s)

$ diff ci_table_BASELINE.csv experiments/all_20260811_183720/stats/ci_table.csv
A/B: ci_table.csv BYTE-IDENTICAL to baseline
```

**RENDER GRANT: lane B spent ZERO scenes.** Pre-declared. The one item that would
need the render env (F-93's known-answer ambisonic test) SKIPS when
`spherical_harmonics_rt` is unimportable, which is this host's state. The
remaining ≤1 scene of RD-17's ≤4 is untouched — but see RD-152, it is
over-subscribed three ways.

**Full suite**, this worktree's `src`, before and after:

```
before (92939e3, pristine):  506 passed, 1 skipped in 62.53s
after  (this branch):        518 passed, 2 skipped in 63.41s
```

The +1 skip is F-93's render-env-gated test. `git merge v3-rebuild` →
`Already up to date.`, so every number above is measured on the tree being handed
over.

---

## 1. The ten "fix applied" rows nobody had re-derived — per-row verdict

`cycle5.yaml` declares one row `awaiting_re_review` (`RR-24`). In fact **ten of
lane B's forty** were cycle-4 fixes never confirmed by anyone (RD-156). CLAUDE.md
is explicit that an unverified fix is a claim, so they were re-derived FIRST,
before any new work, against the current tree.

Probe: `scratchpad/verify_cycle4.py`, reading the live objects rather than the
diff.

| row | verdict | evidence |
|---|---|---|
| **RD-08** | CONFIRMED FIXED | `IRResult.paths: 'PathData \| None'`; producer present in `render()`; `render.py` keys on `result.paths is None`, not on the type |
| **RD-114** | CONFIRMED FIXED | `render_python` redacted; **mutation-checked** — with the redaction deleted the host path leaks (`SOMEONE leaks = True`), so the test is not vacuous |
| **RD-116** | CONFIRMED FIXED | `TestRenderWorkerContract`, 10 tests, compiles the worker, AST-checks its imports and RUNS it under a venv against stubs |
| **RD-117** | CONFIRMED FIXED | `ray_budget` / `leg` / `realization_index` are REQUIRED descriptor keys; a renamed file still identifies its render |
| **RD-120** | CONFIRMED FIXED | path-file size measured in cycle 4 (0.401 MB at `top_k: 5000`); `truncation_qc_flag`'s consumer named at `gsound_sir.py` `_fit_to_window` (Step 4 / RD-14) |
| **RD-123** | CONFIRMED FIXED | exactly ONE `getPathData` call in the worker, kwargs `{energy_percentage: 100.0, max_rays: 0}` — synthesis provably sees every path |
| **RD-24** | fixed WITH F-88 | descriptor machinery present, but its band NAMING was unenforced. Hole closed this cycle by F-88 below; **close the two together** |
| **RD-67** | fixed WITH F-94 | the fill is present, but the speed cross-check ran over the RETAINED subset — ~1 % of the simulated paths. Closed this cycle by F-94; **close the two together** |
| **RD-21** | **CONDITIONAL — do NOT mark CONFIRMED FIXED** | the disclosure is present and correct on both branches (pad: `discarded_tail_db: None`; trim: `-3.0103 dB` known answer, flag fires). But `AC-56` (C6, serial queue) holds the whole QC **structurally dead** — pygsound compiles `maxIRLength = 3.0 s`, so `ir_duration: 4.25` can never be filled — and `F-83` measured that the flag cannot fire under `configs/base.yaml` at all. Verdict: **fix present, operability unresolved pending AC-56/F-83.** Issuing CONFIRMED here would be a false clearance (RD-154) |
| **RR-69** | **NOT FIXED** | see below |

### RR-69 / F-87 — cycle 4 claimed three sites corrected; only two were

Cycle 4's inbox records RR-28 as fixed: "The three in-repo copies of the stale
'retention is native upstream / no custom trimming / getPathData is a separate
call' claim corrected — `_retention_args`, `PathRetention`, and the test
docstring."

Re-derived: the two CODE sites are correct. **The third is not.**

```
stale claim present at: {'_retention_args': False, 'PathRetention': False, 'test docstring': True}
```

`tests/test_simulator_seam.py:744-746` still read:

```python
def test_the_retention_policy_maps_onto_upstream_arguments(self) -> None:
    """Retention is native upstream, so there is no custom trimming to get
    wrong: `path_retention` maps directly onto (energy_percentage, max_rays)."""
```

That is verbatim the pre-RD-102 contract that RD-102 falsified — the stale
contract comment being the thing that produced RD-102 in the first place. FIXED
this cycle: renamed to
`test_the_retention_policy_maps_onto_the_workers_own_arguments` with the actual
contract (the pair is consumed by the worker's `_retain` AFTER synthesis;
`getPathData` is always called unfiltered).

**This is the case for the re-derive rule.** Both rows were one pass from being
deleted on a claim that was two-thirds true.

---

## 2. Rows fixed this cycle, with the evidence

Every fix has a regression guard, and **every guard was mutation-checked**: the
fix was reverted on a copy of the tree and the test had to FAIL. A test that
passes without its fix is worse than no test.

```
non-vacuous  F-88 descriptor band count   — 1 failed
non-vacuous  F-95 dtype at construction   — 1 failed
non-vacuous  F-84 silent leg refused      — 1 failed
non-vacuous  F-92 retention range         — 1 failed
non-vacuous  F-98 both legs checked       — 1 failed
non-vacuous  F-90 artifact digests        — 1 failed
non-vacuous  F-94 speed over full set     — 1 failed
non-vacuous  F-85 undefined share         — 1 failed
non-vacuous  F-86a backend declares       — 2 failed
```

| row | anchor | what landed |
|---|---|---|
| **F-84** | `src/amcd/simulators/gsound_sir.py` `_fit_to_window`, `render()` | `ir_total_energy` is stamped on every leg, and a zero-energy leg RAISES naming `(scene_id, ray_budget)`. It used to return the same all-clear disclosure as a healthy leg — `total_energy == 0` was folded into "nothing was discarded" — and first surfaced as a NaN in a metric |
| **F-85** | `src/amcd/simulators/gsound_sir.py` `_retain`; `src/amcd/simulators/base.py` `PATH_SCALARS` | two defects, one site. (a) `kept_energy_percentage` is `None`, never `0.0`, when total energy is zero — 0.0 read as "we retained almost nothing" for a subset that held every path. (b) the `total > 0.0` guard **changed the selection** away from upstream: dropped, so `_retain` now reproduces `Scene.cpp:193-224` exactly (upstream keeps 1 there, this kept all) |
| **F-86**a | `src/amcd/simulators/base.py`, `gsound_sir.py`, `render.py` | the BACKEND declares its host-scoped keys (`host_scoped_params()` classmethod), the stage ASKS via `simulator_host_scoped_params(config)`. `_HOST_SCOPED_PARAMS` is deleted. **See the C3 handover below — this changes a queued blocker's edit site** |
| **F-86**b | `tests/test_simulator_seam.py` | re-verified non-vacuous by mutation before rewriting (cycle 4's claim was true) |
| **F-87** | `tests/test_simulator_seam.py:744` | the third stale-contract site, above |
| **F-88** | `src/amcd/simulators/base.py` `validate_path_descriptor`; `gsound_sir.py` `render()` | the descriptor must NAME as many bands as `intensities` has columns. `__post_init__` compared two numbers from the same producer, self-consistent by construction; a file declaring 8 centres over 9 columns passed everything. Now checked at the artifact boundary AND in `render()` against upstream's own reported `num_bands` |
| **F-90** | `src/amcd/simulators/render.py` | sha256 of every artifact into `meta.json`. `rng_seeded: false` puts reproducibility on the cached artifacts and they carried no integrity check. **A defect in my own first version was caught by the evidence run — see below** |
| **F-91** | `tests/test_simulator_seam.py` | the worker-failure test passed for the wrong reason: a nonexistent `render_python` raises `FileNotFoundError` from `subprocess.run` and never reaches the `RuntimeError`, which the `match=` alternation absorbed. Now driven by a REAL interpreter that exits non-zero, asserting the worker's own stderr reaches the parent. The nonexistent-path case is a SEPARATE test so neither can absorb the other |
| **F-92** | `src/amcd/simulators/gsound_sir.py` `PathRetention` | mode-conditional validation. `top_k: 0` and `-3` both meant "keep everything", `5000.7` truncated, `top_percent: 150` meant "all", `0` kept one path — silently reinterpreted, not rejected |
| **F-93** + **AC-57** | `tests/test_simulator_seam.py`; `gsound_sir.py` `_AMBISONIC_CONVENTION` | the tautology (`_AMBISONIC_CONVENTION == "acn_n3d"`, the constant against itself) is replaced by a KNOWN-ANSWER measurement through the real synthesizer: ACN channel positions, ratio √3 (N3D) not 1.0 (SN3D), the Condon-Shortley sign pattern, and the global 1/√(4π) scale. **SKIPS without the render env** — the honest state here. AC-57's part (a), the orthonormal-vs-N3D absolute scale, is now documented at the constant; cycle 4's rebuttal answered it on ratios only, and on ratios cycle 4 was right |
| **F-94** + **RD-67** | `src/amcd/simulators/gsound_sir.py` worker | the declared-speed cross-check MOVED INTO THE WORKER, over the unfiltered path set, before retention discards ~99 % of it. `speed_check_num_paths` records how many paths the declaration was falsified against. The parent's retained-set check stays as defence in depth |
| **F-95** | `src/amcd/simulators/base.py` `__post_init__` | declared dtype is now the WRITTEN dtype: float64 in used to become float32 on read with no error and no logged reason |
| **F-96** | `tests/test_simulator_seam.py` | `Scripts/python.exe` fallback — the POSIX-only venv layout made the whole worker-contract suite unrunnable on Windows, a declared supported host |
| **F-98** | `src/amcd/simulators/render.py` | both legs' shapes checked, and `raise` not `assert`. The high leg was checked by nothing, and `python -O` strips an assert |
| **AC-59** | `src/amcd/simulators/gsound_sir.py` | `rng_seeded` split into `ray_rng_seeded` + `synthesis_carrier_seed`. Seed **verified at the pinned source before stamping**, not assumed: `NoiseGenerator(unsigned int seed = 42)` (`auralizer/src/cpp/binding.cpp:141`), constructed defaulted at `:329`, in the checkout at SHA `608ea30f…` — the same SHA `configs/simulators/gsound_sir.yaml` pins. `rng_seeded` is KEPT because `REQUIRED_PROVENANCE_KEYS` binds `dry_run.py`, which is lane M's file |
| **AC-62** | `src/amcd/simulators/base.py:126` | `relative_speeds` documented as a Doppler RADIAL VELOCITY, explicitly not a propagation speed |
| **AC-63** | `src/amcd/simulators/gsound_sir.py` worker | `getPathData` returns per LISTENER with sources distinguished by `source_indices`, not per source-listener pair |
| **RR-70** | `src/amcd/simulators/base.py` | units/semantics on `PathData`'s four scalars; retained-vs-simulated stated at the field; `REQUIRED_PATH_DESCRIPTOR_KEYS` documented as a FLOOR |
| **RR-71** | `src/amcd/simulators/gsound_sir.py` worker `main()` | the cross-process request/response contract, by group, with the three outputs and their shapes |
| **RR-72** | `tests/test_simulator_seam.py:1-30` | contents map + the real covered-rows list |
| **RR-74** | `src/amcd/simulators/base.py` | `describe()` vs `descriptor` near-name trap, and the PATH_SCALARS-are-reserved round-trip invariant |
| **RR-75** | `src/amcd/simulators/base.py` | `path_types` bits are upstream's `gs::PathFlags` and are NOT captured by the descriptor — stated plainly rather than left to be guessed |
| **RR-76** | `src/amcd/simulators/gsound_sir.py` `_fit_to_window` | "along TIME"; the `(ir, disclosure)` return contract named |
| **RR-77** | `src/amcd/simulators/gsound_sir.py` `_WORKER_SRC` header | the Python-version floor the worker must hold under the RENDER interpreter, and why the literal is raw |
| **RR-78** | `tests/test_simulator_seam.py` | `dataclasses.replace` instead of `__class__`/`__dict__`; the no-op `assert sysconfig` and its unused import removed; unused `ChannelLayoutType` stub removed; magic 5 / 8 / 777 / 16 replaced by constants injected into BOTH stub sources from one declaration |
| **RR-84** | `src/amcd/simulators/base.py` `SceneSpec` | class docstring naming the coordinate frame and that a scene is a specification, not a render; `#:` lines giving metres, the [0,1] domain, ALL SIX SURFACES, frequency-independence, and the deriving seed aspect. The absorption line also states that the value is NOMINAL and what a backend realizes from it is that backend's convention — the C6 fact |

### F-90: my own first version was wrong, and the evidence run caught it

The first implementation digested `out_dir.iterdir()`. On this machine the dry
run then produced:

```
"artifact_sha256": {
  "._high.npy": "4c8f201c…",   <-- macOS AppleDouble sidecar
  "._low.npy":  "4c8f201c…",   <-- ditto
  "high.npy":   "c41f4aea…",
  "low.npy":    "8e98baea…"
}
```

A directory scan picks up whatever the HOST left there. Those sidecars do not
exist on Linux, so the same render would have produced a different `meta.json` on
the two supported hosts — **precisely the defect class RD-114 / F-81 exist for**,
reintroduced by the fix for a different row. Corrected to digest only the names
the stage actually wrote, with a regression guard that plants a `._low.npy` and a
`.DS_Store` in the run dir and asserts neither reaches provenance.

Recorded rather than quietly repaired, because it is the argument for running the
canonical dry run even when the change "obviously" cannot affect it.

---

## 3. Assigned rows this lane did NOT close, and why

| row | why not |
|---|---|
| **RD-121** + **RR-83** | **STRUCTURALLY BLOCKED — one defect, two rows.** Both require creating `src/amcd/simulators/_gsound_worker.py`. Lane B cannot: verified against the guard's own matcher, not assumed — `owns('src/amcd/simulators/_gsound_worker.py') = False`. This is the SAME constraint that made the worker a string in cycle 4, and cycle 4's archive tagged it "(integrator)". See RD-155: it is a C9 instance, not a standalone row |
| **RD-122** | Not a defect — a DECISION. The `paths_{low,high}` filename convention is two-leg/one-realization, but RD-117's metadata already makes every file self-identifying, so no written file needs migrating. Belongs with RD-23 at E4. Recommend re-statusing DEFERRED(E4) rather than fixing |
| **RD-33a** | The gate row. Nothing in lane B's files can lift it (see SCOPE). One lane-B-actionable observation: **its anchor text is stale** — it reads "`src/amcd/simulators/gsound_sir.py` (`render` raises NotImplementedError)" and `render()` has not raised since cycle 4. Integrator's to edit |
| **RD-21** | Fix present; verdict CONDITIONAL on C6 (above) |

---

## 4. New findings — ids from lane B's block only (`RD-150..174`, `F-110..134`, `AC-75..99`, `RR-90..114`)

`RD-150`…`RD-159` were raised by **`research-director` reviewing this lane's PLAN
before implementation**, which is the plan-stage review CLAUDE.md requires.
RD-150 was acted on immediately (F-87 had been dropped from the plan — 39 of 40
rows); the rest are recorded here because they are partition- or
integration-level, not lane-level.

```
RD-150 | research-director | major | CLOSED IN LANE | src/amcd/simulators/gsound_sir.py _retention_args, PathRetention; tests/test_simulator_seam.py:744 | F-87 was assigned to lane B and absent from every section of the plan — the one row of 40 silently dropped. | FIXED: F-87 is in section 2 above, and the plan was corrected before implementation. Recorded so the near-miss is visible; the integrator may delete this row.
RD-151 | research-director | major | OPEN | docs/lanes/cycle5.yaml (RD-17 absent from lanes, serial_queue AND integrator_queue); ledger RD-17, RD-33a(ii) | CONDITION (ii)'S DELIVERABLE HAS NO OWNER IN CYCLE 5. (ii) lifts only when the RD-17 probe RUNS. Cycle 4 unblocked it and lifted nothing (RD-89c); cycle 5's partition schedules the probe in no list at all. Protocol planning step 1c ("name the deliverable, not only its requirements") failing a THIRD consecutive cycle — the RD-81/RD-89 shape. RD-33a is assigned to lane B, so the escalation is lane B's. | Either give RD-17 a serial-queue item this cycle, or state in the resume note that cycle 5 lifts NEITHER condition of RD-33a — so cycle 6 is not planned as though the gate had moved.
RD-152 | research-director | major | OPEN | ledger RD-17 PERMISSION CLAUSE; docs/lanes/cycle5.yaml serial_queue (AC-64, AC-54) | THE <=4-SCENE RENDER GRANT IS OVER-SUBSCRIBED AND NOBODY HAS ASKED. <=1 scene remains. RD-17 itself requires >=2 scenes spanning volume/absorption extremes; item 0b (AC-64) requires a re-measure with the artifact RETAINED; item 0's remedy (AC-54 pre-compensation) requires a confirming render. Three claims, one scene, and the grant states it "does not stretch". | Total the renders items 0, 0b and RD-17 need and put the number to the user as ONE request. Lane B spent zero and pre-declared so.
RD-153 | research-director | major | OPEN | src/amcd/simulators/render.py (_HOST_SCOPED_PARAMS, deleted this cycle); src/amcd/simulators/base.py simulator_host_scoped_params; src/amcd/simulators/gsound_sir.py host_scoped_params | LANE B DELETES THE DECLARED EDIT SITE OF QUEUED BLOCKER CLUSTER C3. F-81 (blocker), F-82 and F-100 all resolve by "exclude host-scoped / disclosure-only params from the declaration render.py already uses", with pipeline.py `_render_fingerprint` (lane P) as the consumer. F-86a replaces that constant with a backend classmethod. Not a textual conflict — a semantic one landing on the integrator mid-queue. The abstraction is correct and serves the multiple-raytracers roadmap item; only the handover was missing. | See "HANDOVER: the C3 seam" below, which names the classmethod, the helper and the call site so C3 is implemented against the new seam.
RD-154 | research-director | major | OPEN | src/amcd/simulators/gsound_sir.py _fit_to_window, truncation_qc_flag; ledger RD-21, AC-56, F-83 | RD-21'S VERDICT IS NOT DECIDABLE INSIDE LANE B. AC-56 (C6) holds pygsound's compiled maxIRLength = 3.0 s makes the truncation QC structurally dead; F-83 measured that the flag cannot fire under configs/base.yaml. A CONFIRMED FIXED verdict would clear a row whose remedy the cluster says is inoperative. | Applied: RD-21 is recorded CONDITIONAL above, never CONFIRMED. Close it with C6 or not at all.
RD-155 | research-director | minor | OPEN | docs/lanes/cycle5.yaml lane B rows RD-121, RR-83; .claude/lane.json owns; tests/test_lane_partition.py | THE RD-121 PARTITION DEFECT IS A C9 INSTANCE, NOT A STANDALONE ROW. Cycle 4's archive tagged RD-100 "(integrator)"; the cycle-5 partition then gave it to a lane that cannot create files. The reachability check cannot catch it: `fix:` names an OWNED file, and the test has no notion of a fix that requires CREATING one. A third hole in C9's machine, beside RD-126/RD-142/F-103. | Missing assertion: a row whose remedy is a NEW file must declare that path in `fix:`, which the owns-check then refuses at declaration time.
RD-156 | research-director | minor | OPEN | docs/lanes/cycle5.yaml `awaiting_re_review: [RR-24]`; docs/parallel_protocol.md planning step 3 | TEN OF LANE B'S FORTY ROWS WERE FIX-APPLIED/AWAITING-RE-REVIEW, which the protocol forbids assigning to a lane, while the partition declares exactly ONE such row. RD-146's under-populated list, now measured: RD-08 RD-21 RD-24 RD-67 RD-114 RD-116 RD-117 RD-120 RD-123 RR-69. | The verify-first ordering compensated and found RR-69/F-87 NOT FIXED. Record against the partition, grouped with RD-146/C9, or the same ten reappear next cycle.
RD-157 | research-director | minor | OPEN | docs/ledger_inbox/B.md section 2 (RR-70…RR-84); ledger FINDING CLUSTERS C12 | C12 SPANS FOUR LANES AND THE INTEGRATOR QUEUE (RR-32, RR-86, RR-87), so lane B's nine-row readability pass is a CLUSTER-PARTIAL closure however well executed, and the cluster rule is "close together or not at all". | Treat lane B's RR closures as "C12-partial, lane B's share"; do not delete them before the other lanes' halves land.
RD-158 | research-director | minor | OPEN | src/amcd/simulators/render.py _canonical_meta; configs/simulators/gsound_sir.yaml; ledger RD-144, AC-55, AC-64 | C6 AND C5 WILL EDIT WHAT THIS LANE RESHAPED, AND THEIR PLANS ARE WRITTEN AGAINST THE OLD SCHEMA. RD-144 puts the absorption convention into configs/simulators/gsound_sir.yaml; AC-64 re-measures the truncation index against meta keys this lane added to. Protocol-legal but silent. | See "HANDOVER: the meta.json schema" below.
RD-159 | research-director | minor | OPEN | LANE.md title; docs/lanes/cycle5-B.md; docs/lanes/cycle5.yaml lane B `title` | THE LANE'S TITLE ASKS THE ONE QUESTION THE LANE CANNOT ANSWER. "is the declared population renderable?" is C6's question and all six C6 rows are in the serial queue, so a clean lane-B report reads as an affirmative answer to something never tested — the RD-89 shape one level up. | Applied: this file's SCOPE section says so in its first line. Retitle next cycle.
F-110 | builder (lane B) | minor | OPEN | src/amcd/simulators/base.py PathData.__post_init__ | F-95's cast is SILENT WHERE IT IS LOSSY. The declared dtype is now enforced at construction, which is what F-95 asked for, but a float64 -> float32 narrowing (or a float -> uint32 truncation on `path_types`) still happens without a logged (unit, reason), against "nothing leaves a result silently". Inert for gsound_sir, whose worker already produces the declared dtypes; live for the SECOND raytracer the roadmap wants, which is exactly the reader RD-24 exists for. | Either log the (array, from_dtype, to_dtype) when a cast changes the value, or raise on a narrowing cast and require producers to declare their own dtypes. Deliberately not decided in-lane: it is a contract question for the multi-backend seam, not a bug in this cycle's fix.
```

---

## HANDOVER: the C3 seam (for `F-81`, `F-82`, `F-100`)

`render.py:28`'s `_HOST_SCOPED_PARAMS` **no longer exists**. The replacement:

- **Declaration** — `GsoundSirSimulator.host_scoped_params()` classmethod in
  `src/amcd/simulators/gsound_sir.py`, returns `("render_python",)`.
- **Lookup** — `simulator_host_scoped_params(config)` in
  `src/amcd/simulators/base.py`; registry lookup only, no instantiation, no
  `Params` validation, so it is safe to call from a fingerprint path.
- **Consumer** — `render._canonical_meta`.
- **Default** — a backend that declares nothing returns `()`, i.e. the FULL echo.
  Chosen deliberately: the failure mode of a missing declaration is an
  over-complete provenance record, never a silent redaction.

C3's fix (`pipeline.py` `_render_fingerprint`, lane P's file) should read the
host-scoped set from `simulator_host_scoped_params`, which is the same
declaration `render.py` now uses — satisfying F-81's "sourced from the same
declaration" requirement more directly than the constant did. **F-82's
disclosure-only params (`max_discarded_tail_db`) are a SEPARATE category** and
should NOT be folded into `host_scoped_params()`: host-scoped and
disclosure-only differ in why they are excluded, and one accessor for both would
make a fingerprint bug look like a provenance bug.

## HANDOVER: the `meta.json` schema (for `AC-64`, `RD-144`, `AC-55`)

Keys ADDED to each leg's `IRResult.meta` this cycle:
`ir_total_energy`, `speed_check_num_paths`, `ray_rng_seeded`,
`synthesis_carrier_seed`. `rng_seeded` is unchanged and still present.
Added to canonical `renders/<scene>/meta.json`: `artifact_sha256`
(name → sha256, only for files the stage wrote).

**AC-64's re-measurement plan should record `ir_total_energy` per band alongside
the index it already lists** — it is now free, and it is the quantity that
distinguishes AC-64's two candidate outcomes.

`configs/simulators/gsound_sir.yaml` is UNCHANGED this cycle (no key added, none
removed), so RD-144's absorption-convention key lands on the file as the
integrator last saw it.

## Not mine — recorded, not done

- **`docs/gsound_sir_setup.md`** — still carries RD-103's stale retention claim
  and RD-104's `--sim-python` spelling. Not lane B's file, unchanged since cycle 4.
- **`docs/verbosity.md`** — still needs the `renders/<scene>/paths_*.parquet` row
  (canonical at every save level), and now also `artifact_sha256`. Not lane B's file.
- **`src/amcd/pipeline.py`** — F-65's exempt list, and C3 above.
- **`src/amcd/simulators/dry_run.py`** — lane M's; untouched. It declares no
  `host_scoped_params`, which is the documented default and needs no edit.
