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

Each verdict below names the assertion it rests on. The probe read the LIVE
objects (imported module attributes, `inspect.getsource`, an AST walk over
`_WORKER_SRC`), never the diff, so it describes the tree as it stands.

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

Every fix has a regression guard, and each guard listed below was
mutation-checked: the fix was reverted on a copy of the tree and the test had to
FAIL.

**CORRECTION (F-113).** An earlier draft of this section claimed EVERY guard was
mutation-checked. That was false: the row `F-85 undefined share` covers part (a),
the None-not-0.0 share, and part (b) — the removal of the `total > 0.0` SELECTION
guard — had no test at all, which the falsifier demonstrated by restoring the
guard and watching the suite stay green. A guard now exists
(`test_a_zero_energy_path_set_selects_the_way_upstream_does`) and is
mutation-checked; the claim above is narrowed to the rows actually listed.

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
RD-150 | research-director | major | OPEN | src/amcd/simulators/gsound_sir.py _retention_args, PathRetention; tests/test_simulator_seam.py:744 | F-87 was assigned to lane B and absent from every section of the plan — the one row of 40 silently dropped. | FIXED: F-87 is in section 2 above, and the plan was corrected before implementation. Recorded so the near-miss is visible; the integrator may delete this row.
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
`native_ir_total_energy`, `fitted_ir_total_energy`, `fitted_ir_samples`,
`speed_check_num_paths`, `ray_rng_seeded`, `synthesis_carrier_seed`.
`rng_seeded` is unchanged and still present.
Added to canonical `renders/<scene>/meta.json`: `artifact_sha256`
(name → sha256, only for files the stage wrote).

**AC-64's re-measurement plan can record `native_ir_total_energy` and
`fitted_ir_total_energy` alongside the index it already lists** — both are now
stamped per leg and free. NOTE, correcting an earlier draft of this handover
(AC-84): they are BROADBAND scalars summed over all channels and samples, NOT
per-band. A per-band figure would need the octave filterbank, which
`_fit_to_window` does not run.

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

---

## 5. acoustics-reviewer SELF-CHECK on lane/B-cycle5 (commit ab1f47e) — NOT a clean pass

Domain-physics audit of `src/amcd/simulators/gsound_sir.py`, `src/amcd/simulators/base.py`,
`src/amcd/simulators/render.py`, `tests/test_simulator_seam.py`, against
`external/GSound-SIR` @ `608ea30f6dc4cda149c18947f9cae48bd379fa27` (the SHA
`configs/simulators/gsound_sir.yaml:commit_sha` pins).

**Method.** Upstream's `NoiseGenerator`, `CrossoverFilter`,
`calculate_sh_normalization` and `evaluate_spherical_harmonics` were copied
VERBATIM out of `auralizer/src/cpp/binding.cpp` into standalone C++ and compiled
with `clang++ -std=c++14`, so every predicted value below is measured against
upstream's own code rather than transcribed from it. A Python port
(`scratchpad/port_binding.py`) reproduces the compiled reference bit-for-bit
(`raw_noise[139] = -0.15319705` from both).

### CONFIRMED CORRECT — physics that is right, recorded so it is not re-derived

- **ACN ordering and N3D normalization, ratios (AC-15/AC-57).** Compiled upstream,
  order 1: `+x -> [1, 0, 0, -1.73205]`, `+y -> [1, -1.73205, 0, 0]`,
  `+z -> [1, 0, +1.73205, 0]`. `gsound_sir.py:65-67` is exactly right, and
  `_AMBISONIC_CONVENTION = "acn_n3d"` is the correct stamp. Order 3 confirms the
  per-degree N3D pattern on the zonal channels: ACN 2/6/12 = sqrt(3)/sqrt(5)/sqrt(7).
- **Condon-Shortley == 180-degree yaw (`gsound_sir.py:70-71`).** Verified to be
  exactly true at orders 1, 2 AND 3, not just first order: upstream's basis equals
  the AmbiX/N3D (no-CS) basis evaluated at `(-x, -y, z)`, to 1e-6, for five test
  directions per order. The generalization the comment implies holds.
- **Band edges (`configs/simulators/gsound_sir.yaml:59` vs `:70`).** Each of the 7
  crossovers is the geometric mean of its adjacent ISO octave centres to <=3.7e-8
  relative. `Params.model_post_init` enforces it. `Context.cpp:8` does declare
  exactly those 8 centres. Energy decomposition is consistent, no leakage defect.
  The two ISO eval bands in use (`configs/base.yaml:190` = 500, 1000 Hz) are
  INTERIOR bands of the synthesis bank, so upstream's outermost LP/HP shelves
  (DC-88.7 Hz and 5657 Hz-Nyquist, which are NOT octave bands) do not reach any
  reported metric today.
- **Schroeder direction (`evaluation/room_acoustic.py:227`).**
  `np.cumsum(energy[::-1])[::-1]` — backward integration, correct direction,
  normalized to the total. T30 over [-5, -35] dB x (-60/slope), EDT over
  [0, -10] dB: ISO 3382-1 definitions, correct. C50 split at
  `int(np.ceil(0.050 * sample_rate))` with the late window bounded by the Lundeby
  index (`room_acoustic.py:437-449`): correct, [0, 50) ms / [50 ms, trunc).
- **AC-62 (`base.py:158-161`) CORRECT.** `getRelativeSpeed` at
  `gsSoundPropagator.cpp:4546-4553` is
  `dot(v_source, dirToSource) - dot(v_listener, dirFromListener)` — a projected
  radial velocity in m/s — consumed as `shift = 1 + relativeSpeed/speedOfSound`
  (`:1630`, `:2569`). Zero for static scenes. Not a propagation speed. Correct.
- **AC-63 (`gsound_sir.py:282-289`) CORRECT.** `Scene.cpp:169-274`: `pathDataList`
  is `py::list(n_lis)`, one entry per LISTENER, every source folded in and
  distinguished by `source_indices` (`:258`). `["path_data"][0]` is right for one
  listener. The kwargs `energy_percentage` / `max_rays` / `use_gpu` match the
  `SoundSource&, Listener&` overload registered at `module.cpp:93-96`, and
  `(100.0, 0)` does mean unfiltered.
- **F-85 `_retain` no-guard argument CORRECT.** On an all-zero path set with
  `energy_percentage < 100`, upstream (`Scene.cpp:214-220`) keeps 1;
  `np.searchsorted(cumulative, 0.0)` returns 0, `keep = 1`. The removed
  `total > 0.0` guard really did change the selection.
- **AC-59 carrier SEQUENCE is common-mode.** Measured: the filtered carrier prefix
  is bit-identical for two different `ir_length`s, so the two legs (separate
  worker processes, `render.py:152-153`) see the same carrier samples at the same
  indices. `NoiseGenerator(unsigned int seed = 42)` at `binding.cpp:141`,
  constructed defaulted `NoiseGenerator noise_gen;` at `:329` — the lane's citation
  is exact and 42 is the seed on the path amcd invokes.

### New findings — ids from lane B's `AC-75..99` block

```
AC-75 | acoustics-reviewer | major | OPEN | tests/test_simulator_seam.py:980; src/amcd/simulators/gsound_sir.py:73-78 | F-93/AC-57'S KNOWN-ANSWER TEST ASSERTS A VALUE THE SYNTHESIZER CANNOT PRODUCE, AND HAS NEVER RUN. Line 980 asserts `abs(abs(w) - 1/sqrt(4pi)) < 0.05*abs(w)`. But the late-field synthesis writes `result(c,t) = normalized_sh[c] * weighted_noise_sample` (binding.cpp:423), so W is Y_00 TIMES THE CARRIER SAMPLE AT THE ARRIVAL BIN, never Y_00 alone. Measured with upstream's own NoiseGenerator+CrossoverFilter compiled verbatim, for exactly the call the test makes (1 path, d=1 m, c=344 m/s, intensities all-ones over 8 bands, fs=48000, the yaml's edges): ir_length=2188, arrival bin 139, carrier S = -1.14629734, W = -0.32336451. |W| = 0.32336 vs the asserted 0.28209 — off by 14.6 %, nearly 3x the 5 % tolerance. ASSERTION FAILS. It passes today only because the test SKIPS (`71 passed, 1 skipped`, tests/test_simulator_seam.py:920, render env absent). The skip is honest; an unrun test asserting a wrong number is what makes AC-57/F-93 unclosable, and the first render-host run will fail on it. The same wrong number is documented as measured at gsound_sir.py:73-74 ("W above is 0.28209 = 1/sqrt(4pi) per unit source amplitude"). NOTE: the sibling assertions are FINE — the carrier is a per-bin scalar common to all channels, so every cross-channel ratio cancels it exactly, and the ACN/N3D/Condon-Shortley assertions are correct known answers (confirmed against compiled upstream). ONLY the absolute-scale line is wrong. | Minimal fix: delete the absolute-scale assertion and correct gsound_sir.py:73-78 to "Y_00 = 1/sqrt(4pi) is the SH NORMALIZATION CONSTANT (orthonormal, not textbook N3D's Y_00 = 1); the observed W is Y_00 x the carrier sample, measured -0.32336451 for a 1 m / 344 m/s path". Strong fix: reproduce the carrier in-test (mt19937(42) -> uniform_real<float>(-1,1) -> the LR bank) and assert W against Y_00*S; a port verified bit-identical against compiled upstream is in scratchpad/port_binding.py. Either way the assertion must be re-derived, not re-asserted.
AC-76 | acoustics-reviewer | major | OPEN | src/amcd/simulators/gsound_sir.py:96-102, 778-785 | THE SEEDED CARRIER MAKES THE SEQUENCE COMMON-MODE, NOT THE METRIC ERROR — AND THE PROVENANCE COMMENT CLAIMS THE STRONGER THING. gsound_sir.py:97-101 states the per-realization carrier error "cancels only because the seed is fixed, making the carrier common-mode across the low and high legs of a paired comparison". Half of that is verified (see CONFIRMED above: identical carrier samples at identical indices). The other half does not follow. Upstream multiplies each OCCUPIED delay bin by one carrier sample (binding.cpp:406-425), and the two legs occupy different bin sets with different per-bin band energies because the ray budget differs (5000 vs 200000, configs/base.yaml:55-56) — so the two legs weight the SAME carrier differently and the induced error does not subtract out. MEASURED on a model where both legs are given the IDENTICAL true energy-time envelope (T60 0.6 s, 1 s record, 48 kHz, Poisson bin occupancy, flat 8-band spectrum, 300 realizations, one shared seed-42 carrier reproduced from compiled upstream) so the ideal paired C50 delta is exactly 0 dB: carrier-attributable part of the paired low-minus-high C50 delta = mean -0.0277 dB, sd 0.4525 dB. That sd is 45 % of the project's own `d0b_c50_jnd_db` = 1.0 (configs/base.yaml:187), and it sits on top of the sd 0.3867 dB that IS the physics being studied. Secondary: one fixed seed also means one carrier realization for the WHOLE dataset, so the carrier's contribution is a fixed effect that a bootstrap over scenes cannot see. | Correct the comment to state what is established (identical carrier sequence, identical indexing) and what is not (cancellation of the induced metric error), then MEASURE it: drive `generate_ambisonic_ir` directly on the render host with two synthetic path sets drawn from ONE envelope at N=5000 and N=200000, and compare the paired C50/EDT delta against the same envelope with the carrier replaced by its rms. Costs ZERO renders — no propagation is involved. If the measured residual is a material fraction of the JND, it belongs in the D0b write-up as a floor on paired C50, not in a comment.
AC-77 | acoustics-reviewer | major | OPEN | src/amcd/simulators/gsound_sir.py:88-103, 785; tests/test_simulator_seam.py:1461 | `synthesis_carrier_seed` IS A LITERAL ASSERTING ITSELF — THE EXACT TAUTOLOGY F-93 REMOVED FOR THE AMBISONIC STAMP, LEFT IN PLACE ONE CONSTANT OVER. `_SYNTHESIS_CARRIER_SEED = 42` is a Python literal in amcd's own source; nothing reads it from the installed module. So the stated rationale at gsound_sir.py:100-102 — "A change to entropy seeding upstream would leave every paired metric silently noisier with nothing raising, so the fact belongs in provenance where a diff can catch it" — is NOT achieved: meta.json emits 42 whatever upstream does, and a provenance diff can never move. The only thing that would catch it is the `commit_sha` pin, which is a different guarantee (it detects ANY upstream change, so it cannot attribute one). This is precisely the asymmetry the project already resolved for `speed_of_sound_m_s`, a compiled-in fact that is DECLARED and then empirically falsified against the paths (gsound_sir.py:194-219, F-94) — and the one F-93 closed for `ambisonic_convention`. The guard test at tests/test_simulator_seam.py:1461 asserts `result.meta["synthesis_carrier_seed"] == 42` against the STUB worker, i.e. the constant against itself, and cannot fail. | A falsifying check exists and is free: the carrier is recoverable from a single-path call to `generate_ambisonic_ir` (W at the arrival bin = Y_00 x S), and S is predictable from mt19937(42) through the LR bank — measured -1.14629734 for the AC-75 setup, from compiled upstream. Fold that into the same render-env test AC-75 fixes: one call, both facts. Until then, do not describe the stamp as diff-catchable.
AC-78 | acoustics-reviewer | minor | OPEN | tests/test_simulator_seam.py:942 | THE KNOWN-ANSWER TEST MEASURES ORDER 1; PRODUCTION RENDERS ORDER 3. `configs/base.yaml:46` sets `ambisonics_order: 3` (16 channels, `config.py:724`), and `gsound_sir._ambisonics_order` derives 3 from that. The test hardcodes order 1, so it validates `calculate_sh_normalization` only at l=0 and l=1 and the Condon-Shortley phase only at |m|=1. A per-degree normalization or phase error at l=2,3 — the failure mode AC-57 names, a sqrt(2l+1) error — passes untouched. Low probability given upstream uses a closed-form K(l,m), but the path actually exercised is not the path the known answer covers. | One-line change (order 1 -> 3) plus the predicted ratios, measured from compiled upstream. +z: ACN 2/6/12 = +1.73205/+2.23607/+2.64575 (sqrt(3)/sqrt(5)/sqrt(7), all other channels exactly 0). +x: ACN 3/6/8/13/15 = -1.73205/-1.11803/+1.93649/+1.62019/-2.09165. +y: ACN 1/6/8/9/11 = -1.73205/-1.11803/-1.93649/+2.09165/+1.62019. Note the sign pattern (-1)^|m| across m=1,2,3 is visible at order 3 and invisible at order 1.
AC-79 | acoustics-reviewer | major | OPEN | src/amcd/simulators/gsound_sir.py:594-595, 726-733 | F-84'S SILENT-LEG GUARD MEASURES THE ARRAY IT DOES NOT SHIP, AND C6's PENDING FIX WOULD OPEN THE HOLE. `_fit_to_window` computes `total_energy` over `native` (line 595), `render()` refuses the leg on `truncation["ir_total_energy"] <= 0.0` (line 726) — but the array handed downstream is `ir`, the FITTED (n_channels, n_samples) array. If the native IR's energy lay entirely beyond `n_samples`, the guard passes and a silent `ir` ships: exactly the NaN-in-a-metric failure F-84 exists to prevent, one branch over. Inert TODAY only because pygsound's compiled `maxIRLength = 3.0 s` (AC-56) keeps native <= 146048 samples against `n_samples` 204000, so `_fit_to_window` always takes the PAD branch and the two energies coincide. That is a C6 fact, and C6 is queued to change it: any fix that lifts the 3.0 s cap toward `ir_duration` 4.25 s makes the trim branch live and the hole real. Not re-litigating C6 — flagging that a change this lane landed depends on C6's current, wrong state for its correctness. | Guard the SHIPPED array: check the energy of `ir` (or both, and disclose them separately). Zero cost — `ir` is in hand at line 720. Add a known-answer test that trims all energy past the window and asserts the leg is refused; today's stub cannot reach it because the stub's native IR is shorter than the window. Close WITH C6, not before.
AC-80 | acoustics-reviewer | minor | OPEN | src/amcd/simulators/gsound_sir.py:625; src/amcd/simulators/base.py (IRResult.meta contract) | `ir_total_energy` IS STAMPED WITH NO DECLARED UNIT, NO REFERENCE, AND A NAME THAT MISDESCRIBES IT. It is sum-of-squares over the NATIVE (pre-fit) IR in float32 sample units — dimensionless, uncalibrated, no dB reference — while its sibling in the same dict is correctly named `native_ir_samples`. A reader of meta.json comparing `ir_total_energy` between legs will read it as the shipped IR's energy; today they coincide only because the pad branch always fires (see AC-79). The float64 accumulation (`native.astype(np.float64) ** 2`) is the RIGHT choice and should be kept. | Rename to `native_ir_total_energy` (or stamp both) and state the unit at the site: "sum of squared sample amplitudes over the native IR, float64 accumulation, arbitrary/uncalibrated amplitude units, no dB reference". F-84's question "is zero total energy the physically right test for a dead leg" — YES for exact zero, and float64 sum-of-squares is the right quantity; the defect is which array it is computed over, not the quantity.
AC-81 | acoustics-reviewer | minor | OPEN | src/amcd/simulators/base.py:145-147, 163-164 | `listener_directions` / `source_directions` DECLARE NO COORDINATE FRAME AND NO ARRIVAL-VS-EMISSION SENSE, while `listener_directions` IS the sole spatial input to the ambisonic encoding (gsound_sir.py:299 -> binding.cpp:394-397). `SceneSpec` was given an explicit frame declaration this cycle (RR-84, base.py:20-24); `PathData` says only "(N, 3) unit vectors". The distinction is load-bearing for the very finding this lane stamped: whether the CS phase reads as a 180-degree yaw depends on whether the vector points TOWARD the arrival or AWAY from it, and upstream stores `directionFromListener` for one and `-directionToSource` for the other (gsSoundPropagator.cpp:1416). Anyone filling `evaluation/spatial.py` (RD-25) has to re-derive this from C++. | State at the field: world vs listener-local frame, the sense (direction of arrival at the listener, per upstream `directionFromListener`), and that no listener orientation is set so local == world here. One sentence each; no code change.
AC-82 | acoustics-reviewer | minor | OPEN | src/amcd/simulators/gsound_sir.py:156-191, 528-531 | "_retain REPRODUCES upstream's selection EXACTLY (Scene.cpp:193-224)" IS AN OVERCLAIM ON THE `top_percent` BRANCH. Two divergences, both in that branch: (a) upstream accumulates in `Real` = float32 progressively (`Scene.cpp:213-219`) while `_retain` uses a float64 `np.cumsum` — over ~10^6 paths, float32 running accumulation drifts enough to move the cut index; (b) upstream sorts with `std::sort` (UNSTABLE, `Scene.cpp:202`) while `_retain` uses `np.argsort(kind="stable")`, so tied path energies select a different subset. Neither is exercised today — `configs/simulators/gsound_sir.yaml:123` is `mode: top_k`, so `_retention_args` returns `energy_percentage = 100.0` and the branch is skipped entirely. Retention affects only the saved artifact, never the IR, so nothing physical is at risk now. Reported per the unexercised-path rule: it is a valid config input that would silently retain a different path set than the docstring promises. | Either narrow the claim ("reproduces upstream's selection; the `top_percent` cut index may differ by a path or two from upstream's float32 accumulation and unstable tie-break") or add a known-answer test on a tied/near-tied path set. Do not remove the branch — `top_percent` is a declared config mode.
AC-83 | acoustics-reviewer | minor | OPEN | src/amcd/simulators/base.py:158-161 | `relative_speeds` DECLARES ITS UNIT BUT NOT ITS SIGN. AC-62's semantics are correct (verified, see CONFIRMED above), and m/s is stated, but not whether positive means approaching or receding. Upstream is `sourceSpeed - listenerSpeed` with `sourceSpeed = dot(v_source, directionToSource)` (gsSoundPropagator.cpp:4549-4552), consumed as `shift = 1 + relativeSpeed/c`, and the two call sites pass direction vectors whose orientation the header does not name. Exactly zero and inert for these static scenes; live for any roadmap backend with moving sources, which is the reader RD-24/RR-70 exist for. | One clause at the field naming the sign convention, derived at the source rather than assumed. Needs a guard or known-answer test if a moving-source backend ever lands, not removal.
AC-84 | acoustics-reviewer | minor | OPEN | docs/ledger_inbox/B.md:293-294 (this file, HANDOVER: the meta.json schema); src/amcd/simulators/gsound_sir.py:625 | THE C6/AC-64 HANDOVER PROMISES A PER-BAND QUANTITY THE CODE DOES NOT PRODUCE. The handover tells AC-64's re-measurement to "record `ir_total_energy` per band alongside the index it already lists — it is now free". It is not free and it is not per band: `_fit_to_window` stamps ONE broadband scalar summed over all channels and all samples. Per-band energy of the fitted IR would require running the octave filterbank, which `_fit_to_window` does not do. The integrator will act on this handover mid-serial-queue. | Correct the handover to "broadband, whole-IR, native, sum of squares" (see AC-80), or add the per-band quantity if AC-64 genuinely needs it — but then it must be declared as an octave-band decomposition with named centres, not as a variant of this key.
AC-85 | acoustics-reviewer | minor | OPEN | src/amcd/simulators/gsound_sir.py:609-613, 620-630 | THE FIT DISCLOSURE IS ASYMMETRIC: THE TRIM BRANCH REPORTS WHAT IT COST, THE PAD BRANCH REPORTS NOTHING. Under today's 3.0 s cap the pad branch fires on EVERY leg, appending ~1.2 s (57952 samples at fs 48000, n_samples 204000) of exact digital silence to every IR before any ISO metric sees it — and the disclosure dict records only `truncated: False` and `discarded_tail_db: None`. The pad length is derivable (`n_samples` is in `render._canonical_meta`, `native_ir_samples` in the leg meta) but is not stated as a fact about the record, and a hard zero tail is an input to Lundeby noise-floor estimation and to the Schroeder integration bound. | Stamp `padded_samples` (or `fitted_ir_samples`) alongside `native_ir_samples`, units samples. Close WITH C6 (AC-56/RD-21/F-83): the same cluster decides whether the pad branch stays the only live one.
```

---

## 6. Falsifier self-check on `lane/B-cycle5` @ ab1f47e — adversarial re-derivation

Self-check on an unintegrated branch, NOT a clean pass (`docs/parallel_protocol.md` rule 5).
Every row below was derived from the CURRENT state of the four audited files, not from the diff.
Ids from lane B's `F-110..134` block, starting at F-111 (F-110 already used).

```
F-111 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:595, :620-630, :726-733 | F-84's SILENT-LEG GUARD TESTS THE WRONG ARRAY. `_fit_to_window` computes `total_energy` from `native` (:595), i.e. BEFORE the trim, and stamps it as `ir_total_energy`; `render()` then guards on `truncation["ir_total_energy"] <= 0.0` (:726). The file written to disk is the FITTED array. PROBE (n_samples=500, native (16,5000) with its only arrival at sample 3000): native energy 1.0, STORED ir energy 0.0, `ir_total_energy <= 0.0` is False -> the all-zero leg SHIPS. F-84's headline claim ("a silent leg is refused rather than shipped") is false for the artifact that is actually written. The only thing that fires is `truncation_qc_flag`, which gsound_sir.py:590 states NOTHING CONSUMES YET. | Guard on the FITTED ir (or on both), and add a test whose native IR is non-silent but whose fitted window is zero. Confirming test: assert `render()` raises when `_fit_to_window` returns an all-zero `ir` with non-zero native energy.
F-112 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:604-608, :620-630; docs/ledger_inbox/B.md:293-295 | `ir_total_energy` IS MISNAMED AND THE HANDOVER PROPAGATES IT. It is the NATIVE IR's energy, not the energy of `low.npy`/`high.npy`. PROBE: native [arrival at 0 plus 0.5 at 3000], meta `ir_total_energy` = 1.25, energy of the stored array = 1.0 (ratio 0.8). Section 2's HANDOVER tells AC-64 to "record `ir_total_energy` per band alongside the index", so the ambiguity is being handed to the re-measurement that will read it. Adjacent: `discarded_tail_db` uses `None` for "nothing discarded" and `0.0` for "EVERYTHING discarded" (PROBE: 0.0 dB in the case above) — opposite ends of the scale, adjacent-looking values. | Rename to `native_ir_total_energy` (it already sits beside `native_ir_samples`) or stamp both native and fitted energies. State the None-vs-0.0 semantics at the field.
F-113 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:176-186; tests/test_simulator_seam.py (no covering test); docs/ledger_inbox/B.md:160-171 | F-85(b) HAS NO REGRESSION GUARD, AND SECTION 2's BLANKET MUTATION CLAIM IS FALSE FOR IT. Section 2 says "**every guard was mutation-checked**: the fix was reverted on a copy of the tree and the test had to FAIL"; its table row is "non-vacuous F-85 undefined share — 1 failed", which is part (a) only. MUTATION PROBE (scratchpad copy, `if energy_percentage < 100.0 and total > 0.0:` re-added): `pytest tests/test_simulator_seam.py` -> **71 passed, 1 skipped**; full suite -> **516 passed** (2 failures are the source-tree-isolation guards firing on the scratch copy, unrelated). No test reaches `_retain` with zero total energy — grep for `_retain`/zero-energy over the suite returns only the stub's `AMCD_STUB_SILENT` IR path, not a path-energy case. Compounding: the branch is also unreachable in production, since zero path energy implies a zero IR and F-84's raise fires first. | Add a worker-level test with an all-zero `intensities` set asserting keep == 1 under `energy_percentage < 100`. Until then F-85(b) is an unverified claim, and the section-2 mutation statement must be narrowed to the rows it actually covers.
F-114 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:157, :180-181, :530 | "REPRODUCES `Scene.cpp:193-224` EXACTLY" IS OVERSTATED IN THREE PLACES. I transcribed upstream at SHA 608ea30f and compared `keep` AND the kept index list over 17 edge cases (all-zero energies, empty set, exact boundary, ep 0/100/150, max_rays 0/1/>n/negative, ties): all 17 agree, so the COUNT logic is right and F-85(b) is correct as far as it goes. But (a) `std::sort` at Scene.cpp:202-203 is UNSTABLE, so with ties at the cut upstream's kept indices are unspecified — PROBE: three admissible input orders give upstream {0,1}, {3,2}, {2,0} while `_retain`'s stable argsort always gives {0,1}; (b) upstream accumulates in `Real` = float32 (`gsound/gsConfig.h:375`) while `_retain` sums in float64, so a `top_percent` cut near a boundary can land one path differently; (c) `np.searchsorted` at :183 assumes a MONOTONE cumulative, which fails if any per-path energy is negative, while upstream's loop still works — an unexercised path with no guard. | Reword to "the same selection rule, with a deterministic tie-break and float64 accumulation", and either assert non-negativity in `_retain` or document the assumption. `_retain`'s determinism is preferable to upstream's; only the word "exactly" is wrong.
F-115 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:444-467, :708-718; src/amcd/simulators/base.py:327-337; configs/simulators/gsound_sir.yaml (band_centres_hz, frequency_points) | F-88 CHECKS BAND COUNTS, NOT BAND IDENTITY — A WHOLE FILTERBANK ONE OCTAVE OFF PASSES EVERY GUARD. PROBE: centres doubled to [126, 250, 500, ...] with edges recomputed as their geometric means is ACCEPTED by `Params` (`_check_centre_count`, `_check_band_edges`, `model_post_init`), satisfies `render()`'s new count check against the worker's `num_bands` (8 == 8, 8 == 7+1), and satisfies `validate_path_descriptor`. `frequency_points` is passed to `sh.generate_ambisonic_ir` (:296-306), so a self-consistent-but-wrong bank does not merely misname the columns — it CHANGES THE IR, which is the AC-12 failure mode at full scale. Nothing anywhere anchors the declared centres to pygsound's compiled set (`Context.cpp:8`), and pygsound exposes no accessor for them. Second, smaller hole at base.py:329: the check is `len()` with no type validation — PROBE: `band_centres_hz='63125250'` (a str of len 8) with `band_edges_hz='5000000'` is ACCEPTED, and an int raises `TypeError`, not the intended `ValueError`. | Add a test that parses `Context.cpp:8` from the pinned checkout and asserts `band_centres_hz` equals it, skipping when the checkout is absent (the F-93 pattern). Type-check the descriptor values as sequences of numbers, not just `len()`.
F-116 | falsifier | minor | OPEN | src/amcd/simulators/render.py:67-107, :204-216; src/amcd/pipeline.py:76 `_render_fingerprint`; tests/test_simulator_seam.py:763-777 | F-90's DIGESTS ARE WRITTEN AND NEVER VERIFIED BY ANY PIPELINE CODE. `grep -rn artifact_sha256 --include=*.py` shows exactly one writer (`render.py:97`, `:206`) and no reader outside the tests; `test_a_corrupted_artifact_no_longer_matches_its_provenance` recomputes the digest ITSELF. The cache-reuse path admits a cached render on a config fingerprint alone and never touches the digests, so the row's own rationale ("`rng_seeded: false` puts reproducibility on the cached artifacts and they carried no integrity check") is only half closed: the record exists, the check does not. This is a number in canonical output that contributes nothing to the inferential result. Adjacent: `_canonical_meta`'s `artifact_sha256: dict | None = None` defaults to `{}` (:97), which is indistinguishable from "this render wrote nothing" — reached today only by the test at :1059, but it is a latent unscored-quantity-as-a-number path. | Verify the digests where a cached render is REUSED (the stage-cache path), or state in the docstring that this is a provenance record with no verifier. Make the default `None` raise rather than render `{}`.
F-117 | falsifier | minor | OPEN | src/amcd/simulators/base.py:207-232 (`to_parquet`); src/amcd/simulators/render.py:206 | THE PARQUET DIGESTS ARE LIBRARY-VERSION-SCOPED, SO F-90's RECORD IS NOT HOST-PORTABLE FOR THE PATH ARTIFACTS. PROBE: every file `to_parquet` writes embeds `created_by: parquet-cpp-arrow version 24.0.0`; the same logical PathData written under a different pyarrow gives a different sha256. The `.npy` digests ARE portable (`np.save` pins dtype/endianness). Same defect class as the AppleDouble bug section 2 records — a host/toolchain fact entering canonical provenance — just one layer down, and it is not covered by the "only what the stage wrote" guard, which fixes the file SET, not the file BYTES. | Either state at `_sha256` that a digest is a within-host integrity check and not a cross-host identity, or digest the logical content (arrays + descriptor) rather than the container bytes.
F-118 | falsifier | minor | OPEN | src/amcd/simulators/render.py:139-146, :181-206 | A STALE ARTIFACT INSIDE A RETAINED SCENE DIR SURVIVES PRUNING AND IS INVISIBLE TO PROVENANCE. Pruning removes whole orphan scene DIRS whose name is not in `current_ids`; it never removes stale FILES inside a dir that is kept. F-90 then digests only `written`, so the stale file is not in `artifact_sha256` either. PROBE: plant `renders/scene_0000/paths_low.parquet` before a dry_run render -> after the render, files on disk are ['high.npy','low.npy','meta.json','paths_low.parquet'] while `artifact_sha256` keys are ['high.npy','low.npy']. A backend switch (paths-exporting -> not, or a retention change) therefore leaves a readable path file that no meta.json describes and no prune removes. | Prune unexpected files inside a retained scene dir, or record the (unit, reason) for every file present but not written. Confirming test: the probe above, asserting the stale file is gone or is reported.
F-119 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:632-655, :702, :205-210; tests/test_simulator_seam.py:989-1002; docs/ledger_inbox/B.md:196-199 | THE PARENT-SIDE SPEED CHECK IS DEAD CODE, NOT "DEFENCE IN DEPTH". The worker checks the UNFILTERED set with `np.allclose(..., rtol=1e-3)` and refuses an empty one (:205-218); the parent then re-checks the RETAINED SUBSET of that same set with the same tolerance (:702), which cannot fail if the worker's passed, and its `observed.size == 0` branch cannot be reached because `_retain` returns an empty set only when the worker already exited. MUTATION PROBE: deleting the call at :702 leaves `tests/test_simulator_seam.py` at **71 passed, 1 skipped**. The two tests that appear to cover it (`:989`, `:999`) call `_check_declared_speed` DIRECTLY, so they hold coverage open while proving nothing about the render path. | Either delete the parent-side call and its direct tests, or make it a genuinely independent check (different tolerance / different quantity) and drive it through `render()` with a worker whose retained subset disagrees.
F-120 | falsifier | major | OPEN | tests/test_simulator_seam.py:1375-1399; configs/simulators/gsound_sir.yaml (`render_python: null`) | F-91's REWRITTEN TEST IS HOST-STATE-COUPLED AND FAILS ON A DECLARED SUPPORTED HOST. `test_the_parent_surfaces_a_worker_failure_with_its_stderr` builds `_gsound_sim(render_python=sys.executable)` and depends on the PIPELINE interpreter having no `amcd_gsound_install.json` — an accident of this Mac. On a native x86_64 Ubuntu/Windows host set up per `docs/gsound_sir_setup.md`, `render_python: null` is documented as CORRECT precisely because the render env IS the pipeline env: the receipt is present, the SHA matches, `import pygsound` succeeds, and the worker performs a REAL 4.25 s / 16-channel / 5000-diffuse-ray render inside the unit suite — after which `pytest.raises(RuntimeError)` fails with DID NOT RAISE. Unlike its neighbours (`_stub_env`), this test does not isolate the interpreter it runs. Same defect class as the platform-coupling rule; introduced by this cycle's F-91 rewrite. | Drive it through `_stub_env` with a receipt whose SHA mismatches (a guaranteed non-zero exit on every host), not through `sys.executable`. Confirming test: run this file in an env where `scripts/setup_gsound_sir.py` has been run against the pipeline interpreter.
F-121 | falsifier | minor | OPEN | src/amcd/simulators/base.py:404-462 (`Simulator`), :447-462; src/amcd/simulators/dry_run.py | ADDING `host_scoped_params` TO THE `@runtime_checkable` PROTOCOL BROKE THE SCAFFOLD'S CONFORMANCE, WHILE THE DOCSTRING CALLS IT "Optional". Protocol membership is structural and not optional. PROBE: `issubclass(DryRunSimulator, Simulator)` is now **False**; at 92939e3 the Protocol declared only `render` and `min_source_receiver_distance_m` and it was True. Latent today — nothing isinstance-checks `Simulator` — but the `runtime_checkable` decorator exists to invite exactly that check, and the first caller to add one silently rejects the scaffold and every future backend that legitimately declares nothing. `simulator_host_scoped_params` (:528) already handles absence correctly, which is what makes the Protocol declaration the inconsistent half. | Give the Protocol member a real default body (`return ()`) instead of `...`, or move it out of the Protocol and document it as duck-typed. Confirming test: `assert issubclass(DryRunSimulator, Simulator)`.
F-122 | falsifier | minor | OPEN | src/amcd/simulators/base.py:184-198, :265-270 | CORROBORATES AND WIDENS THE LANE'S OWN F-110: F-95's CAST IS VALUE-DESTROYING, AND FOR SOME INPUTS WARNING-FREE. PROBES at default numpy settings — float64 `[3.7, -1.0, 4.29e9]` -> uint32 `[3, 0, 4290000000]` (a `RuntimeWarning` to stderr, no `(unit, reason)`); float64 `[0.0, nan, 1.5]` -> uint64 `[0, 0, 1]`, i.e. a corrupt value becomes a VALID-LOOKING source index 0; int64 `[-5, 1, 2]` -> uint64 `[18446744073709551611, 1, 2]` with **no warning at all**. `test_the_declared_dtype_is_the_dtype_that_gets_written` (:588) asserts dtype conformance only, with exactly-representable values, so no test covers a lossy cast. Round-trip itself IS faithful (all 8 arrays + descriptor + a None `kept_energy_percentage` verified identical), so F-95's round-trip claim holds. Second, smaller hole: `from_parquet` (:265-270) raises a bare `TypeError` if a non-nullable scalar is null. | Group with F-110 and close together. Raise on a narrowing/lossy cast rather than warn; make the non-nullable-null case raise the same explanatory `ValueError` the missing-descriptor case does.
F-123 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:37-48, :141, :149 | `_RECEIPT_NAME` AND `_RECEIPT_SHA_KEY` ARE DEAD, AND THEIR 8-LINE COMMENT ASSERTS A CONTRACT THAT DOES NOT HOLD. `grep -rn '_RECEIPT_NAME|_RECEIPT_SHA_KEY' src/ tests/ scripts/` returns only the two definition lines. The worker hardcodes `"amcd_gsound_install.json"` at :141 and `["commit_sha"]` at :149 in the raw string. The comment explains at length why the duplication with `scripts/setup_gsound_sir.py` is deliberate, but changing either constant changes nothing — they are a third copy, not a single source of truth, and the comment makes them look load-bearing. | Interpolate them into `_WORKER_SRC` (it is already a formatted literal elsewhere in the file's tooling) or delete them and move the rationale to the worker's own lines. Confirming test: assert `_RECEIPT_NAME in _WORKER_SRC` and that changing it changes the worker.
F-124 | falsifier | minor | OPEN | src/amcd/simulators/gsound_sir.py:89-103; tests/test_simulator_seam.py:1461 | `_SYNTHESIS_CARRIER_SEED = 42` IS THE ONE REMAINING UNFALSIFIABLE UPSTREAM STAMP, AND ITS OWN RATIONALE SAYS IT SHOULD NOT BE. AC-59's comment argues the seed is stamped so "a diff can catch" an upstream change to entropy seeding. It cannot: the value is a literal in OUR module, so provenance reads 42 whatever upstream does. This is exactly the asymmetry F-93/AC-57 closed for `_AMBISONIC_CONVENTION` (now a known-answer measurement) and RD-19 closed for `speed_of_sound_m_s` (cross-checked against the paths' own array). No test asserts the value at all: `:1461` reads `assert result.meta["synthesis_carrier_seed"] == 42`, a literal against a literal, which can only fail if the stamp is unwired, never if the seed is wrong. | Measure it the way F-93 measures the convention — two identical synthesis calls must be bit-identical, and a known-answer carrier fingerprint pinned at the SHA — skipping without the render env. Or state at the constant that it is unfalsifiable and rests entirely on the `commit_sha` pin.
F-125 | falsifier | major | OPEN | src/amcd/simulators/gsound_sir.py:726-733; src/amcd/simulators/render.py:33-41, :148-216 | F-84's RAISE KILLS THE WHOLE BATCH, WITH NO PER-SCENE DROP PATH. `run_render`'s loop has no `try`/`except` and no drop record, so one silent leg at scene 500 of 720 propagates out, the stage sentinel is never written, and the entire multi-hour emulated render is redone from scratch. `render.py:36-41` states this exact cost as the reason `_preflight_separations` collects every offender instead of failing mid-loop — F-84 reintroduces the pattern that docstring exists to forbid, one stage later and after the expensive part rather than before it. There is also no `(unit, reason)` record for a scene that cannot be rendered, so the "log every drop" contract has nothing to log through. | Collect silent legs and fail after the loop with all of them named (the `_preflight_separations` shape), or record the scene as a logged drop with its reason and let the batch complete with the drop visible in the render stage's counts.
```

### What I could NOT break

- **`_retain`'s selection counts match upstream in all 17 edge cases probed**, including the four the brief named (all-zero energies, ties at the cut, `top_percent` 0/100, `max_rays` 0) and the ones the old code got right (empty set, boundary, `max_rays` > n, negative `max_rays`). F-85(b) is correct; see F-113/F-114 for what is unverified and what is overstated about it.
- **`PathData` round-trip is faithful.** All eight arrays, dtypes, shapes, the descriptor dict, a `None` `kept_energy_percentage` and an N=0 file all survive `to_parquet` -> `from_parquet` byte-for-value. F-95's round-trip claim holds.
- **F-90's "only what the stage wrote" fix is real.** `sorted(written)` is deterministic and the AppleDouble/`.DS_Store` guard is genuinely non-vacuous. Digests are stable across two independent dry-run renders (verified: identical `artifact_sha256`, identical full `meta.json`).
- **No platform branch, no hardcoded host path, no `isinstance(..., DryRunSimulator)`, no dry-run-keyed branch in any of the three package files.** `render.py` is `pathlib` throughout; `gsound_sir.py`'s only host seam is the `render_python` config value. The one platform coupling found is in the TEST file (F-120).
- **No leakage or split/seed surface in these files.** `SceneSpec.seed` is documented as drawn from the named `scene_generation` aspect (base.py:27-29); nothing in the render seam touches splits, normalization stats, or checkpoint selection.
- **Suite and pipeline reproduce the lane's claims**: `518 passed, 2 skipped in 64.10s`; the canonical dry run completes all nine stages.

---

## 7. Disposition of the self-check round — what was fixed, what stays OPEN

Sections 5 and 6 are the reviewers' own words, appended by them and left intact.
This section is the lane's response: **44 findings across three reviewers**
(18 readability, 11 acoustics, 15 falsifier). All are SELF-CHECK findings on an
unintegrated branch, not a clean pass.

**Nine were defects in work this lane did THIS CYCLE, and are fixed here.** Five
of those were regressions the lane introduced while fixing something else — the
same shape as the AppleDouble bug in section 2, and the reason a self-check pass
is worth its cost even though it cannot count as clean.

| finding | what was wrong | fix, with its guard |
|---|---|---|
| **F-111** / **AC-79** | F-84's silent-leg guard read the NATIVE energy while the FITTED array is what ships. A leg whose energy lay entirely past the window was trimmed to silence and shipped — the exact NaN-in-a-metric failure F-84 exists to prevent. Inert today ONLY because C6's 3.0 s cap keeps the trim branch dead, so C6's pending fix would have opened it | guard moved to `fitted_ir_total_energy`; BOTH energies stamped and named for the array each describes. `test_a_leg_trimmed_to_silence_is_refused_too` drives it through `render()` via a new `AMCD_STUB_TAIL_ONLY` stub switch. Mutation-checked |
| **AC-75** | **my new known-answer test asserted a value upstream cannot produce.** It pinned \|W\| to 1/√(4π); the late field is `sh[c] * carrier[t]`, so W is Y₀₀ times the noise-carrier sample. Measured by the reviewer against compiled upstream: 0.32336 vs 0.28209 asserted, ~3× the tolerance. **It passed only because it skips** — the first render-host run would have failed | the absolute-scale assertion is deleted, with the reason recorded at the site; `_SH_CONDON_SHORTLEY_PHASE`'s docstring corrected. Every sibling assertion is a RATIO, in which the carrier cancels exactly — those the reviewer confirmed correct against compiled upstream, at orders 1, 2 and 3 |
| **F-120** | F-91's rewritten test used `render_python=sys.executable` and depended on the pipeline env having no install receipt — an accident of THIS Mac. On a native x86_64 host, where `render_python: null` is the documented-correct setting, it would run a real 4.25 s render inside the unit suite and then fail DID NOT RAISE. A cross-platform violation introduced by this cycle | driven through the isolated `_stub_env` with a deliberately mismatched SHA — a guaranteed non-zero exit on every host |
| **F-121** | adding `host_scoped_params` to the `@runtime_checkable` Protocol broke `issubclass(DryRunSimulator, Simulator)`, which was True at `92939e3`. Protocol membership is structural; "optional" is not a thing it can express | removed from the Protocol, documented as duck-typed with `simulator_host_scoped_params` as the accessor that treats absence as "nothing to redact" |
| **F-113** | **section 2's claim that EVERY guard was mutation-checked was false.** F-85(b) — the selection change — had no test; the falsifier restored the `total > 0.0` guard and the suite stayed green | `test_a_zero_energy_path_set_selects_the_way_upstream_does` exercises the worker's `_retain` directly (the branch is unreachable through `render()`, since F-84 raises first). Mutation-checked. The claim in section 2 is corrected in place |
| **F-119** | the parent-side speed check was described as "defence in depth" and was dead: it re-tested a subset of an array the worker had already accepted at the same tolerance, and its empty branch was unreachable. Its two tests called it directly, holding coverage open while proving nothing about the render path | method and call deleted; the two tests repointed at the WORKER's check, which is the one that runs |
| **F-125** | F-84's raise aborted the whole batch — one silent leg at scene 500 of 720 loses the run with no sentinel. `_preflight_separations`' own docstring names that cost as the reason it collects offenders, so this reopened the pattern one stage later and after the expensive part | backend contract failures are collected per scene and reported together; the stage still FAILS (an incomplete dataset is not a success) but every other scene is on disk, so the re-run costs only the refused scenes |
| **AC-76** | the `_SYNTHESIS_CARRIER_SEED` docstring asserted that the carrier's metric error CANCELS in a paired comparison. Only half is established — the sequence is common-mode; the legs occupy different bins and weight it differently. The reviewer's model puts the residual at sd ~0.45 dB on paired C50, ~45 % of the declared JND | docstring split into WHAT IS ESTABLISHED / WHAT IS NOT, pointing at AC-76. **The claim was mine and I did not measure it** |
| **AC-77** / **F-124** | the same docstring said the stamp is falsifiable "where a diff can catch it". It is not: 42 is a literal in our own source, so provenance emits it whatever upstream does | corrected to state plainly that it rests on the `commit_sha` pin alone |

**Also fixed, from the same round:** F-114/AC-82 (the "reproduces upstream
EXACTLY" overclaim, narrowed to the same selection rule with a deterministic
tie-break, float64 accumulation and a stated non-negativity assumption);
F-112/AC-80/RR-100 (`ir_total_energy` renamed and given units);
F-115 (a filterbank one octave off passed every guard AND changed the IR —
`test_the_declared_centres_are_upstreams_compiled_set` now parses
`Context.cpp:8` from the pinned checkout and skips without it, mutation-checked);
RR-90 (contents map missing a whole class, and the row list presented as
exhaustive); RR-93/94/95/98 (comments retelling bug reports, per C12's rule);
RR-96 (`base.py`'s key list documented one backend's value and had ALREADY
diverged from `gsound_sir.py` after AC-57); RR-99 (43.1 % vs 43.2 % for one
measurement); RR-101 (`_STUB_CONSTANTS` renamed `_STUB_CONSTANTS_SRC`, the
docstring-demotion noted, last bare literal removed); RR-102 (four `SceneSpec`
fields RR-84 left undocumented); RR-103 (the render stage docstring named none of
its own artifacts); RR-105/RR-107 (this file: a pointer to a scratchpad outside
the repo, and an invented ledger status `CLOSED IN LANE`).

**RR-102 caught one more of my own errors.** Documenting `sim_params`, I wrote
that it is "merged over the simulator's config block at render time". It is not:
`grep` shows `scenes/generator.py` writes `{}` and NOTHING reads it. The field is
a reserved seam with no consumer, and it now says so. A false contract in a
docstring is worse than no docstring.

### Remaining OPEN, with a reason — not silently dropped

Everything below is anchored in sections 5 and 6 and stays OPEN for the
integrator's pass.

- **Needs the render env, which this lane pre-declared it would not use**:
  `AC-76`'s deciding experiment (zero renders, but it needs
  `spherical_harmonics_rt`), `AC-77`/`F-124`'s carrier falsifier, `AC-78`
  (extend the known-answer test to order 3 — production is order 3, the test is
  order 1; the reviewer supplied the predicted ratios, which I will not paste in
  as fact without running them).
- **Close WITH cluster C6**: `AC-79`'s "close with C6" clause, `AC-85` (the pad
  branch discloses nothing while the trim branch discloses its cost), and
  `RD-21`, already conditional.
- **Cross-lane or bigger than this lane**: `F-116` (the digests have no verifier —
  the cache-reuse path is `pipeline.py`, lane P's), `F-117` (parquet digests embed
  the pyarrow version, so they are a within-host integrity check and not a
  cross-host identity), `F-118` (a stale file inside a RETAINED scene dir survives
  pruning and is absent from provenance), `RR-91` (the scaffold-physics tests
  belong in their own file — same new-file constraint as RD-121/RR-83),
  `RR-104` (paper→code traceability), `RR-106` (this file's evidence blocks).
- **Grouped, awaiting one decision**: `F-110` + `F-122` — the lossy-cast question.
  The falsifier widened my own row with sharper probes (int64 `-5` → uint64
  `18446744073709551611` with NO warning; NaN → uint64 `0`, a valid-looking source
  index). It is a contract question for the multi-backend seam — raise vs log —
  and I am not deciding it inside a lane.
- **Small, deliberate**: `F-123` (`_RECEIPT_NAME`/`_RECEIPT_SHA_KEY` are dead; the
  worker inlines both literals — the comment now says so rather than implying they
  are load-bearing), `AC-81`/`AC-83` (direction-vector frame and sign conventions),
  `RR-92`/`RR-97` (remaining duplicated rationale; upstream `.cpp` citations that
  look like repo paths).

### What all three reviewers independently confirmed sound

Worth recording so it is not re-litigated. `_retain`'s selection counts match a
literal transcription of upstream in all 17 probed edge cases, so **F-85(b) is
correct**. The `PathData` round trip is faithful — eight arrays, dtypes, the
descriptor, a `None` share, and an N=0 file. F-90's "only what the stage wrote"
fix is real and digests are stable across independent runs. The ACN ordering, the
√3 N3D magnitude and the Condon-Shortley phase are correct against compiled
upstream at orders 1, 2 AND 3, and AC-62/AC-63's corrected semantics are right at
source. No platform branch, hardcoded host path or scaffold-keyed branch exists in
any of the three package files — the one platform coupling found was in the test
file, and is F-120 above. No leakage, split or seed surface is touched.
