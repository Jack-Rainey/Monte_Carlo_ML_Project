# Lane P inbox — cycle5

Branch `lane/P-cycle5`. Written by lane P, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/P-cycle5", never "clean".

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

## READ FIRST — 35 of this lane's 40 rows were ALREADY FIXED and never re-verified

Lane P was assigned 40 rows. **37 of them were fixed on `lane/P-cycle4`, merged
into `v3-rebuild`, and never re-review-confirmed.** `docs/ledger_inbox/archive/
cycle4-P.md` says so in its own words — "CLOSED", "FIXED IN THIS BRANCH … Delete
on confirmation", "ADDRESSED".

They are in substance **fix applied, awaiting re-review** rows, which
`docs/parallel_protocol.md` planning step 3 forbids assigning to any lane
("assigning one invites a second fix stacked on a first that nobody checked").

**This is not a lane discovery — it is already OPEN as `RD-146`**, which states
that the cycle-5 partition put every row under `integrator_queue:` and zeroed
`awaiting_re_review:`, emptying the exact distinction step 3 depends on. The
backlog-verification pass (`3498456`) reached 53 rows and left
`awaiting_re_review: [RR-24]`; **it did not reach these 35.**

So this lane's deliverable is VERIFICATION WITH EVIDENCE, not a second round of
fixes — the clause in CLAUDE.md's definition of done that cycle 4 failed:
*zero rows sit in "fix applied, awaiting re-review", because a fix nobody
re-derived is a claim.*

### Result — the arithmetic, stated once so it can be checked rather than recounted

**37 of the 40 assigned rows were claimed CLOSED/ADDRESSED in cycle 4. 35 held; 2
were PARTIAL.** Full decomposition of all 40, disjoint:

| bucket | n | rows |
|---|---|---|
| verified **CONFIRMED FIXED** | **35** | the id list at the end of this file |
| verified **PARTIAL**, completed here | **2** | RR-56, AC-48 |
| genuinely open, fixed here | **1** | F-M11 |
| **escalated**, not fixed | **2** | RD-20, F-102 |
| | **40** | |

Three of the 35 are **CONDITIONAL ON MERGE** — see RD-206.

**Self-check reviewers then found 26 further findings, 12 of which are defects in
the code this lane wrote this cycle. Those 12 are fixed; the rest are recorded
OPEN.** Two of the 26 corrected false statements in THIS FILE (F-169, RR-152) —
both are fixed above and below, and the fact that a permanent record needed
correcting is itself the argument for the verification pass.

### Gate declaration (planning step 1b — stated, not implied)

**Lane P neither LIFTS nor UNBLOCKS RD-33a.** Condition (i)'s path list is
`src/amcd/scenes/**`, `src/amcd/evaluation/**`, `config.py` *split handling*,
`configs/*.yaml` split declarations. **No lane-P row is anchored there** — P's
`config.py` rows are the config-root and banner/`_check` rows, not split handling
— so under RD-128's severity scoping this lane clears **zero of the 20**. What it
advances is CLAUDE.md's definition-of-done clause. Recorded so cycle 6 is not
planned as though the gate had moved (RD-89c).

### Pass condition — declared effect on `ci_table.csv`: NONE. Held.

Declared before starting (RD-91/RD-149). Measured on a **FRESH run dir**, because
this lane edits `data/**` (in preprocess's code scope) and `reporting/` (in
report's), so a reused run_dir would be partially invalidated by construction and
"byte-identical" would prove nothing (RD-204):

Both legs are the canonical dry run, same command, different `--run-dir`, taken
at `92939e3` (A) and at this branch's tip (B). Reproducible as written:

```
P=/Volumes/T7/Monte_Carlo_Research/v3-lane-P
C="-c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml"

# A leg — before any lane-P edit, at v3-rebuild (92939e3)
PYTHONPATH=$P/src .../bin/amcd all $C -r $P/experiments/all_20260811_180530
# B leg — after, on a FRESH run dir (RD-204)
PYTHONPATH=$P/src .../bin/amcd all $C -r $P/experiments/bleg_final

shasum -a 256 <each>/stats/ci_table.csv
  74651cd26663fcc911979d9a7b9ddd8d97433ef4376fd829d35e6277d6f23052   BOTH LEGS
diff  <A>/stats/ci_table.csv <B>/stats/ci_table.csv   → exit 0, byte-identical
```

The A leg is genuinely pre-fix — its `report/summary.txt` has no `Unit` column and
the B leg's does — so the identical hash is not a stale-artifact artefact. The
falsifier independently re-derived both hashes.

Suite: **511 passed, 1 skipped** (baseline 506 + 5 new), zero failures.
`git merge v3-rebuild` → "Already up to date", so the evidence needs no re-run.

---

## Verification method

Every probe patched a **scratch copy of the package** under the session
scratchpad, never the worktree — which is how `representations/spectrogram.py`
(lane M) and `simulators/dry_run.py` (lane M) were patched without touching them.
`git status` stayed clean throughout.

**Control first**, so a refusal means the probe and not the harness: the
unpatched scratch copy on a copy of the baseline run_dir gives **9× `[skip]`,
EXIT=0** with no warnings. The copy sits at a different absolute path and hashes
identically, confirming `code_version` is path-relative.

---

## Rows verified — CONFIRMED FIXED (35)

### A1 · cache holes — F-63, F-64, F-66, RD-66

Each patch is an **uncommitted** working-tree edit, which is the state
`git rev-parse HEAD` cannot see — so RD-66/F-55's content-hash premise is
exercised by all four probes at once.

**F-63** (`stats` had no `code_version`, `report` no fingerprint) — patched
`bootstrap_ci`'s `ci_lower` × 0.001:

```
[FAIL] stats: Stage 'stats' was cached under a DIFFERENT config
    code_version: '55e13636…' → 'f848207f…'
  gen-scenes…eval all [skip], EXIT=1
```

and patching `reporting/tables.py` refuses **`report`** with gen-scenes…**stats**
cached — the second half, which is `_report_fingerprint` + `STAGE_UPSTREAM["report"]`.

**F-64** (`preprocess` had no `code_version`) — patched `spectrogram.encode` (×4):

```
[FAIL] preprocess: … code_version: '81795321…' → '3ad38d37…'
  gen-scenes, render [skip]
```

The refusal names **`preprocess`**, the stage whose artifacts are actually stale.
The original defect refused `train`, so following the message rebuilt the wrong
thing. That is the load-bearing half and it holds.

**F-66** (`eval`/`infer` omitted `data`) — patched `data.normalization.denormalize`
(+10 dB); which scopes move is the finding:

| stage | moved? |
|---|---|
| `preprocess`, `train`, `infer`, `eval` | **YES** — every stage declaring `data` |
| `stats`, `report` | **no** — neither denormalizes |

The scopes are precise, not blanket. Both halves of the row hold.

### A2 · host independence — F-69, F-69-B4, F-79

**F-69** — injected `._probe.py` and `._probe2.py` into `evaluation/` and `data/`
(scoped subpackages): all six stages' `code_version` **bit-identical**
(`diff` exit 0). The cache key describes the source, not the host.

**F-79** — both triggers:
* scope entry naming a `.py`-less directory → `ValueError` naming the entry and
  saying it "would contribute nothing to the hash";
* package copied under an ancestor directory literally named `__pycache__` → the
  six hashes are unchanged and are **five distinct values of six**, not collapsed
  to one constant. The relative-path match is doing its job.
  (Corrected from "six distinct" after the falsifier re-derived it — F-169.
  `train` and `infer` declare the identical scope tuple at `pipeline.py:270,273`,
  so they are equal by construction in every configuration. F-79's substance —
  no collapse to a constant — is unaffected.)

**F-69-B4** (intermittent in ~half of full-suite runs on this host) — the four
host/code-version tests pass in isolation, **and three consecutive randomized
full suites passed 506/506**. Intermittency gone.

### A3 · config layout — F-73, F-80

**F-73** reproduced live and unplanned: the scratch package without a sibling
`configs/` failed at load with a message naming **every** location tried and what
would fix it — not a bare `FileNotFoundError` from inside `_merge_yaml`.

**F-80** — a configs root holding `base.yaml` but missing the plugin dirs:

```
FileNotFoundError: amcd found `base.yaml` in … but not the plugin parameter
directory ['representations']. … a missing directory is indistinguishable from a
parameter-free plugin, so the run would load with empty params and fail later
without naming this.
```

Refused at load, rather than a VALIDATED `Config` with empty plugin params.

### A4/A5 · device and legacy sentinel — F-74, F-76

**F-74** — `versions.json` from the canonical run carries `device: mps` and
`platform_machine: arm64`; and a sweep of every fingerprint payload for
`device`/`mps`/`cuda`/`arm64`/`platform` returns **NONE**, so a device change
cannot discard an expensive checkpoint.

**F-76** — forged a legacy `{"fingerprint": null}` sentinel: raises the actionable
"predates fingerprinted caching for this stage … `--force` … or a fresh
`--run-dir`", a `RuntimeError`, **not** the bare `TypeError` from `set(None)`.

### A6 · the guards, MUTATION-CHECKED — F-77, F-78, RD-103, RD-105

A green suite is not evidence a guard works; cycle 4 found F-77 exactly this way.

**F-77** — removed `"data"` from `STAGE_CODE_SCOPE["eval"]` in the scratch copy:
**3 tests failed**, including the behavioural `test_patching_denormalize_moves_
eval_and_infer`. The walker really does resolve the edges it claims to.

**F-78** — removed `seed_split_assignment` from `_preprocess_fingerprint`:

```
FAILED …::test_perturbing_each_named_seed_invalidates_at_least_one_stage[split_assignment]
AssertionError: seeds.split_assignment moved no stage fingerprint.
```

The per-aspect sweep catches the **leakage-critical** seed by name — the hole the
old master-seed-only probe hid.

**RD-103** — the limit is stated, not implied: the test docstring says coverage is
proved at TOP-LEVEL `Config` field granularity, names which nested models are
dumped wholesale, and flags `Seeds` as the exception swept separately.

**RD-105** — `STAGE_CODE_SCOPE["preprocess"]` names `simulators/base.py` with the
single-module rationale, and the closure test states what it checks and what it
cannot. **Conditional on merge — see RD-206.**

### A7/A8 · declarations and reported tables (verified by reading, claims counted)

| Row | Verdict |
|---|---|
| RD-101 | CONFIRMED — `STAGE_FINGERPRINT`/`STAGE_UPSTREAM` describe the current code; `diagnostics` is named as the only `None`, `report` as chained |
| RD-102 | CONFIRMED — every exemption states a re-entry condition, and `tests/test_config.py:290-293` enforces the "Non-exempt"/"fingerprinted through" marker |
| RD-106 | CONFIRMED — `code_version(("stats","evaluation"))` and `(("reporting",))` are in place, so the integrator's column additions will invalidate cached `stats`/`report` |
| AC-47 | CONFIRMED — eval's scope rationale is now the true one (registry-resolved, invisible to the closure test), not the false "eval decodes" |
| RR-35 | CONFIRMED — `# ── Plugin blocks + layer merge ──` banner present; `_check_shift_splits` extracted |
| RR-36 | CONFIRMED — "blind to" appears **once** in `pipeline.py` (at `STAGE_CODE_SCOPE`); `provenance.py`'s single hit is the different `git rev-parse` claim, not a duplicate |
| RR-42 | CONFIRMED — `ALL_SOURCES`' dot is explained at the constant; the scoping rationale lives in `code_version` and is cited from `STAGE_CODE_SCOPE` |
| RR-46 | CONFIRMED — the four transcripts are cut to rule + ledger id |
| RR-47 | CONFIRMED — F-53 appears 3× in `pipeline.py`: narrated **once** at `STAGE_UPSTREAM:344`, cited at `:125` ("see STAGE_UPSTREAM, which owns that story") and `:154` |
| RR-48 | CONFIRMED — test-class docstrings trimmed; the "WHAT THIS TEST CHECKS / CANNOT CHECK" contract paragraphs kept |
| RR-49 | CONFIRMED — `code_version`'s last paragraph points at the test instead of restating it |
| RR-50 | CONFIRMED — `_DIAGNOSTICS_EXEMPTION` sits above the `#:` block, which documents the dict it precedes |
| RR-51 | CONFIRMED — the "Non-exempt" token contract is stated in the table header |
| RR-52 | CONFIRMED — the three absent stages are declared absent BY DECISION, citing RD-107/RD-108 (the remap reached these citations) |
| RR-53 | CONFIRMED — `TestTheTableProducingStagesAreCacheProtected` exists |
| RR-54 | CONFIRMED — banner is a bare title, matching the other three |
| RR-55 | CONFIRMED — `Config._check` is **eight named calls** and one ordering comment, nothing else |
| RR-57 | CONFIRMED — `select_device` annotated `-> "torch.device"`, lazy import preserved |
| RR-59 | CONFIRMED — module docstring lists the six failure families, no stale row enumeration |
| RR-68 | CONFIRMED — `GeometryFamily.characterization` documents `sabine`/`none` and the roadmap reason the seam exists |

---

## PARTIAL, and fixed here — RR-56

**NOT FIXED as written.** RR-56 asked that ~15 lines in `Config.stamp` re-telling
`provenance.py`'s docstrings be cut to one line each. The git/device comments were
trimmed, but the `code_version` comment had regrown to **13 lines**, of which ~5
restate `provenance.py`'s whole-package-vs-scoped distinction — the same
duplication, by the same standard, one cycle later.

Fixed: the restatement is now a citation to `amcd.provenance`; the F-75 content
(this stamp describes the INVOCATION, not the artifacts) is **kept**, because that
is not in `provenance.py` and is the load-bearing half.

## Fixes applied (3)

**AC-48** — second half. The threshold VALUE was already rendered (first half
fixed in cycle 4); the `Imp mean`/CI/MDES columns still carried **no unit** while
their rows mix seconds and decibels.

The unit **cannot** come from `kind` (RD-201, verified): `kind` is
`match_reference|maximize|minimize`, and `T30` and `C50` share `match_reference`
while differing in unit. So `_METRIC_UNITS` is declared per metric and `_unit_for`
**RAISES** on an unlisted one — a blank unit beside a physical quantity is the
silent exclusion the drop log exists to prevent. `energy_mse` is an
**operand-domain** MSE (`evaluation/signal.py`), so its unit is not a fixed
string: it is resolved from the **preprocess-stamped** `value_domain`, never
inferred from a rep class (F-19).

```
C50            3/3   11.4768   -0.6738 [-1.1648, -0.4035]   1.3903 dB     0.0% (0/3)
EDT            3/3    0.4083   -0.0333 [-0.0573, -0.0011]   0.1193 s      0.0% (0/3)  1 high-variance
T30            3/3    0.2925   -0.0118 [-0.0212, -0.0068]   0.0267 s      0.0% (0/3)
energy_mse     3/3    4.7987   -4.7736 [-5.6727, -4.2481]   2.5535 dB²    0.0% (0/3)
energy_snr_db  3/3    5.8982  -47.8947 [-53.1556,-43.0012] 16.6047 dB     0.0% (0/3)
```

Four new tests in `tests/test_report.py`, **mutation-checked**: reverting
`_unit_for` to the pre-fix "render blank, never raise" makes
`test_a_metric_with_no_declared_unit_is_refused_by_name` fail
("DID NOT RAISE ValueError"). `summary.txt` only — **not** `ci_table.csv`, which
is why the declared "none" still holds. The durable form is filed as RD-201.

**F-M11** — confirmed live: `low_mean`/`low_std` are computed and stamped into
`preprocessed/meta.json` and applied to nothing (both legs use the HIGH stats —
one affine frame, F-02). Not leakage.

**Kept, not deleted**, and the reason is now in the source AND the artifact: they
are the only on-disk record of how the LOW-RAY leg's distribution differs from the
high leg — the axis the roadmap's ray-count sweep (paper §6) varies, and the
evidence behind F-02's own framing decision. Deleting them to satisfy "emits
output, contributes nothing" would delete the evidence for a decision.

`meta.json` now carries a sibling `norm_stats_applied` string saying which of the
four were applied. **The keys themselves could not be renamed**:
`tests/test_invariants.py:113` asserts `low_mean`/`low_std`/`high_mean`/`high_std`
verbatim and is **lane S's file**.

**RR-56** — above.

---

## NOT DONE, deliberately — 2 escalations

**RD-20 — MIS-ASSIGNED TO THIS LANE. Not attempted.** Its resolution puts
`RunContext` in `src/amcd/runtime.py`, which is **lane B's file**, and changes the
dispatch signature across all nine stage entry points spanning lanes M, S and B.
`docs/lanes/cycle5.yaml:176` declares `fix: [src/amcd/pipeline.py]` — not where
the fix lands, which is how it passed the reachability check. **No lane-P-only
subset exists**: `RunContext` cannot be defined outside `runtime.py`, and a
dispatch-signature change is atomic across nine call sites. See RD-208.

**F-102 — REPRODUCED, deliberately NOT implemented.** See RD-200: the ancestor
walk it asks for is dead code the moment RD-107 lands, and **RD-107 is already
decided** ("USER DECISION 2026-08-11: WIRE BOTH"). Its own resolution says "fold
into RD-107's implementation". The reproduction, which the integrator needs:

```
render backend changed (byte edit to simulators/dry_run.py), same run_dir:
  amcd all            → 3 × [warn ] (gen-scenes, render, diagnostics)
  amcd report --force → 0 × [warn ]
```

The reported table is regenerated from renders that predate the change, in
silence. Confirmed: the warning is invocation-scoped, exactly as F-102 states.

---

## New findings

```
| RD-200 | research-director (lane P) | major | OPEN | src/amcd/pipeline.py:657-709 `_warn_if_unprotected_and_stale`, :354-364 STAGE_UPSTREAM | F-102's ancestor walk is dead code the moment RD-107 lands, and RD-107 is DECIDED (ledger:316 "WIRE BOTH"). :680-682 early-returns for any stage carrying code_version, which RD-107 gives gen-scenes and render; the only remaining unprotected stage is diagnostics, whose STAGE_UPSTREAM is None, so it is never any stage's ancestor. Its regression test would also need an unprotected-ancestor fixture — the pattern RD-129 flags as institutionalising AC-45's hole in a test file. Disclosure-vs-refusal reasoning is sound; the SEQUENCING is not. | Fold into RD-107 (serial queue). Reproduction transcript is in docs/ledger_inbox/P.md. If implemented in-lane later: derive the fixture stage from STAGE_FINGERPRINT declaration, never hardcode diagnostics/render. |
| RD-201 | research-director (lane P) | major | OPEN | src/amcd/reporting/tables.py `_METRIC_UNITS`; src/amcd/evaluation/metric_row.py:67 (MetricTriple.kind) | THE DURABLE FIX FOR AC-48 SPANS TWO LANES. A unit cannot be derived from `kind` (match_reference|maximize|minimize) — T30 and C50 share a kind and differ in unit — so lane P declared a metric->unit table in tables.py that RAISES on an unlisted metric. That is correct but it is a SECOND declaration site: the unit belongs beside `kind` on the metric itself, in evaluation/ (lane M), carried through stats/aggregate.py into ci_table.csv. Carrying it would MOVE ci_table.csv, which no lane may do outside the metric lane (rule 2). | Declare `unit` beside `kind` on MetricTriple and carry it to ci_table.csv, as an integrator-queue change with the ci_table movement expected and declared in advance. Until then a new metric fails loud at report (by design). |
| RD-202 | research-director (lane P) | minor | OPEN | docs/lanes/cycle5.yaml:174-214, :344-345; ledger RD-146 | The partition assigned lane P 35 rows that are FIX APPLIED, awaiting re-review, which planning step 3 forbids. This is RD-146's already-filed defect (the partition zeroed `awaiting_re_review:`), not a new one — recorded here so the two are linked rather than duplicated, and so RD-146's remedy is executed rather than restated. `raised_against_this_partition:` is `[]`. | Repopulate `awaiting_re_review:` from the EXPLICIT id list in docs/ledger_inbox/P.md, and record lane P as raised-against-the-partition for cycle 6's accounting (RD-126/RD-142/F-103). |
| RD-203 | research-director (lane P) | minor | OPEN | docs/lanes/cycle5.yaml lane P rows | ROW COUNT IS NOT A WORK ESTIMATE, and lane P is the extreme case: 40 assigned rows collapse to 3 fixes + 2 escalations + 35 verifications. Reporting 40 as the workload misstates both the effort and what the cycle bought. | Record the disjoint partition (35 verify / 3 fix / 2 escalate) in the cycle-5 accounting rather than the raw count. |
| RD-204 | research-director (lane P) | minor | OPEN | docs/lanes/cycle5-P.md "Pass condition"; src/amcd/pipeline.py:267, :287 | A LANE'S ci_table A/B MUST USE A FRESH run_dir WHEN THE LANE EDITS ANY SCOPED SUBPACKAGE. Lane P edits data/** (preprocess's scope) and reporting/ (report's), so a reused run_dir is invalidated by construction and a "byte-identical" result measured off a partially cached run proves nothing. Lane P used a fresh dir; the brief does not require it. | Add "measure the B leg on a fresh --run-dir, and say which" to the pass-condition text in every lane brief. |
| RD-205 | research-director (lane P) | minor | OPEN | docs/lanes/cycle5-P.md; ledger RD-33a, RD-128 | THE 1b DECLARATION WAS MISSING FROM THE BRIEF. No lane-P row is anchored on RD-33a(i)'s path list, so this lane clears zero of the 20 and moves the gate neither way. The brief's pass condition says nothing about which gate conditions the lane lifts or unblocks, which is how a cycle gets planned as though a gate had moved (RD-89c). | Lane P LIFTS nothing and UNBLOCKS nothing on RD-33a; it advances CLAUDE.md's definition-of-done clause. Require the 1b declaration in every generated brief, not just the partition header. |
| RD-206 | research-director (lane P) | minor | OPEN | src/amcd/pipeline.py STAGE_CODE_SCOPE; tests/test_stage_cache.py TestDeclaredScopeCoversWhatTheStageImports | THREE VERDICTS ARE CONDITIONAL ON MERGE. F-66, F-77 and RD-105 are import-closure claims measured on a tree lanes M and B have not merged into. RD-105 states that one new import in either makes P's declared scope insufficient and fails P's OWN test on the integrated tree, after P's pass condition was measured green here. | Re-derive F-66 / F-77 / RD-105 at integration gate step 3 before deleting them. The remedy for a red closure test is a one-line scope update in pipeline.py — NEVER a weakened assertion, because a red test means a real undeclared dependency, which is F-66 itself. |
| RD-207 | research-director (lane P) | minor | OPEN | src/amcd/data/normalization.py; src/amcd/data/preprocess.py `norm_stats_applied` | F-M11's KEEP DECISION IS RIGHT BUT THE ARTIFACT COULD NOT BE MADE FULLY SELF-DESCRIBING. low_mean/low_std serve a documented roadmap item (ray-count sweep, paper §6) and are the evidence behind F-02's single-affine-frame framing, so they are kept. The KEYS could not be renamed to say they are unapplied: tests/test_invariants.py:113 asserts all four names verbatim and is lane S's file. A sibling `norm_stats_applied` string carries the disclosure instead. | Either accept the sibling key as the durable form, or rename the keys together with lane S's assertion in one integrator change. |
| RD-208 | research-director (lane P) | major | OPEN | docs/lanes/cycle5.yaml:176 vs ledger RD-20 anchor; ledger RD-111 | RD-20 IS A RULE-4 SPANNING ROW ASSIGNED TO A SINGLE LANE. Its anchor spans pipeline.py (P), runtime.py (B) and cli.py (integrator-owned); its resolution puts RunContext in runtime.py. The declared `fix: [src/amcd/pipeline.py]` is not where the fix lands, which is how it passed the reachability check — RD-111 already identified that the partition test validates `fix:` paths against `owns` but never against the row's own ANCHOR. No lane-P-only subset exists. | Move RD-20 to the integrator queue and schedule it WITH RD-104 (device.py split, ITEM 4): RD-20's own resolution asks for it to land while the dispatch signature is already being touched. Fix the partition test to check the ANCHOR, not only the declared fix paths (RD-111). |
| F-160 | builder (lane P) | major | OPEN | docs/parallel_protocol.md:82-85; tests/test_lane_partition.py | A GUARD THE PROTOCOL CLAIMS EXISTS DOES NOT EXIST. parallel_protocol.md states that `tests/test_lane_partition.py` asserts "inbox -> ledger coverage — every `\| ID \|` in an inbox findings block must be an OPEN ledger row or explicitly folded into one", and gives the reason: without it "a finding missing from BOTH sides passes silently — which is how cycle 4's fold lost five rows including a blocker (RD-142)". MEASURED: the file has nine tests — two-lane paths, shared-authority, inbox distinctness, row-fixable-in-lane, no-row-id-in-two-places, no-duplicate-ledger-id, disjoint id blocks, partition-covers-ledger, briefs-exist. NONE of them opens an inbox file. The only asserted identity is ledger <-> partition, exactly the gap the passage says is closed. Same class as F-66/F-77 — documentation claiming more than the test checks — on the guard that exists to stop silent row loss at the fold. | Implement the inbox->ledger coverage test, or correct parallel_protocol.md to say the check is manual. Do NOT leave the claim standing: this cycle's fold is the first to depend on it, and lane P alone is handing the integrator 10 new rows. |
```

## The explicit `awaiting_re_review:` id list (RD-146's remedy)

These 35 lane-P rows are CONFIRMED FIXED by this lane's evidence above and are
ready for deletion at gate step 7 — **except the three marked conditional**:

```
RD-66 RD-101 RD-102 RD-103 RD-106
F-63 F-64 F-69 F-69-B4 F-73 F-74 F-76 F-78 F-79 F-80
AC-47
RR-35 RR-36 RR-42 RR-46 RR-47 RR-48 RR-49 RR-50 RR-51 RR-52
RR-53 RR-54 RR-55 RR-57 RR-59 RR-68
CONDITIONAL ON MERGE (re-derive at gate step 3): F-66 F-77 RD-105
```

RR-56 was PARTIAL and is fixed on this branch — it needs re-review like any fix,
not deletion on this lane's say-so. AC-48 and F-M11 likewise.

## Reviewers

`research-director` ran **on the plan, before implementation**, per the
implementation loop. It raised RD-200…RD-208, corrected two items of substance
(F-102 should not be implemented; the AC-48 unit cannot come from `kind`), and all
nine were folded into the work rather than deferred. F-160 is the builder's.

`falsifier`, `acoustics-reviewer` and `readability-reviewer` then ran over the
CURRENT state of the branch. **This is a self-check on `lane/P-cycle5`, NOT a
clean pass** (rule 5) — only the integrator, on the merged tree, can produce one.

They raised **26 findings**. All three independently confirmed the central claim
(the 37 rows were fixed in cycle 4 and merged unverified) and the pass condition;
the falsifier re-derived F-63/F-64/F-66/F-74/F-76/F-77/F-78/F-79 with its own
probes and agreed row for row, and the acoustics-reviewer re-derived every unit in
`_METRIC_UNITS` against known-answer signals and confirmed all five.

### Fixed in response (12) — these were defects in the code THIS lane wrote

* **F-163 / AC-130** (raised independently by two agents — corroboration, not
  noise). My AC-48 guard was **data-dependent**: `_metric_row` returns the
  `unscored` line before it needs a unit, so an undeclared metric that happened to
  be unscored rendered at exit 0 and would crash the report on a later run that
  scored it. Units are now resolved **up front over the metric SET**, before any
  row is formatted. New regression test; the old lazy form fails it.
* **F-169** — this file said F-79 gave "six distinct values"; it is **five of
  six**. Corrected in place above.
* **RR-152** — this file's headline count (34) disagreed with its own id list
  (35). The full disjoint arithmetic is now stated once, near the top.
* **RR-140** — `normalization.py` claimed the stats are "stamped under a key that
  says they are unapplied". They are not — the disclosure is a *sibling* key,
  because the four names are asserted verbatim by lane S's test. My own docstring
  over-claiming about my own fix is the F-66 class, so it is corrected, not left.
* **F-165 / RR-151** — `norm_stats_applied` was a hardcoded string asserting which
  stats were applied, tied to nothing. It is now **derived from
  `_APPLIED_STAT_KEYS`, the same constant the `normalize()` calls read**, so the
  artifact cannot claim a normalization the tensors did not get, and its value is
  structured rather than prose.
* **AC-125** — `amp²` invented a unit and reads as ampere-squared. Now `a.u.²`,
  with the reason recorded; the amplitude domain has no declared unit.
* **AC-126** — the legend said the table mixes "seconds and decibels" while the
  table now also renders `dB²`. The legend names the third unit and says what it
  is: a mean SQUARED level difference, not decibels.
* **AC-128** — one Unit column labelled three different quantities. The legend now
  says what the unit is OF, per kind, and that a negative `Imp mean` means the
  error GREW — which is what the canonical run actually shows.
* **F-164** — a missing `value_domain` key gave a bare `KeyError`; now the same
  shaped, actionable message as the missing-file branch.
* **F-168** — `test_git_is_resolved_from_the_package_not_the_run_dir` asserted
  `git_sha() != "unavailable"` because "this checkout is a git repo" — a property
  of THIS HOST, contradicting `git_sha`'s own contract and the project's
  second-host requirement. Now asserts the contract (40-hex OR "unavailable") plus
  package-not-run_dir resolution.
* **RR-144** — two disagreeing statements of which columns the Unit labels; the
  footer is now the only one.
* **RR-147 / RR-148** — function-local `import pytest` hoisted; module docstring
  names the AC-48 contract it now owns.

### Recorded OPEN, deliberately not fixed here

**F-161 is the one to read.** The falsifier's most consequential finding, and I am
not fixing it at the end of a session: `code_version` hashes raw file BYTES, so a
comment-only edit invalidates every fingerprinted stage. Measured this cycle —
`stats`' key moved although nothing in `stats/` or `evaluation/` changed, purely
because RR-56 trimmed a comment in `config.py` (which is in `_CORE_SOURCES`, i.e.
every scope). Harmless while the dry run takes seconds; the moment **RD-107
(DECIDED, "WIRE BOTH")** gives `render` a `code_version`, a comment edit forces the
multi-hour emulated re-render — and that is exactly the "teaches the operator to
reach for `--force`" compliance failure `provenance.code_version`'s own docstring
gives as the reason for scoping. **It should be decided together with RD-107, not
before it.**

The rest, each a real finding with a concrete anchor, listed in the block below:
F-162, F-166, F-167, AC-127, AC-129, AC-131, AC-132, RR-141, RR-142, RR-143,
RR-145, RR-146, RR-149, RR-150, RR-153, RR-154.

Two deserve the integrator's attention specifically:

* **F-167** — F-76's legacy-sentinel guard reached `_is_done` but **not the
  upstream leg** of `_effective_fingerprint`, which still says a stage "has not
  completed" and its artifacts "do not exist" when both are false. Live population:
  any run_dir predating F-63. F-76's verdict stands **for its own row text**; this
  is the adjacent half, and it is in a file this lane owns — it is recorded rather
  than fixed only because it is new work found after the pass condition was
  measured, and an unverified fix is what this whole cycle exists to stop.
* **AC-129** — the unit disclosure is half-applied *within the file I changed*:
  `report/metrics_table.csv` still carries 21 bare float columns mixing s, dB and
  dB². Not fixed here because adding the column MOVES that artifact, and this lane
  declared its expected effect in advance; re-declaring it silently at the end
  would defeat the interference detector. It needs a stated re-declaration.

```
| F-161 | falsifier (lane P) | major | OPEN | src/amcd/provenance.py:130-145 `code_version`, :46-49 `_CORE_SOURCES`; src/amcd/config.py stamp comment | A COMMENT-ONLY EDIT INVALIDATES EVERY FINGERPRINTED STAGE, AND THIS CYCLE IS THE PROOF. `code_version` hashes raw bytes (`digest.update(path.read_bytes())`), so a comment is indistinguishable from a semantic change, and config.py is in _CORE_SOURCES i.e. every scope. MEASURED v3-rebuild vs lane/P-cycle5: all six stage code_versions moved; `stats` (scope ("stats","evaluation")) moved although NOTHING in stats/ or evaluation/ changed — solely because RR-56 trimmed a comment in config.py. provenance.code_version:118-122 argues for scoping because "a guard that refuses a cached stage for reasons the operator can see are irrelevant teaches them to reach for --force, which disables the guard entirely." Comment granularity is that failure one level down. Bounded only while the dry run is seconds: once RD-107 (DECIDED) gives render a code_version, a comment edit forces the multi-hour emulated re-render. | DECIDE WITH RD-107, not separately. Either hash semantics (e.g. `ast.dump(ast.parse(src))` with docstrings stripped) or price the forced re-render into RD-107 explicitly. TEST: code_version is invariant to an inserted comment and moves on any executable change; then re-run the F-63/F-64/F-66/F-77 mutation probes — all four must still refuse. |
| F-162 | falsifier (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_stamped_value_domain`; src/amcd/pipeline.py:221-232 `_report_fingerprint`, :287 STAGE_CODE_SCOPE["report"] | REPORT READS A PREPROCESS ARTIFACT IT DOES NOT DECLARE. PROBE: copy a complete run_dir, flip preprocessed/meta.json value_domain db->amplitude, `amcd report --force` -> EXIT 0 with every energy_mse row relabelled over numbers computed in dB, while energy_snr_db stays SCORED and labelled dB — impossible under amplitude, where evaluation/signal.py NaNs both SNR legs. THE REALISTIC ROUTE IS PROTECTED (verified: a config-driven rep change is refused through five chain links), so this is an undeclared dependency covered incidentally by chain transitivity, not a live wrong number. | Declare preprocessed/meta.json as an input of `report`, and cross-check the stamp against the rows being labelled (a scored energy_snr_db row is impossible under amplitude). |
| F-166 | falsifier (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_metric_row`, footer | THE TABLE NOW DISCLOSES THE UNIT AND STILL NOT THE KIND. `kind` decides whether Imp mean is pred-low (maximize) or |low-high|-|pred-high| (match_reference). In the canonical run energy_snr_db (-47.8947, maximize) and C50 (-0.6738, match_reference) sit in one column both labelled dB. `kind` IS already carried in stats/summary.json, ci_table.csv and metrics_table.csv — it is dropped exactly at the human-readable artifact. Partially mitigated this cycle by the AC-128 legend text, which states the two meanings but does not say per row which applies. | Add a Kind column; the value is already in the row dict, so this is report-local and does not move ci_table.csv. |
| F-167 | falsifier (lane P) | minor | OPEN | src/amcd/pipeline.py:558-565 (`_effective_fingerprint` upstream leg) vs :606-620 (`_is_done`) | F-76's GUARD WAS APPLIED TO ONE OF THE TWO LEGS. PROBE: forge {"fingerprint": null} into stages/stats.done of a populated run_dir, `amcd report --force` -> "Stage 'report' depends on 'stats', which has not completed in <dir> (no readable fingerprinted sentinel) … running 'report' now would record a provenance chain for artifacts that do not exist." Both claims are FALSE: stats/ is fully populated and the stage DID complete; only its sentinel predates fingerprinting. `_is_done` handles the identical shape correctly and its own comment says the case is guarded "here rather than at the call site" because it "recurs for EVERY stage that gains a fingerprint later (diagnostics next, RD-108/AC-45)" — the upstream call site is the one it did not reach. Live population: any run_dir predating F-63. | Route `_recorded_fingerprint() is None` on the upstream leg to the same actionable message as `_is_done`, distinguishing "never ran" from "ran before fingerprinting". Lane-P-owned file; found after this cycle's pass condition was measured, so recorded rather than fixed unverified. |
| F-168 | falsifier (lane P) | minor | OPEN | tests/test_stage_cache.py `test_git_is_resolved_from_the_package_not_the_run_dir` | FIXED THIS CYCLE — recorded for the audit trail. The test asserted git_sha() != "unavailable" with the reason "this checkout is a git repo", a property of one HOST, contradicting provenance.git_sha's stated contract ("allowed to be unavailable") and its own sibling test. Measured: an exported source copy outside a checkout gave 1 failed, 103 passed on this file. Same class as a platform-keyed branch, in a test file, against CLAUDE.md's second-host requirement. | Now asserts the CONTRACT (40-hex sha OR "unavailable") plus package-not-run_dir resolution. Awaiting re-review. |
| F-169 | falsifier (lane P) | minor | OPEN | docs/ledger_inbox/P.md (F-79 evidence, section A2) | FIXED THIS CYCLE — recorded because it is a correction to a PERMANENT record. P.md claimed F-79's probe left "six distinct values"; it is FIVE of six — train and infer declare the identical scope tuple (pipeline.py:270,273) and are equal by construction. F-79's substance (no collapse to a constant) holds. | Corrected in place, with the reason. Awaiting re-review. |
| AC-125 | acoustics-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_DOMAIN_UNITS` | FIXED THIS CYCLE. "amp²" asserted a unit the package does not declare and reads as ampere-squared; the amplitude domain is raw ambisonic samples in arbitrary units. Now "a.u.²" with the reason recorded. Path is unexercised in production (spectrogram/edr are "db") but reachable via the waveform rep. | Consider declaring the amplitude domain's unit on Representation.value_domain (representations/base.py) and citing it here — metric lane. Awaiting re-review. |
| AC-126 | acoustics-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py footer | FIXED THIS CYCLE. The legend said the table mixes "seconds and decibels" while rendering dB² two lines above. dB² is a MEAN SQUARED level difference: 4.7987 dB² is an RMS level error of 2.19 dB, and reading it as 4.8 dB is a factor-of-2 misreading. | Legend now names the third unit and says to take its square root. Awaiting re-review. |
| AC-127 | acoustics-reviewer (lane P) | major | OPEN | src/amcd/reporting/tables.py `_METRIC_UNITS`; src/amcd/evaluation/evaluator.py:85-98; tests/test_report.py | THE UNIT COLUMN IS AN ASSERTION ABOUT ANOTHER MODULE'S OUTPUT WITH NO TEST BINDING THEM. dB²/dB are correct ONLY BECAUSE evaluator.py denormalizes with high_mean/high_std before computing signal metrics; on normalized tensors the same metric is in (z-scored dB)², a factor high_std² off, and tables.py cannot see which it got. All the new tests feed SYNTHETIC rows to the formatter and check the string; none checks the number is in the unit claimed. Verified correct today by probe (x2 operands -> x4.0 improvement exactly), so an untested invariant, not a live error — but data/normalization.py was edited in this very change. | Known-answer end-to-end test: dB tensors with a KNOWN mean squared level error through eval->stats->report, asserting the rendered Imp mean equals the hand-computed dB². Same for T30 in seconds via an exponential decay of known T60. |
| AC-128 | acoustics-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py footer; src/amcd/evaluation/metric_row.py:119-127 | FIXED THIS CYCLE. One Unit column labelled three quantities: for match_reference metrics Imp mean is a REDUCTION IN |ERROR| vs the high-ray reference, for maximize a plain difference, and Pred mean is the absolute value. "C50 Pred mean 31.7503 dB, Imp mean -0.0669 dB" reads as "C50 moved by -0.07 dB" when it means "the model's |C50 error| GREW by 0.067 dB". | Legend now states what the unit is OF per kind, and that a negative Imp mean means the error grew. Awaiting re-review. |
| AC-129 | acoustics-reviewer (lane P) | major | OPEN | src/amcd/reporting/tables.py (`df.to_csv(report_dir / "metrics_table.csv")`) | THE DISCLOSURE IS HALF-APPLIED INSIDE THE FILE THAT WAS CHANGED. summary.txt gained a Unit column; report/metrics_table.csv, written five lines later by the same lane-owned file, still carries pred_mean, the CIs, improvement_mean and improvement_mdes as bare floats across rows mixing s, dB and dB² — 21 columns, zero units. This is the machine-readable twin a paper table would be built from, and it needs no cross-lane change. | Add a `unit` column from the same `_unit_for` before the to_csv. NOTE: this MOVES report/metrics_table.csv, so the lane's declared "expected effect" must be RE-DECLARED in advance rather than amended after the fact (the ci_table.csv claim is unaffected). |
| AC-131 | acoustics-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py (pred-unresolved legend) vs (high-variance legend) | AC-48's OWN STANDARD IS APPLIED TO ONE CAVEAT AND NOT ITS NEIGHBOUR. The high-variance line renders its threshold's value and unit; the pred-unresolved caveat is governed by metric_band_resolvability_margin x the filter's own T30/EDT in that band (room_acoustic.py:478-487; base.yaml 2.0), and NEITHER the multiplier NOR the per-band floor in seconds appears anywhere in summary.txt. The wording also conflates a hard NaN with falling below the floor. | Render the margin and per-band floors the way the high-variance line does, citing `_band_resolvable_decay_s` as the one place the filter decays are written. |
| AC-132 | acoustics-reviewer (lane P) | minor | OPEN | src/amcd/evaluation/evaluator.py:126-129 (`iso_windows`); metrics/iso_integration_windows.json | THE SCHROEDER INTEGRATION LIMIT IS REPORTED WITH NO DECLARED UNIT. The artifact is {"scene_0001": {"1000": {"set_by_leg": "high", "trunc_idx": 9897}}} — trunc_idx is in SAMPLES and the band key in Hz, and the JSON declares neither, nor the sample_rate needed to convert. Same units-declaration class the Unit column just closed for summary.txt. OUTSIDE lane P's owned files — recorded only. | Rename to trunc_idx_samples (or add a sibling trunc_s) and stamp sample_rate and band_hz. Metric-computation lane. |
| RR-141 | readability-reviewer (lane P) | minor | OPEN | src/amcd/data/normalization.py; src/amcd/data/preprocess.py | The F-M11 keep-argument is written at FULL LENGTH three times — the docstring, the comment above the key, and the artifact string. Same cut-then-regrow class as RR-56/RR-46, which this lane has now done three times. PARTIALLY addressed this cycle (the preprocess comment was cut when F-165 restructured the key), but the argument still appears in two places at length. | Keep it at normalization.py (where the numbers are produced) and in the artifact string (which must stand alone for a meta.json reader); reduce any third statement to a citation. |
| RR-142 | readability-reviewer (lane P) | minor | OPEN | src/amcd/data/preprocess.py; src/amcd/reporting/tables.py `_METRIC_UNITS` | LANE/PROCESS BOOKKEEPING INSIDE PACKAGE SOURCE: "which lane P does not own", "that spans the metric-computation lane, so it is filed for the integrator rather than half-built here". Both are false or meaningless once the branches merge, and neither is the durable reason. Partially addressed (the preprocess instance now says "asserted verbatim by tests/test_invariants.py"); the tables.py instance stands. | Keep the durable constraint, drop the lane vocabulary: "the durable form declares `unit` beside `kind` on the metric itself (RD-201)". |
| RR-143 | readability-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_METRIC_UNITS` docstring; tests/test_report.py test-class docstring | 14 comment lines on a 5-entry dict, two of whose claims are re-told elsewhere: the symptom narration, and "a blank unit beside a physical quantity is exactly the silent exclusion the drop log exists to prevent", which is near-verbatim in the test class. The line that earns its place is "a unit cannot come from `kind`". THE LANE'S OWN PATTERN, third cycle running. | Reduce to three lines (what the mapping is; why not `kind`, RD-201; unlisted -> raises) and have the test docstring cite `_METRIC_UNITS` instead of repeating it. |
| RR-145 | readability-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_OPERAND_DOMAIN_SQUARED`, `_METRIC_UNITS`, `_unit_for` | `_OPERAND_DOMAIN_SQUARED = object()` forces `dict[str, object]`, so the annotation tells a reader nothing about a legal entry, and `_unit_for` is annotated `-> str` while returning that object in one branch. The two legal shapes are discoverable only by reading the body. | Give the sentinel a named singleton type with a docstring and annotate `dict[str, str | _OperandDomainSquared]`. |
| RR-146 | readability-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py `_DOMAIN_UNITS` | PARTIALLY FIXED THIS CYCLE (the comment now names representations/base.py as the declaring site and says an unlisted domain is refused). Recorded so the re-review checks the wording rather than assuming. | Confirm the citation is accurate at re-review. |
| RR-149 | readability-reviewer (lane P) | minor | OPEN | src/amcd/config.py Config.stamp code_version comment | RR-56's DUPLICATION IS GONE — the block is now a citation to amcd.provenance and nothing load-bearing was lost (verified: code_version_unscoped is live at pipeline.py:651, code_version_describes is present in versions.json). But 11 lines remain where 3 would do: part re-narrates the F-75 discovery the ledger id already carries, and part justifies why both a comment and a key exist when the key's own string already says it to the versions.json reader. This lane has trimmed and regrown this block twice (RR-36 -> RR-46 -> RR-56). | Cut to ~3 lines: the citation, "this stamp describes THIS INVOCATION, not the artifacts (F-75)", and "per-stage truth: stages/<stage>.done". Trim to the rule; do not reflow. |
| RR-150 | readability-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py (the "N sc/att" comment above `col_w`) | The comment sits above `col_w`, a column-WIDTH dict it does not describe, and its second half duplicates the footer line the file already prints. Only "the scored count is the paired-improvement population" is not said elsewhere. | Move that clause to where the column is produced; delete the rest. |
| RR-153 | readability-reviewer (lane P) | minor | OPEN | docs/ledger_inbox/P.md pass-condition block | FIXED THIS CYCLE. The block gave two run-dir paths and one sha256 but not the commands, and the A-leg path is a timestamped dir that will not exist for the later reader this file is written for. | Commands, both run dirs and the diff are now in the block. Awaiting re-review. |
| RR-154 | readability-reviewer (lane P) | minor | OPEN | src/amcd/reporting/tables.py `run_report` | The module's public entry point has no docstring while the two private helpers added this cycle each carry one. A caller cannot see that it writes report/summary.txt + report/metrics_table.csv, copies config.yaml/versions.json only at save >= provenance, or that it now raises on a missing stats summary OR a missing preprocess meta. Pre-existing, surfaced by this diff. | Three-line contract docstring: inputs, files written, the verbosity-gated copy, the raise conditions. |
```
