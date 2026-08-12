# Lane S inbox — cycle5

Branch `lane/S-cycle5`. Written by lane S, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/S-cycle5", never "clean".

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

## READ THIS FIRST — three things the integrator must not meet cold

**1. Lane S LIFTS NOTHING AND UNBLOCKS NOTHING on RD-33a.** Checked against
RD-128's severity scoping of condition (i) before any work started, and it holds
after it. `src/amcd/diagnostics/**` is not on the (i) path list at all, so F-72,
RR-63 and RR-64 — including a major — contribute exactly zero. Of the 21 assigned
rows, exactly ONE is major AND on the path list (RD-112), and its contribution is
*re-deriving a cycle-4 fix*, not new work. The other 20 shrink the ledger and
discharge the awaiting-re-review backlog: both are required by the definition of
done, neither is gate movement. **Do not book this lane as progress on (i).**

**2. Two rows are PARTIAL and must NOT be deleted.** `RD-65` and `F-60` each have
a residual in a file lane S does not own, and `generator.py` now looks clean for
both — so a post-merge reviewer could delete them, and deletion is this project's
only "resolved" marker. Marked below in cycle 4's shape:

`PARTIAL — stays OPEN; residual = <file> <what>`

This is exactly the hazard **RD-111** exists for. RD-111 is a *process* row,
though, so it can close without either residual landing — which is why each gets
its own anchored row below (RD-227, F-185).

**3. Nine of the 21 assigned rows were "fix applied, never re-derived".** They
were fixed in cycle 4 and assigned to this lane as ordinary `fix:` rows, contrary
to protocol planning step 3. This pass re-derived each rather than stacking a
second fix on an unchecked first. **Eleven came back CONFIRMED FIXED and two did
not** — RR-60/RR-61 NOT FIXED (re-introduced by this session) and RR-65 regressed
by this session, both then fixed in `5e2a09b`. Full verdict table below. Raised as
RD-226.

**4. F-186 is a major that is NOT lane S's to fix and must reach ITEM 0.** Both
the falsifier and the acoustics-reviewer arrived at it independently: the
record-length gate is evaluated at nominal α while AC-54 holds the backend
realizes α_eff, and over base's declared support that is the difference between
P(over-limit) = 0.000 and 0.018 against a declared tolerance of 0.0. It is the
same physical inconsistency as AC-54/AC-55/AC-56 and must not become a fourth
partial fix (RD-143). Details and a one-render falsification signal below.

## Declared expected effect on `ci_table.csv`: **NONE**

Declared before starting, per RD-91/RD-149 — the detector only discriminates
interference from legitimate change if every non-metric lane declares in advance.

Structural basis: nothing in this lane's diff touches `rng`, `_sample_*`,
`_generation_plan` or scene admission. Scene specs are written inside the
generation loop (`generator.py`); `_disclose_and_gate_record_length` runs after
every spec already exists, so its only failure mode is aborting the whole run —
never partial admission.

## Preflight

```
$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-S/src \
    /Users/nortonrainey/miniconda3/envs/amcd/bin/python scripts/lane_preflight.py
identity : lane S (scenes, QC and diagnostics), cycle cycle5
checkout : /Volumes/T7/Monte_Carlo_Research/v3-lane-S
branch   : lane/S-cycle5
amcd from: /Volumes/T7/Monte_Carlo_Research/v3-lane-S

OK — imports, branch and identity all agree.
```

`git merge v3-rebuild` → `Already up to date.` Nothing arrived, so the evidence
below is valid against the tree it was measured on.

## Pass condition — MET

Fixed-seed A/B against a baseline captured before any edit
(`experiments/all_20260811_180407`):

```
$ diff baseline/stats/ci_table.csv after/stats/ci_table.csv
IDENTICAL
$ diff -r baseline/scenes after/scenes
IDENTICAL
```

⚠️ **The second diff has NO `-x placement_report.json`.** Cycle 4 excluded it,
and it is the one artifact this lane's edits compute — S-F4, S-F5, AC-53 and the
`_flag_counts` change all touch the code producing its numbers, so a slip in the
characterized-denominator logic would have moved `n_scenes`/`fraction` invisibly
to the cycle-4 pass condition. Byte-identity is asserted over it here.

```
$ PYTHONPATH=.../src .../pytest -q
530 passed, 1 skipped in 63.62s        (506 + 1 skipped before this lane; +24 new)

$ PYTHONPATH=.../src .../amcd all -c configs/base.yaml \
    -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
[done] gen-scenes / render / preprocess / diagnostics / train / infer / eval / stats / report
```

Commits: `c799003` (the rows), `5e2a09b` (the reviewer self-checks' findings,
including two majors in `c799003`'s own prose) and `4dfe46c` (AC-153 + RR-166,
the reviewer findings that were in this lane's files after all). The A/B above was
re-run on `4dfe46c`, not carried over.

**Every reviewer finding that lane S can reach is CLOSED.** Of the six new rows
raised against this lane's own work, four are cross-lane or belong to ITEM 0
(F-186, AC-150, plus RD-225/226/227 and F-185 on the partition), one was fixed in
`4dfe46c` (AC-153) alongside RR-166, and exactly one remains OPEN and unreachable:
RR-165 needs a NEW test file, which the ownership hook refuses.

---

## CLOSED — with the evidence, re-derived on THIS tree

### F-71 — the dilution attack is refused (re-derived, not taken on trust)

Fixed in cycle 4, never re-derived until now. Reproduced end to end on the current
tree with a declared `openfield` family (3 scenes, `characterization: none`)
alongside tiny's 26 enclosed scenes, at the tolerance where dilution decides the
outcome (0.9): honest 26/26 = 100 %, diluted 26/29 = 89.7 %.

```
GATE RAISED : ir_duration is 0.1 s, but 26 of 26 scenes (100.000%) exceed it —
              more than scenes.max_t60_over_ir_duration_frac (0.9) allows:
exclusion   : 3 of 29 scenes are excluded from this fraction as uncharacterized (RD-64).
openfield block n_scenes/n_uncharacterized/fraction: 0 3 None
```

The uncharacterized scenes are out of numerator, denominator and gate alike, the
exclusion is itself disclosed, and the block reports `null` rather than `0.0`.

### RD-112 — a gate that scores nothing is UNSCORED, not passed

Re-derived on the REAL generation path, which is the half the cycle-4 test missed
(its hand-built report never reached the corner disclosure, where the
`f"{None:.2f}"` TypeError made the warning unreachable on the only config that
triggers it):

```
WARN: WARNING: the record-length gate scored 0 of 26 scenes — every geometry family
      in this config declares characterization: none, so no closed-form T60 exists
      to compare against ir_duration 0.1 s. The gate is UNSCORED, not passed (RD-112).
all splits scored 0: True
```

No TypeError; warning emitted; every split reports 0 scored.

### RD-65 — see PARTIAL below. The generator half is confirmed; the row STAYS OPEN.

---

## PARTIAL — these two stay OPEN

### RD-65 — warning half confirmed, report-table half never scheduled

`PARTIAL — stays OPEN; residual = src/amcd/reporting/tables.py — carry per-split
over-limit counts into the E1 report table (lane P's file; integrator queue per
RD-82, since a new reported COLUMN spans two lanes by construction).`

Confirmed present and working on this tree — the per-split warning is emitted
unconditionally and BEFORE the overall gate can raise, so a failing run still
names the splits responsible:

```
WARNING: split 'id': 20/20 scenes (100.000%) exceed ir_duration 0.1 s — above this
         config's own scenes.max_t60_over_ir_duration_frac (0.9). ...
WARNING: split 'test_material_shift': 2/2 scenes (100.000%) ...
WARNING: split 'test_placement_shift': 2/2 scenes (100.000%) ...
WARNING: split 'test_geometry_shift': 2/2 scenes (100.000%) ...
WARNING: split 'test_openfield': 0 of 3 scenes are characterized, so its
         over-limit fraction is UNDEFINED — reported as null, never as 0.0.
[then] GATE RAISED
```

The residual is scheduled in NO cycle-5 lane. Raised as **RD-227**.

### F-60 — the doable half done; the realized check is not lane S's to write

`PARTIAL — stays OPEN; residual = src/amcd/evaluation/room_acoustic.py +
configs/base.yaml — flag any scene whose FITTED T30 exceeds a config-declared
fraction of the REALIZED record length, and count it in the eval output. (Lane
M's files.)`

The row's charge is that the gate is called "realized" but gates realized DRAWS
OF AN ESTIMATE, never the rendered IR. Everything that can be said inside
`scenes/` is now said: the docstring no longer calls it realized, and both
mechanisms the row names are stated at the flag's own definition — Sabine's 4V/S
mean-free-path assumption against base.yaml's corridor family, and the scaffold's
rt60 clip (recorded as `rt60_clipped`, which the flag never consults).

The remedy proper needs `evaluation/` and a config key. Neither is in this lane's
owned set, and `docs/parallel_protocol.md:347-351` names F-60 as a *cycle-4
integrator-queue row for exactly this reason* — yet `cycle5.yaml:233` re-assigned
it here with `fix: [scenes/generator.py]`, which cannot host it. Raised as
**RD-225**; the residual proper is **F-185**.

---

## New findings

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| RD-225 | lane S | minor | OPEN | `docs/lanes/cycle5.yaml:233` (row F-60) | F-60 is declared `fix: [src/amcd/scenes/generator.py]`, but its resolution — a FITTED T30 counted in the eval output against a config-declared fraction — needs `src/amcd/evaluation/room_acoustic.py` and `configs/base.yaml`, both lane M's. It cannot be finished in the lane it was assigned to. `docs/parallel_protocol.md:347-351` already records F-60 as a cycle-4 INTEGRATOR-QUEUE row for precisely this reason, so the reassignment reversed a decision the protocol document states. It passed `tests/test_lane_partition.py` because the reachability check validates a row's declared `fix:` paths against `owns` and never against the row's own ANCHOR — which is exactly the hole RD-111 (cycle-4 RD-93) asked the next partition to close, and it was not closed. | Move F-60 to `integrator_queue:` or split it: the `scenes/` half is done (see PARTIAL above), the residual is F-185. Separately, make the partition test compare each row's ledger ANCHOR against its owning lane, not only its declared `fix:` paths — without that, the next spanning row passes declaration time the same way. |
| RD-226 | lane S | minor | OPEN | `docs/lanes/cycle5.yaml:232-252` (rows F-71, F-72, RR-44, RD-112, RR-60, RR-61, RR-65, RR-66, RR-67) | Nine of lane S's 21 rows were FIXED IN CYCLE 4 and never re-derived (`docs/ledger_inbox/archive/cycle4-S.md:52, 119, 164, 274, 432-437`), yet cycle5.yaml assigns them as ordinary `fix:` rows. `docs/parallel_protocol.md:316-318` (planning step 3) says such rows are excluded from every lane, because assigning one invites a second fix stacked on a first nobody checked; `awaiting_re_review:` at `cycle5.yaml:344` lists only `RR-24`. The cycle's arithmetic therefore counts nine confirmations as nine work items, which is a ~40 % overstatement of this lane's remaining effort. | Repopulate `awaiting_re_review:` from the cycle-4 inboxes' CLOSED sections when planning cycle 6. Pairs with RD-146. This lane re-derived them rather than re-fixing them; verdicts in the reviewer sections below. |
| RD-227 | lane S | minor | OPEN | `src/amcd/reporting/tables.py` (RD-65 residual) | RD-65's residual — carrying per-split over-limit counts into the E1 report table — is scheduled in NO cycle-5 lane: not in lane P's rows, not in `serial_queue:`, not in `integrator_queue:`. Its only protection is RD-111, which is a PROCESS row (its remedy is a `partial_residual:` list and a partition-test change), so RD-111 can close with the column having never landed and RD-65 then reads as a generator-only row that is done. | Own row on the integrator queue — rule 2 / RD-82: the lane that computes a number does not own the file that reports it, so a new reported COLUMN spans two lanes by construction. Do not delete RD-65 at integration step 7. |
| F-185 | lane S | minor | OPEN | `src/amcd/evaluation/room_acoustic.py`; `configs/base.yaml` (F-60 residual) | Nothing anywhere compares a scene's FITTED T30 against the record it was rendered into. The only record-length check in the pipeline is the design-time closed-form gate in `scenes/generator.py`, whose instrument is Sabine-from-geometry — so a scene truncated by a decay Sabine under-predicts (the corridor family) or by the scaffold's rt60 clip is never counted anywhere. **Two things this row must carry.** (a) The comparand is the REALIZED record length, NOT the configured `ir_duration`: `docs/lanes/cycle5.yaml:14-15` states pygsound compiles `maxIRLength = 3.0 s` and does not expose it, so `ir_duration 4.25` can never be filled and a check written against the configured value would test a number the backend cannot produce. (b) It is therefore dependent on ITEM 0 (AC-54/AC-55/AC-56), which settles what the backend actually realizes. | Flag any scene whose fitted T30 exceeds a config-declared fraction of the realized record length; count it in the eval output with a scored-vs-attempted denominator. Home: cycle-6 lane M, or cycle-5 integrator queue if ITEM 0 lands first. Test: a scene whose RENDERED T30 exceeds its record must appear in a count, not only in a pre-render estimate. |

---

## What was implemented, and what each new test is worth

`c799003`. Rows S-F4, S-F5, S-F6, S-F7, AC-51, AC-52, AC-53, RR-62, RR-63, RR-64
plus F-60's doable half.

**Honest accounting of the 16 new tests.** Eight were verified to FAIL against the
pre-fix generator (`git stash push -- src/amcd/scenes/generator.py`, run, `git
stash pop`):

```
FAILED TestTheGateDiagnosesAReportKeyItCannotScore::test_an_undeclared_non_split_key_names_itself
FAILED TestTheGateDiagnosesAReportKeyItCannotScore::test_a_declared_non_split_key_is_skipped_not_scored
FAILED TestTheGateDiagnosesAReportKeyItCannotScore::test_the_declared_set_is_empty_today
FAILED TestADeclaredSplitWithNoScenes::test_a_zero_scene_split_is_named_by_the_gate
FAILED TestUncharacterizedRecordLengthIsUncheckedNotMerelyExcluded::test_the_scene_reason_names_record_length
FAILED TestUncharacterizedRecordLengthIsUncheckedNotMerelyExcluded::test_the_record_length_block_says_unchecked
FAILED TestUncharacterizedRecordLengthIsUncheckedNotMerelyExcluded::test_each_block_declares_its_own_consequence
FAILED TestUncharacterizedRecordLengthIsUncheckedNotMerelyExcluded::test_the_clause_is_required_not_defaulted
8 failed, 7 passed
```

**The other seven passed pre-fix, and that is correct — they pin behaviour that
was already right but unguarded.** Said plainly rather than folded into a "new
tests" headline:

- **S-F5** (3 tests). The mixed `0 < n_uncharacterized < n` case is unreachable
  from any shipped config — one geometry family per split makes characterization
  all-or-nothing — so RD-113's pin only ever sees 0 or n. Cycle 4's falsifier
  probed the mixed case directly and found it CORRECT; nothing guarded it. These
  guard it. They pass pre-fix because `_flag_counts` used to swallow the new
  keyword into `**context`.
- **AC-52** (3 tests) + `test_it_is_distinguished_from_an_uncharacterized_split`.
  Known-answer pins on unchanged formulas.

⚠️ **CORRECTION — my first claim about what AC-52's tests are worth was REFUTED
by the falsifier, and the corrected claim is narrower.** Recorded here rather than
quietly restated, because `c799003`'s commit message carries the wrong version and
history cannot be edited.

I claimed the new tests "catch a wrong constant (100.0) that the previous
inequality-only pin accepts". Forcing `_C_TIMES_SABINE_K = 100.0` in fact fails
FOUR tests, **two of them pre-existing**
(`test_the_declared_support_spans_every_declared_material_regime`,
`test_the_individual_corners_still_reproduce`) — and an Eyring-only error
(`log1p` scaled ×2) fails three, one again pre-existing. So the suite was **not**
inequality-only for Eyring, and my demonstration had simply run the one old test
that is.

The claim that survives, verified: switch to the c = 343 convention
(`_C_TIMES_SABINE_K = 343*SABINE_K`), a −0.035 % move that sits **inside every
`abs=0.005` pin in the file**:

```
[probe] _C_TIMES_SABINE_K -> 55.222... (was 55.26204)
FAILED ...::test_d_min_reproduces_the_iso_form_from_the_reported_t60
1 failed, 67 passed
```

**AC-52's real worth is that its tests are the only thing pinning the
speed-of-sound convention `_C_TIMES_SABINE_K` folds in** (AC-49's 343.24 vs 343).
That is a smaller claim than the one I made and it is the one the evidence
supports.

**One deliberate deviation from a row's wording.** S-F4 offered "guard the shape,
or state in the docstring that every top-level key is a split". Neither, exactly:
the guard is a module-level DECLARED set `_NON_SPLIT_REPORT_KEYS` (empty today)
which the gate skips, and an undeclared unscoreable key raises a `ValueError` that
names the constant to add it to. Hardcoding "every top-level key is a split" would
have foreclosed the metadata AC-54/RD-144 (absorption convention), RD-131 (the
AC-54 caveat) and AC-30/AC-50 are all pushing into that artifact — S-F4's own text
says the disclosure work invites it.

**AC-51 and AC-52 are caveated "nominal α, pending AC-54"**, in both the comments
and the test docstring. ITEM 0 is ordered first this cycle and may replace nominal
α with α_eff = 1−sqrt(1−α), under which these closed forms describe a room never
rendered. The caveat is so the pin fails loudly and legibly if that lands, rather
than certifying an unrendered room. Mirrors AC-50's wording on the same numbers.

---

## Reviewer self-checks on `lane/S-cycle5` @ `c799003` (NOT a clean pass)

All three run over the CURRENT state, not just the diff, and each was given its
own awaiting-re-review rows for a per-row verdict. **Every finding they raised
about `c799003` is fixed in `5e2a09b`** except those recorded as new rows below.

### Backlog verdicts — all nine rows re-derived

| row | reviewer | verdict |
|---|---|---|
| F-71 | falsifier | **CONFIRMED FIXED** — rebuilt the dilution attack from scratch (4 enclosed all breaching + 6 non-enclosures): gate raised at 4/4 = 100 %, would have passed at 4/10 = 40 %. |
| F-72 | falsifier | **CONFIRMED FIXED** — by exhaustive AST enumeration of all six whole-entry writer sites in `probe.py`, not by reading the tests. |
| RD-112 | falsifier | **CONFIRMED FIXED**, both halves — no TypeError on the `None` corner, warning reachable on the only config that triggers it. |
| AC-51 | acoustics | **CONFIRMED FIXED** — 1.9074425 and crossover α = 0.7251492 both re-derived independently; my 1.907 / 0.725 correct. |
| AC-52 | acoustics | **CONFIRMED FIXED** — all three assertions verified independently: S = 460.0 m², d_min_eyring = 7.3203620 m, equal-surface pair V = 520 vs 600, AC-49 offset −0.00035331. |
| AC-53 | acoustics | **CONFIRMED FIXED**, with the residual at AC-153 below. |
| RR-44 | readability | **CONFIRMED FIXED** — `flag_key_present` gone from the tree. |
| RR-60 / RR-61 | readability | **NOT FIXED — re-introduced by this session.** See below. |
| RR-62 | readability | **CONFIRMED FIXED** — module grepped for `%`, `median`, `2.51`, `0.894`, `92.5`: no realized-config constants survive. |
| RR-63 | readability | **CONFIRMED FIXED.** |
| RR-64 | readability | **CONFIRMED FIXED**, with one surviving instance of the same defect one key over — fixed in `5e2a09b`. |
| RR-65 | readability | **CONFIRMED FIXED for the banner; REGRESSED by this session** in the docstring. Fixed in `5e2a09b`. |
| RR-66 / RR-67 | readability | **CONFIRMED FIXED.** |

**RR-60/RR-61 in full, because it is this session's own defect.** My AC-53 and
S-F7 docstring additions put the "uncharacterized scenes leave BOTH numerator and
denominator" rule back into two more homes, giving it four —
`_room_acoustics`, `_scene_is_characterized`, `_flag_counts` and the gate
docstring, two of them in near-identical words. That is precisely what RR-60/61
asked to cut to one home each. Fixed in `5e2a09b`: the rule now lives at
`_scene_is_characterized` (the predicate), and the other sites point at it.
`_room_acoustics`'s copy was kept deliberately — it explains why the key is
*omitted rather than False*, at the site of the omission, which is a different
statement.

### What the reviewers could NOT break (stated so it is not re-litigated)

- **Population neutrality.** The falsifier rebuilt the pre-work tree in scratch
  and ran gen-scenes on all four shipped config stacks under both trees:
  `IDENTICAL, scenes/ byte-for-byte`, `placement_report.json` included. Mechanism:
  the only artifact-visible delta is `uncharacterized_note`, emitted only when
  `n_uncharacterized > 0`, which needs a family declaring `characterization: none`
  — and **no shipped config declares one**. It found no reachable config where
  these edits move a number.
- **The F-71 dilution attack** — no route found to get an uncharacterized scene
  into the gate's denominator.
- **RR-64's emit-iff contract — PROVEN, not asserted**, by enumerating all six
  writer sites.
- **AC-53's `uncharacterized_consequence` cannot leak into the artifact**
  (keyword-only, outside `**context`; verified absent from the JSON).
- **The S-F4 guard cannot fire on a legitimate report** — verified end to end on
  all four shipped config stacks.
- **The S-F6 reordering cannot change the gate arithmetic** — `over`/`total`/
  `attempted_total` are summed from `per_split` and untouched by the warning loop.
- **No scaffold or platform coupling** in either file.
- **The §5.3 reduction, both variants, and the AC-30/AC-50 support corners** all
  re-derived correct.

### New findings

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| F-186 | falsifier + acoustics-reviewer | **major** | OPEN | `src/amcd/scenes/generator.py` `t60_exceeds_ir_duration` / `_disclose_and_gate_record_length`; `configs/base.yaml` `max_t60_over_ir_duration_frac` | **THE RECORD-LENGTH GATE PASSES THE EXACT CONFIG IT EXISTS TO REFUSE, IF AC-54 IS RIGHT.** The gate is the only pre-render check and can abort a run, and it is evaluated at NOMINAL α while ITEM 0 holds that the backend realizes α_eff = 1−sqrt(1−α). T60 then scales by α/α_eff = 1.975 at α 0.05, 1.837 at 0.30, 1.447 at 0.80. Largest declared shoebox at α 0.05: **4.200 s nominal → 8.294 s effective**, against a 4.25 s record. Monte-Carlo over base's declared shoebox × `mixed` support (2×10⁵ draws): **P(T60 > ir_duration) = 0.0000 nominal vs 0.0184 at α_eff**, against base's declared tolerance of **0.0**. Everything downstream that assumes an untruncated decay — T30, EDT, the Schroeder bound in `evaluation/room_acoustic.py` — inherits it. Distinct from F-60/F-185: that row is about the gate's instrument being an estimate; this is about the estimate being evaluated at the wrong α. | **Belongs with ITEM 0 (AC-54/AC-55/AC-56), not to a lane** — it is the same physical inconsistency and must not get a fourth partial fix. Falsification signal, one render, no ambiguity: render one 12×10×5 m box at nominal α = 0.05, source-receiver 3 m, fit T30 on channel 0 over −5 → −35 dB. Nominal predicts 4.20 s, α_eff predicts 8.29 s — 2× apart, far outside `d0b_t30_jnd_frac` 0.05. Backend-free precursor: synthesize a T60 = 8.29 s decay, truncate to 4.25 s, confirm the project's own T30 estimator reports the truncation rather than the room. |
| AC-150 | acoustics-reviewer | minor | OPEN | `src/amcd/acoustics.py:17-19`; `configs/simulators/gsound_sir.yaml:78`; `configs/simulators/dry_run.yaml:16` | **Three speeds of sound coexist across stages and only one pair is disclosed.** `SABINE_K = 0.161` implies c = **343.2425 m/s**; `dry_run` renders at **343.0**; the shipped `gsound_sir` backend at **344.0**. `SABINE_K` is a hardcoded module constant with no derivation from the configured `speed_of_sound_m_s`, so every published T60/d_min describes a room at 343.24 m/s whatever the backend is set to. Magnitudes are small — against a c = 344-consistent constant, T60 runs +0.22 % and d_min −0.11 % — so this is a declaration defect, not a numeric one; it would grow silently if c were set for a non-20 °C medium. **Not lane S's files.** | Derive `SABINE_K` from the active `speed_of_sound_m_s` (`24·ln10/c`), or declare in `acoustics.py` the tolerance within which the two may differ and guard it at config load. Integrator queue. |
| AC-153 | acoustics-reviewer + falsifier | minor | **CLOSED in `4dfe46c`** | `src/amcd/scenes/generator.py` | **The per-scene `(unit, reason)` pair AC-53 is about never reached disk.** `room_stats` reaches `placement_report.json` only through `_summarize` (numeric keys) and `_flag_counts` (booleans), and `SceneSpec.to_dict` has no such field, so the string died in memory: `'uncharacterized_reason' in placement_report.json → False`, `in any scene spec → False` over 16 specs. The disclosure was not lost — it reached the artifact per split — but the project's per-unit drop rule was satisfied at SPLIT granularity only, and `test_the_scene_reason_names_record_length` pinned a string no artifact carried. | **Fixed.** Each split record now carries `uncharacterized: [{scene, reason}]` — the same shape as the eval stage's `drops.csv` and `probe.py`'s `dropped` — emitted only when non-empty, so an empty list never reads as "checked and found nothing" and the canonical report is byte-unchanged. `SceneSpec` was the other candidate home; it is lane B's file, and the split record is the better one anyway because the aggregate already lives there. Verified on the openfield config: 3 entries, ids `scene_0026..28`, each reason carrying UNCHECKED, count agreeing with `n_uncharacterized`; the enclosed `id` split carries no key at all. |
| RR-165 | readability-reviewer | minor | OPEN | `tests/test_scene_placement.py` (whole file, now ~1240 lines) | The file is past 1000 lines and its own docstring concedes two subjects: Research-I config fidelity / placement sampling above the topical banner, and the record-length gate + ISO d_min disclosure below it. The banner keeps it navigable for a human, but an LLM asked to find the gate tests must read a config-validation file to get there. | Split at the banner into `tests/test_record_length_gate.py` (the gate/d_min classes plus `_openfield_config` / `_scored_entry`), leaving placement and config fidelity behind. **The ONLY row here lane S genuinely cannot finish**: a new test file is outside this lane's owned set and the ownership hook refuses it. Integrator-cycle move; it also changes `docs/lanes/*.yaml` ownership lists. |
| RR-166 | readability-reviewer | minor | **CLOSED in `4dfe46c`** | `src/amcd/scenes/generator.py` `_disclose_and_gate_record_length` | Even after `5e2a09b`'s docstring trim the function carried four responsibilities — corner disclosure, per-regime accounting, per-split warning, overall gate — in one body. | **Fixed.** `_disclose_declared_support_corner` and `_warn_regimes_over_limit` extracted, each with its own contract; the remaining function's name now matches what it does. The warning helper documents why three states are distinguished rather than collapsed (generated nothing / all uncharacterized / genuinely over limit). |

## Not started — and why each is genuinely out of reach



- **S-F1** — needs a config-declared minimum coverage in `configs/base.yaml`
  (lane M). No hidden defaults, so it cannot be fixed inside lane S. Unchanged
  from cycle 4; it is on the cycle-5 serial queue (ITEM 5).
- **S-1, S-2** — cross-lane consolidations, on the integrator/serial queues.
- **RR-165** — needs a new file in `tests/`, outside the owned set; the ownership
  hook refuses it. Integrator-cycle move.
- **F-186, AC-150, RD-225/226/227, F-185** — ITEM 0, `acoustics.py` (lane M), and
  the partition itself. See the rows for each.

Everything else the reviewers raised is closed in `5e2a09b` / `4dfe46c`. Nothing
on this list is deferred for convenience: each is refused by file ownership or
belongs to a decision (ITEM 0) that must not receive a fourth partial fix.
