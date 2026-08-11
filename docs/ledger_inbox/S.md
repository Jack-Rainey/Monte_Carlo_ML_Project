# Lane S inbox — cycle4

Branch `lane/S-cycle4`. Written by lane S, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/S-cycle4", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

---

## Read this first: three of my six rows are PARTIAL and must NOT be deleted

`RD-65`, `AC-30` and `RR-32` each have a residual in a file lane S does not own.
`generator.py` now looks clean for all three, so a post-merge reviewer could
delete them — and deletion is this project's only "resolved" marker, so the
residuals would be lost silently. Each is marked below in one shape:

`PARTIAL — stays OPEN; residual = <file> <what>`

Full accounting of that hazard is **RD-93** under "New findings" below.

## Preflight

```
$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-S/src \
    /Users/nortonrainey/miniconda3/envs/amcd/bin/python scripts/lane_preflight.py
identity : lane S (scenes, QC and diagnostics), cycle cycle4
checkout : /Volumes/T7/Monte_Carlo_Research/v3-lane-S
branch   : lane/S-cycle4
amcd from: /Volumes/T7/Monte_Carlo_Research/v3-lane-S
owns     : src/amcd/scenes/**, src/amcd/diagnostics/**, tests/test_scene_placement.py,
           tests/test_min_separation.py, tests/test_invariants.py, tests/test_probe.py,
           docs/ledger_inbox/S.md

OK — imports, branch and identity all agree.
```

`git merge v3-rebuild` → `Already up to date.` Nothing arrived, so the evidence
below is valid against the tree it was measured on.

---

## CLOSED

### F-71 — uncharacterized scenes no longer dilute the record-length gate

`src/amcd/scenes/generator.py` — `_room_acoustics` (`characterization: none`
branch) now OMITS `t60_exceeds_ir_duration` instead of setting it `False`, and
`_disclose_and_gate_record_length` scores only characterized scenes.

Reproduced first, with a declared `openfield` family (3 scenes,
`characterization: none`) alongside tiny's 26 enclosed scenes. **Before:**

```
### test_openfield — diffuse_field_validity (honours the rule):
  "n_scenes": 0, "n_uncharacterized": 3, ... "fraction": null
### test_openfield — t60_over_ir_duration (the F-71 sibling):
  "n_scenes": 3, "t60_exceeds_ir_duration": { "count": 0, "fraction": 0.0 }

### overall over-limit: 26 over-limit
###   scored denominator (block n_scenes) : 29  -> 89.655%
### gate at tolerance 0.9, 26 enclosed scenes all over-limit + 3 non-enclosures:
###   PASSED — the dataset bought its pass with uncharacterized scenes
```

**After:**

```
### test_openfield — t60_over_ir_duration (the F-71 sibling):
  "n_scenes": 0, "n_uncharacterized": 3,
  "uncharacterized_note": "scenes whose geometry family declares characterization:
    none are excluded from these fractions — ... (RD-64)",
  "t60_exceeds_ir_duration": { "count": 0, "fraction": null }

### overall over-limit: 26 over-limit
###   scored denominator (block n_scenes) : 26  -> 100.000%
### gate at tolerance 0.9, 26 enclosed scenes all over-limit + 3 non-enclosures:
###   RAISED — ir_duration is 0.1 s, but 26 of 26 scenes (100.000%) exceed it —
###            more than scenes.max_t60_over_ir_duration_frac (0.9) allows:
###       id: 20/20 scenes
###       test_material_shift: 2/2 scenes ...
```

The sibling block now matches `diffuse_field_validity` exactly, and the dilution
attack (26/26 = 100 % honest, 26/29 = 89.7 % diluted, tolerance 0.9) is refused.

Tests: `tests/test_scene_placement.py::TestUncharacterizedScenesLeaveTheRecordLengthGate`
(4 tests, all constructing the failing population). Verified they FAIL against the
pre-fix generator — `git stash push -- src/amcd/scenes/generator.py`, 13 of 14 new
tests failed, restored with `git stash pop`.

**Population-neutrality (RD-91).** The gate cannot change ADMISSION: scene specs are
written at `generator.py:563` inside the generation loop, and
`_disclose_and_gate_record_length` runs at `:625` after every spec already exists,
so its only failure mode is aborting the whole run. Empirical guard on the
canonical dry run, covering this lane's whole diff:

```
$ diff baseline/stats/ci_table.csv after/stats/ci_table.csv
IDENTICAL — ci_table.csv unchanged
$ diff -r -x placement_report.json baseline/scenes after/scenes
IDENTICAL — every scene spec byte-for-byte unchanged
```

⚠️ **Note for the integrator, so this is not met cold:** `ci_table.csv` cannot
*discriminate* for F-71 — no shipped config declares a `characterization: none`
family, so `n_uncharacterized` is 0 everywhere and the table is unchanged by
construction whatever F-71 does. The structural argument above is what establishes
neutrality. Consequence to expect: once a `none` family IS declared, this gate can
newly abort a run that previously passed. That is F-71 working, not a regression.

### F-72 — D0a and D0b no longer drop a scene, or a split, in silence

`src/amcd/diagnostics/probe.py`. Both probes accumulate `(scene, reason)` per
split, report `n_attempted` alongside the scored count, and record an all-failed
split with an `unscored_reason` instead of skipping it.

**Before** (starve `test_geometry_shift` of its 2 scenes' tensors, canonical tiny
config):

```
  D0b — Carrier ceiling test ...
  train    0.0021s(PASS)   0.0027s(PASS)   0.1285dB(PASS)
  ... (test_geometry_shift absent entirely)

  D0b verdict: CARRIER CEILING CLEARS — oracle IR recovers reference metrics within JND
  Proceed to E1.

### test_geometry_shift in d0a_gap.json per_split: False
### test_geometry_shift in d0b_oracle.json per_split: False
```

**After:**

```
  test_geometry_shift    0/2  unscored — all 2 scenes failed to load — see `dropped` ...
  test_geometry_shift    N/A   N/A   N/A  N/A — all 2 scenes lacked a required input ...

  D0b verdict: INDETERMINATE — one or more ISO-3382 metrics unavailable in at least one split

### d0b entry: { "n_scenes": 0, "n_attempted": 2,
  "dropped": [ {"scene": "scene_0024", "reason": "input missing: scene_0024_high.pt"},
               {"scene": "scene_0025", "reason": "input missing: scene_0025_high.pt"} ],
  "unscored_reason": "all 2 scenes lacked a required input — see `dropped` ..." }
```

Tests: new `tests/test_probe.py` (6 tests). Verified all 6 FAIL against the pre-fix
probe (`git stash push -- src/amcd/diagnostics/probe.py`, restored after).

One deliberate deviation from the row's wording: the row asks for
`n_scored`/`n_attempted`, but D0a/D0b's existing `n_scenes` **is** the scored count
and is already consumed by the print branch and by lane P's
`test_dataset_integrity.py`. Adding a second `n_scored` key would be two names for
one number — the AC-24 divergence shape. So: `n_scenes` (scored) + `n_attempted`
(denominator) + `dropped`, with a comment at the write site saying so.

### RR-44 — verified already satisfied; the one unmet clause closed

Evidence is the CURRENT code state, not a diff: `_scene_is_characterized` is at
`generator.py:424`, above its caller `_flag_counts` at `:434`. (Provenance only:
that rename/move landed in `3c78c05` on the base branch, before this lane existed.)
The one clause genuinely unmet was the docstring, which named only `none`; it now
names both `sabine` and `none` and states the contract in one sentence.

---

## PARTIAL — these three stay OPEN

### RD-65 — warning half done

`PARTIAL — stays OPEN; residual = reporting/tables.py — carry per-split over-limit
counts into the E1 report table (lane P's file; integrator queue per RD-82).`

Done: `_disclose_and_gate_record_length` now emits an always-on per-split WARNING
naming any split over the declared limit, computed BEFORE the overall gate can
raise so a failing run still names the splits responsible. The OVERALL gate is
unchanged — kept deliberately, per the row.

```
$ pytest tests/test_scene_placement.py -k PerSplitOverLimit -q
5 passed
```
Covers: a split over its own limit named while the gate passes (1/30 = 3.333 % at
research_i's 0.01 tolerance, the row's own scenario); the warning preceding a gate
failure; survival at `show=0` (warnings bypass the ladder, F-24); a
zero-characterized split named as UNDEFINED rather than 0.0; and RD-94 below.

### AC-30 — `placement_report.json` half done

`PARTIAL — stays OPEN; residual = configs/base.yaml — the justification comment for
`distance_range [1.0, null]` still states "~2.6 m at the largest", which is not a
corner. (Lane M's file.)`

**The correction to apply.** ⚠️ **AC-30's own [0.41, 5.16] m is WRONG, and so was
my first transcription of it — do not write that range into base.yaml.** Caught
independently by the falsifier (S-F2) and the acoustics-reviewer (AC-46) on this
lane's diff. The row sweeps the `mixed` material regime only (α ≤ 0.80), but
base.yaml declares `ceiling_absorptive` α ∈ [0.85, 0.98] over the **same** shoebox
family, and `test_material_shift` selects exactly that regime.

d_min = 2·sqrt(V/(c·T60)); substituting Sabine's T60 gives d_min = 2·sqrt(αS/(c·K)),
and c and K appear only as the product `c·K = 24·ln10`, so d_min is independent of
volume and needs neither constant separately. (Caveat AC-45: `SABINE_K` ships
rounded, so using 24·ln10 assumes c = 343.24 m/s and biases d_min by a constant
−0.035 % — immaterial, but "free of c" is true of the formula, not of the constant.)

Swept over **every declared geometry × material corner**, the way
`Config.worst_case_t60` already sweeps them:

| corner | dims | α | regime | d_min Sabine | d_min Eyring |
|---|---|---|---|---|---|
| smallest, least absorptive | 3.0 × 3.0 × 2.4 | 0.05 | mixed | **0.41 m** | 0.42 m |
| largest, least absorptive | 12.0 × 10.0 × 5.0 | 0.05 | mixed | 1.29 m | 1.31 m |
| largest, `mixed` ceiling | 12.0 × 10.0 × 5.0 | 0.80 | mixed | 5.16 m | 7.32 m |
| **largest, most absorptive** | 12.0 × 10.0 × 5.0 | 0.98 | ceiling_absorptive | **5.71 m** | **11.41 m** |

So the declared support is d_min ∈ **[0.41, 5.71] m by Sabine and [0.42, 11.41] m
by Eyring** — not [0.41, 5.16] m. The 1.0 m floor still sits near the BOTTOM (that
half of the row holds, and more strongly), but writing 5.16 m into base.yaml would
install a fresh understatement of 1.11× (Sabine) / **2.20× (Eyring)** on precisely
the split whose realized shortfall is 92.5–100 %.

Verified: `TestIsoMinimumDistanceDisclosure::test_the_declared_support_spans_every_declared_material_regime`,
which now DERIVES the corners from the `Config` object. My first version pinned the
literals `(3.0, 3.0, 2.4)` / `0.05` / `0.80` and so could not see the omission —
that is why the error survived my own testing (S-F8).

Done in `generator.py`: each characterized scene carries
`iso_min_distance_sabine_m` / `_eyring_m` and the two `below_iso_min_distance_*`
flags; `placement_report.json` gains a per-split `below_iso_min_distance` block
(via the existing `_flag_counts`, so the RD-64 exclusion and its note come free)
beside the existing `d_over_rc` summary. Realized on the canonical dry run:

```
id                     n= 20  floor=1.0 m  d_min median sabine=2.34 m  below: sabine 4/20 (20.0%)  eyring 7/20 (35.0%)
test_material_shift    n=  3  floor=1.0 m  d_min median sabine=3.37 m  below: sabine 2/3 (66.7%)  eyring 3/3 (100.0%)
test_placement_shift   n=  3  floor=1.0 m  d_min median sabine=3.29 m  below: sabine 2/3 (66.7%)  eyring 2/3 (66.7%)
test_geometry_shift    n=  3  floor=1.0 m  d_min median sabine=2.75 m  below: sabine 1/3 (33.3%)  eyring 1/3 (33.3%)
```

(29-scene dry run, so coarse; the shape matches the row's 600-scene measurement —
id 20.0 % Sabine / 35.0 % Eyring here against the row's 25.4 % / 37.2 %.)
Disclosure only: no threshold, no config key, no behaviour change. The per-scene
ISO criterion stays DEFERRED, as the row says.

### RR-32 — generator half done

`PARTIAL — stays OPEN; residual = src/amcd/config.py `_check_split_roles` — the
docstring still replays the F-44 reproduction transcript. (Lane P's file.)`

Done: `_check_regimes_clear_backend_floor`'s docstring no longer embeds
"measured P(d < 0.3 m) = 0.186 %/scene → ~67 % chance a 600-scene run aborts" — a
measurement describing a config base.yaml can no longer produce. Cut to the
contract plus AC-13/F-48/RD-45/RD-57. `_disclose_and_gate_record_length` and
`_sample_positions` untouched, per the row.

Strictly this row spans two lanes and under `parallel_protocol.md` rule 4 should
not have been parallelized at all; see RD-93.

---

## New findings

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| RD-93 | research-director | major | OPEN | `docs/lanes/cycle4.yaml:127-132`; rows RD-65, AC-30, RR-32 | Three of lane S's six rows are half-remedies whose residual halves are declared only in a YAML comment. `cycle4.yaml` gives S `fix:[generator.py]` for each, but none appears in `integrator_queue:`, and `test_no_row_id_appears_in_two_places` forbids listing them twice — so the cycle's accounting (union of four lists == ledger OPEN set) counts all three as fully covered while the base.yaml correction, the E1 report-table counts and the `config.py` docstring exist nowhere a check can see. Since a resolved row is DELETED, a post-merge reviewer seeing `generator.py` clean is invited to delete three rows that are each ~two-thirds unmet. Related: the partition test validates declared `fix:` paths against `owns` but never against the row's own ANCHOR, which is how a spanning row passes rule 4. | Next cycle's partition gets a `partial_residual:` list (or the rows move to `integrator_queue` with the residual named); the partition test flags any row whose ledger anchor names a file outside its owning lane. |
| RD-94 | research-director | major | OPEN → fixed in-lane, awaiting re-review | `src/amcd/scenes/generator.py` `_disclose_and_gate_record_length` | After F-71, `if total and (over/total) > limit` lets a config of entirely `characterization: none` scenes reach `total == 0` with N scenes present, and the run completes with nothing printed and nothing recorded — F-71's own defect one level up, at exactly the outdoor/partially-open configuration the RD-64 seam exists to enable. | Fixed here: the zero-scored case warns "the record-length gate scored 0 of N scenes … UNSCORED, not passed". Test: `TestPerSplitOverLimitWarning::test_a_wholly_uncharacterized_config_is_unscored_not_passed`. |
| RD-97 | research-director | minor | OPEN → pinned in-lane, awaiting re-review | `src/amcd/scenes/generator.py:399-405, 449`; `tests/test_acoustic_validity.py:115-121` | The gate derives the characterized denominator as `entry["n_scenes"] - n_uncharacterized`, which is a second expression for the number `_flag_counts` already publishes as the block's own `n_scenes` — the AC-24 divergence shape in miniature. It is derived rather than read directly because lane P's `test_the_gate_is_overall_not_per_split` hand-builds a report whose `t60_over_ir_duration` block has no `n_scenes` key, and lane S cannot edit that file. | Pinned by `test_the_derived_denominator_agrees_with_the_published_one`. **Ask for lane P:** add `n_scenes` to that fixture's `t60_over_ir_duration` blocks; the derived form can then be replaced by the direct read. |
| S-1 | lane S | minor | OPEN | `src/amcd/acoustics.py` | The ISO 3382-1 §5.3 minimum measurement distance is computed in `scenes/generator.py` (`_C_TIMES_SABINE_K` + two expressions in `_room_acoustics`) because `acoustics.py` is lane M's this cycle. Nothing else computes d_min today, so there is no AC-24 divergence risk yet — but it is scene/statistical-model physics and belongs beside `sabine_rt60` / `critical_distance`. | Move to a `min_measurement_distance()` helper in `amcd/acoustics.py` (lane M) in a later cycle. |
| S-2 | lane S | minor | OPEN | `tests/test_probe.py`; `tests/test_dataset_integrity.py:256` | Probe coverage now lives in two files: F-45's `TestD0bEnumeratesDeclaredSplits` in lane P's `test_dataset_integrity.py`, and F-72's new S-owned `tests/test_probe.py`. Known and accepted for one cycle (RD-83). | Consolidate into `tests/test_probe.py` in a later cycle. |

Also worth the integrator's attention, not a finding: the D0a verdict table's count
column changed from `n` to `scored/att` (e.g. `10/10`, `0/2`), and D0b appends
`[n/N scored, k dropped]` to a partially-covered split's row. Console format only —
no artifact key was renamed.

---

## Not started, deliberately

`AC-41`, `F-60`, `AC-43` — on the integrator's serial queue and touching lane S's
files. Not mine to start this cycle (brief, "Not yours").

## Evidence

```
$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-S/src \
    /Users/nortonrainey/miniconda3/envs/amcd/bin/pytest -q
384 passed in 51.25s          (362 before this lane's work; +22 new)

$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-S/src \
    /Users/nortonrainey/miniconda3/envs/amcd/bin/amcd all \
      -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml \
      -c configs/overlays/dry_run.yaml
[done] gen-scenes / render / preprocess / diagnostics / train / infer / eval / stats / report

$ diff baseline/stats/ci_table.csv after2/stats/ci_table.csv
IDENTICAL — ci_table.csv still unchanged vs the pre-work baseline
```

Three reviewers have since run over the finished diff — `falsifier`,
`acoustics-reviewer` and `readability-reviewer`, sections below. All three are
SELF-CHECKS on `lane/S-cycle4`; none is a clean pass (rule 5).

---

## acoustics-reviewer pass on lane/S-cycle4 @ 5e15293 (SELF-CHECK, not a clean pass)

Scope: the AC-30 closed-form d_min physics and the F-71 gate treatment, reviewed
against the CURRENT tree. Verdict on the core algebra: **correct**. The ISO
3382-1 §5.3 reduction reproduces an independent numeric route to 0.035 % for both
variants over 25 (room, α) combinations, and that residual is entirely SABINE_K's
rounding (AC-45 below). Findings are about the constant's duplication, the stated
support range, a convention juxtaposition, and test coverage — not the formula.

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| AC-45 | acoustics-reviewer | minor | OPEN | `src/amcd/scenes/generator.py:27-32` (`_C_TIMES_SABINE_K`) vs `src/amcd/acoustics.py:19` (`SABINE_K`) | **A SECOND DECLARATION OF SABINE'S CONSTANT THAT DISAGREES WITH THE FIRST — the AC-24 shape, at 0.07 %.** The docstring asserts "c·SABINE_K … is 24·ln10 by SABINE_K's own definition", but `SABINE_K` is the ROUNDED literal `0.161`, so c·SABINE_K = 343 × 0.161 = **55.2230** while `_C_TIMES_SABINE_K` = 24·ln10 = **55.2620** (+0.0707 %). Every reported `iso_min_distance_*_m` is therefore **0.0353 % smaller** than the value a reader recomputing 2·sqrt(V/(c·T60)) from the report's OWN `t60_sabine_s` at c = 343 obtains — no single declared c reconciles the two published numbers. Numerically immaterial (1.8 mm at the 5.16 m corner) but it is literally two expressions of one physical constant in two modules. Related: the speed of sound is declared NOWHERE in config; it survives only implicitly inside the rounded 0.161, so "free of c" is true of the formula and false of the constant. | Derive d_min from `SABINE_K` (`d_min = 2·sqrt(αS/(c·SABINE_K))` with a declared c) or state the rounding in the constant's docstring — "24·ln10, which differs from c·SABINE_K by 0.07 % because SABINE_K is rounded; the resulting d_min bias is 0.035 %". Folds into S-1's move to `acoustics.py`. |
| AC-46 | acoustics-reviewer | major | OPEN | `docs/ledger_inbox/S.md` (AC-30 corner table); `tests/test_scene_placement.py:666-683`; residual target `configs/base.yaml:118-124` | **THE STATED SUPPORT RANGE [0.41, 5.16] m IS THE `mixed`-REGIME SPAN, NOT THE DECLARED SUPPORT.** All three corners reproduce exactly (0.412 / 1.290 / 5.161 m), but α = 0.80 is `mixed`'s ceiling; `ceiling_absorptive` declares α ∈ [0.85, 0.98] on the SAME shoebox family (`test_material_shift` overrides material only). MEASURED over every geometry × material corner of base.yaml: Sabine d_min spans **[0.412, 5.712] m** (max at shoebox 12×10×5, α 0.98) and Eyring **[0.417, 11.413] m** — against the 7.32 m Eyring value at the stated 5.16 m corner. So the correction AC-30 asks the integrator to write into base.yaml would install a NEW understatement, by 1.11× (Sabine) and 2.20× (Eyring), on exactly the split whose realized shortfall is 92.5-100 %. | State the range as computed over the full geometry × material product, both variants: Sabine [0.41, 5.71] m, Eyring [0.42, 11.41] m, max at shoebox 12×10×5 m, α = 0.98 (`ceiling_absorptive`). Have the corner test READ base.yaml's `geometry_families` / `material_regimes` instead of hardcoding (3.0,3.0,2.4)/0.05 and 0.80 — the current test claims "base.yaml's shoebox family at its own extremes" while pinning literals, so widening a declared range silently invalidates the claim without failing. |
| AC-47 | acoustics-reviewer | minor | OPEN | `src/amcd/scenes/generator.py:338, 347-348, 357, 359-360, 371, 381, 385-386` | **TWO REVERBERANT-FIELD RADII ON INCOMPATIBLE ABSORPTION-AREA CONVENTIONS SIT SIDE BY SIDE IN ONE RECORD.** `critical_distance_m` uses the Hopkins–Stryker room constant R = Sα/(1−α); ISO's d_min descends from the Sabine absorption area A = Sα. Their ratio is d_min/r_c = 2·sqrt(16π/24ln10)·sqrt(1−α) = 1.907·sqrt(1−α), so the standard's "d_min ≈ 2× the reverberation radius" rationale holds only below α ≈ 0.725 and INVERTS above it. MEASURED, S = 460 m²: α 0.05 → d_min 1.29 m vs r_c 0.69 m (1.86×); α 0.90 → d_min 5.47 m vs r_c 9.08 m (**0.60×**); α 0.98 → 5.71 m vs 21.18 m (0.27×). The `ceiling_absorptive` split therefore publishes an ISO "minimum measurement distance" that lies deep INSIDE its own reported critical distance, and the two derived flags `receiver_inside_critical_distance` / `below_iso_min_distance_sabine` swap strictness at α = 0.725 with nothing in the record saying why. Neither number is wrong under its own definition; the record states neither definition. | Name the absorption-area convention in the comment for each key (`critical_distance_m`: R = Sα/(1−α); `iso_min_distance_*_m`: ISO's A = Sα), and note the 1.907·sqrt(1−α) relation so a reader is not left to infer a factor-2 that only holds at low α. Belongs beside S-1's `min_measurement_distance()` in `acoustics.py`. |
| AC-48 | acoustics-reviewer | minor | OPEN | `tests/test_scene_placement.py:685-690` (`test_eyring_is_the_stricter_criterion_at_high_absorption`); docstring claim at `:666-669` and `src/amcd/scenes/generator.py:27-31` | **THE EYRING d_min — the variant that carries the headline shortfall — is pinned only by an INEQUALITY.** The Sabine variant has three known-answer corners; Eyring has `eyring > sabine`, which at α = 0.80 is satisfied by ANY denominator < 111 in place of 24·ln10 (the true value is 55.26), so a wrong constant or a wrong absorption term in that one line passes. The reported 100 % below-d_min on `test_material_shift` rides on it. Separately, the volume-independence and c-independence claims (asserted in both the module constant's docstring and the corner test's docstring) are stated and never tested. | Three assertions: (a) `iso_min_distance_eyring_m` at (12,10,5) m, α 0.80 == 7.32 m (abs 0.005); (b) two rooms with EQUAL surface and different volume — e.g. 12×10×5 (S=460) and a non-cuboid-equivalent pair matched on S — give identical d_min; (c) d_min == 2·sqrt(V/(c·T60)) recomputed from the same call's `t60_eyring_s` at c = 24·ln10/SABINE_K, to within the AC-45 rounding. |
| AC-49 | acoustics-reviewer | minor | OPEN | `src/amcd/scenes/generator.py:288-306`, `:469-480` | F-71's omission is **acoustically correct** (see the pass note below), but it leaves a residual with no guard: a `characterization: none` scene is still RENDERED into a record of fixed `ir_duration`, and after F-71 nothing checks its truncation at all — a non-enclosure has a finite decay too, it merely is not Sabine's. The per-scene `uncharacterized_reason` enumerates T60/R/r_c/DRR as undefined but never names record length, and the per-split `uncharacterized_note` speaks only of "these fractions". Not reachable today (no shipped config declares a `none` family, verified) — a guard/known-answer test owed before one does, not a defect now. | Add "and its record-length adequacy is therefore UNCHECKED" to `uncharacterized_reason`, and a known-answer test that a mixed enclosure/non-enclosure config reports the non-enclosure count as unchecked rather than merely excluded. |

**Confirmed correct, no finding (stated so it is not re-litigated):**

- **The §5.3 formula and BOTH reductions.** d_min = 2·sqrt(V/(c·T)) is ISO 3382-1
  §5.3 as written. Substituting Sabine (T = KV/(αS), K ≡ 24ln10/c) gives
  2·sqrt(αS/(24ln10)); substituting Eyring (T = KV/(−S·ln(1−α))) gives
  2·sqrt(−ln(1−α)·S/(24ln10)). Both verified against an INDEPENDENT numeric route
  — 2·sqrt(V/(c·T60)) with the T60 the code itself reports at c = 343 — over 5
  rooms × 5 α (0.05…0.98): agreement to −0.0353 % everywhere, constant across V,
  S and α, i.e. exactly the AC-45 rounding and nothing else. Volume cancellation
  and c cancellation each confirmed independently (the residual is invariant in V).
- **Both variants reported.** Correct. Eyring is stricter at ALL α, not only high
  α (ratio 1.005 at α 0.02 → 1.418 at 0.80 → 1.998 at 0.98) because
  −ln(1−α) > α, and a shorter T60 gives a larger d_min. The code's comment claims
  only "disagree substantially at high α", which is accurate.
- **Omitted, not zeroed, for `characterization: none`.** Acoustically right: d_min
  is a functional of the diffuse-field T60 of an enclosure, so for a non-enclosure
  there is no V/(cT) to evaluate — a 0.0 would read as "no minimum distance
  applies" and a NaN would poison `_summarize`. Omission + `_scene_is_characterized`
  + `_summarize`'s `if any(key in r)` guard keeps it out of numerator, denominator
  and per-split summary alike; `test_an_uncharacterized_split_reports_no_below_d_min_number`
  pins all three.
- **F-71's treatment of a non-enclosure.** Correct. Sabine T60 is undefined for a
  non-enclosure, so `t60_exceeds_ir_duration` is unscorable, not False; scoring it
  False admitted an unscored quantity to the gate denominator as passing. The
  omission + RD-94's UNSCORED warning is the right pair.
- **Units.** Every new key declares its unit in its name (`_m`), consistent with
  `_s` / `_db` / dimensionless `d_over_rc`. No finding.
- **S-1 (AC-24 home) confirmed acceptable for one cycle** on the evidence: grep
  over `src/` shows NO other consumer of `iso_min_distance*` / `below_iso_min*`
  and no other d_min computation, so there is no second implementation to diverge
  from — except the constant itself, which is AC-45.

---

## falsifier pass on lane/S-cycle4 @ 5e15293 (SELF-CHECK, not a clean pass)

Two of the lane's claims refuted; the rest held. What it could NOT break, stated
because a clean bill is worth nothing without the attempt: the `.get`
contract behind F-71's denominator is airtight (`_flag_counts` emits the key iff
nonzero over a non-negative int, and no non-test consumer of
`t60_exceeds_ir_duration` or of `placement_report.json` exists anywhere in `src/`
or `scripts/`); population neutrality holds (zero diff lines touching
`rng|seed|integers|uniform|_sample_*|_generation_plan`, and a failed gen-scenes
leaves no sentinel, which `pipeline.py:442-449` refuses to render past); the RD-65
warning cannot be suppressed; no split can vanish from either D0 artifact; and the
new tests bind rather than being tautological.

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| S-F1 | falsifier | major | OPEN | `src/amcd/diagnostics/probe.py` D0b verdict loop | **F-72 CLOSED THE FILE-MISSING DROP AXIS ONLY; PER-METRIC NaN ATTRITION IS STILL INVISIBLE AND STILL CANNOT MOVE `all_clear`.** `_verdict` degrades to N/A only at `n == 0`, and the new coverage bracket is keyed on `len(dropped)`, so a metric averaged over a subset of SCORED scenes prints as a bare number with `(PASS)`. CONFIRMED on the canonical dry run: `train` reports `n_scenes=12 n_attempted=12 dropped=0` while `T30 n=11` and `C50 n=10`, and the stage prints `CARRIER CEILING CLEARS … Proceed to E1` with no indication of either. The comment above the verdict states the contract as "any missing metric is insufficient coverage"; the code implements "a metric with ZERO scenes". Sub-defect: `t30_thresh` scales a nanmean over SCORED scenes while `t30_r` averages over NON-NaN-RESIDUAL scenes — two populations on the two sides of one ratio test. | Not fixable in lane S: the remedy needs a config-declared minimum coverage (`configs/base.yaml`, lane M) so the verdict is a function of coverage rather than of a hardcoded threshold. Print per-metric `n`. Test: per (split, metric), `n == n_scenes == n_attempted` or the verdict is INDETERMINATE. |
| S-F4 | falsifier | minor | OPEN | `src/amcd/scenes/generator.py` `_disclose_and_gate_record_length` | The gate iterates every top-level key of `report` and indexes `entry["t60_over_ir_duration"]` unconditionally, but `report` IS the `placement_report.json` artifact. A future non-split metadata key — which AC-30's own disclosure work invites — raises KeyError (probed). Loud rather than silent, hence minor. | Guard the shape, or state in the docstring that every top-level key is a split. |
| S-F5 | falsifier | minor | OPEN | `src/amcd/scenes/generator.py` gate denominator; `tests/test_scene_placement.py` RD-97 pin | The derived denominator's interesting case, `0 < n_uncharacterized < n`, is unreachable from any config today — one geometry family per split makes characterization all-or-nothing — so RD-97's pinning test only ever sees 0 or n. The falsifier probed the mixed case directly and it is CORRECT (4 sabine + 6 none → derived 4 == published 4, gate raised at 4/4), but nothing guards it, and the roadmap's outdoor families make it reachable. | Unit test on `_flag_counts` + the gate with a hand-built mixed `room_stats` list. |
| S-F6 | falsifier | minor | OPEN | `src/amcd/scenes/generator.py` per-split warning loop | The gate says nothing about a declared split with ZERO scenes (`attempted == 0` is skipped), while `probe.py` warns for the analogous case. Reachable: `scenes.n_id: 0` is accepted and yields `id: n_scenes 0`, silent in the gate. | Warn on any declared split with zero attempted scenes. |
| S-F7 | falsifier | minor | OPEN | `src/amcd/scenes/generator.py` `_disclose_and_gate_record_length` docstring | RD-65's warning is per generation-plan REGIME, not per declared split: in frac mode `train`/`valid`/`test_id` are pooled under `id` (canonical run keys: `id, test_material_shift, …`). Genuinely per split for `research_i` (count mode), so RD-65's own scenario is covered. Structural — split assignment happens at preprocess, not here — but the docstring says "per split" without saying which split set, and a reader will read `id` as `train`. | One docstring clause naming the split set the warning ranges over. |

## readability-reviewer pass on lane/S-cycle4 @ 5e15293 (SELF-CHECK, not a clean pass)

**RR-32 (generator half) and RR-44 both confirmed closed as specified.** No new
measured constants were introduced in this cycle's prose. Remaining rows below;
RR-46/47/52/53/54 are addressed in the follow-up commit, RR-48/50/51 are
pre-cycle-4 prose in lane-S files left for a later cycle, RR-55 is lane P's.

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| RR-48 | readability-reviewer | minor | OPEN | `src/amcd/scenes/generator.py` `_room_acoustics` docstring and the AC-29 flag comment | Exactly RR-32's charge, in the same module but predating cycle 4: the docstring pins "failed for 100 % of scenes (alpha median 0.894, Sabine/Eyring ratio median 2.51, 23 % with r_c larger than the room's longest dimension)" to a realized config, repeats "median 2.51 … max 3.90" nine lines later, and carries a 9-line AC-29 bug report ("MEASURED over the realized base.yaml set: 92.5 % of test_material_shift and 17.2 % of id … an under-report of ~2.6x"). The 2.51/23 % pair is a third copy of `configs/research_i.yaml`'s comment. | Cut each to the rule plus AC-21/AC-29; let the surviving copy of the measured percentages be the config comment, closest to the config state it describes. Not done this cycle — outside RR-32's stated scope, and it is the kind of edit that should be one row, not a drive-by. |
| RR-50 | readability-reviewer | minor | OPEN | `src/amcd/diagnostics/probe.py` D0b enumeration comment | Same shape, also pre-cycle-4: 8 lines of F-45 reproduction transcript plus a note that the branch below "was DEAD CODE" — a statement about an implementation that no longer exists, sitting above the branch it now describes as live. Separately the `all_clear` point is made three times. | Cut to two lines (enumerate config-declared splits in declaration order; an undeclared split present in the data is appended, F-45) and keep the `all_clear` rule once, at the verdict loop that consumes it. |
| RR-51 | readability-reviewer | minor | OPEN | `src/amcd/diagnostics/probe.py` — the per-split record schema | The record shape this cycle extended (`n_scenes` = SCORED, `n_attempted`, `dropped: [{scene, reason}]`, `unscored_reason` present iff `n_scenes == 0`) is constructed at six sites and declared at none. The two consumers disagree about whether the last key is guaranteed — D0a indexes `info["unscored_reason"]` directly while D0b defends with `.get(..., 'no scenes')` — so a reader cannot tell which is the contract. The artifact is read across lanes (`tests/test_dataset_integrity.py`), which is where an undeclared schema costs most. | Declare the four keys and the emit-iff invariant once in `probe.py`'s module docstring, or factor an `_unscored_split(...)` helper both probes call; make the two consumers agree. |
| RR-55 | readability-reviewer | minor | OPEN | `src/amcd/config.py` `GeometryFamily.characterization` docstring | Says `"none"` means `_room_acoustics` records "a (split, reason) instead of a number". The record is per SCENE (`uncharacterized_reason` on the scene dict); split-level counts are a later aggregation in `_flag_counts`. A reader following this sentence looks for the wrong unit. **Lane P's file — not lane S's to fix.** | "(scene, reason)". |

---

## What I corrected in response (follow-up commit)

Three of the findings above were errors in this lane's own work, not observations
about it. All three are fixed; the rest are recorded, not actioned.

1. **A crash I introduced exposure to, on the exact config RD-94 exists for.**
   `_disclose_and_gate_record_length` formatted `corner['t60_sabine_s']:.2f`
   unconditionally, but `Config.worst_case_t60` returns a reasoned `None` for a
   config where no family declares `characterization: sabine`. So on an
   all-uncharacterized config, gen-scenes raised
   `TypeError: unsupported format string passed to NoneType.__format__` **before**
   RD-94's "UNSCORED, not passed" warning could be emitted — the warning was
   unreachable on the only config that triggers it. My RD-94 test built the report
   by hand and so never exercised the real path. Fixed; the corner is now disclosed
   as UNSCORED with its reason. New test
   `test_the_same_config_survives_the_real_generation_path` drives the real
   `run_gen_scenes` path. (Flagged by `readability-reviewer` as an aside, outside
   its own remit.)

2. **AC-30's declared-support range was wrong, and I had transcribed it into the
   residual queued for lane M.** Corrected above: Sabine [0.41, **5.71**] m, Eyring
   [0.42, **11.41**] m. The corner test now derives from the `Config` object
   instead of restating literals, which is what let the error through
   (S-F2 / AC-46 / S-F8).

3. **The "free of the speed of sound" claim was overstated.** True of the formula,
   false of the constant: `SABINE_K` ships rounded, so `24·ln10` assumes
   c = 343.24 m/s and biases d_min by a constant −0.035 %. The constant's docstring
   now states this rather than claiming independence (AC-45 / S-F3).

Also applied: the readability trims for RR-46/47 (duplicated rule statements in
`_disclose_and_gate_record_length` and `_flag_counts` cut back to one home each),
RR-52 (test module docstring lists the new rows; the section banner is topical
rather than cycle/lane-stamped), RR-53 (`test_probe.py` cites
`docs/review_ledger.md` rather than this transient inbox) and RR-54 (the generator
module docstring now names `placement_report.json` and the gate).

**Not actioned, and why:** S-F1 needs a config-declared coverage floor in
`configs/base.yaml` (lane M) — no hidden defaults, so it cannot be fixed inside
lane S. AC-47/AC-48/AC-49, S-F4..S-F7 and RR-48/50/51 are recorded for a later
cycle rather than swept into a lane whose six assigned rows are done; AC-49 and
S-F5 are both unreachable from any shipped config today.
