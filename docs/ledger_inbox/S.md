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

**The correction to apply, with the arithmetic re-derived and re-verified here.**
d_min = 2·sqrt(V/(c·T60)); substituting Sabine's T60 gives
d_min = 2·sqrt(αS/(c·K)), and since `SABINE_K ≡ 24·ln10/c` the product is
`24·ln10` — so d_min is independent of BOTH volume and the speed of sound. The
shoebox family's real corners:

| corner | dims | α | d_min |
|---|---|---|---|
| smallest, least absorptive | 3.0 × 3.0 × 2.4 | 0.05 | **0.41 m** |
| largest, least absorptive | 12.0 × 10.0 × 5.0 | 0.05 | 1.29 m |
| largest, most absorptive | 12.0 × 10.0 × 5.0 | 0.80 | **5.16 m** |

So the declared support spans d_min ∈ **[0.41, 5.16] m** and the declared 1.0 m
floor sits near its BOTTOM, not "inside the band". All three reproduce to 2 d.p. in
`TestIsoMinimumDistanceDisclosure::test_the_declared_support_corners_reproduce`.

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
382 passed in 51.71s          (362 before this lane's work; +20 new)

$ PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-S/src \
    /Users/nortonrainey/miniconda3/envs/amcd/bin/amcd all \
      -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml \
      -c configs/overlays/dry_run.yaml
[done] gen-scenes / render / preprocess / diagnostics / train / infer / eval / stats / report
```

No reviewer has been run over this lane's finished diff. The
`research-director` pass recorded above ran on the PLAN, before implementation,
and is a self-check on `lane/S-cycle4` either way — not a clean pass.
