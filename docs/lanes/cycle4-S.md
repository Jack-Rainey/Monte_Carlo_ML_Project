# Lane S brief — scenes, QC and diagnostics (cycle 4)

Your rows share a theme: **something leaves a result silently.** The project rule
is that nothing does — every drop, skip and NaN is logged as `(unit, reason)`,
with scored-vs-attempted counts, and an unscored quantity is never rendered as a
number. Each of your findings is a place that rule is broken.

Row text is in `docs/review_ledger.md` — read it there; the ownership hook will
refuse a write to it.

## Order

1. **F-71.** The `characterization: none` branch sets `t60_exceeds_ir_duration`
   to `False` instead of omitting it, so uncharacterized scenes enter the
   record-length gate's denominator as passing. Reproduced: a declared
   `openfield` family reports `n_scenes: 0, n_uncharacterized: 3` in the
   `diffuse_field_validity` block while the sibling `t60_over_ir_duration` block
   reports `n_scenes: 3, count: 0, fraction: 0.0`. N uncharacterized scenes shrink
   the over-limit fraction by N/(N+M) — a dataset whose enclosed scenes breach the
   limit can pass by adding non-enclosures. Omit the flag so
   `_scene_is_characterized` excludes them from both the fraction and the gate.
   Note the same helper honours the rule twenty lines above, which is your model.
2. **RR-44** immediately after — it renames that exact helper
   (`flag_key_present` → `_scene_is_characterized`, moved above `_flag_counts`,
   one-line docstring naming the `sabine` vs `none` distinction). Same function,
   same diff neighbourhood; splitting them means touching it twice.
3. **F-72 — and read this before you start it.** F-45's existing probe test class
   `TestD0bEnumeratesDeclaredSplits` lives in `tests/test_dataset_integrity.py`,
   which is **lane P's**. Do not touch it. A new S-owned `tests/test_probe.py`
   has been pre-declared for you — it does not exist yet, and the ownership hook
   allows you to create it precisely because it is declared (RD-83). Probe tests
   living in two files is a one-cycle cost; note it in your inbox so a later
   cycle can consolidate them.

   Both D0a and D0b `continue` past a scene with missing tensors or
   `renders/<id>/high.npy` without logging `(scene, reason)`, then `continue` past
   a split whose scenes all failed — so the split vanishes from the artifact and
   D0b's `all_clear` stays True over a split it never measured. This is F-45's
   defect on a different axis: F-45 fixed the split enumeration, the per-scene
   drops are still silent. Mirror the eval drop log, report
   `n_scored`/`n_attempted`, and give an all-failed split an `unscored_reason` so
   it reaches the verdict table as INDETERMINATE — never a pass.
4. **RD-65 — the WARNING half only.** The AC-22/RD-56 gate is the OVERALL
   over-limit fraction, the one aggregation invariant #9 forbids for results:
   research_i's 0.01 over 720 scenes permits 7 over-limit scenes that could all
   sit in the 30-scene `test_geometry_shift` (23 % of that split) and still pass.
   **Keep the overall gate** — its reasoning is right, a per-split gate lets the
   smallest split set the tolerance for train. Add an always-emitted per-split
   WARNING naming any split whose own fraction exceeds the declared limit.
   The row's other half — carrying per-split counts into the E1 report table —
   needs `reporting/tables.py`, which is lane P's, so it is on the integrator's
   queue (RD-82). Stop at the warning.
5. **AC-30.** Disclosure only, no behaviour change. Correct base.yaml's third
   d_min number to a real corner and state the full [0.41, 5.16] m range; record
   the realized below-d_min fraction per split in `placement_report.json` beside
   the existing `d_over_rc` summary. **`configs/base.yaml` is lane M's**, so write
   the config-comment half into `docs/ledger_inbox/S.md` for the integrator and
   implement only the `placement_report.json` half yourself.
6. **RR-32** last. Cut two docstrings back to the contract plus the ledger id:
   `_check_regimes_clear_backend_floor` embeds a measured probability describing a
   config that can no longer be produced (base.yaml now declares a 1.0 m minimum),
   and `_check_split_roles` replays the F-44 reproduction transcript. Both facts
   are already in the rows they cite. `_disclose_and_gate_record_length` and
   `_sample_positions` are the right altitude — match them, do not touch them.

## Pass condition

Every row here is about a count or a reason string, so the evidence is the
artifact, not the exit code: show the relevant block of `placement_report.json`,
`d0a_gap.json` or `d0b_oracle.json` before and after. For F-71 and F-72
specifically, the test must construct the *failing* population — a split of
`characterization: none` scenes, a scene with its tensors removed — and assert
the counts, because both defects are invisible on a healthy run.

**And one more, which is not obvious: your changes must be population-neutral.**
You own `scenes/**`, which decides which scenes are ADMITTED, and therefore the
population every reported number is computed over. The integration gate treats a
moved `ci_table.csv` as cross-lane interference, so if your work moves it the
detector cannot tell your legitimate change from a real defect (RD-91). Show an
unchanged fixed-seed `ci_table.csv` on the canonical dry run.

F-71 is the live risk: tightening `_scene_is_characterized` changes a gate
OUTCOME. If it turns out to change admission rather than only the reported
fraction, **stop** — that is an M-class change and belongs in the integrator
queue. Your other rows are disclosure-only and safe.

## Evidence

```
PYTHONPATH=<your-worktree>/src <env>/bin/pytest
PYTHONPATH=<your-worktree>/src <env>/bin/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

The prefix is mandatory — see `LANE.md`.

## Not yours

You own `scenes/**`, `diagnostics/**` and four test files — that is all. **The
simulators moved to lane R this cycle** (`base.py`, `render.py`,
`gsound_sir.py`), because R is building Steps 2+3 in them; `dry_run.py` is lane
M's. `configs/base.yaml` and everything under `evaluation/` and
`representations/` belong to M; `config.py`, `stats/**` and `reporting/**` to P.

Three rows on the integrator's queue touch your files: **AC-41** (the α clip
differing between `_room_acoustics` and `dry_run.render`), **F-60** (the
"realized" gate that gates estimates, whose fix needs a config key and an eval
count), and **AC-43**, which lands its disclosure in your `probe.py` but was
raised against lane M's `dry_run.py`. Do not start any of them. New work in
another lane's file goes to `docs/ledger_inbox/S.md`.
