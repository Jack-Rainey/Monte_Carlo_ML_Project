# Parallel lanes

How several Claude Code sessions work this repository at once without weakening
the two things that make the review loop trustworthy: **evidence before claims**,
and **a clean pass means reviewers ran over the current, integrated state**.

The reviewers were never the bottleneck — they are read-only and already run in
parallel. The serial part is the fix phase: one builder applying one fix at a
time, each gated on its own evidence run. That is what this splits.

A **lane** is a git worktree + a branch + an exclusively-owned set of files + a
subset of ledger rows + its own run dir. One lane, one directory, one session.
The main checkout runs no lane; it is the **integrator**.

---

## The five rules

1. **Ownership is by file, not by finding, and it is exclusive.** A lane may edit
   only the files in its declared set. Textual merge conflicts are then
   impossible by construction — no locks, no polling, no coordination between
   sessions. Enforced by `scripts/lane_guard.py`, not by discipline.

2. **The metric COMPUTATION path is always ONE lane.** Everything that can move a
   number in `ci_table.csv` — `evaluation/`, `representations/`, `acoustics.py`,
   `simulators/dry_run.py` — stays together. Each lane's pass condition is a
   fixed-seed `ci_table.csv` A/B, so two lanes both moving that table invalidate
   both A/Bs at the merge. This rule is what caps the achievable speedup, and it
   is not negotiable for a speedup.

   **The lane that computes a number does not own the file that reports it.**
   `ci_table.csv` is written by `stats/aggregate.py:324` and `summary.txt` by
   `reporting/tables.py:144`. Those live wherever the cache/provenance lane
   lives, because that is where their fingerprints are. So a finding whose fix
   adds a new reported COLUMN spans two lanes by construction, however
   metric-shaped it looks — it goes to the integrator queue under rule 4. Cycle 4
   found three of these (F-70, AC-43, RD-65) only because the reachability check
   below exists (RD-82).

3. **Lanes never edit shared-authority files**: `docs/review_ledger.md`,
   `CLAUDE.md`, `docs/design_spec.md`. One writer — the integrator. Lanes write
   closures and new findings to `docs/ledger_inbox/<lane>.md`, a file only that
   lane touches.

4. **A finding whose anchors span two lanes' files is not parallelized.** It goes
   to the integrator's serial queue, applied after the merge and before the
   reviewers. Declared per cycle under `integrator_queue:` in the partition file.

5. **Reviewers count only on the integrated tree.** A lane may run reviewers on
   its branch as a cheap self-check, but that never counts toward a clean pass.
   CLAUDE.md's definition of done is unchanged.

---

## Setting up a cycle

Declare the partition in `docs/lanes/<cycle>.yaml` — lane ids, titles, owned
paths, assigned rows, and the integrator queue with a reason per row. That file
is the single source of truth: `scripts/new_lane.py` generates each worktree's
`LANE.md` (what the session reads) and `.claude/lane.json` (what the guard
enforces) from it, so the two cannot disagree.

Each row declares the `fix:` and `test:` paths it will actually touch.
`tests/test_lane_partition.py` then asserts:

- no path is owned by two lanes, and no lane claims a shared-authority file;
- **every row's `fix` and `test` paths fall inside its own lane's owned set** —
  so a row that cannot be finished in its lane fails at declaration time rather
  than halfway through a session;
- every row id appears in exactly one of the four lists (lane rows,
  `integrator_queue:`, `awaiting_re_review:`, rows raised against the partition);
- every declared brief exists.

The reachability check is the one that earns its keep. Non-overlap alone does not
give it: a row can sit in the lane that owns the code it *describes* while its
fix or its test lands elsewhere. Cycle 4 shipped with F-72 assigned to lane S and
its test class in a file lane P owns — the hook would have refused the edit
mid-session (RD-83).

Then:

```
python scripts/new_lane.py --partition docs/lanes/cycle4.yaml --all
```

This creates the worktrees as siblings of the main checkout (`v3` → `v3-lane-M`),
branches them from `v3-rebuild`, writes each lane's identity and per-worktree
`.claude/settings.local.json`, and prints the `cd` lines. It does **not** launch
sessions.

Open one terminal per directory and start Claude in each. Order and timing do not
matter, and you do not have to tell a session which lane it is:

```
cd /Volumes/T7/Monte_Carlo_Research/v3          # no LANE.md -> integrator
cd /Volumes/T7/Monte_Carlo_Research/v3-lane-M
cd /Volumes/T7/Monte_Carlo_Research/v3-lane-P
cd /Volumes/T7/Monte_Carlo_Research/v3-lane-S
```

**One session per worktree.** Two sessions in `v3-lane-M` would both correctly
identify as lane M and then collide with each other — the hazard lanes exist to
remove, reintroduced inside a lane. For more parallelism, add a lane.

---

## How a session knows which lane it is

Four independent signals, so no single mechanism has to be right:

| Signal | Where it comes from |
|---|---|
| The directory | `v3-lane-M`; the session's cwd is in its context from the first turn |
| `LANE.md` | Generated per worktree, gitignored; CLAUDE.md tells every session to read it first. Absent = integrator |
| The branch | `lane/M-cycle4` |
| The imported source | `scripts/lane_preflight.py` prints all four together |

Run the preflight before the first edit and paste its output. If the four
disagree, stop rather than work.

---

## Evidence isolation — the trap this exists for

The project is installed with `pip install -e .`, and
`site-packages/__editable__.amcd-0.1.0.pth` contains **one absolute path**: the
main checkout's `src/`. A lane worktree that runs a bare `pytest` or `amcd`
therefore imports the *main tree's* modules while editing its own, and every
number it reports describes code it did not change. Nothing errors. The suite
passes. The evidence is simply about the wrong tree.

Prefix every command in a lane:

```
PYTHONPATH=<worktree>/src <env>/bin/pytest
PYTHONPATH=<worktree>/src <env>/bin/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

`tests/test_source_tree_isolation.py` asserts the imported `amcd` resolves inside
the checkout the tests live in, so a lane that forgets the prefix fails on its
first `pytest` — before any evidence is gathered. In the main checkout it passes
trivially.

Run artifacts need no such care: `_make_run_dir` returns a path relative to the
working directory and `experiments/` is gitignored, so each lane gets its own run
dirs and its own stage cache for free.

---

## Ownership enforced by the harness

`scripts/new_lane.py` writes a **PreToolUse hook** into each worktree's
`.claude/settings.local.json` (gitignored, so it never reaches another checkout)
that runs `scripts/lane_guard.py` before every `Edit` and `Write`. The guard
reads `.claude/lane.json` and denies:

- paths inside another checkout of this repository — an absolute path is the one
  way a lane could reach around its own worktree;
- paths inside this checkout that are not in the lane's owned set, including the
  shared-authority files.

It allows paths outside the repo entirely (scratchpads, temp files — not shared
state), and it allows everything when there is no `.claude/lane.json`, which is
how the integrator's main checkout behaves. It fails **open** and says so on
stderr: a guard that crashes should not also stop real work.

---

## Keeping lanes off stale code

- **The integrator does not edit lane-owned files while lanes are live.** Its
  serial queue — spanning rows, docs, ledger — runs after lanes report and before
  the reviewers. Ownership holds in both directions, not just downward.
- **When the integrator lands anything on `v3-rebuild` mid-cycle it says so, and
  every live lane merges `v3-rebuild` into its branch before continuing.**
- **A sync invalidates evidence that predates it.** If the incoming change
  touches anything the lane imports, the lane re-runs its pass condition before
  reporting. Evidence is only valid against the tree it was measured on — the
  same rule the stage cache enforces on the pipeline, applied to the session.
- **Lanes report on a tip that already has `v3-rebuild` merged in.** That is what
  keeps the common case a fast-forward.

---

## The integration gate

Serial, in the main checkout, after the lanes report:

1. Merge lane branches into `v3-rebuild`, one at a time.
2. Apply the integrator queue (rule 4).
3. Full suite on the merged tree — one run, not three.
4. Canonical dry run, and a fixed-seed `ci_table.csv` A/B against the baseline
   captured before the lanes started.
5. Fold `docs/ledger_inbox/*.md` into the ledger — new findings become rows,
   closures get "fix applied, awaiting re-review". **Delete nothing yet.**
6. Run the four reviewers on the merged tree. **This is the pass that counts.**
7. **Now** delete the rows that pass confirmed clean.

Steps 5 and 7 are separate on purpose. CLAUDE.md deletes a row only when it is
fixed AND re-review-confirmed; deleting at step 5 would authorize deletion on the
lane's own say-so — self-grading, and irreversible in working memory (RD-84).
Folding must still happen *before* step 6, because the reviewers need to see the
new rows.

**Steps 3-4 are mandatory whenever the merge is not a fast-forward.** Lane
*count* is the wrong test for whether a combination was checked: a single lane
merging on top of integrator commits is exactly as unverified as two lanes
merging into each other. What matters is whether anything changed underneath the
lane, and `git merge` already answers that. If it fast-forwards, the lane's own
evidence stands. If it creates a merge commit, re-run 3-4 and fix what the
combination broke — rule 1 makes textual conflicts impossible, so a merge commit
here means semantic overlap, which is exactly the case that needs the run.

Step 4 is also the cross-lane interference detector. Only the metric lane may
have moved `ci_table.csv`; if another lane moved it, that is a defect to
investigate, not a merge artifact.

---

## Planning the next cycle's partition

This is meant to be run every cycle, not just once. The partition is the only
part that changes; the machinery, the guard and the tests are cycle-agnostic.

1. **Start from the gate, not the ledger.** Ask what the current gate needs and
   give that a lane FIRST. Cycle 4's partition was drafted from the ledger alone
   and scheduled none of the declared Steps 2+3 content; the ledger rows it did
   schedule could have all closed without moving RD-33a, because that gate needs
   a probe that unbuilt code cannot run (RD-81). **A cycle whose lanes cannot
   move the gate is a cycle that ends where it started, with a smaller ledger.**
2. **Bucket the OPEN rows into four lists** and write all four down: lane rows,
   `integrator_queue:`, `awaiting_re_review:`, and anything raised against the
   partition itself. Print the arithmetic in the yaml header. A row in no list is
   the silent omission RD-73 exists to prevent; a row in two is two plans.
3. **Exclude "FIX APPLIED, awaiting re-review" rows from every lane.** They are
   confirmations the integration reviewer pass produces. Assigning one invites a
   second fix stacked on a first that nobody checked.
4. **Draw ownership by file, then check the rows fit it** — not the reverse. For
   each row write the `fix:` and `test:` paths first; if they cross a lane
   boundary the row is a rule-4 spanning row, and no amount of redrawing lanes
   makes it otherwise.
5. **Balance on work, not row count.** Lane R exists because folding a new schema
   and a subprocess worker into six QC rows would have made that lane the
   wall-clock floor — not because "render" is a different topic.
6. **Run `pytest tests/test_lane_partition.py` before creating any worktree.** It
   is seconds, and it is the difference between finding a bad partition now and
   finding it three sessions in.

Then commit, run `scripts/new_lane.py`, and open the terminals.

## What this does and does not buy

Expect roughly **1.8-2×** on the fix phase, not 3×, and **less than that end to
end**. Rule 2 forbids splitting the metric lane, so it sets the wall-clock floor.
Splitting it into `representations/` and `evaluation/` halves becomes possible
once the gate has an attribution scheme for two lanes moving `ci_table.csv` —
which does not exist yet, and is not worth inventing before the protocol has run
a cycle.

The end-to-end figure is lower because **integration does not parallelize**. In
cycle 4, four lanes' worth of fresh change plus 22 awaiting-re-review rows all
land on ONE reviewer pass. Given this project's reviewer finding-rate, budget for
that pass to take about as long as a serial cycle's fix phase. The speedup is
real; it is a speedup of one phase.

`configs/base.yaml` is the known contention point: several lanes' fixes want to
declare a config key in it. It is owned by the metric lane, and a row whose fix
is mostly a config declaration elsewhere goes to the integrator queue instead
(cycle 4: RD-18, F-60). That is a real cost of the partition, recorded rather
than worked around.
