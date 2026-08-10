# Lane M brief — the metric path (cycle 4)

You own everything that can move a number in `ci_table.csv`. That is why this
lane is not split: your pass condition is a fixed-seed A/B of that table, and a
second lane moving it concurrently would invalidate both A/Bs at the merge.

Row text lives in `docs/review_ledger.md` — read your rows there. You may read
the ledger freely; you may not write it (the ownership hook will refuse). This
brief carries what the ledger does not: order, pass conditions, and the traps
this file has already sprung twice.

## AC-37 — the remedy is DECIDED; do not re-open it

`min_db` acts as an injected energy floor in the decoded pred: an oracle
prediction — `decode(encode(high), low)`, definitionally perfect — reads T30
**+24.4 % at −30 dB of IR level and +935 % at −40 dB**, with a `min_db: -200`
control clean at ≤ 0.38 %. It is inert under dry_run, so it would first appear on
the emulated gsound render, where it will look like a model failure.

**User decision, 2026-08-10: remedy (a) guard + (c) sweep.** Not (b) — a
per-scene-relative decode floor changes the decode contract, i.e. the Research II
energy-domain reframe itself, and would move every decoded waveform on top of
your other metric changes, muddying the single A/B you are judged on (RD-86).

**Only the guard is yours.** `min_db` lives entirely in `spectrogram.py`, so the
guard belongs in `encode`, failing loud when a scene's per-band peak sits closer
than a config-declared N dB to `min_db`. It must NOT go in `data/preprocess.py` —
that is lane P's file, and putting it there makes this row unfinishable in your
lane. The (c) D0b level sweep needs `diagnostics/probe.py` and is on the
integrator's queue.

Write the known-answer test first; it is remedy-independent, cheap, and needs no
render, so the guard lands as an isolated A/B against a measured baseline:

> `decode(encode(high), low)` must reproduce high's T30 within `d0b_t30_jnd_frac`
> at gain ∈ {0, −20, −30, −40} dB.

## Order

1. **AC-37's known-answer test** (no decision needed — it measures the defect).
2. **Readability rows first among the rest** — RR-37, RR-38, RR-45 are docstring
   and comment work in the same files as the physics rows. Doing them first
   means the physics diffs land in already-tidy files, and RR-38's missing ledger
   id is now AC-36, which you can cite immediately.
3. **AC-38, then AC-39, then AC-27's remainder.** AC-38 changes the resolvability
   OPERAND (decide from a leg-independent quantity — `IRResult.meta["rt60_s"]` or
   the report's Sabine/Eyring T60 — or count-and-disclose instead of suppress).
   AC-39 rewords the drop reason that AC-38 leaves behind. Doing AC-39 first
   would word a reason for a decision you are about to change.
4. **AC-42** (C50 has no resolvability guard and no `pred_unresolved` entry; a
   degenerate pred currently reports +203.4 dB as a scored absolute).
5. **F-68** (rewrite the KNOWN RESIDUAL paragraph to describe the fold and state
   the leg asymmetry as measured; overlaps RR-38, so do them together).
6. **AC-28 — verify and report, do not re-fix.** The cycle-3 resume note records
   it as closed (C50 flat to 0.02 dB over 16× distance → 10.5 dB monotone) but
   the row was never deleted. Check the current code; if it is done, say so in
   your inbox with the evidence, and the integrator deletes the row.

**F-70 and AC-43 were reassigned away from this lane** (RD-82). Both look like M
rows and are not: F-70's imputed-population CI is a new column in
`stats/aggregate.py` and AC-43's disclosure has to reach the D0a/D0b artifacts in
`diagnostics/probe.py` — neither file is yours. They are on the integrator's
queue. If you find yourself needing a new reported column, that is the same trap:
write it to your inbox.

## Pass condition — this lane's is stricter than the others'

A change to `_band_energy` moves **every** reported ISO number. This file has
already produced two changes that were applied and defended before a reviewer
caught them: the `padtype="odd"` reflection (AC-36) and then the single-sample
fold (F-67), whose EDR step moved paired EDT improvement 77 % and deleted a
split's MDES while the evidence offered was entirely C50.

So: **a fixed-seed `ci_table.csv` A/B across all metrics and all splits, with
n_scored and MDES per split — not a known-answer probe on one metric.** Capture
the baseline from your branch point before your first metric change.

## Evidence

```
PYTHONPATH=<your-worktree>/src <env>/bin/pytest
PYTHONPATH=<your-worktree>/src <env>/bin/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

The prefix is mandatory — see `LANE.md`. `tests/test_source_tree_isolation.py`
fails if you forget it.

## Not yours

`scenes/generator.py` is lane S's; `pipeline.py`, `config.py`, **and crucially
`stats/aggregate.py` and `reporting/tables.py`** are lane P's. That last pair is
the one that catches people: they write `ci_table.csv` and `summary.txt`, so
every reported column lives in P even though the numbers in it come from your
code (RD-82).

`simulators/base.py` and `gsound_sir.py` are lane R's this cycle —
`simulators/dry_run.py` is still yours.

AC-41 (the α clip differing between the room described and the room rendered)
spans your files and S's, so it is on the integrator's queue — do not fix it,
even though `acoustics.py` is yours. Same for RR-43. If you find new work in
another lane's file, write it to `docs/ledger_inbox/M.md`.
