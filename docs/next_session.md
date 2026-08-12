# Next session — one integrator, no lanes

Partition: `docs/lanes/cycle6.yaml` (`lanes: []`). Read `docs/review_ledger.md`
for the rows; this file says what to do with them and in what order.

**Why not parallel.** Cycle 5 ran four lanes, produced 144 findings and moved the
gate by zero. The binding constraint is no longer throughput — it is that the
work that matters spans every lane boundary at once. ITEM 0's remainder touches
`simulators/`, `scenes/` and `configs/` together, so no partition can host it.

---

## 1. Finish ITEM 0 (cluster C6). Everything else waits.

`AC-54` is done — the absorption convention is declared on the backend and
confirmed by render (`as_is` 1.841× Eyring, `pre_compensate` 0.971×). The
remainder is not:

- **`AC-184`** *(blocker)* — the record-length problem is NOT the compiled 3.0 s
  cap. No rendered scene came near it. Upstream's adaptive energy trim binds
  first, and it grows far slower than the decay: T60 swept 45×, the record 2.9×,
  so captured decay collapses 436 → 172 → **27.9 dB**. For the largest declared
  room the record is *shorter than the T30 it is measuring*. A declared constant
  record length does not fix this — the deficit is a function of T60.
  **The gate's denominator must become the realized per-scene support, read back
  from `native_ir_samples`.**
- **`AC-185`** — the trim length moves with the ray budget (1.951 s → 2.057 s for
  40×), so absolute ISO metrics inherit a dependence on the axis E4 sweeps.
  Resolve with cluster C5.
- **`AC-55`** — `diffuse_depth` is physically a time bound; declare it as one.
- **`AC-56`** — `ir_duration: 4.25` against a backend that cannot fill it.
- **`AC-66`** — air absorption realized at α_ISO/4; same domain confusion as
  AC-54, second call site, must be fixed with it.
- **`F-186`** — the gate is evaluated at nominal α. Falls out of AC-184's fix.

Artifacts to work from: `experiments/ac175_probe/`, `experiments/item0_probe/`.

## 2. Then cluster C5 — the truncation window

`AC-64` *(blocker)*, `AC-58`, `F-89`, `RD-55`. AC-185 gives this a measured root
cause it did not have: the record itself moves with the budget, not just the
Schroeder window. RD-55's Lundeby extrapolated-tail compensation is the candidate
fix for all four.

## 3. Then the two remaining blockers

`F-75` and `F-81` — render/gen-scenes are not cache-protected, and the render
fingerprint is host-dependent. Both are off condition (i)'s path list, so they do
not gate the dataset render, but F-81 is a cross-platform violation.

## 4. `F-218` — before trusting another A/B

Gate step 4's detector has measured-zero power over part of what it certifies:
the canonical run uses `simulator: dry_run`, so a whole lane's changed surface
never executes, and a scene-acceptance threshold moved 50.0 → 52.0 while the A/B
reported "no change". Report it per lane over the files the run actually
executes, or stop calling it clearance.

---

## Needed from the user

- **A render grant.** The previous six are spent (four on AC-175/AC-184, two
  confirming AC-54). Closing C5 needs ~2 retained-artifact scenes; RD-17's
  convergence probe — the only thing that can lift gate condition (ii) — needs
  ~3 more and has never run.

## Standing rules for this session

- The ledger holds unresolved findings only. A fix means the row is **deleted**,
  not annotated. Nothing about how the work was done goes in it.
- Reviewers run over the final tree and the last pass must be clean. A pass whose
  findings you then fixed is not the last pass.
- `pytest tests/ -q` → 628 passed, 24 skipped. Canonical A/B sha256
  `74651cd26663fcc911979d9a7b9ddd8d97433ef4376fd829d35e6277d6f23052`.
