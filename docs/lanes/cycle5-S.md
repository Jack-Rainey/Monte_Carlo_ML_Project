# Lane S — scenes, QC and diagnostics

Cycle 5. Generated skeleton: the integrator fills the ORDER and the PASS CONDITION
before this lane starts. Assigned rows come from `docs/lanes/cycle5.yaml`, which is
the authority; this file explains them.

## Your row-id block (rule 6)

Allocate NEW ids only from here. Cycle 4 had no such rule and four lanes collided
four ways, including with live rows.

- `AC-150..174`
- `F-185..209`
- `RD-225..249`
- `RR-165..189`

## Pass condition

A fixed-seed `ci_table.csv` A/B against the baseline captured before the lanes
started. **Declare your expected effect on that table before you begin** (RD-91,
RD-149): the detector is the SET of declarations, not the premise that only the metric
lane can move the table — that premise is false by construction, since the lane
owning stats/reporting writes it and the lane owning scenes/ sets the population
(RD-191/RD-261). A row that moves without a pre-registered declaration is a
finding.

## Assigned rows (21)

- **RD-65** — fix: `src/amcd/scenes/generator.py`
- **F-60** — fix: `src/amcd/scenes/generator.py`
- **F-71** — fix: `src/amcd/scenes/generator.py`
- **F-72** — fix: `src/amcd/diagnostics/probe.py`
- **RR-44** — fix: `src/amcd/scenes/generator.py`
- **RD-112** — fix: `src/amcd/scenes/generator.py`
- **S-F4** — fix: `src/amcd/scenes/generator.py`
- **S-F5** — fix: `src/amcd/scenes/generator.py`, `tests/test_scene_placement.py`
- **S-F6** — fix: `src/amcd/scenes/generator.py`
- **S-F7** — fix: `src/amcd/scenes/generator.py`
- **AC-51** — fix: `src/amcd/scenes/generator.py`
- **AC-52** — fix: `src/amcd/scenes/generator.py`, `tests/test_scene_placement.py`
- **AC-53** — fix: `src/amcd/scenes/generator.py`
- **RR-60** — fix: `src/amcd/scenes/generator.py`
- **RR-61** — fix: `src/amcd/scenes/generator.py`
- **RR-62** — fix: `src/amcd/scenes/generator.py`
- **RR-63** — fix: `src/amcd/diagnostics/probe.py`
- **RR-64** — fix: `src/amcd/diagnostics/probe.py`
- **RR-65** — fix: `tests/test_scene_placement.py`
- **RR-66** — fix: `tests/test_probe.py`
- **RR-67** — fix: `src/amcd/scenes/generator.py`

## Files you own

- `src/amcd/scenes/**`
- `src/amcd/diagnostics/**`
- `tests/test_scene_placement.py`
- `tests/test_min_separation.py`
- `tests/test_invariants.py`
- `tests/test_probe.py`

## Not yours

Anything else — including `docs/review_ledger.md`, `CLAUDE.md` and
`docs/design_spec.md`, which have one writer. Record cross-lane findings in
`docs/ledger_inbox/S.md` with a CONCRETE FILE ANCHOR; the anchor is what assigns the row
to a lane next cycle and what the RD-33a gate counts.
