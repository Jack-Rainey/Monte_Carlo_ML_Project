# Lane B — render backend — is the declared population renderable?

Cycle 5. Generated skeleton: the integrator fills the ORDER and the PASS CONDITION
before this lane starts. Assigned rows come from `docs/lanes/cycle5.yaml`, which is
the authority; this file explains them.

## Your row-id block (rule 6)

Allocate NEW ids only from here. Cycle 4 had no such rule and four lanes collided
four ways, including with live rows.

- `AC-75..99`
- `F-110..134`
- `RD-150..174`
- `RR-90..114`

## Pass condition

A fixed-seed `ci_table.csv` A/B against the baseline captured before the lanes
started. **Declare your expected effect on that table before you begin** (RD-91,
RD-149): the detector is the SET of declarations, not the premise that only the metric
lane can move the table — that premise is false by construction, since the lane
owning stats/reporting writes it and the lane owning scenes/ sets the population
(RD-191/RD-261). A row that moves without a pre-registered declaration is a
finding.

## Assigned rows (40)

- **RD-33a** — fix: `src/amcd/simulators/gsound_sir.py`
- **RD-67** — fix: `src/amcd/simulators/gsound_sir.py`
- **RD-08** — fix: `src/amcd/simulators/base.py`
- **RD-21** — fix: `src/amcd/simulators/gsound_sir.py`
- **RD-24** — fix: `src/amcd/simulators/base.py`
- **RD-114** — fix: `configs/simulators/gsound_sir.yaml`, `src/amcd/simulators/render.py`
- **RD-116** — fix: `src/amcd/simulators/gsound_sir.py`, `tests/test_simulator_seam.py`
- **RD-117** — fix: `src/amcd/simulators/render.py`
- **RD-120** — fix: `src/amcd/simulators/gsound_sir.py`
- **RD-121** — fix: `src/amcd/simulators/gsound_sir.py`
- **RD-122** — fix: `src/amcd/simulators/render.py`
- **RD-123** — fix: `src/amcd/simulators/gsound_sir.py`
- **F-84** — fix: `src/amcd/simulators/gsound_sir.py`, `src/amcd/simulators/render.py`
- **F-85** — fix: `src/amcd/simulators/gsound_sir.py`
- **F-86** — fix: `src/amcd/simulators/render.py`, `tests/test_simulator_seam.py`
- **F-87** — fix: `src/amcd/simulators/gsound_sir.py`, `tests/test_simulator_seam.py`
- **F-88** — fix: `src/amcd/simulators/base.py`, `src/amcd/simulators/gsound_sir.py`
- **F-90** — fix: `src/amcd/simulators/gsound_sir.py`, `src/amcd/simulators/render.py`
- **F-91** — fix: `src/amcd/simulators/gsound_sir.py`, `tests/test_simulator_seam.py`
- **F-92** — fix: `src/amcd/simulators/gsound_sir.py`
- **F-93** — fix: `src/amcd/simulators/gsound_sir.py`, `tests/test_simulator_seam.py`
- **F-94** — fix: `src/amcd/simulators/gsound_sir.py`
- **F-95** — fix: `src/amcd/simulators/base.py`
- **F-96** — fix: `tests/test_simulator_seam.py`
- **F-98** — fix: `src/amcd/simulators/render.py`
- **AC-57** — fix: `src/amcd/simulators/base.py`, `src/amcd/simulators/gsound_sir.py`, `tests/test_simulator_seam.py`
- **AC-59** — fix: `src/amcd/simulators/gsound_sir.py`
- **AC-62** — fix: `src/amcd/simulators/base.py`
- **AC-63** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-69** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-70** — fix: `src/amcd/simulators/base.py`
- **RR-71** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-72** — fix: `tests/test_simulator_seam.py`
- **RR-74** — fix: `src/amcd/simulators/base.py`
- **RR-75** — fix: `src/amcd/simulators/base.py`
- **RR-76** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-77** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-78** — fix: `tests/test_simulator_seam.py`
- **RR-83** — fix: `src/amcd/simulators/gsound_sir.py`
- **RR-84** — fix: `src/amcd/simulators/base.py`

## Files you own

- `src/amcd/simulators/base.py`
- `src/amcd/simulators/gsound_sir.py`
- `src/amcd/simulators/render.py`
- `src/amcd/runtime.py`
- `configs/simulators/**`
- `tests/test_simulator_seam.py`
- `tests/test_setup_gsound_sir.py`
- `tests/test_runtime.py`

## Not yours

Anything else — including `docs/review_ledger.md`, `CLAUDE.md` and
`docs/design_spec.md`, which have one writer. Record cross-lane findings in
`docs/ledger_inbox/B.md` with a CONCRETE FILE ANCHOR; the anchor is what assigns the row
to a lane next cycle and what the RD-33a gate counts.
