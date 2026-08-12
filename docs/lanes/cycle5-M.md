# Lane M — metric computation path

Cycle 5. Generated skeleton: the integrator fills the ORDER and the PASS CONDITION
before this lane starts. Assigned rows come from `docs/lanes/cycle5.yaml`, which is
the authority; this file explains them.

## Your row-id block (rule 6)

Allocate NEW ids only from here. Cycle 4 had no such rule and four lanes collided
four ways, including with live rows.

- `AC-100..124`
- `F-135..159`
- `RD-175..199`
- `RR-115..139`

## Pass condition

A fixed-seed `ci_table.csv` A/B against the baseline captured before the lanes
started. **Declare your expected effect on that table before you begin** (RD-91,
RD-149): the detector is the SET of declarations, not the premise that only the metric
lane can move the table — that premise is false by construction, since the lane
owning stats/reporting writes it and the lane owning scenes/ sets the population
(RD-191/RD-261). A row that moves without a pre-registered declaration is a
finding.

## Assigned rows (32)

- **F-68** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-26** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-28** — fix: `src/amcd/simulators/dry_run.py`
- **AC-30** — fix: `configs/base.yaml`
- **AC-37** — fix: `src/amcd/evaluation/evaluator.py`, `src/amcd/representations/spectrogram.py`
- **AC-38** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-39** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-43** — fix: `src/amcd/simulators/dry_run.py`
- **AC-36** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-65** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-68** — fix: `src/amcd/evaluation/room_acoustic.py`
- **RR-37** — fix: `src/amcd/simulators/dry_run.py`
- **RR-38** — fix: `src/amcd/evaluation/room_acoustic.py`
- **RR-45** — fix: `src/amcd/representations/spectrogram.py`
- **F-M3** — fix: `src/amcd/representations/spectrogram.py`
- **F-M9** — fix: `src/amcd/evaluation/room_acoustic.py`
- **F-M10** — fix: `configs/representations/spectrogram.yaml`, `src/amcd/representations/spectrogram.py`
- **F-68-R2** — fix: `src/amcd/evaluation/room_acoustic.py`
- **F-68-R3** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-37-R4** — fix: `src/amcd/representations/spectrogram.py`
- **AC-26-R6** — fix: `configs/base.yaml`
- **AC-19-R7** — fix: `src/amcd/representations/spectrogram.py`
- **RD-93** — fix: `src/amcd/evaluation/room_acoustic.py`
- **RD-96** — fix: `src/amcd/evaluation/room_acoustic.py`
- **RD-98** — fix: `src/amcd/evaluation/room_acoustic.py`
- **RD-99** — fix: `src/amcd/evaluation/room_acoustic.py`
- **S-1** — fix: `src/amcd/acoustics.py`
- **AC-60** — fix: `configs/base.yaml`
- **RR-85** — fix: `src/amcd/evaluation/room_acoustic.py`
- **AC-69** — fix: `src/amcd/representations/spectrogram.py`, `tests/test_filterbank.py`, `tests/test_metrics.py`
- **AC-70** — fix: `src/amcd/evaluation/room_acoustic.py`
- **F-106** — fix: `tests/test_metrics.py`

## Files you own

- `src/amcd/evaluation/**`
- `src/amcd/representations/**`
- `src/amcd/acoustics.py`
- `src/amcd/simulators/dry_run.py`
- `configs/base.yaml`
- `configs/research_i.yaml`
- `configs/representations/**`
- `tests/test_metrics.py`
- `tests/test_filterbank.py`
- `tests/test_acoustic_validity.py`
- `tests/test_eval_improvement.py`
- `tests/test_signal_domain.py`

## Not yours

Anything else — including `docs/review_ledger.md`, `CLAUDE.md` and
`docs/design_spec.md`, which have one writer. Record cross-lane findings in
`docs/ledger_inbox/M.md` with a CONCRETE FILE ANCHOR; the anchor is what assigns the row
to a lane next cycle and what the RD-33a gate counts.
