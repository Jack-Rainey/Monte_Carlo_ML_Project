# Lane P — provenance, stage cache and reported tables

Cycle 5. Generated skeleton: the integrator fills the ORDER and the PASS CONDITION
before this lane starts. Assigned rows come from `docs/lanes/cycle5.yaml`, which is
the authority; this file explains them.

## Your row-id block (rule 6)

Allocate NEW ids only from here. Cycle 4 had no such rule and four lanes collided
four ways, including with live rows.

- `AC-125..149`
- `F-160..184`
- `RD-200..224`
- `RR-140..164`

## Pass condition

A fixed-seed `ci_table.csv` A/B against the baseline captured before the lanes
started. **Declare your expected effect on that table before you begin** (RD-91,
RD-149): the detector is the SET of declarations, not the premise that only the metric
lane can move the table — that premise is false by construction, since the lane
owning stats/reporting writes it and the lane owning scenes/ sets the population
(RD-191/RD-261). A row that moves without a pre-registered declaration is a
finding.

## Assigned rows (40)

- **RD-66** — fix: `src/amcd/pipeline.py`
- **RD-20** — fix: `src/amcd/pipeline.py`
- **F-63** — fix: `src/amcd/pipeline.py`
- **F-64** — fix: `src/amcd/data/preprocess.py`, `src/amcd/pipeline.py`
- **F-66** — fix: `src/amcd/pipeline.py`, `src/amcd/provenance.py`
- **F-69** — fix: `src/amcd/provenance.py`
- **F-73** — fix: `src/amcd/config.py`
- **F-74** — fix: `src/amcd/config.py`, `src/amcd/training/infer.py`, `src/amcd/training/trainer.py`
- **RR-35** — fix: `src/amcd/config.py`
- **RR-36** — fix: `src/amcd/pipeline.py`
- **RR-42** — fix: `src/amcd/provenance.py`
- **F-M11** — fix: `src/amcd/data/normalization.py`, `src/amcd/data/preprocess.py`
- **F-69-B4** — fix: `src/amcd/provenance.py`
- **RD-101** — fix: `src/amcd/pipeline.py`
- **RD-102** — fix: `src/amcd/pipeline.py`
- **RD-103** — fix: `tests/test_stage_cache.py`
- **RD-105** — fix: `src/amcd/pipeline.py`, `tests/test_stage_cache.py`
- **RD-106** — fix: `src/amcd/reporting/tables.py`, `src/amcd/stats/aggregate.py`
- **F-76** — fix: `src/amcd/pipeline.py`
- **F-77** — fix: `tests/test_stage_cache.py`
- **F-78** — fix: `tests/test_stage_cache.py`
- **F-79** — fix: `src/amcd/provenance.py`
- **F-80** — fix: `src/amcd/config.py`
- **AC-47** — fix: `src/amcd/pipeline.py`
- **AC-48** — fix: `src/amcd/reporting/tables.py`
- **RR-46** — fix: `src/amcd/pipeline.py`
- **RR-47** — fix: `src/amcd/pipeline.py`
- **RR-48** — fix: `tests/test_stage_cache.py`
- **RR-49** — fix: `src/amcd/provenance.py`
- **RR-50** — fix: `src/amcd/pipeline.py`
- **RR-51** — fix: `src/amcd/pipeline.py`
- **RR-52** — fix: `src/amcd/pipeline.py`
- **RR-53** — fix: `tests/test_stage_cache.py`
- **RR-54** — fix: `src/amcd/config.py`
- **RR-55** — fix: `src/amcd/config.py`
- **RR-56** — fix: `src/amcd/config.py`
- **RR-57** — fix: `src/amcd/provenance.py`
- **RR-59** — fix: `tests/test_stage_cache.py`
- **RR-68** — fix: `src/amcd/config.py`
- **F-102** — fix: `src/amcd/pipeline.py`

## Files you own

- `src/amcd/pipeline.py`
- `src/amcd/provenance.py`
- `src/amcd/config.py`
- `src/amcd/data/**`
- `src/amcd/stats/**`
- `src/amcd/reporting/**`
- `src/amcd/training/**`
- `tests/test_stage_cache.py`
- `tests/test_config.py`
- `tests/test_dataset_integrity.py`
- `tests/test_stats.py`
- `tests/test_report.py`

## Not yours

Anything else — including `docs/review_ledger.md`, `CLAUDE.md` and
`docs/design_spec.md`, which have one writer. Record cross-lane findings in
`docs/ledger_inbox/P.md` with a CONCRETE FILE ANCHOR; the anchor is what assigns the row
to a lane next cycle and what the RD-33a gate counts.
