# Review ledger

Findings from the review agents (`research-director`, `falsifier`,
`acoustics-reviewer`, `readability-reviewer`). Holds ONLY unresolved findings —
working memory for the loop, not an audit log. One row per finding:
`ID | agent | severity | status | anchor | finding | resolution`.

Status is exactly one of two values: **OPEN** (not yet resolved) or **DEFERRED**
(intentionally out of scope for the current gate, with a one-line reason and the
gate it belongs to). There is no ADDRESSED/RESOLVED status — the moment a
finding is fixed and re-review-confirmed clean, its row is deleted (git history
of this file is the audit trail).

## OPEN findings

From the 2026-07-06 re-review of the implemented 2026-07-05 plan (falsifier and
acoustics-reviewer confirmed all their findings clean, zero new;
readability-reviewer confirmed RR-09/10/11 clean and raised two new minors).

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RR-12 | readability-reviewer | minor | OPEN | src/amcd/stats/aggregate.py:227 | `mdes` is the only ci_table.csv column without a `pred_`/`improvement_` prefix; a CSV reader cannot tell which σ it derives from (the F-18 point is that they diverge). | Fix applied (renamed `improvement_mdes`; tables.py header stays "MDES"), awaiting re-review. |
| RR-13 | readability-reviewer | minor | OPEN | src/amcd/reporting/tables.py:46,55 | Report "N" column prints n_scored, which reads contradictory beside a finite Pred mean for diagnostic-only metrics (N=0 with a value). | Fix applied (header "N scored"), awaiting re-review. |

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. |

### Resume here

**2026-07-06 (Fable): 2026-07-05 plan implemented in full; re-review nearly
closed.** All plan items landed as commits on branch `v3-rebuild` (S-01 → repo
initialized, remote = github.com/Jack-Rainey/Monte_Carlo_ML_Project, branch not
yet pushed): F-18 paired-improvement MDES/CI (+ folded F-15 substreams and F-17
power>alpha guard, each with its own test), F-19 value_domain seam, RR-09/10/11
docs/renames. Re-review: falsifier verified F-18/F-19/F-15/F-17 clean by
independent recompute of every ci_table.csv cell from the raw parquet (machine
precision); acoustics-reviewer zero new findings (probe-verified dB↔linear SNR,
value_domain declarations, paired quantity per metric); readability-reviewer
confirmed RR-09/10/11 clean, raised minors RR-12/RR-13 — fixes applied, awaiting
readability re-confirmation, which is the ONLY step left before zero OPEN rows.
Suite 76 passed; full dry run clean (experiments/all_20260706_*).

**Next concrete build (roadmap, per RD-06):** the
real `gsound_sir` render backend (x86) is the single hard dependency and the
next concrete build. Off it, in order: real D0a/D0b (the dry_run D0b "CARRIER
BOTTLENECK" verdict is a plumbing artifact, scientifically OPEN — RD-07) →
E1 reproduce-the-old-null (waveform) → pre-E1/post-E1 falsifier pass incl.
RD-04 split-seed guard → only then take E2 energy-residual results. Building
the §3 loss seam (metric-consistency + tail-weighting + spatial, loss.py,
F-06 unify) in parallel is fine; taking E2 results before E1 clears is not.
δ=1.0 remains PROVISIONAL, tuned at E3 (§7).
