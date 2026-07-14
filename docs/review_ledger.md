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

(none — 2026-07-13 verbosity-gate pass closed with zero OPEN rows; see Resume here)

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. |

### Resume here

**2026-07-13 (Fable): verbosity gate CLOSED — zero OPEN rows.** F-22 (threading:
frozen `Verbosity(save, show)` cli → Pipeline → all nine stages via the widened
dispatch signature; zero bare `print` outside `runtime.emit`), F-23 (save axis
gates only observability artifacts; falsifier independently verified save=0 vs
save=5 full runs bit-identical across metrics.parquet, stats, diagnostics JSONs,
and best.pt weights), F-24 (IntRange(0,5), always-stderr warnings/errors, visual
TTY guard), RD-09 (defaults 1/1 quarantined to the CLI layer, provenance rung),
RD-10 (level-5 §6 Blender seam reserved, recorded in research-director.md's
forward-looking list), RR-19 (ladder + total per-stage wiring table in
docs/verbosity.md, single `emit` helper, shared `common_options` decorator) all
implemented and confirmed clean; the pass's own findings (RD-11, RR-20..23 —
doc wording, all minor) fixed and re-review-confirmed; rows deleted. Clean-pass
evidence: falsifier zero findings, acoustics zero findings (print→emit faithful,
no physics change), research-director + readability re-confirmed their fixes.
Suite 99 passed; dry run `experiments/all_20260713_194209` (default save=1/show=1
writes provenance quartet incl. git SHA, omits train_log.csv and renders/meta.json
as specced). The dry_run plumbing gate is complete.

**Next concrete build (roadmap, per RD-06):** the
real `gsound_sir` render backend (x86) is the single hard dependency and the
next concrete build. Off it, in order: real D0a/D0b (the dry_run D0b "CARRIER
BOTTLENECK" verdict is a plumbing artifact, scientifically OPEN — RD-07) →
E1 reproduce-the-old-null (waveform) → pre-E1/post-E1 falsifier pass incl.
RD-04 split-seed guard → only then take E2 energy-residual results. Building
the §3 loss seam (metric-consistency + tail-weighting + spatial, loss.py,
F-06 unify) in parallel is fine; taking E2 results before E1 clears is not.
δ=1.0 remains PROVISIONAL, tuned at E3 (§7).
