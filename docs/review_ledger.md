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

(none — 2026-07-08 pass closed with zero OPEN rows; see Resume here)

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. |

### Resume here

**2026-07-08 (Fable): pass CLOSED — zero OPEN rows.** F-20 (metric-`kind`
taxonomy: every metric declares `kind` ∈ {match_reference, maximize, minimize};
eval/stats spine branches on it; `energy_snr_db` now maximize and scored 3/3
everywhere), F-21 (no silent exclusion: producer NaN reasons →
`metrics/drops.csv`, scored-vs-attempted in stats/report), RR-14 (unscored rows
render `unscored`), RR-15..18 (readability minors), and AC-08 (ISO-3382 legs
averaged over the cross-leg intersected band set) all fixed and
re-review-confirmed clean by their raising reviewers; rows deleted. Clean-pass
evidence: falsifier recomputed all 20 ci_table cells from the raw parquet at
machine precision (twice — round 1 and the AC-08 delta, max diff 7.1e-15) and
verified drops.csv completeness (no missing, false, or orphan rows); acoustics
verified SNR legs bit-identical to pre-refactor and confirmed intersection
physics + preserved probe path; readability confirmed rendering and RR-15..18;
research-director gated the plan (maximize/minimize ruled forward-looking for
roadmap perceptual/spatial metrics, module docstring documents this). Current
dry run: `experiments/all_20260708_194959`; suite 86 passed. Falsifier's
out-of-scope observation (not a finding): D0b probe averages oracle/reference
over each IR's own surviving bands, not an intersection — pre-existing,
diagnostic-only, gated to real renders; acoustics-reviewer's call when D0b goes
live.

**2026-07-06 (Fable): 2026-07-05 plan implemented in full; pass CLOSED —
zero OPEN rows.** All plan items landed as commits on branch `v3-rebuild` (S-01
→ repo initialized, remote = github.com/Jack-Rainey/Monte_Carlo_ML_Project;
push is user-run): F-18 paired-improvement MDES/CI (+ folded F-15 substreams
and F-17 power>alpha guard, each with its own test), F-19 value_domain seam,
RR-09/10/11 docs/renames, RR-12/13 label fixes. Clean pass evidence: falsifier
verified F-18/F-19/F-15/F-17 clean by independent recompute of every
ci_table.csv cell from the raw parquet (machine precision), then delta-confirmed
the RR-12/13 rename (no value change, no stale schema consumer);
acoustics-reviewer zero new findings (probe-verified dB↔linear SNR,
value_domain declarations, paired quantity per metric; flagged a future
round-trip probe for any log/exp base edit); readability-reviewer confirmed
RR-09/10/11 and then RR-12/13 clean, zero new. Suite 76 passed; full dry run
clean (experiments/all_20260706_214339).

**Next concrete build (roadmap, per RD-06):** the
real `gsound_sir` render backend (x86) is the single hard dependency and the
next concrete build. Off it, in order: real D0a/D0b (the dry_run D0b "CARRIER
BOTTLENECK" verdict is a plumbing artifact, scientifically OPEN — RD-07) →
E1 reproduce-the-old-null (waveform) → pre-E1/post-E1 falsifier pass incl.
RD-04 split-seed guard → only then take E2 energy-residual results. Building
the §3 loss seam (metric-consistency + tail-weighting + spatial, loss.py,
F-06 unify) in parallel is fine; taking E2 results before E1 clears is not.
δ=1.0 remains PROVISIONAL, tuned at E3 (§7).
