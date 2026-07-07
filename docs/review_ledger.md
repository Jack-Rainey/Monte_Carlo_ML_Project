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

From the 2026-07-05 full re-verify pass (falsifier + acoustics-reviewer +
readability-reviewer over the current codebase; acoustics returned zero new
findings). Implementation plan: `docs/implementation_plan_2026-07-05.md`.

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| F-18 | falsifier | major | OPEN | src/amcd/stats/aggregate.py:146 (mdes on ci["std"]) | MDES uses std of absolute per-scene `pred_val`, not the paired baseline-vs-denoised difference std that design_spec §9 (l.261-263) requires; σ diverges up to ~2.4× in the dry run (test_id C50: 6.27 vs 2.57) and the implicit t-test is against a null of 0, not the improvement null; no CI on the paired improvement exists anywhere. tests/test_stats.py:127-130 locks in the wrong σ. | Plan §1: compute MDES/CI on the per-scene paired improvement; keep pred_val CI as descriptive only. |
| F-19 | falsifier | minor | OPEN | src/amcd/evaluation/signal.py:31 (energy_snr_db) | `10**(x/10)` assumes dB log-energy operands, but the registered waveform rep (waveform.py:31) feeds raw amplitude → meaningless SNR on the E1 waveform path; unexercised today, diagnostic-only. | Plan §2: guard/test that compute_signal_metrics runs only on dB-domain banded reps, or make it rep-domain-aware. |
| S-01 | session (Fable) | major | OPEN | project root (no .git) | The project is NOT a git repository, contradicting this ledger's "git history is the audit trail" premise and spec §9 (l.270), which requires a git SHA in the reporting supplementary bundle. All prior "deleted rows live in git history" claims currently have no backing store. | Plan §3: `git init` + initial commit (user-visible decision; flagged in plan). |
| RR-09 | readability-reviewer | minor | OPEN | src/amcd/models/base.py:8 | `Model` Protocol's `aux: torch.Tensor \| None` parameter is undocumented — it is the forward-looking seam for the path-conditioned variant (research_I_paper.md §4.4 / App. C; design_spec §8) but nothing in code says so. | Plan §4: one docstring line at the seam (base.py + cnn.py:51). |
| RR-10 | readability-reviewer | minor | OPEN | src/amcd/config.py:277 (also configs/base.yaml:10) | `run_id` is declared, defaulted to "", stamped into every run's config.yaml, but has no consumer and no purpose note — the only Config field with neither. | Plan §4: document intended role (experiment-ledger label); do not remove. |
| RR-11 | readability-reviewer | minor | OPEN | src/amcd/stats/aggregate.py:167 (→ reporting/tables.py:37) | Summary column `mdes_80pct` hardcodes 80% power into the label while the power used is `config.bootstrap_power`; a `bootstrap_power: 0.9` run would emit a right number under a wrong name. | Plan §1 (bundled with F-18): rename to `mdes` or derive label from config. |

## DEFERRED backlog

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| RD-04 | research-director | low | DEFERRED | configs/base.yaml (seeds.split_assignment) | Split seed must be pinned stable-for-life; changing it after E1 is a deliberate re-dataset event (leakage/faithfulness risk). | Belongs to the pre-E1 falsifier pass: add a guard against silent change to the split seed. |
| F-06 | falsifier | minor | DEFERRED | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place meanwhile: docstring states δ must be operand-domain; sole current caller (`build_criterion`) already scales it. |
| F-15 | falsifier | minor | DEFERRED | src/amcd/stats/aggregate.py (run_stats bootstrap RNG) | Single bootstrap RNG stream is shared across all groupby (split,metric) groups, so a group's bootstrap CI bounds depend on preceding groups' RNG consumption; adding/removing a metric or split perturbs unchanged groups' CI bounds by MC noise. Reproducible for a fixed group set; point estimates unaffected. | Stats-hardening pass: give each group an independent substream (SeedSequence(seed("bootstrap")).spawn keyed by group, or reseed per (split,metric)). Sub-threshold for the current gate. |
| F-17 | falsifier | minor | DEFERRED | src/amcd/config.py:388 (_check) | `_check` validates `alpha` and `bootstrap_power` in (0,1) independently but does not enforce `power > alpha`; a config with `power ≤ alpha` makes `mdes` silently return ≈0 (achieved power = alpha) rather than raising. Nonsensical config far outside any plausible bootstrap setting (base ships power=0.8 ≫ α=0.05) — flagged so it is not lost. | Stats-hardening pass (with F-15): one-line `power > alpha` guard in `_check`. Sub-threshold for the current gate. |
| RD-08 | research-director | minor | DEFERRED | src/amcd/simulators/base.py:63 (IRResult) vs design_spec §8 l.238 | Spec §8 shows IRResult carrying `paths: PathData`; code omits it. This is the producer half of the path-conditioned-variant seam (consumer half `Model.forward` aux already exists, RR-09). No PathData schema and no populating producer exist yet. | Lands with the gsound_sir build (RD-06): path export defines PathData and populates IRResult.paths. Do NOT add a speculative empty field before the producer exists. |

### Resume here

**2026-07-05 re-verify pass (Fable): reviews RUN, fixes NOT implemented — by
explicit user instruction.** Fable ran the full reviewer set over the current
codebase and wrote the OPEN rows above plus an implementation plan at
`docs/implementation_plan_2026-07-05.md` for a future Opus session to execute.
The next session should: read that plan, implement in its stated order, then
re-run falsifier + readability-reviewer (and acoustics-reviewer if
evaluation/signal.py changes) to confirm clean and delete resolved rows.

Reviewer evidence this pass: falsifier ran `pytest tests/` (70 passed) and a
full `amcd all` dry run, and re-confirmed clean: split routing (0 misroutes),
normalization provenance (train-only stats), no scaffold coupling, no identity
collapse (correction energy 17% of low→high gap), seed discipline.
acoustics-reviewer re-verified all metric/representation physics with
known-answer probes (T30 recovery 0.506/1.172 vs true 0.5/1.2 s; band energy
conservation 1.0000; C50 boundary leakage 0.0) — zero new findings.

The prior (2026-07-04) verification pass remains CLOSED. DEFERRED backlog
unchanged (RD-04 split-seed guard; F-06 loss unify; F-15 bootstrap substreams;
F-17 power>alpha guard) — none on the current gate, though F-18 will likely
subsume/interact with F-15/F-17 in the pre-E1 stats pass.

**Next concrete build (roadmap, per RD-06):** the
real `gsound_sir` render backend (x86) is the single hard dependency and the
next concrete build. Off it, in order: real D0a/D0b (the dry_run D0b "CARRIER
BOTTLENECK" verdict is a plumbing artifact, scientifically OPEN — RD-07) →
E1 reproduce-the-old-null (waveform) → pre-E1/post-E1 falsifier pass incl.
RD-04 split-seed guard → only then take E2 energy-residual results. Building
the §3 loss seam (metric-consistency + tail-weighting + spatial, loss.py,
F-06 unify) in parallel is fine; taking E2 results before E1 clears is not.
δ=1.0 remains PROVISIONAL, tuned at E3 (§7).
