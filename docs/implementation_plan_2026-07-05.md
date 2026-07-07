# Implementation plan — 2026-07-05 review-pass findings

**Written by:** Fable session, 2026-07-05, under an explicit user instruction to
run the reviewers and plan only — **no fixes were implemented**. This document
is the hand-off to a future Opus session, which should execute it top to
bottom, then run the re-review loop to close the ledger.

**Ledger state at hand-off:** 6 OPEN rows (F-18 major, S-01 major, F-19 /
RR-09 / RR-10 / RR-11 minor) in `docs/review_ledger.md`, written by this pass.
DEFERRED backlog (RD-04, F-06, F-15, F-17) is untouched and stays deferred.

**Evidence baseline at hand-off** (reviewer-run, this pass): `pytest tests/` =
70 passed; `amcd all --config configs/base.yaml --config configs/dry_run.yaml
--force` runs clean. acoustics-reviewer re-verified all metric physics with
known-answer probes and returned **zero** new findings — do not touch
`representations/` or `evaluation/room_acoustic.py` under this plan.

---

## Order of work

1. §1 F-18 + RR-11 (one change surface: `stats/aggregate.py` + its tests + reporting)
2. §2 F-19 (guard in `evaluation/signal.py` path)
3. §3 S-01 (git init — do this FIRST if the user confirms, so the fixes land as commits; otherwise last)
4. §4 RR-09 + RR-10 (docstring/comment lines only)
5. §5 re-review loop and definition of done

§3 is ordered flexibly because it needs a user-visible decision; everything
else is unambiguous.

---

## §1 — F-18 (major): MDES/CI on the paired improvement, not the absolute metric
**Spec:** design_spec §9 (lines 261–263): MDES is "the smallest
baseline-vs-denoised **difference** detectable at the current n" — a paired
per-scene quantity. Also §9 lines 255–260 (CIs per split, never pooled) and
invariant #9.
**Also resolves:** RR-11 (the `mdes_80pct` column label hardcodes 80% power
while `config.bootstrap_power` governs; rename in the same edit).

**Defect:** `run_stats` (`src/amcd/stats/aggregate.py:139–146`) feeds
`mdes()` the std of the **absolute** per-scene `pred_val`, and the CI is on
`pred_val` too. The §9 quantity is the per-scene *paired* improvement. The two
dispersions diverge up to ~2.4× in the dry run (falsifier probe: test_id C50
std(pred_val)=6.27 vs std(paired)=2.57; test_geometry_shift energy_mse 9.47 vs
18.71 — MDES can be over- OR under-stated). The implicit noncentral-t null is
"mean = 0" of the wrong variable, so no correctly calibrated statistic tests
the improvement hypothesis at all. `tests/test_stats.py:127–130` locks in the
wrong σ.

**Change (files/functions, in order):**
1. `src/amcd/stats/aggregate.py`, `run_stats` (~line 138): per
   `(split, metric)` group, compute the per-scene paired improvement from the
   existing parquet columns (`evaluator.py:103–105` writes `low_val`,
   `pred_val`, `high_ref`):
   `paired = |low_val − high_ref| − |pred_val − high_ref|`
   (positive ⟺ improved; consistent with `metric_improvement` in
   `evaluation/metric_row.py:36–51`, so `paired > 0 ⟺ improved == True` row
   by row — assert this in a test). Drop NaN triples — this is exactly the
   `improved.notna()` population already used for `pct_improved`, so the
   paired-n equals `n_scored` by construction.
2. Same function: bootstrap CI + `mdes()` now run on `paired`. Keep the
   existing `pred_val` mean/CI columns as *descriptive* (rename to make the
   distinction legible, e.g. `pred_mean`/`pred_ci_lower`/`pred_ci_upper`/
   `pred_std`), and add `improvement_mean`, `improvement_ci_lower`,
   `improvement_ci_upper`, `improvement_std`, and `mdes` (renamed from
   `mdes_80pct` per RR-11 — power is stamped in the run's config.yaml).
   Exact schema is Opus's call; the non-negotiables are (a) MDES and the
   improvement CI are computed from `paired`, (b) the absolute-value CI is
   clearly labeled descriptive, (c) no column name hardcodes a config value.
3. `src/amcd/reporting/tables.py:37` (and any other `mdes_80pct` /
   pred-CI consumers — grep for both): follow the schema rename.
4. `tests/test_stats.py:127–130` plus neighbors: replace the wrong-σ lock
   with (a) a fixture where std(pred_val) ≠ std(paired) and an assertion that
   MDES follows the paired σ, and (b) the `paired > 0 ⟺ improved` consistency
   test from step 1.
5. Note for the commit message / ledger: this is the falsifier's "single most
   consequential risk" — the §9 guard against over-reading sub-noise
   differences was itself mis-calibrated.

**Invariants touched:** #9 (per-split, never pooled — unchanged, keep the
groupby); no-hidden-defaults (power/alpha still flow from config only).
**Interaction with DEFERRED backlog (research-director ruling, this pass):**
**Take F-15** (bootstrap RNG substreams) — §1 rewrites what the bootstrap
consumes, opening F-15's exact change surface; deferring means re-touching the
same lines and paying re-review on the same function twice. F-17 (power>alpha
guard, one line in `config._check`, a different file) is optional — take it or
leave it. Guard-rail if folded: give F-18, F-15 (and F-17 if taken) each their
OWN test so the falsifier can isolate the consequential fix from the bundled
hardening on re-review. Delete DEFERRED rows only on re-review-clean.
F-18's fix must NOT silently change the split seed or any config semantics.

**Evidence to show:** `pytest tests/` output; `amcd all … --force` and the
stats `ci_table.csv` with both descriptive and improvement columns, finite
paired MDES on the n=3 dry-run splits; a one-off probe printing old-σ vs
paired-σ MDES per (split, metric) to demonstrate the divergence is real.

## §2 — F-19 (minor): `energy_snr_db` assumes dB operands; waveform rep feeds amplitude
**Spec:** design_spec §6 (metric triples; diagnostic-only metrics), §3 (rep
seam). The waveform rep is the E1 faithfulness path, so this fires precisely
at the next milestone.

**Defect:** `src/amcd/evaluation/signal.py:31–37` computes
`10**(high_ref/10)` assuming dB log-energy input, but
`WaveformRepresentation.encode` (`representations/waveform.py:31–33`) yields
raw amplitude `(C,1,T)` → meaningless SNR on the waveform path. Unexercised
today (dry run uses spectrogram); diagnostic-only (`low=high=NaN` ⇒ improvement
undefined), so it cannot corrupt headline improvement counts — but it would
emit silently-garbage numbers in the E1 run.

**Change:** falsifier's recommended category is a **guard, not removal**.
Preferred shape (Opus to confirm against the code): give `Representation` (or
its registry entry) an explicit value-domain declaration (e.g. `domain: "db" |
"amplitude"` — check `representations/base.py` for where it fits the existing
seam) and have the eval stage skip-or-branch `compute_signal_metrics`
accordingly, emitting the documented NaN-triple for non-dB reps. Minimal
alternative: a hard assert with a clear message at the `compute_signal_metrics`
call site. Either way add a test that constructs the waveform rep and asserts
the SNR path does not emit a finite-but-garbage value.
**Invariants touched:** scaffold/seam rule — the guard must key on a declared
rep property, NEVER on `isinstance(rep, WaveformRepresentation)`.
**Files:** `src/amcd/evaluation/signal.py`, possibly
`src/amcd/representations/base.py` (+ the three reps), `tests/` (new test).
**Re-review:** touching `evaluation/` requires an **acoustics-reviewer** pass
(§5), not just falsifier.

## §3 — S-01 (major, infra): the project is not a git repository
`ls -a` shows no `.git`; `git log` fails. This contradicts (a) the ledger's
core premise that deleted rows are preserved in git history (the 2026-07-04
pass's audit trail currently does not exist anywhere), and (b) design_spec §9
line 270, which requires a git SHA in the reporting supplementary bundle.

**Change:** `git init` + a single initial commit of the current tree (respect
a sensible `.gitignore`: `__pycache__`, `._*` AppleDouble files, run outputs /
`experiments/` artifacts as appropriate — check what `experiments/` holds
before deciding). **This is a user-visible repository-creation decision: ask
the user before running it** (they may prefer to host it or have a remote
convention). If confirmed early, do it before §1 so every fix is a commit.
**Invariants touched:** none in code; restores the ledger's audit-trail
mechanism and unblocks the spec §9 git-SHA bundle item.

## §4 — RR-09 / RR-10 (minor): two documentation lines
- **RR-09:** `src/amcd/models/base.py:8` (and mirror at
  `src/amcd/models/cnn.py:51`): one docstring line on the `aux` parameter —
  it is the forward-looking seam for the path-conditioned variant
  (research_I_paper.md §4.4 / Appendix C; signature per design_spec §8), unused
  by `vanilla_cnn`, reserved for the gsound_sir path-export roadmap. Document,
  do not remove (CLAUDE.md forward-looking-abstraction rule).
- **RR-10:** `src/amcd/config.py:277` and `configs/base.yaml:10`: purpose note
  for `run_id` (the experiment-ledger label — E1/E2/… — a future run sets;
  currently stamped into each run's config.yaml with no consumer). Document,
  do not remove.

## §5 — Re-review loop and definition of done
Per CLAUDE.md: after implementing §§1–4, invoke by name:
- **falsifier** — re-review F-18/F-19 fixes over the current state,
- **acoustics-reviewer** — required because §2 touches `evaluation/signal.py`
  (and §1 touches the stats consumers of eval output),
- **readability-reviewer** — re-review RR-09/RR-10/RR-11 and the new
  column-schema naming.

Delete each ledger row only when its raising reviewer confirms clean. Done =
zero OPEN rows + zero new findings on a full pass. If stopping early, follow
the CLAUDE.md stopping rule (update ledger, state "NOT complete — N open").

### Notes carried forward (no action in this plan)
- **acoustics-reviewer residual-risk probe for the gsound_sir build** (next
  roadmap item, RD-06): `DryRunSimulator` models ray-budget convergence as
  reverberant-tail *energy* scaling (`dry_run.py:61,66`); real MC convergence
  reduces estimator *variance*, not expected energy. When the real backend
  lands, render the SAME scene at two ray budgets and verify T30 is
  budget-invariant rather than shrinking with N — if it shrinks, a
  dry_run-style energy assumption leaked into the real path. Documented
  scaffold behavior, not a defect (RD-07 already tracks the D0b verdict as a
  plumbing artifact).
- **IRResult.paths divergence — research-director RULED (this pass):**
  design_spec §8 shows `IRResult` carrying `paths: PathData`, absent from
  `simulators/base.py:63`. Verdict: acceptable to defer; do NOT add a
  speculative empty field before the producer (gsound_sir path export) exists.
  Tracked as DEFERRED row RD-08 in the ledger. No action in this plan.

### research-director verdict on this plan (2026-07-05)
DIRECTION: on-track. "Disciplined bug-clearing sitting squarely on the current
gate… No genuine scope creep." F-18 now (before gsound_sir/E1) is the ideal
window, not premature: the stats path is backend-independent and provable with
the dry run today, and slipping it past this gate would mean reading the first
real E1 numbers through a mis-calibrated improvement statistic. §2's rep
value-domain declaration and §4's seam documentation are correctly
forward-looking per the roadmap rule. Keep §1 ordered first.
