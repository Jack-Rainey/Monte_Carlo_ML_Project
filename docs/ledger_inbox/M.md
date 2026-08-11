# Lane M inbox — cycle4

Branch `lane/M-cycle4`. Written by lane M, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/M-cycle4", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

---

## Baseline (Step 0) — the A leg, and the comparison's own noise floor

`research-director` raised at plan time that "ci_table.csv must be bit-identical"
was an assertion, not a measurement (RD-97 below). So the canonical dry run was
run TWICE from scratch, into separate run dirs, before any edit:

```
$ amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml \
      -c configs/overlays/dry_run.yaml -r <scratch>/baseline_a      # 13.2 s
$ amcd all -c ... -r <scratch>/baseline_b                           # 11.1 s
$ diff baseline_a/stats/ci_table.csv baseline_b/stats/ci_table.csv
IDENTICAL: ci_table.csv
$ diff baseline_a/metrics/drops.csv baseline_b/metrics/drops.csv
IDENTICAL: drops.csv
```

**The A/B's noise floor is zero** — the pipeline is bit-reproducible for both
reported artifacts across independent from-scratch runs. Every "unchanged" claim
below is therefore a real measurement, and any moved cell is signal.

Pre-change suite: `362 passed in 49.55s`.

The baseline carries exactly ONE resolvability-floor drop in the whole run, which
is what makes the AC-38 pre-registration below sharp:

```
scene_0022 | test_material_shift | EDT/low  and  EDT/high
  partial: 1/2 bands kept from the physical legs — ... (500 Hz: EDT 0.0121 s is
  below the 0.0191 s the 500 Hz octave band can resolve (2 x the filter's own EDT
  of 0.0096 s) ...)
```

---

## PRE-REGISTRATION of the expected `ci_table.csv` movement (RD-100)

Written after Step 3 and BEFORE implementing AC-38 / AC-39 / AC-42, because this
file has twice shipped a change that was applied and defended on evidence chosen
after the fact (AC-36's padding, F-67's fold). Predicted movement, canonical dry
run, against `baseline_a`:

1. **`test_material_shift` / `EDT` — the ONLY row expected to move.** Count-and-
   disclose stops excluding scene_0022's 500 Hz band from the physical legs, so
   that scene's EDT becomes a 2-band average instead of a 1-band average.
   - `n_partial_band` 1 → 0
   - `pred_mean` expected to FALL from 0.06491 (the 500 Hz EDT, 0.0121 s, is
     smaller than the 1000 Hz one it is being averaged with)
   - `improvement_mean` / CIs / `improvement_std` / `improvement_mdes` all move
   - `n_scored` expected to STAY 3 (pred is not NaN in that band; the baseline
     reports `n_pred_band_unresolved` 0 everywhere)
2. **Every other (split, metric) cell: UNCHANGED.** No other drop exists in the
   run, so nothing else has a floor decision to change.
3. **AC-39** changes a reason STRING only. Under AC-38 the physical legs are no
   longer dropped for this cause, so this drop row leaves `drops.csv` and the
   reworded reason is exercised by pred-side drops and the synthetic test, not by
   the dry run.
4. **AC-42** predicted to move NOTHING in `ci_table.csv`: the canonical run has no
   degenerate pred, so no `late` window sits at the numeric floor.

If anything outside (1) moves, the change is doing something it was not designed
to do and I stop rather than explain it afterwards.

---

## acoustics-reviewer pass — SELF-CHECK on lane/M-cycle4 (not a clean pass)

Domain-physics / DSP only. Probes under
`/private/tmp/claude-501/.../scratchpad/p1..p14`, all run with
`PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-M/src`.
Suite at review time: `77 passed in 15.81s`
(test_metrics, test_filterbank, test_acoustic_validity, test_signal_domain).

`ID | source | severity | status | anchor | finding | resolution`

AC-42-R1 | acoustics-reviewer | blocker | OPEN | `src/amcd/evaluation/room_acoustic.py:466-496` | The AC-42 residue derivation is false, and the guard is an undeclared C50 ceiling. The comment asserts a float32 sum "cannot carry information below ~sqrt(n)*eps*the largest value in the integrated region". float32 is FLOATING point (per-element RELATIVE precision) and `late = energy[split:trunc_idx].sum()` never takes the early peak as an operand, so `max(integrated)` (line 479-484, which spans the EARLY window) bounds no error in it. PROBE: two late windows with bit-identical sums and true relative error 1.66e-07 / 7.07e-08 vs a float64 recompute get opposite verdicts purely from an early peak of 1.0 vs 1e6. At the point the guard actually fires on real data (C50 ~ +60 dB) the late sum is still accurate to 3.5e-08 relative — ~7.5 correct significant digits, so the reason string at :490-496 ("would report ... dB of rounding noise as clarity") is untrue at its own firing point. Algebraically the guard is `C50 >= 10log10(early/peak) - 10log10(sqrt(n_late)*eps)`, i.e. a ceiling set by the DIRECT ARRIVAL'S CREST FACTOR: measured 57.59-62.19 dB varying with arrival shape alone at a fixed tail level, unscoring a 10-sample arrival at C50 +60.13 dB while keeping an ideal impulse at +50.83 dB. The threshold therefore moves the WRONG way — a sharper direct arrival, exactly what legitimately produces a large C50, LOWERS the ceiling. ISO 3382-1 defines C50 = 10 lg[int(0,50ms) p2 dt / int(50ms,inf) p2 dt] with no upper bound and licenses no censoring on the denominator's magnitude. Because `residue` is computed per leg from that leg's own peak, two legs with identical late windows and different direct peaks get different verdicts — leg-asymmetric censoring in a paired comparison, the defect class RD-43/AC-25 exist to prevent. NOT firing inside today's declared support (`ceiling_absorptive` alpha<=0.98, `distance_range` [1.0,null]): worst margin 9.59 dB at alpha 0.98 / d 1.0 m / 3x3x2.4 m / 1000 Hz, C50 +54.00 dB — so no reported number is wrong today, but the margin is 9.6 dB, unmotivated, and calibrated against the scaffold's level convention. | Anchor the residue on the LATE window (`max(energy[split:trunc_idx])`), which makes it a genuine relative-precision statement that essentially never fires on real data; or declare an explicit C50 ceiling as a config field and name it as one. Note the degenerate 5 ms pred it was built to catch is caught by accident — at 500 Hz that late window is the FILTER's own ringing (T30 17.9 ms, `_band_resolvable_decay_s`), real arithmetic and not rounding, so the honest instrument for it is the resolvability floor the comment at :456-462 deliberately declines to use.

F-68-R2 | acoustics-reviewer | major | OPEN | `src/amcd/evaluation/room_acoustic.py:113-126` (docstring), `:198-212` (fold) | The KNOWN RESIDUAL paragraph mis-describes what `_band_energy` does. "The residual that remains is placement, not magnitude: mirroring is exact for an isolated arrival" is false for the arrival that matters. Onset alignment (`channel_per_band_metrics:641-642`) guarantees the direct arrival sits at record index 0, i.e. ON the boundary. Measured against the shift-invariant ground truth (same isolated impulse placed in the record interior, identical filter), the folded post-onset energy is +96.60 % at 500 Hz and +93.32 % at 1000 Hz, against 0.00 % error for the same impulse at index 500. The fold nearly DOUBLES the direct arrival's post-onset in-band energy relative to an identical interior arrival — a magnitude effect on the ISO integral's early window, shift-variant in exactly the position the metric always puts the direct sound. Downstream, C50 against the causal IEC 61260-conformant reference (`sosfilt`, same 4th-order band edges) runs +0.06 to +1.67 dB high, monotone in direct-dominance (alpha 0.10/d 3.0 -> +0.23/+0.06 dB; alpha 0.98/d 1.0 -> +1.67/+1.22 dB), and the NO-FOLD variant is closer to the causal reference in every direct-dominated cell. The bias is common-mode across legs but a function of DRR, hence NOT across scenes: paired improvement largely protected, reported ABSOLUTE C50 carries up to +1.7 dB of convention bias. Separately, the paragraph's own figures (22.17/22.16/20.37 pp, max 46.09) name neither config nor seed and are not reproducible from what it states; a 24-cell resample of the declared support gives mean 14.17 pp (low) / 14.19 pp (high), max 40.02 — same structure (low ~ high to 0.02 pp, tens of pp, max ~40-46), different scenes. | Restate the residual as a MAGNITUDE bias on the early window with its measured size vs a causal reference, not as placement; and anchor the pp figures to the config + seed that produced them. Energy conservation itself is CONFIRMED (folded/full = 1.000000001 over impulse-at-0, impulse-at-mid and exp-decay noise at both bands), so only the characterisation is wrong, not the conservation claim.

F-68-R3 | acoustics-reviewer | minor | OPEN | `src/amcd/evaluation/room_acoustic.py:203`, `:211` | The fold double-counts one pad sample at each end. With `head == guard` (the normal case: n_record 204000 >> guard 4608) the preceding slice `energy[guard-1::-1][:head]` already covers k = 1..guard INCLUSIVE, so `energy[0]` has been folded onto `energy[2*guard]` before line 203 adds it to `energy[guard]` a second time; line 211 is the mirror-image case for `energy[-1]`. MEASURED: folded/full = 1.000000001490 (500 Hz) and 1.000000004612 (1000 Hz); deleting the two lines gives exactly 1.000000000000. The over-count is (e[0]+e[-1])/full ~ 1.7e-28 and is physically irrelevant. The reason it is still worth a row: those two lines are also the ONLY handling of the `head < guard` short-record branch, where pad samples with k > head are silently DISCARDED (energy no longer conserved) and only e[0] is re-added. `ir_duration: 4.25` keeps the code out of that branch today. | Needs a guard or a known-answer test on a record shorter than one guard width (n_record <= guard, i.e. < 96 ms at 500 Hz), not removal — the branch is unexercised, not wrong-by-construction.

AC-37-R4 | acoustics-reviewer | major | OPEN | `src/amcd/representations/spectrogram.py:372`, calibration at `:235-273`, `configs/representations/spectrogram.yaml` `min_db_headroom_db` | The headroom guard is calibrated on one band set and enforced on another. `torch.amax(energy_db, dim=(0,2))` + `argmin` runs over ALL 27 third-octave STFT bands, and MEASURED the limiting band is ALWAYS the 24.8 Hz band — whose own in-band energy fraction is 0.579 (`describe_bands`), i.e. one of the bands `_build_third_octave_filters:33-38` declares "measure Hann leakage rather than band content". The guard raises at a level where the 500 Hz and 1000 Hz bands still carry 60.4 / 59.9 dB of headroom, 13.9 dB above the worst calibrated survivor (46.5 dB). But the calibration table at :240-246 and the stated physical reading at :263-265 ("T30 regresses the EDR over -5 to -35 dB ... 50 dB is that span plus 15") are both about T30 — a 500/1000 Hz OCTAVE-band quantity of the DECODED WAVEFORM (`config.iso_eval_freqs: [500, 1000]`), not a third-octave STFT band. Units/convention mismatch under the "fixed acoustic constants declared and consistent across stages" rule. Direction of error is conservative (over-rejects, fails loudly), which is why this is major and not a blocker. | Declare which band set the guard reads. Either restrict the operand to the bands the reported metrics use, or re-calibrate the 50.0 against the min-over-all-27-bands operand the code actually applies and say so in both the Params docstring and the yaml.

AC-37-R5 | acoustics-reviewer | major | OPEN | `src/amcd/representations/spectrogram.py:372`; consumer `src/amcd/evaluation/room_acoustic.py:704`, `:716` | The AC-37 guard does not protect channel 0 — the only channel every reported ISO-3382 metric reads. `amax(energy_db, dim=(0,2))` maxes over CHANNELS, so one healthy channel masks a W channel sitting on the absolute floor. MEASURED: with W driven 70 dB down and channels 1-3 untouched, W's own headroom is 46.9 dB at 500 Hz and 36.7 dB in its worst band — below the declared 50.0 — and `encode` ACCEPTS. `compute_room_acoustic_metrics` then reads `pred_ir[0]` / `high_ref_ir[0]`, so the exact failure the guard exists to prevent (decode boosting a clamped band up to `min_db`, injecting a non-decaying floor inside a shared Schroeder window the prediction does not control, reporting a T30 that is a property of `min_db`) remains open on the channel the metric reads. Not hypothetical under the real backend: `src/amcd/simulators/base.py:88-94` declares GSound-SIR as `acn_n3d`, and N3D scales degree l by sqrt(2l+1) — sqrt(7) at the configured `ambisonics_order: 3` — so a directional field can legitimately put a higher-order channel above W. | Either take the headroom over the metric-bearing channel (ch 0 / W) as well as the max, or state and test the assumption that W is the largest channel. Today's masking is invisible because `evaluation/spatial.py` is stubbed; it becomes load-bearing with AC-15/RD-25.

AC-26-R6 | acoustics-reviewer | minor | OPEN | `configs/base.yaml:207` `metric_band_resolvability_margin`, `src/amcd/evaluation/room_acoustic.py:244-275` | The resolvability floor is a self-measured criterion, not the standard's, and does not say so. `margin 2.0 x` the filter's own decay gives 35.8 ms (T30, 500 Hz) and 23.5 ms (EDT, 500 Hz). ISO 3382-2:2008 expresses the same concern as a bandwidth-time product, BT > 16, which at the 500 Hz octave (B = 354 Hz) requires T > 45 ms — i.e. the shipped floor is MORE permissive than the standard's guidance by ~9 ms at 500 Hz. Not wrong (a filter-measured floor is defensible and is measured through the very path it governs), but a reader will take "what the band can resolve" for the ISO criterion. | Name the criterion as project-defined in the config comment and cite BT > 16 as the standard's alternative, so the two are not conflated.

AC-19-R7 | acoustics-reviewer | minor | OPEN | `src/amcd/representations/spectrogram.py:33-38`, same figures in `configs/representations/spectrogram.yaml` | Measured in-band fractions have drifted from the quoted ones: docstring says "99.4 % at 500 Hz and above, 93.4 % at 250 Hz, 56.8 % at 125 Hz"; `describe_bands()` on the production framing (48 kHz, n_fft 2048, hop 512, min_bins_per_band 1) now returns 0.9992 / 0.9464 / 0.5771. 0.5-1.2 pp, no decision depends on it. | Refresh the three figures, or state the framing/commit they were measured at.

### Confirmed correct (no finding) — recorded so the next pass need not re-derive

* **C50 is computed before the resolvability verdict and no value depends on the ordering.** Forcing `band_resolvability_margin` 2.0 -> 1e6 (so every band is floor-limited and both T30 and EDT enter the verdict dict) leaves T30/EDT/C50 bit-identical: C50 8.9146858865 both ways at 500 Hz, 10.8781446525 at 1000 Hz. `c50` is read only into the `c50_note` f-string at `room_acoustic.py:515`, and `resolvability` is keyed on T30/EDT only, so C50 never carries a verdict. AC-42's guard is applied inside the C50 branch and is independent of it.
* **AC-38's per-leg-role application does what it specifies, and band-averaging over a common band set (AC-08/AC-25) is intact.** With the physical legs themselves below the EDT floor (alpha 0.98, d 1.0 m, 3x3x2.4 m): `resolvability_limited_hz` [500, 1000], `pred_unresolved_hz` [], all three legs REPORTED (EDT low 0.007516 / pred 0.007470 / high 0.007519). With the physical legs resolving and pred a 5 ms decay: pred NaN in T30/EDT/C50 while low/high keep their values (T30 low 0.23919 / high 0.23930). The kept band set is `all(finite[leg][b] for leg in physical_legs)` (`room_acoustic.py:779`) — pred never votes — and for the physical legs `finite` no longer depends on the floor at all, so the average's composition is unchanged by the disclosure.
* **Disclosing a below-floor T30/EDT for the physical legs is the more ISO-aligned of the two options.** ISO 3382-2 treats a small bandwidth-time product as an uncertainty to be REPORTED, not a validity gate; censoring an estimator on its own magnitude is the distortion AC-38 measured (+7.5 % survivor bias at true T60 0.04 s).
* **AC-39's wording is physically correct.** A large C50 IS diagnostic of a direct-arrival-dominated first 10 dB rather than filter ringing: at alpha 0.98 / d 1.0 m the high leg has EDT 0.00752 s (below both the 0.0235 s and 0.0115 s floors) while T30 reads 0.0678 s — the room decay measures fine and only the first 10 dB is short — with C50 +55.3 dB. T30's -5..-35 dB window sits past the direct arrival's influence, so keeping T30's wording filter-oriented is right. CAVEAT: the C50 the reason cites carries F-68-R2's convention bias (+1.7 dB at that corner), and can be NaN precisely when AC-42-R1 fires, degrading the reason to "C50 unscored in this band" in exactly the direct-dominated regime it was written to explain.
* **Schroeder integration runs backward** — `[4,3,2,1] -> [10,6,3,1]`, `room_acoustic.py:224`.
* **C50 window bounds are ISO-conformant** — `split = ceil(0.050 * 48000) = 2400` = exactly 50.000 ms, boundary sample in the LATE window; late integrates only to the Lundeby index, not the full record (AC-04).
* **Octave band decomposition is clean** — a 500 Hz tone puts 99.93 % in the 500 band and 0.01 % in the 1000 band; 1000 Hz gives 99.97 % / 0.01 %; the 707 Hz crossover splits 25/25 as an 8th-order (filtfilt-doubled) Butterworth should. No leakage between the two eval bands.
* **Third-octave bank conserves energy within its covered range** — 0.9390 of a white-noise IR's STFT power lands in bands, 0.0610 uncovered (matches the declared `uncovered_bin_fraction` reasoning); ZERO FFT bins are claimed by more than one band.
* **The reported metric path is the decoded waveform, not the energy grid** — `evaluation/evaluator.py:103-114` loads `{scene_id}_decoded_ir.npy` and calls `compute_room_acoustic_metrics`; the energy-domain helpers are private and marked training-only.
* **Ambisonic conventions are consistent for every exercised path** — channel count (order+1)^2 at `config.py:656`, ACN puts W at index 0, all scalar metrics use channel 0 where N3D and SN3D agree exactly, and `simulators/base.py:88-94` documents the acn_n3d/SN3D hazard as becoming load-bearing only when `spatial.py` is filled in. No silent mismatch.

---

## Falsifier self-check on `lane/M-cycle4` (NOT a clean pass)

Read-only audit of the current worktree state. The A/B claim was reproduced
independently: `git archive HEAD` extracted to scratch, canonical dry run on both
trees into separate scratch run dirs, `ci_table.csv` diffed cell-by-cell →
**7 cells in exactly 1 of 20 rows moved** (`test_material_shift`/`EDT`:
`pred_mean` 0.0649068→0.0633868, `pred_ci_lower` 0.0231186→0.0185588, `pred_std`
0.0451436→0.0472802, `improvement_mean` −0.0133272→−0.0115544, `improvement_std`
0.0086125→0.0093776, `improvement_mdes` 0.0281117→0.0306090, `n_partial_band`
1→0), `n_scored` 3 on every row in both. Suite: 373 passed, 5 consecutive full
runs. The claim is TRUE as stated. The findings below attack what it licenses.

| ID | source | severity | status | anchor | finding | resolution |
|----|--------|----------|--------|--------|---------|------------|
| F-M1 | falsifier | blocker | OPEN | `src/amcd/evaluation/room_acoustic.py:479-496` | AC-42's `residue` is NOT a float32 accumulation bound and behaves as an undeclared C50 ceiling that deletes the PHYSICAL legs. It uses `max(energy[:trunc_idx])` — the direct-arrival peak — to bound a sum taken over `energy[split:trunc_idx]` only. `energy` is per-element float32, so the accumulation residue of that sum is bounded by the max of its own SUMMANDS. Probe: `late / (sqrt(n)·eps·max(late window))` = 1.2e7 at every tail amplitude tested INCLUDING the 5 ms "degenerate pred" the guard was written for (8.2e6 @500 Hz, 1.8e7 @1 kHz) — the correct bound never fires. What fires is DRR: sweeping true T60 with identical physical legs, C50 is scored at 0.05 s (+60.6 dB) and NaN with the AC-42 reason at T60 ≤ 0.04 s, while `room_acoustic.py:420` states base.yaml's support admits Eyring T60 = 0.0179 s. So AC-38's own thesis (never censor an estimator on its own value) is violated by AC-42 in the same file, on low/high, with censoring probability monotone in absorption — confounded with the `test_material_shift` axis. Also leg-asymmetric: residue tracks each leg's own peak (canonical run, scene_0027/500 Hz: pred margin 58.19 dB vs high 56.89 dB). And the threshold governs which scenes reach a reported metric, so `needs no config key` (l.477) contradicts CLAUDE.md. | |
| F-M2 | falsifier | major | OPEN | `src/amcd/evaluation/room_acoustic.py:799-804`, `src/amcd/evaluation/evaluator.py:190-196`, `src/amcd/stats/aggregate.py:246-262`, `src/amcd/reporting/tables.py:44-47` | The AC-38 disclosure never reaches a reader-facing artifact, and the change DELETED the only marker that was there. `n_bands_resolvability_limited` is written to `metrics.parquet` (scene_0022/EDT = 1) but `ci_table.csv` has no such column and `summary.txt`'s Caveats shows only `3 high-variance`. Before the change the same row read `1 partial-band`. Net effect on the reported table: the caveat became invisible while the row got cleaner. `room_acoustic.py:800-801` asserts the opposite ("a reader must be able to see which numbers carry the caveat"). SPANS LANE P (`stats/aggregate.py`, `reporting/tables.py`) → integrator queue, RD-82. AC-38 must not be reported closed until the column lands. | |
| F-M3 | falsifier | major | OPEN | `src/amcd/representations/spectrogram.py:370-392` | AC-37's guard is a per-band MINIMUM over all 27 ladder bands (25 Hz–20 kHz), so it is a spectral-flatness test, not a level test, and its message blames the wrong cause. Probe on the scaffold: per-band headroom spans 72.6→94.1 dB and RISES with frequency because the dry-run tail is white; applying a 2nd-order 4 kHz lowpass — far gentler than air absorption over a 4.25 s IR — drops the 20158.7 Hz band to 49.4 dB and `encode` RAISES, instructing the operator to "Fix the level (source_power / normalize_ir ...)". Any spectrally sloped render (air absorption, frequency-dependent α — a roadmap item) trips it. The calibration table at `spectrogram.py:229-243` is therefore a calibration of the scaffold's spectral flatness, not of level. | |
| F-M4 | falsifier | major | OPEN | `src/amcd/evaluation/room_acoustic.py:766-789`, `src/amcd/stats/aggregate.py:227-242` | The residual pred-side selection is still selection on the dependent variable, and the fix gave it a NEW confound. pred is NaN for the whole metric when it falls below the floor in a band the physical legs resolve, so the scene leaves `paired_improvement` — and the scenes removed are those where pred decays fastest, i.e. the largest \|pred−high\| errors, so `improvement_mean` is biased OPTIMISTIC. New property: whether an identical degenerate pred is censored now depends on the PHYSICAL legs' decay, so censoring probability is a function of room absorption — again correlated with the `test_material_shift` axis. Zero coverage in the A/B: `n_bands_pred_unresolved == 0` for all 60 (scene, ISO-metric) cells. | |
| F-M5 | falsifier | major | OPEN | `docs/lanes/cycle4-M.md` (pass condition), evidence: `metrics/metrics.parquet` of the canonical dry run | The pass-condition A/B has near-zero power over the code it validates. Of 60 (scene, ISO-metric) cells, exactly ONE has `n_bands_resolvability_limited != 0`; `n_bands_pred_unresolved == 0` everywhere; no C50 leg is within 32.3 dB of AC-42's cliff; AC-37's guard raises on no scene. So "1 of 20 rows moved" is equally consistent with the change being correct and with 3 of the 4 changes never having executed. AC-42 and AC-37 have ZERO coverage; AC-38's changed branch has n=1. It would miss F-M1, F-M3 and the whole of F-M4. Needs a corner-of-declared-support A/B config (high α, small rooms) where the changed branches fire. | |
| F-M6 | falsifier | major | OPEN | `src/amcd/pipeline.py:95-116`, `:272` | AC-37's guard is bypassable through the preprocess cache. `_preprocess_fingerprint` carries no `code_version` while train/infer/eval do (RD-59's own argument), and adding a guard inside `encode` is a code-only change — so on an existing run_dir `preprocess` prints `[skip]` and the guard never runs. The PARAMETER is covered (verified: `min_db_headroom_db` appears in the preprocess and eval fingerprints); the CODE half is not. Separately, `diagnostics` has no fingerprint at all, and the D0b oracle is the artifact `min_db_headroom_db = 50.0` was calibrated from — the calibration evidence itself can be served stale. LANE P file → integrator. | |
| F-M7 | falsifier | major | OPEN | `src/amcd/stats/aggregate.py:44-57` | At n=3 the reported "95 % CI" is identically the SAMPLE RANGE and cannot respond to the middle observation. Measured: 18 of 20 rows have `improvement_ci_lower == min(paired)` and `improvement_ci_upper == max(paired)` exactly. Direct demonstration from this very A/B — the middle scene's paired EDT moved 74 % (−0.012512 → −0.007194), `improvement_mean` moved, and the CI did not change by one digit. Nominal coverage of [min,max] at n=3 is ~0.75, not 0.95, yet `summary.txt` labels it `Imp 95% CI` and every row's interval excludes 0. `bootstrap_ci` already special-cases n==1 and `mdes()` already returns NaN for n≤2 (F-13/F-14), so the analogous small-n guard for the CI is the missing piece. LANE P file → integrator. | |
| F-M8 | falsifier | minor | OPEN | `src/amcd/provenance.py:79`, cf. `src/amcd/evaluation/evaluator.py:40` | F-69 attribution REFUTED as proven, CONFIRMED as contamination. `code_version` hashes `rglob("*.py")`, and 10 of the 27 files in the `eval` scope are AppleDouble `._*.py` sidecars; excluding them changes the digest (`dc98386526b7f131` vs `ad0348e7732e479d`). `evaluator.py:40` already filters `._` for the same host artifact — one place handles it, the other does not. Consequences independent of the flaky test: the cache key is a function of filesystem metadata, so it differs between this exFAT checkout and a native Linux checkout of identical source (cross-platform requirement), and a copy or `git clean -x` moves it. But the intermittency is NOT confirmed: 5 consecutive full-suite runs (1 with `-p no:randomly`, 4 randomised) all gave 373 passed, and `._room_acoustic.py`'s mtime/digest were unchanged across them while `room_acoustic.py`'s mtime moved — so writing an existing file does not rewrite its sidecar. The live mechanism that remains is CREATION of a sidecar for a file that has none today (e.g. `evaluation/evaluator.py`). Discriminating probe: log the (path, sha) list `code_version` hashes and diff it across the failing test's before/after. | |
| F-M9 | falsifier | minor | OPEN | `src/amcd/evaluation/room_acoustic.py:830` vs `:852-866` | Unexercised path that would silently lose a drop reason: line 830 writes `nan_reasons[(metric, leg)]` for a physical leg that caused a band exclusion, and line 860 unconditionally OVERWRITES the same key for every leg floor-limited in a kept band. One reason per (metric, leg) reaches `drops.csv` (`evaluator.py:215-225`), so a leg that both caused an exclusion and is floor-limited loses the exclusion reason. I could not construct an instance inside base.yaml's support (a Schroeder EDR is monotone so "non-decaying EDR" is near-unreachable, and Lundeby's 480-sample floor makes "<2 samples" unreachable), so this is "needs a guard or test", not "remove it". | |
| F-M10 | falsifier | minor | OPEN | `src/amcd/representations/spectrogram.py:361` vs `configs/representations/spectrogram.yaml:48`, `tests/test_filterbank.py:100` | Evidence/config drift in the new guard: the docstring says "Measured on a definitionally perfect oracle at the shipped 55 dB" while the shipped value is 50.0 and the test fixture uses 55.0. A measured claim stated against a value that is not the one in force. | |
| F-M11 | falsifier | minor | OPEN | `src/amcd/data/normalization.py:29-32`, `src/amcd/data/preprocess.py:132-133`, `src/amcd/evaluation/evaluator.py:83-85` | `low_mean`/`low_std` are computed and stamped into `preprocessed/meta.json` but never consumed — both legs are normalised with the HIGH stats. Emits output, contributes nothing to any inferential result. NOT leakage (train-only, verified) and not lane M's file → integrator. | |

### Checks that came back clean (tried and could not break)

- **Split leakage.** `preprocessed/splits.json`: 12 train / 5 valid / 3 test_id /
  3 each shift split, 29 scenes, 29 unique, zero pairwise overlap.
- **Normalization leakage.** `data/normalization.py:1,8-31` — stats from the
  training split only; `preprocess.py:115` passes `train_lows`/`train_highs`.
- **Fingerprint coverage of the NEW parameter.** `min_db_headroom_db` reaches the
  preprocess, train, infer and eval fingerprints through
  `config.representation.params` (verified by calling `_preprocess_fingerprint`
  and `_eval_fingerprint`). The parameter has no default: the `Params` dataclass
  and `__init__` both require it, so an undeclared config raises. (The CODE half
  of preprocess is still uncovered — F-M6.)
- **Scaffold / platform coupling.** No `isinstance(..., DryRunSimulator)`, no
  `platform`/`sys.platform`/`Darwin` branch, no `/Volumes` literal anywhere in
  `src/amcd/`. `training/infer.py:19-24` falls back MPS → CUDA → CPU.
- **Bit-reproducibility.** An independent from-scratch run in a fresh scratch run
  dir reproduced the claimed post-change table exactly
  (`pred_mean` 0.063387, `improvement_mdes` 0.030609, `n_partial_band` 0).

---

## Builder's record — lane M, cycle 4

### Assigned rows: 9 of 9 acted on

Counted against `LANE.md`, not against this session's scope (RD-73's trap).

| row | outcome |
|-----|---------|
| AC-37 | FIX APPLIED + known-answer test. Guard in `spectrogram.encode`. |
| AC-38 | **PARTIALLY DISCHARGED** — parquet half done, reported-column half spans lane P. |
| AC-39 | FIX APPLIED, verified at the corner the row names. |
| AC-42 | FIX APPLIED on the **second** attempt; the first was a blocker, see AC-42-B1. |
| AC-28 | VERIFIED CLOSED (defect gone), but the row's acceptance criterion is wrong — AC-28-B2. |
| F-68 | FIX APPLIED, re-measured; acoustics-reviewer then refined it further (F-68-R2, still OPEN). |
| RR-37 | FIX APPLIED. |
| RR-38 | FIX APPLIED. |
| RR-45 | **ALREADY CLOSED** — verified, no edit needed. Evidence below. |

### Pass condition — the fixed-seed `ci_table.csv` A/B

Baseline captured at the branch point BEFORE any edit; comparison after every
change. **Exactly 1 of 20 (split, metric) rows moved**, and it is the one
pre-registered above:

```
test_material_shift / EDT
    pred_mean            0.0649067603777645 -> 0.0633868393663180
    pred_ci_lower        0.0231185540572927 -> 0.0185587910229530
    pred_std             0.0451436084201951 -> 0.0472802303218847
    improvement_mean    -0.0133271947161191 -> -0.0115544302772808
    improvement_std      0.0086125291622205 -> 0.0093776452094210
    improvement_mdes     0.0281116702578466 -> 0.0306090423564209
    n_partial_band                        1 -> 0

n_scored unchanged on every row (3 everywhere);  n_attempted unchanged.
```

`pred_mean` FALLS, which is the direction that removes an upward survivor bias,
and MDES is preserved rather than deleted. The falsifier independently reproduced
this A/B from `git archive HEAD` and got the same 7 cells.

**The A/B is necessary and NOT sufficient, and I am saying so rather than letting
a quiet table read as a clean result.** The falsifier measured its power (F-M5):
of 60 (scene, metric) cells, one has `n_bands_resolvability_limited != 0`, none
has `n_bands_pred_unresolved != 0`, AC-37's guard fires on no scene, and no C50
leg is near any guard. AC-37 and AC-42 therefore have **zero** coverage in the
A/B and AC-38's changed branch has n=1. The synthetic known-answer tests are the
real evidence for those three; the A/B's job is to prove nothing ELSE moved.

Suite: **377 passed, 1 failed** — the failure is F-69, pre-existing and not this
lane's file. See F-69-B4.

### AC-42-B1 | builder | blocker | RESOLVED IN-LANE — recorded because it shipped briefly

`src/amcd/evaluation/room_acoustic.py` (C50 branch)

My first AC-42 guard was a "float32 accumulation residue",
`sqrt(n_late) * eps * max(energy[:trunc_idx])`. It is WRONG and both reviewers
caught it independently (acoustics-reviewer AC-42-R1, falsifier F-M1).

- `late` is a sum over the LATE window; `max(...)` spans the EARLY one, so it
  bounds no error in that sum. float32 carries per-element RELATIVE precision.
- Algebraically it reduced to a C50 ceiling set by the direct arrival's crest
  factor: measured 57.6-62.2 dB varying with arrival SHAPE alone at a fixed tail
  level, and moving the WRONG way — a sharper arrival lowered the ceiling.
- Worst: it censored the **physical** legs, firing at true T60 <= 0.04 s with a
  probability monotone in absorption, i.e. confounded with `test_material_shift`'s
  own independent variable. That is the censoring-on-the-datum defect AC-38
  removes three blocks higher in the same file.

**Removed.** C50 now inherits **T30's** verdict and only T30's: where T30 is
unresolvable, the energy after the 50 ms split — C50's denominator — is filter
ringing, not reverberation. Not EDT's (RD-98: at AC-39's corner EDT is below floor
while C50 measures +48.8 to +55.3 dB, a real direct-arrival quantity). Under
AC-38's policy this can only ever touch `pred`, and only in bands the physical legs
resolve, so no physical leg is censorable by construction.

Verified with both reviewers' own confirming probes:

```
pred == high == low, sweeping true T60 across the declared support:
   0.0179 s -> C50 +154.65 dB  low/high SCORED      (Eyring min the support admits)
   0.0200 s -> +145.52          SCORED
   0.0300 s -> +97.39           SCORED
   0.0400 s -> +72.41           SCORED
   0.0500 s -> +57.43           SCORED
  physical legs censored at: NOWHERE

AC-42's own case (degenerate 5 ms pred, healthy legs):
  C50 low +7.83  high +9.29  pred nan;  pred_unresolved_hz [500.0, 1000.0]
AC-39's corner (direct-dominated): C50 +38.85 dB, still SCORED
```

Regression guard: `test_a_physical_leg_is_never_censored_for_its_own_c50`,
parametrized over T60 in {0.0179 … 0.05}.

**Lesson worth keeping:** I reached for a "numerical, therefore not a hidden
default" guard specifically to stay inside my lane and avoid a `config.py` span.
That motivation produced a wrong derivation. RD-99 anticipated exactly this and
said not to hardcode a dB threshold to stay in-lane; the trap was subtler — a
threshold that *looked* like arithmetic.

### AC-38 — PARTIALLY DISCHARGED, do not close (RD-94)

Done, in lane M: the floor no longer suppresses. The verdict is returned per band
by `_iso3382_band_metrics`; `compute_room_acoustic_metrics` discloses for the
physical legs and suppresses `pred` only where the physical legs DO resolve the
band. New `metrics.parquet` columns `n_bands_resolvability_limited` and
`n_bands_pred_unresolved_in_floor_limited`.

Evidence that the bias is removed rather than moved — 200 realizations/point:

```
 true T60   suppressed   disclose err   suppress err
  0.02 s     279/400        +8.3 %        +13.4 %
  0.03 s     194/400        +5.4 %         +7.0 %
  0.04 s     100/400        +3.1 %         +7.1 %
  0.05 s      12/400        +2.4 %         +3.3 %
  0.06 s       0/400        +2.6 %         +2.6 %
```

Disclosing is strictly closer to truth wherever the floor bites and IDENTICAL
where it does not.

**NOT done, and it is not mine:** the per-split count must reach `ci_table.csv`
and `summary.txt` Caveats — `stats/aggregate.py` / `reporting/tables.py`, lane P
(RD-82). The falsifier's F-M2 sharpens this into a regression: before this change
`test_material_shift`/EDT read `1 partial-band` in Caveats; now it reads nothing,
because `n_partial_band` went 1 -> 0 and no column replaced it. **The reported
table currently got QUIETER about a caveat that still exists.** That is the F-65
failure mode and it needs the lane-P half before this row closes.

### RD-93 — the plan-time blocker, REALIZED and then removed

The research-director predicted at plan time that count-and-disclose would move
the survivor bias onto the pred leg. It did, exactly, on the first implementation:

```
test_material_shift / EDT, pred suppressed wherever below floor:
  n_scored          3 -> 2
  improvement_mdes  0.0281 -> N/A      (a split's MDES deleted — F-67's symptom)
  pred_mean         0.0649 -> 0.0858   (ROSE: the dropped scene was the low one)
```

Fixed by AC-25's own qualifier — pred is suppressed only in a band **the physics
resolves** — after which the same run reads n_scored 3, MDES 0.0306, pred_mean
0.0634. Without the plan-time review this would have shipped, and the dry-run A/B
would have shown one row moving either way.

**Residual, per falsifier F-M4 (still OPEN):** pred can still leave the paired
comparison when it fails in a resolved band, and whether an identical degenerate
pred is censored now depends on the physical legs' decay — so censoring
probability is a function of room absorption, correlated with the
`test_material_shift` axis. Smaller than before, but a new confound in kind.
**F-70 is a PREREQUISITE of this disclosure, not a sibling** (`stats/aggregate.py`
`n_scored = improved.notna().sum()`), and it is unimplemented.

### AC-28-B2 | builder | minor | OPEN — the row's acceptance criterion is wrong

`docs/review_ledger.md` AC-28; `src/amcd/simulators/dry_run.py`

AC-28's DEFECT is fixed — verified, not assumed. Measured independently of
`tests/test_simulator_seam.py`, through `build_simulator` and the real ISO path,
at the row's own geometry (10x8x3.5 m, alpha 0.2, r_c = 1.193 m):

```
  d (m)   C50 (dB)   per doubling   closed-form DRR
    0.5      11.90                          7.55
    1.0       6.99         -4.91            1.53
    2.0       3.62         -3.37           -4.49
    4.0       2.00         -1.62          -10.51
    8.0       1.42         -0.58          -16.53
  monotone decreasing: True;  total swing 10.48 dB over 16x distance
```

The placement axis is live. But the row's stated known-answer condition — "C50
must fall ~6 dB per doubling of d and cross 0 dB near d = r_c" — is **not met and
should not be**: measured 4.91 / 3.37 / 1.62 / 0.58 dB per doubling, and C50 never
goes below 0 dB. That criterion describes **DRR**, not C50: C50's early window is
the first 50 ms, which at T60 0.78 s contains a large amount of reverberant
energy, lifting C50 and compressing its distance dependence. DRR itself DOES match
the closed form (`test_realized_drr_matches_the_published_closed_form` passes at
abs=1.0). Do not "fix" the scaffold to satisfy the criterion — correct the
criterion.

Also: AC-28's coverage lives in `tests/test_simulator_seam.py:351-445`, which lane
M does NOT own, while `docs/lanes/cycle4.yaml` names `tests/test_acoustic_validity.py`
as AC-28's test file. I ran it as evidence (4 passed) rather than duplicating a
test into a file I do own.

**AC-43's measurement is now stale** (it is on the integrator queue, not mine):
AC-43 records EDT = 0.7504/0.8056/0.7828/0.7803/0.7805 s at d = 0.5/1/2/4/8 m,
"non-monotone, 5.5 % spread". I measure 0.3354/0.8069/0.8006/0.7998/0.7999 —
**66.5 % spread**, the d = 0.5 m value having moved 0.7504 -> 0.3354 since AC-43
was written (the AC-36/F-67 filter work sits in between). The row's conclusion
holds and strengthens; its numbers need restating.

### RR-45 — ALREADY CLOSED, verified, no edit made

`src/amcd/representations/spectrogram.py:74-79` already carries the legend and the
ordering cue RR-45 asked for (F-59's cycle-3 fix covered it):

```
    # It is NOT behaviour-preserving in general — only at 48 kHz (F-59). MEASURED
    # against the old rule at min_bins_per_band=1, as `new (was old)`:
    #     48000/2048 → 27 (was 27), bank bit-identical    48000/512 → 21 (was 21)
    #      8000/256  → 18 (was 19), one band LOST        44100/2048 → 27 (was 28)
```

Columns are labelled (`new (was old)`), and "one band LOST" supplies the direction
that the symmetric 48 kHz row cannot. Safe to delete the row.

### AC-37 — guard calibration, and its one known weakness

Reproduced first, through the REPORTED path. The defect is invisible on a
standalone IR (each leg gets its own Lundeby cut, which truncates the injected
floor away: 0.02 % error) and only appears through the shared window the physical
legs set — 126.8 % at the same gain. A synthetic unit-variance IR also hides it
entirely (<= 1.6 % at every gain to -60 dB) because it carries ~70 dB more
headroom than a render does. Both traps are written into the test's docstring.

```
gain    headroom   T30 high   T30 pred      error
  0 dB   72.6 dB    0.9681 s   0.9682 s     0.01 %
-20 dB   52.6 dB    0.9681 s   0.9804 s     1.27 %
-30 dB   42.6 dB    0.9681 s   2.1959 s   126.82 %
-40 dB   32.6 dB    0.9681 s  10.9364 s  1029.65 %
```

Threshold 50.0 dB calibrated over six scenes spanning the support: breach at
36.7-46.5 dB of headroom, native level 68.7-74.1 dB. A single scalar cannot be
both tight and safe (the admit/reject window is 1 dB wide), so it errs toward
rejecting; that cost is stated in the config comment.

Validated against the FULL 720-scene research_i population — the guard fires on
nothing, tightest margin +15.0 dB (`test_material_shift`, as expected):

```
split                    n   min hr  median   max   margin
test_geometry_shift     30    70.8    76.5   81.4   +20.8
test_id                 60    68.2    76.1   82.7   +18.2
test_material_shift     40    65.0    70.3   76.6   +15.0
test_placement_shift    30    69.2    75.7   85.8   +19.2
train                  500    67.5    75.8   84.8   +17.5
valid                   60    69.5    75.3   82.9   +19.5
```

Fixed after review: the guard maxed over CHANNELS, so channel 0 — the only one the
reported ISO path reads — was unprotected (acoustics-reviewer AC-37-R5). Now
per (channel, band).

**Still OPEN, not fixed by me:** AC-37-R4 / F-M3 — the limiting band is essentially
always the lowest (~25 Hz) band, a single-FFT-bin band the code's own docstring
calls a Hann-leakage measurement, while the calibration rationale is about
500/1000 Hz octave quantities. It fails in the conservative direction (raises
rather than corrupts), but the operand and the justification are about different
band sets, and a legitimately lowpassed IR would trip it with a message telling the
operator to fix the level. Needs a decision I did not want to take unilaterally
mid-session.

### F-69-B4 | builder | minor | OPEN — F-69 makes a SUITE TEST fail intermittently

`src/amcd/provenance.py` (not lane M's file)

`tests/test_stage_cache.py::TestCodeVersionSeesTheWorkingTree::test_editing_metric_code_changes_evals_version`
fails in roughly half of full-suite runs on this host and passes in isolation.
Confirmed contamination: 6 AppleDouble `._*.py` sidecars under
`src/amcd/evaluation/` alone match `code_version`'s `rglob("*.py")`, and the
falsifier measured 10 of the 27 files hashed into `code_version("eval")` are
sidecars — excluding them changes the digest.

**Attribution partly refuted, and I am recording the refutation** (falsifier F-M8):
writing an EXISTING file does not rewrite its sidecar, so the exact intermittency
mechanism is not yet pinned; sidecar CREATION for a file that has none is the live
candidate. Either way F-69's fix (filter `._`, as `evaluator.py:40` already does)
removes it. Same host-artifact class is visible in git itself here:
`error: non-monotonic index .../._pack-*.idx` on every `git fetch`.

### Findings for other lanes / the integrator queue

- **F-70 → PREREQUISITE of AC-38**, not a sibling. `stats/aggregate.py:236`.
- **AC-38's reported column** — `stats/aggregate.py`, `reporting/tables.py` (lane P).
  Without it the caveat is invisible and the table got quieter (F-M2).
- **D0b keeps the censored estimator.** `channel_band_avg_metrics` deliberately
  retains suppression (its callers are `diagnostics/probe.py:256,262`, lane S, and
  `tests/test_simulator_seam.py:395`, lane R), so eval and the D0b oracle now use
  DIFFERENT resolvability policies. Signature freeze was forced; the policy freeze
  is a choice, declared in the docstring rather than inherited (RD-96). Assigning
  the D0b half is the integrator's.
- **F-M7 (falsifier, major):** at n=3 the reported "95 % CI" is identically the
  sample range — 18 of 20 rows have `ci_lower == min(paired)` and
  `ci_upper == max(paired)`. This A/B demonstrates it: the middle scene's paired
  EDT moved 74 % and the CI did not change one digit. `stats/aggregate.py:44-57`,
  lane P.
- **F-M6 (falsifier, major):** the AC-37 guard is bypassable through the preprocess
  cache — `_preprocess_fingerprint` carries no `code_version`, so on an existing
  run_dir `preprocess` skips and the guard never runs. This is F-64, still open.
- **AC-43's numbers are stale** — see AC-28-B2 above.
- **AC-41 / RR-43** — untouched, span other lanes, as the brief directs.

### Reviewer findings still OPEN against lane M's files

Raised this session and NOT addressed, deliberately, because each needs a decision
rather than a keystroke: **F-68-R2** (the KNOWN RESIDUAL paragraph should describe
a MAGNITUDE bias on the early window, +0.06 to +1.67 dB on absolute C50 vs a causal
reference, not a placement residual — the fold nearly doubles an onset-aligned
arrival's in-band energy relative to an identical interior one), **F-68-R3** (the
fold double-counts one pad sample at each end, 1e-28 today, but the same lines are
the only handling of an unexercised short-record branch), **AC-37-R4/F-M3**,
**AC-26-R6** (the floor is project-defined, not ISO 3382-2's BT > 16, and does not
say so), **AC-19-R7** (in-band-fraction figures drifted), **F-M9** (unguarded reason
overwrite at `room_acoustic.py:860`), **F-M11** (`low_mean`/`low_std` computed,
stamped, never consumed).

### Resume here

Lane M is **NOT complete**. The nine assigned rows are acted on and the pass
condition is met, but 12 findings raised during this session remain OPEN against
lane M's files (7 above + F-M4's residual selection + the AC-38 lane-P half + the
three integrator-queue items). None is a correctness regression against the
baseline — the A/B moved exactly the one pre-registered row — but AC-38 must not be
reported closed, and F-68-R2 changes what a reader is told about absolute C50.
