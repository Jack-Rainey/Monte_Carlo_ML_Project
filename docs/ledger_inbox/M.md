# Lane M inbox — cycle5

Branch `lane/M-cycle5`. Written by lane M, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/M-cycle5", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

**Two things this file's format has to get right, both learned in cycle 4:**

- **Give every finding a FILE ANCHOR**, as `path` or `path:line`. The integrator's
  fold copies it into the ledger's anchor column, and that column is what assigns
  the row to a lane next cycle AND what the RD-33a gate counts. Cycle 4 shipped
  116 rows anchored "see inbox" and made the gate's own lift condition
  uncomputable.
- **Number new findings from YOUR id block only** (it is in your `LANE.md`). Every
  lane runs at once, so numbering from the ledger's maximum guarantees collisions.

**This file is PERMANENT, not a scratch pad.** The integrator's fold keeps compact
rows that point back here for the measurements, so it is the primary record for
its findings and is never truncated while an OPEN row cites it — see
`docs/ledger_inbox/README.md`. Write it for a reader in a later cycle who has only
this file and the ledger row that names it.

---

## PRE-REGISTRATION — written before the first edit (RD-91 / RD-149)

Timestamped by its commit: this entry is the first commit on `lane/M-cycle5`, and
every code change follows it. The declaration is worthless written afterwards.

### Expected effect on `ci_table.csv`: **NO ROW MOVES at reporting precision.**

The integration gate's step-4 A/B is the cross-lane interference detector, and it
can only discriminate interference from legitimate change if every lane declares
in advance. Lane M is the only lane permitted to move the table on the metric
path, and it declares that this cycle's work should not move it, because:

- Every planned change is a docstring, a config comment, a new test, a
  declaration, or a numerically inert guard.
- The one change touching arithmetic on the metric path is **F-68-R3**'s
  double-count removal, measured in cycle 4 at folded/full = 1.000000001490
  (500 Hz) / 1.000000004612 (1000 Hz) against exactly 1.0 without it — a relative
  effect of ~1.7e-28, far below float32 resolution.
- The **AC-37 headroom guard's operand changes** (AC-37-R4 / F-M3), but the guard
  raises on **zero** scenes in the canonical dry run today, and narrowing the
  operand can only make it *more* permissive. No scene enters or leaves the
  dataset.
- **F-M9**'s fix changes a drop *reason string* in a branch that is unreachable
  inside `base.yaml`'s declared support — no value moves.

**Any row that does move is a FINDING, not a merge artifact**, and will be
investigated and written up here rather than accepted.

### Known weakness of this pass condition, stated up front

**F-M5** (integrator queue, not lane M's row) measured that the canonical dry-run
A/B has near-zero power over exactly the code it is meant to validate: of 60
(scene, ISO-metric) cells, exactly **one** has `n_bands_resolvability_limited != 0`;
`n_bands_pred_unresolved == 0` everywhere; no C50 leg is within 32.3 dB of AC-42's
cliff; and the AC-37 guard raises on no scene. A green A/B is therefore equally
consistent with these changes being correct and with three of the four never
having executed.

Lane M cannot fix that in-lane: a corner-of-declared-support overlay would live in
`configs/overlays/`, which lane M does not own. **Compensation:** every cluster
carries a unit-level known-answer test at a corner of the declared support where
the changed branch demonstrably fires, and every new test gets a revert/mutation
negative control. **Those tests, not the A/B, are this lane's real evidence**, and
the A/B is reported as a non-interference control only — never as a demonstration
of correctness.

### Baseline

Captured on the untouched tip of `lane/M-cycle5` before any edit, by the canonical
invocation:

```
PYTHONPATH=/Volumes/T7/Monte_Carlo_Research/v3-lane-M/src \
  /Users/nortonrainey/miniconda3/envs/amcd/bin/amcd all \
  -c configs/base.yaml \
  -c configs/overlays/simulator_dry_run.yaml \
  -c configs/overlays/dry_run.yaml
```

Run dir `experiments/all_20260811_183712`; `stats/ci_table.csv` sha256
`74651cd26663fcc911979d9a7b9ddd8d97433ef4376fd829d35e6277d6f23052`, 20 data rows.
"Before" full suite on this checkout: **506 passed, 1 skipped in 62.56s**.

---

## C1 — the `_band_energy` fold  (AC-36 · F-68 · F-68-R2 · F-68-R3 · RR-38)

All anchors `src/amcd/evaluation/room_acoustic.py`. ON-PATH (`evaluation/**`).

### F-68-R3 — CONFIRMED, FIXED, and it was BIGGER than the row states

The row describes a 1.7e-28 double-count and calls the short-record branch
"unexercised". The double-count reproduces exactly:

```
folded/full   impulse@0   500 Hz 1.000000001490   1000 Hz 1.000000004612
              impulse@mid 500 Hz 1.000000001490   1000 Hz 1.000000004612
```

**But the short-record branch is not merely unexercised — it silently discarded up
to 29.7 % of the band energy at record lengths `_iso3382_band_metrics` ADMITS.**
`head = min(guard, n_record - 1)` dropped every pad sample beyond one record
length. MEASURED, `folded/full` by record length (500 Hz, guard 4608):

```
  n_record     32       40       64      128      512     1024    2304    4608
  folded/full  0.7029   0.8558   0.9545  0.9872   0.99999  1.000   1.000   1.000
  (1000 Hz)    0.9512   0.9740   0.9869  0.9992   1.000    1.000   1.000   1.000
```

`_MIN_FILTER_SAMPLES` is 32, and the guard comment beside it says a very-late onset
trim (AC-07) is exactly what leaves too few samples. So this was reachable, and it
left no `(unit, reason)` record — against the project rule that nothing leaves a
result silently.

**FIX:** fold every pad sample, with the mirror index CLAMPED into the record
rather than dropped. Conservation is now exact end to end in float64 —
`folded/full = 1.000000000000000` at both bands at every length tested — and the
~1e-08 visible through the float32 return is cast noise, varying in sign. The
residual becomes a PLACEMENT approximation confined to records shorter than one
guard width, replacing an outright energy loss. Raised as **AC-100** below.

**TEST:** `test_the_energy_fold_conserves_energy_at_every_record_length`,
parametrized over n_record ∈ {32, 40, 64, 128, 512, 2304, 4608, 9216} × both bands,
straddling the branch boundary.

### F-68-R2 — CONFIRMED, docstring was asserting the refuted claim; REWRITTEN

The KNOWN RESIDUAL paragraph still read *"The residual that remains is placement,
not magnitude: mirroring is exact for an isolated arrival."* Reproduced against the
row: onset-aligned arrival vs the identical impulse in the record interior —

```
  total energy      500 Hz onset/interior = 1.0000 (+0.00 %)   1000 Hz 1.0000
  POST-ONSET only   500 Hz             1.9660 (+96.60 %)       1000 Hz 1.9332 (+93.32 %)
```

matching the row's +96.60 / +93.32 to the digit. It is a **magnitude
redistribution into the ISO early window**, i.e. C50's numerator, not a placement
residual.

Absolute C50 against a CAUSAL reference (`sosfilt`, same 4th-order band edges),
**28 (corner, distance, band) cells** over base.yaml's declared support —
`configs/base.yaml` + `configs/overlays/simulator_dry_run.yaml`, **scene seed 7**,
`high_ray_budget` (the row's own figures named no config or seed and were not
reproducible; these are anchored):

```
  delta C50 (folded - causal)   min +0.0122   max +2.2294   mean +0.7280 dB   n=28
  monotone in direct-dominance: +0.0122 dB at d/r_c 1.44 (large, a 0.05)
                                +2.2294 dB at d/r_c 0.15 (small, a 0.98)
```

Wider than the row's +0.06…+1.67 because these corners include the small α = 0.98
room. **The max exceeds this project's own `d0b_c50_jnd_db` of 1.0**, so the
docstring now says absolute C50 carries up to ~2.2 dB of convention bias, common-mode
across LEGS but not across SCENES.

### AC-36 / F-68 — VERIFIED, with a negative control

`test_an_onset_impulse_keeps_the_band_energy_an_interior_one_gets` passes, and it
has teeth: replacing `_band_energy` with each historical variant fails it.

```
pytest tests/test_metrics.py -p negctl_c1 --negctl=<variant> -k "fold or onset_impulse or dc_tail"

  variant nofold       13 failed,  6 passed    (pre-AC-36: guard stripped)
  variant oldfold       9 failed, 10 passed    (pre-F-68-R3 fold)
  variant padtype_odd  13 failed,  6 passed    (scipy default reflecting pad)
```

`oldfold` is the discriminating one: it fails **only** n_record ∈ {32, 40, 64, 128,
512} and passes {2304, 4608, 9216} plus the AC-36 onset test — exactly the claim
that the old fold conserved in the normal case and discarded in the short one. The
harness monkeypatches the module attribute and touches no repo file
(`scratchpad/negctl_c1.py`).

### Verdicts

| row | verdict |
|---|---|
| AC-36 | CONFIRMED FIXED — property re-derived, negative control added |
| F-68 | CONFIRMED FIXED — docstring no longer asserts "paired improvements unaffected"; superseded in substance by F-68-R2's rewrite |
| F-68-R2 | FIXED this session — residual restated as a magnitude bias with config+seed-anchored figures |
| F-68-R3 | FIXED this session — conservation now exact; the branch it guarded was reachable and lossy, see AC-100 |
| RR-38 | FIXED this session — transcript compressed, padding finding carries its AC-36 id |

---

## C2 — the resolvability floor
(AC-65 · AC-26 · AC-38 · AC-39 · AC-26-R6 · F-M9 · RD-93 · RD-96 · RD-98 · RD-99)

Anchors `src/amcd/evaluation/room_acoustic.py` and `configs/base.yaml`. ON-PATH.

### AC-65 — CONFIRMED STALE, restated, and now PINNED

`_band_resolvable_decay_s` declares itself the one place the floors are written
down. Measured by calling it on this tree:

```
             DECLARED (stale)        MEASURED            drift
   500 Hz    T30 17.881 / EDT 11.765 → 20.360 / 9.556   T30 +13.9 %, EDT -18.8 %
  1000 Hz    T30  8.924 / EDT  5.771 → 10.162 / 4.802   T30 +13.9 %, EDT -16.8 %
```

Corroborated from inside the ledger exactly as AC-65 says: AC-27's resolution
quotes f·T30 = 9.85-10.18, and measured here f·T30 = **9.88-10.18** across
125-4000 Hz — the ledger and the docstring had already disagreed. Re-measured
AFTER C1's fold change, so these are current: C1 does not move them (the probe
impulse sits far from either boundary, so no clamping occurs).

Restated in the docstring; **pinned** by
`test_the_band_resolvability_floors_are_the_declared_values` (rel 5e-3, tight
enough to catch the 13.9 % drift) and
`test_the_resolvability_floors_scale_as_one_over_f`. Nothing asserted these
before — `tests/test_metrics.py` called the function without checking its result,
which is precisely how they drifted.

### AC-101 (NEW, from lane M's block) — the margin calibration does NOT reproduce

`configs/base.yaml` claimed: *"the 500 Hz T30 estimator's bias is +12.8 % at
margin 1.0, +4.9 % at margin 2.0 and +2.8 % at 3.0, so 2.0 lands where the bias
crosses d0b_t30_jnd_frac of 0.05"*. Re-measured — 300 realizations/point, true T60
log-spaced 0.015-0.30 s, T30 through the real `_band_energy` path, seed 20260811,
bias of the population left UNFLAGGED at each margin:

```
   margin     1.0    1.5    2.0    2.5    3.0    4.0
   500 Hz   +5.91  +4.15  +3.63  +3.26  +2.97  +2.62  %
  1000 Hz   +2.22  +2.64  +2.62  +2.49  +2.33  +2.09  %
```

The 500 Hz bias crosses the 5 % JND **between margin 1.0 and 1.5, not at 2.0**,
and at 1000 Hz it is under the JND at *every* margin including 0, so the floor
buys nothing there on this criterion. Cause is AC-65's drift: the old figures were
measured against floors the AC-36/F-67 fold then moved, and the declaration never
followed.

**2.0 is KEPT**, but the justification is rewritten and is now weaker, which the
config says in those words. Under AC-38 the floor DISCLOSES rather than
suppresses, so a too-wide margin costs extra caveats and biases nothing; only a
too-narrow one would let an unresolvable band be reported without a caveat. My
protocol necessarily differs from the original's (which named none), so this is
recorded as "does not reproduce", not as a refutation of the original measurement.

### AC-26 / AC-38 — RE-DERIVED AND CONFIRMED CORRECT; NEITHER CLOSES

200 realizations/point, margin 2.0, seed 20260811 — error of the reported mean
against truth:

```
   true T60  band  suppressed   disclose err   suppress err
    0.020 s   500   200/200        +13.4 %     NO VALUE AT ALL
    0.020 s  1000    96/200         +3.3 %         +14.6 %
    0.030 s   500   192/200         +6.4 %         +38.9 %
    0.040 s   500    79/200         +5.7 %         +14.8 %
    0.050 s   500    12/200         +3.7 %          +5.4 %
    0.060 s   500     0/200         +3.7 %          +3.7 %
    0.080 s   500     0/200         +3.4 %          +3.4 %
```

Disclosing is strictly closer to truth wherever the floor bites and **identical**
where it does not — AC-38's central claim, independently re-derived. Stronger than
the row states: at true T60 = 0.02 s the 500 Hz band suppressed **all 200**
realizations, so suppression does not merely bias the split mean, it deletes it.
That corner is inside base.yaml's own declared support (Eyring admits 0.0179 s).

**Neither row closes.**
- **AC-38** — its reported-column half is lane P's (`stats/aggregate.py`,
  `reporting/tables.py`); F-70 is its prerequisite and F-M2 records that the table
  got QUIETER. **Gate: lane P's column lands.**
- **AC-26** — its own resolution reads *"PARTLY CLOSED, NOT CLOSED — see AC-38 …
  Superseded by AC-38"*, so it is chained to AC-38. AC-26 is a **gate-path major**:
  RD-33a condition (i) therefore cannot lift for `evaluation/**` this cycle
  whatever lane M does. **Gate: AC-38.** Flagging this now rather than at the merge.

### AC-39 / RD-98 — CONFIRMED FIXED, at AC-39's own corner

Corner 3×3×2.4 m, α 0.98, d 2.0 m (shoebox dims lo × ceiling_absorptive α hi —
inside the declared support), high leg:

```
   500 Hz  T30 0.0782  EDT 0.0110  C50 +49.01 dB   flagged: ['EDT']
  1000 Hz  T30 0.0737  EDT 0.0059  C50 +53.36 dB   flagged: ['EDT']
```

Exactly AC-39's structure: the room decay measures fine, only the first 10 dB is
short, and the cause is the direct arrival. The EDT reason names **both**
mechanisms and carries the band's C50 (`names direct arrival: True`, `names filter
ringing: True`, `carries the band C50: True`).

RD-98 holds: EDT is flagged in both bands, T30 is not, and **C50 stays SCORED** at
+49.01 / +53.36 dB. C50 inherits T30's verdict and not EDT's, which is the whole
of RD-98.

### RD-93 / RD-96 / RD-99 — CONFIRMED FIXED

- **RD-93** — `pred_unresolved_in_floor_limited_hz` is present and computed in
  `band_accounting` for all three metrics. At this corner EDT shows
  `resolvability_limited_hz = [500.0, 1000.0]` with the physical legs still SCORED
  (low 0.008418, high 0.008418) and the AC-38 caveat attached to all three FINITE
  legs — the disclosure reaches `nan_reasons`, not just a count.
- **RD-96** — the D0b policy divergence is declared in
  `channel_band_avg_metrics`'s docstring as a deliberate choice, naming
  `probe.py:256,262` as the unassigned half.
- **RD-99** — no hardcoded dB threshold survives on the C50 path; the float32
  crest-factor bound is gone and appears only in the comment recording why it was
  wrong.

### F-M9 — FIXED, with a constructed collision test

`nan_reasons[(metric, leg)]` was assigned unconditionally by the AC-38 disclosure,
silently deleting a band-EXCLUSION reason written for the same key. Now appends
(`… | ALSO: …`).

Test `test_a_leg_that_both_excludes_a_band_and_is_floor_limited_keeps_both_reasons`
injects the collision at the per-band layer, because F-M9 records that no instance
is reachable inside base.yaml's support. **Negative control:** reverting to the
overwrite fails it, and the failure output shows the exclusion reason gone
entirely — only the caveat survives, which is exactly the `drops.csv` loss F-M9
predicts.

### AC-26-R6 — FIXED

`configs/base.yaml` now names the floor as project-defined and cites ISO
3382-2:2008's BT > 16 as the standard's alternative. At margin 2.0 the shipped
floor is **40.7 ms** at 500 Hz against the standard's **45 ms** — more permissive
by ~4 ms (the row said ~9 ms against the old 35.8 ms floor; the AC-65 correction
narrows the gap). Stated so the two criteria are not conflated.

---

## C3 — octave-filter conformance  (AC-68)

`src/amcd/evaluation/room_acoustic.py`. ON-PATH.

**CONFIRMED, reproduces the row to the digit.** Rejection through `_band_energy`,
pure tone, dB re the tone's total energy:

```
   fc      -2 oct   -1 oct   lo edge   centre   hi edge   +1 oct   +2 oct
   500 Hz  -46.59   -37.43    -6.00     -0.00    -6.01    -38.49   -47.33
  1000 Hz  -49.59   -40.29    -6.01     -0.00    -6.01    -41.36   -50.48
```

(row: −37.4 / −38.5 at one octave, −46.6 / −47.3 at two.)

**FIXED — all three parts of the row, including the headline half.**

1. Realized figures declared in `_butter_octave_filter`, with the reason the order
   is NOT simply raised: the order also sets the ringing `_band_resolvable_decay_s`
   measures, so a steeper filter buys selectivity with a LONGER unresolvable floor.
   That is a research trade, not a code cleanup.
2. **The module docstring's "the standard ISO-3382 path" claim is now qualified.**
   This was the row's headline and the easiest half to drop: declaring a threshold
   at the measured figure without saying IEC 61260 class 1 is unmet would convert a
   conformance gap into a passing test. The module now lists TWO declared
   departures — this one and AC-26-R6's project-defined resolvability floor — and
   says "the standard ISO-3382 path" names the procedure, not a conformance claim.
3. Known-answer tests: `test_the_octave_filter_meets_its_declared_stopband_rejection`
   (both bands × 1 and 2 octaves × both sides) and
   `test_the_octave_filter_edges_are_minus_six_db_and_power_complementary`, which
   pins what the filter gets RIGHT so a future attempt at class 1 cannot silently
   break energy conservation across the decomposition.

**Negative control:** dropping the filter to 2nd order fails
`test_..._stopband_rejection[500.0-1]` and `[1000.0-1]`. The −6 dB edge test
correctly still passes — band edges are order-independent by design — which is the
discrimination that shows the two tests measure different properties.

**DEPARTURE FROM THE ROW'S REMEDY, deliberate, with the reason.** AC-68 asks for a
"config-declared rejection". `configs/base.yaml` is lane M's but its schema is
`src/amcd/config.py`, which is lane P's and sets `extra: forbid` — so a new key
fails validation without a field lane M cannot add. This is the config-contention
case `docs/parallel_protocol.md` documents. The bound is therefore pinned as
`_DECLARED_STOPBAND_DB` in `tests/test_metrics.py`, the same class as
`_MIN_FILTER_SAMPLES` and `_DECLARED_FLOORS_48K`: nothing in the pipeline reads it
and it governs no experiment, so it is a declared property of the filter design
rather than a hidden experiment default. **The config-declaration half is filed as
RD-186 below (spanning row).**

---

## C4 — the `min_db` headroom guard
(AC-37 · AC-37-R4 · F-M3 · F-M10 · AC-69 · AC-19-R7 · RR-45)

`src/amcd/representations/spectrogram.py`, `configs/representations/spectrogram.yaml`,
`tests/test_metrics.py`, `tests/test_filterbank.py`. **OFF-PATH** (`representations/**`
is not on RD-33a condition (i)'s list) — but this is the Research II decode floor and
is the most consequential work on the lane.

### The decision (user direction: highest scientific accuracy)

Guard the quantity actually at risk; disclose the rest. The operand is now the
ladder bands inside the reported ISO octave span, declared as
`min_db_headroom_octave_centres_hz: [500.0, 1000.0]` — **centres, not a `[lo, hi]`
range**, so a future non-contiguous `iso_eval_freqs` cannot silently re-admit every
band between.

### AC-37-R4 — CONFIRMED in structure, with one correction to the row

The row says the limiting band is "ALWAYS the 24.8 Hz band". **Measured on the
calibration scene it is the 99.2 Hz band** (headroom 72.56 dB; 24.8 Hz is present in
the ladder at in-band fraction 0.5787 but is not the minimum). The row's *substance*
holds — the verdict is set by a low, few-bin band outside the reported span — but
"always the 24.8 Hz band" is too strong. Measured gap between operands on that
scene: **4.91 dB** (ISO-span min 77.48 dB at 396.9 Hz vs all-band min 72.56 dB).

### The re-calibration — 50.0 → 52.0, and it REFUTES the old calibration's shape

Population: **`valid` split ONLY, n=5** (scene_0005/0006/0016/0017/0018 from
`experiments/all_20260811_183712/preprocessed/splits.json`). A threshold deciding
which scenes ENTER the dataset is a tuned value and must not be selected on any
`test_*` scene. Method: 1 dB level sweep; oracle `decode(encode(high), low)` read
through the REPORTED path so it inherits the physical legs' shared Schroeder window.

```
  scene         native   last OK   breach   err at breach
  scene_0005      81.8      51.8     50.8       6.0 %
  scene_0006      72.0      43.0     42.0       6.7 %
  scene_0016      75.0      48.0     47.0       5.5 %
  scene_0017      78.2      48.2     47.2       6.1 %
  scene_0018      77.3      49.3     48.3       6.3 %
```

**No single scalar is both tight and safe, and by far more than the old comment
admitted.** Rejecting every breach needs > 50.8; admitting every survivor needs
≤ 43.0 — an **8 dB inversion**, not the "1 dB window" previously claimed. Scene
dependence is ~9 dB here and was ~8 dB in the old all-band operand (same scenes
breached at 31.0-39.0 dB), so **restricting the operand did not make the criterion
sharper — it made it correct**, which is a weaker and different claim, and is
written that way in both the yaml and the Params docstring.

52.0 errs toward REJECTING per the project's stated asymmetry. Cost is zero in
practice: every valid scene carries 72.0-81.8 dB at native level.

**ACCEPTANCE CRITERION MET.** The pre-existing oracle tests still partition the same
gains: `test_the_decoded_oracle_reproduces_its_targets_t30_at_any_level` passes at
{0, −20} dB and `test_a_scene_that_would_breach_the_t30_jnd_is_refused_by_encode`
still raises at {−30, −40} dB. Narrowing the operand makes the guard strictly more
permissive, so this was the thing that could have broken and did not.

### AC-69 — CONFIRMED NOT FIXED, now fixed and controlled

The row's demonstration reproduces: every prior test encodes a single channel, where
`amax(dim=2)` and `amax(dim=(0,2))` coincide. Added
`test_the_headroom_guard_names_the_offending_channel_not_just_a_band` — 4-channel
field, W attenuated 70 dB, channels 1-3 native; asserts `encode` raises and names
channel 0. Also fixed the stale `dim=(0, 2)` at `tests/test_metrics.py` that sat
beside the guard as a wrong template.

**Negative control:** reverting the guard to `amax(dim=(0,2))` fails it.

### F-M3 — FIXED, with the lowpass case as a regression test

`test_the_headroom_guard_ignores_a_spectral_slope_outside_the_metric_bands` applies
the row's own 2nd-order 4 kHz lowpass and asserts `encode` does NOT raise.
**Negative control:** reverting the operand to all ladder bands fails it — so the
row's scenario is real and the fix removes it.

The raise message no longer blames level unconditionally: it names the bands
checked, the octave spans, and says a merely spectrally-sloped scene no longer
reaches it.

### AC-37-R4 drift guard

`test_the_headroom_guard_reads_exactly_the_reported_metric_bands` reads
`configs/base.yaml` and asserts **EQUALITY** with `iso_eval_freqs` — equality, not
containment, because over-coverage is the defect being fixed. **Negative control:**
adding 2000.0 to the declared centres fails it.

### F-M10 — FIXED

Resolved by the re-calibration: one value is in force (52.0) and every measured
claim beside it is stated against that value. The 55.0 test fixture is gone —
`tests/test_filterbank.py`'s ladder-hang test now sources params from
`tiny_config()`.

**A test that was passing for the wrong reason, found and fixed.** After adding the
required Params field, that test still raised `ValidationError` — but for the
MISSING key, not for the field under test, so all three parametrized cases passed
without exercising anything. It now asserts the baseline is valid before corrupting
a field.

### AC-19-R7 — CONFIRMED, refreshed

Quoted 99.4 / 93.4 / 56.8 %; measured **0.9992 (500 Hz) / 0.9464 (250 Hz) / 0.5771
(125 Hz)**, exactly the row's figures. Refreshed in both
`spectrogram.py` and `spectrogram.yaml`, now **with the framing stated**
(48 kHz, n_fft 2048, hop 512, min_bins_per_band 1) — the previous figures named no
framing, which is why the drift was unattributable.

### RR-45 — CONFIRMED FIXED

The table reads `new (was old)` with "one band LOST" as the ordering cue. Closed.

---

## C5 — the dry-run scaffold  (AC-28 · AC-43 · RR-37)

`src/amcd/simulators/dry_run.py`, `tests/test_metrics.py`. **OFF-PATH.**

### AC-28 — fix CONFIRMED live; test added; ROW'S OWN ACCEPTANCE CRITERION REJECTED

The broadband direct arrival is in place and the axis is live: C50 through the ISO
path swings **+11.602 → +1.708 dB over a 16× distance range** (pre-AC-28: 1.91 dB at
every distance, identical to 3 s.f.).

**I did NOT use the row's prescribed test, and this is a finding (AC-102 below).**
AC-28 asks that "C50 must fall ~6 dB per doubling of d and cross 0 dB near d = r_c".
Measured at 10×8×3.5 m, α 0.2, r_c 1.193 m:

```
   d      d/r_c   C50 ISO   DRR closed-form   dC50 per doubling
   0.5 m   0.42   +11.602      +7.551
   1.0 m   0.84    +6.721      +1.530             -4.88 dB
   2.0 m   1.68    +3.495      -4.490             -3.23 dB
   4.0 m   3.35    +2.114     -10.511             -1.38 dB
   8.0 m   6.71    +1.708     -16.531             -0.41 dB
```

C50 never crosses 0 dB and its slope flattens. That is correct physics: C50's early
window is the first 50 ms, holding the direct arrival PLUS 50 ms of reverberant
tail, so C50 exceeds DRR everywhere and tends to a tail-only asymptote. 6 dB/doubling
and 0-at-r_c are DRR properties; the row conflated the two quantities, and its
criterion **would fail a correct implementation**. `test_the_placement_axis_moves_c50_through_the_iso_path`
asserts monotonicity and a swing > 5 × `d0b_c50_jnd_db` instead.

### AC-43 — in-lane half done, artifact half is not mine

EDT re-measured on the same sweep: **0.5514 / 0.7882 / 0.7989 / 0.7843 / 0.7849 s** —
non-monotone, ~1.5 % across the top four distances, against C50's 9.9 dB swing. The
split verdict (AC-28 made C50 live and left EDT dead) is stated at the `NOT MODELLED`
note with the measurement, and no early-reflection model was added — the row forbids
it. **The D0a/D0b artifact and E1 disclosure land in `diagnostics/probe.py` (lane S)
and `reporting/tables.py` (lane P)** — filed below. **AC-43 does not close.**

### RR-37 — CONFIRMED FIXED

No duplicated AC-28 narration, no three-blank-line run at the imports (checked
mechanically). Closed.

---

## C6 — config declarations  (AC-30 · AC-60)

`configs/base.yaml`, `configs/research_i.yaml`. **AC-30 is ON-PATH**
(`configs/*.yaml`); AC-60 likewise.

### AC-30 — corrected, but the OBVIOUS correction was itself refuted, and it DOES NOT CLOSE

The stale text ("~2.6 m at the largest", "1.0 m sits inside the band") is replaced.
**AC-30's own replacement number [0.41, 5.16] m was NOT written**: AC-50 (major,
OPEN) is titled to refute it — that span is the `mixed` regime only (α ≤ 0.80) while
`ceiling_absorptive` puts α ∈ [0.85, 0.98] on the same shoebox family, which is
exactly what `test_material_shift` selects. Written instead:

```
   Sabine  d_min in [0.412, 5.712] m
   Eyring  d_min in [0.417, 11.413] m   (max at 12x10x5 m, alpha 0.98)
```

tagged **"nominal α as configured; pending AC-54"** — lane B's ITEM 0, live this
cycle, has shown effective absorption is `1−sqrt(1−α)`, so even these describe a room
that is not rendered.

Also corrected: the algebra. `d_min = 2·sqrt(V/(cT))` reduces under Sabine to
`2·sqrt(αS/(c·K))` and is **volume-independent**, so "the largest room" was never a
corner of it.

**AC-30 DOES NOT CLOSE.** RD-111 names it explicitly (*"DO NOT DELETE RD-65, AC-30
or RR-32 at step 7"*) with its residual being exactly this base.yaml range, and it
sits in ledger cluster **C11**, marked "gated on C6", under close-together-or-not-at-all.
**Gate: AC-54 + AC-50 + cluster C11.** The realized per-split below-d_min disclosure
belongs to `scenes/generator.py` (lane S) — filed below.

### `configs/research_i.yaml` — same defect, second file, lane M owns it

Not in my assigned rows and not in the plan until the review caught it. Both
placement regimes declare `distance_range: [1.0, 10.0]`, and **AC-50's Eyring corner
of 11.413 m exceeds the entire range** — no separation this config can draw satisfies
the standard's criterion at that corner. Correcting base.yaml alone would have
installed a fresh base ↔ research_i divergence in the config that claims Research-I
faithfulness. Caveat carried across.

### AC-60 — FIXED

The 1.0 m floor's justification argued from `20·log10(r_c/d)`, textbook
inverse-square, which the pinned upstream backend does not implement
(`1/(4π(1+d²))`, −3.01 dB at exactly d = 1.0 m — the floor sits at the worst point).
The realized spreading law is now noted where the DRR reasoning is written, with the
per-distance error table and the statement that the bias is common-mode across legs
but NOT across placement regimes, so it perturbs `near_corner` vs `interior_random`
specifically. The floor is not moved — that is a research decision about the declared
population.

---

## C7 — `acoustics.py`  (S-1) — DELIBERATELY NOT DONE

**Referred to the integrator queue as a rule-4 spanning row.** S-1 asks for
`min_measurement_distance()` in `src/amcd/acoustics.py` (mine) while its ONLY caller
is `src/amcd/scenes/generator.py` (lane S). Shipping the helper without the caller
switch would make my owned file look clean while the residual sits elsewhere —
**RD-111's exact failure mode**, which that row raises as major against three rows
that already did this. Supporting: `acoustics.py` is off RD-33a (i)'s path list, S-1
is minor, and it sits in cluster C11 which the ledger marks "gated on C6" (lane B's
ITEM 0, live this cycle).

The AC-45 rounding finding travels with it rather than being half-applied:
`c·SABINE_K = 343 × 0.161 = 55.2230` vs `24·ln10 = 55.2620`, +0.0707 %, a 0.035 %
d_min bias. **Gate: cluster C11 / AC-54.**

---

## C8 — naming and traceability  (RR-85 · AC-70 · F-106)

`src/amcd/evaluation/room_acoustic.py`, `tests/test_metrics.py`. **ON-PATH.**

- **RR-85 — CONFIRMED NOT FIXED, now fixed.** The docstring documented a 2-tuple
  while returning four values and never named `band_accounting`. Returns is now a
  4-item list with units: triples (s / dB), nan_reasons (covering both unscored legs
  AND scored-but-caveated ones), `window` (keyed by the STRING `f"{fc:g}"`, valued
  `(sample index from that leg's own onset, source-leg name)`), and `band_accounting`
  with all six keys — noted as the load-bearing return, since it carries the caveat
  columns to `ci_table.csv`.
- **AC-70 — CONFIRMED NOT FIXED, now fixed.** The comment in the energy-domain
  `_c50` helper called `channel_band_avg_metrics` "the reported path", contradicting
  the module docstring, the code, and `evaluator.py`. One-line correction naming
  `compute_room_acoustic_metrics` and why.
- **F-106 — CONFIRMED NOT FIXED, now fixed.** AC-40's known-answer test was
  parametrized over the module constant `_ISO`; it now reads
  `configs/base.yaml`'s `iso_eval_freqs` at collection time.

---

## PASS CONDITION — result

Suite on this checkout, `PYTHONPATH` prefixed: **535 passed, 1 skipped** (before:
506 passed, 1 skipped; +29 tests, no failures, nothing removed).

Canonical dry run `experiments/all_20260811_190542`. Fixed-seed A/B against the
pre-registered baseline:

```
diff ci_table.baseline.csv experiments/all_20260811_190542/stats/ci_table.csv
  -> IDENTICAL, no row moved
```

**As pre-registered.** `git merge-base --is-ancestor v3-rebuild HEAD` confirms
`v3-rebuild` did not move during the session, so the evidence stands against the
tree it was measured on and the merge is a fast-forward.

**WHAT THIS DOES AND DOES NOT DEMONSTRATE.** It is a NON-INTERFERENCE control and
nothing more. F-M5 already measured that this A/B has near-zero power over exactly
this code, and my own changes make that worse, not better: the headroom guard fires
on no scene in the canonical dry run at either threshold, so 50.0 → 52.0 is
unexercised by it; the F-M9 append is unreachable inside the declared support; and
the C1 fold change is inert above one guard width, which every scene here exceeds.
**The unit-level known-answer tests and their negative controls are this lane's real
evidence. The green A/B is not.**

---

## PLAN REVIEW — what it changed before implementation

`research-director` ran on the PLAN, before any edit (CLAUDE.md: it runs on a plan).
It raised eleven items. **Ten were accepted and folded into the work rather than
filed as OPEN rows**, because they were corrections to the plan and are now
discharged; they are recorded here so the reasoning is recoverable:

- **AC-30 was about to be written wrong.** The plan said to write AC-30's own
  [0.41, 5.16] m. AC-50 refutes exactly that number, and AC-54 (lane B, live this
  cycle) makes even AC-50's figures nominal-α. Caught before the edit. This was the
  single highest-value catch of the session.
- **`configs/research_i.yaml` was missing from the plan entirely** — lane M owns it
  and it carries the same defect, where the Eyring corner exceeds the declared range.
- **S-1 was going to ship as an orphan helper**, which is RD-111's named failure
  mode. Referred instead.
- **AC-68's headline half** (the module's own "standard ISO-3382 path" claim) was
  going to be dropped, leaving a conformance gap dressed as a passing test.
- **The coverage test was specified as containment**, which cannot prevent the
  over-coverage defect it exists to prevent; and a `[lo, hi]` range mis-types a list.
  Both changed to equality and to a band set.
- **Negative controls were specified for only 3 of ~8 new tests.** Now all of them.
- **The re-calibration had no declared population**, risking a tuned threshold
  selected on test scenes. Now `valid` only, with split, seed and config named.
- **AC-26's inability to close** and the **ON-PATH / OFF-PATH split** were not stated;
  both are now in the gate accounting above.
- **Per-minor GATES** (RD-128) rather than bare reasons.

**One item was REFUTED and the refutation is recorded.** The review claimed AC-37's
prescribed remedy-independent oracle gain-sweep test was absent from the lane. It
exists at `tests/test_metrics.py` —
`test_the_decoded_oracle_reproduces_its_targets_t30_at_any_level` at gains {0, −20}
dB and `test_a_scene_that_would_breach_the_t30_jnd_is_refused_by_encode` at
{−30, −40} dB. Its existence is what turned the re-calibration into a sharp pass/fail
rather than a judgement call, so the correction improved the plan anyway.

---

## NEW FINDINGS — ids from lane M's block only (rule 6)

| ID | agent | severity | status | anchor | finding | resolution |
|---|---|---|---|---|---|---|
| AC-100 | builder (lane M) | minor | OPEN | `src/amcd/evaluation/room_acoustic.py` `_band_energy` fold | The F-68-R3 fix trades an energy LOSS for a PLACEMENT approximation, and the trade is undisclosed outside the source comment. Folding with the mirror index CLAMPED conserves energy exactly at every record length, but where the mirror would land outside the record the energy is deposited on the boundary sample instead. For records shorter than one guard width (< 96 ms at 500 Hz) that concentrates up to 29.7 % of band energy at the record edge, which lands in C50's early window. It is strictly better than the previous silent discard, but it is not free and no `(unit, reason)` records it. | Either disclose the clamped fraction as a per-band caveat when it is non-zero, or raise/NaN with a reason for records below one guard width. Needs a decision on which, since the second changes admission. |
| AC-101 | builder (lane M) | minor | OPEN | `configs/base.yaml` `metric_band_resolvability_margin` | The margin's stated calibration does not reproduce. Documented +12.8 / +4.9 / +2.8 % at margins 1.0 / 2.0 / 3.0 (500 Hz); re-measured +5.91 / +3.63 / +2.97 %, so the JND crossing lands between margin 1.0 and 1.5, not at 2.0, and at 1000 Hz the bias is under the JND at every margin including 0. Cause is AC-65's drift — the old figures were measured against floors the AC-36/F-67 fold then moved. 2.0 is kept and the justification rewritten as weaker (under AC-38 the floor discloses, so a too-wide margin biases nothing), but the original protocol named no config, seed or population and could not be reproduced exactly, so this is "does not reproduce", not a refutation. | Re-derive the margin against the current floors with a stated protocol, or accept the weaker AC-38-based justification now in the config and close. |
| AC-102 | builder (lane M) | minor | OPEN | `docs/review_ledger.md` AC-28's resolution text | AC-28's prescribed acceptance test is not valid for the quantity it names, and would fail a correct implementation. It asks that "C50 through the ISO path must fall ~6 dB per doubling of d and cross 0 dB near d = r_c". Those are DRR properties. C50's early window is the first 50 ms, holding the direct arrival PLUS 50 ms of reverberant tail, so C50 exceeds DRR everywhere and tends to a tail-only asymptote. MEASURED (10x8x3.5 m, alpha 0.2, r_c 1.193 m): C50 +11.602 / +6.721 / +3.495 / +2.114 / +1.708 dB at d = 0.5/1/2/4/8 m — slope -4.88, -3.23, -1.38, -0.41 dB per doubling, never crossing 0, while the closed-form DRR runs +7.551 to -16.531 dB. | Reword AC-28's remedy to assert monotonicity and a swing >> `d0b_c50_jnd_db` (what `test_the_placement_axis_moves_c50_through_the_iso_path` now does), or move the 6 dB/doubling + 0-at-r_c criterion onto a DRR probe, which is the quantity that has it. |
| RD-186 | builder (lane M) | minor | OPEN | `configs/base.yaml` + `src/amcd/config.py` | AC-68's remedy asks for a "config-declared rejection" threshold, which cannot be done in one lane: `configs/base.yaml` is lane M's but its schema is `src/amcd/config.py` (lane P) with `extra: forbid`, so a new key fails validation. Shipped instead as `_DECLARED_STOPBAND_DB` in `tests/test_metrics.py`, argued as a declared property of the filter design rather than an experiment-governing value. Rule-4 spanning row; also the config-contention case `docs/parallel_protocol.md` documents. | Integrator: add the schema field and move the mapping (octaves-out -> min rejection dB) into base.yaml, or ratify the test-pinned form and close AC-68's config clause. |
| RD-187 | builder (lane M) | minor | OPEN | `src/amcd/representations/base.py:63` (`build_representation`) | `min_db_headroom_octave_centres_hz` is a SECOND declaration of the evaluation band set (`iso_eval_freqs` is the first) — the AC-24 divergence shape, admissible only while `test_the_headroom_guard_reads_exactly_the_reported_metric_bands` forbids the drift. It exists solely because `build_representation` takes `sample_rate` as its only cross-cutting argument, so a representation cannot read the master config; threading `iso_eval_freqs` through would touch `data/preprocess.py` and `training/infer.py` (lane P) and `diagnostics/probe.py` (lane S). | Integrator/next cycle: give `build_representation` the evaluation band set as a second cross-cutting argument and derive the guard's operand, retiring the duplicate declaration. |

## FINDINGS FOR OTHER LANES — anchored, not actionable here

| ID | anchor (owning lane) | finding |
|---|---|---|
| RD-188 | `src/amcd/data/preprocess.py` (lane P) | F-M3's other half. A minimum ACROSS FREQUENCY is a spectral-flatness constraint; restricting the guard's operand removes the false rejection but does not DISCLOSE the slope. The honest treatment is a per-scene spectral-slope disclosure in `preprocessed/meta.json`, which `data/preprocess.py` writes. Without it a strongly sloped render now passes silently where it previously failed loudly. |
| RD-189 | `src/amcd/scenes/generator.py` (lane S) | AC-30's disclosure half: the realized below-d_min fraction per split (25.4 % of id, 42.5 % of `test_material_shift` by Sabine; 37.2 % / 92.5 % by Eyring) belongs in `placement_report.json` beside the existing `d_over_rc` summary, carrying the same "nominal α; pending AC-54" caveat (RD-131). |
| RD-190 | `src/amcd/diagnostics/probe.py:256,262` (lane S); `src/amcd/reporting/tables.py` (lane P) | AC-43's artifact half: the D0a/D0b artifacts and the E1 report must state that the placement axis exercises C50 but NOT EDT under the scaffold (EDT non-monotone, ~1.5 % spread over a 16x distance range against C50's 9.9 dB). Same anchors carry RD-96's D0b resolvability-policy divergence. |
| RD-191 | `docs/lanes/cycle5-M.md:21`; `docs/lanes/cycle5.yaml` (integrator) | The partition's premise "only the metric lane may legitimately move `ci_table.csv`" is false by construction: lane P owns `stats/**` and `reporting/**` — the code that WRITES the table — and the serial queue's F-M2 requires ADDING a column to it; lane S owns `scenes/generator.py`, whose placement/admission sampling moves every fixed-seed downstream number. It held empirically in cycle 4 but is a property of those lanes' declarations, not of the partition, so lane M's non-interference control depends on three other lanes declaring and honouring "none". |

---

## GATE ACCOUNTING (RD-128)

**Lane M carried zero blockers and ten majors**: AC-26, AC-28, AC-37, AC-36, AC-65,
F-M3, F-68-R2, AC-37-R4, RR-85, AC-69. The other 22 rows are minor. All ten are
acted on.

**Only six of the ten are on condition (i)'s path list** (`src/amcd/scenes/**`,
`src/amcd/evaluation/**`, `config.py` split handling, `configs/*.yaml` split
declarations):

| ON-PATH — retiring these moves the gate | OFF-PATH — contributes zero to (i) |
|---|---|
| AC-26, AC-36, AC-65, F-68-R2, RR-85 (`evaluation/room_acoustic.py`); AC-37 (anchor names `evaluation/evaluator.py`) | AC-37-R4, F-M3, AC-69 (`representations/**`); AC-28 (`simulators/dry_run.py`) |

This is **not** a reason to discount C4 — AC-37 is the Research II decode floor and
is the most consequential work here — but the integrator cannot recompute (i)
without the split being stated.

**Clearing lane M's six is necessary, not sufficient:** `evaluation/**` also holds
AC-64 (blocker), F-89 and F-M2 (majors), all on the serial queue.

**AC-26 is an ON-PATH major that CANNOT close this cycle.** Its own resolution
chains it to AC-38, whose reported-column half is lane P's. So RD-33a condition (i)
cannot lift for `evaluation/**` this cycle regardless of lane M's execution. Stated
now rather than discovered at the merge.

### Rows NOT closing, each with a GATE (RD-128 requires a gate, not just a reason)

| row | severity | why it stays open | GATE |
|---|---|---|---|
| AC-26 | major | chained to AC-38 by its own resolution | AC-38 closes |
| AC-38 | minor | reported-column half is `stats/aggregate.py` + `reporting/tables.py` | lane P lands F-70 then the column (F-M2) |
| AC-30 | minor | RD-111 forbids deletion; range is provisional | AC-54 resolves, then AC-50, then cluster C11 together |
| AC-43 | minor | artifact-side disclosure is lanes S and P | RD-190 lands |
| S-1 | minor | spanning row, deliberately not started | cluster C11 / AC-54 |
| AC-68 | minor | config-declaration clause needs `config.py` | RD-186 |
| AC-37 | major | remedy (c), the D0b level sweep, needs `diagnostics/probe.py` | integrator queue (lane S) |

---

## SUMMARY

**32 assigned rows, all acted on; none untouched and unremarked.**

- **19 CONFIRMED FIXED or FIXED this session** and closable by the integrator once
  re-derived on the merged tree: AC-36, F-68, F-68-R2, F-68-R3, RR-38, AC-65,
  AC-39, AC-26-R6, F-M9, RD-93, RD-96, RD-98, RD-99, AC-37-R4, F-M3, F-M10, AC-69,
  AC-19-R7, RR-45, RR-37, RR-85, AC-70, F-106 — with the caveat that four of these
  (AC-37-R4, F-M3, F-M10, AC-69) are one cluster that must close together with the
  re-calibration.
- **7 do NOT close**, each with a gate in the table above.
- **1 deliberately not started** (S-1), referred as a spanning row.
- **6 new findings** raised from lane M's block (AC-100, AC-101, AC-102, RD-186,
  RD-187) plus 4 anchored for other lanes (RD-188..RD-191).
- **2 corrections to existing rows**, recorded rather than quietly worked around:
  AC-37-R4's "always the 24.8 Hz band" is too strong (measured: 99.2 Hz), and
  AC-28's prescribed acceptance test is invalid for C50 (AC-102).

**This is a SELF-CHECK on `lane/M-cycle5`, NOT a clean pass.** The reviewer pass
that counts is the integrator's over the merged tree.
