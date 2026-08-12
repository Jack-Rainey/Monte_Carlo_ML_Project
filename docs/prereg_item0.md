# Pre-registration — ITEM 0 (cluster C6), ITEM 0b (C5) and AC-175

Committed ALONE, before any code change, per RD-192/F-138. The point of a
pre-registration is that git can evidence it preceded the edits; lane M's said so
while one commit held the declaration, eight changed files and the results, and
that is the failure this file exists not to repeat.

Render grant: **6 scenes, 0 spent**, extended by the user 2026-08-12 (RD-267).

---

## What is already established without a render

Both re-derived from pinned upstream at `608ea30`, independently, by the
acoustics reviewer:

- **AC-54** — reflectivity is `sqrt(1-α)` as an AMPLITUDE coefficient
  (`SoundMesh.cpp:221`), accumulated into a variable named `energy`
  (`gsSoundPropagator.cpp:1364,1531`), with the synthesizer taking `sqrt()` of it
  (`binding.cpp:417,454`). The `1/d` pressure law coming out correct is what
  forces the reading. So **α_eff = 1 − √(1−α)** exactly.
- **AC-66** — `gsSoundMedium.cpp:226-228` halves the ISO 9613-1 pressure dB/m,
  then `getAttenuation` converts with `dbToLinear` = `10^(dB/20)` against an
  ENERGY accumulator. Realized air absorption is **α_ISO/4**.
- **AC-175** — `prop_request.maxIRLength = 3.0` is compiled at
  `Context.cpp:33`, and `grep` over `module.cpp` returns NOTHING for it: the
  Python module exposes specular/diffuse counts and depth, threads, sample_rate,
  channel_type and normalize, and **not** the IR length. Confirmed at the pinned
  sha, above.

## The one question a known answer cannot settle

**Is `native_ir_samples` set by the room's decay, or by upstream's IR-length
machinery?** `IR_THRESHOLD` and `ADAPTIVE_IR_LENGTH` are both ON
(`Context.cpp:19,21`), so the native length is chosen adaptively and then capped
at 3.0 s. The single retained observation to date is 44647 samples (0.930 s) in a
room measuring T30 0.5441 s — about 1.6 × T30, consistent with an adaptive trim.

If that ratio holds, AC-175 is a **gate bug**: records are decay-proportional and
only the longest rooms hit the cap. If instead `native_ir_samples` is roughly
CONSTANT across a large T60 swing, every long-T60 record in E1 is bounded by a
constant nobody declared, and AC-175 is a **dataset-wide invalidation**.

### Decision rule, fixed before the render

Three scenes, spanning the declared T60 range as widely as `base.yaml` admits,
artifacts **RETAINED to a real run dir** (never a `TemporaryDirectory` — that is
what destroyed AC-64's evidence). Recorded per scene and band:
`native_ir_samples`, the last non-zero sample, band-energy peak, `noise_power`,
the truncation index, and the fitted T30.

Let `r = native_ir_samples / T30_fitted`.

- **r roughly constant (spread < 2×) across the T60 sweep** → records are
  decay-proportional. AC-175 is a gate bug: the gate's denominator must become
  the backend's realized support. **No E1 invalidation.**
- **native_ir_samples roughly constant (spread < 2×) while T60 varies > 4×** →
  the support is set by upstream machinery, not the room. **AC-175 escalates:
  every T30/EDT above the implied T60 is bounded by an undeclared constant, and
  the E1 dataset render cannot proceed on the current backend configuration.**
- Anything between → report the measured `r` per scene and take neither branch.

**This rule is fixed now and will not be revised after seeing the numbers.**

## Render budget, and the order, which is binding

| # | claimant | scenes | why this order |
|---|---|---|---|
| 1 | **AC-175** | 3 | Decides gate-bug vs dataset-wide invalidation. Runs FIRST because the answer changes whether ITEM 0's remedy is sufficient at all. |
| 2 | ITEM 0 / AC-54 | 2 | α_eff confirmation at two α, AFTER the fix lands, so it measures the fix. |
| 3 | ITEM 0b / AC-64 | 1 | Truncation index re-measured with the artifact retained. |

RD-17's own convergence probe is **not** funded by this grant and does not run
this session; condition (ii) stays unlifted (RD-263).

Renders use the LOW ray budget unless a scene needs the high leg, because none of
these three questions is about convergence.

## ITEM 0's remedy, decided before implementation

Branch already chosen by RD-144 and now recorded in AC-54's own cell: the
absorption convention is a **declared property of the BACKEND** in
`configs/simulators/gsound_sir.yaml`, applied at that backend's own `createbox`
call site. `scenes/generator.py` stays in nominal α and remains backend-agnostic,
because the alternative writes one raytracer's energy/amplitude confusion into
the scene population and forecloses the roadmap's multiple-raytracers item.

Four changes, as one:

1. `absorption_convention` declared in the backend config, applied at `createbox`.
2. The record-length gate's denominator becomes the backend's **realized**
   support, declared in the backend config and falsified per render against
   `native_ir_samples` — the same shape as `speed_of_sound_m_s` (AC-175, F-186).
3. `diffuse_depth` declared as the TIME bound it physically is (AC-55).
4. Air absorption's α_ISO/4 declared with its measured per-band consequence
   (AC-66).

## Expected effect on `ci_table.csv`

**MOVEMENT EXPECTED, and that is the point.** ITEM 0 changes the absorption the
backend realizes, so every closed form in `scenes/generator.py` and every
record-length decision downstream of it can move.

The canonical dry run uses `DryRunSimulator` and does not touch the gsound
backend, so the **canonical A/B should remain byte-identical** — and if it moves,
that is a finding, because it would mean a backend-scoped change reached the
scaffold path. Declared here so the step-4 detector can discriminate.

The gsound path has no A/B baseline to compare against, because it has never
produced a dataset. That is stated rather than papered over.
