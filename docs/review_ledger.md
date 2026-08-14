# Review ledger

**Unresolved findings only.** One row per finding:
`ID | agent | severity | status | anchor | finding | resolution`.

Status is **OPEN** or **DEFERRED** (out of scope for the current gate, with a
reason and the gate it belongs to). There is no resolved state: a finding that is
fixed is DELETED.

This file exists so unsolved work survives a lost context window. **That is its
only job.** It is not a record of process: no cycle narration, no pass counts, no
rows about the ledger itself or the machinery around it. A row that changes
neither what the code does nor what a reported number means does not belong here.

You are not allowed to add anything to this file without the user's permission. You may solve issues and remove things from this file, but adding new rows requires user approval to prevent documentative bloat. That way, true research limitations are documented, rather than being passed here only for the next session to deal with and be unable to solve.

## OPEN

None.

## DEFERRED

Two kinds. **Report-prose obligations** are sentences the E1 or E2 write-up must
contain; nothing can be implemented for them before the report exists, and
together they are that report's disclosure list. **Implementation, gated** is work
whose prerequisite does not exist yet.

### Report-prose obligations (E1 / E2)

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| AC-16 | acoustics-reviewer | minor | DEFERRED (gate: E2 write-up) | src/amcd/representations/spectrogram.py vs configs/simulators/gsound_sir.yaml | The representation resolves 27 third-octave bands over a simulation with only 8 octave bands of spectral freedom: exactly 3 third-octaves per simulated band except the top band, which backs 6, and three bands sit inside the DC-88.7 Hz band. ~22 % of dimensions carry no independent simulated content. | Record-only obligation: ~22 % of representation dimensions carry no independent simulated content. Nothing to implement; it is a disclosure the E2 write-up must carry.  |
| RD-46 | research-director | major | DEFERRED (gate: E1 report) | configs/research_i.yaml `ceiling_absorptive` | The split's absolutes are diagnostic-only. | The user's 2026-08-10 decision was to suppress absolutes below `metric_min_measurable_t60_s`; that threshold is gone, because at this split's Eyring median T60 of 0.059 s it suppressed the MAJORITY of EDT realizations and biased the survivor mean +31.2 %. Censoring an estimator on its own value is a distortion, not a disclosure. The filter-derived floor (~0.010-0.020 s) does not bite, so the numbers ARE now reported — and labelled instead, via the per-split `n_estimator_variance_limited` count. **The change is deliberate and is written into `configs/research_i.yaml` deviation 2.** What remains is E1 REPORT PROSE stating the split is high-variance diagnostic, never physical. |
| RD-27 | research-director | major | DEFERRED (gate: E1 report) | configs/research_i.yaml (deviation: `test_geometry_shift`) | RI's `test_geometry_shift` is DUAL-axis (corridor AND near_wall+far_pair), which violates invariant #10. Reproducing it verbatim imports RI's confound; deviating makes v3's split NOT comparable to RI's 0.61 baseline rel-L2. | ARBITRATED: keep invariant #10. The config-side disclosure is written into `configs/research_i.yaml`; what remains is E1 REPORT PROSE — state the non-comparability per split and relabel the split "v3-corrected single-axis geometry shift". |
| RD-28 | research-director | major | DEFERRED (gate: per-surface materials / E1 report) | configs/research_i.yaml (deviation: `test_material_shift`); src/amcd/simulators/base.py (`SceneSpec.material_absorption`) | RI's material shift is "ceiling_absorptive AND asymmetric_walls"; `asymmetric_walls` needs per-surface absorption, which a scalar `material_absorption` and `createbox(absorp)` cannot express. **Sign corrected:** the scalar α applies to ALL SIX SURFACES, giving Sabine T60 0.117/0.161/0.209 s vs an area-weighted ceiling-only reading's 0.42/0.51/0.65 s — 3-4× STRONGER, not weaker. | Implementation deferred to a per-surface-materials gate; the disclosure is written into `configs/research_i.yaml` and remains an E1 report obligation. **CONTINGENCY:** gsound's `createbox(absorp)` takes scalar-or-per-band and NOT per-surface (confirmed, `docs/gsound_sir_setup.md` §4), so if per-surface proves unreachable this decision reopens — itself an argument for an eventual custom simulator. |
| RD-29 | research-director | major | DEFERRED (gate: E1 report) | configs/research_i.yaml (placement regimes); src/amcd/scenes/generator.py (placement_report.json) | RI never gives `mid_pair`/`far_pair` source-receiver distance sub-ranges numerically. Substituting the global 1.0-10.0 m range changes the distance distribution of five of six splits, and distance sets DRR → C50/D50/EDT. | Do not invent sub-ranges. The quantification exists and is verified: `placement_report.json` records volume, Sabine AND Eyring T60, r_c, d/r_c and DRR per split, with train median DRR −5.75 dB against test_material_shift +4.86 dB at near-identical median distance. What remains is the E1 report stating the realized distribution and exactly what was substituted for RI's unstated sub-ranges. |

### Implementation, gated

| ID | agent | sev | status | anchor | finding | resolution |
|----|-------|-----|--------|--------|---------|------------|
| F-06 | falsifier | minor | DEFERRED (gate: E2) | src/amcd/representations/spectrogram.py (loss); src/amcd/training/loss.py | Dual loss source of truth: `rep.loss(pred,target,delta)` takes a RAW-dB δ; a future caller wiring it against z-scored operands would silently reintroduce F-01. | Belongs to the E2 loss-architecture work (unify `build_criterion` / `rep.loss`). Mitigation in place: docstring states δ must be operand-domain; the sole caller already scales it. |
| RD-12 | research-director | minor | DEFERRED (gate: E4) | src/amcd/reporting/tables.py (ray-budget axis label) | The swept variable is the DIFFUSE ray budget with specular fixed; "reduce ray count" could be read as a total. | Only the E4 report axis label remains, and there is no axis to label until E4. |
| RD-23 | research-director | minor | DEFERRED (gate: E4) | src/amcd/simulators/base.py, src/amcd/simulators/render.py (`realization_index`) | pygsound exposes no RNG seed, so ONE render per (scene, budget) conflates Monte-Carlo realization variance with the budget effect — and MC variance at low budgets IS the phenomenon under study. | E4 needs ≥N realizations per (scene, budget) or an explicit argument that between-scene variance dominates. The layout half is discharged: `realization_index` is a required key of both the path descriptor (`simulators/base.py`) and `renders/<id>/meta.json`, so identity is in metadata rather than filenames and adding realizations is a value change, not a migration. |
| AC-19-value | acoustics-reviewer / research-director | minor | DEFERRED (gate: E2) | configs/representations/spectrogram.yaml (`min_bins_per_band` VALUE) | The MECHANISM is closed — the ladder is config-declared, the band description is recorded, and the in-band fractions are measured. The VALUE is shipped as 1 — today's behaviour — because raising it is a research decision with real cost: at production framing 3 bins drops every band below ~315 Hz, i.e. all low-frequency coverage. Measured basis: the five lowest bands hold ONE FFT bin each and a 63 Hz tone peaks in the WRONG band (57.7 % in the 78.7 Hz band vs 35.6 % in 49.6 Hz, per `configs/representations/spectrogram.yaml`). | Blast radius is bounded: reported ISO-3382 metrics are 500/1000 Hz octave-band quantities computed from DECODED WAVEFORMS, not from these bands, so this cannot corrupt an E1 headline number — it affects the learned representation and low-frequency reconstruction. The roadmap fix is **multi-resolution sampling** (paper §6); merge-or-reframe the low bands is preferred, dropping them is the fallback. Decide at E2 with `preprocessed/meta.json`'s recorded in-band fractions in hand. |
