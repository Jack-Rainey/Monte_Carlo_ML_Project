# Design Spec — Acoustic Monte Carlo Denoising (clean rebuild)

Companion files: `CLAUDE.md` (agent operating context), `.claude/agents/falsifier.md`
(adversarial auditor). This document is the build plan; names are proposals.

---

## 1. Purpose & status

Rebuild the supervised low-ray → high-ray ambisonic IR denoising pipeline from
scratch. Priority order: **(1) a verifiably correct pipeline**, **(2) reproduce
the prior negative result** as a faithfulness check, **(3) test hypotheses.**
The prior pipeline produced an honest null (no variant beat the low-ray baseline
on signal metrics); we cannot tell a real null from a silent bug until the
harness is trustworthy.

## 2. Research framing

- **Question:** to what extent can ML reduce the ray count for Monte Carlo
  geometric-acoustic IR simulation while preserving objective acoustic metrics?
- **Success criterion (unchanged, re-domained):** a denoised output succeeds on
  a scene when it is closer to the high-ray reference than the low-ray input is —
  now measured in the **energy / acoustic-metric domain**, not waveform L2.
- **Hypotheses:** (H1) wrong output domain — waveform-sample regression to an
  incoherent target is ill-posed → identity collapse; (H2) Huber δ inert in raw
  amplitude space; (H3) optimization/data scale (batch=1, ~500 scenes); (H4)
  drop full-prediction, keep residual framing.

## 3. Representation & loss (finalized: D1–D4)

The reframe: **denoise where the target is a convergent expectation (energy),
not an incoherent realization (pressure).** Energy is both learnable and the
basis of every reported metric.

- **D1 — Target:** predict the **per-channel log power-spectrogram** (short-time
  band-energy envelope) of the high-ray IR (the **learning target**). Do not
  predict EDR directly (preserves early-reflection localization). Energy→metric
  derivation survives only as a fast training proxy and the loss term below —
  **not** as the reported metric source (see "Metric source of truth").
- **D2 — Bands:** **third-octave** (~30 bands, ISO-3382-aligned). Octave and mel
  are registered alternatives behind the representation seam.
- **D3 — Reconstruction (decode, REQUIRED):** decode the predicted envelope into
  a waveform by imposing it on the **low-ray IR's own fine structure, rescaled
  per band/frame**; shaped/velvet-noise carriers for the diffuse tail are a later
  option. Decode feeds **both** the objective metrics and the listening track —
  it is *not* listening-only. Reported objective metrics are computed from this
  decoded waveform via the **standard ISO-3382 path** (IIR octave bands,
  Schroeder backward integration, Lundeby tail truncation).
- **D4 — Channels:** per-channel envelopes for v1 (preserves W for omni metrics);
  an inter-channel directional-energy-ratio term is **stubbed** for spatial
  metrics.
- **Output parameterization:** residual — predict the **correction** to the
  low-ray envelope. Full-prediction is dropped (H4).
- **Loss:** L1/Huber on **log-band-energy** (δ is now O(1)-meaningful → resolves
  H2 as a side effect), tail-weighted; **plus** a differentiable
  metric-consistency term (T30/C50/EDT computed from the predicted EDR vs
  reference — an energy-domain **proxy** for the reported metrics, validated once
  against the standard path) so training optimizes toward what is reported;
  **plus** the stubbed spatial term.

**Metric source of truth (resolves the D1/D3 ambiguity).** Reported objective
metrics come from exactly one place: the **decoded waveform**, via the standard
ISO-3382 metric function. The energy domain is the model's *target* and a
training *proxy/loss* only. Headline T30/C50/EDT are **never** computed directly
from the STFT energy grid — that is non-standard and (per the falsifier)
near-circular against an FFT-filtered reference. This rule governs §3, §4 (D0b),
and the eval stage; D1 and D3 are subordinate to it.

**Reported ISO absolutes are paired-comparison quantities, not literature values
(AC-17/RD-44).** Lundeby truncation is noise-floor dependent, and in this study the
noise floor IS the independent variable (ray budget). Truncating each leg at its own
index therefore manufactures a metric difference with no acoustic cause — measured
at a −50 dB floor with identical decay and only the floor scaled by √40 (the
5,000:200,000 ray ratio), the noisier leg read T30 12–16 % short, rising to ~51 %
at −30 dB, against a declared T30 JND of 0.05. All legs of a comparison are
therefore integrated over **one shared Schroeder window per (scene, band)**,
derived from the **physical legs only** (`low`/`high`) and *applied* to `pred`: a
model output must never set the window used to measure its own ground truth
(RD-43).

Two consequences that must be stated wherever these numbers are reported:

1. The shared window is set by the noisier physical leg, so a reported **absolute**
   T30/EDT/C50 is a function of the low-ray budget as well as the room. Absolutes
   are comparable **within** a paired comparison, not against Research I or the
   literature. The window and the leg that set it are recorded per scene and band
   in `metrics/iso_integration_windows.json`.
2. An E4-style "metric vs ray count" curve is therefore confounded by its own
   estimator unless the window is made budget-independent. Closing that needs
   Lundeby extrapolated-tail compensation (ISO 3382-1 Annex), which is deferred to
   the E4 gate.

A decay too short for a band to resolve is reported as **unscored with a reason**,
never as a small number. The floor is the octave filter's **own** decay in that
band times `metric_band_resolvability_margin` — measured per (metric, band), and
independent of the value being tested. Thresholding on the fitted value instead
censors the estimator's low tail and biases the surviving mean (+31.2 % measured
at true T60 = 0.06 s); AC-26/AC-27. EDT below `metric_edt_variance_limited_s` is
variance-limited rather than filter-limited and is **disclosed and counted per
split**, never suppressed.

## 4. Pre-training diagnostics (run BEFORE any training)

Two cheap probes that can falsify the premise for ~zero compute:

1. **Headroom probe.** Low-ray vs high-ray distance in the energy domain across
   the dataset. Large gap → signal to learn. Tiny gap → 5k rays already
   converged band energy; denoising here cannot help, and the bottleneck is
   elsewhere (phase / early-reflection localization / directional detail). Either
   result is publishable.
2. **Oracle upper bound (carrier ceiling).** Decode (D3) the *true* high-ray
   envelope onto the low-ray carrier and measure residual error with the
   **standard ISO-3382 metric function** — the same one the eval stage uses. This
   is the ceiling of the decomposition: the best any model could do, limited by
   the carrier. If the oracle misses the metrics, the carrier — especially its
   sparse early reflections (C50 / D50 / EDT) — is the bottleneck, not the model,
   so stop before training. This **requires `decode()`** and is a carrier test,
   not an energy-path self-check.

## 5. Architecture (seams now, v1 only)

One CLI over a DAG of cached, independently-runnable stages, each with a
`dry_run` path. Plugin seams = the future-work axes (§ paper 6).

```
gen-scenes → render → preprocess → diagnostics → train → infer → eval → stats → report   (+ all)
```

The `diagnostics` stage runs the §4 pre-training probes (D0a headroom, D0b oracle
ceiling) after preprocess and before train — either result is publishable and D0b
can gate a stop before training.

Seams (registries): `simulators/`, `representations/`, `models/`, `metrics/`.
Adding a raytracer / output domain / architecture / metric = drop a file +
register, no pipeline surgery.

## 6. Stage I/O contracts & artifact schemas

Each stage consumes/produces declared artifacts; nothing passes via globals.

| Stage | Input | Output (artifact) |
|---|---|---|
| gen-scenes | scene config | scene specs (JSON), deterministic from seed |
| render | scene spec + simulator backend **(x86)** | paired IRs `(C,T) float32` low+high, retained-path file, QC record |
| preprocess | raw IRs | energy tensors (per-channel log third-octave), **train-only** norm stats, split assignment, carrier refs |
| train | normalized energy tensors | checkpoint (+ resolved config stamp); valid-based selection |
| infer | checkpoint + low-ray inputs | predicted energy envelopes **+ decoded IR (required, D3)** |
| eval | decoded IRs + high-ray refs | **per-scene** metrics table (parquet), computed via the **standard ISO-3382 path** on the decoded waveform, + drop log (`drops.csv`: every NaN/partial metric leg as `(scene, split, metric, leg, reason)` — no silent exclusion) |
| stats | per-scene tables | mean ± CI, MDES, failure-case groups, sweep comparisons |
| report | stats | plots/tables + supplementary export |

Key schemas:
- **scene spec:** geometry family + dims, materials, source/receiver pose, seed,
  sim params. Hashable (leakage checks depend on it).
- **IR array:** channel-first `(C, T)`, float32; 3rd-order → C=16; T = sr·dur.
- **energy tensor:** `(channels, bands, frames)`; bands=third-octave; framing
  params recorded in the stamp.
- **per-scene metric row:** `scene_id, split, metric, kind, low_val, pred_val,
  high_ref, baseline_rel_ratio, improved` — one row per (scene, metric), where the
  three values are that metric's low-ray / predicted / high-ray-reference numbers.
  `kind` is the metric's **declared improvement kind**, set at the metric's
  definition site (required, no default — the eval/stats spine never assumes one):
  `match_reference` (improved ⟺ `|pred_val − high_ref| < |low_val − high_ref|`,
  consumes all three legs), `maximize` (improved ⟺ `pred_val > low_val`; the
  reference leg is structurally absent — e.g. energy SNR, whose
  reference-vs-itself value is +∞), or `minimize` (mirror of maximize).
  `maximize`/`minimize` are the seam for the roadmap's perceptual /
  spatial-error metrics (research_I_paper §6). `improved` is **nullable**:
  `None` when any consumed leg is NaN, so `stats` never misattributes an
  improvement; every such NaN leg is logged to `drops.csv` with a reason.
  `baseline_rel_ratio` = the two reference-errors' ratio (> 1 ⟺ improved;
  match_reference only, NaN otherwise). Per-scene rows are never collapsed
  before `stats`; `split` is always retained.

### 6.1 Evaluation splits & distribution-shift design

Six splits, **declared in config** (geometry family, placement regime, material
regime, seed, count) so new shift axes drop in without code changes. They exist
to separate ordinary held-out generalization from robustness under *structured*
distribution shift — each shift split isolates exactly one axis. The counts and
regimes below are the **Research I reference instantiation**, shown to explain
prior results; all of them (per-split counts, total dataset size, and the
placement / material / geometry regimes) are config-declared and expected to
grow or change in Research II — nothing here is a fixed requirement, and several
(dataset size, ray budget) may become swept axes:

| Split | n (R1) | Geometry | Placement | Material | Tests |
|---|---|---|---|---|---|
| train | 500 | shoebox | interior_random+mid_pair | mixed primary | weights |
| valid | 60 | shoebox | interior_random+mid_pair | mixed primary | selection only |
| test_id | 60 | shoebox | interior_random+mid_pair | mixed primary | in-distribution generalization |
| test_material_shift | 40 | shoebox | interior_random+mid_pair | → ceiling-absorptive + asymmetric walls | material robustness |
| test_placement_shift | 30 | shoebox | near_corner+mid_pair | mixed primary | placement robustness |
| test_geometry_shift | 30 | corridor | near_wall+far_pair | mixed primary | geometry robustness |

> The `n (R1)` column is the Research I size, shown only to explain prior results
> (e.g. why geometry-shift's baseline differed). The **durable** content is which
> single axis each shift isolates — not the counts or regimes. Do not treat any
> count or regime as fixed; dataset size in particular grows in Research II.

Rules:
- **Every test split is evaluated independently** — per-split metrics, per-split
  baseline-relative success, per-split CIs/MDES. The four are **never pooled**
  into one "test" number; the per-shift breakdown *is* the robustness result.
- **Controlled-shift integrity:** a shift split differs from `test_id` only along
  its named axis. Uncontrolled co-variation defeats the design — `falsifier` and
  `research-director` both watch for it.
- **Per-split headroom differs** (e.g. `test_geometry_shift` baseline rel-L2 0.61
  vs ~0.50 elsewhere), so diagnostics and reporting are per-split, not pooled.

**Planned future shift axes (architecture must not preclude):** partially-open
and fully-outdoor scene families (new geometry/material regimes → new shift
splits, e.g. `test_outdoor_shift`); and dynamic scenes — moving / multiple
source–receiver positions and a time dimension — which additionally require a
simulator change (GSound-SIR is static, single source–receiver) and so sit
behind the simulator seam. Because the split set is config-declared, these are
additions, not rewrites.

## 7. Config system — three parameter roles

The rebuild lifts hardcoded constants into config, but **exposing ≠ tuning.**
Every parameter declares a role:

- **`fixed`** — single value, used as-is, not searched. **Still config-declared,
  never a magic number in code:** any fixed value can be changed, or promoted to
  `tuned`/`swept`, at any time. "Fixed" means "not currently varied," not
  "locked." Assume the researcher may want to change any currently-fixed value.
- **`tuned`** — hyperparameter search; explored over a space, **one value
  selected on the validation split**; search trials are scaffolding, not
  results. Only the selected value enters reported models.
- **`swept`** — a **research variable**; every value is run to completion and
  reported on the **held-out test split**; never collapsed to "best." The
  cross-value comparison *is* a primary result.

```yaml
# scalar = fixed
sample_rate: 48000
high_ray_budget: 200000           # reference is fixed
# tuned: optimized on validation
learning_rate: { tune: { space: [1.0e-4, 1.0e-2], scale: log } }
loss_weights:  { tune: { ... } }
# swept: research axis, every value reported on test
low_ray_budget: { sweep: [1000, 2000, 5000, 20000] }
```

**Starter role classification (from the paper's appendices):**

| Role | Parameters |
|---|---|
| fixed | sample_rate 48k, ir_duration **per config** (base.yaml 4.25 s, research_i.yaml 3.0 s RI-pinned — see §11.2), base_seed 42, high_ray_budget 200k, QC thresholds (onset 2 ms, min energy, max path-file 128 MB), normalization scheme |
| tuned (on valid) | learning_rate, batch_size, loss-term weights, Huber δ, model depth/width/kernel/dilation, early-stopping patience |
| swept (research) | low **diffuse** ray budget (was fixed 5k), retained-path count k, band resolution (representation study); ambisonic_order = fixed-3 v1 but sweep-capable |

*The ray budgets are **diffuse** ray counts (RD-12): `low_ray_budget` /
`high_ray_budget` drive GSound's `diffuse_count`, while `specular_count` is
declared in `configs/simulators/gsound_sir.yaml` and held fixed across both legs,
so the swept axis is never a moving total. Both counts are stamped into canonical
render meta. They stay top-level config fields rather than simulator params so the
swept axis survives a simulator name change (RD-40).*

*Scene parameters — per-split counts, total dataset size, and the
placement / material / geometry regimes — are all config-declared. The Research I
values are a starting point, expected to grow or change in Research II; some
(dataset size, ray budget) may become swept axes. Nothing scene-related is
hardcoded.*

**Role interactions (important — the falsifier checks these):**
- **Leakage rule:** `tuned` params are selected on **valid only**; `swept`
  values each receive an **independent test eval** (no selection across values →
  no leakage). Never select any param on a test split.
- **Sweep × tune:** to avoid combinatorial blow-up and keep the sweep
  interpretable, **tune once at a reference operating point, then hold tuned
  values fixed across the sweep** (default). Per-value re-tuning is an option if
  fairness across the swept axis demands it — record which was used.
- **Provenance:** a `swept` param expands into N sibling runs to compare; a
  `tuned` param expands into M trial runs that roll up to one selected run. The
  per-run stamp records role, concrete value, and (tuned) the space + selection
  criterion, (swept) the sweep set + sibling run IDs.

## 8. Plugin interfaces (signatures)

```python
class Simulator(Protocol):
    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult: ...
    # IRResult: ir[C,T] float32, paths: PathData, meta: dict

class Representation(Protocol):                                   # + registry; nested Params schema
    center_freqs: "list[float]"                                   # band metadata ([] if band-less)
    def encode(self, ir: "ndarray[C,T]") -> "Tensor": ...        # → energy domain (dB log energy)
    def decode(self, env: "Tensor", carrier: "ndarray[C,T]") -> "ndarray[C,T]": ...   # env in dB
    def loss(self, pred: "Tensor", target: "Tensor", delta: float) -> "Tensor": ...   # δ in operand domain (cf. F-06)

class Model(Protocol):                                            # nn.Module + registry
    def forward(self, x: "Tensor", aux: "Tensor | None" = None) -> "Tensor": ...

class Metric(Protocol):
    def compute(self, pred, high_ref, low_ref) -> "dict[str, float]": ...  # per scene
```

## 9. Stats & reporting

- **CI for small per-split n** (samples are small now and grow in Research II)**:**
  **bootstrap percentile CI** (default; robust to non-normal metric
  distributions) over per-scene values; report mean ± 95% CI. t-interval only as
  a sanity cross-check. **Computed per test split, including each shift split
  separately — the in-distribution and the three shifted splits are reported side
  by side, never pooled.**
- **Minimum detectable effect size (MDES):** report, per metric/split, the
  smallest baseline-vs-denoised difference detectable at the current n. Guards
  against over-reading sub-noise differences (the professor's point).
- **Paired improvement, keyed on the metric's declared `kind`:** the CI/MDES
  quantity is the per-scene paired improvement — `|low−high| − |pred−high|`
  (match_reference), `pred − low` (maximize), `low − pred` (minimize) — never
  the absolute metric value's σ, and never an assumed match-reference form.
- **Scored vs attempted:** every (split, metric) reports `n_scored/n_attempted`;
  an unscored metric renders as `unscored`, never as a number (no descriptive
  mean in a results column). Per-leg drop reasons live in eval's `drops.csv`.
- **Failure-case grouping:** group poorly-performing scenes by geometry,
  material, placement, baseline low-ray error, reverberation, retained-path
  stats — to test whether failures correlate with identifiable conditions.
- **Swept variables** render as comparison curves/tables across values; **tuned**
  variables render the selected value + a brief sensitivity note (not the trials).
- Reporting also emits the supplementary bundle (resolved configs + versions +
  git SHA) that replaces appendix bulk in the paper.

## 10. Invariants (enforced as `tests/`)

1. Train/valid/test scene sets disjoint (hash scene specs).
2. No test split used for selection / early stopping / tuning.
3. Normalization stats from training split only (low and high stats computed
   separately). The residual framing (pred = low + model(low)) requires input and
   target in one affine frame, so BOTH are normalized with the HIGH stats; the
   separately-computed low stats are stamped for provenance but not applied. The
   invariant that binds is *train-split-only* — no valid/test leakage.
4. `(C,T) ↔ (T,C)` round-trip lossless; channels never mixed with time.
5. Config + seed → reproducible run.
6. Per-scene rows preserved through `eval`; `stats` is the only collapsing point.
7. Every loss term active (no silently zero/NaN/clamped term; δ vs signal scale).
8. `swept` values each get full held-out eval; `tuned` selection touches valid
   only.
9. Every test split is evaluated independently; the four test splits are never
   pooled into a single aggregate "test" result.
10. The split set is config-declared; each shift split differs from `test_id`
    only along its named axis (controlled shift, no uncontrolled co-variation).

## 11. Experiment ledger & sequence

| ID | Run | Provenance | Gate |
|---|---|---|---|
| D0a | Headroom probe — **all splits** (train/valid + test_id/material/placement/geometry) | single | proceed only if gap is real; expect it to differ per split |
| D0b | Oracle upper bound | single | proceed only if ceiling clears metrics |
| E1 | Reproduce old null (waveform, old config) | single | must match prior result (faithfulness) |
| E2 | Energy-domain residual (lead) | single | beat baseline in energy/metric domain? |
| E3 | Tune E2 at reference op-point | tuned → 1 selected | select on valid |
| E4 | Low-ray budget sweep | swept → N siblings | report metric-vs-ray-count on test |

Run D0 before E1; run the `falsifier` after E1 and before trusting E2+. **All
eval-bearing runs (E1–E4) report metrics per test split, including each shift
split separately** — D0a characterizes per-split headroom (it will differ, most
notably for `test_geometry_shift`), and E2+ must show generalization vs
robustness as a per-shift breakdown, not a pooled test number.

### 11.1 Render threshold (user decision, 2026-08-12)

Real `gsound_sir` renders are emulated on this host and are the project's scarcest
resource, so they are bounded — but by a **standing threshold**, not a per-cycle
grant:

- **Any session may render up to 30 scenes.** No permission clause, no allocation
  table, no per-item sub-grant.
- **Beyond 30, the session stops and asks the user**, giving the reason and an
  estimated wall-clock. A full-dataset render for the Research I reproduction is
  an expected and legitimate reason to exceed it; the point of the threshold is
  that a session cannot spend a day of compute without the user knowing first.
- **Every render records measured per-scene wall-clock**, so the estimate in that
  request is measured rather than guessed. Nothing in the repo recorded this
  before, which is why three cycles rationed renders against an unmeasured price.

30 is also the smallest count at which a probe starts to say anything about
statistical significance; the superseded ≤4/≤6 grants could not.

**The dataset-render gate.** No full-dataset (720-scene) `gsound_sir` render until
BOTH conditions hold. This lives here rather than in the review ledger because it
is a standing rule of the study, not an unsolved defect — a gate carried as a
ledger row drifted its own lift condition three times.

- **(i)** Zero OPEN ledger rows anchored on this explicit path list:
  `src/amcd/scenes/**`, `src/amcd/evaluation/**`, `config.py` split handling,
  `configs/*.yaml` split declarations. The list is explicit because the earlier
  free-text version ("the metric path") admitted whatever reading was convenient.
  Severity is not a criterion: the gate lifts at zero OPEN, not at zero
  blocker/major.
- **(ii)** The ray-budget probe has validated the high leg
  (`high_ray_budget`) as a converged reference against the config-declared
  `convergence:` tolerances, on **all three** declared quantities — per-band
  energy, T30 and C50 — through the production ISO path, driven via
  `build_simulator`.

Condition (ii) exists because every paired-improvement number in the project, D0a's
headroom and D0b's carrier test all treat that leg as ground truth, and nothing had
ever checked it. It is a tolerance check over a handful of scenes, never a
CI-backed convergence claim, and must be reported as such.

### 11.2 The reverberant corner is not measurable on GSound-SIR (user decision, 2026-08-12; numbers re-derived 2026-08-13)

**Part of the declared population cannot be scored on this backend, and lengthening
the record does not help.** Upstream's adaptive energy trim closes the IR before the
decay has fallen far enough for an ISO-3382-1 T30 fit, so the captured decay range
`60·support/T60` drops below the standard's 45 dB at the reverberant end. Measured
identical at 3.0 s, 4.25 s and 30 s of `ir_duration`: the binding limit is the trim,
not the buffer.

**What sets the trim was measured wrong until 2026-08-13, and the correction
matters.** The first law was `support ≈ c·T60^k`, fitted on renders whose rooms were
all scalings of one shoebox — so room size and decay time moved together and either
could appear to drive the record. A crossed probe
(`scripts/support_law_probe.py`, 21 renders, `experiments/support_law/`) separates
them and refutes T60 outright: holding geometry fixed and moving absorption across
the declared `mixed` support, a **16× change in T60 moves realized support by
0.00 % / 0.49 % / 0.00 %**. What does drive it is the backend's reflection-order
bound and the room's surface area — `support ≈ c·diffuse_depth^0.688·S^0.464`,
declared in `configs/simulators/gsound_sir.yaml` as a conservative lower envelope.
So the admissible region is **not a T60 ceiling at all**; the earlier "T30 stops being
admissible above T60 ≈ 1.85 s" is withdrawn.

**Decision: this is accepted as a limitation OF GSOUND-SIR.** The study does not
reshape its declared population around it. Consequences, all deliberate:

- Affected scenes still render. Their T30 comes back **unscored with a reason** from
  the estimator's own ISO bound (AC-176) — never as a truncation-biased number — and
  the per-split scored-vs-attempted counts carry that into the report.
- **`configs/base.yaml`: 23/600 = 3.83 %** censored (was reported as 2.8 % under the
  refuted law, which under-counted the censoring it exists to bound). Concentrated in
  the reverberant splits, zero in the near-anechoic `test_material_shift`.
- **`configs/research_i.yaml`: 40/720 = 5.56 %, which EXCEEDS its own declared
  `max_t60_over_ir_duration_frac` of 0.05, so `gen-scenes` now REFUSES it.** The
  Research I population is not fully admissible on this backend. That is an open
  research decision — raise the tolerance with a stated reason, or accept the
  reproduction over a declared subpopulation — and it is deliberately not resolved by
  tuning the number, because the gate exists to surface exactly this.
- `scenes.max_t60_over_ir_duration_frac` is a **regression tripwire on the censoring
  rate**, not an admissibility gate: it trips if the rate roughly doubles.
- Any E1/E2 claim over the reverberant end is a claim over a **censored
  subpopulation**, and must be reported as such.

**This strengthens the case for the custom renderer** in `research_I_paper.md` §6
future work: a raytracer whose record length is a declared parameter rather than an
emergent property of an energy heuristic would remove this ceiling outright. Recorded
here so the eventual recommendation rests on a measurement.

## 12. Environment (cross-platform is a project requirement)

The pipeline must run in two host configurations with identical code — no
platform-keyed branches, no host-specific paths in `src/amcd/`:

- **macOS / Apple Silicon (this machine):** `render` (GSound-SIR) is x86-only —
  invalid-processor on Apple Silicon — so it runs under Rosetta 2 emulation in a
  dedicated `osx-64` conda env (or `arch -x86_64`). The x86/emulation boundary
  lives entirely behind the simulator seam and in environment setup, never in
  package code. **All other stages native arm64 + MPS.** Never emulate training.
- **Native x86_64 (Ubuntu or Windows desktop):** the whole pipeline, including
  `render`, runs natively with no emulation and no code edits. Torch device
  selection falls back MPS → CUDA → CPU; it never assumes MPS.
- **GSound-SIR provenance:** pulled from upstream GitHub
  (https://github.com/yongyizang/GSound-SIR) at a **config-pinned commit SHA**
  for reproducibility — never vendored with local modifications.
- Framework: **PyTorch** (custom losses, energy-domain targets, mature MPS).

## 13. Open calls remaining

- Bootstrap vs t-interval (defaulted to bootstrap — confirm).
- Tune-once-then-hold vs per-value re-tune across the sweep (defaulted to
  tune-once — confirm for fairness needs).
- Target dataset size beyond the prior 500 train (compute/render-time dependent
  — needs your budget).
- Future shift splits (outdoor / partially-open, then dynamic + multi-source-
  receiver): when added, decide whether each is a robustness *shift* split or a
  new in-distribution family — and note that dynamic/time axes are gated on the
  simulator change (§12).
