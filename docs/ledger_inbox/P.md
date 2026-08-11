# Lane P inbox — cycle4

Branch `lane/P-cycle4`. Written by lane P, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on lane/P-cycle4", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

---

## READ FIRST — one test fails on this branch, and it is not mine to fix

`tests/test_simulator_seam.py::TestStageFingerprint::test_undeclared_stage_still_caches_on_bare_sentinel`
FAILS on `lane/P-cycle4`. It asserts `STAGE_FINGERPRINT["report"] is None`, which
is exactly the state F-63 required me to change. `test_simulator_seam.py` is not
in lane P's owned set, so per protocol rule 4 this is the integrator's to apply.

Remedy, **verified working** (I ran the substitution before proposing it —
`diagnostics` is now the only unwired stage, and its bare sentinel still caches):

```python
# tests/test_simulator_seam.py:278-284
# `eval`/`stats` were promoted out of the gap by RD-54 and `report` by F-63, so
# `diagnostics` is the standing example of the unwired path.
assert STAGE_FINGERPRINT["diagnostics"] is None
cfg = tiny_config(scenes={"n_id": 4})
pipe = self._pipeline(cfg, tmp_path)
pipe._mark_done("diagnostics")
assert pipe._is_done("diagnostics")
```

Everything else passes: **415 passed, 1 failed** (`PYTHONPATH=<worktree>/src pytest`).

---

## Rows closed

All four majors were **reproduced on the current code first**, then re-run after
the fix. Source patches were applied to a **scratch copy of the package** under
`/private/tmp/.../scratchpad/pkg2`, never to the worktree — this is how
`representations/spectrogram.py` (lane M) was patched without touching it.
`git status` stayed clean throughout.

### F-64 — `preprocess` had no `code_version` · CLOSED

BEFORE (patched `spectrogram.encode`, power × 4):

```
[skip] preprocess (cached)
[FAIL] train: ... Re-run with --force to rebuild 'train'
```

Following that instruction exactly — force `train`, then `infer`, then `eval`,
then `stats`, each named in turn by the next refusal — reached **exit 0 and a
full report**, while `preprocessed/train/scene_0000_high.pt` stayed
**bit-identical** (`cmp` exit 0) to the pre-change tensor. The reported metrics
derived from an encoding no stage in the run produced.

AFTER:

```
[skip] gen-scenes (cached)
[skip] render (cached)
[FAIL] preprocess: Stage 'preprocess' was cached under a DIFFERENT config
    code_version: 'fb77b496...' → '0d45ec6d...'
  Re-run with --force to rebuild 'preprocess'
```

The refusal now names the stage whose artifacts are actually stale.

Fix: `code_version` in `_preprocess_fingerprint`; scope
`("data", "representations", "simulators/base.py")`. **The third entry is not in
the brief** — see RD-97 below.

### F-63 — `stats` carried no `code_version`, `report` no fingerprint · CLOSED

BEFORE (patched `bootstrap_ci`, `ci_lower` × 0.001): all nine stages `[skip]`,
exit 0, `ci_table.csv` byte-identical to the old code's output (`diff` exit 0).

AFTER, three separate refusals:

* patch `stats/aggregate.py` → `[FAIL] stats`, with gen-scenes…eval all cached.
* patch `reporting/tables.py` → `[FAIL] report`, with gen-scenes…stats cached.
* **the chain**: force `stats` under changed CI code, then run `report` — its own
  inputs are unchanged, and only the chain catches it:

```
[FAIL] report: ...
    upstream.stats: '53c96bad...' → '27b9e4be...'
```

Fix: `code_version("stats")` scope `("stats", "evaluation")`; new
`_report_fingerprint` over `{report_format, code_version(("reporting",))}`; and
`STAGE_UPSTREAM["report"] = "stats"`.

**The chain link goes beyond the row text** and was taken deliberately, approved
by the operator and ruled warranted by `research-director`: a fingerprint alone
leaves `summary.txt` cached across a re-run `stats`, i.e. F-63's own defect one
link downstream. The transcript above is the demonstration.

### F-66 — `eval`/`infer` omitted `data`; the docstring overstated the test · CLOSED

MEASURED, patching `data.normalization.denormalize` (+10 dB):

| stage | before fix | after fix |
|---|---|---|
| `eval` | unchanged (`bbb5f0e2…` both ways) | `09ce8cd2…` → `27b4507b…` ✅ |
| `infer` | unchanged (`89000934…` both ways) | `bf2a1fa0…` → `00a0e224…` ✅ |
| `stats` / `report` | — | correctly **unchanged** (they do not denormalize) |

That last row matters: the new scopes are precise, not blanket.

The docstring claim is now **true and bounded**. `tests/test_stage_cache.py::
TestDeclaredScopeCoversWhatTheStageImports` asserts each declared scope is a
superset of the stage's static transitive `amcd.*` import closure (module- and
function-level) minus `_CORE_SOURCES`. Both `provenance.code_version` and the
test docstring now state what it **cannot** see: plugins loaded by NAME through
the registry (`representations`, `models`, `simulators`) are invisible to static
analysis and remain a declared judgement. I did not want to close an
over-claiming-docstring row by writing a new over-claiming docstring.

Also folded in: `__init__.py` added to `_CORE_SOURCES` — `amcd/__init__.py` is in
every stage's closure and was in no scope, so an edit to it invalidated nothing.

### F-65 — `metric_edt_variance_limited_s` in no fingerprint · CLOSED

BEFORE: 0.15 → 5.0 on a complete run_dir gave all nine `[skip]`, exit 0,
`config.yaml` re-stamped to 5.0, `ci_table.csv` byte-identical. Forcing eval+stats
produced EDT `n_estimator_variance_limited` of **3/3/3/3** against the served
**0/1/3/0** — so the disclosure column really was the old threshold's.
(The row predicted 3/3/2/3; I measure 3/3/3/3. The finding is unaffected.)

AFTER:

```
[FAIL] eval: ...
    metric_edt_variance_limited_s: 0.15 → 5.0
```

**The class-level guard is the real deliverable.** `FINGERPRINT_EXEMPT_FIELDS`
(in `pipeline.py`) + a perturbation test: every `Config` field must either move at
least one stage fingerprint when perturbed, or carry a declared exemption. A new
field fails the test until someone decides which it is.

I **mutation-checked** the guard rather than trusting a green run: deleting the
`metric_edt_variance_limited_s` line from `_eval_fingerprint` makes the
class-level test fail (`test_perturbing_the_field_invalidates_at_least_one_stage
[metric_edt_variance_limited_s]`), not merely the specific one. Deleting `"data"`
from eval's scope fails the closure test. Both restored after.

### F-69 — host-dependent cache key · CLOSED

Verified on real trees, not just by injection: this exFAT worktree carries **42
`._*.py`** sidecars under `src/amcd/`; an `rsync` copy excluding them (0 sidecars,
standing in for APFS / the native x86_64 second host) now produces **identical**
`code_version` for all six scoped stages (`eval 09ce8cd2…`, `train bf2a1fa0…`, …).
Before the fix these differed, so a run_dir carried between hosts was refused with
a diff naming no leaf. `__pycache__` filtered too.

(The row said 46 sidecars; the number that actually mattered is the 42 matching
`*.py`. I corrected the count in the docstring rather than repeat the row's.)

### F-73 — `_CONFIGS_DIR` assumed a source checkout · CLOSED (see caveat)

Closed **as the row is written** — it offers two remedies and I took the second
("make the config root a required input with a clear error"), entirely inside
`config.py`. `_CONFIG_ROOT_CANDIDATES` prefers a packaged `amcd/configs/` and
falls back to the checkout sibling; `_require_configs()` raises a message naming
every location tried and what would fix it, at load, instead of a bare
`FileNotFoundError` from `_merge_yaml`.

Constraint worth knowing: `_BASE_YAML` is imported by `tests/conftest.py` and
`tests/test_simulator_seam.py`, neither of which lane P owns, so the module-level
names had to survive. They do.

The packaging half is a **new row**, not an unfinished half of this one — see
below.

### F-74 — device never recorded · CLOSED

`versions.json` from the canonical dry run now ends:

```json
  "code_version": "76caec5b8375df7b...",
  "device": "mps",
  "platform_machine": "arm64"
```

`select_device()` is now one function shared by `trainer.py`, `infer.py` and the
stamp (it was duplicated verbatim), and is in **no** fingerprint — a test asserts
that, since a device change must not invalidate an expensive artifact.

### RR-35, RR-36, RR-42 · CLOSED

RR-35: `# ── Plugin blocks + layer merge ──` banner added; the shift-split block
extracted from `Config._check` as `_check_shift_splits`, so `_check` reads as a
list of named checks.

RR-36 / RR-42 **appear to contradict each other and do not** — they name two
different claims, and each is now stated once:

* "a config fingerprint is blind to code changes" → stated once at
  `STAGE_CODE_SCOPE`, cited from the other three sites (RR-36).
* the *scoping* rationale (per-stage vs whole-package hash) → stays in
  `provenance.code_version`, and `STAGE_CODE_SCOPE` points at it (RR-42).

Four reproduction transcripts cut to rule + ledger id; the invariant sentences
(sentinel-not-recomputation, F-58 invalidate-before-mutating, per-aspect seeds)
kept.

---

## Counter-check (brief's explicit requirement)

An edit to `search.py` — in no stage's scope — leaves **all nine stages cached at
exit 0**. The guard does not fire on irrelevant changes, which is the whole reason
scopes are per-stage rather than a whole-package hash.

A false start worth recording: my first counter-check appeared to fail, refusing
`report`. It was my own probe residue — a stray blank line left in
`reporting/tables.py` by an append+`sed` revert — not a defect. Found by
`diff -r` against the worktree, fixed, re-run clean. It doubles as evidence the
guard is sensitive to a one-line change in `reporting/` **and** that it refused
only `report`, leaving gen-scenes…stats cached.

---

## New findings for the ledger

```
RD-93 | research-director | major | OPEN | src/amcd/pipeline.py STAGE_FINGERPRINT / STAGE_UPSTREAM docstrings | Both docstrings enumerated a state F-63/F-64 falsify ("the three stages ... train, infer, eval therefore carry code_version()"; "the remaining Nones are diagnostics and report"; "diagnostics and report are terminal and stay unchained"). Shipping them would re-commit F-66's own defect — a provenance docstring asserting other than the code does — inside the cycle convened to close it. | FIXED IN THIS BRANCH, in the same commit as F-63/F-64 rather than the RR-36 pass. Delete on re-review confirmation.
RD-94 | research-director | minor | OPEN | src/amcd/pipeline.py FINGERPRINT_EXEMPT_FIELDS | Exemption reasons must state a re-entry condition, not a present-tense fact. In particular max_onset_ms/min_energy_db are NOT orphans: configs/base.yaml:172-178 declares them as real-gsound_sir render QC gates, unused by dry_run, so a "no consumer" reason would invite deleting a provision for the real-raytracer work. | FIXED IN THIS BRANCH: each exemption says why it is absent today AND what makes it non-exempt; a test enforces that. Delete on confirmation.
RD-95 | research-director | minor | OPEN | tests/test_stage_cache.py TestEveryConfigFieldIsCoveredOrDeclaredExempt | The perturbation guard proves coverage at TOP-LEVEL Config field granularity only; a new leaf inside a nested model is covered only because _preprocess_fingerprint dumps SplitSpec wholesale, while _gen_scenes_fingerprint dumps it selectively (F-50). | FIXED IN THIS BRANCH: limit stated in the test docstring. Probe values live in tests/, exemption table in pipeline.py. Delete on confirmation.
RD-96 | research-director | minor | OPEN | src/amcd/provenance.py select_device | host_platform() is genuinely provenance; select_device() is runtime POLICY, placed in a core module because Config.stamp must record the device and a core module importing amcd.training would drag training/ into every stage's import closure. COST: provenance.py is in _CORE_SOURCES, so any edit to device-selection logic invalidates EVERY fingerprinted stage — and once gen-scenes/render gain a code_version (see RD-99), editing the MPS→CUDA→CPU fallback would force a multi-hour re-render under emulation, on exactly the code the second-host requirement makes someone touch. | Constraint and cost are documented in the source. Proposal: src/amcd/device.py in a cycle whose partition grants lane P a top-level module.
RD-97 | research-director | minor | OPEN | src/amcd/pipeline.py STAGE_CODE_SCOPE["preprocess"]; tests/test_stage_cache.py closure test | TWO CROSS-LANE COUPLINGS THE INTEGRATOR MUST EXPECT AT MERGE. (1) preprocess's scope names "simulators/base.py" — lane R's tree — because data/preprocess.py:17 imports SceneSpec; the brief's ("data","representations") is not a superset of the real closure and would fail the F-66 test added in the same cycle. If R moves SceneSpec, provenance.py raises ValueError: loud, not silent. (2) The closure test asserts scope over evaluation/, representations/ (lane M) and simulators/ (lane R); one new import in either lane makes P's scope insufficient and fails P's test ON THE INTEGRATED TREE, after P's pass condition was measured green here. `git merge v3-rebuild` does not cover it (M and R merge later). | THE REMEDY IS A ONE-LINE SCOPE UPDATE IN pipeline.py, NEVER WEAKENING THE ASSERTION. A red closure test means a real dependency is undeclared — which is precisely F-66.
RD-98 | research-director | minor | OPEN | src/amcd/stats/aggregate.py; src/amcd/reporting/tables.py | Both left UNTOUCHED and able to take a new column, per the brief. Recording the interaction lane P's work creates: code_version(("stats","evaluation")) and code_version(("reporting",)) are now what will make the integrator's post-merge column additions (F-70, RD-65's report half, AC-43/RD-82) correctly invalidate cached stats/report. The machinery those rows need is already in place. | No action; context for the serial queue.
RD-99 | lane P | major | OPEN | src/amcd/pipeline.py STAGE_FINGERPRINT gen-scenes, render | `gen-scenes` and `render` carry NO code_version — the same defect class as F-64, one link further upstream. A change to scenes/generator.py or a simulator backend is invisible to the cache, and preprocess's new code_version does not cover it (different scope). NOT taken in this lane: wiring render means a simulator edit forces a re-render, which under x86 emulation is the multi-hour artifact, so it is a policy call with a cost attached rather than a quiet addition. Interacts with RD-96. | Decide deliberately: scope render to simulators/ (and accept re-render on backend edits), or state in STAGE_FINGERPRINT why these two are exempt.
RD-100 | lane P | minor | OPEN | src/amcd/pipeline.py STAGE_FINGERPRINT["diagnostics"] | `diagnostics` is now the ONLY stage with no fingerprint, and it is what forces the d0a_*/d0b_* entries in FINGERPRINT_EXEMPT_FIELDS. Non-trivial: the D0a verdict ("signal to learn at this ray budget") is a research gate on the ray-count question, and it is currently reachable through the cache under changed thresholds. | Wire it, or record why a research-gate artifact may be served stale.
RD-101 | lane P | minor | OPEN | pyproject.toml; configs/ | F-73's OTHER sanctioned remedy — ship configs/ as package data — needs pyproject.toml and a move of configs/, neither owned by lane P. config.py already prefers a packaged `amcd/configs/` over the checkout sibling, so this is a packaging change with no code change behind it. Filed as a NEW enhancement row, not as F-73 half-done: F-73 is closed as written. | Integrator or a cycle that owns packaging.
RD-102 | lane P | minor | OPEN | .git/objects/pack/._pack-*.idx | Every git command in this worktree prints `error: non-monotonic index .../._pack-e5b872ce....idx` — git is trying to read a macOS AppleDouble sidecar as a pack index. Same root cause as F-69, one layer out: the exFAT volume materialises `._*` files that tools then glob. Harmless (commands succeed) but it buries real output, and it is noise in every future evidence transcript. | Delete the `._pack-*` sidecars in .git/objects/pack/, or set the volume to not create them.
```

---

## Not done, deliberately

* **F-70, RD-65's report-table half, AC-43/RD-82** — terminate in my files but sit
  on the integrator's serial queue. Not started, per the brief. Both modules left
  in a state that can take a new column (RD-98).
* **RR-27** (stale design-spec citations in `config.py` **and**
  `simulators/base.py`) — spans lane R; integrator's queue.
* **`tests/test_simulator_seam.py`** — not owned; the failing assertion and its
  verified remedy are at the top of this file.

## Reviewers

`research-director` was run **on the plan, before implementation**, per the
implementation loop; its six findings are RD-93…98 above, all folded into the work
rather than deferred.

`falsifier`, `acoustics-reviewer` and `readability-reviewer` were then run over
the CURRENT state at b992a78 — a **self-check on an unintegrated branch, not a
clean pass** (rule 5). They raised 22 findings between them. **The lane is NOT
done**: what follows is what I fixed in response, and what remains OPEN.

### The headline claim was overstated, and I am withdrawing it

My write-up above said the four routes to a cached reported number are closed.
Four SPECIFIC routes are. The CLASS claim — "a reported number is no longer
reachable through a cached stage" — **is false**, and the falsifier proved it by
running the probe I had declined to run: a two-character edit to
`simulators/dry_run.py` leaves all nine stages `[skip]` at exit 0 while a fresh
run_dir under the same code reports different CIs (`test_geometry_shift` C50
`improvement_mean` −0.6815340 cached vs −0.6824248 fresh).

I filed that gap myself as RD-99 and deferred it as a policy call. Two things I
got wrong in doing so:

1. I scoped it to "simulators/". It is also `scenes/generator.py` (placement and
   admission sampling) and `simulators/render.py` (QC gates, pruning) — in no
   stage's scope either.
2. I missed that `versions.json` is **re-stamped** on the all-cached run with a
   whole-package `code_version` byte-equal to the fresh run's, and
   `reporting/tables.py` copies it into `report/`. The canonical provenance
   channel positively vouches that the new code produced the old numbers. That is
   worse than the F-53 pattern this cycle was convened to close, and it is the
   part I had not seen. Filed as **F-75 (blocker)** / **AC-44** and awaiting the
   policy decision — a re-render under emulation is the cost on the other side.

### Fixed in response (commit follows this write-up)

* **F-76 (major, a regression I introduced)** — giving `report` a fingerprint made
  cycle-3 `{"fingerprint": null}` sentinels reachable on a path that did
  `set(None)`, so every pre-b992a78 run_dir died with a bare `TypeError` instead
  of the actionable "predates fingerprinted caching" message. `_is_done` now
  treats a recorded `null` as the legacy case. Two regression tests; the fix is
  generic, so `diagnostics` gaining a fingerprint later cannot repeat it.
* **F-77 (major)** — my F-66 closure test resolved relative imports off-by-one for
  a package `__init__.py` and then DROPPED whatever failed to resolve, silently.
  Measured before: `_amcd_imports("amcd.data")` returned nothing at all, and
  preprocess's closure did not contain `representations.spectrogram` — the encoder
  that is F-64's own reproduction. 60 edges were invisible. The walker now anchors
  relative imports on the package and ASSERTS rather than dropping; `amcd.data`
  yields 4 modules, preprocess's closure 11 → 15, and it now contains the encoder.
  **No scope was actually under-declared** (confirmed independently by the
  falsifier's corrected walker and by the test passing), so this was a broken
  guard, not a live hole — but it was my docstring claiming more than my test
  checked, which is precisely F-66 one cycle later. Docstring corrected to say so.
* **F-78 (major)** — the F-65 guard's stated limit was wrong about the one nested
  model that matters: `Seeds` is dumped wholesale by NO fingerprint, and the
  `seeds` probe perturbed only `master`, which moves everything downstream. A new
  per-aspect seed — invariant #5, and `split_assignment` is the leakage-critical
  one — was entirely unguarded. Now swept per-seed over `SEED_NAMES`.
* **F-79 (major)** — `code_version` could hash NOTHING without raising, the exact
  outcome its own ValueError text exists to prevent, because the existence check
  sat inside the per-path loop. Two triggers: a scope entry naming an existing
  `.py`-less directory, and `__pycache__` matched against the ABSOLUTE path so an
  ancestor of that name collapsed every scope to one constant. `_hashable_sources`
  now matches relative and raises on an empty directory.
* **F-80 (minor)** — `_require_configs` checked `base.yaml` only, so a configs root
  missing `models/`/`representations/`/`simulators/` loaded a VALIDATED Config with
  empty plugin params and failed later on a pydantic error naming neither. Now
  checked and named.
* **AC-47 (minor)** — `STAGE_CODE_SCOPE["eval"]`'s rationale was false: eval does
  not decode, `infer` does. The scope entry is right and stays; the reason is now
  the true one (registry-resolved, invisible to the closure test). Same
  over-claiming class as F-66, in the module whose premise is "that claim is
  auditable".
* **AC-46 (minor)** — the `d0b_t30_jnd_frac` exemption now records that it is also
  the calibration criterion behind the eval-fingerprinted
  `metric_band_resolvability_margin`.
* **AC-48 (minor)** — `summary.txt` now prints the threshold's VALUE and unit:
  `high-variance: EDT below metric_edt_variance_limited_s = 0.15 s`. It named the
  key symbolically before, and F-65's evidence is that this key was served at 0.15
  while `config.yaml` stamped 5.0.

**Suite after these fixes: 422 passed, 1 failed** — the failure is still only the
cross-lane `test_simulator_seam.py` row at the top of this file. Canonical dry run
still completes, exit 0.

### F-75: the false-witness half is closed; the staleness is a standing decision

Operator decision: **do not wire `render`/`gen-scenes`** — the re-render cost under
emulation is real and RD-99 stays open as a policy call for the integrator. But
stop the provenance channel vouching for artifacts it did not describe.

Implemented:

* `_mark_done` records `code_version_unscoped` (the whole-package hash of the code
  that actually wrote the artifacts) as a sibling of `fingerprint`, never inside
  it — so it is recorded and never compared, and cannot turn every stage into a
  whole-package hash.
* `Pipeline._warn_if_unprotected_and_stale` warns on stderr, for stages with NO
  scoped `code_version` only, when a cached stage's recorded hash differs from the
  current source. Fingerprinted stages are skipped deliberately: a scoped change
  already refuses them, and a warning there would be noise, which is how an
  operator learns to ignore warnings.
* `versions.json` now carries `code_version_describes`, in the file rather than in
  a comment no reader of the JSON sees.

Verified with the falsifier's own probe (`noise_scale` 1.0 → 2.0 in
`simulators/dry_run.py`, scratch copy):

```
[warn ] gen-scenes is cached and the package source has CHANGED since its
        artifacts were written (295d31ea7f13 → 1e08fc60bdf5), but 'gen-scenes'
        declares no code_version, so nothing refuses it (RD-99). ...
[warn ] render is cached and the package source has CHANGED ...
[warn ] diagnostics is cached and the package source has CHANGED ...
[skip] × 9, EXIT=0
```

and on disk the two records now disagree openly:

```
versions.json  code_version          : 1e08fc60bdf5   (this invocation)
stages/render.done code_version_unscoped: 295d31ea7f13   (what built the artifacts)
```

The staleness remains — that is RD-99's call, unchanged. It is no longer silent,
and `versions.json` no longer asserts the new code produced the old numbers.
**The cycle's cache-protection claim still stands only for `preprocess`…`report`.**

### OPEN, not fixed

* **RD-99 / F-75 / AC-44 (blocker→policy)** — `render` and `gen-scenes` still carry
  no `code_version`. Deliberate, now disclosed at runtime rather than silent.
  Integrator's decision.
* **AC-45 (major)** — acoustics-reviewer rates the unfingerprinted `diagnostics`
  higher than my RD-100 did, and is right about why: the D0b output is a physical
  VERDICT ("CARRIER CEILING CLEARS … Proceed to E1"), invalidated by nothing, and
  measured stale under changed `ir_duration`, `ambisonics_order`, ray budgets and
  `sample_rate`. A stale clearance is a false clearance of the project's own
  premise. RD-100 should be re-rated major.
* **RR-46…RR-57, RR-59 — ADDRESSED** (see below). **RR-58 (README) is cross-lane**
  and stays for the integrator: the README documents only "a cached stage is
  skipped, pass `--force`", which after this cycle is the least important half —
  a mismatch is a LOUD refusal naming the stage, and `--force` DISCARDS artifacts.

### The readability pass (second round)

RR-36 was the fair hit and I own it: I closed it by cutting four reproduction
transcripts and then writing four new ones for the reproductions I ran this cycle
— the same pattern, by the same standard. Now actually done:

* **RR-46** — the four transcripts I added are cut to rule + ledger id
  (`_preprocess_fingerprint`, `_stats_fingerprint`, `_report_fingerprint` 13 lines
  → 6, and the 8-line narration inside `_eval_fingerprint`'s dict literal → 2).
  The evidence lives here in the inbox and in `git log -S`, which is what the ids
  are for.
* **RR-47** — F-53 now narrated once (at `STAGE_UPSTREAM`, which owns the chain),
  with `_train_fingerprint` citing it. The RD-41 "report is terminal" rebuttal went
  from four statements to one at `_report_fingerprint`; the other three cite it.
* **RR-48** — the same trim applied to the test-class docstrings. The
  "WHAT THIS TEST CHECKS / WHAT IT CANNOT CHECK" and LIMIT paragraphs are kept
  verbatim — those are contract, not transcript.
* **RR-49** — `code_version`'s docstring last paragraph no longer restates the test
  docstring; it points at it, in `STAGE_CODE_SCOPE`'s shape.
* **RR-50** — `_DIAGNOSTICS_EXEMPTION` moved above the `#:` block, which now
  documents the dict it precedes rather than the string constant.
* **RR-51** — the "Non-exempt" token contract, previously discoverable only by
  failing a test, is stated in the table's header.
* **RR-52** — `STAGE_CODE_SCOPE` now says the three absent stages are absent by
  decision, and points at RD-99/RD-100.
* **RR-53** — `TestTheChainReachesTheReportedTABLE` →
  `TestTheTableProducingStagesAreCacheProtected`; it no longer differs from its
  neighbour by a shouted last word.
* **RR-54** — the new banner cut to a bare title, matching the other three.
* **RR-55** — `_check_scalar_domains` and `_check_reserved_split_names` extracted;
  `Config._check` is now eight named calls and nothing else.
* **RR-56** — the ~15 lines in `stamp()` re-telling `provenance.py`'s docstrings cut
  to one line each, pointing at the module that owns the rule.
* **RR-57** — `select_device` annotated `-> "torch.device"` (lazy import preserved,
  verified importable) and the two-way-parsing opener reworded.
* **RR-59** — the module docstring's stale row enumeration replaced with the six
  failure families the file now covers.

---

## acoustics-reviewer pass on `lane/P-cycle4` @ b992a78 (self-check, not a clean pass)

Scope: the DECLARATIONS lane P changed (`_eval_fingerprint`, `STAGE_CODE_SCOPE`,
`STAGE_UPSTREAM`, `FINGERPRINT_EXEMPT_FIELDS`), judged on whether they are
PHYSICALLY sufficient to keep a room-acoustic metric from being served from cache
under changed physics. No files modified.

**Confirmed correct (do not re-litigate):**

* Chain-only coverage IS physically sufficient for `ir_duration`, `n_samples`,
  `ambisonics_order`/`n_channels`, `low_ray_budget`, `high_ray_budget` and
  `sample_rate`. Measured on a complete run_dir: each perturbation gives
  `eval=REFUSED  stats=REFUSED  report=REFUSED`, because
  `Pipeline._effective_fingerprint` (src/amcd/pipeline.py:577-593) recurses the
  whole ancestor chain from the current config. None of these belongs in eval's
  own payload; duplicating them would create a second declaration site that can
  drift from `_render_fingerprint`.
* Sub-sample `ir_duration` aliasing (`n_samples = int(sr*ir_duration)` truncates)
  is covered: 3.0 vs 3.000001 leaves `render`/`preprocess` identical but moves
  `gen-scenes`, and the chain carries it to `eval`.
* `metric_edt_variance_limited_s` in `_eval_fingerprint` ONLY is right and
  complete. `stats` reads the boolean column from metrics.parquet
  (src/amcd/stats/aggregate.py:280-281) and `report` the count from ci_table
  (src/amcd/reporting/tables.py:48-51); neither reads the config key, and both
  refuse through the chain.
* The reported ISO-3382 path is the decoded-waveform path
  (src/amcd/training/infer.py:93-97 → src/amcd/evaluation/evaluator.py:103-119),
  not energy directly, and every module `evaluation/` imports is in eval's scope.
* Both band ladders are config-declared, not hardcoded per call site: the ISO
  octave centers in configs/base.yaml:190 (`iso_eval_freqs`, fingerprinted at
  eval) and the third-octave rep ladder in configs/representations/spectrogram.yaml
  (fingerprinted through `representation.params` at preprocess/train/infer/eval).

```
AC-44 | acoustics-reviewer | major | OPEN | src/amcd/pipeline.py:82-89 (`_render_fingerprint`), :283 (`STAGE_CODE_SCOPE["preprocess"]`), :322-326; src/amcd/evaluation/evaluator.py:104-110 | CONFIRMS RD-99 FROM THE PHYSICS SIDE, AND WIDENS ITS BLAST RADIUS. `simulators/` is in NO stage's code scope (preprocess names only `simulators/base.py`, deliberately excluding backends) and `_render_fingerprint` carries no `code_version`. eval reads `renders/<scene>/high.npy` as the ISO-3382 REFERENCE leg and `preprocessed/carrier/<scene>.npy` (a copy of the render's low leg) as the baseline leg, so an edit to `simulators/dry_run.py` — the module that synthesises the decay whose T30/EDT/C50 ARE the ground truth, and a file the metric-computation lane owns — changes every reported absolute AND every paired improvement while all nine stages print `[skip] (cached)` at exit 0. Because render is the chain anchor, preprocess/train/infer/eval/stats/report are all unprotected against it: this is not "the renders are stale", it is "the reference the improvement is measured against is stale". | Decide RD-99 as a policy call (scope `render` to `simulators/`, accepting a re-render on backend edits), or declare the exemption explicitly with the physical consequence stated. CONFIRMING TEST: in a scratch copy of the package, halve the dry_run synthetic IR's decay constant (T60/2), re-run `amcd all` on an existing run_dir — expect nine `[skip]`, exit 0, and `metrics.parquet` `high_ref` for T30 byte-identical to the old decay's.
AC-45 | acoustics-reviewer | major | OPEN | src/amcd/pipeline.py:337 (`STAGE_FINGERPRINT["diagnostics"] = None`), :400-406 (`_DIAGNOSTICS_EXEMPTION`); src/amcd/diagnostics/probe.py:30-34, :351-353, :386-394 | RD-100 IS FILED AT THE WRONG SEVERITY AND THE EXEMPTION UNDERSTATES THE EXPOSURE. The D0b output is not a threshold report, it is a PHYSICAL VERDICT — "CARRIER CEILING CLEARS … Proceed to E1" vs "CARRIER BOTTLENECK" — produced by comparing measured T30/EDT/C50 residuals against JND tolerances. With no fingerprint, `_is_done` returns True on the bare sentinel, so NOTHING invalidates it. Measured on a complete run_dir: `diagnostics=SKIP(cached)` under doubled `ir_duration`, changed `ambisonics_order`, changed `low_ray_budget`/`high_ray_budget` and changed `sample_rate` — every one of which changes the residuals being compared. `probe.py` also imports `evaluation.room_acoustic` with no `code_version`, so a change to the Schroeder window or the octave filter leaves the verdict standing too. The five `d0b_*`/`d0a_*` exemptions read as "five thresholds at risk"; the true exposure is every acoustic key plus the metric code. A stale CARRIER CEILING CLEARS is a false clearance of the project's own physical premise. | Wire `diagnostics` with `code_version(("diagnostics", "evaluation", "representations", "data"))` + the d0a/d0b keys + `iso_eval_freqs`/`metric_onset_rel_db`/`metric_band_resolvability_margin`/`sample_rate`, chained to `preprocess`. Until then the exemption text should name the full exposure, not only the five fields. CONFIRMING TEST: run `amcd all`, then re-run `amcd diagnostics` with `ir_duration` doubled — `d0b_oracle.json` is unchanged while `config.yaml` stamps the new record length.
AC-46 | acoustics-reviewer | minor | OPEN | src/amcd/pipeline.py:436 (`FINGERPRINT_EXEMPT_FIELDS["d0b_t30_jnd_frac"]`); configs/base.yaml:203-207 | The exemption says `d0b_t30_jnd_frac` is "Consumed only by `diagnostics`". It is also the CALIBRATION CRITERION for `metric_band_resolvability_margin`, which IS an eval fingerprint key: base.yaml states margin 2.0 was chosen because "the 500 Hz T30 estimator's bias … lands where the bias crosses this project's own d0b_t30_jnd_frac of 0.05". No code path, so no cache hole — but moving the JND without re-deriving the margin leaves an eval-fingerprinted constant justified by a number that no longer exists, and nothing in the exemption says so. | Extend the re-entry condition: "…and if this moves, `metric_band_resolvability_margin` must be re-derived (configs/base.yaml:203-207)". No code change.
AC-47 | acoustics-reviewer | minor | OPEN | src/amcd/pipeline.py:291-294 (`STAGE_CODE_SCOPE["eval"]` rationale) | The stated reason is false as written: "`representations` is in scope because eval decodes before measuring". eval does NOT decode — `infer` does (src/amcd/training/infer.py:93-97 writes `<scene>_decoded_ir.npy`), and eval loads that array (evaluator.py:103,108). `evaluation/` imports nothing from `representations`. The SCOPE ENTRY IS FINE AND SHOULD STAY (conservative direction, and the rep is loaded BY NAME through the registry, which provenance.py:110-118 says static analysis cannot see) — the RATIONALE is what is wrong, and F-66 was itself an over-claiming provenance docstring, so this is the same class one cycle later, in the module whose premise is "that claim is auditable". | Reword to the true reason: the decoded waveform eval measures is produced by the representation, which is registry-resolved and therefore invisible to the closure test, so the scope is a declared judgement. Docstring only.
AC-48 | acoustics-reviewer | minor | OPEN | src/amcd/reporting/tables.py:138-139, :48-51, :72-81 | UNIT/REFERENCE DISCLOSURE ON A REPORTED ACOUSTIC QUANTITY. The Caveats legend names `metric_edt_variance_limited_s` symbolically and never renders its value or unit, so a reader of summary.txt sees "3 high-variance" with no way to know whether the bound was 0.15 s or 5.0 s — and F-65's own evidence is that this exact key was served at 0.15 while config.yaml stamped 5.0. The CI label two blocks up (tables.py:84) does render its config value numerically (RR-17's rule), so the file is inconsistent with itself. Secondarily, the `Imp mean` / CI / MDES columns carry no unit while their rows mix seconds (T30, EDT) and dB (C50) — `paired_improvement` returns the metric's own units (evaluation/metric_row.py:119-127). | One f-string: `f"  high-variance: EDT below metric_edt_variance_limited_s = {config.metric_edt_variance_limited_s:g} s, where …"`, and a unit column or per-row unit suffix for the improvement columns. Lane-P-owned file; not part of the declaration work, raised because the reviewer judges CURRENT state.
```
