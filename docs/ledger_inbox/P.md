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
rather than deferred. `falsifier`, `acoustics-reviewer` and `readability-reviewer`
have **not** been run on this branch — and a lane-branch review would be a
self-check regardless (rule 5). The reviewer set that counts is the integrator's,
over the merged tree.
