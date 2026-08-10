# Lane P brief — provenance and the stage cache (cycle 4)

Your four majors are one defect with four routes: **a reported number is
reachable through a cached stage.** Cycle 3 claimed the stage cache now protects
the reported result; that is true for train/infer/eval and false for the stages
that actually produce the table. All four were reproduced end to end with exit 0.

Row text is in `docs/review_ledger.md` — read it there. You may read the ledger;
the ownership hook will refuse a write to it.

## Order

1. **F-64 first.** `preprocess` declares no `code_version`, so the encoded
   dataset, the normalization stats and the **leakage-critical split assignment**
   are all served under changed code — and the refusal message names the wrong
   stage, so following its instruction (`--force` on `train`) completes with exit
   0 and a full report over a stale encoding. It is first because it is the
   furthest upstream and the only one that touches split assignment.
   Fix: `code_version` with scope `("data", "representations")`.
2. **F-63.** `stats` carries no `code_version` and `report` no fingerprint at
   all. Give `stats` scope `("stats", "evaluation")` — it imports
   `evaluation.metric_row.paired_improvement` — and give `report` a fingerprint
   over `{report_format, code_version(("reporting",))}`. RD-41's "terminal, only
   its own re-use" rationale is wrong for `report`: its own re-use *is*
   `report/summary.txt`.
3. **F-66.** `eval` and `infer` both call `data.normalization.denormalize` on
   every reported leg and neither declares `data`. Masked today only because
   `data` is in train's scope and the chain refuses upstream first — a
   coincidence. Add it. Then fix the docstring: `provenance.code_version` claims
   `tests/test_stage_cache.py` asserts each declared scope against the modules
   the stage imports, and it does not — it asserts only that the stage's own
   entry-point subpackage is in scope. **Make the claim true rather than
   deleting it**: assert the declared scope is a superset of the stage's
   transitive `amcd.*` import closure minus `_CORE_SOURCES`.
4. **F-65, and then its class-level guard.** `metric_edt_variance_limited_s` — a
   key added in cycle 3 to fix RD-78 — is in no fingerprint, so its disclosure
   column is served under the wrong threshold's provenance stamp. Add it to
   `_eval_fingerprint`. Then add the test the class needs: **every `Config` field
   is either in some stage fingerprint or in an explicitly declared exempt
   list.** That test is the real deliverable of this row; without it the next key
   repeats the failure.
5. **F-69.** The cache key hashes macOS AppleDouble `._*.py` sidecars — 46 of them
   under `src/amcd/` on this exFAT volume — so the same source yields a different
   key on the project's declared second host. Filter `._` and `__pycache__`, or
   hash only git-tracked files.
6. **F-73** (`_CONFIGS_DIR` assumes a source checkout, so a wheel install cannot
   find `base.yaml`) and **F-74** (the compute device is auto-selected, never
   recorded, never fingerprinted — stamp `device` and `platform.machine()` into
   `versions.json` as HUMAN provenance, *not* a cache key: a device change must
   not invalidate an expensive artifact).
7. **RR-35, RR-36, RR-42** last. RR-36 is the largest: four docstrings carry full
   reproduction transcripts, F-53 is narrated twice, and "a config fingerprint is
   blind to code changes" is stated four times. State it ONCE at
   `STAGE_CODE_SCOPE` and have the others cite it. Doing these after 1-6 means
   you are not tidying prose you are about to rewrite.

## Pass condition

Each of F-63/64/65/66 has a **stated reproduction in its ledger row**. Reproduce
it on the current code, apply the fix, then show the same command now refusing
the right stage by name. A refusal that names the wrong stage is the F-64 defect,
not a fix — check the message, not just the exit code.

Do not skip the counter-check: a guard that refuses a stage for visibly
irrelevant reasons teaches the operator to `--force`, which is why the scopes are
per-stage rather than a whole-package hash. Show that an unrelated edit still
leaves your stage cached.

## Evidence

```
PYTHONPATH=<your-worktree>/src <env>/bin/pytest
PYTHONPATH=<your-worktree>/src <env>/bin/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

The prefix is mandatory — see `LANE.md`.

## Not yours

You own `pipeline.py`, `provenance.py`, `config.py`, `data/`, `stats/`,
`reporting/`, `training/`. You do **not** own `evaluation/` or
`representations/` — lane M is changing both, including `stats`' upstream. Your
fingerprint changes must not depend on M's edits landing; declare scopes by
module name, which is stable across what M is doing.

RR-27 (stale design-spec line citations in `config.py` **and**
`simulators/base.py`) spans lane S's files and is on the integrator's queue.
New work in another lane's file goes to `docs/ledger_inbox/P.md`.
