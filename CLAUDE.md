<!--
  This file replaces the operating-rules-through-review-agents portion of your
  existing CLAUDE.md. Keep your project overview, docs/design_spec.md and
  docs/research_I_paper.md pointers, and invariants top-matter above this block.
-->

## Operating rules

- **Read `LANE.md` at the repo root before anything else.** It declares which
  parallel lane this session is and which files it may edit. If it is absent, you
  are the integrator and own everything. See Parallel lanes below.
- **Plan Mode** for any stage implementation or methodology change. Every plan
  must cite the `docs/design_spec.md` section and the invariants it touches, and
  name which files/functions change, in what order.
- **Evidence before claims.** Never report "tests pass", "baseline beaten", or
  "metric improved" without showing the command run and its actual output — for
  results, the `stats` output with CIs, per test split. The builder does not
  assert success; it shows it. And YOU run the command: do not hand the user a
  command and assume its result (see Evidence and Bash below).
- **One hypothesis = one experiment.** Prove plumbing with the canonical dry run
  (`-c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c
  configs/overlays/dry_run.yaml`) before any real render/train. Smallest change
  that tests one thing.
- **Check invariants first** when changing methodology.

## Review agents (in `.claude/agents/`; invoke by name)

The builder does not grade its own work. Four specialists, four different risks —
call them explicitly by name (auto-delegation is unreliable):

- **`research-director`** — strategic alignment: is this advancing the objective,
  on the current ledger gate, free of GENUINE scope creep (as opposed to
  forward-looking design that serves the roadmap). Runs on a PLAN, before
  implementation.
- **`acoustics-reviewer`** — domain-physics correctness (ambisonic conventions,
  band decomposition, Schroeder/EDR, ISO-3382 metrics). After changing
  `representations/`, `metrics/`, or eval code.
- **`falsifier`** — ML-methodology correctness (leakage, normalization, stats,
  degenerate collapse, loss inertness, config/seed discipline, scaffold
  coupling). After any new result.
- **`readability-reviewer`** — human- and LLM-comprehensibility only (README,
  naming, docstrings, paper↔code traceability, comment clutter). NOT correctness
  or scope.

`research-director` checks the right thing is being built; the other three check
it is built right, in three non-overlapping senses.

## Review ledger and definition of done

`docs/review_ledger.md` holds ONLY unresolved findings — it is working memory for
the loop, not an audit log. Resolved findings are DELETED, never marked resolved:
the git history of the file is the audit trail, so nothing is lost and stale rows
do not accumulate to confuse later sessions.

One row per finding:
`ID | agent | severity (blocker|major|minor) | status | anchor | finding | resolution`

- Status is exactly one of two values:
  - OPEN — not yet resolved. A fix applied but not yet re-review-confirmed stays
    OPEN; note "fix applied, awaiting re-review" in `resolution`.
    **"Awaiting re-review" lasts ONE review pass, not one epoch.** The pass that
    follows a fix MUST re-derive it and return CONFIRMED / NOT FIXED / REFUTED —
    you do not wait for some later cycle to notice. Cycle 4 ended with 34 such
    rows, some carried since cycle 3, and when they were finally checked several
    were not fixed and one was refuted outright. An unverified fix is a claim,
    and this project does not run on claims.
  - DEFERRED — intentionally out of scope for the current gate, with a one-line
    reason and the gate it belongs to. This is the live backlog.
- There is NO ADDRESSED/RESOLVED status. The moment a finding is fixed AND
  re-review-confirmed clean, delete its row.
- Keep prose to a minimum: the DEFERRED backlog and a short "resume here" pointer
  are the only durable content. Do NOT accumulate per-pass re-review write-ups
  here — git log and the resume note cover that.

Definition of done: complete only after a full plan → implement → review cycle in
which every invoked reviewer returns zero new findings AND the ledger has zero
OPEN rows (DEFERRED backlog may remain) AND **zero rows sit in "fix applied,
awaiting re-review"** — that last clause because a fix nobody re-derived is a
claim, and a backlog of them makes "zero new findings" look reachable while the
unverified pile grows behind it. Because resolved rows are deleted, "zero OPEN
rows" is now literally a near-empty ledger.

## Implementation loop

Repeat until a clean pass: plan (Plan Mode) → implement → invoke reviewers by
name → write findings to the ledger → address OPEN findings → repeat.

- **A reviewer pass has TWO jobs: find new defects, AND verify the "fix applied,
  awaiting re-review" backlog.** Send each reviewer its own such rows and require
  a per-row verdict. Only then can a row be deleted. A pass that only looks for
  new things guarantees the ledger grows monotonically.
- **A pass whose findings you then fixed is NOT the last pass — re-run it.** The
  loop ends on a pass over the tree that SHIPS, returning zero new findings; it
  does not end on a pass followed by a fix phase. This binds the integrator
  exactly as it binds a lane. Cycle 5 is the demonstration: all four lanes ran
  their reviewers once, fixed what those reviewers raised, and reported, so ~60
  fixes reached integration as claims nobody had re-derived — and when they were
  finally checked, several were not fixed and two lanes had regressed their own
  earlier work. Reviewing tree X and shipping tree Y is not reviewing.
- **Deduplicate before you count.** Running N reviewers over M lanes produces
  N×M passes, and the same defect surfaces repeatedly from different angles.
  Keep every row — a second independent raiser is corroboration, and in cycle 4
  three agents derived one blocker separately — but group them and record which
  rows are ONE defect that must close together. **A row count is not a work
  estimate**: cycle 4's 244 OPEN rows were ~30-40 independent work items, and
  reporting the raw number misleads about both progress and remaining effort.
- One loop is never assumed sufficient. Keep looping while any OPEN finding
  exists and budget remains.
- A clean pass = reviewers run over the CURRENT state and raise zero new
  findings. Only a clean pass ends the loop.
- Never end the loop because you are near a token/time limit — use the Stopping
  rule instead.

## Parallel lanes

Several sessions may work this repo at once, one per git worktree. **Full protocol
and all six rules: `docs/parallel_protocol.md`** — that file owns them, and a
second copy here is what drifts (it already has, three times, in this exact
machinery). What a session must know before reading anything else:

1. **Ownership is by file and exclusive**, so textual merge conflicts are
   impossible; `scripts/lane_guard.py` enforces it, not discipline.
2. **Lanes never edit `docs/review_ledger.md`, `CLAUDE.md`, `docs/design_spec.md`**
   — one writer, the integrator. Lanes write to `docs/ledger_inbox/<lane>.md`.
3. **Reviewers count only on the integrated tree.** A lane-branch review is a
   self-check, never a clean pass. That bounds what a lane review can BUY; what a
   lane OWES is the **lane exit gate**, and the two are not the same rule.
4. **Allocate new row ids only from your lane's declared `id_block`.** Cycle 4 had
   no such rule and four lanes collided four ways, including with live rows.
5. **A lane is not done until all four reviewers have run and the LAST pass came
   back clean over its FINAL commit**, and until every finding it could fix inside
   its own owned files is fixed. Reporting a reachable finding unfixed needs a
   named blocking file outside `owns`, or a named cluster — there is no third
   reason. The six conditions and their evidence are the exit gate in the
   protocol; `LANE.md` carries them and `tests/test_lane_exit.py` checks them.
6. **A cycle must be able to move the gate.** The partition declares `gate:` with
   `lifts:` / `unblocks:`, or an explicit `exception:`. Cycle 5 ran four lanes to
   completion and every one reported, unprompted, that it moved neither RD-33a
   condition — while the on-path blocker/major count rose 20 → 29. Local lane
   progress is not project progress, and discovering that afterwards is the
   failure; choosing a backlog-discharge cycle deliberately is not.

The rest — rule 2's metric-path file list and the reported-column span (RD-82),
rule 4's spanning rows, and the seven-step integration gate — is in the protocol.

In a lane, prefix every command with `PYTHONPATH=<worktree>/src`: the editable
install pins the MAIN checkout, so a bare `pytest`/`amcd` measures the wrong
tree. `tests/test_source_tree_isolation.py` fails if you forget.

## Reporting and stopping

- Report findings quantitatively: "resolved X of Y falsifier findings; Z OPEN."
  Never claim a reviewer's concerns are handled while any of its findings are
  OPEN, and never present a partial fix as complete.
- If you stop before a clean pass (token limit, time, or user pause): write every
  OPEN finding plus a short "resume here" note to `docs/review_ledger.md`, then
  state plainly "NOT complete — N open findings remain" and list them. Do not say
  "done" or "wrapped up."

## Resuming work

A request to "pick up where I left off", "check everything is working",
"continue", or similar is a request to RE-VERIFY, not to trust prior state.

- Reviewers validate the CURRENT state of the code, never just this session's
  diff. The absence of changes to a file is NOT evidence it is correct.
- On every resume, read `docs/review_ledger.md` first to recover OPEN findings,
  then run the FULL relevant reviewer set over the current codebase — including
  reviewers for code you did not touch this session.
- A continuation cue ('continue', 'pick up where I left off') never licenses skipping Plan Mode or re-verify; resume starts at the top of the loop (plan), never mid-loop."

## Evidence and Bash

Evidence-before-claims means YOU run the command and show the output. The project
allow-list (`.claude/settings.json`) pre-approves the safe, repeated commands
(running the pipeline, inspecting rows, tests, read-only git). Only ask the user
to run something if it needs a privilege the allow-list deliberately withholds.

- **Prefer uncompounded commands.** A compound line (`;`, `&&`, `|`) prompts
  whenever ANY sub-command is unapproved, even if each piece would pass alone —
  so split it into single-tool calls that each match an allow-list entry, and
  batch the independent ones in a single message. Compound only when the pipe is
  essential to the result or the whole line is already pre-approved.

## Parameters and configuration

- **No hidden defaults.** Every value that governs an experiment — counts, seeds,
  splits, model dimensions, ray counts, thresholds, tuning ranges, search
  strategy — is set by a CLI argument or a config file, or the code raises an
  error. There is no fourth option. Do not bury experiment-governing values as
  literals in .py files, test fixtures (`conftest.py`), or comments.
- **Output verbosity is not an experiment value.** `--save-verbosity` and
  `--show-verbosity` set only how much a run writes to disk and prints live —
  runtime output levels, never experiment-governing values — so they carry CLI
  defaults (the one sanctioned exception here; a bare invocation must work).
  This does not weaken the rule above: they live in the CLI layer, never the
  config. `--save-verbosity` gates observability artifacts only — canonical
  results, inter-stage data, and stage sentinels are written at every level, and
  no verbosity level may alter what a run produces (ladder + per-stage wiring:
  `docs/verbosity.md`).
- **Per-aspect seeds.** Each stochastic stage (scene generation, per-split
  sampling, model init, shuffling, augmentation) draws from its own named seed.
  Splits in particular must not share a seed.
- **Config, not Python, expresses experiment structure:** which splits exist, how
  many scenes per split, what each contains. Split naming is a convention read
  from config, not hardcoded: `train` / `valid` / `test_id` are the canonical
  training / validation / in-distribution-test sets; every other split is a
  special test set; the number of splits is variable.

## Explicit contracts, no silent exclusion
- When uniform logic spans heterogeneous elements (metrics, splits, models,
  simulators), each element declares the property the logic relies on — never
  assume one that only happens to hold for most (e.g. a metric's improvement
  `kind`). Reviewers check this.
- Nothing leaves a result silently: log every drop/skip/NaN as `(unit, reason)`
  and show scored-vs-attempted counts; never render an unscored quantity as a
  number.

## Cross-platform portability (project requirement)

The pipeline — including the real `gsound_sir` render backend — must run both on
this macOS/Apple-Silicon machine (where the x86-only render step is emulated) and
on a native x86_64 Ubuntu or Windows desktop, from the same code with no edits.
How each host is set up: `docs/gsound_sir_setup.md`.

Rules that follow:

- **Emulation is a host detail, never a code assumption.** No `platform`/arch/
  Rosetta checks, no Darwin-only paths, and no macOS-specific logic inside
  `src/amcd/`. Anything platform-specific (env creation, `arch -x86_64`,
  interpreter paths) lives in documentation and environment setup — behind the
  simulator seam, outside the Python package.
- **Portable path/device handling.** Use `pathlib` (no hardcoded `/Volumes/...`
  or POSIX-only strings in package code); torch device selection must fall back
  cleanly (MPS → CUDA → CPU) rather than assume MPS exists.
- **GSound-SIR is pulled from upstream GitHub at a version pinned by commit SHA
  in config** — never vendored with local modifications. Repository URL and
  installer: `docs/gsound_sir_setup.md`.
- Reviewers treat a platform-coupled branch or hardcoded host path in package
  code as a defect (same class as scaffold coupling).

## Scaffolding and the end goal

Temporary components (e.g. `DryRunSimulator`) exist only behind the same
interface the real component will implement. Downstream code depends on the
interface, never on the scaffold concretely — no `isinstance(sim, DryRunSimulator)`,
no dry-run-keyed branches. Swapping in the real simulator must be a drop-in with
no downstream edits, and deleting the scaffold must leave no dead references.

## Forward-looking abstractions (unmentioned ≠ unwanted)

The absence of an explicit instruction for an abstraction is NOT evidence it is
unwanted. This project is deliberately built toward the paper's future-work
roadmap (`docs/research_I_paper.md` §6), so a generalization, interface seam,
config key, or unused-looking parameter may be a deliberate provision for it.

- Before treating a construct as premature abstraction, scope creep, or clutter,
  check whether it plausibly serves a documented roadmap item. If it does, it is
  forward-looking design, not a defect — at most, document the roadmap item it
  serves.
- This is the correct reading of "resist premature abstraction": a seam a known,
  coming component will fill is not premature — it is the alternative to
  hardcoding. Premature abstraction is a seam nothing on the roadmap will fill.
- `research-director` arbitrates whether a forward-looking abstraction is
  warranted; `readability-reviewer` asks only that its purpose be documented, not
  removed; `falsifier` / `acoustics-reviewer` judge only whether it is correct,
  not whether it is too general.
