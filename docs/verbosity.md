# Run-output verbosity (`--save-verbosity` / `--show-verbosity`)

Two runtime output knobs, each `0–5` (`click.IntRange` — out-of-range is a
usage error, never a silent clamp), available on every stage command and
`all`:

- `--save-verbosity` — how much a run **writes to disk**.
- `--show-verbosity` — how much a run **prints live**.

They set only how much a run writes and prints, **never what it produces**:
canonical results, inter-stage inputs, and stage sentinels are written at
every save level, including 0, and results are bit-identical across all
save/show levels (ledger F-23; enforced by `tests/test_verbosity_gate.py`).
Verbosity is a runtime output level, not an experiment-governing value, so it
lives in the CLI layer — its defaults there are the one sanctioned exception
to no-hidden-defaults, and `Config` never carries it (RD-09; CLAUDE.md
"Output verbosity is not an experiment value"). It is threaded explicitly as
a frozen `Verbosity(save, show)` from `cli.py` through `Pipeline` into every
stage function (F-22) — never a module global.

## The category ladder

Both axes share one monotonic ladder (`amcd/runtime.py:CATEGORY_LEVELS`):
each level adds one category on top of everything below it, so level *n+1*
always emits/saves a superset of level *n*.

| level | category | save axis writes | show axis prints |
|---|---|---|---|
| 0 | — | canonical results only | warnings/errors only (stderr) |
| 1 | provenance / timing | config snapshot + resolved seeds + `versions.json` (git SHA), `timings.json`, report bundle copies | run identity (`Run dir:`), per-stage `[done] (Xs)` durations |
| 2 | progress | *(no artifacts today)* | `[run ]`/`[skip]`, per-stage counts/summaries, epoch progress |
| 3 | metrics | `checkpoints/train_log.csv` | intermediate metrics: tensor shapes, device, best loss, eval headline, stats/report echo, D0a/D0b tables |
| 4 | diagnostics | *(no artifacts today — Step 4's per-criterion render QC record is the next)* | *(no sites today — reserved)* |
| 5 | visual | *(reserved)* | *(reserved — see below)* |

Naming note: the `diagnostics` *category* (level 4, per-unit QC observability) is
unrelated to the `diagnostics` *stage*, whose D0a/D0b tables print at `metrics`
and whose JSON outputs are canonical.

`renders/<id>/meta.json` used to be the category's only artifact. It is now
**canonical, written at every save level** (RD-16): it is the sole record of how
an expensive dataset was made — installed simulator version, both ray budgets,
declared speed of sound, ambisonic convention — and gating it meant the default
`save=1` run produced a dataset nobody could later characterize. Diagnostic
*extras* may still attach behind the gate; none exist yet.

Outside the ladder entirely: **warnings and errors always emit, to stderr,
at every show level** (F-24 — a suppressed fatal error is never acceptable).

**Defaults: `--save-verbosity 1 --show-verbosity 1`.** Level 1 is the
provenance rung, so a bare invocation is non-blocking yet never lacks
reproducibility metadata; `save=0` deliberately omits provenance and is never
a default (RD-09). Quiet levels still satisfy design_spec §9's git-SHA bundle
requirement at every default — only an explicit `save=0` opts out.

**Level 5 (`visual`)** is the reserved slot for the roadmap §6 Blender
authoring/preview front-end (ledger RD-10 — a deliberate forward-looking
seam, not scope creep; do not strip). Nothing emits it today. Its TTY guard
already exists (F-24): live `visual` output requires `show>=5` **and** an
interactive stdout, so a headless run can never block on a preview dismiss —
it degrades to the save axis (render-and-save, or skip).

## Per-stage wiring table

Every stage is wired through the single gated helper
`amcd.runtime.emit(verbosity, category, msg)` (RR-19 — no scattered
`if show >= n: print`). A stage or site **absent from this table is declared
unwired** — treated as not yet honoring verbosity, never as silently done.
As of 2026-07-11 all nine stages plus the CLI/Pipeline shell are wired; the
table is total.

| unit | show sites (category) | save sites (category) |
|---|---|---|
| `cli.py` | `Run dir:` (timing) | `Config.stamp` trio (provenance) |
| `pipeline.py` | `[run ]`/`[skip]` (progress); `[done] (Xs)` (timing); `[FAIL]` (error → stderr, always) | `timings.json` (provenance); `stages/*.done` sentinels **never gated** |
| gen-scenes | generated-count summary (progress) | — (scene specs canonical) |
| render | rendered-count summary (progress) | — (IR pair **and** `renders/<id>/meta.json` provenance are canonical, RD-16) |
| preprocess | count summary (progress); tensor shape (metrics); empty-split `WARNING` (warning → stderr, always) | — (all outputs canonical) |
| diagnostics | D0a + D0b tables and verdicts (metrics) | — (`d0a_gap.json`/`d0b_oracle.json` are the stage's results: canonical) |
| train | epoch lines, early-stop (progress); device, best valid loss (metrics) | `train_log.csv` (metrics); `best.pt` canonical |
| infer | saved-count summary (progress) | — (predictions canonical) |
| eval | scored headline, drop count (metrics) | — (`metrics.parquet`, `drops.csv` canonical — drop logging is invariant-mandated, never gated) |
| stats | summary line (metrics) | — (`ci_table.csv`, `summary.json` canonical) |
| report | full table echo, written-path (metrics) | bundle copies of `config.yaml`/`versions.json` (provenance — same gate as their source); `summary.txt`/`metrics_table.csv` canonical |

Classification rule for new artifacts (F-23): if anything downstream — a
later stage, the stats/report spine, or the audit trail — reads it, it is
canonical and writes at every level. Only pure observability may sit behind
`Verbosity.saves()`, and it must appear in this table when added.
