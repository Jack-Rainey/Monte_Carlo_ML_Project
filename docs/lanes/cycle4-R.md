# Lane R brief — render enablement, Steps 2+3 (cycle 4)

`src/amcd/simulators/gsound_sir.py:124` raising `NotImplementedError` is the
binding constraint on this project. It is yours.

**Be precise about what this cycle does to the gate.** RD-33a lifts on **both**
(i) zero OPEN rows on its declared path list and (ii) the RD-17 probe validating
that the 200,000-ray high leg is a converged reference. Lanes M and S work (i).
This lane **unblocks** (ii) — RD-17 cannot run until Steps 2/3 exist — but it
does **not lift** it, because RD-17 is assigned to no lane this cycle. Cycle 4
removes the blocker; cycle 5 runs the probe (RD-89c).

## Step 0 — the worker itself, before any assigned row

`GsoundSirSimulator.render` must return an `IRResult` through `build_simulator`.
RD-24, RD-08, RD-67 and RD-21 are **requirements ON that render, not substitutes
for it.** A pass condition satisfiable with synthetic PathData and a mocked
gsound leg, while `render()` still raises, would miss this cycle's whole point —
the lane that exists to move the gate would have a pass condition not requiring
the thing that moves it (RD-89).

The subprocess worker behind the existing seam is the deliverable. It had no
ledger row of its own, which is exactly how it nearly went unbuilt.

## Your rows were re-statused to OPEN before you started

RD-08, RD-24, RD-67 and RD-21 read `DEFERRED (gate: Step 2 / Step 3)` until
2026-08-10. Cycle 4 **is** Steps 2+3, so the integrator re-statused them to OPEN
before any worktree existed — a row deferred to a gate that has arrived is not
deferred, and leaving them DEFERRED would have let the cycle report "done" with
all four untouched and `render()` still raising (RD-90).

They are now inside the definition of done. Do not re-status anything yourself —
the ownership hook will refuse the write; record status changes in
`docs/ledger_inbox/R.md` (rule 3).

## Order

1. **RD-24 — the PathData schema, first and on its own.** Everything else in this
   lane consumes it. The row's requirement is that the parquet be
   **self-describing**: band edges/centres, `num_bands`, simulator name and
   commit SHA in the file's own metadata. The reason is concrete — `intensities
   (N,8)` has band meaning that today lives only in the simulator config, so a
   path file from a second raytracer would be uninterpretable without it. The
   roadmap wants multiple raytracers, so this is the seam, not speculation.
2. **RD-08 — populate `IRResult.paths`.** Spec §8 shows `IRResult` carrying
   `paths: PathData` and the code omits it. This is the *producer* half of the
   path-conditioned-variant seam. Its row is explicit that the field must not be
   added speculatively before the producer exists — now the producer exists, so
   add both together.
3. **RD-67 — the provenance fill**, which consolidates RD-16/19/21/12. On the
   gsound leg, `IRResult.meta` must carry: installed upstream SHA **verified
   equal** to the config-pinned `commit_sha`; diffuse AND specular counts;
   `ambisonic_convention: "acn_n3d"` (AC-15 established N3D, not SN3D — check
   `binding.cpp:18,:43` before stamping); native IR length + `truncated` bool +
   discarded-tail energy in dB against a config-declared QC threshold; and a
   `PathData.speeds_of_sound` cross-check against the declared
   `speed_of_sound_m_s`.
4. **RD-21** is the truncation half of the above and lands with it.

## Build against the seam

Everything goes through `build_simulator`, never pygsound directly. That is what
keeps the eventual RD-17 probe an engineering-feasibility artifact rather than an
accidental result — and it is a binding condition of your render permission,
stated in full under Pass condition below.

## Pass condition

Steps 2+3 are structural, so most of the evidence is a shape, not a number:

- the canonical dry run still passes end to end (the scaffold must be unaffected —
  it shares `simulators/base.py` with you);
- a `PathData` parquet written and read back with its self-describing metadata
  intact, asserted in `tests/test_simulator_seam.py`;
- `IRResult.paths` populated on the gsound leg and `None`/absent on the scaffold
  leg, with no downstream edit needed to tolerate either — that is the
  scaffolding rule, and a downstream `isinstance` check would be a defect.

**Plus the step-0 condition, which is the one that matters:** `render()` returns
a real `IRResult` from a **1-scene smoke render**, and you are permitted to run
it — see below. Do not re-open this mid-lane; it is already decided.

## Your render permission, and its exact edges

RD-17 carries a standing ≤4-scene render permission that is **not** gated on
RD-33a. Cycle 4 sub-grants **exactly ONE** of those scenes to this lane, for a
smoke render whose only purpose is to show the worker returns a real `IRResult`.

Binding bounds — all of them:

- **one scene**, into a **throwaway `run_dir`**;
- through **`build_simulator`**, never pygsound directly;
- **not** a dataset render, **not** an E4 result, and it produces **no reported
  number** — nothing from it may reach `ci_table.csv` or an E1/E4 claim;
- it does **not** lift RD-33a, and you must not describe it as having done so;
- the remaining ≤3 scenes belong to the Step-6 probe (RD-17), which is **not
  yours** this cycle.

**If you want more than one scene, or any render beyond this smoke test: STOP and
write it in your inbox.** The grant does not stretch, and a lane that quietly
widens it is the failure the whole ownership scheme exists to prevent.

**No 720-scene render under any circumstances.**

## Evidence

```
PYTHONPATH=<your-worktree>/src <env>/bin/pytest
PYTHONPATH=<your-worktree>/src <env>/bin/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

The prefix is mandatory — see `LANE.md`.

## Not yours

You own `simulators/base.py`, `render.py`, `gsound_sir.py`, `runtime.py`,
`configs/simulators/**` and four test files. `simulators/dry_run.py` is lane
M's — the scaffold's acoustics are being changed there this cycle, so if you need
a change in it, write it to `docs/ledger_inbox/R.md`.

Two rows on the integrator's queue touch your files: **RR-27** (stale design-spec
line citations in `simulators/base.py` and `config.py`) and **RD-20** (the
`RunContext` dispatch change, which spans `pipeline.py`, your `runtime.py` and
`cli.py`). Do not start either — RD-20 in particular is the row whose whole point
is to be done once, while the dispatch signature is already open.
