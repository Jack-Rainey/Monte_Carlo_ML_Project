"""Render-stage quality control: the four Research I admission criteria.

QC decides which rendered scenes are ADMITTED to the dataset, so its thresholds
are config-declared (`configs/base.yaml` §QC thresholds) and they sit in the
render stage's fingerprint — changing one changes the dataset, not just a report.

A scene that fails a GATING criterion is EXCLUDED, per RI §B.4 ("Examples failing
these checks were excluded from the dataset"); it does not discard the batch.
`run_render` records the exclusion in `renders/manifest.json`, which is what
downstream reads as the dataset, and refuses only if attrition breaches the
declared bounds.

Scoring is separated from rendering so it can re-run over persisted artifacts:
changing a threshold re-scores in seconds instead of costing another emulated
render. Nothing here raises — every criterion, pass or fail, comes back as a
record.

Criteria (`docs/research_I_paper.md` §B.4, Figure 5):

  onset_mismatch_ms     pair-level — the same physical arrival must land at a
                        consistent sample location in both legs
  min_energy_db         per leg — W-CHANNEL energy, dB re `min_energy_reference`
  path_file_mb          per leg — retained-path export size
  non_empty_path_file   per leg — the export must carry paths

The two path criteria are keyed on whether the backend EXPORTED paths, never on
which backend it is: a backend with no path export records them skipped with a
reason rather than failing every scene.

THE ENERGY FLOOR IS ON THE W CHANNEL, WHICH IS A DELIBERATE DEPARTURE FROM
RESEARCH I. RI states its floor over the all-channel total. But every reported
ISO-3382 metric reads the W channel alone, so a total-energy floor does not bound
the quantity that decides whether a scene is measurable — and the two are not
convertible by a constant: the theoretical (N+1)² offset under N3D holds only for
a plane wave or a mutually-incoherent field, while the measured offset on real
renders is +3.8 to +8.4 dB and moves with the ray budget on a fixed scene,
because the diffuse contributions share a synthesis carrier and so add
coherently in W while cancelling in the directional channels.

Recorded, never gating:

  total_energy_db       per leg — energy summed over all channels, dB re
                        `min_energy_reference`. What RI's floor was stated over,
                        kept so the two are comparable.

`QCRecord.gating` is what separates the two; nothing branches on a criterion name.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..evaluation.room_acoustic import find_onset

#: Column order of both QC artifacts: `renders/qc_failures.csv` and
#: `renders/qc_record.csv`.
#: `leg` and `ray_budget` are part of the key, not decoration: two criteria are
#: per-leg, `min_energy_db` is budget-dependent by construction (fewer diffuse
#: rays carry less energy), and the ray budget is a swept axis at E4.
QC_COLUMNS = (
    "scene_id", "leg", "ray_budget", "criterion", "gating", "measured",
    "threshold", "passed", "skipped_reason", "adjudication",
)


@dataclass(frozen=True)
class QCRecord:
    """One criterion scored against one unit of one scene.

    `leg` is "low", "high", or "pair" for a criterion over both. `ray_budget` is
    None on a pair-level criterion. `gating` says whether the record bears on
    admission at all.

    THREE STATES, NOT TWO. `passed` is None for a criterion that did not run,
    which is neither a pass nor a failure: `True`/`False` would both assert
    something about a measurement nobody took, and `True` in particular would let
    a skip inflate a "criteria passed" count. `scored` is the predicate every
    consumer should branch on.
    """

    scene_id: str
    leg: str
    ray_budget: int | None
    criterion: str
    #: Whether this record bears on ADMISSION. False for one that discloses a
    #: measurement and nothing more: `qc_failures.csv` carries it for the reader,
    #: and no admission decision reads it. Declared here, at the element, so a
    #: consumer asking "did every gating criterion pass" never has to branch on a
    #: criterion name or on the text of `threshold`.
    gating: bool
    measured: float | str
    threshold: float | str
    #: True, False, or None for a criterion that did not run. A PYTHON bool, never
    #: a numpy one: `np.False_ is False` is False, so a numpy bool here would slip
    #: past the `is False` identity test `failed` uses and never reach the failure
    #: table. `__post_init__` enforces it rather than trusting each call site.
    passed: bool | None
    skipped_reason: str | None = None
    #: Why geometry overruled the onset detector on one or both legs, if it did.
    adjudication: str | None = None

    def __post_init__(self) -> None:
        if self.passed is not None and not isinstance(self.passed, bool):
            object.__setattr__(self, "passed", bool(self.passed))

    @property
    def scored(self) -> bool:
        """Whether this criterion was actually measured."""
        return self.passed is not None

    @property
    def failed(self) -> bool:
        return self.passed is False

    def as_row(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "leg": self.leg,
            "ray_budget": "" if self.ray_budget is None else self.ray_budget,
            "criterion": self.criterion,
            "gating": self.gating,
            "measured": self.measured,
            "threshold": self.threshold,
            "passed": "" if self.passed is None else self.passed,
            "skipped_reason": self.skipped_reason or "",
            "adjudication": self.adjudication or "",
        }


def _energy_db(ir: np.ndarray, reference: float) -> float:
    """Energy of `ir`, in dB re `reference`.

    A silent input is -inf rather than an error: the criterion's whole job is to
    refuse it, and it must reach the failure table as a value.
    """
    energy = float(np.sum(np.asarray(ir, dtype=np.float64) ** 2))
    if energy <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(energy / reference))


def score_scene(
    scene_id: str,
    *,
    legs: dict[str, np.ndarray],        # leg → (C, T) IR
    ray_budgets: dict[str, int],        # leg → the budget it was rendered at
    path_files: dict[str, Path | None],  # leg → its retained-path file, or None
    path_row_counts: dict[str, int | None],  # leg → rows exported, or None
    sample_rate: int,
    onset_rel_db: float,
    #: Per leg, `floor(|src - rcv| / c * fs)` from THAT LEG's own render
    #: provenance, so geometry adjudicates the detector as it does on the reported
    #: metric path. Per leg rather than shared: a shared value would pin both legs
    #: to one position and leave nothing for the criterion to measure, while these
    #: differ exactly when the pair was rendered from different geometry — which is
    #: the defect the criterion is named for. A leg with no geometry makes the
    #: criterion SKIPPED, never scored on the bare detector.
    expected_onset_samples: dict[str, int | None],
    onset_tolerance_samples: int,
    onset_mismatch_tolerance_ms: float,
    min_energy_db: float,
    min_energy_reference: float,
    max_path_file_mb: float,
    require_non_empty_path_file: bool,
) -> list[QCRecord]:
    """Score one scene's rendered pair against every criterion.

    Returns one record per (leg, criterion) — passes, failures and skips alike.
    The caller keeps the failures for `qc_failures.csv` and the whole list for the
    per-scene diagnostics record.
    """
    records: list[QCRecord] = []

    # GEOMETRY ADJUDICATES EACH LEG, as it does on the reported metric path. The
    # bare detector thresholds `onset_rel_db` below that leg's GLOBAL peak, so a
    # small carrier draw at the direct arrival puts it under a bar its own peak
    # set and t=0 lands past the early-reflection gap. Unadjudicated that is an
    # admission decision made on a detector artifact, with a miss rate that rises
    # with distance — the placement split's own axis.
    #
    # What survives adjudication is the defect the criterion is actually for: the
    # two legs' RECORDED geometry disagreeing, i.e. a pair whose halves are not of
    # the same room. A detector miss no longer refuses a batch; a genuinely
    # mismatched pair still does.
    if any(expected_onset_samples.get(leg) is None for leg in legs):
        records.append(QCRecord(
            scene_id=scene_id, leg="pair", ray_budget=None,
            criterion="onset_mismatch_ms", gating=True, measured="n/a",
            threshold=onset_mismatch_tolerance_ms, passed=None,
            skipped_reason=(
                "at least one leg records no source-receiver distance and speed "
                "of sound, so geometry cannot adjudicate the onset detector and "
                "an unadjudicated mismatch is not evidence either way"
            ),
        ))
    else:
        onsets, adjudications = {}, []
        for leg, ir in legs.items():
            onsets[leg], why = find_onset(
                ir[0], onset_rel_db,
                expected_sample=expected_onset_samples[leg],
                tolerance_samples=onset_tolerance_samples,
            )
            if why:
                adjudications.append(f"{leg}: {why}")
        mismatch_ms = 1000.0 * abs(onsets["low"] - onsets["high"]) / sample_rate
        # OVERRULED ON BOTH LEGS MEANS UNSCORED, not passed. When geometry places
        # both onsets, the measured mismatch is the difference between two numbers
        # read from `meta.json` — exactly 0.0 for any pair recording the same
        # distance, however far apart the SIGNALS actually are. RI §B.4 defines
        # this check over the signals, so a 0.0 from metadata is not evidence that
        # it holds; recording it as a pass would admit a genuinely misaligned pair.
        both_overruled = len(adjudications) == len(legs)
        records.append(QCRecord(
            scene_id=scene_id, leg="pair", ray_budget=None,
            criterion="onset_mismatch_ms", gating=True,
            measured=round(mismatch_ms, 6), threshold=onset_mismatch_tolerance_ms,
            passed=None if both_overruled
            else mismatch_ms <= onset_mismatch_tolerance_ms,
            skipped_reason=(
                "geometry overruled the onset detector on BOTH legs, so the "
                "measured mismatch is a difference of recorded distances and "
                "carries no information about the signals"
                if both_overruled else None
            ),
            # Recorded, not swallowed: a pair whose detector was overruled on one
            # leg is still scored, and the reader is owed which leg.
            adjudication="; ".join(adjudications) or None,
        ))

    for leg, ir in legs.items():
        budget = ray_budgets[leg]
        # THE GATE IS THE W CHANNEL — the one every reported ISO metric reads, so
        # the floor bounds the quantity that decides measurability. See the module
        # docstring for why this departs from RI's all-channel total and why the
        # two cannot be converted by a constant on this backend.
        w_db = _energy_db(ir[0], min_energy_reference)
        records.append(QCRecord(
            scene_id=scene_id, leg=leg, ray_budget=budget,
            criterion="min_energy_db", gating=True,
            measured=round(w_db, 6) if np.isfinite(w_db) else w_db,
            threshold=min_energy_db,
            passed=w_db >= min_energy_db,
        ))
        # Recorded, never gating: what RI's floor was stated over, so the
        # comparison against RI remains checkable.
        total_db = _energy_db(ir, min_energy_reference)
        records.append(QCRecord(
            scene_id=scene_id, leg=leg, ray_budget=budget,
            criterion="total_energy_db", gating=False,
            measured=round(total_db, 6) if np.isfinite(total_db) else total_db,
            threshold="n/a", passed=None,
            skipped_reason=(
                "not an admission criterion: recorded because RI states its energy "
                "floor over the all-channel total, which this study gates on the W "
                "channel instead"
            ),
        ))

        path_file = path_files.get(leg)
        if path_file is None:
            skip = (
                "the backend exported no retained paths for this leg, so there is "
                "no file to bound"
            )
            for criterion, threshold in (
                ("path_file_mb", max_path_file_mb),
                ("non_empty_path_file", require_non_empty_path_file),
            ):
                records.append(QCRecord(
                    scene_id=scene_id, leg=leg, ray_budget=budget,
                    criterion=criterion, gating=True, measured="n/a",
                    threshold=threshold, passed=None, skipped_reason=skip,
                ))
            continue

        size_mb = path_file.stat().st_size / (1024.0 * 1024.0)
        records.append(QCRecord(
            scene_id=scene_id, leg=leg, ray_budget=budget,
            criterion="path_file_mb", gating=True,
            measured=round(size_mb, 6), threshold=max_path_file_mb,
            passed=size_mb <= max_path_file_mb,
        ))

        n_rows = path_row_counts.get(leg)
        records.append(QCRecord(
            scene_id=scene_id, leg=leg, ray_budget=budget,
            criterion="non_empty_path_file", gating=True,
            measured=n_rows if n_rows is not None else "unknown",
            threshold=require_non_empty_path_file,
            passed=(not require_non_empty_path_file) or bool(n_rows),
        ))

    return records


def write_qc_csv(path: Path, records: list[QCRecord]) -> None:
    """Write `records` to `path` in `QC_COLUMNS` order, always — including empty.

    Two callers, on opposite sides of the save gate. `renders/qc_failures.csv`
    holds the failures AND the skips, and is canonical at every save level: it is
    the evidence behind the stage's refusal, and what was not scored is evidence
    for an admission decision at the same level as what failed. A raise whose
    evidence file is suppressed at the default save level is not a reportable
    result. `renders/qc_record.csv` holds every criterion, scored or not, and is
    observability — gated at save level 4 (docs/verbosity.md).

    Written even when empty, so "no failures" is a recorded fact rather than a
    missing file.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QC_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
