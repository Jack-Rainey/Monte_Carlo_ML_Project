"""report stage: format summary table + supplementary bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import Config
from ..runtime import Verbosity, emit


#: Metric → the unit `paired_improvement` returns it in (evaluation/metric_row.py).
#: A unit cannot come from `kind`: `T30` and `C50` are both `match_reference` and
#: differ in unit. Unlisted → `_unit_for` raises. The durable form declares `unit`
#: beside `kind` on the metric itself and carries it into `ci_table.csv` (RD-201).
#: The one unit a producer CANNOT state, because it depends on what preprocess
#: encoded in. A metric declares this token and the reporting layer resolves it
#: from the stamped `value_domain`.
_OPERAND_DOMAIN_SQUARED_TOKEN = "operand_domain_squared"

#: `value_domain` → how to render an operand-domain-squared unit. The vocabulary
#: is declared on the representation (`representations/base.py`) and stamped by
#: preprocess; adding a domain there means adding it here, or `_unit_for` refuses
#: the run rather than guessing.
#:
#: The amplitude domain is raw ambisonic samples in arbitrary units — the package
#: declares no unit for them — so it renders as `a.u.²` rather than an invented
#: one. (`amp²` was rejected: it reads as ampere-squared, AC-125.)
_DOMAIN_UNITS = {"db": "dB²", "amplitude": "a.u.²"}


def _unit_for(metric: str, declared: str, value_domain: str) -> str:
    """Render `metric`'s declared unit, or raise naming the metric.

    `declared` is what the PRODUCER stated on its own `MetricTriple` and carried
    through `metrics.parquet`. This layer used to hold a `_METRIC_UNITS` map of its
    own — a second declaration asserting what another module's numbers mean, with
    nothing binding the two, so a metric could change domain and the Unit column
    would keep printing the old one (RD-201/AC-127). The only thing resolved here
    is the operand-domain case, because the domain is a PREPROCESS stamp that the
    producer of the metric cannot see.

    `value_domain` is that stamp, never inferred from a representation class — the
    same rule `evaluation/signal.py` states for the metrics themselves (F-19).
    """
    if not declared:
        raise ValueError(
            f"Metric {metric!r} reaches the report with no declared unit. Its "
            f"improvement mean, CI and MDES are rendered as bare numbers beside "
            f"metrics measured in seconds and in decibels, so a reader cannot tell "
            f"what it is. Declare `unit` on the `MetricTriple` its producer builds."
        )
    if declared != _OPERAND_DOMAIN_SQUARED_TOKEN:
        return declared
    try:
        return _DOMAIN_UNITS[value_domain]
    except KeyError:
        raise ValueError(
            f"Metric {metric!r} is measured in the operand domain, but the "
            f"preprocess-stamped value_domain {value_domain!r} is not one of "
            f"{sorted(_DOMAIN_UNITS)}."
        ) from None


def _stamped_value_domain(run_dir: Path, config=None) -> str:
    """The domain preprocess encoded in, from its own stamp.

    Cross-checked against the CONFIGURED representation's own declaration when a
    config is supplied (F-162). The stamp decides whether an operand-domain metric
    prints as dB^2 or a.u.^2, and it is an artifact of a stage that may have run
    under a different config — so the two are compared rather than either being
    trusted alone. `_report_fingerprint` carries the config side, which is what
    makes a domain change invalidate a cached report at all.
    """
    meta_path = run_dir / "preprocessed" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No preprocessing metadata at {meta_path}, so the operand domain the "
            f"reported metrics are measured in is unknown. Run preprocess first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    try:
        stamped = meta["value_domain"]
    except KeyError:
        # A pre-stamp preprocess run — a real population, and the sibling consumer
        # in evaluation/ already fails loud on it. Say the same thing here rather
        # than raising a bare KeyError with a traceback (F-164).
        raise KeyError(
            f"{meta_path} has no 'value_domain'. It predates the domain stamp, so "
            f"the unit the reported metrics are measured in cannot be established. "
            f"Re-run preprocess."
        ) from None

    if config is not None:
        from ..registry import representation_registry

        declared = str(representation_registry.get(config.representation.name).value_domain)
        if stamped != declared:
            raise ValueError(
                f"{meta_path} stamps value_domain={stamped!r} but the configured "
                f"representation {config.representation.name!r} declares "
                f"{declared!r}. The reported units are rendered from this, so the "
                f"two disagreeing means the table would label numbers produced in "
                f"one domain with the unit of another. Re-run preprocess under this "
                f"config, or point at the run_dir that matches it."
            )
    return stamped


def _record_length_line(config: Config, run_dir: Path, split_name: str) -> str | None:
    """This split's own record-length over-limit count, for its report section.

    RD-65: the gate `scenes/generator.py` applies is the OVERALL over-limit fraction
    across every regime — the one aggregation invariant #9 forbids for results. That
    is the right GATE (a per-split gate lets the smallest split set the tolerance for
    train), but it means research_i's 0.01 over 720 scenes permits 7 over-limit
    scenes that could all sit in the 30-scene `test_geometry_shift`, 23 % of it, and
    still pass. The per-shift breakdown IS the research result, and it reached only
    `placement_report.json` — surfacing in the operator's console solely when the
    gate tripped.

    A scene is over the limit when its closed-form decay does not fit the record, so
    its T30 is the truncation rather than the room. That is a caveat on the numbers
    in the section this line sits under, which is why it is rendered per split rather
    than once for the run.

    Returns None when there is no report to read — a run_dir assembled from stats
    alone, which the report tests do — rather than failing: this is a disclosure
    about scene generation, and its absence is not a reason to refuse a table.
    """
    report_path = run_dir / "scenes" / "placement_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        placement = json.load(f)
    regime, covered = config.generation_regime_of(split_name)
    entry = placement.get(regime)
    if not isinstance(entry, dict) or "record_decay_range" not in entry:
        return None
    block = entry["record_decay_range"]
    scored = block.get("n_scenes", 0)
    over = block.get("decay_range_below_iso_t30", {}).get("count", 0)
    # In frac mode the id-pool splits share one regime entry, so the count covers
    # train+valid+test_id and naming it after this split alone would overstate what
    # was measured here (S-F7).
    scope = (
        f"regime {regime!r}, pooling {'/'.join(covered)}"
        if covered != (split_name,) else f"split {split_name!r}"
    )
    if not scored:
        return (
            f"Record length: UNSCORED for {scope} — no scene was characterized, so "
            f"the over-limit fraction is undefined, not 0 (RD-64/F-71)."
        )
    limit = config.scenes.max_frac_below_iso_t30_decay_range
    verdict = "" if over / scored <= limit else "  ** ABOVE this config's own limit **"
    return (
        f"Record length: {over}/{scored} scenes ({over / scored:.1%}) in {scope} carry "
        f"a decay the record cannot hold, against a declared {limit:.0%} (RD-65). "
        f"The GATE is the OVERALL fraction across regimes, so this split may be far "
        f"over on its own and the run still pass.{verdict}"
    )


def _unconverged_reference_footer(config: Config, unapplied: list[str]) -> list[str]:
    """Footer lines for every metric whose REFERENCE LEG is not converged (AC-187).

    The other caveats in this table describe the scored population — how many bands
    survived, how noisy the estimator is at this decay. This one describes the
    GROUND TRUTH: `improvement` for a match-reference metric is
    |low − high| − |pred − high|, so a high leg that moves with the ray budget moves
    every term in it, and no amount of scoring discipline detects that from inside
    the run.

    Each entry renders its MEASUREMENT beside its TOLERANCE, because "unconverged"
    alone is not actionable — 3.24 dB against a 1.0 dB JND is a different fact from
    1.1 dB, and a reader deciding whether to quote a C50 absolute needs the size.

    `unapplied` names declared metrics this run does not report. Listed rather than
    refused: reporting a subset is legitimate (a run may carry the energy metrics
    and not the ISO ones), but a misspelt metric name would otherwise vanish, and
    this project logs every skip with its reason.

    Returns nothing at all when the map is empty, so clearing it after a future
    probe clears the text — the footer never carries a paragraph about a resolved
    concern.
    """
    unconverged = config.convergence.reference_unconverged
    if not unconverged:
        return []

    applied = sorted(set(unconverged) - set(unapplied))
    lines = [
        "REFERENCE CONVERGENCE — a caveat on the GROUND TRUTH, not on the scored",
        "  population. Every paired improvement here is measured against the high-ray",
        "  leg, and the ray-budget probe measured that it is not a converged one:",
    ]
    for metric in applied:
        m = unconverged[metric]
        lines.append(
            f"  {metric}: worst deviation {m.worst_deviation:g} {m.unit}, "
            f"{m.n_within_tolerance}/{m.n_cells} cells within the declared "
            f"{m.tolerance:g} {m.unit}. Rows carry `reference unconverged`."
        )
    for metric in sorted(unapplied):
        lines.append(
            f"  {metric}: declared unconverged, but SKIPPED — this run does not "
            f"report it, so no row carries the caveat."
        )
    if applied:
        lines += [
            "  The values ARE reported: suppressing them would hide the finding rather",
            "  than disclose it. But no absolute or improvement for these metrics may",
            "  be compared against Research I or the literature on this run's strength.",
            "  Raising high_ray_budget is not a fix — 800k costs 35x for 4x the rays",
            "  and is itself unverified (AC-187).",
        ]
    return lines


def run_report(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    stats_dir = run_dir / "stats"
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = stats_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No stats found at {summary_path}. Run stats first.")

    with open(summary_path) as f:
        summary = json.load(f)

    df = pd.DataFrame(summary)

    col_w = {"metric": 22, "n": 8, "pred": 10, "imp": 10, "ci": 22, "mdes": 10,
             "kind": 16,
             "unit": 6, "improved": 14, "caveat": 18}

    # The domain the operand-domain metrics are measured in, read once from
    # preprocess's own stamp rather than per row.
    value_domain = _stamped_value_domain(run_dir, config)

    # Resolve the unit for EVERY metric present, up front — not lazily per
    # rendered row (F-163/AC-130). `_metric_row` returns early for an unscored
    # row, so a lazy lookup made the "an undeclared metric is refused" contract
    # depend on the DATA: a new metric passed on the run where it happened to be
    # unscored and crashed the report on a later run that scored it, moving the
    # failure away from the change that caused it. The guard is over the metric
    # SET, so it fires on the first run that mentions the metric at all.
    units = {
        row["metric"]: _unit_for(row["metric"], row.get("unit", ""), value_domain)
        for row in summary
    }

    # A declared unconverged-reference caveat that matches no reported metric is a
    # SKIP, and this project logs every skip as (unit, reason) rather than letting
    # it pass unremarked (AC-187). It is not fatal: a run may legitimately report a
    # subset — the energy metrics without the ISO ones — and refusing there would
    # make a correct config fail on a correct run. But it is also how a typo hides,
    # so the footer names it either way.
    unapplied = sorted(set(config.convergence.reference_unconverged) - set(units))

    def _caveats(row: dict) -> str:
        """Composition caveats on the scored population (F-62 / AC-25 / RD-78).

        `4/4` reads as fully scored, and on the RI smoke run it was not: one
        scene's EDT was a ONE-BAND average while the other three were two-band
        averages, so the split's CI pooled improvements computed over different
        band sets. Rendered next to the count rather than left in drops.csv,
        because the count is what a reader takes away.
        """
        parts = []
        if row.get("n_partial_band"):
            parts.append(f"{row['n_partial_band']} partial-band")
        if row.get("n_pred_band_unresolved"):
            parts.append(f"{row['n_pred_band_unresolved']} pred-unresolved")
        if row.get("n_pred_unscored_imputed"):
            # F-70: those scenes are OUT of the CI beside this, so the interval is
            # conditioned on the model having produced something measurable. The
            # bound over the attempted population is rendered below the row.
            parts.append(f"{row['n_pred_unscored_imputed']} model-failed")
        if row.get("n_estimator_variance_limited"):
            # Not a drop: the value is scored, but its ESTIMATOR carries 24-31 %
            # sd in this range, which a bare point estimate does not convey.
            parts.append(f"{row['n_estimator_variance_limited']} high-variance")
        if row.get("ci_calibrated") is False:
            # The interval is reported, not suppressed — but a percentile bootstrap
            # below the declared n cannot reach its nominal coverage, so calling it
            # a "95 % CI" is the overclaim (F-M7/F-105).
            parts.append("CI uncalibrated at this n")
        if row["metric"] in config.convergence.reference_unconverged:
            # NOT a property of this split's data — a property of the REFERENCE
            # every row of this metric is measured against (AC-187). It therefore
            # appears on every C50 row in every split, including fully scored ones,
            # which is the point: `12/12` otherwise reads as a clean result for a
            # comparison whose ground truth is measured not to be ground truth.
            parts.append("reference unconverged")
        if row.get("n_resolvability_limited"):
            # Also not a drop: the PHYSICAL legs reported a value from a band their
            # own octave filter cannot resolve. Scored and disclosed rather than
            # censored, because censoring an estimator on its own value biases the
            # survivors — but a reader must be able to see which numbers carry it.
            parts.append(f"{row['n_resolvability_limited']} band-unresolvable")
        return ", ".join(parts)

    def _metric_row(row: dict) -> str:
        # Inferential columns (imp mean / CI / MDES) are the §9 paired improvement
        # for the metric's declared kind; "Pred mean" is the descriptive absolute
        # value. n_scored == 0 → NOTHING here is a result: render the row as
        # `unscored`, never a number a reader could mistake for an outcome — a
        # descriptive mean included (RR-14).
        n_str = f"{row['n_scored']}/{row['n_attempted']}"
        if row["n_scored"] == 0:
            return (
                f"{row['metric']:<{col_w['metric']}} "
                f"{n_str:>{col_w['n']}} "
                f"unscored — no scene has finite legs (reasons: metrics/drops.csv)"
            )
        imp_mean_str = f"{row['improvement_mean']:.4f}"
        ci_str = f"[{row['improvement_ci_lower']:.4f}, {row['improvement_ci_upper']:.4f}]"
        improved_str = f"{row['pct_improved']:.1f}% ({row['n_improved']}/{row['n_scored']})"
        mdes_val = row["improvement_mdes"]
        mdes_str = f"{mdes_val:.4f}" if mdes_val == mdes_val else "N/A"
        line = (
            f"{row['metric']:<{col_w['metric']}} "
            f"{n_str:>{col_w['n']}} "
            f"{row['pred_mean']:>{col_w['pred']}.4f} "
            f"{imp_mean_str:>{col_w['imp']}} "
            f"{ci_str:<{col_w['ci']}} "
            f"{mdes_str:>{col_w['mdes']}} "
            # AC-48 — the footer line states which columns this labels.
            f"{units[row['metric']]:<{col_w['unit']}} "
            # F-166: the unit says dB, the KIND says what the dB is measured
            # FROM. `Imp mean` is pred-low for `maximize` and
            # &#124;low-high&#124; - &#124;pred-high&#124; for `match_reference`,
            # and the two share a unit — so without this a reader cannot tell
            # which reference point a number carries.
            f"{row.get('kind', '?'):<{col_w['kind']}} "
            f"{improved_str:<{col_w['improved']}} "
            f"{_caveats(row):<{col_w['caveat']}}"
        ).rstrip()
        if not row.get("n_pred_unscored_imputed"):
            return line
        # ── The same improvement over the ATTEMPTED population (F-70) ───────────
        #
        # A second line rather than a column, because it is a second ESTIMATE of the
        # same quantity over a different population — putting it beside the first
        # would invite reading one as a correction of the other. Shown only when the
        # two can differ, so a fully-scored split's table is unchanged.
        att_ci = (
            f"[{row['improvement_ci_lower_attempted']:.4f}, "
            f"{row['improvement_ci_upper_attempted']:.4f}]"
        )
        return (
            f"{line}\n"
            f"{'  └ attempted-population bound':<{col_w['metric']}} "
            f"{row['n_attempted_scorable']:>{col_w['n']}} "
            f"{'':>{col_w['pred']}} "
            f"{row['improvement_mean_attempted']:>{col_w['imp']}.4f} "
            f"{att_ci:<{col_w['ci']}} "
            f"{'':>{col_w['mdes']}} "
            f"{units[row['metric']]:<{col_w['unit']}} "
            f"{row.get('kind', '?'):<{col_w['kind']}} "
            f"{'':<{col_w['improved']}} "
            f"{row['n_pred_unscored_imputed']} imputed at 0"
        ).rstrip()

    # CI level from config, not hardcoded in the label (RR-17, same rule as RR-11).
    ci_label = f"Imp {100 * (1 - config.bootstrap_alpha):g}% CI"
    hdr = (
        f"{'Metric':<{col_w['metric']}} "
        # The scored count IS the paired-improvement population (F-21).
        f"{'N sc/att':>{col_w['n']}} "
        f"{'Pred mean':>{col_w['pred']}} "
        f"{'Imp mean':>{col_w['imp']}} "
        f"{ci_label:<{col_w['ci']}} "
        f"{'MDES':>{col_w['mdes']}} "
        f"{'Unit':<{col_w['unit']}} "
        f"{'Kind':<{col_w['kind']}} "
        f"{'% Improved':<{col_w['improved']}} "
        f"{'Caveats':<{col_w['caveat']}}"
    ).rstrip()

    # One section per split — never pool test splits (invariant #9).
    #
    # Sections are enumerated from the CONFIG-DECLARED test splits in declaration
    # order, not from the splits present in the data (F-45). A declared split that
    # received no scored scene previously vanished from this file entirely, and an
    # absent split is indistinguishable from one that was never declared — the same
    # silent-exclusion class the drop log exists to prevent. Ordering is therefore
    # declaration order rather than the previous alphabetical sort.
    lines = ["=" * 70, f"Run: {run_dir.name}", "=" * 70]
    present_splits = set(df["split"].unique()) if not df.empty else set()
    declared = list(config.test_split_names)
    # Anything scored but not declared would be a routing bug; surface it rather
    # than dropping it off the end of the report.
    undeclared = sorted(present_splits - set(declared))
    for split_name in declared + undeclared:
        split_rows = [r for r in summary if r["split"] == split_name]
        scored_rows = [r for r in split_rows if r.get("n_attempted", 0) > 0]
        suffix = "" if split_name in declared else "  [NOT DECLARED IN CONFIG]"
        lines += [
            "",
            f"Metric results ({split_name}, paired improvement, bootstrap CI):{suffix}",
            "",
        ]
        if not scored_rows:
            # Mirrors _metric_row's n_scored == 0 rule at the split level: nothing
            # here is a result, so render no numbers at all (RR-14).
            lines.append(
                "0 scenes — unscored: this split is declared in config but no scene "
                "reached eval (see preprocessed/meta.json split_counts)."
            )
            continue
        lines += [hdr, "-" * len(hdr)]
        for row in scored_rows:
            lines.append(_metric_row(row))
        # RD-65: this split's own record-length over-limit count, beside the numbers
        # it is a caveat on rather than only in scenes/placement_report.json.
        record_note = _record_length_line(config, run_dir, split_name) if (
            split_name in config.splits
        ) else None
        if record_note:
            lines += ["", record_note]

    lines += [
        "",
        "N sc/att = scenes scored / attempted; per-leg drop reasons: metrics/drops.csv",
        "Unit applies to Pred mean, Imp mean, CI and MDES. Rows mix seconds, decibels",
        "  and dB², so these columns are NOT comparable across rows. dB² is a mean",
        "  SQUARED level difference, not decibels — take its square root for an RMS",
        "  level error (AC-126).",
        "Imp mean is a REDUCTION IN |ERROR| against the high-ray reference for the",
        "  match-reference metrics (T30, EDT, C50, energy_mse) and pred − low for the",
        "  maximize ones (energy_snr_db); Pred mean is the absolute value. Same unit,",
        "  different reference point — a negative Imp mean means the error GREW (AC-128).",
        "Caveats — partial-band: the band average is over fewer bands than declared, so",
        "  this split's CI pools improvements computed over DIFFERENT band sets (F-62).",
        "  pred-unresolved: the model produced no measurable value in a band the physical",
        "  legs resolve; the physical legs keep their own values (AC-25).",
        "  model-failed: the model produced nothing measurable at all, so the scene left",
        "  the CI above — which therefore conditions on the model having WORKED, an",
        "  optimistic direction, and one whose probability correlates with absorption and",
        "  so with test_material_shift's own axis (F-70). The `attempted-population",
        "  bound` line under such a row re-runs the same bootstrap with those scenes",
        "  imputed at ZERO improvement: a lower bound, not a correction. Scenes whose",
        "  PHYSICAL legs failed are in neither population — there is no ground truth",
        "  there to have improved on, so a zero would invent a datum rather than bound one.",
        # The VALUE, not just the key name (AC-48). A reader seeing "3 high-variance"
        # cannot judge it without the bound, and F-65's own evidence is that this
        # key was served at 0.15 while config.yaml stamped 5.0. The CI label above
        # already renders its config value numerically; this now matches it.
        f"  high-variance: EDT below metric_edt_variance_limited_s = "
        f"{config.metric_edt_variance_limited_s:g} s, where the ESTIMATOR's",
        "  sd is 24-31 % of T60 — a scored value, not a precise one (AC-27/RD-78).",
        f"  band-unresolvable: the PHYSICAL legs reported a value from a band whose",
        f"  decay is below what that octave filter can resolve "
        f"({config.metric_band_resolvability_margin:g} x the filter's own decay).",
        "  Scored and disclosed, never censored — censoring an estimator on its own",
        "  value biases the survivors (AC-38) — but the absolute is the least",
        "  trustworthy in the table (F-M2).",
        *_unconverged_reference_footer(config, unapplied),
        "=" * 70,
    ]
    summary_txt = "\n".join(lines)

    (report_dir / "summary.txt").write_text(summary_txt)

    # THE CSV CARRIES THE SAME DISCLOSURE AS THE TEXT TABLE (AC-129). summary.txt
    # gained a Unit column and this file, written five lines later from the same
    # rows, shipped 21 unitless columns — so the machine-readable artifact, which is
    # the one a downstream analysis actually opens, was the one that could not say
    # whether a number was seconds or decibels. `kind` travels with it for the same
    # reason (F-166): unit and reference point are different questions and a
    # dB-valued improvement answers only the first.
    df = df.copy()
    df["unit"] = df["metric"].map(units)
    # AC-187, carried into the machine-readable artifact for the AC-129 reason the
    # unit is: a downstream analysis opens this file, not summary.txt, and a C50
    # improvement whose reference is unconverged must not arrive there bare.
    df["reference_converged"] = ~df["metric"].isin(
        config.convergence.reference_unconverged
    )
    df.to_csv(report_dir / "metrics_table.csv", index=False)

    # Supplementary bundle: copy config stamp + versions. Provenance, same gate
    # as its source (`Config.stamp` runs at save ≥ 1), so a save=0 run — the
    # sanctioned provenance-free level (RD-09) — is self-consistent rather than
    # silently missing a copy.
    if verbosity.saves("provenance"):
        for fname in ["config.yaml", "versions.json"]:
            src = run_dir / fname
            if src.exists():
                shutil.copy(src, report_dir / fname)

    emit(verbosity, "metrics", summary_txt)
    emit(verbosity, "metrics", f"\n  Report written → {report_dir}")
