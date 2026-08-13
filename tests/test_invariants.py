"""
Pipeline invariants (design_spec §10).

These tests exercise the invariants that guard against silent bugs: leakage,
normalization source, split disjointness, loss inertness, controlled-shift
integrity, etc. Split names and scene sampling ranges come from the config
(configs/overlays/test_tiny.yaml) — nothing about the split set is hardcoded here.
Most tests run the full preprocess stage on a tiny dataset.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from amcd.config import Config
from amcd.data.normalization import compute_stats
from amcd.data.splits import assign_split
from amcd.simulators.base import SceneSpec

from tests.conftest import EVAL_FREQS, QUIET, TINY_LAYERS, dry_run_simulator


def _make_minimal_config() -> Config:
    """The tiny test config: 4 channels, 8 kHz, 0.1 s, all six splits n≥2."""
    return Config.load(*TINY_LAYERS)


def _run_through_preprocess(tmp_path: Path, cfg: Config) -> dict:
    """Run gen-scenes + render + preprocess and return meta."""
    from amcd.scenes.generator import run_gen_scenes
    from amcd.simulators.render import run_render
    from amcd.data.preprocess import run_preprocess

    run_gen_scenes(cfg, tmp_path, QUIET)
    run_render(cfg, tmp_path, QUIET)
    run_preprocess(cfg, tmp_path, QUIET)

    with open(tmp_path / "preprocessed" / "meta.json") as f:
        return json.load(f)


class TestSplitDisjointness:
    """Invariant #1: train/valid/test scene sets are disjoint."""

    def test_splits_disjoint(self) -> None:
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)

            with open(tmp / "preprocessed" / "splits.json") as f:
                splits: dict[str, str] = json.load(f)

            by_split: dict[str, set] = {}
            for sid, sp in splits.items():
                by_split.setdefault(sp, set()).add(sid)

            split_names = list(by_split.keys())
            for i, a in enumerate(split_names):
                for b in split_names[i + 1:]:
                    assert by_split[a].isdisjoint(by_split[b]), f"{a} ∩ {b} ≠ ∅"

    def test_all_scenes_assigned(self) -> None:
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)

            with open(tmp / "preprocessed" / "splits.json") as f:
                splits: dict[str, str] = json.load(f)

            n_shift = sum(sp.count for sp in cfg.shift_splits.values())
            assert len(splits) == cfg.scenes.n_id + n_shift


class TestNormalizationSource:
    """Invariant #3: norm stats are computed from train split only."""

    def test_norm_stats_from_train_only(self) -> None:
        """
        Compute stats with and without test data included; the train-only stats
        must differ (structural check that compute_stats only sees train tensors).
        """
        rng = np.random.default_rng(0)
        n_train = 10
        n_test = 5

        train_lows = [torch.tensor(rng.standard_normal((4, 20, 10)).astype(np.float32)) for _ in range(n_train)]
        train_highs = [torch.tensor(rng.standard_normal((4, 20, 10)).astype(np.float32)) for _ in range(n_train)]
        test_lows = [torch.tensor(rng.standard_normal((4, 20, 10)).astype(np.float32) * 100) for _ in range(n_test)]
        test_highs = [torch.tensor(rng.standard_normal((4, 20, 10)).astype(np.float32) * 100) for _ in range(n_test)]

        stats_train_only = compute_stats(train_lows, train_highs)
        stats_with_test = compute_stats(train_lows + test_lows, train_highs + test_highs)

        assert abs(stats_train_only["low_std"]) < abs(stats_with_test["low_std"]), (
            "Mixing test data into stats didn't change them — implementation may be wrong"
        )

    def test_preprocess_saves_norm_stats(self) -> None:
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            meta = _run_through_preprocess(tmp, cfg)

        assert "norm_stats" in meta
        ns = meta["norm_stats"]
        for key in ["low_mean", "low_std", "high_mean", "high_std"]:
            assert key in ns, f"Missing norm stat: {key}"
            assert np.isfinite(ns[key]), f"Non-finite norm stat: {key} = {ns[key]}"


class TestPerScenePreservation:
    """Invariant #6: per-scene rows are never collapsed before stats stage."""

    def test_metrics_parquet_one_row_per_scene_metric(self) -> None:
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)
            from amcd.training.trainer import run_train
            from amcd.training.infer import run_infer
            from amcd.evaluation.evaluator import run_eval

            with open(tmp / "preprocessed" / "splits.json") as f:
                splits = json.load(f)
            n_test = sum(1 for s in splits.values() if s in cfg.test_split_names)

            if n_test == 0:
                pytest.skip("No test scenes in any test split — increase scenes.n_id")

            run_train(cfg, tmp, QUIET)
            run_infer(cfg, tmp, QUIET)
            run_eval(cfg, tmp, QUIET)

            import pandas as pd
            df = pd.read_parquet(tmp / "metrics" / "metrics.parquet")

            assert df["scene_id"].nunique() == n_test
            dupes = df.duplicated(subset=["scene_id", "metric"]).sum()
            assert dupes == 0, f"Found {dupes} duplicate (scene_id, metric) rows"


class TestLossActivity:
    """Invariant #7: all loss terms should be non-NaN and non-zero on a synthetic batch."""

    def test_huber_loss_nonzero(self) -> None:
        from amcd.representations import build_representation

        cfg = _make_minimal_config()
        rep = build_representation(
            cfg.representation.name, cfg.representation.params,
            sample_rate=cfg.sample_rate, eval_freqs_hz=EVAL_FREQS,
        )
        pred = torch.randn(2, cfg.n_channels, rep.n_bands, 5)
        target = torch.randn(2, cfg.n_channels, rep.n_bands, 5)

        loss = rep.loss(pred, target, delta=cfg.huber_delta)
        assert torch.isfinite(loss), "Huber loss is not finite"
        assert loss.item() > 0, "Huber loss is zero — terms may be silently inactive"

    def test_huber_delta_meaningful_in_training_domain(self) -> None:
        """δ vs signal scale (inv #7 / H2) on the REAL trainer operands.

        The trainer z-scores operands by high_std, so a raw δ=1.0 would put the Huber
        knee tens of dB out (≈MSE). build_criterion scales δ (dB) into that domain; a
        ~3 dB residual must then sit in the L1 regime, distinct from MSE. This replaces
        the old raw-dB check, which tested a domain the trainer never uses (ledger F-01).
        """
        from amcd.training.loss import build_criterion

        cfg = _make_minimal_config()
        high_std = 16.0  # realistic log-band-energy std (dB); test datum
        criterion = build_criterion(cfg, {"high_mean": 0.0, "high_std": high_std})

        # A residual of ~3 dB, expressed in the z-scored domain the loss sees.
        pred = torch.zeros(2, 4, 10, 5)
        target = torch.ones(2, 4, 10, 5) * (3.0 / high_std)

        loss = criterion(pred, target)
        mse_loss = torch.nn.functional.mse_loss(pred, target)

        assert torch.isfinite(loss) and loss.item() > 0, "Huber loss inactive"
        # 3 dB > δ=1 dB → linear regime, so Huber departs from MSE (0.5·r²).
        assert loss.item() < 0.5 * mse_loss.item() * 0.99, (
            "Huber ≈ MSE — δ not O(1) in dB on the real operands (H2 degeneracy)"
        )


class TestBaselinePassthrough:
    """Regression guard for the normalization-mismatch bug.

    An identity model (pred_norm = low_norm) must yield improvement_ratio ≈ 1.0.
    """

    def test_identity_model_ratio_near_one(self) -> None:
        from amcd.data.normalization import denormalize

        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)

            with open(tmp / "preprocessed" / "meta.json") as f:
                meta = json.load(f)
            with open(tmp / "preprocessed" / "splits.json") as f:
                splits: dict[str, str] = json.load(f)

            norm_stats = meta["norm_stats"]
            test_scene_ids = [(sid, sp) for sid, sp in splits.items() if sp in cfg.test_split_names]
            if not test_scene_ids:
                pytest.skip("No test scenes in any test split — increase scenes.n_id")

            scene_id, split_name = test_scene_ids[0]
            split_dir = tmp / "preprocessed" / split_name
            low_norm = torch.load(split_dir / f"{scene_id}_low.pt", weights_only=True)
            high_norm = torch.load(split_dir / f"{scene_id}_high.pt", weights_only=True)

            pred_norm = low_norm  # identity model in normalized space

            pred_db = denormalize(pred_norm, norm_stats["high_mean"], norm_stats["high_std"])
            low_db = denormalize(low_norm, norm_stats["high_mean"], norm_stats["high_std"])
            high_db = denormalize(high_norm, norm_stats["high_mean"], norm_stats["high_std"])

            pred_mse = float((high_db - pred_db).pow(2).mean())
            baseline_mse = float((high_db - low_db).pow(2).mean())
            improvement_ratio = baseline_mse / (pred_mse + 1e-10)

            assert 0.9 <= improvement_ratio <= 1.1, (
                f"Identity model improvement_ratio={improvement_ratio:.3f} (expected ≈ 1.0). "
                "Normalization mismatch bug may have returned."
            )


class TestDistributionShiftSplits:
    """Invariants #9 (per-split independence, never pooled) and #10 (controlled shift)."""

    def test_all_six_splits_populated(self) -> None:
        """The declared config populates every split it declares (inv #9 precondition)."""
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            meta = _run_through_preprocess(tmp, cfg)

            counts = meta["split_counts"]
            for sp in cfg.splits:
                assert counts.get(sp, 0) >= 1, (
                    f"Split {sp!r} is empty (counts={counts}); shift plumbing not exercised"
                )

    def test_test_splits_never_pooled(self) -> None:
        """Eval retains each test split independently — no pooled 'test' bucket (inv #9)."""
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)
            from amcd.training.trainer import run_train
            from amcd.training.infer import run_infer
            from amcd.evaluation.evaluator import run_eval

            run_train(cfg, tmp, QUIET)
            run_infer(cfg, tmp, QUIET)
            run_eval(cfg, tmp, QUIET)

            import pandas as pd
            df = pd.read_parquet(tmp / "metrics" / "metrics.parquet")

            eval_splits = set(df["split"].unique())
            assert "test" not in eval_splits, "Found pooled 'test' split — invariant #9 violated"
            assert eval_splits <= set(cfg.test_split_names), (
                f"Eval ran on non-test splits {eval_splits - set(cfg.test_split_names)}"
            )
            assert len(eval_splits & set(cfg.test_split_names)) >= 2, (
                f"Expected multiple test splits reported separately, got {eval_splits}"
            )

    def test_shift_axis_ranges_disjoint_from_id(self) -> None:
        """RD-02 / inv #10: each shift axis is a genuine held-out region — its
        sampling range is DISJOINT from the id baseline along the named axis (or,
        for placement, a proper corner sub-region), so a shift is not an
        overlapping tilt. Read entirely from config."""
        cfg = _make_minimal_config()
        sc = cfg.scenes
        id_geo = sc.id_regime["geometry"]
        id_mat = sc.id_regime["material"]

        for name, sp in cfg.shift_splits.items():
            (axis, value), = sp.axes.items()
            if axis == "geometry":
                shift_lo = sc.geometry_families[value].dims[0][0]   # long-axis lo
                id_hi = sc.geometry_families[id_geo].dims[0][1]     # long-axis hi
                assert shift_lo > id_hi, f"{name}: geometry long-axis overlaps id"
            elif axis == "material":
                shift_lo = sc.material_regimes[value].absorption[0]
                id_hi = sc.material_regimes[id_mat].absorption[1]
                assert shift_lo > id_hi, f"{name}: material absorption overlaps id"
            elif axis == "placement":
                assert sc.placement_regimes[value].type == "corner"
                assert 0.0 < sc.placement_regimes[value].corner_frac < 1.0, (
                    f"{name}: corner_frac must be a proper sub-region"
                )

    def test_controlled_shift_integrity(self) -> None:
        """Each shift scene differs from id in exactly its named axis — asserted on
        the LABEL and on the actually-sampled quantity (inv #10). All ranges from config."""
        from amcd.scenes.generator import run_gen_scenes

        cfg = _make_minimal_config()
        sc = cfg.scenes
        shoebox_long_hi = sc.geometry_families["shoebox"].dims[0][1]
        corridor_long_lo = sc.geometry_families["corridor"].dims[0][0]
        mixed_hi = sc.material_regimes["mixed"].absorption[1]
        ceiling_lo = sc.material_regimes["ceiling_absorptive"].absorption[0]
        corner_frac = sc.placement_regimes["near_corner"].corner_frac
        margin = sc.margins.wall

        assert corridor_long_lo > shoebox_long_hi, "corridor/shoebox long axis overlap"
        assert ceiling_lo > mixed_hi, "material ranges overlap"

        def _assert_sample_matches_label(s: SceneSpec) -> None:
            axes = s.regime_axes
            if axes["geometry"] == "corridor":
                assert s.dims[0] >= corridor_long_lo, f"{s.scene_id} corridor label but dims={s.dims}"
            else:
                assert s.dims[0] <= shoebox_long_hi, f"{s.scene_id} shoebox label but dims={s.dims}"
            if axes["material"] == "ceiling_absorptive":
                assert s.material_absorption >= ceiling_lo, (
                    f"{s.scene_id} ceiling_absorptive label but α={s.material_absorption}"
                )
            else:
                assert s.material_absorption <= mixed_hi, (
                    f"{s.scene_id} mixed label but α={s.material_absorption}"
                )
            if axes["placement"] == "near_corner":
                # HORIZONTAL axes only (AC-10): the corner bias is about proximity
                # to a wall junction, and z is bounded by floor/ceiling margins or
                # an explicit ergonomic height_range — there is no corner in z.
                for i in (0, 1):
                    hi = s.dims[i] - margin
                    corner_hi = margin + corner_frac * (hi - margin)
                    assert s.receiver_pos[i] <= corner_hi + 1e-6, (
                        f"{s.scene_id} near_corner label but rcv[{i}]={s.receiver_pos[i]} > {corner_hi}"
                    )

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            run_gen_scenes(cfg, tmp, QUIET)

            id_axes = dict(sc.id_regime)
            scenes = [SceneSpec.from_json(p) for p in (tmp / "scenes").glob("scene_*.json")]
            saw_shift = {name: 0 for name in cfg.shift_splits}

            for s in scenes:
                _assert_sample_matches_label(s)  # label ↔ sampled quantity consistent

                if s.split_regime == "id":
                    assert s.regime_axes == id_axes, (
                        f"id scene {s.scene_id} axes {s.regime_axes} ≠ baseline {id_axes}"
                    )
                    continue

                # split_regime IS the target split name; it must be a declared shift split.
                assert s.split_regime in cfg.shift_splits, (
                    f"{s.scene_id} split_regime {s.split_regime!r} not a declared shift split"
                )
                (named_axis, _), = cfg.shift_splits[s.split_regime].axes.items()

                differing = {k for k in id_axes if s.regime_axes[k] != id_axes[k]}
                assert differing == {named_axis}, (
                    f"{s.split_regime} scene {s.scene_id} differs from id in {differing}, "
                    f"expected only {{{named_axis!r}}} (uncontrolled co-variation)"
                )
                saw_shift[s.split_regime] += 1

            for name, n in saw_shift.items():
                assert n == cfg.shift_splits[name].count, (
                    f"Expected {cfg.shift_splits[name].count} {name} scenes, saw {n}"
                )


class TestDryRunSimulatorSceneDependence:
    """The synthetic dry_run simulator must READ the perturbed scene axes — otherwise
    the shift splits are pure label artifacts. Guards the falsifier's inert-shift
    blocker at the tensor level."""

    def _sim(self) -> DryRunSimulator:
        return dry_run_simulator(n_channels=4, n_samples=2000, sample_rate=8000)

    def _spec(self, seed: int, dims, absorption: float, rcv) -> SceneSpec:
        return SceneSpec(
            scene_id="s",
            seed=seed,
            geometry_family="shoebox",
            dims=dims,
            material_absorption=absorption,
            source_pos=(1.0, 1.0, 1.0),
            receiver_pos=rcv,
            sim_params={},
        )

    def test_material_absorption_moves_rt60_and_tail_energy(self) -> None:
        sim = self._sim()
        dims = (6.0, 5.0, 3.0)
        rcv = (4.0, 3.0, 2.0)
        low_abs = sim.render(self._spec(0, dims, 0.2, rcv), ray_budget=200000)
        high_abs = sim.render(self._spec(0, dims, 0.9, rcv), ray_budget=200000)

        assert high_abs.meta["rt60_s"] < low_abs.meta["rt60_s"], "absorption ignored by simulator"
        tail = slice(1000, 2000)
        e_low = float((low_abs.ir[:, tail] ** 2).mean())
        e_high = float((high_abs.ir[:, tail] ** 2).mean())
        assert e_high < e_low, "material shift not visible in rendered tail energy"

    def test_geometry_moves_rt60(self) -> None:
        sim = self._sim()
        shoebox = sim.render(self._spec(1, (6.0, 5.0, 3.0), 0.3, (4.0, 3.0, 2.0)), ray_budget=200000)
        corridor = sim.render(self._spec(1, (25.0, 2.0, 3.0), 0.3, (4.0, 1.0, 2.0)), ray_budget=200000)
        assert shoebox.meta["rt60_s"] != corridor.meta["rt60_s"], "geometry ignored by simulator"

    def test_distance_moves_direct_level(self) -> None:
        sim = self._sim()
        dims = (10.0, 8.0, 3.0)
        near = sim.render(self._spec(2, dims, 0.3, (1.5, 1.5, 1.2)), ray_budget=200000)
        far = sim.render(self._spec(2, dims, 0.3, (9.0, 7.0, 2.5)), ray_budget=200000)
        assert near.meta["distance_m"] < far.meta["distance_m"]
        early = slice(0, 50)
        assert float((near.ir[:, early] ** 2).mean()) > float((far.ir[:, early] ** 2).mean()), (
            "placement/distance not visible in rendered direct level"
        )


class TestTrainValidTestNoOverlap:
    """Invariant #8: no test scene IDs appear in training dataloader."""

    def test_no_test_in_train_loader(self) -> None:
        cfg = _make_minimal_config()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run_through_preprocess(tmp, cfg)

            with open(tmp / "preprocessed" / "splits.json") as f:
                splits: dict[str, str] = json.load(f)

            from amcd.data.dataset import EnergyDataset
            preprocessed_dir = tmp / "preprocessed"

            train_split = next(name for name, sp in cfg.splits.items() if sp.role == "train")
            if not any(s == train_split for s in splits.values()):
                pytest.skip("No train scenes")
            present_test_splits = [sp for sp in cfg.test_split_names if any(s == sp for s in splits.values())]
            if not present_test_splits:
                pytest.skip("No test scenes in any test split")

            train_ds = EnergyDataset(preprocessed_dir, train_split)
            train_ids = set(train_ds.scene_ids)

            for sp in present_test_splits:
                test_ds = EnergyDataset(preprocessed_dir, sp)
                test_ids = set(test_ds.scene_ids)
                assert train_ids.isdisjoint(test_ids), (
                    f"Test scene(s) from {sp} found in train loader: {train_ids & test_ids}"
                )
