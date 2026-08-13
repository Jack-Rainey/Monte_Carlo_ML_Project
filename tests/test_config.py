"""Config invariants: loading, validation, role grammar, seed reproducibility."""
import json
import platform
from pathlib import Path

import pytest

from amcd.config import Config

from tests.conftest import (
    CANONICAL_DRY_RUN,
    QUIET,
    dry_run_simulator,
    tiny_config,
)


class TestConfigLoad:
    def test_load_base(self) -> None:
        cfg = Config.load(Path("configs/base.yaml"))
        assert cfg.sample_rate == 48000
        assert cfg.seeds.master == 42
        assert cfg.simulator.name == "gsound_sir"
        # Backend params come from configs/simulators/gsound_sir.yaml, not base.yaml.
        assert cfg.simulator.params["commit_sha"]
        assert cfg.model.name == "vanilla_cnn"

    def test_load_dry_run_overrides_base(self) -> None:
        cfg = Config.load(*CANONICAL_DRY_RUN)
        assert cfg.simulator.name == "dry_run"
        # F-11 scoping: switching the simulator name must not leave gsound_sir's
        # params attached to dry_run — dry_run gets ITS OWN params file, and none
        # of gsound's keys (which its schema would reject) come along.
        assert set(cfg.simulator.params) == {
            "speed_of_sound_m_s", "min_source_receiver_distance_m"}
        assert cfg.scenes.n_id == 20
        assert cfg.ir_duration == 0.25
        # dry_run declares its own small shift counts, overriding the base R1 defaults.
        assert cfg.splits["test_material_shift"].count == 3
        # model params come from configs/models/vanilla_cnn.yaml + inline override.
        assert cfg.model.params["hidden_channels"] == 8

    def test_all_six_splits_declared(self) -> None:
        cfg = Config.load(Path("configs/base.yaml"))
        assert set(cfg.splits) == {
            "train", "valid", "test_id",
            "test_material_shift", "test_placement_shift", "test_geometry_shift",
        }
        # test splits reported separately, never pooled (inv #9).
        assert set(cfg.test_split_names) == {
            "test_id", "test_material_shift", "test_placement_shift", "test_geometry_shift",
        }

    def test_derived_properties(self) -> None:
        cfg = Config.with_overrides(ambisonics_order=3, sample_rate=48000, ir_duration=3.0)
        assert cfg.n_channels == 16   # (3+1)^2
        assert cfg.n_samples == 144000  # 48000 * 3.0

    def test_frac_validation(self) -> None:
        with pytest.raises(Exception):
            # train 0.8 + valid 0.3 > 1.0 (id-pool fracs must sum < 1)
            Config.with_overrides(splits={"train": {"frac": 0.8}, "valid": {"frac": 0.3}})

    def test_shift_split_needs_single_axis(self) -> None:
        with pytest.raises(Exception):
            Config.with_overrides(splits={
                "test_material_shift": {
                    "role": "test", "count": 3,
                    "axes": {"material": "ceiling_absorptive", "geometry": "corridor"},
                }
            })

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            Config.with_overrides(nonexistent_param=42)

    def test_nonpositive_huber_delta_rejected(self) -> None:
        # F-07: δ ≤ 0 is a degenerate loss knee; reject at config load.
        with pytest.raises(Exception):
            Config.with_overrides(huber_delta=0.0)

    def test_nonnegative_onset_threshold_rejected(self) -> None:
        # metric_onset_rel_db is dB BELOW peak → must be negative.
        with pytest.raises(Exception):
            Config.with_overrides(metric_onset_rel_db=5.0)

    def test_nonpositive_bootstrap_resamples_rejected(self) -> None:
        # F-08: n_resamples <= 0 → degenerate CIs (the headline evidence). Fail at load.
        with pytest.raises(Exception):
            Config.with_overrides(bootstrap_n_resamples=0)

    def test_bootstrap_alpha_out_of_range_rejected(self) -> None:
        # F-08: alpha must be a (0,1) significance level.
        with pytest.raises(Exception):
            Config.with_overrides(bootstrap_alpha=1.5)

    def test_bootstrap_power_not_above_alpha_rejected(self) -> None:
        # F-17: power ≤ alpha makes mdes() silently return ≈0 (a zero effect already
        # "achieves" power = alpha) — reject the nonsensical config at load.
        with pytest.raises(Exception):
            Config.with_overrides(bootstrap_power=0.05, bootstrap_alpha=0.05)
        with pytest.raises(Exception):
            Config.with_overrides(bootstrap_power=0.03, bootstrap_alpha=0.05)

    def test_colliding_explicit_seeds_rejected(self) -> None:
        # F-04 / inv #5: two aspects sharing an explicit seed couples them.
        with pytest.raises(Exception):
            Config.with_overrides(
                seeds={"master": 42, "split_assignment": 7, "weight_init": 7}
            )

    def test_missing_model_config_raises(self) -> None:
        # An unknown plugin name has no params file and is not registered → fail loud
        # at load (F-12: the registry, not a file, is the source of truth for names).
        with pytest.raises(Exception):
            Config.with_overrides(model={"name": "no_such_model"})


class TestPluginSeam:
    """Representation/model `{name, params}` seam: name-scoped params (F-11) and
    registry-as-source-of-truth for name validity (F-12)."""

    def test_rep_params_do_not_bleed_across_name_switch(self) -> None:
        """F-11: switching representation.name on a layer that set the prior rep's
        params must NOT carry those params into the new rep. test_tiny sets
        spectrogram n_fft/hop_length; a waveform override must land with clean params
        and build, not fail with `n_fft` extra_forbidden deep in preprocess."""
        from tests.conftest import tiny_config
        from amcd.representations import build_representation

        cfg = tiny_config(representation={"name": "waveform"})
        assert cfg.representation.name == "waveform"
        assert cfg.representation.params == {}, (
            f"spectrogram params bled into waveform: {cfg.representation.params}"
        )
        # And it actually builds (the drop-in claim), no TypeError/ValidationError.
        rep = build_representation(
            cfg.representation.name, cfg.representation.params, sample_rate=cfg.sample_rate
        )
        assert type(rep).__name__ == "WaveformRepresentation"

    def test_registered_stub_needs_no_params_file(self) -> None:
        """F-12: `edr` is registered but ships no configs/representations/edr.yaml.
        Selecting it must load (empty params) and reach the stub's own
        NotImplementedError at build/use — NOT a FileNotFoundError at load."""
        import numpy as np
        from tests.conftest import tiny_config
        from amcd.representations import build_representation

        cfg = tiny_config(representation={"name": "edr"})
        assert cfg.representation.name == "edr" and cfg.representation.params == {}
        rep = build_representation(
            cfg.representation.name, cfg.representation.params, sample_rate=cfg.sample_rate
        )
        with pytest.raises(NotImplementedError):
            rep.encode(np.zeros((cfg.n_channels, 16), dtype=np.float32))

    def test_stamp_writes_files(self, tmp_path: Path) -> None:
        cfg = Config.load()
        cfg.stamp(tmp_path)
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "resolved.yaml").exists()
        assert (tmp_path / "versions.json").exists()


class TestVersionsJsonNamesTheMachine:
    """F-74: the compute device was auto-selected, never recorded, never stamped.

    `versions.json` recorded six package versions, the git sha, the dirty flag and
    `code_version` — and no device or host. Under the project's two-host
    requirement the SAME config and the SAME `code_version` produce different
    weights on this Mac (MPS) and on the x86 box (CUDA/CPU), with identical
    provenance stamps.
    """

    def _versions(self, tmp_path: Path) -> dict:
        """Stamps the way `cli.py` does — the device is SUPPLIED, not selected by
        `Config.stamp`, so that `config.py` (which is in `_CORE_SOURCES`) never
        imports `amcd.device` and drag it into every stage's cache key."""
        from amcd.device import select_device

        Config.load().stamp(tmp_path, device=str(select_device()))
        return json.loads((tmp_path / "versions.json").read_text())

    def test_the_device_is_recorded(self, tmp_path: Path) -> None:
        versions = self._versions(tmp_path)
        assert versions["device"] in {"mps", "cuda", "cpu"} or versions[
            "device"
        ].startswith(("cuda:", "mps:")), versions["device"]

    def test_the_host_architecture_is_recorded(self, tmp_path: Path) -> None:
        assert self._versions(tmp_path)["platform_machine"] == platform.machine()

    def test_it_names_the_device_the_checkpoint_was_trained_on(
        self, tmp_path: Path
    ) -> None:
        """The stamp must agree with the selector `train` and `infer` actually use,
        or the record describes a different run than the one on disk (F-56's rule,
        applied to the device)."""
        from amcd.device import select_device

        assert self._versions(tmp_path)["device"] == str(select_device())

    def test_the_device_is_in_no_fingerprint(self, tmp_path: Path) -> None:
        """Deliberately NOT a cache key: moving machines must not discard an
        expensive artifact."""
        from amcd.pipeline import STAGE_FINGERPRINT

        for stage, fingerprint in STAGE_FINGERPRINT.items():
            if fingerprint is None:
                continue
            payload = json.dumps(fingerprint(tiny_config()), default=str)
            assert "device" not in payload, stage

    def test_one_selector_serves_both_stages(self) -> None:
        """It was duplicated verbatim in trainer.py and infer.py, so the two could
        drift apart while both claimed to describe 'the device'."""
        from amcd.training import infer, trainer
        from amcd.device import select_device

        assert trainer.select_device is select_device
        assert infer.select_device is select_device


class TestConfigRootIsNotAssumedToBeASourceCheckout:
    """F-73: `configs/` was resolved three levels up from the module and nowhere
    else, so a wheel install into site-packages could not find `base.yaml` and
    failed with a bare FileNotFoundError deep inside `_merge_yaml`."""

    def test_a_missing_config_root_fails_actionably(self, monkeypatch) -> None:
        import amcd.config as config_mod

        missing = Path("/nonexistent-amcd-root/configs")
        monkeypatch.setattr(config_mod, "_CONFIG_ROOT_CANDIDATES", (missing,))
        monkeypatch.setattr(config_mod, "_BASE_YAML", missing / "base.yaml")

        with pytest.raises(FileNotFoundError) as excinfo:
            Config.load()
        message = str(excinfo.value)
        # The three things the operator needs: what is missing, where we looked,
        # and what would fix it.
        assert "configs/base.yaml" in message
        assert str(missing / "base.yaml") in message
        assert "pip install -e ." in message

    def test_resolution_never_raises_so_import_cannot_fail_on_layout(
        self, monkeypatch
    ) -> None:
        """`_CONFIGS_DIR` is computed at import. If resolution could raise, a bad
        layout would make `import amcd.config` itself explode — so the resolver
        returns a Path regardless and `_require_configs` reports the problem at
        load time, where the message can be actionable."""
        import amcd.config as config_mod

        monkeypatch.setattr(
            config_mod, "_CONFIG_ROOT_CANDIDATES", (Path("/nonexistent-amcd-root/configs"),)
        )
        resolved = config_mod._resolve_configs_dir()
        assert isinstance(resolved, Path)

    def test_no_candidate_points_somewhere_nothing_ships(self) -> None:
        """RD-109: a search path that can never match is not a provision.

        `amcd/configs/` was listed FIRST as a place a future packaged build might
        put `configs/`. No build ever did — `pyproject.toml` declares no package
        data and `configs/` sits outside the package directory — so its only effect
        was to head the "Tried:" list with a directory that cannot exist, while the
        error text told the operator to install a wheel that would not have worked.

        Asserted as a PROPERTY, not against the current list: every candidate must
        be somewhere `configs/` is actually delivered. If package data is ever
        shipped, this test passes with the entry restored.
        """
        import amcd.config as config_mod

        unshippable = [
            c for c in config_mod._CONFIG_ROOT_CANDIDATES if not (c / "base.yaml").is_file()
        ]
        assert unshippable == [], (
            f"config root candidate(s) {unshippable} hold no base.yaml in this "
            f"installation. A candidate no build populates makes the resolver "
            f"claim support for a layout that does not work (RD-109)."
        )

    def test_the_resolved_root_actually_holds_base_yaml(self) -> None:
        import amcd.config as config_mod

        assert (config_mod._CONFIGS_DIR / "base.yaml").is_file()


class TestFingerprintExemptionsAreWellFormed:
    """F-65's exemption table is a CLAIM about the fingerprint contract, so it has
    to be checkable itself — an exemption naming a field that no longer exists
    would silently widen the guard's blind spot."""

    def test_every_exempt_name_is_a_real_config_field(self) -> None:
        from amcd.pipeline import FINGERPRINT_EXEMPT_FIELDS

        unknown = set(FINGERPRINT_EXEMPT_FIELDS) - set(Config.model_fields)
        assert unknown == set(), (
            f"{sorted(unknown)} are exempted from fingerprinting but are not "
            f"Config fields"
        )

    def test_every_exemption_says_what_would_end_it(self) -> None:
        """A present-tense fact ('nothing consumes it') reads to a later editor as
        permission to delete the field. Each reason must point somewhere."""
        from amcd.pipeline import FINGERPRINT_EXEMPT_FIELDS

        for field, reason in FINGERPRINT_EXEMPT_FIELDS.items():
            assert any(
                marker in reason
                for marker in ("Non-exempt", "non-exempt", "fingerprinted through")
            ), f"exemption for {field!r} does not say what would make it non-exempt"


class TestRoleGrammar:
    """design_spec §7: fixed/tuned/swept leaves resolve to a concrete point."""

    def test_tuned_resolves_to_value(self) -> None:
        cfg = Config.with_overrides(
            learning_rate={"tune": {"space": [1.0e-4, 1.0e-2], "scale": "log"}, "value": 0.003}
        )
        assert cfg.learning_rate == 0.003
        assert cfg.resolved_roles["learning_rate"]["role"] == "tuned"

    def test_tuned_without_value_raises(self) -> None:
        with pytest.raises(Exception):
            Config.with_overrides(learning_rate={"tune": {"space": [1.0e-4, 1.0e-2]}})

    def test_sweep_expands_to_siblings(self) -> None:
        cfg = Config.with_overrides(low_ray_budget={"sweep": [1000, 5000, 20000]})
        assert cfg.low_ray_budget == 1000  # single-run selects index 0
        siblings = cfg.expand_sweeps()
        assert [s.low_ray_budget for s in siblings] == [1000, 5000, 20000]

    def test_tuned_value_outside_space_rejected(self) -> None:
        # F-05: a tuned operating point outside its declared space must fail loudly.
        with pytest.raises(Exception):
            Config.with_overrides(
                learning_rate={"tune": {"space": [1.0e-4, 1.0e-2]}, "value": 0.5}
            )

    def test_sweep_with_stray_value_rejected(self) -> None:
        # F-05: a stray `value` on a sweep node is silently ignored otherwise.
        with pytest.raises(Exception):
            Config.with_overrides(
                low_ray_budget={"sweep": [1000, 5000], "value": 1000}
            )

    def test_bool_tuned_value_rejected(self) -> None:
        # P2-01: bool is an int subclass; reject it so `value: true` can't coerce to 1.
        with pytest.raises(Exception):
            Config.with_overrides(
                learning_rate={"tune": {"space": [0, 1]}, "value": True}
            )


class TestSeedReproducibility:
    """Invariant #5: same config + seed → reproducible outputs."""

    def test_scene_generation_reproducible(self) -> None:
        import json
        from amcd.scenes.generator import run_gen_scenes
        import tempfile, os

        cfg = Config.with_overrides(
            scenes={"n_id": 5},
            seeds={"master": 42},
            simulator={"name": "dry_run"},
            splits={
                "test_material_shift": {"role": "test", "count": 1, "axes": {"material": "ceiling_absorptive"}},
                "test_placement_shift": {"role": "test", "count": 1, "axes": {"placement": "near_corner"}},
                "test_geometry_shift": {"role": "test", "count": 1, "axes": {"geometry": "corridor"}},
            },
        )

        def _load_sorted(path: Path) -> list:
            return sorted(
                (json.loads(f.read_text()) for f in path.glob("scene_*.json")
                 if not f.name.startswith("._")),
                key=lambda d: d["scene_id"],
            )

        with tempfile.TemporaryDirectory() as d1:
            p1 = Path(d1)
            run_gen_scenes(cfg, p1, QUIET)
            scenes1 = _load_sorted(p1 / "scenes")

        with tempfile.TemporaryDirectory() as d2:
            p2 = Path(d2)
            run_gen_scenes(cfg, p2, QUIET)
            scenes2 = _load_sorted(p2 / "scenes")

        assert scenes1 == scenes2, "Scene generation is not reproducible"

    def test_simulator_deterministic(self) -> None:
        import numpy as np
        from amcd.simulators.base import SceneSpec

        sim = dry_run_simulator(n_channels=16, n_samples=4800, sample_rate=48000)
        scene = SceneSpec(
            scene_id="s0", seed=99,
            geometry_family="shoebox",
            dims=(5.0, 4.0, 3.0),
            material_absorption=0.3,
            source_pos=(1.0, 1.0, 1.5),
            receiver_pos=(4.0, 3.0, 1.5),
        )
        ir1 = sim.render(scene, ray_budget=1000).ir
        ir2 = sim.render(scene, ray_budget=1000).ir
        assert np.array_equal(ir1, ir2), "DryRunSimulator is not deterministic"

    def test_model_deterministic(self) -> None:
        import torch
        from amcd.models.cnn import CNNDenoisingModel

        torch.manual_seed(42)
        model = CNNDenoisingModel(n_channels=16, hidden_channels=8, n_layers=2)
        model.eval()

        x = torch.randn(1, 16, 30, 20)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2), "Model forward is not deterministic"


class TestSplitRoleVocabulary:
    """F-44: `role` routes a split through the entire pipeline, so an unrecognised
    role must fail at config load — not silently produce a split that is generated,
    rendered and preprocessed and then appears in no result."""

    def test_unknown_role_rejected_at_load(self) -> None:
        with pytest.raises(ValueError) as exc:
            tiny_config(splits={"test_id": {"role": "tset"}})
        msg = str(exc.value)
        assert "tset" in msg and "test_id" in msg
        # The message must name the vocabulary and where it lives, so the fix is
        # findable without reading the validator (house pattern).
        assert "SPLIT_ROLES" in msg
        assert "train" in msg and "valid" in msg and "test" in msg

    def test_role_typo_is_caught_before_any_expensive_stage(self) -> None:
        """The whole point of moving this to config load: the failure must happen
        before gen-scenes/render, which under research_i.yaml is 60 emulated renders."""
        from amcd.config import Config
        with pytest.raises(ValueError):
            tiny_config(splits={"test_material_shift": {"role": "holdout"}})
        # And a valid config still loads (the guard is not simply rejecting everything).
        cfg = tiny_config()
        assert isinstance(cfg, Config)
        assert cfg.the_split_with_role("train") == "train"
        assert cfg.the_split_with_role("valid") == "valid"

    def test_missing_valid_split_rejected(self) -> None:
        """Pre-fix this surfaced as a bare StopIteration with an empty message, and
        only after render + preprocess had already run."""
        with pytest.raises(ValueError) as exc:
            tiny_config(splits={"valid": {"role": "test", "frac": None, "count": 2,
                                          "seed": 7}})
        assert "valid" in str(exc.value)
        assert "REQUIRED_ROLE_COUNTS" in str(exc.value)

    def test_two_train_splits_rejected(self) -> None:
        """Pre-fix the trainer silently took whichever came first."""
        with pytest.raises(ValueError) as exc:
            tiny_config(splits={"test_id": {"role": "train"}})
        assert "train" in str(exc.value)

    def test_split_names_with_role_rejects_unknown_role(self) -> None:
        """A typo at a CALL SITE must raise rather than return an empty tuple, which
        would read as 'no such splits declared'."""
        cfg = tiny_config()
        with pytest.raises(ValueError) as exc:
            cfg.split_names_with_role("tset")
        assert "SPLIT_ROLES" in str(exc.value)
