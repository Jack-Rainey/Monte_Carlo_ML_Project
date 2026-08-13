"""Shared fixtures for amcd tests.

Test configuration is COMPOSED FROM THREE CONFIG LAYERS rather than hardcoded in
Python — the same "config is the source of truth, no defaults in code" rule the
pipeline follows:

  SIMULATOR_DRY_RUN   configs/overlays/simulator_dry_run.yaml — selects the
                      scaffold backend, so no test needs GSound-SIR.
  TEST_TINY           configs/overlays/test_tiny.yaml — the small framing
                      (few channels, low rate, short record) that keeps the
                      suite fast.
  CANONICAL_DRY_RUN   base + simulator_dry_run + dry_run: the full canonical
                      dry run, for tests that must exercise what an operator
                      actually runs.

Tests that need a variant deep-merge overrides onto the composition via
`tiny_config(**overrides)`.
"""
from pathlib import Path

import pytest
import numpy as np
import yaml

from amcd.config import Config, _BASE_YAML, _merge_layer
from amcd.runtime import Verbosity
from amcd.simulators.base import SceneSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVERLAYS = _REPO_ROOT / "configs" / "overlays"
#: The suite composes the same layers a dry-run invocation does, in the same
#: order, so the backend is declared in exactly one file repo-wide (RD-51) and
#: tests cannot pass against a switch that has drifted from the real one.
SIMULATOR_DRY_RUN = _OVERLAYS / "simulator_dry_run.yaml"
TEST_TINY = _OVERLAYS / "test_tiny.yaml"

#: The canonical dry-run invocation as a layer list, so a test that means "the
#: documented dry run" says so once instead of restating three paths that could
#: fall out of step with the README and the F-51 error message.
CANONICAL_DRY_RUN = (_BASE_YAML, SIMULATOR_DRY_RUN, _OVERLAYS / "dry_run.yaml")

#: The reported metric band set, read from the config that governs it (RD-187).
#:
#: `build_representation` takes it as a cross-cutting argument beside `sample_rate`,
#: so a probe that wants a representation has to supply one. Read rather than
#: written as `[500.0, 1000.0]`: the whole point of RD-187 is that this set is
#: declared ONCE, and a test-file literal would be the third declaration.
EVAL_FREQS: list[float] = [
    float(f) for f in yaml.safe_load(_BASE_YAML.read_text())["iso_eval_freqs"]
]

#: The layers `tiny_config()` composes, for tests that must go through the real
#: `Config.load` or the CLI rather than the in-process merge.
TINY_LAYERS = (SIMULATOR_DRY_RUN, TEST_TINY)


def tiny_cli_args() -> list[str]:
    """`-c <layer>` arguments for the tiny config, in composition order."""
    args: list[str] = []
    for path in (_BASE_YAML, *TINY_LAYERS):
        args += ["-c", str(path)]
    return args

# Stage functions require an explicit Verbosity (no default outside the CLI
# layer — RD-09). Verbosity is a runtime output level, never an
# experiment-governing value (CLAUDE.md), so a shared constant here does not
# violate the no-values-in-conftest rule. save=0 doubles as a standing check
# of the F-23 guarantee: every canonical artifact tests rely on must exist at
# the lowest save level.
QUIET = Verbosity(save=0, show=0)


def tiny_config(**overrides) -> Config:
    """The tiny test config (base + test_tiny), optionally deep-merged with overrides.

    Overrides mirror YAML structure, e.g. tiny_config(scenes={"n_id": 5}).
    """
    merged: dict = {}
    for path in (_BASE_YAML, SIMULATOR_DRY_RUN, TEST_TINY):
        with open(path) as f:
            _merge_layer(merged, yaml.safe_load(f) or {})
    _merge_layer(merged, overrides)
    return Config._from_merged(merged, None)


def dry_run_simulator(*, n_channels: int, n_samples: int, sample_rate: int):
    """Build the dry_run simulator through the registry seam.

    Tensor shape is test scaffolding and stays explicit here, but the backend's
    PARAMETERS (e.g. speed_of_sound_m_s) come from configs/simulators/dry_run.yaml
    — never hardcoded in a fixture, and never constructed around the seam, so
    these tests keep exercising the same `build_simulator` path the render stage
    uses.
    """
    from amcd.simulators.base import build_simulator

    return build_simulator(
        "dry_run",
        tiny_config().simulator.params,
        n_channels=n_channels,
        n_samples=n_samples,
        sample_rate=sample_rate,
    )


@pytest.fixture
def dry_run_config() -> Config:
    return tiny_config()


@pytest.fixture
def sample_scene() -> SceneSpec:
    return SceneSpec(
        scene_id="scene_0000",
        seed=12345,
        geometry_family="shoebox",
        dims=(5.0, 4.0, 3.0),
        material_absorption=0.3,
        source_pos=(1.0, 1.0, 1.5),
        receiver_pos=(4.0, 3.0, 1.5),
    )


@pytest.fixture
def sample_ir(dry_run_config: Config) -> np.ndarray:
    """Random (C, T) float32 IR sized from the tiny test config."""
    rng = np.random.default_rng(0)
    C = dry_run_config.n_channels
    T = dry_run_config.n_samples
    return rng.standard_normal((C, T)).astype(np.float32)
