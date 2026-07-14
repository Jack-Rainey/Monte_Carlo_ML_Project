"""Shared fixtures for amcd tests.

Test configuration is loaded from `configs/test_tiny.yaml` rather than hardcoded
in Python — the same "config is the source of truth, no defaults in code" rule the
pipeline follows. Tests that need a variant deep-merge overrides onto it via
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
TEST_TINY = _REPO_ROOT / "configs" / "test_tiny.yaml"

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
    for path in (_BASE_YAML, TEST_TINY):
        with open(path) as f:
            _merge_layer(merged, yaml.safe_load(f) or {})
    _merge_layer(merged, overrides)
    return Config._from_merged(merged, None)


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
