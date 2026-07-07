"""GSound-SIR simulator stub. Requires x86 (invalid processor on Apple Silicon ARM)."""
from ..registry import simulator_registry
from .base import IRResult, SceneSpec


@simulator_registry.register("gsound_sir")
class GsoundSirSimulator:
    """
    Production simulator using GSound-SIR.

    Requires x86 — run under Rosetta or in a dedicated osx-64 conda env:
        CONDA_SUBDIR=osx-64 conda create -n amcd_render ...
        arch -x86_64 amcd render --config configs/base.yaml

    The x86 boundary lives entirely behind this seam; all other stages run
    native arm64 + MPS.
    """

    def __init__(self, n_channels: int, n_samples: int, sample_rate: int) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sample_rate = sample_rate

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        raise NotImplementedError(
            "GSound-SIR is not yet integrated. "
            "Use --simulator dry_run for pipeline testing, or run this stage "
            "under Rosetta (arch -x86_64) in an osx-64 conda env."
        )
