"""
GSound-SIR render backend — parameter schema (Step 1) ahead of the worker (Step 3).

The backend runs under an x86 interpreter (Rosetta on Apple Silicon, native on
Ubuntu/Windows); that host boundary lives entirely behind this seam and in
environment setup, never in package code (see docs/gsound_sir_setup.md).

This module currently declares only the config contract: `Params` validates
configs/simulators/gsound_sir.yaml so every backend value is config-declared and
typo-proof before any rendering exists. `render()` raises until Step 3 wires the
subprocess worker.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator

from ..registry import simulator_registry
from .base import IRResult, SceneSpec


#: Number of octave bands pygsound's `Context` simulates (63 Hz … 8 kHz centres,
#: ray_generator/src/pygsound/src/Context.cpp:8). `frequency_points` are the
#: CROSSOVERS between them, hence n_bands - 1.
_N_BANDS = 8


@simulator_registry.register("gsound_sir")
class GsoundSirSimulator:
    """Production simulator using GSound-SIR at a config-pinned upstream commit."""

    class Params(BaseModel):
        """gsound_sir's own config schema (design_spec §5/§8).

        Every value here is declared in configs/simulators/gsound_sir.yaml — none
        has a default, so a missing key fails at config load rather than silently
        taking a backend default that no artifact records.
        """
        model_config = {"extra": "forbid"}

        #: Upstream yongyizang/GSound-SIR commit the render env must be built from.
        #: Step 3 hard-errors if the installed SHA differs (reproducibility pin).
        commit_sha: str

        #: Ray counts. `low_ray_budget`/`high_ray_budget` (top-level Config) drive
        #: gsound's DIFFUSE count; specular is declared here and held FIXED across
        #: both legs, so the swept axis is unambiguously the diffuse budget (RD-12).
        specular_count: int
        diffuse_depth: int
        specular_depth: int

        #: Source/listener sphere radii (m) and source power, per gsound's API.
        source_radius: float
        listener_radius: float
        source_power: float

        #: Surface scattering coefficient passed to `createbox`. Scene specs carry
        #: absorption; scattering is a backend-level material property today.
        scattering: float

        #: Filterbank band EDGES (Hz) for SH synthesis — n_bands-1 crossovers, NOT
        #: band centres. See the yaml for the derivation and the upstream trap.
        frequency_points: list[float]

        #: Must stay false: per-IR normalization destroys low↔high energy
        #: comparability, which the paired-improvement spine and the D0b carrier
        #: test both rest on.
        normalize_ir: bool

        #: Which paths are written to the retained-path artifact. Applies ONLY to
        #: that artifact — IR synthesis always uses the full path set, or the
        #: ray-budget axis under study would be confounded.
        path_retention: "PathRetention"

        @field_validator("normalize_ir")
        @classmethod
        def _reject_normalization(cls, v: bool) -> bool:
            if v:
                raise ValueError(
                    "normalize_ir must be false: per-IR normalization destroys the "
                    "low↔high energy comparability that every paired-improvement "
                    "metric and the D0b carrier test depend on."
                )
            return v

        @field_validator("frequency_points")
        @classmethod
        def _check_band_edges(cls, v: list[float]) -> list[float]:
            if len(v) != _N_BANDS - 1:
                raise ValueError(
                    f"frequency_points must be {_N_BANDS - 1} crossover frequencies "
                    f"(band EDGES for {_N_BANDS} bands), got {len(v)}. Upstream's "
                    f"CrossoverFilter enforces n_bands-1; passing band CENTRES (as "
                    f"upstream auralizer/test.py does) shifts every edge ~½ octave."
                )
            if any(b <= a for a, b in zip(v, v[1:])):
                raise ValueError(f"frequency_points must be strictly increasing; got {v}")
            return v

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sample_rate: int,
        **params,
    ) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.params = params

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        raise NotImplementedError(
            "GSound-SIR rendering lands in Step 3 (subprocess worker). The config "
            "contract is in place; use --simulator dry_run for pipeline testing. "
            "Env setup: docs/gsound_sir_setup.md."
        )


class PathRetention(BaseModel):
    """Which simulated paths reach the saved retained-path artifact.

    Maps directly onto upstream `getPathData(energy_percentage=…, max_rays=…)`:
      all          → energy_percentage 100, max_rays 0
      top_percent  → energy_percentage = value
      top_k        → max_rays = value
    Research I used top_k with k = 5,000 (Figure 5).
    """
    model_config = {"extra": "forbid"}

    mode: str
    value: float | None

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in ("all", "top_percent", "top_k"):
            raise ValueError(f"path_retention.mode must be all|top_percent|top_k; got {v!r}")
        return v

    def model_post_init(self, _context) -> None:
        if self.mode == "all":
            if self.value is not None:
                raise ValueError("path_retention.mode 'all' takes no `value` (set it null)")
        elif self.value is None:
            raise ValueError(f"path_retention.mode {self.mode!r} requires a `value`")


GsoundSirSimulator.Params.model_rebuild()
