"""
GSound-SIR render backend — the config contract (Step 1) and the subprocess
worker behind it (Step 3).

The backend runs under an x86 interpreter (Rosetta on Apple Silicon, native on
Ubuntu/Windows); that host boundary lives entirely behind this seam and in
environment setup, never in package code (see docs/gsound_sir_setup.md). There is
no `platform`/arch branch here: the ONLY thing that differs between hosts is the
`render_python` config value, and on a host whose own interpreter can import
pygsound that value is null and the worker runs under `sys.executable`. One code
path, both hosts.

`Params` validates configs/simulators/gsound_sir.yaml so every backend value is
config-declared and typo-proof; `render()` drives the worker and returns an
`IRResult` carrying the IR, the retained `PathData` and this leg's provenance.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator

from ..acoustics import box_volume_and_surface, predicted_support_s, sabine_rt60
from ..registry import simulator_registry
from .base import IRResult, PathData, SceneSpec


#: Number of octave bands pygsound's `Context` simulates (63 Hz … 8 kHz centres,
#: ray_generator/src/pygsound/src/Context.cpp:8). `frequency_points` are the
#: CROSSOVERS between them, hence n_bands - 1.
_N_BANDS = 8

#: Install receipt `scripts/setup_gsound_sir.py` writes into the render env's
#: site-packages, and the JSON key holding the upstream commit it built.
#:
#: DUPLICATED, deliberately: the other definition is
#: `scripts/setup_gsound_sir.py:78` (`RECEIPT_NAME`) and `:300` (`read_receipt`),
#: which package code CANNOT import — `scripts/` has no `__init__.py`, is not
#: installed, and is not on the `PYTHONPATH` the pipeline runs under. Reaching it
#: would take a sys.path hack, i.e. an assumption about repo layout inside package
#: code. Two small constants beat that.
#:
#: SENT TO THE WORKER IN ITS REQUEST (F-123). The worker cannot import `amcd`, so
#: it used to inline both names — a third copy of each. It now reads them from the
#: JSON request it already receives, leaving `scripts/setup_gsound_sir.py` as the
#: only other definition, and that one is unavoidable: package code cannot import
#: `scripts/`, which has no `__init__.py` and is not installed.
_RECEIPT_NAME = "amcd_gsound_install.json"
_RECEIPT_SHA_KEY = "commit_sha"

#: The ambisonic channel ordering + normalization upstream actually produces.
#: **N3D, not SN3D** — verified in the auralizer binding at
#: `auralizer/src/cpp/binding.cpp:18` ("normalization constant K(l, m) for N3D")
#: and `:43` ("N3D/ACN ordering"), re-checked against the pinned SHA before this
#: value was first stamped (AC-15). Getting it wrong is a per-degree sqrt(2l+1)
#: error, invisible while every live scalar metric reads channel 0 (where N3D and
#: SN3D agree exactly) and load-bearing the moment evaluation/spatial.py is filled
#: in (RD-25).
_AMBISONIC_CONVENTION = "acn_n3d"

#: Whether the m != 0 channels carry a Condon-Shortley (-1)^|m| phase on top of
#: ACN/N3D. **MEASURED, not read off the source comments** (AC-57): a single
#: synthetic path pushed through `generate_ambisonic_ir` at order 1, with no
#: propagation involved, gives channel-to-W ratios of
#:
#:     +x -> [1, 0, 0, -1.7321]      +y -> [1, -1.7321, 0, 0]
#:     +z -> [1, 0, +1.7321, 0]
#:
#: The magnitude sqrt(3) confirms N3D (SN3D would be 1.0) and the position of the
#: non-zero entry confirms ACN ordering (W, Y, Z, X) — so AC-15's stamp is right
#: about both. But X and Y are NEGATED relative to W while Z is not, which is
#: exactly (-1)^|m|, and negating X and Y together is a **180 degree yaw**.
#:
#: ABSOLUTE SCALE: upstream's SH normalization constant is the ORTHONORMAL one,
#: Y_00 = 1/sqrt(4pi) = 0.28209, where textbook N3D has Y_00 = 1. That is a single
#: global gain on all channels, so every ratio — which is all any ISO metric or a
#: DOA estimate reads — is unaffected, and no absolute level here is calibrated
#: against a reference pressure. Recorded, not corrected: renormalizing would
#: rewrite every stored IR to fix nothing (AC-57).
#:
#: The OBSERVED |W| is NOT that constant: the late field is
#: `result[c, t] = normalized_sh[c] * carrier[t]` (binding.cpp:423), so a sample of
#: the synthesis noise carrier multiplies it (AC-75). The ratios above are therefore
#: the only absolute-scale-free facts here, which is exactly why they are what the
#: known-answer test asserts.
#:
#: Stamped separately rather than folded into the convention string because the
#: string names ordering + normalization, which are genuinely acn_n3d; the phase
#: is an additional fact about the same data. It is inert today — every live
#: scalar metric reads channel 0, where it has no effect — and becomes
#: load-bearing the moment evaluation/spatial.py estimates a direction, which
#: would otherwise come out 180 degrees wrong in azimuth and look like a bug in
#: the estimator rather than in the encoding (RD-25).
_SH_CONDON_SHORTLEY_PHASE = True

#: Seed of the noise carrier the SH synthesis builds every IR on. Compiled into
#: upstream and not settable, so it is DECLARED here, read from the pinned source
#: rather than assumed: `NoiseGenerator(unsigned int seed = 42)`
#: (auralizer/src/cpp/binding.cpp:141), constructed defaulted as `NoiseGenerator
#: noise_gen;` at `:329` — verified against SHA 608ea30f, the value
#: configs/simulators/gsound_sir.yaml pins.
#:
#: Stamped because it is LOAD-BEARING (AC-59): the carrier is one realization of a
#: random process, and a single realization moves T30 by ~2.5% and C50 by ~1 dB —
#: about one JND.
#:
#: WHAT IS ESTABLISHED: the seed is fixed, so both legs of a scene are built on the
#: IDENTICAL carrier sequence, indexed identically.
#:
#: WHAT IS NOT: that the induced METRIC error therefore cancels in a paired
#: comparison. It does not follow — the two legs occupy different delay bins, so
#: they weight the same carrier differently, and bin occupancy is driven by the ray
#: budget, which is the swept axis itself. A model puts the residual at sd ~0.45 dB
#: on the paired C50 delta, ~45% of `d0b_c50_jnd_db` (AC-76, OPEN, with a
#: zero-render experiment that settles it).
#:
#: This value is also UNFALSIFIABLE from here (AC-77/F-124): it is a literal in
#: amcd's own source, so provenance emits 42 whatever upstream does, and a
#: provenance diff cannot move. Unlike `speed_of_sound_m_s` (cross-checked against
#: the paths, RD-19) and `ambisonic_convention` (measured, F-93), it rests entirely
#: on the `commit_sha` pin.
_SYNTHESIS_CARRIER_SEED = 42

#: How this backend turns an alpha handed to `createbox` into the room it builds.
#: Its per-bounce ENERGY factor is `sqrt(1-alpha)` where the declared physics wants
#: `(1-alpha)` (AC-54, re-derived from pinned upstream), so the absorption it
#: realizes is always LOWER than the one it is given.
def _gsound_realizes(createbox_alpha: float) -> float:
    return 1.0 - math.sqrt(1.0 - createbox_alpha)


def _check_convention(convention: str) -> str:
    if convention not in ("pre_compensate", "as_is"):
        raise ValueError(
            f"absorption_convention {convention!r} is neither 'pre_compensate' nor "
            "'as_is'. It decides which room is rendered from a scene's declared "
            "alpha, so there is no default (AC-54, RD-144)."
        )
    return convention


def _createbox_absorption(alpha_nominal: float, convention: str) -> float:
    """The alpha to HAND `createbox`, given the scene's nominal one.

    Under `pre_compensate` that is `1-(1-alpha)^2`, chosen so what the room realizes
    equals what the scene declared:

        1 - sqrt(1 - (1 - (1-a)^2)) = 1 - sqrt((1-a)^2) = a

    `as_is` passes the nominal value straight through and renders the uncorrected
    room, whose T60 runs 1.14-1.98x longer across `base.yaml`'s declared support. It
    is not a fallback — it is what every measurement before AC-54 was taken under,
    and it must stay reachable so those numbers remain reproducible.

    NOT THE REALIZED ABSORPTION, and the two must not be confused (AC-50): under
    `pre_compensate` this returns 0.5100 for a nominal 0.30 while the room realizes
    0.30, and a closed form evaluated at the wrong one of the two describes a room
    that was never rendered. `realized_absorption` below is the other quantity.
    """
    _check_convention(convention)
    if convention == "pre_compensate":
        return 1.0 - (1.0 - alpha_nominal) ** 2
    return alpha_nominal


#: The worker's SOURCE, read from the module beside this one. It is shipped as text
#: rather than imported because it runs under an interpreter where `amcd` DOES NOT
#: EXIST: the render env holds numpy, pygsound and spherical_harmonics_rt and
#: nothing else. It imports no part of this package and communicates by files — a
#: JSON request in, `ir.npy` + `paths.npz` + `result.json` out — so nothing has to be
#: pickle-compatible across two Pythons of different architectures.
#:
#: It was a 211-line raw string inside this module until the file-ownership rule
#: that forced that dissolved (RR-94). Binding the name to the file's TEXT keeps
#: every consumer unchanged — `_run_worker` still writes it out, and the compile,
#: AST-import and stub-execution tests still read it — while the worker becomes
#: something an editor can lint, a debugger can step, and a diff can show.
_WORKER_SRC = (Path(__file__).with_name("_gsound_worker.py")).read_text(encoding="utf-8")



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

        #: The band CENTRES those edges sit between (Context.cpp:8). Declared rather
        #: than derived: 7 edges do not determine 8 centres without an anchor. They
        #: are not free — `_check_edges_match_centres` requires each edge to be the
        #: geometric mean of its neighbours, exactly as `gs::FrequencyBands` derives
        #: them (gsFrequencyBands.cpp:83-88) — so this is a falsifiable declaration
        #: of a compiled-in fact, not a second tunable. Needed because RD-24 requires
        #: a path file to NAME the frequency of each `intensities` column.
        band_centres_hz: list[float]

        #: gsound's propagation speed. Compiled into C++ and NOT settable (RD-19),
        #: so it is DECLARED here and then cross-checked at render against the
        #: `speeds_of_sound` the paths themselves report — a declaration that would
        #: hard-error if upstream ever changed it, rather than a comment.
        speed_of_sound_m_s: float

        #: RD-21/RD-67's QC threshold, in dB: flag a leg whose discarded tail energy
        #: relative to the native IR's total energy exceeds this. Trimming gsound's
        #: natural IR to `ir_duration` can silently invalidate T30/EDT on the most
        #: reverberant scenes, so the loss is measured and disclosed per (scene, leg).
        max_discarded_tail_db: float

        #: Interpreter that can `import pygsound` — HOST-SCOPED, not an experiment
        #: value. `null` means "this interpreter" (`sys.executable`), which is
        #: correct on a native x86_64 host; on a host where the render env is a
        #: separate x86 interpreter (e.g. Rosetta on Apple Silicon), a config layer
        #: supplies its absolute path. Deliberately kept OUT of the canonical
        #: provenance echo (see `render._canonical_meta`): a machine-local path is
        #: not a property of the dataset, and stamping it would make the same render
        #: differ between hosts. Migrates to `RunContext.host` when RD-20 lands.
        render_python: str | None

        #: HOW THIS BACKEND REALIZES A DECLARED ABSORPTION (AC-54, RD-144).
        #:
        #: GSound sets material reflectivity to `sqrt(1-alpha)` as an AMPLITUDE
        #: coefficient (`SoundMesh.cpp:221`) and then accumulates it into a
        #: quantity it calls `energy`, which the synthesizer takes `sqrt()` of
        #: (`binding.cpp:417,454`). The `1/d` pressure law coming out correct is
        #: what forces the reading: per-bounce ENERGY carries `sqrt(1-alpha)`, so
        #: the realized absorption is `alpha_eff = 1 - sqrt(1 - alpha)`.
        #:
        #: DECLARED ON THE BACKEND, never in `scenes/generator.py` (RD-144): the
        #: generator defines the dataset's acoustics for EVERY simulator, so
        #: re-deriving its closed forms from one raytracer's domain confusion
        #: would make a second raytracer render a different room from the same
        #: scene spec, with nothing declaring it — and would foreclose the
        #: roadmap's multiple-raytracers item.
        #:
        #: `pre_compensate` passes `1-(1-alpha)^2` at the `createbox` call site so
        #: the room REALIZES the scene's declared alpha; `as_is` renders the
        #: uncorrected room and is what every existing measurement was taken
        #: under. No default — this is experiment-governing.
        absorption_convention: Literal["pre_compensate", "as_is"]

        #: THE RECORD THIS BACKEND CAN FILL — the compiled cap (AC-175, AC-56).
        #: `maxIRLength` is compiled at 3.0 s and not exposed by `module.cpp`, so
        #: `ir_duration` cannot govern the native record and must be validated
        #: against this rather than assumed. Declared, not configured: setting a
        #: different value does not change the simulation.
        max_ir_length_s: float

        #: THE LIMIT THAT ACTUALLY BINDS (AC-184). Upstream's adaptive energy trim
        #: closes the record long before the compiled cap — measured across the
        #: retained renders, T60 swept 45.2x while the record grew only 2.89x — so
        #: realized support is a FUNCTION of the scene's decay, not a constant:
        #:
        #:     support_s = coefficient * T60_eff ** exponent
        #:
        #: A constant number of T60-multiples is REFUTED by that data (the measured
        #: multiples span 5.027 / 1.564 / 0.235). The sub-unity exponent is the
        #: finding itself — support grows more slowly than the decay it must hold.
        #:
        #: gen-scenes gates on the PREDICTION, because it runs before any render
        #: exists; `render` then falsifies it against realized `native_ir_samples`
        #: and reports the residual, exactly as `speed_of_sound_m_s` is falsified
        #: against the paths. A prediction that over-reads is a defect HERE, and the
        #: render is where it surfaces — which is what makes an n=3 fit safe to
        #: ship rather than a guess nobody re-checks.
        predicted_support_coefficient_s: float
        predicted_support_depth_exponent: float
        predicted_support_surface_exponent: float
        #: Declared so the law states its own operating point: realized support
        #: rises with the ray budget, so a coefficient carries an implicit budget
        #: unless it says which one it was fitted at (AC-185).
        predicted_support_fitted_at_ray_budget: int

        #: AIR ABSORPTION IS REALIZED AT alpha_ISO/4 (AC-66) — AC-54's domain
        #: confusion at a second call site, which pre-compensating surface alpha
        #: does not reach. Compiled ON and not exposed, so it is DECLARED and
        #: guarded rather than corrected: inert at 500/1000 Hz (<= 0.4% of T60),
        #: ~19% at 8 kHz. `iso_eval_freqs` above the max below is refused.
        precise_early_reflections: bool
        early_reflection_threshold: float
        air_absorption_realized_fraction: float
        air_absorption_max_eval_freq_hz: float
        air_absorption_t60_error_tolerance_frac: float

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

        @field_validator("band_centres_hz")
        @classmethod
        def _check_centre_count(cls, v: list[float]) -> list[float]:
            if len(v) != _N_BANDS:
                raise ValueError(
                    f"band_centres_hz must be {_N_BANDS} octave band centres, got {len(v)}"
                )
            return v

        def model_post_init(self, _context) -> None:
            """Edges and centres must describe ONE filterbank.

            Upstream derives each crossover as the geometric mean of adjacent centres
            (gsFrequencyBands.cpp:83-88). Checking that here is what makes
            `band_centres_hz` a declaration of a compiled-in fact rather than a
            second, independently-settable band definition that could drift from the
            edges actually used for synthesis — the AC-12 failure mode (88.4 vs
            88.7412) one level up.
            """
            expected = [
                (a * b) ** 0.5 for a, b in zip(self.band_centres_hz, self.band_centres_hz[1:])
            ]
            off = [
                (i, e, g) for i, (e, g) in enumerate(zip(self.frequency_points, expected))
                if abs(e - g) > 1e-3 * g
            ]
            if off:
                raise ValueError(
                    f"frequency_points and band_centres_hz describe different "
                    f"filterbanks. Each edge must be the geometric mean of its "
                    f"adjacent centres; disagreements at (index, declared, expected): "
                    f"{off}."
                )

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

    @classmethod
    def min_source_receiver_distance_m(cls, params: dict) -> float:
        """Required pre-render declaration (`Simulator`).

        DERIVED, never separately declared: below `source_radius + listener_radius`
        gsound's source and listener spheres physically overlap, so the floor is
        already implied by two values the config states. A third config field
        holding the same fact could disagree with the geometry it describes —
        the divergence AC-24 was raised about.
        """
        return float(params["source_radius"]) + float(params["listener_radius"])

    @classmethod
    def realized_absorption(cls, params: dict, alpha_nominal: float) -> float:
        """Required pre-render declaration (`Simulator`) — AC-54/AC-50.

        What the ROOM ENDS UP WITH, which is not what `createbox` is handed: this
        backend realizes `1 - sqrt(1 - x)` from an alpha `x`, so the two differ by
        the very correction `pre_compensate` applies. Under that convention the
        round trip is the identity and this returns the nominal value; under `as_is`
        it returns `1 - sqrt(1 - alpha)`, and every closed form the scene report
        publishes describes a room 1.14-1.98x more reverberant than declared.

        Used by the scene report to say which room its ISO 3382-1 distances, T60s
        and DRRs are about, and by this backend's own support falsification, which
        must reason about the decay it actually rendered.
        """
        return _gsound_realizes(
            _createbox_absorption(
                alpha_nominal, _check_convention(str(params["absorption_convention"]))
            )
        )

    @classmethod
    def realized_support_s(cls, params: dict, t60_s: float, volume_m3: float,
                           surface_m2: float, window_s: float) -> float:
        """Required pre-render declaration (`Simulator`) — AC-186, AC-175, AC-56.

        Two limits, and the ADAPTIVE ENERGY TRIM binds first: the compiled 3.0 s
        `maxIRLength` is a ceiling on top of it that no rendered scene has come
        near.

        `t60_s` and `volume_m3` are in the signature and unused HERE, deliberately.
        The gate still needs T60 to turn this answer into a decay range
        (`60 * support / T60`), and a backend whose trim really does track decay —
        which this one measurably does not — must be able to say so without the
        seam changing shape. Passing them and ignoring them is the declaration.

        All three coefficients are config-declared and falsified per render against
        realized `native_ir_samples` (`_support_falsification`), so a prediction
        that over-reads surfaces as a defect in the declaration rather than
        governing the dataset unchallenged.
        """
        trim_s = predicted_support_s(
            float(params["diffuse_depth"]),
            surface_m2,
            float(params["predicted_support_coefficient_s"]),
            float(params["predicted_support_depth_exponent"]),
            float(params["predicted_support_surface_exponent"]),
        )
        return min(trim_s, float(params["max_ir_length_s"]), float(window_s))

    @classmethod
    def max_eval_freq_hz(cls, params: dict) -> float:
        """Highest band this backend renders faithfully enough to measure (AC-66).

        Air absorption is realized at `air_absorption_realized_fraction` of the ISO
        value — a compiled-in domain confusion, declared rather than corrected
        because `AIR_ABSORPTION` is on and not exposed for pre-compensation. The
        resulting T60 error grows with frequency and passes
        `air_absorption_t60_error_tolerance_frac` above this band.
        """
        return float(params["air_absorption_max_eval_freq_hz"])

    @classmethod
    def code_scope(cls) -> tuple[str, ...]:
        """Required declaration (`Simulator`) — the source that decides this
        backend's IRs, and no more.

        Deliberately NOT the whole `simulators` package: that would let an edit to
        the `dry_run` scaffold invalidate a real emulated dataset. `base.py` is in
        scope because the seam it defines — `_fit_to_window`'s window contract and
        `PathData` — shapes what this backend returns.
        """
        return ("simulators/gsound_sir.py", "simulators/base.py")

    @classmethod
    def host_scoped_params(cls) -> tuple[str, ...]:
        """`render_python` is a machine-local interpreter path, not a dataset fact.

        Declared HERE rather than listed inside the render stage (F-86): the stage
        must not know what a gsound is, and a second backend's host-scoped param
        would otherwise be a second entry in a constant that only this backend can
        justify. Redacted from canonical provenance by `render._canonical_meta`;
        moves to `RunContext.host` when RD-20 lands.
        """
        return ("render_python",)

    @property
    def _ambisonics_order(self) -> int:
        """SH order implied by the channel count: n_channels = (order + 1)**2.

        Derived rather than separately declared, for AC-24's reason — a second
        config field holding the same fact could disagree with the channel count it
        describes.
        """
        order = int(round(self.n_channels ** 0.5)) - 1
        if (order + 1) ** 2 != self.n_channels:
            raise ValueError(
                f"n_channels={self.n_channels} is not a whole ambisonic order: "
                f"gsound_sir renders (order+1)**2 channels (1, 4, 9, 16, …). Set "
                f"`ambisonics_order` in config so n_channels lands on one of those."
            )
        return order

    def _retention_args(self) -> tuple[float, int]:
        """`path_retention` → the (`energy_percentage`, `max_rays`) pair.

        Those two numbers are consumed by the worker's own `_retain`, AFTER
        synthesis — NOT by `getPathData`, which is always called at (100.0, 0).
        Upstream's native retention cannot be used here: it filters inside the same
        call that produces the paths, so requesting it would mean a second
        propagation run purely to obtain the unfiltered set the IR is synthesized
        from. `_retain` applies upstream's selection rule, with a deterministic
        tie-break and float64 accumulation (see its docstring; AC-82).

        The split matters: retention applies ONLY to the saved artifact. Filtering
        before synthesis would change the IR itself and confound the ray-budget axis
        under study — measured at 43.1% of path energy on a real scene (RD-123).
        """
        mode = self.params["path_retention"]["mode"]
        value = self.params["path_retention"]["value"]
        if mode == "all":
            return 100.0, 0
        if mode == "top_percent":
            return float(value), 0
        return 100.0, int(value)

    def _run_worker(self, request: dict, out_dir: Path) -> dict:
        """Execute `_WORKER_SRC` under the render interpreter and return its result.

        `render_python` null → `sys.executable`, which is correct wherever the
        pipeline interpreter can itself import pygsound (a native x86_64 host). No
        platform test is involved: one code path, and the host difference is the
        config value alone.
        """
        worker_path = out_dir / "_worker.py"
        worker_path.write_text(_WORKER_SRC)
        request_path = out_dir / "request.json"
        request_path.write_text(json.dumps(request, indent=2))

        interpreter = self.params["render_python"] or sys.executable
        proc = subprocess.run(
            [str(interpreter), str(worker_path), str(request_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"GSound-SIR render worker failed (exit {proc.returncode}) under "
                f"interpreter {interpreter}.\n"
                f"--- worker stderr ---\n{proc.stderr}\n"
                f"--- worker stdout ---\n{proc.stdout}\n"
                f"If this is an architecture or import error, the interpreter cannot "
                f"load pygsound: set `render_python` in the simulator config to an "
                f"x86_64 interpreter built by scripts/setup_gsound_sir.py "
                f"(docs/gsound_sir_setup.md)."
            )
        return json.loads((out_dir / "result.json").read_text())

    def _fit_to_window(self, native: np.ndarray) -> tuple[np.ndarray, dict]:
        """Trim or pad the native IR ALONG TIME to `n_samples`, disclosing the loss.

        Returns `(ir, disclosure)`: the fitted (n_channels, n_samples) float32 array,
        and the per-(scene, leg) record of what fitting cost. The channel axis is
        never touched — `render()` checks it separately (RR-76).

        RD-21: trimming gsound's natural IR to `ir_duration` discards tail energy and
        can silently invalidate T30/EDT on the most reverberant scenes, so what was
        thrown away is MEASURED and reported per (scene, leg) rather than assumed
        negligible. `discarded_tail_db` is the discarded energy relative to the
        native total; it is None — never 0.0 or -inf — when nothing was discarded,
        because an unmeasured quantity must not be rendered as a number.

        `truncation_qc_flag` records the threshold breach. NOTHING CONSUMES IT YET:
        the per-criterion QC record that acts on it is Step 4's (RD-14). It is
        disclosure, not a gate.

        IT IS ALSO STRUCTURALLY UNREACHABLE UNDER `configs/base.yaml`, AND THAT IS
        A PROPERTY OF THE CONFIG PAIR RATHER THAN A DEFECT (F-83). The trim branch
        fires only when the native record EXCEEDS the window, and this backend's
        native record is bounded by the compiled `max_ir_length_s` (3.0 s, plus the
        auralizer's tail padding). Under base's `ir_duration: 4.25` the window is
        always the larger of the two, so `_fit_to_window` always pads and both
        `discarded_tail_db` and this flag are constant.

        It is LIVE under `configs/research_i.yaml`, whose RI-pinned `ir_duration` is
        3.0 s — the same length as the cap, where the padding either side of it
        decides the branch. So this is a backstop for record-shorter-than-cap
        configs, not dead code: deleting it would remove the disclosure exactly
        where the reproduction config needs it. The regression test drives the
        firing case through a CONFIG rather than a hand-built array, so the
        reachability claim is checked rather than asserted.
        """
        n_native = int(native.shape[1])
        total_energy = float(np.sum(native.astype(np.float64) ** 2))

        if n_native > self.n_samples:
            discarded = float(np.sum(native[:, self.n_samples:].astype(np.float64) ** 2))
            ir = native[:, : self.n_samples]
            truncated = True
            # Plain floats/bools throughout: this dict is written into the canonical
            # meta.json, and numpy scalars are not JSON-serializable — a run would
            # fail only at the write, after the expensive part.
            discarded_db = (
                float(10.0 * np.log10(discarded / total_energy))
                if discarded > 0.0 and total_energy > 0.0
                else None
            )
        else:
            ir = np.zeros((native.shape[0], self.n_samples), dtype=np.float32)
            ir[:, :n_native] = native
            truncated = False
            discarded_db = None

        flagged = bool(
            truncated
            and discarded_db is not None
            and discarded_db > float(self.params["max_discarded_tail_db"])
        )
        ir = np.ascontiguousarray(ir, dtype=np.float32)
        return ir, {
            "native_ir_samples": n_native,
            "fitted_ir_samples": int(ir.shape[1]),
            # BOTH energies, and named for the array each describes (F-84/F-111).
            # They differ whenever the trim branch fires, and it is the FITTED one
            # that is written to disk and read by every metric. Sum of squared
            # sample amplitudes, float64 accumulation, uncalibrated — no dB
            # reference, so comparable between legs of one scene and nothing else.
            "native_ir_total_energy": total_energy,
            "fitted_ir_total_energy": float(np.sum(ir.astype(np.float64) ** 2)),
            "truncated": truncated,
            "discarded_tail_db": discarded_db,
            "max_discarded_tail_db": float(self.params["max_discarded_tail_db"]),
            "truncation_qc_flag": flagged,
        }

    def _support_falsification(self, scene: SceneSpec, n_native: int) -> dict:
        """Falsify the declared `predicted_support_*` law against the realized record.

        The declaration in `configs/simulators/gsound_sir.yaml` predicts how much of
        a scene's decay this backend's adaptive energy trim will retain (AC-184).
        gen-scenes gates on that prediction because it runs before any render
        exists; this is the other half — the render reporting what actually
        happened, so an over-reading prediction becomes visible instead of
        governing the dataset unchallenged. Same contract as
        `speed_of_sound_m_s`: declare, then let upstream falsify it.

        Uses the REALIZED absorption, not the scene's nominal alpha — the trim
        responds to the decay that was rendered, and under `pre_compensate` those
        are the same room by construction, while under `as_is` they are not
        (AC-54).
        """
        volume, surface = box_volume_and_surface(scene.dims)
        alpha_realized = self.realized_absorption(
            self.params, float(scene.material_absorption)
        )
        t60_s = sabine_rt60(volume, surface, alpha_realized)
        predicted_s = predicted_support_s(
            float(self.params["diffuse_depth"]),
            surface,
            float(self.params["predicted_support_coefficient_s"]),
            float(self.params["predicted_support_depth_exponent"]),
            float(self.params["predicted_support_surface_exponent"]),
        )
        realized_s = n_native / float(self.sample_rate)
        return {
            "predicted_support_s": predicted_s,
            "realized_support_s": realized_s,
            # >= 1.0 means the backend retained at least what was predicted, so the
            # gen-scenes gate was not optimistic. < 1.0 is the direction that
            # matters: the dataset was admitted against a record it did not get.
            "support_realized_over_predicted": (
                realized_s / predicted_s if predicted_s > 0.0 else float("nan")
            ),
            "support_t60_s": t60_s,
            # What the record holds of its own decay, in dB. This is the quantity
            # AC-176's estimator bound is applied to downstream, reported here so
            # the refusal can be predicted from the render record alone.
            "realized_decay_range_db": (
                60.0 * realized_s / t60_s if t60_s > 0.0 else float("nan")
            ),
        }

    # RD-19's declared-speed cross-check lives in the WORKER, not here (F-94). A
    # parent-side copy over the retained subset was kept briefly as "defence in
    # depth" and was in fact dead: it re-tested a subset of an array the worker had
    # already accepted at the same tolerance, so it could not fail, and its empty
    # branch was unreachable because the worker exits first (F-119).

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        """Render one leg of one scene through the x86 worker.

        `ray_budget` is the DIFFUSE ray count (RD-12); `specular_count` is declared
        in config and held fixed across both legs, so the swept axis is unambiguously
        the diffuse budget.
        """
        energy_percentage, max_rays = self._retention_args()
        request = {
            "commit_sha": self.params["commit_sha"],
            # Cross-checked in the worker, over the unfiltered path set (F-94).
            "speed_of_sound_m_s": float(self.params["speed_of_sound_m_s"]),
            "dims": list(scene.dims),
            # AC-54/RD-144: the scene declares NOMINAL alpha; this backend
            # realizes 1-sqrt(1-alpha), so pre-compensation is what makes the
            # rendered room the room the scene describes.
            "absorption": _createbox_absorption(
                float(scene.material_absorption),
                str(self.params["absorption_convention"]),
            ),
            "absorption_nominal": float(scene.material_absorption),
            "absorption_convention": str(self.params["absorption_convention"]),
            "scattering": float(self.params["scattering"]),
            "source_pos": list(scene.source_pos),
            "receiver_pos": list(scene.receiver_pos),
            "source_radius": float(self.params["source_radius"]),
            "listener_radius": float(self.params["listener_radius"]),
            "source_power": float(self.params["source_power"]),
            "diffuse_count": int(ray_budget),
            "specular_count": int(self.params["specular_count"]),
            "diffuse_depth": int(self.params["diffuse_depth"]),
            "specular_depth": int(self.params["specular_depth"]),
            "sample_rate": int(self.sample_rate),
            # Sent rather than inlined in the worker (F-123): it cannot import
            # amcd, so a literal there would be a third copy of this name.
            "receipt_name": _RECEIPT_NAME,
            "receipt_sha_key": _RECEIPT_SHA_KEY,
            "normalize_ir": bool(self.params["normalize_ir"]),
            "precise_early_reflections": bool(self.params["precise_early_reflections"]),
            "early_reflection_threshold": float(self.params["early_reflection_threshold"]),
            "ambisonics_order": self._ambisonics_order,
            "frequency_points": list(self.params["frequency_points"]),
            "energy_percentage": energy_percentage,
            "max_rays": max_rays,
        }

        with tempfile.TemporaryDirectory(prefix="amcd_gsound_") as tmp:
            out_dir = Path(tmp)
            request["out_dir"] = str(out_dir)
            result = self._run_worker(request, out_dir)
            native = np.load(out_dir / "ir.npy")
            arrays = {k: v for k, v in np.load(out_dir / "paths.npz").items()}

        if native.shape[0] != self.n_channels:
            raise ValueError(
                f"scene {scene.scene_id!r}: worker returned {native.shape[0]} channels, "
                f"expected {self.n_channels} for ambisonic order "
                f"{self._ambisonics_order}."
            )
        # The band axis must be named by as many centres as it has columns (F-88).
        # Checked against the WORKER's reported count, which is upstream's own, so a
        # config whose band declaration drifts from the compiled filterbank fails
        # here rather than producing a path file that misnames its columns.
        n_bands = int(result["num_bands"])
        if n_bands != len(self.params["band_centres_hz"]) or (
            n_bands != len(self.params["frequency_points"]) + 1
        ):
            raise ValueError(
                f"scene {scene.scene_id!r}: upstream simulated {n_bands} bands but "
                f"config declares {len(self.params['band_centres_hz'])} "
                f"band_centres_hz and {len(self.params['frequency_points'])} "
                f"frequency_points (expected {n_bands} and {n_bands - 1}). The "
                f"retained-path file would name bands it does not contain."
            )

        ir, truncation = self._fit_to_window(native)
        # The other half of the AC-184 declaration: gen-scenes gated this scene on a
        # PREDICTED record length; here is what the backend actually produced.
        truncation.update(self._support_falsification(scene, truncation["native_ir_samples"]))

        # A leg with no energy is not a leg (F-84). Tested on the FITTED array — the
        # one written to disk and read by every metric (F-111).
        if truncation["fitted_ir_total_energy"] <= 0.0:
            raise ValueError(
                f"scene {scene.scene_id!r} at diffuse budget {ray_budget}: the "
                f"rendered IR carries zero total energy in the "
                f"{self.n_channels} x {self.n_samples} window that would be written "
                f"(native was {native.shape[0]} x {native.shape[1]}, energy "
                f"{truncation['native_ir_total_energy']:.6g}). The stored leg is "
                f"silent, so every metric computed from it would be undefined."
            )

        # None when total energy is zero — the share is undefined there, and a 0.0
        # would render an unscored quantity as a number (F-85).
        kept_pct = result["kept_energy_percentage"]
        kept_pct = None if kept_pct is None else float(kept_pct)

        paths = PathData(
            **arrays,
            num_paths=int(result["num_paths"]),
            num_bands=n_bands,
            total_energy=float(result["total_energy"]),
            kept_energy_percentage=kept_pct,
            descriptor={
                "simulator": "gsound_sir",
                "commit_sha": result["installed_commit_sha"],
                "band_edges_hz": list(self.params["frequency_points"]),
                "band_centres_hz": list(self.params["band_centres_hz"]),
                "sample_rate": int(self.sample_rate),
                "speed_of_sound_m_s": float(self.params["speed_of_sound_m_s"]),
                "path_retention": dict(self.params["path_retention"]),
                # How many paths the IR was actually synthesized from. Without it a
                # reader of this file sees only the retained count and cannot tell
                # the file is a subset of a much larger simulated set.
                "synthesis_num_paths": int(result["synthesis_num_paths"]),
                "ray_budget": int(ray_budget),
                # The producer does not know the stage's label for this leg; the
                # render stage stamps it when it writes the file.
                "leg": None,
                # RD-23: one render per (scene, budget) today, so the realization
                # axis exists in the artifact but is not yet swept. Present so a
                # file written now stays identifiable once it is.
                "realization_index": 0,
            },
        )

        return IRResult(
            ir=ir,
            paths=paths,
            meta={
                "simulator": "gsound_sir",
                "ray_budget": int(ray_budget),
                "speed_of_sound_m_s": float(self.params["speed_of_sound_m_s"]),
                "ambisonic_convention": _AMBISONIC_CONVENTION,
                "sh_condon_shortley_phase": _SH_CONDON_SHORTLEY_PHASE,
                # Two RNGs, reported separately — see _SYNTHESIS_CARRIER_SEED for
                # what that does and does not establish (AC-59/AC-76).
                "ray_rng_seeded": False,
                "synthesis_carrier_seed": _SYNTHESIS_CARRIER_SEED,
                # Kept: REQUIRED_PROVENANCE_KEYS binds every backend, and the two
                # keys above are gsound-specific refinements of it.
                "rng_seeded": False,
                "commit_sha": self.params["commit_sha"],
                "installed_commit_sha": result["installed_commit_sha"],
                # Both counts, so no artifact is ambiguous about which was varied.
                "diffuse_count": int(ray_budget),
                "specular_count": int(self.params["specular_count"]),
                # Retained vs simulated. Both, because they are very different
                # numbers — a real scene at top_k 5000 retained 43.1% of path energy
                # out of ~10^6 simulated paths — and `num_paths` alone would read as
                # the size of the simulation rather than of the saved artifact.
                "num_paths": int(result["num_paths"]),
                "synthesis_num_paths": int(result["synthesis_num_paths"]),
                "num_bands": n_bands,
                "kept_energy_percentage": kept_pct,
                # How many paths the declared speed was falsified against — the full
                # simulated set, not the retained subset (F-94).
                "speed_check_num_paths": int(result["speed_check_num_paths"]),
                **truncation,
            },
        )


class PathRetention(BaseModel):
    """Which simulated paths reach the saved retained-path artifact.

    Maps onto the (energy_percentage, max_rays) pair that the render worker's
    `_retain` applies — upstream's own selection rule, reproduced there rather than
    requested from `getPathData`, which is always called unfiltered so the IR is
    synthesized from every path (RD-123):
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
            return
        if self.value is None:
            raise ValueError(f"path_retention.mode {self.mode!r} requires a `value`")

        # Range and integrality, per mode (F-92): out-of-domain values are
        # REINTERPRETED by the worker's cut rule, not rejected. The two raise
        # messages below state each domain.
        if self.mode == "top_k":
            if self.value != int(self.value) or int(self.value) < 1:
                raise ValueError(
                    f"path_retention.mode 'top_k' takes a whole number of paths >= 1; "
                    f"got {self.value!r}. Values <= 0 do not mean 'keep everything' — "
                    f"use mode 'all' for that."
                )
        elif self.mode == "top_percent":
            if not 0.0 < float(self.value) <= 100.0:
                raise ValueError(
                    f"path_retention.mode 'top_percent' takes a share in (0, 100]; "
                    f"got {self.value!r}. 0 would keep a single path and >100 is "
                    f"silently the same as 'all'."
                )


GsoundSirSimulator.Params.model_rebuild()
