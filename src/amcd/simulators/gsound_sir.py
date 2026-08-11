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
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, field_validator

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
#: code. Two small constants beat that; the integrator has a note to consolidate
#: them into a shared module (this lane may not create one).
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
#: ACN/N3D. **MEASURED, not read off the source comments** (AC-43): a single
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
#: Stamped separately rather than folded into the convention string because the
#: string names ordering + normalization, which are genuinely acn_n3d; the phase
#: is an additional fact about the same data. It is inert today — every live
#: scalar metric reads channel 0, where it has no effect — and becomes
#: load-bearing the moment evaluation/spatial.py estimates a direction, which
#: would otherwise come out 180 degrees wrong in azimuth and look like a bug in
#: the estimator rather than in the encoding (RD-25).
_SH_CONDON_SHORTLEY_PHASE = True

#: The render worker, as source rather than a module, because it must run under an
#: interpreter where `amcd` DOES NOT EXIST: the render env holds numpy, pygsound and
#: spherical_harmonics_rt and nothing else. It therefore imports no part of this
#: package, and communicates by files — a JSON request in, `ir.npy` + `paths.npz` +
#: `result.json` out — so nothing has to be pickle-compatible across two Pythons of
#: different architectures.
#:
#: It lives in a string because this lane may not create a new file under
#: `src/amcd/simulators/` (exact-file ownership, `.claude/lane.json`); the integrator
#: has a note to promote it to its own module after the merge. `_WORKER_SRC` is
#: compiled and executed against stub backends by `tests/test_simulator_seam.py`, so
#: it has a regression surface that does not require the render host.
#:
#: THE SHA CHECK IS FIRST, BEFORE ANY SIMULATION (RD-67): the whole point of
#: verifying the installed upstream commit is to refuse a render that would produce
#: an artifact of unknown provenance, and under emulation that render can cost hours.
_WORKER_SRC = r'''
"""GSound-SIR render worker. Runs under the x86 render interpreter; imports no amcd."""
import json
import sys
import sysconfig
from pathlib import Path

import numpy as np


def _installed_sha():
    """The upstream commit this env was built from, per its install receipt."""
    receipt = Path(sysconfig.get_paths()["purelib"]) / "amcd_gsound_install.json"
    if not receipt.exists():
        raise SystemExit(
            "no amcd_gsound_install.json in %s - this render env was not installed "
            "by scripts/setup_gsound_sir.py, so the upstream commit it contains is "
            "unknown and no render from it would be reproducible. Re-run the "
            "installer against this interpreter." % receipt.parent
        )
    return json.loads(receipt.read_text())["commit_sha"]


PATH_ARRAYS = ("distances", "intensities", "listener_directions", "source_directions",
               "path_types", "speeds_of_sound", "relative_speeds", "source_indices")


def _retain(paths, energy_percentage, max_rays):
    """Select the retained-path subset, reproducing upstream's own algorithm.

    Upstream applies this INSIDE getPathData (Scene.cpp:193-224): sum `intensities`
    across bands per path, sort descending, keep until the cumulative share reaches
    `energy_percentage`, then cap at `max_rays`. It is reproduced here rather than
    requested from upstream because asking getPathData to filter would mean a SECOND
    propagation run just to obtain the unfiltered set the IR must be synthesized
    from - doubling the cost of every render to get one array twice.

    Returns (kept arrays, total energy over ALL paths, kept share as a percentage).
    """
    intensities = np.asarray(paths["intensities"], dtype=np.float64)
    per_path = intensities.sum(axis=1)
    order = np.argsort(-per_path, kind="stable")
    total = float(per_path.sum())

    keep = per_path.shape[0]
    if energy_percentage < 100.0 and total > 0.0:
        cumulative = np.cumsum(per_path[order])
        reached = np.searchsorted(cumulative, total * (energy_percentage / 100.0))
        keep = int(min(reached + 1, keep))
    if 0 < max_rays < keep:
        keep = int(max_rays)

    selected = order[:keep]
    kept = {name: np.asarray(paths[name])[selected] for name in PATH_ARRAYS}
    kept_pct = 100.0 * float(per_path[selected].sum()) / total if total > 0.0 else 0.0
    return kept, total, kept_pct


def main(request_path):
    req = json.loads(Path(request_path).read_text())
    out_dir = Path(req["out_dir"])

    # Provenance BEFORE physics: never spend an emulated render on an env whose
    # upstream commit does not match the pin.
    installed = _installed_sha()
    if installed != req["commit_sha"]:
        raise SystemExit(
            "installed GSound-SIR commit %s != config-pinned %s. The render env and "
            "configs/simulators/gsound_sir.yaml disagree about which upstream code "
            "produces this dataset; rebuild the env at the pinned SHA or repin."
            % (installed, req["commit_sha"])
        )

    import pygsound as ps
    import spherical_harmonics_rt as sh

    ctx = ps.Context()
    ctx.diffuse_count = int(req["diffuse_count"])
    ctx.specular_count = int(req["specular_count"])
    ctx.diffuse_depth = int(req["diffuse_depth"])
    ctx.specular_depth = int(req["specular_depth"])
    ctx.sample_rate = float(req["sample_rate"])
    ctx.normalize = bool(req["normalize_ir"])

    w, l, h = req["dims"]
    mesh = ps.createbox(float(w), float(l), float(h),
                        float(req["absorption"]), float(req["scattering"]))
    scene = ps.Scene()
    scene.setMesh(mesh)

    src = ps.Source([float(v) for v in req["source_pos"]])
    src.radius = float(req["source_radius"])
    src.power = float(req["source_power"])
    lis = ps.Listener([float(v) for v in req["receiver_pos"]])
    lis.radius = float(req["listener_radius"])

    # ALWAYS the full path set: retention applies ONLY to the saved artifact, never
    # to synthesis. Filtering before synthesis would change the IR itself and so
    # confound the very ray-budget axis under study — top_k 5000 was MEASURED to
    # retain 43.2% of path energy on a real scene, i.e. it would silently delete
    # more than half the response.
    #
    # getPathData returns {"path_data": [<per source-listener pair>, ...]};
    # result[0] raises KeyError. One source and one listener, hence pair 0.
    paths = scene.getPathData(
        [src], [lis], ctx,
        energy_percentage=100.0, max_rays=0, use_gpu=False,
    )["path_data"][0]

    freq_points = np.asarray(req["frequency_points"], dtype=np.float32)
    ir = sh.generate_ambisonic_ir(
        int(req["ambisonics_order"]),
        np.asarray(paths["listener_directions"], dtype=np.float32),
        np.asarray(paths["intensities"], dtype=np.float32),
        np.asarray(paths["distances"], dtype=np.float32),
        np.asarray(paths["speeds_of_sound"], dtype=np.float32),
        freq_points,
        float(req["sample_rate"]),
        normalize=bool(req["normalize_ir"]),
    )
    ir = np.asarray(ir, dtype=np.float32)

    kept, total_energy, kept_pct = _retain(
        paths, float(req["energy_percentage"]), int(req["max_rays"])
    )

    np.save(out_dir / "ir.npy", ir)
    np.savez(out_dir / "paths.npz", **kept)
    (out_dir / "result.json").write_text(json.dumps({
        "installed_commit_sha": installed,
        "num_paths": int(kept["distances"].shape[0]),
        "num_bands": int(paths["num_bands"]),
        "total_energy": total_energy,
        "kept_energy_percentage": kept_pct,
        "native_ir_shape": [int(d) for d in ir.shape],
        "synthesis_num_paths": int(paths["num_paths"]),
    }))


if __name__ == "__main__":
    main(sys.argv[1])
'''


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
        from. `_retain` reproduces upstream's selection exactly (Scene.cpp:193-224).

        The split matters: retention applies ONLY to the saved artifact. Filtering
        before synthesis would change the IR itself and confound the ray-budget axis
        under study — measured at 43.1% of path energy on a real scene (RD-102).
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
        """Trim or pad the native IR to (n_channels, n_samples), disclosing the loss.

        RD-21: trimming gsound's natural IR to `ir_duration` discards tail energy and
        can silently invalidate T30/EDT on the most reverberant scenes, so what was
        thrown away is MEASURED and reported per (scene, leg) rather than assumed
        negligible. `discarded_tail_db` is the discarded energy relative to the
        native total; it is None — never 0.0 or -inf — when nothing was discarded,
        because an unmeasured quantity must not be rendered as a number.

        `truncation_qc_flag` records the threshold breach. NOTHING CONSUMES IT YET:
        the per-criterion QC record that acts on it is Step 4's (RD-14). It is
        disclosure, not a gate.
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
        return np.ascontiguousarray(ir, dtype=np.float32), {
            "native_ir_samples": n_native,
            "truncated": truncated,
            "discarded_tail_db": discarded_db,
            "max_discarded_tail_db": float(self.params["max_discarded_tail_db"]),
            "truncation_qc_flag": flagged,
        }

    def _check_declared_speed(self, speeds: np.ndarray, scene_id: str) -> None:
        """Falsify the declared `speed_of_sound_m_s` against the paths' own speeds.

        RD-19: gsound's speed is compiled into C++ and can only be DECLARED into
        provenance. This is the free empirical check that keeps that declaration
        honest — if upstream ever changes it, the dataset says so instead of
        inheriting a stale number from config.
        """
        declared = float(self.params["speed_of_sound_m_s"])
        observed = np.unique(np.asarray(speeds, dtype=np.float64))
        if observed.size == 0:
            raise ValueError(
                f"scene {scene_id!r}: the render returned no paths, so the declared "
                f"speed_of_sound_m_s={declared} could not be cross-checked and the "
                f"IR has no propagation to synthesize from."
            )
        if not np.allclose(observed, declared, rtol=1e-3):
            raise ValueError(
                f"scene {scene_id!r}: config declares speed_of_sound_m_s={declared} "
                f"but the rendered paths report {observed.tolist()[:5]} m/s. gsound's "
                f"speed is compiled in and can only be declared (RD-19); the "
                f"declaration is now wrong, so every distance/delay in this dataset "
                f"would be described by a speed that did not produce it."
            )

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        """Render one leg of one scene through the x86 worker.

        `ray_budget` is the DIFFUSE ray count (RD-12); `specular_count` is declared
        in config and held fixed across both legs, so the swept axis is unambiguously
        the diffuse budget.
        """
        energy_percentage, max_rays = self._retention_args()
        request = {
            "commit_sha": self.params["commit_sha"],
            "dims": list(scene.dims),
            "absorption": float(scene.material_absorption),
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
            "normalize_ir": bool(self.params["normalize_ir"]),
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
        self._check_declared_speed(arrays["speeds_of_sound"], scene.scene_id)

        ir, truncation = self._fit_to_window(native)

        paths = PathData(
            **arrays,
            num_paths=int(result["num_paths"]),
            num_bands=int(result["num_bands"]),
            total_energy=float(result["total_energy"]),
            kept_energy_percentage=float(result["kept_energy_percentage"]),
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
                # pygsound exposes no RNG seed (RD-23), so reproducibility rests on
                # the cached artifacts, not on re-render bit-identity.
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
                "num_bands": int(result["num_bands"]),
                "kept_energy_percentage": float(result["kept_energy_percentage"]),
                **truncation,
            },
        )


class PathRetention(BaseModel):
    """Which simulated paths reach the saved retained-path artifact.

    Maps onto the (energy_percentage, max_rays) pair that the render worker's
    `_retain` applies — upstream's own selection rule, reproduced there rather than
    requested from `getPathData`, which is always called unfiltered so the IR is
    synthesized from every path (RD-102):
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
