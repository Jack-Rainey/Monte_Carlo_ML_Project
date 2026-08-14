"""GSound-SIR render worker — runs under the x86 render interpreter.

IMPORTS NO PART OF `amcd`, and must not start. The render env holds numpy,
pygsound and spherical_harmonics_rt and nothing else, so this file is read as TEXT
by `gsound_sir._WORKER_SRC` and executed by a separate interpreter. It communicates
by files: a JSON request in, `ir.npy` + `paths.npz` + `result.json` out, so nothing
has to be pickle-compatible across two Pythons of different architectures.

TARGET INTERPRETER: whatever `render_python` names, built by
`scripts/setup_gsound_sir.py` and not necessarily the pipeline's own. Keep to
Python 3.8 syntax and the stdlib plus numpy — no match statements, no `X | Y`
annotations, no walrus in a comprehension target — because a syntax error here
surfaces as a subprocess exit code rather than as an import error the suite
catches.

THE SHA CHECK RUNS FIRST, BEFORE ANY SIMULATION: the point of verifying
the installed upstream commit is to refuse a render that would produce an artifact
of unknown provenance, and under emulation that render can cost hours.
"""
import json
import sys
import sysconfig
from pathlib import Path

import numpy as np


def _installed_sha(receipt_name, sha_key):
    """The upstream commit this env was built from, per its install receipt.

    The receipt's filename and key are PASSED IN rather than repeated here.
    This file cannot import `amcd`, so a literal would be a third copy of a name
    already defined in `gsound_sir.py` and in `scripts/setup_gsound_sir.py` — and
    the parent already sends a JSON request, so it can send these too.
    """
    receipt = Path(sysconfig.get_paths()["purelib"]) / receipt_name
    if not receipt.exists():
        raise SystemExit(
            "no %s in %s - this render env was not installed "
            "by scripts/setup_gsound_sir.py, so the upstream commit it contains is "
            "unknown and no render from it would be reproducible. Re-run the "
            "installer against this interpreter." % (receipt_name, receipt.parent)
        )
    return json.loads(receipt.read_text())[sha_key]


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

    THE SAME SELECTION RULE, not a bit-exact port. Two deliberate
    divergences, both confined to the `top_percent` branch and both in the direction
    of determinism: upstream sorts with `std::sort`, which is UNSTABLE, so its kept
    set is unspecified under tied energies while this one is fixed; and upstream
    accumulates in float32 while this uses float64, so a cut landing on a boundary
    can differ by a path. Counts agree with a transcription of upstream over 17
    edge cases. Assumes non-negative per-path energies, which `searchsorted` needs
    and upstream's running loop does not.

    Returns (kept arrays, total energy over ALL paths, kept share as a percentage or
    None). The share is None - never 0.0 - when total energy is zero, because the
    ratio is undefined there and a 0.0 would read as "almost nothing was retained"
    for a subset that in fact holds every path.
    """
    intensities = np.asarray(paths["intensities"], dtype=np.float64)
    per_path = intensities.sum(axis=1)
    order = np.argsort(-per_path, kind="stable")
    total = float(per_path.sum())

    keep = per_path.shape[0]
    if energy_percentage < 100.0:
        # No `total > 0` guard: upstream has none either, and adding one CHANGES THE
        # SELECTION. On an all-zero-energy path set the cumulative sum is all zeros
        # and the target is 0.0, so searchsorted returns 0 and upstream keeps ONE
        # path; the guard made this branch keep all of them.
        cumulative = np.cumsum(per_path[order])
        reached = np.searchsorted(cumulative, total * (energy_percentage / 100.0))
        keep = int(min(reached + 1, keep))
    if 0 < max_rays < keep:
        keep = int(max_rays)

    selected = order[:keep]
    kept = {name: np.asarray(paths[name])[selected] for name in PATH_ARRAYS}
    kept_pct = 100.0 * float(per_path[selected].sum()) / total if total > 0.0 else None
    return kept, total, kept_pct


def _check_declared_speed(speeds, declared):
    """Falsify the config's declared propagation speed against the paths' own.

    Runs HERE, in the worker, over the UNFILTERED path set: the parent only ever
    sees the retained subset, which under top_k is ~1% of what was simulated, and
    the declaration is stated over the paths, not over a sample of them. gsound's
    speed is compiled into C++ and can only be DECLARED, so this array is the only
    thing that can falsify it.

    Returns the number of paths checked, for the provenance record.
    """
    observed = np.unique(np.asarray(speeds, dtype=np.float64))
    if observed.size == 0:
        raise SystemExit(
            "the render returned no paths, so the declared speed_of_sound_m_s "
            "could not be cross-checked and there is no propagation to synthesize."
        )
    if not np.allclose(observed, declared, rtol=1e-3):
        raise SystemExit(
            "config declares speed_of_sound_m_s=%s but the rendered paths report "
            "%s m/s. gsound's speed is compiled in and can only be declared; the "
            "declaration is now wrong, so every distance and delay in this dataset "
            "would be described by a speed that did not produce it."
            % (declared, observed.tolist()[:5])
        )
    return int(np.asarray(speeds).shape[0])


def main(request_path):
    """Render one leg: JSON request in, three files out.

    REQUEST keys, by group (the parent builds them in `GsoundSirSimulator.render`;
    the two halves must move together):
      provenance  commit_sha, speed_of_sound_m_s, out_dir
      geometry    dims, absorption, scattering, source_pos, receiver_pos,
                  source_radius, listener_radius, source_power
      budgets     diffuse_count, specular_count, diffuse_depth, specular_depth
      synthesis   sample_rate, normalize_ir, ambisonics_order, frequency_points,
                  precise_early_reflections, early_reflection_threshold
      retention   energy_percentage, max_rays  (artifact only, never synthesis)

    RESPONSE, written into out_dir:
      ir.npy       float32 (n_channels, n_native_samples), channel-first
      paths.npz    the RETAINED per-path arrays, PATH_ARRAYS above
      result.json  scalars the parent stamps into provenance
    """
    req = json.loads(Path(request_path).read_text())
    out_dir = Path(req["out_dir"])

    # Provenance BEFORE physics: never spend an emulated render on an env whose
    # upstream commit does not match the pin.
    installed = _installed_sha(req["receipt_name"], req["receipt_sha_key"])
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
    # retain 43.1% of path energy on a real scene, i.e. it would silently delete
    # more than half the response.
    #
    # getPathData returns {"path_data": [<per LISTENER>, ...]}, one entry per
    # listener with every source aggregated into it and distinguished by
    # `source_indices` (Scene.cpp:169,171) - NOT one entry per source-listener pair.
    # result[0] raises KeyError. One listener, hence entry 0.
    paths = scene.getPathData(
        [src], [lis], ctx,
        energy_percentage=100.0, max_rays=0, use_gpu=False,
    )["path_data"][0]

    # Over the FULL path set, before retention throws ~99% of it away.
    speed_check_num_paths = _check_declared_speed(
        paths["speeds_of_sound"], float(req["speed_of_sound_m_s"])
    )

    freq_points = np.asarray(req["frequency_points"], dtype=np.float32)
    ir = sh.generate_ambisonic_ir(
        int(req["ambisonics_order"]),
        np.asarray(paths["listener_directions"], dtype=np.float32),
        np.asarray(paths["intensities"], dtype=np.float32),
        np.asarray(paths["distances"], dtype=np.float32),
        np.asarray(paths["speeds_of_sound"], dtype=np.float32),
        freq_points,
        float(req["sample_rate"]),
        precise_early_reflections=bool(req["precise_early_reflections"]),
        normalize=bool(req["normalize_ir"]),
        early_reflection_threshold=float(req["early_reflection_threshold"]),
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
        "speed_check_num_paths": speed_check_num_paths,
    }))


if __name__ == "__main__":
    main(sys.argv[1])
