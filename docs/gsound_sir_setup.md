# Setting up the GSound-SIR render environment

The `gsound_sir` render backend needs an **x86_64** interpreter: GSound's ray tracer is
x86-only. Every other stage (training, evaluation, stats) runs native on whatever the host
is. That boundary lives entirely behind the simulator seam — the `amcd` package contains no
platform branches, and the render interpreter is named by the simulator config key
`render_python` (`configs/simulators/gsound_sir.yaml`). There is no CLI flag for it: it is
host-scoped, so it is supplied by a host-local config layer that is not committed, and is
redacted from canonical provenance. `null` means "use `sys.executable`", which is correct
on a native x86_64 host.

Setup is two steps:

1. **Create the render environment** — not scriptable portably (conda vs venv, per-OS
   toolchains), so it is documented here.
2. **Build and install GSound-SIR into it** — fully scripted:
   `scripts/setup_gsound_sir.py`.

> This document covers step 1 and the invocation of step 2. The narrative of *why* the
> build needs its two workarounds is encoded in the installer's module docstring.

## 1. Create the render environment

The environment needs **Python 3.10** (x86_64), plus CMake and FFTW.

### macOS on Apple Silicon (this machine) — via Rosetta 2

The x86 render step runs under emulation; everything else stays native arm64 + MPS.

```bash
softwareupdate --install-rosetta --agree-to-license   # once per machine
xcode-select --install                                # C++ toolchain

CONDA_SUBDIR=osx-64 conda create -n amcd-render-x86 python=3.10 -y
conda activate amcd-render-x86
conda config --env --set subdir osx-64     # keep the env osx-64 for later installs
conda install -c conda-forge cmake fftw pkg-config -y
```

Install these *into the render environment*, not just system-wide: the installer points
CMake at the environment's own `include/` and `lib/`, so FFTW must live there.

Verify it really is x86_64 before building anything:

```bash
python -c "import platform; print(platform.machine())"   # must print x86_64
```

### macOS on Intel

As above, without Rosetta and without `CONDA_SUBDIR`.

### Ubuntu / Linux x86_64 (native, no emulation)

```bash
sudo apt install build-essential cmake pkg-config libfftw3-dev zlib1g-dev
conda create -n amcd-render-x86 python=3.10 -y
conda activate amcd-render-x86
conda install -c conda-forge fftw pkg-config -y   # FFTW inside the env, as above
```

### Windows x86_64

Install the MSVC Build Tools ("Desktop development with C++") and CMake, then create the
same conda environment. WSL2 + the Ubuntu instructions above is the tested fallback.

## 2. Build and install GSound-SIR

Run the installer with **any** working Python (the project env is fine); it targets the
render environment through `--env-python`:

```bash
python scripts/setup_gsound_sir.py \
    --env-python ~/miniconda3/envs/amcd-render-x86/bin/python \
    --ref 608ea30f6dc4cda149c18947f9cae48bd379fa27
```

`--ref` accepts a full commit SHA, a branch or tag, or `latest`; anything that is not
already a SHA is resolved to one *before* the build starts, and that concrete SHA is what
gets checked out, verified and recorded. The upstream repository is cloned to
`external/GSound-SIR` (gitignored — it is a build artifact, not vendored source; GSound-SIR
is never modified locally).

The build takes a while, especially under Rosetta. On success the installer prints the
resolved SHA to pin in `configs/simulators/gsound_sir.yaml`, and writes an install receipt
(`amcd_gsound_install.json`) into the environment's `site-packages`. That receipt is the
provenance record of which upstream commit an environment actually contains — `pygsound`
exposes no version of its own — and the render backend compares it against the pinned
`commit_sha`.

Useful flags:

| Flag | Use |
|---|---|
| `--verify-only` | Check an existing render env (imports, architecture, link line, API) without rebuilding. |
| `--rebuild` | Delete `build/` first. Use when a stale CMake cache is suspected. |
| `--clone-dir` | Clone somewhere other than `external/GSound-SIR`. |

## 3. Check it works

```bash
python scripts/setup_gsound_sir.py --env-python <render-env>/bin/python --verify-only
```

This asserts, in the render env: `pygsound` and `spherical_harmonics_rt` both import; the
interpreter is x86_64; the `pygsound` extension links **no** CPython shared library; and
`generate_ambisonic_ir` has the expected signature. Any failure is a hard error with the
reason — the environment is never reported as usable on partial evidence.

## Troubleshooting

**`import pygsound` segfaults.** The `PYBIND11_FINDPYTHON=OFF` fix did not take, usually
because of a stale CMake cache. Re-run with `--rebuild`. (`--verify-only` detects this
before a render does: it inspects the extension's link line for a `libpython`.)

**`ModuleNotFoundError: spherical_harmonics_rt`.** The auralizer was installed by some
means other than this script. Upstream's CMake target is `spherical_harmonics` but the
module it declares is `spherical_harmonics_rt`; the installer exposes both. Re-run it.

**CMake or compiler not found.** The installer prefers the render env's own CMake
(`conda install -c conda-forge cmake` inside that env) and falls back to PATH.

**`source file is not valid UTF-8` while compiling `._omSomething.cpp`.** The clone lives on
a filesystem that cannot store extended attributes natively — exFAT, FAT, or a network
mount. macOS then writes each file's xattrs to a binary AppleDouble sidecar named
`._<name>`, and upstream collects sources with `file(GLOB SOURCES om/*/*.cpp)`
(`ray_generator/src/Om/Om BVH/CMakeLists.txt:7`), which matches the sidecars and hands them
to the compiler. The installer deletes AppleDouble sidecars (verified by magic number, not
by name) after checkout, so a normal run is immune; if you hit this, a stale build tree is
the likely cause — re-run with `--rebuild`. Sidecars inside `.git` cause the related
`non-monotonic index` warnings from git and are cleaned by the same pass.

**Auralizer build fails fetching pybind11.** Its CMake uses `FetchContent` to pull pybind11
v2.11.1, so the build needs network access.

## 4. Upstream API reference (verified by introspection, not from docs)

Relocated here from `docs/review_ledger.md`: these are durable facts about
the pinned upstream, not review findings, and the ledger holds only unresolved
findings. Verified against SHA `608ea30f6dc4cda149c18947f9cae48bd379fa27` on
2026-08-01 by calling into the built render env.

**Upstream** https://github.com/yongyizang/GSound-SIR — declared once, as
`UPSTREAM_URL` in `scripts/setup_gsound_sir.py`. **Pinned SHA**
`608ea30f6dc4cda149c18947f9cae48bd379fa27` (main HEAD at the time). The clone
lives at `external/GSound-SIR`, gitignored — it is a build artifact, not vendored
source (CLAUDE.md forbids a modified local copy).

### `generate_ambisonic_ir` — there is no `path_types` argument

```
sh.generate_ambisonic_ir(order, listener_directions, intensities, distances,
                         speeds, frequency_points, sample_rate,
                         precise_early_reflections=False, normalize=True,
                         early_reflection_threshold=0.01)
```

Upstream's own `auralizer/test.py` passes a `path_types` argument; it does not
exist in the built module.

### `frequency_points` are band EDGES, not centres

They are `n_bands - 1` CROSSOVER frequencies, consumed as
`CrossoverFilter crossover(sample_rate, freq_points)`
(`auralizer/src/cpp/binding.cpp:304,334`), with a hard runtime check
("Number of frequency points must be number of bands - 1").

Traced end to end: `pygsound::Context()`
(`ray_generator/src/pygsound/src/Context.cpp:8`) overrides GSound's log-spaced
defaults with octave band **centres** `{63,125,250,500,1000,2000,4000,8000}` Hz,
and `gs::FrequencyBands` derives crossovers as the geometric mean of adjacent
centres (`gsFrequencyBands.cpp:83-88`). The correct values, as pinned in
`configs/simulators/gsound_sir.yaml`, are therefore

```
[88.7412, 176.7767, 353.5534, 707.1068, 1414.2136, 2828.4271, 5656.8542] Hz
```

⚠️ **The first value is 88.7412, NOT 88.4**. 88.4 is
√(62.5×125) — the base-two *nominal* 62.5 Hz centre — but pygsound uses the
IEC-rounded **63** Hz, so 88.4 disagrees with the ray generator's own band
definition by 0.384 %. Full precision is given here rather than 1 dp because a
rounded first value is exactly how the refuted number survived: this section
originally carried `88.4`, copied forward from a earlier note, while
`gsound_sir.yaml` had already been corrected. The config is the authority; if
these two ever disagree again, the config is right.

Consequence worth knowing: the two reported ISO bands
(`iso_eval_freqs` 500/1000 Hz, edges 353.553 / 707.107 / 1414.214) coincide
**exactly** with simulated bands 3 and 4.

⚠️ **Upstream `auralizer/test.py` is wrong here** — it passes
`[125,250,500,1000,2000,4000,8000]`, i.e. the band centres used as if they were
edges, which shifts every band edge ~½ octave high and misassigns the simulated
per-band energies during SH synthesis. Do not copy it.

### `getPathData` returns a dict wrapper, not a list

```
scene.getPathData(..., energy_percentage=100.0, max_rays=0, use_gpu=False)
  -> {"path_data": [<per-source-listener-pair dict>, ...]}
```

The per-pair dict is `result["path_data"][i]`; `result[0]` raises `KeyError: 0`.

Path retention is native upstream in the API sense — `path_retention {mode:
all|top_percent|top_k, value}` names the same quantities as `energy_percentage` /
`max_rays` — but this backend does **not** use it. Upstream applies retention
inside `getPathData`, which is the same call that supplies the paths the IR is
synthesized from, so filtering there would build the IR from the retained subset
and confound the ray-budget axis under study. Measured when that happened: the IR
was synthesized from 5,000 of 501,492 paths (43.1 % of path energy) and the native
record came back 9,502 samples instead of 46,333.

So the worker calls `getPathData(energy_percentage=100.0, max_rays=0)` once, feeds
the full set to synthesis, and applies retention to the saved ARTIFACT afterwards,
reproducing upstream's own algorithm (`Scene.cpp:193-224`) in `_retain`
(`src/amcd/simulators/gsound_sir.py`). Using upstream's filter instead would need a
second propagation run purely to recover the unfiltered set.

### PathData schema (pinned by the actual keys of `path_data[i]`)

Arrays: `distances` (N,) f32 · `intensities` (N,8) f32 ·
`listener_directions` (N,3) f32 · `source_directions` (N,3) f32 ·
`path_types` (N,) uint32 · `speeds_of_sound` (N,) f32 ·
`relative_speeds` (N,) f32 · `source_indices` (N,) uint64.
Scalars: `num_paths` i64 · `num_bands` i64 · `total_energy` f64 ·
`kept_energy_percentage` f64.

### Other confirmed facts

- `ps.Context` exposes `diffuse_count`, `specular_count`, `diffuse_depth`,
  `specular_depth`, `threads_count`, `channel_type`, `sample_rate`, `normalize` —
  confirming the diffuse/specular split the ray-budget axis rests on.
- `createbox(width, length, height, absorp, scatter)` takes absorption as a scalar
  **or a per-band sequence**, but **not per-surface** — the constraint behind the
  deferred per-surface-materials decision.
- The simulator uses **344 m/s**, not 343. It is compiled into C++ and can only be
  DECLARED into provenance, never configured.

### Observed smoke-test numbers (for comparison when re-verifying)

A 5×4×3 m box at diffuse 5000 / specular 2000 gave 1,001,014 paths over 8 bands;
IR shape **(16, 92859)** float32 (order 3 → 16 channels); onset 6.50 ms observed
vs 6.52 ms predicted for the 2.236 m direct path at 344 m/s.
