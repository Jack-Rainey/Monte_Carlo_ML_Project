# Setting up the GSound-SIR render environment

The `gsound_sir` render backend needs an **x86_64** interpreter: GSound's ray tracer is
x86-only. Every other stage (training, evaluation, stats) runs native on whatever the host
is. That boundary lives entirely behind the simulator seam — the `amcd` package contains no
platform branches, and the render interpreter is selected at runtime with `--sim-python`.

Setup is two steps:

1. **Create the render environment** — not scriptable portably (conda vs venv, per-OS
   toolchains), so it is documented here.
2. **Build and install GSound-SIR into it** — fully scripted:
   `scripts/setup_gsound_sir.py`.

> This document covers step 1 and the invocation of step 2. The narrative of *why* the
> build needs its two workarounds, and how `--sim-python` threads through the pipeline, is
> expanded when the backend itself lands (gate plan Step 7). The workarounds themselves are
> already encoded in the installer's module docstring.

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
