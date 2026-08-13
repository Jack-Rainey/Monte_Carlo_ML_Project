"""Reproducible, ref-addressable installer for GSound-SIR (pygsound + auralizer).

GSound-SIR is pulled from upstream GitHub at a pinned commit SHA and built into a
*render environment* — a separate x86_64 interpreter, since the ray tracer is x86-only.
This script is the executable form of that build: it encodes everything the Step 0 spike
had to discover by hand, so the render env can be rebuilt from scratch on macOS (Rosetta /
osx-64), native x86_64 Linux, or Windows without repeating the discovery.

It lives in ``scripts/``, not in ``src/amcd/``, because it is host tooling: platform
branches (``otool`` vs ``ldd``, per-OS toolchain hints) belong outside the package, behind
the simulator seam. Nothing here modifies upstream source — the pinned SHA remains the
whole story for reproducibility.

Two upstream build defects are worked around, both without touching upstream files:

1. ``import pygsound`` segfaults in ``PyGILState_Ensure``. ``ray_generator``'s
   ``pybind11_add_module(pygsound SHARED ...)`` under pybind11 v3's *new* FindPython mode
   links ``Python::Python``, i.e. the full ``libpython3.10`` shared library. conda's macOS
   python is statically linked (``Py_ENABLE_SHARED = 0``), so that library is a second,
   uninitialised CPython in the process. Fix: configure with
   ``-DPYBIND11_FINDPYTHON=OFF`` (classic mode -> ``pybind11::module`` ->
   ``-undefined dynamic_lookup``, no libpython on the link line). Since
   ``ray_generator/setup.py`` does not forward ``CMAKE_ARGS``, the flag is delivered by
   pre-seeding the CMake cache in the exact build directory setup.py will reuse.

2. ``import spherical_harmonics`` fails with ``PyInit_spherical_harmonics not defined``.
   The auralizer's CMake target is ``spherical_harmonics`` but its ``binding.cpp`` declares
   ``PYBIND11_MODULE(spherical_harmonics_rt, ...)``, and its ``__init__.py`` is empty — so
   upstream's own ``test.py`` cannot work as written. Fix: also expose the extension under
   its true name, ``spherical_harmonics_rt``.

3. ``ld: library 'fftw3_threads' not found``. GSound links FFTW by bare name and upstream's
   ``link_directories(${FFTW3_LIBRARY_DIRS})`` is always empty, so the build only linked
   because ``conda activate`` had exported ``LDFLAGS`` into CMake's cache — invisible shell
   state. Fix: pass the render env's include/library locations explicitly. See
   ``cmake_dependency_args``.

4. AppleDouble sidecars. On filesystems without native extended attributes (exFAT, FAT,
   network mounts) macOS writes a binary ``._name`` companion for every file. Upstream globs
   ``om/*/*.cpp``, so the compiler is handed one (``source file is not valid UTF-8``), and
   setuptools trips over ``._not-zip-safe`` while building the wheel. Fix:
   ``strip_appledouble`` on the clone, then compile from a local staging copy
   (``stage_sources``) because the build regenerates them.

Usage:
    python scripts/setup_gsound_sir.py \\
        --env-python /path/to/render-env/bin/python \\
        --ref 608ea30f6dc4cda149c18947f9cae48bd379fa27

See ``docs/gsound_sir_setup.md`` for creating the render environment itself, which this
script deliberately does not do (env creation is not portable across conda/venv/OS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_URL = "https://github.com/yongyizang/GSound-SIR"

# The auralizer's CMake target name and the name its PYBIND11_MODULE actually declares.
# They disagree upstream; we install under the target name and expose the true one.
AURALIZER_TARGET_NAME = "spherical_harmonics"
AURALIZER_MODULE_NAME = "spherical_harmonics_rt"

# Written into the render env's site-packages. pygsound exposes no version/SHA of its own,
# so this receipt is the provenance source of truth: the render backend reads it to report
# the *installed* SHA and compares it against the pinned `commit_sha` in
# configs/simulators/gsound_sir.yaml. It lives with the env, not with the (gitignored) clone.
RECEIPT_NAME = "amcd_gsound_install.json"

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# Per-OS toolchain hints, printed when preflight fails. Host detail, not code assumption.
_TOOLCHAIN_HINTS = {
    "darwin": "xcode-select --install    # C++ toolchain\n"
              "conda install -c conda-forge cmake fftw    # inside the render env",
    "linux": "sudo apt install build-essential cmake libfftw3-dev zlib1g-dev",
    "win32": "Install 'Desktop development with C++' (MSVC Build Tools) and CMake, "
             "or build under WSL.",
}


# --------------------------------------------------------------------------------------
# Pure helpers (no subprocess, no filesystem) — the unit-tested surface.
# --------------------------------------------------------------------------------------

def is_full_sha(ref: str) -> bool:
    """True for a full 40-character lowercase hex commit SHA."""
    return bool(_FULL_SHA.match(ref))


def parse_ls_remote(output: str, ref: str) -> str:
    """
    Pick the commit SHA for `ref` out of `git ls-remote` output.

    `ref` is "latest" (meaning the remote's default-branch HEAD) or a branch/tag name.
    Raises ValueError rather than guessing, so an unresolvable ref fails here and not
    halfway through a long build.
    """
    # Precedence follows git's own disambiguation: tags before branches, and for an
    # annotated tag the dereferenced commit (`^{}`) before the tag object itself — building
    # a tag object's SHA would fail at checkout.
    wanted = (
        ["HEAD"] if ref == "latest"
        else [f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}", f"refs/heads/{ref}", ref]
    )
    found: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and is_full_sha(parts[0]):
            found[parts[1]] = parts[0]

    for candidate in wanted:
        if candidate in found:
            return found[candidate]
    raise ValueError(
        f"ref {ref!r} did not resolve against {UPSTREAM_URL}; "
        f"remote advertised: {sorted(found)[:10] or 'nothing'}"
    )


def derive_build_temp(ray_generator_dir: Path, probe: dict) -> Path:
    """
    The directory setuptools will configure CMake in, for the *target* interpreter.

    setuptools builds C extensions in ``build/temp.{platform}-{cache_tag}``. We must
    pre-seed the CMake cache in precisely that directory: ``ray_generator/setup.py``
    ignores ``CMAKE_ARGS``, but its re-configure inherits cache entries already there.
    Derived from the target env's own sysconfig, never from the running interpreter's —
    they are different pythons (often different architectures).
    """
    return ray_generator_dir / "build" / f"temp.{probe['platform']}-{probe['cache_tag']}"


def find_libpython_links(link_output: str) -> list[str]:
    """
    Lines in ``otool -L`` / ``ldd`` output that link a CPython shared library.

    A non-empty result on the pygsound extension means the PYBIND11_FINDPYTHON=OFF fix did
    not take, and importing it will segfault against a statically linked python. Returned
    as lines (not a bool) so the failure message can show the offending dependency.
    """
    hits = []
    for line in link_output.splitlines():
        stripped = line.strip()
        # Match libpython3.10.dylib / libpython3.10.so.1.0 / python310.dll, not "python3.10"
        # appearing merely as a path component of the module's own location.
        if re.search(r"(lib)?python\d[\d.]*\.(dylib|so|dll)", stripped, re.IGNORECASE):
            hits.append(stripped)
    return hits


def is_appledouble(path: Path) -> bool:
    """
    True if `path` is a macOS AppleDouble sidecar (``._name``), by magic number.

    On filesystems that cannot store extended attributes natively — exFAT, FAT, many
    network mounts — macOS writes each file's xattrs to a companion ``._name`` file. These
    are binary, and upstream's ``file(GLOB SOURCES om/*/*.cpp)``
    (``ray_generator/src/Om/Om BVH/CMakeLists.txt:7``) happily matches ``._omBVHScene.cpp``,
    so the compiler is handed a sidecar and fails with "source file is not valid UTF-8".

    The magic check matters: name alone would also match a legitimate file someone chose to
    call ``._foo``, and this function's callers delete what it returns True for.
    """
    if not path.name.startswith("._") or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x00\x05\x16\x07"
    except OSError:
        return False


def strip_appledouble(root: Path) -> int:
    """
    Delete AppleDouble sidecars under `root`, returning how many were removed.

    Runs on every host — it is a no-op where the filesystem stores xattrs natively (APFS,
    HFS+, ext4, NTFS), so no platform branch is needed. Sidecars inside ``.git`` are removed
    too: they make git itself report "non-monotonic index" on pack files.
    """
    removed = 0
    for path in root.rglob("._*"):
        if is_appledouble(path):
            path.unlink()
            removed += 1
    return removed


def env_dependency_dirs(probe: dict) -> tuple[list[Path], list[Path]]:
    """
    The render env's include and library directories, as they exist on this host.

    Both conda layouts are covered: ``<prefix>/{include,lib}`` on Unix and
    ``<prefix>/Library/{include,lib}`` on Windows. Only directories that actually exist are
    returned, so the CMake arguments never point at nothing.
    """
    prefix = Path(probe["prefix"])
    include = [p for p in (prefix / "include", prefix / "Library" / "include") if p.is_dir()]
    lib = [p for p in (prefix / "lib", prefix / "Library" / "lib") if p.is_dir()]
    return include, lib


def cmake_dependency_args(probe: dict) -> list[str]:
    """
    CMake arguments locating the render env's own FFTW, without relying on shell activation.

    GSound links FFTW by bare name — ``target_link_libraries(... fftw3_threads fftw3f
    fftw3)`` in ``ray_generator/src/GSound/CMakeLists.txt:62`` — and upstream's
    ``link_directories(${FFTW3_LIBRARY_DIRS})`` is always empty because nothing ever sets
    that variable (there is no ``pkg_check_modules`` call). So the linker finds FFTW only if
    something puts ``-L`` on the command line.

    In the Step 0 spike that something was ``conda activate``: CMake seeds its linker-flag
    cache entries from ``$LDFLAGS`` on first configure, and conda's activation exports
    ``-L<prefix>/lib -Wl,-rpath,<prefix>/lib``. That makes a successful build depend on
    invisible shell state — precisely the folklore this installer replaces — so the flags
    are passed explicitly here instead.

    Deliberately does NOT set ``CMAKE_CXX_FLAGS``: upstream's ``setup.py`` injects
    ``-DVERSION_INFO`` through the ``CXXFLAGS`` environment variable, and CMake reads that
    env var only when the cache entry is unset. Seeding it here would silently drop the
    define. Include directories are supplied via search paths instead, which upstream's
    ``find_path(FFTW3_INCLUDE_DIR fftw3.h)`` honours.
    """
    include_dirs, lib_dirs = env_dependency_dirs(probe)
    prefixes = [str(Path(probe["prefix"])), str(Path(probe["prefix"]) / "Library")]

    args = [
        f"-DCMAKE_PREFIX_PATH={';'.join(prefixes)}",
        f"-DCMAKE_INCLUDE_PATH={';'.join(str(p) for p in include_dirs)}",
        f"-DCMAKE_LIBRARY_PATH={';'.join(str(p) for p in lib_dirs)}",
    ]

    # MSVC takes neither -L nor -rpath; there CMAKE_LIBRARY_PATH above is the mechanism.
    if not probe["sys_platform"].startswith("win"):
        flags = " ".join(f"-L{p} -Wl,-rpath,{p}" for p in lib_dirs)
        args += [
            f"-DCMAKE_SHARED_LINKER_FLAGS={flags}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={flags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={flags}",
        ]
    return args


#: Copied to the staging tree; everything else is build output or version-control state.
BUILD_SUBPACKAGES = ("ray_generator", "auralizer")

#: Never staged: sidecars (the reason for staging), git state, and prior build output.
_STAGING_EXCLUDES = shutil.ignore_patterns("._*", ".git", "build", "*.egg-info")


def stage_sources(clone_dir: Path, dest: Path,
                  subdirs: tuple[str, ...] = BUILD_SUBPACKAGES) -> Path:
    """
    Copy the buildable subpackages out of `clone_dir` into `dest`, returning `dest`.

    Used when the clone lives on a filesystem that cannot store extended attributes, where
    macOS writes AppleDouble sidecars. Cleaning the clone once is *not* sufficient: sidecars
    reappear during the build itself — setuptools writes ``not-zip-safe`` into egg-info,
    macOS immediately creates ``._not-zip-safe`` beside it, and ``bdist_wheel`` then fails
    copying a file that no longer resolves. Building from a local filesystem removes the
    whole class of failure rather than racing it.

    The clone stays authoritative for provenance — it is what was verified at the pinned
    SHA; this is only where compilation happens.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        shutil.copytree(
            clone_dir / subdir, dest / subdir, ignore=_STAGING_EXCLUDES, symlinks=True
        )
    return dest


def build_receipt(*, sha: str, env_python: str, probe: dict, checks: dict) -> dict:
    """Provenance record written into the render env. See RECEIPT_NAME."""
    return {
        "upstream_url": UPSTREAM_URL,
        "commit_sha": sha,
        "env_python": env_python,
        "machine": probe["machine"],
        "python_version": probe["python_version"],
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "installer": "scripts/setup_gsound_sir.py",
        "checks": checks,
    }


def read_receipt(purelib: Path) -> dict:
    """Read the install receipt from a render env's site-packages."""
    path = Path(purelib) / RECEIPT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {RECEIPT_NAME} in {purelib} — this env was not installed by "
            f"scripts/setup_gsound_sir.py, so its GSound-SIR commit SHA is unknown. "
            f"Re-run the installer against it."
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------
# Subprocess glue
# --------------------------------------------------------------------------------------

def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None,
         capture: bool = True) -> str:
    """Run a command, failing loudly with the command line included."""
    # Inline `-c` scripts are multi-line; collapse them so the log stays readable.
    printable = " ".join(
        "<inline script>" if "\n" in str(c) else str(c) for c in cmd
    )
    print(f"  $ {printable}", flush=True)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise RuntimeError(f"command failed ({result.returncode}): {printable}\n{detail}")
    return (result.stdout or "") if capture else ""


_PROBE_SRC = textwrap.dedent(
    """
    import json, platform, sys, sysconfig
    print(json.dumps({
        "machine": platform.machine(),
        "sys_platform": sys.platform,
        "python_version": platform.python_version(),
        "platform": sysconfig.get_platform(),
        "cache_tag": sys.implementation.cache_tag,
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "purelib": sysconfig.get_paths()["purelib"],
        "prefix": sys.prefix,
        "py_enable_shared": sysconfig.get_config_var("Py_ENABLE_SHARED"),
    }))
    """
)


def probe_env(env_python: Path) -> dict:
    """Ask the *target* interpreter to describe itself. All later paths derive from this."""
    return json.loads(_run([env_python, "-c", _PROBE_SRC]))


def resolve_cmake(env_python: Path) -> Path:
    """
    Find the CMake the build will use, preferring the render env's own.

    CMake is typically installed *into* the render env (conda package), so it may not be on
    the controlling shell's PATH — and the env's copy is the one built for the target
    architecture. Falls back to PATH for hosts with a system-wide CMake.
    """
    bindir = env_python.parent
    for candidate in (bindir / "cmake", bindir / "cmake.exe"):
        if candidate.exists():
            return candidate
    found = shutil.which("cmake")
    if found:
        return Path(found)
    hint = _TOOLCHAIN_HINTS.get(sys.platform, "install CMake")
    raise SystemExit(f"cmake not found in {bindir} or on PATH\n{hint}")


def build_env(env_python: Path, probe: dict) -> dict:
    """
    Environment for the build subprocesses.

    The render env's ``bin`` is prepended to PATH so that upstream's ``setup.py`` — which
    invokes a bare ``cmake`` — finds the same CMake this script configured with, rather than
    a different one from the controlling shell.

    ``CMAKE_ARGS`` is ignored by ``ray_generator``'s ``setup.py`` but forwarded by the
    auralizer's, and the auralizer needs it: its ``CMakeLists.txt`` calls
    ``find_package(Python COMPONENTS Interpreter Development REQUIRED)`` directly, and
    CMake's FindPython defaults to ``Python_FIND_STRATEGY=VERSION`` — it takes the *highest*
    Python on the system, not the one on PATH. Upstream passes only the classic
    ``PYTHON_EXECUTABLE``, which new-style FindPython ignores, so an unactivated build
    happily compiles the extension against whatever newest interpreter exists (observed:
    ``spherical_harmonics.cpython-313-darwin.so`` in a 3.10 environment, which cannot be
    imported). Naming the interpreter explicitly pins it to the render env.
    """
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(env_python.parent), env.get("PATH", "")])
    # Space-separated because that is how the auralizer's setup.py splits CMAKE_ARGS;
    # these paths come from sysconfig and contain no spaces.
    env["CMAKE_ARGS"] = " ".join([
        "-DPYBIND11_FINDPYTHON=OFF",
        f"-DPython_EXECUTABLE={env_python}",
        f"-DPython_ROOT_DIR={probe['prefix']}",
        "-DPython_FIND_STRATEGY=LOCATION",
    ])
    return env


def preflight(env_python: Path) -> None:
    """Fail before any long build if the toolchain or target interpreter is missing."""
    if not env_python.exists():
        raise SystemExit(f"--env-python does not exist: {env_python}")
    if shutil.which("git") is None:
        hint = _TOOLCHAIN_HINTS.get(sys.platform, "install git")
        raise SystemExit(f"missing required tool: git\n{hint}")
    resolve_cmake(env_python)  # raises SystemExit with a per-OS hint if absent


def resolve_ref(ref: str) -> str:
    """Resolve a SHA / branch / 'latest' to a concrete SHA, used for everything downstream."""
    if is_full_sha(ref):
        return ref
    print(f"Resolving ref {ref!r} against {UPSTREAM_URL} ...")
    query = ["git", "ls-remote", UPSTREAM_URL]
    if ref != "latest":
        query.append(ref)
    return parse_ls_remote(_run(query), ref)


def sync_clone(clone_dir: Path, sha: str) -> int:
    """
    Clone (or update) upstream and check out exactly `sha`, submodules included.

    Returns the number of AppleDouble sidecars removed — a nonzero count identifies a
    filesystem that will keep producing them during the build, which is what selects the
    staging path in `main`.
    """
    if not (clone_dir / ".git").exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--recurse-submodules", UPSTREAM_URL, str(clone_dir)],
             capture=False)
    else:
        _run(["git", "fetch", "--tags", "origin"], cwd=clone_dir, capture=False)

    _run(["git", "checkout", "--force", sha], cwd=clone_dir, capture=False)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=clone_dir,
         capture=False)

    head = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir).strip()
    if head != sha:
        raise RuntimeError(f"checkout verification failed: HEAD is {head}, expected {sha}")
    print(f"  clone at {sha}")

    # Must happen before CMake configures: upstream globs *.cpp, and on an
    # xattr-less filesystem the glob picks up binary AppleDouble sidecars.
    removed = strip_appledouble(clone_dir)
    if removed:
        print(f"  removed {removed} AppleDouble sidecar file(s) — see strip_appledouble()")
    return removed


def install_ray_generator(clone_dir: Path, env_python: Path, probe: dict,
                          rebuild: bool) -> None:
    """Build+install pygsound with the FINDPYTHON fix delivered via a pre-seeded cache."""
    source = clone_dir / "ray_generator"
    build_temp = derive_build_temp(source, probe)

    if rebuild and (source / "build").exists():
        shutil.rmtree(source / "build")

    build_temp.mkdir(parents=True, exist_ok=True)
    print("Configuring pygsound with PYBIND11_FINDPYTHON=OFF (see module docstring) ...")
    _run(
        [
            resolve_cmake(env_python), "-S", str(source), "-B", str(build_temp),
            "-DPYBIND11_FINDPYTHON=OFF",
            f"-DPYTHON_EXECUTABLE={env_python}",
            "-DCMAKE_BUILD_TYPE=Release",
            *cmake_dependency_args(probe),
        ],
        capture=False,
    )

    print("Building pygsound (slow under emulation) ...")
    _run([env_python, "-m", "pip", "install", str(source)],
         env=build_env(env_python, probe), capture=False)


def install_auralizer(clone_dir: Path, env_python: Path, probe: dict, rebuild: bool) -> Path:
    """
    Build+install the auralizer, then expose it under the name it actually declares.

    Returns the path of the ``spherical_harmonics_rt`` extension that was created.
    """
    source = clone_dir / "auralizer"
    if rebuild and (source / "build").exists():
        shutil.rmtree(source / "build")

    print("Building auralizer (fetches pybind11 v2.11.1 — needs network) ...")
    _run([env_python, "-m", "pip", "install", str(source)],
         env=build_env(env_python, probe), capture=False)

    purelib = Path(probe["purelib"])
    suffix = probe["ext_suffix"]
    built = purelib / f"{AURALIZER_TARGET_NAME}{suffix}"
    if not built.exists():
        # A build against the wrong interpreter still installs an extension — just one
        # tagged for another Python, which the render env cannot import. Naming what WAS
        # produced turns a bare "missing file" into the actual diagnosis.
        others = sorted(p.name for p in purelib.glob(f"{AURALIZER_TARGET_NAME}*.so"))
        raise RuntimeError(
            f"auralizer install produced no {built.name} in {purelib}; cannot expose it as "
            f"{AURALIZER_MODULE_NAME}."
            + (f"\n  found instead: {', '.join(others)}"
               f"\n  That extension targets a different Python than {env_python}."
               if others else "")
        )
    exposed = purelib / f"{AURALIZER_MODULE_NAME}{suffix}"
    shutil.copy2(built, exposed)
    print(f"  exposed {built.name} as {exposed.name} (true PYBIND11_MODULE name)")
    return exposed


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

_IMPORT_CHECK_SRC = textwrap.dedent(
    """
    import json, platform
    import pygsound
    import spherical_harmonics_rt as sh
    print(json.dumps({
        "machine": platform.machine(),
        "pygsound_file": pygsound.__file__,
        "generate_ambisonic_ir_doc": (sh.generate_ambisonic_ir.__doc__ or "").splitlines()[0],
        "has_createbox": hasattr(pygsound, "createbox"),
        "has_context": hasattr(pygsound, "Context"),
    }))
    """
)


def verify(env_python: Path, probe: dict) -> dict:
    """
    Prove the env is usable, or raise. Returns the checks recorded in the receipt.

    Import success alone is not enough: a contaminated link line can import fine on one host
    and segfault on another, and an unexpected upstream ref can change the auralizer API
    under us. Both are checked here so failure lands at setup time, not mid-render.
    """
    checks: dict = {"machine": probe["machine"], "py_enable_shared": probe["py_enable_shared"]}

    print("Verifying imports in the render env ...")
    info = json.loads(_run([env_python, "-c", _IMPORT_CHECK_SRC]))
    checks["imports_ok"] = True
    for attr in ("has_createbox", "has_context"):
        if not info[attr]:
            raise RuntimeError(f"pygsound imported but {attr} is False — unexpected build")

    # API guard: the pinned ref's generate_ambisonic_ir takes band-edge `frequency_points`
    # and NO `path_types`. Upstream's own test.py passes a path_types argument; a ref where
    # that is true would break the render worker, so catch it here.
    signature = info["generate_ambisonic_ir_doc"]
    if "path_types" in signature:
        raise RuntimeError(
            "auralizer API mismatch: generate_ambisonic_ir accepts `path_types`, which the "
            f"render worker does not pass.\n  signature: {signature}"
        )
    if "frequency_points" not in signature:
        raise RuntimeError(
            f"auralizer API mismatch: no `frequency_points` parameter.\n  signature: {signature}"
        )
    checks["auralizer_signature"] = signature

    # Link-line check: no CPython shared library on the pygsound extension.
    pygsound_ext = Path(info["pygsound_file"]).with_name(f"pygsound{probe['ext_suffix']}")
    checks["libpython_check"] = _check_link_line(pygsound_ext, probe)

    print("  verification passed")
    return checks


def _check_link_line(extension: Path, probe: dict) -> str:
    """
    Assert no libpython on `extension`, using the platform's own dependency lister.

    Skipped on Windows, where python is always a DLL and linking it is correct — the defect
    this guards against only exists where the interpreter is statically linked.
    """
    platform_name = probe["sys_platform"]
    if platform_name.startswith("win"):
        return "skipped (windows: shared python is expected)"
    lister = ["otool", "-L"] if platform_name == "darwin" else ["ldd"]
    if shutil.which(lister[0]) is None:
        return f"skipped ({lister[0]} not available)"
    if not extension.exists():
        raise RuntimeError(f"pygsound extension not found at {extension}")

    output = _run([*lister, str(extension)])
    hits = find_libpython_links(output)
    if hits:
        raise RuntimeError(
            "pygsound links a CPython shared library — the PYBIND11_FINDPYTHON=OFF fix did "
            "not take, and importing it will segfault against a statically linked python.\n"
            "  offending: " + "; ".join(hits) + "\n"
            "  Re-run with --rebuild to discard the stale CMake cache."
        )
    return "clean (no libpython)"


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""
            The render environment must already exist; see docs/gsound_sir_setup.md.
            On success the resolved SHA is printed for configs/simulators/gsound_sir.yaml,
            and an install receipt ({RECEIPT_NAME}) is written into the env's site-packages.
            """
        ),
    )
    parser.add_argument("--env-python", required=True, type=Path,
                        help="interpreter of the render environment to install into")
    parser.add_argument("--ref", required=True,
                        help="upstream ref: a full 40-char commit SHA, a branch/tag, or "
                             "'latest' (the remote's default-branch HEAD). Resolved to a "
                             "concrete SHA before anything is built.")
    parser.add_argument("--clone-dir", type=Path, default=None,
                        help="where to clone GSound-SIR "
                             "(default: <repo>/external/GSound-SIR, gitignored)")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete existing build/ directories first (use when a stale "
                             "CMake cache is suspected)")
    parser.add_argument("--verify-only", action="store_true",
                        help="run verification against --env-python without cloning or "
                             "building")
    args = parser.parse_args(argv)

    env_python = args.env_python.resolve()
    clone_dir = (args.clone_dir or _repo_root() / "external" / "GSound-SIR").resolve()

    preflight(env_python)
    probe = probe_env(env_python)
    print(f"Render env: {env_python}")
    print(f"  machine={probe['machine']} python={probe['python_version']} "
          f"platform={probe['platform']} Py_ENABLE_SHARED={probe['py_enable_shared']}")

    if args.verify_only:
        checks = verify(env_python, probe)
        receipt_path = Path(probe["purelib"]) / RECEIPT_NAME
        if receipt_path.exists():
            print(f"  installed SHA: {read_receipt(probe['purelib'])['commit_sha']}")
        else:
            print(f"  WARNING: no {RECEIPT_NAME} — installed SHA unknown")
        print(json.dumps(checks, indent=2))
        return 0

    sha = resolve_ref(args.ref)
    print(f"Resolved ref {args.ref!r} -> {sha}")
    sidecars = sync_clone(clone_dir, sha)

    # A clone that produced sidecars sits on a filesystem without native extended
    # attributes, which breaks the build in ways a one-shot clean cannot fix (see
    # stage_sources). Compile from a local copy instead; the clone remains the SHA-verified
    # provenance record.
    staging = None
    build_root = clone_dir
    try:
        if sidecars:
            staging = Path(tempfile.mkdtemp(prefix="amcd-gsound-build-"))
            print(f"Staging sources on a local filesystem: {staging}")
            build_root = stage_sources(clone_dir, staging)

        install_ray_generator(build_root, env_python, probe, args.rebuild)
        install_auralizer(build_root, env_python, probe, args.rebuild)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    checks = verify(env_python, probe)

    receipt = build_receipt(sha=sha, env_python=str(env_python), probe=probe, checks=checks)
    receipt_path = Path(probe["purelib"]) / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Receipt written: {receipt_path}")

    print("\nDone. Pin this in configs/simulators/gsound_sir.yaml:\n")
    print(f"  commit_sha: {sha}\n")
    # There is no `--sim-python` flag: the interpreter is a HOST fact, so it is a
    # simulator config key supplied by an uncommitted host layer, not a CLI switch.
    print("And name this interpreter in a host-local config layer (not committed):\n")
    print(f"  simulator:\n    params:\n      render_python: {env_python}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
