"""
Unit tests for the GSound-SIR installer (`scripts/setup_gsound_sir.py`).

These cover only the installer's pure logic — ref resolution, build-path derivation,
link-line parsing, receipt round-trip. They never clone, build, or touch a render
environment, so they run everywhere the rest of the suite does (including arm64, where
GSound-SIR cannot even be built). The installer is deliberately factored so that this
logic is separable from the subprocess glue; the glue is exercised by the from-scratch
build recorded in `docs/gsound_sir_setup.md`.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "setup_gsound_sir.py"


def _load_installer():
    """Import the installer by path — `scripts/` is host tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("setup_gsound_sir", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = _load_installer()

PINNED_SHA = "608ea30f6dc4cda149c18947f9cae48bd379fa27"

# A target-env probe as `probe_env` returns it, matching the verified osx-64 render env.
PROBE = {
    "machine": "x86_64",
    "sys_platform": "darwin",
    "python_version": "3.10.18",
    "platform": "macosx-10.15-x86_64",
    "cache_tag": "cpython-310",
    "ext_suffix": ".cpython-310-darwin.so",
    "purelib": "/tmp/env/lib/python3.10/site-packages",
    "py_enable_shared": 0,
}


class TestRefResolution:
    def test_full_sha_recognised(self):
        assert installer.is_full_sha(PINNED_SHA)

    @pytest.mark.parametrize("ref", ["main", "latest", "608ea30", PINNED_SHA.upper(), ""])
    def test_non_full_sha_rejected(self, ref):
        assert not installer.is_full_sha(ref)

    def test_latest_resolves_to_head(self):
        output = f"{PINNED_SHA}\tHEAD\n{'b' * 40}\trefs/heads/main\n"
        assert installer.parse_ls_remote(output, "latest") == PINNED_SHA

    def test_branch_resolves_to_its_ref(self):
        output = f"{'a' * 40}\tHEAD\n{PINNED_SHA}\trefs/heads/main\n"
        assert installer.parse_ls_remote(output, "main") == PINNED_SHA

    def test_annotated_tag_prefers_dereferenced_commit(self):
        # `refs/tags/v1^{}` is the commit an annotated tag points at; the bare ref is the
        # tag object itself, which is not what we want to build.
        output = f"{'a' * 40}\trefs/tags/v1\n{PINNED_SHA}\trefs/tags/v1^{{}}\n"
        assert installer.parse_ls_remote(output, "v1") == PINNED_SHA

    def test_unresolvable_ref_raises(self):
        # Fail at resolution rather than halfway through a long build.
        with pytest.raises(ValueError, match="did not resolve"):
            installer.parse_ls_remote(f"{PINNED_SHA}\trefs/heads/main\n", "nonexistent")

    def test_garbage_output_raises(self):
        with pytest.raises(ValueError):
            installer.parse_ls_remote("fatal: repository not found\n", "latest")


class TestBuildTempDerivation:
    def test_matches_setuptools_layout(self):
        # setuptools builds in build/temp.{platform}-{cache_tag}; the CMake cache must be
        # pre-seeded in exactly that directory or the FINDPYTHON fix is silently dropped.
        got = installer.derive_build_temp(Path("/x/ray_generator"), PROBE)
        assert got == Path("/x/ray_generator/build/temp.macosx-10.15-x86_64-cpython-310")

    def test_uses_target_env_not_host(self):
        # The controlling interpreter is typically arm64 python 3.11; the target is x86_64
        # python 3.10. Deriving from the host would seed the wrong directory.
        linux = dict(PROBE, platform="linux-x86_64", cache_tag="cpython-311")
        got = installer.derive_build_temp(Path("/x/ray_generator"), linux)
        assert got.name == "temp.linux-x86_64-cpython-311"


class TestLinkLineParsing:
    CLEAN_OTOOL = """\
/env/site-packages/pygsound/pygsound.cpython-310-darwin.so:
\t@rpath/pygsound.cpython-310-darwin.so (compatibility version 0.0.0, current version 0.0.0)
\t@rpath/libfftw3f.3.dylib (compatibility version 11.0.0, current version 11.11.0)
\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2100.43.0)
"""
    CONTAMINATED_OTOOL = CLEAN_OTOOL + (
        "\t/env/lib/libpython3.10.dylib (compatibility version 3.10.0, current version 3.10.0)\n"
    )
    CLEAN_LDD = "\tlibstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f00)\n"
    CONTAMINATED_LDD = CLEAN_LDD + "\tlibpython3.10.so.1.0 => /env/lib/libpython3.10.so.1.0\n"

    def test_clean_otool_has_no_hits(self):
        # The module's own name contains "cpython-310" — it must not be mistaken for a
        # libpython dependency.
        assert installer.find_libpython_links(self.CLEAN_OTOOL) == []

    def test_contaminated_otool_detected(self):
        hits = installer.find_libpython_links(self.CONTAMINATED_OTOOL)
        assert len(hits) == 1 and "libpython3.10.dylib" in hits[0]

    def test_clean_ldd_has_no_hits(self):
        assert installer.find_libpython_links(self.CLEAN_LDD) == []

    def test_contaminated_ldd_detected(self):
        hits = installer.find_libpython_links(self.CONTAMINATED_LDD)
        assert len(hits) == 1 and "libpython3.10.so" in hits[0]

    def test_windows_dll_detected(self):
        assert installer.find_libpython_links("    python310.dll => C:\\env\\python310.dll")


class TestCMakeDependencyArgs:
    """
    GSound links FFTW by bare name with no working search path of its own, so a build only
    succeeds if the installer supplies one. Getting this wrong is what made the Step 0 build
    silently depend on `conda activate`.
    """

    @staticmethod
    def _probe_with_env(tmp_path, sys_platform="darwin", windows_layout=False):
        base = tmp_path / "Library" if windows_layout else tmp_path
        (base / "include").mkdir(parents=True)
        (base / "lib").mkdir(parents=True)
        return dict(PROBE, prefix=str(tmp_path), sys_platform=sys_platform)

    def test_unix_layout_dirs_found(self, tmp_path):
        probe = self._probe_with_env(tmp_path)
        include, lib = installer.env_dependency_dirs(probe)
        assert include == [tmp_path / "include"] and lib == [tmp_path / "lib"]

    def test_windows_layout_dirs_found(self, tmp_path):
        probe = self._probe_with_env(tmp_path, "win32", windows_layout=True)
        include, lib = installer.env_dependency_dirs(probe)
        assert include == [tmp_path / "Library" / "include"]
        assert lib == [tmp_path / "Library" / "lib"]

    def test_nonexistent_dirs_omitted(self, tmp_path):
        # Never point CMake at a directory that isn't there.
        probe = dict(PROBE, prefix=str(tmp_path))
        assert installer.env_dependency_dirs(probe) == ([], [])

    def test_linker_flags_carry_search_path_and_rpath(self, tmp_path):
        # -L makes the link succeed; -rpath makes the resulting .so find FFTW at runtime.
        args = installer.cmake_dependency_args(self._probe_with_env(tmp_path))
        shared = next(a for a in args if a.startswith("-DCMAKE_SHARED_LINKER_FLAGS="))
        assert f"-L{tmp_path / 'lib'}" in shared
        assert f"-Wl,-rpath,{tmp_path / 'lib'}" in shared

    def test_windows_gets_no_posix_linker_flags(self, tmp_path):
        # MSVC accepts neither -L nor -rpath; CMAKE_LIBRARY_PATH is the mechanism there.
        args = installer.cmake_dependency_args(
            self._probe_with_env(tmp_path, "win32", windows_layout=True)
        )
        assert not any("LINKER_FLAGS" in a for a in args)
        assert any(a.startswith("-DCMAKE_LIBRARY_PATH=") for a in args)

    def test_cxx_flags_never_seeded(self, tmp_path):
        # Seeding CMAKE_CXX_FLAGS would make CMake ignore the CXXFLAGS env var that
        # upstream's setup.py uses to inject -DVERSION_INFO.
        args = installer.cmake_dependency_args(self._probe_with_env(tmp_path))
        assert not any("CMAKE_CXX_FLAGS" in a for a in args)


class TestAppleDoubleStripping:
    """
    On exFAT/FAT/network mounts macOS stores xattrs in binary `._name` sidecars, and
    upstream globs `om/*/*.cpp` — so the compiler gets handed a sidecar. The installer
    deletes them, which makes the magic-number check load-bearing.
    """
    MAGIC = b"\x00\x05\x16\x07"

    def test_sidecar_detected_by_magic(self, tmp_path):
        sidecar = tmp_path / "._omBVHScene.cpp"
        sidecar.write_bytes(self.MAGIC + b"\x00\x02\x00\x00binary junk")
        assert installer.is_appledouble(sidecar)

    def test_name_alone_is_not_enough(self, tmp_path):
        # A real file that merely happens to be named `._x` must survive — this function's
        # callers delete whatever it approves.
        impostor = tmp_path / "._notasidecar.cpp"
        impostor.write_text("int main() { return 0; }\n")
        assert not installer.is_appledouble(impostor)

    def test_ordinary_file_ignored(self, tmp_path):
        source = tmp_path / "omBVHScene.cpp"
        source.write_bytes(self.MAGIC)  # magic bytes, but not a `._` name
        assert not installer.is_appledouble(source)

    def test_strip_removes_only_sidecars_recursively(self, tmp_path):
        nested = tmp_path / "src" / "Om" / "Om BVH" / "om" / "bvh"
        nested.mkdir(parents=True)
        (nested / "._omBVHScene.cpp").write_bytes(self.MAGIC + b"junk")
        (nested / "omBVHScene.cpp").write_text("// real source\n")
        (tmp_path / "._impostor.txt").write_text("not a sidecar")

        assert installer.strip_appledouble(tmp_path) == 1
        assert not (nested / "._omBVHScene.cpp").exists()
        assert (nested / "omBVHScene.cpp").read_text() == "// real source\n"
        assert (tmp_path / "._impostor.txt").exists()

    def test_noop_on_clean_tree(self, tmp_path):
        # Native-xattr hosts (APFS, ext4, NTFS) have no sidecars; no platform branch needed.
        (tmp_path / "omBVHScene.cpp").write_text("// real source\n")
        assert installer.strip_appledouble(tmp_path) == 0


class TestStaging:
    """
    On a sidecar-producing filesystem the build must run from a local copy: sidecars
    reappear *during* the build (setuptools' `not-zip-safe` gets a `._not-zip-safe`
    companion), which a one-shot clean of the clone cannot prevent.
    """

    @staticmethod
    def _make_clone(root):
        source = root / "ray_generator" / "src"
        source.mkdir(parents=True)
        (source / "Context.cpp").write_text("// real source\n")
        (source / "._Context.cpp").write_bytes(b"\x00\x05\x16\x07junk")
        (root / "ray_generator" / ".git").mkdir()
        (root / "ray_generator" / ".git" / "config").write_text("[core]\n")
        (root / "ray_generator" / "build").mkdir()
        (root / "ray_generator" / "build" / "stale.o").write_text("stale")
        (root / "ray_generator" / "src" / "pygsound.egg-info").mkdir()
        (root / "auralizer").mkdir()
        (root / "auralizer" / "setup.py").write_text("# auralizer\n")
        return root

    def test_real_sources_copied(self, tmp_path):
        clone = self._make_clone(tmp_path / "clone")
        dest = installer.stage_sources(clone, tmp_path / "staging")
        assert (dest / "ray_generator" / "src" / "Context.cpp").read_text() == "// real source\n"
        assert (dest / "auralizer" / "setup.py").exists()

    def test_sidecars_git_and_build_output_excluded(self, tmp_path):
        clone = self._make_clone(tmp_path / "clone")
        dest = installer.stage_sources(clone, tmp_path / "staging")
        assert not (dest / "ray_generator" / "src" / "._Context.cpp").exists()
        assert not (dest / "ray_generator" / ".git").exists()
        assert not (dest / "ray_generator" / "build").exists()
        assert not (dest / "ray_generator" / "src" / "pygsound.egg-info").exists()

    def test_clone_is_left_untouched(self, tmp_path):
        # The clone stays the SHA-verified provenance record; staging only copies from it.
        clone = self._make_clone(tmp_path / "clone")
        installer.stage_sources(clone, tmp_path / "staging")
        assert (clone / "ray_generator" / ".git" / "config").exists()
        assert (clone / "ray_generator" / "src" / "Context.cpp").exists()


class TestReceipt:
    def test_round_trip(self, tmp_path):
        checks = {"imports_ok": True, "libpython_check": "clean (no libpython)"}
        receipt = installer.build_receipt(
            sha=PINNED_SHA, env_python="/env/bin/python", probe=PROBE, checks=checks
        )
        (tmp_path / installer.RECEIPT_NAME).write_text(json.dumps(receipt))

        got = installer.read_receipt(tmp_path)
        assert got["commit_sha"] == PINNED_SHA
        assert got["upstream_url"] == installer.UPSTREAM_URL
        assert got["machine"] == "x86_64"
        assert got["checks"] == checks

    def test_missing_receipt_names_the_consequence(self, tmp_path):
        # An env without a receipt has an unknown installed SHA, so the backend's
        # installed==pinned check cannot run. That must be an error, not a silent skip.
        with pytest.raises(FileNotFoundError, match="commit SHA is unknown"):
            installer.read_receipt(tmp_path)


class TestCLI:
    def test_env_python_and_ref_are_required(self):
        # No hidden defaults: the target interpreter and upstream ref must both be stated.
        with pytest.raises(SystemExit):
            installer.main([])
        with pytest.raises(SystemExit):
            installer.main(["--env-python", "/nonexistent/python"])

    def test_missing_env_python_fails_before_any_work(self, tmp_path):
        with pytest.raises(SystemExit, match="does not exist"):
            installer.main(["--env-python", str(tmp_path / "python"), "--ref", "latest"])
