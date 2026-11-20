"""
Build the b4d Python package.

Blosc2 is located, in order, via:
  1. BLOSC2_DIR=/path/to/c-blosc2 (source tree built in $BLOSC2_DIR/build)
  2. pkg-config blosc2               (e.g. `brew install c-blosc2 pkg-config`)
The build is portable: no -march / -mcpu, only -mtune=native (x86-64 and other non-Apple-arm64 gcc/clang builds) and,
with MSVC, /arch:AVX2. B4D_CFLAGS adds compiler flags (e.g. "-march=native" for a machine-specific build), B4D_CC swaps
the Windows compiler (clang-cl).
"""
import os
import platform
import shutil
import subprocess
import sys
from setuptools import setup, Extension

HERE = os.path.abspath(os.path.dirname(__file__))
CODEC_DIR = "src/c"                                              # the C codec + native zarr writer (setup() wants relative /-paths)
PKG_DIR = "src/b4d"                                              # the Python package (src layout)

include_dirs = [os.path.join(HERE, CODEC_DIR)]
library_dirs, libraries, extra_objects, extra_link = [], [], [], []

blosc_dir = os.environ.get("BLOSC2_DIR")
conda = os.environ.get("CONDA_PREFIX")
vcpkg = os.environ.get("VCPKG_ROOT")
if sys.platform == "win32":
    # Windows: BLOSC2_DIR with include/ and lib/blosc2.lib, or a conda env (conda install -c conda-forge c-blosc2),
    # or vcpkg (vcpkg install blosc2:x64-windows)
    cands = []
    if blosc_dir: cands.append((os.path.join(blosc_dir, "include"), os.path.join(blosc_dir, "lib")))
    if conda: cands.append((os.path.join(conda, "Library", "include"), os.path.join(conda, "Library", "lib")))
    if vcpkg: cands.append((os.path.join(vcpkg, "installed", "x64-windows", "include"), os.path.join(vcpkg, "installed", "x64-windows", "lib")))
    for inc, lib in cands:
        if os.path.exists(os.path.join(inc, "blosc2.h")):
            include_dirs.append(inc); library_dirs.append(lib); libraries.append("blosc2"); break
    else:
        sys.exit("Cannot find blosc2.h: set BLOSC2_DIR=<prefix with include\\blosc2.h and lib\\blosc2.lib>, or run inside a conda env with c-blosc2 installed")
elif blosc_dir:
    include_dirs.append(os.path.join(blosc_dir, "include"))
    static = os.path.join(blosc_dir, "build", "blosc", "libblosc2.a")
    if os.path.exists(static):
        extra_objects.append(static)
    else:
        library_dirs.append(os.path.join(blosc_dir, "build", "blosc")); libraries.append("blosc2")
elif shutil.which("pkg-config"):
    def pc(flag):
        return subprocess.check_output(["pkg-config", flag, "blosc2"]).decode().split()
    for f in pc("--cflags"):
        if f.startswith("-I"): include_dirs.append(f[2:])
    for f in pc("--libs"):
        if f.startswith("-L"): library_dirs.append(f[2:])
        elif f.startswith("-l"): libraries.append(f[2:])
        else: extra_link.append(f)
else:
    sys.exit("Cannot find Blosc2: set BLOSC2_DIR or install pkg-config + c-blosc2")

if sys.platform != "win32":
    libraries += ["m", "pthread"]

cflags = ["-O3", "-fno-math-errno", "-std=gnu11"] if sys.platform != "win32" else ["/O2", "/arch:AVX2", "/std:c11", "/D_CRT_SECURE_NO_WARNINGS"]
if sys.platform != "win32" and not (sys.platform == "darwin" and platform.machine() == "arm64"):
    cflags.append("-mtune=native")          # scheduling only: the binary runs on any CPU of the target architecture
# B4D_CFLAGS: extra compiler flags (space separated, after the defaults), e.g. for clang-cl: "/clang:-O3 /clang:-mtune=native";
# a machine-specific (non-portable) build adds -march=native here
cflags += os.environ.get("B4D_CFLAGS", "").split()

ext = Extension(
    "b4d._b4d",
    sources=[PKG_DIR + "/_b4d.c", CODEC_DIR + "/b4d_blosc2.c", CODEC_DIR + "/b4d_zarr.c", CODEC_DIR + "/b4d_original.c"],
    include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries,
    extra_objects=extra_objects, extra_compile_args=cflags, extra_link_args=extra_link,
)

from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """B4D_CC=clang-cl (or a full path) swaps the Windows compiler: clang-cl is a drop-in for cl.exe (same flags) and
    usually generates tighter code for the codec's integer loops. MSVC's whole-program flags (/GL, /LTCG) are dropped
    for it, and lld-link is used when it sits next to clang-cl: with /GL + link.exe /LTCG the writer compressed 13 %
    slower (Teledyne Kinetix block, 128 threads). On other platforms CC works."""

    def build_extensions(self):
        cc = os.environ.get("B4D_CC")
        if cc and sys.platform == "win32":
            # the MSVC compiler object resolves cl.exe lazily in initialize(); run it first, then replace the executable
            if hasattr(self.compiler, "initialize") and not getattr(self.compiler, "initialized", False):
                self.compiler.initialize()
            self.compiler.cc = cc
            for attr in ("compile_options", "compile_options_debug"):
                if hasattr(self.compiler, attr):
                    setattr(self.compiler, attr, [f for f in getattr(self.compiler, attr) if f != "/GL"])
            for attr in ("ldflags_shared", "ldflags_shared_debug"):
                if hasattr(self.compiler, attr):
                    setattr(self.compiler, attr, [f for f in getattr(self.compiler, attr) if f != "/LTCG"])
            ldflags = getattr(self.compiler, "_ldflags", None)      # newer setuptools: {(target, debug): [flags]}
            if isinstance(ldflags, dict):
                for k, v in ldflags.items():
                    ldflags[k] = [f for f in v if f != "/LTCG"]
            lld = os.path.join(os.path.dirname(cc), "lld-link.exe")
            if os.path.isfile(lld):
                self.compiler.linker = lld
            print(f"[b4d] compiling the extension with {cc}, linking with {self.compiler.linker} (no /GL, /LTCG)",
                  file=sys.stderr)
        super().build_extensions()


# metadata (name, version, dependencies, entry points) lives in pyproject.toml; this file only builds the extension
setup(
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
    packages=["b4d", "b4d.ingest"],
    package_data={"b4d": ["b4d_rice_kernel.h"]},
    ext_modules=[ext],
)
