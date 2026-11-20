"""Build egrabber_ingest.dll and eg_grab.exe with clang-cl (the same compiler the b4d extension uses).

    python egrabber/build.py            # release build into egrabber/bin
    python egrabber/build.py --debug

The DLL is also copied into the package directory (src/egrabber_camera/egrabber_ingest.dll), so an editable install
finds it there and a wheel built afterwards ships it (pyproject package-data).

Requirements: the Euresys eGrabber SDK (headers in %EURESYS_EGRABBER_ROOT%\\include or the default install path),
clang-cl on PATH (conda env b4d has it) with the MSVC toolchain / Windows SDK it auto-detects. The eGrabber C++ API
is header-only over the GenTL producer (coaxlink.cti), which is loaded at run time, so nothing is linked.
"""
import os, subprocess, sys, shutil, time

HERE = os.path.abspath(os.path.dirname(__file__))
SDK = os.environ.get("EURESYS_EGRABBER_ROOT", r"C:\Program Files\Euresys\eGrabber")
INC = os.path.join(SDK, "include")
BIN = os.path.join(HERE, "bin")
OBJ = os.path.join(HERE, "build")


def run(cmd):
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"command failed ({r.returncode})")


def main():
    debug = "--debug" in sys.argv
    if not os.path.exists(os.path.join(INC, "EGrabber.h")):
        sys.exit(f"eGrabber SDK headers not found at {INC}; set EURESYS_EGRABBER_ROOT")
    cc = shutil.which("clang-cl")
    if not cc:
        sys.exit("clang-cl not found on PATH (activate the b4d conda env)")
    os.makedirs(BIN, exist_ok=True); os.makedirs(OBJ, exist_ok=True)
    common = ["-nologo", "-EHsc", "-std:c++17", "-W3", "-D_CRT_SECURE_NO_WARNINGS", "-DNOMINMAX", f"-I{INC}",
              "-Zi" if debug else "-O2", "-MDd" if debug else "-MD"]
    if not debug:
        common += ["/clang:-O2"]
    t0 = time.time()
    print("building egrabber_ingest.dll")
    run([cc] + common + ["-LD", os.path.join(HERE, "src", "cpp", "egrabber_ingest.cpp"),
         f"-Fo{OBJ}\\", f"-Fe{os.path.join(BIN, 'egrabber_ingest.dll')}", "-link", "advapi32.lib", "-DEBUG" if debug else "-RELEASE", f"-IMPLIB:{os.path.join(OBJ, 'egrabber_ingest.lib')}"])
    print("building eg_grab.exe (links the engine statically)")
    run([cc] + common + ["-DEG_INGEST_BUILD", os.path.join(HERE, "src", "cpp", "eg_grab.cpp"), os.path.join(HERE, "src", "cpp", "egrabber_ingest.cpp"),
         f"-Fo{OBJ}\\", f"-Fe{os.path.join(BIN, 'eg_grab.exe')}", "-link", "advapi32.lib"])
    pkg_dll = os.path.join(HERE, "src", "egrabber_camera", "egrabber_ingest.dll")
    shutil.copy2(os.path.join(BIN, "egrabber_ingest.dll"), pkg_dll)
    print(f"done in {time.time() - t0:.1f} s -> {BIN} (DLL also copied to {pkg_dll})")


if __name__ == "__main__":
    main()
