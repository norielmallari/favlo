"""Compile and run the CUDA dispatch check for ``favlo --qwen35_4b kernels``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from favlo.host import cuda_home

_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "dispatch.cu"
_ENGINE = _HERE / "engine.cu"
_BUILD = _HERE / "build"
_EXE = _BUILD / ("dispatch.exe" if os.name == "nt" else "dispatch")
_DLL = _BUILD / ("engine.dll" if os.name == "nt" else "libengine.so")
# RTX 30 is sm_86, RTX 40 is sm_89, RTX 50 is sm_120. The PTX image
# lets a later 12.x device JIT when it has no cubin of its own.
_GENCODE = (
    "arch=compute_86,code=sm_86",
    "arch=compute_89,code=sm_89",
    "arch=compute_120,code=sm_120",
    "arch=compute_120,code=compute_120",
)


def ensure_engine() -> Path:
    """Compile the language step when its source is newer than the library."""
    home = cuda_home()
    if home is None:
        message = "CUDA Toolkit is not installed"
        raise RuntimeError(message)
    return _compile_engine(home)


def run_kernels() -> None:
    """Compile the dispatch when the source is newer, then run it on this device."""
    home = cuda_home()
    if home is None:
        message = "CUDA Toolkit is not installed"
        raise RuntimeError(message)
    exe = _compile(home)
    env = os.environ.copy()
    env["PATH"] = str(home / "bin" / "x64") + os.pathsep + env.get("PATH", "")
    result = subprocess.run([str(exe)], check=False, capture_output=True, text=True, env=env)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"dispatch exited {result.returncode}"
        raise RuntimeError(detail)


def _msvc_bin() -> Path:
    """Directory that holds ``cl.exe``. nvcc does not search Visual Studio on its own."""
    found = _from_vswhere() or _from_install_tree()
    if found is None:
        message = "cl.exe is not installed"
        raise RuntimeError(message)
    return found


def _from_vswhere() -> Path | None:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.is_file():
        return None
    found = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-find",
            r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in found.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return Path(lines[-1]).parent


def _from_install_tree() -> Path | None:
    roots = (
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    )
    compilers = [
        path
        for root in roots
        if root.is_dir()
        for path in root.glob(r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
    ]
    if not compilers:
        return None
    return sorted(compilers)[-1].parent


def _nvcc(home: Path) -> Path:
    for name in ("nvcc.exe", "nvcc"):
        candidate = home / "bin" / name
        if candidate.is_file():
            return candidate
    message = f"nvcc is missing under {home}"
    raise RuntimeError(message)


def _lib_dir(home: Path) -> Path:
    for relative in (Path("lib") / "x64", Path("lib64"), Path("lib")):
        candidate = home / relative
        if candidate.is_dir():
            return candidate
    return home / "lib"


def _gencode() -> list[str]:
    return [f"-gencode={item}" for item in _GENCODE]


def _host_flags() -> list[str]:
    if os.name != "nt":
        return []
    return [f"-ccbin={_msvc_bin()}", "-allow-unsupported-compiler"]


def _compile(home: Path) -> Path:
    nvcc = _nvcc(home)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if _EXE.is_file() and _EXE.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return _EXE
    command = [
        str(nvcc),
        "-O2",
        "-std=c++17",
        *_gencode(),
        *_host_flags(),
        str(_SOURCE),
        "-o",
        str(_EXE),
        f"-I{home / 'include'}",
        f"-L{_lib_dir(home)}",
        "-lcublasLt",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "nvcc failed"
        raise RuntimeError(detail)
    return _EXE


def _compile_engine(home: Path) -> Path:
    nvcc = _nvcc(home)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if _DLL.is_file() and _DLL.stat().st_mtime >= _ENGINE.stat().st_mtime:
        return _DLL
    command = [
        str(nvcc),
        "-O2",
        "-std=c++17",
        "-shared",
        *_gencode(),
        *_host_flags(),
        str(_ENGINE),
        "-o",
        str(_DLL),
        f"-I{home / 'include'}",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "nvcc failed"
        raise RuntimeError(detail)
    return _DLL
