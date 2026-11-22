"""Compile the CUDA step and run its self-check."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

from favlo.host import cuda_home

_HERE = Path(__file__).resolve().parent
_ENGINE = _HERE / "engine.cu"
_BUILD = _HERE / "build"
_DLL = _BUILD / ("engine.dll" if os.name == "nt" else "libengine.so")
_GENCODE = (
    "arch=compute_86,code=sm_86",
    "arch=compute_89,code=sm_89",
    "arch=compute_120,code=sm_120",
    "arch=compute_120,code=compute_120",
)


def run_kernels() -> None:
    """Compile, then check the fp16 MMA, the int4 nibble, LayerNorm, and attention."""
    home = cuda_home()
    if home is None:
        message = "CUDA Toolkit is not installed"
        raise RuntimeError(message)
    dll = _compile(home, _ENGINE, _DLL)
    os.add_dll_directory(str(home / "bin" / "x64"))
    lib = ctypes.CDLL(str(dll))
    lib.engine_open.restype = ctypes.c_int
    lib.engine_selfcheck.restype = ctypes.c_int
    lib.engine_close.restype = None
    if lib.engine_open() != 0:
        message = "CUDA device open failed"
        raise RuntimeError(message)
    try:
        code = lib.engine_selfcheck()
    finally:
        lib.engine_close()
    if code != 0:
        message = f"kernel check failed ({code})"
        raise RuntimeError(message)
    print("kernels ok", flush=True)


def _msvc_bin() -> Path:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.is_file():
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
        if lines:
            return Path(lines[-1]).parent
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
    if compilers:
        return sorted(compilers)[-1].parent
    message = "cl.exe is not installed"
    raise RuntimeError(message)


def _compile(home: Path, source: Path, output: Path) -> Path:
    nvcc = home / "bin" / "nvcc.exe"
    if not nvcc.is_file():
        message = f"nvcc is missing under {home}"
        raise RuntimeError(message)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    command = [
        str(nvcc),
        "-shared",
        "-O2",
        "-std=c++17",
        *[f"-gencode={item}" for item in _GENCODE],
        f"-ccbin={_msvc_bin()}",
        "-allow-unsupported-compiler",
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "nvcc failed"
        raise RuntimeError(detail)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return output
