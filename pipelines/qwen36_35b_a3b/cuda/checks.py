"""Compile the CUDA step and run the dispatch check."""

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
_DLL = _HERE / "engine.dll" if os.name == "nt" else _BUILD / "libengine.so"
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
    return _compile_shared(home, _ENGINE, _DLL)


def run_kernels() -> None:
    """Compile the dispatch when the source is newer, then run it on this device."""
    home = cuda_home()
    if home is None:
        message = "CUDA Toolkit is not installed"
        raise RuntimeError(message)
    exe = _compile_shared(home, _SOURCE, _EXE, shared=False)
    env = os.environ.copy()
    env["PATH"] = str(home / "bin" / "x64") + os.pathsep + env.get("PATH", "")
    result = subprocess.run([str(exe)], check=False, capture_output=True, text=True, env=env)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"dispatch exited {result.returncode}"
        raise RuntimeError(detail)


def _msvc_bin() -> Path:
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


def _gencode() -> list[str]:
    return [f"-gencode={item}" for item in _GENCODE]


def _host_flags() -> list[str]:
    if os.name != "nt":
        return []
    return [f"-ccbin={_msvc_bin()}", "-allow-unsupported-compiler"]


def _compile_shared(home: Path, source: Path, output: Path, *, shared: bool = True) -> Path:
    nvcc = _nvcc(home)
    output.parent.mkdir(parents=True, exist_ok=True)
    newest = source.stat().st_mtime
    sibling = source.parent / "tile.cu"
    if source.name == "engine.cu" and sibling.is_file():
        newest = max(newest, sibling.stat().st_mtime)
    if output.is_file() and output.stat().st_mtime >= newest:
        return output
    command = [
        str(nvcc),
        "-O2",
        "-std=c++17",
        *_gencode(),
        *_host_flags(),
        str(source),
        "-o",
        str(output),
    ]
    if shared:
        command.insert(1, "-shared")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "nvcc failed"
        raise RuntimeError(detail)
    return output
