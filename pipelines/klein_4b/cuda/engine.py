"""Load the CUDA step library and compile it when the source is newer."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DLL = _HERE / "engine.dll"
_CU = _HERE / "engine.cu"
_NVCC = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\nvcc.exe")

_lib: ctypes.CDLL | None = None


class EngineError(RuntimeError):
    pass


def _cuda_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    root = _nvcc().parent
    for folder in (root, root / "x64"):
        if folder.is_dir():
            os.add_dll_directory(str(folder))


def library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        ensure()
        _cuda_dll_dirs()
        _lib = ctypes.CDLL(str(_DLL))
        _bind(_lib)
    return _lib


def ensure() -> None:
    if _DLL.is_file() and _DLL.stat().st_mtime >= _CU.stat().st_mtime:
        return
    compile_source()


def compile_source() -> None:
    nvcc = _nvcc()
    command = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        "-arch=sm_86",
        "--shared",
        "-Xcompiler",
        "/MD",
        "-lcublas",
        "-o",
        str(_DLL),
        str(_CU),
    ]
    env = os.environ.copy()
    if shutil.which("cl") is None:
        _run_with_vcvars(command, env)
        return
    subprocess.run(command, check=True, cwd=_HERE)


def call(name: str, *args: object) -> None:
    function = getattr(library(), name)
    status = int(function(*args))
    if status != 0:
        message = f"{name} failed with status {status}"
        raise EngineError(message)


def pointer(value: int | None) -> ctypes.c_void_p | None:
    if value is None:
        return None
    return ctypes.c_void_p(value)


def alloc(nbytes: int) -> int:
    out = ctypes.c_void_p()
    status = int(library().klein_alloc(ctypes.c_uint64(nbytes), ctypes.byref(out)))
    if status != 0 or not out.value:
        message = f"cudaMalloc failed for {nbytes} bytes"
        raise EngineError(message)
    return int(out.value)


def free(value: int) -> None:
    call("klein_free", pointer(value))


def sync() -> None:
    call("klein_sync")


def _nvcc() -> Path:
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    if _NVCC.is_file():
        return _NVCC
    message = "nvcc is not installed"
    raise FileNotFoundError(message)


def _run_with_vcvars(command: list[str], env: dict[str, str]) -> None:
    script = _vcvars()
    quoted = " ".join(_quote(part) for part in command)
    batch = f'call "{script}" >nul && {quoted}'
    subprocess.run(batch, check=True, cwd=_HERE, env=env, shell=True)


def _vcvars() -> Path:
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(root.glob("**/VC/Auxiliary/Build/vcvars64.bat"))
    if not found:
        message = "cl.exe is not on PATH and vcvars64.bat was not found"
        raise FileNotFoundError(message)
    return found[-1]


def _quote(text: str) -> str:
    if " " in text or "(" in text:
        return f'"{text}"'
    return text


def _bind(lib: ctypes.CDLL) -> None:
    void = ctypes.c_void_p
    i32 = ctypes.c_int
    u64 = ctypes.c_uint64
    f32 = ctypes.c_float
    pairs = {
        "klein_alloc": [u64, ctypes.POINTER(void)],
        "klein_free": [void],
        "klein_copy_h2d": [void, void, u64],
        "klein_copy_d2h": [void, void, u64],
        "klein_copy_d2d": [void, void, u64],
        "klein_zero": [void, u64],
        "klein_sync": [],
        "klein_gemm": [void, void, void, void, void, i32, i32, i32],
        "klein_rmsnorm": [void, void, void, i32, i32, f32],
        "klein_layernorm": [void, void, i32, i32, f32],
        "klein_silu": [void, void, i32],
        "klein_swiglu": [void, void, i32, i32],
        "klein_silu_mul": [void, void, void, i32],
        "klein_modulate": [void, void, void, void, i32, i32],
        "klein_gate": [void, void, void, void, i32, i32],
        "klein_rope_text": [void, void, void, void, i32, i32, i32],
        "klein_rope_dit": [void, void, void, void, i32, i32, i32],
        "klein_attend": [void, void, void, void, i32, i32, i32, i32, i32, i32, f32],
        "klein_slice_cols": [void, void, i32, i32, i32, i32],
        "klein_pad_k": [void, void, i32, i32, i32],
        "klein_embed": [void, void, void, void, i32, i32],
        "klein_euler": [void, void, void, i32, f32],
        "klein_add": [void, void, void, i32],
        "klein_add_f32": [void, void, void, i32],
        "klein_cast_f16_f32": [void, void, i32],
        "klein_cast_f32_f16": [void, void, i32],
        "klein_groupnorm": [void, void, void, void, void, i32, i32, i32, f32, i32, i32],
        "klein_conv": [void, void, void, void, void, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32],
        "klein_concat_cols": [void, void, void, i32, i32, i32],
        "klein_transpose": [void, void, i32, i32, i32, i32],
        "klein_im2col": [void, void, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32],
        "klein_col2im": [void, void, i32, i32, i32, i32],
        "klein_pad_edge": [void, void, i32, i32, i32, i32, i32, i32],
        "klein_nearest2": [void, void, i32, i32, i32, i32],
        "klein_prefix_channels": [void, void, i32, i32, i32],
        "klein_pack_latent": [void, void, void, void, i32, i32, f32],
        "klein_unpack_latent": [void, void, void, void, i32, i32, f32],
    }
    for name, args in pairs.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = ctypes.c_int
