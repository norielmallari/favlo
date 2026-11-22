"""Which device this process can open.

Apple Silicon stays on Metal. Windows stays on CUDA when that package is
shipped and a device answers. The choice is made once per process.
"""

from __future__ import annotations

import sys
from collections.abc import Collection

_METAL: bool | None = None


def select_device(available: Collection[str]) -> str:
    """The device package this host runs. Refuse when that package is missing."""
    shipped = set(available)
    if sys.platform == "darwin":
        if "metal41" in shipped and _metal_ready():
            return "metal41"
    elif sys.platform == "win32" and "cuda" in shipped and _cuda_ready():
        return "cuda"
    message = "this runtime cannot run here"
    raise RuntimeError(message)


def _metal_ready() -> bool:
    global _METAL
    if _METAL is None:
        from favlo.setup import metal_ready

        _METAL = metal_ready()
    return _METAL


def _cuda_ready() -> bool:
    """True when the NVIDIA driver reports a device. The toolkit is not required."""
    import ctypes

    try:
        lib = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return False
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    if lib.cuInit(0) != 0:
        return False
    count = ctypes.c_int()
    lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuDeviceGetCount.restype = ctypes.c_int
    return lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
