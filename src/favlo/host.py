"""Which device this process can open.

Apple Silicon stays on Metal. Windows stays on CUDA when that package is
shipped and a device answers. The choice is made once per process.
"""

from __future__ import annotations

import os
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


def physical_bytes() -> int:
    """Installed memory."""
    if sys.platform == "win32":
        return int(_windows_status().ullTotalPhys)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page * pages


def available_bytes() -> int:
    """Physical bytes Windows still offers this process.

    Darwin reclaimable pages stay in ``memory.reclaimable_bytes``. This probe
    is the Windows count only.
    """
    if sys.platform != "win32":
        message = "available_bytes is the Windows probe"
        raise RuntimeError(message)
    return int(_windows_status().ullAvailPhys)


def page_bytes() -> int:
    """The host page size."""
    if sys.platform == "win32":
        return int(_system_info().dwPageSize)
    return int(os.sysconf("SC_PAGE_SIZE"))


def _windows_status() -> object:
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not ok:
        message = "GlobalMemoryStatusEx failed"
        raise OSError(message)
    return status


def _system_info() -> object:
    import ctypes

    class Info(ctypes.Structure):
        _fields_ = (
            ("wProcessorArchitecture", ctypes.c_ushort),
            ("wReserved", ctypes.c_ushort),
            ("dwPageSize", ctypes.c_ulong),
            ("lpMinimumApplicationAddress", ctypes.c_void_p),
            ("lpMaximumApplicationAddress", ctypes.c_void_p),
            ("dwActiveProcessorMask", ctypes.c_size_t),
            ("dwNumberOfProcessors", ctypes.c_ulong),
            ("dwProcessorType", ctypes.c_ulong),
            ("dwAllocationGranularity", ctypes.c_ulong),
            ("wProcessorLevel", ctypes.c_ushort),
            ("wProcessorRevision", ctypes.c_ushort),
        )

    info = Info()
    ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(info))
    return info
