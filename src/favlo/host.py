"""Which machine this process is running on.

Apple Silicon stays on Metal. Windows stays on CUDA. The CUDA compiler is
the NVIDIA CUDA Toolkit installed on the machine.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# cudaMemGetInfo reports OS free memory and does not guarantee that cudaMalloc
# can take all of it. NVIDIA publishes no byte reserve, so this hold-back is
# the margin kept from that live free count.
VRAM_HOLD_BYTES = 256 << 20


def is_windows() -> bool:
    return sys.platform == "win32"


def runtime_name() -> str:
    """``cuda`` on Windows, ``metal41`` everywhere else."""
    if is_windows():
        return "cuda"
    return "metal41"


def physical_bytes() -> int:
    """Installed memory."""
    if is_windows():
        return int(_windows_status().ullTotalPhys)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page * pages


def available_bytes() -> int:
    """Memory the host can still give this process."""
    if is_windows():
        return int(_windows_status().ullAvailPhys)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page * pages


def page_bytes() -> int:
    if is_windows():
        info = _system_info()
        return int(info.dwPageSize)
    return int(os.sysconf("SC_PAGE_SIZE"))


def vram_bytes() -> int | None:
    """Total VRAM on the first CUDA device. ``None`` when the driver has no device."""
    opened = _cuda_device()
    if opened is None:
        return None
    lib, device = opened
    nbytes = ctypes.c_size_t()
    total = getattr(lib, "cuDeviceTotalMem_v2", None) or getattr(lib, "cuDeviceTotalMem", None)
    if total is None:
        return None
    total.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    total.restype = ctypes.c_int
    if total(ctypes.byref(nbytes), device) != 0:
        return None
    return int(nbytes.value)


def vram_snapshot() -> tuple[int, int] | None:
    """``(free, total)`` from ``cudaMemGetInfo``. Both are read when this is called."""
    lib = _cudart()
    if lib is None or not hasattr(lib, "cudaMemGetInfo"):
        return None
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    lib.cudaMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cudaMemGetInfo.restype = ctypes.c_int
    if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    if free.value < 1 or total.value < 1:
        return None
    return int(free.value), int(total.value)


def _cudart() -> ctypes.WinDLL | None:
    """The toolkit runtime library. ``None`` when that library is not installed."""
    if not is_windows():
        return None
    home = cuda_home()
    if home is None:
        return None
    found = sorted(home.glob("bin/**/cudart64_*.dll"))
    if not found:
        return None
    try:
        return ctypes.WinDLL(str(found[-1]))
    except OSError:
        return None


def usable_bytes(free: int, cap: int | None) -> int:
    """Free bytes the opening may use. ``cap`` lowers that. The hold-back stays."""
    usable = free - VRAM_HOLD_BYTES
    if usable < 1:
        return 0
    if cap is None:
        return usable
    return min(cap, usable)


def device_budget() -> int | None:
    """Bytes the CUDA pipeline may place on the device.

    The count is the live free memory from the driver, minus the hold-back.
    ``device_budget`` in the registry lowers that count. An omitted key does not.
    """
    snap = vram_snapshot()
    if snap is None:
        return None
    free, _total = snap
    from favlo.registry import device_budget_cap

    return usable_bytes(free, device_budget_cap())


def _cuda_device() -> tuple[ctypes.WinDLL, ctypes.c_int] | None:
    """The first CUDA device. ``None`` when this host has no driver device."""
    if not is_windows():
        return None
    try:
        lib = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return None
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    if lib.cuInit(0) != 0:
        return None
    count = ctypes.c_int()
    lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuDeviceGetCount.restype = ctypes.c_int
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
        return None
    device = ctypes.c_int()
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGet.restype = ctypes.c_int
    if lib.cuDeviceGet(ctypes.byref(device), 0) != 0:
        return None
    return lib, device


def cuda_home() -> Path | None:
    """The NVIDIA CUDA Toolkit directory, usually ``CUDA\\v13.5``."""
    override = os.environ.get("CUDA_PATH")
    if override and Path(override).is_dir():
        return Path(override)
    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not root.is_dir():
        return None
    versions = sorted((path for path in root.glob("v*") if path.is_dir()), reverse=True)
    if not versions:
        return None
    return versions[0]


def _windows_status() -> ctypes.Structure:
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


def _system_info() -> ctypes.Structure:
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
