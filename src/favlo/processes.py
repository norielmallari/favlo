"""Pipeline-worker start time, read in Python on each host.

Windows uses the process snapshot and the PEB command line. Linux reads
``/proc``. macOS has no process table in the standard library, so this module
parses ``ps`` elapsed time with the same clock.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

_NEEDLE = "favlo.pipeline_worker"


def process_rows() -> list[tuple[int, int, str]]:
    """Pid, parent pid, and command line for each process this host can see."""
    if sys.platform == "win32":
        return _windows_rows()
    if sys.platform == "linux":
        return _linux_rows()
    return _darwin_rows()


def worker_started() -> float | None:
    """Unix start time of the newest pipeline worker, or None when none is running."""
    if sys.platform == "win32":
        return _windows_started()
    if sys.platform == "linux":
        return _linux_started()
    return _darwin_started()


def etime_seconds(text: str) -> int | None:
    """Parse ``ps`` elapsed time ``[[dd-]hh:]mm:ss``."""
    days = 0
    clock = text.strip()
    if "-" in clock:
        day, clock = clock.split("-", 1)
        try:
            days = int(day)
        except ValueError:
            return None
    try:
        pieces = [int(part) for part in clock.split(":")]
    except ValueError:
        return None
    if len(pieces) == 2:
        hours, minutes, seconds = 0, pieces[0], pieces[1]
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _bind_kernel() -> ctypes.WinDLL:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.ReadProcessMemory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return kernel


def _windows_rows() -> list[tuple[int, int, str]]:
    import ctypes

    kernel = _bind_kernel()
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return []
    entry = _process_entry()
    entry.dwSize = ctypes.sizeof(entry)
    rows: list[tuple[int, int, str]] = []
    try:
        if not kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            return []
        while True:
            pid = int(entry.th32ProcessID)
            parent = int(entry.th32ParentProcessID)
            command = _windows_command(kernel, pid) or str(entry.szExeFile)
            rows.append((pid, parent, command))
            if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel.CloseHandle(snapshot)
    return rows


def _windows_started() -> float | None:
    import ctypes

    kernel = _bind_kernel()
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    entry = _process_entry()
    entry.dwSize = ctypes.sizeof(entry)
    newest: float | None = None
    try:
        if not kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            started = _windows_match(kernel, int(entry.th32ProcessID))
            if started is not None and (newest is None or started > newest):
                newest = started
            if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel.CloseHandle(snapshot)
    return newest


def _process_entry() -> ctypes.Structure:
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = (
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        )

    return Entry()


def _windows_match(kernel: ctypes.WinDLL, pid: int) -> float | None:
    if pid <= 0:
        return None
    command = _windows_command(kernel, pid)
    if _NEEDLE not in command:
        return None
    return _windows_created(kernel, pid)


def _windows_command(kernel: ctypes.WinDLL, pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    access = 0x1000 | 0x0010
    handle = kernel.OpenProcess(access, False, pid)
    if not handle:
        return ""
    try:
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtQueryInformationProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(wintypes.DWORD),
        ]

        class Info(ctypes.Structure):
            _fields_ = (
                ("Reserved1", ctypes.c_void_p),
                ("PebBaseAddress", ctypes.c_void_p),
                ("Reserved2", ctypes.c_void_p * 2),
                ("UniqueProcessId", ctypes.c_void_p),
                ("Reserved3", ctypes.c_void_p),
            )

        class Text(ctypes.Structure):
            _fields_ = (
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("_pad", wintypes.DWORD),
                ("Buffer", ctypes.c_void_p),
            )

        info = Info()
        length = wintypes.DWORD(0)
        status = ntdll.NtQueryInformationProcess(
            handle, 0, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(length)
        )
        if status != 0 or not info.PebBaseAddress:
            return ""
        params = ctypes.c_void_p()
        read = ctypes.c_size_t(0)
        if not kernel.ReadProcessMemory(
            handle, info.PebBaseAddress + 0x20, ctypes.byref(params), 8, ctypes.byref(read)
        ):
            return ""
        text = Text()
        if not params.value or not kernel.ReadProcessMemory(
            handle, params.value + 0x70, ctypes.byref(text), ctypes.sizeof(text), ctypes.byref(read)
        ):
            return ""
        if text.Length < 2 or not text.Buffer:
            return ""
        raw = ctypes.create_unicode_buffer(text.Length // 2)
        if not kernel.ReadProcessMemory(
            handle, text.Buffer, raw, text.Length, ctypes.byref(read)
        ):
            return ""
        return raw.value
    finally:
        kernel.CloseHandle(handle)


def _windows_created(kernel: ctypes.WinDLL, pid: int) -> float | None:
    import ctypes
    from ctypes import wintypes

    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        created = wintypes.FILETIME()
        ignored = wintypes.FILETIME()
        if not kernel.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(ignored),
            ctypes.byref(ignored),
            ctypes.byref(ignored),
        ):
            return None
        ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return ticks / 10_000_000 - 11644473600
    finally:
        kernel.CloseHandle(handle)


def _linux_rows() -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    root = Path("/proc")
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        end = stat.rfind(")")
        fields = stat[end + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            parent = int(fields[1])
        except ValueError:
            continue
        command = raw.decode("utf-8", errors="replace").strip()
        if not command:
            command = stat[stat.find("(") + 1 : end]
        rows.append((int(entry.name), parent, command))
    return rows


def _linux_started() -> float | None:
    boot = _linux_boot()
    if boot is None:
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    newest: float | None = None
    root = Path("/proc")
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        if _NEEDLE not in command:
            continue
        end = stat.rfind(")")
        fields = stat[end + 2 :].split()
        if len(fields) < 20:
            continue
        try:
            started = boot + int(fields[19]) / ticks
        except ValueError:
            continue
        if newest is None or started > newest:
            newest = started
    return newest


def _linux_boot() -> float | None:
    try:
        text = Path("/proc/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("btime "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _darwin_rows() -> list[tuple[int, int, str]]:
    import subprocess

    try:
        completed = subprocess.run(
            ["ps", "-axww", "-o", "pid=,ppid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    rows: list[tuple[int, int, str]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            parent = int(parts[1])
        except ValueError:
            continue
        rows.append((pid, parent, parts[2]))
    return rows


def _darwin_started() -> float | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["ps", "-axww", "-o", "etime=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    newest: float | None = None
    now = time.time()
    for line in completed.stdout.splitlines():
        if _NEEDLE not in line:
            continue
        elapsed = etime_seconds(line.split(None, 1)[0])
        if elapsed is None:
            continue
        started = now - elapsed
        if newest is None or started > newest:
            newest = started
    return newest
