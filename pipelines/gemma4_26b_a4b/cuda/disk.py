"""One-shot record reads on Windows.

The bytes are not used again from the file cache. ``FILE_FLAG_NO_BUFFERING``
moves a sector-aligned span straight into the caller buffer. Several reads stay
outstanding so the drive queue stays full.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

_SECTOR = 4096
_CHUNK = 8 << 20
_DEPTH = 4
_PENDING = 997
_GENERIC_READ = 0x80000000
_SHARE_READ = 1
_OPEN_EXISTING = 3
_NO_BUFFERING = 0x20000000
_OVERLAPPED = 0x40000000
_COMMIT = 0x1000
_RESERVE = 0x2000
_READWRITE = 0x04
_RELEASE = 0x8000

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None


class _Overlapped(ctypes.Structure):
    _fields_ = (
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    )


def alloc_pages(nbytes: int) -> int:
    """Page-aligned address of ``nbytes``. The caller frees it with ``free_pages``."""
    if _kernel32 is None or nbytes < 1:
        message = "page allocation is a Windows read buffer"
        raise RuntimeError(message)
    pointer = _kernel32.VirtualAlloc(None, ctypes.c_size_t(nbytes), _COMMIT | _RESERVE, _READWRITE)
    if not pointer:
        message = f"page allocation of {nbytes} bytes failed"
        raise RuntimeError(message)
    return int(pointer)


def free_pages(pointer: int) -> None:
    if _kernel32 is not None and pointer:
        _kernel32.VirtualFree(ctypes.c_void_p(pointer), 0, _RELEASE)


def read_file(path: Path, destination: int, nbytes: int) -> None:
    """Read ``nbytes`` from the start of ``path`` into ``destination``."""
    read_span(path, 0, destination, nbytes)


def read_span(path: Path, file_offset: int, destination: int, nbytes: int) -> None:
    """Read ``nbytes`` at ``file_offset`` into ``destination``."""
    if nbytes < 1:
        return
    aligned = nbytes % _SECTOR == 0 and file_offset % _SECTOR == 0 and destination % _SECTOR == 0
    if _kernel32 is None or not aligned or not _read_direct(path, file_offset, destination, nbytes):
        _read_cached(path, file_offset, destination, nbytes)


def _read_direct(path: Path, file_offset: int, destination: int, nbytes: int) -> bool:
    assert _kernel32 is not None
    handle = _kernel32.CreateFileW(
        str(path),
        _GENERIC_READ,
        _SHARE_READ,
        None,
        _OPEN_EXISTING,
        _NO_BUFFERING | _OVERLAPPED,
        None,
    )
    if int(handle) == int(wintypes.HANDLE(-1).value):
        return False
    events: list[int] = []
    busy = [False] * _DEPTH
    try:
        slots = []
        for _ in range(_DEPTH):
            event = _kernel32.CreateEventW(None, False, False, None)
            if not event:
                return False
            events.append(int(event))
            slots.append(_Overlapped(hEvent=event))
        chunks = (nbytes + _CHUNK - 1) // _CHUNK
        next_index = 0
        done = 0

        def issue(slot: int, index: int) -> bool:
            offset = file_offset + index * _CHUNK
            size = min(_CHUNK, nbytes - index * _CHUNK)
            item = slots[slot]
            item.Offset = offset & 0xFFFFFFFF
            item.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
            ok = _kernel32.ReadFile(
                handle,
                ctypes.c_void_p(destination + index * _CHUNK),
                size,
                None,
                ctypes.byref(item),
            )
            if ok or ctypes.get_last_error() == _PENDING:
                busy[slot] = True
                return True
            return False

        def finish(slot: int) -> bool:
            got = wintypes.DWORD()
            if not _kernel32.GetOverlappedResult(
                handle, ctypes.byref(slots[slot]), ctypes.byref(got), False
            ):
                return False
            local = ((slots[slot].OffsetHigh << 32) | slots[slot].Offset) - file_offset
            busy[slot] = False
            return int(got.value) == min(_CHUNK, nbytes - local)

        for slot in range(_DEPTH):
            if next_index >= chunks:
                break
            if not issue(slot, next_index):
                return False
            next_index += 1
        while done < chunks:
            active = [events[slot] for slot in range(_DEPTH) if busy[slot]]
            count = len(active)
            wait = _kernel32.WaitForMultipleObjects(
                count, (wintypes.HANDLE * count)(*active), False, 120000
            )
            if wait < 0 or wait >= count:
                return False
            slot = [index for index in range(_DEPTH) if busy[index]][wait]
            if not finish(slot):
                return False
            done += 1
            if next_index < chunks and not issue(slot, next_index):
                return False
            if next_index < chunks:
                next_index += 1
        return done == chunks
    finally:
        if any(busy):
            _kernel32.CancelIoEx(handle, None)
            for slot, pending in enumerate(busy):
                if pending:
                    got = wintypes.DWORD()
                    _kernel32.GetOverlappedResult(handle, ctypes.byref(slots[slot]), ctypes.byref(got), True)
        for event in events:
            _kernel32.CloseHandle(event)
        _kernel32.CloseHandle(handle)


def _read_cached(path: Path, file_offset: int, destination: int, nbytes: int) -> None:
    with path.open("rb") as handle:
        handle.seek(file_offset)
        done = 0
        window = 8 << 20
        while done < nbytes:
            blob = handle.read(min(window, nbytes - done))
            if not blob:
                message = f"{path.name} ended at {done} of {nbytes}"
                raise RuntimeError(message)
            ctypes.memmove(destination + done, blob, len(blob))
            done += len(blob)


def _bind() -> None:
    if _kernel32 is None:
        return
    _kernel32.VirtualAlloc.restype = ctypes.c_void_p
    _kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    _kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    _kernel32.VirtualFree.restype = wintypes.BOOL
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.GetOverlappedResult.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL,
    ]
    _kernel32.GetOverlappedResult.restype = wintypes.BOOL
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.WaitForMultipleObjects.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    _kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    _kernel32.CancelIoEx.restype = wintypes.BOOL
    _kernel32.WaitForMultipleObjects.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


_bind()
