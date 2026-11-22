"""Move a path into the system Trash instead of unlinking it."""

from __future__ import annotations

import os
from pathlib import Path


def trash_path(path: Path) -> None:
    """Put ``path`` in Trash. A missing path is a no-op."""
    if not path.exists():
        return
    if os.name == "nt":
        _trash_windows(path)
        return
    try:
        from Foundation import NSURL, NSFileManager
    except ImportError as exc:
        message = "trash is unavailable"
        raise OSError(message) from exc

    url = NSURL.fileURLWithPath_(str(path.resolve()))
    ok, _result, error = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
        url, None, None
    )
    if ok:
        return
    message = "trash failed"
    if error is not None:
        detail = str(error)
        if detail:
            message = f"trash failed: {detail}"
    raise OSError(message)


def _trash_windows(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    class _Op(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    source = ctypes.create_unicode_buffer(str(path.resolve()) + "\0")
    op = _Op()
    op.wFunc = 3
    op.pFrom = ctypes.cast(source, ctypes.c_wchar_p)
    op.fFlags = 0x40 | 0x10 | 0x4 | 0x400
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        message = f"trash failed: {result}"
        raise OSError(message)
