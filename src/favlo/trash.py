"""Move a path into the system Trash instead of unlinking it."""

from __future__ import annotations

from pathlib import Path


def trash_path(path: Path) -> None:
    """Put ``path`` in Trash. A missing path is a no-op."""
    if not path.exists():
        return
    from Foundation import NSURL, NSFileManager

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
