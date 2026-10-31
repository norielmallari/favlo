"""Open a folder dialog on this Mac. The browser does not read the disk."""

from __future__ import annotations

import subprocess
import sys
import threading

from web_api.store import Rejected

_LOCK = threading.Lock()
_SCRIPT = (
    'try\n'
    '  POSIX path of (choose folder with prompt "Choose a gallery folder")\n'
    'on error number -128\n'
    '  return ""\n'
    'end try'
)


def choose_directory() -> str | None:
    if sys.platform != "darwin":
        raise Rejected("Folder picking is available on this Mac.")
    with _LOCK:
        completed = subprocess.run(
            ["osascript", "-e", _SCRIPT],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise Rejected("The folder picker did not open.")
    path = completed.stdout.strip()
    if path == "":
        return None
    if path.endswith("/") and path != "/":
        return path[:-1]
    return path
