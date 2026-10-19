"""Memory and disk for the idle check. The listener does not import a pipeline.

16 GB is the memory this runtime requires. Qwen3.8-27B is offered when the
machine has that much physical memory, the checkout is not already here, and
the volume can hold the published files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from favlo.logs import favlo_home
from favlo.registry import load_registry, models_dir

MIN_RAM_BYTES = 16 << 30
QWEN38_27B = "qwen38-27b"
QWEN38_27B_REPO = "Qwen/Qwen3.8-27B"
QWEN38_27B_CHECKOUT = "Qwen3.8-27B-bf16"
_PID_NAME = "qwen38-27b.pid"
_SUGGESTIONS_NAME = "suggestions.json"

Check = Callable[[], str | None]
Spawn = Callable[[], int]


def physical_bytes() -> int:
    """Installed memory. Reclaimable pages shrink while a model is loaded."""
    page = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page * pages


def free_bytes(path: Path) -> int:
    """Free bytes on the volume that holds ``path``. The directory is created."""
    path.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(path).free)


def checkout_dir(root: Path | None = None) -> Path:
    return models_dir(root) / QWEN38_27B_CHECKOUT


def suggestions_path(home: Path | None = None) -> Path:
    """``~/.favlo/suggestions.json``. A missing file means reminders are on."""
    root = favlo_home() if home is None else home
    return root / _SUGGESTIONS_NAME


def suggestions_enabled(home: Path | None = None) -> bool:
    """True when idle download suggestions may be spoken."""
    path = suggestions_path(home)
    if not path.is_file():
        return True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        message = f"suggestions file cannot be read: {path}"
        raise ValueError(message) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("enabled"), bool):
        message = f"suggestions file is not an enabled flag: {path}"
        raise ValueError(message)
    return bool(raw["enabled"])


def set_suggestions(enabled: bool, home: Path | None = None) -> str:
    """Remember whether idle download suggestions are spoken."""
    if not isinstance(enabled, bool):
        message = "enabled must be true or false"
        raise ValueError(message)
    path = suggestions_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"enabled": enabled}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    if enabled:
        return "download suggestions enabled"
    return "download suggestions disabled"


def pid_path(home: Path | None = None) -> Path:
    root = favlo_home() if home is None else home
    return root / "downloads" / _PID_NAME


def qwen38_27b_registered(root: Path | None = None) -> bool:
    """True when a registry row is this pipeline, its alias, or its packed folder."""
    data = load_registry(root)
    for folder, entry in data["models"].items():
        if folder == "Qwen3.8-27B-fp4":
            return True
        if not isinstance(entry, dict):
            continue
        if entry.get("pipeline") == QWEN38_27B or entry.get("alias") == QWEN38_27B:
            return True
    return False


def download_state(home: Path | None = None, root: Path | None = None) -> str:
    """``complete``, ``running``, or ``idle``."""
    marker = checkout_dir(root) / "fetch_complete.json"
    if marker.is_file():
        return "complete"
    path = pid_path(home)
    if not path.is_file():
        return "idle"
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError):
        return "idle"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "idle"
    except PermissionError:
        return "running"
    except OSError:
        return "idle"
    return "running"


def situation(
    root: Path | None = None,
    home: Path | None = None,
    *,
    ram: int | None = None,
    check: Check | None = None,
) -> str:
    """Why Qwen3.8-27B is or is not offered. ``ready`` means the download may start."""
    if qwen38_27b_registered(root):
        return "registered"
    state = download_state(home, root)
    if state == "complete":
        return "downloaded"
    if state == "running":
        return "downloading"
    memory = physical_bytes() if ram is None else ram
    if memory < MIN_RAM_BYTES:
        return "ram"
    problem = _disk_check(root) if check is None else check()
    if problem:
        return "disk"
    return "ready"


def machine_report(
    root: Path | None = None,
    home: Path | None = None,
    *,
    ram: int | None = None,
    free: int | None = None,
    check: Check | None = None,
) -> str:
    """Measurements the idle turn reads. Status is the offer decision."""
    memory = physical_bytes() if ram is None else ram
    dest = checkout_dir(root)
    disk = free_bytes(dest) if free is None else free
    status = situation(root, home, ram=memory, check=check)
    return f"ram_bytes {memory}\nfree_bytes {disk}\nstatus {status}"


def offer_qwen38_27b(
    root: Path | None = None,
    home: Path | None = None,
    *,
    ram: int | None = None,
    free: int | None = None,
    check: Check | None = None,
    spawn: Spawn | None = None,
) -> str:
    """Start the checkout when the machine can hold it. Otherwise report why not."""
    report = machine_report(root, home, ram=ram, free=free, check=check)
    status = report.rsplit("status ", 1)[-1]
    if status != "ready":
        return f"{report}\nstarted no"
    if spawn is None:
        _spawn(root, home)
    else:
        spawn()
    return f"{report}\nstarted yes"


def _disk_check(root: Path | None) -> str | None:
    command = [sys.executable, "-m", "favlo", "--qwen38-27b", "download", "--check"]
    env = os.environ.copy()
    if root is not None:
        env["FAVLO_ROOT"] = str(root)
    env["FAVLO_SKIP_PLATE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode == 0:
        return None
    text = (completed.stderr or completed.stdout).strip()
    return text or "the checkout volume is short"


def _spawn(root: Path | None, home: Path | None) -> int:
    base = favlo_home() if home is None else home
    base.joinpath("logs").mkdir(parents=True, exist_ok=True)
    base.joinpath("downloads").mkdir(parents=True, exist_ok=True)
    log_path = base / "logs" / "qwen38-27b-download.log"
    command = [
        sys.executable,
        "-m",
        "favlo",
        "--qwen38-27b",
        "download",
        "--console",
        "plain",
    ]
    env = os.environ.copy()
    if root is not None:
        env["FAVLO_ROOT"] = str(root)
    env["FAVLO_SKIP_PLATE"] = "1"
    handle = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    pid_path(base).write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc.pid
