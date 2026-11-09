"""Plain file names under a session's local root.

Home is ``~/.favlo/<agent id>/channels/home``. A workspace uses the folder the
person picked, or ``~/.favlo/<agent id>/workspaces/<workspace name>``. The
listener still receives an image id or file bytes. A session path stays here.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any

from web_api.store import Rejected

_IMAGE = {".png"}
_VIDEO = {".mp4"}
_AUDIO = {".mp3", ".wav"}
_STILL = {".png", ".jpg", ".jpeg", ".webp"}
_CLAIMS: set[tuple[str, str]] = set()
_LOCK = threading.Lock()


def home_dir(agent_id: str) -> Path:
    """The home chat folder for this agent. The directory is created."""
    from favlo.logs import favlo_home

    root = favlo_home() / agent_id.strip() / "channels" / "home"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_workspace_dir(agent_id: str, name: str) -> Path:
    """The folder used when a workspace is created without a picked location."""
    from favlo.logs import favlo_home

    return favlo_home() / agent_id.strip() / "workspaces" / name


def local_root(store: Any, session_id: str) -> Path | None:
    """The folder for this chat. A workspace uses its saved folder. Home uses the agent id."""
    workspace = store.workspace_for_session(session_id)
    if workspace is not None:
        if not store.has_local_folder(workspace["id"]):
            return None
        return Path(store.local_location(workspace["id"]))
    agent_id = store.agent_for_session(session_id)
    if not agent_id:
        return None
    return home_dir(agent_id)


def remember_source(root: Path, filename: str, link: str) -> None:
    """Remember the ``favlo://`` link written under this file name."""
    from favlo.media_metadata import METADATA_DIR

    side = root / METADATA_DIR
    side.mkdir(parents=True, exist_ok=True)
    (side / f"{filename}.uri").write_text(link.strip() + "\n", encoding="utf-8")


def source_link(root: Path, filename: str) -> str:
    """The ``favlo://`` link stored for this file name, or empty."""
    from favlo.media_metadata import METADATA_DIR

    path = root / METADATA_DIR / f"{filename}.uri"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def file_name(raw: object) -> str:
    """One plain file name at the folder root."""
    return _plain_name(raw)


def reserve_output(root: Path, tool_name: str, arguments: dict[str, Any]) -> str:
    """Claim one new root file name from ``filename``. The name is removed from arguments."""
    raw = arguments.pop("filename", None)
    name = _plain_name(raw)
    suffix = Path(name).suffix.lower()
    if suffix not in _suffixes(tool_name, arguments):
        raise Rejected("That file name does not match this tool.")
    key = _key(root, name)
    with _LOCK:
        if key in _CLAIMS:
            raise Rejected("That file name is already being made.")
        if (root / name).exists():
            raise Rejected("That file is already in this folder.")
        _CLAIMS.add(key)
    return name


def release_output(root: Path, name: str) -> None:
    """Drop a claim when the task ends or the call does not start."""
    with _LOCK:
        _CLAIMS.discard(_key(root, name))


def rewrite_still(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace an image file name with the file bytes. Other values stay."""
    image = arguments.get("image")
    if not isinstance(image, str) or not image or image.startswith(("image_", "data:")):
        return arguments
    suffix = Path(image.strip()).suffix.lower()
    if suffix not in _STILL:
        return arguments
    name = _plain_name(image)
    path = root / name
    if not path.is_file():
        raise Rejected("That file is not in this folder.")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {**arguments, "image": encoded}


def _plain_name(raw: object) -> str:
    if not isinstance(raw, str):
        raise Rejected("Pass a file name at the folder root.")
    name = raw.strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or name.startswith(".")
    ):
        raise Rejected("Pass a file name at the folder root.")
    return name


def _suffixes(tool_name: str, arguments: dict[str, Any]) -> set[str]:
    local = tool_name.split("__")[-1]
    if local == "generate_image" or local.endswith("_image"):
        return set(_IMAGE)
    if "video" in local:
        return set(_VIDEO)
    fmt = str(arguments.get("response_format") or "mp3").strip().lower()
    if f".{fmt}" not in _AUDIO:
        raise Rejected("This folder stores mp3 or wav.")
    return {f".{fmt}"}


def _key(root: Path, name: str) -> tuple[str, str]:
    return (str(root.resolve()), name)
