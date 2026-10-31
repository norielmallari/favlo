"""Local folder adapter. Every path stays inside one local folder."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from web_api.store import Rejected

_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
_NOTE_SUFFIX = ".json"


def _occupied(path: Path) -> bool:
    return any(not child.name.startswith(".") for child in path.iterdir())


class LocalFolder:
    """One local folder on this machine."""

    def __init__(self, location: str) -> None:
        root = Path(location).expanduser()
        if not root.exists() or not root.is_dir():
            raise Rejected("The local folder must be an existing directory.")
        self.root = root.resolve()

    def list(self, relative: str = "") -> dict[str, Any]:
        directory = self.resolve(relative, directory=True)
        folders: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        empty = True
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if child.name.startswith("."):
                continue
            empty = False
            path = self._relative(child)
            if child.is_dir():
                folders.append({"path": path, "name": child.name, "kind": "folder"})
                continue
            if child.suffix.lower() not in _IMAGES:
                continue
            images.append(self.stat(path))
        return {"path": self._relative(directory), "folders": folders, "images": images, "empty": empty}

    def read(self, relative: str) -> bytes:
        path = self.resolve(relative, file=True)
        return path.read_bytes()

    def write(
        self,
        relative: str,
        payload: bytes,
        note: dict[str, Any] | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        path = self.resolve(relative, create=True)
        if path.suffix.lower() not in _IMAGES:
            raise Rejected("The gallery stores png, jpg, jpeg, or webp files.")
        if path.exists() and path.is_dir():
            raise Rejected("That path is a folder.")
        parent = path.parent
        if not parent.exists() or not parent.is_dir():
            raise Rejected("The parent folder does not exist.")
        if path.exists():
            if expected_revision is None:
                raise Rejected("That file already exists.")
            if _revision(path) != expected_revision:
                raise Rejected("That file changed.")
        path.write_bytes(payload)
        if note:
            path.with_name(path.name + _NOTE_SUFFIX).write_text(
                json.dumps(note),
                encoding="utf-8",
            )
        return self.stat(self._relative(path))

    def stat(self, relative: str) -> dict[str, Any]:
        path = self.resolve(relative, file=True)
        return {
            "path": self._relative(path),
            "name": path.name,
            "kind": "image",
            "media_type": _media_type(path.suffix),
            "size": path.stat().st_size,
            "revision": _revision(path),
            "note": _note(path),
        }

    def create_folder(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, create=True, folder=True)
        if path == self.root:
            raise Rejected("The local folder is not a folder the agent can create.")
        if path.exists():
            if path.is_dir():
                raise Rejected("That folder already exists.")
            raise Rejected("That path is a file.")
        path.mkdir(parents=False)
        return {"path": self._relative(path), "name": path.name, "kind": "folder"}

    def rename_folder(self, relative: str, name: str) -> dict[str, str]:
        cleaned = name.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
            raise Rejected("A folder name is required.")
        path = self.resolve(relative, directory=True)
        if path == self.root:
            raise Rejected("The local folder cannot be renamed.")
        target = path.with_name(cleaned)
        if target.resolve().parent != path.parent:
            raise Rejected("That name is not allowed.")
        if target.exists():
            raise Rejected("That folder already exists.")
        path.rename(target)
        return {"path": self._relative(target), "name": target.name, "kind": "folder"}

    def move_folder(self, relative: str, destination: str) -> dict[str, str]:
        path = self.resolve(relative, directory=True)
        if path == self.root:
            raise Rejected("The local folder cannot be moved.")
        target = self.resolve(destination, create=True, folder=True)
        if target == self.root:
            raise Rejected("The local folder cannot be replaced.")
        if target.exists():
            raise Rejected("That folder already exists.")
        if path == target or path in target.parents:
            raise Rejected("A folder cannot be moved into itself.")
        if self.root not in target.parents and target != self.root:
            raise Rejected("That path leaves the local folder.")
        target.parent.mkdir(parents=False, exist_ok=True)
        if not target.parent.is_dir():
            raise Rejected("The parent folder does not exist.")
        shutil.move(str(path), str(target))
        return {"path": self._relative(target), "name": target.name, "kind": "folder"}

    def move(self, relative: str, destination: str) -> dict[str, Any]:
        path = self.resolve(relative, file=True)
        target = self.resolve(destination, create=True)
        if target == self.root or (target.exists() and target.is_dir()):
            raise Rejected("That path is a folder.")
        if self.root not in target.parents:
            raise Rejected("That path leaves the local folder.")
        if not target.parent.exists() or not target.parent.is_dir():
            raise Rejected("The parent folder does not exist.")
        if target.exists():
            raise Rejected("That file already exists.")
        note = path.with_name(path.name + _NOTE_SUFFIX)
        shutil.move(str(path), str(target))
        if note.is_file():
            shutil.move(str(note), str(target.with_name(target.name + _NOTE_SUFFIX)))
        return self.stat(self._relative(target))

    def delete_image(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, file=True)
        if path.suffix.lower() not in _IMAGES:
            raise Rejected("The gallery stores png, jpg, jpeg, or webp files.")
        note = path.with_name(path.name + _NOTE_SUFFIX)
        path.unlink()
        if note.is_file():
            note.unlink()
        return {"path": self._relative(path), "name": path.name, "kind": "image"}

    def delete_folder(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, directory=True)
        if path == self.root:
            raise Rejected("The local folder cannot be deleted.")
        if _occupied(path):
            raise Rejected("That folder is not empty.")
        shutil.rmtree(path)
        return {"path": relative.strip("/")}

    def resolve(
        self,
        relative: str,
        *,
        file: bool = False,
        directory: bool = False,
        create: bool = False,
        folder: bool = False,
    ) -> Path:
        cleaned = relative.strip().strip("/")
        if cleaned in {".", ".."} or ".." in Path(cleaned).parts:
            raise Rejected("That path leaves the local folder.")
        if Path(cleaned).is_absolute() or cleaned.startswith("~"):
            raise Rejected("A path is relative to the local folder.")
        if "\x00" in cleaned:
            raise Rejected("That path is not allowed.")
        candidate = (self.root / cleaned).resolve() if cleaned else self.root
        if candidate != self.root and self.root not in candidate.parents:
            raise Rejected("That path leaves the local folder.")
        if create:
            return candidate
        if not candidate.exists():
            raise Rejected("That path does not exist.")
        if file and not candidate.is_file():
            raise Rejected("That path is not a file.")
        if directory and not candidate.is_dir():
            raise Rejected("That path is not a folder.")
        if folder and candidate.exists() and not candidate.is_dir():
            raise Rejected("That path is a file.")
        return candidate

    def _relative(self, path: Path) -> str:
        if path == self.root:
            return ""
        return path.relative_to(self.root).as_posix()


def folder_for(store: Any, workspace_id: str, folder_id: str | None = None) -> LocalFolder:
    return LocalFolder(store.local_location(workspace_id, folder_id))


def _media_type(suffix: str) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mapping.get(suffix.lower(), "application/octet-stream")


def _revision(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.stat().st_mtime_ns).encode())
    digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def _note(path: Path) -> dict[str, Any] | None:
    sidecar = path.with_name(path.name + _NOTE_SUFFIX)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
