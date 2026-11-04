"""Local folder adapter. Every path stays inside one local folder."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from web_api.gallery_metadata import (
    KIND_SUFFIXES,
    METADATA_DIR,
    kind_for,
    move_sidecar,
    relocate_metadata_prefix,
    remove_metadata_prefix,
    remove_sidecar,
    sidecar_path,
)
from web_api.store import Rejected

_IMAGES = KIND_SUFFIXES["image"]
_MEDIA = {suffix for group in KIND_SUFFIXES.values() for suffix in group}
_NOTE_SUFFIX = ".json"


def _occupied(path: Path) -> bool:
    return any(not child.name.startswith(".") for child in path.iterdir())


def trash_path(path: Path) -> Path:
    """Move one file into the macOS Trash. Put Back restores its original path."""
    from Foundation import NSURL, NSFileManager

    url = NSURL.fileURLWithPath_(str(path))
    ok, landed, _error = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
        url, None, None
    )
    if not ok or landed is None:
        raise Rejected("The file could not be moved to the Trash.")
    restored = landed.path()
    if not isinstance(restored, str) or not restored:
        raise Rejected("The file could not be moved to the Trash.")
    return Path(restored)


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

    def find_images(self, names: list[str]) -> list[str]:
        """Relative paths for named png and jpg files, in the order given.

        A path is used when that file exists. A bare filename is the first
        match in sorted path order. A name that is not in this folder is skipped.
        """
        return [path for _name, path in self.locate(names)]

    def locate(self, names: list[str]) -> list[tuple[str, str]]:
        """Each found name and its relative path, in the order given."""
        index: dict[str, str] | None = None
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for name in names:
            cleaned = _image_name(name)
            if cleaned is None:
                continue
            if "/" in cleaned:
                relative = self._existing_image(cleaned)
            else:
                if index is None:
                    index = self._image_index()
                relative = index.get(cleaned.lower())
            if relative is None or relative in seen:
                continue
            seen.add(relative)
            found.append((name, relative))
        return found

    def _existing_image(self, relative: str) -> str | None:
        try:
            path = self.resolve(relative, file=True)
        except Rejected:
            return None
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return None
        return self._relative(path)

    def _image_index(self) -> dict[str, str]:
        """Lowercase filename to the first relative path in sorted order."""
        grouped: dict[str, list[str]] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved != self.root and self.root not in resolved.parents:
                continue
            relative = resolved.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            grouped.setdefault(relative.name.lower(), []).append(relative.as_posix())
        return {name: sorted(paths, key=str.lower)[0] for name, paths in grouped.items()}

    def write(
        self,
        relative: str,
        payload: bytes,
        note: dict[str, Any] | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        path = self.resolve(relative, create=True)
        if path.suffix.lower() not in _MEDIA:
            raise Rejected("The gallery stores png, jpg, jpeg, webp, mp3, wav, or mp4 files.")
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
        media_kind = kind_for(path) or "image"
        rel = self._relative(path)
        side = sidecar_path(self.root, rel)
        return {
            "path": rel,
            "name": path.name,
            "kind": media_kind,
            "media_type": _media_type(path.suffix),
            "size": path.stat().st_size,
            "revision": _revision(path),
            "note": _note(path),
            "has_sidecar": side.is_file(),
        }

    def walk_media(
        self,
        relative: str = "",
        *,
        name_contains: str = "",
        kinds: list[str] | None = None,
        depth_limit: int = 8,
        record_limit: int = 80,
        visit_limit: int = 5000,
    ) -> dict[str, Any]:
        """Capped recursive media list. Dot dirs and adjacent ``*.json`` notes are skipped."""
        allowed = set(kinds or ["image", "audio", "video"])
        suffixes = {
            suffix
            for kind, group in KIND_SUFFIXES.items()
            if kind in allowed
            for suffix in group
        }
        start = self.resolve(relative, directory=True)
        needle = name_contains.strip().lower()
        records: list[dict[str, Any]] = []
        visited = 0
        truncated = False
        depth_limit_hit = False

        def walk(directory: Path, depth: int) -> None:
            nonlocal visited, truncated, depth_limit_hit
            if truncated:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
            except OSError:
                return
            for child in children:
                if truncated:
                    return
                if child.name.startswith("."):
                    continue
                visited += 1
                if visited > visit_limit:
                    truncated = True
                    return
                try:
                    resolved = child.resolve()
                except OSError:
                    continue
                if resolved != self.root and self.root not in resolved.parents:
                    continue
                if child.is_dir():
                    if depth >= depth_limit:
                        depth_limit_hit = True
                        continue
                    walk(child, depth + 1)
                    continue
                if not child.is_file():
                    continue
                if child.name.endswith(".json"):
                    continue
                if child.suffix.lower() not in suffixes:
                    continue
                if needle and needle not in child.name.lower():
                    continue
                if len(records) >= record_limit:
                    truncated = True
                    return
                path = self._relative(child)
                item = self.stat(path)
                item["folder"] = str(Path(path).parent.as_posix()).lstrip(".") or ""
                if item["folder"] == "":
                    item["folder"] = ""
                records.append(item)

        walk(start, 0)
        return {
            "root": self._relative(start),
            "media": records,
            "images": records,
            "truncated": truncated,
            "depth_limit_hit": depth_limit_hit,
        }

    def walk_images(self, relative: str = "", **kwargs: Any) -> dict[str, Any]:
        """Image-only walk kept for callers that have not moved to ``walk_media``."""
        return self.walk_media(relative, kinds=["image"], **kwargs)

    def create_folder(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, create=True, folder=True)
        if path == self.root:
            raise Rejected("The local folder is not a folder the agent can create.")
        if path.exists():
            if path.is_dir():
                raise Rejected("That folder already exists.")
            raise Rejected("That path is a file.")
        path.mkdir(parents=False)
        (path / METADATA_DIR).mkdir(exist_ok=True)
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
        old_rel = self._relative(path)
        path.rename(target)
        new_rel = self._relative(target)
        relocate_metadata_prefix(self.root, old_rel, new_rel)
        return {"path": new_rel, "name": target.name, "kind": "folder"}

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
        old_rel = self._relative(path)
        shutil.move(str(path), str(target))
        new_rel = self._relative(target)
        relocate_metadata_prefix(self.root, old_rel, new_rel)
        return {"path": new_rel, "name": target.name, "kind": "folder"}

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
        old_rel = self._relative(path)
        shutil.move(str(path), str(target))
        if note.is_file():
            shutil.move(str(note), str(target.with_name(target.name + _NOTE_SUFFIX)))
        new_rel = self._relative(target)
        move_sidecar(self.root, old_rel, new_rel)
        return self.stat(new_rel)

    def delete_image(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, file=True)
        if path.suffix.lower() not in _IMAGES:
            raise Rejected("The gallery stores png, jpg, jpeg, or webp files.")
        note = path.with_name(path.name + _NOTE_SUFFIX)
        rel = self._relative(path)
        trash_path(path)
        if note.is_file():
            trash_path(note)
        remove_sidecar(self.root, rel)
        return {"path": rel, "name": path.name, "kind": "image"}

    def delete_folder(self, relative: str) -> dict[str, str]:
        path = self.resolve(relative, directory=True)
        if path == self.root:
            raise Rejected("The local folder cannot be deleted.")
        if _occupied(path):
            raise Rejected("That folder is not empty.")
        rel = self._relative(path)
        shutil.rmtree(path)
        remove_metadata_prefix(self.root, rel)
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


def _image_name(name: str) -> str | None:
    """A relative png or jpg path, or none when the text is not one."""
    cleaned = name.replace("\\", "/").strip().strip("/")
    if not cleaned or ":" in cleaned or cleaned.startswith("~") or "\x00" in cleaned:
        return None
    parts = Path(cleaned).parts
    if not parts or any(part in {".", ".."} or part.startswith(".") for part in parts):
        return None
    if Path(cleaned).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return None
    return cleaned


def _media_type(suffix: str) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
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
