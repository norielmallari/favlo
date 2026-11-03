"""sqlite-vec index of gallery media metadata prompts."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

import sqlite_vec
from sqlite_vec import serialize_float32

from web_api.events import now

DIM = 1024

_SCHEMA = """
create table if not exists gallery_media (
  workspace_id text not null,
  path text not null,
  kind text not null,
  sidecar_mtime text,
  prompt_hash text not null,
  prompt_text text not null,
  updated_at text not null,
  primary key (workspace_id, path)
);
create virtual table if not exists gallery_media_vec using vec0(
  embedding float[1024] distance_metric=cosine
);
"""


class GalleryIndex:
    """Per-workspace media rows keyed by relative path."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._path)
        db.execute("pragma journal_mode=WAL")
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    def _ensure(self) -> None:
        with self._lock:
            db = self._connect()
            try:
                db.executescript(_SCHEMA)
                db.commit()
            finally:
                db.close()

    def upsert(
        self,
        workspace_id: str,
        path: str,
        *,
        kind: str,
        sidecar_mtime: str,
        prompt_text: str,
        vector: list[float],
    ) -> None:
        if len(vector) != DIM:
            raise ValueError(f"embedding length is not {DIM}")
        digest = hashlib.sha256(prompt_text.encode()).hexdigest()
        stamped = now()
        with self._lock:
            db = self._connect()
            try:
                prior = db.execute(
                    "select rowid from gallery_media where workspace_id = ? and path = ?",
                    (workspace_id, path),
                ).fetchone()
                if prior is not None:
                    rowid = int(prior["rowid"])
                    db.execute("delete from gallery_media_vec where rowid = ?", (rowid,))
                    db.execute(
                        """
                        update gallery_media
                        set kind = ?, sidecar_mtime = ?, prompt_hash = ?,
                            prompt_text = ?, updated_at = ?
                        where rowid = ?
                        """,
                        (kind, sidecar_mtime, digest, prompt_text, stamped, rowid),
                    )
                else:
                    db.execute(
                        """
                        insert into gallery_media (
                            workspace_id, path, kind, sidecar_mtime,
                            prompt_hash, prompt_text, updated_at
                        ) values (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (workspace_id, path, kind, sidecar_mtime, digest, prompt_text, stamped),
                    )
                    rowid = int(
                        db.execute(
                            "select rowid from gallery_media where workspace_id = ? and path = ?",
                            (workspace_id, path),
                        ).fetchone()["rowid"]
                    )
                db.execute(
                    "insert into gallery_media_vec(rowid, embedding) values (?, ?)",
                    (rowid, serialize_float32(vector)),
                )
                db.commit()
            finally:
                db.close()

    def delete(self, workspace_id: str, path: str) -> None:
        with self._lock:
            db = self._connect()
            try:
                row = db.execute(
                    "select rowid from gallery_media where workspace_id = ? and path = ?",
                    (workspace_id, path),
                ).fetchone()
                if row is None:
                    return
                rowid = int(row["rowid"])
                db.execute("delete from gallery_media_vec where rowid = ?", (rowid,))
                db.execute("delete from gallery_media where rowid = ?", (rowid,))
                db.commit()
            finally:
                db.close()

    def clear_workspace(self, workspace_id: str) -> None:
        with self._lock:
            db = self._connect()
            try:
                rows = db.execute(
                    "select rowid from gallery_media where workspace_id = ?",
                    (workspace_id,),
                ).fetchall()
                for row in rows:
                    db.execute("delete from gallery_media_vec where rowid = ?", (int(row["rowid"]),))
                db.execute("delete from gallery_media where workspace_id = ?", (workspace_id,))
                db.commit()
            finally:
                db.close()

    def count(self, workspace_id: str) -> int:
        with self._lock:
            db = self._connect()
            try:
                row = db.execute(
                    "select count(*) as n from gallery_media where workspace_id = ?",
                    (workspace_id,),
                ).fetchone()
                return int(row["n"])
            finally:
                db.close()

    def row(self, workspace_id: str, path: str) -> dict[str, Any] | None:
        """Indexed metadata for one path, or None when missing."""
        with self._lock:
            db = self._connect()
            try:
                found = db.execute(
                    """
                    select path, kind, prompt_hash, prompt_text, sidecar_mtime
                    from gallery_media
                    where workspace_id = ? and path = ?
                    """,
                    (workspace_id, path),
                ).fetchone()
                if found is None:
                    return None
                return {
                    "path": str(found["path"]),
                    "kind": str(found["kind"]),
                    "prompt_hash": str(found["prompt_hash"]),
                    "prompt_text": str(found["prompt_text"]),
                    "sidecar_mtime": str(found["sidecar_mtime"] or ""),
                }
            finally:
                db.close()

    def search(
        self,
        workspace_id: str,
        vector: list[float],
        *,
        limit: int = 12,
        kinds: list[str] | None = None,
        unlike: bool = False,
    ) -> list[dict[str, Any]]:
        """Nearest matches, or farthest when ``unlike`` is set."""
        if len(vector) != DIM:
            raise ValueError(f"embedding length is not {DIM}")
        capped = max(1, min(40, int(limit)))
        with self._lock:
            db = self._connect()
            try:
                total = int(
                    db.execute(
                        "select count(*) as n from gallery_media where workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()["n"]
                )
                # Nearest: a shortlist. Unlike: every row (vec0 only sorts ASC).
                knn = max(total, 1) if unlike else max(capped * 4, 1)
                rows = db.execute(
                    """
                    select m.path, m.kind, m.prompt_text, v.distance
                    from gallery_media_vec v
                    join gallery_media m on m.rowid = v.rowid
                    where m.workspace_id = ?
                      and v.embedding match ?
                      and k = ?
                    order by v.distance
                    """,
                    (workspace_id, serialize_float32(vector), knn),
                ).fetchall()
                allowed = set(kinds) if kinds else None
                ranked: list[dict[str, Any]] = []
                for row in rows:
                    if allowed is not None and row["kind"] not in allowed:
                        continue
                    distance = float(row["distance"])
                    ranked.append(
                        {
                            "path": row["path"],
                            "kind": row["kind"],
                            "score": max(0.0, 1.0 - distance),
                            "prompt_text": row["prompt_text"],
                        }
                    )
                if unlike:
                    return list(reversed(ranked[-capped:]))
                return ranked[:capped]
            finally:
                db.close()
