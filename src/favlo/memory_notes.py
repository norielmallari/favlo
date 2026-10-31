"""Shared observation store for MCP memory tools.

Vectors are 1024-float BGE-M3 dense embeddings. The file is ``~/.favlo/memory.db``.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import sqlite_vec
from sqlite_vec import serialize_float32

from favlo.logs import favlo_home

EMBED_MODEL = "bge-m3-int8"
DIM = 1024
_VEC_KNN = 5
_VEC_MAX = 8

_SCHEMA = """
create table if not exists meta (
  key text primary key,
  value text not null
);
create table if not exists observation (
  id text primary key,
  text text not null,
  created text not null,
  client text,
  deleted text
);
create virtual table if not exists observation_vec using vec0(
  embedding float[1024] distance_metric=cosine
);
create virtual table if not exists observation_fts using fts5(
  text,
  content='observation',
  content_rowid='rowid'
);
create unique index if not exists observation_text_client_live
  on observation(text, ifnull(client, '')) where deleted is null;
"""

_FTS_TRIGGERS = """
create trigger if not exists observation_ai after insert on observation begin
  insert into observation_fts(rowid, text) values (new.rowid, new.text);
end;
create trigger if not exists observation_ad after delete on observation begin
  insert into observation_fts(observation_fts, rowid, text)
    values ('delete', old.rowid, old.text);
end;
create trigger if not exists observation_au after update of text on observation begin
  insert into observation_fts(observation_fts, rowid, text)
    values ('delete', old.rowid, old.text);
  insert into observation_fts(rowid, text) values (new.rowid, new.text);
end;
"""


def memory_db_path(home: Path | None = None) -> Path:
    root = favlo_home() if home is None else home
    return root / "memory.db"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_id() -> str:
    return f"mem_{secrets.token_hex(4)}"


def _fts_query(query: str) -> str | None:
    parts = [piece for piece in query.split() if piece]
    if not parts:
        return None
    return " ".join(parts)


class MemoryStore:
    """SQLite store with vec0 nearest-neighbor search and FTS5 recall."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else memory_db_path()
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._path)
        db.execute("pragma journal_mode=WAL")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    def _ensure_schema(self) -> None:
        with self._lock:
            db = self._connect()
            try:
                db.executescript(_SCHEMA)
                db.execute("drop index if exists observation_text_live")
                db.executescript(_FTS_TRIGGERS)
                row = db.execute("select value from meta where key = 'model'").fetchone()
                if row is None:
                    db.execute(
                        "insert into meta(key, value) values ('model', ?)",
                        (EMBED_MODEL,),
                    )
                db.commit()
            finally:
                db.close()

    def remember(self, text: str, vector: list[float], client: str | None = None) -> str:
        """Insert one observation. Duplicate live text returns the existing id."""
        cleaned = text.strip()
        if not cleaned:
            message = "text is empty"
            raise ValueError(message)
        if len(vector) != DIM:
            message = f"embedding length is not {DIM}"
            raise ValueError(message)
        with self._lock:
            db = self._connect()
            try:
                existing = db.execute(
                    "select id from observation where text = ? and deleted is null "
                    "and ifnull(client, '') = ifnull(?, '')",
                    (cleaned, client),
                ).fetchone()
                if existing is not None:
                    return str(existing[0])
                mem_id = _new_id()
                created = _now()
                db.execute(
                    "insert into observation(id, text, created, client, deleted) "
                    "values (?, ?, ?, ?, null)",
                    (mem_id, cleaned, created, client),
                )
                rowid = db.execute(
                    "select rowid from observation where id = ?",
                    (mem_id,),
                ).fetchone()[0]
                db.execute(
                    "insert into observation_vec(rowid, embedding) values (?, ?)",
                    (rowid, serialize_float32(vector)),
                )
                db.commit()
                return mem_id
            finally:
                db.close()

    def recall(
        self,
        query: str,
        vector: list[float],
        *,
        reembed: Callable[[str], list[float]] | None = None,
        client: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to eight live rows ranked by vector similarity and FTS.

        A client limits the rows to that owner.
        """
        if len(vector) != DIM:
            message = f"embedding length is not {DIM}"
            raise ValueError(message)
        with self._lock:
            db = self._connect()
            try:
                self._sync_model(db, reembed)
                knn = self._knn(db, vector, client)
                by_id = {row["id"]: row for row in knn}
                fts = self._fts_hits(db, query, exclude=set(by_id), client=client)
                merged = knn + fts
                if len(merged) > _VEC_MAX:
                    merged = merged[:_VEC_MAX]
                return merged
            finally:
                db.close()

    def forget(self, mem_id: str, client: str | None = None) -> None:
        """Mark one observation deleted and drop its vector.

        A client can delete only a row stored for that owner.
        """
        with self._lock:
            db = self._connect()
            try:
                row = db.execute(
                    "select rowid, deleted, client from observation where id = ?",
                    (mem_id,),
                ).fetchone()
                if row is None:
                    message = f"unknown memory id {mem_id}"
                    raise KeyError(message)
                rowid, deleted, owner = row
                if client is not None and owner != client:
                    message = f"unknown memory id {mem_id}"
                    raise KeyError(message)
                if deleted is not None:
                    return
                stamp = _now()
                db.execute(
                    "update observation set deleted = ? where id = ?",
                    (stamp, mem_id),
                )
                db.execute("delete from observation_vec where rowid = ?", (rowid,))
                db.commit()
            finally:
                db.close()

    def _sync_model(
        self,
        db: sqlite3.Connection,
        reembed: Callable[[str], list[float]] | None,
    ) -> None:
        row = db.execute("select value from meta where key = 'model'").fetchone()
        stored = row[0] if row is not None else None
        if stored == EMBED_MODEL:
            return
        if reembed is None:
            return
        db.execute(
            "insert or replace into meta(key, value) values ('model', ?)",
            (EMBED_MODEL,),
        )
        rows = db.execute(
            "select rowid, text from observation where deleted is null",
        ).fetchall()
        for rowid, text in rows:
            vec = reembed(str(text))
            if len(vec) != DIM:
                message = f"embedding length is not {DIM}"
                raise ValueError(message)
            db.execute("delete from observation_vec where rowid = ?", (rowid,))
            db.execute(
                "insert into observation_vec(rowid, embedding) values (?, ?)",
                (rowid, serialize_float32(vec)),
            )
        db.commit()

    def _knn(
        self,
        db: sqlite3.Connection,
        vector: list[float],
        client: str | None,
    ) -> list[dict[str, Any]]:
        blob = serialize_float32(vector)
        rows = db.execute(
            """
            select o.id, o.text, o.created, o.client, v.distance
            from observation_vec v
            join observation o on o.rowid = v.rowid
            where v.embedding match ?
              and k = ?
              and o.deleted is null
              and (? is null or o.client = ?)
            """,
            (blob, _VEC_KNN, client, client),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for mem_id, text, created, client, distance in rows:
            score = 1.0 - float(distance)
            item: dict[str, Any] = {
                "id": mem_id,
                "text": text,
                "created": created,
                "score": score,
            }
            if client:
                item["client"] = client
            out.append(item)
        return out

    def _fts_hits(
        self,
        db: sqlite3.Connection,
        query: str,
        *,
        exclude: set[str],
        client: str | None,
    ) -> list[dict[str, Any]]:
        match = _fts_query(query)
        if match is None:
            return []
        try:
            hits = db.execute(
                """
                select o.id, o.text, o.created, o.client
                from observation_fts f
                join observation o on o.rowid = f.rowid
                where observation_fts match ?
                  and o.deleted is null
                  and (? is null or o.client = ?)
                limit ?
                """,
                (match, client, client, _VEC_MAX),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out: list[dict[str, Any]] = []
        for mem_id, text, created, client in hits:
            if mem_id in exclude:
                continue
            item: dict[str, Any] = {
                "id": mem_id,
                "text": text,
                "created": created,
                "score": None,
            }
            if client:
                item["client"] = client
            out.append(item)
            if len(out) + len(exclude) >= _VEC_MAX:
                break
        return out
