"""Domain records. Each method commits one transaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from typing import Any
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

from favlo.config_tools import GROUP_ID
from web_api.conversation import GROUP_ID as CONVERSATION_GROUP_ID
from web_api.cron import ScheduleError, check_cron, check_timezone, due
from web_api.db import FAVLO_AGENT_ID, LOCAL_PROVIDER_ID, connect, migrate
from web_api.events import envelope, now
from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_GROUP_ID
from web_api.ids import mint
from web_api.mcp_groups import GROUP_NAMES, ORDER
from web_api.session_tools import GROUP_ID as SESSION_GROUP_ID


class Conflict(Exception):
    pass


class NotFound(Exception):
    pass


class Rejected(Exception):
    pass


_KNOWN_GROUPS = {GROUP_ID, CONVERSATION_GROUP_ID, HEARTBEAT_GROUP_ID, SESSION_GROUP_ID}


def _agent_record(conn: sqlite3.Connection, row: Any) -> dict[str, Any]:
    agent = dict(row)
    groups = conn.execute(
        "SELECT group_id FROM agent_tool_group WHERE agent_id = ? ORDER BY group_id",
        (agent["id"],),
    ).fetchall()
    agent["tool_groups"] = [item["group_id"] for item in groups]
    return agent


def _heartbeat_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["enabled"] = bool(item["enabled"])
    item.pop("fired_minute", None)
    return item


_KIND_SQL = {
    "all": "s.kind IN ('channel', 'workspace_chat')",
    "agent": "s.kind = 'channel' AND c.kind = 'agent'",
    "channel": "s.kind = 'channel' AND IFNULL(c.kind, '') != 'agent'",
    "workspace_chat": "s.kind = 'workspace_chat'",
}


def _like(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped.lower()}%"


def _conversation_row(row: Any) -> dict[str, Any]:
    session_kind = row["session_kind"]
    channel_kind = row["channel_kind"]
    name = row["channel_name"] or row["session_id"]
    if session_kind == "workspace_chat":
        kind = "workspace_chat"
        if row["channel_name"] is None:
            name = "Workspace chat"
    elif channel_kind == "agent":
        kind = "agent"
    else:
        kind = "channel"
    last_at = row["last_at"] if "last_at" in row.keys() else None
    return {"id": row["session_id"], "kind": kind, "name": name, "last_at": last_at}


def _mcp_fields(body: dict[str, Any]) -> tuple[str, str, str]:
    transport = str(body.get("transport") or "")
    command = str(body.get("command") or "").strip()
    url = str(body.get("url") or "").strip()
    if transport == "stdio":
        if not _command_present(command):
            raise Rejected("A command is required.")
        return transport, command, ""
    if transport == "http":
        return transport, "", _mcp_http_url(url)
    raise Rejected("Transport is http or stdio.")


def _mcp_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise Rejected("An http URL is required.")
    path = parsed.path or "/"
    host = (parsed.hostname or "").lower()
    favlo = host in {"127.0.0.1", "localhost"} and parsed.port == 8129
    if favlo and path.rstrip("/") in {"", "/v1"}:
        path = "/mcp"
    elif path.rstrip("/") == "/mcp":
        path = "/mcp"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _command_present(command: str) -> bool:
    import shlex

    return bool(shlex.split(command)) if command.strip() else False


def _group_switches(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): bool(value) for key, value in parsed.items()}


def _mcp_server(row: Any) -> dict[str, Any]:
    switches = _group_switches(row["group_switches"])
    rank = {group_id: index for index, group_id in enumerate(ORDER)}
    ordered = sorted(switches, key=lambda group_id: rank.get(group_id, len(rank)))
    return {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "name": row["name"],
        "transport": row["transport"],
        "command": row["command"],
        "url": row["url"],
        "enabled": bool(row["enabled"]),
        "status": row["status"],
        "status_message": row["status_message"],
        "revision": row["revision"],
        "groups": [
            {
                "id": group_id,
                "name": GROUP_NAMES.get(group_id, group_id),
                "enabled": switches[group_id],
                "tools": [],
            }
            for group_id in ordered
        ],
    }


def _save_switches(conn: sqlite3.Connection, server_id: str, groups: list[Any]) -> None:
    """Store which groups are on. Names and schemas stay in the catalog cache."""
    row = conn.execute(
        "SELECT group_switches FROM mcp_server WHERE id = ?",
        (server_id,),
    ).fetchone()
    switches = _group_switches(row["group_switches"] if row is not None else "")
    for item in groups:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            continue
        switches[item["id"]] = bool(item.get("enabled"))
    conn.execute(
        "UPDATE mcp_server SET group_switches = ? WHERE id = ?",
        (json.dumps(switches), server_id),
    )


def _schedule(expression: str, timezone: str) -> tuple[str, str]:
    try:
        return check_cron(expression), check_timezone(timezone)
    except ScheduleError as exc:
        raise Rejected(str(exc)) from exc


def _replace_groups(conn: sqlite3.Connection, agent_id: str, groups: list[str]) -> None:
    if any(item not in _KNOWN_GROUPS for item in groups):
        raise Rejected("That tool group is not available.")
    conn.execute("DELETE FROM agent_tool_group WHERE agent_id = ?", (agent_id,))
    for group_id in dict.fromkeys(groups):
        conn.execute(
            "INSERT INTO agent_tool_group (agent_id, group_id) VALUES (?, ?)",
            (agent_id, group_id),
        )


def _owned_sessions(conn: sqlite3.Connection, agent_id: str) -> list[str]:
    """Sessions whose participant is this agent, plus its sole agent channel."""
    found = [
        item["id"]
        for item in conn.execute(
            "SELECT id FROM session WHERE participant_id = ?",
            (agent_id,),
        )
    ]
    channels = conn.execute(
        """
        SELECT c.id FROM channel c
        JOIN channel_member m ON m.channel_id = c.id
        WHERE m.agent_id = ? AND c.kind = 'agent'
        """,
        (agent_id,),
    ).fetchall()
    for channel in channels:
        others = conn.execute(
            """
            SELECT 1 FROM channel_member
            WHERE channel_id = ? AND agent_id != ?
            """,
            (channel["id"], agent_id),
        ).fetchone()
        if others is not None:
            continue
        for item in conn.execute(
            "SELECT id FROM session WHERE channel_id = ?",
            (channel["id"],),
        ):
            if item["id"] not in found:
                found.append(item["id"])
    return found


def _rows(conn: sqlite3.Connection, statement: str, ids: list[str]) -> list[sqlite3.Row]:
    if not ids:
        return []
    marks = ", ".join("?" for _ in ids)
    return list(conn.execute(f"{statement} ({marks})", ids))


def _folder_payload(row: Any) -> dict[str, Any]:
    kind = row["kind"]
    parent = row["parent_id"]
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "parent_id": parent or None,
        "name": row["name"],
        "kind": kind,
        "provider": row["provider"] or None,
        "location": row["location"] if kind == "local" else None,
        "machine_id": row["machine_id"] or "",
        "remote_id": row["remote_id"],
        "revision": row["revision"],
    }


def _folder_name(value: object) -> str:
    name = " ".join(str(value or "").split())
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise Rejected("A name is required.")
    return name


def _folder_in(conn: sqlite3.Connection, workspace_id: str, folder_id: str) -> Any:
    row = conn.execute(
        "SELECT * FROM folder WHERE id = ? AND workspace_id = ?",
        (folder_id, workspace_id),
    ).fetchone()
    if row is None:
        raise NotFound("That folder does not exist.")
    return row


def _reject_sibling(
    conn: sqlite3.Connection,
    workspace_id: str,
    parent_id: str | None,
    name: str,
    folder_id: str | None,
) -> None:
    if parent_id:
        rows = conn.execute(
            """
            SELECT id FROM folder
            WHERE workspace_id = ? AND parent_id = ? AND name = ? COLLATE NOCASE
            """,
            (workspace_id, parent_id, name),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id FROM folder
            WHERE workspace_id = ? AND parent_id IS NULL AND name = ? COLLATE NOCASE
            """,
            (workspace_id, name),
        ).fetchall()
    if any(row["id"] != folder_id for row in rows):
        raise Rejected("That folder name is already here.")


def _folder_fields(
    conn: sqlite3.Connection, workspace_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    from web_api.profile import known_cloud

    kind = str(body.get("kind") or "")
    name = _folder_name(body.get("name"))
    provider = str(body.get("provider") or "").strip() or None
    parent_id = str(body.get("parent_id") or "").strip() or None
    remote_id = str(body.get("remote_id") or "").strip() or None
    if kind == "local":
        if provider:
            raise Rejected("A local folder has no provider.")
        if parent_id:
            raise Rejected("A local folder stays at the workspace root.")
        location = str(body.get("location") or "").strip()
        if not location:
            raise Rejected("A local folder needs a location.")
        from pathlib import Path

        path = Path(location).expanduser()
        if not path.exists() or not path.is_dir():
            raise Rejected("That directory does not exist.")
        return {
            "name": name,
            "kind": "local",
            "provider": None,
            "location": str(path.resolve()),
            "remote_id": None,
            "parent_id": None,
        }
    if kind != "cloud":
        raise Rejected("kind is local or cloud.")
    if not known_cloud(provider):
        raise Rejected("That provider is not in the catalog.")
    if parent_id:
        parent = _folder_in(conn, workspace_id, parent_id)
        if parent["kind"] != "cloud":
            raise Rejected("That would download.")
    return {
        "name": name,
        "kind": "cloud",
        "provider": provider,
        "location": None,
        "remote_id": remote_id,
        "parent_id": parent_id,
    }


def _move_parent(conn: sqlite3.Connection, workspace_id: str, row: Any, parent: object) -> str | None:
    parent_id = str(parent or "").strip() or None
    if row["kind"] == "local":
        if parent_id:
            raise Rejected("That would upload.")
        return None
    if parent_id == row["id"]:
        raise Rejected("A folder cannot be moved into itself.")
    if parent_id:
        parent_row = _folder_in(conn, workspace_id, parent_id)
        if parent_row["kind"] != "cloud":
            raise Rejected("That would download.")
        seen = {str(row["id"])}
        current: str | None = parent_id
        while current:
            if current in seen:
                raise Rejected("A folder cannot be moved into itself.")
            seen.add(current)
            nxt = conn.execute(
                "SELECT parent_id FROM folder WHERE id = ? AND workspace_id = ?",
                (current, workspace_id),
            ).fetchone()
            if nxt is None:
                raise Rejected("That folder is not in this workspace.")
            current = nxt["parent_id"] or None
    return parent_id


def _seed_gallery_skills(conn: sqlite3.Connection, workspace_id: str, created: str) -> None:
    from web_api.workspace_skill_catalog import skills_for

    for name, description, definition in skills_for("gallery"):
        conn.execute(
            """
            INSERT INTO skill (
                id, workspace_id, name, description, definition, revision, created_at
            )
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (mint("skill"), workspace_id, name, description, definition, created),
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, '1')",
            (f"workspace_skill_seed:{workspace_id}:{name}",),
        )


def _copy_default_skills(conn: sqlite3.Connection, agent_id: str, created: str) -> None:
    rows = conn.execute(
        """
        SELECT name, description, definition FROM global_skill
        WHERE add_by_default = 1
        ORDER BY name COLLATE NOCASE, id
        """
    ).fetchall()
    for row in rows:
        conn.execute(
            """
            INSERT INTO agent_skill (
                id, agent_id, name, description, definition, revision, created_at
            )
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (mint("skill"), agent_id, row["name"], row["description"], row["definition"], created),
        )


def _skill_fields(body: dict[str, Any]) -> tuple[str, str, str]:
    name = " ".join(str(body.get("name") or "").split())
    description = str(body.get("description") or "").strip()
    definition = str(body.get("definition") or "").strip()
    if not name:
        raise Rejected("A name is required.")
    return name, description, definition


def _skill_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "name": row["name"],
        "description": row["description"],
        "definition": row["definition"] or "",
        "revision": row["revision"],
    }


def _agent_skill_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "name": row["name"],
        "description": row["description"],
        "definition": row["definition"] or "",
        "revision": row["revision"],
    }


def _global_skill_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "definition": row["definition"] or "",
        "add_by_default": bool(row["add_by_default"]),
        "revision": row["revision"],
    }


def _agent_skills_for(conn: sqlite3.Connection, agent_id: str | None = None) -> list[dict[str, Any]]:
    if agent_id:
        rows = conn.execute(
            """
            SELECT * FROM agent_skill
            WHERE agent_id = ?
            ORDER BY name COLLATE NOCASE, id
            """,
            (agent_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM agent_skill ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
    return [_agent_skill_payload(row) for row in rows]


def _skills_for(conn: sqlite3.Connection, workspace_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM skill
        WHERE workspace_id = ?
        ORDER BY name COLLATE NOCASE, id
        """,
        (workspace_id,),
    ).fetchall()
    return [_skill_payload(row) for row in rows]


def _workspace_payload(conn: sqlite3.Connection, row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "channel_id": row["channel_id"],
        "session_id": row["session_id"],
        "name": row["name"],
        "type": row["type"],
        "revision": row["revision"],
        "skills": _skills_for(conn, row["id"]),
    }


def _require_workspace(conn: sqlite3.Connection, workspace_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM workspace WHERE id = ?", (workspace_id,)).fetchone()
    if row is None:
        raise NotFound("That workspace does not exist.")
    return _workspace_payload(conn, row)


def _execute_in(conn: sqlite3.Connection, statement: str, ids: list[str]) -> None:
    if not ids:
        return
    marks = ", ".join("?" for _ in ids)
    conn.execute(f"{statement} ({marks})", ids)


def _hash(body: dict[str, Any]) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class Store:
    def __init__(self, path: Any) -> None:
        self._lock = threading.Lock()
        self._conn = connect(path)
        migrate(self._conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _write(self, fn: Any) -> Any:
        with self._lock:
            try:
                result = fn(self._conn)
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    def _read(self, fn: Any) -> Any:
        with self._lock:
            return fn(self._conn)

    def _event(
        self,
        conn: sqlite3.Connection,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, Any],
        channel_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        actor: dict[str, str] | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM event WHERE stream_id = ?",
            (stream_id,),
        ).fetchone()
        sequence = int(row[0])
        event_id = mint("evt")
        body = envelope(
            event_id=event_id,
            stream_id=stream_id,
            event_type=event_type,
            payload=payload,
            sequence=sequence,
            channel_id=channel_id,
            session_id=session_id,
            run_id=run_id,
            actor=actor,
            correlation_id=correlation_id,
        )
        conn.execute(
            "INSERT INTO event (id, stream_id, sequence, type, body) VALUES (?, ?, ?, ?, ?)",
            (event_id, stream_id, sequence, event_type, json.dumps(body)),
        )
        return body

    def _claim(
        self, conn: sqlite3.Connection, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]] | None:
        if not key:
            raise Rejected("An Idempotency-Key header is required.")
        digest = _hash(body)
        row = conn.execute(
            "SELECT body_hash, status, result FROM command WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["body_hash"] != digest:
            raise Conflict("This idempotency key was used with a different body.")
        return int(row["status"]), json.loads(row["result"])

    def _remember(
        self,
        conn: sqlite3.Connection,
        key: str,
        body: dict[str, Any],
        status: int,
        result: dict[str, Any],
    ) -> None:
        conn.execute(
            "INSERT INTO command (key, body_hash, status, result) VALUES (?, ?, ?, ?)",
            (key, _hash(body), status, json.dumps(result)),
        )

    def providers(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            providers = [dict(row) for row in conn.execute("SELECT * FROM provider ORDER BY name")]
            rows = conn.execute("SELECT * FROM provider_model ORDER BY display_name")
            models = [dict(row) for row in rows]
            for provider in providers:
                provider["enabled"] = bool(provider["enabled"])
                provider["models"] = [
                    {**model, "enabled": bool(model["enabled"])}
                    for model in models
                    if model["provider_id"] == provider["id"]
                ]
            return providers

        return self._read(read)

    def provider_url(self, provider_id: str) -> str:
        def read(conn: sqlite3.Connection) -> str:
            row = conn.execute(
                "SELECT base_url, enabled FROM provider WHERE id = ?",
                (provider_id,),
            ).fetchone()
            if row is None or not row["enabled"]:
                raise NotFound("That provider does not exist.")
            return str(row["base_url"])

        return self._read(read)

    def sync_models(self, provider_id: str, model_ids: list[str]) -> list[dict[str, Any]]:
        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            provider = conn.execute(
                "SELECT id FROM provider WHERE id = ?",
                (provider_id,),
            ).fetchone()
            if provider is None:
                raise NotFound("That provider does not exist.")
            cleaned: list[str] = []
            seen: set[str] = set()
            for model_id in model_ids:
                item = model_id.strip()
                if item and item not in seen:
                    seen.add(item)
                    cleaned.append(item)
            for item in cleaned:
                existing = conn.execute(
                    "SELECT id FROM provider_model WHERE provider_id = ? AND model_id = ?",
                    (provider_id, item),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO provider_model (
                            id, provider_id, model_id, display_name, enabled, revision
                        )
                        VALUES (?, ?, ?, ?, 1, 1)
                        """,
                        (mint("model"), provider_id, item, item),
                    )
                else:
                    conn.execute(
                        "UPDATE provider_model SET enabled = 1 WHERE id = ?",
                        (existing["id"],),
                    )
            if cleaned:
                marks = ", ".join("?" for _ in cleaned)
                conn.execute(
                    f"""
                    UPDATE provider_model
                    SET enabled = 0
                    WHERE provider_id = ? AND model_id NOT IN ({marks})
                    """,
                    (provider_id, *cleaned),
                )
            else:
                conn.execute(
                    "UPDATE provider_model SET enabled = 0 WHERE provider_id = ?",
                    (provider_id,),
                )
            rows = conn.execute(
                """
                SELECT * FROM provider_model
                WHERE provider_id = ? AND enabled = 1
                ORDER BY display_name
                """,
                (provider_id,),
            ).fetchall()
            return [{**dict(row), "enabled": True} for row in rows]

        return self._write(write)

    def add_model(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            provider = conn.execute(
                "SELECT id FROM provider WHERE id = ?",
                (body["provider_id"],),
            ).fetchone()
            if provider is None:
                raise NotFound("That provider does not exist.")
            model_id = body["model_id"].strip()
            display = body["display_name"].strip() or model_id
            if not model_id:
                raise Rejected("A model id is required.")
            existing = conn.execute(
                "SELECT id FROM provider_model WHERE provider_id = ? AND model_id = ?",
                (body["provider_id"], model_id),
            ).fetchone()
            if existing is not None:
                raise Conflict("That model is already listed.")
            record = {
                "id": mint("model"),
                "provider_id": body["provider_id"],
                "model_id": model_id,
                "display_name": display,
                "enabled": True,
                "revision": 1,
            }
            conn.execute(
                """
                INSERT INTO provider_model (
                    id, provider_id, model_id, display_name, enabled, revision
                )
                VALUES (?, ?, ?, ?, 1, 1)
                """,
                (record["id"], record["provider_id"], model_id, display),
            )
            self._remember(conn, key or "", body, 201, record)
            return 201, record, []

        return self._write(write)

    def update_model(self, model_row: str, expected: int, enabled: bool) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("SELECT * FROM provider_model WHERE id = ?", (model_row,)).fetchone()
            if row is None:
                raise NotFound("That model does not exist.")
            if int(row["revision"]) != expected:
                raise Conflict("The model was updated. Reload and try again.")
            revision = expected + 1
            conn.execute(
                "UPDATE provider_model SET enabled = ?, revision = ? WHERE id = ?",
                (1 if enabled else 0, revision, model_row),
            )
            updated = dict(row)
            updated["enabled"] = enabled
            updated["revision"] = revision
            return updated

        return self._write(write)

    def ensure_initial_model(self, model_id: str) -> None:
        """Store the registry model as the application default when none is set."""

        def write(conn: sqlite3.Connection) -> None:
            settings = conn.execute(
                "SELECT default_model_id FROM settings WHERE id = 1",
            ).fetchone()
            if settings["default_model_id"]:
                return
            cleaned = model_id.strip()
            if not cleaned:
                return
            existing = conn.execute(
                """
                SELECT id, enabled FROM provider_model
                WHERE provider_id = ? AND model_id = ?
                """,
                (LOCAL_PROVIDER_ID, cleaned),
            ).fetchone()
            if existing is None:
                record_id = mint("model")
                conn.execute(
                    """
                    INSERT INTO provider_model (
                        id, provider_id, model_id, display_name, enabled, revision
                    )
                    VALUES (?, ?, ?, ?, 1, 1)
                    """,
                    (record_id, LOCAL_PROVIDER_ID, cleaned, cleaned),
                )
            else:
                record_id = str(existing["id"])
                if not existing["enabled"]:
                    conn.execute(
                        "UPDATE provider_model SET enabled = 1 WHERE id = ?",
                        (record_id,),
                    )
            conn.execute(
                """
                UPDATE settings
                SET default_model_id = ?, revision = revision + 1
                WHERE id = 1
                """,
                (record_id,),
            )

        self._write(write)

    def settings(self) -> dict[str, Any]:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT default_model_id, exchange_budget, revision FROM settings WHERE id = 1",
            ).fetchone()
            return dict(row)

        return self._read(read)

    def update_settings(
        self, expected: int, default_model_id: str | None, exchange_budget: int
    ) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("SELECT revision FROM settings WHERE id = 1").fetchone()
            if int(row["revision"]) != expected:
                raise Conflict("Settings were updated. Reload and try again.")
            if exchange_budget < 1:
                raise Rejected("The exchange budget is a positive integer.")
            if default_model_id is not None:
                model = conn.execute(
                    "SELECT enabled FROM provider_model WHERE id = ?",
                    (default_model_id,),
                ).fetchone()
                if model is None or not model["enabled"]:
                    raise Rejected("Choose an enabled model.")
            revision = expected + 1
            conn.execute(
                """
                UPDATE settings
                SET default_model_id = ?, exchange_budget = ?, revision = ?
                WHERE id = 1
                """,
                (default_model_id, exchange_budget, revision),
            )
            return {
                "default_model_id": default_model_id,
                "exchange_budget": exchange_budget,
                "revision": revision,
            }

        return self._write(write)

    def use_model(self, provider_id: str, model_id: str, expected: int) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("SELECT revision FROM settings WHERE id = 1").fetchone()
            if int(row["revision"]) != expected:
                raise Conflict("Settings were updated. Reload and try again.")
            cleaned = model_id.strip()
            if not cleaned:
                raise Rejected("Choose a model.")
            existing = conn.execute(
                """
                SELECT id, enabled FROM provider_model
                WHERE provider_id = ? AND model_id = ?
                """,
                (provider_id, cleaned),
            ).fetchone()
            if existing is None or not existing["enabled"]:
                raise Rejected("That model is not available.")
            record_id = str(existing["id"])
            revision = expected + 1
            conn.execute(
                """
                UPDATE settings
                SET default_model_id = ?, revision = ?
                WHERE id = 1
                """,
                (record_id, revision),
            )
            budget = conn.execute(
                "SELECT exchange_budget FROM settings WHERE id = 1",
            ).fetchone()
            return {
                "default_model_id": record_id,
                "model_id": cleaned,
                "exchange_budget": int(budget["exchange_budget"]),
                "revision": revision,
            }

        return self._write(write)

    def agents(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute("SELECT * FROM agent ORDER BY name").fetchall()
            return [_agent_record(conn, row) for row in rows]

        return self._read(read)

    def create_agent(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            name = body["name"].strip()
            if not name:
                raise Rejected("A name is required.")
            model_id = body.get("default_model_id")
            if model_id is not None:
                self._require_model(conn, model_id)
            agent_id = mint("agent")
            channel_id = mint("channel")
            session_id = mint("session")
            persona = body.get("persona") or ""
            conn.execute(
                """
                INSERT INTO agent (id, name, persona, default_model_id, revision)
                VALUES (?, ?, ?, ?, 1)
                """,
                (agent_id, name, persona, model_id),
            )
            conn.execute(
                "INSERT INTO channel (id, kind, name, revision) VALUES (?, 'agent', ?, 1)",
                (channel_id, name),
            )
            conn.execute(
                "INSERT INTO channel_member (channel_id, agent_id) VALUES (?, ?)",
                (channel_id, agent_id),
            )
            created = now()
            conn.execute(
                """
                INSERT INTO session (id, kind, channel_id, participant_id, created_at)
                VALUES (?, 'channel', ?, ?, ?)
                """,
                (session_id, channel_id, agent_id, created),
            )
            _copy_default_skills(conn, agent_id, created)
            groups = body.get("tool_groups")
            if groups is not None:
                _replace_groups(conn, agent_id, groups)
            agent = _agent_record(
                conn,
                {
                    "id": agent_id,
                    "name": name,
                    "persona": persona,
                    "default_model_id": model_id,
                    "revision": 1,
                },
            )
            result = {
                "agent": agent,
                "channel": {
                    "id": channel_id,
                    "kind": "agent",
                    "name": name,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "revision": 1,
                },
                "session": {"id": session_id, "kind": "channel", "channel_id": channel_id},
            }
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="agent.created",
                    payload=result,
                    channel_id=channel_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_agent(
        self,
        agent_id: str,
        expected: int,
        name: str,
        persona: str,
        default_model_id: str | None,
        tool_groups: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM agent WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise NotFound("That agent does not exist.")
            if int(row["revision"]) != expected:
                raise Conflict("The agent was updated. Reload and try again.")
            cleaned = name.strip()
            if not cleaned:
                raise Rejected("A name is required.")
            if default_model_id is not None:
                self._require_model(conn, default_model_id)
            revision = expected + 1
            conn.execute(
                """
                UPDATE agent
                SET name = ?, persona = ?, default_model_id = ?, revision = ?
                WHERE id = ?
                """,
                (cleaned, persona, default_model_id, revision, agent_id),
            )
            conn.execute(
                """
                UPDATE channel SET name = ?, revision = revision + 1
                WHERE id IN (SELECT channel_id FROM channel_member WHERE agent_id = ?)
                """,
                (cleaned, agent_id),
            )
            if tool_groups is not None:
                _replace_groups(conn, agent_id, tool_groups)
            updated = _agent_record(
                conn,
                {
                    "id": agent_id,
                    "name": cleaned,
                    "persona": persona,
                    "default_model_id": default_model_id,
                    "revision": revision,
                },
            )
            return updated, []

        return self._write(write)

    def delete_agent(
        self, agent_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Remove one agent and the rows that belong to it.

        Tasks name a server and a session, so they go first. Tool rows go
        before servers. Heartbeats name a session, so they go before sessions.
        Messages, runs, and events go before the sessions they sit on. Membership
        goes before an agent channel that would otherwise be empty. The agent
        row is last.
        """

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
            row = conn.execute("SELECT * FROM agent WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise NotFound("That agent does not exist.")
            if agent_id == FAVLO_AGENT_ID:
                raise Rejected("Favlo stays.")
            sessions = _owned_sessions(conn, agent_id)
            running = [
                item["id"]
                for item in conn.execute(
                    "SELECT id FROM run WHERE status = 'running' AND agent_id = ?",
                    (agent_id,),
                )
            ]
            active = _rows(
                conn,
                "SELECT id FROM run WHERE status = 'running' AND session_id IN",
                sessions,
            )
            for item in active:
                if item["id"] not in running:
                    running.append(item["id"])
            conn.execute("DELETE FROM mcp_task WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM mcp_server WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM heartbeat WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agent_skill WHERE agent_id = ?", (agent_id,))
            _execute_in(conn, "DELETE FROM message WHERE session_id IN", sessions)
            conn.execute(
                "DELETE FROM message WHERE actor_kind = 'agent' AND actor_id = ?",
                (agent_id,),
            )
            _execute_in(conn, "DELETE FROM run WHERE session_id IN", sessions)
            conn.execute("DELETE FROM run WHERE agent_id = ?", (agent_id,))
            _execute_in(
                conn,
                "DELETE FROM event WHERE stream_id IN",
                [f"session:{session_id}" for session_id in sessions],
            )
            _execute_in(conn, "DELETE FROM session WHERE id IN", sessions)
            workspaces = [
                item["id"]
                for item in conn.execute(
                    """
                    SELECT w.id FROM workspace w
                    JOIN channel_member m ON m.channel_id = w.channel_id
                    WHERE m.agent_id = ?
                    """,
                    (agent_id,),
                )
            ]
            _execute_in(conn, "DELETE FROM skill WHERE workspace_id IN", workspaces)
            _execute_in(conn, "DELETE FROM folder WHERE workspace_id IN", workspaces)
            _execute_in(conn, "DELETE FROM workspace WHERE id IN", workspaces)
            channels = [
                item["channel_id"]
                for item in conn.execute(
                    """
                    SELECT channel_id FROM channel_member
                    WHERE agent_id = ?
                    """,
                    (agent_id,),
                )
            ]
            conn.execute("DELETE FROM channel_member WHERE agent_id = ?", (agent_id,))
            _execute_in(
                conn,
                "DELETE FROM channel WHERE id IN",
                [
                    channel_id
                    for channel_id in channels
                    if conn.execute(
                        "SELECT 1 FROM channel_member WHERE channel_id = ?",
                        (channel_id,),
                    ).fetchone()
                    is None
                ],
            )
            conn.execute("DELETE FROM agent_tool_group WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agent WHERE id = ?", (agent_id,))
            result = {"id": agent_id, "name": row["name"]}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="agent.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events, running

        return self._write(write)

    def heartbeats(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT * FROM heartbeat
                ORDER BY cron_expression, id
                """
            ).fetchall()
            return [_heartbeat_record(row) for row in rows]

        return self._read(read)

    def create_heartbeat(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent = conn.execute(
                "SELECT id FROM agent WHERE id = ?",
                (body["agent_id"],),
            ).fetchone()
            if agent is None:
                raise NotFound("That agent does not exist.")
            schedule = _schedule(body["cron_expression"], body["timezone"])
            heartbeat_id = mint("heartbeat")
            result = {
                "id": heartbeat_id,
                "agent_id": body["agent_id"],
                "cron_expression": schedule[0],
                "timezone": schedule[1],
                "instructions": body.get("instructions") or "",
                "enabled": bool(body.get("enabled", True)),
                "revision": 1,
            }
            conn.execute(
                """
                INSERT INTO heartbeat (
                    id, agent_id, cron_expression, timezone, instructions, enabled, revision
                )
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    result["id"],
                    result["agent_id"],
                    result["cron_expression"],
                    result["timezone"],
                    result["instructions"],
                    int(result["enabled"]),
                ),
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="heartbeat.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_heartbeat(
        self, heartbeat_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM heartbeat WHERE id = ?", (heartbeat_id,)).fetchone()
            if row is None:
                raise NotFound("That heartbeat does not exist.")
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("The heartbeat was updated. Reload and try again.")
            schedule = _schedule(body["cron_expression"], body["timezone"])
            revision = expected + 1
            result = {
                "id": heartbeat_id,
                "agent_id": row["agent_id"],
                "cron_expression": schedule[0],
                "timezone": schedule[1],
                "instructions": body.get("instructions") or "",
                "enabled": bool(body["enabled"]),
                "revision": revision,
            }
            conn.execute(
                """
                UPDATE heartbeat
                SET cron_expression = ?, timezone = ?, instructions = ?, enabled = ?, revision = ?
                WHERE id = ?
                """,
                (
                    result["cron_expression"],
                    result["timezone"],
                    result["instructions"],
                    int(result["enabled"]),
                    revision,
                    heartbeat_id,
                ),
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="heartbeat.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_heartbeat(
        self, heartbeat_id: str, agent_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM heartbeat WHERE id = ?", (heartbeat_id,)).fetchone()
            if row is None or row["agent_id"] != agent_id:
                raise NotFound("That heartbeat does not exist.")
            conn.execute("DELETE FROM heartbeat WHERE id = ?", (heartbeat_id,))
            result = {"id": heartbeat_id, "agent_id": agent_id}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="heartbeat.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def abandoned_runs(self) -> list[str]:
        """Runs still marked running. The process that owned them is gone."""

        def read(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute("SELECT id FROM run WHERE status = 'running'").fetchall()
            return [row["id"] for row in rows]

        return self._read(read)

    def dispatch_heartbeats(
        self, moment: datetime
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Start each enabled heartbeat whose cron matches this minute.

        A minute already considered is left alone. A heartbeat that still has a
        running reply consumes the minute and does not start another run.
        """
        if moment.tzinfo is None:
            raise ScheduleError("A heartbeat time needs a timezone.")

        def write(
            conn: sqlite3.Connection,
        ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
            rows = conn.execute("SELECT * FROM heartbeat WHERE enabled = 1").fetchall()
            started: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for row in rows:
                local = moment.astimezone(ZoneInfo(row["timezone"]))
                if not due(row["cron_expression"], local):
                    continue
                slot = local.strftime("%Y-%m-%dT%H:%M%z")
                if row["fired_minute"] == slot:
                    continue
                session_id = self._ensure_heartbeat_session(conn, row)
                active = conn.execute(
                    "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                    (session_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE heartbeat SET fired_minute = ? WHERE id = ?",
                    (slot, row["id"]),
                )
                if active is not None:
                    continue
                started.append(self._start_heartbeat_run(conn, row, session_id))
            return started

        return self._write(write)

    def begin_heartbeat_run(
        self, key: str | None, heartbeat_id: str
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        body = {"heartbeat_id": heartbeat_id, "action": "run"}

        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            row = conn.execute("SELECT * FROM heartbeat WHERE id = ?", (heartbeat_id,)).fetchone()
            if row is None:
                raise NotFound("That heartbeat does not exist.")
            session_id = self._ensure_heartbeat_session(conn, row)
            active = conn.execute(
                "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                (session_id,),
            ).fetchone()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            result, events = self._start_heartbeat_run(conn, row, session_id)
            self._remember(conn, key or "", body, 202, result)
            return 202, result, events

        return self._write(write)

    def _ensure_heartbeat_session(self, conn: sqlite3.Connection, row: sqlite3.Row) -> str:
        session_id = row["session_id"]
        if session_id:
            return str(session_id)
        session_id = mint("session")
        conn.execute(
            """
            INSERT INTO session (id, kind, channel_id, participant_id, created_at)
            VALUES (?, 'heartbeat', NULL, ?, ?)
            """,
            (session_id, row["agent_id"], now()),
        )
        conn.execute(
            "UPDATE heartbeat SET session_id = ? WHERE id = ?",
            (session_id, row["id"]),
        )
        return session_id

    def _start_heartbeat_run(
        self, conn: sqlite3.Connection, row: sqlite3.Row, session_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        text = str(row["instructions"] or "").strip() or "Run this heartbeat."
        message_id = mint("message")
        run_id = mint("run")
        command_id = mint("cmd")
        created = now()
        actor = {"kind": "user", "id": "local"}
        conn.execute(
            """
            INSERT INTO message (id, session_id, role, body, actor_kind, actor_id, created_at)
            VALUES (?, ?, 'user', ?, 'user', 'local', ?)
            """,
            (message_id, session_id, text, created),
        )
        conn.execute(
            """
            INSERT INTO run (id, session_id, agent_id, status, error, started_at, finished_at)
            VALUES (?, ?, ?, 'running', NULL, ?, NULL)
            """,
            (run_id, session_id, row["agent_id"], created),
        )
        stream_id = f"session:{session_id}"
        events = [
            self._event(
                conn,
                stream_id=stream_id,
                event_type="message.created",
                payload={"message_id": message_id, "role": "user", "text": text},
                session_id=session_id,
                actor=actor,
                correlation_id=command_id,
            ),
            self._event(
                conn,
                stream_id=stream_id,
                event_type="run.started",
                payload={"status": "running"},
                session_id=session_id,
                run_id=run_id,
                actor=actor,
                correlation_id=command_id,
            ),
        ]
        result = {
            "command_id": command_id,
            "status": "accepted",
            "session_id": session_id,
            "run_id": run_id,
            "message_id": message_id,
            "text": text,
        }
        return result, events

    def list_conversations(self, kind: str, limit: int) -> list[dict[str, Any]]:
        clause = _KIND_SQL[kind]

        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                f"""
                SELECT s.id AS session_id, s.kind AS session_kind,
                       c.kind AS channel_kind, c.name AS channel_name,
                       (
                           SELECT created_at FROM message
                           WHERE session_id = s.id
                           ORDER BY rowid DESC
                           LIMIT 1
                       ) AS last_at
                FROM session s
                LEFT JOIN channel c ON c.id = s.channel_id
                WHERE {clause}
                ORDER BY COALESCE(last_at, s.created_at) DESC, s.id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [_conversation_row(row) for row in rows]

        return self._read(read)

    def conversations_matching(self, conversation: str) -> list[dict[str, Any]]:
        needle = conversation.strip()

        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT s.id AS session_id, s.kind AS session_kind,
                       c.kind AS channel_kind, c.name AS channel_name,
                       NULL AS last_at
                FROM session s
                LEFT JOIN channel c ON c.id = s.channel_id
                WHERE s.kind IN ('channel', 'workspace_chat')
                  AND (
                      s.id = ?
                      OR s.channel_id = ?
                      OR lower(c.name) = lower(?)
                  )
                ORDER BY s.id
                """,
                (needle, needle, needle),
            ).fetchall()
            return [_conversation_row(row) for row in rows]

        return self._read(read)

    def search_messages(self, query: str, kind: str, limit: int) -> list[dict[str, Any]]:
        clause = _KIND_SQL[kind]

        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                f"""
                SELECT m.role, m.body, m.created_at,
                       s.id AS session_id, s.kind AS session_kind,
                       c.kind AS channel_kind, c.name AS channel_name
                FROM message m
                JOIN session s ON s.id = m.session_id
                LEFT JOIN channel c ON c.id = s.channel_id
                WHERE {clause}
                  AND lower(m.body) LIKE ? ESCAPE '\\'
                ORDER BY m.created_at DESC, m.rowid DESC
                LIMIT ?
                """,
                (_like(query), limit),
            ).fetchall()
            found = []
            for row in rows:
                item = _conversation_row(row)
                item["role"] = row["role"]
                item["body"] = row["body"]
                item["created_at"] = row["created_at"]
                found.append(item)
            return found

        return self._read(read)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT role, body, created_at
                FROM message
                WHERE session_id = ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

        return self._read(read)

    def mcp_servers(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM mcp_server ORDER BY name, id"
            ).fetchall()
            return [_mcp_server(row) for row in rows]

        return self._read(read)

    def mcp_server(self, server_id: str) -> dict[str, Any] | None:
        found = [row for row in self.mcp_servers() if row["id"] == server_id]
        return found[0] if found else None

    def create_mcp_server(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent = conn.execute(
                "SELECT id FROM agent WHERE id = ?",
                (body["agent_id"],),
            ).fetchone()
            if agent is None:
                raise NotFound("That agent does not exist.")
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            server_id = mint("mcp")
            conn.execute(
                """
                INSERT INTO mcp_server (
                    id, agent_id, name, transport, command, url, enabled,
                    status, status_message, revision, group_switches
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'unchecked', '', 1, '{}')
                """,
                (
                    server_id,
                    body["agent_id"],
                    name,
                    transport,
                    command,
                    url,
                    int(bool(body.get("enabled", True))),
                ),
            )
            groups = body.get("groups")
            if isinstance(groups, list):
                _save_switches(conn, server_id, groups)
            result = _mcp_server(
                conn.execute("SELECT * FROM mcp_server WHERE id = ?", (server_id,)).fetchone(),
            )
            self._remember(conn, key or "", body, 201, result)
            return 201, result, []

        return self._write(write)

    def update_mcp_server(self, server_id: str, body: dict[str, Any]) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            if row is None:
                raise NotFound("That MCP server does not exist.")
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("The MCP server was updated. Reload and try again.")
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            revision = expected + 1
            conn.execute(
                """
                UPDATE mcp_server
                SET name = ?, transport = ?, command = ?, url = ?, enabled = ?, revision = ?
                WHERE id = ?
                """,
                (
                    name,
                    transport,
                    command,
                    url,
                    int(bool(body["enabled"])),
                    revision,
                    server_id,
                ),
            )
            groups = body.get("groups")
            if isinstance(groups, list):
                _save_switches(conn, server_id, groups)
            updated = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            return _mcp_server(updated)

        return self._write(write)

    def mark_mcp_connected(self, server_id: str) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            if row is None:
                raise NotFound("That MCP server does not exist.")
            conn.execute(
                """
                UPDATE mcp_server
                SET status = 'connected', status_message = 'Connection succeeded',
                    revision = revision + 1
                WHERE id = ?
                """,
                (server_id,),
            )
            updated = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            return _mcp_server(updated)

        return self._write(write)

    def mark_mcp_failed(self, server_id: str) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            if row is None:
                raise NotFound("That MCP server does not exist.")
            conn.execute(
                """
                UPDATE mcp_server
                SET status = 'failed', status_message = 'Connection failed',
                    revision = revision + 1
                WHERE id = ?
                """,
                (server_id,),
            )
            updated = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            return _mcp_server(updated)

        return self._write(write)

    def enabled_mcp_servers(self, agent_id: str) -> list[dict[str, Any]]:
        """Enabled servers and the group switches. Tool definitions are not here."""

        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT * FROM mcp_server
                WHERE agent_id = ? AND enabled = 1
                ORDER BY name, id
                """,
                (agent_id,),
            ).fetchall()
            return [
                {
                    "server_id": row["id"],
                    "server_name": row["name"],
                    "transport": row["transport"],
                    "command": row["command"],
                    "url": row["url"],
                    "switches": _group_switches(row["group_switches"]),
                }
                for row in rows
            ]

        return self._read(read)

    def begin_mcp_task(self, body: dict[str, Any]) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            task_id = mint("mctask")
            created = now()
            conn.execute(
                """
                INSERT INTO mcp_task (
                    id, remote_id, server_id, agent_id, session_id, tool_name,
                    status, poll_interval_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'working', ?, ?)
                """,
                (
                    task_id,
                    body["remote_id"],
                    body["server_id"],
                    body["agent_id"],
                    body["session_id"],
                    body["tool_name"],
                    int(body["poll_interval_ms"]),
                    created,
                ),
            )
            return {
                "id": task_id,
                "remote_id": body["remote_id"],
                "server_id": body["server_id"],
                "agent_id": body["agent_id"],
                "session_id": body["session_id"],
                "tool_name": body["tool_name"],
                "status": "working",
                "poll_interval_ms": int(body["poll_interval_ms"]),
            }

        return self._write(write)

    def working_mcp_tasks(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM mcp_task WHERE status = 'working' ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]

        return self._read(read)

    def deliver_mcp_result(self, task_pk: str, status: str) -> bool:
        """Mark a working task finished. The result is not a chat message."""

        def write(conn: sqlite3.Connection) -> bool:
            row = conn.execute("SELECT * FROM mcp_task WHERE id = ?", (task_pk,)).fetchone()
            if row is None or row["status"] != "working":
                return False
            conn.execute(
                "UPDATE mcp_task SET status = ? WHERE id = ?",
                (status, task_pk),
            )
            return True

        return self._write(write)

    def clear_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Delete the stored turns for one session and say that it was cleared."""

        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            session = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise NotFound("That session does not exist.")
            conn.execute("DELETE FROM message WHERE session_id = ?", (session_id,))
            conn.execute(
                "DELETE FROM run WHERE session_id = ? AND status != 'running'",
                (session_id,),
            )
            return [
                self._event(
                    conn,
                    stream_id=f"session:{session_id}",
                    event_type="session.cleared",
                    payload={"session_id": session_id},
                    channel_id=session["channel_id"],
                    session_id=session_id,
                    actor={"kind": "agent", "id": session["participant_id"] or ""},
                )
            ]

        return self._write(write)

    def begin_task_correction(
        self, session_id: str, agent_id: str, cue: str
    ) -> tuple[str | None, list[dict[str, Any]], bool]:
        """Start a turn that answers a finished task.

        The cue is the model prompt. It is not a chat message. A running reply
        returns waiting so the caller can try again after that reply ends.
        """

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[str | None, list[dict[str, Any]], bool]:
            text = cue.strip()
            if not text:
                return None, [], False
            session = conn.execute(
                "SELECT * FROM session WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None, [], False
            active = conn.execute(
                "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                (session_id,),
            ).fetchone()
            if active is not None:
                return None, [], True
            run_id = mint("run")
            created = now()
            conn.execute(
                """
                INSERT INTO run (
                    id, session_id, agent_id, status, error, started_at, finished_at, cue
                )
                VALUES (?, ?, ?, 'running', NULL, ?, NULL, ?)
                """,
                (run_id, session_id, agent_id, created, text),
            )
            actor = {"kind": "agent", "id": agent_id}
            events = [
                self._event(
                    conn,
                    stream_id=f"session:{session_id}",
                    event_type="run.started",
                    payload={"status": "running"},
                    channel_id=session["channel_id"],
                    session_id=session_id,
                    run_id=run_id,
                    actor=actor,
                )
            ]
            return run_id, events, False

        return self._write(write)

    def channels(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT c.id, c.kind, c.name, c.revision, m.agent_id, s.id AS session_id
                FROM channel c
                JOIN channel_member m ON m.channel_id = c.id
                JOIN session s ON s.channel_id = c.id AND s.kind = 'channel'
                WHERE c.kind = 'agent'
                ORDER BY c.name COLLATE NOCASE, c.id
                """
            ).fetchall()
            return [dict(row) for row in rows]

        return self._read(read)

    def list_workspaces(self, channel_id: str | None = None) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if channel_id:
                channel = conn.execute(
                    "SELECT id FROM channel WHERE id = ?",
                    (channel_id,),
                ).fetchone()
                if channel is None:
                    raise NotFound("That channel does not exist.")
                rows = conn.execute(
                    """
                    SELECT * FROM workspace
                    WHERE channel_id = ?
                    ORDER BY created_at, id
                    """,
                    (channel_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM workspace
                    ORDER BY name COLLATE NOCASE, id
                    """
                ).fetchall()
            return [_workspace_payload(conn, row) for row in rows]

        return self._read(read)

    def workspace(self, workspace_id: str) -> dict[str, Any]:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            return _require_workspace(conn, workspace_id)

        return self._read(read)

    def create_workspace(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            channel_id = str(body.get("channel_id") or "")
            name = str(body.get("name") or "").strip()
            kind = str(body.get("type") or "gallery")
            if not name:
                raise Rejected("A name is required.")
            if kind != "gallery":
                raise Rejected("type is gallery.")
            channel = conn.execute("SELECT id FROM channel WHERE id = ?", (channel_id,)).fetchone()
            if channel is None:
                raise NotFound("That channel does not exist.")
            member = conn.execute(
                "SELECT agent_id FROM channel_member WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if member is None:
                raise Rejected("That channel has no agent.")
            workspace_id = mint("workspace")
            session_id = mint("session")
            created = now()
            conn.execute(
                """
                INSERT INTO session (
                    id, kind, channel_id, participant_id, workspace_id, created_at
                )
                VALUES (?, 'workspace_chat', ?, ?, ?, ?)
                """,
                (session_id, channel_id, member["agent_id"], workspace_id, created),
            )
            conn.execute(
                """
                INSERT INTO workspace (
                    id, channel_id, session_id, name, type, revision, created_at
                )
                VALUES (?, ?, ?, ?, 'gallery', 1, ?)
                """,
                (workspace_id, channel_id, session_id, name, created),
            )
            _seed_gallery_skills(conn, workspace_id, created)
            result = _require_workspace(conn, workspace_id)
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="workspace.created",
                    payload=result,
                    channel_id=channel_id,
                    session_id=session_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def delete_workspace(
        self, workspace_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        def write(
            conn: sqlite3.Connection,
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
            row = conn.execute("SELECT * FROM workspace WHERE id = ?", (workspace_id,)).fetchone()
            if row is None:
                raise NotFound("That workspace does not exist.")
            session_id = row["session_id"]
            running = [
                item["id"]
                for item in conn.execute(
                    "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                    (session_id,),
                )
            ]
            conn.execute("DELETE FROM message WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM run WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM event WHERE stream_id = ?", (f"session:{session_id}",))
            conn.execute("DELETE FROM folder WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM skill WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
            conn.execute("DELETE FROM workspace WHERE id = ?", (workspace_id,))
            result = {"id": workspace_id, "name": row["name"]}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="workspace.deleted",
                    payload=result,
                    channel_id=row["channel_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events, running

        return self._write(write)

    def list_folders(self, workspace_id: str, parent_id: str | None = None) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            _require_workspace(conn, workspace_id)
            if parent_id:
                rows = conn.execute(
                    """
                    SELECT * FROM folder
                    WHERE workspace_id = ? AND parent_id = ?
                    ORDER BY name COLLATE NOCASE, id
                    """,
                    (workspace_id, parent_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM folder
                    WHERE workspace_id = ? AND parent_id IS NULL
                    ORDER BY name COLLATE NOCASE, id
                    """,
                    (workspace_id,),
                ).fetchall()
            return [_folder_payload(row) for row in rows]

        return self._read(read)

    def add_folder(
        self, key: str | None, workspace_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            workspace = _require_workspace(conn, workspace_id)
            if workspace["type"] != "gallery":
                raise Rejected("That workspace is not a gallery.")
            fields = _folder_fields(conn, workspace_id, body)
            _reject_sibling(conn, workspace_id, fields["parent_id"], fields["name"], None)
            folder_id = mint("folder")
            created = now()
            conn.execute(
                """
                INSERT INTO folder (
                    id, workspace_id, parent_id, name, kind, provider, location,
                    machine_id, remote_id, revision, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, 1, ?)
                """,
                (
                    folder_id,
                    workspace_id,
                    fields["parent_id"],
                    fields["name"],
                    fields["kind"],
                    fields["provider"],
                    fields["location"],
                    fields["remote_id"],
                    created,
                ),
            )
            result = _folder_payload(
                conn.execute("SELECT * FROM folder WHERE id = ?", (folder_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="folder.created",
                    payload=result,
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_folder(
        self, workspace_id: str, folder_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(conn, workspace_id)
            row = _folder_in(conn, workspace_id, folder_id)
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("That folder was updated. Reload and try again.")
            name = " ".join(str(body.get("name") or row["name"]).split())
            if not name:
                raise Rejected("A name is required.")
            parent_id = row["parent_id"]
            if body.get("move"):
                parent_id = _move_parent(conn, workspace_id, row, body.get("parent_id"))
            _reject_sibling(conn, workspace_id, parent_id, name, folder_id)
            conn.execute(
                """
                UPDATE folder
                SET name = ?, parent_id = ?, revision = revision + 1
                WHERE id = ?
                """,
                (name, parent_id, folder_id),
            )
            result = _folder_payload(
                conn.execute("SELECT * FROM folder WHERE id = ?", (folder_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="folder.updated",
                    payload=result,
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_folder_row(
        self, workspace_id: str, folder_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(conn, workspace_id)
            row = _folder_in(conn, workspace_id, folder_id)
            child = conn.execute(
                "SELECT 1 FROM folder WHERE parent_id = ?",
                (folder_id,),
            ).fetchone()
            if child is not None:
                raise Rejected("Move the folders inside it first.")
            conn.execute("DELETE FROM folder WHERE id = ?", (folder_id,))
            conn.execute(
                """
                UPDATE workspace SET focus_folder_id = NULL
                WHERE id = ? AND focus_folder_id = ?
                """,
                (workspace_id, folder_id),
            )
            result = {"id": folder_id, "name": row["name"]}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="folder.deleted",
                    payload=result,
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def set_focus(self, workspace_id: str, folder_id: str | None) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            _require_workspace(conn, workspace_id)
            if folder_id:
                row = _folder_in(conn, workspace_id, folder_id)
                if row["kind"] != "local":
                    raise Rejected("Open a local folder.")
            conn.execute(
                "UPDATE workspace SET focus_folder_id = ? WHERE id = ?",
                (folder_id, workspace_id),
            )
            return {"folder_id": folder_id}

        return self._write(write)

    def has_local_folder(self, workspace_id: str) -> bool:
        def read(conn: sqlite3.Connection) -> bool:
            _require_workspace(conn, workspace_id)
            row = conn.execute(
                "SELECT 1 FROM folder WHERE workspace_id = ? AND kind = 'local'",
                (workspace_id,),
            ).fetchone()
            return row is not None

        return self._read(read)

    def local_location(self, workspace_id: str, folder_id: str | None = None) -> str:
        def read(conn: sqlite3.Connection) -> str:
            _require_workspace(conn, workspace_id)
            if folder_id:
                row = _folder_in(conn, workspace_id, folder_id)
                if row["kind"] != "local" or not row["location"]:
                    raise Rejected("That folder is not on this machine.")
                return str(row["location"])
            workspace = conn.execute(
                "SELECT focus_folder_id FROM workspace WHERE id = ?",
                (workspace_id,),
            ).fetchone()
            focus = workspace["focus_folder_id"] if workspace is not None else None
            if focus:
                focused = conn.execute(
                    "SELECT * FROM folder WHERE id = ? AND workspace_id = ?",
                    (focus, workspace_id),
                ).fetchone()
                if focused is not None and focused["kind"] == "local" and focused["location"]:
                    return str(focused["location"])
            rows = conn.execute(
                """
                SELECT * FROM folder
                WHERE workspace_id = ? AND kind = 'local'
                ORDER BY created_at, id
                """,
                (workspace_id,),
            ).fetchall()
            if len(rows) == 1 and rows[0]["location"]:
                return str(rows[0]["location"])
            if not rows:
                raise Rejected("Save a local folder first.")
            raise Rejected("Open a local folder.")

        return self._read(read)

    def list_agent_skills(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if agent_id:
                agent = conn.execute("SELECT id FROM agent WHERE id = ?", (agent_id,)).fetchone()
                if agent is None:
                    raise NotFound("That agent does not exist.")
            return _agent_skills_for(conn, agent_id)

        return self._read(read)

    def create_agent_skill(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent_id = str(body.get("agent_id") or "")
            agent = conn.execute("SELECT id FROM agent WHERE id = ?", (agent_id,)).fetchone()
            if agent is None:
                raise NotFound("That agent does not exist.")
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                "SELECT id FROM agent_skill WHERE agent_id = ? AND name = ?",
                (agent_id, name),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already on this agent.")
            skill_id = mint("skill")
            created = now()
            conn.execute(
                """
                INSERT INTO agent_skill (
                    id, agent_id, name, description, definition, revision, created_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (skill_id, agent_id, name, description, definition, created),
            )
            result = _agent_skill_payload(
                conn.execute("SELECT * FROM agent_skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="agent_skill.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_agent_skill(
        self, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM agent_skill WHERE id = ?", (skill_id,)).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                """
                SELECT id FROM agent_skill
                WHERE agent_id = ? AND name = ? AND id != ?
                """,
                (row["agent_id"], name, skill_id),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already on this agent.")
            revision = expected + 1
            conn.execute(
                """
                UPDATE agent_skill
                SET name = ?, description = ?, definition = ?, revision = ?
                WHERE id = ?
                """,
                (name, description, definition, revision, skill_id),
            )
            result = _agent_skill_payload(
                conn.execute("SELECT * FROM agent_skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="agent_skill.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_agent_skill(self, skill_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM agent_skill WHERE id = ?", (skill_id,)).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            conn.execute("DELETE FROM agent_skill WHERE id = ?", (skill_id,))
            result = {"id": skill_id, "agent_id": row["agent_id"]}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="agent_skill.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def list_global_skills(self) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT * FROM global_skill
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()
            return [_global_skill_payload(row) for row in rows]

        return self._read(read)

    def create_global_skill(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                "SELECT id FROM global_skill WHERE name = ?",
                (name,),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already in the catalog.")
            skill_id = mint("skill")
            created = now()
            conn.execute(
                """
                INSERT INTO global_skill (
                    id, name, description, definition, add_by_default, revision, created_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (skill_id, name, description, definition, int(bool(body.get("add_by_default"))), created),
            )
            result = _global_skill_payload(
                conn.execute("SELECT * FROM global_skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="global_skill.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_global_skill(
        self, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM global_skill WHERE id = ?", (skill_id,)).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                "SELECT id FROM global_skill WHERE name = ? AND id != ?",
                (name, skill_id),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already in the catalog.")
            revision = expected + 1
            add_by_default = row["add_by_default"]
            if "add_by_default" in body and body["add_by_default"] is not None:
                add_by_default = int(bool(body["add_by_default"]))
            conn.execute(
                """
                UPDATE global_skill
                SET name = ?, description = ?, definition = ?, add_by_default = ?, revision = ?
                WHERE id = ?
                """,
                (name, description, definition, add_by_default, revision, skill_id),
            )
            result = _global_skill_payload(
                conn.execute("SELECT * FROM global_skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="global_skill.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_global_skill(self, skill_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = conn.execute("SELECT * FROM global_skill WHERE id = ?", (skill_id,)).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            conn.execute("DELETE FROM global_skill WHERE id = ?", (skill_id,))
            result = {"id": skill_id}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="global_skill.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def copy_global_skill(
        self, key: str | None, skill_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            row = conn.execute("SELECT * FROM global_skill WHERE id = ?", (skill_id,)).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            agent_id = str(body.get("agent_id") or "").strip()
            workspace_id = str(body.get("workspace_id") or "").strip()
            if bool(agent_id) == bool(workspace_id):
                raise Rejected("Choose an agent or a workspace.")
            name = str(row["name"])
            description = str(row["description"])
            definition = str(row["definition"] or "")
            created = now()
            copy_id = mint("skill")
            if agent_id:
                agent = conn.execute("SELECT id FROM agent WHERE id = ?", (agent_id,)).fetchone()
                if agent is None:
                    raise NotFound("That agent does not exist.")
                taken = conn.execute(
                    "SELECT id FROM agent_skill WHERE agent_id = ? AND name = ?",
                    (agent_id, name),
                ).fetchone()
                if taken is not None:
                    raise Rejected("That skill name is already on this agent.")
                conn.execute(
                    """
                    INSERT INTO agent_skill (
                        id, agent_id, name, description, definition, revision, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (copy_id, agent_id, name, description, definition, created),
                )
                result = _agent_skill_payload(
                    conn.execute("SELECT * FROM agent_skill WHERE id = ?", (copy_id,)).fetchone()
                )
                events = [
                    self._event(
                        conn,
                        stream_id="app",
                        event_type="agent_skill.created",
                        payload=result,
                        actor={"kind": "user", "id": "local"},
                    )
                ]
            else:
                workspace = _require_workspace(conn, workspace_id)
                taken = conn.execute(
                    "SELECT id FROM skill WHERE workspace_id = ? AND name = ?",
                    (workspace_id, name),
                ).fetchone()
                if taken is not None:
                    raise Rejected("That skill name is already on this workspace.")
                conn.execute(
                    """
                    INSERT INTO skill (
                        id, workspace_id, name, description, definition, revision, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (copy_id, workspace_id, name, description, definition, created),
                )
                result = _skill_payload(
                    conn.execute("SELECT * FROM skill WHERE id = ?", (copy_id,)).fetchone()
                )
                events = [
                    self._event(
                        conn,
                        stream_id="app",
                        event_type="skill.created",
                        payload={"workspace_id": workspace_id, "skill": result},
                        channel_id=workspace["channel_id"],
                        session_id=workspace["session_id"],
                        actor={"kind": "user", "id": "local"},
                    )
                ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def list_skills(self, workspace_id: str) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            _require_workspace(conn, workspace_id)
            return _skills_for(conn, workspace_id)

        return self._read(read)

    def create_skill(
        self, key: str | None, workspace_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            workspace = _require_workspace(conn, workspace_id)
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                "SELECT id FROM skill WHERE workspace_id = ? AND name = ?",
                (workspace_id, name),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already on this workspace.")
            skill_id = mint("skill")
            created = now()
            conn.execute(
                """
                INSERT INTO skill (
                    id, workspace_id, name, description, definition, revision, created_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (skill_id, workspace_id, name, description, definition, created),
            )
            result = _skill_payload(
                conn.execute("SELECT * FROM skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="skill.created",
                    payload={"workspace_id": workspace_id, "skill": result},
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(conn, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_skill(
        self, workspace_id: str, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(conn, workspace_id)
            row = conn.execute(
                "SELECT * FROM skill WHERE id = ? AND workspace_id = ?",
                (skill_id, workspace_id),
            ).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row["revision"]) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = conn.execute(
                """
                SELECT id FROM skill
                WHERE workspace_id = ? AND name = ? AND id != ?
                """,
                (workspace_id, name, skill_id),
            ).fetchone()
            if taken is not None:
                raise Rejected("That skill name is already on this workspace.")
            revision = expected + 1
            conn.execute(
                """
                UPDATE skill
                SET name = ?, description = ?, definition = ?, revision = ?
                WHERE id = ?
                """,
                (name, description, definition, revision, skill_id),
            )
            result = _skill_payload(
                conn.execute("SELECT * FROM skill WHERE id = ?", (skill_id,)).fetchone()
            )
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="skill.updated",
                    payload={"workspace_id": workspace_id, "skill": result},
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_skill(
        self, workspace_id: str, skill_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(conn, workspace_id)
            row = conn.execute(
                "SELECT * FROM skill WHERE id = ? AND workspace_id = ?",
                (skill_id, workspace_id),
            ).fetchone()
            if row is None:
                raise NotFound("That skill does not exist.")
            conn.execute("DELETE FROM skill WHERE id = ?", (skill_id,))
            result = {"id": skill_id, "workspace_id": workspace_id}
            events = [
                self._event(
                    conn,
                    stream_id="app",
                    event_type="skill.deleted",
                    payload=result,
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def workspace_for_session(self, session_id: str) -> dict[str, str] | None:
        def read(conn: sqlite3.Connection) -> dict[str, str] | None:
            session = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if session is None or session["kind"] != "workspace_chat":
                return None
            workspace_id = session["workspace_id"]
            if not workspace_id:
                return None
            workspace = conn.execute(
                "SELECT id, type FROM workspace WHERE id = ?",
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                return None
            return {"id": str(workspace["id"]), "type": str(workspace["type"])}

        return self._read(read)

    def gallery_for_session(self, session_id: str) -> str | None:
        def read(conn: sqlite3.Connection) -> str | None:
            session = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if session is None or session["kind"] != "workspace_chat":
                return None
            workspace_id = session["workspace_id"]
            if not workspace_id:
                return None
            workspace = conn.execute(
                "SELECT type FROM workspace WHERE id = ?",
                (workspace_id,),
            ).fetchone()
            if workspace is None or workspace["type"] != "gallery":
                return None
            folder = conn.execute(
                "SELECT 1 FROM folder WHERE workspace_id = ? AND kind = 'local'",
                (workspace_id,),
            ).fetchone()
            if folder is None:
                return None
            return str(workspace_id)

        return self._read(read)

    def note_gallery(self, workspace_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            row = conn.execute("SELECT * FROM workspace WHERE id = ?", (workspace_id,)).fetchone()
            if row is None:
                raise NotFound("That workspace does not exist.")
            return [
                self._event(
                    conn,
                    stream_id=f"session:{row['session_id']}",
                    event_type="gallery.updated",
                    payload=payload,
                    channel_id=row["channel_id"],
                    session_id=row["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]

        return self._write(write)

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            session = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise NotFound("That session does not exist.")
            messages = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM message WHERE session_id = ? ORDER BY rowid",
                    (session_id,),
                )
            ]
            runs = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM run WHERE session_id = ? ORDER BY rowid",
                    (session_id,),
                )
            ]
            stream_id = f"session:{session_id}"
            cursor = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM event WHERE stream_id = ?",
                (stream_id,),
            ).fetchone()
            return {
                "session": dict(session),
                "messages": messages,
                "runs": runs,
                "cursor": int(cursor[0]),
            }

        return self._read(read)

    def events_after(self, stream_id: str, cursor: int) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT body FROM event WHERE stream_id = ? AND sequence > ? ORDER BY sequence",
                (stream_id, cursor),
            ).fetchall()
            return [json.loads(row["body"]) for row in rows]

        return self._read(read)

    def begin_message(
        self,
        key: str | None,
        session_id: str,
        text: str,
        folder: str | None = None,
        image: str | None = None,
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        body = {"session_id": session_id, "text": text, "folder": folder, "image": image}

        def write(conn: sqlite3.Connection) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(conn, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            session = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise NotFound("That session does not exist.")
            cleaned = text.strip()
            if not cleaned:
                raise Rejected("A message is required.")
            if len(cleaned.encode()) > 8 * 1024 * 1024:
                raise Rejected("A message is at most 8 MiB.")
            active = conn.execute(
                "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                (session_id,),
            ).fetchone()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            message_id = mint("message")
            run_id = mint("run")
            command_id = mint("cmd")
            created = now()
            actor = {"kind": "user", "id": "local"}
            conn.execute(
                """
                INSERT INTO message (id, session_id, role, body, actor_kind, actor_id, created_at)
                VALUES (?, ?, 'user', ?, 'user', 'local', ?)
                """,
                (message_id, session_id, cleaned, created),
            )
            conn.execute(
                """
                INSERT INTO run (
                    id, session_id, agent_id, status, error, started_at, finished_at,
                    gallery_folder, gallery_image
                )
                VALUES (?, ?, ?, 'running', NULL, ?, NULL, ?, ?)
                """,
                (run_id, session_id, session["participant_id"], created, folder, image),
            )
            stream_id = f"session:{session_id}"
            channel_id = session["channel_id"]
            events = [
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload={"message_id": message_id, "role": "user", "text": cleaned},
                    channel_id=channel_id,
                    session_id=session_id,
                    actor=actor,
                    correlation_id=command_id,
                ),
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="run.started",
                    payload={"status": "running"},
                    channel_id=channel_id,
                    session_id=session_id,
                    run_id=run_id,
                    actor=actor,
                    correlation_id=command_id,
                ),
            ]
            result = {
                "command_id": command_id,
                "status": "accepted",
                "session_id": session_id,
                "run_id": run_id,
                "message_id": message_id,
                "text": cleaned,
            }
            self._remember(conn, key or "", body, 202, result)
            return 202, result, events

        return self._write(write)

    def run_context(self, run_id: str) -> dict[str, Any]:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise NotFound("That run does not exist.")
            session = conn.execute(
                "SELECT * FROM session WHERE id = ?",
                (run["session_id"],),
            ).fetchone()
            agent_row = conn.execute(
                "SELECT * FROM agent WHERE id = ?",
                (run["agent_id"],),
            ).fetchone()
            agent = _agent_record(conn, agent_row) if agent_row is not None else None
            history = [
                {
                    "role": row["role"],
                    "text": row["body"],
                    "reasoning": row["reasoning"] or "",
                }
                for row in conn.execute(
                    """
                    SELECT role, body, reasoning FROM message
                    WHERE session_id = ? AND role IN ('user', 'assistant')
                    ORDER BY rowid
                    """,
                    (run["session_id"],),
                )
            ]
            settings = conn.execute("SELECT default_model_id FROM settings WHERE id = 1").fetchone()
            model_row_id = agent["default_model_id"] or settings["default_model_id"]
            model = None
            base_url = None
            if model_row_id:
                found = conn.execute(
                    """
                    SELECT m.model_id, m.enabled AS model_enabled,
                           p.base_url, p.enabled AS provider_enabled
                    FROM provider_model m
                    JOIN provider p ON p.id = m.provider_id
                    WHERE m.id = ?
                    """,
                    (model_row_id,),
                ).fetchone()
                if found is not None and found["model_enabled"] and found["provider_enabled"]:
                    model = str(found["model_id"])
                    base_url = str(found["base_url"])
            cue = str(run["cue"] or "").strip()
            if cue:
                user_text = cue
                prior = history
            else:
                user_text = history[-1]["text"] if history else ""
                prior = history[:-1]
            folder = run["gallery_folder"]
            image = run["gallery_image"]
            return {
                "run": dict(run),
                "session": dict(session),
                "agent": agent,
                "model": model,
                "base_url": base_url,
                "prior": prior,
                "user_text": user_text,
                "gallery_folder": None if folder is None else str(folder),
                "gallery_image": None if image is None else str(image),
            }

        return self._read(read)

    def complete_run(self, run_id: str, text: str, reasoning: str = "") -> list[dict[str, Any]]:
        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            if run is None or run["status"] != "running":
                return []
            session = conn.execute(
                "SELECT * FROM session WHERE id = ?",
                (run["session_id"],),
            ).fetchone()
            message_id = mint("message")
            created = now()
            stored = reasoning.strip()
            conn.execute(
                """
                INSERT INTO message (
                    id, session_id, role, body, actor_kind, actor_id, created_at, reasoning
                )
                VALUES (?, ?, 'assistant', ?, 'agent', ?, ?, ?)
                """,
                (message_id, run["session_id"], text, run["agent_id"], created, stored or None),
            )
            conn.execute(
                "UPDATE run SET status = 'completed', finished_at = ? WHERE id = ?",
                (created, run_id),
            )
            stream_id = f"session:{run['session_id']}"
            actor = {"kind": "agent", "id": run["agent_id"]}
            payload: dict[str, Any] = {"message_id": message_id, "role": "assistant", "text": text}
            if stored:
                payload["reasoning"] = stored
            return [
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload=payload,
                    channel_id=session["channel_id"],
                    session_id=run["session_id"],
                    run_id=run_id,
                    actor=actor,
                ),
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="run.completed",
                    payload={"status": "completed"},
                    channel_id=session["channel_id"],
                    session_id=run["session_id"],
                    run_id=run_id,
                    actor=actor,
                ),
            ]

        return self._write(write)

    def note_empty_reply(self, run_id: str, reasoning: str) -> list[dict[str, Any]]:
        """Store reasoning from a reply that drew no text, then fail the run."""

        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            if run is None or run["status"] != "running":
                return []
            session = conn.execute(
                "SELECT * FROM session WHERE id = ?",
                (run["session_id"],),
            ).fetchone()
            message_id = mint("message")
            created = now()
            stored = reasoning.strip()
            conn.execute(
                """
                INSERT INTO message (
                    id, session_id, role, body, actor_kind, actor_id, created_at, reasoning
                )
                VALUES (?, ?, 'assistant', '', 'agent', ?, ?, ?)
                """,
                (message_id, run["session_id"], run["agent_id"], created, stored),
            )
            conn.execute(
                "UPDATE run SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                ("The model returned an empty reply.", created, run_id),
            )
            stream_id = f"session:{run['session_id']}"
            actor = {"kind": "agent", "id": run["agent_id"]}
            return [
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload={
                        "message_id": message_id,
                        "role": "assistant",
                        "text": "",
                        "reasoning": stored,
                    },
                    channel_id=session["channel_id"],
                    session_id=run["session_id"],
                    run_id=run_id,
                    actor=actor,
                ),
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="run.failed",
                    payload={
                        "status": "failed",
                        "message": "The model returned an empty reply.",
                    },
                    channel_id=session["channel_id"],
                    session_id=run["session_id"],
                    run_id=run_id,
                    actor=actor,
                ),
            ]

        return self._write(write)

    def finish_run(self, run_id: str, status: str, error: str | None) -> list[dict[str, Any]]:
        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            if run is None or run["status"] != "running":
                return []
            session = conn.execute(
                "SELECT channel_id FROM session WHERE id = ?",
                (run["session_id"],),
            ).fetchone()
            finished = now()
            conn.execute(
                "UPDATE run SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, finished, run_id),
            )
            event_type = "run.cancelled" if status == "cancelled" else "run.failed"
            payload: dict[str, Any] = {"status": status}
            if error:
                payload["message"] = error
            return [
                self._event(
                    conn,
                    stream_id=f"session:{run['session_id']}",
                    event_type=event_type,
                    payload=payload,
                    channel_id=session["channel_id"],
                    session_id=run["session_id"],
                    run_id=run_id,
                    actor={"kind": "agent", "id": run["agent_id"]},
                )
            ]

        return self._write(write)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            run = conn.execute(
                "SELECT id, status, session_id FROM run WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise NotFound("That run does not exist.")
            return dict(run)

        return self._read(read)

    def _require_model(self, conn: sqlite3.Connection, model_row: str) -> None:
        row = conn.execute(
            "SELECT enabled FROM provider_model WHERE id = ?",
            (model_row,),
        ).fetchone()
        if row is None or not row["enabled"]:
            raise Rejected("Choose an enabled model.")
