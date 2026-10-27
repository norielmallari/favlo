"""Domain records. Each method commits one transaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any
from urllib.parse import urlparse, urlunparse

from favlo.config_tools import GROUP_ID
from web_api.conversation import GROUP_ID as CONVERSATION_GROUP_ID
from web_api.cron import ScheduleError, check_cron, check_timezone
from web_api.db import LOCAL_PROVIDER_ID, connect, migrate
from web_api.events import envelope, now
from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_GROUP_ID
from web_api.ids import mint
from web_api.mcp_groups import ORDER as MCP_GROUP_ORDER
from web_api.mcp_groups import group_for


class Conflict(Exception):
    pass


class NotFound(Exception):
    pass


class Rejected(Exception):
    pass


_KNOWN_GROUPS = {GROUP_ID, CONVERSATION_GROUP_ID, HEARTBEAT_GROUP_ID}


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


def _mcp_server(conn: sqlite3.Connection, row: Any) -> dict[str, Any]:
    tools = conn.execute(
        """
        SELECT name, group_id FROM mcp_tool
        WHERE server_id = ?
        ORDER BY name
        """,
        (row["id"],),
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for tool in tools:
        grouped.setdefault(tool["group_id"], []).append(tool["name"])
    groups = conn.execute(
        "SELECT * FROM mcp_tool_group WHERE server_id = ?",
        (row["id"],),
    ).fetchall()
    rank = {group_id: index for index, group_id in enumerate(MCP_GROUP_ORDER)}
    ordered = sorted(groups, key=lambda item: rank.get(item["group_id"], len(rank)))
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
                "id": item["group_id"],
                "name": item["name"],
                "enabled": bool(item["enabled"]),
                "tools": grouped.get(item["group_id"], []),
            }
            for item in ordered
        ],
    }


def _enabled_map(groups: Any) -> dict[str, bool]:
    found: dict[str, bool] = {}
    if not isinstance(groups, list):
        return found
    for item in groups:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            found[item["id"]] = bool(item.get("enabled"))
    return found


def _replace_catalog(
    conn: sqlite3.Connection,
    server_id: str,
    tools: list[Any],
    enabled: dict[str, bool],
) -> None:
    conn.execute("DELETE FROM mcp_tool WHERE server_id = ?", (server_id,))
    conn.execute("DELETE FROM mcp_tool_group WHERE server_id = ?", (server_id,))
    seen: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        group_id, group_name = group_for(tool["name"])
        seen[group_id] = group_name
        schema = tool.get("schema", tool.get("input_schema"))
        if not isinstance(schema, dict):
            schema = {}
        description = tool.get("description")
        conn.execute(
            """
            INSERT INTO mcp_tool (server_id, name, group_id, description, schema)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                server_id,
                tool["name"],
                group_id,
                description if isinstance(description, str) else "",
                json.dumps(schema),
            ),
        )
    for group_id, group_name in seen.items():
        conn.execute(
            """
            INSERT INTO mcp_tool_group (server_id, group_id, name, enabled)
            VALUES (?, ?, ?, ?)
            """,
            (server_id, group_id, group_name, int(enabled.get(group_id, True))),
        )


def _set_mcp_groups(conn: sqlite3.Connection, server_id: str, groups: list[Any]) -> None:
    for item in groups:
        if not isinstance(item, dict):
            continue
        conn.execute(
            """
            UPDATE mcp_tool_group SET enabled = ?
            WHERE server_id = ? AND group_id = ?
            """,
            (int(bool(item.get("enabled"))), server_id, item.get("id")),
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
            result = {
                "agent": {
                    "id": agent_id,
                    "name": name,
                    "persona": persona,
                    "default_model_id": model_id,
                    "revision": 1,
                    "tool_groups": [],
                },
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
            session_id = row["session_id"]
            created = now()
            if not session_id:
                session_id = mint("session")
                conn.execute(
                    """
                    INSERT INTO session (id, kind, channel_id, participant_id, created_at)
                    VALUES (?, 'heartbeat', NULL, ?, ?)
                    """,
                    (session_id, row["agent_id"], created),
                )
                conn.execute(
                    "UPDATE heartbeat SET session_id = ? WHERE id = ?",
                    (session_id, heartbeat_id),
                )
            active = conn.execute(
                "SELECT id FROM run WHERE session_id = ? AND status = 'running'",
                (session_id,),
            ).fetchone()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            text = str(row["instructions"] or "").strip() or "Run this heartbeat."
            message_id = mint("message")
            run_id = mint("run")
            command_id = mint("cmd")
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
            self._remember(conn, key or "", body, 202, result)
            return 202, result, events

        return self._write(write)

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
            return [_mcp_server(conn, row) for row in rows]

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
                    status, status_message, revision
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'unchecked', '', 1)
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
            tools = body.get("tools")
            if isinstance(tools, list):
                _replace_catalog(conn, server_id, tools, _enabled_map(body.get("groups")))
                conn.execute(
                    """
                    UPDATE mcp_server
                    SET status = 'connected', status_message = 'Connection succeeded'
                    WHERE id = ?
                    """,
                    (server_id,),
                )
            result = _mcp_server(
                conn,
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
            tools = body.get("tools")
            if isinstance(tools, list):
                _replace_catalog(conn, server_id, tools, _enabled_map(body.get("groups")))
                conn.execute(
                    """
                    UPDATE mcp_server
                    SET status = 'connected', status_message = 'Connection succeeded'
                    WHERE id = ?
                    """,
                    (server_id,),
                )
            elif isinstance(body.get("groups"), list):
                _set_mcp_groups(conn, server_id, body["groups"])
            updated = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            return _mcp_server(conn, updated)

        return self._write(write)

    def store_mcp_catalog(self, server_id: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            if row is None:
                raise NotFound("That MCP server does not exist.")
            prior = {
                item["group_id"]: bool(item["enabled"])
                for item in conn.execute(
                    "SELECT group_id, enabled FROM mcp_tool_group WHERE server_id = ?",
                    (server_id,),
                )
            }
            conn.execute("DELETE FROM mcp_tool WHERE server_id = ?", (server_id,))
            conn.execute("DELETE FROM mcp_tool_group WHERE server_id = ?", (server_id,))
            seen: dict[str, str] = {}
            for tool in tools:
                group_id, group_name = tool["group_id"], tool["group_name"]
                seen[group_id] = group_name
                conn.execute(
                    """
                    INSERT INTO mcp_tool (server_id, name, group_id, description, schema)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        server_id,
                        tool["name"],
                        group_id,
                        tool["description"],
                        json.dumps(tool["schema"]),
                    ),
                )
            for group_id, group_name in seen.items():
                conn.execute(
                    """
                    INSERT INTO mcp_tool_group (server_id, group_id, name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (server_id, group_id, group_name, int(prior.get(group_id, True))),
                )
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
            return _mcp_server(conn, updated)

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
            return _mcp_server(conn, updated)

        return self._write(write)

    def enabled_mcp_tools(self, agent_id: str) -> list[dict[str, Any]]:
        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT s.id AS server_id, s.name AS server_name, s.transport, s.command, s.url,
                       t.name, t.description, t.schema, g.group_id
                FROM mcp_server s
                JOIN mcp_tool t ON t.server_id = s.id
                JOIN mcp_tool_group g ON g.server_id = s.id AND g.group_id = t.group_id
                WHERE s.agent_id = ? AND s.enabled = 1 AND g.enabled = 1
                ORDER BY s.name, t.name
                """,
                (agent_id,),
            ).fetchall()
            found = []
            for row in rows:
                found.append(
                    {
                        "server_id": row["server_id"],
                        "server_name": row["server_name"],
                        "transport": row["transport"],
                        "command": row["command"],
                        "url": row["url"],
                        "name": row["name"],
                        "description": row["description"],
                        "schema": json.loads(row["schema"]),
                        "group_id": row["group_id"],
                    }
                )
            return found

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

    def deliver_mcp_result(
        self, task_pk: str, status: str, text: str
    ) -> list[dict[str, Any]]:
        def write(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            row = conn.execute("SELECT * FROM mcp_task WHERE id = ?", (task_pk,)).fetchone()
            if row is None or row["status"] != "working":
                return []
            conn.execute(
                "UPDATE mcp_task SET status = ? WHERE id = ?",
                (status, task_pk),
            )
            session = conn.execute(
                "SELECT * FROM session WHERE id = ?",
                (row["session_id"],),
            ).fetchone()
            if session is None:
                return []
            message_id = mint("message")
            created = now()
            conn.execute(
                """
                INSERT INTO message (id, session_id, role, body, actor_kind, actor_id, created_at)
                VALUES (?, ?, 'assistant', ?, 'agent', ?, ?)
                """,
                (message_id, row["session_id"], text, row["agent_id"], created),
            )
            return [
                self._event(
                    conn,
                    stream_id=f"session:{row['session_id']}",
                    event_type="message.created",
                    payload={"message_id": message_id, "role": "assistant", "text": text},
                    channel_id=session["channel_id"],
                    session_id=row["session_id"],
                    actor={"kind": "agent", "id": row["agent_id"]},
                )
            ]

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
        self, key: str | None, session_id: str, text: str
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        body = {"session_id": session_id, "text": text}

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
                INSERT INTO run (id, session_id, agent_id, status, error, started_at, finished_at)
                VALUES (?, ?, ?, 'running', NULL, ?, NULL)
                """,
                (run_id, session_id, session["participant_id"], created),
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
                {"role": row["role"], "text": row["body"]}
                for row in conn.execute(
                    """
                    SELECT role, body FROM message
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
            user_text = history[-1]["text"] if history else ""
            prior = history[:-1]
            return {
                "run": dict(run),
                "session": dict(session),
                "agent": agent,
                "model": model,
                "base_url": base_url,
                "prior": prior,
                "user_text": user_text,
            }

        return self._read(read)

    def complete_run(self, run_id: str, text: str) -> list[dict[str, Any]]:
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
            conn.execute(
                """
                INSERT INTO message (id, session_id, role, body, actor_kind, actor_id, created_at)
                VALUES (?, ?, 'assistant', ?, 'agent', ?, ?)
                """,
                (message_id, run["session_id"], text, run["agent_id"], created),
            )
            conn.execute(
                "UPDATE run SET status = 'completed', finished_at = ? WHERE id = ?",
                (created, run_id),
            )
            stream_id = f"session:{run['session_id']}"
            actor = {"kind": "agent", "id": run["agent_id"]}
            return [
                self._event(
                    conn,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload={"message_id": message_id, "role": "assistant", "text": text},
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
