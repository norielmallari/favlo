"""SQLite file for the web API. One connection, one lock."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from favlo.config_tools import GROUP_ID
from favlo.persona import PERSONA
from web_api.events import now
from web_api.session_tools import GROUP_ID as SESSION_GROUP_ID

SCHEMA = """
CREATE TABLE IF NOT EXISTS provider (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    base_url TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_model (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    UNIQUE (provider_id, model_id)
);
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    default_model_id TEXT,
    exchange_budget INTEGER NOT NULL,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agent (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    persona TEXT NOT NULL,
    default_model_id TEXT,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS channel (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS channel_member (
    channel_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    PRIMARY KEY (channel_id, agent_id)
);
CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    channel_id TEXT,
    participant_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    body TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    actor_id TEXT,
    created_at TEXT NOT NULL,
    reasoning TEXT
);
CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_id TEXT,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    cue TEXT
);
CREATE TABLE IF NOT EXISTS event (
    id TEXT PRIMARY KEY,
    stream_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (stream_id, sequence)
);
CREATE TABLE IF NOT EXISTS command (
    key TEXT PRIMARY KEY,
    body_hash TEXT NOT NULL,
    status INTEGER NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_tool_group (
    agent_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    PRIMARY KEY (agent_id, group_id)
);
CREATE TABLE IF NOT EXISTS mcp_server (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,
    command TEXT NOT NULL,
    url TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    status TEXT NOT NULL,
    status_message TEXT NOT NULL,
    revision INTEGER NOT NULL,
    group_switches TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS mcp_task (
    id TEXT PRIMARY KEY,
    remote_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    poll_interval_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS heartbeat (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    timezone TEXT NOT NULL,
    instructions TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    session_id TEXT,
    fired_minute TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workspace (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS folder (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    provider TEXT,
    location TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    remote_id TEXT,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skill (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, name)
);
CREATE TABLE IF NOT EXISTS agent_skill (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (agent_id, name)
);
"""

LOCAL_PROVIDER_ID = "provider_local"
LOCAL_BASE_URL = "http://127.0.0.1:8129/v1"
FAVLO_AGENT_ID = "agent_favlo"
FAVLO_CHANNEL_ID = "channel_favlo"
FAVLO_SESSION_ID = "session_favlo"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "secrets").mkdir(exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _drop_mcp_definitions(conn: sqlite3.Connection) -> None:
    """Drop catalog tables. Group on/off switches stay on the server row."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(mcp_server)")}
    if "group_switches" not in columns:
        conn.execute(
            "ALTER TABLE mcp_server ADD COLUMN group_switches TEXT NOT NULL DEFAULT '{}'"
        )
    present = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'mcp_tool_group'"
    ).fetchone()
    if present is not None:
        grouped: dict[str, dict[str, bool]] = {}
        query = "SELECT server_id, group_id, enabled FROM mcp_tool_group"
        for row in conn.execute(query):
            server_id = str(row["server_id"])
            grouped.setdefault(server_id, {})[str(row["group_id"])] = bool(row["enabled"])
        for server_id, switches in grouped.items():
            current = conn.execute(
                "SELECT group_switches FROM mcp_server WHERE id = ?",
                (server_id,),
            ).fetchone()
            if current is None:
                continue
            merged: dict[str, bool] = {}
            raw = current["group_switches"]
            if isinstance(raw, str) and raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    merged = {str(key): bool(value) for key, value in parsed.items()}
            merged.update(switches)
            conn.execute(
                "UPDATE mcp_server SET group_switches = ? WHERE id = ?",
                (json.dumps(merged), server_id),
            )
        conn.execute("DROP TABLE mcp_tool_group")
    conn.execute("DROP TABLE IF EXISTS mcp_tool")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _drop_mcp_definitions(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(heartbeat)")}
    if "session_id" not in columns:
        conn.execute("ALTER TABLE heartbeat ADD COLUMN session_id TEXT")
    if "fired_minute" not in columns:
        conn.execute("ALTER TABLE heartbeat ADD COLUMN fired_minute TEXT")
    message_columns = {row[1] for row in conn.execute("PRAGMA table_info(message)")}
    if "reasoning" not in message_columns:
        conn.execute("ALTER TABLE message ADD COLUMN reasoning TEXT")
    run_columns = {row[1] for row in conn.execute("PRAGMA table_info(run)")}
    if "cue" not in run_columns:
        conn.execute("ALTER TABLE run ADD COLUMN cue TEXT")
    session_columns = {row[1] for row in conn.execute("PRAGMA table_info(session)")}
    if "workspace_id" not in session_columns:
        conn.execute("ALTER TABLE session ADD COLUMN workspace_id TEXT")
    row = conn.execute("SELECT id FROM settings WHERE id = 1").fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO settings (id, default_model_id, exchange_budget, revision)
            VALUES (1, NULL, 10, 1)
            """
        )
    provider = conn.execute("SELECT id FROM provider WHERE id = ?", (LOCAL_PROVIDER_ID,)).fetchone()
    if provider is None:
        conn.execute(
            """
            INSERT INTO provider (id, name, type, base_url, enabled, revision)
            VALUES (?, 'Favlo', 'openai_compatible', ?, 1, 1)
            """,
            (LOCAL_PROVIDER_ID, LOCAL_BASE_URL),
        )
    _seed_favlo(conn)
    _seed_config_group(conn)
    _seed_session_group(conn)
    conn.commit()


def _seed_favlo(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT id FROM agent WHERE id = ?", (FAVLO_AGENT_ID,)).fetchone()
    if row is not None:
        return
    conn.execute(
        """
        INSERT INTO agent (id, name, persona, default_model_id, revision)
        VALUES (?, 'Favlo', ?, NULL, 1)
        """,
        (FAVLO_AGENT_ID, PERSONA),
    )
    conn.execute(
        "INSERT INTO channel (id, kind, name, revision) VALUES (?, 'agent', 'Favlo', 1)",
        (FAVLO_CHANNEL_ID,),
    )
    conn.execute(
        "INSERT INTO channel_member (channel_id, agent_id) VALUES (?, ?)",
        (FAVLO_CHANNEL_ID, FAVLO_AGENT_ID),
    )
    conn.execute(
        """
        INSERT INTO session (id, kind, channel_id, participant_id, created_at)
        VALUES (?, 'channel', ?, ?, ?)
        """,
        (FAVLO_SESSION_ID, FAVLO_CHANNEL_ID, FAVLO_AGENT_ID, now()),
    )


def _seed_config_group(conn: sqlite3.Connection) -> None:
    """Enable the configuration group for Favlo once. A later choice stays."""
    marked = conn.execute("SELECT key FROM meta WHERE key = 'favlo_config_seed'").fetchone()
    if marked is not None:
        return
    exists = conn.execute("SELECT id FROM agent WHERE id = ?", (FAVLO_AGENT_ID,)).fetchone()
    if exists is not None:
        conn.execute(
            "INSERT OR IGNORE INTO agent_tool_group (agent_id, group_id) VALUES (?, ?)",
            (FAVLO_AGENT_ID, GROUP_ID),
        )
    conn.execute("INSERT INTO meta (key, value) VALUES ('favlo_config_seed', '1')")


def _seed_session_group(conn: sqlite3.Connection) -> None:
    """Enable the session group for Favlo once. A later choice stays."""
    marked = conn.execute("SELECT key FROM meta WHERE key = 'favlo_session_seed'").fetchone()
    if marked is not None:
        return
    exists = conn.execute("SELECT id FROM agent WHERE id = ?", (FAVLO_AGENT_ID,)).fetchone()
    if exists is not None:
        conn.execute(
            "INSERT OR IGNORE INTO agent_tool_group (agent_id, group_id) VALUES (?, ?)",
            (FAVLO_AGENT_ID, SESSION_GROUP_ID),
        )
    conn.execute("INSERT INTO meta (key, value) VALUES ('favlo_session_seed', '1')")
