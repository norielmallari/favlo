"""Domain records. Each method commits one transaction."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, literal_column, or_, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from favlo.config_tools import GROUP_ID
from web_api.conversation import GROUP_ID as CONVERSATION_GROUP_ID
from web_api.cron import ScheduleError, check_cron, check_timezone, due
from web_api.events import envelope, now
from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_GROUP_ID
from web_api.ids import FAVLO_AGENT_ID, LOCAL_PROVIDER_ID, guid
from web_api.infrastructure.db.database import open_session
from web_api.infrastructure.db.defaults import copy_default_mcp, copy_default_skills
from web_api.infrastructure.db.models import (
    AgentRow,
    AgentSkillRow,
    AgentToolGroupRow,
    ChannelMemberRow,
    ChannelRow,
    CommandRow,
    EventRow,
    FolderRow,
    GlobalMcpServerRow,
    GlobalSkillRow,
    HeartbeatRow,
    McpServerRow,
    McpTaskRow,
    MessageRow,
    MetaRow,
    CatalogElementRow,
    ProviderModelRow,
    ProviderRow,
    RunRow,
    SessionRow,
    SettingsRow,
    ShellElementRow,
    ShellRow,
    SkillRow,
    ThreadSnapshotRow,
    WorkspaceRow,
)
from web_api.infrastructure.db.row import as_dict
from web_api.mcp_groups import GROUP_NAMES, ORDER, default_switch_payload
from web_api.session_commands import ARCHIVE_COMMAND, ARCHIVE_CUE, CLEAR_COMMAND

# SQLite insertion order. rowid is not a mapped column.
_ROW_ID = literal_column("rowid")
_MESSAGE_ROW_ID = literal_column("message.rowid")


class Conflict(Exception):
    pass


class NotFound(Exception):
    pass


class Rejected(Exception):
    pass


_KNOWN_GROUPS = {GROUP_ID, CONVERSATION_GROUP_ID, HEARTBEAT_GROUP_ID}


def _stored_flag(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _flag_column(value: Any) -> int | None:
    flag = _stored_flag(value)
    if flag is None:
        return None
    return int(flag)


def _stored_effort(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _effective_effort(db: Session, agent: dict[str, Any] | None) -> str | None:
    """Agent effort, else the Providers effort. None leaves the field off the request."""
    if agent is not None:
        effort = _stored_effort(agent.get("reasoning_effort"))
        if effort:
            return effort
    settings = db.get(SettingsRow, 1)
    if settings is None:
        return None
    return _stored_effort(settings.reasoning_effort)


def _stored_expand(value: Any, *, default: str) -> str | None:
    """Store ``hidden``, ``collapsed``, ``running``, or ``always``. The default is null."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned not in {"hidden", "collapsed", "running", "always"} or cleaned == default:
        return None
    return cleaned


def _view(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return as_dict(row)


def _agent_record(db: Session, row: Any) -> dict[str, Any]:
    agent = _view(row)
    agent["enable_thinking"] = _stored_flag(agent.get("enable_thinking"))
    agent["preserve_thinking"] = _stored_flag(agent.get("preserve_thinking"))
    agent["reasoning_effort"] = _stored_effort(agent.get("reasoning_effort"))
    agent["expand_thinking"] = _stored_expand(agent.get("expand_thinking"), default="collapsed")
    agent["expand_tools"] = _stored_expand(agent.get("expand_tools"), default="hidden")
    groups = db.scalars(
        select(AgentToolGroupRow.group_id)
        .where(AgentToolGroupRow.agent_id == agent["id"])
        .order_by(AgentToolGroupRow.group_id)
    )
    agent["tool_groups"] = list(groups)
    return agent


def _resolve_provider_model(db: Session, model_row_id: str | None) -> tuple[str | None, str | None]:
    """Listener model id and base URL for a provider_model row id."""
    if not model_row_id:
        return None, None
    found = db.execute(
        select(
            ProviderModelRow.model_id,
            ProviderModelRow.enabled,
            ProviderRow.base_url,
            ProviderRow.enabled,
        )
        .join(ProviderRow, ProviderRow.id == ProviderModelRow.provider_id)
        .where(ProviderModelRow.id == model_row_id)
    ).one_or_none()
    if found is None or not found[1] or not found[3]:
        return None, None
    return str(found[0]), str(found[2])


def _provider_model(db: Session, agent_id: str | None) -> tuple[str, str]:
    """Listener model id and base URL.

    The agent's saved provider model, or the settings default when the agent
    has none. A missing or disabled choice is invalid.
    """
    model_row_id = None
    if agent_id:
        agent = db.get(AgentRow, agent_id)
        if agent is not None and agent.default_model_id:
            model_row_id = agent.default_model_id
    if not model_row_id:
        settings = db.get(SettingsRow, 1)
        model_row_id = None if settings is None else settings.default_model_id
    if not model_row_id:
        raise Rejected("Choose an enabled model in configuration.")
    model, base = _resolve_provider_model(db, model_row_id)
    if not model or not base:
        raise Rejected("Choose an enabled model in configuration.")
    return model, base


def _chat_model(db: Session, agent_id: str) -> tuple[str, str]:
    """Listener model id and base URL for an agent."""
    return _provider_model(db, agent_id)


def _workspace_agent_id(db: Session, workspace_id: str | None) -> str | None:
    if not workspace_id:
        return None
    workspace = db.get(WorkspaceRow, workspace_id)
    if workspace is None:
        return None
    chat = db.get(SessionRow, workspace.session_id)
    if chat is None:
        return None
    return chat.participant_id


def _gallery_vision_model(db: Session, workspace_id: str | None = None) -> tuple[str, str]:
    """The workspace agent's model, else the settings default. Thinking stays off on the request."""
    return _provider_model(db, _workspace_agent_id(db, workspace_id))


def _heartbeat_record(row: Any) -> dict[str, Any]:
    item = _view(row)
    item["enabled"] = bool(item["enabled"])
    item.pop("fired_minute", None)
    return item


def _kind_clause(kind: str) -> Any:
    if kind == "all":
        return SessionRow.kind.in_(("channel", "workspace_chat"))
    if kind == "agent":
        return and_(SessionRow.kind == "channel", ChannelRow.kind == "agent")
    if kind == "channel":
        return and_(
            SessionRow.kind == "channel",
            func.coalesce(ChannelRow.kind, "") != "agent",
        )
    if kind == "workspace_chat":
        return SessionRow.kind == "workspace_chat"
    raise KeyError(kind)


def _like(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped.lower()}%"


def _last_message_at() -> Any:
    return (
        select(MessageRow.created_at)
        .where(MessageRow.session_id == SessionRow.id)
        .order_by(_ROW_ID.desc())
        .limit(1)
        .scalar_subquery()
    )


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
    data = _view(row)
    switches = _group_switches(data["group_switches"])
    rank = {group_id: index for index, group_id in enumerate(ORDER)}
    ordered = sorted(switches, key=lambda group_id: rank.get(group_id, len(rank)))
    return {
        "id": data["id"],
        "agent_id": data["agent_id"],
        "name": data["name"],
        "transport": data["transport"],
        "command": data["command"],
        "url": data["url"],
        "enabled": bool(data["enabled"]),
        "status": data["status"],
        "status_message": data["status_message"],
        "revision": data["revision"],
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


def _global_mcp_server(row: Any) -> dict[str, Any]:
    data = _view(row)
    switches = _group_switches(data["group_switches"])
    rank = {group_id: index for index, group_id in enumerate(ORDER)}
    ordered = sorted(switches, key=lambda group_id: rank.get(group_id, len(rank)))
    return {
        "id": data["id"],
        "name": data["name"],
        "transport": data["transport"],
        "command": data["command"],
        "url": data["url"],
        "enabled": bool(data["enabled"]),
        "add_by_default": bool(data["add_by_default"]),
        "revision": data["revision"],
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


def _switch_payload(groups: list[Any], prior: str = "") -> str:
    switches = _group_switches(prior)
    for item in groups:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            continue
        switches[item["id"]] = bool(item.get("enabled"))
    return json.dumps(switches)


def _save_switches(db: Session, server_id: str, groups: list[Any]) -> None:
    """Store which groups are on. Names and schemas stay in the catalog cache."""
    row = db.get(McpServerRow, server_id)
    if row is not None:
        row.group_switches = _switch_payload(groups, row.group_switches)


def _schedule(expression: str, timezone: str) -> tuple[str, str]:
    try:
        return check_cron(expression), check_timezone(timezone)
    except ScheduleError as exc:
        raise Rejected(str(exc)) from exc


def _replace_groups(db: Session, agent_id: str, groups: list[str]) -> None:
    if any(item not in _KNOWN_GROUPS for item in groups):
        raise Rejected("That tool group is not available.")
    _drop(db, AgentToolGroupRow, AgentToolGroupRow.agent_id == agent_id)
    db.flush()
    for group_id in dict.fromkeys(groups):
        db.add(AgentToolGroupRow(agent_id=agent_id, group_id=group_id))


def _owned_sessions(db: Session, agent_id: str) -> list[str]:
    """Sessions whose participant is this agent, plus its sole agent channel."""
    found = list(db.scalars(select(SessionRow.id).where(SessionRow.participant_id == agent_id)))
    channels = db.scalars(
        select(ChannelRow.id)
        .join(ChannelMemberRow, ChannelMemberRow.channel_id == ChannelRow.id)
        .where(ChannelMemberRow.agent_id == agent_id, ChannelRow.kind == "agent")
    )
    for channel_id in channels:
        others = db.scalars(
            select(ChannelMemberRow.agent_id).where(
                ChannelMemberRow.channel_id == channel_id,
                ChannelMemberRow.agent_id != agent_id,
            )
        ).first()
        if others is not None:
            continue
        for session_id in db.scalars(
            select(SessionRow.id).where(SessionRow.channel_id == channel_id)
        ):
            if session_id not in found:
                found.append(session_id)
    return found


def _drop(db: Session, model: Any, *criteria: Any) -> None:
    for row in list(db.scalars(select(model).where(*criteria))):
        db.delete(row)


def _drop_ids(db: Session, model: Any, column: Any, ids: list[str]) -> None:
    if ids:
        _drop(db, model, column.in_(ids))


def _folder_payload(row: Any) -> dict[str, Any]:
    data = _view(row)
    kind = data["kind"]
    parent = data["parent_id"]
    return {
        "id": data["id"],
        "workspace_id": data["workspace_id"],
        "parent_id": parent or None,
        "name": data["name"],
        "kind": kind,
        "provider": data["provider"] or None,
        "location": data["location"] if kind == "local" else None,
        "machine_id": data["machine_id"] or "",
        "remote_id": data["remote_id"],
        "revision": data["revision"],
    }


def _folder_name(value: object) -> str:
    name = " ".join(str(value or "").split())
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise Rejected("A name is required.")
    return name


def _folder_in(db: Session, workspace_id: str, folder_id: str) -> FolderRow:
    row = db.get(FolderRow, folder_id)
    if row is None or row.workspace_id != workspace_id:
        raise NotFound("That folder does not exist.")
    return row


def _reject_sibling(
    db: Session,
    workspace_id: str,
    parent_id: str | None,
    name: str,
    folder_id: str | None,
) -> None:
    parent = FolderRow.parent_id == parent_id if parent_id else FolderRow.parent_id.is_(None)
    rows = db.scalars(
        select(FolderRow).where(
            FolderRow.workspace_id == workspace_id,
            parent,
            FolderRow.name.collate("NOCASE") == name,
        )
    )
    if any(row.id != folder_id for row in rows):
        raise Rejected("That folder name is already here.")


def _folder_fields(db: Session, workspace_id: str, body: dict[str, Any]) -> dict[str, Any]:
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
        parent = _folder_in(db, workspace_id, parent_id)
        if parent.kind != "cloud":
            raise Rejected("That would download.")
    return {
        "name": name,
        "kind": "cloud",
        "provider": provider,
        "location": None,
        "remote_id": remote_id,
        "parent_id": parent_id,
    }


def _move_parent(db: Session, workspace_id: str, row: FolderRow, parent: object) -> str | None:
    parent_id = str(parent or "").strip() or None
    if row.kind == "local":
        if parent_id:
            raise Rejected("That would upload.")
        return None
    if parent_id == row.id:
        raise Rejected("A folder cannot be moved into itself.")
    if parent_id:
        parent_row = _folder_in(db, workspace_id, parent_id)
        if parent_row.kind != "cloud":
            raise Rejected("That would download.")
        seen = {str(row.id)}
        current: str | None = parent_id
        while current:
            if current in seen:
                raise Rejected("A folder cannot be moved into itself.")
            seen.add(current)
            nxt = db.execute(
                select(FolderRow.parent_id).where(
                    FolderRow.id == current,
                    FolderRow.workspace_id == workspace_id,
                )
            ).one_or_none()
            if nxt is None:
                raise Rejected("That folder is not in this workspace.")
            current = nxt[0] or None
    return parent_id


def _mark_meta(db: Session, key: str) -> None:
    if db.get(MetaRow, key) is None:
        db.add(MetaRow(key=key, value="1"))


def _seed_workspace_skills(db: Session, workspace_id: str, kind: str, created: str) -> None:
    from web_api.workspace_skill_catalog import skills_for

    for name, description, definition in skills_for(kind):
        db.add(
            SkillRow(
                id=guid(),
                workspace_id=workspace_id,
                name=name,
                description=description,
                definition=definition,
                revision=1,
                created_at=created,
            )
        )
        _mark_meta(db, f"workspace_skill_seed:{workspace_id}:{name}")


def _skill_fields(body: dict[str, Any]) -> tuple[str, str, str]:
    name = " ".join(str(body.get("name") or "").split())
    description = str(body.get("description") or "").strip()
    definition = str(body.get("definition") or "").strip()
    if not name:
        raise Rejected("A name is required.")
    return name, description, definition


def _skill_payload(row: Any) -> dict[str, Any]:
    data = _view(row)
    return {
        "id": data["id"],
        "workspace_id": data["workspace_id"],
        "name": data["name"],
        "description": data["description"],
        "definition": data["definition"] or "",
        "revision": data["revision"],
    }


def _agent_skill_payload(row: Any) -> dict[str, Any]:
    data = _view(row)
    return {
        "id": data["id"],
        "agent_id": data["agent_id"],
        "name": data["name"],
        "description": data["description"],
        "definition": data["definition"] or "",
        "revision": data["revision"],
    }


def _global_skill_payload(row: Any) -> dict[str, Any]:
    data = _view(row)
    return {
        "id": data["id"],
        "name": data["name"],
        "description": data["description"],
        "definition": data["definition"] or "",
        "add_by_default": bool(data["add_by_default"]),
        "revision": data["revision"],
    }


def _agent_skills_for(db: Session, agent_id: str | None = None) -> list[dict[str, Any]]:
    stmt = select(AgentSkillRow).order_by(AgentSkillRow.name.collate("NOCASE"), AgentSkillRow.id)
    if agent_id:
        stmt = stmt.where(AgentSkillRow.agent_id == agent_id)
    return [_agent_skill_payload(row) for row in db.scalars(stmt)]


def _skills_for(db: Session, workspace_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(SkillRow)
        .where(SkillRow.workspace_id == workspace_id)
        .order_by(SkillRow.name.collate("NOCASE"), SkillRow.id)
    )
    return [_skill_payload(row) for row in rows]


def _shell_payload(db: Session, row: ShellRow) -> dict[str, Any]:
    members = db.scalars(
        select(ShellElementRow)
        .where(ShellElementRow.shell_id == row.id)
        .order_by(ShellElementRow.sort, ShellElementRow.element_name)
    )
    elements: list[dict[str, Any]] = []
    for member in members:
        element = db.get(CatalogElementRow, member.element_name)
        elements.append(
            {
                "name": member.element_name,
                "slot": member.slot,
                "initial_props": json.loads(member.initial_props),
                "version": element.version if element is not None else 1,
                "max_count": element.max_count if element is not None else 1,
                "gestures": json.loads(element.gestures) if element is not None else [],
            }
        )
    return {
        "id": row.id,
        "label": row.label,
        "icon": row.icon,
        "setup": row.setup,
        "setup_hint": row.setup_hint,
        "slots": json.loads(row.slots),
        "renderer": row.renderer,
        "version": row.version,
        "elements": elements,
    }


def _workspace_payload(db: Session, row: Any) -> dict[str, Any]:
    data = _view(row)
    expand_thinking = None
    expand_tools = None
    chat = db.get(SessionRow, data["session_id"])
    if chat is not None and chat.participant_id:
        agent = db.get(AgentRow, chat.participant_id)
        if agent is not None:
            expand_thinking = _stored_expand(agent.expand_thinking, default="collapsed")
            expand_tools = _stored_expand(agent.expand_tools, default="hidden")
    shell = db.get(ShellRow, data["type"])
    return {
        "id": data["id"],
        "channel_id": data["channel_id"],
        "session_id": data["session_id"],
        "name": data["name"],
        "type": data["type"],
        "revision": data["revision"],
        "skills": _skills_for(db, data["id"]),
        "expand_thinking": expand_thinking,
        "expand_tools": expand_tools,
        "theme": _workspace_theme(db, data["id"], data["type"]),
        "icon": shell.icon if shell is not None else "box",
        "renderer": shell.renderer if shell is not None else "chat",
        "provisioning_state": data.get("provisioning_state") or "provisioned",
        "shell": _shell_payload(db, shell) if shell is not None else None,
    }


_WORKSPACE_THEMES = frozenset({"favlo", "rose-pine", "catppuccin"})


def _workspace_theme(db: Session, workspace_id: str, workspace_type: str) -> str:
    row = db.get(MetaRow, f"workspace_theme:{workspace_id}")
    if row is None:
        row = db.get(MetaRow, f"gallery_theme:{workspace_id}")
    default = "catppuccin" if workspace_type == "coding" else "rose-pine"
    if row is None or row.value not in _WORKSPACE_THEMES:
        return default
    return str(row.value)


def _require_workspace(db: Session, workspace_id: str) -> dict[str, Any]:
    row = db.get(WorkspaceRow, workspace_id)
    if row is None:
        raise NotFound("That workspace does not exist.")
    return _workspace_payload(db, row)


def _session_view(row: Any) -> dict[str, Any]:
    viewed = _view(row)
    viewed.pop("maf_state", None)
    return viewed


def _json_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [row for row in parsed if isinstance(row, dict)]


def _tool_calls(raw: Any) -> list[dict[str, Any]]:
    return _json_list(raw)


def _trace(raw: Any) -> list[dict[str, Any]]:
    return _json_list(raw)


def _hash(body: dict[str, Any]) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _next_sequence(db: Session, stream_id: str) -> int:
    current = db.scalar(select(func.max(EventRow.sequence)).where(EventRow.stream_id == stream_id))
    return int(current or 0) + 1


_MEDIA_DONE = frozenset({"completed", "failed", "cancelled"})


def _open_media_tasks(db: Session, session_id: str) -> list[dict[str, str]]:
    """In-flight generate tasks, oldest first, from this session's event log."""
    stream_id = f"session:{session_id}"
    rows = db.scalars(
        select(EventRow)
        .where(
            EventRow.stream_id == stream_id,
            EventRow.type.in_(("media.pending", "message.created")),
        )
        .order_by(EventRow.sequence)
    )
    order: list[str] = []
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            body = json.loads(row.body)
        except json.JSONDecodeError:
            continue
        payload = body.get("payload") if isinstance(body, dict) else None
        if not isinstance(payload, dict):
            continue
        if row.type == "message.created":
            finished = payload.get("task_ids")
            if isinstance(finished, list):
                for item in finished:
                    by_id.pop(str(item or "").strip(), None)
            continue
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            continue
        status = str(payload.get("status") or "working").strip() or "working"
        if status in _MEDIA_DONE:
            by_id.pop(task_id, None)
            continue
        if task_id not in by_id:
            order.append(task_id)
        by_id[task_id] = {
            "task_id": task_id,
            "kind": str(payload.get("kind") or "image").strip() or "image",
            "status": status,
        }
    return [by_id[task_id] for task_id in order if task_id in by_id]


class Store:
    def __init__(self, path: Any) -> None:
        self._lock = threading.Lock()
        self.path = Path(path) if not isinstance(path, Path) else path
        self._session = open_session(self.path)
        bind = self._session.get_bind()
        self._engine = bind if isinstance(bind, Engine) else bind.engine

    def close(self) -> None:
        with self._lock:
            self._session.close()
            self._engine.dispose()

    def _write(self, fn: Any) -> Any:
        with self._lock:
            try:
                self._session.expire_all()
                result = fn(self._session)
                self._session.commit()
                return result
            except Exception:
                self._session.rollback()
                raise

    def _read(self, fn: Any) -> Any:
        with self._lock:
            try:
                self._session.expire_all()
                return fn(self._session)
            finally:
                self._session.rollback()

    def _event(
        self,
        db: Session,
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
        sequence = _next_sequence(db, stream_id)
        event_id = guid()
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
        db.add(
            EventRow(
                id=event_id,
                stream_id=stream_id,
                sequence=sequence,
                type=event_type,
                body=json.dumps(body),
            )
        )
        return body

    def _claim(
        self, db: Session, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]] | None:
        if not key:
            raise Rejected("An Idempotency-Key header is required.")
        digest = _hash(body)
        row = db.get(CommandRow, key)
        if row is None:
            return None
        if row.body_hash != digest:
            raise Conflict("This idempotency key was used with a different body.")
        return int(row.status), json.loads(row.result)

    def _remember(
        self,
        db: Session,
        key: str,
        body: dict[str, Any],
        status: int,
        result: dict[str, Any],
    ) -> None:
        db.add(
            CommandRow(
                key=key,
                body_hash=_hash(body),
                status=status,
                result=json.dumps(result),
            )
        )

    def providers(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            providers = [
                _view(row) for row in db.scalars(select(ProviderRow).order_by(ProviderRow.name))
            ]
            models = [
                _view(row)
                for row in db.scalars(
                    select(ProviderModelRow).order_by(ProviderModelRow.display_name)
                )
            ]
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
        def read(db: Session) -> str:
            row = db.get(ProviderRow, provider_id)
            if row is None or not row.enabled:
                raise NotFound("That provider does not exist.")
            return str(row.base_url)

        return self._read(read)

    def sync_models(self, provider_id: str, model_ids: list[str]) -> list[dict[str, Any]]:
        def write(db: Session) -> list[dict[str, Any]]:
            provider = db.get(ProviderRow, provider_id)
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
                existing = db.scalars(
                    select(ProviderModelRow).where(
                        ProviderModelRow.provider_id == provider_id,
                        ProviderModelRow.model_id == item,
                    )
                ).first()
                if existing is None:
                    db.add(
                        ProviderModelRow(
                            id=guid(),
                            provider_id=provider_id,
                            model_id=item,
                            display_name=item,
                            enabled=1,
                            revision=1,
                        )
                    )
                else:
                    existing.enabled = 1
            if cleaned:
                db.execute(
                    update(ProviderModelRow)
                    .where(
                        ProviderModelRow.provider_id == provider_id,
                        ProviderModelRow.model_id.not_in(cleaned),
                    )
                    .values(enabled=0)
                )
            else:
                db.execute(
                    update(ProviderModelRow)
                    .where(ProviderModelRow.provider_id == provider_id)
                    .values(enabled=0)
                )
            rows = db.scalars(
                select(ProviderModelRow)
                .where(
                    ProviderModelRow.provider_id == provider_id,
                    ProviderModelRow.enabled == 1,
                )
                .order_by(ProviderModelRow.display_name)
                .execution_options(populate_existing=True)
            )
            return [{**_view(row), "enabled": True} for row in rows]

        return self._write(write)

    def add_model(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            provider = db.get(ProviderRow, body["provider_id"])
            if provider is None:
                raise NotFound("That provider does not exist.")
            model_id = body["model_id"].strip()
            display = body["display_name"].strip() or model_id
            if not model_id:
                raise Rejected("A model id is required.")
            existing = db.scalars(
                select(ProviderModelRow).where(
                    ProviderModelRow.provider_id == body["provider_id"],
                    ProviderModelRow.model_id == model_id,
                )
            ).first()
            if existing is not None:
                raise Conflict("That model is already listed.")
            record = {
                "id": guid(),
                "provider_id": body["provider_id"],
                "model_id": model_id,
                "display_name": display,
                "enabled": True,
                "revision": 1,
            }
            db.add(
                ProviderModelRow(
                    id=record["id"],
                    provider_id=record["provider_id"],
                    model_id=model_id,
                    display_name=display,
                    enabled=1,
                    revision=1,
                )
            )
            self._remember(db, key or "", body, 201, record)
            return 201, record, []

        return self._write(write)

    def update_model(self, model_row: str, expected: int, enabled: bool) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(ProviderModelRow, model_row)
            if row is None:
                raise NotFound("That model does not exist.")
            if int(row.revision) != expected:
                raise Conflict("The model was updated. Reload and try again.")
            revision = expected + 1
            row.enabled = 1 if enabled else 0
            row.revision = revision
            updated = _view(row)
            updated["enabled"] = enabled
            updated["revision"] = revision
            return updated

        return self._write(write)

    def ensure_initial_model(self, model_id: str) -> None:
        """Store the registry model as the application default when none is set."""

        def write(db: Session) -> None:
            settings = db.get(SettingsRow, 1)
            if settings.default_model_id:
                return
            cleaned = model_id.strip()
            if not cleaned:
                return
            existing = db.scalars(
                select(ProviderModelRow).where(
                    ProviderModelRow.provider_id == LOCAL_PROVIDER_ID,
                    ProviderModelRow.model_id == cleaned,
                )
            ).first()
            if existing is None:
                record_id = guid()
                db.add(
                    ProviderModelRow(
                        id=record_id,
                        provider_id=LOCAL_PROVIDER_ID,
                        model_id=cleaned,
                        display_name=cleaned,
                        enabled=1,
                        revision=1,
                    )
                )
            else:
                record_id = str(existing.id)
                if not existing.enabled:
                    existing.enabled = 1
            settings.default_model_id = record_id
            settings.revision = settings.revision + 1

        self._write(write)

    def settings(self) -> dict[str, Any]:
        def read(db: Session) -> dict[str, Any]:
            row = db.get(SettingsRow, 1)
            return {
                "default_model_id": row.default_model_id,
                "gallery_vision_model_id": row.gallery_vision_model_id,
                "reasoning_effort": _stored_effort(row.reasoning_effort),
                "exchange_budget": row.exchange_budget,
                "revision": row.revision,
            }

        return self._read(read)

    def update_settings(
        self,
        expected: int,
        default_model_id: str | None,
        exchange_budget: int,
        gallery_vision_model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(SettingsRow, 1)
            if int(row.revision) != expected:
                raise Conflict("Settings were updated. Reload and try again.")
            if exchange_budget < 1:
                raise Rejected("The exchange budget is a positive integer.")
            if default_model_id is not None:
                model = db.get(ProviderModelRow, default_model_id)
                if model is None or not model.enabled:
                    raise Rejected("Choose an enabled model.")
            if gallery_vision_model_id is not None:
                vision = db.get(ProviderModelRow, gallery_vision_model_id)
                if vision is None or not vision.enabled:
                    raise Rejected("Choose an enabled gallery vision model.")
            revision = expected + 1
            row.default_model_id = default_model_id
            row.gallery_vision_model_id = gallery_vision_model_id
            row.reasoning_effort = _stored_effort(reasoning_effort)
            row.exchange_budget = exchange_budget
            row.revision = revision
            return {
                "default_model_id": default_model_id,
                "gallery_vision_model_id": gallery_vision_model_id,
                "reasoning_effort": _stored_effort(reasoning_effort),
                "exchange_budget": exchange_budget,
                "revision": revision,
            }

        return self._write(write)

    def use_model(self, provider_id: str, model_id: str, expected: int) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(SettingsRow, 1)
            if int(row.revision) != expected:
                raise Conflict("Settings were updated. Reload and try again.")
            cleaned = model_id.strip()
            if not cleaned:
                raise Rejected("Choose a model.")
            existing = db.scalars(
                select(ProviderModelRow).where(
                    ProviderModelRow.provider_id == provider_id,
                    ProviderModelRow.model_id == cleaned,
                )
            ).first()
            if existing is None or not existing.enabled:
                raise Rejected("That model is not available.")
            record_id = str(existing.id)
            revision = expected + 1
            row.default_model_id = record_id
            row.revision = revision
            return {
                "default_model_id": record_id,
                "gallery_vision_model_id": row.gallery_vision_model_id,
                "reasoning_effort": _stored_effort(row.reasoning_effort),
                "model_id": cleaned,
                "exchange_budget": int(row.exchange_budget),
                "revision": revision,
            }

        return self._write(write)

    def agents(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(select(AgentRow).order_by(AgentRow.name))
            return [_agent_record(db, row) for row in rows]

        return self._read(read)

    def create_agent(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            name = body["name"].strip()
            if not name:
                raise Rejected("A name is required.")
            model_id = body.get("default_model_id")
            if model_id is not None:
                self._require_model(db, model_id)
            agent_id = guid()
            channel_id = guid()
            session_id = guid()
            persona = body.get("persona") or ""
            db.add(
                AgentRow(
                    id=agent_id,
                    name=name,
                    persona=persona,
                    default_model_id=model_id,
                    enable_thinking=_flag_column(body.get("enable_thinking")),
                    preserve_thinking=_flag_column(body.get("preserve_thinking")),
                    reasoning_effort=_stored_effort(body.get("reasoning_effort")),
                    expand_thinking=_stored_expand(body.get("expand_thinking"), default="collapsed"),
                    expand_tools=_stored_expand(body.get("expand_tools"), default="hidden"),
                    revision=1,
                )
            )
            db.add(ChannelRow(id=channel_id, kind="agent", name=name, revision=1))
            db.add(ChannelMemberRow(channel_id=channel_id, agent_id=agent_id))
            created = now()
            db.add(
                SessionRow(
                    id=session_id,
                    kind="channel",
                    channel_id=channel_id,
                    participant_id=agent_id,
                    created_at=created,
                )
            )
            copy_default_skills(db, agent_id, created)
            copy_default_mcp(db, agent_id)
            groups = body.get("tool_groups")
            if groups is not None:
                _replace_groups(db, agent_id, groups)
            agent = _agent_record(db, db.get(AgentRow, agent_id))
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
                    db,
                    stream_id="app",
                    event_type="agent.created",
                    payload=result,
                    channel_id=channel_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
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
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        expand_thinking: str | None = None,
        expand_tools: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(AgentRow, agent_id)
            if row is None:
                raise NotFound("That agent does not exist.")
            if int(row.revision) != expected:
                raise Conflict("The agent was updated. Reload and try again.")
            cleaned = name.strip()
            if not cleaned:
                raise Rejected("A name is required.")
            if default_model_id is not None:
                self._require_model(db, default_model_id)
            revision = expected + 1
            row.name = cleaned
            row.persona = persona
            row.default_model_id = default_model_id
            row.enable_thinking = _flag_column(enable_thinking)
            row.preserve_thinking = _flag_column(preserve_thinking)
            row.reasoning_effort = _stored_effort(reasoning_effort)
            row.expand_thinking = _stored_expand(expand_thinking, default="collapsed")
            row.expand_tools = _stored_expand(expand_tools, default="hidden")
            row.revision = revision
            for channel_id in db.scalars(
                select(ChannelMemberRow.channel_id).where(ChannelMemberRow.agent_id == agent_id)
            ):
                channel = db.get(ChannelRow, channel_id)
                if channel is not None:
                    channel.name = cleaned
                    channel.revision = channel.revision + 1
            if tool_groups is not None:
                _replace_groups(db, agent_id, tool_groups)
            updated = _agent_record(db, db.get(AgentRow, agent_id))
            return updated, []

        return self._write(write)

    def delete_agent(self, agent_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Remove one agent and the rows that belong to it.

        Tasks name a server and a session, so they go first. Tool rows go
        before servers. Heartbeats name a session, so they go before sessions.
        Messages, runs, and events go before the sessions they sit on. Membership
        goes before an agent channel that would otherwise be empty. The agent
        row is last.
        """

        def write(
            db: Session,
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
            row = db.get(AgentRow, agent_id)
            if row is None:
                raise NotFound("That agent does not exist.")
            if agent_id == FAVLO_AGENT_ID:
                raise Rejected("Favlo stays.")
            name = row.name
            sessions = _owned_sessions(db, agent_id)
            running = list(
                db.scalars(
                    select(RunRow.id).where(RunRow.status == "running", RunRow.agent_id == agent_id)
                )
            )
            if sessions:
                for run_id in db.scalars(
                    select(RunRow.id).where(
                        RunRow.status == "running",
                        RunRow.session_id.in_(sessions),
                    )
                ):
                    if run_id not in running:
                        running.append(run_id)
            _drop(db, McpTaskRow, McpTaskRow.agent_id == agent_id)
            _drop(db, McpServerRow, McpServerRow.agent_id == agent_id)
            _drop(db, HeartbeatRow, HeartbeatRow.agent_id == agent_id)
            _drop(db, AgentSkillRow, AgentSkillRow.agent_id == agent_id)
            _drop_ids(db, MessageRow, MessageRow.session_id, sessions)
            _drop(
                db,
                MessageRow,
                MessageRow.actor_kind == "agent",
                MessageRow.actor_id == agent_id,
            )
            _drop_ids(db, RunRow, RunRow.session_id, sessions)
            _drop(db, RunRow, RunRow.agent_id == agent_id)
            _drop_ids(
                db,
                EventRow,
                EventRow.stream_id,
                [f"session:{session_id}" for session_id in sessions],
            )
            _drop_ids(db, SessionRow, SessionRow.id, sessions)
            workspaces = list(
                db.scalars(
                    select(WorkspaceRow.id)
                    .join(
                        ChannelMemberRow,
                        ChannelMemberRow.channel_id == WorkspaceRow.channel_id,
                    )
                    .where(ChannelMemberRow.agent_id == agent_id)
                )
            )
            _drop_ids(db, SkillRow, SkillRow.workspace_id, workspaces)
            _drop_ids(db, FolderRow, FolderRow.workspace_id, workspaces)
            _drop_ids(db, WorkspaceRow, WorkspaceRow.id, workspaces)
            channels = list(
                db.scalars(
                    select(ChannelMemberRow.channel_id).where(ChannelMemberRow.agent_id == agent_id)
                )
            )
            _drop(db, ChannelMemberRow, ChannelMemberRow.agent_id == agent_id)
            empty_channels = [
                channel_id
                for channel_id in channels
                if db.scalars(
                    select(ChannelMemberRow.channel_id).where(
                        ChannelMemberRow.channel_id == channel_id
                    )
                ).first()
                is None
            ]
            _drop_ids(db, ChannelRow, ChannelRow.id, empty_channels)
            _drop(db, AgentToolGroupRow, AgentToolGroupRow.agent_id == agent_id)
            db.delete(row)
            result = {"id": agent_id, "name": name}
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="agent.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events, running

        return self._write(write)

    def heartbeats(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(
                select(HeartbeatRow).order_by(HeartbeatRow.cron_expression, HeartbeatRow.id)
            )
            return [_heartbeat_record(row) for row in rows]

        return self._read(read)

    def create_heartbeat(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent = db.get(AgentRow, body["agent_id"])
            if agent is None:
                raise NotFound("That agent does not exist.")
            schedule = _schedule(body["cron_expression"], body["timezone"])
            heartbeat_id = guid()
            result = {
                "id": heartbeat_id,
                "agent_id": body["agent_id"],
                "cron_expression": schedule[0],
                "timezone": schedule[1],
                "instructions": body.get("instructions") or "",
                "enabled": bool(body.get("enabled", True)),
                "revision": 1,
            }
            db.add(
                HeartbeatRow(
                    id=result["id"],
                    agent_id=result["agent_id"],
                    cron_expression=result["cron_expression"],
                    timezone=result["timezone"],
                    instructions=result["instructions"],
                    enabled=int(result["enabled"]),
                    revision=1,
                )
            )
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="heartbeat.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_heartbeat(
        self, heartbeat_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(HeartbeatRow, heartbeat_id)
            if row is None:
                raise NotFound("That heartbeat does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The heartbeat was updated. Reload and try again.")
            schedule = _schedule(body["cron_expression"], body["timezone"])
            revision = expected + 1
            result = {
                "id": heartbeat_id,
                "agent_id": row.agent_id,
                "cron_expression": schedule[0],
                "timezone": schedule[1],
                "instructions": body.get("instructions") or "",
                "enabled": bool(body["enabled"]),
                "revision": revision,
            }
            row.cron_expression = result["cron_expression"]
            row.timezone = result["timezone"]
            row.instructions = result["instructions"]
            row.enabled = int(result["enabled"])
            row.revision = revision
            events = [
                self._event(
                    db,
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
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(HeartbeatRow, heartbeat_id)
            if row is None or row.agent_id != agent_id:
                raise NotFound("That heartbeat does not exist.")
            db.delete(row)
            result = {"id": heartbeat_id, "agent_id": agent_id}
            events = [
                self._event(
                    db,
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

        def read(db: Session) -> list[str]:
            return list(db.scalars(select(RunRow.id).where(RunRow.status == "running")))

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
            db: Session,
        ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
            rows = db.scalars(select(HeartbeatRow).where(HeartbeatRow.enabled == 1))
            started: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for row in rows:
                local = moment.astimezone(ZoneInfo(row.timezone))
                if not due(row.cron_expression, local):
                    continue
                slot = local.strftime("%Y-%m-%dT%H:%M%z")
                if row.fired_minute == slot:
                    continue
                session_id = self._ensure_heartbeat_session(db, row)
                active = db.scalars(
                    select(RunRow.id).where(
                        RunRow.session_id == session_id,
                        RunRow.status == "running",
                    )
                ).first()
                row.fired_minute = slot
                if active is not None:
                    continue
                started.append(self._start_heartbeat_run(db, row, session_id))
            return started

        return self._write(write)

    def begin_heartbeat_run(
        self, key: str | None, heartbeat_id: str
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        body = {"heartbeat_id": heartbeat_id, "action": "run"}

        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            row = db.get(HeartbeatRow, heartbeat_id)
            if row is None:
                raise NotFound("That heartbeat does not exist.")
            session_id = self._ensure_heartbeat_session(db, row)
            active = db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            ).first()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            result, events = self._start_heartbeat_run(db, row, session_id)
            self._remember(db, key or "", body, 202, result)
            return 202, result, events

        return self._write(write)

    def _ensure_heartbeat_session(self, db: Session, row: HeartbeatRow) -> str:
        session_id = row.session_id
        if session_id:
            return str(session_id)
        session_id = guid()
        db.add(
            SessionRow(
                id=session_id,
                kind="heartbeat",
                channel_id=None,
                participant_id=row.agent_id,
                created_at=now(),
            )
        )
        row.session_id = session_id
        return session_id

    def _start_heartbeat_run(
        self, db: Session, row: HeartbeatRow, session_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prompt = str(row.instructions or "").strip() or "Run this heartbeat."
        message_id = guid()
        run_id = guid()
        command_id = guid()
        created = now()
        actor = {"kind": "user", "id": "local"}
        # Each heartbeat run re-reads the owning agent's model (else Providers default).
        # Drop a prior freeze so stream_reply does not keep a stale listener id.
        chat = db.get(SessionRow, session_id)
        if chat is not None:
            chat.definition = None
            chat.definition_hash = None
        db.add(
            MessageRow(
                id=message_id,
                session_id=session_id,
                role="user",
                body=prompt,
                actor_kind="user",
                actor_id="local",
                created_at=created,
            )
        )
        db.add(
            RunRow(
                id=run_id,
                session_id=session_id,
                agent_id=row.agent_id,
                status="running",
                error=None,
                started_at=created,
                finished_at=None,
            )
        )
        stream_id = f"session:{session_id}"
        events = [
            self._event(
                db,
                stream_id=stream_id,
                event_type="message.created",
                payload={"message_id": message_id, "role": "user", "text": prompt},
                session_id=session_id,
                actor=actor,
                correlation_id=command_id,
            ),
            self._event(
                db,
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
            "text": prompt,
        }
        return result, events

    def list_conversations(self, kind: str, limit: int) -> list[dict[str, Any]]:
        clause = _kind_clause(kind)

        def read(db: Session) -> list[dict[str, Any]]:
            last_at = _last_message_at()
            rows = db.execute(
                select(
                    SessionRow.id.label("session_id"),
                    SessionRow.kind.label("session_kind"),
                    ChannelRow.kind.label("channel_kind"),
                    ChannelRow.name.label("channel_name"),
                    last_at.label("last_at"),
                )
                .outerjoin(ChannelRow, ChannelRow.id == SessionRow.channel_id)
                .where(clause)
                .order_by(func.coalesce(last_at, SessionRow.created_at).desc(), SessionRow.id)
                .limit(limit)
            ).all()
            return [_conversation_row(row._mapping) for row in rows]

        return self._read(read)

    def conversations_matching(self, conversation: str) -> list[dict[str, Any]]:
        needle = conversation.strip()

        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.execute(
                select(
                    SessionRow.id.label("session_id"),
                    SessionRow.kind.label("session_kind"),
                    ChannelRow.kind.label("channel_kind"),
                    ChannelRow.name.label("channel_name"),
                )
                .outerjoin(ChannelRow, ChannelRow.id == SessionRow.channel_id)
                .where(
                    SessionRow.kind.in_(("channel", "workspace_chat")),
                    or_(
                        SessionRow.id == needle,
                        SessionRow.channel_id == needle,
                        func.lower(ChannelRow.name) == needle.lower(),
                    ),
                )
                .order_by(SessionRow.id)
            ).all()
            return [_conversation_row(row._mapping) for row in rows]

        return self._read(read)

    def search_messages(self, query: str, kind: str, limit: int) -> list[dict[str, Any]]:
        clause = _kind_clause(kind)

        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.execute(
                select(
                    MessageRow.role,
                    MessageRow.body,
                    MessageRow.created_at,
                    SessionRow.id.label("session_id"),
                    SessionRow.kind.label("session_kind"),
                    ChannelRow.kind.label("channel_kind"),
                    ChannelRow.name.label("channel_name"),
                )
                .join(SessionRow, SessionRow.id == MessageRow.session_id)
                .outerjoin(ChannelRow, ChannelRow.id == SessionRow.channel_id)
                .where(
                    clause,
                    func.lower(MessageRow.body).like(_like(query), escape="\\"),
                )
                .order_by(MessageRow.created_at.desc(), _MESSAGE_ROW_ID.desc())
                .limit(limit)
            ).all()
            found = []
            for row in rows:
                item = _conversation_row(row._mapping)
                item["role"] = row.role
                item["body"] = row.body
                item["created_at"] = row.created_at
                found.append(item)
            return found

        return self._read(read)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.execute(
                select(MessageRow.role, MessageRow.body, MessageRow.created_at)
                .where(MessageRow.session_id == session_id)
                .order_by(_ROW_ID.desc())
                .limit(limit)
            ).all()
            return [
                {"role": row.role, "body": row.body, "created_at": row.created_at}
                for row in reversed(rows)
            ]

        return self._read(read)

    def mcp_servers(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(select(McpServerRow).order_by(McpServerRow.name, McpServerRow.id))
            return [_mcp_server(row) for row in rows]

        return self._read(read)

    def mcp_server(self, server_id: str) -> dict[str, Any] | None:
        found = [row for row in self.mcp_servers() if row["id"] == server_id]
        return found[0] if found else None

    def create_mcp_server(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent = db.get(AgentRow, body["agent_id"])
            if agent is None:
                raise NotFound("That agent does not exist.")
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            server_id = guid()
            db.add(
                McpServerRow(
                    id=server_id,
                    agent_id=body["agent_id"],
                    name=name,
                    transport=transport,
                    command=command,
                    url=url,
                    enabled=int(bool(body.get("enabled", True))),
                    status="unchecked",
                    status_message="",
                    revision=1,
                    group_switches="{}",
                )
            )
            groups = body.get("groups")
            if isinstance(groups, list):
                _save_switches(db, server_id, groups)
            result = _mcp_server(db.get(McpServerRow, server_id))
            self._remember(db, key or "", body, 201, result)
            return 201, result, []

        return self._write(write)

    def update_mcp_server(self, server_id: str, body: dict[str, Any]) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(McpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The MCP server was updated. Reload and try again.")
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            row.name = name
            row.transport = transport
            row.command = command
            row.url = url
            row.enabled = int(bool(body["enabled"]))
            row.revision = expected + 1
            groups = body.get("groups")
            if isinstance(groups, list):
                _save_switches(db, server_id, groups)
            return _mcp_server(row)

        return self._write(write)

    def mark_mcp_connected(self, server_id: str) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(McpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            row.status = "connected"
            row.status_message = "Connection succeeded"
            row.revision = row.revision + 1
            return _mcp_server(row)

        return self._write(write)

    def mark_mcp_failed(self, server_id: str) -> dict[str, Any]:
        def write(db: Session) -> dict[str, Any]:
            row = db.get(McpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            row.status = "failed"
            row.status_message = "Connection failed"
            row.revision = row.revision + 1
            return _mcp_server(row)

        return self._write(write)

    def list_global_mcp_servers(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(
                select(GlobalMcpServerRow).order_by(
                    GlobalMcpServerRow.name.collate("NOCASE"), GlobalMcpServerRow.id
                )
            )
            return [_global_mcp_server(row) for row in rows]

        return self._read(read)

    def global_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        found = [row for row in self.list_global_mcp_servers() if row["id"] == server_id]
        return found[0] if found else None

    def create_global_mcp_server(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            taken = db.scalars(
                select(GlobalMcpServerRow.id).where(GlobalMcpServerRow.name == name)
            ).first()
            if taken is not None:
                raise Rejected("That MCP name is already in the catalog.")
            server_id = guid()
            db.add(
                GlobalMcpServerRow(
                    id=server_id,
                    name=name,
                    transport=transport,
                    command=command,
                    url=url,
                    enabled=int(bool(body.get("enabled", True))),
                    add_by_default=int(bool(body.get("add_by_default"))),
                    revision=1,
                    group_switches="{}",
                    created_at=now(),
                )
            )
            groups = body.get("groups")
            if isinstance(groups, list):
                row = db.get(GlobalMcpServerRow, server_id)
                if row is not None:
                    row.group_switches = _switch_payload(groups, row.group_switches)
            result = _global_mcp_server(db.get(GlobalMcpServerRow, server_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="global_mcp.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_global_mcp_server(
        self, server_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(GlobalMcpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The MCP server was updated. Reload and try again.")
            transport, command, url = _mcp_fields(body)
            name = str(body.get("name") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            taken = db.scalars(
                select(GlobalMcpServerRow.id).where(
                    GlobalMcpServerRow.name == name,
                    GlobalMcpServerRow.id != server_id,
                )
            ).first()
            if taken is not None:
                raise Rejected("That MCP name is already in the catalog.")
            add_by_default = row.add_by_default
            if "add_by_default" in body and body["add_by_default"] is not None:
                add_by_default = int(bool(body["add_by_default"]))
            row.name = name
            row.transport = transport
            row.command = command
            row.url = url
            row.enabled = int(bool(body["enabled"]))
            row.add_by_default = add_by_default
            row.revision = expected + 1
            groups = body.get("groups")
            if isinstance(groups, list):
                row.group_switches = _switch_payload(groups, row.group_switches)
            result = _global_mcp_server(row)
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="global_mcp.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_global_mcp_server(
        self, server_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(GlobalMcpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            db.delete(row)
            result = {"id": server_id}
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="global_mcp.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def copy_global_mcp_server(
        self, key: str | None, server_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            row = db.get(GlobalMcpServerRow, server_id)
            if row is None:
                raise NotFound("That MCP server does not exist.")
            agent_id = str(body.get("agent_id") or "").strip()
            if not agent_id:
                raise Rejected("Choose an agent.")
            agent = db.get(AgentRow, agent_id)
            if agent is None:
                raise NotFound("That agent does not exist.")
            taken = db.scalars(
                select(McpServerRow.id).where(
                    McpServerRow.agent_id == agent_id,
                    McpServerRow.name == row.name,
                )
            ).first()
            if taken is not None:
                raise Rejected("That MCP name is already on this agent.")
            copy_id = guid()
            db.add(
                McpServerRow(
                    id=copy_id,
                    agent_id=agent_id,
                    name=row.name,
                    transport=row.transport,
                    command=row.command,
                    url=row.url,
                    enabled=int(row.enabled),
                    status="unchecked",
                    status_message="",
                    revision=1,
                    group_switches=(
                        default_switch_payload()
                        if row.name == "Favlo media"
                        else row.group_switches or "{}"
                    ),
                )
            )
            result = _mcp_server(db.get(McpServerRow, copy_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="mcp.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def enabled_mcp_servers(self, agent_id: str) -> list[dict[str, Any]]:
        """Enabled servers and the group switches. Tool definitions are not here."""

        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(
                select(McpServerRow)
                .where(McpServerRow.agent_id == agent_id, McpServerRow.enabled == 1)
                .order_by(McpServerRow.name, McpServerRow.id)
            )
            return [
                {
                    "server_id": row.id,
                    "server_name": row.name,
                    "transport": row.transport,
                    "command": row.command,
                    "url": row.url,
                    "switches": _group_switches(row.group_switches),
                }
                for row in rows
            ]

        return self._read(read)

    def clear_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Clear the chat and mint a new chat GUID for that slot."""

        def write(db: Session) -> list[dict[str, Any]]:
            _result, events = self._clear_chat(db, session_id)
            return events

        return self._write(write)

    def clear_chat(self, session_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Drop the chat and point its slot at a new GUID."""

        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return self._clear_chat(db, session_id)

        return self._write(write)

    def _clear_chat(
        self, db: Session, session_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        chat = db.get(SessionRow, session_id)
        if chat is None:
            raise NotFound("That session does not exist.")
        kind = chat.kind
        channel_id = chat.channel_id
        participant_id = chat.participant_id
        workspace_id = chat.workspace_id
        running = list(
            db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            )
        )
        _drop(db, MessageRow, MessageRow.session_id == session_id)
        _drop(db, RunRow, RunRow.session_id == session_id)
        _drop(db, EventRow, EventRow.stream_id == f"session:{session_id}")
        _drop(db, ThreadSnapshotRow, ThreadSnapshotRow.thread_id == session_id)
        new_id = guid()
        created = now()
        db.add(
            SessionRow(
                id=new_id,
                kind=kind,
                channel_id=channel_id,
                participant_id=participant_id,
                workspace_id=workspace_id,
                created_at=created,
            )
        )
        db.delete(chat)
        if kind == "workspace_chat":
            for workspace in db.scalars(
                select(WorkspaceRow).where(WorkspaceRow.session_id == session_id)
            ):
                workspace.session_id = new_id
        if kind == "heartbeat":
            for beat in db.scalars(
                select(HeartbeatRow).where(HeartbeatRow.session_id == session_id)
            ):
                beat.session_id = new_id
        events = [
            self._event(
                db,
                stream_id=f"session:{new_id}",
                event_type="session.cleared",
                payload={
                    "session_id": new_id,
                    "previous_session_id": session_id,
                },
                channel_id=channel_id,
                session_id=new_id,
                actor={"kind": "agent", "id": participant_id or ""},
            )
        ]
        result = {
            "status": "cleared",
            "session_id": new_id,
            "previous_session_id": session_id,
            "cleared": True,
            "cancelled_run_ids": running,
        }
        return result, events

    def maf_session(self, session_id: str) -> dict[str, Any] | None:
        """The Agent Framework session stored for this chat, if one has finished."""

        def read(db: Session) -> dict[str, Any] | None:
            row = db.get(SessionRow, session_id)
            if row is None or not row.maf_state:
                return None
            loaded = json.loads(row.maf_state)
            return loaded if isinstance(loaded, dict) else None

        return self._read(read)

    def save_maf_session(self, session_id: str, payload: dict[str, Any]) -> None:
        """Store the Agent Framework session that this turn actually sent."""
        encoded = json.dumps(payload)

        def write(db: Session) -> None:
            found = db.get(SessionRow, session_id)
            if found is None:
                return
            found.maf_state = encoded

        self._write(write)

    def definition(self, session_id: str) -> dict[str, Any] | None:
        """The frozen agent definition for this chat, if one exists."""

        def read(db: Session) -> dict[str, Any] | None:
            row = db.get(SessionRow, session_id)
            if row is None or not row.definition:
                return None
            loaded = json.loads(row.definition)
            return loaded if isinstance(loaded, dict) else None

        return self._read(read)

    def freeze_definition(self, session_id: str, payload: dict[str, Any]) -> bool:
        """Store the definition once. Later calls leave the existing row."""
        encoded = json.dumps(payload)
        digest = str(payload.get("hash") or "")

        def write(db: Session) -> bool:
            row = db.get(SessionRow, session_id)
            if row is None:
                return False
            if row.definition:
                return False
            row.definition = encoded
            row.definition_hash = digest
            return True

        return self._write(write)

    def agui_state(self, session_id: str) -> dict[str, Any] | None:
        """The last validated workspace placement for this chat."""

        def read(db: Session) -> dict[str, Any] | None:
            row = db.get(SessionRow, session_id)
            if row is None or not row.agui_state:
                return None
            loaded = json.loads(row.agui_state)
            return loaded if isinstance(loaded, dict) else None

        return self._read(read)

    def save_agui_state(self, session_id: str, payload: dict[str, Any]) -> None:
        """Store the validated workspace placement."""
        encoded = json.dumps(payload)

        def write(db: Session) -> None:
            found = db.get(SessionRow, session_id)
            if found is None:
                return
            found.agui_state = encoded

        self._write(write)

    def save_thread_snapshot(self, scope: str, thread_id: str, document: dict[str, Any]) -> None:
        """Replace the AG-UI snapshot for one thread."""
        encoded = json.dumps(document)

        def write(db: Session) -> None:
            row = db.scalars(
                select(ThreadSnapshotRow).where(
                    ThreadSnapshotRow.scope == scope,
                    ThreadSnapshotRow.thread_id == thread_id,
                )
            ).first()
            if row is None:
                db.add(
                    ThreadSnapshotRow(
                        id=guid(),
                        scope=scope,
                        thread_id=thread_id,
                        document=encoded,
                    )
                )
                return
            row.document = encoded

        self._write(write)

    def append_thread_snapshot_message(
        self, scope: str, thread_id: str, message: dict[str, Any]
    ) -> None:
        """Append one server-owned message to the durable AG-UI thread snapshot.

        When ``message`` carries an ``id`` that is already present, the row is
        left unchanged so a run-start user row and the library save stay one row.
        """

        def write(db: Session) -> None:
            message_id = message.get("id") if isinstance(message, dict) else None
            row = db.scalars(
                select(ThreadSnapshotRow).where(
                    ThreadSnapshotRow.scope == scope,
                    ThreadSnapshotRow.thread_id == thread_id,
                )
            ).first()
            if row is None or not row.document:
                document: dict[str, Any] = {
                    "messages": [message],
                    "state": None,
                    "interrupt": None,
                    "session_state": None,
                }
                db.add(
                    ThreadSnapshotRow(
                        id=guid(),
                        scope=scope,
                        thread_id=thread_id,
                        document=json.dumps(document),
                    )
                )
                return
            document = json.loads(row.document)
            if not isinstance(document, dict):
                document = {
                    "messages": [],
                    "state": None,
                    "interrupt": None,
                    "session_state": None,
                }
            messages = document.get("messages")
            if not isinstance(messages, list):
                messages = []
            if message_id:
                for existing in messages:
                    if isinstance(existing, dict) and existing.get("id") == message_id:
                        return
            messages.append(message)
            document["messages"] = messages
            row.document = json.dumps(document)

        self._write(write)

    def thread_snapshot(self, scope: str, thread_id: str) -> dict[str, Any] | None:
        def read(db: Session) -> dict[str, Any] | None:
            row = db.scalars(
                select(ThreadSnapshotRow).where(
                    ThreadSnapshotRow.scope == scope,
                    ThreadSnapshotRow.thread_id == thread_id,
                )
            ).first()
            if row is None or not row.document:
                return None
            loaded = json.loads(row.document)
            return loaded if isinstance(loaded, dict) else None

        return self._read(read)

    def delete_thread_snapshot(self, scope: str, thread_id: str) -> bool:
        def write(db: Session) -> bool:
            row = db.scalars(
                select(ThreadSnapshotRow).where(
                    ThreadSnapshotRow.scope == scope,
                    ThreadSnapshotRow.thread_id == thread_id,
                )
            ).first()
            if row is None:
                return False
            db.delete(row)
            return True

        return self._write(write)

    def clear_thread_snapshots(self, scope: str | None) -> None:
        def write(db: Session) -> None:
            if scope is None:
                _drop(db, ThreadSnapshotRow)
                return
            _drop(db, ThreadSnapshotRow, ThreadSnapshotRow.scope == scope)

        self._write(write)

    def settle_agui_run(self, run_id: str, status: str, error: str | None = None) -> list[dict[str, Any]]:
        """Close a library turn without writing a second assistant row."""

        def write(db: Session) -> list[dict[str, Any]]:
            run = db.get(RunRow, run_id)
            if run is None or run.status != "running":
                return []
            chat = db.get(SessionRow, run.session_id)
            run.status = status
            run.error = error
            run.finished_at = now()
            if status == "completed":
                event_type = "run.completed"
            elif status == "cancelled":
                event_type = "run.cancelled"
            else:
                event_type = "run.failed"
            payload: dict[str, Any] = {"status": status}
            if error:
                payload["message"] = error
            return [
                self._event(
                    db,
                    stream_id=f"session:{run.session_id}",
                    event_type=event_type,
                    payload=payload,
                    channel_id=chat.channel_id if chat is not None else None,
                    session_id=run.session_id,
                    run_id=run_id,
                    actor={"kind": "agent", "id": run.agent_id},
                )
            ]

        return self._write(write)

    def loaded_mcp_tools(self, session_id: str) -> dict[str, list[str]]:
        """Remote MCP tool names loaded via progressive disclosure for this chat."""

        def read(db: Session) -> dict[str, list[str]]:
            row = db.get(SessionRow, session_id)
            if row is None or not row.loaded_mcp:
                return {}
            loaded = json.loads(row.loaded_mcp)
            if not isinstance(loaded, dict):
                return {}
            out: dict[str, list[str]] = {}
            for server_id, names in loaded.items():
                if isinstance(names, list):
                    out[str(server_id)] = [str(name) for name in names]
            return out

        return self._read(read)

    def save_loaded_mcp_tools(self, session_id: str, payload: dict[str, list[str]]) -> None:
        """Persist progressively loaded MCP remote names until the chat is cleared."""
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

        def write(db: Session) -> None:
            found = db.get(SessionRow, session_id)
            if found is None:
                return
            found.loaded_mcp = encoded

        self._write(write)

    def compaction_percent(self, session_id: str) -> int:
        """Live compaction trigger for this chat. Not part of the frozen prompt."""

        def read(db: Session) -> int:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                return 80
            workspace_percent: int | None = None
            if chat.workspace_id:
                workspace = db.get(WorkspaceRow, chat.workspace_id)
                if workspace is not None and workspace.compaction_percent is not None:
                    workspace_percent = int(workspace.compaction_percent)
            agent_percent = 80
            if chat.participant_id:
                agent = db.get(AgentRow, chat.participant_id)
                if agent is not None and agent.compaction_percent is not None:
                    agent_percent = int(agent.compaction_percent)
            value = workspace_percent if workspace_percent is not None else agent_percent
            return max(50, min(95, value))

        return self._read(read)

    def channels(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.execute(
                select(
                    ChannelRow.id,
                    ChannelRow.kind,
                    ChannelRow.name,
                    ChannelRow.revision,
                    ChannelMemberRow.agent_id,
                    SessionRow.id.label("session_id"),
                    AgentRow.expand_thinking,
                    AgentRow.expand_tools,
                )
                .join(ChannelMemberRow, ChannelMemberRow.channel_id == ChannelRow.id)
                .join(
                    SessionRow,
                    and_(SessionRow.channel_id == ChannelRow.id, SessionRow.kind == "channel"),
                )
                .join(AgentRow, AgentRow.id == ChannelMemberRow.agent_id)
                .where(ChannelRow.kind == "agent")
                .order_by(ChannelRow.name.collate("NOCASE"), ChannelRow.id)
            ).all()
            return [
                {
                    "id": row.id,
                    "kind": row.kind,
                    "name": row.name,
                    "revision": row.revision,
                    "agent_id": row.agent_id,
                    "session_id": row.session_id,
                    "expand_thinking": _stored_expand(row.expand_thinking, default="collapsed"),
                    "expand_tools": _stored_expand(row.expand_tools, default="hidden"),
                }
                for row in rows
            ]

        return self._read(read)

    def list_shells(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(select(ShellRow).order_by(ShellRow.sort, ShellRow.id))
            return [_shell_payload(db, row) for row in rows]

        return self._read(read)

    def list_workspaces(self, channel_id: str | None = None) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            if channel_id:
                channel = db.get(ChannelRow, channel_id)
                if channel is None:
                    raise NotFound("That channel does not exist.")
                rows = db.scalars(
                    select(WorkspaceRow)
                    .where(WorkspaceRow.channel_id == channel_id)
                    .order_by(WorkspaceRow.created_at, WorkspaceRow.id)
                )
            else:
                rows = db.scalars(
                    select(WorkspaceRow).order_by(
                        WorkspaceRow.name.collate("NOCASE"), WorkspaceRow.id
                    )
                )
            return [_workspace_payload(db, row) for row in rows]

        return self._read(read)

    def workspace(self, workspace_id: str) -> dict[str, Any]:
        def read(db: Session) -> dict[str, Any]:
            return _require_workspace(db, workspace_id)

        return self._read(read)

    def create_workspace(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            channel_id = str(body.get("channel_id") or "")
            name = str(body.get("name") or "").strip()
            kind = str(body.get("type") or "gallery")
            location = str(body.get("location") or "").strip()
            if not name:
                raise Rejected("A name is required.")
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise Rejected("That workspace name cannot be a folder.")
            shell = db.get(ShellRow, kind)
            if shell is None:
                raise Rejected("That workspace shell does not exist.")
            if shell.setup != "local_folder" and location:
                raise Rejected("This workspace has no local folder.")
            channel = db.get(ChannelRow, channel_id)
            if channel is None:
                raise NotFound("That channel does not exist.")
            member = db.scalars(
                select(ChannelMemberRow).where(ChannelMemberRow.channel_id == channel_id)
            ).first()
            if member is None:
                raise Rejected("That channel has no agent.")
            if shell.setup == "local_folder" and not location:
                from web_api.workspace_files import default_workspace_dir

                path = default_workspace_dir(str(member.agent_id), name)
                resolved = str(path.resolve())
                taken = db.scalars(
                    select(FolderRow.id).where(FolderRow.location == resolved)
                ).first()
                if taken is not None:
                    raise Rejected("That folder is already a workspace.")
                path.mkdir(parents=True, exist_ok=True)
                location = resolved
            workspace_id = guid()
            session_id = guid()
            created = now()
            db.add(
                SessionRow(
                    id=session_id,
                    kind="workspace_chat",
                    channel_id=channel_id,
                    participant_id=member.agent_id,
                    workspace_id=workspace_id,
                    created_at=created,
                )
            )
            workspace = WorkspaceRow(
                id=workspace_id,
                channel_id=channel_id,
                session_id=session_id,
                name=name,
                type=kind,
                provisioning_state="new",
                revision=1,
                created_at=created,
            )
            db.add(workspace)
            _seed_workspace_skills(db, workspace_id, kind, created)
            if shell.setup == "local_folder":
                fields = _folder_fields(
                    db,
                    workspace_id,
                    {"kind": "local", "location": location, "name": "Library"},
                )
                folder_id = guid()
                db.add(
                    FolderRow(
                        id=folder_id,
                        workspace_id=workspace_id,
                        parent_id=None,
                        name=fields["name"],
                        kind="local",
                        provider=None,
                        location=fields["location"],
                        machine_id="",
                        remote_id=None,
                        revision=1,
                        created_at=created,
                    )
                )
                workspace.focus_folder_id = folder_id
            result = _require_workspace(db, workspace_id)
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="workspace.created",
                    payload=result,
                    channel_id=channel_id,
                    session_id=session_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def claim_provisioning(self, workspace_id: str) -> bool:
        """Move ``new`` to ``provisioning``. A later replay does not start again."""

        def write(db: Session) -> bool:
            row = db.get(WorkspaceRow, workspace_id)
            if row is None or row.provisioning_state != "new":
                return False
            row.provisioning_state = "provisioning"
            row.revision += 1
            return True

        return self._write(write)

    def finish_provisioning(self, workspace_id: str) -> list[dict[str, Any]]:
        """Mark the workspace provisioned and tell the channel and the workspace page."""

        def write(db: Session) -> list[dict[str, Any]]:
            row = db.get(WorkspaceRow, workspace_id)
            if row is None or row.provisioning_state == "provisioned":
                return []
            row.provisioning_state = "provisioned"
            row.revision += 1
            payload = {"id": row.id, "provisioning_state": "provisioned", "name": row.name}
            actor = {"kind": "agent", "id": "local"}
            return [
                self._event(
                    db,
                    stream_id="app",
                    event_type="workspace.provisioned",
                    payload=payload,
                    channel_id=row.channel_id,
                    session_id=row.session_id,
                    actor=actor,
                ),
                self._event(
                    db,
                    stream_id=f"session:{row.session_id}",
                    event_type="workspace.provisioned",
                    payload=payload,
                    channel_id=row.channel_id,
                    session_id=row.session_id,
                    actor=actor,
                ),
            ]

        return self._write(write)

    def begin_greeting(self, session_id: str, run_id: str, cue: str) -> dict[str, Any]:
        """Open a run whose cue is the greeting.

        The host writes that cue as a fenced, UI-hidden user row on the thread
        snapshot (see ``host_cues``). It is not the person's typed message.
        """

        def write(db: Session) -> dict[str, Any]:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That chat does not exist.")
            cleaned = cue.strip()
            if not cleaned or cleaned == ARCHIVE_CUE:
                raise Rejected("A greeting is required.")
            active = db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            ).first()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            created = now()
            db.add(
                RunRow(
                    id=run_id,
                    session_id=session_id,
                    agent_id=chat.participant_id,
                    status="running",
                    error=None,
                    started_at=created,
                    finished_at=None,
                    cue=cleaned,
                )
            )
            events = [
                self._event(
                    db,
                    stream_id=f"session:{session_id}",
                    event_type="run.started",
                    payload={"status": "running"},
                    channel_id=chat.channel_id,
                    session_id=session_id,
                    run_id=run_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return {"run_id": run_id, "session_id": session_id, "events": events}

        return self._write(write)

    def record_assistant(
        self,
        session_id: str,
        text: str,
        *,
        task_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Store one assistant message. Used when provisioning fails before a reply.

        ``task_ids`` marks a background media delivery so the UI does not clear
        Working for an open run.
        """

        def write(db: Session) -> list[dict[str, Any]]:
            chat = db.get(SessionRow, session_id)
            cleaned = text.strip()
            if chat is None or not cleaned:
                return []
            message_id = guid()
            created = now()
            db.add(
                MessageRow(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    body=cleaned,
                    actor_kind="agent",
                    actor_id=chat.participant_id,
                    created_at=created,
                )
            )
            payload: dict[str, Any] = {
                "message_id": message_id,
                "role": "assistant",
                "text": cleaned,
            }
            if task_ids:
                payload["task_ids"] = [item for item in task_ids if item]
            return [
                self._event(
                    db,
                    stream_id=f"session:{session_id}",
                    event_type="message.created",
                    payload=payload,
                    channel_id=chat.channel_id,
                    session_id=session_id,
                    actor={"kind": "agent", "id": chat.participant_id or ""},
                )
            ]

        return self._write(write)

    def note_media_pending(
        self,
        session_id: str,
        task_id: str,
        kind: str,
        status: str,
    ) -> list[dict[str, Any]]:
        """Publish a host media placeholder event keyed by task id (no run_id)."""

        def write(db: Session) -> list[dict[str, Any]]:
            chat = db.get(SessionRow, session_id)
            cleaned = task_id.strip()
            if chat is None or not cleaned:
                return []
            return [
                self._event(
                    db,
                    stream_id=f"session:{session_id}",
                    event_type="media.pending",
                    payload={
                        "task_id": cleaned,
                        "kind": kind.strip() or "image",
                        "status": status.strip() or "working",
                    },
                    channel_id=chat.channel_id,
                    session_id=session_id,
                    actor={"kind": "agent", "id": chat.participant_id or ""},
                )
            ]

        return self._write(write)

    def delete_workspace(
        self, workspace_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        def write(
            db: Session,
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
            row = db.get(WorkspaceRow, workspace_id)
            if row is None:
                raise NotFound("That workspace does not exist.")
            session_id = row.session_id
            name = row.name
            channel_id = row.channel_id
            running = list(
                db.scalars(
                    select(RunRow.id).where(
                        RunRow.session_id == session_id,
                        RunRow.status == "running",
                    )
                )
            )
            _drop(db, MessageRow, MessageRow.session_id == session_id)
            _drop(db, RunRow, RunRow.session_id == session_id)
            _drop(db, EventRow, EventRow.stream_id == f"session:{session_id}")
            _drop(db, FolderRow, FolderRow.workspace_id == workspace_id)
            _drop(db, SkillRow, SkillRow.workspace_id == workspace_id)
            chat = db.get(SessionRow, session_id)
            if chat is not None:
                db.delete(chat)
            db.delete(row)
            result = {"id": workspace_id, "name": name}
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="workspace.deleted",
                    payload=result,
                    channel_id=channel_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events, running

        return self._write(write)

    def list_folders(self, workspace_id: str, parent_id: str | None = None) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            _require_workspace(db, workspace_id)
            parent = (
                FolderRow.parent_id == parent_id if parent_id else FolderRow.parent_id.is_(None)
            )
            rows = db.scalars(
                select(FolderRow)
                .where(FolderRow.workspace_id == workspace_id, parent)
                .order_by(FolderRow.name.collate("NOCASE"), FolderRow.id)
            )
            return [_folder_payload(row) for row in rows]

        return self._read(read)

    def add_folder(
        self, key: str | None, workspace_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            workspace = _require_workspace(db, workspace_id)
            if workspace["type"] != "gallery":
                raise Rejected("That workspace is not a gallery.")
            fields = _folder_fields(db, workspace_id, body)
            _reject_sibling(db, workspace_id, fields["parent_id"], fields["name"], None)
            folder_id = guid()
            created = now()
            db.add(
                FolderRow(
                    id=folder_id,
                    workspace_id=workspace_id,
                    parent_id=fields["parent_id"],
                    name=fields["name"],
                    kind=fields["kind"],
                    provider=fields["provider"],
                    location=fields["location"],
                    machine_id="",
                    remote_id=fields["remote_id"],
                    revision=1,
                    created_at=created,
                )
            )
            result = _folder_payload(db.get(FolderRow, folder_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="folder.created",
                    payload=result,
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_folder(
        self, workspace_id: str, folder_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(db, workspace_id)
            row = _folder_in(db, workspace_id, folder_id)
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("That folder was updated. Reload and try again.")
            name = " ".join(str(body.get("name") or row.name).split())
            if not name:
                raise Rejected("A name is required.")
            parent_id = row.parent_id
            if body.get("move"):
                parent_id = _move_parent(db, workspace_id, row, body.get("parent_id"))
            _reject_sibling(db, workspace_id, parent_id, name, folder_id)
            row.name = name
            row.parent_id = parent_id
            row.revision = row.revision + 1
            result = _folder_payload(row)
            events = [
                self._event(
                    db,
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
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(db, workspace_id)
            row = _folder_in(db, workspace_id, folder_id)
            children = db.scalar(
                select(func.count()).select_from(FolderRow).where(FolderRow.parent_id == folder_id)
            )
            if children:
                raise Rejected("Move the folders inside it first.")
            name = row.name
            db.delete(row)
            home = db.get(WorkspaceRow, workspace_id)
            if home is not None and home.focus_folder_id == folder_id:
                home.focus_folder_id = None
            result = {"id": folder_id, "name": name}
            events = [
                self._event(
                    db,
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
        def write(db: Session) -> dict[str, Any]:
            _require_workspace(db, workspace_id)
            if folder_id:
                row = _folder_in(db, workspace_id, folder_id)
                if row.kind != "local":
                    raise Rejected("Open a local folder.")
            workspace = db.get(WorkspaceRow, workspace_id)
            if workspace is not None:
                workspace.focus_folder_id = folder_id
            return {"folder_id": folder_id}

        return self._write(write)

    def set_workspace_theme(
        self, workspace_id: str, theme: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(db, workspace_id)
            if workspace["type"] not in ("gallery", "coding"):
                raise Rejected("Themes are for gallery and coding workspaces.")
            if theme not in _WORKSPACE_THEMES:
                raise Rejected("Choose Favlo, Rosé Pine, or Catppuccin.")
            meta_key = f"workspace_theme:{workspace_id}"
            row = db.get(MetaRow, meta_key)
            if row is None:
                legacy = db.get(MetaRow, f"gallery_theme:{workspace_id}")
                if legacy is not None:
                    db.delete(legacy)
                db.add(MetaRow(key=meta_key, value=theme))
            else:
                row.value = theme
            event = self._event(
                db,
                stream_id=f"session:{workspace['session_id']}",
                event_type="workspace.theme_set",
                payload={"id": workspace_id, "theme": theme},
                channel_id=workspace["channel_id"],
                session_id=workspace["session_id"],
            )
            return {"id": workspace_id, "theme": theme}, [event]

        return self._write(write)

    def set_gallery_theme(
        self, workspace_id: str, theme: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self.set_workspace_theme(workspace_id, theme)

    def local_location(self, workspace_id: str, folder_id: str | None = None) -> str:
        def read(db: Session) -> str:
            _require_workspace(db, workspace_id)
            if folder_id:
                row = _folder_in(db, workspace_id, folder_id)
                if row.kind != "local" or not row.location:
                    raise Rejected("That folder is not on this machine.")
                return str(row.location)
            workspace = db.get(WorkspaceRow, workspace_id)
            focus = workspace.focus_folder_id if workspace is not None else None
            if focus:
                focused = db.get(FolderRow, focus)
                if (
                    focused is not None
                    and focused.workspace_id == workspace_id
                    and focused.kind == "local"
                    and focused.location
                ):
                    return str(focused.location)
            rows = list(
                db.scalars(
                    select(FolderRow)
                    .where(FolderRow.workspace_id == workspace_id, FolderRow.kind == "local")
                    .order_by(FolderRow.created_at, FolderRow.id)
                )
            )
            if len(rows) == 1 and rows[0].location:
                return str(rows[0].location)
            if not rows:
                raise Rejected("Save a local folder first.")
            raise Rejected("Open a local folder.")

        return self._read(read)

    def list_agent_skills(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            if agent_id:
                agent = db.get(AgentRow, agent_id)
                if agent is None:
                    raise NotFound("That agent does not exist.")
            return _agent_skills_for(db, agent_id)

        return self._read(read)

    def create_agent_skill(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            agent_id = str(body.get("agent_id") or "")
            agent = db.get(AgentRow, agent_id)
            if agent is None:
                raise NotFound("That agent does not exist.")
            name, description, definition = _skill_fields(body)
            taken = db.scalars(
                select(AgentSkillRow.id).where(
                    AgentSkillRow.agent_id == agent_id,
                    AgentSkillRow.name == name,
                )
            ).first()
            if taken is not None:
                raise Rejected("That skill name is already on this agent.")
            skill_id = guid()
            created = now()
            db.add(
                AgentSkillRow(
                    id=skill_id,
                    agent_id=agent_id,
                    name=name,
                    description=description,
                    definition=definition,
                    revision=1,
                    created_at=created,
                )
            )
            result = _agent_skill_payload(db.get(AgentSkillRow, skill_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="agent_skill.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_agent_skill(
        self, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(AgentSkillRow, skill_id)
            if row is None:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = db.scalars(
                select(AgentSkillRow.id).where(
                    AgentSkillRow.agent_id == row.agent_id,
                    AgentSkillRow.name == name,
                    AgentSkillRow.id != skill_id,
                )
            ).first()
            if taken is not None:
                raise Rejected("That skill name is already on this agent.")
            row.name = name
            row.description = description
            row.definition = definition
            row.revision = expected + 1
            result = _agent_skill_payload(row)
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="agent_skill.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_agent_skill(self, skill_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(AgentSkillRow, skill_id)
            if row is None:
                raise NotFound("That skill does not exist.")
            agent_id = row.agent_id
            db.delete(row)
            result = {"id": skill_id, "agent_id": agent_id}
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="agent_skill.deleted",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def list_global_skills(self) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(
                select(GlobalSkillRow).order_by(
                    GlobalSkillRow.name.collate("NOCASE"), GlobalSkillRow.id
                )
            )
            return [_global_skill_payload(row) for row in rows]

        return self._read(read)

    def create_global_skill(
        self, key: str | None, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            name, description, definition = _skill_fields(body)
            taken = db.scalars(select(GlobalSkillRow.id).where(GlobalSkillRow.name == name)).first()
            if taken is not None:
                raise Rejected("That skill name is already in the catalog.")
            skill_id = guid()
            created = now()
            db.add(
                GlobalSkillRow(
                    id=skill_id,
                    name=name,
                    description=description,
                    definition=definition,
                    add_by_default=int(bool(body.get("add_by_default"))),
                    revision=1,
                    created_at=created,
                )
            )
            result = _global_skill_payload(db.get(GlobalSkillRow, skill_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="global_skill.created",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_global_skill(
        self, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(GlobalSkillRow, skill_id)
            if row is None:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = db.scalars(
                select(GlobalSkillRow.id).where(
                    GlobalSkillRow.name == name,
                    GlobalSkillRow.id != skill_id,
                )
            ).first()
            if taken is not None:
                raise Rejected("That skill name is already in the catalog.")
            add_by_default = row.add_by_default
            if "add_by_default" in body and body["add_by_default"] is not None:
                add_by_default = int(bool(body["add_by_default"]))
            row.name = name
            row.description = description
            row.definition = definition
            row.add_by_default = add_by_default
            row.revision = expected + 1
            result = _global_skill_payload(row)
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="global_skill.updated",
                    payload=result,
                    actor={"kind": "user", "id": "local"},
                )
            ]
            return result, events

        return self._write(write)

    def delete_global_skill(self, skill_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            row = db.get(GlobalSkillRow, skill_id)
            if row is None:
                raise NotFound("That skill does not exist.")
            db.delete(row)
            result = {"id": skill_id}
            events = [
                self._event(
                    db,
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
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            row = db.get(GlobalSkillRow, skill_id)
            if row is None:
                raise NotFound("That skill does not exist.")
            agent_id = str(body.get("agent_id") or "").strip()
            workspace_id = str(body.get("workspace_id") or "").strip()
            if bool(agent_id) == bool(workspace_id):
                raise Rejected("Choose an agent or a workspace.")
            name = str(row.name)
            description = str(row.description)
            definition = str(row.definition or "")
            created = now()
            copy_id = guid()
            if agent_id:
                agent = db.get(AgentRow, agent_id)
                if agent is None:
                    raise NotFound("That agent does not exist.")
                taken = db.scalars(
                    select(AgentSkillRow.id).where(
                        AgentSkillRow.agent_id == agent_id,
                        AgentSkillRow.name == name,
                    )
                ).first()
                if taken is not None:
                    raise Rejected("That skill name is already on this agent.")
                db.add(
                    AgentSkillRow(
                        id=copy_id,
                        agent_id=agent_id,
                        name=name,
                        description=description,
                        definition=definition,
                        revision=1,
                        created_at=created,
                    )
                )
                result = _agent_skill_payload(db.get(AgentSkillRow, copy_id))
                events = [
                    self._event(
                        db,
                        stream_id="app",
                        event_type="agent_skill.created",
                        payload=result,
                        actor={"kind": "user", "id": "local"},
                    )
                ]
            else:
                workspace = _require_workspace(db, workspace_id)
                taken = db.scalars(
                    select(SkillRow.id).where(
                        SkillRow.workspace_id == workspace_id,
                        SkillRow.name == name,
                    )
                ).first()
                if taken is not None:
                    raise Rejected("That skill name is already on this workspace.")
                db.add(
                    SkillRow(
                        id=copy_id,
                        workspace_id=workspace_id,
                        name=name,
                        description=description,
                        definition=definition,
                        revision=1,
                        created_at=created,
                    )
                )
                result = _skill_payload(db.get(SkillRow, copy_id))
                events = [
                    self._event(
                        db,
                        stream_id="app",
                        event_type="skill.created",
                        payload={"workspace_id": workspace_id, "skill": result},
                        channel_id=workspace["channel_id"],
                        session_id=workspace["session_id"],
                        actor={"kind": "user", "id": "local"},
                    )
                ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def list_skills(self, workspace_id: str) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            _require_workspace(db, workspace_id)
            return _skills_for(db, workspace_id)

        return self._read(read)

    def create_skill(
        self, key: str | None, workspace_id: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            workspace = _require_workspace(db, workspace_id)
            name, description, definition = _skill_fields(body)
            taken = db.scalars(
                select(SkillRow.id).where(
                    SkillRow.workspace_id == workspace_id,
                    SkillRow.name == name,
                )
            ).first()
            if taken is not None:
                raise Rejected("That skill name is already on this workspace.")
            skill_id = guid()
            created = now()
            db.add(
                SkillRow(
                    id=skill_id,
                    workspace_id=workspace_id,
                    name=name,
                    description=description,
                    definition=definition,
                    revision=1,
                    created_at=created,
                )
            )
            result = _skill_payload(db.get(SkillRow, skill_id))
            events = [
                self._event(
                    db,
                    stream_id="app",
                    event_type="skill.created",
                    payload={"workspace_id": workspace_id, "skill": result},
                    channel_id=workspace["channel_id"],
                    session_id=workspace["session_id"],
                    actor={"kind": "user", "id": "local"},
                )
            ]
            self._remember(db, key or "", body, 201, result)
            return 201, result, events

        return self._write(write)

    def update_skill(
        self, workspace_id: str, skill_id: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(db, workspace_id)
            row = db.get(SkillRow, skill_id)
            if row is None or row.workspace_id != workspace_id:
                raise NotFound("That skill does not exist.")
            expected = int(body["expected_revision"])
            if int(row.revision) != expected:
                raise Conflict("The skill was updated. Reload and try again.")
            name, description, definition = _skill_fields(body)
            taken = db.scalars(
                select(SkillRow.id).where(
                    SkillRow.workspace_id == workspace_id,
                    SkillRow.name == name,
                    SkillRow.id != skill_id,
                )
            ).first()
            if taken is not None:
                raise Rejected("That skill name is already on this workspace.")
            row.name = name
            row.description = description
            row.definition = definition
            row.revision = expected + 1
            result = _skill_payload(row)
            events = [
                self._event(
                    db,
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
        def write(db: Session) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            workspace = _require_workspace(db, workspace_id)
            row = db.get(SkillRow, skill_id)
            if row is None or row.workspace_id != workspace_id:
                raise NotFound("That skill does not exist.")
            db.delete(row)
            result = {"id": skill_id, "workspace_id": workspace_id}
            events = [
                self._event(
                    db,
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

    def agent_for_session(self, session_id: str) -> str | None:
        """The agent that owns this chat, or none when the session is missing."""

        def read(db: Session) -> str | None:
            chat = db.get(SessionRow, session_id)
            if chat is None or not chat.participant_id:
                return None
            return str(chat.participant_id)

        return self._read(read)

    def workspace_for_session(self, session_id: str) -> dict[str, str] | None:
        def read(db: Session) -> dict[str, str] | None:
            chat = db.get(SessionRow, session_id)
            if chat is None or chat.kind != "workspace_chat":
                return None
            workspace_id = chat.workspace_id
            if not workspace_id:
                return None
            workspace = db.get(WorkspaceRow, workspace_id)
            if workspace is None:
                return None
            return {"id": str(workspace.id), "type": str(workspace.type)}

        return self._read(read)

    def gallery_for_session(self, session_id: str) -> str | None:
        """Gallery workspace id for this chat. Bound even before a local folder exists."""

        def read(db: Session) -> str | None:
            chat = db.get(SessionRow, session_id)
            if chat is None or chat.kind != "workspace_chat":
                return None
            workspace_id = chat.workspace_id
            if not workspace_id:
                return None
            workspace = db.get(WorkspaceRow, workspace_id)
            if workspace is None or workspace.type != "gallery":
                return None
            return str(workspace_id)

        return self._read(read)

    def has_local_folder(self, workspace_id: str) -> bool:
        def read(db: Session) -> bool:
            row = db.scalars(
                select(FolderRow.id).where(
                    FolderRow.workspace_id == workspace_id,
                    FolderRow.kind == "local",
                )
            ).first()
            return row is not None

        return self._read(read)

    def workspace_session_id(self, workspace_id: str) -> str | None:
        def read(db: Session) -> str | None:
            row = db.get(WorkspaceRow, workspace_id)
            return None if row is None else str(row.session_id)

        return self._read(read)

    def note_gallery(self, workspace_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        def write(db: Session) -> list[dict[str, Any]]:
            row = db.get(WorkspaceRow, workspace_id)
            if row is None:
                raise NotFound("That workspace does not exist.")
            return [
                self._event(
                    db,
                    stream_id=f"session:{row.session_id}",
                    event_type="gallery.updated",
                    payload=payload,
                    channel_id=row.channel_id,
                    session_id=row.session_id,
                    actor={"kind": "user", "id": "local"},
                )
            ]

        return self._write(write)

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        def read(db: Session) -> dict[str, Any]:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That session does not exist.")
            row_messages = []
            for row in db.scalars(
                select(MessageRow).where(MessageRow.session_id == session_id).order_by(_ROW_ID)
            ):
                item = _view(row)
                item["tool_calls"] = _tool_calls(item.get("tool_calls"))
                item["trace"] = _trace(item.get("trace"))
                row_messages.append(item)
            messages = list(row_messages)
            runs = [
                _view(row)
                for row in db.scalars(
                    select(RunRow).where(RunRow.session_id == session_id).order_by(_ROW_ID)
                )
            ]
            stream_id = f"session:{session_id}"
            cursor = db.scalar(
                select(func.coalesce(func.max(EventRow.sequence), 0)).where(
                    EventRow.stream_id == stream_id
                )
            )
            state = None
            if chat.agui_state:
                loaded = json.loads(chat.agui_state)
                state = loaded if isinstance(loaded, dict) else None
            snapshot_row = db.scalars(
                select(ThreadSnapshotRow).where(ThreadSnapshotRow.thread_id == session_id)
            ).first()
            if snapshot_row is not None and snapshot_row.document:
                from web_api.snapshots import project_messages

                document = json.loads(snapshot_row.document)
                if isinstance(document, dict):
                    recorded = document.get("messages")
                    if isinstance(recorded, list):
                        messages = project_messages(recorded)
                        # While a run is open the library has not saved the user
                        # turn yet. Surface that MessageRow so a reload keeps it.
                        if any(str(run.get("status") or "") == "running" for run in runs):
                            seen = {str(item.get("id") or "") for item in messages}
                            for item in row_messages:
                                item_id = str(item.get("id") or "")
                                if item_id and item_id not in seen and item.get("role") == "user":
                                    messages.append(item)
                                    seen.add(item_id)
                    shared = document.get("state")
                    if isinstance(shared, dict) and shared:
                        state = shared
            if not state:
                state = {}
                if chat.kind == "workspace_chat" and chat.workspace_id:
                    workspace = db.get(WorkspaceRow, chat.workspace_id)
                    if workspace is not None:
                        shell = db.get(ShellRow, workspace.type)
                        if shell is not None:
                            from web_api.placement import initial_document

                            built = _shell_payload(db, shell)
                            state = initial_document(built["id"], built["elements"])
            return {
                "session": _session_view(chat),
                "messages": messages,
                "runs": runs,
                "cursor": int(cursor or 0),
                "state": state,
                "pending_media": _open_media_tasks(db, session_id),
            }

        return self._read(read)

    def begin_agui_run(
        self,
        session_id: str,
        run_id: str,
        text: str,
        *,
        archive: bool = False,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Record the user message (unless archive) and open a running run.

        ``message_id`` is the client AG-UI message id when present so the
        snapshot row at run start and the library save keep one row for that id.
        """

        def write(db: Session) -> dict[str, Any]:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That chat does not exist.")
            active = db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            ).first()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            created = now()
            actor = {"kind": "user", "id": "local"}
            stored_message_id = None
            events: list[dict[str, Any]] = []
            if archive:
                db.add(
                    RunRow(
                        id=run_id,
                        session_id=session_id,
                        agent_id=chat.participant_id,
                        status="running",
                        error=None,
                        started_at=created,
                        finished_at=None,
                        cue=ARCHIVE_CUE,
                    )
                )
            else:
                cleaned = text.strip()
                if not cleaned:
                    raise Rejected("A user message is required.")
                stored_message_id = (message_id or "").strip() or guid()
                db.add(
                    MessageRow(
                        id=stored_message_id,
                        session_id=session_id,
                        role="user",
                        body=cleaned,
                        actor_kind="user",
                        actor_id="local",
                        created_at=created,
                    )
                )
                db.add(
                    RunRow(
                        id=run_id,
                        session_id=session_id,
                        agent_id=chat.participant_id,
                        status="running",
                        error=None,
                        started_at=created,
                        finished_at=None,
                    )
                )
                events.append(
                    self._event(
                        db,
                        stream_id=f"session:{session_id}",
                        event_type="message.created",
                        payload={
                            "message_id": stored_message_id,
                            "role": "user",
                            "text": cleaned,
                        },
                        channel_id=chat.channel_id,
                        session_id=session_id,
                        actor=actor,
                    )
                )
            events.append(
                self._event(
                    db,
                    stream_id=f"session:{session_id}",
                    event_type="run.started",
                    payload={"status": "running"},
                    channel_id=chat.channel_id,
                    session_id=session_id,
                    run_id=run_id,
                    actor=actor,
                )
            )
            return {
                "run_id": run_id,
                "message_id": stored_message_id,
                "session_id": session_id,
                "events": events,
            }

        return self._write(write)

    def session_exists(self, session_id: str) -> bool:
        """True when this chat GUID is still in the database."""

        def read(db: Session) -> bool:
            return db.get(SessionRow, session_id) is not None

        return self._read(read)

    def arm_archive(self, session_id: str, run_id: str) -> dict[str, Any]:
        """Mark the chat as archiving and return the run id for the AG-UI wrap-up."""

        def write(db: Session) -> dict[str, Any]:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That chat does not exist.")
            active = db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            ).first()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            if not run_id.strip():
                raise Rejected("run_id is required.")
            db.add(
                RunRow(
                    id=run_id,
                    session_id=session_id,
                    agent_id=chat.participant_id,
                    status="running",
                    error=None,
                    started_at=now(),
                    finished_at=None,
                    cue=ARCHIVE_CUE,
                )
            )
            self._event(
                db,
                stream_id=f"session:{session_id}",
                event_type="run.started",
                payload={"status": "running"},
                channel_id=chat.channel_id,
                session_id=session_id,
                run_id=run_id,
                actor={"kind": "user", "id": "local"},
            )
            return {"status": "accepted", "run_id": run_id, "session_id": session_id}

        return self._write(write)

    def events_after(self, stream_id: str, cursor: int) -> list[dict[str, Any]]:
        def read(db: Session) -> list[dict[str, Any]]:
            rows = db.scalars(
                select(EventRow.body)
                .where(EventRow.stream_id == stream_id, EventRow.sequence > cursor)
                .order_by(EventRow.sequence)
            )
            return [json.loads(body) for body in rows]

        return self._read(read)

    def begin_message(
        self,
        key: str | None,
        session_id: str,
        text: str,
        folder: str | None = None,
        image: str | None = None,
        attachment: str | None = None,
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        body = {
            "session_id": session_id,
            "text": text,
            "folder": folder,
            "image": image,
            "attachment": attachment,
        }

        def write(db: Session) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            prior = self._claim(db, key, body)
            if prior is not None:
                return prior[0], prior[1], []
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That session does not exist.")
            cleaned = text.strip()
            if not cleaned:
                raise Rejected("A message is required.")
            if len(cleaned.encode()) > 8 * 1024 * 1024:
                raise Rejected("A message is at most 8 MiB.")
            if cleaned == CLEAR_COMMAND:
                result, events = self._clear_chat(db, session_id)
                self._remember(db, key or "", body, 200, result)
                return 200, result, events
            active = db.scalars(
                select(RunRow.id).where(
                    RunRow.session_id == session_id,
                    RunRow.status == "running",
                )
            ).first()
            if active is not None:
                raise Conflict("A reply is already in progress.")
            if cleaned == ARCHIVE_COMMAND:
                return self._begin_archive(db, key, body, chat)
            message_id = guid()
            run_id = guid()
            command_id = guid()
            created = now()
            actor = {"kind": "user", "id": "local"}
            db.add(
                MessageRow(
                    id=message_id,
                    session_id=session_id,
                    role="user",
                    body=cleaned,
                    actor_kind="user",
                    actor_id="local",
                    created_at=created,
                )
            )
            db.add(
                RunRow(
                    id=run_id,
                    session_id=session_id,
                    agent_id=chat.participant_id,
                    status="running",
                    error=None,
                    started_at=created,
                    finished_at=None,
                    gallery_folder=folder,
                    gallery_image=image,
                    gallery_attachment=attachment,
                )
            )
            stream_id = f"session:{session_id}"
            channel_id = chat.channel_id
            events = [
                self._event(
                    db,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload={"message_id": message_id, "role": "user", "text": cleaned},
                    channel_id=channel_id,
                    session_id=session_id,
                    actor=actor,
                    correlation_id=command_id,
                ),
                self._event(
                    db,
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
            self._remember(db, key or "", body, 202, result)
            return 202, result, events

        return self._write(write)

    def _begin_archive(
        self,
        db: Session,
        key: str | None,
        body: dict[str, Any],
        chat: SessionRow,
    ) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        session_id = chat.id
        run_id = guid()
        command_id = guid()
        created = now()
        db.add(
            RunRow(
                id=run_id,
                session_id=session_id,
                agent_id=chat.participant_id,
                status="running",
                error=None,
                started_at=created,
                finished_at=None,
                cue=ARCHIVE_CUE,
            )
        )
        actor = {"kind": "user", "id": "local"}
        events = [
            self._event(
                db,
                stream_id=f"session:{session_id}",
                event_type="run.started",
                payload={"status": "running"},
                channel_id=chat.channel_id,
                session_id=session_id,
                run_id=run_id,
                actor=actor,
                correlation_id=command_id,
            )
        ]
        result = {
            "command_id": command_id,
            "status": "accepted",
            "session_id": session_id,
            "run_id": run_id,
            "cleared": False,
        }
        self._remember(db, key or "", body, 202, result)
        return 202, result, events

    def reasoning_effort(self, agent_id: str | None) -> str | None:
        """Agent effort when set, otherwise the Providers effort."""

        def read(db: Session) -> str | None:
            agent = None
            if agent_id:
                row = db.get(AgentRow, agent_id)
                if row is not None:
                    agent = _agent_record(db, row)
            return _effective_effort(db, agent)

        return self._read(read)

    def chat_model(self, agent_id: str) -> tuple[str, str]:
        """Listener model id and base URL for an agent."""

        def read(db: Session) -> tuple[str, str]:
            return _chat_model(db, agent_id)

        return self._read(read)

    def gallery_vision_model(self, workspace_id: str | None = None) -> tuple[str, str]:
        """Listener model id and base URL for gallery vision on one workspace."""

        def read(db: Session) -> tuple[str, str]:
            return _gallery_vision_model(db, workspace_id)

        return self._read(read)

    def gallery_vision_effort(self, workspace_id: str | None = None) -> str | None:
        """Reasoning effort this workspace's agent would send. None when unset."""

        def read(db: Session) -> str | None:
            from web_api.chat import sampling_options

            agent_id = _workspace_agent_id(db, workspace_id)
            if not agent_id:
                return None
            row = db.get(AgentRow, agent_id)
            if row is None:
                return None
            agent = _agent_record(db, row)
            options = sampling_options(
                agent.get("enable_thinking"),
                agent.get("preserve_thinking"),
                _effective_effort(db, agent),
            )
            if not options:
                return None
            reasoning = options.get("reasoning")
            if not isinstance(reasoning, dict):
                return None
            effort = reasoning.get("effort")
            return effort if isinstance(effort, str) and effort else None

        return self._read(read)

    def run_context(self, run_id: str) -> dict[str, Any]:
        def read(db: Session) -> dict[str, Any]:
            run = db.get(RunRow, run_id)
            if run is None:
                raise NotFound("That run does not exist.")
            chat = db.get(SessionRow, run.session_id)
            agent_row = db.get(AgentRow, run.agent_id) if run.agent_id else None
            agent = _agent_record(db, agent_row) if agent_row is not None else None
            history = [
                {
                    "role": row.role,
                    "text": row.body,
                    "reasoning": row.reasoning or "",
                }
                for row in db.scalars(
                    select(MessageRow)
                    .where(
                        MessageRow.session_id == run.session_id,
                        MessageRow.role.in_(("user", "assistant")),
                    )
                    .order_by(_ROW_ID)
                )
            ]
            if agent is None:
                raise Rejected("Choose an enabled model in configuration.")
            model, base_url = _chat_model(db, str(agent["id"]))
            agent["reasoning_effort"] = _effective_effort(db, agent)
            cue = str(run.cue or "").strip()
            if cue:
                user_text = cue
                prior = history
            else:
                user_text = history[-1]["text"] if history else ""
                prior = history[:-1]
            folder = run.gallery_folder
            image = run.gallery_image
            attached = run.gallery_attachment
            return {
                "run": _view(run),
                "session": _session_view(chat),
                "agent": agent,
                "model": model,
                "base_url": base_url,
                "prior": prior,
                "user_text": user_text,
                "gallery_folder": None if folder is None else str(folder),
                "gallery_image": None if image is None else str(image),
                "gallery_attachment": None if attached is None else str(attached),
            }

        return self._read(read)

    def complete_run(
        self,
        run_id: str,
        text: str,
        reasoning: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        def write(db: Session) -> list[dict[str, Any]]:
            run = db.get(RunRow, run_id)
            if run is None or run.status != "running":
                return []
            chat = db.get(SessionRow, run.session_id)
            if str(run.cue or "") == ARCHIVE_CUE:
                return self._finish_archive_run(db, run, chat)
            message_id = guid()
            created = now()
            stored = reasoning.strip()
            calls = tool_calls or []
            parts = trace or []
            db.add(
                MessageRow(
                    id=message_id,
                    session_id=run.session_id,
                    role="assistant",
                    body=text,
                    actor_kind="agent",
                    actor_id=run.agent_id,
                    created_at=created,
                    reasoning=stored or None,
                    tool_calls=json.dumps(calls) if calls else None,
                    trace=json.dumps(parts) if parts else None,
                )
            )
            run.status = "completed"
            run.finished_at = created
            stream_id = f"session:{run.session_id}"
            actor = {"kind": "agent", "id": run.agent_id}
            payload: dict[str, Any] = {"message_id": message_id, "role": "assistant", "text": text}
            if stored:
                payload["reasoning"] = stored
            if calls:
                payload["tool_calls"] = calls
            if parts:
                payload["trace"] = parts
            return [
                self._event(
                    db,
                    stream_id=stream_id,
                    event_type="message.created",
                    payload=payload,
                    channel_id=chat.channel_id,
                    session_id=run.session_id,
                    run_id=run_id,
                    actor=actor,
                ),
                self._event(
                    db,
                    stream_id=stream_id,
                    event_type="run.completed",
                    payload={"status": "completed"},
                    channel_id=chat.channel_id,
                    session_id=run.session_id,
                    run_id=run_id,
                    actor=actor,
                ),
            ]

        return self._write(write)

    def finish_archive(self, run_id: str) -> list[dict[str, Any]] | None:
        """Clear an archive turn that stopped without text. Other runs stay."""

        def write(db: Session) -> list[dict[str, Any]] | None:
            run = db.get(RunRow, run_id)
            if run is None or run.status != "running":
                return None
            if str(run.cue or "") != ARCHIVE_CUE:
                return None
            chat = db.get(SessionRow, run.session_id)
            if chat is None:
                return None
            return self._finish_archive_run(db, run, chat)

        return self._write(write)

    def _finish_archive_run(
        self, db: Session, run: RunRow, chat: SessionRow
    ) -> list[dict[str, Any]]:
        created = now()
        run.status = "completed"
        run.finished_at = created
        actor = {"kind": "agent", "id": run.agent_id}
        events = [
            self._event(
                db,
                stream_id=f"session:{run.session_id}",
                event_type="run.completed",
                payload={"status": "archived"},
                channel_id=chat.channel_id,
                session_id=run.session_id,
                run_id=run.id,
                actor=actor,
            )
        ]
        _result, clear_events = self._clear_chat(db, chat.id)
        events.extend(clear_events)
        return events

    def finish_run(self, run_id: str, status: str, error: str | None) -> list[dict[str, Any]]:
        def write(db: Session) -> list[dict[str, Any]]:
            run = db.get(RunRow, run_id)
            if run is None or run.status != "running":
                return []
            chat = db.get(SessionRow, run.session_id)
            run.status = status
            run.error = error
            run.finished_at = now()
            event_type = "run.cancelled" if status == "cancelled" else "run.failed"
            payload: dict[str, Any] = {"status": status}
            if error:
                payload["message"] = error
            return [
                self._event(
                    db,
                    stream_id=f"session:{run.session_id}",
                    event_type=event_type,
                    payload=payload,
                    channel_id=chat.channel_id,
                    session_id=run.session_id,
                    run_id=run_id,
                    actor={"kind": "agent", "id": run.agent_id},
                )
            ]

        return self._write(write)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        def read(db: Session) -> dict[str, Any]:
            run = db.get(RunRow, run_id)
            if run is None:
                raise NotFound("That run does not exist.")
            return {"id": run.id, "status": run.status, "session_id": run.session_id}

        return self._read(read)

    def running_run_ids(self, session_id: str) -> list[str]:
        """Every run still marked running on this chat GUID."""

        def read(db: Session) -> list[str]:
            chat = db.get(SessionRow, session_id)
            if chat is None:
                raise NotFound("That session does not exist.")
            return list(
                db.scalars(
                    select(RunRow.id).where(
                        RunRow.session_id == session_id,
                        RunRow.status == "running",
                    )
                )
            )

        return self._read(read)

    def _require_model(self, db: Session, model_row: str) -> None:
        row = db.get(ProviderModelRow, model_row)
        if row is None or not row.enabled:
            raise Rejected("Choose an enabled model.")
