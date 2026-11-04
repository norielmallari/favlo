"""SQLite tables for the web app. Infrastructure only."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProviderRow(Base):
    __tablename__ = "provider"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    base_url: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class ProviderModelRow(Base):
    __tablename__ = "provider_model"
    __table_args__ = (UniqueConstraint("provider_id", "model_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    provider_id: Mapped[str] = mapped_column(String, nullable=False)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class SettingsRow(Base):
    __tablename__ = "settings"
    __table_args__ = (CheckConstraint("id = 1"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    default_model_id: Mapped[str | None] = mapped_column(String)
    gallery_vision_model_id: Mapped[str | None] = mapped_column(String)
    reasoning_effort: Mapped[str | None] = mapped_column(String)
    exchange_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class AgentRow(Base):
    __tablename__ = "agent"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    default_model_id: Mapped[str | None] = mapped_column(String)
    enable_thinking: Mapped[int | None] = mapped_column(Integer)
    preserve_thinking: Mapped[int | None] = mapped_column(Integer)
    reasoning_effort: Mapped[str | None] = mapped_column(String)
    expand_thinking: Mapped[str | None] = mapped_column(String)
    expand_tools: Mapped[str | None] = mapped_column(String)
    compaction_percent: Mapped[int | None] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class ChannelRow(Base):
    __tablename__ = "channel"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class ChannelMemberRow(Base):
    __tablename__ = "channel_member"

    channel_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, primary_key=True)


class SessionRow(Base):
    __tablename__ = "session"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    channel_id: Mapped[str | None] = mapped_column(String)
    participant_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String)
    maf_state: Mapped[str | None] = mapped_column(Text)
    definition: Mapped[str | None] = mapped_column(Text)
    definition_hash: Mapped[str | None] = mapped_column(String)
    agui_state: Mapped[str | None] = mapped_column(Text)
    loaded_mcp: Mapped[str | None] = mapped_column(Text)


class MessageRow(Base):
    __tablename__ = "message"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    actor_kind: Mapped[str] = mapped_column(String, nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[str | None] = mapped_column(Text)
    trace: Mapped[str | None] = mapped_column(Text)


class RunRow(Base):
    __tablename__ = "run"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[str | None] = mapped_column(String)
    finished_at: Mapped[str | None] = mapped_column(String)
    cue: Mapped[str | None] = mapped_column(Text)
    gallery_folder: Mapped[str | None] = mapped_column(Text)
    gallery_image: Mapped[str | None] = mapped_column(Text)
    gallery_attachment: Mapped[str | None] = mapped_column(Text)


class EventRow(Base):
    __tablename__ = "event"
    __table_args__ = (UniqueConstraint("stream_id", "sequence"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    stream_id: Mapped[str] = mapped_column(String, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)


class CommandRow(Base):
    __tablename__ = "command"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    body_hash: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(Text, nullable=False)


class AgentToolGroupRow(Base):
    __tablename__ = "agent_tool_group"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    group_id: Mapped[str] = mapped_column(String, primary_key=True)


class McpServerRow(Base):
    __tablename__ = "mcp_server"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    transport: Mapped[str] = mapped_column(String, nullable=False)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    status_message: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    group_switches: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class McpTaskRow(Base):
    __tablename__ = "mcp_task"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    remote_id: Mapped[str] = mapped_column(String, nullable=False)
    server_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    poll_interval_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class HeartbeatRow(Base):
    __tablename__ = "heartbeat"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String, nullable=False)
    timezone: Mapped[str] = mapped_column(String, nullable=False)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String)
    fired_minute: Mapped[str | None] = mapped_column(String)


class MetaRow(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class ShellRow(Base):
    __tablename__ = "shell"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    icon: Mapped[str] = mapped_column(String, nullable=False)
    setup: Mapped[str] = mapped_column(String, nullable=False)
    setup_hint: Mapped[str] = mapped_column(Text, nullable=False)
    slots: Mapped[str] = mapped_column(Text, nullable=False)
    renderer: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    sort: Mapped[int] = mapped_column(Integer, nullable=False)


class CatalogElementRow(Base):
    __tablename__ = "catalog_element"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    slots: Mapped[str] = mapped_column(Text, nullable=False)
    max_count: Mapped[int] = mapped_column(Integer, nullable=False)
    props_schema: Mapped[str] = mapped_column(Text, nullable=False)
    gestures: Mapped[str] = mapped_column(Text, nullable=False)


class ShellElementRow(Base):
    __tablename__ = "shell_element"
    __table_args__ = (UniqueConstraint("shell_id", "element_name"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    shell_id: Mapped[str] = mapped_column(String, nullable=False)
    element_name: Mapped[str] = mapped_column(String, nullable=False)
    slot: Mapped[str] = mapped_column(String, nullable=False)
    sort: Mapped[int] = mapped_column(Integer, nullable=False)
    initial_props: Mapped[str] = mapped_column(Text, nullable=False)


class WorkspaceRow(Base):
    __tablename__ = "workspace"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    provisioning_state: Mapped[str] = mapped_column(String, nullable=False, default="new")
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    focus_folder_id: Mapped[str | None] = mapped_column(String)
    compaction_percent: Mapped[int | None] = mapped_column(Integer)


class FolderRow(Base):
    __tablename__ = "folder"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str | None] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(Text)
    machine_id: Mapped[str] = mapped_column(String, nullable=False)
    remote_id: Mapped[str | None] = mapped_column(String)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class SkillRow(Base):
    __tablename__ = "skill"
    __table_args__ = (UniqueConstraint("workspace_id", "name"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class AgentSkillRow(Base):
    __tablename__ = "agent_skill"
    __table_args__ = (UniqueConstraint("agent_id", "name"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class ThreadSnapshotRow(Base):
    """One AG-UI thread snapshot. The library owns the document shape."""

    __tablename__ = "thread_snapshot"
    __table_args__ = (UniqueConstraint("scope", "thread_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    thread_id: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class GlobalSkillRow(Base):
    __tablename__ = "global_skill"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    add_by_default: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

