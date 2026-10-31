"""Request bodies for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    persona: str = ""
    default_model_id: str | None = None
    tool_groups: list[str] | None = None


class AgentUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    persona: str = ""
    default_model_id: str | None = None
    tool_groups: list[str] | None = None


class ModelChoice(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=1)


class ModelCreate(BaseModel):
    provider_id: str
    model_id: str = Field(min_length=1, max_length=200)
    display_name: str = ""


class ModelUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    enabled: bool


class SettingsUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    default_model_id: str | None = None
    exchange_budget: int = Field(ge=1)


class MessageCreate(BaseModel):
    text: str = Field(min_length=1)
    folder: str | None = None
    image: str | None = None


class McpGroupUpdate(BaseModel):
    id: str
    name: str = ""
    enabled: bool


class McpCreate(BaseModel):
    agent_id: str
    name: str = Field(min_length=1, max_length=80)
    transport: str
    command: str = ""
    url: str = ""
    enabled: bool = True
    groups: list[McpGroupUpdate] | None = None


class McpUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    transport: str
    command: str = ""
    url: str = ""
    enabled: bool = True
    groups: list[McpGroupUpdate] | None = None


class McpTest(BaseModel):
    transport: str
    command: str = ""
    url: str = ""


class HeartbeatCreate(BaseModel):
    agent_id: str
    cron_expression: str = Field(min_length=1, max_length=120)
    timezone: str = Field(min_length=1, max_length=80)
    instructions: str = Field(default="", max_length=8000)
    enabled: bool = True


class HeartbeatUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    cron_expression: str = Field(min_length=1, max_length=120)
    timezone: str = Field(min_length=1, max_length=80)
    instructions: str = Field(default="", max_length=8000)
    enabled: bool = True


class WorkspaceCreate(BaseModel):
    channel_id: str
    name: str = Field(min_length=1, max_length=80)
    type: str = "gallery"


class FolderRecord(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: str
    provider: str | None = None
    location: str | None = None
    remote_id: str | None = None
    parent_id: str | None = None


class FolderRecordUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, max_length=80)
    parent_id: str | None = None
    move: bool = False


class FocusUpdate(BaseModel):
    folder_id: str | None = None


class ImageMove(BaseModel):
    path: str = Field(min_length=1)
    to: str = Field(min_length=1)


class ImageDelete(BaseModel):
    path: str = Field(min_length=1)


class SkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=8000)
    definition: str = Field(default="", max_length=32000)


class AgentSkillCreate(BaseModel):
    agent_id: str
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=8000)
    definition: str = Field(default="", max_length=32000)


class GlobalSkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=8000)
    definition: str = Field(default="", max_length=32000)
    add_by_default: bool = False


class SkillCopy(BaseModel):
    agent_id: str | None = None
    workspace_id: str | None = None


class SkillUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=8000)
    definition: str = Field(default="", max_length=32000)
    add_by_default: bool | None = None


class ImageWrite(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(min_length=1)
    data: str | None = Field(default=None, alias="bytes")
    source: str | None = None
    note: dict[str, Any] | None = None
    expected_revision: str | None = None


class FolderCreate(BaseModel):
    path: str = Field(min_length=1)


class FolderPatch(BaseModel):
    path: str = Field(min_length=1)
    name: str | None = None
    to: str | None = None


class FolderDelete(BaseModel):
    path: str = Field(min_length=1)
