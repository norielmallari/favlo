"""Request bodies for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    persona: str = ""
    default_model_id: str | None = None


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


class HeartbeatInterval(BaseModel):
    minutes: int = Field(ge=1, le=1440)


class SettingsUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    default_model_id: str | None = None
    exchange_budget: int = Field(ge=1)


class MessageCreate(BaseModel):
    text: str = Field(min_length=1)


class McpGroupUpdate(BaseModel):
    id: str
    enabled: bool


class McpToolInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")


class McpCreate(BaseModel):
    agent_id: str
    name: str = Field(min_length=1, max_length=80)
    transport: str
    command: str = ""
    url: str = ""
    enabled: bool = True
    groups: list[McpGroupUpdate] | None = None
    tools: list[McpToolInput] | None = None


class McpUpdate(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    transport: str
    command: str = ""
    url: str = ""
    enabled: bool = True
    groups: list[McpGroupUpdate] | None = None
    tools: list[McpToolInput] | None = None


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
