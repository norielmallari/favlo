"""Rows a new database contains. An existing row is left as it is."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from favlo.config_tools import GROUP_ID
from favlo.persona import PERSONA
from web_api.events import now
from web_api.global_skill_catalog import catalog_skills
from web_api.ids import (
    FAVLO_AGENT_ID,
    FAVLO_CHANNEL_ID,
    FAVLO_GLOBAL_MCP_ID,
    FAVLO_MCP_ID,
    LOCAL_PROVIDER_ID,
    guid,
)
from web_api.infrastructure.db.defaults import copy_default_mcp, copy_default_skills
from web_api.infrastructure.db.models import (
    AgentRow,
    AgentToolGroupRow,
    CatalogElementRow,
    ChannelMemberRow,
    ChannelRow,
    GlobalMcpServerRow,
    GlobalSkillRow,
    MetaRow,
    ProviderRow,
    SessionRow,
    SettingsRow,
    ShellElementRow,
    ShellRow,
)
from web_api.infrastructure.db.session import LOCAL_BASE_URL, LOCAL_MCP_URL
from web_api.mcp_groups import default_switch_payload
from web_api.shell_catalog import catalog_elements, dump, shell_elements, shells


def seed(session: Session) -> None:
    _seed_shells(session)
    if session.get(SettingsRow, 1) is None:
        session.add(
            SettingsRow(
                id=1,
                default_model_id=None,
                gallery_vision_model_id=None,
                exchange_budget=10,
                revision=1,
            )
        )
    if session.get(ProviderRow, LOCAL_PROVIDER_ID) is None:
        session.add(
            ProviderRow(
                id=LOCAL_PROVIDER_ID,
                name="Favlo",
                type="openai_compatible",
                base_url=LOCAL_BASE_URL,
                enabled=1,
                revision=1,
            )
        )
    created = now()
    _seed_global_skills(session, created)
    _seed_global_mcp(session, created)
    if session.get(AgentRow, FAVLO_AGENT_ID) is None:
        session.add(
            AgentRow(
                id=FAVLO_AGENT_ID,
                name="Favlo",
                persona=PERSONA,
                default_model_id=None,
                revision=1,
            )
        )
        session.add(ChannelRow(id=FAVLO_CHANNEL_ID, kind="agent", name="Favlo", revision=1))
        session.add(ChannelMemberRow(channel_id=FAVLO_CHANNEL_ID, agent_id=FAVLO_AGENT_ID))
        session.add(
            SessionRow(
                id=guid(),
                kind="channel",
                channel_id=FAVLO_CHANNEL_ID,
                participant_id=FAVLO_AGENT_ID,
                created_at=created,
            )
        )
        copy_default_skills(session, FAVLO_AGENT_ID, created)
        copy_default_mcp(session, FAVLO_AGENT_ID, prefer_ids={"Favlo media": FAVLO_MCP_ID})
    if session.get(MetaRow, "favlo_config_seed") is None:
        if session.get(AgentRow, FAVLO_AGENT_ID) is not None:
            session.add(AgentToolGroupRow(agent_id=FAVLO_AGENT_ID, group_id=GROUP_ID))
        session.add(MetaRow(key="favlo_config_seed", value="1"))
    session.commit()


def _seed_global_skills(session: Session, created: str) -> None:
    for name, description, definition in catalog_skills():
        mark = f"global_skill_seed:{name}"
        if session.get(MetaRow, mark) is not None:
            continue
        if session.scalars(select(GlobalSkillRow).where(GlobalSkillRow.name == name)).first() is None:
            session.add(
                GlobalSkillRow(
                    id=guid(),
                    name=name,
                    description=description,
                    definition=definition,
                    add_by_default=1,
                    revision=1,
                    created_at=created,
                )
            )
        session.add(MetaRow(key=mark, value="1"))


def _seed_global_mcp(session: Session, created: str) -> None:
    if session.get(MetaRow, "global_mcp_seed:Favlo media") is not None:
        return
    if session.get(GlobalMcpServerRow, FAVLO_GLOBAL_MCP_ID) is None:
        taken = session.scalars(
            select(GlobalMcpServerRow).where(GlobalMcpServerRow.name == "Favlo media")
        ).first()
        if taken is None:
            session.add(
                GlobalMcpServerRow(
                    id=FAVLO_GLOBAL_MCP_ID,
                    name="Favlo media",
                    transport="http",
                    command="",
                    url=LOCAL_MCP_URL,
                    enabled=1,
                    add_by_default=1,
                    revision=1,
                    group_switches=default_switch_payload(),
                    created_at=created,
                )
            )
    session.add(MetaRow(key="global_mcp_seed:Favlo media", value="1"))


def _seed_shells(session: Session) -> None:
    for row in shells():
        if session.get(ShellRow, row["id"]) is not None:
            continue
        session.add(
            ShellRow(
                id=row["id"],
                label=row["label"],
                icon=row["icon"],
                setup=row["setup"],
                setup_hint=row["setup_hint"],
                slots=dump(row["slots"]),
                renderer=row["renderer"],
                version=row["version"],
                sort=row["sort"],
            )
        )
    for row in catalog_elements():
        if session.get(CatalogElementRow, row["name"]) is not None:
            continue
        session.add(
            CatalogElementRow(
                name=row["name"],
                version=row["version"],
                slots=dump(row["slots"]),
                max_count=row["max_count"],
                props_schema=dump(row["props_schema"]),
                gestures=dump(row["gestures"]),
            )
        )
    for row in shell_elements():
        taken = session.scalars(
            select(ShellElementRow).where(
                ShellElementRow.shell_id == row["shell_id"],
                ShellElementRow.element_name == row["element_name"],
            )
        ).first()
        if taken is not None:
            continue
        session.add(
            ShellElementRow(
                id=guid(),
                shell_id=row["shell_id"],
                element_name=row["element_name"],
                slot=row["slot"],
                sort=row["sort"],
                initial_props=dump(row["initial_props"]),
            )
        )
