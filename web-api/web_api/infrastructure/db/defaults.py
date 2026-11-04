"""Copy global catalog rows onto a new agent."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from web_api.ids import guid
from web_api.infrastructure.db.models import (
    AgentSkillRow,
    GlobalMcpServerRow,
    GlobalSkillRow,
    McpServerRow,
)
from web_api.mcp_groups import default_switch_payload


def _copied_switches(row: GlobalMcpServerRow) -> str:
    """A new agent starts with every Favlo media tool on."""
    if row.name == "Favlo media":
        return default_switch_payload()
    return row.group_switches or "{}"


def copy_default_skills(db: Session, agent_id: str, created: str) -> None:
    rows = db.scalars(
        select(GlobalSkillRow)
        .where(GlobalSkillRow.add_by_default == 1)
        .order_by(GlobalSkillRow.name.collate("NOCASE"), GlobalSkillRow.id)
    )
    for row in rows:
        taken = db.scalars(
            select(AgentSkillRow.id).where(
                AgentSkillRow.agent_id == agent_id,
                AgentSkillRow.name == row.name,
            )
        ).first()
        if taken is not None:
            continue
        db.add(
            AgentSkillRow(
                id=guid(),
                agent_id=agent_id,
                name=row.name,
                description=row.description,
                definition=row.definition,
                revision=1,
                created_at=created,
            )
        )


def copy_default_mcp(
    db: Session,
    agent_id: str,
    *,
    prefer_ids: dict[str, str] | None = None,
) -> None:
    preferred = prefer_ids or {}
    rows = db.scalars(
        select(GlobalMcpServerRow)
        .where(GlobalMcpServerRow.add_by_default == 1)
        .order_by(GlobalMcpServerRow.name.collate("NOCASE"), GlobalMcpServerRow.id)
    )
    for row in rows:
        taken = db.scalars(
            select(McpServerRow.id).where(
                McpServerRow.agent_id == agent_id,
                McpServerRow.name == row.name,
            )
        ).first()
        if taken is not None:
            continue
        db.add(
            McpServerRow(
                id=preferred.get(row.name) or guid(),
                agent_id=agent_id,
                name=row.name,
                transport=row.transport,
                command=row.command,
                url=row.url,
                enabled=int(row.enabled),
                status="unchecked",
                status_message="",
                revision=1,
                group_switches=_copied_switches(row),
            )
        )
