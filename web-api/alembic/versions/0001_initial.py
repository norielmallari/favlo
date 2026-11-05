"""Initial web schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from web_api.infrastructure.db.models import Base

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
