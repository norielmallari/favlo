"""Open the web database from the single Alembic revision, then seed it."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from web_api.infrastructure.db.seed import seed
from web_api.infrastructure.db.session import make_engine, session_factory

_ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def prepare(path: Path) -> None:
    """Apply Alembic revisions before the store opens the file.

    This is the startup call. It applies pending revision files, the same role
    as EF Core ``Database.Migrate()``. It does not generate a revision.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _stamped(path):
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix) if suffix else path
            if candidate.exists():
                candidate.unlink()
    url = f"sqlite:///{path}"
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


def open_session(path: Path) -> Session:
    prepare(path)
    engine = make_engine(Path(path))
    session = session_factory(engine)()
    seed(session)
    return session


def _stamped(path: Path) -> bool:
    engine = make_engine(path)
    try:
        return inspect(engine).has_table("alembic_version")
    except DatabaseError:
        return False
    finally:
        engine.dispose()
