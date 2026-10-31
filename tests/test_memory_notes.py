"""Host tests for the shared memory store."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from favlo.memory_notes import DIM, EMBED_MODEL, MemoryStore


def _unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


def _embed_map(text: str) -> list[float]:
    if text == "north":
        return _unit(0)
    if text == "east":
        return _unit(1)
    if text == "south":
        return _unit(2)
    if text == "west":
        return _unit(3)
    if text == "alpha bravo":
        return _unit(4)
    if text == "charlie delta":
        return _unit(5)
    if text == "echo foxtrot":
        return _unit(6)
    if text == "golf hotel":
        return _unit(7)
    if text == "india juliet":
        return _unit(8)
    if "bravo" in text.lower():
        return _unit(4)
    return _unit(9)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def test_remember_and_duplicate(store: MemoryStore) -> None:
    first = store.remember("Likes tea.", _embed_map("tea"))
    second = store.remember("Likes tea.", _embed_map("tea"))
    assert first == second
    assert store.remember("Likes coffee.", _embed_map("coffee")) != first


def test_forget_drops_vector(store: MemoryStore) -> None:
    mem_id = store.remember("Old fact.", _embed_map("old"))
    store.forget(mem_id)
    rows = store.recall("old", _embed_map("old"))
    assert rows == []
    with store._lock:
        db = store._connect()
        try:
            count = db.execute("select count(*) from observation_vec").fetchone()[0]
        finally:
            db.close()
    assert count == 0


def test_recall_ranks_by_inner_product(store: MemoryStore) -> None:
    store.remember("north star", _unit(0))
    store.remember("east wind", _unit(1))
    rows = store.recall("north", _unit(0))
    assert rows[0]["text"] == "north star"
    assert rows[0]["score"] == pytest.approx(1.0)


def test_fts_supplements_knn(store: MemoryStore) -> None:
    for label in (
        "alpha bravo",
        "charlie delta",
        "echo foxtrot",
        "golf hotel",
        "india juliet",
    ):
        store.remember(label, _embed_map(label))
    rows = store.recall("bravo", _embed_map("query"))
    texts = {row["text"] for row in rows}
    assert "alpha bravo" in texts


def test_model_change_reembeds(store: MemoryStore) -> None:
    mem_id = store.remember("north star", _unit(0))
    with store._lock:
        db = store._connect()
        try:
            db.execute(
                "update meta set value = ? where key = 'model'",
                ("other-model",),
            )
            db.commit()
        finally:
            db.close()
    rows = store.recall("north", _unit(0), reembed=_embed_map)
    assert rows and rows[0]["id"] == mem_id
    with store._lock:
        db = store._connect()
        try:
            model = db.execute("select value from meta where key = 'model'").fetchone()[0]
        finally:
            db.close()
    assert model == EMBED_MODEL


def test_concurrent_remember(store: MemoryStore) -> None:
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            store.remember(f"fact {index}", _unit(index % DIM))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    with store._lock:
        db = store._connect()
        try:
            count = db.execute(
                "select count(*) from observation where deleted is null",
            ).fetchone()[0]
        finally:
            db.close()
    assert count == 8


def test_recall_and_forget_stay_with_one_client(store: MemoryStore) -> None:
    own = store.remember("Likes tea.", _unit(0), client="agent_a")
    other = store.remember("Likes tea.", _unit(0), client="agent_b")
    assert own != other
    rows = store.recall("tea", _unit(0), client="agent_a")
    assert [row["id"] for row in rows] == [own]
    with pytest.raises(KeyError):
        store.forget(other, client="agent_a")
    still = store.recall("tea", _unit(0), client="agent_b")
    assert [row["id"] for row in still] == [other]


def test_mcp_lists_memory_tools() -> None:
    import asyncio

    from favlo.mcp_server import build_mcp

    mcp = build_mcp(lambda *_args: {}, {}, lambda _text: _unit(0))
    tools = asyncio.run(mcp.list_tools(run_middleware=False))
    names = {tool.name for tool in tools}
    assert {"remember", "recall", "forget", "generate_image"}.issubset(names)
