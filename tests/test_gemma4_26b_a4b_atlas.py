"""Atlas slot counts. No device."""

from pathlib import Path

from gemma4_26b_a4b import atlas as atlas_mod
from gemma4_26b_a4b.atlas import ExpertAtlas, kv_bytes, layer_slots


def test_one_layer_is_the_minimum() -> None:
    assert layer_slots(0, 100, 30) == 0
    assert layer_slots(99, 100, 30) == 0
    assert layer_slots(100, 100, 30) == 1
    assert layer_slots(1000, 100, 30) == 10
    assert layer_slots(10_000, 100, 30) == 30


def test_kv_grows_with_capacity() -> None:
    assert kv_bytes(0) == 0
    assert kv_bytes(256) == 256 * kv_bytes(1)
    assert kv_bytes(1) > 0


class _Session:
    def __init__(self) -> None:
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


def test_a_short_budget_reuses_the_oldest_layer(tmp_path: Path, monkeypatch) -> None:
    layer_bytes = 64
    experts = tmp_path / "experts"
    experts.mkdir()
    for layer in range(4):
        (experts / f"{layer}.bin").write_bytes(bytes([layer + 1]) * layer_bytes)
    live: dict[int, bytearray] = {}
    next_id = 0

    def allocate(nbytes: int, *, owner: str) -> int:
        del owner
        nonlocal next_id
        handle = next_id
        next_id += 1
        live[handle] = bytearray(nbytes)
        return handle

    def release(handle: int) -> None:
        live.pop(handle)

    def read(handle: int, path: Path) -> None:
        live[handle][:] = path.read_bytes()

    monkeypatch.setattr(atlas_mod.memory, "allocate", allocate)
    monkeypatch.setattr(atlas_mod.memory, "release", release)
    resident = ExpertAtlas("owner", tmp_path, layer_bytes, 3, read)
    resident.load(lambda _line: None)
    assert resident.slots == 3
    assert resident.complete is False
    session = _Session()
    resident.retune(session, layer_bytes)
    assert session.flushed == 1
    assert resident.slots == 1
    assert resident.complete is False
    kept = next(iter(resident._files))
    handle = resident.bind(session, 3)
    assert session.flushed == 2
    assert kept not in resident._files
    assert 3 in resident._files
    assert live[handle][0] == 4
