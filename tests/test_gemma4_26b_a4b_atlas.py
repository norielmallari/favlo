"""Atlas slot counts. No device."""

from pathlib import Path

from gemma4_26b_a4b import atlas as atlas_mod
from gemma4_26b_a4b.atlas import ExpertAtlas, kv_bytes, kv_mode, layer_slots, plain_cache


def test_kv_mode_defaults_to_fp8() -> None:
    assert kv_mode(None) == "fp8_e4m3"
    assert kv_mode("") == "fp8_e4m3"
    assert kv_mode("fp8_e4m3") == "fp8_e4m3"
    assert kv_mode("fp8") == "fp8_e4m3"
    assert kv_mode("e4m3") == "fp8_e4m3"
    assert kv_mode("half") == "half"
    assert kv_mode("f16") == "half"
    assert kv_mode("fp16") == "half"
    assert plain_cache("half") is True
    assert plain_cache("fp8_e4m3") is False
    try:
        kv_mode("f32")
    except ValueError as exc:
        assert "f32" in str(exc)
    else:
        raise AssertionError("f32 must be rejected")


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
    assert kv_bytes(1, "fp8_e4m3") < kv_bytes(1, "half")


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

    def allocate(owner: str, nbytes: int, *, zero: bool = False) -> int:
        del owner, zero
        nonlocal next_id
        handle = next_id
        next_id += 1
        live[handle] = bytearray(nbytes)
        return handle

    def release(handle: int) -> None:
        live.pop(handle)

    def read(handle: int, path: Path) -> None:
        live[handle][:] = path.read_bytes()

    monkeypatch.setattr(atlas_mod, "device_buffer", allocate)
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
    resident.admit(session, layer_bytes * 4)
    assert resident.slots == 4
    assert resident.complete is False
    resident.admit(session, layer_bytes)
    assert resident.slots == 4
