"""Plane, promotion, catalog, and prefix checks for Gemma 4 26B-A4B."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from favlo.prefix import PrefixHeader, RegionRef
from gemma4_26b_a4b.metal41.checkpoint import load_tensor, tensor_names
from gemma4_26b_a4b.metal41.geometry import GLOBAL_LAYERS, expected, local_layers
from gemma4_26b_a4b.metal41.planes import (
    align_k,
    expert_bytes,
    expert_planes,
    pack_matrix,
    relative_rms,
    restore_matrix,
    scale_stride,
)
from gemma4_26b_a4b.metal41.prefix import (
    CacheAdapter,
    global_nbytes,
    local_nbytes,
    pack_global,
    pack_head_major,
    pack_local,
)
from gemma4_26b_a4b.metal41.promote import promotion_count, select_promoted


def test_fp8_roundtrip_and_zero_block() -> None:
    weight = np.zeros((4, 128), dtype=np.float32)
    weight[0, 0] = 1.0
    weight[1, 17] = -0.5
    restored = restore_matrix(pack_matrix(weight, "fp8"), 4, 128, "fp8")
    assert restored.shape == weight.shape
    assert restored[2, 0] == 0.0
    assert relative_rms(weight, restored) < 0.05
    zeros = np.zeros((2, 64), dtype=np.float32)
    assert np.all(restore_matrix(pack_matrix(zeros, "fp8"), 2, 64, "fp8") == 0.0)


def test_fp4_midpoint_keeps_the_even_magnitude() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 3.5
    restored = restore_matrix(pack_matrix(weight, "fp4"), 1, 32, "fp4")
    assert restored[0, 0] == 4.0


def test_fp4_scale_search_beats_the_peak_scale() -> None:
    weight = np.full((1, 32), 0.35, dtype=np.float32)
    weight[0, 0] = 6.2
    packed = pack_matrix(weight, "fp4")
    restored = restore_matrix(packed, 1, 32, "fp4")
    peak = float(np.max(np.abs(weight)))
    coarse = 2.0 ** np.ceil(np.log2(max(peak / 6.0, np.finfo(np.float64).tiny)))
    unit = weight / coarse
    edges = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    index = np.clip(np.digitize(np.abs(unit), edges, right=True), 0, 7)
    magnitude = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[index]
    peak_restore = np.sign(weight) * magnitude * coarse
    assert relative_rms(weight, restored) <= relative_rms(weight, peak_restore)


def test_fp4_low_nibble_is_element_zero() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 0.5
    weight[0, 1] = 1.0
    blob = pack_matrix(weight, "fp4")
    first = blob[0]
    assert first & 0x0F != 0
    restored = restore_matrix(blob, 1, 32, "fp4")
    assert abs(float(restored[0, 0]) - 0.5) < 0.26
    assert abs(float(restored[0, 1]) - 1.0) < 0.26


def test_q4_0_pairs_the_block_ends() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 8.0
    weight[0, 16] = -4.0
    blob = pack_matrix(weight, "q4_0")
    assert len(blob) == 18
    pair = blob[2]
    assert pair & 0x0F == 0
    assert pair >> 4 == 12
    restored = restore_matrix(blob, 1, 32, "q4_0")
    assert abs(float(restored[0, 0]) - 8.0) < 1e-5
    assert abs(float(restored[0, 16]) + 4.0) < 1e-5
    zeros = np.zeros((2, 64), dtype=np.float32)
    assert np.all(restore_matrix(pack_matrix(zeros, "q4_0"), 2, 64, "q4_0") == 0.0)


def test_q4_0_recovers_a_qat_step() -> None:
    step = np.float32(2**-6)
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 3.0 * step
    weight[0, 1] = -2.0 * step
    weight[0, 16] = step
    restored = restore_matrix(pack_matrix(weight, "q4_0"), 1, 32, "q4_0")
    assert np.max(np.abs(restored - weight)) < 1e-6


def test_k_pad_and_tight_scale_row() -> None:
    assert align_k(704, "fp4") == 768
    assert align_k(704, "fp8") == 768
    assert align_k(2112, "fp4") == 2304
    assert align_k(2112, "fp8") == 2176
    assert align_k(1152, "fp8") == 1152
    assert align_k(1152, "fp4") == 1280
    assert align_k(704, "q4_0") == 704
    assert align_k(4304, "q4_0") == 4320
    assert scale_stride(2816) == 2816 // 32
    assert scale_stride(1024) == 1024 // 32


def test_expert_images_start_on_a_128_byte_boundary() -> None:
    for plane in ("fp4", "fp8"):
        parts = expert_planes(plane)
        assert all(part % 128 == 0 for part in parts)
    assert expert_bytes(4) == expert_planes("fp4")[-1]
    assert expert_bytes(8) == expert_planes("fp8")[-1]
    assert expert_bytes(8) > expert_bytes(4)


def test_promotion_count_and_tie_break() -> None:
    assert promotion_count(3840, 20) == 768
    assert promotion_count(3840, 0) == 0
    scores = [(1.0, 2, 0), (1.0, 1, 4), (9.0, 0, 1), (1.0, 1, 3)]
    chosen = select_promoted(scores, 75)
    assert chosen[0] == (0, 1)
    assert chosen[1] == (1, 3)
    assert chosen[2] == (1, 4)


def test_catalog_matches_the_checkpoint_map() -> None:
    catalog = expected()
    assert len(catalog) == 1013
    ups = [name for name, (role, _shape) in catalog.items() if role == "expert-up"]
    assert len(ups) == 30
    local = []
    missing_v = []
    for layer in range(30):
        name = f"model.language_model.layers.{layer}.self_attn.v_proj.weight"
        if layer in GLOBAL_LAYERS:
            assert name not in catalog
            missing_v.append(layer)
        else:
            assert name in catalog
            local.append(layer)
    assert len(local) == 25
    assert missing_v == list(GLOBAL_LAYERS)
    assert len(local_layers()) == 25


def test_bf16_reader_slices_one_expert(tmp_path: Path) -> None:
    values = np.array([[[1.0, -2.0], [0.5, 4.0]], [[3.0, 0.0], [-1.0, 8.0]]], dtype=np.float32)
    raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    header = {
        "experts": {
            "dtype": "BF16",
            "shape": [2, 2, 2],
            "data_offsets": [0, len(raw)],
        }
    }
    payload = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(payload)) + payload + raw)
    names = tensor_names(tmp_path)
    loaded = load_tensor(tmp_path, names["experts"], "experts", 1)
    assert loaded.shape == (2, 2)
    assert abs(float(loaded[0, 0]) - 3.0) < 0.05


def test_head_major_pack_matches_the_host_planes() -> None:
    count = 4
    rng = np.random.default_rng(0)
    local_k = rng.standard_normal((25, 8, count, 256)).astype(np.float16)
    local_v = rng.standard_normal((25, 8, count, 256)).astype(np.float16)
    global_k = rng.standard_normal((5, 2, count, 512)).astype(np.float16)
    global_v = rng.standard_normal((5, 2, count, 512)).astype(np.float16)
    assert _packed_layers(local_k) + _packed_layers(local_v) == pack_local(local_k, local_v, count)
    assert _packed_layers(global_k) + _packed_layers(global_v) == pack_global(
        global_k, global_v, count
    )


def _packed_layers(plane: np.ndarray) -> bytes:
    layers, heads, count, dim = plane.shape
    parts: list[bytes] = []
    for layer in range(layers):
        packed = np.ascontiguousarray(plane[layer]).tobytes()
        parts.append(pack_head_major(packed, heads, dim, count))
    return b"".join(parts)


def test_prefix_pack_length_and_exact_file_refusal() -> None:
    count = 4
    keys = np.zeros((25, 8, count, 256), dtype=np.float16)
    values = np.zeros((25, 8, count, 256), dtype=np.float16)
    global_keys = np.zeros((5, 2, count, 512), dtype=np.float16)
    global_values = np.zeros((5, 2, count, 512), dtype=np.float16)
    local = pack_local(keys, values, count)
    packed_global = pack_global(global_keys, global_values, count)
    assert len(local) == local_nbytes(count)
    assert len(packed_global) == global_nbytes(count)
    header = PrefixHeader(
        path=Path("prefix.bin"),
        model="gemma4_26b_a4b",
        digest=b"\0" * 32,
        tag="half",
        tokens=(1, 2, 3, 4),
        regions=(
            RegionRef("local", 0, len(local)),
            RegionRef("global", len(local), len(packed_global)),
        ),
        file_size=len(local) + len(packed_global),
    )
    adapter = CacheAdapter.__new__(CacheAdapter)
    assert adapter.accept(header, [1, 2, 3, 4]) is False
    assert adapter.accept(header, [1, 2, 3, 4, 5]) is True
    assert adapter.accept(header, [9, 2, 3, 4, 5]) is False


def test_has_vision() -> None:
    from gemma4_26b_a4b.metal41.chat import Pipeline

    assert Pipeline().has_vision() is True


def test_vision_disk_cache_uses_the_shared_helper(tmp_path: Path, monkeypatch) -> None:
    from favlo import vision_cache
    from gemma4_26b_a4b.metal41.geometry import HIDDEN
    from gemma4_26b_a4b.metal41 import vision

    monkeypatch.setattr(vision_cache, "cache_dir", lambda pipeline: tmp_path / pipeline)
    root = tmp_path / "record"
    digest = b"\x01" * 32
    blob = bytes(HIDDEN * 4)
    vision._store_vision_disk(root, digest, blob, 1, 4)
    assert vision._load_vision_disk(root, digest) == (blob, 1, 4)
    path = vision_cache.cache_path("gemma4_26b_a4b", str(root), digest, 0, 0)
    assert path.is_file()
    assert vision_cache.cache_path("gemma4_26b_a4b", str(root), digest, 1, 2) != path


def _spans() -> tuple[dict[int, bytearray], object, object]:
    spans: dict[int, bytearray] = {}
    issued = 0

    def allocate(owner: str, nbytes: int, *, zero: bool = False) -> int:
        nonlocal issued
        del owner, zero
        issued += 1
        spans[issued] = bytearray(nbytes)
        return issued

    def map_span(handle: int, offset: int, nbytes: int) -> memoryview:
        return memoryview(spans[handle])[offset : offset + nbytes]

    return spans, allocate, map_span


def test_vision_rows_fill_the_bound_span(tmp_path: Path, monkeypatch) -> None:
    from gemma4_26b_a4b.metal41 import forward, vision
    from gemma4_26b_a4b.metal41.geometry import VISION_HEAD, VISION_LAYERS, VISION_WIDTH

    specs: list[tuple[str, str, list[int], int]] = [
        ("model.vision_tower.patch_embedder.input_proj.weight", "fp8", [1, 128], 32),
        ("model.embed_vision.embedding_projection.weight", "fp8", [1, 128], 32),
        ("model.vision_tower.std_scale", "half", [4], 8),
        ("model.vision_tower.std_bias", "half", [4], 8),
    ]
    for layer in range(VISION_LAYERS):
        base = f"model.vision_tower.encoder.layers.{layer}"
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ):
            specs.append((f"{base}.{name}", "half", [4], 8))
        for name in (
            "self_attn.q_proj.linear.weight",
            "self_attn.k_proj.linear.weight",
            "self_attn.v_proj.linear.weight",
            "self_attn.o_proj.linear.weight",
            "mlp.gate_proj.linear.weight",
            "mlp.up_proj.linear.weight",
            "mlp.down_proj.linear.weight",
        ):
            specs.append((f"{base}.{name}", "fp8", [1, 128], 32))
    rows: list[dict[str, object]] = []
    blob = bytearray()
    for name, plane, shape, nbytes in specs:
        rows.append(
            {
                "name": name,
                "plane": plane,
                "shape": shape,
                "nbytes": nbytes,
                "offset": len(blob),
            }
        )
        blob.extend(bytes((len(blob) + index) % 251 for index in range(nbytes)))
    pos_offset = len(blob)
    rows.append(
        {
            "name": "model.vision_tower.patch_embedder.position_embedding_table",
            "plane": "half",
            "shape": [2, vision._POS_ROWS, 4],
            "nbytes": 8,
            "offset": pos_offset,
        }
    )
    (tmp_path / "resident.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "resident.bin").write_bytes(blob)
    spans, allocate, map_span = _spans()
    released: dict[int, bytes] = {}
    stored: list[bytes] = []
    starts: list[int] = []
    real_fill = forward._fill

    def spy(raw: object, view: memoryview, start: int) -> None:
        starts.append(start)
        real_fill(raw, view, start)

    def release(handle: int) -> None:
        released[handle] = bytes(spans.pop(handle))

    class _Session:
        def __init__(self) -> None:
            self.dispatches: list[tuple[object, object]] = []
            self.encoded: list[str] = []

        def zeros(self, nbytes: int) -> tuple[str, int]:
            return ("scratch", nbytes)

        def dispatch(
            self,
            name: str,
            handles: list[object],
            constants: list[bytes],
            groups: tuple[int, int, int],
            threads: tuple[int, int, int],
        ) -> None:
            del name, constants, groups, threads
            self.dispatches.append((handles[0], handles[1]))

        def encode(
            self,
            name: str,
            handles: list[object],
            constants: list[bytes],
            groups: tuple[int, int, int],
            threads: tuple[int, int, int],
        ) -> None:
            del handles, constants, groups, threads
            self.encoded.append(name)

        def _u32(self, value: int) -> bytes:
            return struct.pack("<I", value)

    def remember(_handle: object, _offset: int, data: bytes) -> None:
        stored.append(bytes(data))

    monkeypatch.setattr(vision, "device_buffer", allocate)
    monkeypatch.setattr(vision.memory, "map", map_span)
    monkeypatch.setattr(vision.memory, "store", remember)
    monkeypatch.setattr(vision.memory, "release", release)
    monkeypatch.setattr(forward, "_fill", spy)
    session = _Session()
    tower = vision.VisionTower(session, tmp_path, "gemma4:test")
    tower._open()
    assert tower._ready is True
    assert tower._pos_offset == pos_offset
    assert pos_offset not in starts
    assert bytes(spans[tower.input_proj.data]) == bytes(blob[:32])
    sources = [handle for handle, _dest in session.dispatches]
    assert sources
    assert all(isinstance(handle, int) and handle in released for handle in sources)
    ones = struct.pack("<f", 1.0)
    assert stored == [ones * VISION_WIDTH, ones * VISION_HEAD]
    row_bytes = VISION_WIDTH * 2
    y_at = pos_offset + vision._POS_ROWS * row_bytes
    path = tmp_path / "resident.bin"
    with path.open("r+b") as handle:
        handle.seek(y_at + row_bytes - 1)
        handle.write(b"\x00")
        handle.seek(pos_offset)
        handle.write(b"\x11" * row_bytes)
        handle.seek(y_at)
        handle.write(b"\x22" * row_bytes)
    before = len(session.dispatches)
    tower._positions(1, 1)
    assert starts[-2:] == [pos_offset, y_at]
    assert session.encoded == ["vision_positions"]
    assert session.dispatches[before][1] is tower.pos_x
    assert session.dispatches[before + 1][1] is tower.pos_y
    assert released[session.dispatches[before][0]] == b"\x11" * row_bytes
    assert released[session.dispatches[before + 1][0]] == b"\x22" * row_bytes


def test_a_short_vision_row_stays_named(tmp_path: Path, monkeypatch) -> None:
    from gemma4_26b_a4b.metal41 import vision

    rows = [
        {
            "name": "model.vision_tower.std_scale",
            "plane": "half",
            "shape": [4],
            "nbytes": 8,
            "offset": 0,
        }
    ]
    (tmp_path / "resident.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "resident.bin").write_bytes(b"\x00\x00")
    spans, allocate, map_span = _spans()
    monkeypatch.setattr(vision, "device_buffer", allocate)
    monkeypatch.setattr(vision.memory, "map", map_span)
    monkeypatch.setattr(vision.memory, "release", lambda handle: spans.pop(handle, None))
    tower = vision.VisionTower(object(), tmp_path, "gemma4:test")
    try:
        tower._open()
    except RuntimeError as exc:
        assert str(exc) == "model.vision_tower.std_scale is short"
    else:
        raise AssertionError("a short vision row must fail")
    assert spans == {}


def test_a_text_open_skips_vision_rows(tmp_path: Path, monkeypatch) -> None:
    from gemma4_26b_a4b.metal41 import forward
    from gemma4_26b_a4b.metal41.forward import Gemma4Runtime

    index = [
        {
            "name": "model.vision_tower.std_bias",
            "nbytes": 32,
            "offset": 0,
            "shape": [16],
            "plane": "half",
        },
        {
            "name": "model.embed_vision.embedding_projection.weight",
            "nbytes": 32,
            "offset": 32,
            "shape": [1, 128],
            "plane": "fp8",
        },
        {
            "name": "model.language_model.embed_tokens.weight",
            "nbytes": 16,
            "offset": 64,
            "shape": [1, 128],
            "plane": "fp8",
        },
    ]
    (tmp_path / "resident.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "resident.bin").write_bytes(b"\xab" * 32 + b"\xcd" * 32 + b"\xef" * 16)
    spans, allocate, map_span = _spans()
    starts: list[int] = []
    real_fill = forward._fill

    def spy(raw: object, view: memoryview, start: int) -> None:
        starts.append(start)
        real_fill(raw, view, start)

    monkeypatch.setattr(forward, "device_buffer", allocate)
    monkeypatch.setattr(forward.memory, "map", map_span)
    monkeypatch.setattr(forward, "_fill", spy)

    class _Host:
        _owner = "gemma4:1"

        def _note_load(self, nbytes: int) -> None:
            del nbytes

    try:
        Gemma4Runtime._load(_Host(), tmp_path)
    except KeyError as exc:
        assert exc.args[0] == "model.language_model.norm.weight"
    else:
        raise AssertionError("the text map must stop before a full record")
    assert starts == [64]
    assert list(spans.values()) == [bytearray(b"\xef" * 16)]


def test_expert_widths_fill_the_span(tmp_path: Path, monkeypatch) -> None:
    from gemma4_26b_a4b.metal41 import forward
    from gemma4_26b_a4b.metal41.forward import Gemma4Runtime
    from gemma4_26b_a4b.metal41.geometry import EXPERTS, HALF, LAYERS, RESIDENT, expected

    rows = []
    for name, (role, shape) in expected().items():
        if role not in {RESIDENT, HALF}:
            continue
        vision = "vision_tower" in name or "embed_vision" in name
        rows.append(
            {
                "name": name,
                "plane": "half" if role == HALF else "fp8",
                "shape": list(shape),
                "nbytes": 64 if vision else 0,
                "offset": 0,
            }
        )
    (tmp_path / "resident.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "resident.bin").write_bytes(b"\x00")
    experts = tmp_path / "experts"
    experts.mkdir()
    width_bytes = LAYERS * EXPERTS
    (experts / "widths.bin").write_bytes(bytes([4]) * width_bytes)
    for layer in range(LAYERS):
        (experts / f"{layer}.bin").write_bytes(b"\x00" * (EXPERTS * 8))
    spans, allocate, map_span = _spans()
    sizes: list[int] = []

    def allocate_sized(owner: str, nbytes: int, *, zero: bool = False) -> int:
        sizes.append(nbytes)
        return allocate(owner, nbytes, zero=zero)

    class _Session:
        def __init__(self) -> None:
            self.uploads: list[object] = []

        def zeros(self, nbytes: int) -> object:
            del nbytes
            return object()

        def dispatch(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def _u32(self, value: int) -> bytes:
            del value
            return b""

        def upload(self, data: object) -> int:
            self.uploads.append(data)
            return len(self.uploads)

    def load(handle: object, offset: int, nbytes: int) -> bytes:
        if isinstance(handle, int):
            return bytes(spans[handle][offset : offset + nbytes])
        return struct.pack("<f", 1.0) * (nbytes // 4)

    monkeypatch.setattr(forward, "expert_bytes", lambda width: 8 if int(width) == 4 else 16)
    monkeypatch.setattr(forward, "device_buffer", allocate_sized)
    monkeypatch.setattr(forward.memory, "map", map_span)
    monkeypatch.setattr(forward.memory, "load", load)
    monkeypatch.setattr(forward.memory, "store", lambda _handle, _offset, _data: None)
    monkeypatch.setattr(forward.memory, "release", lambda _handle: None)
    session = _Session()
    host = Gemma4Runtime.__new__(Gemma4Runtime)
    host._owner = "gemma4:1"
    host.session = session
    host._note_load = lambda _nbytes: None
    host._load(tmp_path)
    assert 64 not in sizes
    assert bytes(spans[host.expert_widths]) == bytes([4]) * width_bytes
    assert len(session.uploads) == 1
    offsets = session.uploads[0]
    assert int(offsets[0, 0]) == 0
    assert int(offsets[0, 1]) == 8
    assert int(offsets[LAYERS - 1, EXPERTS - 1]) == (EXPERTS - 1) * 8
