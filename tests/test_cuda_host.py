"""Host detection and the CUDA pack chosen for a device budget."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from favlo.cli import pipeline_entries
from favlo.host import runtime_name

ROOT = Path(__file__).resolve().parents[1]


def _layout():
    path = ROOT / "pipelines" / "qwen35_4b" / "cuda" / "layout.py"
    spec = importlib.util.spec_from_file_location("qwen35_cuda_layout", path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_four_gigabytes_packs_int4_without_vision() -> None:
    layout = _layout()
    pack = layout.pack_for_budget(4 << 30)
    assert pack.plane == "int4"
    assert pack.vision is False
    assert pack.context == layout.CONTEXT_4GB


def test_six_gigabytes_packs_int8_with_vision() -> None:
    layout = _layout()
    pack = layout.pack_for_budget(6 << 30)
    assert pack.plane == "int8"
    assert pack.vision is True
    assert pack.context >= 32_768


def test_eight_gigabytes_keeps_int8_and_a_longer_cache() -> None:
    layout = _layout()
    six = layout.pack_for_budget(6 << 30)
    eight = layout.pack_for_budget(8 << 30)
    assert eight.plane == "int8"
    assert eight.vision is True
    assert eight.context > six.context
    assert layout.CONTEXT_CAP == 131_072
    assert eight.context == layout.CONTEXT_CAP


def test_the_same_stem_follows_the_host(tmp_path: Path) -> None:
    model = tmp_path / "pipelines" / "demo_model"
    model.mkdir(parents=True)
    (model / "__init__.py").write_text("", encoding="utf-8")
    for runtime in ("metal41", "cuda"):
        tree = model / runtime
        tree.mkdir()
        (tree / "__init__.py").write_text('PIPELINE = "demo_model"\n', encoding="utf-8")
    path = pipeline_entries(tmp_path)["demo_model"]
    assert path.name == runtime_name()


def test_usable_bytes_keeps_the_hold_back_and_a_lower_cap() -> None:
    from favlo.host import VRAM_HOLD_BYTES, usable_bytes

    free = 8 << 30
    assert usable_bytes(free, None) == free - VRAM_HOLD_BYTES
    assert usable_bytes(free, 6 << 30) == 6 << 30
    assert usable_bytes(VRAM_HOLD_BYTES, None) == 0


@pytest.mark.skipif(sys.platform != "win32", reason="CUDA driver is a Windows host")
def test_windows_reports_a_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from favlo.host import device_budget, usable_bytes, vram_bytes, vram_snapshot

    monkeypatch.setattr("favlo.registry.device_budget_cap", lambda root=None: None)
    total = vram_bytes()
    snap = vram_snapshot()
    assert total is not None and snap is not None
    assert total >= 4 << 30
    free, reported = snap
    assert reported > 0 and free > 0
    budget = device_budget()
    assert budget == usable_bytes(free, None)
    assert budget < free
    assert runtime_name() == "cuda"


@pytest.mark.skipif(sys.platform != "win32", reason="CUDA driver is a Windows host")
def test_registry_cap_limits_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from favlo.host import device_budget, usable_bytes, vram_snapshot

    monkeypatch.setattr("favlo.registry.device_budget_cap", lambda root=None: 6 << 30)
    snap = vram_snapshot()
    assert snap is not None
    budget = device_budget()
    assert budget == usable_bytes(snap[0], 6 << 30)
