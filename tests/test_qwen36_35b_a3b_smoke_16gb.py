"""Carrot smoke with the runtime capped at 16GiB.

The cases are the ones in ``test_qwen36_35b_a3b_smoke``. The cap is the same
``max_memory`` value as ``serve --max-memory 16GiB``. The atlas then holds
fewer than every expert image, and a miss reads that image when it is routed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from qwen36_35b_a3b.forward import Qwen36Runtime

_CAP = "16GiB"
_BYTES = 16 << 30


def _smoke():
    path = Path(__file__).with_name("test_qwen36_35b_a3b_smoke.py")
    spec = importlib.util.spec_from_file_location("qwen36_35b_a3b_smoke_cases", path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path.name}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CASES = _smoke()


@pytest.fixture(scope="module")
def carrot_runtime():
    if not (_CASES._ROOT / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = Qwen36Runtime(_CASES._ROOT, max_memory=_CAP)
    if runtime.max_memory != _BYTES:
        message = f"memory cap is {runtime.max_memory}, not {_BYTES}"
        raise AssertionError(message)
    if runtime.atlas.slots >= 40 * 256:
        message = f"16GiB atlas held all {runtime.atlas.slots} expert images"
        raise AssertionError(message)
    try:
        yield runtime
    finally:
        runtime.close()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    _CASES.pytest_generate_tests(metafunc)


def test_carrot_instruct_greedy(carrot_runtime: Qwen36Runtime) -> None:
    _CASES.test_carrot_instruct_greedy(carrot_runtime)


def test_carrot_instruct_recommended(carrot_runtime: Qwen36Runtime) -> None:
    _CASES.test_carrot_instruct_recommended(carrot_runtime)


def test_carrot_thinking_recommended(carrot_runtime: Qwen36Runtime) -> None:
    _CASES.test_carrot_thinking_recommended(carrot_runtime)


def test_carrot_thinking_effort(carrot_runtime: Qwen36Runtime, carrot_effort: str) -> None:
    _CASES.test_carrot_thinking_effort(carrot_runtime, carrot_effort)
