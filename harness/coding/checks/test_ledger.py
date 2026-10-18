"""The suite that grades ``ledger.py``. It is not copied into the workspace."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _ledger():
    root = Path(os.environ["CODING_WORKSPACE"])
    path = root / "ledger.py"
    spec = importlib.util.spec_from_file_location("ledger", path)
    if spec is None or spec.loader is None:
        message = "ledger.py is missing"
        raise AssertionError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_is_the_difference() -> None:
    assert _ledger().collect(15, 21) == 6


def test_collect_zero() -> None:
    assert _ledger().collect(0, 0) == 0


def test_collect_negative() -> None:
    assert _ledger().collect(21, 15) == -6
