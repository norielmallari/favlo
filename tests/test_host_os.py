"""Host seams that differ by operating system. No pipeline and no GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from favlo.host import available_bytes, page_bytes, physical_bytes, select_device
from favlo.local_uri import read_file_uri
from favlo.memory import reclaimable_bytes
from favlo.prefix import _pread, _pwrite
from favlo.processes import etime_seconds, process_rows
from favlo.setup import host_ready


def test_this_host_reports_pages_and_installed_memory() -> None:
    assert page_bytes() > 0
    assert physical_bytes() > page_bytes()
    assert reclaimable_bytes() > 0


def test_available_bytes_is_refused_off_windows() -> None:
    if sys.platform == "win32":
        assert available_bytes() > 0
        return
    try:
        available_bytes()
    except RuntimeError as exc:
        assert "Windows" in str(exc)
        return
    raise AssertionError("available_bytes returned on this host")


def test_windows_reclaimable_bytes_is_the_physical_probe(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("favlo.host.available_bytes", lambda: 4096)
    assert reclaimable_bytes() == 4096


def test_host_ready_follows_the_platform(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("favlo.host._cuda_ready", lambda: False)
    assert host_ready() is False
    monkeypatch.setattr("favlo.host._cuda_ready", lambda: True)
    assert host_ready() is True
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("favlo.setup.metal_ready", lambda: False)
    assert host_ready() is False
    monkeypatch.setattr("favlo.setup.metal_ready", lambda: True)
    assert host_ready() is True


def test_select_device_refuses_a_windows_host_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    try:
        select_device(["metal41"])
    except RuntimeError as exc:
        assert str(exc) == "this runtime cannot run here"
        return
    raise AssertionError("metal41 was selected on Windows")


def test_select_device_accepts_cuda_when_the_driver_answers(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("favlo.host._cuda_ready", lambda: True)
    assert select_device(["metal41", "cuda"]) == "cuda"


def test_process_rows_see_this_process() -> None:
    assert any(pid == os.getpid() for pid, _parent, _command in process_rows())


def test_elapsed_time_parses_on_every_host() -> None:
    assert etime_seconds("01:02") == 62
    assert etime_seconds("2-01:02:03") == 2 * 86400 + 3723
    assert etime_seconds("nope") is None


def test_positioned_io_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "prefix.bin"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _pwrite(fd, b"abcd", 0)
        _pwrite(fd, b"XYZ", 1)
        assert _pread(fd, 4, 0) == b"aXYZ"
    finally:
        os.close(fd)


def test_a_file_uri_roundtrips_on_this_host(tmp_path: Path) -> None:
    path = tmp_path / "still.bin"
    path.write_bytes(b"png")
    assert read_file_uri(path.resolve().as_uri()) == b"png"


def test_close_runtime_skips_the_objc_pool_off_darwin(monkeypatch) -> None:
    from favlo.pipelines import close_runtime

    monkeypatch.setattr(sys, "platform", "win32")
    closed: list[str] = []

    class Runtime:
        def close(self) -> None:
            closed.append("closed")

    close_runtime(Runtime())
    assert closed == ["closed"]
