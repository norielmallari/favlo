"""file:// reads one absolute local file."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from favlo.local_uri import file_uri, materialize, read_file_uri


def test_an_absolute_file_uri_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "face.png"
    path.write_bytes(b"png")
    uri = file_uri(path)
    assert uri.startswith("file:///")
    assert read_file_uri(uri) == b"png"
    assert materialize(uri) == base64.b64encode(b"png").decode("ascii")
    assert materialize({"uri": uri}) == {"b64_json": materialize(uri)}


def test_a_file_uri_is_a_still(tmp_path: Path) -> None:
    from favlo.mcp_server import _still

    raw = b"\xff" * 8
    path = tmp_path / "face.png"
    path.write_bytes(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    assert encoded.startswith("/")
    assert _still(file_uri(path), {}) == encoded
    assert _still({"type": "image", "data": file_uri(path)}, {}) == encoded
    with pytest.raises(RuntimeError, match="bare path"):
        _still(str(path), {})


def test_a_prompt_with_a_slash_is_not_a_still() -> None:
    from favlo.mcp_server import _argument
    from fastmcp.exceptions import ToolError

    prompt = "85mm f/1.8, shallow depth of field"
    assert _argument(prompt, {}) == prompt
    with pytest.raises(ToolError, match="bare path"):
        _argument(str(Path("/tmp/face.png")), {}, still=True)


def test_an_image_list_keeps_each_still(tmp_path: Path) -> None:
    from favlo.mcp_server import _image_list
    from fastmcp.exceptions import ToolError

    raw = b"\xff" * 8
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(raw)
    second.write_bytes(b"\xfe" * 4)
    encoded = base64.b64encode(raw).decode("ascii")
    other = base64.b64encode(b"\xfe" * 4).decode("ascii")
    assert _image_list([file_uri(first), file_uri(second)], {}) == [encoded, other]
    with pytest.raises(ToolError, match="image is required"):
        _image_list([], {})
    with pytest.raises(ToolError, match="bare path"):
        _image_list([str(first)], {})


def test_a_file_uri_still_argument_survives_base64_slashes(tmp_path: Path) -> None:
    from favlo.mcp_server import _argument

    raw = b"\xff" * 8
    path = tmp_path / "face.png"
    path.write_bytes(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    assert "/" in encoded or encoded.startswith("/")
    assert _argument(file_uri(path), {}, still=True) == encoded


def test_a_host_or_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        read_file_uri("file://localhost/tmp/a.png")
    with pytest.raises(ValueError, match="absolute"):
        read_file_uri("file://relative.png")
    missing = file_uri(tmp_path / "gone.png")
    with pytest.raises(ValueError, match="missing"):
        read_file_uri(missing)
