"""Plain HTTP download. A local server stands in for the hub."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qwen36_35b_a3b.fetch import download, list_files, remaining_download, shortfall


class _Hub(BaseHTTPRequestHandler):
    files = {"config.json": b"hello", "weights.bin": b"01234567"}

    def log_message(self, fmt: str, *args: object) -> None:
        del fmt, args

    def do_HEAD(self) -> None:  # noqa: N802
        name = self.path.rsplit("/", 1)[-1]
        body = self.files.get(name)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/models/"):
            siblings = [
                {"rfilename": name, "size": len(body)} for name, body in self.files.items()
            ]
            blob = json.dumps({"siblings": siblings}).encode()
            self._send(200, blob, {})
            return
        name = self.path.rsplit("/", 1)[-1]
        body = self.files.get(name)
        if body is None:
            self._send(404, b"", {})
            return
        start = 0
        status = 200
        headers: dict[str, str] = {}
        piece = body
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=", 1)[1].split("-", 1)[0])
            piece = body[start:]
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
        self._send(status, piece, headers)

    def _send(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_download_resumes_a_short_file(tmp_path: Path) -> None:
    server, host = _serve()
    try:
        files = list_files(host, "Qwen/Qwen3.6-35B-A3B")
        assert files[0] == ("config.json", 5)
        dest = tmp_path / "checkout"
        (dest).mkdir()
        partial = dest / "weights.bin"
        partial.write_bytes(b"0123")
        seen: list[tuple[int, int, str]] = []
        download(dest, files, lambda done, total, name: seen.append((done, total, name)), host=host)
        assert (dest / "config.json").read_bytes() == b"hello"
        assert partial.read_bytes() == b"01234567"
        assert seen[-1][0] == seen[-1][1] == 13
        assert remaining_download(dest, files) == 0
    finally:
        server.shutdown()


def test_a_file_longer_than_the_listing_is_kept_when_the_server_agrees(tmp_path: Path) -> None:
    server, host = _serve()
    try:
        dest = tmp_path / "checkout"
        dest.mkdir()
        body = _Hub.files["weights.bin"]
        (dest / "weights.bin").write_bytes(body)
        listed = [("weights.bin", 4)]
        seen: list[int] = []
        download(dest, listed, lambda done, total, _name: seen.append(total), host=host)
        assert (dest / "weights.bin").read_bytes() == body
        assert seen[-1] == len(body)
    finally:
        server.shutdown()


def test_shortfall_names_the_free_space(tmp_path: Path) -> None:
    message = shortfall(1 << 60, 1, tmp_path / "src", tmp_path / "out")
    assert message is not None
    assert "free" in message
    assert shortfall(1, 1, tmp_path / "src", tmp_path / "out") is None
