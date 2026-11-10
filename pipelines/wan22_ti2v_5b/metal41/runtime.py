"""Open the packed DiT, the shared umT5 record, and the float32 VAE."""

from __future__ import annotations

from collections.abc import Sequence

import json
from pathlib import Path

from favlo.cli import repo_root
from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaTool, MediaCall, MediaPaused, MediaPipeline
from favlo.memory import memory
from favlo.registry import models_dir
from wan22_ti2v_5b.metal41.clip import Clip, ClipHold
from wan22_ti2v_5b.metal41.dit import DiT
from wan22_ti2v_5b.metal41.gpu import Gpu
from wan22_ti2v_5b.metal41.request import VideoRequest, VideoRequestError, parse_request
from wan22_ti2v_5b.metal41.text import TextEncoder
from wan22_ti2v_5b.metal41.vae import WanVae

_MIN_FRAME = 256 * (704 // 2) * (1280 // 2) * 4


class Runtime:
    def __init__(
        self,
        record: Path,
        owner: str,
        umt5: Path,
        vae_path: Path,
        tokenizer: Path,
    ) -> None:
        self.record = record
        self._owner = owner
        self._umt5 = umt5
        self._vae_path = vae_path
        self._tokenizer = tokenizer
        self._closed = True
        self._ceiling = 0
        self.frame_tile = 1
        self.gpu: Gpu | None = None
        self.dit: DiT | None = None
        self.text: TextEncoder | None = None
        self.vae: WanVae | None = None
        self.clip: Clip | None = None
        open_gate(_MIN_FRAME * 3, detail="packed record")
        self._bind()

    def ensure_open(self, request: VideoRequest) -> None:
        """Wait for prior releases, reload when closed, then check this clip fits."""
        need = self.working_set_bytes(request)
        detail = f"{request.frames} frames at {request.width}x{request.height}"
        open_gate(need, detail=detail)
        if self._closed:
            self._bind()
        _require_fit(self, request)

    def working_set_bytes(self, request: VideoRequest) -> int:
        """Bytes the frame tile for this clip needs, plus the load floor."""
        wide = 256 * (request.height // 2) * (request.width // 2) * 4
        clip = int(request.frames * max(wide, 1) / 0.45)
        return max(_MIN_FRAME * 3, clip)

    def _bind(self) -> None:
        if not self._closed:
            return
        self.gpu = Gpu(self._owner)
        try:
            self.text = TextEncoder(self._umt5, self.gpu, self._tokenizer)
            self.vae = WanVae(self._vae_path, self.gpu)
            self.dit = DiT(self.record, self.gpu)
            self.clip = Clip(self.dit, self.text, self.vae)
            self._closed = False
            self.refresh_budget()
        except Exception:
            self._closed = False
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        before, held = begin_release(self._owner)
        self._closed = True
        try:
            vae = self.vae
            gpu = self.gpu
            self.vae = None
            self.gpu = None
            self.dit = None
            self.text = None
            self.clip = None
            if vae is not None:
                vae.close()
            if gpu is not None:
                gpu.close()
        finally:
            finish_release(self._owner, before, held)

    def halt(self) -> None:
        self.close()

    def refresh_budget(self) -> int:
        """The largest ledger answer this opening has seen. A dip does not shrink it."""
        current = memory.budget()
        if current > self._ceiling:
            self._ceiling = current
        tile = max(1, int(self._ceiling * 0.45 / _MIN_FRAME))
        if tile > self.frame_tile:
            self.frame_tile = tile
        return self._ceiling

    def max_frames(self, height: int, width: int) -> int:
        wide = 256 * (height // 2) * (width // 2) * 4
        return max(1, int(self.refresh_budget() * 0.45 / max(wide, 1)))


def open_runtime(
    path: Path,
    kv_cache: str,
    console: object,
    resident_percent: float | None,
    owner: str | None = None,
    max_memory: str | None = None,
    parts: dict | None = None,
) -> Runtime:
    del kv_cache, console, resident_percent
    memory.set_max_memory(max_memory)
    record = Path(path)
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    umt5 = _umt5(parts, record)
    print(f"umt5 {umt5.name}", flush=True)
    # The tokenizer sits in the int8 pack. The BF16 encoder file is not opened.
    tokenizer = umt5 / "tokenizer.json"
    if not tokenizer.is_file():
        message = "umT5 pack is missing tokenizer.json"
        raise FileNotFoundError(message)
    vae_path = repo_root() / str(manifest["vae"])
    owner_name = stamp_owner("wan22_ti2v_5b", owner, id(record))
    return Runtime(record, owner_name, umt5, vae_path, tokenizer)


class Pipeline(MediaPipeline):
    """Wan TI2V's listener contract. The clip sampler stays in this package."""

    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"
    id_hex = 4

    def parse(self, body: dict) -> VideoRequest:
        return parse_request(body)

    def describe(self, request: VideoRequest) -> dict:
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "size": request.size,
            "frames": request.frames,
            "quality": request.quality,
        }

    def generate(self, runtime, request, hold, job, stop, call: MediaCall):
        kept = hold if isinstance(hold, ClipHold) else None
        try:
            return runtime.clip.sample(
                request,
                video_id=job["id"],
                home=call.home,
                videos=call.table,
                hold=kept,
                stop=stop,
            )
        except MediaPaused as paused:
            if isinstance(paused.hold, ClipHold):
                job["progress"] = min(99, int(100 * paused.hold.step / 4))
                job["size"] = request.size
                job["seconds"] = request.seconds
            raise

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, job, home, table, call
        if isinstance(result, dict) and result.get("status") == "completed":
            result["progress"] = 100
        if isinstance(result, dict):
            return result
        return None



    def main(self, argv: Sequence[str]) -> int:
        from wan22_ti2v_5b.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from wan22_ti2v_5b.metal41.runtime import open_runtime as open_rt
        from favlo.pipelines import bind_open_runtime

        return bind_open_runtime(open_rt, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        from wan22_ti2v_5b.metal41 import _video

        rows = [
            _video(
                "generate_video_t2v",
                "MP4 from text when the user asked for video or motion, not a still.",
                still=False,
            ),
            _video(
                "generate_video_i2v",
                "MP4 from a still when the user asked for video or motion, not a still.",
                still=True,
            ),
        ]
        return [
            MediaTool(
                name=str(row["name"]),
                description=str(row.get("description") or ""),
                parameters=row.get("parameters") or {"type": "object", "properties": {}},
                handler=str(row["handler"]),
                result=str(row.get("result") or "video"),
                task=bool(row.get("task", True)),
            )
            for row in rows
        ]

pipeline = Pipeline()
handle_video = pipeline
def handle_chat(reply, runtime, body, selection, args, state) -> None:
    del runtime, body, selection, args, state
    reply.json(400, {"error": "wan22_ti2v_5b does not complete a chat"})


def _require_fit(runtime: Runtime, request: VideoRequest) -> None:
    allowed = runtime.max_frames(request.height, request.width)
    if request.frames <= allowed:
        return
    message = "the clip does not fit the memory budget"
    raise VideoRequestError(message)


def _umt5(parts: dict | None, record: Path) -> Path:
    """The int8 pack on the registry row. The BF16 encoder file is not a pack."""
    named = _named_umt5(parts)
    if named is None:
        from favlo.registry import parts_for_directory

        named = parts_for_directory(record).get("umt5")
    if named is None:
        local = record / "umt5"
        shared = models_dir() / "google" / "umt5-xxl-int8"
        named = local if (local / "manifest.json").is_file() else shared
    return _require_int8_pack(Path(named))


def _named_umt5(parts: dict | None) -> Path | None:
    if not isinstance(parts, dict):
        return None
    raw = parts.get("umt5")
    if not raw:
        return None
    return Path(raw)


def _require_int8_pack(path: Path) -> Path:
    if path.suffix == ".pth" or "bf16" in path.name.lower():
        message = "umT5 weights must be the int8 pack"
        raise RuntimeError(message)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        message = "umT5 int8 pack is missing"
        raise FileNotFoundError(message)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("weight") != "int8":
        message = "umT5 pack is not int8"
        raise RuntimeError(message)
    return path
