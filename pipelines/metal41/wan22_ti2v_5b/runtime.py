"""Open the packed DiT, the shared umT5 record, and the float32 VAE."""

from __future__ import annotations

import json
from pathlib import Path

from favlo.cli import repo_root
from favlo.media_pipeline import MediaCall, MediaPaused, MediaPipeline
from favlo.memory import memory
from favlo.registry import models_dir
from wan22_ti2v_5b.clip import Clip, ClipHold
from wan22_ti2v_5b.dit import DiT
from wan22_ti2v_5b.gpu import Gpu
from wan22_ti2v_5b.request import VideoRequest, VideoRequestError, parse_request
from wan22_ti2v_5b.text import TextEncoder
from wan22_ti2v_5b.vae import WanVae

_MIN_FRAME = 256 * (704 // 2) * (1280 // 2) * 4


class Runtime:
    def __init__(
        self,
        record: Path,
        owner: str,
        budget: int,
        dit: DiT,
        text: TextEncoder,
        vae: WanVae,
        gpu: Gpu,
    ) -> None:
        self.record = record
        self._owner = owner
        self.budget = budget
        self.dit = dit
        self.text = text
        self.vae = vae
        self.gpu = gpu
        self.clip = Clip(dit, text, vae)
        self.frame_tile = max(1, int(budget * 0.45 / _MIN_FRAME))

    def close(self) -> None:
        self.vae.close()
        self.gpu.close()

    def halt(self) -> None:
        self.close()

    def max_frames(self, height: int, width: int) -> int:
        wide = 256 * (height // 2) * (width // 2) * 4
        return max(1, int(self.budget * 0.45 / max(wide, 1)))


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
    record = Path(path)
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    budget = _budget(max_memory)
    if budget < _MIN_FRAME * 3:
        message = "minimum working set does not fit"
        raise MemoryError(message)
    owner_name = owner or "wan22-ti2v-5b"
    gpu = Gpu(owner_name)
    umt5 = _umt5(parts, record)
    print(f"umt5 {umt5.name}", flush=True)
    # The tokenizer json sits beside the source download. The BF16 encoder file is not opened.
    tokenizer = models_dir() / "google" / "umt5-xxl" / "tokenizer.json"
    text = TextEncoder(umt5, gpu, tokenizer)
    vae_path = repo_root() / str(manifest["vae"])
    vae = WanVae(vae_path, gpu)
    return Runtime(record, owner_name, budget, DiT(record, gpu), text, vae, gpu)


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
        _require_fit(runtime, request)
        kept = hold if isinstance(hold, ClipHold) else None

        def gate(pause: bool = False) -> bool:
            reason = stop()
            if reason == "cancel":
                return True
            return bool(pause and reason == "preempt")

        payload, paused = runtime.clip.sample(
            request,
            video_id=job["id"],
            home=call.home,
            videos=call.table,
            hold=kept,
            stop=gate,
        )
        if paused is not None:
            job["progress"] = min(99, int(100 * paused.step / 4))
            job["size"] = request.size
            job["seconds"] = request.seconds
            raise MediaPaused("preempt", paused)
        if isinstance(payload, dict) and payload.get("status") == "cancelled":
            raise MediaPaused("cancel", None)
        return payload

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, job, home, table, call
        if isinstance(result, dict) and result.get("status") == "completed":
            result["progress"] = 100
        if isinstance(result, dict):
            return result
        return None


handle_video = Pipeline()


def handle_chat(reply, runtime, body, selection, args, state) -> None:
    del runtime, body, selection, args, state
    reply.json(400, {"error": "wan22-ti2v-5b does not complete a chat"})


def _require_fit(runtime: Runtime, request: VideoRequest) -> None:
    allowed = runtime.max_frames(request.height, request.width)
    if request.frames <= allowed:
        return
    message = "the clip does not fit the memory budget"
    raise VideoRequestError(message)


def _budget(limit: str | None) -> int:
    available = memory.budget()
    if not limit:
        return available
    text = limit.strip().lower()
    scale = 1
    for suffix, factor in (("gib", 1 << 30), ("gb", 10**9), ("mib", 1 << 20), ("mb", 10**6)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            scale = factor
            break
    try:
        capped = int(float(text) * scale)
    except ValueError as exc:
        message = "max_memory is not a byte count"
        raise RuntimeError(message) from exc
    return min(available, capped)


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
