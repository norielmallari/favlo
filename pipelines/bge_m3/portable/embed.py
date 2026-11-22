"""Embeddings shell. The device opens from ``open_runtime``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bge_m3.portable.sampling as _sampling_mod
from favlo.embeddings_pipeline import EmbeddingsPipeline


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    return frozenset(
        name for name in ("metal41", "cuda") if (root / name / "__init__.py").is_file()
    )


class Pipeline(EmbeddingsPipeline):
    """``POST /v1/embeddings``. One string in, one dense vector out."""

    context_length = _sampling_mod.CONTEXT_LENGTH
    variants = _sampling_mod.VARIANTS
    default_variant = _sampling_mod.DEFAULT_VARIANT

    def embed(self, runtime: Any, text: str) -> tuple[list[float], int]:
        result = runtime.embed(text)
        return result.vector, result.tokens

    def main(self, argv: Sequence[str]) -> int:
        from bge_m3.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from bge_m3.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)


pipeline = Pipeline()
handle_embed = pipeline
