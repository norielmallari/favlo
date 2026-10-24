"""BGE-M3 embeddings on the Neural Engine."""

from bge_m3.commands import main
from bge_m3.runtime import handle_embed, open_runtime

__all__ = ["handle_embed", "main", "open_runtime"]
