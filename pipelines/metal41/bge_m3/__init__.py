"""BGE-M3 dense embeddings from the int8 GPU record."""

from bge_m3.commands import main
from bge_m3.runtime import handle_embed, open_runtime

__all__ = ["handle_embed", "main", "open_runtime"]
