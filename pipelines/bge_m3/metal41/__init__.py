"""BGE-M3 dense embeddings from the int8 GPU record."""


PIPELINE = "bge_m3"


from bge_m3.metal41.commands import main
from bge_m3.metal41.runtime import handle_embed, open_runtime

__all__ = ["handle_embed", "main", "open_runtime"]
