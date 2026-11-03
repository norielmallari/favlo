"""Gallery media metadata under ``.favlo_metadata``.

Schema and creators live in ``favlo.media_metadata``. This module re-exports
them for gallery callers.
"""

from __future__ import annotations

from favlo.media_metadata import (  # noqa: F401
    KIND_SUFFIXES,
    METADATA_DIR,
    build_index_sidecar,
    clamp,
    from_generation,
    is_media,
    kind_for,
    merge_analyzed,
    move_sidecar,
    prompt_text,
    read_sidecar,
    relocate_metadata_prefix,
    remove_metadata_prefix,
    remove_sidecar,
    sidecar_path,
    write_sidecar,
)
