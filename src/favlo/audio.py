"""Shared folder for music and voice clips.

Both kinds keep their files under ``<home>/audio/``. The id prefix
(``music_`` or ``voice_``) is what separates them.
"""

from __future__ import annotations

from pathlib import Path

_DIR = "audio"


def audio_dir(home: Path) -> Path:
    return home / _DIR
