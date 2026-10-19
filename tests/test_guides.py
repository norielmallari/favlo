"""The setup pages the terminal chat is allowed to quote."""

from __future__ import annotations

import pytest

from favlo.guides import read_guide


def test_clients_guide_names_this_server() -> None:
    text = read_guide("clients")
    assert "http://127.0.0.1:8129/v1" in text
    assert "qwen36-35b-a3b" in text
    assert "API key" in text
    assert "/quit" in text
    assert "save_intro" in text
    assert "intros.json" in text
    assert "30 minutes" in text
    assert "Qwen/Qwen3.8-27B" in text
    assert "download_suggestions" in text


def test_models_guide_names_the_official_checkpoints() -> None:
    text = read_guide("models")
    assert "Qwen/Qwen3.6-35B-A3B" in text
    assert "favlo --qwen36-35b-a3b setup" in text
    assert "Qwen/Qwen3.8-27B" in text
    assert "favlo --qwen38-27b download" in text
    assert "download_suggestions" in text
    assert "favlo --qwen38-27b quantize` does not download" in text
    assert "Qwen/Qwen3.8-Flash-Next" in text
    assert "--source` is required" in text


def test_history_is_the_carrot_naming() -> None:
    text = read_guide("history")
    assert "Amanda" in text
    assert "Celestia" in text
    assert "carrots" in text
    assert "gave him his name" in text


def test_an_unknown_guide_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown guide"):
        read_guide("../clients")
    with pytest.raises(ValueError, match="unknown guide"):
        read_guide("clients.md")
