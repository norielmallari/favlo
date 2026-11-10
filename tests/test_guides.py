"""The setup pages the configuration tools are allowed to quote."""

from __future__ import annotations

import pytest

from favlo.guides import read_guide, read_setup, setup_pipelines


def test_clients_guide_names_this_server() -> None:
    text = read_guide("clients")
    assert "http://127.0.0.1:8129/v1" in text
    assert "qwen35_4b" in text
    assert "API key" in text
    assert "POST /v1/pipelines/reload" in text
    assert "does not call a model" in text
    assert "Ctrl+C" in text
    assert "Qwen3.5-4B" in text


def test_models_guide_names_the_official_checkpoints() -> None:
    text = read_guide("models")
    assert "Qwen/Qwen3.5-4B" in text
    assert "favlo serve` with an empty registry downloads that checkpoint" in text
    assert "favlo --qwen35_4b setup" in text
    assert "packs `models/Qwen3.5-4B-fp8`" in text
    assert "Qwen/Qwen3.6-35B-A3B" in text
    assert "favlo --qwen36_35b_a3b setup" in text
    assert "google/gemma-4-26B-A4B-it" in text
    assert "favlo --gemma4_26b_a4b quantize" in text
    assert "models/gemma-4-26B-A4B-it-fp8-e20" in text
    assert "Qwen/Qwen3.8-27B" in text
    assert "favlo --qwen38_27b download" in text
    assert "download_suggestions" in text
    assert "favlo --qwen38_27b quantize` does not download" in text


def test_history_is_the_carrot_naming() -> None:
    text = read_guide("history")
    assert "Amanda" in text
    assert "Celestia" in text
    assert "carrots" in text
    assert "gave him his name" in text


def test_setup_guides_load_by_pipeline_id() -> None:
    assert "qwen36_35b_a3b" in setup_pipelines()
    assert "gemma4_26b_a4b" in setup_pipelines()
    assert "qwen38_27b" in setup_pipelines()
    assert "z_image_turbo" in setup_pipelines()
    assert "wan22_i2v" in setup_pipelines()
    assert "ace_step_15_turbo" in setup_pipelines()
    gemma = read_setup("gemma4_26b_a4b")
    assert "uv run favlo --gemma4_26b_a4b quantize" in gemma
    assert "uv run favlo --gemma4_26b_a4b quantize --src models/example-model" in gemma
    assert "google/gemma-4-26B-A4B-it" in gemma
    assert "models/gemma-4-26B-A4B-it" in gemma
    text = read_setup("qwen36_35b_a3b")
    assert "uv run favlo --qwen36_35b_a3b setup" in text
    assert "--keep-bf16" in text
    assert "delete" in text
    later = read_setup("qwen38_27b")
    assert "uv run favlo --qwen38_27b download" in later
    assert "uv run favlo --qwen38_27b delete qwen38_27b" in later
    image = read_setup("z_image_turbo")
    assert "uv run favlo --z_image_turbo setup" in image
    assert "--weights models/example-model.safetensors" in image
    video = read_setup("wan22_i2v")
    assert "uv run favlo --umt5_xxl setup" in video
    assert "uv run favlo --umt5_xxl quantize" in video
    assert "uv run favlo --wan22_i2v setup" in video
    ti2v = read_setup("wan22_ti2v_5b")
    assert "uv run favlo --umt5_xxl setup" in ti2v
    assert "uv run favlo --umt5_xxl quantize" in ti2v
    assert "uv run favlo --wan22_ti2v_5b setup" in ti2v
    umt5 = read_setup("umt5_xxl")
    assert "uv run favlo --umt5_xxl setup" in umt5
    assert "uv run favlo --umt5_xxl quantize" in umt5
    music = read_setup("ace_step_15_turbo")
    assert "uv run favlo --ace_step_15_turbo setup" in music
    assert "uv run favlo --ace_step_15_turbo quantize" in music
    assert "uv run favlo --ace_step_15_turbo delete acestep-v15-xl-turbo-fp8-fp4" in music
    with pytest.raises(ValueError, match="unknown setup guide"):
        read_setup("../qwen36_35b_a3b")

def test_an_unknown_guide_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown guide"):
        read_guide("../clients")
    with pytest.raises(ValueError, match="unknown guide"):
        read_guide("clients.md")
