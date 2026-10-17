"""Shader checks for the Qwen3.6-35B-A3B pipeline. They do not load a checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qwen36_35b_a3b.atlas import atlas_slots
from qwen36_35b_a3b.cache import kv_buffer_bytes, kv_mode, max_position, parse_memory
from qwen36_35b_a3b.checks import run_checks
from qwen36_35b_a3b.formats import EXPERT_IMAGE, dequant_fp4, fp4_image, quantize_fp4
from qwen36_35b_a3b.tools import function_has_parameters, parse_tool_calls, required_function


def test_fp4_tie_picks_the_even_code() -> None:
    block = np.zeros((1, 32), dtype=np.float32)
    block[0, 0] = 6.0
    block[0, 1] = 0.75
    packed, scales = quantize_fp4(block)
    restored = dequant_fp4(packed, scales, 1, 32)
    assert restored[0, 0] == np.float32(6.0)
    assert restored[0, 1] == np.float32(1.0)


def test_expert_image_is_two_fp4_planes() -> None:
    gate = np.zeros((1024, 2048), dtype=np.float32)
    down = np.zeros((2048, 512), dtype=np.float32)
    assert len(fp4_image(gate) + fp4_image(down)) == EXPERT_IMAGE


def test_tool_call_text_becomes_an_openai_call() -> None:
    text = (
        "ok\n<tool_call>\n<function=create_file>\n"
        "<parameter=path>\nnote.txt\n</parameter>\n"
        "<parameter=content>\nThe harbor log is open.\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == "ok"
    assert calls[0]["function"]["name"] == "create_file"
    assert "note.txt" in calls[0]["function"]["arguments"]
    choice = {"type": "function", "function": {"name": "create_file"}}
    assert required_function(choice) == "create_file"


def test_empty_function_has_no_parameters() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert function_has_parameters(tools, "read_file") is False
    assert function_has_parameters(tools, "missing") is True
    with_path = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    assert function_has_parameters(with_path, "read_file") is True


def test_kv_widths_and_atlas_budget() -> None:
    assert kv_mode(None) == "e4m3"
    assert kv_mode("fp4") == "e2m1"
    assert kv_mode("f32") == "f32"
    positions = 262144
    assert kv_buffer_bytes("e4m3", positions) == 2 * positions * 264
    assert kv_buffer_bytes("e2m1", positions) == 2 * positions * 136
    assert kv_buffer_bytes("f32", positions) == 2 * positions * 1024
    file_bytes = 427_819_008
    reserved = 1 << 30
    assert atlas_slots(reserved + file_bytes * 40, file_bytes, reserved) == 40
    assert atlas_slots(reserved + file_bytes * 4, file_bytes, reserved) == 4
    assert parse_memory("32GiB") == 32 << 30
    assert parse_memory("512MiB") == 512 << 20
    try:
        atlas_slots(reserved, file_bytes, reserved)
    except ValueError:
        pass
    else:
        raise AssertionError("a cap below one layer file must fail")
    root = Path("models/Qwen3.6-35B-A3B-madmetal-c256-fp4")
    if (root / "config.json").is_file():
        assert max_position(root) == 262144


def test_omitted_sampling_uses_the_pipeline_card() -> None:
    from qwen36_35b_a3b.chat import _presence, _sampling
    from qwen36_35b_a3b.sampling import mode_card

    thinking, card = mode_card(None, None)
    assert thinking is True
    assert card is not None
    assert card["presence_penalty"] == 1.5
    closed, instruct = mode_card(None, False)
    assert closed is False
    assert instruct is not None
    assert instruct["temperature"] == 0.7

    assert _sampling({}, True, None) == (1.0, 0.95, 20)
    assert _presence({}, True, None) == 1.5
    assert _sampling({}, False, None) == (0.7, 0.8, 20)
    assert _presence({}, False, None) == 1.5
    assert _presence({"presence_penalty": 0}, True, None) == 0.0


def test_kernels_match_the_host_formulas() -> None:
    report = run_checks()
    failed = [name for name, ok, _detail in report["rows"] if not ok]
    assert report["ok"], failed
