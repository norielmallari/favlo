"""Checkpoint shapes for Qwen3.8-27B. Values come from a config object."""

from qwen38_27b.formats import text_shapes


def _config() -> dict:
    layers = (["linear_attention"] * 3 + ["full_attention"]) * 16
    return {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "output_gate_type": "swish",
            "layer_types": layers,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "rope_theta": 10_000_000,
            },
        }
    }


def test_text_shapes_follow_the_checkpoint_config() -> None:
    shapes = text_shapes(_config())
    assert shapes.hidden == 5120
    assert shapes.layers == 64
    assert shapes.vocab == 248320
    assert shapes.intermediate == 17408
    assert shapes.full_heads == 24
    assert shapes.kv_heads == 4
    assert shapes.head_dim == 256
    assert shapes.linear_q_heads == 16
    assert shapes.linear_v_heads == 48
    assert shapes.linear_head_dim == 128
    assert shapes.conv_kernel == 4
    assert shapes.rotary_pairs == 32
    assert shapes.mrope_section == (11, 11, 10)
    assert shapes.context == 262144
    assert shapes.output_gate == "swish"
    assert shapes.layer_types.count("full_attention") == 16
    assert shapes.layer_types.count("linear_attention") == 48
    assert shapes.layer_types[:4] == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
