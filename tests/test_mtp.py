"""Host formula for the MTP projection mix."""

import numpy as np
from qwen38_flash_next.checks import mtp_mix


def test_mtp_mix_adds_the_branch_mean_to_the_embedding() -> None:
    residual = np.ones((4, 8), dtype=np.float32)
    embed = np.ones(8, dtype=np.float32)
    norm_hidden = np.zeros(32, dtype=np.float32)
    norm_embed = np.zeros(8, dtype=np.float32)
    hidden = np.eye(8, dtype=np.float32)
    embed_proj = np.eye(8, dtype=np.float32) * 2
    got = mtp_mix(residual, embed, norm_hidden, norm_embed, hidden, embed_proj, 1e-6)
    inv = np.float32(1.0 / np.sqrt(np.float32(1.0) + np.float32(1e-6)))
    assert got.shape == (8,)
    assert np.allclose(got, np.float32(3.0) * inv)
