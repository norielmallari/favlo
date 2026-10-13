"""Grouping of router choices. No device and no checkpoint."""

import numpy as np

from test_packing import pipeline


def test_group_assignments_keep_slot_order() -> None:
    ids = np.asarray([2, 0, 2, 1], dtype=np.uint32)
    experts, offsets, tokens, inverse = pipeline.group_assignments(ids, 2, 2)
    assert experts.tolist() == [0, 1, 2]
    assert offsets.tolist() == [0, 1, 2, 4]
    assert tokens.tolist() == [0, 1, 0, 1]
    assert inverse.tolist() == [2, 0, 3, 1]
