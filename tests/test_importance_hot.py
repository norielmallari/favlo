"""Hot-id selection from an importance graph."""

from qwen38_flash_next.convert import importance_hot_ids


def test_importance_hot_ids_keep_the_highest_frequency() -> None:
    graph = {
        "experts": {
            "0": {
                "0": {"frequency": 0.1},
                "1": {"frequency": 0.9},
                "2": {"frequency": 0.5},
                "3": {"frequency": 0.5},
            }
        }
    }
    assert importance_hot_ids(graph, layers=1, experts=4, keep=2) == [[1, 2]]
