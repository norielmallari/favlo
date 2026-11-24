"""The prefix budget is global under the store root."""

from pathlib import Path

from favlo.prefix import PrefixRecord, PrefixStore, Region


def test_budget_evicts_across_models(tmp_path: Path) -> None:
    digest_a = bytes(range(32))
    digest_b = bytes(range(32, 64))
    first = PrefixStore(
        tmp_path, model="model-a", digest=digest_a, tag="half", budget=1
    )
    path_a = first.save(PrefixRecord([1, 2, 3], [Region("kv", b"a" * 64)]))
    assert path_a is not None and path_a.is_file()
    second = PrefixStore(
        tmp_path, model="model-b", digest=digest_b, tag="half", budget=1
    )
    path_b = second.save(PrefixRecord([4, 5, 6], [Region("kv", b"b" * 64)]))
    assert path_b is not None and path_b.is_file()
    assert not path_a.is_file()


def test_budget_evicts_across_checkpoints(tmp_path: Path) -> None:
    digest_a = bytes(range(32))
    digest_b = bytes(range(32, 64))
    first = PrefixStore(
        tmp_path, model="example-model", digest=digest_a, tag="half", budget=1
    )
    path_a = first.save(PrefixRecord([1, 2, 3], [Region("kv", b"a" * 64)]))
    assert path_a is not None and path_a.is_file()
    second = PrefixStore(
        tmp_path, model="example-model", digest=digest_b, tag="half", budget=1
    )
    path_b = second.save(PrefixRecord([4, 5, 6], [Region("kv", b"b" * 64)]))
    assert path_b is not None and path_b.is_file()
    assert not path_a.is_file()


def test_pin_survives_global_eviction(tmp_path: Path) -> None:
    digest_a = bytes(range(32))
    digest_b = bytes(range(32, 64))
    first = PrefixStore(
        tmp_path, model="model-a", digest=digest_a, tag="half", budget=1
    )
    path_a = first.save(PrefixRecord([1, 2, 3], [Region("kv", b"a" * 64)]))
    assert path_a is not None
    first.pin(path_a)
    second = PrefixStore(
        tmp_path, model="model-b", digest=digest_b, tag="half", budget=1
    )
    path_b = second.save(PrefixRecord([4, 5, 6], [Region("kv", b"b" * 64)]))
    assert path_b is not None and path_b.is_file()
    assert path_a.is_file()
