"""A pinned prefix file survives a later save that is over the budget."""

from pathlib import Path

from favlo.prefix import PrefixRecord, PrefixStore, Region


def test_pin_survives_a_one_byte_budget(tmp_path: Path) -> None:
    digest = bytes(range(32))
    store = PrefixStore(
        tmp_path, model="example-model", digest=digest, tag="half", budget=1
    )
    short = store.save(PrefixRecord([1, 2, 3], []))
    assert short is not None
    store.pin(short)
    long = store.save(PrefixRecord([1, 2, 3, 4, 5], [Region("kv", b"x" * 64)]))
    assert long is not None and long.is_file()
    assert short.is_file()
    newer = store.save(PrefixRecord([9, 9, 9, 9], [Region("kv", b"y" * 64)]))
    assert newer is not None and newer.is_file()
    assert short.is_file()
    assert not long.is_file()
    again = PrefixStore(
        tmp_path, model="example-model", digest=digest, tag="half", budget=1
    )
    assert again.pinned_path() == short
