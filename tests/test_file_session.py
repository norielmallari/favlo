"""File-agent ladder: file selection, rate parsing, and prefix extension."""

from pathlib import Path

import pytest
from harness.file_session import (
    RequestRate,
    _open_agent,
    boundary_reloaded,
    extensions_hold,
    log_since,
    model_stem,
    parse_requests,
    pipeline_log,
    prefix_was_hit,
    preserved_reloaded,
    printed_requests,
    read_repository_file,
    rewrite_note,
    rewritten_loaded,
    source_files,
)

_LOG = """\
listening on http://127.0.0.1:9
prompt tokens 100
cache miss
incoming tokens 100
prefill 50.00 tok/s (100 tok in 2.00s)
output tokens 20
decode 10.00 tok/s (20 tok in 2.00s)
prefix load 120 tokens
prompt tokens 180
cache hit 120
incoming tokens 60
prefill 100.00 tok/s (60 tok in 0.60s)
output tokens 4
decode 8.00 tok/s (4 tok in 0.50s)
prompt tokens 300
cache hit 184
incoming tokens 116
prefill 80.00 tok/s (116 tok in 1.45s)
output tokens 4
decode 8.00 tok/s (4 tok in 0.50s)
"""


def test_a_replayed_log_prints_each_request_once() -> None:
    once = parse_requests(_LOG)
    replayed = printed_requests(parse_requests(_LOG + _LOG))
    assert [row.prompt_tokens for row in replayed] == [row.prompt_tokens for row in once]
    pending = "prompt tokens 100\ncache miss\n"
    closed = pending + "prefill 50.00 tok/s (100 tok in 2.00s)\ndecode 10.00 tok/s (20 tok in 2.00s)\n"
    rows = printed_requests(parse_requests(pending + closed + closed))
    assert len(rows) == 1
    assert rows[0].decode.startswith("decode ")


def test_a_rolled_log_keeps_one_row_when_the_prefill_line_changes() -> None:
    first = """\
prompt tokens 505
cache miss
prefill 763.24 tok/s (505 tok in 0.66s)
decode 68.43 tok/s (27 tok in 0.39s)
prompt tokens 11795
cache hit 532
prefill 883.45 tok/s (11263 tok in 12.75s)
decode 64.08 tok/s (41 tok in 0.64s)
"""
    replay = """\
prompt tokens 11795
cache hit 532
prefill 763.24 tok/s (505 tok in 0.66s)
decode 68.43 tok/s (27 tok in 0.39s)
prompt tokens 505
cache miss
prefill 763.24 tok/s (505 tok in 0.66s)
decode 68.43 tok/s (27 tok in 0.39s)
prompt tokens 11795
cache hit 532
prefill 110.58 tok/s (11516 tok in 104.14s)
decode 8.56 tok/s (42 tok in 4.91s)
"""
    rows = printed_requests(parse_requests((first + replay) * 1500))
    assert [row.prompt_tokens for row in rows] == [505, 11795]
    assert rows[1].prefill.startswith("prefill 110.58 tok/s")
    assert rows[1].decode_tokens == 42


def test_model_stem_drops_a_variant_suffix() -> None:
    assert model_stem("qwen38_27b") == "qwen38_27b"
    assert model_stem("qwen38_27b:thinking") == "qwen38_27b"


def test_source_files_read_shared_package_largest_first(tmp_path: Path) -> None:
    package = tmp_path / "src" / "favlo"
    package.mkdir(parents=True)
    (package / "a.py").write_text("a" * 10)
    (package / "b.py").write_text("b" * 30)
    (package / "empty.py").write_text("")
    (package / "note.bin").write_text("no")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "c.py").write_text("c" * 100)
    (tmp_path / "pipelines").mkdir()
    (tmp_path / "pipelines" / "big.py").write_text("d" * 200)
    found = source_files(tmp_path)
    assert [path.name for path in found] == ["b.py", "a.py"]
    allowed = {path.relative_to(package).as_posix() for path in found}
    assert read_repository_file(tmp_path, "b.py", allowed) == "b" * 30
    assert read_repository_file(tmp_path, "B.PY", allowed) == "b" * 30
    assert read_repository_file(tmp_path, "missing.py", allowed, "b.py") == "b" * 30
    with pytest.raises(ValueError):
        read_repository_file(tmp_path, "../secret.txt", allowed)
    with pytest.raises(ValueError):
        read_repository_file(tmp_path, "pipelines/big.py", allowed)


def test_instruct_model_pins_the_instruct_card() -> None:
    from harness.file_session import instruct_model

    assert instruct_model("qwen36_35b_a3b") == "qwen36_35b_a3b:instruct"
    assert instruct_model("qwen36_35b_a3b:thinking") == "qwen36_35b_a3b:instruct"


def test_parse_requests_and_prefix_extension() -> None:
    requests = parse_requests(_LOG)
    assert len(requests) == 3
    assert requests[0].missed
    assert requests[0].prefix_hit is None
    assert requests[0].prefix_load is None
    assert requests[0].decode_tokens == 20
    assert requests[0].prefill.startswith("prefill ")
    assert requests[0].steady == ()
    assert requests[1].prefix_load == 120
    assert requests[1].prefix_hit == 120
    assert requests[1].prompt_tokens == 180
    assert requests[2].prefix_load is None
    assert requests[2].prefix_hit == 184
    assert extensions_hold(requests)
    assert not extensions_hold(requests[:1])
    missed = parse_requests(_LOG.replace("cache hit 120", "cache miss"))
    assert not extensions_hold(missed)
    short = parse_requests(_LOG.replace("cache hit 120", "cache hit 100"))
    assert not extensions_hold(short)
    memory = parse_requests(_LOG.replace("prefix load 120 tokens\n", ""))
    assert not extensions_hold(memory)
    later = _LOG.replace(
        "prompt tokens 300\ncache hit 184",
        "prefix load 184 tokens\nprompt tokens 300\ncache hit 184",
    )
    assert extensions_hold(parse_requests(later))
    wrong = later.replace(
        "prefix load 184 tokens\nprompt tokens 300\ncache hit 184",
        "prefix load 100 tokens\nprompt tokens 300\ncache hit 100",
    )
    assert not extensions_hold(parse_requests(wrong))
    after_prompt = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 20
prefix load 12 tokens
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    loaded = parse_requests(after_prompt)
    assert loaded[1].prefix_load == 12
    assert extensions_hold(loaded)
    followed = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prefix load 24 tokens
prompt tokens 40
cache hit 24
prefill 1.00 tok/s (16 tok in 16.00s)
decode 1.00 tok/s (3 tok in 3.00s)
"""
    agent = parse_requests(followed)
    assert agent[1].prefix_load is None
    assert agent[1].prefix_hit == 12
    assert agent[2].prefix_load == 24
    assert extensions_hold(agent)
    warm = """\
prefix load 50 tokens
prompt tokens 50
cache hit 50
prefill 1.00 tok/s (0 tok in 0.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 80
cache hit 52
prefill 1.00 tok/s (28 tok in 2.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prefix load 84 tokens
prompt tokens 100
cache hit 84
prefill 1.00 tok/s (16 tok in 2.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    assert extensions_hold(parse_requests(warm))


def test_prefix_was_hit_requires_a_loaded_prefix() -> None:
    miss = RequestRate(10, None, True, "", "", "", 1, "", ())
    cold = RequestRate(10, 0, False, "", "", "", 1, "", ())
    hit = RequestRate(12, 8, False, "", "", "", 1, "", ())
    assert prefix_was_hit(miss) is False
    assert prefix_was_hit(cold) is False
    assert prefix_was_hit(hit) is True
    first = RequestRate(365, None, True, "", "", "", 25, "", ())
    loaded = RequestRate(365, 361, False, "", "", "", 25, "", (), 361)
    longer = RequestRate(13581, 440, False, "", "", "", 100, "", ())
    memory = RequestRate(365, 361, False, "", "", "", 25, "", ())
    assert boundary_reloaded(first, loaded) is True
    assert boundary_reloaded(first, longer) is False
    assert boundary_reloaded(first, memory) is False
    previous = RequestRate(13531, 390, False, "", "", "", 475, "", ())
    applied = RequestRate(15000, 14006, False, "", "", "", 10, "", (), 14006)
    suffix = RequestRate(15000, 13527, False, "", "", "", 10, "", (), 13527)
    short = RequestRate(15000, 361, False, "", "", "", 10, "", (), 361)
    assert preserved_reloaded(previous, applied) is True
    assert preserved_reloaded(previous, suffix) is True
    assert preserved_reloaded(previous, short) is False
    stored = RequestRate(50102, None, True, "", "", "", 292, "", ())
    resumed = RequestRate(50278, 31136, False, "", "", "", 10, "", (), 31136)
    full = RequestRate(50278, 50102, False, "", "", "", 10, "", (), 50102)
    cold = RequestRate(50278, None, True, "", "", "", 10, "", ())
    assert rewritten_loaded(stored, resumed) is True
    assert rewritten_loaded(stored, full) is False
    assert rewritten_loaded(stored, cold) is False
    ladder = [
        RequestRate(160, None, True, "", "", "", 13, "", ()),
        RequestRate(70824, 173, False, "", "", "", 116, "", (), 173),
    ]
    shorter = RequestRate(70970, 46358, False, "", "", "", 13, "", (), 46358)
    same = RequestRate(70970, 70824, False, "", "", "", 13, "", (), 70824)
    assert rewrite_note(ladder, shorter) == ""
    assert extensions_hold([*ladder, shorter]) is False
    assert rewrite_note(ladder, same) == "prefix was not reloaded"


def test_the_file_agent_replays_the_transcript() -> None:
    _client, agent = _open_agent(
        "http://127.0.0.1:8129/v1",
        "inkling_2_mini",
        None,
        mtp=None,
        mtp_depth=3,
    )
    assert agent.default_options["store"] is False


def test_log_since_keeps_only_the_lines_after_the_mark() -> None:
    mark = "prompt tokens 10\ncache miss\n"
    older = "request open qwen36_35b_a3b\n" + mark
    fresh = (
        "request open qwen36_35b_a3b\n"
        "prompt tokens 12\n"
        "prefix load 8 tokens\n"
        "cache hit 8\n"
        "prefill 1.00 tok/s (4 tok in 4.00s)\n"
        "decode 1.00 tok/s (1 tok in 1.00s)\n"
    )
    parsed = parse_requests(pipeline_log(log_since(older + fresh, mark), "qwen36_35b_a3b"))
    assert len(parsed) == 1
    assert parsed[0].prompt_tokens == 12
    assert prefix_was_hit(parsed[0])
    assert log_since(fresh, "missing mark") == fresh

