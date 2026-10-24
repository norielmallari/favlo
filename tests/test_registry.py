"""Flavor registry. Directories are temporary. No device."""

import json

import pytest

from favlo.registry import (
    add_model,
    choose_complete,
    format_listing,
    list_unregistered,
    load_registry,
    model_objects,
    refuse_conflicting_destination,
    register_quantized,
    resolve,
    save_registry,
    set_alias,
)

PIPELINE = "qwen38-flash-next"


def _record(root, name: str) -> None:
    path = root / "models" / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"complete": True}))


def test_pipeline_id_names_the_only_row_of_that_pipeline(tmp_path) -> None:
    folder = "Qwen-h128-fp8-c384-fp4"
    _record(tmp_path, folder)
    add_model(folder, alias="flash", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    selection = resolve(PIPELINE, root=tmp_path)
    assert selection.folder == folder
    assert selection.requested_id == PIPELINE
    thinking = resolve("flash:thinking", root=tmp_path)
    assert thinking.folder == folder
    assert thinking.variant == "thinking"
    assert thinking.sampling is not None
    assert thinking.sampling["temperature"] == 1.0
    instruct = resolve(f"{folder}:instruct", root=tmp_path)
    assert instruct.variant == "instruct"
    assert instruct.parent == folder
    assert instruct.sampling is not None
    assert instruct.sampling["presence_penalty"] == 1.5
    _record(tmp_path, "other")
    add_model("other", alias="other-name", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(LookupError, match="unknown model"):
        resolve(PIPELINE, root=tmp_path)
    assert resolve("flash", root=tmp_path).folder == folder


def test_unknown_suffix_names_the_registered_ids(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(LookupError, match="short") as raised:
        resolve("short:draft", root=tmp_path)
    assert "packed" in str(raised.value)


def test_listing_includes_folder_alias_and_variants(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    rows = model_objects(tmp_path, resident="packed")
    ids = [row["id"] for row in rows]
    assert ids == [
        "packed",
        "packed:thinking",
        "packed:instruct",
        "short",
        "short:thinking",
        "short:instruct",
    ]
    assert all(row["loaded"] for row in rows)
    assert rows[0]["owned_by"] == "favlo"
    assert rows[0]["object"] == "model"
    assert rows[0]["context_length"] == 131072
    thinking = rows[1]
    assert thinking["variant"] == "thinking"
    assert thinking["parent"] == "packed"
    assert thinking["sampling"]["top_k"] == 20


def test_registration_stores_the_passed_max_prompt_tokens(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model(
        "packed",
        root=tmp_path,
        pipeline=PIPELINE,
        runtime="metal41",
        max_prompt_tokens=65536,
    )
    assert load_registry(tmp_path)["models"]["packed"]["max_prompt_tokens"] == 65536
    source = tmp_path / "bf16"
    source.mkdir()
    _record(tmp_path, "again")
    register_quantized(
        "again",
        source=source,
        alias=None,
        directory=tmp_path / "models" / "again",
        root=tmp_path,
        pipeline=PIPELINE, runtime="metal41",
        max_prompt_tokens=65536,
    )
    assert load_registry(tmp_path)["models"]["again"]["max_prompt_tokens"] == 65536


def test_model_row_includes_configured_max_prompt_tokens(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    data = load_registry(tmp_path)
    data["models"]["packed"]["max_prompt_tokens"] = 65536
    save_registry(data, tmp_path)
    rows = model_objects(tmp_path)
    assert rows
    assert all(row["max_prompt_tokens"] == 65536 for row in rows)
    assert resolve("short", root=tmp_path).max_prompt_tokens == 65536


def test_a_finished_record_stays_unregistered(tmp_path) -> None:
    folder = "Qwen3.8-Flash-Next-h128-fp4-c384-fp4"
    _record(tmp_path, folder)
    data = load_registry(tmp_path)
    assert data == {"version": 1, "models": {}}
    assert list_unregistered(tmp_path) == [folder]
    with pytest.raises(LookupError, match="unknown model"):
        resolve(PIPELINE, root=tmp_path)
    path = tmp_path / "models" / "partial"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"complete": False}))
    assert list_unregistered(tmp_path) == [folder]


def test_choose_complete_lists_models_when_default_is_unset(tmp_path) -> None:
    with pytest.raises(LookupError, match="no registered model"):
        choose_complete(model=None, record=None, root=tmp_path, pipeline=PIPELINE)
    _record(tmp_path, "one")
    add_model("one", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    path, selection = choose_complete(model=None, record=None, root=tmp_path, pipeline=PIPELINE)
    assert path.name == "one"
    assert selection is not None
    _record(tmp_path, "two")
    add_model("two", alias="second", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    data = load_registry(tmp_path)
    assert "default" not in data
    data["extra"] = {"kept": True}
    save_registry(data, tmp_path)
    with pytest.raises(LookupError, match="pass --model"):
        choose_complete(model=None, record=None, root=tmp_path, pipeline=PIPELINE)
    explicit, none = choose_complete(model=None, record=path, root=tmp_path)
    assert explicit == path
    assert none is None
    set_alias("one", "first", root=tmp_path)
    assert load_registry(tmp_path)["extra"] == {"kept": True}


def test_alias_must_be_one_unused_name(tmp_path) -> None:
    _record(tmp_path, "one")
    _record(tmp_path, "two")
    add_model("one", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(ValueError, match="folder"):
        add_model("two", alias="one", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(ValueError, match="already used"):
        add_model("two", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(ValueError, match="single name"):
        set_alias("one", "bad:name", root=tmp_path)
    set_alias("one", "renamed", root=tmp_path)
    assert resolve("renamed", root=tmp_path).folder == "one"


def test_a_second_pipeline_is_listed_without_a_default(tmp_path) -> None:
    flash = "Qwen-h128-fp8-c384-fp4"
    dense = "Qwen3.8-27B-fp4"
    _record(tmp_path, flash)
    _record(tmp_path, dense)
    add_model(flash, alias="qwen38-flash-next", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    add_model(dense, alias="qwen38-27b", root=tmp_path, pipeline="qwen38-27b", runtime="metal41")
    data = load_registry(tmp_path)
    assert "default" not in data
    assert resolve("qwen38-flash-next", root=tmp_path).folder == flash
    assert resolve("qwen38-flash-next", root=tmp_path).pipeline == "qwen38-flash-next"
    selected = resolve("qwen38-27b", root=tmp_path)
    assert selected.folder == dense
    assert selected.pipeline == "qwen38-27b"
    ids = [row["id"] for row in model_objects(tmp_path)]
    assert "qwen38-flash-next" in ids
    assert "qwen38-27b" in ids
    assert "qwen38-27b:thinking" in ids
    listing = format_listing(tmp_path)
    assert "pipeline qwen38-27b" in listing
    assert "pipeline qwen38-flash-next" in listing


def test_unregistered_complete_directory_is_listed(tmp_path) -> None:
    _record(tmp_path, "loose")
    assert list_unregistered(tmp_path) == ["loose"]
    listing = format_listing(tmp_path)
    assert listing == "unregistered loose"


def test_quantize_registration_refuses_a_different_source(tmp_path) -> None:
    source = tmp_path / "bf16"
    source.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    packed = tmp_path / "models" / "packed"
    register_quantized(
        "packed",
        source=source,
        alias="short",
        directory=packed,
        root=tmp_path,
        pipeline=PIPELINE,
        runtime="metal41",
    )
    stored = load_registry(tmp_path)
    assert "default" not in stored
    assert list(stored["models"]) == ["packed"]
    register_quantized(
        "packed",
        source=source,
        alias="short",
        directory=packed,
        root=tmp_path,
        pipeline=PIPELINE,
        runtime="metal41",
    )
    with pytest.raises(ValueError, match="different source"):
        refuse_conflicting_destination("packed", other, root=tmp_path)


def test_registration_copies_the_pipeline_card(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    flash = load_registry(tmp_path)["models"]["packed"]
    assert flash["default_variant"] == "thinking"
    assert flash["variants"]["thinking"]["presence_penalty"] == 0.0
    assert flash["variants"]["instruct"]["presence_penalty"] == 1.5
    bare = resolve("packed", root=tmp_path)
    assert bare.sampling is None
    assert bare.variant is None
    assert bare.default_variant == "thinking"
    assert bare.variants is not None
    assert bare.variants["thinking"]["presence_penalty"] == 0.0
    assert bare.variants["instruct"]["presence_penalty"] == 1.5


def test_qwen36_registration_uses_that_pipeline_card(tmp_path) -> None:
    _record(tmp_path, "a3b")
    add_model(
        "a3b",
        alias="qwen36-35b-a3b",
        root=tmp_path,
        pipeline="qwen36-35b-a3b",
        runtime="metal41",
    )
    entry = load_registry(tmp_path)["models"]["a3b"]
    assert entry["default_variant"] == "thinking"
    assert entry["variants"]["thinking"]["presence_penalty"] == 1.5
    assert entry["variants"]["thinking"]["temperature"] == 1.0
    assert entry["variants"]["instruct"]["temperature"] == 0.7
    assert entry["variants"]["instruct"]["presence_penalty"] == 1.5
    bare = resolve("qwen36-35b-a3b", root=tmp_path)
    assert bare.sampling is None
    assert bare.variant is None
    assert bare.default_variant == "thinking"
    assert bare.variants is not None
    assert bare.variants["thinking"]["presence_penalty"] == 1.5
    forced = resolve("qwen36-35b-a3b:instruct", root=tmp_path)
    assert forced.variant == "instruct"
    assert forced.sampling is not None
    assert forced.sampling["temperature"] == 0.7
    assert forced.sampling["presence_penalty"] == 1.5


def test_a_row_without_directory_opens_the_models_folder(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    data = load_registry(tmp_path)
    data["models"]["packed"].pop("directory")
    save_registry(data, tmp_path)
    assert resolve("packed", root=tmp_path).path == tmp_path / "models" / "packed"


def test_reregister_refreshes_the_card_from_the_pipeline(tmp_path) -> None:
    source = tmp_path / "bf16"
    source.mkdir()
    packed = tmp_path / "elsewhere" / "packed"
    register_quantized(
        "packed",
        source=source,
        alias="short",
        directory=packed,
        root=tmp_path,
        pipeline=PIPELINE,
        runtime="metal41",
    )
    data = load_registry(tmp_path)
    data["models"]["packed"]["variants"]["thinking"]["presence_penalty"] = 9
    data["models"]["packed"]["default_variant"] = "instruct"
    save_registry(data, tmp_path)
    register_quantized(
        "packed",
        source=source,
        alias="short",
        directory=packed,
        root=tmp_path,
        pipeline=PIPELINE,
        runtime="metal41",
    )
    entry = load_registry(tmp_path)["models"]["packed"]
    assert entry["directory"] == str(packed)
    assert resolve("packed", root=tmp_path).path == packed
    path, selection = choose_complete(
        model=None, record=None, root=tmp_path, pipeline=PIPELINE
    )
    assert path == packed
    assert selection is not None
    assert selection.path == packed
    assert entry["default_variant"] == "thinking"
    assert entry["variants"]["thinking"]["presence_penalty"] == 0.0
