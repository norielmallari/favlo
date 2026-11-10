"""Flavor registry. Directories are temporary. No device."""

import json

import pytest

from favlo.registry import (
    add_model,
    choose_complete,
    default_llm_id,
    format_listing,
    list_unregistered,
    load_registry,
    model_objects,
    one_model,
    refuse_conflicting_destination,
    register_quantized,
    resolve,
    save_registry,
    served_model_objects,
    set_alias,
    set_model_mtp,
)

PIPELINE = "qwen36_35b_a3b"


def _record(root, name: str) -> None:
    path = root / "models" / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"complete": True}))


def test_resolve_uses_the_public_id_not_the_folder_or_pipeline(tmp_path) -> None:
    folder = "Qwen-h128-fp8-c384-fp4"
    _record(tmp_path, folder)
    add_model(folder, alias="flash", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(LookupError, match="unknown model"):
        resolve(PIPELINE, root=tmp_path)
    with pytest.raises(LookupError, match="unknown model"):
        resolve(folder, root=tmp_path)
    with pytest.raises(LookupError, match="unknown model"):
        resolve(f"{folder}:instruct", root=tmp_path)
    thinking = resolve("flash:thinking", root=tmp_path)
    assert thinking.folder == folder
    assert thinking.variant == "thinking"
    assert thinking.sampling is not None
    assert thinking.sampling["temperature"] == 1.0
    instruct = resolve("flash:instruct", root=tmp_path)
    assert instruct.variant == "instruct"
    assert instruct.parent == "flash"
    assert instruct.sampling is not None
    assert instruct.sampling["presence_penalty"] == 1.5
    assert resolve("flash", root=tmp_path).folder == folder


def test_unknown_suffix_names_the_registered_ids(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    with pytest.raises(LookupError, match="short") as raised:
        resolve("short:draft", root=tmp_path)
    assert "packed" not in str(raised.value)


def test_default_llm_id_uses_the_registry_row(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    assert default_llm_id(tmp_path) == "short:instruct"
    data = load_registry(tmp_path)
    data["default"] = "short:thinking"
    save_registry(data, tmp_path)
    assert default_llm_id(tmp_path) == "short:thinking"


def test_listing_uses_the_alias_once_with_variants(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    rows = model_objects(tmp_path, resident="packed")
    ids = [row["id"] for row in rows]
    assert ids == [
        "short",
        "short:thinking",
        "short:instruct",
    ]
    assert "packed" not in ids
    assert all(row["loaded"] for row in rows)
    assert rows[0]["owned_by"] == "favlo"
    assert rows[0]["object"] == "model"
    assert rows[0]["root"] == "packed"
    assert rows[0]["context_length"] == 131072
    thinking = rows[1]
    assert thinking["variant"] == "thinking"
    assert thinking["parent"] == "short"
    assert thinking["sampling"]["top_k"] == 20
    with pytest.raises(LookupError, match="unknown model"):
        resolve("packed", root=tmp_path)
    assert resolve("short", root=tmp_path).folder == "packed"


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
    assert data["default"] == "one:instruct"
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
    small = "Qwen3.6-35B-A3B-fp4"
    dense = "Qwen3.8-27B-fp4"
    _record(tmp_path, small)
    _record(tmp_path, dense)
    add_model(small, alias="qwen36_35b_a3b", root=tmp_path, pipeline="qwen36_35b_a3b", runtime="metal41")
    add_model(dense, alias="qwen38_27b", root=tmp_path, pipeline="qwen38_27b", runtime="metal41")
    data = load_registry(tmp_path)
    assert data["default"] == "qwen36_35b_a3b:instruct"
    assert resolve("qwen36_35b_a3b", root=tmp_path).folder == small
    assert resolve("qwen36_35b_a3b", root=tmp_path).pipeline == "qwen36_35b_a3b"
    selected = resolve("qwen38_27b", root=tmp_path)
    assert selected.folder == dense
    assert selected.pipeline == "qwen38_27b"
    ids = [row["id"] for row in model_objects(tmp_path)]
    assert "qwen36_35b_a3b" in ids
    assert "qwen38_27b" in ids
    assert "qwen38_27b:thinking" in ids
    assert small not in ids
    assert dense not in ids
    listing = format_listing(tmp_path)
    assert "pipeline qwen38_27b" in listing
    assert "pipeline qwen36_35b_a3b" in listing


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
    assert stored["default"] == "short:instruct"
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
    assert flash["variants"]["thinking"]["presence_penalty"] == 1.5
    assert flash["variants"]["instruct"]["presence_penalty"] == 1.5
    bare = resolve("packed", root=tmp_path)
    assert bare.sampling is None
    assert bare.variant is None
    assert bare.default_variant == "thinking"
    assert bare.variants is not None
    assert bare.variants["thinking"]["presence_penalty"] == 1.5
    assert bare.variants["instruct"]["presence_penalty"] == 1.5


def test_qwen36_registration_uses_that_pipeline_card(tmp_path) -> None:
    _record(tmp_path, "a3b")
    add_model(
        "a3b",
        alias="qwen36_35b_a3b",
        root=tmp_path,
        pipeline="qwen36_35b_a3b",
        runtime="metal41",
    )
    entry = load_registry(tmp_path)["models"]["a3b"]
    assert entry["default_variant"] == "thinking"
    assert entry["variants"]["thinking"]["presence_penalty"] == 1.5
    assert entry["variants"]["thinking"]["temperature"] == 1.0
    assert entry["variants"]["thinking"]["reasoning_effort"] == "high"
    assert entry["variants"]["instruct"]["temperature"] == 0.7
    assert entry["variants"]["instruct"]["presence_penalty"] == 1.5
    assert entry["variants"]["instruct"]["reasoning_effort"] == "none"
    bare = resolve("qwen36_35b_a3b", root=tmp_path)
    assert bare.sampling is None
    assert bare.variant is None
    assert bare.default_variant == "thinking"
    assert bare.reasoning_effort == "high"
    assert bare.variants is not None
    assert bare.variants["thinking"]["presence_penalty"] == 1.5
    forced = resolve("qwen36_35b_a3b:instruct", root=tmp_path)
    assert forced.variant == "instruct"
    assert forced.sampling is not None
    assert forced.sampling["temperature"] == 0.7
    assert "reasoning_effort" not in forced.sampling
    assert forced.reasoning_effort == "none"
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
    with pytest.raises(LookupError, match="unknown model"):
        resolve("packed", root=tmp_path)
    assert resolve("short", root=tmp_path).path == packed
    path, selection = choose_complete(
        model=None, record=None, root=tmp_path, pipeline=PIPELINE
    )
    assert path == packed
    assert selection is not None
    assert selection.path == packed
    assert entry["default_variant"] == "thinking"
    assert entry["variants"]["thinking"]["presence_penalty"] == 1.5


def test_model_mtp_is_stored_on_the_row(tmp_path) -> None:
    _record(tmp_path, "packed")
    add_model("packed", alias="short", root=tmp_path, pipeline=PIPELINE, runtime="metal41")
    assert resolve("short", root=tmp_path).mtp is False
    assert resolve("short", root=tmp_path).mtp_depth == 3
    assert "mtp off depth 3" in format_listing(tmp_path)
    text = set_model_mtp("short", enabled=True, depth=4, root=tmp_path)
    assert text == "packed mtp on depth 4"
    selected = resolve("short", root=tmp_path)
    assert selected.mtp is True
    assert selected.mtp_depth == 4
    row = model_objects(tmp_path)[0]
    assert row["mtp"] is True
    assert row["mtp_depth"] == 4
    with pytest.raises(LookupError, match="unknown model"):
        set_model_mtp("packed", enabled=False, root=tmp_path)
    kept = set_model_mtp("short", enabled=False, root=tmp_path)
    assert kept == "packed mtp off depth 4"


def test_the_first_image_row_fills_an_empty_media_default(monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    class _Image:
        kind = "image"

    module = SimpleNamespace(
        mcp_tools=lambda: [{"name": "generate_image", "handler": "handle_image"}],
        handle_image=_Image,
    )
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    data: dict = {"models": {}}
    registry._ensure_media_default(data, "folder-a", "pipe", "metal41", None)
    assert data["media"]["image"] == "folder-a"
    registry._ensure_media_default(data, "folder-b", "pipe", "metal41", None)
    assert data["media"]["image"] == "folder-a"


def test_an_empty_media_key_is_filled_from_a_registered_row(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    class _Image:
        kind = "image"

    module = SimpleNamespace(
        mcp_tools=lambda: [{"name": "generate_image", "handler": "handle_image"}],
        handle_image=_Image,
    )
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    root = tmp_path
    (root / "models").mkdir()
    registry.save_registry(
        {
            "version": 1,
            "models": {
                "still-a": {"pipeline": "pipe", "runtime": "metal41", "directory": "models/still-a"},
                "still-b": {"pipeline": "pipe", "runtime": "metal41", "directory": "models/still-b"},
            },
        },
        root,
    )
    assert registry.media_default("image", root) == "still-a"
    assert registry.load_registry(root)["media"]["image"] == "still-a"


def test_a_stored_media_default_is_kept(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    class _Image:
        kind = "image"

    module = SimpleNamespace(
        mcp_tools=lambda: [{"name": "generate_image", "handler": "handle_image"}],
        handle_image=_Image,
    )
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    root = tmp_path
    (root / "models").mkdir()
    registry.save_registry(
        {
            "version": 1,
            "media": {"image": "still-b"},
            "models": {
                "still-a": {"pipeline": "pipe", "runtime": "metal41"},
                "still-b": {"pipeline": "pipe", "runtime": "metal41"},
            },
        },
        root,
    )
    assert registry.media_default("image", root) == "still-b"


def test_an_embeddings_row_fills_an_empty_registry_default(monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    module = SimpleNamespace(handle_embed=lambda *args: None)
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    data: dict = {"models": {}}
    registry._ensure_embeddings_default(data, "folder-a", "pipe", "metal41", False)
    assert data["embeddings"] == "folder-a"
    registry._ensure_embeddings_default(data, "folder-b", "pipe", "metal41", False)
    assert data["embeddings"] == "folder-a"


def test_an_empty_embeddings_key_is_filled_from_a_registered_row(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    module = SimpleNamespace(handle_embed=lambda *args: None)
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    root = tmp_path
    (root / "models").mkdir()
    registry.save_registry(
        {
            "version": 1,
            "models": {
                "vec-a": {"pipeline": "pipe", "runtime": "metal41", "directory": "models/vec-a"},
                "vec-b": {"pipeline": "pipe", "runtime": "metal41", "directory": "models/vec-b"},
            },
        },
        root,
    )
    assert registry.embeddings_default(root) == "vec-a"
    assert registry.load_registry(root)["embeddings"] == "vec-a"
    assert registry.embeddings_default(root) == "vec-a"


def test_a_stored_embeddings_default_is_kept(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo import registry

    module = SimpleNamespace(handle_embed=lambda *args: None)
    monkeypatch.setattr("favlo.cli.load_pipeline", lambda *args, **kwargs: module)
    root = tmp_path
    (root / "models").mkdir()
    registry.save_registry(
        {
            "version": 1,
            "embeddings": "vec-b",
            "models": {
                "vec-a": {"pipeline": "pipe", "runtime": "metal41"},
                "vec-b": {"pipeline": "pipe", "runtime": "metal41"},
            },
        },
        root,
    )
    assert registry.embeddings_default(root) == "vec-b"


def test_served_models_list_language_and_embeddings_from_registry_only(
    tmp_path, monkeypatch
) -> None:
    def refuse_load(*_args, **_kwargs):
        raise AssertionError("served listing must not import a pipeline")

    monkeypatch.setattr("favlo.cli.load_pipeline", refuse_load)
    (tmp_path / "models").mkdir()
    save_registry(
        {
            "version": 1,
            "embeddings": "vec-pack",
            "models": {
                "chat-pack": {
                    "pipeline": "chat_pipe",
                    "runtime": "metal41",
                    "alias": "chat",
                    "variants": {
                        "thinking": {"temperature": 1.0},
                        "instruct": {"temperature": 0.7},
                    },
                    "default_variant": "thinking",
                    "context_length": 8192,
                },
                "vec-pack": {
                    "pipeline": "embed_pipe",
                    "runtime": "metal41",
                    "alias": None,
                    "variants": {"embed": {}},
                    "default_variant": "embed",
                    "context_length": 4096,
                },
                "still-pack": {
                    "pipeline": "image_pipe",
                    "runtime": "metal41",
                    "alias": None,
                    "variants": {"generate": {}},
                    "default_variant": "generate",
                    "context_length": 512,
                },
            },
        },
        tmp_path,
    )
    all_ids = [row["id"] for row in model_objects(tmp_path)]
    assert "still-pack" in all_ids
    served = served_model_objects(tmp_path)
    assert [row["id"] for row in served] == [
        "chat",
        "chat:thinking",
        "chat:instruct",
        "vec-pack",
        "vec-pack:embed",
    ]
    assert one_model("chat:instruct", root=tmp_path)["pipeline"] == "chat_pipe"
    assert one_model("vec-pack", root=tmp_path)["pipeline"] == "embed_pipe"
    with pytest.raises(LookupError, match="not found"):
        one_model("still-pack", root=tmp_path)

