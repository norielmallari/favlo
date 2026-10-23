"""Install copies plugins into the default home and links them from each profile."""

from __future__ import annotations

from pathlib import Path

from favlo.hermes_plugins import edit_config, hermes_root, install, main

SOURCE = Path(__file__).resolve().parents[1] / "hermes-agent-plugins"


def test_edit_config_enables_both_backends_once() -> None:
    original = """\
known_builtin_toolsets:
  cli:
    - image_gen
platform_toolsets:
  cli:
    - file
    - image_gen
    - memory
  discord:
    - hermes-discord
plugins:
  disabled: []
  enabled:
    - homeassistant
    - image_gen/favlo
image_gen:
  provider: other
  model: kept
"""
    once = edit_config(original)
    twice = edit_config(once)
    assert twice == once
    assert once.count("- image_gen/favlo") == 1
    assert once.count("- video_gen/favlo") == 1
    assert "    - image_gen\n    - video_gen\n    - memory" in once
    assert "    - hermes-discord" in once
    assert "known_builtin_toolsets:\n  cli:\n    - image_gen\n" in once
    assert "provider: other" in once
    assert "model: kept" in once
    assert "video_gen:\n  provider: favlo_video_gen\n  model: wan22-i2v\n" in once
    bare = edit_config("platform_toolsets:\n  cli:\n    - file\n")
    assert "    - file\n    - image_gen\n    - video_gen\n" in bare
    shared = edit_config(
        "image_gen:\n  provider: favlo\n  model: z-image-turbo-fp4\n"
        "video_gen:\n  provider: favlo\n  model: wan22-i2v\n"
    )
    assert "provider: favlo\n" not in shared
    assert "provider: favlo_image_gen" in shared
    assert "provider: favlo_video_gen" in shared
    assert edit_config(shared) == shared


def test_install_links_each_profile_at_the_default_copy(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    profile = root / "profiles" / "atlas"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    (profile / "config.yaml").write_text(
        "platform_toolsets:\n  cli:\n    - image_gen\n",
        encoding="utf-8",
    )
    labels = install(root, SOURCE)
    assert labels == ["default", "atlas"]
    for home in (root, profile):
        assert (home / "plugins" / "image_gen" / "favlo" / "plugin.yaml").is_file()
        assert (home / "plugins" / "video_gen" / "favlo" / "__init__.py").is_file()
        text = (home / "config.yaml").read_text(encoding="utf-8")
        assert "image_gen/favlo" in text
        assert "video_gen/favlo" in text
    assert hermes_root(profile) == root.resolve()
    default_image = root / "plugins" / "image_gen" / "favlo"
    default_video = root / "plugins" / "video_gen" / "favlo"
    assert default_image.is_symlink() is False
    linked = profile / "plugins" / "image_gen" / "favlo"
    linked_video = profile / "plugins" / "video_gen" / "favlo"
    assert linked.is_symlink()
    assert linked_video.is_symlink()
    assert linked.resolve() == default_image.resolve()
    assert linked_video.resolve() == default_video.resolve()
    install(root, SOURCE)
    enabled = (root / "config.yaml").read_text(encoding="utf-8")
    assert enabled.count("image_gen/favlo") == 1
    assert enabled.count("video_gen/favlo") == 1
    assert linked.is_symlink()


def test_missing_home_is_an_error(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing"
    assert main(["--home", str(missing)]) == 1
    captured = capsys.readouterr()
    assert "hermes home is missing" in captured.err
