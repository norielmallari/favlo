from madmetal.cli import main


def test_help_lists_the_model_flag(capsys) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--qwen38-flash-next" in captured.out


def test_missing_flag_is_an_error(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "--qwen38-flash-next" in captured.err


def test_unknown_model_is_an_error(capsys) -> None:
    assert main(["--another-model"]) == 2
    captured = capsys.readouterr()
    assert "unknown model flag" in captured.err


def test_pipeline_help_comes_from_the_hyphenated_file(capsys) -> None:
    assert main(["--qwen38-flash-next", "--help"]) == 0
    captured = capsys.readouterr()
    assert "probe" in captured.out
