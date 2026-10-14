import re
from pathlib import Path

import pytest

import refusalscope
from refusalscope.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", refusalscope.__version__)


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert refusalscope.__version__ in capsys.readouterr().out


def test_cli_without_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "refusalscope" in capsys.readouterr().err


def test_license_names_author():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Ammon Jackson" in text


def test_gitignore_protects_user_data_and_mock_output():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in lines
    assert "results/MOCK*" in lines
    assert "runs/" in lines


def test_readme_states_results_have_been_run():
    # Two real pairs have landed under results-20261014; the status banner must
    # name them rather than claiming the suite has never run.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Results: RUN" in readme
    assert "qwen3-4b-rocm-1.json" in readme and "qwen3-8b-rocm-1.json" in readme


def test_example_configs_validate(capsys):
    from refusalscope.config import load_config

    example = load_config(ROOT / "configs" / "example_qwen3.json")
    assert example.pair_ids() == ["qwen3-4b", "qwen3-8b"]
    assert not example.is_mock
    mock = load_config(ROOT / "configs" / "mock_study.json")
    assert mock.is_mock


def test_check_config_command(capsys):
    assert main(["check-config", "--config", str(ROOT / "configs" / "example_qwen3.json")]) == 0
    out = capsys.readouterr().out
    assert "pair qwen3-4b" in out and "pair qwen3-8b" in out
    assert main(["check-config", "--config", str(ROOT / "configs" / "mock_study.json")]) == 0
    assert "MOCK configuration" in capsys.readouterr().out


def test_check_config_reports_errors(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["check-config", "--config", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
