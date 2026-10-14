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


def test_readme_states_results_not_run():
    assert "Results: NOT YET RUN" in (ROOT / "README.md").read_text(encoding="utf-8")
