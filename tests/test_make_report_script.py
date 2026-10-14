import json
import subprocess
import sys
from pathlib import Path

import pytest

from refusalscope.demo import run_demo

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_report.py"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_script_refuses_when_there_are_no_results(tmp_path):
    (tmp_path / "results").mkdir()
    result = run_script("--results", str(tmp_path / "results"), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "NOT YET RUN" in result.stderr
    assert not (tmp_path / "out").exists()


def test_script_refuses_the_repository_results_directory_today():
    assert not list((ROOT / "results").glob("*.json"))
    result = run_script("--out", str(ROOT / "tmp" / "should-not-exist"))
    assert result.returncode == 2 and "NOT YET RUN" in result.stderr
    assert not (ROOT / "tmp" / "should-not-exist").exists()


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    demo = run_demo(out, limit=10, n_boot=100)
    return demo["results_path"].parent


def test_script_refuses_mock_results_unless_allowed(demo_results, tmp_path):
    refused = run_script("--results", str(demo_results), "--out", str(tmp_path / "out"))
    assert refused.returncode == 2 and "only MOCK result files" in refused.stderr
    allowed = run_script(
        "--results", str(demo_results), "--out", str(tmp_path / "out"), "--allow-mock"
    )
    assert allowed.returncode == 0
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names and all(n.startswith("MOCK") for n in names)


def test_script_never_touches_the_readme_with_mock_results(demo_results, tmp_path):
    readme = tmp_path / "README.md"
    original = (ROOT / "README.md").read_text(encoding="utf-8")
    readme.write_text(original, encoding="utf-8")
    result = run_script("--results", str(demo_results), "--out", str(tmp_path / "out"), "--allow-mock",
                        "--update-readme", str(readme))  # fmt: skip
    assert result.returncode == 2 and "refusing to write MOCK numbers" in result.stderr
    assert readme.read_text(encoding="utf-8") == original


def test_script_rejects_malformed_results(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    result = run_script("--results", str(tmp_path / "results"), "--out", str(tmp_path / "out"))
    assert result.returncode == 2 and "not a refusalscope results file" in result.stderr
