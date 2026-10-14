"""Repository-level promises: honesty rules, docs that match the code, no leftovers."""

import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import tomllib

import refusalscope
from refusalscope.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".toml",
    ".yml",
    ".yaml",
    ".mmd",
    ".sh",
    ".txt",
    ".cfg",
    "",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "report",
    "tmp",
}


def tracked_text_files():
    out = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts
        ):
            continue
        if (
            path.suffix in TEXT_SUFFIXES
            and path.name != ".coverage"
            and path.stat().st_size < 5_000_000
        ):
            out.append(path)
    return out


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


# ------------------------------------------------------------------ honesty rules


def test_readme_says_results_not_yet_run_and_has_fillable_markers():
    text = read("README.md")
    assert "<!-- STATUS:START -->**Results: NOT YET RUN.**<!-- STATUS:END -->" in text
    assert "<!-- RESULTS:START -->" in text and "<!-- RESULTS:END -->" in text
    block = text.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    assert block.count("not run") >= 4 and not re.search(r"\d+\.\d+%", block)


def test_no_real_results_are_committed():
    results = [p for p in (ROOT / "results").iterdir() if p.suffix == ".json"]
    assert results == [], "real result files may only be added after a real run"
    assert not list((ROOT / "results").glob("MOCK*"))


def test_gitignore_keeps_mock_output_and_user_data_out():
    ignored = read(".gitignore").splitlines()
    for pattern in ("results/MOCK*", "data/", "runs/", "report/", "*.harmful.jsonl"):
        assert pattern in ignored


def test_no_harmful_dataset_is_bundled():
    for path in ROOT.rglob("*.csv"):
        if not any(part in SKIP_DIRS for part in path.parts):
            pytest.fail(f"unexpected CSV in repository: {path}")
    assert not (ROOT / "data").exists()


def test_launch_copy_claims_no_result():
    files = sorted((ROOT / "docs" / "launch").glob("*.md"))
    assert {f.name for f in files} == {"LINKEDIN.md", "X_THREAD.md", "DISCORD_SHORT.md", "FAQ.md"}
    claims = re.compile(
        r"\b(we|i) (found|measured|showed|observed|discovered)\b|\bresults? (show|reveal|indicate)|"
        r"\bour results\b|\bsignificantly (lower|higher|worse|better)\b|\boutperform",
        re.IGNORECASE,
    )
    for file in files:
        text = file.read_text(encoding="utf-8")
        assert "%" not in text, (
            f"{file.name}: numbers with percent signs are not allowed before results exist"
        )
        assert not claims.search(text), f"{file.name}: reads like a result claim"
        assert re.search(r"NOT YET RUN|no results? yet|No result", text), (
            f"{file.name}: must state the status"
        )


# ------------------------------------------------------------------ pre-registration order


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def test_preregistration_was_committed_before_any_analysis_code():
    def first(path):
        return (git("log", "--diff-filter=A", "--format=%H", "--", path) or "").splitlines()

    prereg, stats, analysis = (
        first(p)
        for p in (
            "docs/PREREGISTRATION.md",
            "src/refusalscope/stats.py",
            "src/refusalscope/analysis.py",
        )
    )
    if not (prereg and stats and analysis):
        pytest.skip("git history not available")
    for later in (stats[-1], analysis[-1]):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", prereg[-1], later], cwd=ROOT, check=False
        ).returncode
        assert ancestor == 0, "the pre-registration must be an ancestor of the analysis code"


# ------------------------------------------------------------------ leftovers


def test_no_todo_markers_or_model_identifiers_in_the_repository():
    todo = re.compile(r"\b(TODO|FIXME|XXX|TBD|HACK)\b|lorem ipsum", re.IGNORECASE)
    ai = re.compile(r"\b(claude|anthropic|chatgpt|gpt-?\d\w*|gemini)\b", re.IGNORECASE)
    for path in tracked_text_files():
        if path.name == "test_repo_hygiene.py" or path.suffix == ".jsonl":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not todo.search(text), f"{path.relative_to(ROOT)} contains a to-do marker"
        assert not ai.search(text), f"{path.relative_to(ROOT)} names an AI model or vendor product"


def test_snapshots_contain_no_model_identifiers():
    ai = re.compile(r"\b(claude|anthropic|chatgpt|gpt-?\d\w*|gemini)\b", re.IGNORECASE)
    for path in (ROOT / "suites").glob("*.jsonl"):
        assert not ai.search(path.read_text(encoding="utf-8")), path.name


# ------------------------------------------------------------------ docs match the code


def test_version_is_consistent():
    pyproject = tomllib.loads(read("pyproject.toml"))
    assert pyproject["project"]["version"] == refusalscope.__version__
    assert f"## {refusalscope.__version__} - " in read("CHANGELOG.md")
    assert pyproject["project"]["scripts"]["refusalscope"] == "refusalscope.cli:main"
    assert pyproject["project"]["dependencies"] == ["numpy>=1.24"]
    assert (
        pyproject["project"]["name"] == "refusalscope"
        and pyproject["project"]["requires-python"] == ">=3.10"
    )


def test_ci_matrix_and_commands():
    ci = read(".github/workflows/ci.yml")
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f'"{version}"' in ci
    assert "ruff check ." in ci and "pytest --cov" in ci
    assert "make_report.py" in ci  # the refusal path is exercised in CI


def test_makefile_targets_named_in_the_readme_exist():
    makefile = read("Makefile")
    for target in re.findall(r"^make (\w[\w-]*)", read("README.md"), flags=re.MULTILINE):
        assert re.search(rf"^{target}:", makefile, flags=re.MULTILINE), target


def test_documented_files_exist():
    for relative in (
        "docs/PREREGISTRATION.md", "docs/METHODOLOGY.md", "docs/THREATS_TO_VALIDITY.md", "docs/RUNBOOK_WINDOWS.md",
        "docs/FLOWCHARTS.md", "docs/diagrams/usage.mmd", "docs/diagrams/troubleshooting.mmd",
        "docs/diagrams/usage.svg", "docs/diagrams/troubleshooting.svg", "scripts/render_diagrams.sh",
        "scripts/make_report.py", "CONTRIBUTING.md", "CHANGELOG.md", "Makefile", "Dockerfile", "LICENSE",
    ):  # fmt: skip
        assert (ROOT / relative).is_file(), relative
    readme = read("README.md")
    for link in re.findall(r"\]\(([^)#]+)\)", readme):
        if not link.startswith("http"):
            assert (ROOT / link).exists(), f"README links to missing {link}"


def test_flowchart_svgs_are_valid_and_embedded():
    for name in ("usage", "troubleshooting"):
        svg = ROOT / "docs" / "diagrams" / f"{name}.svg"
        root = ET.fromstring(svg.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg") and svg.stat().st_size > 5000
        assert f"docs/diagrams/{name}.svg" in read("README.md")
    doc = read("docs/FLOWCHARTS.md")
    assert doc.count("```mermaid") == 2
    for name in ("usage", "troubleshooting"):
        source = read(f"docs/diagrams/{name}.mmd").rstrip()
        assert source in doc, f"FLOWCHARTS.md is out of date for {name}.mmd"


def test_troubleshooting_covers_the_required_symptoms():
    source = read("docs/diagrams/troubleshooting.mmd").lower()
    for needle in (
        "plain text", "--jinja", "context size", "connection refused", "top_logprobs",
        "accuracy is 0", "grader", "--run-id", "ablation-unverified", "not yet run",
    ):  # fmt: skip
        assert needle in source, needle
    assert source.count("fix") >= 12


def test_usage_chart_covers_the_workflow():
    source = read("docs/diagrams/usage.mmd")
    for needle in ("install", "serve", "config", "doctor", "run", "analyze", "report"):
        assert needle in source.lower(), needle


def command_lines():
    """Every ``refusalscope ...`` command inside fenced code blocks of the docs."""
    for relative in (
        "README.md",
        "docs/RUNBOOK_WINDOWS.md",
        "CONTRIBUTING.md",
        "docs/launch/FAQ.md",
    ):
        in_block, pending = False, ""
        for line in read(relative).splitlines():
            if line.startswith("```"):
                in_block = not in_block
                pending = ""
                continue
            if not in_block:
                continue
            text = pending + line.strip()
            if text.endswith(("\\", "`")):
                pending = text[:-1].strip() + " "
                continue
            pending = ""
            if text.startswith("refusalscope "):
                yield relative, text


def test_every_documented_command_parses():
    parser = build_parser()
    seen = 0
    for relative, line in command_lines():
        argv = shlex.split(line.split("#")[0].replace("\\", "/"))[1:]
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 0, f"{relative}: `{line}` is not a valid command"
        seen += 1
    assert seen >= 15


def test_documented_json_keys_exist_in_the_example_config():
    from refusalscope.config import load_config

    config = load_config(ROOT / "configs" / "example_qwen3.json")
    assert config.suites.refusal_path == "data/harmful_behaviors.csv"
    assert "data/harmful_behaviors.csv" in read("README.md") and "harmful_behaviors.csv" in read(
        "docs/RUNBOOK_WINDOWS.md"
    )
    assert "--jinja" in read("README.md") and "--ctx-size 8192" in read("README.md")
    assert "--jinja" in read("docs/RUNBOOK_WINDOWS.md") and "--ctx-size" in read(
        "docs/RUNBOOK_WINDOWS.md"
    )
    for pair in config.pair_ids():
        stock, ablated = config.pair(pair)
        assert stock.extra_body == ablated.extra_body and stock.quant == ablated.quant
