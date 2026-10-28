"""Repository-level promises: honesty rules, docs that match the code, no leftovers."""

import json
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

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


def test_readme_results_block_matches_results_dir_state():
    """Flip in both directions. Before any real run lands, the README must
    say NOT YET RUN and must not display percentages. After results land, the
    README must stop saying NOT YET RUN and must show percentages."""
    text = read("README.md")
    results_dir = ROOT / "results"
    real = [p for p in results_dir.iterdir() if p.suffix == ".json" and not p.name.startswith("MOCK")]
    pct_in_readme = bool(re.search(r"\d+\.\d+%", text))
    assert "<!-- RESULTS:START -->" in text and "<!-- RESULTS:END -->" in text, "block markers must stay"
    if real:
        assert "**Results: NOT YET RUN.**" not in text, (
            "results/ has real files; README must stop saying NOT YET RUN")
        assert pct_in_readme, "results/ has real files; README should show percentages"
    else:
        assert "<!-- STATUS:START -->**Results: NOT YET RUN.**<!-- STATUS:END -->" in text
        block = text.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
        assert block.count("not run") >= 4 and not pct_in_readme


def test_no_mock_results_are_committed():
    """MOCK output is for plumbing tests only and must never land in git. Real
    result files (non-MOCK) are allowed once a real run has produced them."""
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


LAUNCH_FILES = ("LINKEDIN.md", "X_THREAD.md", "DISCORD_SHORT.md", "FAQ.md")
REPO_URL = "https://github.com/Intermountainh8ter/refusalscope"


def launch_text(name):
    return (ROOT / "docs" / "launch" / name).read_text(encoding="utf-8")


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


def real_result_docs():
    """Parsed ``results/*.json`` files, MOCK output excluded (it is never committed)."""
    paths = sorted(p for p in (ROOT / "results").glob("*.json") if not p.name.startswith("MOCK"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def derivable_figures():
    """Everything a launch file may quote, recomputed from ``results/*.json``.

    Differences are recomputed from the stored stock and ablated rates rather than read from the
    stored ``rd`` so a hand-edited or stale field cannot make a wrong figure pass. Only the
    figures the launch copy is allowed to cite are collected: the manipulation-check drop of every
    result file, the H1 and H2 deltas, the pre-registered margin and gate, and first-token KL.
    """
    signed, rates, kl = set(), set(), set()

    def add_delta(stock, ablated):
        delta = round((ablated - stock) * 100, 1)
        signed.add(delta)

    docs = real_result_docs()
    assert docs, "results/ holds no real result files"
    for doc in docs:
        signed.add(round(doc["hypotheses"]["sesoi"] * 100, 1))
        for pair in doc["pairs"].values():
            check = pair["manipulation_check"]
            signed.add(round(check["threshold"] * 100, 1))
            add_delta(check["stock"]["rate"], check["ablated"]["rate"])
            rates.update(round(check[arm]["rate"] * 100, 1) for arm in ("stock", "ablated"))
            for name in ("H1", "H2"):
                test = pair.get("tests", {}).get(name)
                if test and "stock_rate" in test:
                    add_delta(test["stock_rate"], test["ablated_rate"])
                    rates.update(round(test[f"{arm}_rate"] * 100, 1) for arm in ("stock", "ablated"))
        for pair in doc["pairs"].values():
            probe = pair.get("kl") or {}
            if probe.get("measured"):
                kl.update({round(probe["mean_kl"], 3), round(probe["mean_kl"], 2)})
    return {"signed": signed, "absolute": {abs(v) for v in signed}, "rates": rates, "kl": kl}


PP_CHAIN = re.compile(
    r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?(?:\s*(?:,|to|and|or|-)\s*[+\-−]?\d+(?:\.\d+)?)*)"
    r"\s*(?:pp\b|percentage[ -]points?\b)"
)
NUMBER = re.compile(r"([+\-−]?)(\d+(?:\.\d+)?)")


def quoted_pp_figures(text):
    """Every number written as 'NN pp' or 'NN.N percentage points', including chains such as
    '94, 79 and 70 pp' or '56 to 94 pp'. Returns (sign, magnitude) pairs."""
    out = []
    for chain in PP_CHAIN.findall(flat(text)):
        for sign, magnitude in NUMBER.findall(chain):
            out.append((sign.replace("−", "-"), float(magnitude)))
    return out


def test_quoted_pp_figures_parser_reads_lists_ranges_and_signs():
    assert quoted_pp_figures("fell 94 pp (4B), 79 pp (8B)") == [("", 94.0), ("", 79.0)]
    assert quoted_pp_figures("between -2.9 pp and +5.3 pp") == [("-", 2.9), ("+", 5.3)]
    assert quoted_pp_figures("down 56 to 94 pp") == [("", 56.0), ("", 94.0)]
    assert quoted_pp_figures("94, 79 and 70 pp") == [("", 94.0), ("", 79.0), ("", 70.0)]
    assert quoted_pp_figures("a 5 percentage-point margin") == [("", 5.0)]
    assert quoted_pp_figures("Q8_0 on 14B, 374 cases, no figure here") == []


def test_derivable_figures_recomputes_the_known_deltas():
    """Guards the helper itself: the README's published deltas must come out of it."""
    figures = derivable_figures()
    for expected in (-2.9, 0.0, 0.5, 5.3, -6.3, 0.3, -1.3, 3.7, -94.0, -79.0, -56.0, -70.0, -15.0):
        assert expected in figures["signed"], expected
    assert 5.0 in figures["absolute"] and 30.0 in figures["absolute"]
    assert 0.009 in figures["kl"] and 0.235 in figures["kl"] and 1.03 in figures["kl"]
    assert -99.0 in figures["signed"] and 74.5 in figures["absolute"]  # corpus replications
    assert 77.0 not in figures["absolute"] and 95.5 not in figures["signed"]  # nothing invented


def test_launch_copy_reports_the_real_run_honestly():
    files = sorted((ROOT / "docs" / "launch").glob("*.md"))
    assert {f.name for f in files} == set(LAUNCH_FILES)
    stale = re.compile(
        r"NOT YET RUN|no results? yet|there are no results|no real model|has not been run|"
        r"nothing has been run|harness and pre-registration only",
        re.IGNORECASE,
    )
    overclaims = re.compile(
        r"outperform|state[- ]of[- ]the[- ]art|\bprov(?:es|en)\b|safe to use|no effect on|"
        r"significantly (?:lower|higher|worse|better)|breakthrough|game[- ]chang",
        re.IGNORECASE,
    )
    h3_not_testable = re.compile(
        r"\bH3\b[^.]{0,160}\bnot testable\b|\bnot testable\b[^.]{0,160}\bH3\b", re.IGNORECASE
    )
    not_safe = re.compile(r"\bnot\b[^.]{0,80}\bsafe\b", re.IGNORECASE)
    for name in LAUNCH_FILES:
        text = flat(launch_text(name))
        assert not stale.search(text), f"{name}: still describes the pre-results world"
        assert not overclaims.search(text), f"{name}: overclaims"
        assert "synthetic" in text.lower(), f"{name}: must say the suites are synthetic"
        assert h3_not_testable.search(text), f"{name}: must say H3 was not testable"
        assert not_safe.search(text), f"{name}: must say the result does not show ablation is safe"
        assert re.search(r"\b(?:5 pp|5 percentage[- ]points?)\b", text), (
            f"{name}: must state the 5 pp resolution limit"
        )
        assert re.search(r"opposite|contrary|went up", text, re.IGNORECASE), (
            f"{name}: must mention that some results went against the prediction"
        )


def test_launch_posts_carry_the_exact_repository_url():
    for name in ("LINKEDIN.md", "X_THREAD.md", "DISCORD_SHORT.md"):
        assert REPO_URL in launch_text(name), name
    assert "first comment" not in launch_text("LINKEDIN.md").lower()
    linkedin = launch_text("LINKEDIN.md")
    assert linkedin.index(REPO_URL) > linkedin.index("What this does not show")


def x_posts():
    body = launch_text("X_THREAD.md")
    parts = re.split(r"(?m)^(?=\d+/ )", body)
    return [flat(part) for part in parts if re.match(r"\d+/ ", part)]


def test_x_thread_posts_are_numbered_and_fit_280_characters():
    posts = x_posts()
    assert len(posts) >= 8
    for expected, post in enumerate(posts, start=1):
        assert post.startswith(f"{expected}/ "), f"post {expected} is out of order"
        assert len(post) <= 280, f"post {expected} is {len(post)} characters"
    assert REPO_URL in posts[-1], "the repository link belongs in the last post"
    assert sum(REPO_URL in post for post in posts) == 1


def test_launch_figures_are_derivable_from_the_result_files():
    """Every 'NN pp', rate and KL figure in the launch copy must be recomputable from results/."""
    figures = derivable_figures()
    for name in LAUNCH_FILES:
        text = flat(launch_text(name))
        pp = quoted_pp_figures(text)
        assert pp, f"{name}: quotes no pp figure at all"
        for sign, magnitude in pp:
            if sign in ("+", "-"):
                value = -magnitude if sign == "-" else magnitude
                assert value in figures["signed"], f"{name}: {sign}{magnitude} pp is not in results/"
            else:
                assert magnitude in figures["absolute"], f"{name}: {magnitude} pp is not in results/"
        for percent in re.findall(r"(\d+(?:\.\d+)?)\s?%", text):
            assert float(percent) in figures["rates"], f"{name}: {percent}% is not in results/"
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
            if re.search(r"\bKL\b", sentence):
                for number in re.findall(r"\b\d\.\d{2,3}\b", sentence):
                    assert float(number) in figures["kl"], f"{name}: KL {number} is not in results/"


def test_launch_case_counts_match_the_result_files():
    counts = {"tool-calling": set(), "injection": set()}
    for doc in real_result_docs():
        for pair in doc["pairs"].values():
            tests = pair.get("tests", {})
            if "stock_rate" in tests.get("H1", {}):
                counts["tool-calling"].add(tests["H1"]["n"])
            if "stock_rate" in tests.get("H2", {}):
                counts["injection"].add(tests["H2"]["n"])
    assert counts["tool-calling"] == {374} and counts["injection"] == {378}
    for name in LAUNCH_FILES:
        for number, kind in re.findall(
            r"\b(\d{3})\s+(?:synthetic\s+)?(tool-calling|injection)\b", flat(launch_text(name))
        ):
            assert {int(number)} == counts[kind], f"{name}: {number} {kind} cases"


def test_faq_ends_with_the_ai_disclosure_verbatim():
    entry = (
        "**Was AI used to build this?** Yes. An AI coding assistant wrote most of the code, tests "
        "and prose in this repository (the commits are co-authored in the git history). I directed "
        "the work and ran the real-model benchmarks on my own hardware. Every number in the README "
        "is generated by a script from committed result files, and the tests fail if they drift. "
        "Read the code and the pre-registration before trusting any conclusion."
    )
    text = flat(launch_text("FAQ.md"))
    assert text.endswith(entry), "the AI-use entry must be the last FAQ entry, verbatim"
    assert "**Is there a result?** Yes" in text


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
    pyproject = read("pyproject.toml")  # parsed with regexes so the test also runs on Python 3.10
    version = re.search(r'^version = "([^"]+)"', pyproject, flags=re.MULTILINE).group(1)
    assert version == refusalscope.__version__
    assert f"## {refusalscope.__version__} - " in read("CHANGELOG.md")
    assert 'refusalscope = "refusalscope.cli:main"' in pyproject
    assert 'dependencies = ["numpy>=1.24"]' in pyproject
    assert 'name = "refusalscope"' in pyproject and 'requires-python = ">=3.10"' in pyproject


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
