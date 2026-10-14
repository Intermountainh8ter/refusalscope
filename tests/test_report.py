import copy
import json
import xml.etree.ElementTree as ET

import pytest

from refusalscope.analysis import analyze_run
from refusalscope.mock import PROFILES, MockServer
from refusalscope.report import (
    MOCK_BANNER,
    NOT_RUN,
    README_RESULTS_END,
    README_RESULTS_START,
    README_STATUS_END,
    README_STATUS_START,
    ReportError,
    categories_svg,
    forest_svg,
    interval,
    load_results,
    make_report,
    odds,
    p_text,
    pct,
    pp,
    rate_cell,
    render_markdown,
    results_block,
    table,
    update_readme,
)
from refusalscope.runner import Runner
from tests.helpers import make_study


@pytest.fixture(scope="module")
def mock_doc(provider, behaviours_file, tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    with (
        MockServer(profile=PROFILES["reference"], provider=provider) as a,
        MockServer(profile=PROFILES["degraded"], provider=provider) as b,
    ):
        config = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file)
        Runner(
            config, root / "MOCK-rep", progress=lambda m: None, run_id="MOCK-rep", workers=4
        ).run(["all"], limit=80)
    results, _ = analyze_run(root / "MOCK-rep", None, n_boot=300)
    return results


def as_real(doc):
    """TEST ONLY: relabel a mock results document as if it came from a real run, to exercise the real-results path."""
    real = copy.deepcopy(doc)
    real["mock"], real["label"], real["run_id"] = False, "real-run", "real-run"
    return real


def write(directory, name, doc):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(doc), encoding="utf-8")


# ------------------------------------------------------------------ formatting


def test_formatters():
    assert pct(0.1234) == "12.3%" and pct(None) == "n/a" and pct(1.0, 0) == "100%"
    assert pp(-0.0523) == "-5.2 pp" and pp(None) == "n/a" and pp(0.1) == "+10.0 pp"
    assert interval([-0.1, 0.05]) == "[-10.0, +5.0]" and interval(None) == "n/a"
    assert p_text(None) == "n/a" and p_text(0.00001) == "<0.0001" and p_text(0.0123) == "0.0123"
    assert rate_cell({"n": 10, "rate": 0.5, "lo": 0.2, "hi": 0.8}) == "50.0% [20.0%, 80.0%] (n=10)"
    assert rate_cell(None) == "n/a" and rate_cell({"n": 0, "rate": None}) == "n/a"
    assert odds({"or": 2.5, "lo": 0.7, "hi": 8.5, "haldane": False}) == "2.50 [0.70, 8.50]"
    assert odds({"or": 11.0, "lo": 0.9, "hi": None, "haldane": True}) == "11.00* [0.90, inf]"
    assert odds(None) == "n/a" and odds({"or": None}) == "n/a"
    assert table(["a"], []) == "_No data._\n"
    assert table(["a", "b"], [["1", "2"]]).splitlines()[1] == "|---|---|"


# ------------------------------------------------------------------ refusing to render


def test_refuses_when_there_are_no_results(tmp_path):
    with pytest.raises(ReportError, match="NOT YET RUN"):
        make_report(tmp_path / "empty", tmp_path / "out")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "results" / "README.md").write_text("notes", encoding="utf-8")
    with pytest.raises(ReportError, match="NOT YET RUN"):
        make_report(tmp_path / "results", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_mock_results_by_default(tmp_path, mock_doc):
    write(tmp_path / "results", "MOCK-rep.json", mock_doc)
    with pytest.raises(ReportError, match="only MOCK result files"):
        make_report(tmp_path / "results", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_files_that_are_not_results(tmp_path):
    write(tmp_path / "results", "bad.json", {"hello": "world"})
    with pytest.raises(ReportError, match="not a refusalscope results file"):
        make_report(tmp_path / "results", tmp_path / "out")
    (tmp_path / "results" / "bad.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(ReportError, match="cannot read"):
        load_results(tmp_path / "results" / "bad.json")


# ------------------------------------------------------------------ MOCK labelling


def test_mock_report_is_stamped_everywhere(tmp_path, mock_doc):
    write(tmp_path / "results", "MOCK-rep.json", mock_doc)
    written = make_report(tmp_path / "results", tmp_path / "out", allow_mock=True)
    assert written and all(p.name.startswith("MOCK") for p in written)
    markdown = next(p for p in written if p.suffix == ".md").read_text(encoding="utf-8")
    assert markdown.startswith("# MOCK - refusalscope report") and markdown.count(MOCK_BANNER) == 2
    for svg in (p for p in written if p.suffix == ".svg"):
        text = svg.read_text(encoding="utf-8")
        assert "MOCK" in text.split("</title>")[0] and "NOT EVIDENCE" in text


def test_every_mock_heading_and_table_title_says_mock(mock_doc):
    markdown = render_markdown(mock_doc)
    headings = [line for line in markdown.splitlines() if line.startswith("##")]
    bold_titles = [
        line for line in markdown.splitlines() if line.startswith("**") and line.endswith("**")
    ]
    assert len(headings) >= 8 and bold_titles
    assert all(line.endswith("(MOCK)") for line in headings), headings
    assert all(line.endswith("(MOCK)**") for line in bold_titles), bold_titles
    assert "(MOCK)" not in render_markdown(as_real(mock_doc))


def test_mock_label_is_added_when_the_results_file_lacks_it(tmp_path, mock_doc):
    doc = copy.deepcopy(mock_doc)
    doc["label"] = "unprefixed"
    write(tmp_path / "results", "unprefixed.json", doc)
    written = make_report(tmp_path / "results", tmp_path / "out", allow_mock=True)
    assert all(p.name.startswith("MOCK-unprefixed") for p in written)


def test_mock_numbers_never_reach_the_readme(tmp_path, mock_doc):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{README_STATUS_START}{NOT_RUN}{README_STATUS_END}\n{README_RESULTS_START}\nnone\n{README_RESULTS_END}\n",
        encoding="utf-8",
    )
    write(tmp_path / "results", "MOCK-rep.json", mock_doc)
    with pytest.raises(ReportError, match="refusing to write MOCK numbers"):
        make_report(tmp_path / "results", tmp_path / "out", allow_mock=True, readme=readme)
    assert NOT_RUN in readme.read_text(encoding="utf-8")


def test_real_results_are_rendered_without_mock_marks(tmp_path, mock_doc):
    write(tmp_path / "results", "real-run.json", as_real(mock_doc))
    written = make_report(tmp_path / "results", tmp_path / "out")
    assert all("MOCK" not in p.name for p in written)
    markdown = next(p for p in written if p.suffix == ".md").read_text(encoding="utf-8")
    assert "MOCK" not in markdown and markdown.startswith("# refusalscope report")


def test_mixed_directory_renders_only_real_results_by_default(tmp_path, mock_doc):
    write(tmp_path / "results", "real-run.json", as_real(mock_doc))
    write(tmp_path / "results", "MOCK-rep.json", mock_doc)
    names = {p.name for p in make_report(tmp_path / "results", tmp_path / "out")}
    assert names and all(not n.startswith("MOCK") for n in names)
    both = {p.name for p in make_report(tmp_path / "results", tmp_path / "out2", allow_mock=True)}
    assert any(n.startswith("MOCK") for n in both) and any(not n.startswith("MOCK") for n in both)


# ------------------------------------------------------------------ content


def test_markdown_has_every_section_and_the_numbers_from_the_results(mock_doc):
    markdown = render_markdown(mock_doc)
    for heading in ("## Ablation check", "## Pre-registered hypotheses", "## Suite A: tool calling",
                    "## Suite B: indirect prompt injection", "## Suite C", "## Breakdowns", "## Cost and latency"):  # fmt: skip
        assert heading in markdown, heading
    h1 = mock_doc["pairs"]["p"]["tests"]["H1"]
    assert pct(h1["stock_rate"]) in markdown and pp(h1["rd"]) in markdown
    assert "Haldane" in markdown and "THREATS_TO_VALIDITY" in markdown
    assert "not testable" in markdown  # H3 with one pair


def test_excluded_pairs_are_listed_separately(mock_doc):
    doc = copy.deepcopy(mock_doc)
    doc["pairs"]["p"]["in_family"] = False
    doc["pairs"]["p"]["status"] = "ablation-unverified"
    markdown = render_markdown(doc)
    assert "Descriptive only" in markdown
    assert "ablation-unverified" in markdown


def test_forest_plot_is_valid_svg_with_one_row_per_test(mock_doc):
    svg = forest_svg(mock_doc)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert svg.count("<circle") == len(
        [t for t in ("H1", "H2", "H4") if t in mock_doc["pairs"]["p"]["tests"]]
    )
    assert "supported" in svg or "inconclusive" in svg


def test_forest_plot_marks_excluded_pairs(mock_doc):
    doc = copy.deepcopy(mock_doc)
    doc["pairs"]["p"]["in_family"] = False
    assert "(excluded)" in forest_svg(doc)


def test_category_chart_is_valid_svg(mock_doc):
    svg = categories_svg(mock_doc, "p")
    ET.fromstring(svg)
    assert svg.count("<rect") > 30 and "simple_string" in svg
    empty = copy.deepcopy(mock_doc)
    empty["pairs"]["p"]["breakdowns"]["toolcall_by_category"] = {}
    assert categories_svg(empty, "p") is None


def test_figure_text_is_xml_escaped(mock_doc):
    doc = copy.deepcopy(mock_doc)
    doc["pairs"] = {"a<&>b": doc["pairs"].pop("p")}
    ET.fromstring(forest_svg(doc))
    ET.fromstring(categories_svg(doc, "a<&>b"))


# ------------------------------------------------------------------ README block


README = f"""# x

{README_STATUS_START}{NOT_RUN}{README_STATUS_END}

{README_RESULTS_START}
| nothing yet |
{README_RESULTS_END}

tail
"""


def test_readme_update_replaces_only_the_marked_blocks(mock_doc):
    updated = update_readme(README, as_real(mock_doc))
    assert NOT_RUN not in updated and "**Results: RUN**" in updated
    assert "nothing yet" not in updated and "results/real-run.json" in updated
    assert updated.startswith("# x") and updated.rstrip().endswith("tail")
    assert updated.count(README_RESULTS_START) == 1 and updated.count(README_STATUS_END) == 1


def test_readme_without_markers_is_an_error(mock_doc):
    with pytest.raises(ReportError, match="missing the marker"):
        update_readme("# no markers", as_real(mock_doc))


def test_results_block_has_a_row_per_pair(mock_doc):
    block = results_block(as_real(mock_doc))
    assert "| p |" in block and "H1 tool-call accuracy" in block and "->" in block


def test_make_report_updates_the_readme_for_real_results(tmp_path, mock_doc):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    write(tmp_path / "results", "real-run.json", as_real(mock_doc))
    written = make_report(tmp_path / "results", tmp_path / "out", readme=readme)
    assert readme in written and "Results: RUN" in readme.read_text(encoding="utf-8")
