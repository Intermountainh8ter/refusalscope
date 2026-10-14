"""The pre-registration is part of the product; keep its required structure from rotting."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PREREG = (DOCS / "PREREGISTRATION.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("label", ["H1", "H1b", "H2", "H3", "H4"])
def test_each_hypothesis_has_a_section(label):
    assert re.search(rf"^### {label}[ .]", PREREG, flags=re.MULTILINE), label


@pytest.mark.parametrize("label", ["H1", "H2", "H3", "H4"])
def test_each_confirmatory_hypothesis_states_how_it_is_falsified(label):
    start = re.search(rf"^### {label}[ .]", PREREG, flags=re.MULTILINE).start()
    nxt = re.search(r"^### ", PREREG[start + 5 :], flags=re.MULTILINE)
    section = PREREG[start : start + 5 + nxt.start()] if nxt else PREREG[start:]
    assert "Falsified if" in section


def test_required_top_level_sections_exist():
    for heading in (
        "## 4. Sample sizes",
        "## 6. Smallest effect size of interest",
        "## 8. Manipulation check and exclusion rules",
        "## 9. Statistical analysis plan",
        "## 11. Explicit null and negative outcomes",
        "## 13. Deviations and amendments",
    ):
        assert heading in PREREG, heading


def test_prereg_commits_to_tests_and_corrections():
    for needle in ("McNemar", "Holm", "bootstrap", "Clopper-Pearson", "Wilson", "Spearman"):
        assert needle in PREREG, needle


def test_prereg_states_no_results_exist():
    assert "Nothing in this repository reports an empirical result" in PREREG


def test_methodology_documents_failure_taxonomy():
    text = (DOCS / "METHODOLOGY.md").read_text(encoding="utf-8")
    for label in ("text_tool_call", "malformed_arguments", "wrong_value", "spurious_call"):
        assert f"`{label}`" in text
