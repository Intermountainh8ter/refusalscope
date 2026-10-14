"""Tables and figures built only from ``results/*.json``.

The generator refuses to render when no result file exists, and refuses MOCK result files unless
asked, in which case every output name, title and table carries ``MOCK``. It never invents or
smooths a number: each figure and cell is read straight from a results document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from refusalscope.analysis import RESULT_SCHEMA
from refusalscope.stats import SESOI
from refusalscope.suites import toolcall

README_RESULTS_START = "<!-- RESULTS:START -->"
README_RESULTS_END = "<!-- RESULTS:END -->"
README_STATUS_START = "<!-- STATUS:START -->"
README_STATUS_END = "<!-- STATUS:END -->"
NOT_RUN = "**Results: NOT YET RUN.**"

MOCK_BANNER = (
    "> **MOCK - NOT EVIDENCE.** This report was produced against the bundled mock server, which "
    "returns scripted answers with tunable error rates. It only shows that the pipeline works. "
    "It says nothing about any real model."
)
STOCK_COLOR, ABLATED_COLOR, ACCENT = "#3b6ea5", "#d1603d", "#b00020"


class ReportError(RuntimeError):
    """The report cannot be built (no results, only MOCK results, malformed file)."""


# --------------------------------------------------------------------------- loading


def load_results(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"{path}: cannot read results ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != RESULT_SCHEMA or "pairs" not in doc:
        raise ReportError(f"{path}: not a refusalscope results file (schema {RESULT_SCHEMA})")
    return doc


def _file_label(doc: dict[str, Any]) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(doc["label"]))
    return label if not doc.get("mock") or label.startswith("MOCK") else f"MOCK-{label}"


# --------------------------------------------------------------------------- formatting


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f} pp"


def rate_cell(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("rate") is None:
        return "n/a"
    return f"{pct(summary['rate'])} [{pct(summary['lo'])}, {pct(summary['hi'])}] (n={summary['n']})"


def interval(ci: list[float] | None) -> str:
    return "n/a" if not ci else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def odds(ratio: dict[str, Any] | None) -> str:
    if not ratio or ratio.get("or") is None:
        return "n/a"
    hi = "inf" if ratio["hi"] is None else f"{ratio['hi']:.2f}"
    lo = "n/a" if ratio["lo"] is None else f"{ratio['lo']:.2f}"
    mark = "*" if ratio.get("haldane") else ""
    return f"{ratio['or']:.2f}{mark} [{lo}, {hi}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def p_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


# --------------------------------------------------------------------------- markdown


def _title(doc: dict[str, Any], text: str) -> str:
    return f"MOCK - {text}" if doc.get("mock") else text


def _manipulation_table(doc: dict[str, Any]) -> str:
    rows = []
    for pair_id, pair in doc["pairs"].items():
        check = pair["manipulation_check"]
        rows.append(
            [
                pair_id,
                pair["status"],
                rate_cell(check.get("stock")),
                rate_cell(check.get("ablated")),
                pp(-check["refusal_drop"]) if "refusal_drop" in check else "n/a",
                check["reason"],
                "; ".join(pair["flags"]) or "-",
            ]
        )
    return table(
        ["Pair", "Status", "Stock refusal", "Ablated refusal", "Change", "Check", "Flags"], rows
    )


def _test_row(pair_id: str, test_id: str, test: dict[str, Any]) -> list[str]:
    ratio = test.get("odds_ratio_paired")
    p_raw = test.get("p_mcnemar", test.get("p_boot"))
    return [
        pair_id,
        test_id,
        str(test["n"]),
        pct(test.get("stock_rate")) if "stock_rate" in test else pp(test.get("rd_toolcall")),
        pct(test.get("ablated_rate")) if "ablated_rate" in test else pp(test.get("rd_capability")),
        f"{pp(test['rd'])} {interval(test['rd_ci'])}",
        f"{test.get('b', '-')}/{test.get('c', '-')}",
        odds(ratio) if ratio else "-",
        p_text(p_raw),
        p_text(test.get("p_holm")),
        test.get("label", "-"),
    ]


def _confirmatory(doc: dict[str, Any]) -> str:
    rows, descriptive = [], []
    for pair_id, pair in doc["pairs"].items():
        for test_id in ("H1", "H2", "H4"):
            test = pair["tests"].get(test_id)
            if not test:
                continue
            (rows if pair["in_family"] else descriptive).append(_test_row(pair_id, test_id, test))
    headers = ["Pair", "Test", "n", "Stock", "Ablated", "Risk difference [95% CI]", "b/c", "Paired OR [CI]",
               "p (raw)", "p (Holm)", "Decision"]  # fmt: skip
    out = "### Confirmatory family\n\n" + table(headers, rows)
    out += (
        "\nFor H4 the Stock and Ablated columns show the risk difference of the tool-call suite and of "
        "the capability control. `b/c` = cases where only the stock / only the ablated arm succeeded. "
        "`*` = Haldane-Anscombe corrected zero cell.\n"
    )
    if descriptive:
        out += "\n### Descriptive only (excluded from the family)\n\n" + table(headers, descriptive)
    return out


def _hypothesis_summary(doc: dict[str, Any]) -> str:
    h = doc["hypotheses"]
    lines = [
        f"Family size m = {h['m']} (Holm-Bonferroni, alpha = {h['alpha']}, SESOI = {100 * h['sesoi']:.0f} pp).\n"
    ]
    for hid, counts in sorted(h["summary"].items()):
        lines.append(
            f"- {hid}: " + ", ".join(f"{label} x{n}" for label, n in sorted(counts.items()))
        )
    h3 = h["H3"]
    if h3["testable"]:
        lines.extend(
            f"- {key}: rho = {h3[key]['rho']:.3f}, exact p = {p_text(h3[key]['p_value'])}"
            for key in ("H3a", "H3b")
            if key in h3
        )
    else:
        lines.append(f"- H3: not testable ({h3.get('reason', 'too few pairs')})")
    return "\n".join(lines) + "\n"


def _secondary(doc: dict[str, Any]) -> str:
    """Secondary tests, the S8 sensitivity analysis and paired latency (all exploratory)."""
    rows, latency = [], []
    for pair_id, pair in doc["pairs"].items():
        entries = {k: v for k, v in pair["tests"].items() if k not in ("H1", "H2", "H4")}
        entries.update(pair.get("sensitivity", {}))
        for test_id, test in entries.items():
            if "stock_rate" not in test:
                continue
            row = _test_row(pair_id, test_id, test)
            rows.append(
                [
                    *row[:8],
                    row[8],
                    p_text(test.get("p_holm_if_added")) if "p_holm_if_added" in test else "-",
                ]
            )
        for name, item in pair.get("latency", {}).items():
            latency.append([pair_id, name, str(item["n"]), f"{item['stock_mean']:.2f}", f"{item['ablated_mean']:.2f}",
                            f"{item['mean_diff']:+.2f} [{item['ci'][0]:+.2f}, {item['ci'][1]:+.2f}]",
                            "n/a" if item["cohens_dz"] is None else f"{item['cohens_dz']:.2f}"])  # fmt: skip
    headers = ["Pair", "Test", "n", "Stock", "Ablated", "Risk difference [95% CI]", "b/c", "Paired OR [CI]",
               "p (raw, unadjusted)", "p if added to the family"]  # fmt: skip
    out = table(headers, rows)
    out += "\nExploratory: no Holm correction is applied to these rows. S8 repeats H1 and H2 without cases that had request errors.\n"
    out += "\n**Paired request time per case (seconds)**\n\n"
    return out + table(
        ["Pair", "Suite", "n", "Stock mean", "Ablated mean", "Difference [95% CI]", "Cohen's d_z"],
        latency,
    )


def _suite_a(doc: dict[str, Any]) -> str:
    rows, tax_rows = [], []
    for name, model in doc["models"].items():
        s = model["summary"].get("toolcall")
        if not s:
            continue
        rows.append([name, model["role"], rate_cell(s["valid"]), rate_cell(s["schema_valid"]), rate_cell(s["correct"]),
                     rate_cell(s["lenient_correct"]), pct(s["text_tool_call_rate"])])  # fmt: skip
        counts = {k: v for k, v in s["taxonomy"].items() if v}
        tax_rows.append([name, ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"])
    out = table(
        [
            "Model",
            "Role",
            "Valid",
            "Schema-valid",
            "Correct (strict)",
            "Correct (lenient)",
            "Text tool calls",
        ],
        rows,
    )
    return out + "\n**First-failure taxonomy**\n\n" + table(["Model", "Counts"], tax_rows)


def _suite_b(doc: dict[str, Any]) -> str:
    rows = []
    for name, model in doc["models"].items():
        s = model["summary"].get("injection")
        if not s:
            continue
        rows.append(
            [
                name, model["role"],
                rate_cell(s["control"]["tool_use"]), rate_cell(s["control"]["utility"]),
                rate_cell(s["attack"]["utility"]), rate_cell(s["attack"]["asr"]),
                rate_cell(s["attack"]["asr_incl_text"]), rate_cell(s["asr_given_task_completed"]),
                str(s["asr_control_false_positives"]),
            ]
        )  # fmt: skip
    return table(
        ["Model", "Role", "Tool use (control)", "Utility (control)", "Utility (attack)", "ASR", "ASR incl. text",
         "ASR given task done", "ASR in control (must be 0)"],
        rows,
    )  # fmt: skip


def _suite_c(doc: dict[str, Any]) -> str:
    rows = []
    for name, model in doc["models"].items():
        s = model["summary"]
        harmful, benign = s.get("refusal_harmful"), s.get("refusal_benign")
        rows.append(
            [
                name, model["role"],
                rate_cell(harmful) if harmful else "not run",
                pct(harmful["ambiguous_share"]) if harmful else "n/a",
                rate_cell(benign) if benign else "not run",
                rate_cell(s.get("capability")) if s.get("capability") else "not run",
            ]
        )  # fmt: skip
    out = table(
        [
            "Model",
            "Role",
            "Refusal on behaviours file",
            "Ambiguous share",
            "Refusal on benign set",
            "Capability control",
        ],
        rows,
    )
    kl_rows = []
    for pair_id, pair in doc["pairs"].items():
        k = pair["kl"]
        if k.get("measured"):
            kl_rows.append([pair_id, f"{k['mean_kl']:.4f} [{k['ci'][0]:.4f}, {k['ci'][1]:.4f}]", str(k["n"]),
                            pct(k["top1_agreement"]), "; ".join(k["warnings"]) or "-"])  # fmt: skip
        else:
            kl_rows.append([pair_id, "not measured", "0", "n/a", k.get("reason", "")])
    return (
        out
        + "\n**First-token KL(stock || ablated), harmless prompts (top-k approximation)**\n\n"
        + table(["Pair", "Mean KL [95% CI]", "Prompts", "Top-1 agreement", "Warnings"], kl_rows)
    )


def _breakdowns(doc: dict[str, Any]) -> str:
    out = ""
    for pair_id, pair in doc["pairs"].items():
        for key, title in (("toolcall_by_category", "Tool-call accuracy by category"),
                           ("injection_asr_by_vector", "ASR by injection vector"),
                           ("injection_asr_by_environment", "ASR by environment")):  # fmt: skip
            data = pair["breakdowns"].get(key) or {}
            if not data:
                continue
            rows = [
                [g, rate_cell(c["stock"]), rate_cell(c["ablated"]), pp(c.get("rd"))]
                for g, c in data.items()
            ]
            out += (
                f"**{title} - {pair_id}**\n\n"
                + table(["Group", "Stock", "Ablated", "Difference"], rows)
                + "\n"
            )
    return out


def _usage(doc: dict[str, Any]) -> str:
    rows = []
    for name, model in doc["models"].items():
        u = model["summary"]["usage"]
        median = "n/a" if u["latency_median_s"] is None else f"{u['latency_median_s']:.2f}"
        p95 = "n/a" if u["latency_p95_s"] is None else f"{u['latency_p95_s']:.2f}"
        rows.append([name, str(u["trajectories"]), str(u["requests"]), str(u["trajectories_with_errors"]),
                     f"{u['prompt_tokens']:,}", f"{u['completion_tokens']:,}", median, p95,
                     f"{u['latency_total_s']:.0f}", f"{u['cost']:.4f}"])  # fmt: skip
    return table(["Model", "Trajectories", "Requests", "With errors", "Prompt tokens", "Completion tokens",
                  "Median latency (s)", "p95 latency (s)", "Total request time (s)", "Cost"], rows)  # fmt: skip


def render_markdown(doc: dict[str, Any]) -> str:
    """Full Markdown report for one results document."""
    mock = bool(doc.get("mock"))
    parts = [f"# {_title(doc, 'refusalscope report')}: {doc['label']}\n"]
    if mock:
        parts.append(MOCK_BANNER + "\n")
    meta_rows = [
        ["Study", doc["study"]], ["Run id", doc["run_id"]], ["Generated", doc["generated_at"]],
        ["Harness version", doc["tool"]["version"]], ["Primary temperature", str(doc["primary_temperature"])],
        ["Trajectories", str(doc["n_trajectories"])], ["Config sha256", (doc.get("config_sha256") or "n/a")[:16]],
        ["Python", doc["environment"]["python"]],
    ]  # fmt: skip
    parts.append(table(["Field", "Value"], meta_rows))
    parts.append("## Ablation check\n\n" + _manipulation_table(doc))
    parts.append(
        "## Pre-registered hypotheses\n\n" + _confirmatory(doc) + "\n" + _hypothesis_summary(doc)
    )
    parts.append("## Secondary and sensitivity analyses\n\n" + _secondary(doc))
    parts.append("## Suite A: tool calling\n\n" + _suite_a(doc))
    parts.append("## Suite B: indirect prompt injection\n\n" + _suite_b(doc))
    parts.append("## Suite C: refusal, divergence and capability control\n\n" + _suite_c(doc))
    parts.append("## Breakdowns\n\n" + _breakdowns(doc))
    parts.append("## Cost and latency\n\n" + _usage(doc))
    if doc.get("notes"):
        parts.append("## Notes\n\n" + "\n".join(f"- {n}" for n in doc["notes"]) + "\n")
    parts.append(
        "## How to read this\n\nThe suites are synthetic and the comparison is within pair; absolute levels "
        "do not transfer to other benchmarks. Read `docs/THREATS_TO_VALIDITY.md` before drawing conclusions. "
        "A decision of `equivalent` means no change larger than the 5 pp margin was detected, not that "
        "ablation is safe.\n"
    )
    text = "\n".join(parts + ([MOCK_BANNER + "\n"] if mock else []))
    if mock:  # stamp every section and table title, not just the document title
        text = re.sub(r"^(#{2,3} .+)$", r"\1 (MOCK)", text, flags=re.MULTILINE)
        text = re.sub(r"^\*\*(.+)\*\*$", r"**\1 (MOCK)**", text, flags=re.MULTILINE)
    return text


# --------------------------------------------------------------------------- SVG figures


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )
    return (
        "\n".join(
            [
                head,
                f"<title>{escape(title)}</title>",
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                *body,
                "</svg>",
            ]
        )
        + "\n"
    )


def _text(
    x: float,
    y: float,
    text: str,
    anchor: str = "start",
    size: int = 12,
    color: str = "#222",
    weight: str = "normal",
) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}">{escape(text)}</text>')  # fmt: skip


def forest_svg(doc: dict[str, Any]) -> str:
    """Risk differences with 95% intervals for the pre-registered tests, one row per pair and test."""
    rows = [
        (f"{pair_id} {test_id}" + ("" if pair["in_family"] else " (excluded)"), test)
        for pair_id, pair in doc["pairs"].items()
        for test_id in ("H1", "H2", "H4")
        if (test := pair["tests"].get(test_id))
    ]
    title = _title(doc, "Risk difference (ablated minus stock), 95% bootstrap interval")
    width, left, right, top, row_h = 820, 230, 120, 70, 34
    height = top + max(len(rows), 1) * row_h + 60
    values = [abs(v) for _, t in rows for v in (t["rd"], *t["rd_ci"])] or [SESOI]
    span = max(0.1, round(max(values) * 1.15, 2))

    def sx(v: float) -> float:
        return left + (v + span) / (2 * span) * (width - left - right)

    body = [
        _text(width / 2, 24, title, "middle", 14, ACCENT if doc.get("mock") else "#222", "bold")
    ]
    if doc.get("mock"):
        body.append(
            _text(
                width / 2, 44, "MOCK - NOT EVIDENCE ABOUT ANY MODEL", "middle", 12, ACCENT, "bold"
            )
        )
    band_x0, band_x1 = sx(-SESOI), sx(SESOI)
    body.append(
        f'<rect x="{band_x0:.1f}" y="{top - 10}" width="{band_x1 - band_x0:.1f}" height="{len(rows) * row_h + 10}" fill="#eeeeee"/>'
    )
    body.append(
        f'<line x1="{sx(0):.1f}" y1="{top - 10}" x2="{sx(0):.1f}" y2="{top + len(rows) * row_h}" stroke="#555" stroke-dasharray="4 3"/>'
    )
    for i, (label, test) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        color = {"supported": ACCENT, "contrary": "#6a3d9a", "equivalent": "#2c7a4b"}.get(
            test.get("label", ""), "#555"
        )
        lo, hi = test["rd_ci"]
        body.append(_text(left - 12, y + 4, label, "end"))
        body.append(
            f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(f'<circle cx="{sx(test["rd"]):.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        body.append(
            _text(width - right + 4, y + 4, test.get("label", "descriptive"), "start", 11, color)
        )
    axis_y = top + len(rows) * row_h + 14
    for tick in (-span, -SESOI, 0.0, SESOI, span):
        body.append(
            f'<line x1="{sx(tick):.1f}" y1="{axis_y - 6}" x2="{sx(tick):.1f}" y2="{axis_y}" stroke="#555"/>'
        )
        body.append(_text(sx(tick), axis_y + 14, f"{100 * tick:+.0f} pp", "middle", 11))
    body.append(
        _text(
            (left + width - right) / 2,
            axis_y + 38,
            "grey band = pre-registered margin (+/- 5 pp)",
            "middle",
            11,
            "#555",
        )
    )
    return _svg(width, height, title, body)


def categories_svg(doc: dict[str, Any], pair_id: str) -> str | None:
    """Grouped bars of strict tool-call accuracy by category for one pair."""
    data = doc["pairs"][pair_id]["breakdowns"].get("toolcall_by_category") or {}
    rows = [(c, data[c]) for c in toolcall.CATEGORIES if c in data]
    if not rows:
        return None
    title = _title(doc, f"Tool-call accuracy by category - {pair_id}")
    width, left, top, bar_h, gap = 720, 190, 70, 11, 10
    height = top + len(rows) * (2 * bar_h + gap) + 50
    scale = (width - left - 60) / 1.0
    body = [
        _text(width / 2, 24, title, "middle", 14, ACCENT if doc.get("mock") else "#222", "bold")
    ]
    if doc.get("mock"):
        body.append(
            _text(
                width / 2, 44, "MOCK - NOT EVIDENCE ABOUT ANY MODEL", "middle", 12, ACCENT, "bold"
            )
        )
    for i, (category, cell) in enumerate(rows):
        y = top + i * (2 * bar_h + gap)
        body.append(_text(left - 10, y + bar_h + 4, category, "end", 11))
        for j, (arm, color) in enumerate((("stock", STOCK_COLOR), ("ablated", ABLATED_COLOR))):
            rate = cell[arm]["rate"] or 0.0
            body.append(
                f'<rect x="{left}" y="{y + j * bar_h}" width="{rate * scale:.1f}" height="{bar_h - 1}" fill="{color}"/>'
            )
            body.append(
                _text(
                    left + rate * scale + 4,
                    y + j * bar_h + bar_h - 2,
                    pct(cell[arm]["rate"], 0),
                    "start",
                    10,
                )
            )
    legend_y = top + len(rows) * (2 * bar_h + gap) + 14
    body.append(
        f'<rect x="{left}" y="{legend_y - 9}" width="10" height="10" fill="{STOCK_COLOR}"/>'
        + _text(left + 16, legend_y, "stock", "start", 11)
    )
    body.append(
        f'<rect x="{left + 80}" y="{legend_y - 9}" width="10" height="10" fill="{ABLATED_COLOR}"/>'
        + _text(left + 96, legend_y, "ablated", "start", 11)
    )
    return _svg(width, height, title, body)


# --------------------------------------------------------------------------- README block


def _result_cell(tests: dict[str, Any], test_id: str) -> tuple[str, str]:
    t = tests.get(test_id)
    if not t:
        return "not run", "-"
    label = t.get("label", "descriptive")
    if "stock_rate" in t:
        return f"{pct(t['stock_rate'])} -> {pct(t['ablated_rate'])} ({pp(t['rd'])})", label
    return pp(t["rd"]), label


def results_block(doc: dict[str, Any]) -> str:
    """Compact results table for the README (real results only)."""
    rows = []
    for pair_id, pair in doc["pairs"].items():
        tests = pair["tests"]

        h1, h2, h4 = (_result_cell(tests, t) for t in ("H1", "H2", "H4"))
        kl = pair["kl"]
        rows.append([pair_id, pair["status"], h1[0], h1[1], h2[0], h2[1], h4[1],
                     f"{kl['mean_kl']:.3f}" if kl.get("measured") else "n/a"])  # fmt: skip
    head = f"Source: `results/{doc['label']}.json` (generated {doc['generated_at']}, harness {doc['tool']['version']}).\n\n"
    return head + table(
        [
            "Pair",
            "Status",
            "H1 tool-call accuracy (stock -> ablated)",
            "H1",
            "H2 attack success (stock -> ablated)",
            "H2",
            "H4",
            "KL",
        ],
        rows,
    )


def update_readme(text: str, doc: dict[str, Any]) -> str:
    """Fill the README markers with a real-results block and flip the status line."""
    for marker in (
        README_RESULTS_START,
        README_RESULTS_END,
        README_STATUS_START,
        README_STATUS_END,
    ):
        if marker not in text:
            raise ReportError(f"README is missing the marker {marker}")
    status = f"**Results: RUN** - see the table below and `results/{doc['label']}.json`."
    text = re.sub(
        re.escape(README_STATUS_START) + r".*?" + re.escape(README_STATUS_END),
        lambda _: f"{README_STATUS_START}{status}{README_STATUS_END}", text, flags=re.DOTALL,
    )  # fmt: skip
    return re.sub(
        re.escape(README_RESULTS_START) + r".*?" + re.escape(README_RESULTS_END),
        lambda _: f"{README_RESULTS_START}\n{results_block(doc)}{README_RESULTS_END}", text, flags=re.DOTALL,
    )  # fmt: skip


# --------------------------------------------------------------------------- entry point


def make_report(
    results_dir: str | Path,
    out_dir: str | Path,
    allow_mock: bool = False,
    readme: str | Path | None = None,
) -> list[Path]:
    """Render every results document found in ``results_dir``. Returns the files written."""
    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        raise ReportError(
            f"no result files found in {results_dir}/*.json. Results: NOT YET RUN. Produce them with "
            "`refusalscope run` and `refusalscope analyze` against real models; the mock server only tests the harness."
        )
    docs = [(path, load_results(path)) for path in files]
    real = [d for _, d in docs if not d.get("mock")]
    if not real and not allow_mock:
        raise ReportError(
            f"only MOCK result files found in {results_dir}. They are not evidence about any model; pass "
            "--allow-mock to render them stamped MOCK."
        )
    chosen = [d for _, d in docs if not d.get("mock") or allow_mock]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in chosen:
        label = _file_label(doc)
        md = out / f"{label}-report.md"
        md.write_text(render_markdown(doc), encoding="utf-8")
        written.append(md)
        forest = out / f"{label}-forest.svg"
        forest.write_text(forest_svg(doc), encoding="utf-8")
        written.append(forest)
        for pair_id in doc["pairs"]:
            svg = categories_svg(doc, pair_id)
            if svg:
                path = out / f"{label}-categories-{pair_id}.svg"
                path.write_text(svg, encoding="utf-8")
                written.append(path)
    if readme is not None:
        if not real:
            raise ReportError("refusing to write MOCK numbers into the README")
        target = Path(readme)
        target.write_text(
            update_readme(target.read_text(encoding="utf-8"), real[-1]), encoding="utf-8"
        )
        written.append(target)
    return written
