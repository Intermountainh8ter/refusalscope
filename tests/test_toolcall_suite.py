import collections
import json
import re
from pathlib import Path

import pytest

from refusalscope.client import ChatClient, ToolCall
from refusalscope.mock import MockProfile, MockServer, mutate_call
from refusalscope.schema import LENIENT, STRICT, materialize, validate
from refusalscope.suites import rows_digest, write_snapshots
from refusalscope.suites.toolcall import (
    CATEGORIES,
    DEFAULT_PER_CATEGORY,
    TAXONOMY,
    ToolcallOracle,
    build_cases,
    case_key,
    grade_toolcall,
    scenario_for,
    summarize_grades,
)
from refusalscope.suites.toolcall_lib import TOOLS, date_words, number_words, time_words
from refusalscope.turn import Turn
from refusalscope.util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    return build_cases()


@pytest.fixture(scope="module")
def by_id(cases):
    return {c["id"]: c for c in cases}


def call(name, arguments, raw=None):
    text = raw if raw is not None else json.dumps(arguments)
    return ToolCall(
        id="c", name=name, arguments_raw=text,
        arguments=arguments if raw is None else None,
        parse_error=None if raw is None else "invalid JSON: x",
    )  # fmt: skip


def turn(*calls, content=""):
    return Turn(ok=True, content=content, tool_calls=list(calls))


def expected_turn(case):
    return turn(*[call(c["name"], materialize(c["arguments"])) for c in case["expected"]["calls"]],
                content="" if case["expected"]["calls"] else "Plain text answer.")  # fmt: skip


def first(cases, category):
    return next(c for c in cases if c["category"] == category)


# ------------------------------------------------------------------ generator


def test_suite_size_and_categories(cases):
    assert len(cases) == len(CATEGORIES) * DEFAULT_PER_CATEGORY == 374
    assert len(CATEGORIES) == 17
    counts = collections.Counter(c["category"] for c in cases)
    assert set(counts.values()) == {22}
    assert len({c["id"] for c in cases}) == 374


def test_generation_is_deterministic_and_seed_sensitive(cases):
    assert build_cases() == cases
    other = build_cases(seed=7)
    assert other != cases
    assert len(other) == 374


def test_per_category_size_is_configurable():
    small = build_cases(per_category=2)
    assert len(small) == 34
    assert [c["id"] for c in small][:2] == ["A-simple_string-000", "A-simple_string-001"]


def test_case_keys_are_unique(cases):
    keys = [case_key(c) for c in cases]
    assert len(set(keys)) == len(keys)


def test_case_structure(cases):
    for case in cases:
        assert case["messages"][0]["role"] == "system"
        assert case["messages"][-1]["role"] == "user"
        assert case["tools"] and all(t["type"] == "function" for t in case["tools"])
        assert case["expected"]["reason"] in (None, "irrelevant", "missing_required")
        assert case["version"] == 1 and case["suite"] == "toolcall"
        names = [t["function"]["name"] for t in case["tools"]]
        assert len(names) == len(set(names))


def test_tool_library_schemas_are_wellformed():
    assert len(TOOLS) >= 30
    for tool in TOOLS.values():
        fn = tool.schema()["function"]
        params = fn["parameters"]
        assert params["type"] == "object" and fn["description"]
        assert set(params["required"]) <= set(params["properties"])
        for prop in params["properties"].values():
            assert prop["type"] in {"string", "integer", "number", "boolean", "array", "object"}
            assert prop["description"]


def test_expected_calls_satisfy_their_own_schema(cases):
    for case in cases:
        functions = {t["function"]["name"]: t["function"] for t in case["tools"]}
        for expected in case["expected"]["calls"]:
            args = materialize(expected["arguments"])
            issues, _ = validate(functions[expected["name"]]["parameters"], args, STRICT)
            assert not issues, (case["id"], issues)


def test_expected_strings_appear_in_the_request(cases):
    """The request must contain every string the model is expected to pass through."""
    skip = {
        "enum_synonym",
        "type_coercion",
        "multi_turn_context",
        "irrelevance",
        "missing_required",
    }
    for case in cases:
        if case["category"] in skip:
            continue
        text = case["messages"][-1]["content"].casefold()
        for expected in case["expected"]["calls"]:
            for key, value in materialize(expected["arguments"]).items():
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if isinstance(item, str):
                        if tool_param_is_display_mapped(expected["name"], key):
                            continue
                        assert item.casefold() in text, (case["id"], key, item)


def tool_param_is_display_mapped(tool, key):
    return bool(TOOLS[tool].params[key].display)


def test_every_category_has_the_expected_shape(cases):
    for case in cases:
        n_calls = len(case["expected"]["calls"])
        category = case["category"]
        if category in {"irrelevance", "missing_required"}:
            assert n_calls == 0
        elif category == "parallel_same":
            assert 2 <= n_calls <= 4 and len({c["name"] for c in case["expected"]["calls"]}) == 1
        elif category == "parallel_multiple":
            assert (
                2 <= n_calls <= 3 and len({c["name"] for c in case["expected"]["calls"]}) == n_calls
            )
        else:
            assert n_calls == 1


def test_toolset_sizes_by_category(cases):
    sizes = collections.defaultdict(list)
    for case in cases:
        sizes[case["category"]].append(case["meta"]["n_offered"])
    assert max(sizes["simple_string"]) == 1 and max(sizes["simple_numeric"]) == 1
    assert 4 <= min(sizes["multiple_choice"]) and max(sizes["multiple_choice"]) <= 6
    assert 12 <= min(sizes["large_toolset"]) and max(sizes["large_toolset"]) <= 16


def test_long_arguments_are_long_and_special_chars_are_special(cases):
    longs = [c for c in cases if c["category"] == "long_argument"]
    assert all(c["meta"]["text_chars"] >= 400 for c in longs)
    assert max(c["meta"]["text_chars"] for c in longs) < 1100
    specials = "".join(
        materialize(c["expected"]["calls"][0]["arguments"])[c["meta"]["verbatim_param"]]
        for c in cases
        if c["category"] == "special_chars"
    )
    for needle in ('"', "\\", "\n", "é", "{", "'"):
        assert needle in specials


def test_many_params_cases_pass_six_or_more_parameters(cases):
    many = [c for c in cases if c["category"] == "many_params"]
    assert all(c["meta"]["n_params"] >= 6 for c in many)


def test_multi_turn_cases_carry_the_prior_tool_result(cases):
    for case in (c for c in cases if c["category"] == "multi_turn_context"):
        roles = [m["role"] for m in case["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
        context = json.dumps(case["messages"])
        arguments = materialize(case["expected"]["calls"][0]["arguments"])
        for value in arguments.values():
            for piece in str(value).split("/"):
                assert piece in context, (case["id"], piece)


def test_surface_helpers():
    assert (
        number_words(0) == "zero"
        and number_words(21) == "twenty-one"
        and number_words(90) == "ninety"
    )
    with pytest.raises(ValueError):
        number_words(100)
    assert date_words("2026-03-05") == "March 5th, 2026"
    assert date_words("2027-11-22") == "November 22nd, 2027"
    assert date_words("2026-01-13") == "January 13th, 2026"
    assert date_words("2026-02-01") == "February 1st, 2026"
    assert time_words("15:30") == "3:30 pm" and time_words("09:00") == "9 am"
    assert time_words("00:15") == "12:15 am" and time_words("12:00") == "12 pm"


def test_type_coercion_cases_spell_values_out(cases):
    ints = [c for c in cases if c["category"] == "type_coercion" and c["meta"]["coercion"] == "int"]
    dates = [
        c for c in cases if c["category"] == "type_coercion" and c["meta"]["coercion"] == "date"
    ]
    assert ints and dates
    assert all(not re.search(r"\d", c["messages"][-1]["content"]) or c["meta"]["function"] == "book_hotel"
               for c in ints)  # fmt: skip
    assert all(not re.search(r"\d{4}-\d{2}-\d{2}", c["messages"][-1]["content"]) for c in dates)


def test_snapshot_matches_generator(tmp_path):
    committed = list(read_jsonl(ROOT / "suites" / "toolcall.jsonl"))
    assert rows_digest(committed) == rows_digest(json.loads(json.dumps(build_cases())))
    counts = write_snapshots(tmp_path)
    assert counts["toolcall"] == 374
    assert (tmp_path / "toolcall.jsonl").read_bytes() == (
        ROOT / "suites" / "toolcall.jsonl"
    ).read_bytes()
    assert write_snapshots(tmp_path, only=["nothing"]) == {}


# ------------------------------------------------------------------ grader: oracle and taxonomy


def test_perfect_answers_score_correct_in_both_modes(cases):
    for case in cases:
        for mode in (STRICT, LENIENT):
            grade = grade_toolcall(case, expected_turn(case), mode)
            assert grade["correct"] and grade["valid"] and grade["schema_valid"], (
                case["id"],
                grade,
            )
            assert grade["failure"] is None


def test_request_error():
    case = build_cases(per_category=1)[0]
    grade = grade_toolcall(case, Turn(ok=False, error="boom"))
    assert grade["failure"] == "request_error" and not grade["valid"]


def test_unknown_mode_rejected():
    case = build_cases(per_category=1)[0]
    with pytest.raises(ValueError, match="grading mode"):
        grade_toolcall(case, expected_turn(case), "sloppy")


def test_text_tool_call_is_not_valid(by_id):
    case = by_id["A-simple_string-000"]
    name = case["expected"]["calls"][0]["name"]
    body = json.dumps(
        {"name": name, "arguments": materialize(case["expected"]["calls"][0]["arguments"])}
    )
    grade = grade_toolcall(case, turn(content=f"<tool_call>{body}</tool_call>"))
    assert grade["failure"] == "text_tool_call" and grade["text_tool_call"] and not grade["valid"]


def test_malformed_arguments(by_id):
    case = by_id["A-simple_string-000"]
    name = case["expected"]["calls"][0]["name"]
    grade = grade_toolcall(case, turn(call(name, None, raw='{"name": "El')))
    assert grade["failure"] == "malformed_arguments" and not grade["valid"] and not grade["correct"]


def test_missed_call_when_a_call_was_expected(by_id):
    grade = grade_toolcall(
        by_id["A-simple_string-000"], turn(content="I can look that up if you like.")
    )
    assert grade["failure"] == "missed_call" and not grade["valid"]


def test_spurious_and_premature_calls(cases):
    irrelevant = first(cases, "irrelevance")
    tool = irrelevant["tools"][0]["function"]["name"]
    grade = grade_toolcall(irrelevant, turn(call(tool, {})))
    assert grade["failure"] == "spurious_call" and grade["valid"] and not grade["correct"]
    missing = first(cases, "missing_required")
    tool = missing["tools"][0]["function"]["name"]
    grade = grade_toolcall(missing, turn(call(tool, {"x": 1})))
    assert grade["failure"] == "premature_call"


def test_no_call_cases_reject_text_tool_calls(cases):
    irrelevant = first(cases, "irrelevance")
    name = irrelevant["tools"][0]["function"]["name"]
    grade = grade_toolcall(irrelevant, turn(content=f'{{"name": "{name}", "arguments": {{}}}}'))
    assert grade["failure"] == "text_tool_call" and not grade["valid"]


def test_wrong_call_count(cases):
    parallel = first(cases, "parallel_same")
    full = expected_turn(parallel)
    grade = grade_toolcall(parallel, turn(full.tool_calls[0]))
    assert grade["failure"] == "wrong_call_count" and grade["valid"]
    grade = grade_toolcall(parallel, turn(*full.tool_calls, full.tool_calls[0]))
    assert grade["failure"] == "wrong_call_count"


def test_parallel_calls_are_order_insensitive(cases):
    parallel = first(cases, "parallel_same")
    full = expected_turn(parallel)
    reordered = turn(*reversed(full.tool_calls))
    assert grade_toolcall(parallel, reordered)["correct"]


def test_parallel_failure_reports_the_first_mismatching_label(cases):
    case = first(cases, "parallel_multiple")
    calls = expected_turn(case).tool_calls
    wrong = ToolCall("c", calls[0].name, "{}", {}, None)
    grade = grade_toolcall(case, turn(wrong, *calls[1:]))
    assert grade["failure"] == "missing_required_arg"


def _single_arg_case(cases):
    return next(c for c in cases if c["id"] == "A-simple_string-000")


def test_failure_labels_for_each_mutation(cases):
    case = next(
        c
        for c in cases
        if c["category"] == "optional_args" and len(c["expected"]["calls"][0]["arguments"]) >= 3
    )
    expected = case["expected"]["calls"][0]
    good = {"name": expected["name"], "arguments": materialize(expected["arguments"])}
    labels = {}
    for mode in ("wrong_value", "missing_arg", "wrong_type", "unknown_arg", "wrong_function"):
        mutated, _ = mutate_call(good, case["tools"], mode)
        labels[mode] = grade_toolcall(case, turn(call(mutated["name"], mutated["arguments"])))[
            "failure"
        ]
    assert labels["wrong_value"] == "wrong_value"
    assert labels["missing_arg"] == "missing_required_arg"
    assert labels["wrong_type"] == "type_mismatch"
    assert labels["unknown_arg"] == "unknown_argument"
    assert labels["wrong_function"] in {"wrong_function", "unknown_function"}


def test_unknown_function(by_id):
    case = by_id["A-simple_string-000"]
    grade = grade_toolcall(case, turn(call("does_not_exist", {})))
    assert grade["failure"] == "unknown_function" and not grade["schema_valid"]


def test_wrong_function_when_the_function_is_offered(cases):
    case = first(cases, "multiple_choice")
    expected = case["expected"]["calls"][0]["name"]
    other = next(t["function"] for t in case["tools"] if t["function"]["name"] != expected)
    args = {k: "x" for k in other["parameters"]["required"]}
    grade = grade_toolcall(case, turn(call(other["name"], args)))
    assert grade["failure"] == "wrong_function"


def test_enum_violation_and_lenient_forgiveness(cases):
    case = next(
        c for c in cases if c["category"] == "enum_synonym" and c["meta"]["enum_param"] == "units"
    )
    exp = case["expected"]["calls"][0]
    args = materialize(exp["arguments"])
    args["units"] = args["units"].upper()
    strict = grade_toolcall(case, turn(call(exp["name"], args)), STRICT)
    assert strict["failure"] == "enum_violation" and not strict["schema_valid"]
    assert grade_toolcall(case, turn(call(exp["name"], args)), LENIENT)["correct"]
    args["units"] = "kelvin"
    assert (
        grade_toolcall(case, turn(call(exp["name"], args)), LENIENT)["failure"] == "enum_violation"
    )


def test_numeric_strings_are_type_mismatch_strict_but_fine_lenient(cases):
    case = next(
        c
        for c in cases
        if c["category"] == "simple_numeric" and c["meta"]["function"] == "set_timer"
    )
    exp = case["expected"]["calls"][0]
    args = {**materialize(exp["arguments"])}
    args["minutes"] = str(args["minutes"])
    assert grade_toolcall(case, turn(call(exp["name"], args)), STRICT)["failure"] == "type_mismatch"
    assert grade_toolcall(case, turn(call(exp["name"], args)), LENIENT)["correct"]


def handmade(tool_name, args):
    tool = TOOLS[tool_name]
    return {
        "id": "hand",
        "tools": [tool.schema()],
        "messages": [{"role": "user", "content": "hi"}],
        "expected": {
            "calls": [{"name": tool_name, "arguments": args}],
            "reason": None,
            "ordered": False,
        },
    }


def test_extra_and_missing_optional_arguments():
    case = handmade("get_weather", {"city": "Paris", "days": 3})
    exact = turn(call("get_weather", {"city": "Paris", "days": 3}))
    assert grade_toolcall(case, exact)["correct"]
    extra = turn(call("get_weather", {"city": "Paris", "days": 3, "units": "celsius"}))
    assert grade_toolcall(case, extra)["failure"] == "extra_optional_arg"
    assert grade_toolcall(case, extra, LENIENT)["correct"]
    omitted = turn(call("get_weather", {"city": "Paris"}))
    assert grade_toolcall(case, omitted)["failure"] == "missing_optional_arg"


def test_default_valued_optional_argument_is_not_an_extra():
    case = handmade("web_search", {"query": "tides"})
    ok = turn(call("web_search", {"query": "tides", "recency": "any"}))
    assert grade_toolcall(case, ok)["correct"]
    other = turn(call("web_search", {"query": "tides", "recency": "week"}))
    assert grade_toolcall(case, other)["failure"] == "extra_optional_arg"


def test_verbatim_arguments_must_be_exact(cases):
    case = first(cases, "special_chars")
    exp = case["expected"]["calls"][0]
    args = materialize(exp["arguments"])
    param = case["meta"]["verbatim_param"]
    assert grade_toolcall(case, turn(call(exp["name"], args)))["correct"]
    mangled = {**args, param: args[param].replace('"', "'").replace("\\", "/") + "!"}
    assert grade_toolcall(case, turn(call(exp["name"], mangled)))["failure"] == "wrong_value"
    crlf = {**args, param: args[param].replace("\n", "\r\n") + "\n"}
    assert grade_toolcall(case, turn(call(exp["name"], crlf)))["correct"]


def test_schema_valid_is_reported_separately_from_correct(by_id):
    case = by_id["A-simple_string-000"]
    exp = case["expected"]["calls"][0]
    args = {k: v + " (wrong)" for k, v in materialize(exp["arguments"]).items()}
    grade = grade_toolcall(case, turn(call(exp["name"], args)))
    assert grade["valid"] and grade["schema_valid"] and not grade["correct"]
    assert grade["failure"] == "wrong_value"


def test_taxonomy_labels_are_unique_and_documented():
    assert len(set(TAXONOMY)) == len(TAXONOMY) == 16
    text = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
    for label in TAXONOMY:
        assert f"`{label}`" in text, label


def test_summarize_grades(cases):
    rows = [(case["category"], grade_toolcall(case, expected_turn(case))) for case in cases[:34]]
    rows.append(("simple_string", grade_toolcall(cases[0], turn(content="no"))))
    summary = summarize_grades(rows)
    assert summary["n"] == 35
    assert summary["taxonomy"]["missed_call"] == 1
    assert summary["by_category"]["simple_string"]["n"] == 23
    assert summary["valid"] > summary["correct"] or summary["valid"] == summary["correct"]
    assert summarize_grades([])["correct"] is None


def test_scenario_for_builds_matching_scenarios(cases):
    positive = scenario_for(first(cases, "simple_string"))
    negative = scenario_for(first(cases, "irrelevance"))
    missing = scenario_for(first(cases, "missing_required"))
    assert positive.calls and not hasattr(negative, "calls")
    assert "more information" in missing.reply


# ------------------------------------------------------------------ integration with the mock


def run_through_mock(cases, profile):
    oracle = ToolcallOracle(cases)
    grades = []
    with MockServer(profile=profile, provider=oracle.lookup) as server:
        client = ChatClient(server.base_url, "mock-model", retries=0)
        for case in cases:
            result = client.chat(case["messages"], tools=case["tools"], seed=1)
            assert result.ok and result.mock
            grades.append((case, grade_toolcall(case, Turn.from_result(result))))
    return grades


def test_oracle_ignores_unknown_requests(cases):
    oracle = ToolcallOracle(cases)
    assert oracle.lookup({"messages": [{"role": "user", "content": "zzz"}], "tools": []}) is None
    assert oracle.lookup({"messages": [], "tools": [{"function": {"name": "a"}}]}) is None
    unknown = {
        "messages": [{"role": "user", "content": "zzz"}],
        "tools": [{"function": {"name": "a"}}],
    }
    assert oracle.lookup(unknown) is None


def test_full_suite_through_perfect_mock_scores_100_percent(cases):
    grades = run_through_mock(cases, MockProfile(p_correct=1.0))
    assert all(g["correct"] for _, g in grades)


@pytest.mark.parametrize(
    ("mode", "labels"),
    [
        ("text_call", {"text_tool_call", "spurious_call", "premature_call"}),
        ("malformed_json", {"malformed_arguments", "spurious_call", "premature_call"}),
        ("no_call", {"missed_call", "spurious_call", "premature_call"}),
        ("unknown_arg", {"unknown_argument", "spurious_call", "premature_call"}),
        ("missing_arg", {"missing_required_arg", "spurious_call", "premature_call", "wrong_value"}),
        ("wrong_type", {"type_mismatch", "spurious_call", "premature_call"}),
    ],
)
def test_injected_failure_modes_are_classified_correctly(cases, mode, labels):
    subset = [c for i, c in enumerate(cases) if i % 5 == 0]
    grades = run_through_mock(subset, MockProfile(p_correct=0.0).only_failure(mode))
    seen = collections.Counter(g["failure"] for _, g in grades)
    assert set(seen) <= labels | {None}, seen
    assert sum(seen.values()) >= len(subset) - seen.get(None, 0)
    assert seen.get(None, 0) <= len(subset) * 0.15


def test_mixed_failure_rate_is_measured(cases):
    grades = run_through_mock(cases, MockProfile(p_correct=0.7, seed=11))
    accuracy = sum(g["correct"] for _, g in grades) / len(grades)
    assert 0.55 <= accuracy <= 0.85
    summary = summarize_grades([(c["category"], g) for c, g in grades])
    assert sum(summary["taxonomy"].values()) == summary["n"] - sum(g["correct"] for _, g in grades)
