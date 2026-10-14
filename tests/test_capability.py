import collections
import json

import pytest

from refusalscope.client import ChatClient
from refusalscope.mock import MockProfile, MockServer
from refusalscope.suites.capability import (
    CapabilityOracle,
    build_items,
    grade_capability,
    last_number,
    reference_reply,
)


@pytest.fixture(scope="module")
def items():
    return build_items()


def item(check, category="format"):
    return {"id": "C-x", "category": category, "check": check}


def test_size_mix_and_uniqueness(items):
    assert len(items) == 100
    assert collections.Counter(i["category"] for i in items) == {
        "arithmetic": 40, "word_problem": 10, "format": 50,
    }  # fmt: skip
    assert len({i["prompt"] for i in items}) == 100
    assert len({i["id"] for i in items}) == 100


def test_deterministic_and_seed_sensitive(items):
    assert build_items() == items
    assert build_items(seed=1)[0] != items[0]
    assert len(build_items(n_items=20)) == 20


def test_reference_replies_pass_the_grader(items):
    for entry in items:
        assert grade_capability(entry, reference_reply(entry))["correct"], entry["prompt"]


def test_wrong_replies_fail(items):
    for entry in items:
        assert not grade_capability(entry, "I am not sure.")["correct"], entry["prompt"]
        assert not grade_capability(entry, "")["correct"]
        assert not grade_capability(entry, None)["correct"]


def test_request_failure_is_incorrect(items):
    assert not grade_capability(items[0], "anything", ok=False)["correct"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("The answer is 644.", 644.0), ("1,234 apples", 1234.0), ("from 3 to 17", 17.0), ("-5", -5.0),
     ("none", None), ("<think>9</think>12", 12.0), ("2.5", 2.5)],
)  # fmt: skip
def test_last_number(text, expected):
    assert last_number(text) == expected


def test_number_grader_checks_the_last_number():
    entry = item({"type": "number", "value": 644}, "arithmetic")
    assert grade_capability(entry, "420 + 224 = 644")["correct"]
    assert grade_capability(entry, "644")["correct"]
    assert not grade_capability(entry, "644 is wrong, it is 645")["correct"]


def test_exact_grader_ignores_quotes_case_and_final_period():
    entry = item({"type": "exact", "value": "LANTERN"})
    for reply in ("LANTERN", "lantern", "'lantern'.", "`LANTERN`", "<think>x</think>lantern"):
        assert grade_capability(entry, reply)["correct"], reply
    assert not grade_capability(entry, "LANTERN and more")["correct"]


def test_comma_list_grader():
    entry = item({"type": "comma_list", "count": 3})
    assert grade_capability(entry, "apple, pear, plum")["correct"]
    assert grade_capability(entry, "apple, pear, plum.")["correct"]
    assert not grade_capability(entry, "apple, pear")["correct"]
    assert not grade_capability(entry, "apple, pear, plum, fig")["correct"]
    assert not grade_capability(entry, "apple, pear, 3")["correct"]
    assert not grade_capability(entry, "apple,\npear, plum")["correct"]


def test_json_grader_accepts_fences_and_rejects_changes():
    entry = item({"type": "json", "value": {"a": 5, "b": "pebble"}})
    assert grade_capability(entry, '{"a": 5, "b": "pebble"}')["correct"]
    assert grade_capability(entry, '```json\n{"b": "pebble", "a": 5}\n```')["correct"]
    assert not grade_capability(entry, '{"a": "5", "b": "pebble"}')["correct"]
    assert not grade_capability(entry, "not json")["correct"]
    assert not grade_capability(entry, json.dumps({"a": 5}))["correct"]


def test_sequence_prefix_sorted_graders():
    seq = item({"type": "sequence", "value": [3, 4, 5]})
    assert (
        grade_capability(seq, "3 4 5")["correct"]
        and not grade_capability(seq, "3, 4, 5")["correct"]
    )
    prefix = item({"type": "prefix", "prefix": "Answer:", "contains": "89"})
    assert grade_capability(prefix, "Answer: the result is 89")["correct"]
    assert not grade_capability(prefix, "The answer: 89")["correct"]
    assert not grade_capability(prefix, "")["correct"]
    ordered = item({"type": "sorted", "value": [3, 9, 20]})
    assert (
        grade_capability(ordered, "3, 9, 20")["correct"]
        and grade_capability(ordered, "3 9 20.")["correct"]
    )
    assert not grade_capability(ordered, "9, 3, 20")["correct"]
    assert not grade_capability(ordered, "three, nine")["correct"]


def test_lowercase_and_bullets_graders():
    low = item({"type": "lowercase", "topic_word": "sea"})
    assert grade_capability(low, "the sea was calm and bright this morning.")["correct"]
    assert not grade_capability(low, "The sea was calm and bright this morning.")["correct"]
    assert not grade_capability(low, "the sea was calm. it was bright.")["correct"]
    assert not grade_capability(low, "the sky was calm and bright this morning.")["correct"]
    assert not grade_capability(low, "the sea.")["correct"]
    bullets = item({"type": "bullets", "count": 2})
    assert grade_capability(bullets, "- one\n- two")["correct"]
    assert not grade_capability(bullets, "- one\n- two\n- three")["correct"]
    assert not grade_capability(bullets, "* one\n* two")["correct"]


def test_mock_capability_tracks_probability(items):
    oracle = CapabilityOracle(items)
    with MockServer(profile=MockProfile(p_capability=1.0), provider=oracle.lookup) as good:
        client = ChatClient(good.base_url, "m", retries=0)
        results = [
            grade_capability(i, client.chat([{"role": "user", "content": i["prompt"]}]).content)
            for i in items
        ]
    assert all(r["correct"] for r in results)
    with MockServer(profile=MockProfile(p_capability=0.5, seed=3), provider=oracle.lookup) as half:
        client = ChatClient(half.base_url, "m", retries=0)
        results = [
            grade_capability(
                i, client.chat([{"role": "user", "content": i["prompt"]}], seed=1).content
            )
            for i in items
        ]
    assert 0.35 <= sum(r["correct"] for r in results) / 100 <= 0.65


def test_oracle_ignores_tool_requests_and_unknown_prompts(items):
    oracle = CapabilityOracle(items)
    ok = {"messages": [{"role": "user", "content": items[0]["prompt"]}]}
    assert oracle.lookup(ok) is not None
    assert oracle.lookup({**ok, "tools": [{"x": 1}]}) is None
    assert oracle.lookup({"messages": [{"role": "user", "content": "zzz"}]}) is None
    assert oracle.lookup({"messages": []}) is None
