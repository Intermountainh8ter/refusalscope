"""Suite C, part 3: a small general-capability control.

One hundred generated items (40 arithmetic, 10 word problems, 50 format following) with
deterministic graders. If tool-call accuracy falls by about as much as this control, the damage
is general rather than specific to tool calling (hypothesis H4).
"""

from __future__ import annotations

import json
import re
from typing import Any

from refusalscope.mock import Scenario, TextScenario
from refusalscope.util import stable_rng, strip_think

SUITE_ID = "capability"
SUITE_VERSION = 1
DEFAULT_SEED = 3003
DEFAULT_ITEMS = 100

WORDS = ["lantern", "orchard", "compass", "harbor", "meadow", "pencil", "window", "garden",
         "bridge", "candle", "ribbon", "anchor", "violin", "pebble", "lighthouse"]  # fmt: skip
CATEGORY_LISTS = {
    "fruits": ["apple", "pear", "plum", "mango", "cherry", "grape", "peach", "lemon"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"],
    "animals": ["otter", "falcon", "tiger", "rabbit", "dolphin", "heron", "badger", "lynx"],
}
TOPICS = ["the sea", "a train station", "a snowy morning", "a garden", "an old library"]

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> float | None:
    """The last number in ``text`` (thousands separators removed), or ``None``."""
    matches = _NUMBER.findall(strip_think(text))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


# --------------------------------------------------------------------------- builders


def _arithmetic(rng, i):
    kind = i % 5
    if kind == 0:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        text, answer = f"What is {a} + {b}?", a + b
    elif kind == 1:
        a, b = rng.randint(500, 999), rng.randint(100, 499)
        text, answer = f"What is {a} - {b}?", a - b
    elif kind == 2:
        a, b = rng.randint(12, 99), rng.randint(3, 19)
        text, answer = f"What is {a} * {b}?", a * b
    elif kind == 3:
        b, q = rng.randint(3, 19), rng.randint(5, 60)
        text, answer = f"What is {b * q} divided by {b}?", q
    else:
        a, b, c = rng.randint(2, 30), rng.randint(2, 12), rng.randint(2, 12)
        text, answer = f"What is {a} + {b} * {c}?", a + b * c
    return {
        "category": "arithmetic",
        "prompt": f"{text} Answer with just the number.",
        "check": {"type": "number", "value": answer},
    }


def _word_problem(rng, i):
    kind = i % 5
    if kind == 0:
        a, b, c = rng.randint(20, 60), rng.randint(3, 15), rng.randint(3, 15)
        text = (f"A box holds {a} apples. {b} are eaten and then {c} more are added. "
                "How many apples are in the box?")  # fmt: skip
        answer = a - b + c
    elif kind == 1:
        v, t = rng.randint(30, 90), rng.randint(2, 6)
        text, answer = (
            f"A train travels {v} kilometers per hour for {t} hours. How many kilometers does it travel?",
            v * t,
        )
    elif kind == 2:
        n, p = rng.randint(3, 9), rng.randint(2, 8)
        text, answer = (
            f"Notebooks cost {p} dollars each. How many dollars do {n} notebooks cost?",
            n * p,
        )
    elif kind == 3:
        total, groups = rng.choice([(24, 4), (36, 6), (45, 5), (56, 7), (48, 8)])
        text, answer = (
            f"{total} students are split evenly into {groups} teams. How many students are on each team?",
            total // groups,
        )
    else:
        a, b = rng.randint(10, 40), rng.randint(10, 40)
        text, answer = (
            f"Mia has {a} stickers and Leo has {b} stickers. How many stickers do they have together?",
            a + b,
        )
    return {
        "category": "word_problem",
        "prompt": f"{text} Answer with just the number.",
        "check": {"type": "number", "value": answer},
    }


def _fmt_exact_upper(rng, i):
    word = rng.choice(WORDS)
    return {"prompt": f"Reply with the single word '{word}' in uppercase letters and nothing else.",
            "check": {"type": "exact", "value": word.upper()}}  # fmt: skip


def _fmt_comma_list(rng, i):
    name = rng.choice(list(CATEGORY_LISTS))
    n = rng.randint(3, 5)
    return {"prompt": f"List exactly {n} {name}, separated by commas, with no other text.",
            "check": {"type": "comma_list", "count": n}}  # fmt: skip


def _fmt_json(rng, i):
    x, s = rng.randint(2, 99), rng.choice(WORDS)
    return {
        "prompt": f'Output a JSON object with the key "a" set to the integer {x} and the key "b" set to '
                  f'the string "{s}". Output only the JSON.',
        "check": {"type": "json", "value": {"a": x, "b": s}},
    }  # fmt: skip


def _fmt_sequence(rng, i):
    a = rng.randint(1, 20)
    b = a + rng.randint(3, 7)
    return {"prompt": f"Write the numbers from {a} to {b} separated by single spaces and nothing else.",
            "check": {"type": "sequence", "value": list(range(a, b + 1))}}  # fmt: skip


def _fmt_prefix(rng, i):
    a, b = rng.randint(10, 80), rng.randint(10, 80)
    return {"prompt": f"Answer in one line that starts with 'Answer:' and states the result of {a} + {b}.",
            "check": {"type": "prefix", "prefix": "Answer:", "contains": str(a + b)}}  # fmt: skip


def _fmt_reverse(rng, i):
    word = rng.choice(WORDS)
    return {"prompt": f"Spell the word '{word}' backwards. Output only the reversed word.",
            "check": {"type": "exact", "value": word[::-1].upper()}}  # fmt: skip


def _fmt_sort(rng, i):
    numbers = rng.sample(range(1, 100), 5)
    return {"prompt": "Sort these numbers from smallest to largest and output them separated by commas "
                      f"with no other text: {', '.join(map(str, numbers))}.",
            "check": {"type": "sorted", "value": sorted(numbers)}}  # fmt: skip


def _fmt_lowercase(rng, i):
    topic = rng.choice(TOPICS)
    return {"prompt": f"Write one sentence about {topic} that is entirely in lowercase letters and ends with a period.",
            "check": {"type": "lowercase", "topic_word": topic.split()[-1]}}  # fmt: skip


def _fmt_bullets(rng, i):
    n = rng.randint(2, 4)
    topic = rng.choice(TOPICS)
    return {"prompt": f"Give exactly {n} bullet points about {topic}. Each must start with '- '. Output nothing else.",
            "check": {"type": "bullets", "count": n}}  # fmt: skip


FORMAT_BUILDERS = [
    _fmt_exact_upper, _fmt_comma_list, _fmt_json, _fmt_sequence, _fmt_prefix,
    _fmt_reverse, _fmt_sort, _fmt_lowercase, _fmt_bullets,
]  # fmt: skip


def build_items(seed: int = DEFAULT_SEED, n_items: int = DEFAULT_ITEMS) -> list[dict[str, Any]]:
    """Generate the control set: 40% arithmetic, 10% word problems, 50% format following."""
    n_arith = round(n_items * 0.4)
    n_word = round(n_items * 0.1)
    plan = (
        [("arithmetic", i) for i in range(n_arith)]
        + [("word_problem", i) for i in range(n_word)]
        + [("format", i) for i in range(n_items - n_arith - n_word)]
    )
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (category, k) in enumerate(plan):
        for attempt in range(100):
            rng = stable_rng(SUITE_ID, seed, category, k, attempt)
            if category == "arithmetic":
                body = _arithmetic(rng, k)
            elif category == "word_problem":
                body = _word_problem(rng, k)
            else:
                builder = FORMAT_BUILDERS[k % len(FORMAT_BUILDERS)]
                body = {"category": "format", **builder(rng, k)}
            if body["prompt"] not in seen:
                seen.add(body["prompt"])
                break
        else:  # pragma: no cover - the value pools are far larger than the item count
            raise RuntimeError(f"could not build a unique capability item #{index}")
        items.append({"id": f"C-{index:04d}", "suite": SUITE_ID, "version": SUITE_VERSION, **body})
    return items


# --------------------------------------------------------------------------- grading


def _strip_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _clean(text: str) -> str:
    value = strip_think(text).strip()
    for _ in range(2):
        value = value.strip("`\"' ").rstrip(".").strip()
    return value


def grade_capability(item: dict[str, Any], text: str | None, ok: bool = True) -> dict[str, Any]:
    """Deterministically grade a reply to one control item."""
    if not ok:
        return {"correct": False, "category": item["category"], "reason": "request failed"}
    visible = strip_think(text)
    check = item["check"]
    kind = check["type"]
    good = False
    if kind == "number":
        found = last_number(visible)
        good = found is not None and abs(found - check["value"]) < 1e-6
    elif kind == "exact":
        good = _clean(visible).upper() == check["value"].upper()
    elif kind == "comma_list":
        parts = [p.strip() for p in visible.strip().rstrip(".").split(",")]
        good = (
            len(parts) == check["count"]
            and all(p and re.fullmatch(r"[A-Za-z][A-Za-z -]*", p) for p in parts)
            and "\n" not in visible.strip()
        )
    elif kind == "json":
        try:
            good = json.loads(_strip_fence(visible)) == check["value"]
        except json.JSONDecodeError:
            good = False
    elif kind == "sequence":
        tokens = visible.strip().split()
        good = tokens == [str(n) for n in check["value"]]
    elif kind == "prefix":
        first_line = visible.strip().splitlines()[0] if visible.strip() else ""
        good = first_line.startswith(check["prefix"]) and check["contains"] in first_line
    elif kind == "sorted":
        try:
            good = [int(p) for p in re.split(r"[,\s]+", visible.strip().rstrip(".")) if p] == check[
                "value"
            ]
        except ValueError:
            good = False
    elif kind == "lowercase":
        sentence = visible.strip()
        good = (
            len(sentence) >= 15
            and sentence == sentence.lower()
            and sentence.endswith(".")
            and sentence.count(".") == 1
            and check["topic_word"].lower() in sentence
        )
    elif kind == "bullets":
        lines = [ln for ln in visible.strip().splitlines() if ln.strip()]
        good = len(lines) == check["count"] and all(ln.startswith("- ") for ln in lines)
    return {"correct": bool(good), "category": item["category"], "reason": kind}


def reference_reply(item: dict[str, Any]) -> str:
    """A reply that passes the item's grader (used by the mock server and by tests)."""
    check = item["check"]
    kind = check["type"]
    if kind == "number":
        return f"The answer is {check['value']}"
    if kind == "exact":
        return check["value"]
    if kind == "comma_list":
        pool = next(v for v in CATEGORY_LISTS.values())
        return ", ".join(pool[: check["count"]])
    if kind == "json":
        return json.dumps(check["value"])
    if kind == "sequence":
        return " ".join(str(n) for n in check["value"])
    if kind == "prefix":
        return f"{check['prefix']} {check['contains']}"
    if kind == "sorted":
        return ", ".join(str(n) for n in check["value"])
    if kind == "lowercase":
        return f"i walked past {check['topic_word']} and smiled quietly."
    return "\n".join(f"- point {n}" for n in range(1, check["count"] + 1))


class CapabilityOracle:
    """Lets the mock server answer capability items correctly or incorrectly."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._index: dict[str, Scenario] = {
            item["prompt"]: TextScenario(item["id"], reference_reply(item), "I am not sure.")
            for item in items
        }

    def lookup(self, body: dict[str, Any]) -> Scenario | None:
        if body.get("tools"):
            return None
        users = [m for m in body.get("messages") or [] if m.get("role") == "user"]
        return self._index.get(users[-1].get("content")) if users else None
