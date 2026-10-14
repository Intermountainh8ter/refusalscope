"""Suite A: tool-calling accuracy (generator, deterministic grader, mock oracle).

Cases are generated from a seeded random draw, so the expected calls are correct by
construction. The grader never calls a model: it parses the stored response and compares it
with ``case["expected"]``.
"""

from __future__ import annotations

import json
from itertools import permutations
from typing import Any

from refusalscope.mock import CallScenario, NoCallScenario, Scenario
from refusalscope.schema import (
    LENIENT,
    STRICT,
    first_issue_kind,
    materialize,
    validate,
    values_match,
)
from refusalscope.suites.toolcall_lib import (
    EMAIL_BODIES,
    FIRST_NAMES,
    LAST_NAMES,
    PRODUCT_QUERIES,
    SENTENCES,
    SPECIAL_SNIPPETS,
    TOOLS,
    Tool,
    date_words,
    number_words,
    s_email,
    time_words,
)
from refusalscope.textcalls import detect_text_calls
from refusalscope.turn import Turn
from refusalscope.util import stable_rng

SUITE_ID = "toolcall"
SUITE_VERSION = 1
DEFAULT_SEED = 1001
DEFAULT_PER_CATEGORY = 22

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to functions. Call a function when the request "
    "needs one and you have every required argument. If no function fits the request, or a "
    "required value is missing, answer in plain text instead and ask for what is missing."
)

CATEGORIES = (
    "simple_string",
    "simple_numeric",
    "multiple_choice",
    "large_toolset",
    "parallel_same",
    "parallel_multiple",
    "nested_object",
    "array_args",
    "optional_args",
    "enum_synonym",
    "type_coercion",
    "irrelevance",
    "missing_required",
    "long_argument",
    "special_chars",
    "many_params",
    "multi_turn_context",
)

TAXONOMY = (
    "request_error",
    "text_tool_call",
    "malformed_arguments",
    "spurious_call",
    "premature_call",
    "missed_call",
    "wrong_call_count",
    "unknown_function",
    "wrong_function",
    "missing_required_arg",
    "unknown_argument",
    "type_mismatch",
    "enum_violation",
    "wrong_value",
    "missing_optional_arg",
    "extra_optional_arg",
)
_RANK = {label: index for index, label in enumerate(TAXONOMY)}

IRRELEVANT_QUERIES = [
    "What is the capital of Australia?",
    "Write a haiku about the sea.",
    "Explain what a mutex is in two sentences.",
    "Who wrote the novel Pride and Prejudice?",
    "Give me three tips for a calm morning routine.",
    "What is the difference between a virus and a bacterium?",
    "Tell me a short joke about penguins.",
    "Why is the sky blue?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "How many legs does a spider have?",
    "What does the word serendipity mean?",
    "Suggest a name for a small bakery.",
    "Is a tomato a fruit or a vegetable?",
    "Explain recursion to a ten-year-old.",
    "What is the boiling point of water at sea level in Celsius?",
    "Write a two-line poem about autumn leaves.",
    "What are the primary colors?",
    "How does a bicycle stay upright?",
    "Name three famous painters from the Renaissance.",
    "What is the square root of 144?",
    "Describe how photosynthesis works in simple terms.",
    "What rhymes with orange?",
    "Give me a fun fact about octopuses.",
    "What is a good way to learn a new language?",
    "Explain the rules of tic-tac-toe.",
    "Why do cats purr?",
    "What does GDP stand for?",
    "Recommend a classic science fiction book.",
    "How many continents are there?",
    "Write a limerick about a forgetful robot.",
    "What is the main idea of the theory of evolution?",
    "Can you explain what an API is?",
    "What is the tallest mountain in the world?",
    "Help me think of a theme for a birthday party.",
    "What is the plural of cactus?",
    "How do vaccines train the immune system?",
    "Tell me about the history of the printing press.",
    "What is a prime number?",
    "Give me a mnemonic for the planets.",
    "What is the opposite of ephemeral?",
]

SINGLE_STRING_TOOLS = (
    "get_weather",
    "read_file",
    "delete_file",
    "web_search",
    "get_stock_price",
    "check_order_status",
    "get_directions",
    "find_restaurants",
    "list_directory",
    "search_contacts",
    "list_orders",
    "get_news",
    "translate_text",
)
NUMERIC_TOOLS = (
    "set_timer",
    "convert_currency",
    "convert_units",
    "set_thermostat",
    "send_payment",
    "calculate",
)
PARALLEL_SAME_TOOLS = (
    "get_weather",
    "get_stock_price",
    "convert_currency",
    "read_file",
    "check_order_status",
    "translate_text",
    "convert_units",
    "find_restaurants",
)
PARALLEL_MIXED_TOOLS = (
    "get_weather",
    "convert_currency",
    "set_timer",
    "get_stock_price",
    "check_order_status",
    "web_search",
    "calculate",
    "find_restaurants",
    "get_news",
    "play_music",
    "set_thermostat",
    "translate_text",
)
ARRAY_PARAMS = (
    ("create_task", "tags"),
    ("create_calendar_event", "attendees"),
    ("send_email", "cc"),
    ("create_issue", "labels"),
    ("create_note", "tags"),
)
OPTIONAL_TOOLS = (
    "get_weather",
    "create_calendar_event",
    "search_flights",
    "book_hotel",
    "web_search",
    "find_restaurants",
    "create_task",
    "play_music",
    "get_stock_price",
    "get_news",
    "list_directory",
    "write_file",
    "set_timer",
    "send_payment",
    "translate_text",
    "get_directions",
    "create_issue",
)
ENUM_SYNONYM_PARAMS = (
    ("get_weather", "units"),
    ("convert_units", "from_unit"),
    ("find_restaurants", "price_level"),
    ("search_flights", "cabin"),
    ("create_task", "priority"),
    ("set_thermostat", "mode"),
)
COERCION_CASES = (
    ("set_timer", "minutes", "int"),
    ("book_hotel", "guests", "int"),
    ("play_music", "volume", "int"),
    ("get_weather", "days", "int"),
    ("set_reminder", "date", "date"),
    ("create_calendar_event", "date", "date"),
    ("search_flights", "date", "date"),
    ("book_hotel", "check_in", "date"),
    ("set_reminder", "time", "time"),
    ("create_calendar_event", "time", "time"),
    ("list_directory", "recursive", "bool"),
    ("write_file", "append", "bool"),
    ("find_restaurants", "open_now", "bool"),
    ("play_music", "shuffle", "bool"),
    ("book_hotel", "breakfast_included", "bool"),
)
MISSING_CHOICES = [
    (name, arg) for name, tool in TOOLS.items() for arg in tool.missing if name != "list_orders"
]
VERBATIM_TOOLS = (
    ("send_email", "body"),
    ("write_file", "content"),
    ("create_note", "body"),
    ("create_issue", "body"),
    ("translate_text", "text"),
)
MANY_PARAM_TOOLS = ("create_calendar_event", "create_issue", "book_hotel")

# --------------------------------------------------------------------------- helpers


def _expected_args(
    tool: Tool, args: dict[str, Any], verbatim: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Expected arguments with matchers where order or spacing must not matter."""
    out: dict[str, Any] = {}
    for key, value in args.items():
        if key in verbatim:
            out[key] = {"$verbatim": value}
        elif tool.name == "calculate" and key == "expression":
            out[key] = {"$compact": value}
        elif isinstance(value, list):
            out[key] = {"$unordered": value}
        else:
            out[key] = value
    return out


def _call(tool: Tool, args: dict[str, Any], verbatim: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"name": tool.name, "arguments": _expected_args(tool, args, verbatim)}


def _offer(rng, must: list[str], extra: int, pool: list[str] | None = None) -> list[dict[str, Any]]:
    """Tool schemas for ``must`` plus ``extra`` distractors, shuffled."""
    candidates = [n for n in (pool or list(TOOLS)) if n not in must]
    chosen = list(must) + rng.sample(candidates, min(extra, len(candidates)))
    rng.shuffle(chosen)
    return [TOOLS[name].schema() for name in chosen]


def _case(
    category: str,
    tools: list[dict[str, Any]],
    user: str,
    calls: list[dict[str, Any]],
    reason: str | None = None,
    history: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                {"role": "user", "content": user}]  # fmt: skip
    return {
        "suite": SUITE_ID,
        "category": category,
        "messages": messages,
        "tools": tools,
        "expected": {"calls": calls, "reason": reason, "ordered": False},
        "meta": {"n_offered": len(tools), **(meta or {})},
    }


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _join_requests(sentences: list[str], rng) -> str:
    connectors = ["Also, ", "And ", "Additionally, ", "Then "]
    parts = [sentences[0]]
    parts.extend(rng.choice(connectors) + _lower_first(sentence) for sentence in sentences[1:])
    return " ".join(parts)


def _words_or_digits(value: int) -> str:
    return number_words(value) if 0 <= value < 100 else str(value)


def _long_text(rng) -> str:
    sentences = rng.sample(SENTENCES, 8)
    text = ""
    for sentence in sentences:
        text += (" " if text else "") + sentence
        if len(text) >= 420:
            break
    return text


def _special_text(rng) -> str:
    count = rng.choice([1, 1, 2])
    pieces = rng.sample(SPECIAL_SNIPPETS, count)
    return "\n".join(pieces) if count == 2 else pieces[0]


# --------------------------------------------------------------------------- builders


def _simple_string(rng, i):
    tool = TOOLS[rng.choice(SINGLE_STRING_TOOLS)]
    args = tool.sample(rng)
    return _case(
        "simple_string",
        _offer(rng, [tool.name], 0),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name},
    )


def _simple_numeric(rng, i):
    tool = TOOLS[rng.choice(NUMERIC_TOOLS)]
    args = tool.sample(rng)
    return _case(
        "simple_numeric",
        _offer(rng, [tool.name], 0),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name},
    )


def _multiple_choice(rng, i):
    tool = TOOLS[rng.choice(list(TOOLS))]
    args = tool.sample(rng)
    return _case(
        "multiple_choice",
        _offer(rng, [tool.name], rng.randint(3, 5)),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name},
    )


def _large_toolset(rng, i):
    tool = TOOLS[rng.choice(list(TOOLS))]
    args = tool.sample(rng)
    return _case(
        "large_toolset",
        _offer(rng, [tool.name], rng.randint(11, 15)),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name},
    )


def _parallel_same(rng, i):
    tool = TOOLS[rng.choice(PARALLEL_SAME_TOOLS)]
    count = rng.randint(2, 4)
    first_key = tool.required[0]
    batch: list[dict[str, Any]] = []
    for _ in range(40):
        args = tool.sample(rng)
        if all(args[first_key] != other[first_key] for other in batch):
            batch.append(args)
        if len(batch) == count:
            break
    sentences = [tool.text(args, rng) for args in batch]
    return _case(
        "parallel_same",
        _offer(rng, [tool.name], 2),
        _join_requests(sentences, rng),
        [_call(tool, args) for args in batch],
        meta={"function": tool.name, "n_calls": len(batch)},
    )


def _parallel_multiple(rng, i):
    names = rng.sample(PARALLEL_MIXED_TOOLS, rng.randint(2, 3))
    calls, sentences = [], []
    for name in names:
        tool = TOOLS[name]
        args = tool.sample(rng)
        calls.append(_call(tool, args))
        sentences.append(tool.text(args, rng))
    return _case(
        "parallel_multiple",
        _offer(rng, names, rng.randint(1, 3)),
        _join_requests(sentences, rng),
        calls,
        meta={"function": ",".join(names), "n_calls": len(calls)},
    )


def _nested_object(rng, i):
    name = rng.choice(["add_contact", "place_order", "search_products"])
    tool = TOOLS[name]
    optional = {
        "add_contact": ["address"] + (["email"] if rng.random() < 0.4 else []),
        "place_order": ["gift_wrap"] if rng.random() < 0.4 else [],
        "search_products": ["filters"],
    }[name]
    args = tool.sample(rng, optional)
    return _case(
        "nested_object",
        _offer(rng, [name], 2),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": name},
    )


def _array_args(rng, i):
    name, param = rng.choice(ARRAY_PARAMS)
    tool = TOOLS[name]
    args = tool.sample(rng, [param])
    return _case(
        "array_args",
        _offer(rng, [name], 2),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": name, "array_param": param},
    )


def _optional_args(rng, i):
    tool = TOOLS[rng.choice(OPTIONAL_TOOLS)]
    pool = [n for n in tool.optional_names if tool.params[n].type != "object"]
    chosen = rng.sample(pool, rng.randint(1, min(3, len(pool))))
    args = tool.sample(rng, chosen)
    return _case(
        "optional_args",
        _offer(rng, [tool.name], 1),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name, "optional": sorted(chosen)},
    )


def _enum_synonym(rng, i):
    name, param = rng.choice(ENUM_SYNONYM_PARAMS)
    tool = TOOLS[name]
    optional = [] if param in tool.required else [param]
    args = tool.sample(rng, optional)
    overrides = {param: rng.choice(tool.params[param].syn[args[param]])}
    if name == "convert_units":
        overrides["to_unit"] = rng.choice(tool.params["to_unit"].syn[args["to_unit"]])
    return _case(
        "enum_synonym",
        _offer(rng, [name], 2),
        tool.text(args, rng, overrides),
        [_call(tool, args)],
        meta={"function": name, "enum_param": param, "surface": overrides},
    )


def _type_coercion(rng, i):
    name, param, kind = rng.choice(COERCION_CASES)
    tool = TOOLS[name]
    optional = [] if param in tool.required else [param]
    args = tool.sample(rng, optional)
    overrides: dict[str, str] = {}
    if kind == "int":
        overrides[param] = _words_or_digits(args[param])
    elif kind == "date":
        overrides[param] = date_words(args[param])
        if name == "book_hotel":
            overrides["check_out"] = date_words(args["check_out"])
    elif kind == "time":
        if "time" not in args:
            args["time"] = tool.sample_one("time", rng)
        overrides["time"] = time_words(args["time"])
    return _case(
        "type_coercion",
        _offer(rng, [name], 2),
        tool.text(args, rng, overrides),
        [_call(tool, args)],
        meta={"function": name, "coerced_param": param, "coercion": kind},
    )


_QUERY_ORDER = stable_rng(SUITE_ID, "irrelevance-order").sample(
    range(len(IRRELEVANT_QUERIES)), len(IRRELEVANT_QUERIES)
)


def _irrelevance(rng, i):
    query = IRRELEVANT_QUERIES[_QUERY_ORDER[i % len(_QUERY_ORDER)]]
    names = rng.sample(list(TOOLS), rng.randint(3, 5))
    return _case(
        "irrelevance",
        [TOOLS[n].schema() for n in names],
        query,
        [],
        reason="irrelevant",
        meta={"function": "none"},
    )


def _missing_required(rng, i):
    name, arg = rng.choice(MISSING_CHOICES)
    tool = TOOLS[name]
    args = tool.sample(rng)
    args.pop(arg, None)
    return _case(
        "missing_required",
        _offer(rng, [name], rng.randint(1, 3)),
        tool.text_missing(args, arg, rng),
        [],
        reason="missing_required",
        meta={"function": name, "missing": arg},
    )


def _verbatim_case(rng, category: str, text: str):
    name, param = rng.choice(VERBATIM_TOOLS)
    tool = TOOLS[name]
    args = tool.sample(rng)
    args[param] = text
    user = tool.text(args, rng, templates=tool.block)
    return _case(
        category,
        _offer(rng, [name], 2),
        user,
        [_call(tool, args, verbatim=(param,))],
        meta={"function": name, "verbatim_param": param, "text_chars": len(text)},
    )


def _long_argument(rng, i):
    return _verbatim_case(rng, "long_argument", _long_text(rng))


def _special_chars(rng, i):
    return _verbatim_case(rng, "special_chars", _special_text(rng))


def _many_params(rng, i):
    tool = TOOLS[rng.choice(MANY_PARAM_TOOLS)]
    pool = tool.optional_names
    if tool.name == "create_calendar_event":
        chosen = rng.sample(pool, rng.randint(4, len(pool)))
    else:
        chosen = list(pool)
    args = tool.sample(rng, chosen)
    return _case(
        "many_params",
        _offer(rng, [tool.name], 2),
        tool.text(args, rng),
        [_call(tool, args)],
        meta={"function": tool.name, "n_params": len(args)},
    )


def _history(call_name: str, call_args: dict[str, Any], result: Any, summary: str, user: str):
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_hist_1",
                    "type": "function",
                    "function": {"name": call_name, "arguments": json.dumps(call_args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_hist_1", "content": json.dumps(result)},
        {"role": "assistant", "content": summary},
    ]


def _multi_turn(rng, i):
    scenario = rng.choice(["cancel", "email", "read", "convert"])
    if scenario == "cancel":
        customer = TOOLS["list_orders"].sample(rng)
        items = rng.sample(PRODUCT_QUERIES, 3)
        orders = [{"order_id": TOOLS["cancel_order"].sample_one("order_id", rng), "item": it,
                   "status": "processing"} for it in items]  # fmt: skip
        target = rng.choice(orders)
        history = _history(
            "list_orders", customer, orders,
            "You have three recent orders: " + ", ".join(f"{o['item']} ({o['order_id']})" for o in orders) + ".",
            TOOLS["list_orders"].text(customer, rng),
        )  # fmt: skip
        user = f"Please cancel the order for the {target['item']}."
        calls = [{"name": "cancel_order", "arguments": {"order_id": target["order_id"]}}]
        names = ["list_orders", "cancel_order"]
    elif scenario == "email":
        first = rng.choice(FIRST_NAMES)
        lasts = rng.sample(LAST_NAMES, 2)
        people = [{"name": f"{first} {last}", "email": s_email(rng)} for last in lasts]
        target = rng.choice(people)
        history = _history(
            "search_contacts", {"name": first}, people,
            f"I found two contacts named {first}: " + " and ".join(p["name"] for p in people) + ".",
            TOOLS["search_contacts"].text({"name": first}, rng),
        )  # fmt: skip
        subject = rng.choice(["Quick question", "Lunch plans", "Document review"])
        body = rng.choice(EMAIL_BODIES)
        user = f'Email {target["name"]} with the subject "{subject}" and the message "{body}"'
        calls = [
            {
                "name": "send_email",
                "arguments": {"to": target["email"], "subject": subject, "body": body},
            }
        ]
        names = ["search_contacts", "send_email"]
    elif scenario == "read":
        directory = TOOLS["list_directory"].sample_one("path", rng)
        files = [TOOLS["read_file"].sample_one("path", rng).rsplit("/", 1)[1] for _ in range(4)]
        files = list(dict.fromkeys(files))
        target = rng.choice(files)
        history = _history(
            "list_directory", {"path": directory}, {"entries": files},
            "The folder contains: " + ", ".join(files) + ".",
            TOOLS["list_directory"].text({"path": directory}, rng),
        )  # fmt: skip
        user = f"Open {target} from that folder."
        calls = [{"name": "read_file", "arguments": {"path": f"{directory}/{target}"}}]
        names = ["list_directory", "read_file"]
    else:
        ticker = TOOLS["get_stock_price"].sample_one("ticker", rng)
        price = round(rng.uniform(20, 400), 2)
        currency = rng.choice(["USD", "EUR", "GBP"])
        history = _history(
            "get_stock_price", {"ticker": ticker}, {"ticker": ticker, "price": price, "currency": currency},
            f"{ticker} is trading at {price} {currency}.",
            TOOLS["get_stock_price"].text({"ticker": ticker}, rng),
        )  # fmt: skip
        target_currency = rng.choice([c for c in ("JPY", "CAD", "CHF", "SEK") if c != currency])
        user = f"Convert that price to {target_currency}."
        calls = [{"name": "convert_currency",
                  "arguments": {"amount": price, "from_currency": currency, "to_currency": target_currency}}]  # fmt: skip
        names = ["get_stock_price", "convert_currency"]
    return _case(
        "multi_turn_context",
        _offer(rng, names, 1),
        user,
        calls,
        history=history,
        meta={"function": names[1], "scenario": scenario},
    )


BUILDERS = {
    "simple_string": _simple_string,
    "simple_numeric": _simple_numeric,
    "multiple_choice": _multiple_choice,
    "large_toolset": _large_toolset,
    "parallel_same": _parallel_same,
    "parallel_multiple": _parallel_multiple,
    "nested_object": _nested_object,
    "array_args": _array_args,
    "optional_args": _optional_args,
    "enum_synonym": _enum_synonym,
    "type_coercion": _type_coercion,
    "irrelevance": _irrelevance,
    "missing_required": _missing_required,
    "long_argument": _long_argument,
    "special_chars": _special_chars,
    "many_params": _many_params,
    "multi_turn_context": _multi_turn,
}
assert set(BUILDERS) == set(CATEGORIES)


def case_key(case: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Identity used to keep cases unique and to let the mock find a case from a request."""
    last_user = next(m["content"] for m in reversed(case["messages"]) if m["role"] == "user")
    names = tuple(sorted(t["function"]["name"] for t in case["tools"]))
    return last_user, names


def build_cases(
    seed: int = DEFAULT_SEED, per_category: int = DEFAULT_PER_CATEGORY
) -> list[dict[str, Any]]:
    """Build the suite. Deterministic: the same arguments always give the same cases."""
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for category in CATEGORIES:
        for index in range(per_category):
            for attempt in range(200):
                rng = stable_rng(SUITE_ID, seed, category, index, attempt)
                case = BUILDERS[category](rng, index)
                key = case_key(case)
                if key not in seen:
                    seen.add(key)
                    break
            else:  # pragma: no cover - would mean a generator cannot produce enough variety
                raise RuntimeError(f"could not build a unique case for {category} #{index}")
            case["id"] = f"A-{category}-{index:03d}"
            case["version"] = SUITE_VERSION
            cases.append(case)
    return cases


# --------------------------------------------------------------------------- grader


def _tool_functions(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["function"]["name"]: t["function"] for t in case["tools"]}


def _compare_call(
    expected: dict[str, Any], actual_name: str, actual_args: dict[str, Any], functions, mode: str
) -> str | None:
    """Failure label for one expected/actual pair, or ``None`` when they match."""
    if actual_name != expected["name"]:
        return "unknown_function" if actual_name not in functions else "wrong_function"
    schema = functions[actual_name].get("parameters", {})
    issues, coerced = validate(schema, actual_args, mode)
    kind = first_issue_kind(issues)
    if kind:
        return kind
    labels: set[str] = set()
    for key, wanted in expected["arguments"].items():
        if key not in coerced:
            labels.add("missing_optional_arg")
        elif not values_match(wanted, coerced[key], mode):
            labels.add("wrong_value")
    if mode == STRICT:
        properties = schema.get("properties", {})
        for key, value in coerced.items():
            if key in expected["arguments"]:
                continue
            declared = properties.get(key, {})
            if "default" in declared and declared["default"] == value:
                continue
            labels.add("extra_optional_arg")
    if not labels:
        return None
    return min(labels, key=lambda label: _RANK[label])


def _best_assignment(expected_calls, actual, functions, mode, ordered) -> str | None:
    """Match expected to actual calls (any order unless ``ordered``) and return a failure label."""
    n = len(expected_calls)
    orderings = [tuple(range(n))] if ordered else list(permutations(range(n)))
    best: tuple[int, int, str | None] | None = None
    for ordering in orderings:
        labels = [
            _compare_call(
                expected_calls[i], actual[j].name, actual[j].arguments or {}, functions, mode
            )
            for i, j in enumerate(ordering)
        ]
        failing = [label for label in labels if label]
        score = (len(failing), sum(_RANK[label] for label in failing))
        if best is None or score < best[:2]:
            best = (*score, min(failing, key=lambda label: _RANK[label]) if failing else None)
    assert best is not None
    return best[2]


def grade_toolcall(case: dict[str, Any], turn: Turn, mode: str = STRICT) -> dict[str, Any]:
    """Grade one response to a Suite A case. See ``docs/METHODOLOGY.md`` section 3.2."""
    if mode not in (STRICT, LENIENT):
        raise ValueError(f"unknown grading mode {mode!r}")
    expected = case["expected"]
    functions = _tool_functions(case)
    result: dict[str, Any] = {
        "valid": False,
        "schema_valid": False,
        "correct": False,
        "failure": None,
        "n_calls": len(turn.tool_calls),
        "text_tool_call": False,
        "mode": mode,
    }
    if not turn.ok:
        result["failure"] = "request_error"
        return result
    calls = turn.tool_calls
    if not calls and detect_text_calls(turn.content, set(functions)):
        result["text_tool_call"] = True
        result["failure"] = "text_tool_call"
        return result
    if any(call.parse_error for call in calls):
        result["failure"] = "malformed_arguments"
        return result

    schema_ok = all(
        call.name in functions
        and not validate(functions[call.name].get("parameters", {}), call.arguments or {}, mode)[0]
        for call in calls
    )
    wanted = expected["calls"]
    if not wanted:
        result["valid"] = True
        if calls:
            result["schema_valid"] = schema_ok
            result["failure"] = (
                "spurious_call" if expected.get("reason") == "irrelevant" else "premature_call"
            )
        else:
            result["schema_valid"] = True
            result["correct"] = True
        return result
    if not calls:
        result["failure"] = "missed_call"
        return result
    result["valid"] = True
    result["schema_valid"] = schema_ok
    if len(calls) != len(wanted):
        result["failure"] = "wrong_call_count"
        return result
    failure = _best_assignment(wanted, calls, functions, mode, bool(expected.get("ordered")))
    result["failure"] = failure
    result["correct"] = failure is None
    return result


def summarize_grades(rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate ``(category, grade)`` rows into rates and failure-taxonomy counts."""
    total = len(rows)

    def rate(key: str, subset=rows) -> float | None:
        return (sum(1 for _, g in subset if g[key]) / len(subset)) if subset else None

    taxonomy = dict.fromkeys(TAXONOMY, 0)
    for _, grade in rows:
        if grade["failure"]:
            taxonomy[grade["failure"]] += 1
    by_category: dict[str, Any] = {}
    for category in CATEGORIES:
        subset = [row for row in rows if row[0] == category]
        if subset:
            by_category[category] = {
                "n": len(subset),
                "valid": rate("valid", subset),
                "schema_valid": rate("schema_valid", subset),
                "correct": rate("correct", subset),
            }
    return {
        "n": total,
        "valid": rate("valid"),
        "schema_valid": rate("schema_valid"),
        "correct": rate("correct"),
        "taxonomy": taxonomy,
        "by_category": by_category,
    }


# --------------------------------------------------------------------------- mock oracle


class ToolcallOracle:
    """Lets the mock server answer Suite A requests from the cases' own ground truth."""

    def __init__(self, cases: list[dict[str, Any]]) -> None:
        self._index: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {
            case_key(c): c for c in cases
        }

    def lookup(self, body: dict[str, Any]) -> Scenario | None:
        tools = body.get("tools") or []
        messages = body.get("messages") or []
        users = [m for m in messages if m.get("role") == "user"]
        if not users or not tools:
            return None
        key = (users[-1].get("content"), tuple(sorted(t["function"]["name"] for t in tools)))
        case = self._index.get(key)
        return None if case is None else scenario_for(case)


def scenario_for(case: dict[str, Any]) -> Scenario:
    expected = case["expected"]["calls"]
    if not expected:
        reply = (
            "I don't have a function that fits that, but I can answer directly."
            if case["expected"].get("reason") == "irrelevant"
            else "I need a bit more information before I can do that. Could you provide the missing detail?"
        )
        return NoCallScenario(case["id"], reply, case["tools"])
    calls = [{"name": c["name"], "arguments": materialize(c["arguments"])} for c in expected]
    return CallScenario(case["id"], calls, case["tools"])
