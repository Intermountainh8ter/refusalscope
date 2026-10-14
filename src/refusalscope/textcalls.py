"""Detect tool calls that a model wrote as plain text instead of structured ``tool_calls``.

This is the signature of a serving problem (for example ``llama-server`` started without
``--jinja``, or a chat template without tool support). It is separated from the model's own
tool-calling ability so the two are never confused.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from refusalscope.client import ToolCall, parse_arguments

_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(
    r"```(?:json|tool_call|tool_code|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
_PY_CALL = re.compile(r"^\s*\[?\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*\]?\s*$", re.DOTALL)


def _balanced_objects(text: str) -> list[str]:
    """Substrings that look like top-level ``{...}`` objects (string-aware brace matching)."""
    found: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                found.append(text[start : index + 1])
    return found


def _as_call(obj: Any, index: int) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    function = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = function.get("name") or function.get("tool") or function.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    for key in ("arguments", "parameters", "args", "input"):
        if key in function:
            raw = function[key]
            break
    else:
        return None
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    arguments, error = parse_arguments(raw_text)
    return ToolCall(
        id=f"text_{index}",
        name=name,
        arguments_raw=raw_text,
        arguments=arguments,
        parse_error=error,
    )


def _python_style(text: str, names: set[str], index: int) -> ToolCall | None:
    match = _PY_CALL.match(text.strip().strip("`"))
    if not match or match.group(1) not in names:
        return None
    try:
        node = ast.parse(f"f({match.group(2)})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or node.args:
        return None
    try:
        arguments = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    except (ValueError, SyntaxError):
        return None
    return ToolCall(
        id=f"text_{index}",
        name=match.group(1),
        arguments_raw=json.dumps(arguments, ensure_ascii=False),
        arguments=arguments,
        parse_error=None,
    )


def detect_text_calls(content: str | None, tool_names: set[str] | None = None) -> list[ToolCall]:
    """Return tool calls written as text inside ``content`` (empty if none are found).

    Recognised shapes: ``<tool_call>{...}</tool_call>`` blocks, fenced or bare JSON objects with
    ``name`` plus ``arguments``/``parameters``, and Python-style ``name(key=value)`` calls to a
    function in ``tool_names``.
    """
    if not content:
        return []
    names = tool_names or set()
    calls: list[ToolCall] = []

    def add(call: ToolCall | None) -> None:
        if call is not None:
            calls.append(call)

    tagged = _TAG.findall(content)
    for chunk in tagged:
        decoded = None
        try:
            decoded = json.loads(chunk.strip())
        except json.JSONDecodeError:
            for candidate in _balanced_objects(chunk):
                try:
                    decoded = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
        items = decoded if isinstance(decoded, list) else [decoded]
        for item in items:
            add(_as_call(item, len(calls)))
    if calls:
        return calls
    sources = [m.strip() for m in _FENCE.findall(content)] or [content]
    for source in sources:
        for candidate in _balanced_objects(source):
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            add(_as_call(decoded, len(calls)))
        if names:
            for line in source.splitlines():
                add(_python_style(line, names, len(calls)))
    return calls
