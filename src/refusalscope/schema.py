"""A small JSON-Schema subset validator and an argument value matcher.

Only what the tool-calling and injection suites need is implemented: ``type``, ``enum``,
``properties``, ``required``, ``items`` and rejection of undeclared properties. The matcher
compares an expected value, which may contain matcher objects, with an actual value.

Matcher objects (a dict with a single ``$``-prefixed key) allowed in expected values:

``{"$any_of": [v, ...]}``       any alternative matches
``{"$verbatim": "text"}``       exact text; only newline style and outer whitespace are forgiven
``{"$compact": "text"}``        equal after removing all whitespace, case-insensitive
``{"$unordered": [v, ...]}``    list equal as a multiset
``{"$approx": x, "tol": t}``    numeric within a tolerance
``{"$contains": "text"}``       actual string contains the text, case-insensitive
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import permutations
from typing import Any

STRICT = "strict"
LENIENT = "lenient"
MODES = (STRICT, LENIENT)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SchemaIssue:
    """One problem found while validating arguments against a function schema."""

    kind: str  # missing_required_arg | unknown_argument | type_mismatch | enum_violation
    path: str
    detail: str


# --------------------------------------------------------------------------- normalisation


def norm_text(value: str, mode: str = STRICT) -> str:
    """Case-fold and collapse whitespace. Lenient mode also drops punctuation."""
    text = _WS.sub(" ", value).strip().casefold()
    if mode == LENIENT:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = _WS.sub(" ", text).strip()
    return text


def norm_verbatim(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


MATCHER_KEYS = frozenset({"$any_of", "$verbatim", "$compact", "$unordered", "$approx", "$contains"})


def is_matcher(value: Any) -> bool:
    return isinstance(value, dict) and any(key in MATCHER_KEYS for key in value)


def materialize(expected: Any) -> Any:
    """Turn an expected value with matchers into one concrete value (first alternative)."""
    if is_matcher(expected):
        if "$any_of" in expected:
            return materialize(expected["$any_of"][0])
        if "$verbatim" in expected:
            return expected["$verbatim"]
        if "$compact" in expected:
            return expected["$compact"]
        if "$unordered" in expected:
            return [materialize(v) for v in expected["$unordered"]]
        if "$approx" in expected:
            return expected["$approx"]
        if "$contains" in expected:
            return expected["$contains"]
    if isinstance(expected, dict):
        return {k: materialize(v) for k, v in expected.items()}
    if isinstance(expected, list):
        return [materialize(v) for v in expected]
    return expected


# --------------------------------------------------------------------------- matching


def values_match(expected: Any, actual: Any, mode: str = STRICT) -> bool:
    """Whether ``actual`` satisfies ``expected`` after normalisation."""
    if is_matcher(expected):
        return _matcher_match(expected, actual, mode)
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if _is_number(expected):
        return _is_number(actual) and math.isclose(
            float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and norm_text(expected, mode) == norm_text(actual, mode)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(values_match(e, a, mode) for e, a in zip(expected, actual))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(values_match(expected[k], actual[k], mode) for k in expected)
        )
    return expected == actual


def _matcher_match(matcher: dict[str, Any], actual: Any, mode: str) -> bool:
    if "$any_of" in matcher:
        return any(values_match(option, actual, mode) for option in matcher["$any_of"])
    if "$verbatim" in matcher:
        return isinstance(actual, str) and norm_verbatim(actual) == norm_verbatim(
            matcher["$verbatim"]
        )
    if "$compact" in matcher:
        return (
            isinstance(actual, str)
            and _WS.sub("", actual).casefold() == _WS.sub("", matcher["$compact"]).casefold()
        )
    if "$unordered" in matcher:
        want = matcher["$unordered"]
        if not isinstance(actual, list) or len(actual) != len(want):
            return False
        return any(
            all(values_match(w, a, mode) for w, a in zip(want, ordering))
            for ordering in permutations(actual)
        )
    if "$approx" in matcher:
        return _is_number(actual) and abs(float(actual) - float(matcher["$approx"])) <= float(
            matcher.get("tol", 1e-6)
        )
    if "$contains" in matcher:
        return isinstance(actual, str) and matcher["$contains"].casefold() in actual.casefold()
    return False  # pragma: no cover - guarded by is_matcher


# --------------------------------------------------------------------------- validation


def _type_ok(kind: str, value: Any, mode: str) -> tuple[bool, Any]:
    """Return ``(ok, coerced_value)``; coercion only happens in lenient mode."""
    if kind == "string":
        return isinstance(value, str), value
    if kind == "integer":
        if isinstance(value, bool):
            return False, value
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if mode == LENIENT and isinstance(value, str):
            try:
                return True, int(value.strip())
            except ValueError:
                try:
                    as_float = float(value.strip())
                except ValueError:
                    return False, value
                return (True, int(as_float)) if as_float.is_integer() else (False, value)
        return False, value
    if kind == "number":
        if _is_number(value):
            return True, value
        if mode == LENIENT and isinstance(value, str):
            try:
                return True, float(value.strip())
            except ValueError:
                return False, value
        return False, value
    if kind == "boolean":
        if isinstance(value, bool):
            return True, value
        if (
            mode == LENIENT
            and isinstance(value, str)
            and value.strip().lower() in ("true", "false")
        ):
            return True, value.strip().lower() == "true"
        return False, value
    if kind == "array":
        return isinstance(value, list), value
    if kind == "object":
        return isinstance(value, dict), value
    if kind == "null":
        return value is None, value
    return True, value


def validate(
    schema: dict[str, Any], value: Any, mode: str = STRICT, path: str = ""
) -> tuple[list[SchemaIssue], Any]:
    """Validate ``value`` against ``schema``; return the issues and the (coerced) value."""
    issues: list[SchemaIssue] = []
    kind = schema.get("type")
    coerced = value
    if kind is not None:
        ok, coerced = _type_ok(kind, value, mode)
        if not ok:
            issues.append(
                SchemaIssue(
                    "type_mismatch", path or "$", f"expected {kind}, got {_describe(value)}"
                )
            )
            return issues, value
    if "enum" in schema:
        options = schema["enum"]
        if not _enum_member(options, coerced, mode):
            issues.append(
                SchemaIssue("enum_violation", path or "$", f"{coerced!r} is not one of {options}")
            )
        elif mode == LENIENT and isinstance(coerced, str):
            coerced = next(
                o for o in options if isinstance(o, str) and o.casefold() == coerced.casefold()
            )
    if kind == "object" and isinstance(coerced, dict):
        issues_obj, coerced = _validate_object(schema, coerced, mode, path)
        issues.extend(issues_obj)
    elif kind == "array" and isinstance(coerced, list) and "items" in schema:
        items: list[Any] = []
        for index, item in enumerate(coerced):
            sub_issues, sub_value = validate(schema["items"], item, mode, f"{path}[{index}]")
            issues.extend(sub_issues)
            items.append(sub_value)
        coerced = items
    return issues, coerced


def _enum_member(options: list[Any], value: Any, mode: str) -> bool:
    for option in options:
        if isinstance(option, str) and isinstance(value, str):
            if option == value or (mode == LENIENT and option.casefold() == value.casefold()):
                return True
        elif type(option) is type(value) and option == value:
            return True
    return False


def _validate_object(
    schema: dict[str, Any], value: dict[str, Any], mode: str, path: str
) -> tuple[list[SchemaIssue], dict[str, Any]]:
    issues: list[SchemaIssue] = []
    properties = schema.get("properties", {})
    here = path or "$"
    issues.extend(
        SchemaIssue("missing_required_arg", f"{path}.{name}".lstrip("."), "missing")
        for name in schema.get("required", [])
        if name not in value
    )
    coerced: dict[str, Any] = {}
    for name, item in value.items():
        child_path = f"{path}.{name}".lstrip(".")
        if name not in properties:
            if schema.get("additionalProperties", False) is not True:
                issues.append(
                    SchemaIssue(
                        "unknown_argument", child_path, f"{name!r} is not declared in {here}"
                    )
                )
            coerced[name] = item
            continue
        sub_issues, sub_value = validate(properties[name], item, mode, child_path)
        issues.extend(sub_issues)
        coerced[name] = sub_value
    return issues, coerced


def _describe(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null" if value is None else type(value).__name__


ISSUE_ORDER = ("missing_required_arg", "unknown_argument", "type_mismatch", "enum_violation")


def first_issue_kind(issues: list[SchemaIssue]) -> str | None:
    """The most fundamental problem among ``issues`` according to :data:`ISSUE_ORDER`."""
    kinds = {issue.kind for issue in issues}
    for kind in ISSUE_ORDER:
        if kind in kinds:
            return kind
    return None
