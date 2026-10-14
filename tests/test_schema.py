import pytest

from refusalscope.schema import (
    LENIENT,
    STRICT,
    first_issue_kind,
    is_matcher,
    materialize,
    norm_text,
    norm_verbatim,
    validate,
    values_match,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "price": {"type": "number"},
        "flag": {"type": "boolean"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {
            "type": "object",
            "properties": {"street": {"type": "string"}, "zip": {"type": "string"}},
            "required": ["street"],
        },
    },
    "required": ["city"],
}


def kinds(value, mode=STRICT):
    issues, _ = validate(SCHEMA, value, mode)
    return sorted(i.kind for i in issues)


# ------------------------------------------------------------------ normalisation


def test_norm_text_strict_and_lenient():
    assert norm_text("  New   YORK ") == "new york"
    assert norm_text("New-York!", LENIENT) == "new york"
    assert norm_text("Hello, World.", LENIENT) == "hello world"
    assert norm_text("a.b") == "a.b"


def test_norm_verbatim_only_forgives_newline_style_and_outer_whitespace():
    assert norm_verbatim("  a\r\nb\r\n") == "a\nb"
    assert norm_verbatim("a  b") == "a  b"


# ------------------------------------------------------------------ validation


def test_valid_arguments_pass():
    ok = {"city": "Paris", "days": 3, "price": 2.5, "flag": True, "units": "celsius"}
    assert kinds(ok) == []


def test_missing_required_and_unknown_argument():
    assert kinds({}) == ["missing_required_arg"]
    assert kinds({"city": "x", "bogus": 1}) == ["unknown_argument"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("city", 5),
        ("days", "3"),
        ("days", 2.5),
        ("days", True),
        ("price", "9.5"),
        ("price", False),
        ("flag", "true"),
        ("flag", 1),
        ("tags", "a,b"),
        ("address", "home"),
    ],
)
def test_strict_type_mismatches(field, value):
    assert kinds({"city": "x", field: value}) == ["type_mismatch"]


def test_integer_valued_float_is_a_valid_integer():
    issues, coerced = validate(SCHEMA, {"city": "x", "days": 3.0})
    assert not issues and coerced["days"] == 3 and isinstance(coerced["days"], int)


def test_enum_is_case_sensitive_when_strict_and_forgiving_when_lenient():
    assert kinds({"city": "x", "units": "Celsius"}) == ["enum_violation"]
    issues, coerced = validate(SCHEMA, {"city": "x", "units": "Celsius"}, LENIENT)
    assert not issues and coerced["units"] == "celsius"
    assert kinds({"city": "x", "units": "kelvin"}, LENIENT) == ["enum_violation"]


def test_lenient_coerces_numeric_and_boolean_strings():
    value = {"city": "x", "days": "3", "price": "9.5", "flag": "TRUE"}
    issues, coerced = validate(SCHEMA, value, LENIENT)
    assert not issues
    assert coerced["days"] == 3 and coerced["price"] == 9.5 and coerced["flag"] is True
    assert kinds({"city": "x", "days": "three"}, LENIENT) == ["type_mismatch"]
    assert kinds({"city": "x", "days": "3.5"}, LENIENT) == ["type_mismatch"]
    assert kinds({"city": "x", "days": "4.0"}, LENIENT) == []
    assert kinds({"city": "x", "price": "cheap"}, LENIENT) == ["type_mismatch"]
    assert kinds({"city": "x", "flag": "maybe"}, LENIENT) == ["type_mismatch"]


def test_nested_object_and_array_validation():
    assert kinds({"city": "x", "address": {}}) == ["missing_required_arg"]
    assert kinds({"city": "x", "address": {"street": "s", "po": 1}}) == ["unknown_argument"]
    assert kinds({"city": "x", "address": {"street": 5}}) == ["type_mismatch"]
    assert kinds({"city": "x", "tags": ["a", 2]}) == ["type_mismatch"]
    issues, _ = validate(SCHEMA, {"city": "x", "tags": ["a", 2]})
    assert issues[0].path == "tags[1]"


def test_schema_without_type_accepts_anything_and_null_type():
    assert validate({}, {"a": 1})[0] == []
    assert validate({"type": "null"}, None)[0] == []
    assert validate({"type": "null"}, 0)[0][0].kind == "type_mismatch"
    assert validate({"enum": [1, 2]}, True)[0][0].kind == "enum_violation"
    assert validate({"enum": [1, 2]}, 2)[0] == []


def test_additional_properties_true_allows_extras():
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    assert validate(schema, {"anything": 1})[0] == []


def test_first_issue_kind_precedence():
    issues, _ = validate(SCHEMA, {"bogus": 1, "days": "x"})
    assert first_issue_kind(issues) == "missing_required_arg"
    assert first_issue_kind([]) is None
    issues, _ = validate(SCHEMA, {"city": "x", "bogus": 1, "days": "x"})
    assert first_issue_kind(issues) == "unknown_argument"


# ------------------------------------------------------------------ matching


def test_values_match_basic_types():
    assert values_match("Paris", " paris ")
    assert not values_match("Paris", "Lyon")
    assert values_match(3, 3.0)
    assert not values_match(3, "3")
    assert not values_match(True, 1)
    assert values_match(True, True)
    assert values_match(None, None)
    assert not values_match(None, 0)
    assert values_match(1.0000000001, 1.0)
    assert not values_match("a", 5)


def test_values_match_lists_and_dicts():
    assert values_match([1, "A"], [1, "a"])
    assert not values_match([1, 2], [2, 1])
    assert not values_match([1], [1, 2])
    assert values_match({"a": 1, "b": "X"}, {"b": "x", "a": 1})
    assert not values_match({"a": 1}, {"a": 1, "b": 2})
    assert not values_match({"a": 1}, [1])
    assert not values_match([1], {"a": 1})


def test_lenient_string_matching_ignores_punctuation():
    assert values_match("New York", "new-york", LENIENT)
    assert not values_match("New York", "new-york", STRICT)


def test_matcher_any_of_and_contains():
    assert values_match({"$any_of": ["NYC", "New York"]}, "new york")
    assert not values_match({"$any_of": ["NYC"]}, "Boston")
    assert values_match({"$contains": "report"}, "Final REPORT v2")
    assert not values_match({"$contains": "x"}, 5)


def test_matcher_verbatim_is_exact_apart_from_newline_style():
    expected = {"$verbatim": 'She said "hi"\nbye'}
    assert values_match(expected, 'She said "hi"\r\nbye ')
    assert not values_match(expected, 'She said "HI"\nbye')
    assert not values_match(expected, 'She said "hi"  bye')
    assert not values_match(expected, 3)


def test_matcher_compact_ignores_all_whitespace_and_case():
    assert values_match({"$compact": "(12 + 7) * 3"}, "(12+7)*3")
    assert not values_match({"$compact": "1+2"}, "1+3")
    assert not values_match({"$compact": "1+2"}, 12)


def test_matcher_unordered_handles_strings_and_objects():
    assert values_match({"$unordered": ["a", "b", "c"]}, ["c", "A", "b"])
    assert not values_match({"$unordered": ["a", "b"]}, ["a", "a"])
    assert not values_match({"$unordered": ["a"]}, ["a", "b"])
    assert not values_match({"$unordered": ["a"]}, "a")
    items = {"$unordered": [{"sku": "A", "q": 1}, {"sku": "B", "q": 2}]}
    assert values_match(items, [{"sku": "B", "q": 2}, {"sku": "A", "q": 1}])


def test_matcher_approx():
    assert values_match({"$approx": 3.14, "tol": 0.01}, 3.145)
    assert not values_match({"$approx": 3.14, "tol": 0.001}, 3.15)
    assert not values_match({"$approx": 3.14}, "3.14")


def test_is_matcher_and_materialize():
    assert is_matcher({"$any_of": [1]})
    assert not is_matcher({"a": 1})
    assert not is_matcher([1])
    nested = {
        "a": {"$any_of": ["x", "y"]},
        "b": {"$verbatim": "v"},
        "c": {"$compact": "1 + 2"},
        "d": {"$unordered": [{"$any_of": [1]}, 2]},
        "e": {"$approx": 1.5, "tol": 0.1},
        "f": {"$contains": "z"},
        "g": [1, {"$verbatim": "w"}],
        "h": 5,
    }
    assert materialize(nested) == {
        "a": "x",
        "b": "v",
        "c": "1 + 2",
        "d": [1, 2],
        "e": 1.5,
        "f": "z",
        "g": [1, "w"],
        "h": 5,
    }
