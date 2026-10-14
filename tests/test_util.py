import json

import pytest

from refusalscope.util import (
    canonical_json,
    percent,
    read_jsonl,
    sha256_file,
    sha256_text,
    stable_hash,
    stable_rng,
    strip_think,
    utc_now_iso,
    write_jsonl,
)


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert canonical_json({"é": 1}) == '{"é":1}'


def test_sha256_known_vector():
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_file_matches_text(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_text("abc")


def test_stable_hash_and_rng_are_reproducible():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash("a", 2)
    assert stable_rng("x", 3).random() == stable_rng("x", 3).random()
    assert stable_rng("x", 3).random() != stable_rng("x", 4).random()


def test_utc_now_iso_shape():
    stamp = utc_now_iso()
    assert stamp.endswith("Z") and "T" in stamp


def test_jsonl_roundtrip(tmp_path):
    rows = [{"b": 2, "a": "é"}, {"c": [1, 2, 3]}]
    path = tmp_path / "sub" / "rows.jsonl"
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == '{"a": "é", "b": 2}'


def test_read_jsonl_skips_blank_lines_and_rejects_bad_rows(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": 2}]
    path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        list(read_jsonl(path))
    path.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(read_jsonl(path))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("plain", "plain"),
        ("<think>hidden</think>answer", "answer"),
        ("<think>\nline1\nline2\n</think>\n\nfinal", "final"),
        ("<THINK>x</THINK> ok", "ok"),
        ("<think>never closed because max tokens", ""),
        ("before <think>unfinished", "before"),
        ("reasoning only</think>answer", "answer"),
    ],
)
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


def test_percent():
    assert percent(1, 4) == 25.0
    assert percent(1, 0) == 0.0


def test_write_jsonl_is_valid_json_per_line(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [{"k": i} for i in range(5)])
    assert [json.loads(line)["k"] for line in path.read_text().splitlines()] == list(range(5))
