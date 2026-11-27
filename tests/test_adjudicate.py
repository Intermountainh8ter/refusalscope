import json

from refusalscope import adjudicate


def test_degenerate_stubs_have_no_flips_and_null_ratio():
    r = adjudicate.run(7, 3, 200)["adjudication"]
    for s in ("always_refuse", "always_comply"):
        assert r["flips"][s]["total_raw"] == 0 and r["flips"][s]["total_genuine"] == 0
        assert r["raw_to_genuine_ratio"][s] is None


def test_genuine_never_exceeds_raw_and_deterministic():
    a = adjudicate.run(7, 3, 200)
    assert a == adjudicate.run(7, 3, 200)
    for v in a["adjudication"]["flips"].values():
        assert v["total_genuine"] <= v["total_raw"]


def test_cli(tmp_path):
    from refusalscope.cli import main

    out = tmp_path / "o.json"
    assert main(["adjudicate", "--seed", "7", "--draws", "3", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "rare_batch" in d["adjudication"]["raw_to_genuine_ratio"]
