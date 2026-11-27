import json

from refusalscope import dosetemplate


def test_deterministic_and_stubs_never_detected():
    g = [0.0, 0.5, 1.0]
    a = dosetemplate.run(7, 20, g)
    assert a == dosetemplate.run(7, 20, g)
    assert a["template"]["degenerate_stub_detected_strength"] == {"always_refuse": None, "always_comply": None}


def test_mismatch_never_detects_later_than_matched():
    r = dosetemplate.run(7, 30, [0.0, 0.2, 0.5, 1.0])["template"]
    if r["min_strength_matched"] is not None:
        assert r["min_strength_mismatch"] is not None
        assert r["min_strength_mismatch"] <= r["min_strength_matched"]


def test_cli(tmp_path):
    from refusalscope.cli import main

    out = tmp_path / "o.json"
    assert main(["dosetemplate", "--seed", "7", "--draws", "10", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "min_strength_mismatch" in d["template"]
