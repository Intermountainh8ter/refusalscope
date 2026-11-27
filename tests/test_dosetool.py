import json

from refusalscope import dosetool


def test_deterministic_and_stubs_never_detected():
    a = dosetool.run(7, 20, [0.0, 0.5, 1.0])
    assert a == dosetool.run(7, 20, [0.0, 0.5, 1.0])
    assert a["dose_response"]["degenerate_stub_detected_strength"] == {"always_refuse": None, "always_comply": None}


def test_true_crossing_and_text_refusal():
    assert dosetool.tool_success(0.0) == dosetool.BASE_TOOL
    assert abs(dosetool.text_refusal(1.0) - 0.10) < 1e-12
    r = dosetool.run(7, 20, [0.0, 0.35, 1.0])
    assert r["dose_response"]["true_loss_crossing_strength"] == 0.35


def test_cli(tmp_path):
    from refusalscope.cli import main

    out = tmp_path / "o.json"
    assert main(["dosetool", "--seed", "7", "--draws", "10", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "benign_tool_loss_min_strength" in d["dose_response"]
