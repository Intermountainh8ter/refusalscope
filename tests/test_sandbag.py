import json

from refusalscope import sandbag


def test_constant_stubs_zero_gap():
    r = sandbag.run(7, 50, 2, 10, 5)["sandbag"]
    for c in ("always_correct", "always_wrong"):
        assert r["stubs"][c]["accuracy_gap_evaluation_framed"] == 0.0
        assert r["stubs"][c]["accuracy_gap_cue_removed"] == 0.0


def test_eval_gap_positive():
    r = sandbag.run(7, 200, 4, 50, 10)["sandbag"]
    assert r["accuracy_gap_evaluation_framed"] > 0


def test_deterministic():
    assert sandbag.run(7, 20, 2, 10, 3) == sandbag.run(7, 20, 2, 10, 3)


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["sandbag", "--seed", "7", "--replicates", "3", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True
