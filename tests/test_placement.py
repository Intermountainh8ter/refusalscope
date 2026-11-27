import json

from refusalscope import placement


def test_run_is_deterministic():
    assert placement.run(3, 60, 50) == placement.run(3, 60, 50)


def test_controls_have_zero_spread():
    r = placement.run(7, 100, 100)["placement"]["per_stub"]
    assert r["always_wrong"]["spread"] == 0.0
    assert r["coin_flip"]["spread"] == 0.0


def test_positional_bias_has_positive_spread():
    r = placement.run(7, 200, 100)["placement"]
    assert r["per_stub"]["positional_bias"]["spread"] > 0
    assert r["spread_over_interval"]["always_wrong"] == 0.0


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-stub.json"
    assert placement.main(["--seed", "1", "--items", "50", "--n-boot", "50", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "positional_bias" in d["placement"]["spread_over_interval"]
