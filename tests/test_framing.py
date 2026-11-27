import json

from refusalscope import framing


def test_controls_flip_zero_and_keyed_in_range(tmp_path):
    r = framing.run(7, 200, 400)["framing"]
    for c in ("always_comply", "always_refuse", "coin_flip"):
        assert r["paired_flip_rate"][c] == 0.0
    assert 0.08 <= r["paired_flip_rate"]["wording_keyed"] <= 0.25
    assert r["bootstrap_95"]["wording_keyed"][0] > 0


def test_cli_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-stub.json"
    assert framing.main(["--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True
    assert d == json.loads(out.read_text())


def test_deterministic():
    assert framing.run(7, 50, 50) == framing.run(7, 50, 50)
