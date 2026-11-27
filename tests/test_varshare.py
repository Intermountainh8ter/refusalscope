import json

from refusalscope import varshare


def test_shares_in_range_and_controls():
    r = varshare.run(7, 200, 400)["variance_share"]
    assert 0.70 <= r["template"]["factorial_stub"] <= 0.95
    assert r["cache"]["factorial_stub"] < 0.05
    for c in ("always_refuse", "always_comply"):
        assert r["total_variance"][c] == 0.0
        for f in varshare.FACTORS:
            assert r[f][c] is None


def test_cli_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-stub.json"
    assert varshare.main(["--seed", "7", "--items", "50", "--draws", "20", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert varshare.run(7, 50, 20) == varshare.run(7, 50, 20)
