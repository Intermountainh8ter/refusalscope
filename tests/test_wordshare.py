import json

from refusalscope import wordshare


def test_shares_in_range_and_controls():
    r = wordshare.run(7, 200, 400)["variance_share"]
    assert r["wording"]["factorial_stub"] > 0.70
    assert 0.60 <= r["wording_minus_serving"]["factorial_stub"] <= 0.90
    for c in ("always_refuse", "always_comply"):
        assert r["total_variance"][c] == 0.0
        assert r["wording_minus_serving"][c] is None
        for f in wordshare.FACTORS:
            assert r[f][c] is None


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["word-share", "--seed", "7", "--items", "50", "--draws", "20", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert wordshare.run(7, 50, 20) == wordshare.run(7, 50, 20)
