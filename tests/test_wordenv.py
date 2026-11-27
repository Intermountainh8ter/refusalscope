import json

from refusalscope import wordenv


def test_stubs_zero_variance_null_shares():
    r = wordenv.run(7, 6, 8, 100, 5)["wordenv"]
    for c in ("always_refuse", "always_comply"):
        assert r["stubs"][c]["total_ss"] == 0.0
        assert r["stubs"][c]["wording_share"] is None
        assert r["stubs"][c]["environment_share"] is None


def test_shares_in_unit_interval():
    r = wordenv.run(7, 6, 8, 100, 20)["wordenv"]
    assert 0 < r["wording_share"] < 1
    assert 0 < r["environment_share"] < 1
    assert r["wording_share"] + r["environment_share"] <= 1.0 + 1e-9


def test_deterministic():
    assert wordenv.run(7, 4, 4, 50, 5) == wordenv.run(7, 4, 4, 50, 5)


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["wordenv", "--seed", "7", "--replicates", "3", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True
