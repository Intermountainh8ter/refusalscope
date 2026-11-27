import json

from refusalscope import rareflip


def test_degenerate_stubs_have_zero_power():
    r = rareflip.run(7, 5, [100, 300])["rare_flip"]["power"]
    for s in ("always_refuse", "always_comply"):
        assert all(v == 0 for v in r[s].values())


def test_deterministic_and_cli(tmp_path):
    assert rareflip.run(7, 5, [100]) == rareflip.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["rareflip", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "300" in d["rare_flip"]["power"]["rare_batch"]
