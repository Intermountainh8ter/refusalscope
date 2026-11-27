import json

from refusalscope import stacksources


def test_degenerate_stubs_have_zero_rates():
    r = stacksources.run(7, 5, [100, 300])["stacked_sources"]["degenerate_rate"]
    for m in stacksources.MODES:
        for s in ("always_refuse", "always_comply"):
            assert all(v == 0 for v in r[m][s].values())


def test_deterministic_and_cli(tmp_path):
    assert stacksources.run(7, 5, [100]) == stacksources.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["stacksources", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "1000" in d["stacked_sources"]["power"]["stacked"]["shift_0.04"]
