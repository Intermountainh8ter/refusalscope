import json

from refusalscope import floorci


def test_wilson_bounds():
    lo, hi = floorci.wilson(0, 100)
    assert lo == 0 or lo < 1e-9
    assert 0 < hi < 0.1


def test_degenerate_stubs_have_zero_floor():
    r = floorci.run(7, 5, [30, 100])["floor_ci"]["degenerate_max_floor"]
    for s in ("always_refuse", "always_comply"):
        assert all(v == 0 for v in r[s].values())


def test_deterministic_and_cli(tmp_path):
    assert floorci.run(7, 5, [100]) == floorci.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["floorci", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "100" in d["floor_ci"]["coverage"]["identical_stack"]
