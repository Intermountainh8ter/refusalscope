import json

from refusalscope import controlfloor


def test_zero_stub_exact():
    s = controlfloor.run(7, 16, 5)["control_floor"]["stubs"]["zero_effect"]
    assert s["designed_effect"] == 0.0 and s["random_effect"] == 0.0
    assert s["random_reaches_designed_fraction"] is None


def test_fraction_in_range():
    f = controlfloor.run(7, 64, 400)["control_floor"]["random_reaches_designed_fraction"]
    assert 0.005 <= f <= 0.03


def test_deterministic():
    assert controlfloor.run(7, 8, 3) == controlfloor.run(7, 8, 3)


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["controlfloor", "--seed", "7", "--replicates", "3", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True
