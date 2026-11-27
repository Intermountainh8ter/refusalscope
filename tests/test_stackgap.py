import json

from refusalscope import stackgap


def test_fraction_in_range_and_controls():
    r = stackgap.run(7, 200, 400)["noise_floor"]
    assert 0.05 <= r["explained_fraction"]["template_shift"] <= 0.35
    assert r["explained_fraction"]["identical_stack"] == 1.0
    for c in ("always_refuse", "always_comply"):
        assert r["observed_gap"][c] == 0.0 and r["noise_gap"][c] == 0.0


def test_cli_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-stub.json"
    assert stackgap.main(["--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert stackgap.run(7, 50, 50) == stackgap.run(7, 50, 50)
