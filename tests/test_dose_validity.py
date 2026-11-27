import json

from refusalscope import dose_validity as dv


def test_slope_in_range_and_floors_zero():
    r = dv.run(7, 400, 200)["dose_response"]
    assert -0.36 <= r["tool_call_validity_slope"] <= -0.24
    assert r["slope_bootstrap_95"][1] < 0
    assert all(s == 0.0 for s in r["floor_slopes"].values())
    for k in ("always_refuse", "never_call"):
        assert all(v == 0.0 for v in r["validity_by_dose"][k])


def test_cli_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-mock.json"
    assert dv.main(["--seed", "7", "--n-boot", "50", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert dv.run(7, 50, 20) == dv.run(7, 50, 20)
