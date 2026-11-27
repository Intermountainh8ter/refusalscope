from refusalscope import adjpower


def test_adjudication_beats_raw_and_stubs_flat():
    adjpower.GRID[:] = [200, 600, 1500]
    r = adjpower.run(7, 30)["power"]
    assert r["adjudicated"]["600"] >= r["raw"]["600"]
    assert all(v < 0.8 for v in r["degenerate_stub_max_power"].values())


def test_cli_dispatch(tmp_path):
    from refusalscope.cli import main

    adjpower.GRID[:] = [200]
    out = tmp_path / "o.json"
    assert main(["adjpower", "--draws", "3", "--out", str(out)]) == 0
    assert "started_at" in out.read_text()
