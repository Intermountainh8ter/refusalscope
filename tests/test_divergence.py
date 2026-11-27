import json

from refusalscope import divergence


def test_degenerate_stubs_have_zero_rates():
    r = divergence.run(7, 5, [100])["divergence"]["degenerate_rate"]
    for s in ("always_refuse", "always_comply"):
        for rho in r[s].values():
            for cell in rho.values():
                assert all(v == 0 for v in cell.values())


def test_deterministic_and_cli(tmp_path):
    assert divergence.run(7, 5, [100]) == divergence.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["divergence", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "300" in d["divergence"]["power"]["rho_0.3"]["divergence_0.15"]
