import json

from refusalscope import mdd


def test_mcnemar_exact_p():
    assert mdd.mcnemar_exact_p(0, 0) == 1.0
    assert mdd.mcnemar_exact_p(5, 5) == 1.0
    assert abs(mdd.mcnemar_exact_p(0, 10) - 2 / 1024) < 1e-12


def test_degenerate_stubs_have_zero_power():
    r = mdd.run(7, 20)["mdd"]
    for s in ("always_refuse", "always_comply"):
        assert r["required_prompts"][s] is None
        assert all(v == 0 for v in r["power"][s].values())


def test_deterministic_and_cli(tmp_path):
    assert mdd.run(7, 10) == mdd.run(7, 10)
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["mdd", "--seed", "7", "--draws", "10", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "required_prompts" in d["mdd"]
