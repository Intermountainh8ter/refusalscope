import json

from refusalscope import orderdrift


def test_degenerate_stubs_have_zero_false_positives():
    r = orderdrift.run(7, 5, [100, 300])["order_drift"]["degenerate_false_positive_rate"]
    for d in orderdrift.DESIGNS:
        for s in ("always_refuse", "always_comply"):
            assert all(v == 0 for v in r[d][s].values())


def test_deterministic_and_cli(tmp_path):
    assert orderdrift.run(7, 5, [100]) == orderdrift.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["orderdrift", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "1000" in d["order_drift"]["false_positive_rate"]["fixed_order"]
