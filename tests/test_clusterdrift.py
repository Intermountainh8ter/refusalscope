import json

from refusalscope import clusterdrift


def test_degenerate_stubs_have_zero_false_positives():
    r = clusterdrift.run(7, 5, [100, 300])["cluster_drift"]["degenerate_false_positive_rate"]
    for c in clusterdrift.CLUSTERS:
        for s in ("always_refuse", "always_comply"):
            assert all(v == 0 for v in r[f"cluster_{c}"][s].values())


def test_deterministic_and_cli(tmp_path):
    assert clusterdrift.run(7, 5, [100]) == clusterdrift.run(7, 5, [100])
    out = tmp_path / "o.json"
    from refusalscope.cli import main

    assert main(["clusterdrift", "--seed", "7", "--draws", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "1000" in d["cluster_drift"]["false_positive_rate"]["cluster_20"]
