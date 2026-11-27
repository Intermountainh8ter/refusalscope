import json

from refusalscope import singleshot


def test_stubs_agree_exactly():
    r = singleshot.run(7, 60, 31, 10)["sampling"]["stubs"]
    for c in ("always_refuse", "always_comply"):
        assert r[c]["single_shot_majority_agreement"] == 1.0


def test_bimodal_agreement_below_one():
    a = singleshot.run(7, 200, 31, 10)["sampling"]["single_shot_majority_agreement"]
    assert 0.5 < a < 1.0


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["singleshot", "--seed", "7", "--prompts", "50", "--studies", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert singleshot.run(7, 50, 31, 5) == singleshot.run(7, 50, 31, 5)
