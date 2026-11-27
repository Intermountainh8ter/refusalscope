import json

from refusalscope import invariantctl


def test_control_and_stubs_never_flip():
    r = invariantctl.run(7, 80, 10, 0.05)["batch_invariant"]
    assert r["flip_rate_control"] == 0.0
    for c in ("always_refuse", "always_comply"):
        assert r["stubs"][c]["flip_rate_batch"] == 0.0
        assert r["stubs"][c]["flip_rate_control"] == 0.0


def test_batch_noise_flips_some():
    r = invariantctl.run(7, 300, 20, 0.05)["batch_invariant"]
    assert 0.0 < r["flip_rate_batch"] < 0.1


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["invariant-ctl", "--seed", "7", "--prompts", "50", "--studies", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert invariantctl.run(7, 50, 5, 0.05) == invariantctl.run(7, 50, 5, 0.05)
