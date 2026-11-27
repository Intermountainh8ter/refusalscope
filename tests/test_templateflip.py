import json

from refusalscope import templateflip


def test_stubs_never_flip_and_ratio_null():
    r = templateflip.run(7, 80, 10, 0.2)["template"]
    for c in ("always_refuse", "always_comply"):
        assert r["stubs"][c]["flip_rate_template"] == 0.0
        assert r["stubs"][c]["flip_rate_sampling"] == 0.0
        assert r["stubs"][c]["flip_rate_vs_sampling_ratio"] is None


def test_shift_flips_some():
    r = templateflip.run(7, 300, 20, 0.2)["template"]
    assert 0.0 < r["flip_rate_template"] < 0.2
    assert r["flip_rate_vs_sampling_ratio"] > 0


def test_zero_shift_no_template_flips():
    assert templateflip.run(7, 100, 5, 0.0)["template"]["flip_rate_template"] == 0.0


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["templateflip", "--seed", "7", "--prompts", "50", "--studies", "5", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert templateflip.run(7, 50, 5, 0.2) == templateflip.run(7, 50, 5, 0.2)
