import json

from refusalscope import batchnoise


def test_stubs_never_flip_and_never_detect():
    r = batchnoise.run(7, 60, 20, 0.05)["batch_noise"]["stubs"]
    for c in ("always_refuse", "always_comply"):
        assert r[c]["mean_batch_only_flips_per_study"] == 0.0
        assert r[c]["min_detectable_step"] is None


def test_noisy_stub_flips_and_zero_noise_does_not():
    assert batchnoise.run(7, 200, 20, 0.05)["batch_noise"]["mean_batch_only_flips_per_study"] > 0
    assert batchnoise.run(7, 200, 20, 0.0)["batch_noise"]["mean_batch_only_flips_per_study"] == 0


def test_cli_writes_provenance(tmp_path):
    from refusalscope import cli

    out = tmp_path / "x-flip-floor-stub.json"
    assert cli.main(["batch-noise", "--seed", "7", "--prompts", "50", "--studies", "10", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and d["mock"] is True


def test_deterministic():
    assert batchnoise.run(7, 50, 10, 0.05) == batchnoise.run(7, 50, 10, 0.05)
