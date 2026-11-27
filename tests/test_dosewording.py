import json

from refusalscope import dosewording


def test_deterministic_and_stubs_never_detected():
    g = [0.0, 0.5, 1.0]
    a = dosewording.run(7, 20, g)
    assert a == dosewording.run(7, 20, g)
    assert a["dose_response"]["degenerate_stub_detected_strength"] == {"always_refuse": None, "always_comply": None}


def test_floor_never_detects_earlier():
    r = dosewording.run(7, 30, [0.0, 0.2, 0.5, 1.0])["dose_response"]
    if r["min_strength_wordingfree"] is None:
        assert r["min_strength_wording_floor"] is None
    elif r["min_strength_wording_floor"] is not None:
        assert r["min_strength_wording_floor"] >= r["min_strength_wordingfree"]


def test_wording_floor_value():
    assert abs(dosewording.wording_floor(5, 0.04) - 0.04) < 1e-9


def test_cli(tmp_path):
    from refusalscope.cli import main

    out = tmp_path / "o.json"
    assert main(["dose-wording", "--seed", "7", "--studies", "10", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "min_strength_wording_floor" in d["dose_response"]
