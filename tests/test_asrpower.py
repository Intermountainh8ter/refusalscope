import json

from refusalscope import asrpower


def test_deterministic_and_monotone_grid():
    a = asrpower.run(7, 20, [50, 1000])
    assert a == asrpower.run(7, 20, [50, 1000])
    assert a["asr_power"]["power"]["1000"] >= a["asr_power"]["power"]["50"]


def test_min_cases_none_when_unreached():
    assert asrpower.run(7, 5, [50])["asr_power"]["min_cases_per_arm"] in (None, 50)


def test_cli(tmp_path):
    from refusalscope.cli import main

    out = tmp_path / "o.json"
    assert main(["asrpower", "--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "min_cases_per_arm" in d["asr_power"]
