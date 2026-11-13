from refusalscope import dose_floor


def test_suppression_is_monotone_and_half_at_midpoint():
    vals = [dose_floor.suppression(d) for d in dose_floor.GRID]
    assert vals == sorted(vals)
    assert abs(dose_floor.suppression(0.5) - 0.5) < 1e-12


def test_dose_zero_repeat_has_no_flips():
    out = dose_floor.run(7, 80, 100)
    assert out["dose_0_repeat"]["flip_rate"] == 0.0
    assert out["dose_response"]["per_dose"]["0.0"]["flip_rate"] == 0.0


def test_run_is_deterministic():
    assert dose_floor.run(3, 80, 100) == dose_floor.run(3, 80, 100)


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "x-flip-floor-mock.json"
    assert dose_floor.main(["--seed", "1", "--draws", "80", "--n-boot", "100", "--out", str(out)]) == 0
    import json

    d = json.loads(out.read_text())
    assert d["provenance"]["started_at"] and "min_dose_above_floor" in d["dose_response"]
