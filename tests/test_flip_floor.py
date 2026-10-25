import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "flip_floor.py"
spec = importlib.util.spec_from_file_location("flip_floor", SCRIPT)
flip_floor = importlib.util.module_from_spec(spec)
sys.modules["flip_floor"] = flip_floor
spec.loader.exec_module(flip_floor)


def test_flip_stats_identical_arms_have_zero_flips():
    out = flip_floor.flip_stats([True, False, True, False], [True, False, True, False], 200, 1)
    assert out["flip_rate"] == 0 and out["mcnemar"]["p_value"] == 1.0


def test_flip_stats_counts_directional_flips():
    a = [True] * 10 + [False] * 10
    b = [False] * 4 + [True] * 6 + [False] * 10
    out = flip_floor.flip_stats(a, b, 200, 1)
    assert out["flip_rate"] == 0.2 and out["refusal_only_in_first"] == 4
    assert out["refusal_only_in_second"] == 0
    assert out["ci95"][0] <= 0.2 <= out["ci95"][1]


def test_flip_stats_rejects_mismatched_arms():
    with pytest.raises(ValueError):
        flip_floor.flip_stats([True], [True, False], 10, 1)


def test_mock_run_is_seeded_and_cache_off_repeat_is_exact():
    first = flip_floor.run(3, 80, 0.2, 100)
    again = flip_floor.run(3, 80, 0.2, 100)
    assert first == again
    assert first["cache_off_vs_off"]["flip_rate"] == 0
    assert first["cache_on_vs_off"]["flip_rate"] > 0


def test_zero_jitter_gives_zero_flip_rate():
    assert flip_floor.run(3, 40, 0.0, 50)["cache_on_vs_off"]["flip_rate"] == 0


def test_script_requires_mock_flag(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path / "x.json")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2 and not (tmp_path / "x.json").exists()


def test_script_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--mock", "--seed", "5", "--draws", "40", "--n-boot", "50",
         "--out", str(out)], capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["mock"] is True and doc["provenance"]["started_at"].endswith("Z")
    assert "cache_on_vs_off" in doc["refusal_label_flip_rate"]


def test_slots_run_is_seeded_and_pinned_arm_is_exact():
    first = flip_floor.run_slots(3, 80, 0.2, 100)
    assert first == flip_floor.run_slots(3, 80, 0.2, 100)
    assert first["slots_1_vs_1"]["flip_rate"] == 0
    assert first["slots_4_pinned_vs_1"]["flip_rate"] == 0
    assert first["slots_4_vs_1"]["flip_rate"] > 0


def test_slots_zero_jitter_gives_zero_flip_rate():
    assert flip_floor.run_slots(3, 40, 0.0, 50)["slots_4_vs_1"]["flip_rate"] == 0
