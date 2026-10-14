import json
from pathlib import Path

import pytest

from refusalscope.cli import build_parser, main
from refusalscope.mock import PROFILES, MockServer
from tests.helpers import make_study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_config(tmp_path, provider, behaviours_file):
    a = MockServer(profile=PROFILES["reference"], provider=provider).start()
    b = MockServer(profile=PROFILES["degraded"], provider=provider).start()
    config = make_study({"mock-a": a.base_url, "mock-b": b.base_url}, behaviours_file)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    yield path
    a.stop()
    b.stop()


def test_parser_lists_every_command():
    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "command").choices
    assert set(choices) == {
        "check-config",
        "generate",
        "doctor",
        "run",
        "analyze",
        "report",
        "audit",
        "power",
        "mock-serve",
    }


def test_generate_writes_all_snapshots(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "toolcall.jsonl", "injection.jsonl", "borderline.jsonl", "kl_prompts.jsonl", "capability.jsonl",
    }  # fmt: skip
    assert "wrote 374 rows for toolcall" in capsys.readouterr().out
    assert (tmp_path / "toolcall.jsonl").read_bytes() == (
        ROOT / "suites" / "toolcall.jsonl"
    ).read_bytes()


def test_generate_only_selected(tmp_path):
    assert main(["generate", "--out", str(tmp_path), "--only", "capability"]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["capability.jsonl"]


def test_doctor_passes_on_mock_servers(mock_config, capsys):
    code = main(["doctor", "--config", str(mock_config)])
    out = capsys.readouterr().out
    assert code == 0 and "0 failed" in out and "MOCK endpoint" in out


def test_doctor_fails_with_fixes_when_servers_are_down(tmp_path, behaviours_file, capsys):
    config = make_study(
        {"a": "http://127.0.0.1:9/v1", "b": "http://127.0.0.1:9/v1"}, behaviours_file
    )
    path = tmp_path / "down.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert main(["doctor", "--config", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "fix:" in out


def test_doctor_rejects_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["doctor", "--config", str(bad)]) == 2
    assert main(["run", "--config", str(bad)]) == 2
    assert capsys.readouterr().err.count("error:") == 2


def test_full_cli_flow_run_resume_analyze_report(mock_config, tmp_path, capsys):
    out_dir = tmp_path / "runs"
    argv = ["run", "--config", str(mock_config), "--suite", "toolcall", "capability", "refusal", "--out", str(out_dir),
            "--run-id", "MOCK-cli", "--limit", "30", "--workers", "3"]  # fmt: skip
    assert main(argv) == 0
    first = capsys.readouterr()
    assert "run MOCK-cli" in first.out and "next: refusalscope analyze" in first.out
    assert (out_dir / "MOCK-cli" / "trajectories.jsonl").exists()

    assert main(argv) == 0  # resume: nothing left to do
    assert "0 executed" in capsys.readouterr().out

    results_dir = tmp_path / "results"
    assert (
        main(
            [
                "analyze",
                "--run",
                str(out_dir / "MOCK-cli"),
                "--out",
                str(results_dir),
                "--bootstrap",
                "200",
            ]
        )
        == 0
    )
    analyze_out = capsys.readouterr().out
    assert "MOCK run: these numbers say nothing about any real model" in analyze_out
    assert (results_dir / "MOCK-cli.json").exists()

    assert main(["report", "--results", str(results_dir), "--out", str(tmp_path / "report")]) == 2
    assert "only MOCK result files" in capsys.readouterr().err
    assert (
        main(
            [
                "report",
                "--results",
                str(results_dir),
                "--out",
                str(tmp_path / "report"),
                "--allow-mock",
            ]
        )
        == 0
    )
    written = capsys.readouterr().out
    assert "MOCK-cli-report.md" in written and "MOCK-cli-forest.svg" in written

    assert (
        main(["audit", "--run", str(out_dir / "MOCK-cli"), "--suite", "toolcall", "--n", "2"]) == 0
    )
    audit = capsys.readouterr().out
    assert "request:" in audit and "grade:" in audit
    assert main(["audit", "--run", str(out_dir / "MOCK-cli"), "--suite", "kl"]) == 0
    assert "no kl trajectories" in capsys.readouterr().out


def test_run_ids_for_mock_configs_always_carry_the_label(mock_config, tmp_path, capsys):
    assert main(["run", "--config", str(mock_config), "--suite", "capability", "--out", str(tmp_path / "r"),
                 "--run-id", "plain", "--limit", "3"]) == 0  # fmt: skip
    assert (tmp_path / "r" / "MOCK-plain").is_dir() and not (tmp_path / "r" / "plain").exists()
    assert main(["run", "--config", str(mock_config), "--suite", "capability", "--out", str(tmp_path / "r"),
                 "--limit", "3"]) == 0  # fmt: skip
    assert any(
        p.name.startswith("MOCK-mock-pipeline") or p.name.startswith("MOCK-unit")
        for p in (tmp_path / "r").iterdir()
    )
    capsys.readouterr()


def test_run_with_unknown_suite_is_an_error(mock_config, capsys):
    assert main(["run", "--config", str(mock_config), "--suite", "nonsense"]) == 2
    assert "unknown suite" in capsys.readouterr().err


def test_run_without_behaviours_file_reports_the_error(tmp_path, mock_config, capsys):
    raw = json.loads(mock_config.read_text(encoding="utf-8"))
    raw["suites"]["refusal"]["harmful_path"] = str(tmp_path / "nope.jsonl")
    mock_config.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["run", "--config", str(mock_config), "--out", str(tmp_path / "r")]) == 2
    assert "not found" in capsys.readouterr().err


def test_run_refuses_mislabelled_endpoints(mock_config, tmp_path, capsys):
    raw = json.loads(mock_config.read_text(encoding="utf-8"))
    for model in raw["models"]:
        model["mock"] = False
    mock_config.write_text(json.dumps(raw), encoding="utf-8")
    assert (
        main(
            ["run", "--config", str(mock_config), "--out", str(tmp_path / "r"), "--run-id", "real"]
        )
        == 2
    )
    assert "announces itself as the MOCK server" in capsys.readouterr().err


def test_run_reports_aborted_runs_with_a_distinct_exit_code(tmp_path, behaviours_file, capsys):
    config = make_study({"a": "http://127.0.0.1:9/v1", "b": "http://127.0.0.1:9/v1"}, behaviours_file,
                        sampling={"temperatures": [0.0], "seeds": [1], "retries": 0, "timeout_s": 1})  # fmt: skip
    raw = config.to_dict()
    for model in raw["models"]:
        model["mock"] = False
    path = tmp_path / "down.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    code = main(["run", "--config", str(path), "--suite", "capability", "--out", str(tmp_path / "r"),
                 "--run-id", "down", "--limit", "40", "--workers", "1"])  # fmt: skip
    assert code == 3
    assert "failed 10 requests in a row" in capsys.readouterr().err


def test_analyze_errors(tmp_path, capsys):
    assert main(["analyze", "--run", str(tmp_path)]) == 2
    assert "not a run directory" in capsys.readouterr().err
    assert main(["audit", "--run", str(tmp_path), "--suite", "toolcall"]) == 2


def test_report_without_results_says_not_yet_run(tmp_path, capsys):
    assert (
        main(["report", "--results", str(tmp_path / "none"), "--out", str(tmp_path / "out")]) == 2
    )
    assert "NOT YET RUN" in capsys.readouterr().err


def test_power_prints_the_preregistered_numbers(capsys):
    assert main(["power", "--n", "374", "--p-discordant", "0.1", "--rd", "0.05", "0.08"]) == 0
    out = capsys.readouterr().out
    assert "0.847" in out and "1.000" in out
    assert main(["power", "--n", "378", "--p-discordant", "0.05", "--rd", "0.10"]) == 0
    assert "n/a" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage: refusalscope" in capsys.readouterr().err


def test_python_dash_m_entry_point():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "refusalscope", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "refusalscope 0.1.0" in result.stdout
