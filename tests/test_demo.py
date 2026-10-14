import json

from refusalscope.cli import main
from refusalscope.demo import PLACEHOLDER_BEHAVIOURS, run_demo


def all_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_demo_labels_every_output_file_mock(tmp_path):
    demo = run_demo(tmp_path / "out", limit=20, n_boot=200)
    names = {p.name for p in all_files(tmp_path / "out") if p.suffix in {".json", ".md", ".svg"}}
    assert names and all(
        n.startswith("MOCK") or n in {"run.json", "trajectories.jsonl"} for n in names
    ), names
    assert all("MOCK" in str(p.relative_to(tmp_path / "out")) for p in all_files(tmp_path / "out"))
    results = json.loads(demo["results_path"].read_text(encoding="utf-8"))
    assert results["mock"] is True and results["label"] == "MOCK-demo"
    for path in demo["reports"]:
        assert path.name.startswith("MOCK")
    markdown = next(p for p in demo["reports"] if p.suffix == ".md").read_text(encoding="utf-8")
    assert markdown.startswith("# MOCK - refusalscope report") and "NOT EVIDENCE" in markdown


def test_demo_exercises_every_suite_and_the_confirmatory_tests(tmp_path):
    demo = run_demo(tmp_path / "out", limit=60, n_boot=300)
    rows = [
        json.loads(line)
        for line in (demo["run_dir"] / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {r["suite"] for r in rows} == {"toolcall", "injection", "refusal", "kl", "capability"}
    assert all(r["mock"] for r in rows)
    pair = demo["results"]["pairs"]["mock-pair"]
    assert pair["status"] == "confirmatory" and {"H1", "H2", "H4"} <= set(pair["tests"])


def test_placeholder_behaviours_are_obviously_not_real_requests():
    assert len(PLACEHOLDER_BEHAVIOURS) == 60
    assert all(
        "Placeholder" in text and "not a real behaviour" in text for text in PLACEHOLDER_BEHAVIOURS
    )


def test_mock_demo_command(tmp_path, capsys):
    assert (
        main(["mock-demo", "--out", str(tmp_path / "demo"), "--limit", "10", "--bootstrap", "100"])
        == 0
    )
    out = capsys.readouterr().out
    assert "MOCK demo:" in out and "labelled MOCK" in out and "MOCK-demo.json" in out
