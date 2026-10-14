import json

import pytest

from refusalscope.client import ChatClient
from refusalscope.config import parse_config
from refusalscope.mock import PROFILES, Fault, MockServer
from refusalscope.runner import (
    RUN_FILE,
    TRAJECTORY_FILE,
    RunError,
    Runner,
    SuiteData,
    plan_jobs,
    resolve_suites,
)
from refusalscope.turn import Turn
from tests.helpers import make_study


def quiet(config, run_dir, **kwargs):
    return Runner(config, run_dir, progress=lambda m: None, **kwargs)


def records(run_dir):
    with open(run_dir / TRAJECTORY_FILE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture
def servers(provider):
    started = []

    def make(*profiles):
        started.extend(
            MockServer(profile=profile, provider=provider).start() for profile in profiles
        )
        return started[-len(profiles) :]

    yield make
    for server in started:
        server.stop()


@pytest.fixture
def config_pair(servers, behaviours_file):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    return make_study({"a": a.base_url, "b": b.base_url}, behaviours_file), a, b


# ------------------------------------------------------------------ planning


def test_resolve_suites_aliases_and_errors():
    assert resolve_suites("all") == ["toolcall", "injection", "refusal", "kl", "capability"]
    assert resolve_suites("ablation") == ["refusal", "kl"]
    assert resolve_suites(["toolcall", "ablation", "toolcall"]) == ["toolcall", "refusal", "kl"]
    with pytest.raises(RunError, match="unknown suite"):
        resolve_suites("everything")


def test_plan_counts_for_one_pair(behaviours_file):
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"}, behaviours_file)
    data = SuiteData.from_config(config)
    jobs = plan_jobs(config, data, resolve_suites("all"))
    by_suite = {s: sum(1 for j in jobs if j.suite == s) for s in resolve_suites("all")}
    assert by_suite == {
        "toolcall": 748,
        "injection": 1512,
        "refusal": 200,
        "kl": 120,
        "capability": 200,
    }
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert {j.condition for j in jobs if j.suite == "injection"} == {"control", "attack"}


def test_plan_expands_temperatures_and_seeds_but_not_for_greedy(behaviours_file):
    config = make_study(
        {"a": "http://x/v1", "b": "http://y/v1"},
        behaviours_file,
        sampling={"temperatures": [0.0, 0.7], "seeds": [11, 22, 33]},
    )
    data = SuiteData.from_config(config)
    jobs = [j for j in plan_jobs(config, data, ["toolcall"], ["a"], limit=10) if j.model == "a"]
    assert sorted({(j.temperature, j.seed) for j in jobs}) == [
        (0.0, 11),
        (0.7, 11),
        (0.7, 22),
        (0.7, 33),
    ]
    assert len(jobs) == 40


def test_limit_takes_an_even_spread_across_categories(behaviours_file):
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"}, behaviours_file)
    data = SuiteData.from_config(config)
    jobs = plan_jobs(config, data, ["toolcall"], ["a"], limit=34)
    categories = {data.toolcall[j.case_id]["category"] for j in jobs}
    assert len(jobs) == 34 and len(categories) == 17


def test_limit_sampling_is_deterministic_and_spreads_injection_vectors(behaviours_file):
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"}, behaviours_file)
    data = SuiteData.from_config(config)
    first = plan_jobs(config, data, ["injection"], ["a"], limit=28)
    assert first == plan_jobs(config, data, ["injection"], ["a"], limit=28)
    vectors = [
        data.injection[j.case_id]["meta"]["vector"] for j in first if j.condition == "attack"
    ]
    assert len(vectors) == 28 and set(vectors) == {v for v in vectors} and len(set(vectors)) == 14
    assert {vectors.count(v) for v in set(vectors)} == {2}
    assert plan_jobs(config, data, ["capability"], ["a"], limit=1000) == plan_jobs(
        config, data, ["capability"], ["a"]
    )


def test_model_filter_and_stable_ids(behaviours_file):
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"}, behaviours_file)
    data = SuiteData.from_config(config)
    first = plan_jobs(config, data, ["capability"], ["b"])
    again = plan_jobs(config, data, ["capability"], ["b"])
    assert first == again and {j.model for j in first} == {"b"}


def test_suite_data_requires_a_readable_behaviours_file(tmp_path):
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"}, tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="not found"):
        SuiteData.from_config(config)


def test_suite_data_without_behaviours_file_notes_the_consequence():
    config = make_study({"a": "http://x/v1", "b": "http://y/v1"})
    data = SuiteData.from_config(config)
    assert data.harmful is None and "no pair can pass the manipulation check" in data.notes[0]
    assert all(i.startswith("N-") for i in data.refusal)


def test_suite_data_uses_configured_sizes(behaviours_file):
    config = make_study(
        {"a": "http://x/v1", "b": "http://y/v1"},
        behaviours_file,
        suites={"toolcall": {"n_per_category": 2}, "injection": {"n_cases": 14},
                "capability": {"n_items": 20}, "refusal": {"harmful_path": str(behaviours_file), "max_items": 5}},
    )  # fmt: skip
    data = SuiteData.from_config(config)
    assert (len(data.toolcall), len(data.injection), len(data.capability)) == (34, 14, 20)
    assert len([i for i in data.refusal if i.startswith("H-")]) == 5


# ------------------------------------------------------------------ running


def test_run_writes_complete_trajectories(config_pair, tmp_path):
    config, *_ = config_pair
    runner = quiet(config, tmp_path / "MOCK-x", workers=3, run_id="MOCK-x")
    summary = runner.run(resolve_suites("all"), limit=20)
    rows = records(tmp_path / "MOCK-x")
    assert summary.executed == len(rows) == summary.planned and summary.errors == 0
    assert {r["suite"] for r in rows} == {"toolcall", "injection", "refusal", "kl", "capability"}
    for row in rows:
        assert row["mock"] is True and row["turns"] and row["requests"] == len(row["turns"])
        first = row["turns"][0]
        assert first["request"]["messages"] and first["response"]["choices"]
        assert row["pair_id"] == "p" and row["role"] in ("stock", "ablated")
    toolcall_row = next(r for r in rows if r["suite"] == "toolcall")
    assert set(toolcall_row["grade"]) == {"strict", "lenient"}
    kl_row = next(r for r in rows if r["suite"] == "kl")
    assert kl_row["temperature"] == 1.0 and kl_row["turns"][0]["request"]["max_tokens"] == 1
    assert kl_row["turns"][0]["request"]["logprobs"] is True
    injection_rows = [r for r in rows if r["suite"] == "injection"]
    assert {r["condition"] for r in injection_rows} == {"control", "attack"}
    assert max(r["requests"] for r in injection_rows) >= 2


def test_run_metadata(config_pair, tmp_path):
    config, _, _ = config_pair
    runner = quiet(config, tmp_path / "MOCK-m", run_id="MOCK-m")
    runner.run(["capability"], limit=5)
    meta = json.loads((tmp_path / "MOCK-m" / RUN_FILE).read_text(encoding="utf-8"))
    assert meta["mock"] is True and meta["version"] and meta["config_sha256"] == config.digest()
    assert meta["harmful_file"]["n_used"] == 60 and len(meta["suite_hashes"]) == 5
    assert "sk-" not in json.dumps(meta)
    assert parse_config(meta["config"]).digest() == config.digest()


def test_requests_carry_sampling_settings(config_pair, tmp_path):
    config, a, _ = config_pair
    quiet(config, tmp_path / "r", run_id="MOCK-r").run(["toolcall"], models=["a"], limit=5)
    body = a.requests[0]
    assert body["temperature"] == 0.0 and body["seed"] == 1 and body["top_p"] == 1.0
    assert (
        body["parallel_tool_calls"] is True
        and body["tool_choice"] == "auto"
        and body["max_tokens"] == 128
    )


def test_resume_skips_finished_jobs(config_pair, tmp_path):
    config, a, b = config_pair
    first = quiet(config, tmp_path / "r", run_id="MOCK-r").run(["toolcall", "capability"], limit=15)
    seen = len(a.requests) + len(b.requests)
    second = quiet(config, tmp_path / "r", run_id="MOCK-r").run(
        ["toolcall", "capability"], limit=15
    )
    assert second.executed == 0 and second.already_done == first.planned
    assert len(a.requests) + len(b.requests) == seen
    third = quiet(config, tmp_path / "r", run_id="MOCK-r").run(
        ["toolcall", "capability", "kl"], limit=15
    )
    assert third.executed > 0 and third.already_done == first.planned


def test_resume_repairs_a_truncated_final_line(config_pair, tmp_path):
    config, _, _ = config_pair
    run_dir = tmp_path / "r"
    runner = quiet(config, run_dir, run_id="MOCK-r")
    runner.run(["capability"], limit=10)
    path = run_dir / TRAJECTORY_FILE
    raw = path.read_bytes()
    path.write_bytes(raw[:-40])  # simulate a crash mid-write
    messages: list[str] = []
    resumed = Runner(config, run_dir, progress=messages.append, run_id="MOCK-r")
    summary = resumed.run(["capability"], limit=10)
    assert summary.executed == 1 and any("truncated" in m for m in messages)
    assert len(records(run_dir)) == 20  # 10 per model, the lost one re-run
    assert path.read_bytes().endswith(b"\n")


def test_corrupt_middle_line_is_an_error(config_pair, tmp_path):
    config, _, _ = config_pair
    run_dir = tmp_path / "r"
    quiet(config, run_dir, run_id="MOCK-r").run(["capability"], limit=5)
    path = run_dir / TRAJECTORY_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = "{broken"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RunError, match="corrupt trajectory line"):
        quiet(config, run_dir, run_id="MOCK-r").run(["capability"], limit=5)


def test_errored_jobs_are_retried_on_resume(servers, behaviours_file, tmp_path):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    config = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file)
    a.queue_fault(Fault("status", status=500), times=3)
    run_dir = tmp_path / "r"
    first = quiet(config, run_dir, workers=1, run_id="MOCK-r").run(
        ["capability"], models=["a"], limit=10
    )
    assert first.errors == 3
    again = quiet(config, run_dir, workers=1, run_id="MOCK-r").run(
        ["capability"], models=["a"], limit=10
    )
    assert again.executed == 3 and again.errors == 0
    latest = {}
    for row in records(run_dir):
        latest[row["job_id"]] = row
    assert len(latest) == 10 and all(r["errors"] == 0 for r in latest.values())


def test_server_that_moves_ports_can_still_resume(servers, behaviours_file, tmp_path):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    c, d = servers(PROFILES["reference"], PROFILES["degraded"])
    run_dir = tmp_path / "r"
    quiet(
        make_study({"a": a.base_url, "b": b.base_url}, behaviours_file), run_dir, run_id="MOCK-r"
    ).run(["capability"], limit=8)
    moved = make_study({"a": c.base_url, "b": d.base_url}, behaviours_file)
    summary = quiet(moved, run_dir, run_id="MOCK-r").run(["capability"], limit=8)
    assert summary.executed == 0 and summary.already_done == 16


def test_result_affecting_changes_are_refused_unless_forced(config_pair, tmp_path, behaviours_file):
    config, a, b = config_pair
    run_dir = tmp_path / "r"
    quiet(config, run_dir, run_id="MOCK-r").run(["capability"], limit=5)
    changed = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file,
                         sampling={"temperatures": [0.0], "seeds": [2], "max_tokens": 128})  # fmt: skip
    with pytest.raises(RunError, match="different result-affecting settings"):
        quiet(changed, run_dir, run_id="MOCK-r").run(["capability"], limit=5)
    quiet(changed, run_dir, run_id="MOCK-r").run(["capability"], limit=5, force=True)
    meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    assert meta["config_history"] and meta["config"]["sampling"]["seeds"] == [2]


def test_changed_behaviours_file_is_refused(config_pair, tmp_path, behaviours_file):
    config, _, _ = config_pair
    run_dir = tmp_path / "r"
    quiet(config, run_dir, run_id="MOCK-r").run(["capability"], limit=5)
    behaviours_file.write_text(
        behaviours_file.read_text(encoding="utf-8") + '{"prompt": "extra"}\n', encoding="utf-8"
    )
    try:
        with pytest.raises(RunError, match="different result-affecting settings"):
            quiet(config, run_dir, run_id="MOCK-r").run(["capability"], limit=5)
    finally:
        lines = behaviours_file.read_text(encoding="utf-8").splitlines()[:-1]
        behaviours_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_circuit_breaker_stops_a_dead_model(servers, behaviours_file, tmp_path):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    config = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file)
    a.queue_fault(Fault("status", status=500), times=500)
    summary = quiet(config, tmp_path / "r", workers=1, breaker=4, run_id="MOCK-r").run(
        ["capability"], models=["a"], limit=30
    )
    assert summary.aborted and "failed 4 requests in a row" in summary.aborted
    assert summary.executed == 4 and "same --run-id" in summary.aborted


def test_parallel_workers_match_serial_results(servers, behaviours_file, tmp_path):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    config = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file)
    quiet(config, tmp_path / "s", workers=1, run_id="MOCK-s").run(
        ["toolcall", "capability"], limit=12
    )
    quiet(config, tmp_path / "p", workers=6, run_id="MOCK-p").run(
        ["toolcall", "capability"], limit=12
    )
    serial = {r["job_id"]: r["grade"] for r in records(tmp_path / "s")}
    parallel = {r["job_id"]: r["grade"] for r in records(tmp_path / "p")}
    assert serial == parallel and len(serial) == 48


def test_default_client_uses_model_settings(behaviours_file):
    config = make_study(
        {"a": "http://127.0.0.1:1/v1", "b": "http://127.0.0.1:2/v1"}, behaviours_file
    )
    runner = quiet(config, "unused-dir")
    client = runner.client("a")
    assert isinstance(client, ChatClient) and client.base_url == "http://127.0.0.1:1/v1"
    assert client.timeout_s == 10 and client.retries == 0
    assert runner.client("a") is client


def test_api_key_comes_from_the_environment(behaviours_file, monkeypatch):
    raw = {
        "study": "k",
        "models": [
            {
                "name": "a",
                "base_url": "http://127.0.0.1:1/v1",
                "role": "stock",
                "pair_id": "p",
                "api_key_env": "RS_KEY",
            },
            {"name": "b", "base_url": "http://127.0.0.1:2/v1", "role": "ablated", "pair_id": "p"},
        ],
    }
    config = parse_config(raw)
    monkeypatch.setenv("RS_KEY", "token-123")
    assert quiet(config, "unused-dir").client("a").api_key == "token-123"


# ------------------------------------------------------------------ mock labelling safety


def test_mock_endpoint_must_be_configured_as_mock(servers, behaviours_file):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    raw = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file).to_dict()
    raw["models"][0]["mock"] = False
    with pytest.raises(RunError, match="announces itself as the MOCK server"):
        quiet(parse_config(raw), "unused-dir").check_mock_flags()


def test_real_endpoint_cannot_be_labelled_mock(servers, behaviours_file, monkeypatch):
    a, b = servers(PROFILES["reference"], PROFILES["degraded"])
    config = make_study({"a": a.base_url, "b": b.base_url}, behaviours_file)
    runner = quiet(config, "unused-dir")
    real_client = ChatClient(a.base_url, "m", retries=0)
    monkeypatch.setattr(real_client, "get_json", lambda path: (200, {}, {}))
    runner._clients["a"] = real_client
    with pytest.raises(RunError, match='marked "mock": true but'):
        runner.check_mock_flags()


def test_unreachable_endpoints_are_left_to_doctor(behaviours_file):
    config = make_study(
        {"a": "http://127.0.0.1:1/v1", "b": "http://127.0.0.1:2/v1"}, behaviours_file
    )
    quiet(config, "unused-dir").check_mock_flags()  # must not raise


def test_matching_mock_flags_pass(config_pair):
    config, _, _ = config_pair
    quiet(config, "unused-dir").check_mock_flags()


# ------------------------------------------------------------------ helpers


def test_turn_from_record_roundtrip(config_pair, tmp_path):
    config, _, _ = config_pair
    quiet(config, tmp_path / "r", run_id="MOCK-r").run(["toolcall"], models=["a"], limit=6)
    row = records(tmp_path / "r")[0]
    turn = Turn.from_record(row["turns"][0])
    assert turn.ok and (turn.tool_calls or turn.content)
    failed = Turn.from_record({"ok": False, "error": "boom", "response": None})
    assert not failed.ok and failed.error == "boom"
    unusable = Turn.from_record({"ok": True, "response": {"choices": []}})
    assert not unusable.ok and "unusable" in unusable.error
