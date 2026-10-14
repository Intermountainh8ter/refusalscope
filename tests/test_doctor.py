import pytest

from refusalscope.client import ChatClient
from refusalscope.config import parse_config
from refusalscope.doctor import Check, _explain_error, check_model, run_doctor
from refusalscope.mock import PROFILES, Fault, MockProfile, MockServer
from tests.helpers import make_study


def provider_with_weather(base):
    """The shared provider already answers the doctor's probes; kept as a hook for other tests."""
    return base


def model_for(server, mock=True):
    config = make_study({"a": server.base_url, "b": server.base_url})
    raw = config.to_dict()
    raw["models"][0]["mock"] = mock
    return parse_config(raw).models[0]


def checks_by_name(checks):
    return {c.name: c for c in checks}


@pytest.fixture
def served(provider):
    started = []

    def make(profile=None, wrap=True, **kwargs):
        server = MockServer(
            profile=profile or PROFILES["perfect"],
            provider=provider_with_weather(provider) if wrap else provider,
            **kwargs,
        ).start()
        started.append(server)
        return server

    yield make
    for server in started:
        server.stop()


def client_for(server):
    return ChatClient(server.base_url, "m", retries=0, timeout_s=5)


def test_healthy_endpoint_passes_every_check(served):
    server = served()
    checks = checks_by_name(check_model(model_for(server), client_for(server), 5))
    assert {c.status for c in checks.values()} == {"ok"}
    assert {"endpoint", "mock flag", "chat", "tool calls", "context", "logprobs"} <= set(checks)


def test_text_tool_calls_fail_with_the_jinja_fix(served):
    server = served(MockProfile(p_correct=0.0).only_failure("text_call"))
    checks = checks_by_name(check_model(model_for(server), client_for(server)))
    assert checks["tool calls"].status == "fail" and "--jinja" in checks["tool calls"].fix
    assert "plain text" in checks["tool calls"].message


def test_model_that_answers_in_text_gets_a_warning(served):
    server = served(MockProfile(p_correct=0.0).only_failure("no_call"))
    checks = checks_by_name(check_model(model_for(server), client_for(server)))
    assert checks["tool calls"].status == "warn" and "chat template" in checks["tool calls"].fix


def test_malformed_arguments_warn(served):
    server = served(MockProfile(p_correct=0.0).only_failure("malformed_json"))
    checks = checks_by_name(check_model(model_for(server), client_for(server)))
    assert checks["tool calls"].status == "warn" and "valid JSON" in checks["tool calls"].message


def test_missing_logprobs_warns_that_kl_will_not_be_measured(served):
    server = served(MockProfile(logprobs_supported=False))
    checks = checks_by_name(check_model(model_for(server), client_for(server)))
    assert checks["logprobs"].status == "warn" and "not measured" in checks["logprobs"].fix


def test_context_overflow_is_diagnosed(served):
    server = served()
    server.queue_fault(
        Fault("status", status=400, body="request exceeds the available context size", skip=2)
    )
    checks = checks_by_name(check_model(model_for(server), client_for(server)))
    assert checks["context"].status == "fail" and "--ctx-size" in checks["context"].fix


def test_connection_refused_is_explained(served):
    server = served()
    client = client_for(server)
    model = model_for(server)
    server.stop()
    checks = check_model(model, client)
    assert len(checks) == 1 and checks[0].status == "fail"
    assert "connection refused" in checks[0].message and "start the server" in checks[0].fix


def test_wrong_base_url_suggests_v1(served):
    server = served()
    bare = ChatClient(server.base_url + "/wrong", "m", retries=0, timeout_s=5)
    checks = check_model(model_for(server), bare)
    assert checks[0].status == "fail" and "/v1" in checks[0].fix


def test_mock_flag_mismatch_is_a_failure(served):
    server = served()
    checks = checks_by_name(check_model(model_for(server, mock=False), client_for(server)))
    assert checks["mock flag"].status == "fail" and "MOCK" in checks["mock flag"].message


def test_chat_failure_stops_further_checks(served):
    server = served()
    server.queue_fault(Fault("status", status=401, body="no key"))
    checks = check_model(model_for(server), client_for(server))
    assert [c.name for c in checks][-1] == "chat" and checks[-1].status == "fail"
    assert "api_key_env" in checks[-1].fix


def test_empty_reply_from_a_thinking_model_is_flagged(served):
    from refusalscope.mock import MockReply

    server = MockServer(
        responder=lambda body: MockReply(content="", finish_reason="length")
    ).start()
    try:
        checks = checks_by_name(check_model(model_for(server), client_for(server)))
        assert checks["chat"].status == "warn" and "enable_thinking" in checks["chat"].fix
    finally:
        server.stop()


def test_thinking_blocks_are_reported():
    from refusalscope.mock import MockReply

    server = MockServer(responder=lambda body: MockReply(content="<think>hm</think>OK")).start()
    try:
        checks = checks_by_name(check_model(model_for(server), client_for(server)))
        assert checks["thinking"].status == "warn"
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("error", "status", "needle"),
    [
        ("URLError: [Errno 111] Connection refused", None, "connection refused"),
        ("URLError: [WinError 10061] No connection could be made", None, "connection refused"),
        ("TimeoutError: timed out", None, "timed out"),
        ("HTTP 401: x", 401, "HTTP 401"),
        ("HTTP 404: x", 404, "HTTP 404"),
        ("HTTP 400: request exceeds the available context size", 400, "context size exceeded"),
        ("HTTP 500: boom", 500, "HTTP 500"),
    ],
)
def test_error_explanations(error, status, needle):
    reason, fix = _explain_error(error, status)
    assert needle in reason and fix


def test_check_line_formatting():
    ok = Check("m", "chat", "ok", "fine", "ignored")
    bad = Check("m", "chat", "fail", "broken", "do this")
    assert ok.line().startswith("[  OK ]") and "fix:" not in ok.line()
    assert "fix: do this" in bad.line() and bad.line().startswith("[FAIL ]")


def test_run_doctor_checks_every_model_and_pair_warnings(served):
    a, b = served(), served()
    config = make_study({"a": a.base_url, "b": b.base_url})
    raw = config.to_dict()
    raw["models"][1]["quant"] = "different"
    checks = run_doctor(parse_config(raw))
    assert {c.target for c in checks} == {"a", "b", "pairs"}
    assert any(c.target == "pairs" and "quantisation differs" in c.message for c in checks)
