import json
import time

import pytest

from refusalscope.client import (
    ChatClient,
    ChatResult,
    ToolCall,
    parse_arguments,
    parse_completion,
)
from refusalscope.mock import (
    CallScenario,
    Fault,
    FirstTokenScenario,
    MockProfile,
    NoCallScenario,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CALL = CallScenario("w1", [{"name": "get_weather", "arguments": {"city": "Paris"}}], [WEATHER])
USER = [{"role": "user", "content": "weather in Paris"}]


# ------------------------------------------------------------------ parse_arguments / ToolCall


@pytest.mark.parametrize(
    ("raw", "args", "has_error"),
    [
        ('{"a": 1}', {"a": 1}, False),
        ("", {}, False),
        ("   ", {}, False),
        (None, {}, False),
        ("{bad", None, True),
        ("[1, 2]", None, True),
        ('"{\\"a\\": 1}"', None, True),
        ("3", None, True),
    ],
)
def test_parse_arguments(raw, args, has_error):
    parsed, error = parse_arguments(raw)
    assert parsed == args
    assert (error is not None) is has_error


def test_toolcall_from_wire_variants():
    call = ToolCall.from_wire({"id": "c1", "function": {"name": "f", "arguments": '{"x": 1}'}})
    assert (call.id, call.name, call.arguments, call.parse_error) == ("c1", "f", {"x": 1}, None)
    as_dict = ToolCall.from_wire({"function": {"name": "f", "arguments": {"x": 2}}}, index=3)
    assert as_dict.arguments == {"x": 2} and as_dict.id == "call_3"
    none_args = ToolCall.from_wire({"function": {"name": "f", "arguments": None}})
    assert none_args.arguments == {} and none_args.arguments_raw == ""
    weird = ToolCall.from_wire({"function": {"name": 5, "arguments": 7}})
    assert weird.name == "" and weird.parse_error is not None
    bad = ToolCall.from_wire({"function": {"name": "f", "arguments": "{oops"}})
    assert bad.arguments is None and "invalid JSON" in bad.parse_error


def test_toolcall_roundtrip_to_wire():
    call = ToolCall.from_wire({"id": "c9", "function": {"name": "f", "arguments": '{"a":1}'}})
    wire = call.to_wire()
    assert wire == {
        "id": "c9",
        "type": "function",
        "function": {"name": "f", "arguments": '{"a":1}'},
    }


# ------------------------------------------------------------------ parse_completion


def completion(message, **extra):
    return {"choices": [{"message": message, "finish_reason": "stop", **extra}]}


def test_parse_completion_content_parts_and_reasoning():
    parsed = parse_completion(
        completion(
            {
                "content": [{"type": "text", "text": "ab"}, "cd", {"type": "x"}],
                "reasoning_content": "thinking",
            }
        )
    )
    assert parsed["raw_content"] == "abcd"
    assert parsed["reasoning"] == "thinking"


def test_parse_completion_strips_think_but_keeps_raw():
    parsed = parse_completion(completion({"content": "<think>x</think>Hello"}))
    assert parsed["content"] == "Hello"
    assert parsed["raw_content"] == "<think>x</think>Hello"


def test_parse_completion_usage_and_missing_content():
    body = completion({"content": None}, logprobs=None)
    body["usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "weird": "x"}
    parsed = parse_completion(body)
    assert parsed["content"] == "" and parsed["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
    }
    assert parsed["logprobs"] is None


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [3]}, {"choices": [{}]}])
def test_parse_completion_rejects_unusable(body):
    with pytest.raises(ValueError):
        parse_completion(body)


def test_parse_completion_logprobs_normalised():
    body = completion(
        {"content": "A"},
        logprobs={
            "content": [
                {
                    "token": "A",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "A", "logprob": -0.1},
                        {"token": "B", "logprob": -2},
                    ],
                },
                {"junk": True},
            ]
        },
    )
    lp = parse_completion(body)["logprobs"]
    assert lp == [
        {
            "token": "A",
            "logprob": -0.1,
            "top": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}],
        }
    ]
    assert (
        parse_completion(completion({"content": "A"}, logprobs={"content": []}))["logprobs"] is None
    )


# ------------------------------------------------------------------ constructor validation


def test_client_validates_arguments():
    with pytest.raises(ValueError, match="http"):
        ChatClient("localhost:80", "m")
    with pytest.raises(ValueError, match="retries"):
        ChatClient("http://x", "m", retries=-1)
    with pytest.raises(ValueError, match="timeout"):
        ChatClient("http://x", "m", timeout_s=0)
    assert ChatClient("http://x/v1/", "m").base_url == "http://x/v1"


# ------------------------------------------------------------------ requests


def test_chat_returns_structured_tool_call(make_server, make_client):
    server = make_server(scenarios={"weather in Paris": CALL})
    result = make_client(server).chat(USER, tools=[WEATHER], seed=7, temperature=0.0)
    assert isinstance(result, ChatResult)
    assert result.ok and result.mock and result.status == 200
    assert [c.name for c in result.tool_calls] == ["get_weather"]
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.finish_reason == "tool_calls"
    assert result.usage["total_tokens"] > 0
    assert result.latency_s >= 0 and result.attempts == 1


def test_request_body_contents(make_server, make_client):
    server = make_server(scenarios={"weather in Paris": CALL})
    client = make_client(server, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    client.chat(
        USER,
        tools=[WEATHER],
        seed=42,
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        parallel_tool_calls=True,
        extra={"min_p": 0.0},
    )
    body = server.requests[0]
    assert body["seed"] == 42 and body["temperature"] == 0.7 and body["top_p"] == 0.9
    assert body["max_tokens"] == 64 and body["stream"] is False
    assert body["tools"] == [WEATHER] and body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["min_p"] == 0.0
    assert body["model"] == "mock-model"


def test_request_without_tools_omits_tool_fields(make_server, make_client):
    server = make_server()
    make_client(server).chat(USER)
    body = server.requests[0]
    assert "tools" not in body and "tool_choice" not in body and "seed" not in body


def test_authorization_header_and_default_key(make_server, make_client):
    server = make_server()
    make_client(server, api_key="secret-key").chat(USER)
    make_client(server).chat(USER)
    first, second = server.request_headers
    assert first["Authorization"] == "Bearer secret-key"
    assert second["Authorization"].startswith("Bearer sk-no-key")
    assert first["User-Agent"].startswith("refusalscope/")


def test_logprobs_are_captured(make_server, make_client):
    server = make_server(scenarios={"hi": FirstTokenScenario("k1")})
    result = make_client(server).chat(
        [{"role": "user", "content": "hi"}], logprobs=True, top_logprobs=5, max_tokens=1
    )
    assert result.logprobs is not None
    assert len(result.logprobs[0]["top"]) == 5
    assert result.logprobs[0]["token"] == result.logprobs[0]["top"][0]["token"]
    request = server.requests[0]
    assert request["logprobs"] is True and request["top_logprobs"] == 5


def test_logprobs_absent_when_server_does_not_support(make_server, make_client):
    server = make_server(
        profile=MockProfile(logprobs_supported=False),
        scenarios={"hi": FirstTokenScenario("k1")},
    )
    result = make_client(server).chat([{"role": "user", "content": "hi"}], logprobs=True)
    assert result.ok and result.logprobs is None


def test_assistant_message_and_record(make_server, make_client):
    server = make_server(scenarios={"weather in Paris": CALL})
    result = make_client(server).chat(USER, tools=[WEATHER])
    message = result.assistant_message()
    assert message["role"] == "assistant" and message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    record = result.record()
    assert record["ok"] is True and record["mock"] is True
    assert record["request"]["messages"] == USER
    json.dumps(record)  # must be serialisable for trajectory logs


def test_get_json_models(make_server, make_client):
    server = make_server(model_names=("a", "b"))
    status, payload, headers = make_client(server).get_json("/models")
    assert status == 200 and [m["id"] for m in payload["data"]] == ["a", "b"]
    assert headers["x-refusalscope-mock"] == "1"


# ------------------------------------------------------------------ retries and failures


def test_retries_transient_status_then_succeeds(make_server, make_client):
    server = make_server(scenarios={"weather in Paris": CALL})
    server.queue_fault(Fault("status", status=503), times=2)
    delays: list[float] = []
    client = make_client(server, retries=3, sleep=delays.append, backoff_s=0.5)
    result = client.chat(USER, tools=[WEATHER])
    assert result.ok and result.attempts == 3
    assert delays == [0.5, 1.0]
    assert len(server.requests) == 3


def test_retry_after_header_is_honoured_up_to_cap(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("status", status=429, retry_after="3"))
    delays: list[float] = []
    result = make_client(server, retries=1, sleep=delays.append, backoff_s=0.1).chat(USER)
    assert result.ok and delays == [3.0]
    server.queue_fault(Fault("status", status=429, retry_after="999"))
    delays.clear()
    make_client(server, retries=1, sleep=delays.append, max_backoff_s=8.0).chat(USER)
    assert delays == [8.0]


def test_backoff_is_capped(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("status", status=500), times=4)
    delays: list[float] = []
    result = make_client(server, retries=4, sleep=delays.append, backoff_s=1, max_backoff_s=3).chat(
        USER
    )
    assert result.ok
    assert delays == [1, 2, 3, 3]


def test_client_error_is_not_retried_and_reports_body(make_server, make_client):
    server = make_server()
    server.queue_fault(
        Fault("status", status=400, body='{"error":"request exceeds the available context size"}')
    )
    result = make_client(server, retries=3).chat(USER)
    assert not result.ok and result.attempts == 1 and result.status == 400
    assert "context size" in result.error
    assert result.response is None and result.request["model"] == "mock-model"


def test_retries_exhausted_reports_last_error(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("status", status=503, body="busy"), times=10)
    result = make_client(server, retries=2).chat(USER)
    assert not result.ok and result.attempts == 3 and result.status == 503
    assert "busy" in result.error


def test_dropped_connection_is_retried(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("drop"))
    result = make_client(server, retries=1).chat(USER)
    assert result.ok and result.attempts == 2


def test_garbage_body_is_reported(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("garbage"))
    result = make_client(server, retries=0).chat(USER)
    assert not result.ok and "JSONDecodeError" in result.error


def test_timeout_is_reported(make_server, make_client):
    server = make_server()
    server.queue_fault(Fault("delay", delay_s=0.6))
    started = time.perf_counter()
    result = make_client(server, retries=0, timeout_s=0.1).chat(USER)
    assert not result.ok and result.status is None
    assert time.perf_counter() - started < 3


def test_connection_refused_is_reported_not_raised(make_server, make_client):
    server = make_server()
    client = make_client(server, retries=1)
    server.stop()
    result = client.chat(USER)
    assert not result.ok and result.attempts == 2
    assert "URLError" in result.error or "Connection" in result.error


def test_unusable_response_is_not_retried(make_server, make_client):
    server = make_server(responder=lambda body: None)  # type: ignore[arg-type]
    result = make_client(server, retries=3).chat(USER)
    assert not result.ok
    assert result.attempts <= 4


def test_loopback_requests_bypass_environment_proxy(monkeypatch, make_server, make_client):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = make_server()
    assert make_client(server).chat(USER).ok


def test_no_call_scenario_roundtrip(make_server, make_client):
    server = make_server(scenarios={"hello": NoCallScenario("n1", "Hi there!", [WEATHER])})
    result = make_client(server).chat([{"role": "user", "content": "hello"}], tools=[WEATHER])
    assert result.ok and result.content == "Hi there!" and result.tool_calls == []
