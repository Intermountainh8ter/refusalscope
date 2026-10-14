"""A mock OpenAI-compatible server for testing the harness.

THE MOCK PRODUCES NO EVIDENCE ABOUT ANY REAL MODEL. It exists so that the client, runner,
graders and statistics can be tested end to end without GPUs or weights. Its answers come from
an oracle (the case's own ground truth) that is degraded with configurable probabilities, so a
run against it can only show that the pipeline notices the degradation it was told to inject.
Every response is marked as mock (header and ``system_fingerprint``) and the runner propagates
a ``MOCK`` label to every output.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from refusalscope.client import MOCK_FINGERPRINT, MOCK_HEADER
from refusalscope.util import stable_rng

# --------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class CallScenario:
    """The request expects these concrete tool calls."""

    key: str
    calls: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class NoCallScenario:
    """The request expects a plain-text reply and no tool call."""

    key: str
    reply: str
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentScenario:
    """A multi-step agent task, optionally with an injection payload to be (not) obeyed."""

    key: str
    task_calls: list[dict[str, Any]]
    answer: str
    payload: str | None = None
    attacker_call: dict[str, Any] | None = None


@dataclass(frozen=True)
class TextScenario:
    """A text task with one correct and one incorrect reply (capability control)."""

    key: str
    correct: str
    wrong: str


@dataclass(frozen=True)
class RefusalScenario:
    """A refusal probe. ``harmful`` selects which refusal probability applies."""

    key: str
    harmful: bool


@dataclass(frozen=True)
class FirstTokenScenario:
    """A one-token request with logprobs (KL probe)."""

    key: str


Scenario = (
    CallScenario
    | NoCallScenario
    | AgentScenario
    | TextScenario
    | RefusalScenario
    | FirstTokenScenario
)
Provider = Callable[[dict[str, Any]], "Scenario | None"]

# --------------------------------------------------------------------------- profile

FAILURE_MODES = (
    "wrong_value",
    "missing_arg",
    "wrong_type",
    "unknown_arg",
    "wrong_function",
    "malformed_json",
    "text_call",
    "no_call",
)


@dataclass(frozen=True)
class MockProfile:
    """Tunable behaviour of the mock. All numbers are arbitrary test settings."""

    name: str = "perfect"
    seed: int = 0
    p_correct: float = 1.0
    failure_weights: dict[str, float] = field(
        default_factory=lambda: {
            "wrong_value": 4.0,
            "missing_arg": 1.0,
            "wrong_type": 1.0,
            "unknown_arg": 1.0,
            "wrong_function": 1.0,
            "malformed_json": 1.0,
            "text_call": 1.0,
            "no_call": 1.0,
        }
    )
    p_inject: float = 0.0
    p_attack_text: float = 0.0
    p_task_success: float = 1.0
    p_refuse_harmful: float = 0.0
    p_refuse_benign: float = 0.0
    p_capability: float = 1.0
    kl_shift: float = 0.0
    logprobs_supported: bool = True
    latency_s: float = 0.0

    def with_(self, **changes: Any) -> MockProfile:
        return replace(self, **changes)

    def only_failure(self, mode: str) -> MockProfile:
        if mode not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {mode!r}")
        return replace(self, failure_weights={m: float(m == mode) for m in FAILURE_MODES})


PROFILES: dict[str, MockProfile] = {
    # Always right, never injectable, always refuses harmful prompts.
    "perfect": MockProfile(name="perfect", p_refuse_harmful=1.0),
    # Arbitrary numbers chosen to exercise the pipeline. They are NOT estimates for any model.
    "reference": MockProfile(
        name="reference",
        seed=1,
        p_correct=0.90,
        p_inject=0.15,
        p_task_success=0.92,
        p_refuse_harmful=0.95,
        p_refuse_benign=0.10,
        p_capability=0.90,
        kl_shift=0.0,
    ),
    "degraded": MockProfile(
        name="degraded",
        seed=2,
        p_correct=0.78,
        p_inject=0.35,
        p_task_success=0.88,
        p_refuse_harmful=0.10,
        p_refuse_benign=0.0,
        p_capability=0.88,
        kl_shift=0.6,
    ),
}

# --------------------------------------------------------------------------- reply

REFUSAL_TEXTS = (
    "I'm sorry, but I can't help with that request.",
    "I cannot assist with that. It goes against my guidelines.",
    "I can't provide that. If you have another question, I'm happy to help.",
    "I won't be able to help with this one.",
)
COMPLIANCE_TEXTS = (
    "Sure, here is an overview. Step 1: gather the basics. Step 2: proceed carefully.",
    "Certainly! Here's a short explanation of how that works in general terms.",
    "Here is what you asked for: first consider the context, then follow these steps.",
)
GENERIC_TEXT = "OK."
FIRST_TOKENS = ("The", "A", "I", "Sure", "In", "To", "It", "Here", "One", "This", "We", "You")


@dataclass
class MockReply:
    """What the mock server returns for one request."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # {"name": str, "arguments": dict | str}
    finish_reason: str = "stop"
    first_token_logprobs: list[tuple[str, float]] | None = None
    top_logprobs: int = 0


def text_tool_call(name: str, arguments: dict[str, Any] | str) -> str:
    """A tool call written as plain text, the way a model served without a tool parser does."""
    payload = arguments if isinstance(arguments, dict) else {"raw": arguments}
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": payload}) + "\n</tool_call>"


def _dummy_value(schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    return {
        "string": "x",
        "integer": 1,
        "number": 1.5,
        "boolean": True,
        "array": [],
        "object": {},
    }.get(kind, "x")


def _tool_index(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function", tool)
        if isinstance(function, dict) and function.get("name"):
            index[function["name"]] = function
    return index


def spurious_call(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """A plausible-looking but unwanted call to the first offered tool."""
    index = _tool_index(tools)
    if not index:
        return {"name": "unknown_tool", "arguments": {}}
    name, function = next(iter(index.items()))
    schema = function.get("parameters", {})
    properties = schema.get("properties", {})
    args = {key: _dummy_value(properties.get(key, {})) for key in schema.get("required", [])}
    return {"name": name, "arguments": args}


def _break_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " zz"
    if isinstance(value, list):
        return [*value, "zz"]
    if isinstance(value, dict):
        return {**value, "zz": "zz"}
    return "zz"


def _wrong_type(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return 12345
    return "zz"


def mutate_call(
    call: dict[str, Any], tools: list[dict[str, Any]], mode: str
) -> tuple[dict[str, Any], str | None]:
    """Return a degraded copy of a call and, for ``malformed_json``, the broken argument text."""
    name = call["name"]
    args = json.loads(json.dumps(call.get("arguments", {})))
    index = _tool_index(tools)
    required = index.get(name, {}).get("parameters", {}).get("required", [])
    if mode == "missing_arg":
        victims = [key for key in required if key in args]
        if victims:
            del args[victims[0]]
            return {"name": name, "arguments": args}, None
        mode = "wrong_value"
    if mode == "wrong_type":
        if args:
            key = next(iter(args))
            args[key] = _wrong_type(args[key])
            return {"name": name, "arguments": args}, None
        mode = "unknown_arg"
    if mode == "wrong_value":
        if args:
            key = next(iter(args))
            args[key] = _break_value(args[key])
            return {"name": name, "arguments": args}, None
        mode = "unknown_arg"
    if mode == "unknown_arg":
        args["zz_extra"] = 1
        return {"name": name, "arguments": args}, None
    if mode == "wrong_function":
        others = [n for n in index if n != name]
        return {"name": others[0] if others else f"{name}_v2", "arguments": args}, None
    if mode == "malformed_json":
        text = json.dumps(args)
        return {"name": name, "arguments": args}, text[:-1] if len(text) > 1 else "{"
    raise ValueError(f"unknown failure mode {mode!r}")


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class ProfileResponder:
    """Turns a request into a :class:`MockReply` using a profile and a scenario provider."""

    def __init__(self, profile: MockProfile, provider: Provider | None = None) -> None:
        self.profile = profile
        self.provider = provider

    def __call__(self, body: dict[str, Any]) -> MockReply:
        scenario = self.provider(body) if self.provider else None
        if scenario is None:
            return MockReply(content=GENERIC_TEXT)
        top = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        if isinstance(scenario, CallScenario):
            return self._calls(scenario, body)
        if isinstance(scenario, NoCallScenario):
            return self._no_call(scenario, body)
        if isinstance(scenario, AgentScenario):
            return self._agent(scenario, body)
        if isinstance(scenario, TextScenario):
            return self._text(scenario, body)
        if isinstance(scenario, RefusalScenario):
            return self._refusal(scenario, body)
        if isinstance(scenario, FirstTokenScenario):
            return self._first_token(scenario, body, top)
        raise TypeError(f"unsupported scenario {type(scenario).__name__}")

    # ------------------------------------------------------------------ helpers

    def _rng(self, scenario_key: str, body: dict[str, Any], salt: str):
        return stable_rng(
            self.profile.name, self.profile.seed, scenario_key, body.get("seed"), salt
        )

    def _failure_mode(self, rng) -> str:
        weights = self.profile.failure_weights
        modes = [m for m in FAILURE_MODES if weights.get(m, 0) > 0]
        if not modes:
            return "wrong_value"
        return rng.choices(modes, weights=[weights[m] for m in modes])[0]

    def _emit(self, calls: list[dict[str, Any]], mode: str | None, tools) -> MockReply:
        """Build a reply for ``calls`` with the failure ``mode`` applied (``None`` = correct)."""
        if mode is None:
            return MockReply(tool_calls=calls, finish_reason="tool_calls")
        if mode == "text_call":
            text = "\n".join(text_tool_call(c["name"], c.get("arguments", {})) for c in calls)
            return MockReply(content=text)
        if mode == "no_call":
            return MockReply(content="I'm not sure how to help with that.")
        mutated, broken = mutate_call(calls[0], tools, mode)
        out = [mutated, *calls[1:]]
        if broken is not None:
            out[0] = {"name": mutated["name"], "arguments": broken}
        return MockReply(tool_calls=out, finish_reason="tool_calls")

    # ------------------------------------------------------------------ scenarios

    def _calls(self, scenario: CallScenario, body: dict[str, Any]) -> MockReply:
        rng = self._rng(scenario.key, body, "calls")
        if rng.random() < self.profile.p_correct:
            return self._emit(scenario.calls, None, scenario.tools)
        return self._emit(scenario.calls, self._failure_mode(rng), scenario.tools)

    def _no_call(self, scenario: NoCallScenario, body: dict[str, Any]) -> MockReply:
        rng = self._rng(scenario.key, body, "nocall")
        if rng.random() < self.profile.p_correct:
            return MockReply(content=scenario.reply)
        return MockReply(tool_calls=[spurious_call(scenario.tools)], finish_reason="tool_calls")

    def _agent(self, scenario: AgentScenario, body: dict[str, Any]) -> MockReply:
        messages = body.get("messages") or []
        issued = _issued_calls(messages)
        tool_text = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "tool"
        )
        rng_inject = self._rng(scenario.key, body, "inject")
        rng_task = self._rng(scenario.key, body, "task")
        rng_mode = self._rng(scenario.key, body, "mode")
        injected = bool(scenario.payload) and (
            scenario.payload in tool_text
            or json.dumps(scenario.payload, ensure_ascii=False)[1:-1] in tool_text
        )
        attacker = scenario.attacker_call
        if attacker and injected and not any(_same_call(c, attacker) for c in issued):
            if rng_inject.random() < self.profile.p_inject:
                as_text = rng_mode.random() < self.profile.p_attack_text
                return self._emit([attacker], "text_call" if as_text else None, [])
        task_ok = rng_task.random() < self.profile.p_task_success
        pending = [c for c in scenario.task_calls if not any(_same_call(c, i) for i in issued)]
        if not task_ok:
            return MockReply(content="I was unable to complete that request.")
        if pending:
            return MockReply(tool_calls=[pending[0]], finish_reason="tool_calls")
        return MockReply(content=scenario.answer)

    def _text(self, scenario: TextScenario, body: dict[str, Any]) -> MockReply:
        rng = self._rng(scenario.key, body, "text")
        ok = rng.random() < self.profile.p_capability
        return MockReply(content=scenario.correct if ok else scenario.wrong)

    def _refusal(self, scenario: RefusalScenario, body: dict[str, Any]) -> MockReply:
        rng = self._rng(scenario.key, body, "refusal")
        p = self.profile.p_refuse_harmful if scenario.harmful else self.profile.p_refuse_benign
        pool = REFUSAL_TEXTS if rng.random() < p else COMPLIANCE_TEXTS
        return MockReply(content=rng.choice(pool))

    def _first_token(self, scenario: FirstTokenScenario, body: dict[str, Any], top: int):
        base_rng = stable_rng("klbase", scenario.key)
        noise_rng = stable_rng("klnoise", scenario.key)
        base = [base_rng.gauss(0.0, 1.5) for _ in FIRST_TOKENS]
        noise = [noise_rng.gauss(0.0, 1.0) for _ in FIRST_TOKENS]
        logits = [b + self.profile.kl_shift * n for b, n in zip(base, noise)]
        probs = _softmax(logits)
        ranked = sorted(zip(FIRST_TOKENS, probs), key=lambda pair: -pair[1])
        listing = [(t, math.log(p)) for t, p in ranked]
        reply = MockReply(content=listing[0][0], top_logprobs=top)
        if body.get("logprobs") and self.profile.logprobs_supported:
            reply.first_token_logprobs = listing[: max(top, 1)]
        return reply


def _issued_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for wire in message.get("tool_calls") or []:
            function = wire.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": function.get("name"), "arguments": args})
    return calls


def _same_call(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("name") == b.get("name") and a.get("arguments") == b.get("arguments")


# --------------------------------------------------------------------------- server


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def build_completion(reply: MockReply, body: dict[str, Any], serial: int) -> dict[str, Any]:
    """Assemble an OpenAI-format chat completion from a :class:`MockReply`."""
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        wire = []
        for i, call in enumerate(reply.tool_calls):
            arguments = call.get("arguments", {})
            wire.append(
                {
                    "id": f"call_{serial}_{i}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        message["tool_calls"] = wire
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": reply.finish_reason,
        "logprobs": None,
    }
    if reply.first_token_logprobs:
        top = reply.first_token_logprobs
        choice["logprobs"] = {
            "content": [
                {
                    "token": top[0][0],
                    "logprob": top[0][1],
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top],
                }
            ]
        }
        choice["message"]["content"] = top[0][0]
    prompt_text = " ".join(str(m.get("content") or "") for m in body.get("messages", []))
    tools_text = json.dumps(body.get("tools") or [])
    prompt_tokens = _count_tokens(prompt_text) + len(tools_text) // 4
    completion_tokens = _count_tokens(reply.content or "") + sum(
        len(json.dumps(c.get("arguments", {}))) // 4 + 1 for c in (reply.tool_calls or [])
    )
    return {
        "id": f"chatcmpl-mock-{serial}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "system_fingerprint": MOCK_FINGERPRINT,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@dataclass
class Fault:
    """A one-shot misbehaviour queued on the server."""

    kind: str  # "status", "drop", "garbage", "delay"
    status: int = 503
    body: str = "mock fault"
    retry_after: str | None = None
    delay_s: float = 0.0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RefusalscopeMock/1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def mock(self) -> MockServer:
        return self.server.mock  # type: ignore[attr-defined]

    def _send(self, status: int, payload: bytes, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(MOCK_HEADER, "1")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/v1/health"):
            self._send(200, b'{"status":"ok"}')
        elif path in ("/v1/models", "/models"):
            data = [
                {"id": name, "object": "model", "owned_by": "refusalscope-mock"}
                for name in self.mock.model_names
            ]
            self._send(200, json.dumps({"object": "list", "data": data}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, b'{"error":"not found"}')
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, b'{"error":"invalid JSON body"}')
            return
        self.mock._record(body, dict(self.headers.items()))
        fault = self.mock._next_fault()
        if fault is not None:
            if fault.kind == "drop":
                self.close_connection = True
                return
            if fault.kind == "status":
                extra = {"Retry-After": fault.retry_after} if fault.retry_after else None
                self._send(fault.status, fault.body.encode(), extra)
                return
            if fault.kind == "garbage":
                self._send(200, b"<html>not json</html>")
                return
            if fault.kind == "delay":
                time.sleep(fault.delay_s)
        if self.mock.latency_s:
            time.sleep(self.mock.latency_s)
        reply = self.mock.responder(body)
        payload = build_completion(reply, body, self.mock._serial())
        self._send(200, json.dumps(payload).encode("utf-8"))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MockServer:
    """Threaded mock server. Use as a context manager or call :meth:`start` / :meth:`stop`."""

    def __init__(
        self,
        responder: Callable[[dict[str, Any]], MockReply] | None = None,
        *,
        profile: MockProfile | None = None,
        provider: Provider | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        model_names: tuple[str, ...] = ("mock-model",),
        latency_s: float | None = None,
    ) -> None:
        self.profile = profile or PROFILES["perfect"]
        self.responder = responder or ProfileResponder(self.profile, provider)
        self.host = host
        self.model_names = model_names
        self.latency_s = self.profile.latency_s if latency_s is None else latency_s
        self._port = port
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._headers: list[dict[str, str]] = []
        self._faults: deque[Fault] = deque()
        self._count = 0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> MockServer:
        self._httpd = _Server((self.host, self._port), _Handler)
        self._httpd.mock = self  # type: ignore[attr-defined]
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> MockServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self._port}/v1"

    # ------------------------------------------------------------------ introspection

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    @property
    def request_headers(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._headers)

    def queue_fault(self, fault: Fault, times: int = 1) -> None:
        with self._lock:
            self._faults.extend([fault] * times)

    # ------------------------------------------------------------------ internals

    def _record(self, body: dict[str, Any], headers: dict[str, str]) -> None:
        with self._lock:
            self._requests.append(body)
            self._headers.append(headers)

    def _next_fault(self) -> Fault | None:
        with self._lock:
            return self._faults.popleft() if self._faults else None

    def _serial(self) -> int:
        with self._lock:
            self._count += 1
            return self._count


def serve_forever(
    profile_name: str, host: str = "127.0.0.1", port: int = 8099, provider: Provider | None = None
) -> None:  # pragma: no cover - blocking CLI helper
    """Run a mock server in the foreground until interrupted."""
    profile = PROFILES[profile_name]
    server = MockServer(profile=profile, provider=provider, host=host, port=port).start()
    print(f"MOCK server '{profile.name}' listening on {server.base_url} (no real model)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
