"""Preflight checks against each configured endpoint (``refusalscope doctor``).

Every failure carries a concrete fix. The checks are the real symptoms people hit when serving a
stock and an ablated model side by side; the troubleshooting flowchart in ``docs/diagrams``
mirrors them.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass

from refusalscope.client import MOCK_HEADER, ChatClient
from refusalscope.config import ModelConfig, StudyConfig, pair_warnings
from refusalscope.mock import CallScenario, FirstTokenScenario, Scenario
from refusalscope.suites import toolcall
from refusalscope.textcalls import detect_text_calls

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


PROBE_PROMPT = "What's the weather in Paris?"
LOGPROBS_PROMPT = "Say hello."


class DoctorOracle:
    """Lets the mock server answer the doctor's own probes like a healthy model would."""

    def lookup(self, body: dict) -> Scenario | None:
        users = [m for m in body.get("messages") or [] if m.get("role") == "user"]
        prompt = users[-1].get("content") if users else None
        if prompt == PROBE_PROMPT and body.get("tools"):
            call = {"name": "get_weather", "arguments": {"city": "Paris"}}
            return CallScenario("doctor-weather", [call], [WEATHER_TOOL])
        if prompt == LOGPROBS_PROMPT and body.get("logprobs"):
            return FirstTokenScenario("doctor-hello")
        return None


@dataclass
class Check:
    """One preflight result. ``status`` is ``ok``, ``warn`` or ``fail``."""

    target: str
    name: str
    status: str
    message: str
    fix: str = ""

    def line(self) -> str:
        tag = {"ok": "  OK ", "warn": " WARN", "fail": "FAIL "}[self.status]
        text = f"[{tag}] {self.target}: {self.name} - {self.message}"
        return text + (f"\n         fix: {self.fix}" if self.fix and self.status != "ok" else "")


def _explain_error(error: str, status: int | None) -> tuple[str, str]:
    lowered = error.lower()
    if "connection refused" in lowered or "errno 111" in lowered or "10061" in lowered:
        return "connection refused", (
            "start the server for this model (README quickstart) and check the port in base_url"
        )
    if "timed out" in lowered or "timeout" in lowered:
        return "request timed out", (
            "the model may still be loading or the machine is overloaded; raise sampling.timeout_s "
            "or wait for the server log to say it is listening"
        )
    if status in (401, 403):
        return (
            f"HTTP {status}",
            "set api_key_env to the name of an environment variable holding the key",
        )
    if status == 404:
        return "HTTP 404", "base_url should end with /v1 (for example http://127.0.0.1:8080/v1)"
    if "context" in lowered and status == 400:
        return "context size exceeded", (
            "restart the server with a larger context (llama-server --ctx-size 8192 or higher)"
        )
    return error[:160], "see the server log for the cause"


def check_model(model: ModelConfig, client: ChatClient, top_logprobs: int = 5) -> list[Check]:
    """Run every endpoint check for one model."""
    out: list[Check] = []
    name = model.name

    # 1. reachability and mock marker
    try:
        _, _, headers = client.get_json("/models")
    except urllib.error.HTTPError as exc:
        reason, fix = _explain_error(str(exc), exc.code)
        out.append(Check(name, "endpoint", "fail", reason, fix))
        return out
    except Exception as exc:
        reason, fix = _explain_error(str(getattr(exc, "reason", exc)), None)
        out.append(Check(name, "endpoint", "fail", reason, fix))
        return out
    out.append(Check(name, "endpoint", "ok", f"{model.base_url} answers /models"))
    announces_mock = headers.get(MOCK_HEADER.lower()) == "1"
    if announces_mock != model.mock:
        out.append(
            Check(
                name,
                "mock flag",
                "fail",
                "endpoint announces the MOCK server"
                if announces_mock
                else 'marked "mock": true but the endpoint is real',
                'set "mock" in the configuration to match the endpoint',
            )
        )
    elif announces_mock:
        out.append(
            Check(name, "mock flag", "ok", "MOCK endpoint: every output will be labelled MOCK")
        )

    # 2. plain chat
    chat = client.chat(
        [{"role": "user", "content": "Reply with the single word OK."}],
        max_tokens=64,
        temperature=0.0,
    )
    if not chat.ok:
        reason, fix = _explain_error(chat.error or "", chat.status)
        out.append(Check(name, "chat", "fail", reason, fix))
        return out
    if not chat.content and chat.finish_reason == "length":
        out.append(
            Check(
                name,
                "chat",
                "warn",
                "empty reply with finish_reason=length",
                'the model spent its tokens thinking; add {"chat_template_kwargs": {"enable_thinking": false}} '
                "to extra_body or raise sampling.max_tokens",
            )
        )
    else:
        out.append(Check(name, "chat", "ok", f"replied in {chat.latency_s:.2f}s"))
    if "<think>" in chat.raw_content:
        out.append(
            Check(name, "thinking", "warn", "model emits <think> blocks (stripped before grading)",
                  "set enable_thinking false in extra_body to save tokens and time"),
        )  # fmt: skip

    # 3. structured tool calls versus text
    probe = client.chat(
        [{"role": "user", "content": PROBE_PROMPT}],
        tools=[WEATHER_TOOL], tool_choice="auto", max_tokens=200, temperature=0.0,
    )  # fmt: skip
    if not probe.ok:
        reason, fix = _explain_error(probe.error or "", probe.status)
        out.append(Check(name, "tool calls", "fail", reason, fix))
    elif probe.tool_calls:
        bad = [c for c in probe.tool_calls if c.parse_error]
        status = "warn" if bad else "ok"
        out.append(Check(name, "tool calls", status, "structured tool_calls returned" if not bad else "arguments are not valid JSON",
                         "check the chat template and quantisation" if bad else ""))  # fmt: skip
    elif detect_text_calls(probe.content, {"get_weather"}):
        out.append(
            Check(
                name,
                "tool calls",
                "fail",
                "the model wrote the tool call as plain text instead of a structured tool_calls entry",
                "restart llama-server with --jinja (and a model whose chat template supports tools); "
                "vLLM needs --enable-auto-tool-choice and --tool-call-parser",
            )
        )
    else:
        out.append(
            Check(name, "tool calls", "warn", "the model answered in text and made no tool call",
                  "check that the chat template supports tools; try the same prompt in the server's web UI"),
        )  # fmt: skip

    # 4. context size, using the largest Suite A request
    biggest = max(toolcall.build_cases(per_category=3), key=lambda c: len(json.dumps(c["tools"])))
    ctx = client.chat(biggest["messages"], tools=biggest["tools"], max_tokens=16, temperature=0.0)
    if ctx.ok:
        out.append(Check(name, "context", "ok", "a request with 16 tool definitions fits"))
    else:
        reason, fix = _explain_error(ctx.error or "", ctx.status)
        out.append(Check(name, "context", "fail", reason, fix))

    # 5. logprobs for the KL probe
    lp = client.chat([{"role": "user", "content": LOGPROBS_PROMPT}], max_tokens=1, logprobs=True,
                     top_logprobs=top_logprobs, temperature=1.0)  # fmt: skip
    if lp.ok and lp.logprobs and lp.logprobs[0]["top"]:
        out.append(
            Check(name, "logprobs", "ok", f"top-{len(lp.logprobs[0]['top'])} logprobs returned")
        )
    else:
        out.append(
            Check(name, "logprobs", "warn", "no top_logprobs in the response",
                  "the KL probe will be reported as not measured; use llama-server or vLLM if you need it"),
        )  # fmt: skip
    return out


def run_doctor(
    config: StudyConfig, client_factory: Callable[[ModelConfig], ChatClient] | None = None
) -> list[Check]:
    """Check every model, then the pair-level consistency warnings."""
    sampling = config.sampling
    factory = client_factory or (
        lambda m: ChatClient(
            m.base_url, m.served_name, api_key=m.api_key(), timeout_s=sampling.timeout_s,
            retries=0, extra_body=m.extra_body,
        )
    )  # fmt: skip
    checks: list[Check] = []
    for model in config.models:
        checks.extend(check_model(model, factory(model), config.suites.kl_top_logprobs))
    checks.extend(
        Check(
            "pairs",
            "consistency",
            "warn",
            warning,
            "make both arms identical except for the weights",
        )
        for warning in pair_warnings(config)
    )
    return checks
