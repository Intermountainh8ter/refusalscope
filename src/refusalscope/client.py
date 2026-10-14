"""OpenAI-compatible chat completions client built on urllib.

Works against llama.cpp ``llama-server``, vLLM, Ollama's ``/v1`` endpoint and the bundled
mock server. It never raises on transport problems: failures are returned inside
:class:`ChatResult` so a long study can record them and continue.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from refusalscope import __version__
from refusalscope.util import strip_think

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MOCK_HEADER = "X-Refusalscope-Mock"
MOCK_FINGERPRINT = "refusalscope-mock"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})


def parse_arguments(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the ``arguments`` string of a tool call.

    Returns ``(arguments, None)`` on success and ``(None, reason)`` when the text is not a JSON
    object. An empty string is accepted as an empty argument object because several servers
    send it for functions without parameters.
    """
    if raw is None:
        return {}, None
    text = raw.strip()
    if text == "":
        return {}, None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at position {exc.pos}"
    if not isinstance(value, dict):
        return None, f"arguments JSON is a {type(value).__name__}, not an object"
    return value, None


@dataclass
class ToolCall:
    """One structured tool call as returned by the server."""

    id: str
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None
    parse_error: str | None

    @classmethod
    def from_wire(cls, wire: dict[str, Any], index: int = 0) -> ToolCall:
        function = wire.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments")
        if isinstance(raw, dict):
            raw = json.dumps(raw, ensure_ascii=False)
        elif raw is None:
            raw = ""
        elif not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        arguments, error = parse_arguments(raw)
        return cls(
            id=str(wire.get("id") or f"call_{index}"),
            name=name if isinstance(name, str) else "",
            arguments_raw=raw,
            arguments=arguments,
            parse_error=error,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_raw},
        }


@dataclass
class ChatResult:
    """Normalised outcome of one chat completion request."""

    ok: bool
    content: str = ""
    raw_content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    logprobs: list[dict[str, Any]] | None = None
    latency_s: float = 0.0
    elapsed_s: float = 0.0
    attempts: int = 1
    status: int | None = None
    error: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    mock: bool = False

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn to append to a conversation after this response."""
        message: dict[str, Any] = {"role": "assistant", "content": self.raw_content or None}
        if self.tool_calls:
            message["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        return message

    def record(self) -> dict[str, Any]:
        """A JSON-serialisable record of the exchange for trajectory logs."""
        return {
            "request": self.request,
            "response": self.response,
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "latency_s": round(self.latency_s, 6),
            "elapsed_s": round(self.elapsed_s, 6),
            "attempts": self.attempts,
            "usage": self.usage,
            "mock": self.mock,
        }


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _extract_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    block = choice.get("logprobs")
    if not isinstance(block, dict):
        return None
    content = block.get("content")
    if not isinstance(content, list) or not content:
        return None
    out: list[dict[str, Any]] = []
    for entry in content:
        if not isinstance(entry, dict) or "logprob" not in entry:
            continue
        top = [
            {"token": str(alt.get("token", "")), "logprob": float(alt["logprob"])}
            for alt in entry.get("top_logprobs") or []
            if isinstance(alt, dict) and "logprob" in alt
        ]
        out.append(
            {
                "token": str(entry.get("token", "")),
                "logprob": float(entry["logprob"]),
                "top": top,
            }
        )
    return out or None


def parse_completion(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields graders need from a chat completion body.

    Raises ``ValueError`` when the body has no usable ``choices[0].message``.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message")
    raw_content = _flatten_content(message.get("content"))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    wire_calls = message.get("tool_calls") or []
    calls = [ToolCall.from_wire(w, i) for i, w in enumerate(wire_calls) if isinstance(w, dict)]
    usage_raw = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
    return {
        "raw_content": raw_content,
        "content": strip_think(raw_content),
        "reasoning": reasoning if isinstance(reasoning, str) else "",
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "logprobs": _extract_logprobs(choice),
    }


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host.lower() in LOOPBACK_HOSTS or host.startswith("127.")


class ChatClient:
    """Chat completions client for one model on one endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        retries: int = 3,
        backoff_s: float = 0.5,
        max_backoff_s: float = 8.0,
        extra_body: dict[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://: {base_url!r}")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "sk-no-key-required"
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.extra_body = dict(extra_body or {})
        self._sleep = sleep
        self._clock = clock
        # A system proxy would otherwise intercept traffic to a local llama-server.
        handlers: list[urllib.request.BaseHandler] = []
        if _is_loopback(self.base_url):
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    # ------------------------------------------------------------------ low level

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"refusalscope/{__version__}",
        }

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        delay = min(self.backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), self.max_backoff_s))
            except ValueError:
                pass
        return delay

    def _post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=self._headers(), method="POST"
        )
        with self._opener.open(request, timeout=self.timeout_s) as handle:
            payload = handle.read()
            headers = {k.lower(): v for k, v in handle.headers.items()}
            return handle.status, json.loads(payload.decode("utf-8")), headers

    # ------------------------------------------------------------------ public API

    def get_json(self, path: str) -> tuple[int, Any, dict[str, str]]:
        """GET a JSON document (used for ``/models``). Raises on transport failure."""
        request = urllib.request.Request(
            f"{self.base_url}{path}", headers=self._headers(), method="GET"
        )
        with self._opener.open(request, timeout=self.timeout_s) as handle:
            payload = handle.read()
            headers = {k.lower(): v for k, v in handle.headers.items()}
            return handle.status, json.loads(payload.decode("utf-8")), headers

    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                body["parallel_tool_calls"] = parallel_tool_calls
        if logprobs:
            body["logprobs"] = True
            if top_logprobs:
                body["top_logprobs"] = top_logprobs
        body.update(self.extra_body)
        if extra:
            body.update(extra)
        return body

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        """Send one chat completion request, retrying transient failures."""
        body = self.build_request(messages, **kwargs)
        started = self._clock()
        attempts = 0
        last_error = "no attempt made"
        last_status: int | None = None
        while attempts <= self.retries:
            attempts += 1
            attempt_start = self._clock()
            retry_after: str | None = None
            try:
                status, payload, headers = self._post("/chat/completions", body)
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if exc.code not in RETRY_STATUS:
                    break
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                last_status = None
                reason = getattr(exc, "reason", exc)
                last_error = f"{type(exc).__name__}: {reason}"
            else:
                try:
                    parsed = parse_completion(payload)
                except ValueError as exc:
                    last_status = status
                    last_error = f"unusable response: {exc}"
                    break
                latency = self._clock() - attempt_start
                mock = (
                    headers.get(MOCK_HEADER.lower()) == "1"
                    or payload.get("system_fingerprint") == MOCK_FINGERPRINT
                )
                return ChatResult(
                    ok=True,
                    latency_s=latency,
                    elapsed_s=self._clock() - started,
                    attempts=attempts,
                    status=status,
                    request=body,
                    response=payload,
                    mock=mock,
                    **parsed,
                )
            if attempts > self.retries:
                break
            self._sleep(self._delay(attempts, retry_after))
        return ChatResult(
            ok=False,
            latency_s=self._clock() - started,
            elapsed_s=self._clock() - started,
            attempts=attempts,
            status=last_status,
            error=last_error,
            request=body,
        )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        text = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - reading an already-closed error body
        text = ""
    text = " ".join(text.split())
    return text[:500] if text else (exc.reason if isinstance(exc.reason, str) else "")
