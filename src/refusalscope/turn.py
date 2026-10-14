"""A model turn reduced to what graders need, rebuilt from live results or stored records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from refusalscope.client import ChatResult, ToolCall, parse_completion


@dataclass
class Turn:
    """One assistant response: visible text plus structured tool calls."""

    ok: bool
    error: str | None = None
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    logprobs: list[dict[str, Any]] | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: ChatResult) -> Turn:
        return cls(
            ok=result.ok,
            error=result.error,
            content=result.content,
            tool_calls=list(result.tool_calls),
            finish_reason=result.finish_reason,
            logprobs=result.logprobs,
            usage=dict(result.usage),
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Turn:
        """Rebuild a turn from the ``record()`` of a stored exchange (no network needed)."""
        if not record.get("ok") or not isinstance(record.get("response"), dict):
            return cls(ok=False, error=record.get("error") or "no response recorded")
        try:
            parsed = parse_completion(record["response"])
        except ValueError as exc:
            return cls(ok=False, error=f"unusable stored response: {exc}")
        return cls(
            ok=True,
            content=parsed["content"],
            tool_calls=parsed["tool_calls"],
            finish_reason=parsed["finish_reason"],
            logprobs=parsed["logprobs"],
            usage=parsed["usage"],
        )
