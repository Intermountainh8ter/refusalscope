"""Small shared helpers: hashing, stable randomness, JSON Lines, text clean-up."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no extra whitespace, non-ASCII preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    """Hash any JSON-serialisable parts into a short stable hex string."""
    return sha256_text(canonical_json(list(parts)))[:16]


def stable_rng(*parts: Any) -> random.Random:
    """A `random.Random` seeded deterministically from the given parts.

    Unlike `hash()`, the seed does not depend on the interpreter session, so replays of the
    same request produce the same draw on any machine.
    """
    return random.Random(int(stable_hash(*parts), 16))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSON Lines file, skipping blank lines."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object per line")
            yield value


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as canonical-ish JSON Lines (stable key order) and return the row count."""
    count = 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str | None) -> str:
    """Remove reasoning blocks from model output so graders see only the visible answer.

    Handles three shapes: a closed ``<think>...</think>`` block, an unterminated block
    (the model ran out of tokens while thinking), and a bare closing tag when the chat
    template already opened the block in the prompt.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    lower = cleaned.lower()
    open_at = lower.find("<think>")
    if open_at != -1:
        cleaned = cleaned[:open_at]
    else:
        close_at = lower.rfind("</think>")
        if close_at != -1:
            cleaned = cleaned[close_at + len("</think>") :]
    return cleaned.strip()


def percent(part: float, whole: float) -> float:
    return 0.0 if whole == 0 else 100.0 * part / whole
