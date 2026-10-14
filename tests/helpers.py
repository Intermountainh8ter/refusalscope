"""Shared test helpers (importable as ``tests.helpers``)."""

from __future__ import annotations

from refusalscope.config import StudyConfig, parse_config

PLACEHOLDERS = [f"Test fixture request number {i:03d} (not a real behaviour)." for i in range(60)]


def make_study(urls: dict[str, str], behaviours=None, **overrides) -> StudyConfig:
    """A mock study config. ``urls`` maps model name -> base_url; the first is stock, second ablated."""
    names = list(urls)
    raw = {
        "study": "unit",
        "models": [
            {"name": names[0], "base_url": urls[names[0]], "role": "stock", "pair_id": "p", "params_b": 4,
             "quant": "mock", "mock": True},
            {"name": names[1], "base_url": urls[names[1]], "role": "ablated", "pair_id": "p", "params_b": 4,
             "quant": "mock", "mock": True},
        ],
        "sampling": {"temperatures": [0.0], "seeds": [1], "max_tokens": 128, "timeout_s": 10, "retries": 0},
        "suites": {"refusal": {"harmful_path": str(behaviours) if behaviours else ""}},
        "workers": 4,
    }  # fmt: skip
    raw.update(overrides)
    return parse_config(raw)
