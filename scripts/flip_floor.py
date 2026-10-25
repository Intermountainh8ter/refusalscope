#!/usr/bin/env python3
"""Paired refusal-label flip rate between two serving settings (noise-floor estimator).

Runs against the bundled MOCK server only. The mock re-labels a reply with a seeded jitter that
is applied only when ``cache_prompt`` is on, so the estimator is validated against a known
injected effect before any real llama.cpp run. Output is a MOCK-labelled estimator check, not
evidence about any model.

    python scripts/flip_floor.py --mock --seed 11 --out results/45-flip-floor-cache-mock.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusalscope.mock import COMPLIANCE_TEXTS, REFUSAL_TEXTS, MockReply, MockServer  # noqa: E402
from refusalscope.stats import mcnemar_exact  # noqa: E402
from refusalscope.suites.refusal import borderline_items, classify_refusal  # noqa: E402
from refusalscope.util import stable_rng  # noqa: E402

SCHEMA = "refusalscope.flip-floor/1"
BASE_P_REFUSE = 0.8
DRAW_TAG = re.compile(r"\[case (\S+) draw (\d+)\]")


def flip_stats(a: list[bool], b: list[bool], n_boot: int, seed: int) -> dict:
    """Flip rate of two paired boolean label vectors, exact McNemar and a seeded bootstrap CI."""
    if len(a) != len(b) or not a:
        raise ValueError("arms must be non-empty and the same length")
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    flips = (x != y).astype(float)
    n_ab = int((x & ~y).sum())  # refusal only in the first arm
    n_ba = int((~x & y).sum())
    idx = np.random.default_rng(seed).integers(0, flips.size, size=(n_boot, flips.size))
    samples = flips[idx].mean(axis=1)
    return {
        "n": int(flips.size),
        "flip_rate": float(flips.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n_boot": n_boot,
        "refusal_only_in_first": n_ab,
        "refusal_only_in_second": n_ba,
        "mcnemar": mcnemar_exact(n_ab, n_ba),
    }


def make_responder(seed: int, jitter: float):
    def responder(body: dict) -> MockReply:
        text = body["messages"][-1]["content"]
        case, draw = DRAW_TAG.search(text).groups()
        refuse = stable_rng("flip-base", seed, case).random() < BASE_P_REFUSE
        if body.get("cache_prompt") and stable_rng("flip-jitter", seed, case, draw).random() < jitter:
            refuse = False
        pool = REFUSAL_TEXTS if refuse else COMPLIANCE_TEXTS
        return MockReply(content=stable_rng("flip-text", seed, case, draw, bool(refuse)).choice(pool))

    return responder


def ask(url: str, prompt: str, cache_prompt: bool) -> bool:
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "cache_prompt": cache_prompt,
    }
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = json.load(resp)["choices"][0]["message"]["content"]
    return classify_refusal(content)["label"] == "refusal"


def run(seed: int, draws: int, jitter: float, n_boot: int) -> dict:
    items = borderline_items()
    reps = draws // len(items)
    arms: dict[str, list[bool]] = {"off": [], "off_repeat": [], "on": []}
    with MockServer(responder=make_responder(seed, jitter)) as server:
        for rep in range(reps):
            for item in items:
                prompt = f"{item['prompt']} [case {item['id']} draw {rep}]"
                arms["off"].append(ask(server.base_url, prompt, False))
                arms["off_repeat"].append(ask(server.base_url, prompt, False))
                arms["on"].append(ask(server.base_url, prompt, True))
    return {
        "cache_off_vs_off": flip_stats(arms["off"], arms["off_repeat"], n_boot, seed),
        "cache_on_vs_off": flip_stats(arms["off"], arms["on"], n_boot, seed),
        "refusal_rate": {k: float(np.mean(v)) for k, v in arms.items()},
        "injected": {"jitter": jitter, "base_p_refuse": BASE_P_REFUSE, "direction": "refusal->compliance"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="required: only the MOCK server is supported")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.mock:
        print("flip_floor.py only supports --mock; real runs are not implemented", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    result = run(args.seed, args.draws, args.jitter, args.n_boot)
    doc = {
        "schema": SCHEMA,
        "mock": True,
        "label": "MOCK flip-floor cache_prompt estimator check",
        "refusal_label_flip_rate": {
            "cache_on_vs_off": result["cache_on_vs_off"]["flip_rate"],
            "cache_off_vs_off": result["cache_off_vs_off"]["flip_rate"],
        },
        **result,
        "provenance": {
            "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seeds": [args.seed],
            "command": " ".join(["python3", "scripts/flip_floor.py", *sys.argv[1:]]),
            "hardware": f"{platform.machine()} cpu, {platform.system()}",
            "wall_s": round(time.monotonic() - t0, 3),
            "data_sources": [
                {
                    "url": "https://arxiv.org/abs/2605.27763",
                    "license": "arXiv paper cited for method only; no data used; run uses bundled MOCK server",
                }
            ],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
