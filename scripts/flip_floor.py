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


def make_slot_responder(seed: int, jitter: float):
    """Jitter only when n_parallel is 4 and no id_slot is pinned (a server-chosen slot)."""

    def responder(body: dict) -> MockReply:
        text = body["messages"][-1]["content"]
        case, draw = DRAW_TAG.search(text).groups()
        refuse = stable_rng("flip-base", seed, case).random() < BASE_P_REFUSE
        unpinned = body.get("n_parallel") == 4 and body.get("id_slot") is None
        if unpinned and stable_rng("flip-slot-jitter", seed, case, draw).random() < jitter:
            refuse = False
        pool = REFUSAL_TEXTS if refuse else COMPLIANCE_TEXTS
        return MockReply(content=stable_rng("flip-text", seed, case, draw, bool(refuse)).choice(pool))

    return responder


def make_template_responder(seed: int, jitter: float):
    """Jitter only when the chat-template variant is B."""

    def responder(body: dict) -> MockReply:
        text = body["messages"][-1]["content"]
        case, draw = DRAW_TAG.search(text).groups()
        refuse = stable_rng("flip-base", seed, case).random() < BASE_P_REFUSE
        if body.get("template") == "B" and stable_rng("flip-template-jitter", seed, case, draw).random() < jitter:
            refuse = False
        pool = REFUSAL_TEXTS if refuse else COMPLIANCE_TEXTS
        return MockReply(content=stable_rng("flip-text", seed, case, draw, bool(refuse)).choice(pool))

    return responder


def make_batch_responder(seed: int, jitter: float):
    """Jitter only when batch size is 8 and batch-invariant kernels are off."""

    def responder(body: dict) -> MockReply:
        text = body["messages"][-1]["content"]
        case, draw = DRAW_TAG.search(text).groups()
        refuse = stable_rng("flip-base", seed, case).random() < BASE_P_REFUSE
        varies = body.get("batch_size") == 8 and not body.get("batch_invariant")
        if varies and stable_rng("flip-batch-jitter", seed, case, draw).random() < jitter:
            refuse = False
        pool = REFUSAL_TEXTS if refuse else COMPLIANCE_TEXTS
        return MockReply(content=stable_rng("flip-text", seed, case, draw, bool(refuse)).choice(pool))

    return responder


def ask(url: str, prompt: str, cache_prompt: bool, extra: dict | None = None) -> bool:
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "cache_prompt": cache_prompt,
        **(extra or {}),
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


def run_slots(seed: int, draws: int, jitter: float, n_boot: int) -> dict:
    items = borderline_items()
    reps = draws // len(items)
    arms: dict[str, list[bool]] = {"slots_1": [], "slots_1_repeat": [], "slots_4": [], "slots_4_pinned": []}
    with MockServer(responder=make_slot_responder(seed, jitter)) as server:
        for rep in range(reps):
            for item in items:
                prompt = f"{item['prompt']} [case {item['id']} draw {rep}]"
                one = {"n_parallel": 1, "id_slot": 0}
                arms["slots_1"].append(ask(server.base_url, prompt, False, one))
                arms["slots_1_repeat"].append(ask(server.base_url, prompt, False, one))
                arms["slots_4"].append(ask(server.base_url, prompt, False, {"n_parallel": 4}))
                arms["slots_4_pinned"].append(
                    ask(server.base_url, prompt, False, {"n_parallel": 4, "id_slot": 0})
                )
    return {
        "slots_1_vs_1": flip_stats(arms["slots_1"], arms["slots_1_repeat"], n_boot, seed),
        "slots_4_vs_1": flip_stats(arms["slots_1"], arms["slots_4"], n_boot, seed),
        "slots_4_pinned_vs_1": flip_stats(arms["slots_1"], arms["slots_4_pinned"], n_boot, seed),
        "refusal_rate": {k: float(np.mean(v)) for k, v in arms.items()},
        "injected": {"jitter": jitter, "base_p_refuse": BASE_P_REFUSE, "direction": "refusal->compliance",
                     "applies_when": "n_parallel == 4 and no id_slot"},
    }


def run_template(seed: int, draws: int, jitter: float, n_boot: int) -> dict:
    items = borderline_items()
    reps = draws // len(items)
    arms: dict[str, list[bool]] = {"template_a": [], "template_a_repeat": [], "template_b": []}
    with MockServer(responder=make_template_responder(seed, jitter)) as server:
        for rep in range(reps):
            for item in items:
                prompt = f"{item['prompt']} [case {item['id']} draw {rep}]"
                a = {"template": "A"}
                arms["template_a"].append(ask(server.base_url, prompt, False, a))
                arms["template_a_repeat"].append(ask(server.base_url, prompt, False, a))
                arms["template_b"].append(ask(server.base_url, prompt, False, {"template": "B"}))
    return {
        "template_a_vs_a": flip_stats(arms["template_a"], arms["template_a_repeat"], n_boot, seed),
        "template_b_vs_a": flip_stats(arms["template_a"], arms["template_b"], n_boot, seed),
        "refusal_rate": {k: float(np.mean(v)) for k, v in arms.items()},
        "injected": {"jitter": jitter, "base_p_refuse": BASE_P_REFUSE, "direction": "refusal->compliance",
                     "applies_when": "template == B"},
    }


def run_batch(seed: int, draws: int, jitter: float, n_boot: int) -> dict:
    items = borderline_items()
    reps = draws // len(items)
    names = ("batch_1", "batch_1_repeat", "batch_8", "batch_8_invariant")
    arms: dict[str, list[bool]] = {k: [] for k in names}
    with MockServer(responder=make_batch_responder(seed, jitter)) as server:
        for rep in range(reps):
            for item in items:
                prompt = f"{item['prompt']} [case {item['id']} draw {rep}]"
                one = {"batch_size": 1}
                arms["batch_1"].append(ask(server.base_url, prompt, False, one))
                arms["batch_1_repeat"].append(ask(server.base_url, prompt, False, one))
                arms["batch_8"].append(ask(server.base_url, prompt, False, {"batch_size": 8}))
                arms["batch_8_invariant"].append(
                    ask(server.base_url, prompt, False, {"batch_size": 8, "batch_invariant": True})
                )
    floor = flip_stats(arms["batch_1"], arms["batch_1_repeat"], n_boot, seed)
    paired = flip_stats(arms["batch_1"], arms["batch_8"], n_boot, seed)
    invariant = flip_stats(arms["batch_1"], arms["batch_8_invariant"], n_boot, seed)
    return {
        "batch": {
            "repeat_floor": floor["flip_rate"],
            "paired_flip": paired["flip_rate"],
            "paired_flip_over_floor": paired["flip_rate"] - floor["flip_rate"],
            "invariant_flip": invariant["flip_rate"],
        },
        "batch_1_vs_1": floor,
        "batch_8_vs_1": paired,
        "batch_8_invariant_vs_1": invariant,
        "refusal_rate": {k: float(np.mean(v)) for k, v in arms.items()},
        "injected": {"jitter": jitter, "base_p_refuse": BASE_P_REFUSE, "direction": "refusal->compliance",
                     "applies_when": "batch_size == 8 and batch_invariant is off"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="required: only the MOCK server is supported")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--slots", action="store_true", help="parallel-slot dimension instead of cache_prompt")
    parser.add_argument("--template", action="store_true", help="chat-template-variant dimension")
    parser.add_argument("--batch", action="store_true", help="batch-size dimension with a batch-invariant arm")
    parser.add_argument("--jitter", type=float, default=None, help="default 0.05 (cache) or 0.06 (--slots)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.mock:
        print("flip_floor.py only supports --mock; real runs are not implemented", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    if args.batch:
        jitter = 0.06 if args.jitter is None else args.jitter
        result = run_batch(args.seed, args.draws, jitter, args.n_boot)
        label = "MOCK flip-floor batch-size estimator check"
        rates = {
            "batch_8_vs_1": result["batch"]["paired_flip"],
            "batch_8_invariant_vs_1": result["batch"]["invariant_flip"],
            "batch_1_vs_1": result["batch"]["repeat_floor"],
        }
    elif args.template:
        jitter = 0.07 if args.jitter is None else args.jitter
        result = run_template(args.seed, args.draws, jitter, args.n_boot)
        label = "MOCK flip-floor chat-template variant estimator check"
        rates = {
            "template_b_vs_a": result["template_b_vs_a"]["flip_rate"],
            "template_a_vs_a": result["template_a_vs_a"]["flip_rate"],
        }
    elif args.slots:
        jitter = 0.06 if args.jitter is None else args.jitter
        result = run_slots(args.seed, args.draws, jitter, args.n_boot)
        label = "MOCK flip-floor parallel-slot estimator check"
        rates = {
            "slots_4_vs_1": result["slots_4_vs_1"]["flip_rate"],
            "slots_4_pinned_vs_1": result["slots_4_pinned_vs_1"]["flip_rate"],
            "slots_1_vs_1": result["slots_1_vs_1"]["flip_rate"],
        }
    else:
        jitter = 0.05 if args.jitter is None else args.jitter
        result = run(args.seed, args.draws, jitter, args.n_boot)
        label = "MOCK flip-floor cache_prompt estimator check"
        rates = {
            "cache_on_vs_off": result["cache_on_vs_off"]["flip_rate"],
            "cache_off_vs_off": result["cache_off_vs_off"]["flip_rate"],
        }
    doc = {
        "schema": SCHEMA,
        "mock": True,
        "label": label,
        "refusal_label_flip_rate": rates,
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
