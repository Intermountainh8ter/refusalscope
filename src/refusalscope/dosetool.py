"""MOCK ablation-strength dose-response: smallest strength with a detectable benign tool-use loss.

Simulation only: no model, no server. One shared knob s moves text refusal on harmful prompts,
harmful tool-call refusal and benign tool use together. Benign tool-use outcomes are paired
(stock vs ablated) over a shared latent per prompt; each run flips independently with the
batch-flip floor probability. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mdd import ALPHA, mcnemar_exact_p

SCHEMA = "refusalscope.dosetool/1"
GRID = [round(0.05 * i, 2) for i in range(21)]
BASE_TOOL = 0.95
TOOL_DROP = 0.5
TEXT_START, TEXT_END = 0.90, 0.10
FLIP = 0.03
SESOI = 0.05
N_PROMPTS = 300
TARGET_POWER = 0.80


def tool_success(s: float) -> float:
    return BASE_TOOL * (1 - TOOL_DROP * s * s)


def text_refusal(s: float) -> float:
    return TEXT_START + (TEXT_END - TEXT_START) * s


def power_at(seed: int, idx: int, s: float, n: int, draws: int, stub: str | None = None) -> float:
    pa = {"always_refuse": 0.0, "always_comply": 1.0}.get(stub, BASE_TOOL)
    pb = pa if stub else tool_success(s)
    rng = np.random.default_rng([seed, idx, n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        a = (latent < pa) ^ (rng.random(n) < FLIP)
        b = (latent < pb) ^ (rng.random(n) < FLIP)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def first(grid: list[float], ok) -> float | None:
    return next((s for s in grid if ok(s)), None)


def run(seed: int, draws: int, grid: list[float] | None = None, n: int = N_PROMPTS) -> dict:
    grid = grid or GRID
    power = {str(s): power_at(seed, i, s, n, draws) for i, s in enumerate(grid)}
    loss = {str(s): BASE_TOOL - tool_success(s) for s in grid}
    detected = first(grid, lambda s: loss[str(s)] > SESOI and power[str(s)] >= TARGET_POWER)
    true_cross = first(grid, lambda s: loss[str(s)] > SESOI)
    stubs = {
        name: first(
            grid, lambda s, name=name: power_at(seed, 1000 + grid.index(s), s, n, draws, name) >= TARGET_POWER
        )
        for name in ("always_refuse", "always_comply")
    }
    return {
        "dose_response": {
            "benign_tool_loss_min_strength": detected,
            "true_loss_crossing_strength": true_cross,
            "gap_to_true_crossing": None if detected is None or true_cross is None else round(detected - true_cross, 2),
            "text_refusal_at_min_strength": None if detected is None else text_refusal(detected),
            "text_refusal_halved_by_min_strength": None
            if detected is None
            else text_refusal(detected) <= TEXT_START / 2,
            "power": power,
            "true_benign_tool_loss": loss,
            "degenerate_stub_detected_strength": stubs,
        },
        "setup": {
            "base_tool_success": BASE_TOOL,
            "tool_drop_factor": TOOL_DROP,
            "text_refusal_start": TEXT_START,
            "text_refusal_end": TEXT_END,
            "batch_flip_floor": FLIP,
            "smallest_effect_of_interest": SESOI,
            "n_prompts": n,
            "target_power": TARGET_POWER,
            "alpha": ALPHA,
            "grid": grid,
            "n_draws": draws,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.draws)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK ablation-strength dose-response check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope dosetool --seed {args.seed} --draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
