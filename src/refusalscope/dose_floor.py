"""MOCK dose-response of ablation strength against a mock serving flip floor.

Simulation only: no model, no server. A seeded logistic curve gives the probability that a
refusal is suppressed at each ablation dose. Output is an estimator check, not evidence about
any model.

    PYTHONPATH=src python3 -m refusalscope.dose_floor --seed 7 --out results/155-dose-flip-floor-sigmoid-mock.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .stats import mcnemar_exact
from .suites.refusal import borderline_items
from .util import stable_rng

SCHEMA = "refusalscope.dose-flip-floor/1"
BASE_P_REFUSE = 0.8
FLOOR = 0.07
MIDPOINT = 0.5
STEEPNESS = 10.0
GRID = [round(i / 10, 1) for i in range(11)]


def suppression(dose: float) -> float:
    """Logistic probability that a refusal is suppressed at this dose."""
    return 1.0 / (1.0 + math.exp(-STEEPNESS * (dose - MIDPOINT)))


def labels(seed: int, dose: float, draws: int, tag: str = "") -> list[bool]:
    """Refusal labels for ``draws`` draws spread over the borderline cases."""
    items = borderline_items()
    out = []
    for k in range(draws):
        case = items[k % len(items)]["id"]
        rep = k // len(items)
        refuse = stable_rng("dose-base", seed, case).random() < BASE_P_REFUSE
        if refuse and dose > 0 and stable_rng("dose-jit", seed, case, rep, dose, tag).random() < suppression(dose):
            refuse = False
        out.append(refuse)
    return out


def flip_stats(a: list[bool], b: list[bool], n_boot: int, seed: int) -> dict:
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    flips = (x != y).astype(float)
    idx = np.random.default_rng(seed).integers(0, flips.size, size=(n_boot, flips.size))
    samples = flips[idx].mean(axis=1)
    n_ab = int((x & ~y).sum())
    n_ba = int((~x & y).sum())
    return {
        "n": int(flips.size),
        "flip_rate": float(flips.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n_boot": n_boot,
        "mcnemar": mcnemar_exact(n_ab, n_ba),
    }


def run(seed: int, draws: int, n_boot: int) -> dict:
    ref = labels(seed, 0.0, draws)
    repeat = flip_stats(ref, labels(seed, 0.0, draws, "repeat"), n_boot, seed)
    per_dose = {}
    min_dose = None
    for dose in GRID:
        st = flip_stats(ref, labels(seed, dose, draws), n_boot, seed)
        st["suppression_prob"] = suppression(dose)
        st["clears_floor"] = st["ci95"][0] > FLOOR
        per_dose[f"{dose:.1f}"] = st
        if min_dose is None and st["clears_floor"]:
            min_dose = dose
    return {
        "dose_response": {"min_dose_above_floor": min_dose, "floor": FLOOR, "per_dose": per_dose},
        "dose_0_repeat": repeat,
        "injected": {
            "curve": "logistic",
            "midpoint": MIDPOINT,
            "steepness": STEEPNESS,
            "base_p_refuse": BASE_P_REFUSE,
            "grid": GRID,
            "draws_per_dose": draws,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.draws, args.n_boot)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK dose-response flip-floor estimator check (logistic curve)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": "PYTHONPATH=src python3 -m refusalscope.dose_floor "
                f"--seed {args.seed} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [
                    {"url": "https://github.com/ggml-org/llama.cpp", "license": "MIT (mock server only, no external data)"}
                ],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
