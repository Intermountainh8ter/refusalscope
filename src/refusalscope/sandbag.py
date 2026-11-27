"""MOCK evaluation-framing accuracy gap on a fixed question set.

Simulation only: no model, no server. Each item has a logit accuracy z_i. A framing (neutral,
evaluation-framed, cue-removed) adds a logit shift and a per-paraphrase effect; each
(item, paraphrase) cell is answered `draws` times as Bernoulli. The gap is neutral accuracy minus
framed accuracy, averaged over replicates. Constant responders have gap exactly 0. Not evidence
about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.sandbag/1"
EVAL_SHIFT = -0.3
CUE_SHIFT = 0.0
PARAPHRASE_SD = 0.2
STUBS = ("effects_stub", "always_correct", "always_wrong")


def _accuracy(rng, stub, z, shift, paraphrases, draws):
    if stub == "always_correct":
        return 1.0
    if stub == "always_wrong":
        return 0.0
    pe = rng.normal(0.0, PARAPHRASE_SD, (len(z), paraphrases))
    p = 1.0 / (1.0 + np.exp(-(z[:, None] + shift + pe)))
    return float((rng.binomial(draws, p) / draws).mean())


def simulate(seed, stub, items, paraphrases, draws, replicates):
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    ge, gc = [], []
    for _ in range(replicates):
        z = rng.normal(0.0, 1.0, items)
        neutral = _accuracy(rng, stub, z, 0.0, paraphrases, draws)
        ge.append(neutral - _accuracy(rng, stub, z, EVAL_SHIFT, paraphrases, draws))
        gc.append(neutral - _accuracy(rng, stub, z, CUE_SHIFT, paraphrases, draws))
    return {
        "accuracy_gap_evaluation_framed": float(np.mean(ge)),
        "accuracy_gap_cue_removed": float(np.mean(gc)),
    }


def run(seed, items, paraphrases, draws, replicates):
    sims = {s: simulate(seed, s, items, paraphrases, draws, replicates) for s in STUBS}
    return {
        "sandbag": {**sims["effects_stub"], "stubs": {s: sims[s] for s in STUBS if s != "effects_stub"}},
        "injected": {
            "eval_shift": EVAL_SHIFT,
            "cue_shift": CUE_SHIFT,
            "paraphrase_sd": PARAPHRASE_SD,
            "items": items,
            "paraphrases": paraphrases,
            "draws": draws,
            "replicates": replicates,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--paraphrases", type=int, default=4)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(a.seed, a.items, a.paraphrases, a.draws, a.replicates)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK evaluation-framing accuracy gap (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [a.seed],
                "command": f"python3 -m refusalscope sandbag --seed {a.seed} --items {a.items} "
                f"--paraphrases {a.paraphrases} --draws {a.draws} --replicates {a.replicates} --out {a.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
