"""MOCK agreement between one sampled refusal label and the many-sample majority label.

Simulation only: no model, no server. Each prompt has a refusal probability drawn from a bimodal
Beta; the majority label comes from n samples and the single-shot label is one further
independent sample. Degenerate stubs always agree. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.singleshot/1"
BETA = (0.2, 0.2)
STUBS = ("bimodal_stub", "always_refuse", "always_comply")


def _probs(rng: np.random.Generator, stub: str, prompts: int) -> np.ndarray:
    if stub == "always_refuse":
        return np.ones(prompts)
    if stub == "always_comply":
        return np.zeros(prompts)
    return rng.beta(BETA[0], BETA[1], prompts)


def simulate(seed: int, stub: str, prompts: int, samples: int, studies: int) -> dict:
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    agree = []
    for _ in range(studies):
        p = _probs(rng, stub, prompts)
        majority = rng.binomial(samples, p) * 2 > samples
        single = rng.random(prompts) < p
        agree.append(float((majority == single).mean()))
    return {
        "single_shot_majority_agreement": float(np.mean(agree)),
        "agreement_sd_across_studies": float(np.std(agree)),
    }


def run(seed: int, prompts: int, samples: int, studies: int) -> dict:
    sims = {s: simulate(seed, s, prompts, samples, studies) for s in STUBS}
    primary = sims["bimodal_stub"]
    return {
        "sampling": {
            "single_shot_majority_agreement": primary["single_shot_majority_agreement"],
            "agreement_sd_across_studies": primary["agreement_sd_across_studies"],
            "stubs": {s: sims[s] for s in STUBS if s != "bimodal_stub"},
        },
        "injected": {"beta": list(BETA), "prompts": prompts, "samples": samples, "studies": studies},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", type=int, default=500)
    ap.add_argument("--samples", type=int, default=31)
    ap.add_argument("--studies", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.prompts, args.samples, args.studies)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK single-shot vs majority refusal label agreement (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope singleshot --seed {args.seed} "
                f"--prompts {args.prompts} --samples {args.samples} "
                f"--studies {args.studies} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
