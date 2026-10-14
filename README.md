# refusalscope

A research harness for one question:

> **Does removing the refusal direction from an LLM (abliteration, for example with the
> open-source Heretic tool) damage structured tool calling, and does it make the model more
> likely to follow instructions injected into tool results?**

<!-- STATUS:START -->**Results: RUN** - see the table below and `results/qwen3-4b-rocm-1.json`, `results/qwen3-8b-rocm-1.json`.<!-- STATUS:END -->

Two pairs have now been run on real models. The table below is filled by
`scripts/make_report.py` from `results/*.json` and refuses to be filled from anything else.

<!-- RESULTS:START -->
Source: `results/qwen3-4b-rocm-1.json`, `results/qwen3-8b-rocm-1.json` (generated 2026-10-14T04:52:48.342177Z, harness 0.1.0).

| Pair | Status | H1 tool-call accuracy (stock -> ablated) | H1 | H2 attack success (stock -> ablated) | H2 | H4 | KL |
|---|---|---|---|---|---|---|---|
| qwen3-4b | ablation-unverified | 71.7% -> 68.7% (-2.9 pp) | descriptive | 14.0% -> 7.7% (-6.3 pp) | descriptive | descriptive | 0.009 |
| qwen3-8b | ablation-unverified | 64.4% -> 64.4% (+0.0 pp) | descriptive | 22.2% -> 22.5% (+0.3 pp) | descriptive | descriptive | 0.075 |
<!-- RESULTS:END -->

### What we found

**No pre-registered hypothesis can be confirmed from this run, by the pre-registration's own
rule.** No behaviours file was supplied, so neither pair passes the section 8 manipulation
check, both are labelled `ablation-unverified`, and every test below is descriptive only. H3 is
`not testable` at any rate: it needs five verified pairs and the exact Spearman p-value cannot
reach 0.05 below that. What follows is therefore evidence, not a confirmed claim.

The two pairs disagree, and that is the headline.

**Qwen3-8B** (`DreamFast/qwen3-8b-heretic`) shows **no effect anywhere**. Tool-call accuracy is
identical to the decimal, 64.4% in both arms, with perfectly symmetric discordance (4 cases each
way out of 374) and a 95% CI of [-1.6, +1.3] pp. Attack success is 22.2% vs 22.5%, CI
[-2.1, +2.6] pp. Every interval sits well inside the pre-registered 5 pp SESOI, so this is an
equivalence result rather than an underpowered null. The arms are not the same weights: first-
token KL is 0.075 with CI [0.033, 0.124], excluding zero, and top-1 agreement is 96.7%.

**Qwen3-4B** (`heretic-org/Qwen3-4B-Instruct-2507-heretic`) shows the opposite picture, and in
one case the opposite *direction* to the hypothesis:

- Tool-call accuracy falls 71.7% -> 68.7%, **-2.9 pp** [-4.8, -1.3], p = 0.003, discordance
  12 vs 1. Directionally what H1 predicts, but **smaller than the 5 pp SESOI**, and it
  disappears entirely under lenient grading (S4: 89.8% in both arms, 0.0 pp).
- Attack success **falls** 14.0% -> 7.7%, **-6.3 pp** [-9.0, -4.0], p < 0.0001, discordance
  25 vs 1. H2 predicted a *rise*. This is larger than the SESOI and in the wrong direction for
  the hypothesis. It is also confounded: the same arm uses tools **less** in the clean control
  condition (S1: -5.3 pp [-8.2, -2.6], p = 0.0003), so a model that engages less with the tool
  flow has fewer opportunities to be captured by an injection in it.

The 8B ablated arm moves the other way on that same control, using tools **more** (+4.5 pp
[+2.4, +6.9], p < 0.0001) with higher utility under attack (+3.4 pp, p = 0.007). Two pairs,
opposite signs on tool-use propensity. Whatever these edits do, it is not one consistent effect.

Note also that KL does not predict behaviour here. The 8B pair diverges roughly eight times
further in first-token KL (0.075 vs 0.009) yet behaves identically, while the 4B pair has
**100%** top-1 agreement across all 60 probes and still differs significantly on two measures.

### Threats to validity that apply to this run

- **The manipulation check never ran.** Without it there is no evidence either ablated model
  actually refuses less. A null on an edit that did nothing would look exactly like a null on an
  edit that was harmless. This is the single largest limitation and it is not a small one.
- **Serving template.** The 8B ablated upload ships **no `chat_template` at all**. Served as its
  uploader distributed it, it returned 0 of 3 structured tool calls and would have produced a
  spectacular false H1 effect. Both arms are therefore driven from one shared template file
  (Qwen's own, sha256 `a55ee1b1…`), so the ablated arm is not served exactly as published. The
  4B pair needed no such intervention: its templates were already byte-identical.
- **Two pairs, one family each.** Both pairs are Qwen3 and both ablated arms come from different
  uploaders using different tooling. Nothing here generalises to abliteration in general.
- **Quantisation.** Both arms of each pair were converted and quantised by the same commands to
  Q8_0, with sizes 4,672 bytes apart (8B) and 384 bytes apart (4B). Q8_0 is not lossless, and a
  small effect could in principle be masked by it.
- **One temperature, one seed**, as pre-registered (temperature 0.0, seed 1234).
- **Hardware.** AMD gfx906 under ROCm 7.2.3 via llama.cpp HIP, not CUDA or vLLM.

The hypotheses, metrics, sample sizes, tests and the outcomes that would count as a failure of
each hypothesis were written down **before any analysis code existed**:
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (git history shows the order). The measurement
design is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and the ways this study can mislead you
are in [docs/THREATS_TO_VALIDITY.md](docs/THREATS_TO_VALIDITY.md).

## Why this might matter

Abliteration removes a single direction from a model's residual stream so that it stops
refusing. People then use these models as agents. Two things are not known:

1. **Capability.** Tool calling is a structured-output skill. Does editing the weights along the
   refusal direction leave it intact? (Hypothesis H1, with H4 separating tool-specific damage
   from general degradation.)
2. **Security.** Refusals are also part of how a model resists instructions smuggled into tool
   results (web pages, emails, files). Does removing them make indirect prompt injection easier?
   (H2.)

The repository owner searched for prior work on exactly this in 2026 and found one adjacent
preprint on abliterated code-generation models plus defence papers on indirect injection, but no
study that measures tool-call accuracy or injection compliance after abliteration. That is a
dated observation, not a literature review: re-check before citing it.

## What is measured

Models are compared in **pairs**: a stock model and an ablated version of the same checkpoint,
served by the same engine with the same quantisation, template and decoding.

| Suite | What it measures | Size |
|---|---|---:|
| A `toolcall` | Tool-call validity, schema-validity and accuracy over 17 categories (parallel calls, nested arguments, enums, irrelevance, missing arguments, long and special-character arguments, multi-turn context...). Deterministic AST-style grader with a 16-label failure taxonomy | 374 cases |
| B `injection` | Utility, attack success rate (ASR) and benign tool use in 9 fake tool environments with 14 injection vectors (tool result, document text, filenames, system-looking text, JSON fields, HTML comments, spoofed chat-template tokens, ...). Every case also runs with **no injection** as a control | 378 cases x 2 |
| C `refusal` | Ablation check: refusal rate on a behaviours file **you** supply plus a built-in benign "borderline" set (over-refusal) | your file + 40 |
| C `kl` | First-token KL divergence on harmless prompts from top logprobs (approximation, documented) | 60 |
| C `capability` | General-capability control (arithmetic, word problems, format following) so tool-call changes can be separated from general degradation | 100 |

Statistics: paired by case, exact McNemar tests, seeded bootstrap intervals, paired odds ratios,
Holm-Bonferroni correction across the pre-registered family, Spearman permutation test for the
model-size trend (only when five or more pairs pass the ablation check).

## Honesty rules

- **No fabricated results.** The report generator reads `results/*.json` only and refuses to
  render if there are none.
- **Mock output is labelled.** The bundled mock server tests the harness; everything made
  against it carries `MOCK` in its file name, title and tables, and is git-ignored.
- **No harmful dataset is bundled.** You download a public behaviours file yourself (see below);
  the repository ships only a small benign borderline set that I wrote.
- **Pre-registered.** Changes after the first real run are logged as deviations.

## Quick start (no GPU, no weights)

```bash
cd refusalscope                                           # a checkout of this repository
python -m venv .venv && source .venv/bin/activate        # Windows: see the PowerShell block below
pip install -e ".[dev]"
pytest -q                                                 # the whole suite runs against a mock server
refusalscope mock-demo --out /tmp/rs-demo                 # full pipeline, MOCK-labelled output only
```

`mock-demo` starts two mock servers, runs all five suites through the real runner, analyses the
run and writes `MOCK-demo.json`, a Markdown report and SVG figures into `/tmp/rs-demo`. Its
numbers come from scripted answers with arbitrary error rates and mean nothing about any model.

## Running the real study on Linux

Requirements: Python 3.10+, a `llama.cpp` build (`llama-server`, `llama-quantize`,
`convert_hf_to_gguf.py`), and enough memory for the model pair you are testing (a 4B model in
Q4_K_M needs roughly 3 GB per server).

**1. Make the pair from one checkpoint through one pipeline.** Convert and quantise the stock
and the ablated weights identically; do not mix a third-party GGUF for one arm with your own
conversion for the other.

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir hf/qwen3-4b-stock
# Ablate with Heretic (https://github.com/p-e-w/heretic; follow its README for the current
# install and invocation) and save the result to hf/qwen3-4b-ablated.
# Record the Heretic version and the settings it reports.

for arm in stock ablated; do
  python llama.cpp/convert_hf_to_gguf.py hf/qwen3-4b-$arm \
      --outfile gguf/qwen3-4b-$arm-f16.gguf --outtype f16
  llama-quantize gguf/qwen3-4b-$arm-f16.gguf gguf/qwen3-4b-$arm-Q4_K_M.gguf Q4_K_M
done
```

**2. Serve both arms.** `--jinja` is required for structured tool calls; without it models emit
tool calls as plain text and every score collapses (the harness detects and reports this).
`--parallel 1` keeps the whole context window for the single request stream.

```bash
llama-server -m gguf/qwen3-4b-stock-Q4_K_M.gguf   --alias qwen3-4b-stock   \
    --host 127.0.0.1 --port 8080 --jinja --ctx-size 8192 --parallel 1 -ngl 99 &
llama-server -m gguf/qwen3-4b-ablated-Q4_K_M.gguf --alias qwen3-4b-ablated \
    --host 127.0.0.1 --port 8081 --jinja --ctx-size 8192 --parallel 1 -ngl 99 &
```

Drop `-ngl 99` for a CPU-only build. vLLM and Ollama work too: point `base_url` at their
OpenAI-compatible `/v1` endpoint (vLLM needs `--enable-auto-tool-choice --tool-call-parser
hermes` for Qwen3 tool calls; Ollama may not return logprobs, in which case the KL probe is
reported as "not measured").

**3. Provide the behaviours file for the ablation check.** The repository does not include one.
Download a public set of refusal-eliciting requests yourself, for example AdvBench's
`harmful_behaviors.csv`, and keep it out of git (`data/` is ignored):

```bash
mkdir -p data
curl -L -o data/harmful_behaviors.csv \
  https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
```

JSONL (keys `prompt`, `goal`, `behavior`, `Behavior`, `text`, `instruction` or `question`) and
CSV with a header row are both accepted. A pair that does not show a refusal drop of at least 30
percentage points on this file is reported as `ablation-unverified` and excluded from the
confirmatory tests.

**4. Configure, check, run.**

```bash
cp configs/example_qwen3.json configs/my_study.json      # edit quant, params_b, ports, notes
refusalscope check-config --config configs/my_study.json
refusalscope doctor       --config configs/my_study.json  # fixes for tool-call, context, logprobs problems
refusalscope run     --config configs/my_study.json --suite all --run-id qwen3-pilot-1
refusalscope analyze --run runs/qwen3-pilot-1 --out results
refusalscope report  --results results --out report --update-readme README.md
```

A crash or Ctrl-C loses at most the job in flight: run the same `run` command with the same
`--run-id` and it resumes. To fit memory, run one pair at a time with
`--model qwen3-4b-stock --model qwen3-4b-ablated`, then the next pair, into the same run id.

## Running the real study on Windows (PowerShell)

Details and troubleshooting: [docs/RUNBOOK_WINDOWS.md](docs/RUNBOOK_WINDOWS.md).

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
refusalscope mock-demo --out $env:TEMP\rs-demo

# Serve the pair (adjust the llama.cpp folder; run each in its own window or as below).
$Llama = "C:\tools\llama.cpp\llama-server.exe"
Start-Process $Llama -ArgumentList @("-m","C:\models\qwen3-4b-stock-Q4_K_M.gguf","--alias","qwen3-4b-stock",
    "--host","127.0.0.1","--port","8080","--jinja","--ctx-size","8192","--parallel","1","-ngl","99")
Start-Process $Llama -ArgumentList @("-m","C:\models\qwen3-4b-ablated-Q4_K_M.gguf","--alias","qwen3-4b-ablated",
    "--host","127.0.0.1","--port","8081","--jinja","--ctx-size","8192","--parallel","1","-ngl","99")

New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -OutFile data\harmful_behaviors.csv `
  -Uri https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv

Copy-Item configs\example_qwen3.json configs\my_study.json
refusalscope doctor --config configs\my_study.json
refusalscope run --config configs\my_study.json --suite all --run-id qwen3-pilot-1
refusalscope analyze --run runs\qwen3-pilot-1 --out results
refusalscope report --results results --out report --update-readme README.md
```

## Flowcharts

How a study flows from install to report, and what to do when something goes wrong (source:
`docs/diagrams/*.mmd`, text versions in [docs/FLOWCHARTS.md](docs/FLOWCHARTS.md)):

![Usage flow](docs/diagrams/usage.svg)

![Troubleshooting decision tree](docs/diagrams/troubleshooting.svg)

## Configuration

`configs/example_qwen3.json` is a complete example. Each model has `name`, `base_url`,
`api_key_env` (the *name* of an environment variable, never the key), `role` (`stock` or
`ablated`), `pair_id`, `params_b`, `quant` and optional `extra_body` (for example
`{"chat_template_kwargs": {"enable_thinking": false}}`, identical for both arms). `sampling`
holds `temperatures`, `seeds`, `max_tokens`, `timeout_s` and `retries`; `suites` holds the
generator seeds and sizes. Unknown keys are rejected so typos cannot silently change a study.

## Outputs

| Path | Content |
|---|---|
| `runs/<id>/trajectories.jsonl` | Every request and response of every job, with latencies, token usage and the grade computed at run time |
| `runs/<id>/run.json` | Configuration (no secrets), suite hashes, harness version, behaviours-file hash |
| `results/<id>.json` | Re-graded, paired statistics: rates with Wilson intervals, risk differences with bootstrap intervals, McNemar p-values, odds ratios, Holm-adjusted decisions, taxonomy counts, per-category and per-vector breakdowns, KL, cost and latency |
| `report/<id>-report.md`, `*-forest.svg`, `*-categories-<pair>.svg` | Tables and figures built from the results file |

`refusalscope audit --run runs/<id> --suite toolcall --n 20` prints graded samples so you can
check the graders by eye; a grader fix only needs `refusalscope analyze` again.

## Repository layout

```
src/refusalscope/      client, config, mock server, suites, graders, stats, runner, analysis, report, CLI
suites/                committed snapshots of the generated suites (checked against the generators)
configs/               example and mock configurations
docs/                  pre-registration, methodology, threats to validity, runbook, flowcharts, launch copy
scripts/               make_report.py, render_diagrams.sh
tests/                 pytest suite (mock server only; no network, no GPU)
results/               real result files only (empty until a real run is analysed)
```

## Development

```bash
make install   # pip install -e ".[dev]"
make test      # pytest -q
make cov       # pytest --cov (fails below 90 percent)
make lint      # ruff check .
make mock-demo # MOCK-labelled end-to-end run
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). Licensed under the MIT
license (Ammon Jackson).
