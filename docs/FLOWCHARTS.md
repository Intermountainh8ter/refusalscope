# Flowcharts

Two diagrams, kept as Mermaid source (`docs/diagrams/*.mmd`) and rendered to SVG by
`scripts/render_diagrams.sh` (`make diagrams`). The SVGs are embedded in the README.

## Using the harness: install to report

![Usage flow](diagrams/usage.svg)

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "wrappingWidth": 420, "nodeSpacing": 28, "rankSpacing": 38, "curve": "basis"}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart TD
    A["1. Install<br/>pip install -e '.[dev]'"] --> B["2. Prove the harness works<br/>refusalscope mock-demo<br/>(MOCK output only)"]
    B --> C["3. Make the pair from ONE checkpoint<br/>stock + Heretic-ablated<br/>same conversion, same quantisation"]
    C --> D["4a. Serve stock<br/>llama-server --jinja --ctx-size 8192<br/>--parallel 1 --port 8080"]
    C --> E["4b. Serve ablated<br/>llama-server --jinja --ctx-size 8192<br/>--parallel 1 --port 8081"]
    D --> F["5. Write the config<br/>copy configs/example_qwen3.json<br/>set quant, params_b, ports, notes"]
    E --> F
    F --> G["6. Download a behaviours file yourself<br/>data/harmful_behaviors.csv<br/>(never committed)"]
    G --> H{"7. refusalscope doctor<br/>--config my_study.json"}
    H -->|"FAIL or WARN"| T["Troubleshooting flowchart<br/>docs/diagrams/troubleshooting.svg"]
    T --> H
    H -->|"all checks pass"| I["8. refusalscope run --suite all<br/>--run-id ID"]
    I -->|"crash, Ctrl-C or abort"| R["Run the same command with<br/>the same --run-id: it resumes"]
    R --> I
    I -->|"finished"| J["9. refusalscope analyze --run runs/ID<br/>re-grades, paired statistics, Holm"]
    J --> K{"Ablation check passed?<br/>refusal drop of 30+ points"}
    K -->|"no: ablation-unverified"| U["Report it as a finding about the<br/>checkpoint; no confirmatory claim"]
    K -->|"yes"| L["results/ID.json"]
    L --> M["10. refusalscope report<br/>--update-readme README.md"]
    M --> N["report/*.md and *.svg<br/>README results table filled"]
```

## Troubleshooting: real symptoms and the fix for each

Every leaf is a concrete action. Most symptoms are caught by `refusalscope doctor`, which prints
the same fixes. Run `refusalscope audit` to look at graded samples before blaming a grader.

![Troubleshooting decision tree](diagrams/troubleshooting.svg)

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "wrappingWidth": 215, "nodeSpacing": 14, "rankSpacing": 30, "curve": "basis"}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart LR
    S(["What do<br/>you see?"])

    S --> A["Tool calls come back<br/>as plain text<br/>(doctor FAIL, text_tool_call)"]
    A --> A2["FIX: restart llama-server<br/>with --jinja. vLLM: add<br/>--enable-auto-tool-choice<br/>--tool-call-parser hermes.<br/>Re-run doctor"]

    S --> B["HTTP 400 context size<br/>exceeded, or replies<br/>are cut off"]
    B --> B2["FIX: --ctx-size 8192+<br/>and --parallel 1 (slots<br/>split the window), or<br/>lower sampling.max_tokens"]

    S --> C["Connection refused<br/>or timed out"]
    C --> C1("Server window open<br/>and 'listening'?")
    C1 -->|no| C2["FIX: start it, check the<br/>port in base_url, allow<br/>the firewall prompt"]
    C1 -->|yes| C3["FIX: base_url ends in /v1;<br/>raise sampling.timeout_s<br/>while the model loads"]

    S --> D["KL 'not measured'<br/>(no top_logprobs)"]
    D --> D2["FIX: use llama-server or<br/>vLLM for the KL probe.<br/>Never impute a value"]

    S --> E["Accuracy is 0 or near 0<br/>for every model"]
    E --> E1("audit --suite toolcall<br/>--n 10: what do the<br/>replies show?")
    E1 -->|"call as text"| E5["FIX: same as the first<br/>branch: restart llama-server<br/>with --jinja"]
    E1 -->|"empty, length"| E2["FIX: thinking used the<br/>token budget. Set<br/>enable_thinking false in<br/>extra_body, both arms,<br/>or raise max_tokens"]
    E1 -->|"401 / 404"| E3["FIX: api_key_env,<br/>base_url, model name"]
    E1 -->|"wrong values"| E4["Likely real behaviour:<br/>read the taxonomy first"]

    S --> F["Strict and lenient grades<br/>differ, or a grade<br/>looks wrong"]
    F --> F1("audit --n 20:<br/>is the grader wrong?")
    F1 -->|yes| F2["FIX the grader, add a<br/>test, run analyze again.<br/>Trajectories are re-graded,<br/>no model re-run"]
    F1 -->|no| F3["Report the strict versus<br/>lenient gap as a finding"]

    S --> G["Run crashed, Ctrl-C,<br/>or the circuit breaker<br/>aborted it"]
    G --> G1["FIX: run the identical<br/>command with the same<br/>--run-id. It repairs the last<br/>line, skips done jobs and<br/>retries failed ones"]

    S --> J["Resume refused: different<br/>result-affecting settings"]
    J --> J1["You changed models, sampling, suites or the behaviours file. Revert it or start a new run id; --force only with a logged deviation. Ports, timeouts and workers may change freely"]

    S --> H["Pair shows<br/>ablation-unverified"]
    H --> H1("Behaviours file<br/>configured and found?")
    H1 -->|no| H2["FIX: set<br/>suites.refusal.harmful_path<br/>to the downloaded file"]
    H1 -->|yes| H3["Refusal drop under 30<br/>points: check ports are<br/>not swapped and the<br/>ablated weights loaded.<br/>Else report ablation failure"]

    S --> I["report says<br/>NOT YET RUN or<br/>only MOCK files"]
    I --> I2["FIX: run and analyze a<br/>real study. --allow-mock<br/>only renders MOCK output"]
```

## Where each leaf comes from

| Symptom | Detected by | Documented in |
|---|---|---|
| Tool calls as plain text | `doctor` tool-call probe, taxonomy label `text_tool_call`, 5 percent serving-problem rule | `docs/METHODOLOGY.md` 3.2, `docs/PREREGISTRATION.md` 8 |
| Context too small | `doctor` context probe (largest Suite A request) | `docs/RUNBOOK_WINDOWS.md` |
| Connection refused or timeout | `doctor` endpoint check | `docs/RUNBOOK_WINDOWS.md` |
| No logprobs | `doctor` logprobs probe; KL reported as "not measured" | `docs/METHODOLOGY.md` 5.2 |
| All-zero accuracy | `refusalscope audit`, taxonomy counts, `doctor` thinking warning | `docs/THREATS_TO_VALIDITY.md` 2 |
| Grader disagreement | strict/lenient columns, `refusalscope audit`, `refusalscope analyze` re-grades | `docs/METHODOLOGY.md` 3.2 and 7 |
| Resume after a crash | resumable runner, `run.json` fingerprint | `docs/METHODOLOGY.md` 6 |
| Ablation-unverified pair | manipulation check in `analyze` | `docs/PREREGISTRATION.md` 8 |
| NOT YET RUN or only MOCK | `scripts/make_report.py` refusal rules | `README.md` |
