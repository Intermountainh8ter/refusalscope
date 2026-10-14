# Changelog

All notable changes are listed here. Results from different suite versions must not be pooled.

## 0.1.0 - 2026-10-14

First release of the harness. **No real model has been evaluated; there are no results.**

### Added

- Pre-registration (`docs/PREREGISTRATION.md`) written before any analysis code, plus methodology,
  threats to validity, a Windows runbook and flowcharts.
- OpenAI-compatible chat client on `urllib` with retries, timeouts, seeds, loopback proxy bypass
  and logprobs capture; JSON study configuration with strict validation and stock/ablated pair
  checking.
- Mock OpenAI-compatible server that emulates correct tool calls, malformed arguments, tool calls
  written as text, injected-instruction compliance with a tunable probability, refusals and
  first-token logprobs. Everything produced with it is labelled `MOCK`.
- Suite A: 374 tool-calling cases (17 categories) from a seeded generator, deterministic AST-style
  grader reporting validity, schema validity and accuracy separately, 16-label failure taxonomy,
  strict and lenient modes.
- Suite B: 378 indirect prompt-injection cases (9 environments, 14 injection vectors), each run
  with and without the injection; utility, attack success rate, benign tool use.
- Suite C: refusal probe (user-supplied behaviours file loader, 40-item benign borderline set,
  rule-based classifier), first-token KL estimator on 60 harmless prompts, 100-item capability
  control.
- Statistics: Wilson and Clopper-Pearson intervals, exact McNemar, paired and marginal odds
  ratios, seeded paired bootstrap, difference-in-differences bootstrap, Holm-Bonferroni, Spearman
  permutation test, exact power calculation; all unit-tested against textbook or hand values.
- Resumable runner with JSONL trajectory logging, crash repair, circuit breaker and stratified
  `--limit`; `doctor`, `analyze`, `report`, `audit`, `power`, `generate`, `mock-serve` and
  `mock-demo` commands; `scripts/make_report.py` that refuses to render without real results.
- CI (ruff and pytest with coverage on Python 3.10 to 3.13, mock only), Makefile, Dockerfile.
