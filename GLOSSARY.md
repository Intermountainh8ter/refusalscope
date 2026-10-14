# Glossary

Every abbreviation and piece of jargon used in this repository, in plain words.
Terms are listed once; where a short form means two different things, both are
given, because that collision causes real confusion.

## Terms specific to this study

**abliteration** — A technique for removing a language model's refusal
behaviour by editing its weights directly, without retraining it. Also sold
under brand names such as "uncensored", "heretic" or "unaligned". This repo
exists to measure whether that edit damages anything else.

**ablated arm / stock arm** — The two halves of a comparison. "Stock" is the
original published model; "ablated" is the copy with its refusal behaviour
removed. Each is tested on identical inputs so the only difference is the model.

**ASR — Attack Success Rate** — The share of prompt-injection attempts that
succeeded in making the model do the attacker's bidding. Lower is better.

**confirmatory vs descriptive** — A *confirmatory* result is one the study
committed to testing in advance and is entitled to call a finding. A
*descriptive* result is a number you may report but may not treat as a tested
claim. In this study all results are descriptive, because the manipulation check
(below) was never run.

**DiD — Difference in Differences** — Comparing how much one measurement moved
against how much a second, unrelated measurement moved. Used here (H4) to ask
whether tool-calling got worse *beyond* any general drop in capability.

**H1, H1b, H2, H3, H4** — The numbered hypotheses this study registered before
running anything. H1: ablation reduces tool-calling accuracy. H1b: it reduces
the share of well-formed tool calls. H2: it raises prompt-injection success.
H3: the effect grows with model size. H4: the damage is specific to
tool-calling rather than general.

**Holm–Bonferroni** — A procedure for adjusting p-values when several
hypotheses are tested at once, so that testing more things does not make a
false positive more likely.

**KL — Kullback–Leibler divergence** — A number measuring how far apart two
probability distributions are. Used here to measure how differently the stock
and ablated models predict the next word. A key finding: it did not predict
behaviour.

**manipulation check** — The test that proves the experiment's *cause* really
happened: before claiming an "uncensored" model behaves differently, show it
actually refuses less. It needs a file of harmful requests, supplied by hand.
Section 8 of the pre-registration requires it.

**McNemar test** — A statistical test for paired yes/no data: of the cases where
the two models disagreed, were the disagreements lopsided or balanced?

**pp — percentage points** — The plain difference between two percentages. From
20% to 25% is 5 pp, not 25%.

**pre-registration** — Writing down the hypotheses, measurements and thresholds
before seeing any results, so they cannot be adjusted afterwards to fit what
came out. `docs/PREREGISTRATION.md`.

**RD — risk difference** — The ablated model's rate minus the stock model's
rate, in percentage points.

**SESOI — Smallest Effect Size Of Interest** — The smallest difference that
would actually matter, fixed in advance. Here, 5 pp. A result can be
statistically significant and still be below the SESOI, which means real but
too small to care about.

**Spearman (rank) correlation** — A measure of whether two things rise and fall
together, using their rank order rather than their values. Used for H3.

**suite** — One family of test cases: `toolcall`, `injection`, `refusal`, `kl`,
`capability`.

**trajectory** — One complete run of one test case against one model, including
any tool calls it made along the way.

## Statistics and measurement

**bootstrap** — Estimating uncertainty by resampling your own data thousands of
times and seeing how much the answer moves.

**CI** — Two meanings, both used in this repo:
1. **Confidence Interval** — a range of plausible values for a measurement, here
   always 95%. If it contains zero, you cannot rule out "no difference".
2. **Continuous Integration** — the automated checks that run on every change.
Context distinguishes them: `[−1.6, +1.3]` is the first, `.github/workflows` the
second.

**n** — The number of items measured.

**null result** — A measurement that found no difference. A real result, not a
failure.

**p-value** — The probability of seeing a difference at least this large if
there were truly no difference at all. Small means surprising-if-nothing.

**paired comparison** — Comparing two models on the *same* case and recording
which won, rather than comparing two separate averages. Much more sensitive.

## Machine learning

**AI / LLM — Large Language Model** — The kind of AI that predicts text, such as
the Qwen3 models used here.

**capability** — Whether the model can do ordinary tasks at all (arithmetic,
following a format), measured here as a control.

**inference** — Running a trained model to get answers. The opposite of
training.

**prompt injection** — An attack where instructions are hidden in data the model
reads — a document, a tool's output, a web page — hoping the model obeys them as
if they came from its user.

**quantisation** — Storing a model's numbers at lower precision to save memory.
**Q8_0** is the 8-bit format used here, applied identically to both arms so it
cannot explain any difference between them.

**refusal** — The model declining a request.

**over-refusal** — Declining a request that was actually harmless. Measured here
on a benign set, so that "safer" cannot be achieved by refusing everything.

**temperature / top-p / top-k** — Settings controlling how randomly the model
picks its next word. Temperature 0 means always pick the most likely one, which
makes runs repeatable.

**token** — The chunk of text a model reads and writes — roughly a short word or
part of one.

**tool call** — The model asking the system to run a function, in a structured
format, rather than replying in prose.

## Files, formats and systems

**AMD / CUDA / HIP / ROCm** — CUDA is NVIDIA's GPU programming system; HIP is
AMD's near-equivalent; ROCm is AMD's surrounding software stack. This study ran
on AMD.

**AST — Abstract Syntax Tree** — A program's structure represented as data.
Used here to check that generated tool calls are well-formed.

**bf16 / fp16 / fp32** — Number formats of decreasing size and precision used in
model weights and arithmetic.

**CLI — Command-Line Interface** — The terminal commands this tool provides.

**CPU / GPU** — The general-purpose processor, and the parallel processor that
runs models quickly.

**CSV / JSON / JSONL** — Data file formats. CSV is comma-separated rows; JSON is
structured records; JSONL is one JSON record per line.

**GB / MB / GiB** — Units of storage. A GiB is 1024³ bytes, slightly larger than
a GB.

**GGUF** — The file format used by llama.cpp to store a quantised model.

**HTML / SVG** — Web page format, and a vector image format (used for the
diagrams in `docs/diagrams/`).

**llama.cpp** — The program that runs the models on this hardware.

**MIT** — The permissive open-source licence this repo uses.

**MOCK** — A run using a fake model, to test the plumbing. Mock output is
labelled and must never be reported as a result.

**README** — The repository's front page.

**SHA-256** — A fingerprint of a file: same contents, same fingerprint.
