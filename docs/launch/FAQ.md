# FAQ

**Is there a result?** Yes, a first one, and it is narrow. I ran five Qwen3 stock/ablated pairs
(0.6B, 1.7B, 4B, 8B, 14B) through llama.cpp. Four passed the pre-registered ablation check:
refusals on a 100-item author-written behaviours file fell 94 pp (4B), 79 pp (8B), 56 pp (1.7B)
and 70 pp (14B). The 0.6B fell only 15 pp because its stock model barely refuses that file, so it
is reported as ablation-unverified and kept out of the tests. On the four verified pairs the
predicted harm did not appear. Tool-call accuracy, ablated minus stock, was -2.9 pp (4B), 0.0 pp
(8B), +0.5 pp (1.7B) and +5.3 pp (14B). Injection attack success was -6.3 pp (4B), +0.3 pp (8B),
-1.3 pp (1.7B) and +3.7 pp (14B). Two results went the opposite way to the prediction (14B tool
calling, 4B injection) and the 14B injection result is inconclusive. H3, the model-size trend, was
not testable: it needs five verified pairs and there are four. The README has the table.

**What does this not show?** That ablation is safe. The suites are synthetic. Each pair was run
once (temperature 0, one seed) on one machine (AMD gfx906, ROCm, llama.cpp) with Q8_0 GGUFs and
one shared chat template, so some ablated models were not served exactly as their uploaders
published them. All pairs are Qwen3, and the 0.6B, 1.7B and 14B ablated arms share one uploader
and method. With 374 tool-calling and 378 injection cases, differences under about 5 pp are not
resolvable. The refusal drop was also checked against JailbreakBench, HarmBench and AdvBench on
the 4B and 8B pairs (74.5 to 99 pp); those are the only replications.

**What confounds are known?** Two are visible in the data. First, on a clean control with no
injection the ablated arms used tools more than stock on the 8B and 14B pairs and less on the 4B
pair, so whatever these edits do to tool-use propensity has no single sign, and it can move both
headline measures. Second, strict grading counts a refusal as a failed tool call, so the 14B
improvement is largely stock refusals that the ablated model no longer emits, and the 4B dip
disappears under lenient grading. Separately, first-token KL between the arms ranged from 0.009
(4B) to 0.235 (1.7B) across the Qwen3 pairs and did not predict the size or direction of either
effect.

**What about the Llama pair?** A sixth, cross-family pair (Llama-3.1-8B against mlabonne's
abliteration) passed the ablation check (96 pp) with capability unchanged and a first-token KL of
1.03, far above the Qwen3 pairs. Its tool-call and injection suites could not be scored: llama.cpp
returned server errors whenever the model emitted parallel tool calls. It is not in the README
table and does not count toward H3.

**What is abliteration?** Editing a model's weights so that its activations lose the direction
associated with refusing requests (Arditi et al., 2024, "Refusal in Language Models Is Mediated by
a Single Direction"). Tools such as Heretic automate the choice of how much to remove.

**What question does this answer?** Two: whether the ablated model is worse than its stock twin at
producing valid and correct tool calls, and whether it executes attacker instructions hidden in
tool results more often. Each is a pre-registered hypothesis with a stated way to be wrong.

**Has someone already answered it?** I looked in 2026 and found one adjacent preprint on
abliterated code-generation models and defence papers on indirect injection, but no study
measuring either quantity after abliteration. That is my dated observation, not a literature
review. Please open an issue if I missed one.

**Why synthetic suites?** They are self-contained, seeded and checkable: the expected answer comes
from the same random draw that wrote the request. The price is realism; see
`docs/THREATS_TO_VALIDITY.md`. The generators are public so anyone can swap in different cases.

**Why a mock server?** So the whole pipeline can be tested without a GPU or weights. It returns
the correct answer, or a configured kind of wrong answer, so a run against it only shows that the
pipeline notices what it was told to inject. Everything it produces says MOCK and is git-ignored.
None of the reported results came from it.

**Will you publish whatever comes out?** That is the point of the pre-registration, and the first
run is an example: a failed ablation check (0.6B), two results opposite to my prediction and one
inconclusive result are in the README next to the nulls.

**Does the repository contain harmful prompts?** No. The ablation check reads a file you supply.
The reported runs used the author-written 100-item file plus public lists downloaded separately
(JailbreakBench, HarmBench, AdvBench); none is committed, and each results file records the file's
hash. The only bundled refusal data is 40 benign requests that merely sound edgy.

**Do I need a GPU?** Not for the tests or the mock demo. A real study needs a machine that can run
the models; a 4B model in Q4_K_M quantisation fits in a few gigabytes. The reported runs used one
AMD gfx906 GPU with Q8_0 GGUFs.

**Which servers work?** Anything with an OpenAI-compatible `/v1/chat/completions`: `llama-server`
(use `--jinja`), vLLM (tool-call parser flags), Ollama. Logprobs, needed only for the KL probe, are
not available everywhere; the report then says "not measured". The reported runs all used
llama.cpp.

**Why do both arms have to be converted the same way?** Quantisation and chat-template differences
can look like ablation effects. The runbook converts both through one pipeline and the config
records `quant` so mismatches are flagged. Several ablated uploads shipped no chat template, which
is why every arm in the reported runs was driven from one shared template file.

**What does "equivalent" mean in the report?** The whole 95 percent interval of the difference lies
within the pre-registered 5 percentage-point margin. It means no larger difference was detected,
not that ablation is safe. "Contrary" means a difference of at least that margin in the opposite
direction from the hypothesis, and "inconclusive" means neither reading is established.

**How do I contribute a pair or a replication?** See `CONTRIBUTING.md`. Run the study, keep the
trajectories, and open a pull request that adds only `results/<id>.json` plus the config you used.

**Was AI used to build this?** Yes. An AI coding assistant wrote most of the code, tests and prose in this repository (the commits are co-authored in the git history). I directed the work and ran the real-model benchmarks on my own hardware. Every number in the README is generated by a script from committed result files, and the tests fail if they drift. Read the code and the pre-registration before trusting any conclusion.
