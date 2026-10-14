# FAQ

**Is there a result?** No. The harness and the pre-registration are done; no real model has been
run. The README says "Results: NOT YET RUN" until real result files exist in `results/`.

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

**Will you publish whatever comes out?** That is the point of the pre-registration: no detectable
damage, a result in the opposite direction, mixed results and failed ablations are all reported.

**Does the repository contain harmful prompts?** No. The ablation check reads a file you download
yourself (for example a public behaviours list). The only bundled refusal data is 40 benign
requests that merely sound edgy.

**Do I need a GPU?** Not for the tests or the mock demo. A real study needs a machine that can run
the models; a 4B model in Q4_K_M quantisation fits in a few gigabytes.

**Which servers work?** Anything with an OpenAI-compatible `/v1/chat/completions`: `llama-server`
(use `--jinja`), vLLM (tool-call parser flags), Ollama. Logprobs, needed only for the KL probe, are
not available everywhere; the report then says "not measured".

**Why do both arms have to be converted the same way?** Quantisation and chat-template differences
can look like ablation effects. The runbook converts both through one pipeline and the config
records `quant` so mismatches are flagged.

**What does "equivalent" mean in the report?** The whole 95 percent interval of the difference lies
within the pre-registered 5 percentage-point margin. It means no larger difference was detected,
not that ablation is safe.

**How do I contribute a pair or a replication?** See `CONTRIBUTING.md`. Run the study, keep the
trajectories, and open a pull request that adds only `results/<id>.json` plus the config you used.
