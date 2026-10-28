# X / Twitter thread (draft for the repository owner)

Every figure is a difference in percentage points (pp), ablated minus stock, taken from
`results/*.json`. `tests/test_repo_hygiene.py` recomputes them from those files and checks that
each post is 280 characters or fewer.

1/ Does removing a model's refusal direction ("abliteration", e.g. with Heretic) break tool calling, or make it obey instructions hidden in tool results? I ran a pre-registered test on five Qwen3 pairs. Where the ablation worked, I did not see the damage I predicted. Caveats follow.

2/ Setup: each pair is a stock model and its ablated twin, same Q8_0 quantisation, chat template and decoding, scored case by case. 374 tool-calling cases, 378 injection cases (each also run with no injection). Both suites are synthetic. llama.cpp on one AMD gfx906 GPU.

3/ First, did the ablation take? Refusals on a 100-item author-written behaviours file fell 94 pp (4B), 79 pp (8B), 56 pp (1.7B), 70 pp (14B). The 0.6B fell 15 pp: its stock model barely refuses that file, so the 30 pp gate was unreachable. Reported, but kept out of the tests.

4/ Tool-call accuracy, ablated minus stock: 4B -2.9 pp, 8B 0.0 pp, 1.7B +0.5 pp, 14B +5.3 pp. No pair fell past the pre-registered 5 pp margin. The 14B went up, largely because strict grading counts stock refusals as failed calls. The 4B dip vanishes under lenient grading.

5/ Injection attack success, ablated minus stock: 4B -6.3 pp, 8B +0.3 pp, 1.7B -1.3 pp, 14B +3.7 pp. The 4B ablated model was less susceptible, which I did not predict. The 14B result is inconclusive: it leans toward more compliance and I cannot rule out a rise above 5 pp.

6/ H3, the model-size trend, was not testable: it needs five verified pairs and I have four. First-token KL between arms (0.009 to 0.235) did not predict either effect's size or direction. A Llama-3.1-8B pair passed the ablation check but its suites could not be scored here.

7/ This does not show that ablation is safe. One run per pair (temperature 0, one seed), one machine, Q8_0 GGUFs, one shared chat template, one model family, synthetic suites, and differences under about 5 pp are not resolvable with this many cases.

8/ Known confounds: on a clean no-injection control the ablated arms used tools more (8B, 14B) or less (4B) than stock, so tool-use propensity shifts with no consistent sign. And strict grading counts refusals as wrong calls. Either can move the headline numbers.

9/ Hypotheses, tests and what would count as wrong were written before the analysis code existed (git history shows the order). An AI coding assistant wrote most of the code and prose; I directed it and ran the models. Read the code and pre-registration before trusting any conclusion.

10/ MIT licensed, Windows and Linux runbooks, CI on Python 3.10 to 3.13. Replications on other model families, quantisations, hardware and real (non-synthetic) tasks would help most. Code, pre-registration and results: https://github.com/Intermountainh8ter/refusalscope
