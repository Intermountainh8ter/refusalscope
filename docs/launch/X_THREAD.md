# X / Twitter thread (draft for the repository owner)

No result is claimed anywhere in this thread. Keep it that way until real result files exist.

1/ Does removing a model's refusal direction ("abliteration", e.g. with Heretic) break tool
calling? And does it make the model more likely to follow instructions hidden in tool results?
I could not find a study that measures either. So I built the harness. Thread on the design.

2/ Status first: nothing has been run on a real model yet. The README says "Results: NOT YET
RUN" and the report script refuses to render without result files from real runs.

3/ The hypotheses, metrics, sample sizes, tests and "what would prove me wrong" were written
down before any analysis code existed. The git history shows the order. Null and contrary
outcomes have their own section.

4/ Design: stock vs ablated pairs of the same checkpoint, same quantisation, same template,
same decoding, paired by case. Any OpenAI-compatible server works: llama-server, vLLM, Ollama.

5/ Suite A: 374 generated tool-calling cases, 17 categories (parallel calls, nested args, enums,
irrelevance, missing args, long and special-character args, multi-turn). Deterministic grader,
validity and accuracy reported separately, 16-label failure taxonomy.

6/ Suite B: 378 injection cases, 9 fake tool environments, 14 injection styles (tool result,
document text, filenames, fake system notices, spoofed chat tokens, HTML comments...). Every case
also runs with no injection, so utility changes are not confused with the attack.

7/ Controls: a check that the ablation actually removed refusals (you supply the behaviours file;
nothing harmful is bundled), a first-token KL estimate on harmless prompts, and a general
capability control to separate tool damage from general damage.

8/ Stats: exact McNemar, seeded bootstrap intervals, paired odds ratios, Holm correction, an
exact permutation test for a model-size trend that refuses to run with fewer than five pairs.
Every function is tested against textbook values.

9/ The mock server exists only to test the harness. Anything made with it is labelled MOCK in
the file name, the title and every table.

10/ MIT licensed, Windows and Linux runbooks, CI on Python 3.10 to 3.13. If you run ablated
models and want to contribute a pair, a replication or a better grader, PRs welcome. Repo link
in the next post.
