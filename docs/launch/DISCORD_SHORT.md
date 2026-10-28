# Discord / short announcement (draft for the repository owner)

**refusalscope** is a research harness that asks: does abliteration (for example with Heretic)
damage structured tool calling, and does it make models follow instructions injected into tool
results? It compares stock and ablated pairs through any OpenAI-compatible endpoint (llama-server,
vLLM, Ollama).

First real runs are in: five Qwen3 pairs, 0.6B to 14B. On the four pairs where the ablation
demonstrably worked (refusals down 56 to 94 pp), I did not see the damage I predicted. Tool-call
accuracy, ablated minus stock, ranged from -2.9 pp to +5.3 pp, and injection attack success from
-6.3 pp to +3.7 pp. Two results went the opposite way to my prediction (14B tool calling got
better, 4B injection success got lower); the 14B injection result is inconclusive. The model-size
trend (H3) was not testable with four verified pairs.

That does not show ablation is safe: the suites are synthetic (374 tool-calling cases, 378
injection cases), each pair was run once on one machine with Q8_0 GGUFs, all pairs are Qwen3, and
differences under about 5 pp are not resolvable. Results, caveats and the pre-registration:
https://github.com/Intermountainh8ter/refusalscope

If you have ablated checkpoints and a GPU, replications on other model families and quantisations
would help most. MIT licensed, Windows and Linux runbooks.
