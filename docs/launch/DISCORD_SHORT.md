# Discord / short announcement (draft for the repository owner)

**refusalscope** is a research harness that asks: does abliteration (for example with Heretic)
damage structured tool calling, and does it make models follow instructions injected into tool
results? It compares stock and ablated pairs through any OpenAI-compatible endpoint (llama-server,
vLLM, Ollama).

Status: harness and pre-registration only. **No results yet**; the README says "Results: NOT YET
RUN" and the report generator refuses to render without real result files. The built-in mock
server is for testing and everything it makes is stamped MOCK.

It ships 374 tool-calling cases, 378 injection cases (each also run without the injection), a
refusal-drop check (bring your own behaviours file), a KL estimate and a capability control, with
paired exact tests and Holm correction. MIT, Windows and Linux runbooks.

If you have ablated checkpoints and a GPU, a pair run and a replication would help most.
