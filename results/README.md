# results/

Result files produced by **real** runs live here, one JSON file per run, named after the run id
(`refusalscope analyze` writes them). `scripts/make_report.py` and `refusalscope report` build
every table and figure from these files and from nothing else.

Status: **real results are committed here**: one file per analysed run, covering five Qwen3 pairs
(0.6B, 1.7B, 4B, 8B, 14B), the 4B and 8B ablation-check replications on three public corpora
(`*-jbb.json`, `*-hb.json`, `*-advb.json`) and a Llama-3.1-8B pair (`llama31-8b-abliterated-rocm.json`).
The README table is generated from these files. With no real result file in this directory, the
report generator refuses to render (`Results: NOT YET RUN`), by design.

Rules:

- Files named `MOCK-*.json` come from the mock server, which only tests the harness. They are
  git-ignored and must never be committed or cited.
- Do not edit a result file by hand. Re-run `refusalscope analyze` if a grader changes; stored
  trajectories are re-graded, so models do not need to be queried again.
- Publish the matching `runs/<id>/trajectories.jsonl` (large) as a release asset so others can
  re-grade it.
