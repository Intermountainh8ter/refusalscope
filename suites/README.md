# suites/

Committed JSON Lines snapshots of the default suites. They are generated, not hand-written:

| File | Rows | Generator |
|---|---:|---|
| `toolcall.jsonl` | 374 | `refusalscope.suites.toolcall.build_cases(seed=1001, per_category=22)` |
| `injection.jsonl` | 378 | `refusalscope.suites.injection.build_cases(seed=2002, n_cases=378)` |
| `borderline.jsonl` | 40 | `refusalscope.suites.refusal.borderline_items()` (benign, written by hand) |
| `kl_prompts.jsonl` | 60 | `refusalscope.suites.kl.kl_items()` (harmless prompts) |
| `capability.jsonl` | 100 | `refusalscope.suites.capability.build_items(seed=3003, n_items=100)` |

Regenerate with `refusalscope generate --out suites` (or `make suites`). A test fails if the
generators stop reproducing these files byte for byte, so a change to a generator is always
visible in review. Any change after the first real run creates a new suite version: results from
different versions must not be pooled (see `docs/PREREGISTRATION.md`).

There is deliberately no harmful-content dataset here. The refusal probe reads a file you supply.
