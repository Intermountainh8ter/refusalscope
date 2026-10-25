# Pre-registration additions

## 2026-10-25T22:48:19Z flip-floor-cache-mock proposal:8caa45109eeefb16573cd6cb7e50c3a5ff1e9e5013f9eda3443125fb4a86e309

Hypothesis: on the MOCK server with a seeded 5% per-draw label jitter applied only when cache_prompt is on, the paired refusal-label flip rate between cache_prompt off and on over the 40-case borderline set repeated to 400 draws lies in 0.02 to 0.09, while the cache_off versus cache_off repeat flip rate is exactly 0 and the exact McNemar p-value for the off-vs-on pair is below 0.05.

Metric: /refusal_label_flip_rate/cache_on_vs_off (expected between 0.02 and 0.09; cache_off_vs_off equals 0).

Falsifier: cache_on_vs_off outside 0.02 to 0.09, or cache_off_vs_off nonzero, or McNemar p-value 0.05 or higher.
