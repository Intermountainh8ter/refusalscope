# Pre-registration additions

## 2026-10-25T22:48:19Z flip-floor-cache-mock proposal:8caa45109eeefb16573cd6cb7e50c3a5ff1e9e5013f9eda3443125fb4a86e309

Hypothesis: on the MOCK server with a seeded 5% per-draw label jitter applied only when cache_prompt is on, the paired refusal-label flip rate between cache_prompt off and on over the 40-case borderline set repeated to 400 draws lies in 0.02 to 0.09, while the cache_off versus cache_off repeat flip rate is exactly 0 and the exact McNemar p-value for the off-vs-on pair is below 0.05.

Metric: /refusal_label_flip_rate/cache_on_vs_off (expected between 0.02 and 0.09; cache_off_vs_off equals 0).

Falsifier: cache_on_vs_off outside 0.02 to 0.09, or cache_off_vs_off nonzero, or McNemar p-value 0.05 or higher.

## 2026-10-25T22:58:29Z flip-floor-slots-mock proposal:36c7533c66aab83940f02900d32eb4df9c90de081570870514b7aa9804eda947

Hypothesis: on the MOCK server with a seeded 6% per-draw label jitter applied only when parallel slots equal 4 and no slot id is pinned, the paired refusal-label flip rate between slots=1 and slots=4 over the 40-case borderline set repeated to 400 draws lies in 0.02 to 0.10, the slots=4 pinned-slot versus slots=1 flip rate is exactly 0, and the exact McNemar p-value for the unpinned pair is below 0.05.

Metric: /refusal_label_flip_rate/slots_4_vs_1 (expected between 0.02 and 0.10; pinned-slot arm flip rate equals 0).

Falsifier: slots_4_vs_1 outside 0.02 to 0.10, or the pinned-slot arm flip rate nonzero, or McNemar p-value 0.05 or higher.

## 2026-11-11T10:53:53Z flip-floor-template-mock proposal:c50315a44fff5a1ee845ffaf3748b4ec85fa0ec39e6bfc256a80e32dd6a7a90d

Hypothesis: on the MOCK server with a seeded 7% per-draw label jitter applied only when the template variant is B and not A, the paired refusal-label flip rate between template A and template B over the 40-case borderline set repeated to 400 draws lies in 0.03 to 0.11, the A-versus-A repeat flip rate is exactly 0, and the exact McNemar p-value for the A-vs-B pair is below 0.05.

Metric: /refusal_label_flip_rate/template_b_vs_a (expected between 0.03 and 0.11; template_a_vs_a equals 0).

Falsifier: template_b_vs_a outside 0.03 to 0.11, or template_a_vs_a nonzero, or McNemar p-value 0.05 or higher.

## 2026-11-13T12:00:02Z dose-flip-floor-sigmoid-mock proposal:4a49e2f88f8c939c3b9379a527734525598c01dd5e3620745d6bbbd2791bd200

Hypothesis: on the MOCK server with a seeded logistic refusal-suppression curve (midpoint 0.5, steepness 10) over an 11-point dose grid 0.0 to 1.0 and 400 draws per dose on the 40-case borderline set, the paired refusal-label flip rate versus dose 0.0 first has a bootstrap 95% lower bound above the 0.07 mock serving flip floor at dose 0.4, with dose 0.0 versus a dose 0.0 repeat flip rate exactly 0.

Metric: /dose_response/min_dose_above_floor (expected 0.4, within one grid step, range 0.3 to 0.5).

Falsifier: min_dose_above_floor outside 0.3 to 0.5, or the dose 0.0 repeat flip rate is nonzero, or no dose clears the floor.

## 2026-11-16T11:14:31Z flip-floor-batch-mock proposal:4bf3d69562dd1583d99d1dd26290a8e784994ea98e1bd8c7d5903cd0d45c0a7a

Hypothesis: on the MOCK server with a seeded 0.06 per-draw label jitter applied only when batch size is 8 and batch-invariant kernels are off, the paired refusal-label flip rate between batch 1 and batch 8 over the 40-case borderline set at 400 draws exceeds the batch-1 repeat flip rate (exactly 0) with a bootstrap lower bound above 0, while the batch-invariant arm at batch 8 flips at exactly 0.

Metric: /batch/paired_flip_over_floor (expected between 0.02 and 0.10; the batch-1 repeat floor and the batch-invariant arm flip rate both equal 0).

Falsifier: paired_flip_over_floor outside 0.02 to 0.10, or the batch-invariant arm flip rate nonzero, or the batch-1 repeat floor nonzero.
