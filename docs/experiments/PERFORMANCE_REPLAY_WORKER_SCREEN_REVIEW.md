# Ordered CPU worker screen — 2026-09-07

`EXP-9EP1-PREFETCH-SCREEN-002`, independently imported by
`EXP-9EP1-GPU-EVIDENCE-011`, completed in the isolated RTX 3090 checkout.
Four workers are the candidate for a longer confirmation; this one-repeat
screen does not establish production training readiness.

## Measured result

All arms use the original warm TRAIN preparation, the same optimized update,
B2, 40 discarded warmup draws and 200 measured draws. Arms execute in fixed
order 0/1/2/4 on the same 219 distinct TRAIN records. No compact prepared cache
is used in this comparison.

| Workers | Measured records/s | Ratio to direct | Measured seconds | Preparation seconds | Startup seconds |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.468938 | 1.000000 | 81.006501 | 66.113907 | 2.396093 |
| 1 | 1.913034 | 0.774841 | 104.545950 | 67.018626 | 6.123136 |
| 2 | 2.842572 | 1.151334 | 70.358828 | 66.718945 | 6.179350 |
| 4 | 3.282563 | 1.329545 | 60.928005 | 66.960540 | 6.351564 |

Four workers improve measured throughput by **32.95%** in this screen. One
worker is 22.52% slower; two improve it by 15.13%. These are individual
measurements, not medians or confidence intervals. Worker transfer overhead
must not be ignored, and the data alone do not isolate its causal contribution.

Preparation, worker startup, discarded warmup and cleanup are excluded from
the measured window. Warmup takes 17.061/21.693/14.809/14.287 seconds; cleanup
takes approximately 0.000002/0.226/0.274/0.328 seconds. The sum of these reported
arm phases and measurement is 166.577/199.606/158.340/148.855 seconds. Thus the
four-worker advantage in that sum is **11.91%**, not 32.95%. This sum is not a
complete run timer: initial binding/model setup and gaps between reported
phases remain outside it. The registered source elapsed time is 751.649 seconds.

## Correctness and observed resources

The verifier checked all 400 measured update rows and the four complete
warmup records: 480 applied updates and 960 draws including discarded warmup.
Initial/final training-state, final RNG, loss trace and measured schedule
fingerprints agree across all four arms. Seven actual worker startup
observations confirm the recorded worker runtime configuration. This is an
uninterrupted C0 identity comparison; no C2 or checkpoint-resume acceptance
follows from it. Captured state hashes were checked, not independently loaded
checkpoint tensors.

Across 704 resource samples, the maximum owned-process RSS sum was
3,963,428,864 bytes; minimum effective memory headroom was 54,677,135,360 bytes.
Swap-I/O counter growth and resource stops were zero. Shared pages are counted
repeatedly in RSS sums, and sampling does not establish a true peak or general
RAM bound. The controller and its recorded subtree exited without signals.

The V11 desktop policy produced 143 accepted observations plus five producer
boundary observations. Admitted desktop compute memory stayed at 288 MiB,
including Nautilus at 28 MiB. This does not prove absence of graphics-only
work or measure per-process GPU utilization. No user desktop application was
closed. Source access observations remain tri-state in the original record;
the offline verifier read no corpus, model checkpoints, VALIDATION or TEST.

## Provenance and next action

The target gate passed 114 tests with no failures or skips. V11 introduces
only an explicit optional GPU-policy callback in one orchestration source
file; its default behavior and numerical preparation/update path are unchanged.
All other 255 source bindings and frozen class weights were checked. Historical
V9 receipts remain bound to their original source, and reservation002 remains
applicable because its preparer and loader bindings are unchanged.

Archive SHA-256:
`624339b51e9843c33c0072519e3e2edbbf78342cb4222121a2b03e4656009ef1`
(470,521 bytes, 60 files). The independent verifier checked 28 run artifacts,
the exact controller manifest and externally accepted receipt, proposal and
test bindings. Imported executable files were not run.

Next: register fresh sizing for the longer 800-measured/160-warmup draw
population, then compare direct preparation with four workers in three
rotated repetitions. Keep the existing 20% median throughput target. The
previous compact-cache result of 2.900269 uses another preparation strategy
and population; neither multiply these ratios nor infer their combined speed.
Dedicated real-TRAIN historical C2 equivalence and actual scientific-runner
integration remain required before recommending new accelerated C0/C2 seeds.
