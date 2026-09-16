# Real TRAIN performance replay review — 2026-09-06

Engineering phase 9E-P1; no model-quality conclusion or TEST/VALIDATION access.
Primary runs: `EXP-9EP1-GPU-DIAGNOSTIC-002` and
`EXP-9EP1-GPU-THROUGHPUT-001`. Verified import:
`EXP-9EP1-GPU-EVIDENCE-002`. The original terminal records and artifacts are
immutable. Canonical C0/C2 and legacy were not changed.

## Result

The prepared-cache variant preserves exact B2 training results but is slower
on the predeclared real TRAIN draw stream. Do not deploy it as an accepted
performance improvement based on the positive synthetic or short profiler run.

| Arm | Repeat 0 records/s | Repeat 1 | Repeat 2 | Median |
|---|---:|---:|---:|---:|
| Reference B2 | 1.679919 | 1.649032 | 1.642707 | 1.649032 |
| Prepared-cache B2 | 1.192079 | 1.197482 | 1.185884 | 1.192079 |
| Prepared-cache B4 | 1.242424 | 1.240876 | 1.240711 | 1.240876 |
| Prepared-cache B8 | 1.277486 | 1.276667 | 1.270385 | 1.276667 |

B2 median ratio is **0.722896**: 27.71% lower throughput, not an acceleration.
All three B2 pairs share exact initial/final full-state, RNG, loss-trace and
schedule fingerprints; primary per-update loss/gradient-norm and draw rows
also match. Checkpoint/resume and logits/gradient equality have separate unit
gates; this real replay is uninterrupted, so it is not a new real-data resume
experiment. B4/B8 do not preserve B2 learning semantics.

The run used the existing Torch 2.2.2+cu118/Python 3.12.8 runtime on RTX 3090,
800 measured draws per arm, 160 discarded warmup draws, three rotated repeats,
an 8 GiB accounted LRU bound and no cold source conversion. Across 12 arms:
9,600 measured draws, 1,920 warmup draws, 3,960 applied/attempted updates.

## Preparation defeats the update savings

Every measured optimized arm has 189 hits and 611 misses, with 609 evictions.
The bound holds about 309 entries at the end. The population prewarm does not
imply that the whole population remains resident through warmup/measurement.

For the first B2 pair, top-level CPU wall stages were:

| Stage | Reference seconds | Prepared-cache seconds |
|---|---:|---:|
| Load + collation + identity / prepare_batch | 325.370 | 571.460 |
| H2D | 7.364 | 7.786 |
| Full training update | 137.809 | 86.566 |
| Complete measured window | 476.213 | 671.096 |

Other repeats show the same direction. Full-update savings are 51–60 seconds
per 400 updates, while preparation dominates. A cache miss repeats frozen
preparation/singleton collation and adds graph validation, defensive deepcopy
of the prepared object including expanded sidecar, recursive memory accounting
and another sidecar copy on batch delivery. Hits also copy expanded sidecars.
Existing timers do not isolate all these operations, so individual cost
attribution remains a hypothesis. Nested stages must not be added to their
parents as separate elapsed work.

The profiler diagnostic had only 16 draws per arm and measured about 2.897x B2
ratio under instrumentation. It is a mechanics/trace result, not representative
unprofiled throughput. The negative 800-draw result takes precedence for speed.

## Evidence and limits

The archive is 324,769,659 bytes, SHA-256
`13338367f206d0437564bfb5beb36284d83c9e86b37850d4d955131b2c82bfea`.
It is retained in ignored content-addressed storage. Verified 96 record-listed
artifacts, 11 controller entries, sealed lifecycle records, primary metric
accounting and reconstructed loss-trace fingerprints. No imported code ran.
Full verification and per-arm stage totals are under
`outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-002/verification.json`.

The full source population is prewarmed before every arm, including reference.
Reference holds an unused prepared cache under this historical protocol.
Prewarm and state hashes are excluded from measured windows; logging is
included. These are replay TRAIN rates, not estimates of complete 120k runtime
including validation/checkpoints. RSS is a process-lifetime high-water mark;
reserved CUDA memory includes allocator history. No corpuswide worst-case
memory claim follows from these draws.

## Next isolated question

Use `--optimized-data-strategy frozen --batch-sizes 2` to keep original data
preparation in both arms and change only update. This mode constructs no
prepared graph cache and declares its separate prewarm policy. The registered
evidence wrapper requires explicit strategy fields to match the protocol.

`EXP-9EP1-GPU-UPDATE-ONLY-001` is one unprofiled 800-draw repeat with 160 warmup
draws per arm, linked to the negative throughput result. It tests attribution;
it is not the three-repeat >=20% acceptance. Subsequent cache remediation can
test storing original sidecar bytes and avoiding repeated Python-object copies,
without increasing the memory limit or dropping labels/provenance/validators.
Any such change needs exact ownership/equality tests and new experiment IDs.
C2 historical-transposition equivalence remains a separate gate before using
any optimized runner in a future matched scientific series.

## Update-only diagnostic completed

`EXP-9EP1-GPU-UPDATE-ONLY-001` completed at 11:30 UTC and was independently
verified by `EXP-9EP1-GPU-EVIDENCE-003`. Both arms use frozen data and create
no prepared cache. Each measures 800 draws / 400 updates after 160 discarded
warmup draws. Total execution accounting includes 960 updates with warmup.

| Measured quantity | Reference B2 | Optimized update B2 |
|---|---:|---:|
| Records/s | 1.856601 | 2.185625 |
| Complete window, seconds | 430.895 | 366.028 |
| Frozen loading, seconds | 267.502 | 266.967 |
| Full update, seconds | 144.533 | 80.236 |

The throughput ratio is **1.177219**, a **17.72%** increase, while elapsed
time falls **15.05%**. Full update accounts for 64.30 of the 64.87 seconds
saved. All 400 primary row pairs and independently reconstructed loss-trace
hashes agree; full-state/RNG/schedule fingerprints match. This is one
baseline-first pair, not the three-repeat speed acceptance. Load now consumes
about 73% of the optimized measured window.

The import verifies 16 record artifacts, eight controller entries, nine
accepted source bindings and the GPU CSV (1,660 rows). Archive SHA-256:
`4f9811f0248f302e4ae707a236d43882d3cc857bcd3a65f79fe6c062ec31d56d`.
The median GPU utilization samples are 4% / 3%; these include the exempt
desktop process and are not kernel occupancy measurements. Throughput is the
acceptance quantity. Measured CUDA allocated peaks are approximately 1.15 GB
in both arms; these draws do not establish corpuswide memory readiness.

The next candidate retains the exact verified source JSON bytes in the cache
instead of expanded sidecar containers. Public consumers still receive fresh
decoded dictionaries, and graph ownership, validators and bounded accounting
remain. Newly separated copy/accounting/materialization timings will assess
the implementation; no production gain is claimed before measurement.

The next package benchmark must also remove the historical unused prepared
cache from baseline arms and disclose the new prewarm policy. An inactive
candidate cache is not part of original training and can affect host memory
and garbage collection. Historical terminal records remain unchanged. The
new comparison tests the whole cache-plus-update package, not a cache-only
ablation; absolute baseline differences across separate runs cannot be used
as paired evidence of an individual optimization.
