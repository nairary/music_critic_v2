# Compact prepared-cache replay — 2026-09-06

Engineering result `EXP-9EP1-GPU-THROUGHPUT-002`, independently imported by
`EXP-9EP1-GPU-EVIDENCE-004`. The source run completed at 13:03:39 UTC on the
isolated RTX 3090 checkout with Python 3.12.8 / Torch 2.2.2+cu118. Canonical
C0/C2, their checkpoints and datasets were not modified. No musical-quality
conclusion follows from these measurements.

## Verified result

| B2 arm | Repeat 0 records/s | Repeat 1 | Repeat 2 | Median |
|---|---:|---:|---:|---:|
| Original preparation and update | 1.857620 | 1.838590 | 1.834440 | 1.838590 |
| Compact prepared cache and optimized update | 5.332404 | 5.294579 | 5.410702 | 5.332404 |

The ratio of medians is **2.900269**: 190.03% higher measured throughput,
or approximately 65.52% less time for the same measured work. Individual
paired ratios are 2.870557, 2.879695 and 2.949512. The >=20% throughput
threshold is passed for this declared warm-cache workload.

Each arm has 800 measured and 160 discarded warmup draws, fixed B2 and the
same schedule. Three rotated pairs give 4,800 measured draws, 960 warmup
draws and 2,880 applied updates including warmup. All 2,400 measured rows
were checked independently: record IDs, scheduled/applied shifts, losses,
gradient norms and graph counts match across each pair. Typed loss-trace
hashes were reconstructed. Initial/final full-state, RNG and schedule
fingerprints agree across all arms. Full tensor snapshots were not exported;
their equality is supported by the accepted execution code and saved hashes,
not a second independent tensor rehash. The replay is uninterrupted and
identity-only; it does not replace checkpoint-resume or C2-path gates.

## Important workload boundary

The union of warmup and measured schedules contains **690 unique TRAIN
records**; the measured 800-draw suffix uses 602 distinct records. The compact
cache holds all of them: **8,420,704,345 accounted bytes** beneath the
8,589,934,592-byte cap. Every measured optimized arm has **800 hits, zero
misses and zero evictions**. The original baseline holds no unused prepared
cache. Both paths prewarm the same source population; optimized arms alone
receive a fresh bounded LRU, and cleanup happens outside measurement.

This is a strong all-hit steady-state result, not proof that a full 120k run
will be 2.9 times faster. TRAIN contains 1,295 records, while this cache also
has a 1,024-entry limit, so all 1,295 records cannot simultaneously be retained.
Full-TRAIN byte distribution and the long-run miss pattern have not been
established. The optimized population prewarm takes roughly
519–526 seconds per arm, excluded from the measured 148–151-second window.
Startup, later misses, validation and checkpoint time must not disappear
from an eventual end-to-end estimate.

The process lifetime RSS high-water mark reaches approximately 10.13 GB.
That counter carries history across arms and is not a fresh per-arm peak.
An 8 GiB accounted cache limit is not an 8 GiB process-RSS limit. The measured
Torch CUDA allocation peak is about 1.15 GB on this selected stream; finite
extreme TRAIN cases and aggregate prefetch memory remain untested here.

## Where time changed

For the first pair, reference load/collation/identity preparation takes
275.496 seconds. Optimized batch preparation takes 30.762 seconds, including
27.212 seconds decoding fresh sidecar dictionaries from original verified
JSON bytes. The full update takes 143.690 versus 105.607 seconds. Complete
measured windows are 430.659 versus 150.026 seconds.

These are within-pair measurements of the combined cache/update package.
Do not subtract timings from the separate update-only experiment to claim
an isolated cache effect. Nested timers overlap and are not additive.
Whole-device GPU samples average about 12.5–12.7% in baseline windows and
24.5–28.6% in optimized windows; they include the exempt desktop process and
are not a measurement of kernel occupancy.

## Evidence and next action

Archive SHA-256:
`cb0463d8c56e788fc9ec6cae1a009a2630de68ac5aa17fef0ea981a9f97cf791`
(2,048,054 bytes). Independent verification covers 36 run artifacts, nine
controller-manifest entries, nine accepted source bindings, primary rows,
cache/stage accounting and 4,602 GPU telemetry rows. Original terminal
records and artifacts remain immutable. Historical access flags are retained
as source evidence; the verifier itself uses no model, data loader or TEST.

The requested ordered CPU prefetch experiment remains next: direct worker
zero versus 1/2/4 workers on the same original preparation plus optimized
update. First run the registered target-CUDA fixture/resume gate and full
TRAIN size index, then the exact-job reservation gate and short screen.
Select a configuration only after inspecting throughput and worker memory;
the following three-repeat comparison needs its own sizing evidence.
Full-TRAIN memory checks and historical C2 transposition equivalence remain
required before recommending an optimized scientific training series.

The earlier expanded-cache slowdown and the update-only +17.72% result are
retained in their own immutable records; this result does not rewrite them.
