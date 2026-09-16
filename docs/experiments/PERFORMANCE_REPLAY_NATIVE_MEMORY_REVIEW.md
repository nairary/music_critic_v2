# Finite native B2 memory gate — 2026-09-06

`EXP-9EP1-B2-MEMORY-REPLAY-001` completed on RTX 3090 at
15:19:49 UTC, after 513.229 seconds of registered execution. The fixed
TRAIN index supplied five extremal records and all 25 ordered pairs,
including repeated records. Each pair ran in two fresh processes with
two updates per arm: 50 children, 100 applied updates and 200 derived draws.
The original planned upper bound of 256 updates remains unchanged.

Every pair has identical initial state and identical per-update loss,
gradient norm, model, AdamW, scheduler, scaler and RNG fingerprints.
All 18 head reports are present, preserving missing-target masks and null
losses. This is identity-only original CPU preparation with either the
reference or optimized update. It includes no prepared cache or prefetch.

Independent import `EXP-9EP1-GPU-EVIDENCE-008` authenticates 506 regular
archive files: 483 run artifacts, 22 controller files (21 manifest entries)
and the original terminal record. It verifies all 100 durable applied rows,
100 state/loss rows, 25 comparisons and the unchanged source256 mapping.
Memory peaks are recomputed from exported samples and child snapshots.
No imported code, corpus payload, checkpoint or tensor object is executed
or loaded by this offline verification.

## Observed memory

Values below are maxima over the 25 children of each arm, in GiB (2^30 bytes).

| Observation | Reference | Optimized |
|---|---:|---:|
| Parent-sampled child RSS | 1.451691 | 1.466621 |
| Child-reported getrusage high-water mark | 1.470757 | 1.466545 |
| Torch CUDA peak allocated | 1.841464 | 1.841478 |
| Torch CUDA peak reserved | 2.080078 | 2.080078 |

The largest CUDA allocation occurs for the duplicate Liszt Dante pair
(`case-018`): 1,977,256,960 bytes reference and 1,977,271,808 optimized.
Both reserve 2,233,466,880 bytes. The largest sampled child RSS occurs
for Kozeluh followed by Liszt (`case-013`): 1,558,740,992 and 1,574,772,736
bytes. The maximum observed child getrusage high-water mark is
1,579,212,800 bytes. These differing measures and sampling times must not
be combined into one supposedly exact peak. All 50 children have at least
one RSS sample during the native-preparation phase; sampling remains a
lower bound, not allocation attribution.

The finite cases completed without an OOM. No material GPU-memory increase
appears in this comparison. It does not establish memory safety for all
TRAIN pairs, long runs, shifts, B4/B8, caches or simultaneous workers.
The 52.81 GB host availability observed before launch is an instantaneous
preflight value, not the experiment's aggregate memory limit. Imports,
setup and state hashing are included in child process RSS observations.
Torch allocator counters exclude other CUDA allocations and processes;
device-free counters concern the whole device.

## Provenance and limits

Runtime is Python 3.12.8, Torch 2.2.2+cu118, CUDA build 11.8; all child
reports agree. Actual child Torch intra/inter-op getters and the four
native environment limits are one. PyG 2.6.1 is bound by the target test
receipt, not separately queried in every child. Controller ownership
records contain 57 process identities and no cleanup signals; their
absence was checked again before export. This is not a historical OS I/O
audit. Original source TEST/VALIDATION observations remain null under a
closed declared policy. This import accessed neither dataset.

The source hashes are observations of the reviewed producer; the importer
checks their consistency, not the underlying tensor computations. Two
updates in fresh processes test finite allocation and state agreement;
they do not test a 120k trajectory or sustained memory growth.

Root controller tests: 119 passed, one Linux skip, two warnings in 28.40s.
Target Linux controller tests: 120 passed, zero skips, 39 upstream warnings
in 18.90s (19.495s wrapper). The unchanged target source had already passed
the preceding 513-test integration gate. Verifier tests: author and root
each 71 passed, zero skips/warnings in 24.65s. Controller and verifier
received independent review; real archive verification passed on attempt001.

Next: deploy the separately tested worker runtime observation variant,
remeasure its exact job reservations, then run the fixed direct0/1/2/4
worker screen. Selected three-repeat throughput and aggregate worker RAM
checks remain required, along with historical real TRAIN C2 equivalence
before scientific use. This result changes no model-quality conclusion.

Archive SHA-256:
`b1961937dc97dcd30376f885ba8d632bbc4d4dffb532acfa3af2eee2c255dc3f`.
