# Historical C2 CUDA and fresh-object restoration — 2026-09-06

`EXP-9EP1-C2-CUDA-EQUIVALENCE-001` completed on RTX 3090. It executed
**108 attempted/applied updates and 216 draws**, with no skipped updates:
12 B2 jobs in each of nine branches. The branches are the original update
reference and optimized direct0/1/2/4 paths, uninterrupted and resumed.
The registered run took 63.529 seconds; this instrumented diagnostic is not
a production throughput measurement.

Independent import `EXP-9EP1-GPU-EVIDENCE-007` verifies all 108 complete
trace rows, 18 run artifacts, 19 controller entries and the complete 256-file
source mapping. The mapping preserves the earlier 254 files and adds only
the separate C2 fixture implementation and CLI. The four own checkpoint
files receive byte hashes; no checkpoint objects are loaded by the importer.

Every verified step reports all 18 heads with two valid rows, one forward
and one native target alignment. Raw inputs, sidecars, logits, alignment,
losses, gradients before/after clipping, model/AdamW/LR/scaler/RNG state
fingerprints agree across all nine branches. The synthetic schedule covers
12 shift pairs and duplicate record IDs at shift 6. It preserves the
historical stale track mean/min/max features; WS1 is not introduced.

At the five-update checkpoint boundary, the resumed branches discard their
training objects and construct fresh model, optimizer, scheduler and scaler.
The fresh state differs from the saved state. After restoration it equals
the saved state, which also matches the post-update-5 trace. The remaining
updates agree with uninterrupted execution. Nonzero-worker branches exercise
pending speculative jobs. Separate negative tests reject a no-op restore,
omitted Adam restoration and omitted RNG restoration.

This closes the observed fresh-object restoration coverage gap for this
shared restoration helper on the synthetic historical C2 protocol. It does
not retroactively strengthen the old v5 fixture report. It also does not
establish real TRAIN C2 correctness, full vocabulary/AN coverage, production
memory bounds or scientific training readiness. The raw arithmetic oracle
and tensor/state fingerprints are observations from the reviewed producer;
the offline importer checks their consistency without independent inference.

Runtime: Python 3.12.8, Torch 2.2.2+cu118, PyG 2.6.1. Parent Torch
intra/inter-op getters return 1 and the four inherited native limits are 1.
The worker code sets both Torch limits to 1, but the exported worker fields
are constants rather than acknowledgments read back from each worker.
This fixture therefore does not provide per-worker runtime telemetry;
production worker acceptance must record those actual observations.
TF32 flags are retained as observed: matmul false, cuDNN true. Deterministic
algorithms are enabled. The accepted controller observes 23 process identities
and normal completion without cleanup signals; a separate post-run check
confirmed their absence before export. This is not a system-wide I/O audit.
The original source record's unknown TEST/VALIDATION access fields remain
unchanged; this import opens neither dataset and runs no model.

Validation: local C2 integration **181 passed, five CUDA skips, two warnings
in 91.78s**; local controller **70 passed, one Linux-only skip in 2.33s**.
The combined target gate passes **513 tests, zero skips, 40 warnings in
140.66s**, including the Linux process ownership test. Verifier tests:
author **53 passed in 2.74s**, root **53 passed in 2.80s**. Independent
source review accepted the controller, C2 helper and verifier; real archive
verification passed on its first attempt.

Next: the separate finite fresh-process memory experiment from the accepted
full TRAIN index, then the fixed direct0/1/2/4 production worker screen with
actual worker runtime observations. Any changed preparation implementation
requires new source-bound reservation evidence before that screen.
Selected three-repeat acceptance needs a new schedule and reservation record;
historical real TRAIN C2 equivalence remains required before scientific use.
Archive SHA-256:
`8a8155003f95e501cccae8b617cdde41ef84c372d58fb26e4da711e363d18c3d`.
