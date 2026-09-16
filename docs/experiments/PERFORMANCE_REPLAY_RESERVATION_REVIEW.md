# Exact B2 serialization reservation gate — 2026-09-06

`EXP-9EP1-PREFETCH-RESERVATIONS-001` completed 120 sequential CPU jobs
in 130.595 seconds. Independent import `EXP-9EP1-GPU-EVIDENCE-006` checks
the archive, 12 run artifacts, 16 controller entries, complete 254-file source
mapping, prior full TRAIN index bindings and every schedule/size row.
The original terminal record and artifacts are retained unchanged.

The schedule contains 240 draws from 219 TRAIN records: 20 warmup B2 jobs
and 100 future measured B2 jobs. Applied shifts are zero. All 120 reported
serialized payload lengths fit their unchanged 477,102,080-byte (455 MiB)
reservations. Maximum length is **41,565,463 bytes**; total lengths are
**1,874,422,579 bytes**. The total is cumulative serialized volume, not RAM.
The five-slot global reservation cap remains **2,385,510,400 bytes**
(2,275 MiB). It must not be resized in the subsequent screen because the
reservation value is part of each serialized job.

The source ran Python 3.12.8, Torch 2.2.2+cu118 and PyG 2.6.1. Its recorded
Torch intra-op setting is 1 and inter-op is 28; it created no worker processes.
This is sizing evidence, not acceptance of the later worker/thread settings.
The worker replay must observe its own required settings and enforce its
payload reservations at runtime.

The importer checks exported lengths, hashes, metadata and arithmetic without
loading corpus, payload or checkpoint objects. Actual serialized payload bytes
are not retained, so the importer does not independently remeasure those
lengths. It verifies the exported schedule and full-index subset, but does not
resample the historical orbit permutation without the eligibility payload.
Historical source access fields remain unknown; the offline import itself
opens neither TEST nor VALIDATION.

No model updates or throughput measurements occur in this sizing gate.
Reservations do not bound decoded objects, IPC copies, worker/process-tree
RSS, activations or GPU memory. The old fixture's fresh-runtime restoration
gap also remains open. Next are the separately registered historical C2 CUDA
restoration gate and finite fresh-process memory gate, followed by the fixed
one-repeat direct0/1/2/4 screen. A selected three-repeat comparison requires
a new 800+160-draw schedule and its own sizing record.

Validation: verifier author **53 tests passed in 2.27s**; root **53 passed
in 2.05s**, with independent source review. Real archive verification passed
on its first attempt. Archive SHA-256:
`7f810378ad0bbad0a96cc657242236c95c21fc9059cbafbcc16e5881f81d639d`.
