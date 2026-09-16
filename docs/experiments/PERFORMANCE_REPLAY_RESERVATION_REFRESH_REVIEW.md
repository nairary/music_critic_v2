# Worker telemetry and reservation refresh — 2026-09-06

Source experiment: `EXP-9EP1-PREFETCH-RESERVATIONS-002`.
Independent import: `EXP-9EP1-GPU-EVIDENCE-009`.
This is a preparation gate for the requested worker comparison, not a speed,
memory-capacity or model-quality result.

## Accepted scope

The approved telemetry overlay replaced exactly two executable modules and
their two test files in the isolated performance checkout. All other 254
executable source files remain unchanged. Original file bytes and modes were
saved before replacement. The new complete source fingerprint is
`2e924725ce67ded218c87188bb17a13b98a621cf943b467be6eb2c99b9735ef3`.

The subsequent sequential CPU measurement completed all **120 reported B2
jobs / 240 scheduled draws**, covering **219 TRAIN records**, in
**131.416819 seconds**. Its schedule and frozen loader bindings match the
accepted previous sizing run. No optimizer updates or worker pool are part
of this sizing protocol. The controller finished at **16:13:05 UTC**; its
PID and observed owned process had exited before export.

| Quantity | Verified report |
|---|---:|
| Maximum serialized job | 41,565,463 bytes |
| Sum over 120 serialized jobs | 1,874,422,579 bytes |
| Unchanged reservation per job | 477,102,080 bytes (455 MiB) |
| Maximum uncommitted jobs | 5 |
| Global serialized reservation cap | 2,385,510,400 bytes (2,275 MiB) |

The reported maximum and sum equal the previous sizing result. The new
preparer base fingerprint is
`b1628e584470d4ebd04b5937bfd80efe705ca7f38b736989601b5d088531cd4a`;
the reservation manifest fingerprint is
`0e204bc562ba8fa92d33775dfe4e346bfa68df457a5fb38fed0c55371c96eeae`.
The final bound preparer fingerprint is
`ce91ab6cd075b7b3a90d8ac82787693ed930536fafed128df2e7352991da7782`.
Old reservations were not silently reused after changing implementation bytes.

Sizing used the original Python 3.12.8 / Torch 2.2.2+cu118 / PyG 2.6.1 runtime.
Actual parent Torch getters were intra-op **1**, inter-op **28**. This is
sequential parent telemetry, not evidence of worker thread settings.

## Failed test attempt and remedy

The first target gate returned **302 passed, 10 failed, zero skipped,
39 warnings in 155.40s**. Installed PyG 2.6.1 checks CUDA tensor operations
when imported after pytest. Custom fault-injection workers encountered that
import side effect before reaching their intended test scenario. A bounded
diagnostic recorded the call site; its interception was diagnostic only and
was not used for runtime acceptance.

A separate test bootstrap imports PyG before pytest, including when Python
re-executes the bootstrap in spawned children. No project module, project
test, worker guard, device visibility or model behavior was changed for this
remedy. The same **312 tests passed with zero skips or warnings in 156.24s**
(157.956874s wrapper). Both attempts and the diagnostic are retained unchanged.
The accepted runtime receipt binds the second command, bootstrap and launcher
bytes, while preserving the first failed receipt and its logs.

## Evidence and limits

The exported archive has **44 regular files**, including all **12 source
artifacts**, **30 controller manifest entries** plus that manifest, and the
original terminal record. SHA-256:
`1566ed5f43ff63a279e8dca2728c7bd2f99c6f708cfd0f9dc23fce2b9061ab4d`;
size **148,019 bytes**. The independent verifier accepted its first actual
import attempt and authenticated all 256 source bindings and the prior
memory/index/reservation evidence. Verifier tests: author **109 passed in
12.23s**, root **109 passed in 12.21s**, no skips or warnings.

Serialized objects are not included. Sizes and payload digests are
authenticated as reported; aggregates, schedule and manifest identity hashes
were recomputed. This import does not independently reserialize the jobs.
The global cap covers serialized reservations; decoded batches, worker
preparation, IPC copies, model state and GPU memory are additional. It does
not establish aggregate RAM readiness, throughput or universal OOM safety.

Original source access observations remain unknown independently of the
declared TRAIN-only policy. The offline importer read no evaluation or TEST
data, executed no imported code and deserialized no tensors. Canonical runs,
model/data semantics, historical C2 track statistics and legacy remain unchanged.

Next is the separately registered **one-repeat direct0/1/2/4 worker screen**:
B2, 200 measured plus 40 warmup draws per arm, the same frozen preparation
and optimized parent update. Selection from that screen still requires a
separately sized three-repeat acceptance and the real TRAIN C2 gate.
