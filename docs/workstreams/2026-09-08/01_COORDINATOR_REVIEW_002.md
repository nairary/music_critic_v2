# Coordinator review 002 — context diagnostics revision r1, 2026-09-08

Disposition: **one resource-supervisor correction required before CUDA**.
The three prior findings are closed: durable forward accounting, honest
checkpoint selection provenance, and a new sealed package including STATUS.
No CUDA or musical-quality acceptance is inferred.

## Independent verification

- Verified manifest SHA-256
  `adece583566e6d0c7075250487c54932a4ad2c72373215c0128761150ba7b446`,
  patch SHA-256
  `60be8626551193b3c4c62046f9e4ecb47747f238c216d817d4a9fe7554328e52`,
  and archive SHA-256
  `86bb53c2bcdbabc50c9c7631f1163c30404f47fba36f878217a046ebc074e77f`.
- All 18 archive members are unique safe regular files and match their
  manifest hash/size. Before-hashes match the common source snapshot; current
  worktree STATUS matches the new sealed STATUS.
- A separate copy built from the common snapshot and verified archive bytes
  passed five context modules plus repository contract: **138 passed,
  0 failed/skipped, 2 existing Torch JIT warnings, 4.71 seconds; exit 0**.
  The first test command used a wrong directory glob and did not invoke pytest;
  the corrected command used the explicit manifest test paths.
- Independent code review confirmed fsynced started receipts before inference,
  completed counts only after successful explicit CUDA synchronization, unknown
  in-flight work, exact physical GPU UUID binding and fail-closed monitoring.

Evidence: `outputs/coordination/research_parallel_20260908/context_review_r1/`.
The preselected C2 export is SHA-bound; best115k/seed17 remains declared parent
metadata, not independently recovered historical selection/update. The earlier
artifact-only results were not recalculated or imported into the root registry.

## Required narrow revision r2

`scripts/context_diagnostics_resource_wrapper.py:205` calls `child.poll()`,
which can reap an exited process. Its finally block at line224 then calls
`killpg(child.pid, SIGKILL)` via line125 without retaining ownership of that
numeric PID/PGID. PID reuse between reaping and signaling can target an unrelated
process group. Probability does not establish ownership.

Retain the leader identity until group cleanup is complete, for example with
supported non-reaping exit observation followed by group cleanup and then wait,
or another verified ownership mechanism. A second numeric PID check is not a
race-free fix. Cover normal exit, ordinary descendant cleanup and simulated
reuse/ownership regression without signaling real unrelated processes.

This correction was sent to the verified originating task. Preserve handoff-r1
and terminal artifacts; deliver a newly sealed r2 and reseal the still-planned
GPU protocol against final source/wrapper bytes. No GPU job was launched.

The external memory threshold is sampled whole-device usage, not a hard CUDA
allocation cap; transient peaks remain possible. Dispatch also requires an
exclusive slot, fresh evidence paths and verified native thread limits. The
real Dilemma local-versus-C2 comparison still needs authoritative raw/target
exports, official component closure and identical candidate joins. No main
model code, legacy, TEST, SSH, commit, push or merge changed in this review.
