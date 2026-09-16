# Prospective paired wall budget, handoff006

The coordinator reports native005 acceptance: 706 tests passed in143.26s,
with metadata/reference admission successful. The reference remains unlaunched
while SSL occupies the reserved GPU. These are coordinator observations;
this local task does not claim a new native run.

The current SSL timings supplied by the coordinator, about4seconds/update
and8minutes per full162-record validation, suggest that1024updates plus
six full validations may need about6976seconds. They are a planning estimate,
not measured P512 throughput or a guarantee of completion within the limit.

Only the prospective pair wall bound changes from2400 to7200seconds:
the strict public request admission, generated paired request, private attempt
default and the pair command for the existing external resource supervisor.
Reference admission/request/finalization and its supervisor command remain
600seconds. The GPU ceiling remains16384MiB. The in-process deadline still
checks at Python boundaries; the existing external wrapper also covers a
blocked native call. Timeout behavior and failure accounting are unchanged.

Scientific fields remain byte-identical:512updates/arm, B2, seed17,
optimizer/loss/warmup, shared full-record draw schedule, full validation at
0/256/512, checkpoint cadence, source data pins and model state. This is an
operational budget adjustment, with no architectural decision or new gate.
No experiment ID is allocated by this change.

Keep sealed005 and its unlaunched runtime/source/reference-plan snapshots.
Deploy the complete006 candidate to the new `fusion-006-source` directory,
or apply the incremental patch to an exact005 source baseline. Generate fresh
006 runtime/source metadata and requests;005 requests retain their old source
pins and must not be dispatched as006. `EXP-FUSION-TRAIN-REFERENCE-001` is
still unused according to the coordinator and is retained, as are the agreed
P512 IDs. Synchronize the regenerated planned record before launch while
preserving the earlier unlaunched planned copy. Do not overwrite any terminal
record or launched attempt. The006 README contains complete commands.

Changed files are the pilot/dispatch modules, their three targeted test files,
this note and STATUS. Validation is limited to those targeted tests plus
compile, diff, registry/ledger and package-integrity checks; the706-test suite
is not repeated. Exact counts, logs and manifests are in the006 handoff.
The coordinator's unrelated SSL test-fixture integration fix is outside this
delta and outside this isolated full candidate.

No legacy logic was inspected, reused or rejected. No main-checkout edit,
SSH, CUDA execution, training, real payload/checkpoint tensor loading or TEST
access occurs in this task. Next phase: coordinator checks the006 delta on
the target, regenerates metadata/plans, then launches the existing registered
600-second reference and7200-second P512 pair when the GPU is available.
