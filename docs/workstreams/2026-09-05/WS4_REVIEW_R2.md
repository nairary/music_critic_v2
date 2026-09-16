# WS4 independent review R2 — 2026-09-05

Status: **accepted for isolated integration**, not integrated into main.
Worktree: /Users/aavalsamakin/.codex/worktrees/46dc/music-critic-v2.
Base/HEAD: f0e0886161c9f67786011508006db61c803c1752.
Accepted content fingerprint:
d89fa514991a436958fd24a1fc3aea7728a3eb9b372d5a82721f0006041666b9.

The coordinator reproduced the fingerprint over the 14 sorted code/test/
contract/STATUS/DECISIONS files using path, NUL, complete bytes, NUL for each.
The two result reports are excluded. This binds the accepted uncommitted revision.

## R1 closure
WS4-R1-001 is resolved. targets_from_chord_evidence now counts only spans with
kind=trailing_unannotated. Two added regressions use production evidence
dataclasses and inspect the per-piece record plus both view summaries:
- expected_missing/missing_chord_targets: missing pieces/groups=1, trailing=0;
- available/trailing_unannotated: missing pieces/groups=0, trailing=1.

The coordinator read the changed implementation/tests and repeated the full
focused CPU gate from [R1](WS4_REVIEW_R1.md), including the two new cases.
Result: **117 passed, 1 CUDA-only case deselected, 2 warnings in 5.16s**.
Warnings are upstream Torch JIT deprecations; no required test was skipped for
a missing dependency. git diff --check passed.

Root37-to-PC12 sum-softmax, full17 quality winner versus subset9 gold,
independent masks and coverage, exact source-event-start alignment,
native/metadata_removed closure and restricted model-only loading passed R1
review. R2 changes only the reported coverage classification and its tests/
status reporting. No additional blocking findings remain.

## Boundaries
Acceptance establishes synthetic CPU mechanics only. Actual C0/C2 envelope
compatibility and real POP909 transfer metrics are unverified. TEST, real
sources/checkpoints, training, SSH, legacy and experiment records were not
used or changed. This is not a critic/PLL/quality score or full 18-task common
ontology. R1 is retained as review history; acceptance requires this new revision.

Next: preserve this worktree, integrate with the other accepted workstreams in
an isolated copy, and verify the combined result before marking integrated.
