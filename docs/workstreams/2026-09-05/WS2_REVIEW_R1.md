# WS2 independent review R1 — 2026-09-05

Status: **changes requested; independent review complete**.
Worktree: /Users/aavalsamakin/.codex/worktrees/437e/music-critic-v2.
Base/HEAD: f0e0886161c9f67786011508006db61c803c1752.
Verified author's four-file implementation manifest:
954985d47b8a12d24161f67effd69f16d7be1c409ee32031defc386eb51ad440.
Coordinator seven-file content fingerprint:
dbc8edb57bc467f5652979075df9a59465fd77d0a5c9cc4d2b853a34cd14c233.
The latter hashes compact sorted JSON of relative path to file SHA-256 for
all changed/new files, excluding the two WS_RESULT/WS2_RESULT handoff files.

## WS2-R1-001 (P2): masked rows index a zero-candidate head
The harness passes original alignment into the historical corrected loss,
which gathers candidate indices before applying validity masks. A valid
synthetic canonical graph with no notes/onsets still supports raw inference.
A fully masked sidecar row referring to a missing note is correctly diagnosed
by alignment as exact_alignment_failed with valid_mask=False, candidate=0.
The subsequent gather on logits with shape (0,C) raises RuntimeError instead
of omitting that head's loss. Root independently reproduced the failure via
the public harness and synthetic helpers; no real input was opened.

Requested local correction: preserve raw predictions and original alignment/
diagnostics/counts; safely omit invalid zero-candidate rows only in a private
loss view, without fake logits/class-zero supervision or changes to frozen
corrected_training.py. Regressions must cover all-masked no-loss, remaining
valid harmonic heads with gradients, and mixed SSL with a separate normal batch.

## WS2-R1-002 (P2): stale contract fingerprint is accepted
Python dictionary equality treats active_head_count=18 and 18.0 as equal.
Changing that supplied contract field while retaining its old fingerprint
therefore passes load_theory_warmstart although the actual canonical contract
fingerprint differs. Root independently reproduced acceptance and the stale
hash. Require type-sensitive canonical comparison and verification of the
supplied self-fingerprint before destination mutation, with atomic rejection.

## WS2-R1-003 (P2): distinct Parameter objects can alias storage
The alias preflight checks only Python object identity. Replacing destination
inactive encoder.local_baseline.task_heads.heads.task_00.0.bias with a new
Parameter(destination.feature_mask_token.detach()) retains compatible shape
and passes transfer. The inactive parameter is frozen, but its storage aliases
the active mask token: updating that token changes the supposedly excluded
weights. Root independently reproduced all three observations. Require a
preflight storage-alias check for distinct parameters within the supplied
source/destination models, while retaining intentional post-transfer shared
objects across the returned branches. Add atomicity regressions.

## Independent gate and limits
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q -p no:cacheprovider \
tests/ssl/test_theory_warmstart.py tests/ssl/test_theory_retention_harness.py \
tests/ssl/test_transfer.py tests/ssl/test_model.py tests/ssl/test_decoder_objective.py \
tests/ssl/test_prepared_mask_binding.py tests/ssl/test_hierarchical_prepared_binding.py \
tests/experiments/test_phase9eb5c_corrected_model.py \
tests/experiments/test_phase9eb5c_corrected_training.py
```
Result: **213 passed, 6 CUDA-only skips, 2 warnings in 32.87s**.
git diff --check passed. Warnings are upstream Torch JIT deprecation.
Independent bridge, harness, and documentation reviews found no additional
blockers. Existing tests correctly establish object sharing, ordinary transfer,
active gradients, target detach, mode restoration, and sidecar independence,
but did not cover the three cases above. Scientific effectiveness, actual
checkpoint compatibility, corpus readiness and forgetting resistance remain
unmeasured. No source/legacy changes, real-data runs, checkpoints or TEST access
were performed by the coordinator. Await corrected revision.
