# WS4 independent review R1 — 2026-09-05

Status: **changes requested**. Worktree:
 /Users/aavalsamakin/.codex/worktrees/46dc/music-critic-v2.
Base/HEAD: f0e0886161c9f67786011508006db61c803c1752.
Author revision fingerprint:
320dd4fa82b6865c5238ce43ebdaf70c7c59748afbe60a8657d6aecb06c7a674.

## Finding WS4-R1-001 (P2)
targets_from_chord_evidence in evaluation.py counts all evidence.trailing_spans
as trailing_unannotated_span_count. The existing POP adapter uses that tuple
also for a span with kind=missing_chord_targets when status=expected_missing.
The resulting aggregate wrongly counts an entirely missing annotation source
as a trailing-unannotated region.

The coordinator reproduced the error using the production evidence dataclasses,
with status=expected_missing, no blocks/no_chord_spans, and one
Pop909ClCoverageSpan(kind=missing_chord_targets, start_tick=0, end_tick=1920,
ppqn=480, available=False, value=None). Current result is 1; expected is 0.
This does not change accuracy/F1, but violates the separate coverage categories.

Requested correction: count only kind=trailing_unannotated. Regression must
exercise actual adapter evidence shape through conversion and report aggregation,
checking missing-piece/group counts, zero trailing count for missing source,
and a positive trailing count for a true trailing_unannotated span.

## Other review evidence
Root/projection, quality full17/subset9 confusion, independent masks,
event-start alignment, raw metadata closure, restricted model-only loading,
RNG preservation and synthetic/TEST access guards were reviewed from actual
new files. Three additional independent static reviews found no other blockers.

Independent CPU gate:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q -p no:cacheprovider \
tests/experiments/test_pop909_cl_frozen_checkpoint.py \
tests/experiments/test_pop909_cl_frozen_projection.py \
tests/experiments/test_pop909_cl_frozen_views.py \
tests/experiments/test_pop909_cl_frozen_evaluation.py \
tests/experiments/test_phase9eb5c_corrected_model.py \
tests/evaluation/test_checkpoint_evaluation.py \
-k 'not bounded_evaluation_uses_concrete_cuda_runtime'
```
Result: **115 passed, 1 deselected, 2 warnings in 5.18s**. The excluded existing
case requires CUDA; warnings are upstream Torch JIT deprecations. Passing
existing tests did not cover the adapter evidence edge case above.

Actual trained-checkpoint compatibility, real-data evaluation and scientific
quality claims remain outside scope. Await corrected revision and targeted
regression before acceptance. Preserve this review as R1 history.
