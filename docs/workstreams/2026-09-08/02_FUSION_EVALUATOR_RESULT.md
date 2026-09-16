# Fusion-aware POP evaluator: CPU acceptance

Date: 2026-09-08. Scope: the isolated evaluator subtask of workstream 02.
Worktree: `/Users/aavalsamakin/.codex/worktrees/a0bd/music-critic-v2`.
Inherited snapshot manifest SHA-256:
`9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf`.

## Own files and behavior

- `src/music_critic/experiments/analysisgnn/harmonic_fusion_evaluation.py`
  adds an enabled-fusion model/evidence binding and separate synthetic and
  registered TRAIN/VALIDATION report/preflight/protocol schemas.
- `tests/experiments/test_harmonic_fusion_evaluation.py` adds 29 synthetic
  CPU cases.
- This handoff is the third evaluator-owned file. The workstream parent owns
  STATUS, trainer/resume, scientific protocol and coordinator reporting.

The new core uses the existing `_predict_piece`, `align_event_start`,
`quality_gold_class_id` and `_summarize` functions. It never impersonates the
corrected model to enter the old core. Root37 probability marginalization,
full17 quality winners/subset9 gold, exact source-event starts, independent
root/quality masks, all confusion classes, same-beat conflicts and null
denominators retain the frozen arithmetic. Both views predict before each
target reader; native stays primary and metadata_removed stays secondary.
The enabled wrapper enforces all 12 harmonic heads' per-candidate alignment.

`fusion_transfer_preflight(request)` reuses the original complete global split
and explicit membership validation, but emits a distinct fusion binding.
`evaluate_fusion_transfer(request, experiment_record=...)` requires the matching
planned record under its `protocol.pop909_transfer_binding_fingerprint`, then
uses `load_fusion_checkpoint` before opening source inputs. Baseline records or
baseline checkpoints are rejected. The old loader and evaluator are unchanged.

The source-free entry points are `evaluate_fusion_synthetic` and
`evaluate_fusion_from_readers`. Evidence supplied by a reader must bind the
actual model contract, state and frozen flags before the source callback is
invoked. TEST is denied before readers in both execution paths.

## Verification

Final command, from this worktree:

```sh
PYTHONPATH=src:/private/tmp/music-critic-fusion-20260908-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-pop-eval-20260908/bin/python3 -m pytest -q tests/experiments/test_harmonic_fusion_evaluation.py tests/experiments/test_harmonic_logit_fusion.py tests/experiments/test_harmonic_fusion_checkpoint.py tests/experiments/test_pop909_cl_frozen_evaluation.py tests/experiments/test_pop909_cl_transfer.py tests/experiments/test_pop909_cl_frozen_checkpoint.py tests/experiments/test_pop909_cl_frozen_projection.py tests/experiments/test_pop909_cl_frozen_views.py tests/experiments/test_phase9eb5c_corrected_model.py tests/test_repository_contract.py
```

Result: **229 passed, 2 existing Torch JIT deprecation warnings in 14.14s**;
no skips or deselections. Full output is ignored
`outputs/fusion_evaluator_acceptance/pytest.log`.
`git diff --check` and compilation of both evaluator-owned Python files passed.

Actual nonzero fusion weights survive the strict checkpoint roundtrip and
change fixture predictions from C to D and from major to a full17 class
outside the accepted gold subset. This changes confusion cells, root/quality
accuracy and joint accuracy while preserving the base tensors and raw input
fingerprints. Zero fusion reproduces complete old view reports exactly.
Aligned row permutations preserve predictions attached to their correct beats;
cross-head permutations fail before targets. Changed targets leave both raw
input/logit bindings unchanged. Forged or missing checkpoint evidence and
model tensor/mode mutation are rejected.

The first new-test run had 23 passes and four fixture errors (two missing
`return_layers` keyword adapters and two synthetic record filename/ID
mismatches); these were corrected before acceptance. An initial broader
collection failed because the reused environment lacked the already-declared
Hydra training extra. The parent supplied an isolated dependency directory;
no project dependency or shared environment was modified by this subtask.

## Training-checkpoint handoff and claim boundary

This evaluator accepts the existing strict **model-only** fusion schema.
The full trainer checkpoint must first pass the new full resume validator with
explicit config, bindings, table and arm identity. A new immutable export from
that verified session can then use `harmonic_fusion_checkpoint.create_model_payload`
for fusion, or the original `create_model_only_payload(session.model.base)` for
the equally continued control. Retain the full-checkpoint SHA/size and export
SHA/size/state linkage in the parent run evidence. The export does not restore
optimizer, scheduler, sampler or training RNG, and must never be presented as
historical C2 resume.

Only tiny synthetic CPU tests ran. No real corpus, trained checkpoint, SSH,
GPU or TEST access occurred; no standalone substantial-experiment record is
required for these ordinary tests. Legacy was neither inspected nor reused.
Real evaluation remains gated on separately registered verified artifacts and
the coordinator's resource queue. POP is a development transfer diagnostic;
it is not checkpoint selection or a new independent zero-shot benchmark.

Independent trainer review reproduced one parent-owned issue: introducing a
validation callback at a non-cadence resumed update appended a validation row
which the strict checkpoint correctly rejected. The reproducer and suggested
pre-callback guard were sent to the parent; no trainer/resume files were edited
by this subtask. The parent owns its remediation and final combined gate.
