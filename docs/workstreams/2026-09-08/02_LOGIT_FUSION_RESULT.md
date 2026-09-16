# Logit fusion integration — first-stage handoff, 2026-09-08

## Outcome and isolation

The requested first mechanics stage is implemented and CPU-tested: bounded
control/fusion training sessions, complete strict resume, Dilemmadata metric
collection, fusion-aware POP evaluation, fixed pilot config and an executable
synthetic CUDA queue package. No scientific fusion result is claimed.

Worktree: `/Users/aavalsamakin/.codex/worktrees/a0bd/music-critic-v2`.
Bootstrap verified 752 regular files / 13,124,819 bytes, source HEAD
`da8c9dcd999a4b522f87f9a20bf3506ad26497b7` plus dirty `codex/develop`, against
manifest `9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf`.
The worktree Git base is `f0e0886161c9f67786011508006db61c803c1752`; its full
Git diff includes inherited work and is not the task-owned patch.
Bootstrap receipt is `outputs/coordination/source-bootstrap.json`.
Own patch and file hashes are in `outputs/fusion/handoff/`.

## Implemented behavior

- `harmonic_fusion_training.py`: opt-in sessions around the unchanged update,
  loss, B2 sampler and historical C2 transposition. Each arm owns all common
  parameters and independent fresh AdamW/scheduler/RNG state. Reader work does
  not consume the dropout stream. `paired_sessions_from_checkpoint` uses the
  old restricted SHA/size loader, binds the source tensor fingerprint and
  supplies identical base weights to both arms. In-memory synthetic sessions
  are explicitly less restrictive than the future registered corpus boundary.
- `harmonic_fusion_resume.py`: separate full checkpoint schema with complete
  ordered model/classes; source/data/config bindings; named optimizer step
  counters and moment slots; exact scheduler; disabled scaler; primitive/tensor
  Python/NumPy/Torch RNG; sampler table/cursor; update/draw and validation
  history. Same checked bytes feed `weights_only=True`, with no unsafe
  fallback. Rejection occurs before a new session is returned; caller RNG
  and live objects survive failures. Immutable atomic save refuses old paths.
- Dilemmadata evaluation retains the old eight-head primary, all 18 heads,
  joint correctness, per-class confusion/support and component predictions.
  The extra consistency observation is exactly whether argmax spelled-root
  PC belongs to argmax absolute pitch-class set, over all raw beat candidates;
  it is separate from correctness and music quality. Cadence/metrical are
  retained. Native-TRAIN-unseen retains its original meaning.
- `harmonic_fusion_evaluation.py`: separate strict enabled-fusion binding for
  existing POP source readers, projection, rational alignment, masks and
  arithmetic. Nonzero weights change predictions in acceptance tests. Both
  native primary and metadata_removed secondary stay fixed. Real transfer
  needs its new matching planned binding; old loaders/APIs remain unchanged.
  Full training resumes can emit a new immutable model-only export with the
  parent full-checkpoint SHA/size, as demonstrated by the smoke.
- `harmonic_fusion_run.py` and `run_harmonic_fusion_pilot.py`: source-free
  executable gate with six uninterrupted and 3+3 restored updates per arm,
  complete final-state comparison, strict fusion export/reload, artifact hashes
  and actual counts. CUDA CLI requires a planned engineering record and
  inherited thread limits. The CLI intentionally exposes only this synthetic
  gate; it does not pretend a missing registered corpus dispatcher exists.

Use the public session API with an explicitly attested TRAIN reader returning
`(single_record_raw_batch, sidecar)`, frozen weights and `FullOrbitPair` table.
`run_until(..., validate=..., checkpoint=...)` provides fixed validation/save
cadence and best selection with earliest exact tie. All scientific callers
must supply validation from update zero; adding it after unvalidated updates
fails before callbacks. Checkpointing without validation remains supported for
synthetic mechanics only. The final scientific dispatcher must additionally
require all scheduled evaluations and save both final and selected best.

Zero-init establishes equal first forward/loss and pre-clipping common
gradients. Global clipping includes new fusion gradients; post-step base
weights are therefore not promised identical. This expected difference is
documented rather than weakening the shared clipping rule.

## Exact validation and artifacts

Final combined gate: **357 passed, 0 failed/skipped/deselected, 2 existing Torch
JIT deprecation warnings in 39.86s**. This includes all 100 task-owned tests,
the existing fusion/model-only cases, corrected model/training, full-orbit,
synthetic learning, frozen POP loader/core/evaluation/projection/views,
transfer/export, and repository contracts. No pre-existing implementation or
test was modified. Command and full log: `outputs/fusion/final-acceptance/`.

Runtime: Python 3.12, Torch 2.13.0, PyG 2.8.0.post1; CPU thread limits one.
Python executable: `/private/tmp/music-critic-pop-eval-20260908/bin/python3`.
The already declared `training` extra `hydra-core` was absent; it was installed
only to `/private/tmp/music-critic-fusion-20260908-deps`, included with
`PYTHONPATH=src:/private/tmp/music-critic-fusion-20260908-deps`. Project dependency
files and the shared Python environment were not changed.

Final CPU CLI smoke: `outputs/fusion/cpu-smoke-002/report.json` and
`artifact-manifest.json`, **passed**, 24 actual synthetic updates / 48 TRAIN
fixture draws, 12 single-fixture validation reader calls, 4.541112625000096 s.
Each restored final model/optimizer/scheduler/RNG/sampler/history equals its
uninterrupted arm; saved full checkpoints and the enabled model-only export
are SHA/size bound. No target CUDA claim follows from that wall time.
The preceding successful smoke is retained separately as `cpu-smoke-001`.
Ordinary synthetic tests/smokes are not registered as scientific experiments.

`compileall`, `git diff --check`, experiment registry validation (47 records),
and generated ledger consistency also pass. The sole new experiment record is
`EXP-FUSION-CUDA-SYNTHETIC-001`, **planned / not launched**. Its observed access
fields remain null; declared policy remains separate.

## Own changed files

New implementation files (under `src/music_critic/experiments/analysisgnn/`):
`harmonic_fusion_training.py`, `harmonic_fusion_resume.py`,
`harmonic_fusion_evaluation.py`, `harmonic_fusion_run.py`.
New CLI: `scripts/run_harmonic_fusion_pilot.py`.
New tests (under `tests/experiments/`): `test_harmonic_fusion_training.py`,
`test_harmonic_fusion_resume.py`, `test_harmonic_fusion_evaluation.py`,
`test_harmonic_fusion_run.py`, `test_harmonic_fusion_pilot_config.py`.
New config: `configs/experiments/harmonic_fusion_pilot.json`.
New docs: this result, `02_FUSION_EVALUATOR_RESULT.md`, and
`02_LOGIT_FUSION_GPU_PACKAGE.md` in this workstream directory.
New planned record: `docs/experiments/records/EXP-FUSION-CUDA-SYNTHETIC-001.json`.
Updated inherited docs only: `docs/STATUS.md`, `docs/DECISIONS.md`,
`docs/HARMONIC_LOGIT_FUSION_ABLATION.md`, generated `docs/EXPERIMENT_LEDGER.md`.

## Boundaries and next executable stage

The [queue package](02_LOGIT_FUSION_GPU_PACKAGE.md) fixes P=512 per arm, LR
0.0003, warmup64, B2, full identity VALIDATION 0/256/512, fixed stopping and
separate added compute accounting. It binds the existing C2 export candidate
and records the verified provenance JSON without reading checkpoint tensors.
The source filename suggests historical best, but its selected update is not
independently established; this is an explicit preselected export.

Next: coordinator review/source sealing and one 10-minute exclusive GPU slot
for `EXP-FUSION-CUDA-SYNTHETIC-001` using the exact command in the queue package.
CPU corruption/identity/evaluator tests are prerequisite checks; the CLI
executes the new target-runtime trajectory/export check. Preserve and finalize
any negative result before another substantial run.

Still unresolved: target CUDA execution, real TRAIN historical C2 equivalence,
source-bound worker reservations if optimized preparation is adopted, and
registered corpus dispatch with complete validation/best/final persistence.
The prior 108-update CUDA fixture and C0 speedup do not close these gates.
Cross-version/CUDA resume exactness and pre-initialization CUDA caller-context
preservation are not established by CPU evidence. No full120k, real data
training/evaluation, raw-corpus preprocessing, SSH/GPU, TEST, legacy inspection
or reuse, commit/push/merge, baseline promotion, new decoder or SSL occurred.
