# Logit fusion review remediation — increment 002

Date: 2026-09-08. Worktree:
`/Users/aavalsamakin/.codex/worktrees/a0bd/music-critic-v2`.

This is the narrow second increment requested by the coordinator, following
independent verification of the first 19-file patch and its 100 new tests.
It fixes all four review findings without a new loss, encoder, corpus
dispatcher or production optimization. No GPU/SSH/real-data/TEST access,
checkpoint tensor import, legacy inspection/reuse, commit, push or merge.

## Four review findings addressed

1. **CUDA RNG inventory.** A CUDA payload requires a nonempty ordered state
   list for every visible initialized device. Count and each ordinal's scratch
   generator state are verified before any saved RNG is applied. Missing,
   malformed or unverifiable inventory rejects before a fresh model is even
   constructed. CPU-empty inventory remains valid without CUDA calls. The
   ordered list binds visible ordinals; physical GPU UUID identity and actual
   CUDA trajectory equality are not inferred from mocked CPU tests.

2. **Active learned fusion.** The gate separately requires changed model state,
   changed fusion weights, nonzero optimizer step counters for those weights,
   and a nonzero output residual against the same base on a raw fixture probe.
   It reports initial/final weight fingerprints, changed scalar count/L2,
   per-head residual and optimizer counters, plus descriptive validation
   support/CE/accuracy/F1. No per-head loss or accuracy improvement is required.
   Replacing `HarmonicLogitFusion.forward` with identity now fails explicitly;
   matching full-state resume and export alone can no longer accept bypass.

3. **Required validation lifecycle.** Configuration defaults to
   `validation_mode="required"`; the pilot JSON states that mode explicitly.
   The exact expected history prefix includes update zero, each interval and
   the final update even when the interval does not divide the budget. Only
   the current boundary can remain pending. A failed callback leaves that
   boundary pending and can be checkpointed; resume evaluates the same model
   at that boundary before another update. Missing earlier points fail closed.
   Omitting the callback between calls fails before training, and `best_update`
   is unavailable while validation is pending or its prefix is incomplete.
   The checkpoint's explicit pending marker must agree with history/cursor.
   Callback scores use the same strict finite-float/null contract as resume.
   Synthetic mechanics without evaluation use `synthetic_disabled`, which
   forbids attaching a validation history or later enabling a callback.

4. **Failure accounting and publication.** A durable append-only progress
   journal records started/completed updates and validations, arm/trajectory,
   attempted and applied counts, schedule draw slots and actual reader calls.
   Exception receipts retain earlier completed trajectories and reconcile
   the active session's committed cursor. A seven-attempt/six-applied failure
   remains distinct from a skipped draw or a completed gate. Success report
   publication is the final atomic/exclusive commit after preparation of the
   artifact manifest. Manifest or report-fsync failure cannot leave a success
   report. A receipt-storage failure preserves the primary exception and any
   already durable journal; filesystem failure or hard process termination
   cannot guarantee a complete terminal receipt. Uncompleted in-flight work
   is not upgraded into a confirmed successful update.

The isolated training/full checkpoint schemas advance to **1.1.0** and strict
loading rejects earlier full checkpoints rather than silently supplying new
required state. The model-only fusion format and old corrected APIs remain
unchanged. Earlier 1.0.0 files are retained as historical artifacts.

## Exact CPU acceptance

- Final combined gate: **401 passed**, **0 failed/skipped/deselected**,
  **2 existing Torch JIT deprecation warnings**, **67.96 seconds**.
- This adds **44 cases** to the first stage: 19 CUDA-inventory CPU/fake-runtime
  cases, 15 validation lifecycle cases, and 10 additional runner/failure cases.
  The combined selection includes all 144 fusion-workstream cases plus the
  original fusion mechanics/model-only and corrected/POP/full-orbit/learning/
  repository regressions used at stage one. The CUDA-negative cases never
  initialize a real CUDA runtime.
- Focused checks also passed: CUDA RNG + resume **67 passed / 12.79s**;
  lifecycle **15 passed / 8.89s**; runner **13 passed / 23.07s**. These all
  retain only the two existing Torch warnings.
- Runtime is the unchanged local Python 3.12 / Torch2.13.0 / PyG2.8.0.post1
  environment, native/Torch threads one, with the existing temporary Hydra
  overlay. No dependencies were added in this increment.
- Exact full command/log:
  `outputs/fusion/review002-acceptance/command.txt` and `pytest-001.log`.
  Compileall, diff-check, 47-record registry validation and ledger --check pass.

## Retained current and negative artifacts

| Fixture acceptance artifact | Outcome and actual counts |
| --- | --- |
| `outputs/fusion/cpu-smoke-003/` | Passed; 24 attempted/applied updates, 48 attempted/applied/read draws, 12 fixture validation reads; interrupted/uninterrupted states equal in both arms; 4.598448708999058s. |
| `outputs/fusion/review002-bypass-001/` | Expected rejection `missing_fusion_learning_signal`; 18 attempted/applied updates, 36 draws. Completed control trajectories remain recorded; fusion weight change and residual are zero. No success report. |
| `outputs/fusion/review002-prefix-failure-001/` | Expected injected update failure; 7 attempted, 6 applied updates; 14 attempted schedule slots, 12 applied/read draws. The completed six-update control trajectory remains intact. No success report. |

The successful smoke observed **216,886 changed fusion scalars**, L2 delta
**0.4389684949348881**, **12 nonzero-output harmonic heads** and six optimizer
steps for each fusion matrix. These are fixture-learning mechanics, not
musical improvement or a CUDA measurement. Each root contains SHA/size
manifests; failed roots retain `failure-receipt.json` and `progress.jsonl`.
The tests also cover failure during validation, restore, manifest publication,
report fsync and receipt storage. Ordinary synthetic acceptance checks do not
constitute new substantial scientific experiments.

Prior `outputs/fusion/handoff/`, `cpu-smoke-001/`, `cpu-smoke-002/`, first-stage
logs and handoff documents were not replaced. Stage-one source bytes were
verified against all 19 old manifest hashes before editing and copied into
`outputs/fusion/handoff-002/before/` for an exact incremental patch.

## Review package and next action

New package: `outputs/fusion/handoff-002/`. It contains an incremental patch
relative to the verified stage-one bytes, its before/after manifest, a
cumulative task-owned patch relative to the installed snapshot, a source
candidate manifest and artifact/integrity verification. The inherited source
snapshot remains bound by
`9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf`.
Only the task-owned fusion files/tests/config, current docs, the unlaunched
planned record and generated ledger changed. The first handoff/result docs
and all earlier terminal evidence remain unchanged.

The existing **planned**, **not launched** `EXP-FUSION-CUDA-SYNTHETIC-001`
record now separates `cpu_prerequisite_checks` from `cuda_trajectory_checks`.
The original planned record bytes are preserved in `before/`. No status is
advanced and access observations remain null. The CUDA queue still requests
one GPU / 10 minutes, 24 synthetic updates / 48 draws; the target CLI checks
the new six versus three-plus-three trajectories, full CUDA RNG restoration,
active learned fusion, required fixture validation and export. The POP suite
is a CPU prerequisite, not a duplicated GPU workload.

After coordinator review and source/runtime sealing, the existing command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src python3 scripts/run_harmonic_fusion_pilot.py \
  --smoke --device cuda \
  --experiment-record docs/experiments/records/EXP-FUSION-CUDA-SYNTHETIC-001.json \
  --output-root outputs/fusion/EXP-FUSION-CUDA-SYNTHETIC-001
```

Do not dispatch until the coordinator grants the shared slot. CUDA execution,
real TRAIN C2 equivalence and registered corpus dispatch remain unresolved;
the scientific P512 pair has not run and its baseline has not been promoted.
