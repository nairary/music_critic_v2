# Task 03 — review-002 corrections — 2026-09-08

Status: four coordinator-requested correctness fixes completed and CPU verified.
CUDA and scientific experiments remain planned. This result supersedes only the
runner correctness claims of the [first-stage result](TASK03_RESULT.md); its
retained `outputs/ssl_group/review/` and earlier fixture artifacts are unchanged.

## Corrected behavior

1. Finite FP32 gradients can overflow Adam squared moments. The runner now
   validates all model tensors and optimizer state after a normally returned
   step, before confirming a row. Overflow/nonfinite state poisons the instance
   and prevents checkpoint publication. The regression injects a finite 1e30
   gradient. No clipping, loss, optimizer or model change was introduced.
2. Actual runtime is checked before update, before/after Adam, before checkpoint
   creation and before resume read/application. Bindings include thread count,
   deterministic/CUDNN/TF32 flags, explicit float32 matmul precision, autocast
   enabled/dtype state, Python/Torch/CUDA versions, and CUDA-specific
   CUBLAS_WORKSPACE_CONFIG. CPU tests inspect CUDA settings without initializing
   CUDA. Schema 1.1 leaves the original 1.0 dropout seed domain pinned.
3. Successful rows have a strict schema and exact active/inactive paths, finite
   losses and aggregation, draw/forward/mask/density/unit/head counters and
   diagnostics. The schedule binds compact input identities and sizes plus each
   head's actual source sidecar population, alongside the digest of full raw
   graphs/sidecars/weight tensors. `valid + masked + unsupported` must match that
   population. Rehashed negative counters, empty paired masks and coherent
   billion-row head/aggregate inflation reject before resume mutates state.
   Repeated native targets mapping to one raw candidate remain valid.
4. The CLI durably appends started/progress/failure/completion receipts to
   `attempts.jsonl`, with completed encoder forwards and stage. The optimizer
   outcome is not_applied until entered, unknown on an entered call that raises,
   and applied only on normal return. Fully confirmed updates and successfully
   written rows are separate. A wrapper that executes real Adam then raises
   KeyboardInterrupt reports zero confirmed updates, three observed forwards
   and one unknown outcome. A returned but invalid Adam is applied/unconfirmed;
   receipt or successful-row write failures retain already confirmed work.

The 600-second limit is a pre-step elapsed-time check, not a hard timeout.
An external coordinator timeout/resource wrapper must handle an in-flight stall
or process kill. Durable receipts cannot make host/device work atomic with
process termination; a last unknown outcome must remain unknown.

## Files and verification

Changed for this remediation: `group_masking_pilot.py`, `group_masking_cli.py`,
their tests, new `group_masking_accounting.py` and its tests; protocol, STATUS,
DECISIONS, the two still-planned records, generated ledger and this result.
The cumulative handoff includes the unchanged voice and bridge work from stage1.
No legacy logic was inspected, adapted or changed; no dependency was added.

Environment: existing Python 3.12.14, Torch 2.13.0 and pytest 9.1.1, with:

```text
PYTHONPATH=src:/private/tmp/music-critic-fusion-20260908-deps
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/private/tmp/music-critic-pop-eval-20260908/bin/python
```

| Arguments after Python | Result | Saved log |
| --- | --- | --- |
| `-m pytest tests/ssl/test_group_masking_pilot.py tests/ssl/test_group_masking_accounting.py tests/ssl/test_group_masking_cli.py -q` | 128 passed, 2 warnings in 31.98s | `outputs/ssl_group/verification-review-002/focused.log` |
| `-m pytest tests/ssl/test_theory_warmstart.py tests/ssl/test_theory_retention_harness.py tests/ssl/test_masking.py tests/ssl/test_model.py tests/ssl/test_hierarchical_masking.py tests/ssl/test_hierarchical_prepared_binding.py -q` | 225 passed, 5 skipped, 2 warnings in 28.40s | `outputs/ssl_group/verification-review-002/existing-mechanisms.log` |

**353 passed, 5 skipped** across the two final disjoint invocations. Skips need
CUDA; warnings are existing Torch JIT deprecation. Independent read-only review
found and reproduced the coherent head-population inflation and high/medium
runtime alias; both were resolved before these final checks. Tiny synthetic
regressions are engineering checks, not scientific experiment evidence.

Standalone command:

```text
python scripts/run_group_masking_pilot.py --config configs/ssl_group/random_vs_beat.json \
  --mode cpu-fixture --output outputs/ssl_group/cpu-fixture-review-002
```

Both policies resume exactly. Actual totals are 8 confirmed updates, 24 observed
encoder forwards, 16 SSL and 16 theory draws including replay. All successful
row diagnostics, losses and masks equal `cpu-fixture-002` after excluding the
changed batch binding/evidence fields. The new result is 87,330 bytes, SHA-256
`ae7672be128b3fe30a02b4975be94aeab360560effce016c2fe440b782f6fb23`.
Code fingerprint:
`259cf496e99a4d355bb051d68192107fd86e2b7f357b0d11609ebd6d81d7a7bd`.

`EXP-SSL-GROUP-CUDA-001` and `EXP-SSL-GROUP-PILOT-001` remain planned and were
resealed after code changes. Registry check and deterministic render/check:
48 valid records, registry fingerprint
`756e791fe33c1988b8f648ed07624db9c985ef5e9384f820b25dabdcbc79e8be`.
Existing terminal records remain byte-identical to the bootstrap snapshot.

## Handoff and remaining boundary

`outputs/ssl_group/review-002/own-changes.json` describes cumulative task files
against the verified bootstrap manifest; `delta-from-review001.json` identifies
only this remediation. Both patches and source copies support independent
review; `artifacts.json` binds all retained new evidence. Previous review
artifacts are separately verified unchanged. These ignored local artifacts do
not imply durable cross-host backup or integration into the main checkout.

Compact validation establishes schema, arithmetic and population consistency;
it cannot authenticate exact mask selections, measured losses or reclassification
within a fixed target population without original evidence/replay. Runtime
checks at boundaries cannot detect a flag toggled and restored entirely between
them. No cross-host bitwise resume guarantee is made.

Next phase remains coordinator-owned CUDA fixture acceptance under the exact
resealed code/config/C2 identities and external resource supervision, followed
by resolving the real TRAIN/VALIDATION/schedule/replay/evaluator bindings before
a scientific pilot. No CUDA, SSH, real checkpoint load, real-data dispatcher,
PDMX, TEST, baseline promotion, commit, push or merge occurred in this remediation.
