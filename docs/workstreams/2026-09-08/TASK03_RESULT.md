# Task 03 first-stage result — 2026-09-08

Status: engineering implementation and CPU verification completed; CUDA and
real-data scientific runs not started. Worktree:
`/Users/aavalsamakin/.codex/worktrees/582d/music-critic-v2`.
Coordinator task: `01a06e25-7dba-7430-86da-202cacede488`.

## Delivered behavior

- Existing independent-note versus raw-beat pitch masking over shared WS2,
  with identical full initial weights, common policy eligibility and batch
  schedule, explicit density/forward/replay accounting and strict full resume.
- A separate model-only artifact entrypoint binding the actual restricted
  loader evidence to complete corrected C2 state; old synthetic validation and
  original model/ontology/losses stay intact.
- Separate source-scoped voice continuation candidate/target mapping with
  polyphonic crossing and multivoice-per-track fixtures, masks and confidence.
- Concrete executable CUDA fixture gate plus a scientific protocol/config whose
  missing real-data bindings are explicit.

No duplication of the existing full-view stop-gradient mechanism or masking
planners, no EMA, voice encoder edge, new latent loss, harmonic fusion, PLL,
preference/degradation training, corpus preprocessing, TEST or legacy reuse.

## Own source changes relative to the verified bootstrap manifest

Modified: `src/music_critic/ssl/theory_warmstart.py`, `docs/STATUS.md`,
`docs/DECISIONS.md`, `docs/EXPERIMENT_LEDGER.md`, `docs/WS2_THEORY_SSL_HARNESS.md`.

New implementation/config/CLI:

- `src/music_critic/ssl/group_masking_pilot.py`
- `src/music_critic/ssl/group_masking_fixture.py`
- `src/music_critic/ssl/group_masking_cli.py`
- `src/music_critic/data/voice_continuation.py`
- `scripts/run_group_masking_pilot.py`
- `configs/ssl_group/random_vs_beat.json`

New tests: `tests/ssl/test_group_masking_pilot.py`,
`tests/ssl/test_group_masking_cli.py`, `tests/data/test_voice_continuation.py`.

New documentation: this result, `SSL_GROUP_PILOT_PROTOCOL.md` and
`VOICE_CONTINUATION_CONTRACT.md` in this directory; new planned records
`EXP-SSL-GROUP-CUDA-001.json` and `EXP-SSL-GROUP-PILOT-001.json` under
`docs/experiments/records/`. No pre-existing terminal record was modified
relative to the coordinator snapshot. No commits were made.

## Exact verification

All commands ran with:

```text
PYTHONPATH=src:/private/tmp/music-critic-fusion-20260908-deps
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/private/tmp/music-critic-pop-eval-20260908/bin/python
```

This is Python 3.12.14, Torch 2.13.0, PyG 2.8.0.post1 and pytest 9.1.1. The
read-only overlay supplies already-declared optional training dependencies.
No packages or project dependency declarations were changed.

| Arguments after Python | Result | Saved full log |
| --- | --- | --- |
| `-m pytest tests/ssl/test_group_masking_pilot.py tests/ssl/test_group_masking_cli.py -q` | 25 passed, 2 warnings in 18.77s | `outputs/ssl_group/verification/pilot-and-cli.log` |
| `-m pytest tests/ssl/test_theory_warmstart.py tests/ssl/test_theory_retention_harness.py tests/ssl/test_masking.py tests/ssl/test_model.py tests/ssl/test_hierarchical_masking.py tests/ssl/test_hierarchical_prepared_binding.py -q` | 225 passed, 5 skipped, 2 warnings in 28.07s | `outputs/ssl_group/verification/existing-mechanisms.log` |
| `-m pytest tests/data -q` | 292 passed, 2 warnings in 1.17s | `outputs/ssl_group/verification/data.log` |

Total across disjoint final invocations: **542 passed, 5 skipped**, no failures.
Five CUDA parity cases require actual CUDA. The six warnings across processes
are the existing Torch JIT deprecation. Ordinary earlier test failures identified
PyG metadata handling and resume-validation gaps; regressions now cover their
fixes. The original no-overlay test collection failed for missing Hydra;
re-running with the existing dependency overlay resolved it. These development
checks are not scientific experiment evidence.

Standalone final-code command:

```text
python scripts/run_group_masking_pilot.py \
  --config configs/ssl_group/random_vs_beat.json \
  --mode cpu-fixture --output outputs/ssl_group/cpu-fixture-002
```

Result: success, synthetic source/batches, exact interrupted resume for both
policies. Actual instrumented totals: eight updates, 24 encoder forwards,
16 SSL draws and 16 theory draws including replays. The final code fingerprint is
`efb7a65c6de8459d3c3003bd19c8fb5fb012929464bd282c532d73549629d5f9`.
This is not a hardware benchmark, trained-C2 check or representation-quality
result. Earlier `cpu-fixture-001` remains a separate development artifact and
is not presented as final-code evidence.

Final report `outputs/ssl_group/cpu-fixture-002/result.json`: 79,706 bytes,
SHA-256 `d4d870828a5afbcd3d60b03185655ffe3516ba85a4ce49d9e5901846ed09ef2f`.
It links immutable per-arm initial/ownership bindings and checkpoints; the full
`execution.jsonl` separates uninterrupted updates from resume replays.

Registry validation and deterministic ledger render/check pass: **48 records**,
registry fingerprint `e564d689d08ad4bd0b3287db57ea6f240b85fa9ede87b793b8e53b0db619cc90`.
Both new experiment records remain `planned`; no experiment result was imported
or launched in this task.

## Review package and next gated action

`outputs/ssl_group/review/own-changes.patch` and `own-changes.json` distinguish
this task's 19 files from the 752-file inherited snapshot. A file copy under
`outputs/ssl_group/review/source/` supports independent inspection. Patch hashes
and evidence hashes are recorded in `outputs/ssl_group/review/artifacts.json`.
These ignored artifacts are local evidence, not durable cross-machine backup.

The next runnable coordinator-owned step is `EXP-SSL-GROUP-CUDA-001`, with the
exact command and resource accounting in [the protocol](SSL_GROUP_PILOT_PROTOCOL.md).
The checkpoint path is an explicit host-root substitution under a fixed SHA.
Code/config/source/record bindings must pass before loading it or initializing
CUDA. Retain failures and finalize each launched attempt before any retry.

The first scientific run is `EXP-SSL-GROUP-PILOT-001`: 200 updates per arm,
fixed Dilemmadata TRAIN/VALIDATION bindings, same replay, all 18 tasks, original
C2 reference and random-mask control. Current blockers are the exact real-data
factory/index/split/paired-eligibility/schedule/sidecar/weight bindings and a
fixed VALIDATION/recovery-stage evaluator. The delivered fixture CLI does not
pretend to supply those. PDMX release/license/lineage readiness and a real voice
annotation importer are separate unresolved work, described in the protocols.
