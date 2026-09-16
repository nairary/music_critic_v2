# WS1 result — complete raw inputs

Status: **implemented and ready for coordinator review**, not accepted or
integrated. Worktree:
`/Users/aavalsamakin/.codex/worktrees/ef79/music-critic-v2`.
Base and current HEAD: `f0e0886161c9f67786011508006db61c803c1752`.
No commit, push, PR, merge, remote access or other worktree modification.

Review content SHA-256:
`a2bb43ea55053aa2bcd6bb7f3b04372ab381ae398c9183caea319250ee7bcf2c`.
This hashes compact sorted JSON mapping each changed tracked/untracked file
path to SHA-256 of its current bytes, excluding only this report and
`WS_RESULT.md` to avoid a self-reference. It includes the nine files below,
including STATUS and DECISIONS.

## Changes and behavior

| File | Result |
| --- | --- |
| `src/music_critic/experiments/analysisgnn/transposition_inputs.py` | Separate complete directed raw transposition API, registry-bound identity, all eight pitch-derived fields, exact builder-compatible statistics/masks, ownership checks and detached copies. |
| `src/music_critic/adapters/hooktheory.py` | Adapter 2.0.0; annotation-independent duration; clipped/masked outside targets; explicit empty-melody policy; exact mandatory raw bytes/fingerprint version 1.0.0. |
| `src/music_critic/tasks/corpus.py` | Coordinator-approved HookTheory-only adapter-version/raw-key/raw-config binding through existing cache APIs. No generic helpers/schema changes. |
| `tests/experiments/test_complete_transposition_inputs.py` | Reproduces historical stale statistics; complete canonical-rebuild oracle, all shifts/inverses, boundaries/tritone, empty/constant/percussion tracks, source integrity, invalid ownership/target rejection. |
| `tests/adapters/test_hooktheory_raw_timeline.py` | Chord/visibility invariance across exact raw bytes, actual streaming cache and graph; out-of-range reasons/masks; empty/rest cases; historical artifact protection; pitch/timing sensitivity. |
| `tests/adapters/test_hooktheory_adapter.py` | Duration-extension test now exercises raw melody extent; chord annotation cannot extend geometry. |
| `docs/INPUT_CORRECTNESS_CONTRACT.md` | Public APIs, old/new behavior, cache identities, guarantees and limitations. |
| `docs/STATUS.md` | Separate WS1 result. |
| `docs/DECISIONS.md` | Append-only ADR-WS1-20260905. |

`CompleteRawGraphTransposition@1.0.0` fingerprint:
`1b2b65c4348cbc5394bb7ea450722e871f2dad68e6e854d049b79de19a3ccdbf`.
The API is `transpose_complete_raw_graph_view(graph, transform=...)` in the
new module. Future consumers bind its contract fingerprint explicitly;
existing trainer routing does not change.

All complete-view checks use exact full graph fingerprints, including all
features and availability. No floating tolerance is needed: recomputation
uses the builder's fmean/population standard deviation before float32 packing.
The original directed inverse `(6, -6)` is reused; the B5F/B5G tritone history
is not rewritten or claimed as a new discovery.

HookTheory raw config identity contains dataset name, adapter version and raw
projection version. `include_targets` is retained in canonical conversion
provenance. Full source SHA/source manifest fingerprint, canonical SHA and
target values remain separately accessible. Different label/visibility
artifacts share the appropriate raw cache key while retaining distinct
content-addressed canonical paths. Tests reload all earlier artifacts and
verify their bytes/targets after later builds. Historical adapter pieces are
rejected by the new raw projection helper and historical cache bytes remain
readable and unchanged, with historical version identity.

## Exact validation commands

All commands ran from this worktree using the supplied Python runtime without
changing it, with explicit `PYTHONPATH=src` and CPU thread limits.

Primary suite:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q \
tests/experiments/test_complete_transposition_inputs.py \
tests/experiments/test_phase9eb5a_transposition.py \
tests/experiments/test_phase9eb5f_transposition_correctness.py \
tests/experiments/test_phase9eb5g_directed_inverse.py \
tests/experiments/test_phase9eb5h_full_orbit_profile.py \
tests/adapters/test_hooktheory_raw_timeline.py \
tests/adapters/test_hooktheory_adapter.py \
tests/test_repository_contract.py
```

Result: **185 passed, 2 warnings in 3.16s**.

Supplemental regression:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q \
tests/adapters/test_hooktheory_parser.py \
tests/exporters/test_midi_exporter.py \
tests/tasks/test_multisource_dataset.py \
tests/tasks/test_multisource_contract.py \
-k 'not test_dataloader_num_workers_zero_and_two_have_schedule_parity'
```

Result: **124 passed, 1 deselected, 2 warnings in 2.59s**.

The deselected test was separately run by the HookTheory subagent with
authorized sandbox escalation:

```bash
env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q \
tests/tasks/test_multisource_dataset.py::test_dataloader_num_workers_zero_and_two_have_schedule_parity
```

Result: **1 passed, 2 warnings in 12.03s**. An earlier sandboxed attempt failed
because `torch_shm_manager` was denied shared-memory operations
(`Operation not permitted`); the isolated rerun passed. There is no outstanding
test failure. Warnings in all runs are upstream Torch JIT deprecations.
There are **310 distinct passed tests**, zero dependency skips. The full suite
and opt-in real-data integration tests were not run.

Additional development gates: original stale-statistics reproduction
**1 passed**; new transposition suite **37 passed**; independent frozen
transposition/repository suite **51 passed**. These overlap the primary suite
and are not added to its total. A separate read-only subagent review found no
correctness blocker in either implementation. The initial stream-cache
regression exposed eight annotation-sensitive keys; the final test verifies
one raw key across those eight source/visibility cases.

```bash
git diff --check
git diff --quiet -- src/music_critic/experiments/analysisgnn/transposition.py \
  configs tests/fixtures docs/experiments docs/EXPERIMENT_LEDGER.md
```

Both checks passed. Frozen transposition code, profile configs, fixture
fingerprints, experiment records and ledger remain unchanged.

## Boundaries, issues and next phase

Legacy scope: no files inspected, copied, adapted, imported or modified.
All reused behavior is V2 builder/registry/directed-token logic. Missing
labels remain masked; percussion, exact timing, source provenance and raw
inference constraints are retained. No dependency added or changed.

No real corpus, TEST input/target/loader, trained checkpoint, training run,
evaluation run, SSH session or expensive experiment was used. Temporary source
and cache artifacts were synthetic. These tests need no standalone scientific
experiment record, and none was created.

Unresolved: impact on historical C2 is unknown; no augmentation/quality or
corpus-readiness claim follows. The complete graph API supports one validated
default-registry raw graph, not a batch/custom registry. No throughput claim
is made. Relative HookTheory pitch reconstruction still uses accepted key/scale
interpretation and is outside this correction.

Next phase: coordinator actual-diff review and isolated joint integration.
Any training adoption or rerun requires a separately registered experiment
under the new contract; no automatic rerun is part of WS1.
