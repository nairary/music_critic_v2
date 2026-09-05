# WS4 result — frozen POP909-CL evaluator

Status: INT-R1-001 remediation ready for coordinator re-review. R2 had accepted
the prior revision for isolated integration; the later joint-gate finding
returned status to `changes_requested`. Not integrated.

## Revision and ownership

- Worktree: `/Users/aavalsamakin/.codex/worktrees/46dc/music-critic-v2`.
- Base and current HEAD: `f0e0886161c9f67786011508006db61c803c1752`.
- Detached HEAD; no commits, staging, push, PR, merge or foreign-worktree edits.
- Review content fingerprint:
  `ee6f39a55184542c9031ecd6adfdac1566cad2df00ab0afb006c3cd41b3f9a5b`.
  It is SHA-256 over each sorted repo-relative path, NUL, complete file bytes,
  NUL, for the 15 code/test/contract/STATUS/DECISIONS files below, including the
  newly authorized isolated adapter helper. This report
  and the root `WS_RESULT.md` pointer are excluded to avoid self-reference.
  New files are untracked and must be inspected in addition to `git diff`.

Files added:

- `src/music_critic/adapters/pop909_cl_synthetic.py` (INT-R1-001 authorized extension)
- `src/music_critic/experiments/pop909_cl_frozen/__init__.py`
- `src/music_critic/experiments/pop909_cl_frozen/contracts.py`
- `src/music_critic/experiments/pop909_cl_frozen/checkpoint.py`
- `src/music_critic/experiments/pop909_cl_frozen/projection.py`
- `src/music_critic/experiments/pop909_cl_frozen/views.py`
- `src/music_critic/experiments/pop909_cl_frozen/evaluation.py`
- `src/music_critic/experiments/pop909_cl_frozen/run.py`
- `tests/experiments/test_pop909_cl_frozen_checkpoint.py`
- `tests/experiments/test_pop909_cl_frozen_projection.py`
- `tests/experiments/test_pop909_cl_frozen_views.py`
- `tests/experiments/test_pop909_cl_frozen_evaluation.py`
- `docs/POP909_CL_FROZEN_EVALUATION.md`
- This report and `WS_RESULT.md`.

Files updated: append-only WS4 entries in `docs/STATUS.md` and
`docs/DECISIONS.md` (ADR-WS4). No shared API or dependency change was necessary.

## Implemented behavior

- Restricted CPU model-only loader: explicit new schema and limited historical
  envelope compatibility, expected SHA/size checks on the same bytes loaded,
  exact canonical model-contract validation, complete state key/shape/dtype/
  finiteness checks, state fingerprint verification and strict frozen eval
  loading. Caller RNG remains stable on success and failure. Saved optimizer,
  scheduler/scaler/sampler/RNG is never restored; unsafe loading has no fallback.
- Root37 spelled softmax is marginalized to PC12 before prediction. Quality17
  chooses its winner before subset9 gold comparison. Explicit interval
  templates, ordered registry lock, deterministic ties and unsupported gold
  counts are versioned. Outside-subset quality winners remain full confusion
  errors and are never dropped or renormalized away.
- Native primary and metadata_removed secondary views remove exactly the five
  declared canonical fields, then rebuild via the public graph API. Derived
  availability and track.mean_velocity close correctly; raw musical events,
  timing, ownership and other metadata remain unchanged.
- Source block onset/PPQN maps by exact RationalTime to one half-open raw beat.
  Fractional starts, end-exclusive boundaries, unavailable/ambiguous beats,
  multiple same-beat events and conflicting source labels are explicit. Each
  event remains one observation. Root and quality masks stay independent;
  joint accuracy alone uses their intersection.
- Deterministic report includes observed-true-class macro-F1/accuracy, per-class
  support and full confusion, independent event/piece/group coverage axes,
  source-event/input/model/registry/protocol bindings and access observations.
  Null denominators retain reasons. No two-task aggregate is named or used as
  corrected-primary score or music-quality score.
- CLI is help-only unless `--smoke` is explicit. Its generated channel-0
  polyphonic score excludes channel 1. Chord normalization/target extraction
  is deferred until both raw view predictions. TEST is rejected before source,
  cache or checkpoint callbacks; real TRAIN/VALIDATION execution is disabled.

## Verification

All commands ran from this worktree using the shared environment only as a
runtime, with local `PYTHONPATH=src` and OMP/MKL threads limited to one.

Final focused CPU acceptance command (repository gate added for INT-R1-001):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest \
  tests/experiments/test_pop909_cl_frozen_checkpoint.py \
  tests/experiments/test_pop909_cl_frozen_projection.py \
  tests/experiments/test_pop909_cl_frozen_views.py \
  tests/experiments/test_pop909_cl_frozen_evaluation.py \
  tests/experiments/test_phase9eb5c_corrected_model.py \
  tests/evaluation/test_checkpoint_evaluation.py \
  tests/test_repository_contract.py \
  -q -k 'not bounded_evaluation_uses_concrete_cuda_runtime'
```

Current INT-R1-001 result: **124 passed, 1 deselected, 2 warnings in 6.05s**.
R1 result (before adding the repository gate): **117 passed, 1 deselected,
2 warnings in 5.64s**.
Before R1: **115 passed, 1 deselected, 2 warnings in 5.32s**. The one deselected
existing test explicitly requires a real CUDA runtime, outside the CPU-only
scope. The initial six-module command without `-k` yielded **115 passed, 1 skipped,
2 warnings in 5.66s**; no missing dependency caused a required test to skip.
Warnings are existing upstream `torch.jit.script` deprecations.

Current new tests: **101 passed** (checkpoint 34, projection 35, views 4,
evaluation 28). R1 had 100 passed, including 27 evaluation tests; before R1:
98 passed, including 25 evaluation tests.
They cover enharmonic-sum counterexample/conservation/order/drift/ties;
full17 outside-subset FP/FN; independent masks/null counts; rational alignment;
target/channel1 changes with identical raw fingerprints/candidates/all18 eval
logits; metadata closure/nonmutation; strict synthetic checkpoint rejection;
forbidden resume/optimizer/saved-RNG calls; RNG success/failure stability;
report order invariance; reader ordering; TEST and default-CLI guards.

Executable CLI check:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/music-critic-v2-integration-venv/bin/python \
  -m music_critic.experiments.pop909_cl_frozen.run \
  --smoke --output /private/tmp/ws4-pop909-frozen-smoke-20260905.json
```

Exit 0; generated report is 46,339 bytes and remains outside Git.
Its protocol fingerprint is
`fa111f1dda7d2f3f930daf8cc7887d461a800b37f1a9a5506927b6816e1e1ff0`;
report fingerprint is
`f63ffff4bc1b5dbb0d398d5bd436c34451969225f647443e057c27e50a848b69`.
It contains one synthetic piece/event, both fixed views, and false TEST
input/target/metric access. Repeated smoke reports are byte-identical in the
test runtime. Random-model scores are plumbing evidence only.

`git diff --check` passes. Only the listed ownership paths changed. No full
suite or opt-in real-data integration was run. Ordinary synthetic contract
tests did not require standalone experiment records or ledger regeneration.

Internal review corrected an omitted raw-batch validation argument, exact
model-contract scalar-type validation, evaluator checkpoint-evidence bindings,
independent unaligned coverage, and delayed smoke chord extraction. Initial
integration tests failed on the missing validation argument; the final CPU
gate above includes regression coverage for the corrected paths.

## Coordinator review R1 history — 2026-09-05

- Reviewed content fingerprint:
  `320dd4fa82b6865c5238ce43ebdaf70c7c59748afbe60a8657d6aecb06c7a674`.
  Coordinator outcome: `changes_requested`, one P2 coverage correction.
- Finding: `targets_from_chord_evidence` counted `len(evidence.trailing_spans)`.
  The adapter's normal `expected_missing` representation stores a full-piece
  `missing_chord_targets` marker in that collection. The evaluator therefore
  misreported one trailing unannotated span for absent chord supervision.
- Fix: count only spans whose `kind` is `trailing_unannotated`. No schema,
  projection, checkpoint, raw view, alignment or shared API change.
- Regression: real `Pop909ClChordEvidence`/`Pop909ClCoverageSpan` dataclasses
  reproduce the adapter marker shape. Both report views retain
  `missing_target_pieces=1`, `missing_target_groups=1`,
  `trailing_unannotated_spans=0`; the genuine trailing-kind case remains 1.
  The tests also check the piece-level count, zero chord/no-chord events and
  null metric denominators.
- Affected-module command:

  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest \
    tests/experiments/test_pop909_cl_frozen_evaluation.py -q
  ```

  Result: **27 passed, 2 warnings in 2.03s**. The then-current six-module
  focused command returned **117 passed, 1 deselected, 2 warnings in 5.64s**.
- Independent synthetic R1 review: **2 passed, 25 deselected, 2 warnings in
  0.94s** using the same evaluation module with `-k 'missing_chord or trailing'`;
  no remaining R1 code/test findings. `git diff --check` passes.
- R1 edits are limited to `evaluation.py`, its test module, `docs/STATUS.md`
  and the two result documents. The initial implementation record and prior
  review identity are retained here. No real data/checkpoint/TEST access,
  legacy use, experiment records, commits or external-worktree edits.

## Coordinator review R2 history — 2026-09-05

- Accepted for isolated integration at the independently reproduced 14-file
  content fingerprint
  `d89fa514991a436958fd24a1fc3aea7728a3eb9b372d5a82721f0006041666b9`.
- Coordinator independently ran **117 passed, 1 CUDA case deselected,
  2 warnings in 5.16s**, closed `WS4-R1-001`, and retained R1 history.
- Coordinator review artifact is `docs/workstreams/2026-09-05/WS4_REVIEW_R2.md`
  in the main working copy; it was not modified here. Accepted did not mean
  integrated. The following joint-gate finding reopens only its named scope.

## Integration finding INT-R1-001 — 2026-09-05

- Coordinator reported a joint integration gate result of **1 failed,
  812 passed, 6 CUDA skips, 2 deselected, 2 upstream warnings in 53.72s**.
  Both new cross-workstream tests passed. The failing existing test was
  `tests/test_repository_contract.py::test_package_has_no_legacy_or_heavy_imports`:
  direct `import mido` is not permitted under `experiments`.
- Authorization explicitly extended WS4 ownership to new isolated
  `src/music_critic/adapters/pop909_cl_synthetic.py`. Only fixed tiny
  `mido.MidiFile` construction moved there; the helper has no file access.
  `run.synthetic_input(directory)` retains its API, serialization, score
  projection, provenance and deferred target callback. The import is a normal
  static adapter-helper import, not a dynamic bypass.
- Existing adapter files/exports/shared APIs/registries, repository-contract
  tests, dependencies and evaluator protocols are unchanged. This correction
  edits the new helper, `run.py`, its existing evaluation test module, STATUS
  and the two result documents. No ADR change was required.
- Before moving code, captured generated-file SHA/size, source group and smoke
  report identity; the same checks after the move are equal. New regression
  `test_synthetic_input_preserves_pre_helper_midi_bytes` locks:

  | Synthetic artifact | Bytes | SHA-256 |
  | --- | --- | --- |
  | 001.mid | 127 | `678226cc01c74815b35788997881f3a943658552b578dbde79eac86b3ad447e6` |
  | score-only.mid | 90 | `21d31a9fc9823b6f83c821b26c5d032fa55e80b2f60ab4c0c883103550dda0b2` |

  Source group remains `pop909-cl-score:21d31a9fc9823b6f83c821b26c5d032fa55e80b2f60ab4c0c883103550dda0b2`;
  smoke report fingerprint remains
  `f63ffff4bc1b5dbb0d398d5bd436c34451969225f647443e057c27e50a848b69`.
  Only compact hashes/sizes are tracked; generated MIDI remains temporary.
- The complete command in Verification, now including the unmodified
  repository-contract module, returns **124 passed, 1 deselected, 2 warnings
  in 6.05s**. All 101 WS4 tests pass. The single deselection remains the
  existing CUDA-only runtime case. `git diff --check` passes.
- Independent read-only review found no remaining issue. It reproduced both
  MIDI hashes, native/metadata_removed graph/model-input/view fingerprints and
  the original source-event values before/after the move. Its focused
  repository/evaluation selection (`-k 'package_has_no_legacy_or_heavy_imports
  or preserves_pre_helper_midi_bytes or source_extraction_is_deferred or smoke'`)
  returned **5 passed, 29 deselected, 2 warnings in 1.74s**.
- This worktree did not rerun the cross-workstream joint gate. INT-R1-001
  remediation awaits coordinator re-review and joint integration validation;
  no integration completion is claimed. No real-data/checkpoint/TEST access,
  legacy use, new dependencies, commits, push, merge or cleanup occurred.

## Limits and next phase

- Actual C0/C2 checkpoints and corpora were not opened. Historical training
  payloads can contain NumPy/environment objects rejected by restricted load;
  real compatibility is unknown. Synthetic historical envelopes are the only
  historical-schema fixtures tested. No unsafe workaround is implemented.
- Runtime evaluation is intentionally synthetic-only and processes one piece
  at a time. Full-corpus throughput and zero-shot transfer quality remain
  unmeasured. Batch-order acceptance concerns deterministic item aggregation;
  a new multi-piece inference batch implementation would need its own checks.
- Source provenance is explicitly caller-declared synthetic; generic provider
  objects do not attest real-data identity or authorize a run.
- No TEST, SSH, training, real-data evaluation, current C0 process inspection,
  frozen fingerprints, terminal records or dataset/cache changes occurred.
- Legacy scope: none inspected, imported, reused or adapted. Existing V2
  public APIs were reused; gold accompaniment reconstruction, argmax-first
  root projection, quality9 renormalization and training resume were rejected.
- Next: coordinator actual-diff review and isolated joint integration. The
  next scientific phase is separately registered TRAIN/VALIDATION evaluation
  after obtaining and verifying authorized checkpoint artifacts. It is not
  authorized by this fixture-only completion; TEST remains closed.
