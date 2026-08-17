# Phase 6D-A supervised evaluation

## Scope

Phase 6D-A evaluates the existing feature-only, local-GNN, and hierarchical
supervised auxiliary heads. It is not a critic score, calibration result, SSL
objective, or new checkpoint-selection policy. It changes no model, graph,
adapter, target, corpus, cache, ontology, or encoding semantics.

Evaluation and evaluation-artifact contracts are version `1.3.0`. The
profiler contract is `1.1.0`, the macro-summary sub-contract is `1.0.0`, and
the unchanged train-prior contract remains `1.0.0`.

## Candidate-first and split contract

The evaluator reconstructs a fresh model from checkpoint metadata, validates
that metadata against current contracts, and loads only `model_state`.
Optimizer/scheduler/scaler state and checkpoint RNG are never applied. Caller
Python, CPU torch, and CUDA torch RNG states are restored after loading.

The ordering is mandatory:

1. load one raw `MultiSourceBatch`;
2. call `model.predict(batch.raw_graph_batch)`;
3. only then join the target sidecars;
4. stream eligible rows into metrics and fixed train-derived baselines.

Eligibility requires a fully supervised active task, `availability_mask=true`,
and `entity_index_mask=true`. Alignment conflicts are emitted as unavailable
rows and counted separately. Masked, unavailable, unaligned, conflict, and
open-vocabulary/deferred rows never enter a metric denominator.

Validation is the default. Training and evaluation import one neutral
`fixed_validation_membership_v1` contract. Its ranking and membership payloads
use the exact compact UTF-8 JSON bytes, without a terminal newline, used by
Phase 6C checkpoints. Hash-selected membership is without replacement and is
emitted in canonical view order; `limit=0` means the complete view. This byte
rule is intentionally separate from the newline-bearing global evaluation
`canonical_fingerprint`, whose semantics did not change.

Evaluation contract `1.1.0` documentation incorrectly claimed exact Phase 6C
membership parity while evaluation used the global newline-bearing
fingerprint. Contract `1.1.1` restores actual backward compatibility without
changing or deprecating existing checkpoints and without weakening any index,
split, composition, or membership check.

Test evaluation fails before checkpoint access unless
`acknowledge_test_evaluation=true`; every test artifact states that test was
not used for checkpoint selection.

## Fingerprint evidence

`checkpoint_evidence.json` contains checkpoint SHA-256 and size, exact model
contract and fingerprint, checkpoint kind, historical Phase 6C data binding
when present, and the comparison result. The model contract binds canonical
and graph schema/builder, raw feature registry, target ontology and encoding,
and active source-native heads.

The evaluation data binding records:

- corpus index fingerprints;
- a cache fingerprint derived from index-bound cache version, cache keys, and
  canonical artifact SHA-256 values;
- split-manifest fingerprint;
- effective dataset-subset split-view fingerprint when a single-corpus config
  is derived from a globally bound multi-corpus manifest;
- train and selected-split composition fingerprints;
- fixed train and selected-split membership fingerprints;
- ontology and encoding versions/fingerprints;
- cache validation policy.

Every selected production cache artifact is read-only and SHA-validated by the
existing indexed dataset. A Phase 6C validation checkpoint must match its
historical index/split/train+validation composition and validation membership.
Phase 6A/6B model-only checkpoints predate those fields, so the evidence says
historical data binding is unavailable instead of claiming a false match.

## Metrics and isolation

All accumulators are keyed by `(dataset_id, task_id)`. The task's ontology
source adapter must match the dataset. No HookTheory metric is combined with a
POP909-CL metric, even where the concepts have a documented crosswalk.

Closed categorical heads report NLL, top-1, top-3 when at least three classes
exist, balanced accuracy, macro/micro F1, per-class precision/recall/F1/
support/predicted count, and the complete confusion matrix.

Closed multilabel heads use fixed `sigmoid(logit) >= 0.5` and report BCE/NLL,
micro and macro precision/recall/F1, per-class TP/FP/FN/TN/support, and
exact-match accuracy.

Precision and recall retain their own denominator rules. Per-class F1 does
not depend on whether those two separately reported values are defined:

```text
F1 = 2 TP / (2 TP + FP + FN)
```

A supported class with no predicted positive and an unsupported class with
false-positive predictions therefore both have defined F1 `0`. Per-class F1
is undefined only when `2 TP + FP + FN == 0`, meaning that the class occurs
neither in eligible truth nor in predictions. Macro-F1 is the unweighted mean
of every defined per-class F1, including defined zeros.

An undefined denominator produces:

```json
{
  "value": null,
  "undefined": {
    "category": "zero_f1_denominator",
    "reason": "the class is absent from both eligible truth and thresholded predictions"
  }
}
```

Absence of positive labels is therefore not reported as zero quality.
Likelihood sums use exact accumulation of the observed binary64 values, and
confusion/count state is fixed by the ontology size. Multilabel AP retains
only CPU scalar `(score, positive-count, total-count)` groups per label for an
exact global ordering; prediction tensors are not retained across batches.

Primary evidence remains under `datasets[dataset_id][task_id]`. The versioned
`macro_summaries` view groups task-level normalized metrics by
`(dataset_id, encoding_kind)`. Every metric records `value`/`null`,
`included_task_ids`, `undefined_task_ids`, defined/undefined task counts, and
the exact rule: an unweighted arithmetic mean over defined task-level values,
with undefined tasks excluded and counted. Defined zeros are included.
HookTheory and POP909-CL are never combined; categorical and multilabel heads
are never combined. NLL and BCE/NLL are explicitly omitted with
`scientifically_incomparable` reasons because distinct vocabularies and label
dimensions do not share one probability space. Multilabel exact-match is also
omitted because its difficulty depends on the task-specific label-set
dimension.

## Train-only baselines

`train_priors.json` is built from the selected train view before held-out
metrics:

- categorical majority class, with lowest ontology index as the deterministic
  tie-break;
- categorical empirical class probabilities for prior NLL;
- multilabel per-class prevalence;
- multilabel fixed majority prediction at prevalence `>= 0.5`.

The artifact binds the train view's index/cache/split/membership and current
ontology/encoding. It is not stored in graphs, canonical cache, or checkpoints.
A supplied `train_priors_path` is accepted only when its version, fingerprint,
source split, and complete bindings match.

## CLI

For an installed checkout, fixed validation is:

```bash
python -m music_critic.evaluation.run \
  checkpoint=/absolute/path/to/best.pt \
  data=mixed \
  data.index_paths='[/absolute/path/hooktheory.index.json,/absolute/path/pop909_cl.index.json]' \
  data.cache_roots='[/absolute/path/hooktheory,/absolute/path/pop909_cl]' \
  data.split_manifest=/absolute/path/global.split.json \
  split=validation \
  output_dir=/absolute/path/to/evaluation
```

From an uninstalled source checkout, prefix the command with `PYTHONPATH=src`.
Use explicit absolute production paths so the worktree cannot accidentally
resolve another cache. Evaluation only reads them.

Test requires:

```bash
python -m music_critic.evaluation.run \
  checkpoint=/absolute/path/to/best.pt \
  data=mixed \
  split=test \
  acknowledge_test_evaluation=true \
  output_dir=/absolute/path/to/test-evaluation
```

Positive `data.max_train_samples` or `data.max_evaluation_samples` creates an
explicitly fingerprinted bounded smoke subset. Zero means the complete view.
Do not use a bounded validation subset as evidence for a checkpoint trained
against a different validation membership: the evaluator rejects that mismatch.

Managed output collisions fail unless `overwrite_output=true`. The evaluator
writes exactly:

- `resolved_evaluation_config.json`;
- `checkpoint_evidence.json`;
- `train_priors.json`;
- `metrics.json`;
- `evaluation_report.json`.

## Bounded profiler

The profiler is disabled by default. Synthetic fixtures are the default
plumbing mode. A matrix is explicitly enabled with:

```bash
PYTHONPATH=src python -m music_critic.evaluation.profile \
  enabled=true \
  output_path=/tmp/music-critic-phase6d-performance.json
```

Defaults cover HookTheory, POP909-CL, and mixed fixtures; feature-only,
local-GNN, and hierarchical models; batch sizes 1/2/4; workers 0/2; and two
batches per cell. An unavailable worker configuration is reported rather than
silently substituted.

Each cell names independent measurement passes; values from different passes
must not be summed as one decomposition:

- `serial_exclusive_preparation` (`workers=0`) is one result-flow chain:
  per-sample canonical read, per-sample graph construction, per-batch target
  projection/alignment/tensorization, then per-batch metadata/statistics
  collation. The final collation stage consumes already tensorized targets and
  never repeats alignment. With `workers>0`, exact preparation-stage
  attribution is structured `unavailable`.
- `prepared_batch_compute` starts from prepared CPU batches and measures the
  exclusive device-transfer, model-forward, loss, backward, and optimizer-step
  chain. Its throughput excludes all dataset and loader work.
- `prepared_validation_compute` is a separate inference-only pass and is not
  added to the training chain.
- `full_loader_traversal` starts before `iter(loader)`, exhausts the loader
  without model compute, and reports first-batch latency plus total traversal.
  With workers, startup, IPC, prefetch, preparation, and collation overlap, so
  their individual attribution is explicitly unavailable.
- `end_to_end_loader_and_training_compute` starts before `iter(loader)` and
  includes startup, every loader iteration, canonical reads/preparation/
  collation, delivery, and compute through `optimizer.step`.

Every percentile series declares one unit (`per_sample` or `per_batch`).
Memory is labelled as the process-level `ru_maxrss` high-water mark, not an
isolated per-cell allocation. The report retains summaries, not per-batch
histories, and detailed timing remains outside normal training and checkpoint
determinism.

Optional `input_mode=production_read_only` requires explicit absolute index,
cache-root, and split-manifest paths plus a positive
`production_max_samples_per_dataset` capped at 32. Membership is selected by a
seeded deterministic hash and fingerprinted separately for HookTheory,
POP909-CL, and mixed cells. It reads only those indexed canonical artifacts,
writes no cache, loads no checkpoint, and never scans cache directories or
canonical corpus contents. Index and split metadata are validated in full.
A full-corpus profile is not Phase 6D-A acceptance.

For example, a one-cell read-only smoke is:

```bash
PYTHONPATH=src python -m music_critic.evaluation.profile \
  enabled=true \
  input_mode=production_read_only \
  output_path=/tmp/music-critic-phase6d-production-profile.json \
  dataset_values='[hooktheory]' \
  model_values='[feature_only]' \
  batch_sizes='[1]' \
  worker_values='[0]' \
  max_batches=1 \
  production_index_paths='[/absolute/path/hooktheory.index.json,/absolute/path/pop909_cl.index.json]' \
  production_cache_roots='[/absolute/path/hooktheory,/absolute/path/pop909_cl]' \
  production_split_manifest=/absolute/path/global.split.json \
  production_max_samples_per_dataset=1
```

## Ordinary epoch timing

Normal multi-epoch training writes `epoch_performance.jsonl` with train and
validation `wall_seconds`, `samples_per_second`, and `batches_per_second`.
Detailed stage timing remains disabled.

Wall time is nondeterministic. Consequently this sidecar is not part of the
training checkpoint, run-manifest compatibility binding, or deterministic
`metrics.jsonl` journal. Existing uninterrupted/resumed/crash-recovered model,
optimizer, RNG, checkpoint, and metric-journal evidence remains bit-exact. If
a crash commits a checkpoint before timing is written, recovery emits an
explicit unavailable timing row rather than replaying the mathematical epoch.

## Phase 8B.2A downstream comparison

Phase 8B.2A invokes this evaluator only after one supervised checkpoint exists.
Raw candidate logits are still produced before target joining, train priors
remain train-only, and every metric stays isolated by dataset/task/encoding.
The comparison layer does not create a cross-source head or a hidden global
score.

Contract `1.3.0` additionally writes `piece_statistics.json`. Each independent
piece/task/dataset/encoding row contains mergeable CPU scalars: categorical
confusion, correct/eligible and NLL-sum counts, or multilabel TP/FP/FN/TN,
support, eligible-label and BCE-sum counts. Comparison bootstrap resamples
pieces and recomputes corpus endpoints from merged counts after each draw; it
never treats an average of per-piece macro-F1 as corpus macro-F1.

Exact average precision remains a descriptive corpus metric. Scalar score
groups are ordered globally, equal scores share one threshold, and per-label AP
is undefined only without eligible positive support. Exact bootstrap AP would
require retaining prediction-score rows, so it is excluded from inferential
claims. No evaluation artifact retains CUDA tensors.

All downstream cells use one fixed, fingerprinted validation view: the full
split by default or an explicitly bounded subset without replacement. The
training checkpoint, evaluation verification, and comparison protocol must
carry the same membership. Validation artifacts are aggregated over every
declared paired seed for `(variant_id, transfer_mode)` before ranking; test
metrics cannot participate. The comparison test-lock requires the complete
selected seed-checkpoint manifest, acknowledgement, a new output directory,
test-membership evidence recorded before inference, and a single-use identity
for the chosen seed checkpoint. Ordinary `acknowledge_test_evaluation` alone
does not bypass that comparison-level lock. See
`PHASE8B2_COMPARISON_PROTOCOL.md`.

Comparison planning may resolve test membership metadata before unlock. This
means only the membership fingerprint, count, per-dataset counts and split
binding are available; full test piece identities are not serialized. Plans,
locks and final reports distinguish this from model/data access with
`test_membership_metadata_resolved=true`, `test_inference_performed=false`,
`test_targets_accessed=false`, and `test_metrics_accessed=false`. No test model
forward or target/metric read is permitted before single-use authorization.

## Phase 9B.2B Dilemmadata source-entry evaluation

Dilemmadata evaluation calls `model.predict(raw_graph_batch)` before reading a
target sidecar. Candidate log-probabilities belonging to one source entry are
averaged, preserving the training denominator. Each of the four active tasks
reports mean NLL, top-1 accuracy, macro-F1 over classes with true support
(`supported_true_classes_v1`), weighted-F1, balanced accuracy, per-class
precision/recall/F1/support and confusion; quality tasks additionally report
top-3 accuracy. Zero support and zero denominators use structured undefined
reasons. Eligible expanded rows, effective source entries, masked, conflict
and available-unaligned counts remain explicit.

Metrics are projected globally, per record and per split-atomic raw-equivalence
component. Alternative analysis views remain separate entries. Paired
confidence intervals resample components, never expanded candidate rows.
Majority and empirical-prior baselines are fingerprinted train-only artifacts;
zero train probability is reported rather than smoothed from validation/test.
Selection defaults to validation. Test inference requires an explicit unlock
bound to the exact test-membership fingerprint.

## Phase 9B.2C bounded validation evidence

The executable smoke selects its validation identities/component groups with
seed 17, without replacement and without reading validation labels or target
artifacts. It requires AN and DLC, remains disjoint from the train smoke
membership, then runs the existing official evaluator only after checkpoint
reload. The sealed report retains per-task source-entry NLL, top-1, macro-F1,
balanced accuracy, quality top-3, record/component projections, undefined
reasons, and train-only prior evidence. Test inference, targets, metrics, and
unlock remain false. These bounded metrics carry no scientific-quality claim.
Evaluation after reload additionally requires the same observed exact
target-index fingerprint stored by the run checkpoint; another semantically
equivalent physical index requires a new run rather than silent rebinding.
