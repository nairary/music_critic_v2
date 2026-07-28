# Phase 6D-A supervised evaluation

## Scope

Phase 6D-A evaluates the existing feature-only, local-GNN, and hierarchical
supervised auxiliary heads. It is not a critic score, calibration result, SSL
objective, or new checkpoint-selection policy. It changes no model, graph,
adapter, target, corpus, cache, ontology, or encoding semantics.

Evaluation/artifact, train-prior, and profiler contracts are version `1.0.0`.

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

Validation is the default. It uses the same fixed full-view or hash-selected
membership policy as Phase 6C. Test evaluation fails before checkpoint access
unless `acknowledge_test_evaluation=true`; every test artifact states that
test was not used for checkpoint selection.

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

An undefined denominator produces:

```json
{
  "value": null,
  "undefined": {
    "category": "zero_true_positive",
    "reason": "no eligible positive label exists"
  }
}
```

Absence of positive labels is therefore not reported as zero quality.
Likelihood sums use exact accumulation of the observed binary64 values, and
confusion/count state is fixed by the ontology size. Prediction tensors are
not retained across batches.

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

The profiler is synthetic and disabled by default. The full requested matrix
is explicitly enabled with:

```bash
PYTHONPATH=src python -m music_critic.evaluation.profile \
  enabled=true \
  output_path=/tmp/music-critic-phase6d-performance.json
```

Defaults cover HookTheory, POP909-CL, and mixed fixtures; feature-only,
local-GNN, and hierarchical models; batch sizes 1/2/4; workers 0/2; and two
batches per cell. An unavailable worker configuration is reported rather than
silently substituted.

Each completed cell separates canonical artifact read, graph construction,
target alignment/tensorization, collation, device transfer, model forward,
loss construction, backward, optimizer step, and validation forward. It
reports samples/s, batches/s, nodes/s, edges/s, eligible target rows/s,
mean/p50/p90/p95/p99 batch time, CPU peak RSS, and a fingerprint of dataset
membership, model configuration, batch size, and workers.

The report retains summaries, not per-batch histories. It does not add CUDA
synchronization. A production smoke must use a separate explicitly bounded,
read-only command; a full corpus profile is not Phase 6D-A acceptance.

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
