# Phase 6C reproducible baseline training

Phase 6C is a supervised training harness for the already accepted
feature-only, local-GNN, and hierarchical baselines. It does not change model,
loss, target, adapter, graph, canonical, or corpus semantics. It does not
implement SSL, corruption/remasking, preference learning, PLL, or Phase 7.

POP909-CL runtime adapter `2.0.0` separates source-record piece identity from
score-only raw-input equivalence. The splitter therefore keeps exact score
duplicates atomic without collapsing their distinct target observations into
one Dataset sample.

Install the optional training dependency:

```bash
python -m pip install -e '.[training]'
```

## Structured configuration

Hydra composes only registered structured groups:

- `model=feature_only|local_gnn|hierarchical`;
- `data=bounded|hooktheory|pop909_cl|mixed`;
- `experiment=one_batch|smoke|train|supervised_baseline|joint_visible_reconstruction`;
- `optimizer=adamw`;
- `objective=preset|one_batch_joint|supervised_harmonic|joint_visible_reconstruction`;
- `scheduler=none|cosine`;
- `device=cpu|cuda|auto`; an explicit index uses the registered CUDA group plus
  an exact name override, for example `device=cuda device.name=cuda:0`.

The schema has explicit values for seed, batch size, worker count, epoch size,
mixture weights, learning rate, weight decay, gradient clipping, epochs, AMP,
output directory, checkpoint interval, and validation interval. There are no
environment-derived or timestamp-derived training defaults. Every run writes
the fully resolved application configuration to `resolved_config.json`.
Fresh runs reject an output directory that already contains managed training
artifacts. Set `experiment.overwrite_output=true` to remove the complete
managed artifact set before starting; unknown files are preserved. Resume and
overwrite are mutually exclusive.

The presets are intentionally different:

| experiment | default LR | harmonic weight | reconstruction weight | purpose |
| --- | ---: | ---: | ---: | --- |
| `one_batch` | `0.02` | `1` | `1` | bounded overfit/plumbing evidence |
| `smoke`, `train`, `supervised_baseline` | `3e-4` | `1` | `0` | supervised harmonic baseline |
| `joint_visible_reconstruction` | `3e-4` | `1` | `1` | separately named visible-input ablation |

`objective.harmonic_weight`, `objective.reconstruction_weight`, and
`objective.task_weights` are explicit configuration. For example,
`objective=supervised_harmonic
+objective.task_weights={theory.chord.extent:2.0}` changes one accepted
fully-supervised task weight. Positive-unlabeled and deferred open-vocabulary
tasks are rejected. The resolved objective and task weights participate in
the configuration and model/checkpoint fingerprints.

## Cache and split preparation

Cache building remains the explicit Phase 5B.2 offline operation. These
examples are bounded; remove `--limit` only when intentionally performing a
full build:

```bash
python scripts/build_multisource_cache.py \
  --cache-root /data/cache/hooktheory \
  --index-output /data/cache/hooktheory.index.json \
  --report-output /data/cache/hooktheory.report.json \
  --limit 4 hooktheory \
  --raw-path /data/hooktheory/4_merged.json \
  --structure-root /data/hooktheory/structure

python scripts/build_multisource_cache.py \
  --cache-root /data/cache/pop909_cl \
  --index-output /data/cache/pop909_cl.index.json \
  --report-output /data/cache/pop909_cl.report.json \
  --limit 4 pop909_cl \
  --corpus-root /data/pop909-cl
```

Create one target-blind, globally validated split. Repeated indices are
validated together, so source and lineage components cannot cross splits:

```bash
python -m music_critic.training.make_split \
  --index /data/cache/hooktheory.index.json \
  --index /data/cache/pop909_cl.index.json \
  --ratio train=0.8 \
  --ratio validation=0.1 \
  --ratio test=0.1 \
  --seed 42 \
  --output /data/cache/global.split.json
```

The accepted local full-corpus run retained all 908 POP samples and one
quarantine (`172`), produced 907 raw-equivalence groups and POP index
fingerprint
`b2008221fa59ddd0df31289561b22341db9c2eac527e1a503eac57b74da27daf`.
The unchanged 26,175-record HookTheory index plus POP index produced manifest
fingerprint
`b0546316acb225bb95439dab78fab95232b0a7a758316b69b85dc87f733c384d`.
The complete audit of 27,083 active corpus records passed; records 543 and 553
share one component and are both in `train`. Physical immutable cache artifacts
are more numerous because generations from older adapter versions remain
retained. Generated cache/index/report/split files remain local and ignored.

## One-batch optimization evidence

The bounded CPU command is:

```bash
python -m music_critic.training.run \
  experiment=one_batch \
  model=hierarchical \
  data=bounded \
  device=cpu
```

The CUDA equivalent is:

```bash
python -m music_critic.training.run \
  experiment=one_batch \
  model=hierarchical \
  data=bounded \
  device=cuda \
  device.amp=true \
  device.non_blocking=true
```

Use the first actual train batch from existing cache artifacts by selecting a
production data group and overriding its paths:

```bash
python -m music_critic.training.run \
  experiment=one_batch \
  model=hierarchical \
  data=mixed \
  data.index_paths='[/data/cache/hooktheory.index.json,/data/cache/pop909_cl.index.json]' \
  data.cache_roots='[/data/cache/hooktheory,/data/cache/pop909_cl]' \
  data.split_manifest=/data/cache/global.split.json \
  device=cpu \
  output_dir=/data/runs/phase6c-real-one-batch
```

The runner repeats exactly that one batch, logs harmonic, reconstruction, and
total loss separately, checks finite losses/gradients, clips gradients, records
candidate and availability counts plus parameter gradient coverage, and
requires both active objectives to decrease. It writes
`one_batch_report.json` and `one_batch.pt`, reloads the checkpoint, and requires
bit-exact eval logits. This is optimization-plumbing evidence, never
generalization or quality evidence.

## Multi-epoch supervised baseline

```bash
python -m music_critic.training.run \
  experiment=train \
  model=local_gnn \
  data=mixed \
  data.index_paths='[/data/cache/hooktheory.index.json,/data/cache/pop909_cl.index.json]' \
  data.cache_roots='[/data/cache/hooktheory,/data/cache/pop909_cl]' \
  data.split_manifest=/data/cache/global.split.json \
  data.mixture_weights.hooktheory=1 \
  data.mixture_weights.pop909_cl=1 \
  scheduler=cosine \
  device=auto \
  output_dir=/data/runs/phase6c-train
```

Training uses the existing `CorpusIndex`, `CorpusCacheConfig`,
`SplitManifest`, `MultiCorpusDataset`, `DeterministicQuotaSampler`, and
`make_multisource_dataloader` contracts. Each epoch records per-task losses,
availability counts, realized dataset counts, aggregate losses, validation
metrics, `learning_rate_used` for optimizer steps in that epoch, and
`next_learning_rate` after `scheduler.step()` in `metrics.jsonl`. There is no
ambiguous `learning_rate` field. The run also writes resolved
configuration, mixture statistics, corpus/index/split/composition
fingerprints, and the existing model-contract fingerprint.

Training membership may change deterministically with epoch through
`DeterministicQuotaSampler`. Validation never does. By default,
`data.validation_epoch_size=0` visits the complete validation view exactly
once in canonical order, without replacement, repetition, or omission. A
positive value selects one seed-bound hash-ranked subset once, restores its
canonical order, and records selected identities, full/selected counts,
per-dataset counts, and a membership fingerprint. `best.pt` is selected only
from this fixed procedure.

Epoch metrics do not average batch means. Each task records
`loss_numerator`, exact `eligible_row_count`, and their ratio. The harmonic
epoch scalar is the task-weighted mean of active epoch task means. The visible
reconstruction scalar is the mean of active field-level means. The final
configured objective is:

```text
harmonic_weight * harmonic_epoch_mean
  + reconstruction_weight * reconstruction_epoch_mean
```

Unavailable terms are omitted rather than turned into zero or negative
examples. The same numerator/denominator accounting is emitted per dataset,
so validation is independent of batch size, partitioning, and order within
the documented floating-point tolerance. Each batch reduces all task and
reconstruction values into dataset/field scalars on device, packs them, and
performs at most one device-to-host transfer. Only CPU numerator/denominator
buckets are retained across batches. Persistent GPU metric storage is
therefore zero (and in particular bounded by
`O(dataset_count * task_or_field_count)`); transient device storage is bounded
by the current batch. The report distinguishes actual CUDA-to-host transfers
from packed host materializations, and records actual packed scalar and
retained tensor/byte counts.

`last.pt`, `best.pt`, and interval `epoch-NNNN.pt` checkpoints contain model,
optimizer, scheduler, AMP scaler, next epoch, best validation metric, and
Python/CPU-torch/CUDA-torch RNG state. Resume is accepted only at an epoch
boundary:

```bash
python -m music_critic.training.run \
  experiment=train \
  model=local_gnn \
  data=mixed \
  experiment.resume_from=/data/runs/phase6c-train/last.pt \
  output_dir=/data/runs/phase6c-train \
  data.index_paths='[/data/cache/hooktheory.index.json,/data/cache/pop909_cl.index.json]' \
  data.cache_roots='[/data/cache/hooktheory,/data/cache/pop909_cl]' \
  data.split_manifest=/data/cache/global.split.json
```

All contract-bound configuration and data fingerprints must match the saved
checkpoint and the existing `run_manifest.json`. Resume validates the
manifest and fingerprints before writing or journal recovery. Loading
prevalidates the complete payload, requires
`0 <= next_epoch <= experiment.epochs`, validates the matching committed-row
count, model/optimizer structure, auxiliary state application, and RNG
shape/availability as far as possible. If live application still fails,
model, optimizer, scheduler, scaler, and Python/CPU/CUDA RNG are rolled back
bit-exactly.

Each epoch is committed through an atomic pending metric, `last.pt` with its
`committed_metric_rows`, and an atomic per-epoch metric. `metrics.jsonl` is
deterministically rebuilt from committed records. Resume discards a metric
staged before a missing checkpoint and replays that epoch, or finalizes a
pending metric already authorized by `last.pt`; it never duplicates or loses
an epoch row. Mid-epoch resume is deliberately unsupported. Under the
supervised preset, a batch without eligible harmonic rows is skipped
gracefully. It optimizes reconstruction only in the explicitly selected joint
ablation; missing labels never become negative or zero-loss examples.

## Device-transfer boundary and CUDA acceptance

`move_multisource_batch(batch, device, non_blocking=...)` is the official
non-mutating boundary under device-transfer contract `1.0.2`. One shared
runtime resolver canonicalizes CPU to `cpu`, bare CUDA to the concrete
`cuda:<torch.cuda.current_device()>`, and preserves an explicit `cuda:N` only
when the index is smaller than `torch.cuda.device_count()`. The current index
is checked against the same count. Every CUDA request fails structurally when
CUDA is unavailable; a nonexistent explicit or current index fails before
`.to(...)` with `runtime.device.cuda_index_out_of_range`, requested-device
evidence, and the visible count. Runtime device-resolution contract `1.0.1`
performs no tensor allocation or value materialization. It deep-copies
the raw PyG batch and transfers only
tensor attributes, preserving tuple-valued graph metadata. Model-facing target
tensors move to the same device, while strings, provenance, diagnostics,
statistics, and other CPU sidecars remain CPU objects. Targets are never added
to the raw graph. Full graph/target binding validation runs on the CPU source
before transfer. The normal device path performs only structural
device/shape/task-order checks without data-dependent CUDA predicates. Full
post-transfer semantic validation is available through
`debug_validate_device=True`.

Post-transfer checks compare the complete `torch.device`, including its CUDA
index. They never accept a tensor merely because its device type is `cuda`;
an expected `cuda:1` rejects an actual `cuda:0`. Evaluation uses the same
resolver and transfer boundary. Device evidence reports the concrete runtime
device, for example `cuda:0`, rather than the unresolved selector `cuda`.
Training, SSL, and evaluation accept `cpu`, `cuda`, `cuda:N`, and `auto`
through the shared resolver. AMP is accepted only when the resolved device
type is CUDA, so `device.name=cuda:0` and CUDA-resolving `auto` are valid while
CPU AMP is rejected.

Phase 7A representation loss `1.0.1` keeps exact shape/device validation but
allows FP16/BF16/FP32 prediction-target pairs by computing cosine, numerator,
mean, multi-view aggregation, and the combined SSL objective in FP32 with
autocast disabled. Only matching FP64 pairs remain FP64. Targets are detached;
the out-of-place FP32 prediction cast remains differentiable. Anti-collapse
diagnostics `1.1.1` use the same normalization. Multi-view loss and combined
SSL objective are `1.0.1`; the umbrella SSL contract is `1.2.2`.

Normal multi-epoch training does not collect parameter-by-parameter gradient
evidence. The default engine/device hot path has no tensor-to-Python
conversion. Joint visible reconstruction computes field availability
device-side without one data-dependent host predicate per family. Metrics use
one explicit packed device-to-host transfer per non-empty batch and retain no
device tensors afterward; these are runtime counters, while static
source-contract tests guard the prohibited conversion sites. Gradient
evidence is reserved for one-batch mode or an explicit
`experiment.collect_gradient_evidence=true` diagnostic run.

`tests/training/test_cuda_acceptance.py` executes the documented CLI itself
with `device.amp=true`, checks the enabled scaler, backward/optimizer steps,
finite loss curves, checkpoint save/reload, and reported VRAM. A second test
perturbs only sample-zero note rows whose velocity availability bit is true,
keeps unavailable placeholders bit-exact, revalidates the raw graph, requires
the changed sample's fused embeddings and logits to change, and requires every
other sample's logits and fused embeddings to remain bit-exact. The CUDA
resume test compares JSON-loaded and in-memory membership through canonical
JSON normalization while separately preserving the exact membership
fingerprint, counts, limits, ordered identities, and byte-identical
`metrics.jsonl`; production selection and resume binding are unchanged. A
CPU-only CI runner reports explicit skips. Manual GPU
evidence must name the actual device and report peak allocated/reserved bytes
from `one_batch_report.json`; GPU or VRAM results must not be inferred when
CUDA hardware is absent. Additional optional tests exercise a real
hierarchical supervised train/validation path with AMP, fixed validation,
uninterrupted versus epoch-boundary-resumed equivalence, finite losses and
gradient scans, checkpoint reload, and bounded retained metric VRAM.
