# Multi-source Alignment, Encoding, and Collator Contract

Status: **IMPLEMENTED PHASE 5B.1 CONTRACT**

Target encoding registry version: `1.0.0`

Implementation: `music_critic.tasks`

## Scope

Phase 5B.1 turns an ordered, non-empty sequence of already prepared
`MultiSourceSample` values into one validated `MultiSourceBatch`. It implements
exact target alignment, versioned encoding, tensorization, normal PyG
collation, local-to-global offsets, and deterministic statistics. Corpus
loading/indexing, `Dataset`, `DataLoader`, workers, mixture sampling, splits,
models, heads, and losses remain Phase 5B.2 or Phase 6 work.

The collator never adds supervision or sample identity to PyG stores. The
`Batch` contains only the Phase 3A raw graph contract plus PyG's allowed
per-node `batch` and `ptr`. Values, masks, confidence, provenance, diagnostics,
dataset identity, and lineage remain sidecars.

## Public API

`music_critic.tasks` exports:

- `align_sample_targets`, `AlignedTargetFamily`, `AlignedTargetRow`, and
  `TargetAlignmentError`;
- immutable `AlignmentIndex`, `build_alignment_index`,
  `align_targets_with_index`, and optional `AlignmentOperationCounts`;
- `TARGET_ENCODING_REGISTRY_VERSION`, `TARGET_ENCODINGS`,
  `TARGET_ENCODING_BY_TASK`, `TargetEncodingSpec`, deterministic serialization,
  and an encoding fingerprint;
- `tensorize_aligned_targets` and `collate_multisource_samples`;
- immutable `BatchTarget`, `BatchStatistics`, `TaskBatchStatistics`, and
  `MultiSourceBatch`;
- `prepare_multisource_sample`, verified `build_multisource_sample`, and
  graph-free `project_multisource_targets` for inventory audits;
- raw-only `benchmark_multisource_collator`/`CollatorBenchmark` and
  target-heavy `benchmark_target_alignment`/`TargetAlignmentBenchmark`.

`prepare_multisource_sample(piece)` builds the Phase 3A raw graph internally
and stores its complete deterministic fingerprint as an immutable sidecar
binding. `build_multisource_sample(piece, graph)` is the external-graph
compatibility path and accepts the graph only when its complete fingerprint
equals a fresh `build_raw_graph(piece)` projection. There is no public
verification bypass. The collator recomputes the fingerprint before each use,
so categorical-feature, continuous-feature, or topology mutation after
preparation fails with `multisource.raw_graph_binding_mismatch`.
`project_multisource_targets` deliberately has no raw graph and is restricted
to inventory/audit work. No binding value is written into PyG stores.

## Exact alignment

No graph float feature is read to reconstruct canonical time. Alignment uses
canonical entity IDs, exact `RationalTime`, and the deterministic entity order
emitted by the graph builder.

One immutable `AlignmentIndex` is built before task rows are processed. It
contains O(1) note-ID and annotation-ID mappings, O(1) exact-time mappings for
onset/beat/bar boundary candidates, and sorted rational candidate times with
corresponding local indices. A half-open span uses
`[bisect_left(start), bisect_left(end))`; no target-row loop reconstructs an
index or scans a complete candidate store. For the fixed 18-task registry,
total alignment work is:

```text
O(piece entities + target entries * log(candidate count) + emitted rows)
```

Note identity and annotation lookup are O(1), exact event lookup is O(1) plus
output, and span lookup is O(log C + output). Deterministic merge walks each
task's allowed candidate stores in canonical local-index order; across the
fixed registry this is part of the O(piece entities) term. Optional operation
counts expose index builds, index entries, lookups, bisections, candidate
matches, merge candidate-slot visits, and emitted rows for non-timing scaling
tests.

| Policy | Candidate rule |
|---|---|
| Note identity | Exact canonical `note_id` to the `note` store. |
| Region/coverage span | Expand to every onset point and beat/bar start anchor satisfying `start_qn <= time < end_qn`. |
| Boundary event | Expand to every onset/beat/bar candidate whose exact canonical time equals the span start. No snapping, tolerance, nearest neighbour, or node-type priority. |

An available annotation creates one row for every allowed typed candidate.
The node type is explicit, so the same integer in `onset` and `beat` identifies
different nodes. An available annotation with no candidate remains one row
with value and source availability intact, `entity_indices=-1`,
`entity_index_mask=false`, and null node type. It is not
supervision-eligible. A
masked source entry also remains one row, but is not candidate-expanded and
has a null value.

Equal normalized values that address the same typed candidate merge
deterministically while retaining every source entity/provenance reference.
Different values produce one unavailable conflict row with diagnostic
`multisource.alignment_conflict`; no source priority, last-write rule, or
majority vote is used.

## Encoding registry

All 18 source-native ontology tasks stay independent.

| Encoding kind | Values | Sentinel | Model-ready |
|---|---|---|---|
| `closed_categorical_index` | `torch.long [N]`, ontology vocabulary order | `-1` under false availability | yes |
| `closed_multilabel` | `torch.bool [N, C]`, ontology vocabulary order | all false under false availability | yes |
| `open_string_cpu` | deterministic CPU `tuple[str \| None, ...]` | `None` | no |

The open tasks in registry `1.0.0` are `theory.local_key.mode` and
`theory.chord.borrowed`. Their strings are preserved losslessly. No
per-batch/per-worker vocabulary, Python hash, or fixture-derived numeric ID is
created.

Each `BatchTarget` carries its task/source semantics, encoding kind and
version, values, independent availability and entity-index masks, explicit
node types, global entity indices, sample indices, optional confidence plus
confidence mask, CPU provenance/diagnostics, source annotation count,
candidate-expanded row count, model-readiness metadata, and one semantic
`supervision_regime`: `fully_supervised`, `positive_unlabeled`, or
`deferred_open_vocabulary`. Encoding declares representation, not a concrete
training loss. Supervision eligibility is exactly:

```text
availability_mask & entity_index_mask & model_ready
```

An all-false multilabel row is a true negative only when availability is true.
When availability is false it is a sentinel and is not eligible.

## PyG offsets and validation

Graphs are collated by `Batch.from_data_list`. For each aligned row and its
explicit node type:

```text
global_index = local_index + batch[node_type].ptr[sample_index]
batch[node_type].batch[global_index] == sample_index
```

The second equality is checked during tensorization and again by
`MultiSourceBatch`. The existing exact raw-batch validator also checks store
allowlists, metadata, shapes/dtypes, `ptr`/`batch`, endpoints, reverse edges,
cross-graph isolation, and source-graph reconstruction. Malformed offsets and
injected global/node/edge fields are rejected.

## Boundary and statistics

`pop909_cl.chord.boundary` contains only annotated positive events and has
`supervision_regime=positive_unlabeled`. The collator does not enumerate other
candidates or synthesize an absent/negative class. Phase 5B.1 exposes no
CE/BCE/focal/PU choice. Expected Phase 6 decisions are: likely CrossEntropy
for closed multiclass, likely BCEWithLogits for closed multilabel, CE or BCE
for binary presence, a PU-compatible objective or disabled task for POP
boundary, and disabled open-vocabulary tasks until a versioned codec exists.

`BatchStatistics` distinguishes source annotation count from expanded target
row count. Per-task and aggregate records separately count
`model_encodable_row_count`, `supervision_eligible_row_count`,
`masked_row_count`, `available_unaligned_row_count`, `conflict_row_count`, and
`deferred_open_vocabulary_row_count`. Eligibility equals the exact sum of
each task's `supervision_eligibility_mask`; model-encodable rows may still be
masked, conflicting, or unaligned. The partition and aggregate/per-task sums
are constructor invariants.

## Verification and benchmark policy

Tests use bounded HookTheory and POP909-CL records plus a synthetic raw-only
sample. They cover typed alignment, half-open/exact boundaries, duplicate
merge/conflicts, masks and sentinels, open strings, offsets, raw-store leakage,
malformed batches, deterministic repetition, and mixed-batch statistics.

The raw-only baseline runs dozens of small prepared graphs:

```bash
PYTHONPATH=src python scripts/benchmark_multisource_collator.py \
  --samples 32 --repeats 3
```

It remains graph/collator baseline evidence and is not evidence for
target-alignment scaling. The separate target-heavy benchmark contains many
note-identity targets, annotation spans expanded over onset/beat/bar
candidates, exact boundary events, a masked entry, and an
available-but-unaligned entry at small/medium/large sizes:

```bash
PYTHONPATH=src python scripts/benchmark_multisource_collator.py \
  --target-heavy --repeats 3
```

It reports index construction, target lookup, emitted rows, complete
collation, and deterministic operation counts separately. The heavy command
is not in default CI; CI uses a small instrumentation-based scaling test with
no timing threshold. Neither benchmark is a corpus acceptance test. Phase
5B.1 performs no full HookTheory scan and does not repeat the 909-file
POP909-CL acceptance.
