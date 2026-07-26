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
- `TARGET_ENCODING_REGISTRY_VERSION`, `TARGET_ENCODINGS`,
  `TARGET_ENCODING_BY_TASK`, `TargetEncodingSpec`, deterministic serialization,
  and an encoding fingerprint;
- `tensorize_aligned_targets` and `collate_multisource_samples`;
- immutable `BatchTarget`, `BatchStatistics`, `TaskBatchStatistics`, and
  `MultiSourceBatch`;
- `benchmark_multisource_collator` and `CollatorBenchmark`.

`MultiSourceSample` owns the validated `CanonicalPiece` used for alignment as
well as its raw graph and source-native target sidecars. This prevents a caller
from aligning targets against a different canonical piece.

## Exact alignment

No graph float feature is read to reconstruct canonical time. Alignment uses
canonical entity IDs, exact `RationalTime`, and the deterministic entity order
emitted by the graph builder.

| Policy | Candidate rule |
|---|---|
| Note identity | Exact canonical `note_id` to the `note` store. |
| Region/coverage span | Expand to every onset point and beat/bar start anchor satisfying `start_qn <= time < end_qn`. |
| Boundary event | Expand to every onset/beat/bar candidate whose exact canonical time equals the span start. No snapping, tolerance, nearest neighbour, or node-type priority. |

An available annotation creates one row for every allowed typed candidate.
The node type is explicit, so the same integer in `onset` and `beat` identifies
different nodes. An available annotation with no candidate remains one row
with value and source availability intact, `entity_indices=-1`,
`entity_index_mask=false`, and null node type. It is not loss-eligible. A
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
candidate-expanded row count, and model-readiness metadata. Future
supervision eligibility is exactly:

```text
availability_mask & entity_index_mask
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

`pop909_cl.chord.boundary` contains only annotated positive events. The
collator does not enumerate other candidates or synthesize an absent/negative
class. Encoding metadata sets ordinary BCE eligibility to false; Phase 6 must
choose an explicit PU-compatible objective or omit this loss.

`BatchStatistics` distinguishes source annotation count from expanded target
row count and records sample/graph, node, edge, dataset, task, node-type,
aligned, unaligned, masked, conflict, model-ready, and deferred-open counts.
It is deterministic CPU metadata and does not mutate graphs or targets.

## Verification and benchmark policy

Tests use bounded HookTheory and POP909-CL records plus a synthetic raw-only
sample. They cover typed alignment, half-open/exact boundaries, duplicate
merge/conflicts, masks and sentinels, open strings, offsets, raw-store leakage,
malformed batches, deterministic repetition, and mixed-batch statistics.

The lightweight benchmark runs dozens of small prepared graphs:

```bash
PYTHONPATH=src python scripts/benchmark_multisource_collator.py \
  --samples 32 --repeats 3
```

It reports alignment, PyG graph construction/validation, complete collation,
and node/edge/target counts. It is not a corpus acceptance test and is not a
mandatory heavy CI job. Phase 5B.1 performs no full HookTheory scan and does
not repeat the 909-file POP909-CL acceptance.
