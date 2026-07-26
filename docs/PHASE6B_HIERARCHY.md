# Phase 6B Hierarchy and Coarse-Context Contract

Status: implemented and locally verified on the bounded Phase 6B fixtures.
This document describes the production Phase 6B baseline; it does not define
SSL, a likelihood model, or a music-quality critic.

## Public API and versions

Phase 6B adds, without changing the Phase 6A API:

- `HierarchicalBaselineConfig`;
- `HierarchicalHeterogeneousBaseline`;
- `HierarchicalBaselineOutput`;
- `ContextualEncoderOutput`;
- strict hierarchical checkpoint save/load functions;
- the controlled Phase 6B benchmark and single-note diagnostic.

The following independently inspectable serialized contracts are introduced at
version `1.0.0`:

- hierarchy pooling;
- coarse token sequence;
- hierarchical encoder output;
- top-down fusion;
- hierarchical model/output;
- hierarchical checkpoint.

The Phase 6A model, output, loss, encoder-output, and checkpoint contracts are
unchanged. The target ontology, target encoding, adapters, production
manifests, graph schema, and corpus contracts are also unchanged.

The default hierarchical configuration is hidden size 128, three local GNN
layers, two pre-norm Transformer layers, four attention heads, a four-times
hidden feed-forward dimension, and dropout 0.1. Bounded tests and evidence use
hidden size 32, one local layer, one Transformer layer, four heads, and zero
dropout.

## Raw hierarchy ownership

Hierarchy is derived only from validated raw graph edges:

- `beat -> bar`;
- `onset -> bar`;
- `note -> bar`;
- `note -> track`;
- `bar -> song`;
- `track -> song`.

Every child row must have exactly one owner. Forward ownership rows must be in
canonical child-row order, reverse edges must be the exact transpose, owner
indices must be in range, and child and owner membership must identify the
same sample. Membership tensors must be rank-one `torch.long`, non-negative,
non-decreasing, and cardinality-aligned with node rows. Every sample must have
exactly one song row.

Missing, duplicate, reordered, out-of-range, cross-sample, or inconsistent
reverse ownership raises `HierarchyContractError`. Production does not silently
repair or regroup malformed hierarchy. No target-only fields, semantic spans,
split metadata, or provenance participate in ownership.

Extraction first verifies that its input is a PyG `HeteroData`/`Batch`, then
checks all mandatory node stores, all twelve ownership/containment edge stores,
and each existing `edge_index` before indexing a store. Missing stores are
therefore never created as a PyG read side effect. Input type, node store, edge
store, edge attribute, tensor type/dtype/rank/shape/device, missing child,
duplicate child, reordered child, owner range, reverse mismatch, and
cross-sample failures have distinct stable `HierarchyContractError`
categories. Failed validation leaves `node_types`, `edge_types`, and every
store's attribute-key set unchanged.

Externally supplied `HierarchyOwnership` is revalidated for exact ordered keys,
sample count, complete membership maps, tensor dtype/rank/shape/device, owner
ranges, child/parent sample agreement, and exact equality with both raw edge
directions. The production model performs one complete raw ownership scan
before Phase 6A graph encoding and passes that result through an internal
local-row/device consistency check; it does not repeat the six relation scans.

## Deterministic sparse pooling

A bar token combines its own local bar row with separate beat, onset, and note
families. A track token combines its own local track row with the note family.
Each child family contributes:

- mean;
- elementwise maximum;
- `log1p(count)`;
- an explicit availability embedding;
- a learned projection.

An explicit projected parent residual is retained. Empty child families have
count zero and availability false; they do not acquire synthetic child rows.
Pooling uses sparse index accumulation over ownership indices. It creates
neither a dense membership matrix nor a child-by-parent tensor.

For `N` child rows, `P` parents, and hidden width `D`, a family accumulation is
`O(ND + PD)` time. Its output and accumulators are `O(PD)` because the required
parent-token result itself has that size; there is no `O(NP)` object.

## Coarse token sequence and Transformer

Each sample receives exactly one deterministic padded sequence:

```text
[SONG] + bars in canonical row order + tracks in canonical row order
```

SONG, bar, and track have distinct learned type embeddings. Bar ordinals and
track ordinals each restart at zero and use runtime deterministic sinusoidal
positions, so supported length is not a learned fixed table. Padding has an
explicit key-padding mask. A sample is never concatenated with another sample;
batched attention therefore cannot cross sample boundaries.

Packing computes per-sample bar/track counts with `bincount`, boundaries with
`cumsum`, local family ordinals with tensor arithmetic, and padded positions
with indexed tensor placement:

```text
song_position = 0
bar_position = 1 + bar_ordinal
track_position = 1 + bar_count[sample] + track_ordinal
```

There is no Python scan over bar or track rows and no `.tolist()`/`.cpu()` or
per-row `.item()` in production packing. Determining the padded allocation
length performs exactly one `lengths.max().item()` device-to-host
synchronization per constructed batch, independent of coarse-row count.
Indexed placement preserves gradients to song, bar, and track inputs.

The implementation is `batch_first=True`, pre-norm
`torch.nn.TransformerEncoder`. It returns contextual song, bar, and track rows
in their original graph-row order. With sample sequence lengths `L_s`, hidden
width `D`, and feed-forward width `M`, attention costs
`O(sum_s L_s^2 D)` and feed-forward processing costs
`O(sum_s L_s D M)`. Padded activation storage follows the bounded batch tensor
shape `B x max(L_s) x D` plus the attention implementation's internal state.
Before attention, count/position construction is `O(B + N_bar + N_track)` and
feature placement/positional encoding is
`O(B * max(L_s) * D + (N_bar + N_track) * D)`. Required padded output storage
is `O(B * max(L_s) * D)` with `O(B + N_bar + N_track)` tensor metadata; no
child-by-parent matrix is involved.

The contextual SONG row is a learned representation of the raw piece under
this baseline. It is not a scalar quality judgement.

## Top-down fusion and predictions

Phase 6B keeps the complete Phase 6A local multi-scale output and creates a
separate fused row for every raw node:

- note receives contextual bar, contextual track, and contextual song;
- onset and beat receive contextual bar and contextual song;
- bar and track receive their contextual parent token and contextual song;
- song receives contextual song.

Each node type uses a learned gated residual, so local evidence remains
explicitly accessible. Track context is not invented for onset or beat because
the raw graph does not own such relations.

The existing 14 source-native heads run on fused candidate rows. Candidate
identity, candidate ordering, target-only lookup, masks, loss semantics, and
reconstruction semantics remain those of Phase 6A. The bounded tiny batch
still emits 237 candidate rows, and the isolated raw-only piece still emits 79.
Changing, deleting, masking, or adding target-only values does not change
evaluation logits.

## Checkpoints

Hierarchical checkpoint contract `1.0.0` binds all six Phase 6B contracts, the
unchanged Phase 6A contracts, full hierarchical/local configuration, graph and
feature fingerprints, ontology and encoding fingerprints, and the exact
ordered head specifications.

Loading validates metadata, keys, shapes, dtypes, model state, and optimizer
state before application. If either application mutates live state and then
fails, the complete model and optimizer snapshots are restored bit-exactly.
Saving uses a same-directory temporary followed by atomic replacement.
Checkpoints remain external artifacts.

## Controlled bounded evidence

The controlled comparison uses the same deterministic batch, task registry,
loss computation, hidden width 32, one local layer where applicable, and zero
dropout. It measures feasibility only; these CPU timings are not a quality
claim or a statistically powered speed comparison.

| Variant | Parameters | Tiny forward / backward (s) | Larger forward / backward (s) |
|---|---:|---:|---:|
| feature-only | 98,757 | 0.06806 / 0.00782 | 0.19795 / 0.01843 |
| local GNN | 132,101 | 0.02017 / 0.01171 | 0.06036 / 0.01336 |
| hierarchy + Transformer | 189,701 | 0.04158 / 0.01638 | 0.04792 / 0.01791 |

Default-config reference parameter counts are 712,581, 2,292,357, and
3,384,581 respectively. The tiny batch contains 3 graphs, 28 nodes, 98 edges,
3 bars, 4 tracks, and coarse lengths `[3, 4, 3]`; the larger batch contains 9
graphs, 85 nodes, 302 edges, 9 bars, 13 tracks, and padded shape
`[9, 4, 32]`.

On the tiny batch, separately measured hierarchy stages were pooling 0.00146 s,
Transformer 0.00400 s, fusion 0.00052 s, and complete hierarchical
forward/backward 0.05796 s. On the larger batch they were 0.00189 s, 0.00164 s,
0.00051 s, and 0.06584 s. No acceptance threshold or performance conclusion is
derived from these bounded measurements.

A separate 16-repeat uneven-sequence remediation benchmark used coarse lengths
`[3, 4, 3]`, padded shape `[3, 4, 32]`, and retained 237 candidate rows.
Tensorized sequence construction averaged 0.000259 s and complete hierarchical
eval forward averaged 0.025124 s. This is bounded plumbing evidence with no
speed threshold and no production-throughput claim.

Every local node encoder, child/parent pooler, Transformer attention and
feed-forward block, all six fusion modules, and all 14 task heads receive
gradients in the bounded training check. A 30-step feasibility run reduced the
selected harmonic loss from 1.79136 to 0.00000354 and reconstruction loss from
3.11003 to 0.000325. This proves optimization plumbing, not generalization.

## Hierarchical single-note diagnostic

The deterministic diagnostic changes one canonical note pitch while preserving
entity identity, topology, ownership, candidate cardinality, and local-output
retention. The bounded run produced these non-zero L2 deltas:

| Evidence | L2 delta |
|---|---:|
| local note | 0.42141 |
| pooled bar / track | 0.10921 / 0.23121 |
| contextual bar / track / song | 0.13988 / 0.19839 / 0.02796 |
| fused note / onset / beat | 0.90333 / 0.21826 / 0.65703 |
| fused bar / track | 0.08809 / 0.18145 |
| reconstruction logits | 1.37096 |

Every listed path is asserted separately rather than through an aggregate
`any(delta > 0)` check. An unrelated sample processed in the same batch remains
bit-exact at local, pooled, contextual, and fused stages; an end-to-end
two-batch perturbation test additionally preserves its fused embeddings and all
candidate logits bit-exactly. These values show that the expected hierarchy
path is live; they do not rank music quality.

## Phase boundary

Phase 6B does not implement masking, corruption, GraphMAE-style remasking,
latent prediction, PLL, PU objectives, preference learning, scalar quality
scoring, calibration, or deployment inference. Those remain later roadmap
work. In particular, Phase 7 has not started.
