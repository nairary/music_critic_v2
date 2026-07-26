# Phase 6A Local Baseline Contract

## Role and boundary

Phase 6A is the first learned Music Critic V2 phase. It establishes a
CPU-compatible raw-graph representation baseline and source-specific auxiliary
harmony heads. It is not SSL, a corruption detector, a probabilistic
likelihood/PLL model, an anomaly detector, a quality critic, or an aesthetic
model. Small visible-input reconstruction is only a trainability and overfit
plumbing check.

The model consumes exactly the validated Phase 3A `song`, `track`, `bar`,
`beat`, `onset`, and `note` stores. Dataset/piece identity, split, source and
lineage groups, provenance, target availability, target values, theory labels,
semantic roles, and target-derived topology are not encoder inputs.

## Controlled variants

Both variants use `LocalBaselineConfig`, the same per-feature encoders, local
task heads, losses, reconstruction fields, and batches:

- `feature_only`: per-node-type feature encoding and local heads, with no
  message-passing parameters;
- `local_gnn`: the same path plus a separate learned projection for each of
  the 26 ordered Phase 3A edge types.

The reference configuration is hidden dimension 128, three local layers,
dropout 0.1, residuals, LayerNorm, GELU, and sum relation aggregation. Small
CPU tests use hidden dimensions 16/32 and one or two layers.

Categorical columns have independent versioned embeddings and learned
availability embeddings. Continuous columns have independent scalar
projections and learned availability embeddings. Log-count fields use signed
`log1p`; all other scalars use `x / (1 + |x|)`, including values already
z-scored by the raw feature registry. Unavailable values are zeroed only after
that transform and receive a distinct learned availability signal, so
unavailable and observed zero are not equivalent.

Each local GNN layer sums relation-specific vectorized messages, combines a
self projection and configurable residual, then applies LayerNorm, GELU, and
dropout. Forward and reverse relations remain different parameters. Empty
edge stores are valid and no node row is added, removed, pooled, or reordered.
The final representation fuses the original feature scale with the last local
scale, so the deepest GNN layer is not the only surviving evidence.

## Local output and heads

Encoder output contract `1.0.0` retains one final row and batch membership per
original node, plus the feature scale and optional per-layer local embeddings.
There is no song-only or mean-pooled head in Phase 6A. Task heads enumerate
every candidate in each task's allowed raw node stores before consulting any
target sidecar. Candidate identity is the tensor tuple
`(node_type_code, global_entity_index, sample_index)` and candidate logits are
therefore present for raw-only inference. A learned node-type embedding
disambiguates onset, beat, and bar candidates. Targets join to those existing
identities only to compute training losses; replacing, deleting, masking, or
adding target rows cannot change candidate identities or eval logits.

Exactly these source-native fully supervised heads exist:

- HookTheory: `theory.melody.scale_degree`,
  `theory.local_key.tonic_pc`, `theory.chord.presence`,
  `theory.chord.root_degree`, `theory.chord.extent`,
  `theory.chord.inversion`, `theory.chord.adds`, `theory.chord.omits`,
  `theory.chord.alterations`, and `theory.chord.suspensions`;
- POP909-CL: `pop909_cl.chord.root`, `pop909_cl.chord.quality`,
  `pop909_cl.chord.bass`, and `pop909_cl.chord.inversion`.

There are no shared HookTheory/POP heads or pitch-class-set head. Open-string
`theory.local_key.mode` and `theory.chord.borrowed`, plus positive-unlabeled
`pop909_cl.chord.boundary` and `pop909_cl.chord.no_chord`, have no instantiated
head and contribute no loss. No absent event/span becomes a negative.

## Candidate prediction, loss, and reconstruction

Model/output contract `1.1.0`, candidate prediction contract `1.0.0`, loss
contract `1.1.0`, and `BatchTarget` contract `1.1.0` make this split explicit.
`BatchTarget.entity_node_type_codes` is the tensor routing key; CPU node-type
strings remain validation/provenance sidecars. Candidate enumeration, target
join, and task/node-type/sample grouping use tensor indexing, `torch.unique`,
and `index_add_`. Python work is bounded by fixed task and node-type
registries; model forward/loss has no per-target-row or per-candidate-row host
materialization or row list comprehension.

Loss uses unreduced cross entropy for closed categorical
tasks and unreduced BCE-with-logits, averaged across classes per row, for
closed multi-label tasks. Eligibility is exactly availability, valid entity
index, model readiness, and `fully_supervised` regime. Reduction is:

1. mean eligible row loss within each task/node-type/sample group;
2. mean active group loss within each task;
3. configurable weighted mean over active tasks.

A task with no eligible row is absent from the aggregate rather than receiving
an artificial target or producing NaN. Predictions retain all raw candidate
identities and logits; the separate supervision join retains target-row and
candidate indices plus unreduced local losses. A raw-only batch has non-empty
candidate logits and no harmonic loss.

Raw reconstruction contract `1.0.0` predicts one existing inference-safe local
field per mandatory node type: song duration, track program, bar meter
numerator, beat downbeat flag, onset time, and note pitch. Categorical fields
use CE; normalized continuous fields use masked Smooth L1. It does not mask or
remask visible inputs and is not an SSL, likelihood, anomaly, or quality
objective.

## Diagnostics and checkpoints

The deterministic single-note diagnostic changes one pitch in a validator-clean
`CanonicalPiece`, preserves its stable note ID, and sends both original and
perturbed pieces through the production graph builder and validator. It
requires different graph fingerprints and identical entity/topology identity,
reports every changed raw store/feature/entity, local feature/layer/final L2
and cosine delta, changed-node counts, and pitch-reconstruction logit/loss
delta. Oversmoothing is computed separately for every
`(sample, node_type, scale)`. For row-normalized embeddings `u_i`, the exact
dense-convention off-diagonal sum is computed without an `N x N` matrix:

```text
diagonal_sum = sum_i ||u_i||²
pair_sum = ||sum_i u_i||² - diagonal_sum
mean = pair_sum / (N * (N - 1))
```

Production first validates the feature-scale rank-one `torch.long` membership
for non-negative, non-decreasing sample IDs and exact embedding cardinality.
It scans membership once per node type to create `S+1` deterministic
contiguous boundaries, including equal adjacent boundaries for an empty
sample. Other scales must have exactly the same membership before their
embeddings are read; malformed or non-contiguous membership raises
`OversmoothingContractError` rather than being regrouped.

Each group is selected with the basic slice `embeddings[start:end]`, which is
a view. Production therefore creates no boolean mask-selected
`N_group x D` feature copy. Each embedding row is streamed exactly once per
scale into one D-dimensional sum and one scalar; neither a normalized `N x D`
copy nor a cosine `N x N` matrix is materialized. Per group this is `O(ND)`
time and `O(D)` accumulator memory and remains exact when
`F.normalize` leaves a zero vector at zero. `zero_norm_count` is the
deterministic count of input rows whose L2 norm is exactly zero before
normalization; it exposes complete or partial zero collapse even though
PyTorch cosine assigns zero similarity to a zero row. Groups with fewer than
two nodes are explicitly unavailable but still report the count. No statistic
mixes graphs or node types. These diagnostics prove only local accessibility,
not which graph is better. There is no separately versioned diagnostic-policy
contract, so this formula correction does not change model/output/loss
versions.

For `T` node types, `S` samples, `K` encoder scales, hidden size `D`, and
`N_t` rows in node type `t`, boundary construction is
`O(sum_t N_t + T*S)` time once and `O(T*S)` CPU metadata. Cross-scale
membership validation is `O(K*sum_t N_t)`. Cosine work is
`O(K*sum_t N_t*D)` because every embedding row is processed exactly once per
scale. Traversing and constructing every requested group record adds
`O(K*T*S)` time, and the returned report necessarily stores `O(K*T*S)`
records. Empty and one-row groups do not allocate a D-dimensional
accumulator; temporary cosine accumulation for a non-trivial group remains
`O(D)`. These metadata/output terms are separate from feature memory, and no
production step allocates an `N_group x D` group copy.

Checkpoint contract `1.1.0` binds model contract/configuration, canonical
schema, graph schema/builder, feature registry version and fingerprint, target
ontology version and fingerprint, target encoding version and fingerprint, and
the exact ordered active-head specifications. Loading validates payload
structure, metadata, exact model keys/shapes/dtypes, and optimizer groups/state
tensors before mutation. Any failure leaves model and optimizer unchanged.
This includes an optimizer application that mutates live state and then raises:
the already-applied model state and complete optimizer state are restored.
Saving writes a same-directory temporary and atomically replaces the
destination. Checkpoints remain external artifacts.

## Phase 6B boundary

Phase 6B now implements deterministic hierarchy pooling, bar and track tokens,
the bar+track Transformer, contextual song rows, and top-down fusion while
preserving this complete Phase 6A output unchanged. Its controlled ablation
compares feature-only, local GNN, and local GNN plus hierarchy on the same
data. The additive contract is documented in `PHASE6B_HIERARCHY.md`.

A future critic must compare global context with retained local or top-k worst
evidence. A shared pitch-class-set head remains blocked until a versioned
lossless renderer/crosswalk is accepted. Phase 6B does not start Phase 7 SSL.
