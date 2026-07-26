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
There is no song-only or mean-pooled head in Phase 6A. Task heads gather by
explicit node type, validated global entity index, and sample index. A learned
node-type embedding disambiguates tasks routed from onset, beat, or bar rows.

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

## Loss and reconstruction

Loss contract `1.0.0` uses unreduced cross entropy for closed categorical
tasks and unreduced BCE-with-logits, averaged across classes per row, for
closed multi-label tasks. Eligibility is exactly availability, valid entity
index, model readiness, and `fully_supervised` regime. Reduction is:

1. mean eligible row loss within each task/node-type/sample group;
2. mean active group loss within each task;
3. configurable weighted mean over active tasks.

A task with no eligible row is absent from the aggregate rather than receiving
an artificial target or producing NaN. Every task output retains row indices,
node types, sample indices, logits, eligibility, and unreduced local losses.

Raw reconstruction contract `1.0.0` predicts one existing inference-safe local
field per mandatory node type: song duration, track program, bar meter
numerator, beat downbeat flag, onset time, and note pitch. Categorical fields
use CE; normalized continuous fields use masked Smooth L1. It does not mask or
remask visible inputs and is not an SSL, likelihood, anomaly, or quality
objective.

## Diagnostics and checkpoints

The deterministic single-note diagnostic reports the changed stable note ID,
feature/final/per-layer L2 and cosine deltas, affected onset/beat/bar rows,
changed-node counts, pitch-reconstruction logit/loss delta, and mean pairwise
cosine per scale. It proves only that local evidence remains accessible; it
does not decide which graph is better.

Checkpoint contract `1.0.0` binds model contract/configuration, canonical
schema, graph schema/builder, feature registry version and fingerprint, target
ontology version and fingerprint, target encoding version and fingerprint, and
the exact ordered active-head specifications. Compatibility is checked before
weights or optimizer state are loaded. Checkpoints remain external artifacts.

## Phase 6B boundary

Phase 6B owns deterministic hierarchy pooling, bar and track tokens, the
bar+track Transformer, song embedding, and top-down fusion. It must preserve
local rows and cannot make mean-only aggregation the final evidence path.
Its ablation must compare feature-only, local GNN, and local GNN plus hierarchy
on the same data. A future critic must compare global context with retained
local or top-k worst evidence. A shared pitch-class-set head remains blocked
until a versioned lossless renderer/crosswalk is accepted.
