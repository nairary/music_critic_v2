# Phase 8B.1 independently ablatable multi-level SSL objectives

## Status and claim boundary

Phase 8B.1 is implemented on branch
`phase/8b1-multilevel-objectives` from merged Phase 8A main commit
`e97377c450a368d6b46d7ba8bc1c7697bdd5dd63`. It adds bounded
representation-recovery mechanics for onset, beat, bar, and track rows. It
does not establish better musical representations, downstream performance,
likelihood, preference, critic, or quality-score behavior.

This increment neither changes the canonical/raw graph nor reads theory
targets, dataset identity, split, provenance, annotations, diagnostics, or
quality flags as objective inputs. No full corpus was scanned or trained.

The initial draft registered the Hydra objective group, public builder, and a
bounded comparison runner, but the official `music_critic.ssl.run` /
`ssl.engine` path still called `build_ssl_model()` and the Phase 7A forward
unconditionally. Consequently, an explicit Phase 8B config could be present
without controlling official training. That pre-remediation runner was useful
bounded mechanics evidence, but it was not a production training path.

The remediated official dispatcher now has two fail-closed branches:

- absent or null `phase8b_objective` enters the literal pre-remediation Phase
  7A builder, binding, forward, state, loss, checkpoint, and artifact path;
- an explicit objective requires an explicit compatible `phase8b_masking`,
  materializes both configs, builds through
  `build_phase8b_model_from_config()`, and validates the model/binding/forward
  contract before the first optimizer step.

No explicit Phase 8B run silently falls back to Phase 7A.

## Separation of concerns

| Concern | Current role |
| --- | --- |
| Masking policy | Phase 8A chooses exact raw entities and pitch descendants to hide. |
| Note reconstruction | The existing Phase 7A decoder reconstructs selected note rows. |
| Latent objective | Phase 8B.1 predicts detached full-view rows at onset/beat/bar/track level. |
| Harmonic supervision | A separate future supervised path with masked auxiliary labels; never an SSL input here. |
| Critic or quality score | A separate future preference/quality contract; not inferred from these losses. |

Phase 7A's note/bar/song family names remain distinct from the new hierarchy
bar family. `phase7a_bar_latent` is the accepted integration objective over
fused bars; `hierarchy_bar_latent` is the new independently ablatable family
over contextual coarse bar rows.

## Encoder rows and objective formula

The versioned objective registry binds each family to one existing encoder
surface:

| Family | Masked online and full-view source |
| --- | --- |
| `onset_latent` | saved `fused.embeddings["onset"]` contextual/local rows |
| `beat_latent` | saved `fused.embeddings["beat"]` contextual/local rows |
| `hierarchy_bar_latent` | `coarse.bar_embeddings` contextual rows |
| `track_latent` | `coarse.track_embeddings` contextual rows |

For level `l`, a separate `LatentProjectorPredictor` computes

```text
L_l = mean(1 - cosine(P_l(z_masked), stopgrad(T_l(z_full))))
```

The online path uses the Phase 8A pitch overlay. The full-view target reuses
the same encoder with no overlay, `eval()` behavior, `torch.no_grad()`, and no
EMA teacher. Both are indexed by the same exact global raw-graph row indices.

## Exact entity eligibility and alignment

`PreparedPhase8BObjectiveBinding@1.0.0` is a portable sidecar over one exact
prepared Phase 7A/8A binding. It derives rows only from canonical plan indices
and already-attested per-node-type batch pointers.

| Phase 8A policy | Eligible Phase 8B.1 entities |
| --- | --- |
| `independent_note_pitch` | no new family; unchanged Phase 7A note control |
| `onset_pitch_descendants` | each selected onset once |
| `beat_pitch_descendants` | each selected beat once |
| `contiguous_bar_pitch_span` | each selected bar once |
| `track_bar_pitch_span` | the selected track once and each corresponding selected bar once |

Rows are canonical triples `(sample_index, local_index, global_index)`, sorted
and deduplicated. A descendant note set may feed the existing note decoder but
never multiplies one onset/beat/bar/track target. Every global index must lie
within that sample's exact node pointer interval. There is no temporal
snapping, nearest-neighbour matching, theory-derived topology, inferred PyG
store, cross-sample row, dense membership matrix, or dense pairwise
similarity matrix.

## Family loss and weighting contract

Each `Phase8BFamilyLoss@1.1.0` carries:

- summed numerator;
- eligible entity denominator;
- mean loss when available and active;
- availability and explicit unavailable reason;
- configured fixed weight;
- active/inactive state;
- zero-norm count.

A zero denominator is `unavailable` with `numerator=None` and
`mean_loss=None`; it is never a fake zero. A zero weight is
`inactive_zero_weight`: its new head is not executed and receives no gradient
path.

For one CPU batch, every scheduled policy view runs first. For each active
family `f`, the engine then computes

```text
N_f = sum_v numerator_(f,v)
D_f = sum_v eligible_denominator_(f,v)
mean_f = N_f / D_f                         when D_f > 0
total_ssl_loss = sum_(available f) configured_weight_f * mean_f
```

One configured family weight is applied once after all of that family's views
have been aggregated. There is no division by scheduled-policy count, active
weight sum, or available-family count, and no redistribution when another
family is unavailable. The same raw entity predicted in two distinct views is
two prediction observations in `N_f` and `D_f`; it is not deduplicated across
views. In particular, hierarchy-bar rows from contiguous-bar and track/bar
passes share one family-global mean and one bar-weight application.

Metrics pack all available family numerators and the optimizer total into at
most one device-to-host transfer per CPU batch, then retain only fixed-size
detached CPU scalar/O(D) sufficient state. Reports retain neither prediction
tensors, graph tensors, nor CUDA tensors.

## Configuration and public API

Hydra group `phase8b_objective` provides:

- `phase7a_control`;
- `onset_only`;
- `beat_only`;
- `bar_only`;
- `track_only`;
- `multilevel_equal_weight`.

The unchanged Phase 7A root defaults to `phase8b_objective=null`; select a mode
with `+phase8b_objective=<mode>`. An official explicit run must also select
`+phase8b_masking=<mode>`. The masking group provides the same six names plus
`phase8a_mask_only`. The only additional compatible pair is
`phase8b_objective=phase7a_control` with
`phase8b_masking=phase8a_mask_only`; it uses the exact old model/objectives
with the four Phase 8A hierarchy policies.

| Execution mode | Scheduled Phase 8A policy passes |
| --- | --- |
| Phase 7A control | independent note pitch |
| mask-only control | onset, beat, contiguous bar, track/bar |
| onset only | onset descendants |
| beat only | beat descendants |
| bar only | contiguous bar span |
| track only | track/bar span |
| equal weight | onset, beat, contiguous bar, track/bar |

Every resolved objective config binds the registry fingerprint and has its own
canonical fingerprint. Explicit Hydra weight overrides are materialized with
`Phase8BObjectiveConfig.from_hydra()`; their resolved values, including
zero-valued toggles, enter that fingerprint. The masking config independently
binds the Phase 8A policy-mixture fingerprint, fixed scheduled policies, span
parameters, and its own fingerprint. Incompatible mode/policy pairs reject;
there are no policy substitutions.

Exact bounded CPU CLI examples are:

```bash
# Literal Phase 7A control: no Phase 8 config is present.
.venv/bin/python -m music_critic.ssl.run \
  experiment=one_batch model=hierarchical data=bounded device=cpu \
  output_dir=/tmp/phase7a-control

# Official onset-only Phase 8B.1 route.
.venv/bin/python -m music_critic.ssl.run \
  +phase8b_objective=onset_only +phase8b_masking=onset_only \
  experiment=one_batch model=hierarchical data=bounded device=cpu \
  output_dir=/tmp/phase8b1-onset
```

The principal APIs are:

- `Phase8BObjectiveConfig.for_mode(...)`;
- `prepare_phase8b_objective_binding(...)`;
- `Phase8BMultilevelSSLModel.forward_multilevel(...)`;
- `build_phase8b_model(...)`;
- `build_phase8b_model_from_config(...)`;
- `Phase8BObjectiveAccumulator`;
- `aggregate_phase8b_policy_pass_losses(...)`;
- `ResolvedPhase8BMaskingConfig` and `run_phase8b_training(...)`;
- `transfer_phase7a_checkpoint_to_phase8b(...)`;
- `run_phase8b_bounded_comparison(...)`.

`build_phase8b_model(..., phase7a_control)` constructs the literal old
`MaskedGraphSSLModel`, with no new heads, state keys, or wrapper output. Fixed
seed tests prove its model-facing state, binding, and loss are bit-exact.

## Parameters and checkpoints

There are four separate projector/predictor heads. For encoder width `D` and
projector width `P`, each family owns two MLPs; its scalar parameter count is
`2 * (2DP + 3P + D)`. The four heads therefore add `266,240` parameters at
the default `D=P=128`, and `1,280` in the bounded `D=P=8` fixture.

New checkpoints continue to use the existing failure-atomic SSL container.
Official Phase 8B checkpoint metadata and its resolved config bind:

- objective registry version and fingerprint;
- complete objective config, weights, and fingerprint;
- target mode and family-global scheduled-view aggregation rule;
- exact new-head parameter count;
- Phase 8B checkpoint-binding contract version;
- concrete model class and execution mode;
- active families and exact active weights;
- masking config and Phase 8A policy-mixture fingerprints.

The explicit Phase 7A transfer validates the complete old model contract,
keys, shapes, and dtypes before mutation. It loads every old encoder/decoder/
projector tensor, leaves every `phase8b_latent_heads.*` tensor at its separate
initialization, and lists both sets plus counts and source checkpoint SHA-256
in `Phase7AToPhase8BTransferReport@1.1.0`. New checkpoint save/load/resume is
strict; an incompatible objective fingerprint rejects before mutation.
Changing objective mode/weight, masking mode/span policy, model contract, or
active weights also rejects before checkpoint application. An old Phase 7A
checkpoint cannot be resumed as Phase 8B; the separately named explicit
transfer API starts a new run instead. Phase 8B engine/report/checkpoint
bindings created before the family-global remediation are incompatible and
reject fail-closed; the null Phase 7A checkpoint path is unchanged.

## Official training, validation, and accounting

The official path supports one-batch, multi-epoch train/validation, best and
last checkpoints, metric journal, fixed epoch-zero validation schedule, and
exact epoch-boundary resume. Validation membership, global seed, policy order,
and per-sample plan coordinates do not depend on validation loader order or
batch partition. `Phase8BObjectiveAccumulator` combines family numerators and
eligible denominators across both policies and CPU batches without retaining
prediction or CUDA tensors. Training batches, validation, epoch aggregates,
best selection, and resume journals all use the same family-global formula. A
zero denominator stays unavailable and never rescales another family.

Every official report records the concrete model class, active families,
resolved policies, registry/objective/masking/mixture fingerprints, eligible
counts, retained-tensor counts, optimizer steps, model forwards, scheduled
policy passes, family-view passes, eligible prediction rows, objective
evaluations, packed D2H counters, and primary/collateral masked entities.
The variants intentionally have different forward-pass counts: control and
single-family modes schedule one pass per batch, while mask-only and equal
weight schedule four. These runs are therefore not compute matched and are
not an effectiveness comparison. Phase 8B.2 owns scientific comparison and
model selection.

## Contracts

Contracts that bind the corrected cross-policy semantics are `1.1.0`:

- objective registry and objective config;
- family loss and combined objective loss;
- model metadata and forward output;
- metric aggregate;
- official engine, masking config, run manifest, and training report;
- Phase 8B checkpoint binding and Phase 7A transfer report;
- bounded comparison report.

Exact-identity eligible entities, prepared objective bindings, latent
prediction rows, and the newly introduced batch-objective aggregate remain
`1.0.0` because their local identity/prediction contracts did not encode the
superseded pass-average rule. Existing Phase 7A/8A and checkpoint-container
contracts are unchanged.

Objective registry fingerprint:
`47a9e38c3a82107956b2225c82fece50d841e3250afaec43d758964207dbadc3`.
No existing Phase 7A, Phase 8A, graph, canonical, cache, target, model-output,
or checkpoint-container contract is revised.

## Bounded comparison

Run:

```bash
.venv/bin/python scripts/accept_phase8b_multilevel_objectives.py \
  --output /tmp/phase8b1-bounded.json --steps 12
```

The fixed CPU protocol uses seed `42`, the accepted six-piece Phase 8A
fixture, four train and two disjoint held-out pieces, hidden/projector width
`8`, AdamW at `0.02`, zero weight decay, 12 steps, fixed batch membership,
and a versioned five-policy schedule. Every variant reconstructs its model
from the same seed. Base-component initialization is identical across all
seven variants; all new-model head initializations are also identical.

This standalone bounded runner predates official-engine integration. It
remains a deterministic mechanics audit, not the supported training entry
point. Its variants use one scheduled forward for the Phase 7A/single-family
cases and four scheduled forwards for mask-only/equal cases. Equal optimizer
step counts do not make those variants compute matched.

The pre-remediation 783,207-byte report and its SHA-256 are invalid evidence
for this contract. Two fresh reports are byte-identical at 976,674 bytes with
report fingerprint
`651c00f33dfcfe52aa2e2e9729f78d9d7a2e2b55ba5fce984b1eebb735374b46`
and file SHA-256
`13e0a5b931fe70bc948ffc2540a8f1f8c2439757b154a6cffc0c47d0c32aa653`.
The unchanged mask-schedule fingerprint is
`dd1527b66dd8ba41b10f66f176bea77c305b2ba772496a6142a7252ec52ad6b7`.
Train initial to final family means were:

| Variant | Train family means, initial -> final |
| --- | --- |
| Phase 7A control | note `0.899657 -> 0.026215`; bar `0.625743 -> 0.071998`; song `0.599986 -> 0.048333` |
| Phase 8A masks, old objectives | note `0.878062 -> 0.036044`; bar `0.633308 -> 0.060953`; song `0.601537 -> 0.037271` |
| onset only | onset `1.079324 -> 0.115920` |
| beat only | beat `1.000747 -> 0.042573` |
| bar only | hierarchy bar `1.119422 -> 0.140495` |
| track only | track `0.905188 -> 0.020764` |
| equal weight | onset `1.079324 -> 0.147759`; beat `1.000747 -> 0.021558`; hierarchy bar `1.133926 -> 0.062153`; track `0.905188 -> 0.024600` |

The artifact also records initial/final held-out family means, exact
denominators, O(D) anti-collapse diagnostics, and finite/non-zero gradient
coverage. Every available train family decreased and every report retained
zero CUDA/prediction tensors. These are deliberately bounded overfit and
mechanics observations, not a model-selection or musical-quality result.

## Acceptance and remaining limits

Tests cover the independent cross-policy arithmetic oracle (`bar=(6+15)/
(3+5)=2.625`, equal-weight total `6.875`, superseded pass average `2.3125`),
bar-weight-once gradients, policy-order invariance, unavailable-family and
single-policy semantics, mask-only old-family global aggregation, all five
policy eligibility rules; exact masked/full row identity;
sample boundaries; empty eligibility; independent toggles; loss arithmetic;
batch partition/order invariance; target/provenance blindness; graph/binding
immutability; stop-gradient behavior; finite non-zero head and encoder
gradients; Phase 7A bit identity; checkpoint transfer/round-trip/atomic
fingerprint rejection; per-family and combined CPU overfit; optional CUDA AMP
through the shared deterministic runtime; and zero retained report tensors.

Official-engine tests additionally invoke the real CLI subprocess for the
null Phase 7A route, every single family, equal weight, and mask-only; verify
weight/fingerprint changes, structured incompatibility before output/optimizer
mutation, one-batch decrease, manifest/checkpoint bindings, fixed validation,
and uninterrupted two epochs versus stop/resume bit-exact state and journal.
Optional onset/equal CUDA+AMP subprocesses use this same official engine and
skip honestly on CPU-only hosts.

CUDA is optional in Phase 8B.1 and an unavailable local CUDA test is an honest
skip, not hardware evidence. Phase 8B.1 does not run full HookTheory/POP909-CL
SSL training, PDMX, Dilemmadata, Phase 8B.2 scientific comparison, Phase 9,
PLL, preference critic, or quality scoring.
