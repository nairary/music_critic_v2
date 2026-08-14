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

Each `Phase8BFamilyLoss@1.0.0` carries:

- summed numerator;
- eligible entity denominator;
- mean loss when available and active;
- availability and explicit unavailable reason;
- configured fixed weight;
- active/inactive state;
- zero-norm count.

A zero denominator is `unavailable` with `mean_loss=None`; it is never a fake
zero. A zero weight is `inactive_zero_weight`: its new head is not executed
and receives no gradient path. The total is

```text
sum_f configured_weight_f * available_mean_f
```

There is no division by active-weight sum and no redistribution when another
family is unavailable. Multi-policy bounded training divides by the fixed
scheduled pass count, not by the number of available families.

Metrics aggregate only fixed-size detached CPU scalar/O(D) sufficient state.
They retain neither prediction tensors nor CUDA tensors.

## Configuration and public API

Hydra group `phase8b_objective` provides:

- `phase7a_control`;
- `onset_only`;
- `beat_only`;
- `bar_only`;
- `track_only`;
- `multilevel_equal_weight`.

The unchanged Phase 7A root defaults to `phase8b_objective=null`; select a mode
with `+phase8b_objective=<mode>`. Every resolved objective config binds the
registry fingerprint and has its own canonical fingerprint. Explicit Hydra
weight overrides are materialized with `Phase8BObjectiveConfig.from_hydra()`;
their resolved values, including zero-valued toggles, enter that fingerprint.

The principal APIs are:

- `Phase8BObjectiveConfig.for_mode(...)`;
- `prepare_phase8b_objective_binding(...)`;
- `Phase8BMultilevelSSLModel.forward_multilevel(...)`;
- `build_phase8b_model(...)`;
- `build_phase8b_model_from_config(...)`;
- `Phase8BObjectiveAccumulator`;
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
Additive model metadata binds:

- objective registry version and fingerprint;
- complete objective config, weights, and fingerprint;
- target mode and fixed aggregation rule;
- exact new-head parameter count.

The explicit Phase 7A transfer validates the complete old model contract,
keys, shapes, and dtypes before mutation. It loads every old encoder/decoder/
projector tensor, leaves every `phase8b_latent_heads.*` tensor at its separate
initialization, and lists both sets plus counts and source checkpoint SHA-256
in `Phase7AToPhase8BTransferReport@1.0.0`. New checkpoint save/load/resume is
strict; an incompatible objective fingerprint rejects before mutation.

## Contracts

All new contracts begin at `1.0.0`:

- objective registry and objective config;
- eligible entities and prepared objective binding;
- family loss and combined objective loss;
- latent prediction, model metadata, and forward output;
- metric aggregate;
- Phase 8B checkpoint binding and Phase 7A transfer report;
- bounded comparison report.

Objective registry fingerprint:
`39af7500c6cee09d5d84c73f3968572eb5408e557fda0c9b094cf6e4cc660b7e`.
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

The complete report fingerprint is
`a6c94fb685dd3116b090e64ef0f777f78519df2bd7c5b73373d19624c45d9470`;
its mask-schedule fingerprint is
`dd1527b66dd8ba41b10f66f176bea77c305b2ba772496a6142a7252ec52ad6b7`.
Two fresh serialized reports were byte-identical at 783,207 bytes and file
SHA-256
`4417a45921971af272c47c3f087abf8988f53ad6df4c0eab1158a28f8c380f4e`.
Train initial to final family means were:

| Variant | Train family means, initial -> final |
| --- | --- |
| Phase 7A control | note `0.899657 -> 0.026215`; bar `0.625743 -> 0.071998`; song `0.599986 -> 0.048333` |
| Phase 8A masks, old objectives | note `0.878062 -> 0.036332`; bar `0.633308 -> 0.060867`; song `0.601537 -> 0.037862` |
| onset only | onset `1.079324 -> 0.115920` |
| beat only | beat `1.000747 -> 0.042573` |
| bar only | hierarchy bar `1.119422 -> 0.140495` |
| track only | track `0.905188 -> 0.020764` |
| equal weight | onset `1.079324 -> 0.148207`; beat `1.000747 -> 0.014227`; hierarchy bar `1.133926 -> 0.062851`; track `0.905188 -> 0.034363` |

The artifact also records initial/final held-out family means, exact
denominators, O(D) anti-collapse diagnostics, and finite/non-zero gradient
coverage. Every available train family decreased and every report retained
zero CUDA/prediction tensors. These are deliberately bounded overfit and
mechanics observations, not a model-selection or musical-quality result.

## Acceptance and remaining limits

Tests cover all five policy eligibility rules; exact masked/full row identity;
sample boundaries; empty eligibility; independent toggles; loss arithmetic;
batch partition/order invariance; target/provenance blindness; graph/binding
immutability; stop-gradient behavior; finite non-zero head and encoder
gradients; Phase 7A bit identity; checkpoint transfer/round-trip/atomic
fingerprint rejection; per-family and combined CPU overfit; optional CUDA AMP
through the shared deterministic runtime; and zero retained report tensors.

CUDA is optional in Phase 8B.1 and an unavailable local CUDA test is an honest
skip, not hardware evidence. Phase 8B.1 does not run full HookTheory/POP909-CL
SSL training, PDMX, Dilemmadata, Phase 8B.2 scientific comparison, Phase 9,
PLL, preference critic, or quality scoring.
