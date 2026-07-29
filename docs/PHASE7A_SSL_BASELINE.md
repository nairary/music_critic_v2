# Phase 7A deterministic masked-graph SSL baseline

Status: **IMPLEMENTED ON DRAFT PR #15; BOUNDED ACCEPTANCE COMPLETE**.

Phase 7A adds the first trainable self-supervised objective over the existing
raw-only PyG graph. It is GraphMAE2-inspired, not a faithful reproduction of
GraphMAE2. The phase proves deterministic masking, leakage-safe model plumbing,
representation reconstruction, latent prediction, checkpoint/resume, and
encoder transfer on bounded pre-PDMX data. It does not establish full-scale SSL
effectiveness or any improvement in musical quality.

## Scientific boundary

Phase 7A implements representation learning, not a probabilistic music model.
Its losses and diagnostics are not:

- masked-note or pitch-set likelihood;
- perplexity or pseudo-log-likelihood (PLL);
- a critic, reward, preference, aesthetic, or quality score;
- evidence that SSL improves the Phase 6 supervised baseline;
- evidence of PDMX-scale effectiveness.

Hierarchical bar/span/track masking belongs to Phase 8. PDMX projection and a
full-scale rerun of accepted SSL objectives belong to Phase 10. A normalized
probabilistic decoder and deterministic PLL protocol require a separate design
gate and ablation.

## Unchanged raw-data contract

Phase 7A does not change canonical schema, raw graph features or topology,
adapters, target ontology or encoding, split manifests, corpus indices, cache
keys, Phase 6 supervised outputs, or Phase 6 checkpoints. A mask is an
immutable model-side sidecar. Raw `HeteroData`/`Batch` stores are not mutated,
serialized, or cached as masked graphs, so the source graph fingerprint remains
unchanged.

`SSLBatch` exposes only:

- `raw_graph_batch`;
- `dataset_ids`;
- `piece_ids`;
- aggregate sample, node, and edge counts.

It contains no target, annotation, provenance, diagnostic, source-group, or
lineage sidecar. The bounded compatibility adapter does not inspect target
contents. Production-cache execution instead uses `IndexedSSLRawDataset` and
`collate_ssl_samples`: each record is read with `load_cached_piece`, converted
with `build_raw_graph`, fingerprint-checked, and collated directly into an
`SSLBatch`. No harmonic target bundle is projected, validated, or passed
through that production SSL path. Existing group-safe training membership,
fixed no-replacement validation membership, index/cache/split paths, and their
fingerprints remain authoritative. Device transfer deep-copies the PyG batch
and moves only tensor attributes.

## Feature-dependency audit

SSL maskable-field registry `1.0.0` resolves semantic names through raw feature
registry `1.0.0`; it contains no hard-coded column indices. Its fingerprint is:

```text
97836b2adb610529994ae609e89913eb6b21ad0f07d4bf695c911251d5f8ac85
```

The only Phase 7A group is `note_pitch_group`.

| Role | Node | Kind | Field | Availability hidden |
|---|---|---|---|---|
| primary reconstruction target | `note` | categorical | `pitch` | yes |
| primary reconstruction target | `note` | categorical | `pitch_class` | yes |
| primary reconstruction target | `note` | categorical | `octave` | yes |
| primary reconstruction target | `note` | continuous | `track_relative_pitch` | yes |
| collateral only, unselected peers | `note` | continuous | `track_relative_pitch` | yes |
| collateral only | `track` | continuous | `mean_pitch` | yes |
| collateral only | `track` | continuous | `pitch_std` | yes |
| collateral only | `track` | continuous | `min_pitch` | yes |
| collateral only | `track` | continuous | `max_pitch` | yes |

For every selected note, the plan identifies its owner track through the raw
`track contains_note note` relation. Every unselected note peer in any affected
owner track receives a collateral mask for `track_relative_pitch` and its
availability contribution. All four aggregate pitch fields and their
availability contributions are also masked on every affected owner track.
Together these close shortcuts through track-relative peer values and track
statistics, including the exact singleton-track shortcut. Neither peer-note
nor owner-track collateral fields are reconstruction targets.
The corresponding `MaskPlan` reasons are
`owner_track_peer_relative_pitch` and `owner_track_pitch_statistics`.

Duration, velocity, timing, topology, complete bars, and complete tracks are
not masked in Phase 7A.

## Deterministic `MaskPlan`

`MaskPlan` and mask policy contracts are `1.0.0`. The policy is
`uniform_note_without_replacement`.

Each plan binds:

- `(dataset_id, piece_id)` for deterministic seed derivation only;
- train or validation stage;
- canonical epoch, encoder-view index, and global seed;
- selected note-local indices and `note_pitch_group`;
- explicit collateral unselected-peer-note and owner-track masks;
- requested/maskable/selected counts and realized rate;
- a portable SHA-256-derived 64-bit seed;
- a deterministic SHA-256 fingerprint.

Sampling uses no Python `hash()` and no global RNG. Per-index SHA-256 keys form
a stable permutation; selection has no replacement. A positive rate selects at
least one note whenever a sample has a note, rate zero selects none, and rate
one selects all. Train masks advance deterministically by epoch when a
different subset is possible. Validation canonicalizes every requested epoch
to zero. Plans are therefore independent of batch order, DataLoader worker
count, targets, annotations, and dataset-specific labels.
The model regenerates canonical encoder-view-zero plans from only the raw
batch, identities, seed, stage, epoch, and configured rate. A supplied plan is
accepted only when its complete tuple is exactly equal to that canonical
result; a validly fingerprinted alternate selection or nonzero encoder view
fails closed before encoding.

## Model-side overlay and leakage boundary

The raw feature encoder accepts an optional duck-typed overlay without
registering SSL parameters or importing `music_critic.ssl`. For each named
feature it supplies the value and availability contributions separately. On a
masked row, the bound Phase 7A overlay:

1. replaces the value contribution with an SSL-model-owned learnable mask
   token;
2. replaces the availability contribution with zeros;
3. leaves every unmasked contribution object unchanged.

The encoder then adds value followed by availability in the existing order.
When the overlay is absent, the original two additions remain unchanged and
the Phase 6 no-mask path and state-dict surface are preserved.

Consequently, with a fixed plan, changing a masked raw value can change the
full-view target and reconstruction loss but cannot change the online masked
feature contribution. The same holds for collateral
`track_relative_pitch`/availability slots on unselected peers and aggregate
pitch-statistic/availability slots on affected owner tracks; these collateral
slots do not become target rows. Changing a truly unmasked value can still
change online representations. The mask indicator never contains the original
value.

## Online and target paths

The target mode is exactly:

```text
shared_stop_gradient_full_view
```

The logical flow is:

```text
full raw-only graph
  -> shared hierarchical encoder in eval/no-grad mode
  -> detached note, bar, and song representation targets

same full raw-only graph
  -> deterministic feature overlay
  -> online hierarchical encoder
  -> selected note latents + online structural context
  -> decoder re-mask views -> representation decoder
  -> all bar/song latents -> projector/predictor latent objectives
```

The target path uses the same hierarchical encoder parameters, receives the
complete raw view, disables training-time dropout for target extraction, and
returns detached tensors. There is no EMA target encoder in Phase 7A. Gradients
flow through the online encoder, feature mask token, representation decoder,
bar/song projectors, and bar/song predictors, but not through target outputs.
Supervised task and visible-feature reconstruction heads are not Phase 7A
objectives.

Online optimization uses the configured dropout and the globally seeded,
checkpointed RNG state. The one-batch acceptance compares its initial and
final loss measurements in `eval_no_grad` mode, so that the reported trajectory
does not compare different dropout masks; the intervening optimizer steps
remain ordinary train-mode forwards.

## Decoder re-masking

Decoder re-mask contract `1.0.0` operates only on the compact latent rows
selected by the encoder plans. It never receives raw values. Each decoder view
derives an independent SHA-256 seed from the encoder-plan fingerprint, stable
seed, and decoder-view index, then performs deterministic Bernoulli selection
over compact row positions.

Decoder context mode is exactly
`online_owner_track_bar_song_temporal_neighbors`. For each selected row it
combines online owner-track, available owner-bar, song, and previous/next
in-track note representations. This context is derived only from the masked
online path; it contains neither full-view targets nor original masked raw
slots. The contextual projection is added after latent re-masking, so a fully
re-masked view does not reduce every prediction to a function of one shared
constant mask token.

Supported named modes are:

| Mode | `decoder_views` | `decoder_remask_prob` |
|---|---:|---:|
| simple masking baseline | 1 | 0.00 |
| Phase 7A main preset | 3 | 0.20 |

Both modes recover the same detached full-view targets. Multi-view loss is the
sum over every view/row divided by the corresponding combined count. Phase 7A
makes no superiority claim for the multi-view preset.

## Objective and diagnostics

The representation-loss formula is versioned as `one_minus_cosine`:

```text
loss_sum = sum_i (1 - cosine(prediction_i, stopgrad(target_i), eps=1e-8))
loss_mean = loss_sum / row_count
```

Reduction is `sum_count_mean`. Every component reports numerator,
denominator, mean, zero-norm count, and an unavailable reason. Zero-vector rows
remain in the numerator and denominator; cosine uses `eps=1e-8` and such rows
are counted rather than silently dropped. Zero eligible rows produce an
explicit `no_eligible_rows` state, not a fabricated scientific zero or NaN.

The independent components are:

- selected-note multi-view representation reconstruction;
- bar-level projector/predictor latent loss;
- song-level projector/predictor latent loss.

The total is:

```text
L_ssl = w_note * L_note + w_bar * L_bar + w_song * L_song
```

Weights are finite and non-negative, with at least one positive. A positively
weighted unavailable component makes the total unavailable; its weight is not
silently redistributed.

Anti-collapse diagnostics report row count, embedding dimension, target and
prediction embedding variance, mean norm, zero-norm count, and mean
off-diagonal cosine. Pairwise cosine uses normalized-vector sums and exact
sufficient statistics in `O(ND)` time; the final aggregate is `O(D)`, and the
implementation does not construct a production `N x N` matrix. These
diagnostics are not quality scores.

## Contracts and public APIs

All new Phase 7A contracts begin at `1.0.0`:

| Contract family | Main API/module |
|---|---|
| SSL, MaskPlan, mask policy, feature overlay | `ssl.contracts`, `ssl.masking`, `ssl.views` |
| maskable-field registry | `ssl.field_registry` |
| decoder re-mask and representation decoder | `ssl.decoder` |
| representation/multi-view loss and diagnostics | `ssl.objective` |
| SSL model, output, and representation target | `ssl.model` |
| SSL checkpoint and epoch journal | `ssl.checkpoint` |
| pretrained encoder export | `ssl.transfer` |
| run manifest, report, metric row, performance row | `ssl.engine` |

The principal APIs are `build_mask_plan`, `build_batched_mask_plans`,
`build_feature_mask_overlay`, `MaskedGraphSSLModel`, `build_ssl_model`,
`representation_cosine_loss`, `anti_collapse_diagnostics`,
`save_ssl_checkpoint`, `load_ssl_checkpoint`,
`export_pretrained_encoder_state`, `load_pretrained_encoder_state`,
`IndexedSSLRawDataset`, `collate_ssl_samples`, `build_ssl_data_runtime`, and
`run_ssl_training`.

## Training, checkpoint, resume, and transfer

The separate Hydra root supports:

```bash
python -m music_critic.ssl.run \
  experiment=one_batch \
  model=hierarchical \
  data=bounded \
  device=cpu

python -m music_critic.ssl.run \
  experiment=pretrain \
  model=hierarchical \
  data=mixed \
  device=cpu
```

The first command is bounded overfit/plumbing evidence. The second uses the
same versioned production cache/index/split paths as Phase 6C, but its
Phase 7A-specific dataset/collator reads canonical pieces and builds only raw
graphs; it never projects supervised targets. Reading a production cache is
not itself a claim that production or full-corpus SSL training occurred, and a
full corpus run is not part of Phase 7A acceptance.

Checkpoints bind the complete SSL/model contract, maskable-field registry
version and fingerprint, resolved-config fingerprint, data index/split/
composition/fixed-validation fingerprints, model/optimizer/scheduler/scaler
state, RNG state, and an ordered epoch journal. Saves use atomic same-directory
replacement. Loads validate first and apply model, optimizer, scheduler,
scaler, and RNG state failure-atomically. Resume is exact only at an epoch
boundary; mid-epoch resume is unsupported.

Encoder export contains only these hierarchical representation prefixes:

- `local_baseline.encoder.`;
- `context_encoder.pooling.`;
- `context_encoder.transformer.`;
- `context_encoder.fusion.`.

Loading is strict and failure-atomic. Supervised task/reconstruction heads are
reported as untouched and are not overwritten.

Reports separate deterministic metric/checkpoint evidence from
nondeterministic stage timing. They include total and component losses,
per-decoder-view loss, requested/realized masking and counts, anti-collapse
diagnostics, learning rates, sample/node/edge counts, unavailable batches,
gradient coverage, device evidence, and bounded retained-memory counters.
Report provenance explicitly distinguishes `evidence_kind`,
`data_source_kind`, `production_cache_data_used`, the one-batch
`run_scope=one_batch_plumbing` boundary, and the
`production_ssl_training_performed` and
`full_corpus_ssl_training_performed` claims. Production training is true only
when a non-bounded cache source actually completed optimizer steps. Full-corpus
coverage is false for bounded/one-batch execution and otherwise remains
explicitly unavailable unless identity coverage is tracked; cache access alone
does not imply either claim.

## First supervised baseline context and backlog

The first real supervised baseline is context for later ablations, not evidence
for or against Phase 7A:

- strong signal for HookTheory tonic and scale degree;
- strong signal for POP909-CL root and bass;
- weak or collapsed signal for the remaining heads;
- HookTheory multilabel heads at threshold `0.5` produce all-negative output
  and `F1=0`;
- POP909-CL validation evidence is limited to 18 independent pieces;
- scientific evaluation hardening remains in the backlog before final
  ablations, but does not block Phase 7A;
- the ambiguous field `test_not_used_for_checkpoint_selection` remains a
  registered evaluation-backlog item.

No downstream improvement is claimed from these observations.

## Final acceptance evidence

The bounded acceptance used the default 40-step CPU command with
`experiment=one_batch model=hierarchical data=bounded device=cpu`. Initial and
final losses were measured in `eval_no_grad` mode. Outputs were written below
`/tmp`, not to the repository, and no production cache was read.

| Evidence | Final value |
|---|---|
| Branch and implementation commits | `phase/7a-graphmae2-ssl-baseline`; start `07bee14`; implementation `125252b54d51e4644ed5848f1077d163df0c0a12` |
| Contract versions/fingerprints emitted by final run | all new Phase 7A contracts `1.0.0`; maskable registry `97836b2adb610529994ae609e89913eb6b21ad0f07d4bf695c911251d5f8ac85`; model contract `44c8ba546608bbba4accab65d7c5733db15d86965709ad8c78dae16221a2b296`; resolved config `50f93cb2fa1b5e180dfd7e121ee5dff5c39b65e58cdd9be440f6f824c32fa225`; data binding `dc2861e59dffa378da10db4e6c44af2f3538ce5eb06f80ab5ffb832efc139c01` |
| Samples, nodes, and edges | 3 samples; 28 nodes; 98 directed edges |
| Primary and collateral masked counts | requested `0.30`; realized `1.0` for three singleton-note fixtures; 3 primary notes; 0 peer-note collateral rows; 3 owner-track collateral rows. Multi-note peer collateral is exercised separately by leakage tests. |
| Initial/final total SSL loss | `3.0867743492126465` → `0.001336899003945291` |
| Initial/final note reconstruction loss | `0.9426748752593994` → `0.0005422499380074441` |
| Initial/final bar latent loss | `1.051714301109314` → `0.0003143151698168367` |
| Initial/final song latent loss | `1.0923850536346436` → `0.0004803339543286711` |
| Gradient coverage | finite/nonzero gradients reached the feature-mask token and every required online group: local encoder, hierarchy pooling, Transformer, fusion, decoder, and both bar/song projector-predictors; all 81 supervised-head parameter tensors remained gradient-free |
| Masked-value leakage mutation evidence | PASS: fixed plan and raw stores remained bit-exact; masked-value mutation left online embeddings/predictions bit-exact while changing the full-view target and reconstruction loss |
| Deterministic repeat | PASS: mask plans, online embeddings, decoder predictions, and loss were bit-exact |
| Checkpoint reload and epoch-resume evidence | one-batch reload bit-exact; exact uninterrupted/resumed state and metrics passed with dropout `0.2`, cosine scheduling, and CPU AMP; atomic save/load, corrupt-journal, crash-window recovery, and rejected-resume RNG rollback tests passed |
| Encoder-transfer evidence | export `1.0.0`; 470 parameter tensors loaded; 81 supervised-head tensors untouched and bit-exact |
| CPU timing and retained memory | 6.643101028003002 s total; 0.012225073998706648 s transfer, 3.119837647991517 s forward, 2.3773286959985853 s backward; one live batch, zero retained predictions, two retained metric rows |
| CUDA/VRAM evidence or honest unavailability | CUDA unavailable; CUDA+AMP test skipped explicitly; no device name or VRAM values fabricated |
| Focused and complete test results | focused SSL: 79 passed, 1 CUDA skip; full default suite: 911 passed, 20 skipped; model/graph/dataset regression slice: 265 passed, 1 skipped plus 2 isolated worker tests passed; `compileall`, `git diff --check`, and implementation `git show --check` passed |
| Required GitHub CI | PASS: `Required test suite` run #85 completed successfully for implementation commit `125252b54d51e4644ed5848f1077d163df0c0a12`; the final documentation-head result is also recorded in the draft PR evidence comment |

Production SSL training was not authorized as Phase 7A acceptance. Phase 8 was
not started, PDMX was not added, PLL was not implemented, and no critic or
quality score was implemented.
