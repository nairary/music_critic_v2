# Phase 7A deterministic masked-graph SSL baseline

Status: **MERGED IN PR #15 AT `a850207`; POST-MERGE CUDA
DEVICE/EVIDENCE HOTFIX IN PROGRESS; PHASE 8 NOT STARTED**.

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

## Post-merge CUDA device-canonicalization hotfix

Independent execution on an RTX 3090 after the Phase 7A merge produced
`157 passed, 2 failed, 8 warnings` for `python -m pytest -q tests/ssl`.
Both failures ended at the unchanged strict category
`ssl.data.device_transfer_tensor_mismatch`: the prepared CUDA+AMP path and
the bounded CUDA+AMP smoke path received abstract `torch.device("cuda")`,
while PyTorch placed tensors on concrete `cuda:0`. Those device objects do
not compare equal because the former has no index.

The first independent rerun of draft PR #17 at `fb54e85` confirmed that this
root cause was fixed: both `resolve_runtime_device("cuda")` and exact
graph/prepared-binding assertions resolved to `cuda:0`, and
`ssl.data.device_transfer_tensor_mismatch` disappeared. That run produced
`165 passed, 2 failed, 1 skipped` for `tests/ssl` and
`7 passed, 2 failed, 1 skipped` for the training CUDA tests. The remaining
failures were distinct remediation items: AMP decoder predictions were FP16
against FP32 detached targets; explicit `cuda:N` was not range-checked or
accepted consistently by engines; a velocity perturbation test modified an
unavailable placeholder; and a resume assertion compared JSON lists directly
with in-memory tuples.

The next independent RTX 3090 run at exact head `145ee10` confirmed those
remediations: full `tests/ssl` produced
`195 passed, 1 failed, 1 skipped`, the training CUDA suite produced
`15 passed, 1 skipped`, and the prepared CUDA AMP path passed. The sole failure
was `test_bounded_cuda_amp_smoke`. Its strict mutation facts were all green:
raw stores remained bit-exact, runtime-source binding passed, MaskPlan and
prepared-binding fingerprint remained fixed, masked-online embeddings and
predictions remained bit-exact, the full-view target and reconstruction loss
changed, metrics were finite, and target distance was positive. The only false
field was the former `positive_margin` gate:
`correct_minus_mutated_margin=-0.04540175199508667` from an FP16 prediction
with FP16-derived floor `0.0078125`. This is intermediate-head execution
evidence, not leakage and not acceptance evidence for the future final head.

The hotfix resolves every runtime request before transfer. CPU, including an
indexed CPU spelling, canonicalizes to `cpu`; bare CUDA resolves through
`torch.cuda.current_device()`; and explicit `cuda:N` preserves `N` only when
`0 <= N < torch.cuda.device_count()`. The current device is checked against
the same visible count. CUDA requests fail structurally when CUDA is
unavailable, while an invisible explicit or current index fails before
transfer as `runtime.device.cuda_index_out_of_range`. Validation remains
exact: `cuda:0` and `cuda:1` are distinct, and a tensor on the wrong index is
rejected. Training, SSL, and evaluation accept `cpu`, `cuda`, `cuda:N`, and
`auto` through the shared resolver; AMP eligibility is based on the resolved
CUDA device type. No resolver or transfer check calls `.cpu()`, `.item()`,
`.tolist()`, reads tensor values, allocates a validation tensor, or introduces
graph-sized host materialization.

The same resolver governs the SSL graph, prepared selected-index sidecar,
Phase 6C graph and target transfer, evaluation runtime, and direct evaluation
checkpoint model placement. SSL mismatch evidence keeps its stable category
and adds one concrete location plus expected/actual devices:
`global:<attribute>`, `node:<node-type>:<attribute>`,
`edge:<source>|<relation>|<destination>:<attribute>`, or
`binding:<field>`.

Under AMP, any FP16/BF16/FP32 representation prediction-target pair is computed
in FP32 with autocast disabled. Only matching FP64 pairs remain FP64.
Prediction and detached target keep exact shape and concrete-device checks;
the out-of-place prediction cast preserves gradients and the target remains
stop-gradient. Empty numerators, zero-vector policy, multi-view reduction, the
combined note/bar/song objective, and immediate/streaming anti-collapse
diagnostics use the same compute-dtype rule.

Mutation reporting now separates two independent, versioned, fingerprinted
objects. `no_leakage_mutation_evidence@1.0.0` accepts only mutation
applicability, exact raw/runtime-source/plan/binding invariants, strict
`torch.equal` online embeddings and predictions, a changed hidden target, and
finite metrics. `pitch_sensitive_reconstruction_evidence@1.0.0` accepts an
applicable mutation that changes the hidden target and reconstruction loss,
has positive target distance, and keeps metrics finite. Correct-target
preference is recorded separately through the two cosines, signed margin,
`observed|not_observed` status, and
`preference_is_acceptance_criterion=false`. Cosine, L2, margin, and floors use
an autocast-disabled FP32 diagnostic kernel; `margin_floor` is
`8 * finfo(float32).eps`, never a device/source-dtype adjustment intended to
change the margin sign.

Patch versions are runtime resolution `1.0.1`, device transfer `1.0.2`,
representation loss `1.0.1`, multi-view representation loss `1.0.1`, SSL
objective `1.0.1`, anti-collapse diagnostics `1.1.1`, and umbrella SSL
`1.2.2`. SSL training report advances from `1.2.1` to `1.2.2` because its
serialized evidence schema is split. The two new evidence subcontracts begin
at `1.0.0`. Prepared binding remains `1.1.0`; SSL model/output,
checkpoint/journal/metric-row, run-manifest/performance-row, masking, decoder,
registry, fixture, and encoder-export versions remain unchanged. The umbrella
SSL bump changes newly generated model-contract and checkpoint-binding
fingerprints. Existing Phase 7A hashes below remain historical `1.2.0`
evidence and are not rewritten. Exact checkpoint metadata means historical
bounded SSL `1.2.0` checkpoints are not resumable under the remediated umbrella
contract; this hotfix adds no migration.

The velocity CUDA test now mutates only available sample-zero velocity values,
preserves unavailable placeholders bit-exactly, and reruns raw-graph
validation before checking model isolation. The resume CUDA test separately
asserts membership fingerprint/count/limit evidence and compares ordered
identities through canonical JSON normalization. Production raw validation,
placeholder policy, membership selection/fingerprint, resume binding, and
byte-identical `metrics.jsonl` evidence are unchanged.

CPU verification cannot establish CUDA correctness. The hotfix draft must
remain unmerged until the exact final commit passes Required CI and the
independent RTX 3090 SSL, training CUDA, prepared-binding AMP, and bounded AMP
acceptance commands pass with recorded passed/skipped counts and bounded-smoke
peak allocated/reserved VRAM.

On the CPU-only development host, the pre-fix regression first failed exactly
on abstract-versus-concrete CUDA resolution. Final remediation verification
passed focused runtime/config/device checks
`73 passed, 1 skipped, 2 warnings`, focused objective/diagnostic/CUDA
collection `53 passed, 5 skipped, 2 warnings`, the complete SSL suite
`191 passed, 6 skipped, 8 warnings`, related training/evaluation device checks
`60 passed, 6 skipped, 2 warnings`, resume/checkpoint checks
`33 passed, 2 warnings`, and the complete default suite
`1059 passed, 27 skipped, 10 warnings`. Repository/import plus deterministic
membership checks passed `12 passed, 2 warnings`; compileall and
`git diff --check` passed. CUDA-dependent cases account for the relevant
skips, so these counts are CPU regression evidence rather than exact-final RTX
evidence.

The subsequent evidence-semantics remediation passed its focused truth-table,
fingerprint, FP32-diagnostic, checkpoint, and optional-CUDA collection with
`18 passed, 1 skipped, 2 warnings`; the complete SSL suite with
`206 passed, 6 skipped, 8 warnings`; related training/evaluation CUDA-device
tests with `41 passed, 6 skipped, 2 warnings`; the complete repository with
`1074 passed, 27 skipped, 10 warnings`; and the explicit deterministic
repository/resume audit with `12 passed, 2 warnings`. Compileall and
`git diff --check` passed. These remain CPU/skip evidence; an independent RTX
3090 rerun is required on the exact pushed final SHA.

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

Production train and validation construct canonical encoder-view-zero plans
from the fully validated CPU `SSLBatch` before device transfer. Prepared
binding contract `1.1.0` binds ordered identities, node/edge structure,
note-track ownership, stage, canonical epoch, seed, rate, plan fingerprints,
overlay fingerprint, and selected global indices. Its constructor regenerates
the canonical plans from the CPU graph before signing the binding, so a
validly fingerprinted alternate selection fails closed even if supplied
through the internal constructor.

The binding also holds a complete process-local runtime descriptor for the
validated model-facing input. It binds the graph and every global/node/edge
store by strong object reference, object identity, and type; ordered
`node_types`/`edge_types`; and the exact attribute set of every store. The
current schema contains 65 graph tensors: global `raw_only`; each mandatory
node store's `x_cat`, `x_cat_available`, `x_cont`, `x_cont_available`,
`batch`, and `ptr`; `candidate_slot` on beat/onset; and every mandatory
relation's `edge_index`. Each tensor is held by a strong reference and attested
by object identity, `_version`, shape, dtype, and device. The compact selected
note-index tensor has the same separate evidence. A typed hash binds all
non-tensor metadata, including schema/registry/builder values, feature-name
collections, `num_nodes`, and `entity_id` collections.

Device transfer first revalidates the complete source surface, deep-copies the
stores, moves tensor attributes, compares the transferred metadata, shape,
dtype, and device surface, and then replaces the CPU descriptor with a fresh
descriptor over the moved graph. Post-prepare mutation, tensor replacement,
attribute injection/deletion, a foreign graph, or a forged binding therefore
fails before encoder computation. The private object identities, strong
references, version counters, device metadata, HMACs, and capability tokens
never enter `to_dict()`, deterministic binding fingerprints, checkpoints,
reports, graph serialization, or caches.

There is no free boolean validation bypass. Ordinary public Phase 6
`forward`/`encode` APIs always execute the existing full raw-graph validator.
Only the private prepared path accepts an opaque, process-local HMAC-backed
token issued for one exact batch, graph, binding, runtime attestation, and mask
rate. The full-target and masked-online paths independently issue and re-attest
that token immediately before their encoder execution. CPU and CUDA share this
path without graph-sized `.cpu()`, `.tolist()`, `.item()`, or per-note SHA-256
work after transfer. CPU plan preparation, transfer, forward, and backward
remain separate timing fields.

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

Anti-collapse diagnostics contract `1.1.0` reports row count, embedding
dimension, target and prediction embedding variance, mean norm, exact
zero-norm count, and global mean off-diagonal cosine separately for note, bar,
and song. Mergeable float64 sufficient statistics retain no embeddings or
per-batch prediction history: state is `O(D)` per family/side and no production
`N x N` matrix is constructed. Fewer than two rows produce structured
unavailability. Dense-oracle, merge, batch-partition, batch-order, and worker
tests cover emitted `anti_collapse_aggregate`; the rejected
`anti_collapse_last_batch` field is not used.

`O(D)` describes retained accumulator state only. The current
`_StreamingEmbeddingStatistics.from_values` implementation materializes a
float64 `N x D` `values64` working tensor and a normalized `N x D` temporary
for each input batch. No `O(D)` peak-temporary-memory guarantee is made. Their
real CUDA cost has not been measured; production SSL on an RTX 3090 requires a
separate profiler/optimization gate before training is authorized.

Bounded non-collapse acceptance requires finite initial/final held-out values,
zero zero-norm counts, variance and mean norm above dtype-aware numerical
floors, and off-diagonal cosine below the near-identical gate. It is a
mechanics diagnostic, not a quality score.

## Contracts and public APIs

Remediation advances only contracts whose public meaning or numerical
semantics changed:

| Contract family | Version |
|---|---:|
| umbrella SSL | `1.2.2` |
| SSL model/output | `1.2.0` |
| anti-collapse diagnostics | `1.1.1` |
| checkpoint, epoch journal, metric row | `1.2.0` |
| run manifest and performance row | `1.2.0` |
| training report | `1.2.2` |
| no-leakage mutation evidence | `1.0.0` |
| pitch-sensitive reconstruction evidence | `1.0.0` |
| runtime-device resolution | `1.0.1` |
| shared device transfer | `1.0.2` |
| prepared MaskPlan binding | `1.1.0` |
| MaskPlan/policy and feature overlay | `1.0.0` |
| bounded fixture and pitch-mutation policy | `1.0.0` |
| maskable-field registry | `1.0.0` |
| decoder/remask and representation target | `1.0.0` |
| representation loss, multi-view loss, SSL objective | `1.0.1` |
| pretrained encoder export | `1.0.0` |

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
per-decoder-view loss, requested/realized masking and counts, aggregate
anti-collapse diagnostics, learning rates, sample/node/edge counts,
unavailable batches, gradient coverage, device evidence, and explicitly
scoped `O(D)` diagnostic-accumulator retained state.
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

The remediation starts from `791ef19b1dbd7c26b7a2ef87f36d4ee5b08391a6`.
The complete ordered compare list before this documentation commit is:

1. `ab9477888bc39312e8501bbf18685f45cf1d5630` — acceptance remediation;
2. `64f63997141b9a2e5eb9c718af992e62b01f5b9f` — remediation evidence; this
   documentation commit was the commit omitted from the earlier three-commit
   final-comment list;
3. `ba458697599b03395b4a720888e7e7ce9d99c3bb` — cross-environment pitch
   acceptance stabilization;
4. `3713ee4b5d51f5511699633784996a153fd86e07` — documentation-only post-CI
   evidence correction;
5. `c0f0478be880a8e43415d0716d78cadc573a8025` — prepared-input security
   attestation and contract remediation;
6. `38ae6ccbee4d089171e2d3e58f38c8d67b9baa26` — test-only completion of the
   structured mutation matrix.

The final documentation commit is added to this ordered list in PR evidence
after publication. Outputs were written only below `/tmp`; no production cache
was read.

### Fixture and masking

| Evidence | Train | Fixed validation |
|---|---:|---:|
| identities | 3 disjoint pieces | 2 disjoint pieces |
| notes / tracks / bars | 48 / 7 / 7 | 36 / 5 / 5 |
| graph nodes / directed edges | 114 / 740 | 83 / 546 |
| requested / realized mask rate | `0.30` / `13/48 = 0.2708333333333333` | `0.30` / `10/36 = 0.2777777777777778` |
| primary / peer-note / owner-track rows | 13 / 35 / 7 | 10 / 26 / 5 |

Fixture fingerprint is
`9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922`;
split `89715a23b35ead69a1a314845414d01c6b56bdfbcc913e931719f17020bbef8d`;
train composition
`218b51f2a212b5158b244bb22f8b28952ec79d8ecf9fc2ff5861dc24b9e770bf`;
validation composition
`5730dfa44b90912cfca10bdacf489800054da8331f6a030e8dd7ab7cb461d7cd`.
Fixed-validation membership is
`eeefb2ef9e34e0221a2d025603d4ac6967d31583db78d625c25e8e850a725353`;
the complete data binding is
`35cc77297b1acf484695ef7f5a7c5fdcd072f013747be7d2a0338643efd776bc`.
Model contract fingerprint is
`7a1ece2b44dc6b52aef6f7c7532238d4716b1a45c38b8ca66957225a24b76774`;
resolved-config fingerprint is
`0667a5cd6f87780fd0bc0affd8bdda06080229cffe11da3d6c1da7069649cd4c`;
the maskable registry remains
`97836b2adb610529994ae609e89913eb6b21ad0f07d4bf695c911251d5f8ac85`.
Train plan fingerprints are
`f07c83364859e4f28b499d821985f9fb20c3be866c4d5e6f4bea237d3e16647c`,
`3b5c90bc0016a528cb840ee9c3a3214e52cbd2d0eafbad2aa6ded52e0729da5d`,
and `42da3df81221b200303fd9184097e59bc7d4b85eca94a26ac7648f14bc120751`;
their prepared binding is
`f400906c311313edc58802aea8283adb7de3b4a1c2d2abfd8b2c28bb8dd36b76`
at train epoch zero.
Validation plans are
`3d53144db3405b3d504d186ae6e6dfa4bf9f154afded82563f6a7575a41459db`
and `3f135a44278feff1d7af514895f924d796988521403b13573266cd2f7af823e8`;
their fixed binding is
`cbf820a5ae2022ce53da05a7d5bb2ef769c13fb618a848a66f40f6c5bd8c7bf9`.

### One-batch plumbing and pitch sensitivity

The default 40-step CPU/no-AMP run used the full 128-dimensional model and
AdamW learning rate `3e-4`. The SSL runner uses this rate when the one-batch
optimizer rate is unset; explicit overrides remain authoritative. Eval/no-grad
total loss changed `3.122128486633301 → 0.04193296656012535`;
note/bar/song components changed respectively
`0.9796208143234253 → 0.020008785650134087`,
`0.9931425452232361 → 0.012395143508911133`, and
`1.1493650674819946 → 0.009529034607112408`.

Pitch mutation contract `1.0.0` uses fixed policy
`midi_axis_reflection_v1` (`pitch -> 127 - pitch`), fingerprint
`55c9c82b10153c21d158fb3287c3c01deea10b2a427b08d1266e1c89cdc32227`.
For the historical merged CPU evidence under the same plans,
`cos(prediction, correct_target)=0.9806331396102905`,
`cos(prediction, mutated_target)=0.9797264933586121`, so the margin is
`+0.0009066462516784668` above dtype floor `9.5367431640625e-7`.
Target L2 distance is `1.1555137634277344`; cosine distance is
`0.005206167697906494`. Actual runtime-source fingerprints matched the
rebuilt canonical sources; original/mutated CPU/device graph stores remained
bit-exact; masked online embeddings/predictions remained bit-exact while the
full-view target and reconstruction loss changed.

That positive historical margin is retained as an observed diagnostic, not a
portable acceptance constant. The report `1.2.2` split accepts the strict
no-leakage invariants and effective target/loss challenge independently of
margin sign; a scientific correct-target-preference claim requires a trained
checkpoint and held-out evaluation rather than this one-batch plumbing run.

This overfit run is only plumbing/pitch-sensitivity evidence. Its final
one-batch embeddings are not the non-collapse acceptance source.

### Held-out trajectory and aggregate non-collapse

The fixed validation baseline was measured with optimizer step count zero.
Three one-batch train epochs then produced:

| Measurement | Train loss | Fixed-validation loss |
|---|---:|---:|
| initial | — | `3.1229397773742678` |
| epoch 0 | `3.137899176978366` | `2.5964468638102214` |
| epoch 1 | `2.6812487155089886` | `2.2769506017367043` |
| epoch 2 | `2.3729584487803255` | `2.0780126730600994` |

`best.pt` is epoch 2, selected only by minimum fixed-validation loss. Initial
and final aggregate diagnostics are:

| Stage/level | rows | target/pred variance | target/pred mean norm | target/pred zero count | target/pred offdiag cosine |
|---|---:|---:|---:|---:|---:|
| initial note | 10 | `0.17252324824920307 / 0.048302809353796286` | `11.313566954223944 / 5.127291991705188` | `0 / 0` | `0.8083027001321981 / 0.7714759246451488` |
| initial bar | 5 | `0.016285389384962688 / 0.028396703350398934` | `6.45909049406117 / 6.503982809362448` | `0 / 0` | `0.9380904694959945 / 0.8939212061215471` |
| initial song | 2 | `0.017167156929002482 / 0.015808369519164787` | `6.490804980887502 / 6.7738515048479675` | `0 / 0` | `0.8961429433180865 / 0.9124933745309369` |
| final note | 10 | `0.11253555792667458 / 0.01523628125833525` | `11.319743739268711 / 6.464376256994916` | `0 / 0` | `0.8750934437723483 / 0.9496680268784783` |
| final bar | 5 | `0.006837738670964865 / 0.005255709954242996` | `7.257334453798848 / 6.677702690592231` | `0 / 0` | `0.9793280490012577 / 0.981249392210251` |
| final song | 2 | `0.011029292593605558 / 0.006030926095875147` | `7.301032834267348 / 7.1842883725925155` | `0 / 0` | `0.9472672009595868 / 0.9700898056740257` |

All finite/non-collapse gates pass. Two fresh overwrite runs produced
identical semantic artifact bytes. SHA-256 values were: resolved config
`554c09dd93245d173580e1861e91486bffae4b765eeb6bbdf2ae3ec1659b800f`,
fingerprints
`484af62d67e999a10582668733f528875d82776de5ecf876d38237f298c1dd05`,
run manifest
`b003cd18b941870c3e7812e47ef1125fa0595f353dc9f628cb9f97315b1f1572`,
initial validation
`92c81aae2a16d1cb96f8e4a951ea06e36abf0373fa5871bdd57c9c41e9ba56f7`,
and metrics journal
`eb0f4b27bbbdf336539ae757c9bc68d56a41d6f63adaefba0e076217389e713a`.
Loaded `last.pt` and `best.pt` states were recursively bit-exact; raw ZIP
container bytes and timing files are intentionally outside that contract.
The two exact-path overwrite runs used
`/tmp/music-critic-v2-phase7a-final-heldout`; their numerical loss trajectory
and initial/final diagnostics were unchanged.

### Timing, transfer, tests, and CI boundary

No current CPU/GPU speed or memory-performance claim is made. Encoder export
continues to load 470 tensors while leaving all 81 supervised-head tensors
bit-exact.

Post-matrix local checks are: focused prepared-binding/model/masking/bounded
leakage `95 passed, 1 skipped, 2 warnings`; complete SSL
`157 passed, 2 skipped, 2 warnings`. The structured mutation matrix contains
28 cases, including onset `candidate_slot` and split-like attribute injection.
It covers in-place feature/availability/beat-and-onset-candidate/edge/ptr/batch/
`raw_only` changes; same-metadata and changed-shape/dtype replacements;
global/node/edge attribute addition or removal; target/theory/split/provenance/
diagnostic fields; unknown node/edge stores; and entity-ID, feature-name,
schema, and `num_nodes` metadata. Separate cases reject a foreign graph and an
internally forged binding. Every failure asserts `SSLContractError`, zero
encoder calls, and unchanged graph, binding, and model snapshots.
Source-identical Phase 6 model/graph regressions passed
`146 passed, 1 skipped`; checkpoint/resume/transfer passed `19 passed`. The
final head-relative complete suite passed
`989 passed, 21 skipped, 2 warnings in 84.16s`. The automated held-out rerun
passed once, followed by two byte-identical exact-path overwrites.
`compileall`, `git diff --check`, and `git show --check` pass. CUDA is
unavailable locally, so CUDA acceptance is an explicit skip and no CUDA/VRAM
number is fabricated. Required GitHub CI is historical head-relative
operational evidence recorded in the final PR #15 evidence comment. PR #15 is
merged at `a850207`; the post-merge CUDA hotfix requires its own head-relative
CI and RTX 3090 evidence.

Production SSL training was not authorized as Phase 7A acceptance. Phase 8 was
not started, PDMX was not added, PLL was not implemented, and no critic or
quality score was implemented.
