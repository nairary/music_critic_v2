# Music Critic V2 Status

## Current phase

- Date: 2026-07-30
- Completed phase: Phase 1 — canonical data schema and serialization
- Phase 1A: Completed
- Phase 1B.1: Completed
- Phase 1B.2: Completed
- Phase 1B.3: Completed
- Phase 1 merge SHA: `37edf76889730980aa6ce9e9ec981e362c3480a9`
- Phase 2: Accepted and Completed
- Phase 2A.1: Accepted and Completed
- Accepted Phase 2A.1 implementation SHA:
  `32d68e8cb446d9b5dd57bfea1d28b94ccce46274`
- Phase 2B.0: Accepted and Completed
- Accepted Phase 2B.0 implementation SHA:
  `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`
- Phase 2B.1: Accepted and Completed
- Accepted Phase 2B.1 implementation SHA:
  `3898b168063094b87e5ca5d88aae0317c1562c3f`
- Phase 2B.1 closure SHA:
  `6111d3d062e02897e3f8ebdca7e4388f80ef434e`
- Phase 2B.1 merge SHA:
  `b1df77737f641b705e3c48724b2741c7a022a2e4`
- Phase 2B.2: Accepted and Completed
- Phase 2B.2 starting SHA: `3d814a2e2db7434ee6c666619dc287e5eb101101`
- Phase 2B.2 initial implementation SHA:
  `f3799765b74b17cc3a493430dc11f2a64a781b74`
- Accepted Phase 2B.2 implementation HEAD:
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`
- Phase 2B.2 closure SHA:
  `bb94e2972f94a4e092331ebd240781263656dea1`
- Phase 2B.2 merge SHA:
  `1d8a5ecf217ebd466018a1f845eedfab7e1f7828`
- Phase 3A: Completed
- Phase 3A branch: `phase/3a-raw-graph-contract`
- Phase 4A: Completed
- Phase 4A branch: `phase/4a-pop909-evidence-contract`
- Phase 4A POP909-CL identity/leakage remediation: Completed
- Phase 4A POP909-CL semantic remediation: Completed
- Harmonic supervision documentation contract: Accepted in ADR-034
- Phase 4B: Accepted and Completed
- Phase 4B branch: `phase/4b-pop909-cl-adapter`
- POP909-CL runtime adapter version: `2.0.0`
- POP909-CL corpus/production manifest version: `2.0.0`
- Model-input fingerprint contract: `1.0.0`
- Phase 5A: Accepted and Completed
- Phase 5A branch: `phase/5a-multisource-contract`
- Multi-source target ontology version: `1.0.1`
- Phase 5B.1: Completed
- Phase 5B.1 branch: `phase/5b1-target-tensorizer-collator`
- Target encoding registry version: `1.0.0`
- Phase 5B.2: Accepted and Completed
- Phase 5B.2 branch: `phase/5b2-corpus-dataset-loader`
- Multi-source corpus index version: `1.0.0`
- Multi-source canonical cache version: `1.0.0`
- Split manifest version: `1.0.0`
- Dataset view contract version: `1.0.0`
- Mixture sampler version: `1.0.0`
- Documentation branch: `docs/harmonic-supervision-contract`
- Documentation base commit: `681abbdf331c032e34cc7541224ca98f13e19a86`
- Pre-merge clarification base: `4f5f1e32f0244cbbfedd3a0cd4dbaa9047a82e51`
- Phase 6A: Accepted and merged
- Phase 6A branch: `phase/6a-trainable-local-gnn`
- Phase 6A merge SHA: `875dac3f83ab1a6cb3b3ece4875a5f55e3751409`
- Model/output `1.1.0`; encoder/candidate-prediction/reconstruction `1.0.0`;
  loss/checkpoint `1.1.0`; `BatchTarget` `1.1.0`
- Phase 6B: Accepted and merged
- Phase 6B branch: `phase/6b-hierarchy-transformer`
- Phase 6B merge SHA: `b0e8e05ea0b11a06769475468af75b8438b4d45c`
- Hierarchy pooling, coarse token sequence, hierarchical encoder output,
  top-down fusion, hierarchical model/output, and hierarchical checkpoint:
  `1.0.0`
- Phase 6C: Accepted and merged in PR #11 at
  `05501d8247f60d540e79841f89da42988a76b3e3`
- POP909-CL identity hotfix: Accepted and merged in PR #12 at
  `d3590d18550ba4a47bb8386786295d4905544fb5`
- Device-transfer contract: `1.0.2` after indexed-CUDA remediation
- Training-checkpoint contract: `1.0.0`
- Phase 6D-A: Accepted and merged in PR #13 at
  `18ebf5b69797f5d40ff38607cf8e8b5dad2f86e7`
- Phase 6D-A validation-membership parity hotfix: Accepted and merged in PR
  #14 at `bded77ff1a923f391623d735b5ad4ce290d9d2d2`
- Evaluation/artifact contracts: `1.1.1`; profiler: `1.1.0`; macro-summary
  sub-contract: `1.0.0`; unchanged train-prior: `1.0.0`
- Phase 7A branch: `phase/7a-graphmae2-ssl-baseline`
- Phase 7A: accepted and merged in PR #15
- Phase 7A merge SHA:
  `a850207897b5abf6eebccf72d44b8814260323c6`
- Phase 7A final-remediation base:
  `791ef19b1dbd7c26b7a2ef87f36d4ee5b08391a6`
- Ordered Phase 7A commits after that base are:
  `ab9477888bc39312e8501bbf18685f45cf1d5630` (acceptance remediation),
  `64f63997141b9a2e5eb9c718af992e62b01f5b9f` (final evidence),
  `ba458697599b03395b4a720888e7e7ce9d99c3bb` (cross-environment
  acceptance-profile fix),
  `3713ee4b5d51f5511699633784996a153fd86e07` (documentation-only
  post-CI evidence),
  `c0f0478be880a8e43415d0716d78cadc573a8025` (complete prepared-input
  attestation), and
  `38ae6ccbee4d089171e2d3e58f38c8d67b9baa26` (test-only completion of the
  mutation matrix). The earlier three-commit final comment omitted
  `64f63997141b9a2e5eb9c718af992e62b01f5b9f`; it did include the other three
  commits then present. The final documentation commits follow this list.
- Phase 7A deterministic GraphMAE2-inspired masked-graph SSL implementation
  and bounded acceptance are merged. Required CI for that historical head is
  recorded in the final PR #15 evidence comment.
- Phase 7A umbrella SSL contract: `1.2.2`; training-report contract: `1.2.2`
- Phase 7A no-leakage mutation evidence: `1.0.0`; pitch-sensitive
  reconstruction evidence: `1.0.0`
- Phase 7A model/output, checkpoint/journal/metric-row, and
  run-manifest/performance-row contracts remain `1.2.0`
- Phase 7A prepared binding: `1.1.0`; anti-collapse diagnostics: `1.1.1`
- Phase 7A MaskPlan/policy/overlay, maskable registry, decoder,
  representation target, pitch-mutation fixture, and encoder export contracts
  remain `1.0.0`; representation loss, multi-view loss, and SSL objective are
  `1.0.1`
- Phase 7A maskable-field registry fingerprint:
  `97836b2adb610529994ae609e89913eb6b21ad0f07d4bf695c911251d5f8ac85`
- Blocking post-merge hotfix branch:
  `hotfix/cuda-device-canonicalization`
- Runtime device-resolution contract: `1.0.1`
- Draft hotfix PR: #17
- Next gate: Required CI on the exact remediation head and independent RTX
  3090 verification; Phase 8 has not started

## CUDA device-canonicalization hotfix status

- Independent RTX 3090 execution of `python -m pytest -q tests/ssl` against
  merged Phase 7A produced `157 passed, 2 failed, 8 warnings`.
- Both failures retained category
  `ssl.data.device_transfer_tensor_mismatch`. The confirmed root cause is exact
  comparison of abstract `torch.device("cuda")` (`index=None`) with tensors
  concretely placed by PyTorch on `cuda:0`.
- Independent execution of the initial PR #17 head `fb54e85` confirmed the
  original correction: bare CUDA resolved to `cuda:0`, exact graph and
  prepared-binding checks passed, and the device-transfer mismatch category
  disappeared. The result was `165 passed, 2 failed, 1 skipped` for SSL and
  `7 passed, 2 failed, 1 skipped` for the training CUDA tests.
- Those four remaining failures were separate: explicit CUDA indices were not
  range-checked/accepted end to end; AMP decoder prediction FP16 did not match
  detached target FP32; the velocity acceptance test modified an unavailable
  placeholder; and a resume assertion compared JSON list containers directly
  with live tuples.
- One shared resolver now canonicalizes CPU, resolves bare CUDA through the
  current device, validates explicit/current indices against visible device
  count, and rejects unavailable or out-of-range CUDA structurally before
  transfer. SSL graph transfer, prepared binding sidecars, Phase 6C
  graph/target transfer, evaluation, and direct evaluation checkpoint
  placement use it. All three engines accept `cpu`, `cuda`, `cuda:N`, and
  `auto`; AMP eligibility is based on the resolved CUDA type.
- Strict validation is preserved. Type-only CUDA acceptance is forbidden,
  wrong indices are rejected, and SSL mismatch evidence names exact
  global/node/edge/binding location plus expected and actual devices.
- Low-precision representation pairs now compute cosine loss, empty and
  ordinary numerators, means, multi-view reduction, combined objective, and
  diagnostics in FP32 with autocast disabled. Matching FP64 remains FP64;
  shapes and concrete devices remain exact, prediction gradients survive the
  cast, and targets stay detached.
- The velocity test now mutates only available rows and revalidates the raw
  graph. Resume membership retains exact fingerprint/count/limit assertions,
  canonicalizes only JSON container representation for comparison, and keeps
  byte-identical metric journals. Production validation, placeholder,
  membership, fingerprint, and resume contracts are unchanged.
- Independent RTX 3090 execution at exact head `145ee10` produced
  `195 passed, 1 failed, 1 skipped` for the complete SSL suite,
  `15 passed, 1 skipped` for the training CUDA suite, and a passing prepared
  CUDA AMP test. The sole bounded-smoke failure had every strict no-leakage
  invariant true, finite metrics, changed target/loss, and positive target
  distance. Its only false field was the old correct-target-preference gate:
  signed margin `-0.04540175199508667`, FP16 source, and FP16-derived floor
  `0.0078125`. This is not leakage evidence and is not final-head acceptance.
- Training report `1.2.2` now emits independent, canonical-fingerprinted
  `no_leakage_mutation_evidence@1.0.0` and
  `pitch_sensitive_reconstruction_evidence@1.0.0` objects. The first requires
  strict raw/source/plan/binding and `torch.equal` online invariants plus a
  changed hidden target; the second requires an effective target-distance and
  reconstruction-loss challenge. Neither requires a positive margin.
- Correct-target preference remains visible through finite FP32 cosines,
  signed margin, FP32-epsilon floor, boolean observation, and
  `observed|not_observed` status, with
  `preference_is_acceptance_criterion=false`. A trained-preference claim is
  deferred to held-out evaluation after real training.
- Umbrella SSL `1.2.2` changes newly generated model/checkpoint binding
  fingerprints. Historical Phase 7A `1.2.0` hashes remain historical and are
  not rewritten. Prepared binding, masking, model/output, graph, ontology,
  encoding, canonical, and checkpoint container contracts are unchanged.
- Local development is CPU-only, so all real-CUDA tests remain unverified
  locally. The hotfix must remain draft pending new RTX 3090 evidence.
- Previous indexed-CUDA/AMP local remediation verification passed
  runtime/config/device checks
  (`73 passed, 1 skipped, 2 warnings`), focused objective/diagnostic/CUDA
  collection (`53 passed, 5 skipped, 2 warnings`), complete SSL
  (`191 passed, 6 skipped, 8 warnings`), related training/evaluation device
  checks (`60 passed, 6 skipped, 2 warnings`), resume/checkpoint checks
  (`33 passed, 2 warnings`), and the complete default suite
  (`1059 passed, 27 skipped, 10 warnings`). Repository/import plus
  deterministic membership checks passed `12 passed, 2 warnings`; compileall
  and diff checks passed. These are CPU/skip results, not hardware
  verification.
- Current evidence-semantics remediation passed focused truth-table,
  fingerprint, FP32-diagnostic, checkpoint, and optional-CUDA collection
  (`18 passed, 1 skipped, 2 warnings`), complete SSL
  (`206 passed, 6 skipped, 8 warnings`), related training/evaluation
  CUDA-device checks (`41 passed, 6 skipped, 2 warnings`), the complete default
  suite (`1074 passed, 27 skipped, 10 warnings`), and the explicit
  deterministic repository/resume audit (`12 passed, 2 warnings`). Compileall
  and diff checks passed. These are also CPU/skip results; exact-final RTX 3090
  evidence remains required.
- The representation objective formula, weights, zero-vector policy, masking
  policies, model architecture, graph schema, ontology, dataset, cache, and
  production-training behavior are unchanged. Only the required AMP compute
  dtype semantics changed. No Phase 8 implementation is part of this hotfix.

- Phase 8A branch: `phase/8a-hierarchical-masking`
- Phase 8A status: implemented for draft-PR review; Required `push` and
  `pull_request` workflow runs on the final head are the merge gate. Do not
  merge from this implementation task.
- Phase 8A hierarchical plan/policy/config/mixture/unit-evidence/
  unavailable-reason/prepared-hierarchy-binding/prepared-profile/
  hierarchy-output/leakage-audit/fixture/acceptance/benchmark contracts:
  `1.0.0`
- Phase 8A policy contract fingerprint:
  `b188e90a60d3ec6184dfdb3233ef37b1a0ea133cd5957a10fad3eddf58d77ccd`
- Phase 8A pitch-leakage audit fingerprint:
  `27fc135b61649e5b892036dd0aacc92f679493ff671320c8235d33396a7c9949`
- Phase 8A hierarchy fixture fingerprint:
  `ffd0d4c7db80323b8f1f8d72c1e4b7e530151c1b95dd68033e1a30273dd98a1b`
- Phase 8A uses distinct `PreparedHierarchyMaskBinding@1.0.0` and
  `Phase8AHierarchySSLForwardOutput@1.0.0` portable envelopes over the shared
  Phase 7A attestation/encoder kernel. It preserves
  `PreparedMaskBinding@1.1.0`, `SSLForwardOutput@1.2.0`, all other existing
  Phase 7A and Phase 6 contract versions, checkpoint metadata/state, raw
  graph/canonical/cache/split contracts, and independent-control artifacts.
- Next gate: review and merge of the Phase 8A draft PR by an authorized
  maintainer. Phase 8B has not started.

## Phase 7A implementation status

- The implementation is GraphMAE2-inspired, not a faithful GraphMAE2
  reproduction. Target mode is `shared_stop_gradient_full_view`; no EMA target
  encoder is present.
- `note_pitch_group` is the only mask family. Primary note fields are `pitch`,
  `pitch_class`, `octave`, and `track_relative_pitch`, including availability
  contributions. Every unselected note peer in an affected owner track
  collateral-masks `track_relative_pitch` and availability. Collateral
  owner-track fields are `mean_pitch`, `pitch_std`, `min_pitch`, and
  `max_pitch`, including availability. Peer-note and owner-track collateral
  fields close redundant pitch leakage and are not reconstruction targets.
- Per-sample train MaskPlans use deterministic SHA-256 selection without
  replacement and vary by epoch when possible. Fixed validation plans use
  canonical epoch zero. Plans and decoder views are independent of target
  sidecars, batch order, worker count, and Python `hash()`. Supplied plans must
  equal the complete canonical target-independent view-zero plan. Production
  builds and attests plans from validated CPU batches before device transfer;
  prepared accelerator forward performs no graph-sized host materialization.
  Binding construction regenerates canonical plans and fails closed on a
  validly fingerprinted alternative.
- Prepared binding `1.1.0` additionally captures complete private,
  process-local runtime evidence for the validated model input: strong
  graph/store references plus identity and type, ordered node/edge types,
  exact global/node/edge attribute sets, and all 65 graph tensors. Every
  tensor is bound by strong reference, object identity, `_version`, shape,
  dtype, and device; the compact selected-index tensor receives the same
  evidence. Typed immutable evidence covers non-tensor metadata, including
  `entity_id` collections. Transfer first re-attests the complete source
  surface, compares the destination surface, and replaces the source
  descriptor with a fresh destination descriptor.
- Runtime identities, strong references, version counters, HMAC material, and
  capability tokens are deliberately excluded from deterministic binding
  fingerprints, serialization, checkpoints, reports, and caches. Public Phase
  6 `forward`/`encode` paths always retain the full raw-graph validator; there
  is no caller-controlled boolean bypass. Only an opaque process-local token
  can enter the private prepared path, and both the target and online encoder
  calls independently issue and immediately re-attest it.
- Masking is an immutable model-side contribution overlay. It does not mutate,
  serialize, or cache masked raw graphs, and the no-mask Phase 6 path retains
  its existing outputs and state-dict surface.
- Selected note rows use deterministic latent decoder re-masking and
  representation decoding. Decoder context mode
  `online_owner_track_bar_song_temporal_neighbors` combines only masked-online
  owner-track, available owner-bar, song, and temporal-neighbor
  representations, preventing fully re-masked rows from predicting from one
  constant token alone. Bar and song rows use separate projector/predictor
  latent losses. All use `1-cosine` with `eps=1e-8`, explicit sum/count/mean
  and unavailability, and retained zero-norm rows. Exact stage-level
  `anti_collapse_aggregate` values use float64 mergeable `O(D)` retained state
  for note/bar/song target and prediction. They retain no embedding history,
  build no production pairwise matrix, and are invariant to partition/order/
  worker changes. This is an `O(D)` retained-state bound, not an `O(D)` peak
  temporary-memory claim: current `from_values` materializes float64 `N x D`
  `values64` and normalized working temporaries. Real CUDA cost has not been
  measured; a separate RTX 3090 profiler/optimization gate is required before
  production SSL.
- The simple ablation is one decoder view with remask probability zero. The
  main preset is three views with probability `0.20`; no superiority claim is
  made.
- The production raw-only loader uses a dedicated dataset/collator around
  `load_cached_piece` and `build_raw_graph`; it never projects supervised
  targets. It preserves the existing group-safe train/fixed-validation
  membership. Reports distinguish one-batch plumbing, bounded held-out/
  non-collapse, named production-cache execution, and production/full-corpus
  claims. SSL checkpoints bind model, objective, mask registry, resolved config,
  data/split/composition/fixed-validation,
  optimizer/scheduler/scaler, RNG, and epoch journal state; resume is
  failure-atomic and epoch-boundary-only. Encoder transfer loads only the local
  encoder, hierarchy pooling, Transformer, and fusion, leaving supervised
  heads untouched.
- The multi-note fixture has 3 train pieces / 48 notes and 2 disjoint fixed
  validation pieces / 36 notes. Train graphs have 114 nodes / 740 directed
  edges; validation has 83 / 546. Requested mask rate `0.30` realizes
  `13/48` train and `10/36` validation, with primary/peer/owner counts
  `13/35/7` and `10/26/5`. Fixture fingerprint is
  `9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922`.
- Forty-step one-batch plumbing uses the Phase 7A-specific default AdamW rate
  `3e-4`; explicit overrides remain supported. Loss changed
  `3.122128486633301 -> 0.04193296656012535`. The fixed
  `midi_axis_reflection_v1` coherent pitch target produced correct-vs-mutated
  cosine margin `+0.0009066462516784668`, target cosine distance
  `0.005206167697906494`, and mean target L2 distance `1.1555137634277344`;
  runtime-source binding, raw-store immutability,
  no-leakage, deterministic repeat, and checkpoint reload passed. This
  historical positive margin is diagnostic rather than an acceptance
  constant; the one-batch final state is not used as non-collapse evidence.
- Fixed held-out loss was `3.1229397773742678` before optimizer step zero,
  then `2.5964468638102214`, `2.2769506017367043`, and
  `2.0780126730600994` over three epochs. Initial/final note/bar/song aggregate
  diagnostics were finite and noncollapsed, with all zero-norm counts zero.
  Best checkpoint was epoch 2 and was selected only by fixed-validation loss.
  Two fresh runs had identical semantic artifacts and recursively bit-exact
  loaded checkpoint states.
- Security-remediation fingerprints are model
  `7a1ece2b44dc6b52aef6f7c7532238d4716b1a45c38b8ca66957225a24b76774`,
  train epoch-zero prepared binding
  `f400906c311313edc58802aea8283adb7de3b4a1c2d2abfd8b2c28bb8dd36b76`,
  and validation prepared binding
  `cbf820a5ae2022ce53da05a7d5bb2ef769c13fb618a848a66f40f6c5bd8c7bf9`.
  The exact-path held-out rerun under
  `/tmp/music-critic-v2-phase7a-final-heldout` retained resolved-config hash
  `554c09dd93245d173580e1861e91486bffae4b765eeb6bbdf2ae3ec1659b800f`
  and produced fingerprints/run/initial/metrics hashes
  `484af62d67e999a10582668733f528875d82776de5ecf876d38237f298c1dd05`,
  `b003cd18b941870c3e7812e47ef1125fa0595f353dc9f628cb9f97315b1f1572`,
  `92c81aae2a16d1cb96f8e4a951ea06e36abf0373fa5871bdd57c9c41e9ba56f7`,
  and
  `eb0f4b27bbbdf336539ae757c9bc68d56a41d6f63adaefba0e076217389e713a`.
  The numerical trajectory and diagnostics remained unchanged.
- Checkpoint reload was bit-exact; exact dropout/cosine/CPU-AMP epoch resume,
  atomicity, crash recovery, and RNG rollback tests passed. Encoder transfer
  loaded 470 parameter tensors and left all 81 supervised-head tensors
  untouched. Earlier pre-security-remediation timing is not treated as
  head-relative performance evidence; no current speed claim is made. CUDA was
  unavailable, so CUDA/VRAM evidence is an explicit skip rather than
  fabricated data.
- Post-matrix focused prepared-binding/model/masking/bounded-leakage tests
  passed `95 passed, 1 skipped, 2 warnings`; complete SSL passed
  `157 passed, 2 skipped, 2 warnings`. The structured mutation matrix has 28
  cases, including onset `candidate_slot` and split-like attribute injection.
  Source-identical Phase 6 model/graph regressions passed
  `146 passed, 1 skipped`, and checkpoint/resume/transfer passed `19 passed`.
  The final head-relative complete suite passed
  `989 passed, 21 skipped, 2 warnings in 84.16s`. The automated held-out check
  passed once and two exact-path runs were byte-identical. `compileall`,
  `git diff --check`, and `git show --check` passed. CUDA was unavailable, so
  its prepared-forward test skipped and no GPU evidence is claimed.
- Production SSL training has not been performed for Phase 7A acceptance.
  Phase 8A adds bounded hierarchy-aware mask/view mechanics only; Phase 8B
  has not started. PDMX has not been added, PLL has not been implemented, and
  no critic or quality score has been implemented.

## Phase 8A implementation status

- Exactly five policies are versioned: the bit-exact Phase 7A
  `independent_note_pitch` control plus onset descendants, beat descendants,
  one start-anchored contiguous bar span, and the sparse intersection of one
  raw track with one start-anchored bar span.
- Every new policy hides only the four note pitch fields and availability,
  then applies the unchanged Phase 7A peer-relative-pitch and owner-track
  pitch-statistic collateral closure. The fail-closed audit classifies all 68
  raw registry fields as four primary identities, four unique owner-track
  collateral identities, and an exact ordered 60-field visible remainder.
- The target-blind CPU index validates note/onset/beat/bar/track ownership,
  including beat-bar ownership and agreement between an onset's direct bar and
  its owning beat's bar. Fixed-width stable radix ordering, linear scans,
  bounded `max_span_bars <= 8` enumeration, and sparse occupied track/bar cells
  avoid pairwise or dense hierarchy matrices.
- Unavailable policies return a structured reason and never silently fall
  back. Mixture resolution records the full eligibility set, exact normalized
  weights, stable resolution seed, selected policy, and explicit realized
  denominator/frequency.
- Hierarchy execution returns distinct
  `PreparedHierarchyMaskBinding@1.0.0` and
  `Phase8AHierarchySSLForwardOutput@1.0.0` artifacts over the shared Phase 7A
  full-attestation/HMAC/token/transfer implementation. Normal `forward()`
  remains Phase 7A-only; `forward_hierarchy()` is explicit. An
  independent-only configuration returns the exact old Phase 7A binding,
  overlay, logits/embeddings, and losses.
- The bounded fixture has 6 pieces, 14 tracks, 15 bars, 60 beats, 42 onsets,
  93 notes, 39 polyphonic onsets, one multi-onset beat, one cross-bar
  sustained note, and 34 occupied track/bar cells. Train/validation identities
  are disjoint.
- The final compact acceptance report is byte-exact across two fresh
  processes: 32,229 bytes, SHA-256
  `e6915779f21784a1907c930da7967d2d6c1dae4cfd72fbb0ed5c24bec37cc03a`.
  All five single-policy runs resolved `4/4`, produced finite loss, and had
  380/474 finite gradient tensors; 361 were nonzero for independent/onset/
  beat/bar and 376 for track/bar. A separate four-draw mixed-policy smoke
  resolved track/bar `1/4`, independent `2/4`, and onset `1/4`; it is
  accounting evidence, not a frequency-quality claim.
- Final local verification: Phase 8A focused `84 passed, 1 skipped`; workers
  `0/2` parity `1 passed`; complete SSL `241 passed, 3 skipped`; Phase 6
  model/graph/leakage `146 passed, 1 skipped`; checkpoint/resume/transfer
  `19 passed`; full repository `1073 passed, 22 skipped`. CUDA was unavailable
  and no GPU evidence is claimed.
- The no-threshold bounded CPU benchmark measured 147 nodes, 920 total edges,
  208 relevant edges, candidate counts `57/27/26/14/35`, emitted overlay
  entries `138/147/147/162/93`, and peak simultaneously retained compact plan
  JSON bytes `5604/10621/10612/10555/10445` for independent/onset/beat/bar/
  track-bar. These are mechanics and retained-serialization observations, not
  Python allocator, temporary-memory, throughput, CUDA, or quality evidence.
- No legacy source was inspected. No HookTheory or POP909-CL corpus scan,
  production cache rebuild, PDMX projection, production/full-corpus SSL
  training, PLL, critic training, or Phase 8B objective work was performed.

## Scientific context and evaluation backlog

- Strong signal for HookTheory tonic and scale degree.
- Strong signal for POP909-CL root and bass.
- Weak or collapsed signal for the remaining heads.
- HookTheory multilabel heads at threshold `0.5` produce all-negative output
  and `F1=0`.
- POP909-CL validation evidence is limited to 18 independent pieces.
- Scientific evaluation hardening remains in the backlog before final
  ablations, but does not block Phase 7A.
- The ambiguous field `test_not_used_for_checkpoint_selection` remains a
  registered evaluation-backlog item.

## Phase 6D-A supervised evaluation result

- The validation-membership parity hotfix was accepted and merged in PR #14 at
  `bded77ff1a923f391623d735b5ad4ce290d9d2d2`. Training and evaluation import
  one neutral `fixed_validation_membership_v1` implementation whose compact
  UTF-8 JSON bytes have no terminal newline, exactly matching existing Phase
  6C checkpoint fingerprints. The global evaluation canonical fingerprint
  remains unchanged.
- The previous `1.1.0` documentation incorrectly asserted exact Phase 6C
  membership parity: evaluation ranking and membership had used a
  newline-bearing fingerprint and rejected valid partial-validation Phase 6C
  checkpoints. Evaluation/artifact `1.1.1` corrects that compatibility defect
  without modifying checkpoints or weakening index/split/composition checks.
- Read-only acceptance of
  `data/runs/hierarchical-cpu-pilot-10e-2/best.pt` with seed `42` and 512 fixed
  validation samples matched its legacy membership fingerprint
  `52633d79c29498e9f865668121b5454b33baed879e66bc4fd379dbf61a0f2593`.
  All checkpoint data-binding fields verified. The checkpoint SHA-256 remained
  `31b621a214bc31caaf6d76c99b62f5e6b2913d038b9dcaf25900d18ace8a6f3b`.
  The smoke used only two train samples for prior plumbing, so it is not a
  scientific validation-metric claim.

- Existing Phase 6A/6B model-only and Phase 6C training checkpoints load into
  a fresh model from their strict model contract. Only model weights are
  applied; optimizer, scheduler, scaler, checkpoint RNG, and caller RNG remain
  untouched. Checkpoint SHA-256, model contract, current ontology/encoding, and
  current data bindings are reported.
- The evaluator defaults to fixed validation. Test requires
  `acknowledge_test_evaluation=true` and is explicitly marked unavailable for
  checkpoint selection. Phase 6C validation checkpoints are matched against
  index, split, composition, and membership evidence; cache artifacts remain
  index-SHA-validated on read.
- Raw graphs reach `model.predict` before target sidecars are joined. Only
  available, aligned, conflict-free, fully supervised rows enter streaming
  metrics. Results remain keyed by dataset and exact source-native task;
  HookTheory and POP909-CL heads never share metric or macro buckets.
- Categorical and multilabel metrics, class evidence, counts, undefined-value
  reasons, train-only trivial baselines, and model-minus-baseline comparisons
  are fixed-memory. Train priors are a separate artifact bound to train
  membership and index/cache/split/ontology/encoding evidence; held-out label
  mutation cannot change their fingerprint.
- Per-class categorical and multilabel F1 is `2TP/(2TP+FP+FN)` and is
  undefined only when truth and predictions both omit the class. Defined zero
  values remain in class macro-F1. Versioned task macro summaries group only
  by exact dataset and encoding kind, expose included and undefined task IDs
  and counts, and explicitly omit scientifically incomparable cross-vocabulary
  likelihood aggregates.
- The explicitly enabled bounded profiler separates serial exclusive
  preparation, prepared training compute, prepared validation, loader-only
  traversal, and loader-plus-compute end-to-end passes. It does not repeat
  alignment inside measured assembly, does not assign overlapping worker time
  to collation, declares per-sample/per-batch units, and labels RSS as a
  process-level high-water mark. Optional deterministic production-read-only
  subsets require explicit absolute index/cache/split paths. Normal training
  never enables detailed profiling.
- Ordinary training writes bounded per-epoch train/validation wall time and
  sample/batch throughput to `epoch_performance.jsonl`. This sidecar is
  deliberately excluded from the deterministic checkpoint and metric journal,
  so existing byte-exact Phase 6C resume evidence is preserved.
- No production cache, checkpoint, or `metrics.jsonl` was written, rebuilt, or
  deleted. Synthetic fixtures and temporary directories supplied default
  acceptance. The initial Phase 6D-A smoke read 2 train plus 2 validation
  artifacts per dataset; the later membership-parity hotfix read the explicitly
  requested completed checkpoint and 512 fixed validation artifacts through
  absolute production paths. Both wrote outputs only under `/tmp`; neither ran
  a full corpus evaluation/training pass. Legacy code was not inspected or
  reused. Phase 7 has not started.

## Phase 6D-A verification

- Validation-membership hotfix focused suite: `63 passed`; it covers literal
  legacy Phase 6C bytes/SHA-256, multiple limits/seeds and mixed identities,
  shared training/evaluation selection, production checkpoint-writer E2E,
  deterministic repeated artifacts, negative seed/limit/split/index/
  composition binding, all Phase 6D evaluation tests, and Phase 6C
  checkpoint/resume/epoch-performance regressions.
- Public canonical-data API plus hotfix tests after keeping the shared contract
  neutral and non-public: `20 passed`.
- Final local hotfix suite: `832 passed, 19 skipped` in `58.23 s`; skips remain
  the existing opt-in real-data/CUDA guards. Warnings are the existing PyTorch
  JIT deprecations and the intentional Python 3.13 worker-fork warning.
- The Phase 6C-writer/Phase 6D-evaluator deterministic artifact regression was
  rerun alone after the full suite: `1 passed`.
- `python -m compileall -q src tests` and `git diff --check`: passed.
- Draft PR #14 targets `main` from
  `hotfix/phase6d-validation-membership-parity`. Both `full-suite` checks
  passed at code commit `28c42fb` in `1m48s` and `1m54s`; no merge was
  attempted.

- Remediation metric/summary/checkpoint oracle tests: `15 passed`. The direct
  confusion-count oracle covers categorical and multilabel supported misses,
  unsupported false positives, absent truth+prediction, the old macro-F1
  overestimate, and row-order/batch-partition invariance.
- Remediation profiler tests: `6 passed`. They cover the serial exclusive
  result-flow chain, one alignment per scheduled sample, consistent units,
  honest `workers>0` unavailable attribution, a loader delay inside the
  end-to-end timer, and immutable temporary indexed-cache contents in
  production-read-only mode.
- Combined evaluation/data/prior/checkpoint plus epoch-performance/resume/
  atomicity regressions: `44 passed`. The repeated checkpoint evaluation
  remains bit-exact and raw-byte `metrics.jsonl`, checkpoint, optimizer, and
  RNG resume comparisons pass unchanged.
- The repeated-evaluation artifact check was also rerun alone after the full
  suite: `1 passed`; both artifact directories were byte-identical.
- Final default suite on the completed remediation tree: `813 passed,
  19 skipped` in `807.18 s`; skips are existing opt-in real-data/CUDA guards.
  Warnings comprise PyTorch JIT deprecations, the Python 3.13 fork warning in
  the intentional worker test, and one non-failing DataLoader cleanup warning.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- Pre-remediation bounded CLI acceptance wrote all five required evaluation artifacts for
  3 validation samples across both datasets, evaluated 63 eligible rows,
  retained zero prediction tensors, and completed one profiler cell. The
  superseding profiler `1.1.0` evidence uses separate measurement passes
  rather than calling the former overlapping timings independent.
- Optional read-only production smoke used a temporary model-only checkpoint
  and temporary outputs. HookTheory and POP909-CL each loaded exactly 2 train
  and 2 validation cache artifacts through explicit absolute index/cache/
  manifest paths. Results remained dataset-isolated. Historical data
  verification was correctly `false` because a Phase 6A model-only checkpoint
  has no Phase 6C data binding.
- PR #13 was merged into `main` at `18ebf5b`; both remediation `full-suite`
  GitHub checks passed at commit `e9e22f0` in `1m51s` and `1m55s`. The
  validation-membership compatibility correction is intentionally isolated in
  its own hotfix branch and new draft PR.

## Phase 6C reproducible baseline training result

- Registered Hydra configuration groups select feature-only, local-GNN, or
  hierarchical models; bounded, HookTheory, POP909-CL, or mixed data;
  one-batch, smoke, supervised, train, or named joint-visible-reconstruction
  experiments; explicit objective weights; AdamW; no/cosine scheduling; and
  CPU/CUDA/auto devices. One-batch alone defaults to LR `0.02` and joint loss;
  supervised training defaults to LR `3e-4`, harmonic weight `1`, and
  reconstruction weight `0`. Each run writes the resolved configuration, and
  objective/task weights enter model and checkpoint fingerprints. PU and
  open-vocabulary tasks cannot be enabled.
- `move_multisource_batch` is non-mutating. It deep-copies PyG, transfers only
  graph tensors and model-facing target tensors, preserves tuple/string graph
  metadata, and keeps provenance/diagnostics/statistics/strings on CPU.
  Full semantic graph binding is validated on CPU before transfer. The normal
  device path checks structure without data-dependent CUDA predicates; full
  post-transfer validation is an explicit debug option. Targets remain outside
  graph stores.
- One-batch mode repeats exactly one bounded or first cached train batch,
  separately reports harmonic/reconstruction/total loss, proves finite
  gradients and clipping, records candidates/availability/gradient coverage,
  requires both active losses to decrease, and requires checkpoint reload to
  reproduce eval logits bit-exactly. It is not generalization evidence.
- Multi-epoch mode uses the existing Phase 5B.2 cache/index/global split,
  lazy multi-corpus Dataset, deterministic quota sampler, and collator.
  Train membership remains epoch-dependent. Validation is a fixed,
  fingerprinted, no-replacement full view by default or one fixed bounded
  subset. Metrics accumulate per-task loss numerators and exact eligible-row
  denominators, recompute the weighted objective from epoch task means, and
  emit per-dataset counts/metrics. Each non-empty batch makes at most one
  packed metric device-to-host transfer and folds into bounded CPU buckets;
  the accumulator retains zero device tensors/bytes regardless of epoch
  length. Under the supervised preset, batches without harmonic rows skip;
  reconstruction is active only in the named joint ablation. Missing labels
  never become negatives.
- Training checkpoints contain model, optimizer, scheduler, scaler, next
  epoch, best fixed-validation metric, committed metric-row count,
  Python/CPU-torch/CUDA-torch RNG, resolved-config
  fingerprint, data/index/split/composition fingerprints, and the existing
  model contract. Payload fields and detached auxiliary application are
  prevalidated; any live optimizer/scheduler/application failure rolls back
  model, optimizer, scheduler, scaler, and RNG bit-exactly. Atomic per-epoch
  records plus `committed_metric_rows` recover either epoch write-order crash
  without duplicate/lost metric rows. `next_epoch` is bounded by configured
  epochs before live mutation. Resume is epoch-boundary only.
- Fresh runs reject managed output collisions. Explicit overwrite removes the
  known journal, metric, report, last/best, and interval checkpoint set while
  preserving unknown files. Resume validates the run manifest and existing
  evidence before any write or journal recovery; incompatible resume preserves
  RNG and every existing artifact byte/mtime.
- Epoch rows now record the LR actually used as `learning_rate_used` and the
  post-scheduler value as `next_learning_rate`; the ambiguous old field is not
  emitted. Runtime evidence distinguishes actual CUDA-to-host metric transfers
  from packed host materializations, and counts packed scalars and retained
  tensor storage. Static guards cover engine/device tensor-to-Python
  conversions and joint-reconstruction per-family host predicates.
- The split CLI wraps target-blind `plan_group_hash_split` and then performs
  existing complete global source/lineage validation before atomic output.
- This environment has CPU-only PyTorch `2.13.0+cpu`, no CUDA runtime, and no
  `nvidia-smi`; the optional CUDA acceptance therefore skips. No RTX 3090
  loss/VRAM evidence is claimed. No production cache/index/split artifacts
  were present, so only bounded generated caches were exercised and no full
  corpus scan/build/training was run.
- Phase 7/SSL, model/loss semantics, ontology/encoding, adapters, production
  manifests, canonical/graph semantics, and corpus contracts are unchanged.

## Phase 6C bounded evidence and verification

- The exact default hierarchical CPU command used seed 42, hidden size 128,
  three local layers, two Transformer layers, four heads, batch size 3, 40
  AdamW steps at learning rate 0.02, clipping at 1.0, no AMP, and no scheduler.
  Its resolved configuration was stored with the temporary run. On the final
  remediation tree it completed in `4.81744 s` under PyTorch `2.13.0+cpu`
  with deterministic algorithms enabled.
- The bounded mixed batch contained three graphs and 237 candidates.
  Eval harmonic loss decreased from `1.7528455257` to `0.0`; visible-input
  reconstruction decreased from `2.5410101414` to `0.0010602905`. The saved
  checkpoint reproduced all eval logits bit-exactly. Final-step nonzero finite
  gradients covered 487/551 trainable parameters across both
  `local_baseline` and `context_encoder`; zero-gradient parameters remain
  explicitly listed in the report.
- The bounded two-epoch regression compares an uninterrupted run with a
  checkpointed stop after epoch one plus resume. `metrics.jsonl` and complete
  final model, optimizer, scheduler, scaler, RNG, epoch, and best-metric states
  match bit-exactly.
- Focused Phase 6C training tests pass `45 passed, 4 optional CUDA skips`; the
  direct CUDA acceptance invocation reports the same four honest skips on this
  CPU-only host. After the POP909-CL identity hotfix, the full default suite
  passes `784 passed, 19 skipped`, with
  two existing upstream PyTorch JIT deprecation warnings. New coverage
  includes 1,000 synthetic metric batches with constant retained device
  tensors/bytes, fresh collision, explicit overwrite,
  artifact/RNG-atomic incompatible resume, cosine used/next LR evidence,
  pre-mutation future-epoch rejection, and static synchronization guards.
  Deterministic target audit `--check`, compileall over `src`, `scripts`, and
  `tests`, and `git diff --check` passed. Required GitHub CI subsequently
  passed before Phase 6C merged in PR #11.

## Phase 6C full-corpus POP909-CL identity remediation

- The full POP909-CL cache blocker was a collision between source-record
  identity and content identity. The generic MIDI adapter correctly used the
  score-only payload hash for its default piece ID, but POP909-CL records 543
  and 553 have byte-identical score projections and distinct source files.
  POP909-CL now supplies deterministic record IDs
  `piece:pop909-cl-<song-id>` while preserving the common score-only
  equivalence group
  `pop909-cl-score:4585134e3f7a70c105a3bb678a04ab2bc4522c04e11183f6fd6c59046be25286`.
  The independent lineage groups remain `pop909-lineage:<song-id>`.
- Source SHA-256 values are
  `7dc63700fb5e58d2d12b580aa53614413317232caa151920d6079ad2440b662b`
  for 543 and
  `618b99761e750edfaffb4053cc3ad073661fd5c969bfea840481f466a03ec07a`
  for 553. Their score projection bytes, canonical raw projection after
  excluding record/path/lineage/provenance/targets, node counts, and all edge
  counts are equal. Their strict graph fingerprints differ:
  `0a4fa698ed7748ebee855424f38c967bd04cf6b10e792b8b6a4e0aceb9230ed6`
  and
  `605072317c4029380d14d73a45be8f506a8edc45b6dca841ebb5b6e5d8920531`.
  Their common model-input fingerprint is
  `2c03b1a37a722173a72ce6fd0ce74a58f3a03627907ac4fd04702ddee07b9c7f`.
- Both records contain 163 chord blocks. Their target bundle fingerprints are
  `9962345d2a47e6c412c05a83c38c59b7f38b5901df6c19ce41079510ac77ea5b`
  and
  `eac8f6d2e1b616375343a9de71108efb84b9fee80e9a89ff75cf5cd6076e57d0`.
  Boundary and no-chord values/masks are equal. Bass values differ while its
  all-available mask agrees; root, quality, and inversion values/masks differ
  with 154 versus 152 available rows. Pairing, repeated-pitch, mixed-end,
  overlap, and unsupported counts are zero for both, while ambiguous-block
  counts are 9 and 11. This is therefore recorded conservatively as multiple
  observed target views for one score input, not as a full duplicate and not
  as a proven alternative-harmonization relationship.
- Strict duplicate `(dataset_id, piece_id)` rejection remains in force and now
  emits deterministic cluster size plus portable source identity/relative-path
  evidence. Exact score equivalents instead close transitively through
  `source_group_id`, so both records are kept as separate samples but are
  split-atomic. Strict graph fingerprints retain the song record ID;
  `model_input_fingerprint@1.0.0` excludes entity identity. The score-only
  source group, not either fingerprint, is authoritative for split closure.
- POP909-CL adapter and corpus/production manifest versions are `2.0.0`.
  Corpus index, cache, split, canonical, graph schema/feature/topology,
  ontology, encoding, model, output, loss, and checkpoint versions are
  unchanged.
- The runtime-2.0 full POP909-CL build accepted 908 records and quarantined
  only song 172. It produced 908 unique record piece IDs, 907 raw-input groups,
  exactly one two-record raw-equivalence cluster `[543, 553]`, and index
  fingerprint
  `b2008221fa59ddd0df31289561b22341db9c2eac527e1a503eac57b74da27daf`.
  The first build after the breaking adapter-version change reported zero hits
  and 908 misses. Its required deterministic rerun retained the same
  fingerprint and reported 908 hits with zero misses. Artifact count grew from
  1,816 to 2,724 (~3.0 GB), so both older immutable generations remain.
- The pre-existing complete HookTheory index remained byte-identical with
  fingerprint
  `77a1a146e6ed2f3a8af4762ef2e5ada82323b6865a09903c335814d3cc3cfd4f`.
  The deterministic seed-42 joint split has manifest fingerprint
  `b0546316acb225bb95439dab78fab95232b0a7a758316b69b85dc87f733c384d`
  and file SHA-256
  `a5b49cd7f48f87c66ed6656a223e576629373158b9f64b783c47d65e512e5385`.
  Records 543 and 553 share component
  `1e16b6a3d471ebd411cd49b57f6fdad8ac0030f43fb525570744d0178d53f41a`
  and both land in `train`. The audited split contains HookTheory
  20,993/2,577/2,605 and POP909-CL 701/101/106 records in
  train/validation/test, with no source, lineage, or raw-equivalence leakage.
  Generated cache, index, split, reports, and training outputs are not
  committed. Phase 7/SSL has not started.
- Final post-merge identity hotfix verification passed
  `232 passed, 3 opt-in skips` for focused
  MIDI/POP/graph-binding/corpus/split regressions.
- The final evidence remediation passes `24 passed, 3 opt-in skips` for
  focused POP audit/adapter/acceptance tests and `786 passed, 19 skipped` for
  full default pytest. Deterministic target audit `--check`, compileall, and
  diff checks pass. The saved fresh 909-file streaming report was revalidated
  without a cache rebuild or corpus rescan: `ready=true`, 908 accepted, only
  `172` quarantined, and zero mismatches/fatal failures. Its eight
  pairing-anomaly rows use stable corpus-relative paths and evidence fingerprint
  `603ca5eb9fa248ef3e718b0f5d6ddce166b310860473e89e7e35be0a1158662b`.
  This value is now shared by the public current constant and production
  manifest, and acceptance checks its calculated value against both with
  separate mismatch categories. The historical Phase 4A/v1 source-path
  representation remains recorded as
  `d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`.
  The real 543/553 evidence and full-cache/joint-split opt-in tests pass
  `2 passed` from the identity hotfix; those scans/builds were not repeated.

## Phase 6B deterministic hierarchy and coarse-context result

- `HierarchicalHeterogeneousBaseline` composes the unchanged Phase 6A local
  baseline with additive hierarchy pooling, a coarse per-sample Transformer,
  and gated top-down fusion. The complete Phase 6A multi-scale output remains
  present in `ContextualEncoderOutput`; no mean-only final path replaces it.
- Exact ownership comes only from validated raw beat/onset/note-to-bar,
  note-to-track, and bar/track-to-song edges. Each child has exactly one owner,
  reverse edges must transpose the forward mapping, and membership must be
  cardinality-aligned, monotonic, and cross-sample clean. Malformed hierarchy
  raises `HierarchyContractError`.
- Extraction checks graph type, every mandatory node/edge store, and
  `edge_index` presence before PyG indexing. Stable categories distinguish
  missing stores/attributes, invalid dtype/rank/shape/device, missing,
  duplicate, reordered or out-of-range children/owners, reverse mismatch, and
  cross-sample ownership. Failures do not change store inventories or
  attribute-key sets. Externally supplied ownership is fully revalidated
  against raw relations and local rows; the standard path scans each of the
  six relations once.
- Bar pooling uses own+beat/onset/note families and track pooling uses own+note.
  Each family exposes mean, max, `log1p(count)`, availability, and a learned
  projection; the parent residual is explicit. Sparse indexed reductions
  create no dense child/parent membership matrix.
- Each sample sequence is `[SONG] + canonical bars + canonical tracks` with
  separate type embeddings, runtime sinusoidal ordinals, and padding masks.
  The batch-first pre-norm Transformer never attends across samples. Its
  contextual SONG row is representation evidence, not a quality score.
- Coarse counts, family ordinals, and padded positions are computed by
  `bincount`/`cumsum`/indexed tensor placement. Production performs no per-row
  `.item()`/`.tolist()`/`.cpu()` and exactly one batch-level synchronization
  for maximum padded length. Gradients flow through song/bar/track placement.
- Note fusion receives contextual bar+track+song; onset/beat receive bar+song;
  bar/track receive their contextual row+song; song receives contextual song.
  All paths are gated residuals. The existing 14 Phase 6A heads consume fused
  raw candidates, preserving 237 tiny and 79 isolated raw-only candidate rows,
  target-only joins, masks, losses, and reconstruction.
- Hierarchical checkpoint `1.0.0` binds all six Phase 6B contracts plus the
  unchanged Phase 6A, graph, feature, ontology, encoding, configuration, and
  ordered-head contracts. Prevalidation and application-time failures restore
  complete model and optimizer state bit-exactly; save remains atomic.
- Phase 7, SSL/corruption, PLL, PU objectives, shared harmonic semantics,
  preference/quality scoring, adapters, manifests, ontology/encoding, and
  corpus contracts were not changed or started.

## Phase 6B bounded evidence and verification

- Controlled hidden-32/one-local-layer/one-Transformer-layer variants have
  98,757 feature-only, 132,101 local-GNN, and 189,701 hierarchical parameters;
  default-config references are 712,581, 2,292,357, and 3,384,581.
- Tiny forward/backward observations were 0.06806/0.00782 s,
  0.02017/0.01171 s, and 0.04158/0.01638 s. Larger observations were
  0.19795/0.01843 s, 0.06036/0.01336 s, and 0.04792/0.01791 s. These are
  bounded CPU plumbing observations without a speed or quality threshold.
- Pooling/Transformer/fusion stage observations were
  0.00146/0.00400/0.00052 s on tiny and 0.00189/0.00164/0.00051 s on larger.
  Coarse lengths were `[3, 4, 3]` on tiny and padded `[9, 4, 32]` on larger.
- A separate 16-repeat uneven `[3, 4, 3]` sequence benchmark with padded shape
  `[3, 4, 32]` observed 0.000259 s mean sequence construction and 0.025124 s
  mean complete hierarchical eval forward while retaining 237 candidates.
  It is plumbing evidence without a speed or throughput threshold.
- Every pooler, Transformer attention/feed-forward block, all six fusion
  modules, every local node encoder, and all 14 heads receive gradients. Thirty
  bounded steps reduced harmonic loss 1.79136 to 0.00000354 and reconstruction
  3.11003 to 0.000325.
- One-note L2 evidence is nonzero at local note (0.42141), pooled bar/track
  (0.10921/0.23121), contextual bar/track/song
  (0.13988/0.19839/0.02796), fused note/onset/beat
  (0.90333/0.21826/0.65703), fused bar/track
  (0.08809/0.18145), and reconstruction logits (1.37096). Topology, ownership,
  cardinality, and local retention remain fixed; an unrelated co-batched
  sample remains bit-exact at every stage. Tests require each named delta
  separately and preserve unrelated fused embeddings and candidate logits
  bit-exactly end to end.
- Focused Phase 6B tests: 45 passed, 1 optional CUDA skip. All model tests
  including Phase 6A: 85 passed, 1 optional CUDA skip.
  Graph/leakage/repository regressions: 63 passed. Dataset/collator
  regressions: 119 passed. Full default suite: 723 passed, 13 skipped.
  Deterministic target audit, compileall, and diff checks pass.
  The only warnings are the two pre-existing upstream PyTorch JIT deprecation
  warnings.

## Phase 6A trainable local baseline result

- `music_critic.models` provides one controlled implementation with
  `feature_only` and `local_gnn` configurations. Both use the exact Phase 3A
  six-store feature encoder, source-native heads, losses, reconstruction
  fields, and data. The GNN adds one distinct projection for each of the 26
  ordered forward/reverse raw relations and never changes node cardinality.
- Categorical and continuous columns have per-feature modules and learned
  availability signals. Continuous inputs use the documented bounded
  transforms. Encoder output `1.0.0` retains feature-scale, optional layer,
  and final local rows with exact batch membership; final skip fusion preserves
  the original feature scale.
- Exactly 14 fully supervised, model-ready source-native heads are
  instantiated: ten HookTheory and four POP909-CL tasks. Open mode/borrowed and
  positive-unlabeled boundary/no-chord have no head or ordinary loss. No shared
  cross-source or pitch-class-set output exists.
- Model/output `1.1.0` emits candidate logits for every raw node allowed by
  each active task before target access. Raw-only batches retain those logits
  and have no harmonic loss. Target replace/delete/mask/add leaves candidate
  identities and eval logits unchanged. `BatchTarget` `1.1.0` adds validated
  tensor node-type codes for the separate supervision join.
- Loss contract `1.1.0` exposes unreduced local CE/BCE rows, means within each
  task/node-type/sample group, means groups within task, then takes a
  configurable weighted mean of active tasks. Candidate routing, target join,
  and group reduction use tensors; Python work is bounded by the fixed
  task/node-family registry. Empty tasks add neither targets, zeros, nor NaN.
- Visible-input reconstruction `1.0.0` predicts one inference-safe local field
  per mandatory node type with availability-aware CE or Smooth L1. It is only
  gradient/overfit plumbing and is not SSL, likelihood, anomaly, or quality.
- Checkpoint `1.1.0` binds all model, canonical, graph, feature, ontology,
  encoding, and ordered-head contracts, validates complete model/optimizer
  structure before mutation, restores both states on application failure, and
  saves by atomic replace. The canonical single-note diagnostic rebuilds and
  validates original/perturbed production graphs, preserves stable identities
  and topology, reports exact raw-feature/local changes, and separates exact
  linear oversmoothing by sample/node type/scale without assigning quality.
- Phase 6A itself does not include hierarchy/Transformer, PU objectives, shared
  harmonic heads, SSL, PLL, critic/quality, production training/splits,
  adapters, or manifests. Phase 6B composes this output additively.

## Phase 6A bounded evidence and verification

- Reference configuration `(hidden_dim=128, gnn_layers=3, dropout=0.1)` has
  712,581 parameters for `feature_only` and 2,292,357 for `local_gnn`.
  The bounded benchmark uses `(32, 2, 0.0)` and reports 98,757 and 165,445
  parameters respectively.
- The tiny HookTheory + POP909-CL + raw-only batch has 3 graphs, 28 nodes,
  98 directed edges, 237 raw candidate logits and 63 supervised rows. Its
  isolated raw-only graph emits 79 candidates over all 14 tasks, zero
  supervision rows, and no harmonic loss. The larger target-heavy synthetic
  batch has 9 graphs, 85 nodes, 302 edges, 711 candidates, and 252 supervised
  rows. The corrected-evidence CPU observation was 0.0419/0.0474 seconds for
  feature-only/local-GNN on tiny and 0.0588/0.0606 seconds on larger. These
  are diagnostic observations without a threshold or corpus-feasibility claim.
- Forty deterministic local-GNN overfit steps reduced harmonic loss from
  1.816520 to 0.000000219 and visible-input reconstruction from 2.581742 to
  0.000221. Every mandatory node feature encoder and every one of the 14
  active task heads had nonzero gradients. Representative HookTheory
  categorical/multilabel and POP categorical task losses decreased.
- The one-semitone canonical perturbation preserves note/entity identity and
  topology while changing the graph fingerprint. Exact raw changes are track
  `mean_pitch`/`min_pitch`/`max_pitch` and note `pitch`/`pitch_class`; local
  note L2 is 0.600571 at feature scale, 0.382855/0.300158 after two local
  layers, and 0.603081 after final skip; final onset/beat/bar L2 is
  0.132812/0.165358/0.059973 and pitch reconstruction-logit L2 is 0.923962.
  Oversmoothing evidence never mixes either of the two graphs or node types.
  For sample 0/1 beat stores, the exact dense-convention means are
  0.981216/0.981216 at feature scale, 0.800983/0.802777 at layer 1,
  0.716754/0.699026 at layer 2, and 0.898329/0.894792 at final skip. All 48
  diagnostic groups have `zero_norm_count=0` in this evidence; one-node groups
  remain unavailable. Independent tests cover one-zero and all-zero groups so
  zero collapse remains visible. This carries no quality label.
- The linear statistic subtracts the actual normalized diagonal
  `sum_i ||u_i||²`, not `N`, matching an independent dense cosine-matrix oracle
  for random non-zero, mixed zero/non-zero, and all-zero embeddings. An
  application-time checkpoint test mutates live Adam state and raises during
  the first optimizer load; the second load restores the full model and
  optimizer snapshot bit-for-bit.
- Final complexity remediation validates monotonic rank-one long membership
  and builds contiguous boundaries once per node type. Every group is a basic
  slice view, every embedding row is processed once per scale, and production
  creates no boolean-selected `N_group x D` copy or `N x N` cosine matrix.
  Boundaries use `O(T*S)` CPU metadata, cosine accumulation uses `O(D)`
  temporary memory per non-trivial group, and report traversal/storage use
  `O(K*T*S)` time/memory. Malformed non-monotonic membership fails with
  `OversmoothingContractError`.
- Focused model/head/loss/checkpoint/diagnostic/benchmark suite: 40 passed.
  Graph/leakage/repository regressions: 63 passed. Dataset/collator
  regressions: 119 passed. Full default suite: 678 passed, 12 skipped, with
  two existing upstream PyTorch deprecation warnings. Deterministic target
  audit `--check`, `compileall` over `src`, `scripts`, and `tests`, and
  `git diff --check` passed. Required CI remains the remote merge gate. No
  corpus scan or full training run was performed.

## Phase 5B.2 corpus Dataset and loader result

- Portable deterministic index headers bind source, adapter/config, canonical,
  graph/feature, ontology, and encoding contracts. Accepted records and
  structured quarantine are separate; absolute/traversing paths, duplicate
  piece identities, stale versions, and fingerprint mismatches fail closed.
- Offline HookTheory streaming and POP909-CL discovery/adapter builders write
  one SHA-addressed canonical JSON artifact at a time with atomic rename.
  Partial writes are invalid; graphs and tensors are not cached. HookTheory
  quarantines only the expected `HookTheoryAdapterError` under stable category
  `hooktheory.record_conversion_invalid`; unexpected failures propagate and
  abort. Both builder limits reject zero, negative, bool, float, and string.
- `IndexedMultiSourceDataset` loads metadata only and reads/verifies one
  artifact in `__getitem__`, then invokes `prepare_multisource_sample`.
  Canonical/prepared identity, source/lineage, and recomputed availability
  must match indexed sidecars. Raw-only canonical pieces and spawn/pickle
  graph binding are supported.
- One external global split manifest binds the exact complete constituent
  indices, all pieces, and transitive source/lineage components across dataset
  boundaries. It is validated before views are derived; missing, extra,
  duplicate, stale, or independently manifested constituents fail closed.
  Suggested source splits are diagnostics only. No production ratios/seed
  were selected.
- Every view/composition fingerprint binds manifest, split, constituent index
  fingerprints, and exact ordered `(dataset_id, piece_id)` membership.
  Single-split multi-corpus composition, exact largest-remainder quotas,
  deterministic shuffled local cycles, `set_epoch`, worker seeding, and the
  unchanged Phase 5B.1 collator provide epoch-level reproducibility without
  target-dependent split or sampling. Sampler evidence hashes resolved piece
  identities plus its version/seed/epoch/weights/quotas, not integer offsets.
- The workers=0/2 regression compares complete graph serialization and
  fingerprint, all target tensors/masks/indices/routing/confidence/supervision
  metadata, provenance, diagnostics, identities, and deterministic
  `BatchStatistics`; no CPU-only contract field is excluded.
- Default evidence is bounded/synthetic only. No full HookTheory cache build,
  909-file POP909-CL acceptance rerun, training corpus build, adapter change,
  production manifest change, or Phase 6 implementation is part of this task.

## Phase 5B.2 verification

- Base `main`: `c56bfaff2bbbb1f2d5ba249327274fa950648034`.
- Focused corpus/cache/global-split/view/sampler/worker tests: 80 passed.
- Phase 5B.1 collator and graph-leakage regressions: 28 passed.
- Full default suite: 638 passed, 12 skipped; skips are opt-in real-corpus
  integration tests.
- Deterministic target-contract audit and bounded multi-index
  corpus/cache/global-split audit both passed `--check`.
- The bounded synthetic two-index audit produced index fingerprints
  `7295b01ce8e6517cc311289e084b3b217614a645dc52459cf8a1df25e19992d6`
  and
  `b5b7b8ec30d1ec65a84bbb83e247ba65323cc06fc92c119307fa2d419a8945a1`,
  global manifest fingerprint
  `3d5581820ed5b9802623b1d2c858d2ad27bdb5615a9e761df43b6e26f66caf6e`,
  and composition fingerprint
  `ed469225c644dc785c9c0ed14f416ba3ee6e6fc4d1e1f21448b4822ab7c07467`.
  Alpha/beta view fingerprints were
  `925f895998e7e530aaa889bd9763835ad3da06ebfbe3a7797ddbab361353ebfa`
  and
  `80beb70be6d6a6a848e5c1d80b060befb82270dd25ade1f434e847b09d630e4f`.
- A 2:1 explicit bounded mixture at epoch size 6 realized alpha/beta quotas
  4/2 and emitted six samples, 54 nodes, 180 edges, and 294 target rows. Its
  resolved-piece schedule fingerprint was
  `a1faf5009ddf4dd2ed1ccc46e2e12204ee43c782695565e96403846b7c0c4d17`;
  timing remains diagnostic with no threshold.
- `compileall` and `git diff --check` passed. No full HookTheory build,
  909-file POP909-CL acceptance, real cache, adapter/manifest change, legacy
  inspection, or Phase 6 work was performed during remediation.

## Phase 5B.1 exact alignment, tensorizer, and collator result

- `music_critic.tasks` now exposes exact canonical alignment, target encoding
  registry `1.0.0`, tensorization, a production
  `collate_multisource_samples`, strict `BatchTarget`/`MultiSourceBatch`
  validation, deterministic statistics, and a lightweight benchmark.
- Encoding registry fingerprint:
  `386aceef18b6ba7da5e91d406cefdcdc21d46b6839ded873312402940b507e01`.
  Deterministic bounded audit report fingerprint:
  `7303164a65d034127bd5e685b582384c3b1462d4d18c142ce356eb9001be3982`.
- Pre-merge remediation builds one immutable alignment index per piece.
  O(1) note/annotation/exact-time mappings and rational-time bisect span lookup
  have strict complexity `O(P + C log C + T log C + R + F*C)`, including
  candidate sorting during index construction. For the fixed registry, `F*C`
  is linear in temporal candidate count.
  Instrumentation verifies one index build and counts index entries, lookups,
  bisections, candidate matches, merge visits, and emitted rows.
- Notes align by exact entity identity. Region/coverage spans expand to every
  onset point and beat/bar start anchor under exact half-open containment.
  Boundary events expand only to exact-time candidates, with no snapping,
  tolerance, or node-type priority. Equal typed values merge; conflicts are
  masked with `multisource.alignment_conflict`.
- Local indices become global through
  `local_index + batch[node_type].ptr[sample_index]`, then the collator checks
  `batch[node_type].batch[global_index] == sample_index`. All values, masks,
  confidence, source identity, provenance, and diagnostics remain outside raw
  PyG stores.
- Closed categorical targets are ontology-order `torch.long [N]` with masked
  sentinel `-1`; closed multilabel targets are `torch.bool [N, C]`; open
  `theory.local_key.mode` and `theory.chord.borrowed` strings remain lossless
  CPU tuples with `model_ready=false`. No dynamic vocabulary or Python hash is
  used.
- Bounded HookTheory + POP909-CL + raw-only acceptance produces three graphs,
  all 18 stable task sidecars, 17 source target entries, 76 expanded rows,
  76 aligned available rows, zero unaligned/masked/conflict rows, 66
  model-encodable and supervision-eligible rows, and 10 deferred open-string
  rows. POP909 boundary event detection and no-chord coverage detection are
  distinct `positive_unlabeled` tasks. Bounded fixtures produce only explicit
  `present` or `N` rows and zero synthetic negatives.
- Encoding registry `1.0.0` now declares only value representation plus
  `fully_supervised`, `positive_unlabeled`, or
  `deferred_open_vocabulary` semantics. It exposes no ordinary-BCE
  eligibility API and leaves every CE/BCE/focal/PU decision to Phase 6.
- Target ontology `1.0.1` corrects `pop909_cl.chord.no_chord` to
  `positive_unlabeled_coverage_detection` while preserving vocabulary
  `("N",)`, adapter targets, masks, and production manifest counts. A one-class
  representation is not fully-supervised classification. Chord spans,
  uncovered candidates, and absent annotations remain unlabeled; Phase 6 must
  accept a no-chord-specific PU objective or leave the task disabled.
- Production sample preparation builds and fingerprints the exact Phase 3A
  graph. External graphs must match a fresh canonical projection, and
  collation rejects categorical-feature, continuous-feature, or topology
  mutation after preparation with
  `multisource.raw_graph_binding_mismatch`. Audit inventory uses a graph-free
  target projection; no binding enters PyG stores.
- Statistics separately count model-encodable, supervision-eligible, masked,
  available-but-unaligned, conflict, and deferred-open-vocabulary rows.
  Aggregate values are validated against the per-task values, and eligibility
  exactly sums `availability & entity_index & model_ready`.
- A separate synthetic conflict fixture produces four masked conflicts.
  Available-but-unaligned boundary and masked-source fixtures each retain one
  row with `entity_index=-1`; masked entries are never candidate-expanded or
  converted to negatives.
- The 32-graph lightweight benchmark (three repeats) produced 640 nodes, 3,456
  edges, and zero raw-only target rows. On this runner, mean per-repeat exact
  alignment was `0.0633 s`, PyG construction/validation `0.1288 s`, and full
  collation `0.3602 s`. This is bounded performance evidence, not corpus
  acceptance.
- The separate target-heavy benchmark (three repeats) produced:
  small `78` source entries / `170` emitted rows, index `0.00133 s`, lookup
  `0.00115 s`, full collation `0.06295 s`; medium `610` / `1,346`,
  `0.00911 s`, `0.00878 s`, `0.06736 s`; large `2,434` / `5,378`,
  `0.03698 s`, `0.06561 s`, `0.23564 s`. Every size recorded exactly one
  index build. This heavier evidence is excluded from default CI.
- Target/provenance/diagnostic changes leave raw graph fingerprints and raw
  PyG store allowlists unchanged. Ontology `1.0.1` fingerprint is
  `86ea17b016eafb7109fe050f9332c57f8e0f3399046debc01f4d8ac5d19d9613`.
- No full HookTheory scan, manual corpus read, or repeated POP909-CL 909-file
  acceptance was run. Adapters, production manifests, graph semantics,
  dependencies, and legacy code were unchanged. Phase 5B.2 and Phase 6 were
  not started.

## Phase 5B.1 verification

- Focused alignment/tensorizer/collator, Phase 5A contract, and deterministic
  contract suite: `39 passed, 2 warnings in 3.50s`.
- Focused tasks/graph/leakage/audit regression suite:
  `96 passed, 2 warnings in 4.63s`.
- Full default suite:
  `558 passed, 12 skipped, 2 warnings in 5.66s`; skips are opt-in real-corpus
  integrations and warnings are existing upstream PyTorch deprecations.
- Deterministic multi-source audit `--check`, compileall over `src`, `scripts`,
  and `tests`, and `git diff --check` passed.
- Compileall, diff, deterministic audit, and GitHub Actions results are
  recorded in the Phase 5B.1 PR handoff.

## Phase 5A multi-source target contract result

- `music_critic.tasks` now exposes a versioned immutable registry for all 12
  HookTheory and six POP909-CL source-native target families, including value
  spaces, exact entity/time semantics, supervision context, adapter/view,
  missing semantics, required masks/provenance, confidence policy, alignment
  policy, and cross-source-sharing permission.
- Ontology `1.0.1` fingerprint:
  `86ea17b016eafb7109fe050f9332c57f8e0f3399046debc01f4d8ac5d19d9613`.
  Stable adapter task IDs remain unchanged.
- No cross-source pair is classified `exact_shared` or accepted as
  `derived_lossless_subset`. Functional versus absolute root, extent versus
  quality, ordinal versus semitone inversion, presence versus boundary, and
  presence/rest versus `N` are `incompatible`. Absolute-root and
  pitch-class-set renderer paths are `deferred`; unpaired families remain
  `source_specific`.
- Exact per-task alignment policies cover note identity; half-open containment
  of onset points and beat/bar anchors; exact span-start boundary events; and
  explicitly available coverage spans. Typed candidates have no implicit
  priority, equal multi-span values merge, conflicts are masked with
  `multisource.alignment_conflict`, and unmatched boundary events retain a
  masked index without snapping. POP909-CL boundary event detection and
  no-chord coverage detection are distinct positive-unlabeled tasks with no
  synthetic absent/not-N classes.
- `MultiSourceSample` defines an opaque raw graph plus source/piece/group/
  lineage identity and separate target, availability, full target-provenance
  ancestor, confidence, and diagnostic sidecars. `BatchTarget` and
  `MultiSourceBatch` reserve future tensor and CPU metadata fields without
  implementing a collator. Completely empty families use zero entries. A real
  two-graph `Batch.from_data_list` now passes an exact batch-aware Phase 3A
  validator; only node `batch`/`ptr` are added to the raw allowlists, while
  production metadata, shapes/dtypes, offsets, endpoints, and reconstructed
  source graphs are checked.
- Group assignments validate both source and lineage split safety, reject
  every repeated dataset/piece identity, and treat provenance lineage as
  authoritative with source-group fallback. Validation and ordering share the
  same atomic transitive component builder, so a train/`None`/validation bridge
  is rejected. Deterministic ordering hashes components with an explicit seed
  and stable internal identities; future dataset weights require a non-empty
  ID and finite positive non-boolean number. One POP909 song remains one
  sample.
- The deterministic machine artifact contains registry/crosswalk data,
  contract-source hashes, 18 converted real-source HookTheory fixture excerpts
  from 19 accounted cases, and the accepted POP909-CL production-manifest
  counts. Its report fingerprint is
  `fff45335e789f5a28acc8d2ec342970dc47a653dc7a5a63619f8bd61f41c73f8`.
  HookTheory corpus-wide target totals are not claimed.
- No manual corpus reads, full HookTheory corpus scan, or repeated 909-file
  acceptance was run. Existing bounded fixtures and accepted manifest
  aggregates supplied every Phase 5A-required field/count.
- Canonical schema `2.0.0`, graph schema/builder `1.0.0`, both adapters,
  targets, manifests, raw graphs, and inference behavior remain unchanged.
  No dependency was added.

## Phase 5A verification

- Expanded focused ontology/audit/adapter/graph/repository suite:
  `142 passed, 2 warnings in 3.11s`.
- Full default suite:
  `533 passed, 12 skipped, 2 warnings in 4.10s`; the skips are explicitly
  gated real-corpus integrations and warnings are the existing upstream
  PyTorch deprecations.
- Deterministic audit `--check`, `.venv/bin/python -m compileall -q src scripts
  tests`, and `git diff --check`: passed.
- GitHub commit, draft PR, and Actions results are reported in the final Phase
  5A handoff.
- Phase 5B.1 now owns exact entity-index tensorization, PyG batching/offset
  validation, collation, and statistics. Phase 5B.2 retains production corpus
  loading/indexing, worker seeds, configurable mixture weights, and split
  consumption; any changed task semantics require a later evidence-backed
  ontology version.
- Final splits, cache, renderer, applied/borrowed crosswalks, models, losses,
  SSL, PLL, quality critic, and PDMX/Dilemmadata adapters were not started.

## Phase 4B production POP909-CL adapter result

- Public API from `music_critic.adapters`: `Pop909ClCorpusIdentity`,
  `Pop909ClAdapterConfig`, `Pop909ClCorpusRecord`,
  `discover_pop909_cl_corpus`, `convert_pop909_cl_file`,
  `iter_pop909_cl_corpus`, typed accepted/missing/quarantine results, typed
  adapter/corpus/conversion errors, and source/lineage group helpers.
- Deterministic discovery supports direct and nested `POP909_processed`,
  preserves `043 .mid`, excludes AppleDouble noise, diagnoses malformed,
  missing, duplicate, and unexpected IDs, and requires the pinned 909-file
  fingerprint.
- Instrument routing uses only channel-bearing events. Channel 0 plus
  conductor/meta tracks form the raw score projection; channel 1 is excluded
  completely and parsed separately as target evidence.
- Six stable target tasks cover boundary, root, quality, bass, inversion, and
  no-chord. Exact block evidence retains ticks/PPQN, pitch multisets and
  per-note ends, candidates, source track/path/hash, pairing, repeat,
  mixed-end, overlap, and gap diagnostics. Bass and inversion have independent
  masks.
- `367` and `658` return `Pop909ClExpectedTargetAbsence` with six explicit
  one-entry `mask=false` arrays. `172` returns `Pop909ClQuarantine` only when
  the generic adapter actually raises
  `midi_adapter.meter_change_inside_bar`; a successful conversion or different
  failure is fatal.
- `include_targets=False` removes target spans, arrays, and target-only
  provenance/diagnostics while preserving the raw piece. Source chord
  mutation, replacement, and deletion leave score projection bytes, canonical
  tracks/notes, and graph fingerprints unchanged.
- Exact chord intervals remain lossless in structured evidence. Canonical
  target-alignment spans are intersected with raw duration only when a target
  extends beyond channel-0 score end; provenance records that projection and
  raw duration never depends on channel 1.
- Production code imports no audit or legacy module and writes score
  projections only to short-lived temporary files outside the dataset root.
  No dependency or canonical/graph schema version changed.
- Pre-merge remediation removed the public file-verification opt-out from
  `Pop909ClAdapterConfig`. Conversion now always compares the SHA-256 of the
  payload read from disk with the discovery record before parsing MIDI; a
  post-discovery mutation is rejected as
  `pop909_cl.file_fingerprint_mismatch`.

## Phase 4B verification

- Focused production adapter plus Phase 4A audit:
  `20 passed, 1 skipped, 2 warnings in 2.00s`.
- MIDI, HookTheory, graph leakage/builder/strict validation, canonical
  serialization, and repository-contract regressions:
  `237 passed, 2 warnings in 2.44s`.
- Fresh streaming 909-file production acceptance:
  `ready=true` in `1249.285s`; the detailed report is only at
  `/tmp/music-critic-v2-phase4b-production-acceptance.json`.
- Acceptance counts: 909 logical files; 908 accepted; 906 with chord evidence;
  `367` and `658` accepted with explicit masked absence; `172` is the sole
  quarantine; 907 chord instruments; zero fatal failures.
- Target/evidence counts: 116,055 blocks; root/inversion 109,668; quality
  109,800; boundary/bass 116,055; 5,801 ambiguous; 586 unsupported; 947
  derived `N`; 151 trailing masked spans.
- All 908 accepted pieces passed canonical validation, deterministic
  target-visible and target-hidden JSON round trips, raw equality, and graph
  fingerprint equality. Pairing anomaly evidence reproduced the historical
  Phase 4A/v1 source-path fingerprint
  `d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`;
  this is not the current portable `2.0.0` production fingerprint.
- Full default suite: `491 passed, 12 skipped, 2 warnings in 3.46s`; all skips
  are explicitly gated real-corpus integrations and both warnings are the
  existing upstream PyTorch deprecations.
- The 909-file acceptance was not repeated for the fingerprint remediation:
  the preceding successful run already used mandatory verification through
  the then-default `verify_file_sha256=True`.
- `.venv/bin/python -m compileall -q src scripts tests` and
  `git diff --check`: passed with no output.
- Commit/push and GitHub Actions are reported in the final Phase 4B handoff.
- Phase 5, collator, common ontology, models, GNN, SSL, training, PLL,
  preference/quality critic, splits, partial-bar support, and chord rendering
  were not started.

## Harmonic supervision documentation result

- Added `docs/HARMONIC_SUPERVISION.md` as the central contract separating
  harmonic-semantic recognition, melody-conditioned harmonization, actual
  performed/score accompaniment likelihood, and preference/quality scoring.
- HookTheory melody-only raw graphs and POP909-CL channel-0 combined-score raw
  graphs may later train compatible auxiliary harmonic heads through
  dataset-specific annotation views, masks, and per-target provenance.
- HookTheory chord annotations may produce derived root, quality,
  pitch-class-set, bass, inversion, boundary/span, and supported semantic
  targets. Bass and inversion are separate target families with independent
  masks; joint/factorized prediction is a future ablation. A derivation is safe
  while target-only; derived notes remain banned from raw canonical content,
  graph input/topology, raw-input serialization, graph serialization,
  raw-input cache identity, graph fingerprints, and inference. Separate
  target/annotation/diagnostic artifacts may serialize derived targets with
  provenance without becoming raw/graph identity or inference input.
- POP909-CL Phase 4A/4B evidence, masks, provenance, channel contract, audit
  counts, acceptance criteria, and song-172 quarantine remain unchanged.
- Role-agnostic production inference requires no melody, accompaniment, bass,
  chord, voice, staff, or semantic-segmentation labels. Future robustness work
  covers track permutation/merging and metadata removal.
- Masked conditional likelihood and PLL remain future probabilistic-decoder,
  normalization, calibration, and ablation work. Neither chord confidence nor
  GraphMAE reconstruction loss is a quality score.
- Roadmap Phases 5–12 and 14–15 now carry the future ontology, head, masking,
  corpus-projection, likelihood, critic, inference, and ablation boundaries.
- Phases 7–8 validate SSL mechanics on bounded pre-PDMX data. Phase 10 must
  enable a full-scale rerun/evaluation of their accepted objectives on the PDMX
  raw-compatible corpus before scaled SSL or Phase 11 objective conclusions.
- No production code, canonical/graph schema, adapters, fixtures, manifests,
  dependencies, data, models, training, or inference behavior changed. Phase
  4B and model/PLL implementation were not started.

## Pre-merge harmonic clarification verification

- `.venv/bin/python -m pytest -q tests/test_repository_contract.py`: `5 passed
  in 0.08s`.
- Full default suite: `478 passed, 11 skipped, 2 warnings in 3.33s`; real-corpus
  integrations remained opt-in and skipped, and the warnings are the existing
  upstream PyTorch deprecations.
- `.venv/bin/python -m compileall -q src scripts tests`: passed with no output.
- `git diff --check`: passed with no output.
- Changed-path and artifact checks: only seven Markdown documents changed; no
  code, schema, adapter, fixture, manifest, dependency, data, MIDI, report,
  cache, checkpoint, or generated output changed or was added.
- No POP909-CL, HookTheory, original POP909, PDMX, or Dilemmadata corpus scan
  was run.

## Harmonic supervision documentation verification

- `.venv/bin/python -m pytest -q tests/test_repository_contract.py`: `5 passed
  in 0.08s`.
- Full default suite: `478 passed, 11 skipped, 2 warnings in 3.23s`; the skips
  are opt-in real-corpus integrations and the warnings are the existing
  upstream PyTorch deprecations.
- `.venv/bin/python -m compileall -q src scripts tests`: passed with no output.
- `git diff --check`: passed with no output.
- Semantic terminology/link scan and changed-path checks: passed; every changed
  or added path is Markdown under `docs/`.
- No opt-in POP909-CL, HookTheory, original POP909, PDMX, or Dilemmadata corpus
  scan was run. No MIDI, report, cache, checkpoint, dataset, or generated output
  was added.

## Phase 4A POP909-CL remediation result

- Corrected the production corpus identity to `pop909_cl`, specifically
  `POP909_processed` at upstream commit
  `be9094392903c471a930519e1c0bacf8b6be5d62`. All 909 installed MIDI files
  match upstream byte-for-byte. Content fingerprint:
  `b34f07d9a2678abdb6f0dcf5db1c3aec3f35caca813f1fac80c0717cfc8e0c65`.
- Separated 910 AppleDouble files from the 909-file content contract. The full
  1,819-file installation fingerprint remains
  `af623705a375c419751e4ba6456224b8b700f50fc1a09a32af57e1620d1ff4dd`.
- Measured one unique channel-0 combined-score instrument in every file and a
  unique channel-1 chord instrument in 907. Songs `367` and `658` have no
  chord instrument; their structured `missing_chord_instrument` observations
  map to expected all-false chord-target availability rather than fatal corpus
  failures.
- Added a score-only projection boundary. Channel-1 chord notes cannot enter
  canonical musical tracks/notes, raw statistics, graph structure/features,
  raw-input serialization, graph serialization, raw-input cache identity,
  graph fingerprints, or inference inputs. Separate target/annotation artifacts
  may retain chord evidence and provenance without defining raw/graph identity.
  Synthetic chord mutation, replacement, and deletion leave projected bytes,
  canonical score content, and raw graph fingerprints unchanged.
- Score-only generic conversion is 908/909. Song `172` is the sole conversion
  failure: its 4/4→6/8 event at tick 85,080 is 600 ticks inside the active
  1,920-tick bar. The later 6/8→4/4 event is also 480 ticks inside its segment
  bar. The Phase 4B MVP policy is locked to preserve `172` as the sole
  quarantine at 908/909 accepted coverage; a general partial-bar policy is an
  optional later enhancement.
- Score-only warnings total 126,163, including 123,439 same-pitch overlaps.
  Unsafe complete-file diagnostics total 126,605, including 123,873 overlaps
  and eight chord-note pairing warnings. The channel-1 contamination delta is
  434 overlaps plus those eight warnings; the remaining high count belongs to
  the flattened combined score and remains event-level rather than file-level.
- Audited 116,055 exact-tick chord blocks: 109,668 unambiguously supported,
  5,801 ambiguous, and 586 unsupported. The report preserves 261 raw
  pitch-class sets, 340 selected root/quality/bass labels, 947 upstream-compatible
  leading/internal `N` spans, 151 trailing masked/unannotated spans, 691
  overlaps, 87 repeated-pitch blocks, and 313 mixed-end blocks.
- Split provenance correctly: raw chord blocks, directly observed boundary,
  and bass use source `human` with `human_corrected`/`expert_reviewed`; normalized
  root/quality/inversion and inferred `N` use source `derived` with pinned
  upstream derivation chains.
- Added independent task masks: boundary 116,055 available; bass 116,055
  available; root 109,668 available and 6,387 unavailable; inversion 109,668
  available and 6,387 unavailable; quality 109,800 available and 6,255
  unavailable; `N` 947 available with 151 trailing spans unavailable.
- Preserved the four dangling note-ons and four unmatched note-offs as exact
  event evidence with tick, pitch, velocity/channel, ordinal, path/hash, and
  affected block/span markers. Historical Phase 4A/v1 manifest evidence
  SHA-256 using its former source-path representation:
  `d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`.
- Strict readiness now reports `evidence_contract_ready=true` separately from
  `production_adapter_ready=false`; the unimplemented Phase 4B adapter is the
  sole production blocker because the song-172 MVP quarantine policy is locked.
- Original POP909 is retained under `pop909_original` only for
  lineage/ablation evidence. CL and original use separate source groups and
  share `pop909-lineage:<song-id>` when both appear in a later split.
- No production adapter, canonical-meter change, dataset/split, graph schema,
  model, SSL, training, or inference code was added.

## Phase 4A remediation verification

Final pre-merge verification after locking the song-172 MVP quarantine:

- Fresh complete 909-file integration from the final manifest, with no
  `MUSIC_CRITIC_POP909_CL_EXISTING_REPORT`: `1 passed in 220.73s`. The new
  `/tmp` report independently reproduced 909/909 upstream matches, the
  909-entry corpus-wide block-count distribution, 908 accepted scores plus the
  `172` quarantine, `evidence_contract_ready=true`, and only
  `phase_4b_production_adapter_not_implemented` as a production blocker.
- Full default suite: `478 passed, 11 skipped, 2 warnings in 3.33s`.
- `.venv/bin/python -m compileall -q src scripts tests`: passed.
- `git diff --check`, manifest JSON parsing, and empty adapter/graph diff:
  passed.

Semantic-remediation verification on top of `65f6580`:

- Focused CL/original/repository plus saved-report acceptance:
  `20 passed, 2 warnings in 4.96s`.
- The single permitted new 909-file pass completed the audit, then the test
  stopped at the intentionally stale manifest key: `1 failed in 216.16s`.
  It was not rerun. Corrected aggregates were reconstructed deterministically
  from the existing detailed Phase 4A report, with only the four already-known
  anomaly files (`076`, `084`, `086`, `088`) reparsed for exact event evidence.
- Updated-manifest acceptance using that `/tmp` report:
  `1 passed in 3.16s`.
- Full default suite: `478 passed, 11 skipped, 2 warnings in 3.65s`.
- `.venv/bin/python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed; the production-adapter diff remains empty.

- Focused original-lineage, CL synthetic/invariance, and repository-contract
  suites: `18 passed, 2 warnings`; warnings are the
  existing upstream PyTorch deprecations.
- Single explicit full POP909-CL audit/integration:
  `1 passed in 210.85s`; detailed output was written only under `/tmp` and was
  not committed.
- Final manifest was revalidated from that existing report without a second
  corpus parse: `1 passed in 1.06s`.
- Full default suite: `477 passed, 11 skipped, 2 warnings in 3.36s`; real-data
  integrations remain explicitly gated.
- `.venv/bin/python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed with no output.
- `git diff -- src/music_critic/adapters`: empty.

## Phase 3A raw graph result

- Public API from `music_critic.graph`: `build_raw_graph`,
  `validate_raw_graph`, `graph_to_dict`, `dumps_graph`, `dump_graph`, and
  `graph_fingerprint`, plus the feature/relation/version registries.
- Every build returns PyG `HeteroData` with mandatory `song`, `track`, `bar`,
  `beat`, `onset`, and `note` stores. Canonical schema `2.0.0`, graph schema
  `1.0.0`, feature registry `1.0.0`, and graph builder `1.0.0` are stored on
  each graph.
- Exact onset-based containment, chronological/reverse relations, and sparse
  note-to-beat sustained activity are deterministic. Beat and onset candidate
  slots are raw unconditional positions for future direct theory heads.
- Separate categorical, continuous, and availability tensors contain only raw
  MIDI-observable or deterministic raw-derived fields. Targets, theory/gold
  annotations, dataset/source grouping, split, source path, provenance,
  confidence, and quality flags are not read for features or topology.
- Graph validation now enforces exact global, node-store, and edge-store
  attribute allowlists. Serialization and fingerprinting validate first, so an
  injected target, theory, split, provenance, or edge-label field is rejected
  rather than silently ignored. False availability masks require canonical
  placeholders.
- Categorical encoding is owned by `FeatureSpec`. MIDI program/channel `0`
  remain valid observations and are distinct from dedicated unavailable IDs
  `128`/`16`; known out-of-vocabulary meter values are rejected. These
  pre-merge corrections retain feature/schema version `1.0.0`.
- Simultaneous notes share onset/beat intermediaries; no pairwise simultaneous
  clique is built. Bisect ownership, pre-grouped track/bar/beat/onset indices,
  and onset sweep-line activity replace repeated interval scans. Construction
  is output-sensitive in emitted graph and sustained-note incidence; long
  sustains can still produce many `active_at` edges.
- `build_raw_graph` validates the complete `CanonicalPiece` by default.
  `assume_valid=True` is an explicit fast path for a caller that already has an
  error-free canonical validation report.
- PyTorch and PyG imports are graph-isolated but are currently global project
  dependencies. The canonical data layer remains standard-library-only, and
  optional compiled PyG extensions are not required.
- `scripts/benchmark_graph_builder.py` reports validation/build time, per-type
  node/edge counts, output tensor size, and peak memory indicators for 100,
  1,000, and 10,000 sequential notes, dense same-onset polyphony, long
  sustained notes, canonical JSON, and optional POP909/PDMX/HookTheory smoke
  inputs.
- `.github/workflows/ci.yml` exposes the stable `full-suite` check on every
  push and pull request; it runs the complete tests and source compilation.
- Non-goals remain GNNs, SSL objectives, masking/corruptions, semantic nodes,
  graph caches/collation, models, training, preference, and scoring inference.

## Phase 3A verification

- Focused graph suite: `29 passed`.
- Full default repository suite: `464 passed, 9 skipped`; all skips remain
  explicitly gated local-corpus integrations. PyG emits two upstream
  `torch.jit.script` deprecation warnings; there are no test failures.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed with no output.
- One-repeat full synthetic benchmark completed without a timing assertion.
  Sequential 100/1,000/10,000-note cases produced 237/2,317/23,127 nodes,
  1,584/15,754/157,494 directed edges, and approximately
  0.040/0.395/3.943 MB of output tensors. The dense 10,000-note same-onset case
  produced one onset and 100,020 directed edges rather than a note clique. The
  100-note, 1,000-beat sustain case emitted exactly 100,000 `active_at` edges.
  Observed build times were `0.055`, `0.573`, `6.149`, `3.323`, and `0.899`
  seconds respectively on the implementation environment; they are reported
  diagnostically and are not unit-test thresholds. Traced Python peak was
  approximately 41.5 MB and cumulative process peak RSS approximately 466 MB.
- Installed verification runtime: Python 3.13, CPU-only PyTorch `2.13.0`, and
  PyG `2.8.0.post1`. Declared compatibility remains bounded by
  `torch>=2.8,<3` and `torch-geometric>=2.7,<3`.

## Phase 2 migration status

- The HookTheory migration contract in `docs/HOOKTHEORY_MIGRATION.md` is
  **Accepted**. Evidence is classified as observed corpus,
  upstream Sheet Sage, V1 compatibility, project decision, or unresolved.
- HookTheory remediation uses piecewise meter-aware qn timing, compound
  felt-pulse tempo, active-scale pitch, and the upstream MIDI-60 anchor with
  provenance method `hooktheory_scale_degree_to_midi_upstream`. MIDI 72 is
  legacy compatibility history only.
- Applied harmony is deferred from the first HookTheory adapter.
- The Phase 2B.1 HookTheory adapter is **Accepted and Completed** at
  `3898b168063094b87e5ca5d88aae0317c1562c3f`.
- Phase 2B.0 is **Accepted and Completed** at implementation SHA
  `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`.
- Phase 2B.2 is **Accepted and Completed** at implementation HEAD
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`; the accepted chain includes the
  initial implementation and every review remediation.
- At Phase 2 closure, no graph, dataset, model, SSL, training, preference,
  quality, inference, or GRPO work had started; Phase 3A is now completed as
  recorded above.

## Phase 2B.2 canonical MIDI renderer result

- Public API from `music_critic.exporters`: `MidiRenderConfig`,
  `MidiRenderReport`, `MidiRenderError`, `piece_to_midi_bytes`, and
  `write_piece_midi`. It is generic and imports neither HookTheory, Sheet Sage,
  nor the legacy repository.
- Existing `mido>=1.3,<2` is reused; no dependency changed. The canonical data
  layer remains isolated from `mido` and the exporter.
- Validated canonical qn timing selects the denominator LCM up to PPQ 32767.
  Explicit or fallback quantization is forbidden unless
  `require_exact_timing=False`; enabled quantization uses deterministic half-up
  rounding and reports its exact rational maximum error.
- Format-1 MIDI contains a conductor track, non-percussion canonical melody
  track(s), and an optional final percussion click track. Canonical tempo and
  time-signature values are written directly. Clicks derive from
  `CanonicalBeat`; key/chord targets become optional marker text only. No chord
  notes are synthesized.
- `scripts/render_hooktheory_midi.py` supports one clip, golden manifests,
  target hiding, click/marker toggles, explicit PPQ, explicit quantization, and
  deterministic samples covering every observed mode, 6/8, 9/8, 12/8,
  multiple meters/tempos, fractional timing, and shared `ori_uid`. It writes
  exact canonical JSON, MIDI, per-clip reports, a batch manifest, and a
  listening manifest, plus independent comparison, audio-disagreement, and
  ambiguity reports in one reproducible review package.
- The real golden batch selected 19 cases, rendered all 18 usable cases, and
  reported the required missing payload as an expected skip. Seventeen cases
  are strictly exact. `ANmplRlZmyM` requires PPQ 500000000000000; the explicit
  PPQ-960 fallback reports maximum error `29/1500000000000000` qn.
- The independent simplified-source audit imports no production HookTheory
  adapter. It derives the single-endpoint bound `1/(2*PPQ)` and derived-duration
  bound `1/PPQ` from each parsed MIDI instead of trusting the exporter report as
  a tolerance. Exact mode permits no nonzero note endpoint/duration,
  tempo/meter-onset, or piece-duration error; the reported pointwise maximum is
  only cross-checked against observed endpoints. Across 1,383
  rendered/reference notes it reports 18/18 accepted clips, 17 strictly exact
  clips, one independently quantization-bounded clip, zero pitch mismatches,
  zero note-count mismatches, zero meter disagreements, and zero audit/report
  cross-check violations.
- Simplified meter reporting now separates exact identity from acceptance.
  Exact requires identical count, onset, numerator, and denominator; accepted
  requires identical count/signature and onset within zero in exact mode or the
  half-tick endpoint bound in quantized mode. Aggregate mismatch and CLI exit
  use acceptance while retaining exact and quantization-accepted counts.
- Eligible constant-meter/constant-tempo/non-swing audio comparison covers
  1,236 notes. Onset absolute error is median 0.0328056 s, p90 0.96854 s, p95
  1.667565 s; duration absolute error is median 0.00120975 s, p90 0.013095 s,
  p95 0.04021 s. Nine clips exceed the report's 50 ms onset-p95 diagnostic;
  seven agree, nine disagree, and two are ineligible. Disagreement details are
  a separate artifact and remain alignment/tempo evidence, not exporter errors.
- A streaming ambiguity audit covers all 26,175 usable records and 1,228,022
  notes without corpus-wide MIDI rendering. It finds 1,802 same-pitch overlap
  pairs across 102 clips, including 1,627 nested pairs, and zero simultaneous
  different-program conflicts on one channel. The exporter reports these
  ambiguities without rejecting, shifting, or rewriting notes/channels/programs.
- The guarantee is deliberately split: the HookTheory semantic comparison
  covers pitch, onset, duration, tempo, meter, and piece duration; generic MIDI
  rendering preserves representable pitch/timing/tempo/meter but does not
  promise full canonical identity for overlaps, program conflicts,
  unrepresentable data, provenance, targets, or annotations.
- Generated listening artifacts are outside Git at
  `/tmp/music-critic-v2-phase2b2-remediation/listening-manifest.json`. No generated
  MIDI or canonical batch output is tracked.
- Non-goals remain chord voicing and deferred harmony interpretation, audio or
  SoundFont rendering, graphs, datasets, models, training, inference, and
  Phase 3.

## Phase 2B.2 verification

- Exporter unit tests remain `20 passed` (including all nine observed scale
  families); the production exporter API and event architecture are unchanged.
- Focused independent-comparison tests: `13 passed in 0.07s`; renderer CLI:
  `2 passed in 0.12s`; ambiguity/conflict audit: `2 passed in 0.05s`.
- Opt-in real golden renderer/round-trip/review-package plus full-corpus
  ambiguity integration: `3 passed`; 18 renders/reloads, one
  required missing-payload skip, every required report, and all 26,175 usable
  canonical clips audited without corpus MIDI rendering.
- Full default repository suite: `435 passed, 9 skipped in 1.04s`; every skip
  is an explicitly gated local-corpus integration.
- Full suite with every HookTheory, semantic-crosswalk, renderer, and real-MIDI
  integration enabled: `444 passed in 383.95s`.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed with no output.
- Production dependency/import scan: passed through repository-contract and
  import-isolation tests. `mido` is allowed only in adapters/exporters; the data
  layer imports neither it nor rendering, and production rendering imports no
  HookTheory or legacy module.
- Absolute-path scan found only the pre-existing, deliberate legacy-check and
  legacy-contract references; no new production absolute path was introduced.
- The external legacy snapshot check remains exit 1 under the documented
  ADR-023 resolution-C waiver. Its current 29-entry staged state is unchanged
  by Phase 2B.2 and remains detailed in `docs/LEGACY_DRIFT_REPORT.md`.

## Phase 2B.1 production HookTheory adapter result

- Public API from `music_critic.adapters`: `HookTheoryAdapterConfig`,
  `HookTheoryAdapterError`, `convert_hooktheory_record`, and
  `load_hooktheory_piece`.
- Production input is only
  `data/HookTheory/Hooktheory_Raw.json/4_merged.json`, with optional
  `HookTheoryStructure.<split>.jsonl` group metadata. The adapter does not read
  the simplified crosswalk, HTCanon, Sheet Sage, or the legacy repository.
- The incremental production parser supports complete top-level objects and
  legacy fragments, preserves decimal lexemes, detects duplicate requested
  IDs, and has bounded memory use.
- Melody and chord timing use exact `Fraction(str(value))` arithmetic and a
  piecewise timeline: one qn per `beatUnit=1` raw beat and one-half qn per
  `beatUnit=3` raw beat, including spans crossing changes and `endBeat`.
  Sounding pitch uses active scale steps, true accidentals, and MIDI 60 for
  relative octave zero; rests and malformed/unresolved notes create no note.
- Tempo uses exact quarter-pulse BPM in simple meter and three-eighth
  felt-pulse BPM in compound meter, with final half-up rounding. Bars and
  denominator-unit beats preserve exact meter changes and incomplete
  boundaries without padding duration. Structure metadata must match clip stem
  and split before `ori_uid` may affect grouping.
- Local keys and chord spans are target-alignment annotations only. The 12
  target tasks are melody scale degree; local-key tonic and mode; and chord
  presence, root degree, extent, inversion, adds, omits, alterations,
  suspensions, and borrowed value. Applied, alternate, pedal, and section
  semantics remain deferred.
- `include_targets=False` removes annotations, targets, and annotation-only
  provenance without changing identity, grouping, split, duration, tracks,
  notes, tempo, meter, bars, beats, diagnostics, or their IDs/timing.
- Full-corpus smoke: 26,178 raw records, three missing payloads, 26,175 usable
  records attempted, 26,175 valid pieces, and zero unexpected failures.
  Remediated totals are 1,228,022 notes, 302,619 bars, 1,229,208 beats, 26,315 tempo
  events, 27,171 meter events, 476,347 target-alignment spans, and 314,100
  target arrays. All 32 deterministic spread samples passed serialization and
  target-visible/hidden comparisons. A second full-corpus hidden-target pass
  produced the same raw-content and quality-flag totals with zero annotations,
  zero targets, and zero unexpected failures.
- Old to remediated metric totals: notes 1,228,022 -> 1,228,022 (0);
  bars 304,230 -> 302,619 (-1,611); beats 1,242,480 -> 1,229,208
  (-13,272); tempo events 26,315 -> 26,315 (0); meter events 27,171 ->
  27,171 (0). Visible annotations and target arrays remain 476,347 and
  314,100; hidden mode remains zero for both.
- Quality-flag totals: alternate unresolved 14; applied deferred 19,540;
  borrowed unknown string 1; non-rest root zero 6; invalid chord timing 4;
  default tempo 3; duration extended 23; negative rest
  root anomaly 20; invalid note duration 296; invalid note timing 23;
  structure alignment unresolved 11,515; unmatched structure 14,660; and
  invalid tempo 3.
- All 19 Phase 2B.0 golden cases pass against the raw production source: 18
  usable cases convert and the missing-payload case raises the required adapter
  error.
- Semantic audit: 27,216/27,216 paired meter regions match; 1,211,093 melody
  pairs have zero pitch-class and zero relative-octave mismatches. Candidate
  pitch conversion changes 1,227,982 of 1,228,022 production sounding-note
  pitches. Candidate timing changes 6,443 note and 2,009 chord intervals,
  1,611 bars, and 13,272 beats. Compound tempo hypothesis C has 0.39% median
  error across 72 eligible user-alignment intervals, versus 50.04% for A and
  200.07% for B.
- Closure regressions confirm one complete 12/8 bar is 6 qn with 12 half-qn
  canonical beats, one complete 6/8 bar is 3 qn with six beats, compound 12/8
  at 120 BPM renders 6 qn in 1,999,998 microseconds after required integer
  tempo rounding, and simple 4/4 at 120 BPM renders 4 qn in 2,000,000
  microseconds. Production code was unchanged by closure.

## Phase 2B.1 verification

All Python commands used the project-local Python 3.13.5 interpreter.

- HookTheory parser unit tests: `10 passed`.
- HookTheory adapter unit tests after closure regressions: `53 passed`.
- HookTheory validation regression tests: `112 passed`.
- HookTheory semantic-audit and golden-fixture audit tests: `10 passed`.
- Opt-in real golden adapter integration: `1 passed` (all 19 manifest cases;
  18 conversions and one required missing-payload error).
- Opt-in corpus semantic crosswalk integration: `1 passed in 121.24s`.
- Data-layer tests: `247 passed`.
- MIDI adapter tests: `62 passed, 2 skipped`; the skips are gated real-corpus
  integrations.
- Full default repository suite after closure regressions: `397 passed, 6
  skipped`; all skips are explicitly gated real-corpus integrations. The final
  full suite with both HookTheory corpus integrations enabled: `399 passed, 4
  skipped in 131.13s`.
- Full target-visible corpus smoke: 26,175 valid pieces, zero unexpected
  failures, `32/32` serialization round trips, and `32/32` target-hiding
  comparisons.
- Full target-hidden corpus smoke: 26,175 valid pieces, zero unexpected
  failures, zero annotations, and zero targets.
- `python -m compileall src scripts tests`: passed.
- `git show --check --oneline
  3898b168063094b87e5ca5d88aae0317c1562c3f`: passed and printed
  `3898b16 Remediate HookTheory timing and pitch semantics`.
- `git diff --check
  47812f6cea2d8183b3543798ba1a252bb1380f85..HEAD`: passed with no output.
- Closure commit verification `git show --check --oneline HEAD` passed on the
  phase branch and printed `6111d3d Close Phase 2B.1 HookTheory adapter`;
  `git diff --check main..HEAD` passed with no output. After merging,
  `git show --check --oneline HEAD` printed
  `b1df777 merge: complete Phase 2B.1 HookTheory adapter`, and the base-to-main
  diff check again passed with no output.
- Production dependency/import scan: passed; the HookTheory production adapter
  imports only the standard library, its private production JSON reader, and
  `music_critic.data`.
- Added-line and new-file forbidden absolute-path scan: passed.
- Legacy unchanged check: remains intentionally failing under the explicit
  resolution-C waiver in `docs/LEGACY_DRIFT_REPORT.md`. The report records all
  staged added, removed/renamed, and modified paths with recorded/current Git
  blob hashes. Phase 2B.1 did not modify the external checkout or refresh the
  snapshot.

## Phase 2B.0 HookTheory audit result

- `scripts/audit_hooktheory_legacy.py` is a deterministic, read-only,
  standard-library audit CLI for complete JSON objects, legacy top-level
  fragments, and JSONL. It preserves decimal lexemes with `Decimal`, inventories
  and hashes sources, profiles bounded field evidence, runs named corpus-wide
  anomaly/duplicate/pitch/meter checks, crosswalks the simplified schema,
  audits structure joins, and reports `ori_uid` leakage.
- The raw merged source has 26,178 records: train 21,233, val 2,184, and test
  2,761. Three train records have no `json` payload. Existing processed and
  canonical full outputs each contain the remaining 26,175 records.
- Primary hashes: raw merged
  `8ab601050d0b8c8752c3b6bf190d63edefa5fce07735ce823bca6a3922dff833`,
  processed full
  `18421660eada680a223666f8e9af6b193900d91292b2ea7148e5c0687d2d42fe`,
  and canonical full
  `2b78e7d90bd81bd6a9d9ce946bc1ebff259d6967dcda1ad7b139bfbc5a5d8dc8`.
  The upstream simplified source hash is
  `5e7457df5640170337c6e320d32fe90d6355b5ab96f15dbd3567180a05be9c08`.
  The complete source/processed hash inventory is in
  `docs/HOOKTHEORY_FIELD_AUDIT.md` and the fixture manifest.
- Structure joins by normalized split plus `audio_path` stem match all 11,515
  structure rows: train 9,498, val 927, and test 1,090. Symbolic-only counts are
  train 11,735, val 1,257, and test 1,671; there are no structure-only or
  duplicate structure IDs and no missing structure `ori_uid` values.
- There are 2,714 original-song groups with multiple clips. Twenty-three
  `ori_uid` values cross split boundaries and are explicit leakage findings
  that must be resolved atomically before training.
- The pinned upstream Sheet Sage evidence commit is
  `bbdd7b7b6a5fb845828f82790acdceb03a197779`. The simplified-schema crosswalk
  has 26,175 matches, three raw-only missing-payload records, no
  simplified-only records, and no identifier/split mismatches.
- The crosswalk semantically compares meter regions for every matched record:
  27,217 raw regions, 27,216 simplified regions, 27,216 compared regions, and
  27,216 exact matches. It reports zero missing-raw regions, one
  missing-simplified region, one record count mismatch, and zero value
  mismatches. The bounded coverage discrepancy is clip `nvgy-WaRgkA`; key,
  melody, and harmony are inventoried but were not corpus-wide semantically
  compared.
- Nineteen bounded cases cover major/minor/modal examples, integer and
  fractional timing, first-beat conversion, rests and derived pitches,
  multiple key/tempo/meter regions, root-zero rest and malformed non-rest zero,
  chord types/inversions/decorations, borrowed null/empty/mode/list/unknown
  forms, applied raw evidence, matched and unmatched symbolic structure,
  shared `ori_uid`, a missing payload, `beatUnit=3`, `numBeats=8`, negative
  roots, null note beats/octaves, alternate `_`, and `bb1`.
- Not observed and not fabricated: raw root `8`, stringified borrowed lists,
  unexpected borrowed runtime types, derived out-of-range pitch, non-null
  pedal, exact duplicate regions, duplicate structure IDs, structure-only rows,
  or missing structure `ori_uid`.
- The semantic meter crosswalk accepts canonical numerator `numBeats`, with
  denominator 4 for `beatUnit=1` and 8 for `beatUnit=3`; the one omitted
  simplified region is coverage loss rather than a value counterexample. Still
  unresolved or intentionally deferred: `alternate`, non-null `pedal`, applied
  harmony, and audio-seconds-to-symbolic alignment. Structure timestamps remain
  audio seconds with `section_alignment_status=unresolved_audio_seconds`.
- Phase 2B.0 intentionally added no production adapter or canonical conversion
  entry point. Its evidence gate preceded the Phase 2B.1 implementation above;
  it also added no graph, dataset, model, SSL, training, preference, evaluation,
  inference, or GRPO work.

## Phase 2B.0 remediation verification

All Python commands used the project-local Python 3.13.5 interpreter.

- Corpus-wide audit CLI: passed; the final report was written outside the
  repository under `/tmp` and asserted all named counts, pitch-accounting
  totals, and crosswalk totals.
- Static audit and golden-fixture tests: `17 passed`.
- Opt-in raw/simplified/processed/canonical/structure integration, including a
  full `build_report` count assertion: `2 passed`.
- Full default suite: `331 passed, 4 skipped`; the skips are explicitly gated
  local real-data tests.
- `compileall src scripts tests`: passed.
- `git diff --check`: passed.
- The repository `make check` wrapper was unavailable because `make` is not
  installed; its two commands were run directly and passed as reported above.
- The legacy snapshot checker reports that the external read-only legacy
  worktree's pre-existing staged/dirty state differs from the recorded
  snapshot. The legacy commit remains pinned and this task did not modify,
  format, stage, reset, clean, or restore any legacy file.

## Phase 2A.1 generic MIDI result

- Public API: `MidiAdapterConfig`, `MidiAdapterError`, and `load_midi_piece`
  from `music_critic.adapters`.
- Added the sole runtime dependency `mido>=1.3,<2`. The accepted Phase 1 data
  layer remains standard-library-only and importing `music_critic.data` does
  not import `mido`.
- Supported input: Standard MIDI type 0 and type 1 files with PPQN timing,
  multiple source tracks, multiple channels per source track, empty source
  tracks, note-on/off and velocity-zero note-off, tempo/meter/key metadata,
  names, instruments, programs, percussion channel 9, and empty/no-note files.
- Timing remains exact: absolute source ticks are integers and canonical onset,
  duration, bar, and beat positions use `RationalTime` without float conversion,
  rounding, epsilon comparison, or note splitting at bar/tempo/meter changes.
- Canonical track identity is `(source_track_index, MIDI channel)`. Note pairing
  is FIFO per `(source_track_index, channel, pitch)` and never crosses source
  tracks, channels, or pitches. Unmatched note-offs and dangling note-ons are
  diagnosed without invented notes; real same-tick pairs are preserved as
  grace-like zero-duration notes.
- Tempo defaults to `500000` microseconds per quarter at tick 0 when absent or
  first observed later. Meter defaults to `4/4` at tick 0 under the same policy.
  Defaults use `kind=default` provenance; observed source events use
  `kind=source`, the accepted observed equivalent.
- Global metadata events use deterministic `(tick, source track, message)`
  ordering. Exact duplicates are removed and conflicting same-tick values keep
  the first deterministic value plus a namespaced quality flag.
- Generic MIDI emits `annotations=()` and `targets=()`. Every successful
  conversion passes `validate_piece` and both string/file JSON round trips
  preserve exact equality. No canonical cache is written by default.
- Rejected input: MIDI type 2, SMPTE/non-PPQN timing, non-positive PPQN,
  unreadable/corrupt files, and meter changes inside an active bar.
- Intentionally unsupported: MIDI 2.0, proprietary sequencer/SysEx semantics,
  lyric alignment, sustain-pedal reconstruction, voice/role/pickup inference,
  chord or key detection from notes, section detection, and aesthetic scoring.

## Phase 2A.1 remediation review

- Finding: an observed time signature with `numerator=0` and a positive source
  duration could enter metric-grid construction with a zero nominal bar length,
  preventing the bar loop from advancing.
- Fix: every selected observed/default meter is now validated before boundary
  checking, canonical meter creation, or metric-grid construction. Numerator
  and denominator must be positive integers and denominator must be a power of
  two. Invalid source values raise `MidiAdapterError` with source path, event
  tick, raw numerator/denominator, and the reason; they are never clamped,
  normalized, or replaced with `4/4`.
- Meter-boundary checking now uses one exact rational divisibility calculation
  per meter region instead of iterating across intervening bars.
- Metric-grid materialization has a deterministic combined bar-and-beat limit
  of `1,000,000` records. The adapter computes the exact record count per meter
  interval with integer/rational arithmetic before allocating any bar or beat.
  A rejection reports the source path, active meter, interval, estimated count,
  and configured limit.
- A serialized meter with denominator `2**127` and positive duration is rejected
  by the safety policy without iterative materialization. Ordinary power-of-two
  denominators including `2`, `4`, `8`, and `16` remain supported.
- Smoke discovery remains recursive and case-insensitive, excludes symlinks
  resolving outside the requested root, and now supports `first` and `spread`
  sampling. `first` remains the default. `spread` uses deterministic evenly
  spaced ceiling indices, includes both endpoints when selecting more than one
  file, and never duplicates a selected path.

## Phase 2A.1 verification

All commands used the project-local Python 3.13.5 interpreter at
`.venv/bin/python`.

- `tests/data/test_timing.py`: `28 passed`.
- `tests/data/test_schema.py`: `13 passed`.
- `tests/data/test_validation.py`: `110 passed`.
- `tests/data/test_serialization.py`: `94 passed`.
- `tests/adapters/test_midi.py`: `62 passed`.
- Full suite without the opt-in real-data variable: `314 passed, 2 skipped`;
  both skips are the explicitly gated local real-data cases.
- Explicit real-data integration with
  `MUSIC_CRITIC_RUN_REAL_MIDI_TESTS=1`: `2 passed`. This strictly converted,
  validated, and JSON-round-tripped 20 spread-selected POP909 files and 20
  spread-selected PDMX files without skipping any selected file.
- `.venv/bin/python -m compileall src scripts tests/integration`: passed.
- Data-layer import isolation: `data import isolation passed`.
- Adapter public imports: `adapter imports passed`.
- `git diff --check`: passed.
- Synthetic smoke root: `/tmp/music-critic-midi-smoke.W2edj4`.
- Synthetic smoke: `files_seen=3`, `attempted=3`, `converted=3`, `failed=0`,
  `warnings=10`, `notes=3`, `tracks=5`, `type_0=2`, `type_1=1`.

## Real-MIDI validation

Both source datasets were read recursively and remained unmodified.

- POP909-CL complete-file diagnostic root:
  `/home/str/music-critic-v2/data/pop909-cl/POP909_processed/POP909_processed`.
- POP909-CL recursive discovery: `files_seen=909`.
- POP909-CL unsafe complete-file 100-file spread smoke: `attempted=100`,
  `converted=100`, `failed=0`,
  `warnings=14475`, `notes=209228`, `tracks=300`, `type_0=0`, `type_1=100`.
- POP909-CL selected-path coverage: `selected_parent_dirs=1`,
  `selected_min_depth=1`, `selected_max_depth=1`.
- PDMX root: `/home/str/music-critic-v2/data/pdmx/mid`.
- PDMX recursive discovery: `files_seen=254035` across the complete branched
  MIDI tree.
- PDMX 100-file spread smoke: `attempted=100`, `converted=99`, `failed=1`,
  `warnings=378`, `notes=47459`, `tracks=246`, `type_0=0`, `type_1=99`.
- PDMX selected-path coverage: `selected_parent_dirs=100`,
  `selected_min_depth=3`, `selected_max_depth=3`.

Failure triage for both 100-file diagnostic runs:

- unreadable/corrupt MIDI: `0`;
- MIDI type 2: `0`;
- SMPTE/non-PPQN: `0`;
- invalid meter values: `0`;
- meter change inside a bar: `1`, represented by
  `2/31/QmcmH3b8xr1N9KSEu5zS4HG7f6Beq1fENiy3bdZ9D3FXrE.mid` at tick `8970`
  under active meter `75/4`;
- metric-grid safety rejection: `0`;
- canonical validation failure: `0`;
- serialization round-trip failure: `0`;
- unexpected exception: `0`.

The one diagnostic PDMX failure is an explicitly unsupported MVP condition;
the adapter contract was not broadened merely to force 100% conversion. No
hang, uncontrolled memory growth, parser bug, validation escape, or
serialization mismatch was observed.

The mid-bar meter-change rejection and its single PDMX diagnostic failure are
accepted for the Phase 2A.1 MVP. POP909's warning total requires later
warning-code analysis before making training-data quality decisions, but it is
not a Phase 2A.1 merge blocker.

## Phase 2A.1 scope confirmation

- Phase 1 production code, Phase 1 data tests, the accepted schema/data
  contract, and the normative fixture were not modified.
- Project dependencies were not modified by the remediation.
- The Phase 0 repository-contract test was updated to allow `mido` only inside
  `music_critic.adapters`; its bans remain active everywhere else, and the
  adapter/document packages are now required repository structure.
- The read-only legacy repository remains at
  `2d8281f31cc9ad9c8fecaf332da0c61e0e949415` with the same pre-existing dirty
  status observed before this task. No legacy file was modified.
- HookTheory remains documentation-only. No graph, dataset, model, SSL,
  training, preference, quality, inference, or GRPO code was added.
- Phase 2A.1 is accepted and Completed. The later Phase 2B.0 evidence and
  remediation work described above does not alter the accepted MIDI adapter.

## Final Phase 1 result

- Accepted and implemented canonical schema version `2.0.0` with an exact,
  explicit public `music_critic.data` API.
- Implemented normalized exact quarter-note timing with frozen, slotted
  `RationalTime` values and no float-equality timing contract.
- Implemented deeply immutable frozen canonical records. Collection fields are
  tuples, optional observations preserve `None` versus empty values, and raw
  note/track records contain no theory-label or semantic-role leakage.
- Implemented complete deterministic validation with structured errors and
  warnings, exact RFC 6901 paths, reference and ordering checks, target masks,
  confidence and provenance, exact musical timing semantics, and warning-only
  valid pieces.
- Implemented strict field-by-field decoding and validated deterministic JSON
  encoding. Unknown, missing, type, rational, version, and semantic failures
  retain their accepted error-code boundaries.
- Compact and indented JSON are deterministic; file output is UTF-8 with exactly
  one terminal newline, and public operations do not mutate canonical records
  or caller-owned mappings and lists.
- The normative `tests/fixtures/data/canonical_piece_v2.json` mapping decodes,
  validates with warnings only, re-encodes exactly, and remains equal through
  `dumps_piece` and `loads_piece`. Rational fields and immutable collections
  retain their exact Python types; masks, unknown confidence, provenance, and
  alternative annotation views are preserved.
- At Phase 1 completion the data layer used only the Python standard library and
  project runtime dependencies were empty. Phase 2A.1 preserves that data-layer
  isolation while adding `mido` only for adapters.
- No adapter, MIDI parser, graph, dataset, model, training, evaluation, or
  inference implementation was added in Phase 1.

The final float-decoding review fix in commit `396a2b5` was accepted. Huge
positive or negative integers supplied for float-valued mapping fields now
produce `VALUE_NOT_FINITE` at the exact path through
`CanonicalValidationError`; raw `OverflowError` cannot escape and inputs are
not clamped or mutated.

## Phase 1 commit history

- Phase 1A contract review and closure: `241d0e5`, `30ba3f9`, merged by
  `7ca1ce0`.
- Phase 1B.1 timing and schema types: `0ca7b95`.
- Phase 1B.2 validation: `b5c31c6`, with review fixes in `2c16d72`.
- Phase 1B.3 serialization: `1dd4e00`, with accepted float-decoding fix in
  `396a2b5`.
