# Music Critic V2 Engineering Roadmap

The scientific detail in `IMPLEMENTATION_PLAN.md` remains authoritative. This
document is the phase execution checklist.

The accepted future-facing boundary among auxiliary harmonic supervision,
actual accompaniment likelihood, and preference/quality scoring is in
[`HARMONIC_SUPERVISION.md`](HARMONIC_SUPERVISION.md). It changes no completed
phase status.

## Phase 0 — Clean repository bootstrap and legacy audit

- Status: Completed
- Goal: establish an independent, documented, tested repository.
- Dependencies: read-only V1 checkout and source implementation plan.
- Outputs: package scaffold, audit, architecture/data-contract proposals,
  snapshot verifier, and repository-contract tests.
- Tests: imports, repository contract, compile, legacy unchanged check.
- Non-goals: schema, adapters, graphs, models, training, inference.
- Acceptance: all bootstrap checks pass and V1 state matches the captured
  snapshot.

## Phase 1 — Canonical data schema and serialization

- Status: Completed
- Goal: implement exact typed canonical data, validation, and JSON round trips.
- Dependencies: Phase 0 data-contract decisions.
- Outputs: rational timing, schema types, validation reports, serialization.
- Tests: timing, malformed data, target alignment/masks, versioned round trips.
- Non-goals: MIDI parsing and graph construction.
- Acceptance: a synthetic two-track piece validates and round-trips exactly.

### Phase 1A — Canonical schema API and JSON contract

- Status: Completed
- Goal: settle the exact standard-library Python API, immutable record fields,
  validation policy, entity IDs, target encodings, and deterministic JSON
  contract before production implementation.
- Outputs: reviewed and accepted `DATA_CONTRACT.md`, Phase 1 schema ADRs, and
  synchronized roadmap/status documentation, including alternative annotation
  views, unknown target confidence, extensible adapter diagnostics, and complete
  semantic-value validation coverage.
- Tests: canonical example JSON/contract checks plus the existing repository
  test and compile checks; no new tests.
- Non-goals: production modules, unit tests, adapters, MIDI parsing, graph
  construction, dependencies, and legacy changes.
- Acceptance: the Phase 1B implementation can proceed without inventing fields,
  annotation-view lexical behavior, semantic validation codes, confidence
  semantics, diagnostics, or compatibility behavior; the normative fixture
  contains three targets including two analyses of one theory task.

### Phase 1B — Schema implementation and tests

- Status: Completed
- Goal: implement the accepted Phase 1A API and validation/serialization
  behavior.
- Outputs: `music_critic.data.timing`, `schema`, `validation`, and
  `serialization`.
- Tests: rational timing, malformed data, target alignment/masks, versioned and
  deterministic round trips.
- Non-goals: MIDI parsing and graph construction.
- Acceptance: the documented synthetic two-track piece validates and
  round-trips exactly.

#### Phase 1B.1 — Canonical timing and schema types

- Status: Completed
- Goal: implement exact rational timing, immutable canonical schema records,
  explicit public exports, and the normative synthetic fixture.
- Outputs: `music_critic.data.timing`, `music_critic.data.schema`, stable
  `music_critic.data` exports, and the canonical fixture.
- Tests: rational normalization/arithmetic/type behavior, schema
  fields/types/immutability, fixture/document consistency, target views/masks,
  raw/theory separation, and lightweight imports.
- Non-goals: validation, serialization, adapters, graphs, datasets, models,
  training, and inference.
- Acceptance: exact timing and schema APIs are implemented with standard-library
  imports only, and the fixture remains identical to the accepted contract.

#### Phase 1B.2 — Canonical validation

- Status: Completed
- Goal: implement structured validation reports, deterministic issue ordering,
  and raising validation.
- Outputs: `music_critic.data.validation`.
- Tests: complete semantic, reference, timing, target, provenance, error, and
  warning behavior from the accepted contract.
- Non-goals: serialization and dataset adapters.
- Review closure: canonical note/provenance ordering, exact issue
  deduplication, and scalable same-pitch overlap detection are covered by
  regression tests.

#### Phase 1B.3 — Canonical serialization

- Status: Completed
- Goal: implement strict field-by-field canonical JSON encoding and decoding.
- Outputs: `music_critic.data.serialization`.
- Tests: malformed data, exact-version behavior, deterministic bytes, and
  canonical round trips.
- Non-goals: MIDI parsing and graph construction.

## Phase 2 — Generic MIDI and HookTheory adapters

- Status: Accepted and Completed
- Sequence: Phase 2A.1, Phase 2B.0, Phase 2B.1, and Phase 2B.2 are accepted and
  completed.
- Goal: map unlabeled MIDI and HookTheory into the same canonical schema.
- Dependencies: Phase 1.
- Outputs: adapter interface, generic MIDI adapter, HookTheory adapter.
- Tests: missing tempo/meter, type-0/type-1 MIDI, annotation masking.
- Non-goals: graph neural networks.
- Acceptance: labels can be hidden and raw canonical inputs remain valid.

### Phase 2A.1 — Generic MIDI adapter MVP

- Status: Completed
- Implementation: accepted after synthetic tests, strict bounded POP909/PDMX
  integration, and separate 100-file real-data diagnostic smoke runs.
- Goal: convert type-0/type-1 PPQN MIDI into valid canonical pieces with exact
  tick timing, deterministic note pairing, and serialization round trips.
- Outputs: the minimal public MIDI adapter API, synthetic tests, and a bounded
  smoke CLI.
- Acceptance includes strict 20-file spread samples from both POP909 and PDMX,
  plus separate diagnostic 100-file spread smoke runs over each recursive
  corpus tree.
- Non-goals: HookTheory implementation, graph construction, semantic analysis,
  and model or training work.

### Phase 2B.0 — HookTheory legacy audit and golden fixtures

- Status: Accepted and Completed
- Accepted implementation SHA:
  `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`
- Sequence: completed after Phase 2A.1 closure.
- Goal: verify the documented migration contract against bounded real examples
  and lock golden fixtures before production conversion code is written.
- Outputs: a deterministic read-only legacy/data audit CLI; an evidence-backed
  field audit with source hashes, evidence hierarchy, joins, domains, grouping,
  simplified-schema crosswalk, and leakage; 19 bounded real-data golden cases;
  dataset-independent contract tests; and an opt-in verifier against raw,
  simplified, processed, canonical, and structure sources.
- Acceptance: exact 1-based timing, derived pitch, roots, chord decorations,
  borrowed variants, multiple regions, structure seconds, `ori_uid` grouping,
  missing/malformed evidence, and not-observed categories are executable and
  traceable without production conversion code.
- Non-goals: a production HookTheory adapter.

### Phase 2B.1 — HookTheory adapter

- Status: Accepted and Completed
- Accepted implementation:
  `3898b168063094b87e5ca5d88aae0317c1562c3f`
- Closure: `6111d3d062e02897e3f8ebdca7e4388f80ef434e`
- Merged to `main`: `b1df77737f641b705e3c48724b2741c7a022a2e4`
- Dependencies: Phase 2B.0.
- Goal: implement the accepted HookTheory migration contract without exposing
  theory labels as raw inference inputs.
- Outputs: production record converter and incremental loader, exact melody and
  metric conversion, 12 target tasks, complete target hiding, bounded-memory
  JSON parsing, golden integration tests, a read-only corpus smoke CLI, and a
  deterministic raw/simplified semantic audit.
- Corpus result: all 26,175 usable raw records convert to validator-clean
  canonical pieces; the three missing-payload records are counted and skipped.
  Remediation maps compound raw beats to half-qn, integrates crossing durations,
  uses compound felt-pulse tempo, reconstructs scale-aware MIDI-60 pitch, and
  rejects mismatched structure rows.
- Non-goals: MIDI rendering, chord-note synthesis, section alignment, deferred
  chord-field interpretation, graphs, datasets, models, or training.

### Phase 2B.2 — Canonical MIDI renderer

- Status: Accepted and Completed
- Accepted implementation:
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`
- Closure: `bb94e2972f94a4e092331ebd240781263656dea1`
- Merged to `main`: `1d8a5ecf217ebd466018a1f845eedfab7e1f7828`
- Initial implementation:
  `f3799765b74b17cc3a493430dc11f2a64a781b74`
- Acceptance covers the complete implementation and review-remediation chain
  ending at `97eda0d8fdb7c884bd3d22f0027fb872b2034399`.
- Dependencies: accepted Phase 2A.1 and Phase 2B.1 adapters.
- Goal: render any valid `CanonicalPiece` to diagnostic standard MIDI while
  preserving representable rational timing, canonical tempo, canonical meter,
  melody notes, optional canonical-beat clicks, and optional target markers.
- Outputs: a generic exporter API, HookTheory rendering CLI, semantic MIDI
  round trips, a deterministic listening sampler, and a separate audit-only
  simplified/alignment comparison. Independent comparison derives a half-tick
  `1/(2*PPQ)` bound for single endpoints and a full-tick `1/PPQ` bound for
  derived note duration; exact mode permits no observed timing error. Meter
  reports preserve strict identity separately from bounded onset acceptance,
  and aggregate/CLI decisions use the latter. Corpus ambiguity and
  channel/program diagnostics are report-only and do not alter exporter
  policy. All are
  implemented on `phase/2b2-canonical-midi-renderer`; generated MIDI remains
  untracked.
- Verification: all 18 usable real golden cases render and reload; 17 are
  strictly exact and the one excessive-LCM case matches within its explicit
  PPQ-960-derived bound. Independent simplified evidence has zero pitch,
  note-count, meter, or audit-cross-check mismatches across those 18 cases.
  The streaming corpus ambiguity audit finds 1,802 same-pitch overlap pairs in
  102 of 26,175 usable clips (1,627 nested) and zero channel/program conflicts;
  these findings limit generic round-trip/timbre guarantees but do not fail
  rendering.
- Non-goals: chord voicing, audio synthesis, graph construction, models, SSL,
  training, preference scoring, Phase 3 implementation, and treating renderer
  output as independent source truth.

The model and training phases remain pending.

## Phase 3 — Raw graph builder

- Status: Completed
- Completed task: Phase 3A — Raw graph contract and research-scope correction.
- Branch: `phase/3a-raw-graph-contract`
- Goal: construct inference-safe heterogeneous graphs.
- Dependencies: Phases 1–2.
- Outputs: versioned PyG `song/track/bar/beat/onset/note` graph, raw feature
  registry, deterministic serialization, validation, and benchmark.
- Tests: strict attribute allowlists, edge validity/reverses, temporal order,
  sustained activity, candidate slots, target/provenance leakage, adapter
  schema parity, serialization, categorical sentinels, invalid input, and
  output-sensitive growth.
- Non-goals: learned encoders.
- Acceptance: HookTheory and generic MIDI produce the same raw model-facing
  schema (not necessarily the same data), with every target and provenance
  mutation leaving inputs/topology unchanged and extra graph fields rejected.

## Phase 4 — POP909-CL evidence and adapter

### Phase 4A — Evidence audit and adapter contract

- Status: Completed after POP909-CL identity/leakage remediation.
- Goal: establish the exact POP909-CL corpus, embedded-chord, timing,
  instrument, provenance, grouping, and warning evidence before production
  code is written; retain original POP909 only as lineage/ablation evidence.
- Dependencies: Phases 1–3.
- Outputs: separate deterministic read-only CL and original-lineage audit
  CLIs, CL field audit and Phase 4B contract, lineage notes, and separate
  hashed manifests.
- Tests: CL discovery/instrument/chord/timing/no-write coverage, score-only raw
  graph leakage invariance, original-audit regressions, and an explicitly
  gated complete POP909-CL audit.
- Non-goals: production adapter, graph changes, datasets, models, SSL,
  training, and split assignment.
- Acceptance: all 909 CL files match the pinned upstream snapshot; chord
  instruments are target-only; score warnings and chord diagnostics are
  separate; embedded chord evidence is completely inventoried; and no source
  dataset file is changed.

### Phase 4B — Production adapter implementation

- Status: Completed.
- Goal: implement the evidence-backed POP909-CL adapter over the combined
  channel-0 score and target-only embedded channel-1 chord instrument.
- Dependencies: Phase 4A. The MVP retains song `172` as the documented
  quarantine at 908/909 accepted coverage; a general partial-bar-meter policy
  is optional later work.
- Outputs: score-only canonical projection, exact-tick chord blocks and masked
  targets, qualified provenance, and source/lineage-group interfaces.
- Tests: golden CL cases, instrument ambiguity, exact chord timing,
  target hiding, source/lineage grouping, and raw-graph leakage invariance.
- Non-goals: large-scale training and final split assignment.
- Acceptance: all accepted CL scores convert or fail under a documented
  general rule, channel-1 annotation cannot affect raw graphs, and leakage-safe
  POP909-CL graphs pass validation.
- Result: runtime adapter `2.0.0` and corpus manifest `2.0.0` reproduce 909 logical
  files, 908 accepted pieces, only song `172` quarantined, and fully masked
  targets for `367`/`658`. All 908 accepted visible/hidden pieces validate,
  round-trip deterministically, preserve equal raw projections, and have equal
  raw graph fingerprints. The complete pinned block/mask/anomaly aggregates
  match Phase 4A.
- Full-corpus remediation: all 908 accepted source records have unique
  `piece_id`; score-only equivalence has 907 groups with exactly one
  `[543, 553]` cluster. Both target views are retained and forced into one
  split component. Corpus-index/split versions remain unchanged.
- Integrity remediation: strict graph fingerprints retain all entity IDs;
  numerical-equivalence evidence uses separate
  `model_input_fingerprint@1.0.0`. Score-projection `source_group_id`, not a
  graph fingerprint, remains authoritative for split closure.

## Phase 5 — Multi-source dataset and collator

- Goal: batch heterogeneous task availability across datasets.
- Dependencies: adapter and graph phases.
- Phase 5A result: target ontology `1.0.1`, exact source inventories,
  conservative crosswalk, exact sidecar alignment policies, future
  sample/batch API with structural validation, provenance-authoritative
  grouping, atomic transitive source/lineage ordering, distinct
  positive-unlabeled POP909-CL boundary-event and no-chord-coverage
  supervision, and machine-readable bounded evidence. No
  current HookTheory/POP909-CL pair is declared exact-shared or an accepted
  lossless subset.
- Phase 5B.1 result: exact canonical-ID/`RationalTime` alignment, target
  encoding registry `1.0.0`, one immutable output-sensitive per-piece index,
  verified canonical-to-raw-graph fingerprint binding, strict tensor
  sidecars, production PyG collator, typed local-to-global `ptr` offsets, and
  deterministic batch statistics that distinguish encodable from
  supervision-eligible rows. Open strings remain deferred CPU values,
  positive-unlabeled boundary events and explicit positive `N` coverage spans
  have no synthetic negatives, and the encoding registry does not choose a
  loss.
- Phase 5B.2 result: portable index/cache and external split manifest contracts
  `1.0.0`, streaming offline HookTheory and POP909-CL cache builders,
  one-artifact lazy Dataset, dataset-view contract `1.0.0`, globally validated
  single-manifest/single-split multi-corpus composition,
  target-blind largest-remainder quota sampler with deterministic shuffled
  cycles, view-bound resolved-piece schedule evidence, and spawn-safe
  DataLoader routing through the Phase 5B.1 collator. Cross-corpus
  source/lineage components remain atomic; separate per-corpus manifests
  cannot bypass the global constraint.
  Production split ratios/seed and training weights remain explicit Phase 6
  configuration decisions.
- Outputs: datasets, samplers, collator, task routing, the versioned
  source-native target ontology plus any future evidence-backed normalized
  views, dataset-specific annotation views, availability masks, and per-target
  provenance for mixed HookTheory/POP909-CL batches; bass and inversion are
  separate target families with independent masks.
- Tests: Phase 5B.1 covers exact/half-open alignment, typed offsets, masks,
  empty/conflicting/unaligned tasks, encoding sentinels, open strings, leakage,
  raw feature/topology mutation, malformed PyG batches, deterministic
  collation, instrumentation-based scaling, and separate raw-only and
  target-heavy benchmarks.
  Phase 5B.2 covers bounded dataset/cache/global-split/view/sampler/worker
  determinism, forged-sidecar and corruption rejection, narrow HookTheory
  quarantine, raw-only support, and complete 0/2-worker parity.
- Non-goals: advanced SSL.
- Acceptance: one mixed batch routes only available targets and preserves
  source/lineage grouping.

## Phase 6A — Trainable feature-only and local HeteroGNN baselines

- Status: Accepted and merged at
  `875dac3f83ab1a6cb3b3ece4875a5f55e3751409`.
- Goal: establish the first learned, CPU-compatible, raw-only local encoder
  and auxiliary-harmony baseline.
- Dependencies: Phase 5.
- Outputs: comparable feature-only and exact-relation local-GNN variants,
  one-row-per-node multiscale output, target-independent raw-candidate logits
  from source-native fully supervised task heads, tensorized supervision joins
  and group-balanced losses, local visible-input reconstruction,
  failure-atomic checkpoints, a canonical single-note diagnostic, and bounded
  CPU evidence.
- Tests: all node/edge stores, availability, routing and exclusions, local
  losses, raw-only candidate prediction, target-sidecar invariance,
  row-scaling operations, leakage, reconstruction/gradient coverage,
  checkpoint corruption/atomicity and round trip, deterministic one-batch
  overfit, canonical single-note sensitivity, separated oversmoothing, and
  benchmark.
- Non-goals: hierarchy/Transformer, shared cross-source heads, PU objectives,
  GraphMAE2/Hi-GMAE/UGMAE, PLL, critic, or quality scoring.
- Acceptance: a small raw graph batch trains end to end with target-independent
  candidate logits, masked harmonic loss joins, and preserved local rows.
  Forward/loss routing is tensorized over rows. POP909-CL boundary and no-chord
  remain excluded; absence of an observed event/span is never an implicit
  negative.

## Phase 6B — Deterministic hierarchy and bar+track Transformer

- Status: Accepted and merged at
  `b0e8e05ea0b11a06769475468af75b8438b4d45c`.
- Goal: add coarse context without erasing isolated local evidence.
- Dependencies: accepted Phase 6A.
- Outputs: deterministic hierarchy pooling, bar/track tokens, bar+track
  Transformer, song embedding, top-down fusion, and the controlled
  feature-only/local-GNN/hierarchical ablation.
- Tests: exact ownership, stable structured missing/invalid-store categories,
  non-mutating failures, forged-ownership rejection, deterministic membership
  and empty groups, sparse pooling, tensorized uneven packing/reference/CUDA
  parity, padding and end-to-end cross-sample isolation, retained local and
  top-down cardinality, candidate/target invariance, gradients/overfit, strict
  checkpoint round-trip/atomicity, per-path hierarchical sensitivity, and the
  controlled three-way plus uneven-sequence benchmarks.
- Non-goals: GraphMAE2 SSL, PLL, preference/quality critic, or shared
  pitch-class-set semantics.
- Acceptance: global context is available alongside retained note/onset/beat
  rows and isolated-note evidence remains directly inspectable. All six new
  Phase 6B contracts start at `1.0.0`; the 237 tiny and 79 isolated raw-only
  candidate counts remain unchanged.

## Phase 6C — Reproducible baseline training harness

- Status: Accepted and merged in PR #11 at
  `05501d8247f60d540e79841f89da42988a76b3e3`; the POP909-CL identity
  remediation is merged in PR #12 at
  `d3590d18550ba4a47bb8386786295d4905544fb5`.
- Goal: make the three unchanged Phase 6A/6B baselines reproducibly trainable
  on bounded fixtures and existing versioned corpus caches on CPU or CUDA.
- Dependencies: accepted and merged Phase 6B.
- Outputs: structured Hydra groups, official non-mutating batch device
  transfer, one-batch overfit evidence, minimal train/validation epochs,
  epoch-boundary resume, JSON/JSONL artifacts, compatibility-bound training
  checkpoints, and a global split-planning CLI.
- Tests: configuration/preset/objective composition, all three model
  selections, CPU-first device/sidecar/raw-graph boundaries, deterministic
  one-batch repetition, fixed no-replacement validation, row-weighted
  batch-partition-invariant metrics, fully failure-atomic checkpoint load,
  both epoch-commit crash windows, split isolation, artifacts/fingerprints,
  normal-hot-path synchronization instrumentation, and actual CLI-driven
  hierarchical CUDA AMP acceptance.
- Non-goals: Phase 7 SSL, corruption/remasking, PLL, PU objectives,
  preference/quality scoring, long corpus training, or semantic changes to
  models, targets, adapters, graphs, manifests, and caches.
- Acceptance: bounded CPU runs reproduce; production training defaults to
  supervised harmonic LR `3e-4`, while joint visible reconstruction is a
  named ablation; checkpoint reload and application failure are bit-exact;
  uninterrupted, resumed, and crash-recovered epoch metrics match without
  duplicate/lost rows; best selection uses only fixed validation; CUDA runs
  report device/VRAM evidence when hardware exists and otherwise skip
  explicitly.
- Full-corpus data gate: the 908-record POP index and unchanged 26,175-record
  HookTheory index pass one complete cache/split audit; the only exact score
  duplicate cluster is split-atomic and no full-corpus training is implied.

## Phase 6D-A — Supervised evaluation and performance evidence

- Status: Implemented on branch `phase/6d-supervised-evaluation`; bounded
  acceptance and Required CI are the merge gate.
- Goal: establish whether existing supervised heads outperform honest
  train-only trivial baselines and provide bounded evidence for CPU cost.
- Dependencies: accepted and merged Phase 6C and PR #12 identity remediation.
- Outputs: versioned candidate-first evaluator; fixed validation and explicit
  test CLI; categorical/multilabel dataset-task-class metrics; provenance-bound
  train priors; true dataset/encoding task macro summaries; deterministic JSON
  artifacts; bounded opt-in performance matrix with exclusive preparation,
  compute-only and end-to-end boundaries; optional deterministic indexed
  production-read-only subsets; non-binding epoch timing sidecar.
- Tests: hand-computed metrics, partition/order invariance, mask/target
  boundaries, target-mutation logit invariance, train-prior isolation,
  repeated bit-exact evaluation, test acknowledgement, source-native
  dataset/task isolation, direct-count F1 undefined semantics, true macro
  grouping, fixed memory, no repeated serial alignment, honest worker
  attribution, end-to-end loader timing, profiler opt-in, timing/checkpoint
  separation, and Phase 6C resume regression.
- Non-goals: full-corpus scan/training, new checkpoint selection, Phase 7 SSL,
  PLL, calibration, preference/critic work, or any model/data semantics.
- Acceptance: bounded synthetic evaluation and profiling complete; all
  artifacts expose contracts/fingerprints and no incompatible source heads are
  averaged. Optional real-cache smoke is explicitly bounded and read-only.

## Phase 7 — GraphMAE2-style SSL

- Goal: add masked observable-feature representation learning.
- Dependencies: Phase 6D-A evaluation evidence.
- Phase 7A status: accepted and merged in PR #15 at merge commit
  `a850207897b5abf6eebccf72d44b8814260323c6`; its concrete-CUDA,
  indexed-device, FP32-under-AMP, and split-evidence remediation is accepted
  and merged through PR #17 at main `5afec305cfa62ab2c200c5b1e7270ae35cd8a102`.
  The implementation is GraphMAE2-inspired, not a faithful reproduction.
  Exact bounded-run evidence belongs to `PHASE7A_SSL_BASELINE.md`.
- Outputs: masking views, remasked representation decoder, latent prediction
  losses, and a design gate before any normalized probabilistic
  masked-note/pitch-set decoder or deterministic PLL protocol.
- Phase 7A scope: only `note_pitch_group`, with note pitch/pitch-class/octave/
  track-relative-pitch and availability hidden, plus
  track-relative-pitch/availability on every unselected peer in an affected
  owner track and collateral owner-track mean/std/min/max pitch/availability.
  Peer and track collateral fields are not reconstruction targets. It uses
  deterministic no-replacement per-sample MaskPlans,
  `shared_stop_gradient_full_view`, note decoder re-mask views, and bar/song
  latent prediction.
- Prepared-plan boundary: plans and a versioned binding are derived from the
  validated CPU batch before device transfer. The binding covers identity,
  structure/ownership, stage, canonical epoch, seed, and plan fingerprints;
  prepared accelerator forward neither reconstructs plans from graph tensors
  nor materializes those tensors on the host.
- Execution: context mode
  `online_owner_track_bar_song_temporal_neighbors` keeps fully re-masked
  decoder predictions contextual. The production path is a raw-only
  dataset/collator over cached canonical pieces and never projects supervised
  targets. Reports distinguish source/cache/one-batch scope from production or
  full-corpus training. A fixed, disjoint validation set is evaluated before
  the first optimizer step and after every epoch; only its loss selects the
  best checkpoint. Atomic checkpoint/reload, exact epoch-boundary resume, and
  strict encoder-only transfer are included. Simple one-view/no-remask and
  main three-view/0.20-remask modes remain separately configurable.
- Tests: no masked-value leakage, raw-graph immutability, deterministic views,
  forged-binding rejection, no accelerator graph-to-host plan construction,
  worker/order/partition parity, dense-oracle parity for exact stage-wide
  note/bar/song anti-collapse aggregates, coherent pitch-mutation target and
  reconstruction-loss sensitivity, sign-agnostic correct-target-preference
  diagnostics, stop-gradient behavior, finite online gradients,
  checkpoint/resume, validation-only best selection, and untouched supervised
  heads on transfer.
- Deferred inside broader Phase 7: EMA target encoder remains an explicit
  ablation rather than part of Phase 7A.
- Non-goals: masked conditional likelihood, perplexity, PLL, preference,
  critic, or quality scoring.
- Acceptance: a deterministic multi-piece, multi-note fixture has disjoint
  train/validation identities, multitrack/multibar coverage, multiple primary
  masked rows, and nonzero peer-note/owner-track collateral masks. One-batch
  loss decreases under the Phase 7A-specific `3e-4` default rate. Under a
  fixed plan, coherent pitch mutation must leave the raw/masked-online path
  bit-exact while changing the hidden full-view target and reconstruction
  loss with positive target distance. Its signed correct-minus-mutated margin
  is recorded but is not a bounded acceptance criterion. Explicit optimizer
  overrides remain supported. Initial/final held-out note/bar/song target and
  prediction diagnostics remain finite and noncollapsed. Exact mergeable
  stage-wide aggregates retain no embedding history and are invariant to batch
  partition/order/workers. Reconstruction loss remains separate from masked
  conditional likelihood, and no probability factorization is assumed.
  One-batch plumbing, bounded held-out/non-collapse evidence,
  production-cache execution, and production/full-corpus claims remain four
  distinct scopes. Before PDMX, Phase 7 validates SSL mechanics only rather
  than full-scale effectiveness.

## Phase 8A — hierarchy-aware mask contracts, planners, and overlays

- Status: pre-merge remediation is implemented on branch
  `phase/8a-hierarchical-masking` in draft PR #16. Final-head review, both
  Required workflow runs, and independent exact-final RTX 3090 CUDA/AMP
  evidence are merge gates. This task does not merge the PR.
- Goal: extend the accepted Phase 7A view generator from independent note
  rows to coherent raw hierarchy descendants without changing objectives.
- Dependencies: accepted Phase 7A prepared-input and pitch-leakage contracts.
- Outputs: exact `independent_note_pitch` control dispatch plus
  onset-descendant, beat-descendant, contiguous-bar-span, and sparse
  track/bar-span policies; start-anchored semantics; deterministic policy
  mixtures; structured unavailable evidence; bounded near-optimal span
  selection whose retained pool is the seed-ranked top-K over the complete
  tolerance set and whose final choice uses a separate rank domain; unchanged
  pitch-only overlay;
  distinct hierarchy binding/output envelopes over the shared prepared-input
  attestation kernel; bounded oracle/benchmark; and a separate optional
  CUDA/AMP hardware-evidence artifact with per-policy/per-node-type bounded
  CPU-FP32/CUDA-FP32 numerical diagnostics.
- Tests: exact descendant/collateral oracles, polyphony, sustained-note
  exclusion, sample/track boundaries, mixtures/unavailability, worker/batch
  and fresh-process invariance, crafted epochs-0..255 positional-bias
  regression with canonical-prefix escape, late-bar/multi-track actual
  selections and complete retained-pool reachability, pool/error bounds,
  prepared mutation rejection, documented direct-CLI subprocess success and
  failure-closed source/report preflight, fixed numerical-tolerance boundary
  tests, Phase 7A bit-exact CPU/CUDA-AMP compatibility, and bounded
  existing-objective forward/backward smoke. Cross-backend floating outputs
  are bounded diagnostics; all semantic/security invariants and same-device
  replay remain exact.
- Non-goals: new objective heads, quality improvement claims, theory/PDMX
  integration, PLL, critic learning, or production training.
- Acceptance: hierarchy-aware views are deterministic, sparse, leakage-closed,
  failure-closed, and model-ready on bounded variable graphs; the raw graph,
  canonical/cache/split contracts, model state/checkpoints, and Phase 7A
  control artifacts remain unchanged. Portable CPU evidence and optional
  hardware-dependent CUDA identity/timing/VRAM evidence remain separate.

## Phase 8B — multi-level objectives and comparison

- Status: Phase 8B.1 is merged through PR #18. Commit `7365286` added Phase
  8B.2A control-plane primitives; the current remediation completes the
  executable official-engine DAG, actual-schedule/data attestation,
  fixed-validation evaluation, sufficient-statistics aggregation, paired-seed
  configuration selection, resumable cell manifests, and real bounded CLI
  acceptance on `phase/8b2a-scientific-comparison-protocol`. Production/PDMX
  effectiveness evidence and optional current-head CUDA confirmation remain
  pending. Before the
  original integration remediation, the Hydra
  group/builder/bounded runner existed but `ssl.run`/`ssl.engine` still used
  the old model and Phase 7A forward unconditionally.
- Goal: add independently ablatable onset/beat/bar/track objective families
  and compare them against Phase 7A/8A controls.
- Dependencies: accepted Phase 8B.1 merge through PR #18 at main
  `387b5bc`.
- Phase 8B.1 outputs: versioned exact-identity eligibility/binding,
  independently weighted multi-level heads/losses, six Hydra modes, old
  checkpoint transfer, strict new checkpoint binding, and deterministic
  bounded train/held-out mechanics comparison; remediated official one-batch/
  multi-epoch/fixed-validation/resume routing; an independent masking group;
  the old-model Phase 8A mask-only control; and corrected family-global
  cross-policy numerator/denominator aggregation with each family weight
  applied once; plus FP32-safe new latent heads, honest applied/skipped step
  accounting, active/inactive parameter-update evidence, and an independent
  archived RTX FP32/AMP runner.
- Phase 8B.2A outputs: `Phase8B2ComparisonProtocol@1.2.0`, distinct natural
  and encoder-forward-matched analyses, actual target-free paired schedules,
  metadata-derived data identities, exact step/forward instrumentation,
  official frozen/full/scratch transfer, fixed candidate-first validation,
  sufficient-statistics piece bootstrap, paired-seed configuration selection,
  multi-checkpoint single-use test lock, resumable cell DAG, immutable complete
  artifacts, executable Hydra actions/presets, and optional CUDA/AMP evidence.
  Production paths additionally bind metadata planning and official-engine
  evidence through `Phase8B2DataSemanticProjection@1.0.0`; test membership
  metadata is resolved for the lock without test inference, targets, metrics,
  or serialized full test identities.
- Phase 8B.2 future output: execute the locked protocol at appropriate scale,
  including the Phase 10 PDMX rerun, before any effectiveness or curriculum
  conclusion.
- Tests: level-specific target/denominator semantics, unavailable-level
  handling, exact alignment/sample isolation, independent ablation,
  checkpoint compatibility/atomicity, bounded optimization, optional CUDA AMP,
  zero retained report tensors, real CLI routing, fail-before-optimizer
  incompatibility, complete step/forward/policy/objective/masked-entity
  accounting, independent repeated-bar manual oracle, one packed objective-
  metrics D2H transfer at most per CPU batch, CPU FP16 overflow/safe-scale and
  public scaler skip/apply oracles, all-family CUDA+AMP real-update checks,
  FP32/AMP structural parity, and exact two-epoch resume.
- Non-goals: claiming scaled effectiveness before the Phase 10
  raw-compatible PDMX projection and rerun; Phase 9, PLL,
  preference critic, quality scoring, or production/full-corpus SSL training.
- Acceptance: objective families remain independently ablatable and their
  held-out mechanics/comparisons are reported without likelihood, critic, or
  quality-score interpretation. Bounded variants with different scheduled
  forward counts are not described as compute matched or as effectiveness
  evidence.
- Phase 8B.2A acceptance: the public bounded CPU CLI covers two paired seeds,
  8 SSL cells, 8 encoder exports, 18 downstream cells, 18 validation cells,
  actual schedule parity, interrupted/resumed execution, deterministic launch
  permutation, real checkpoint-to-evaluation verification and aggregation,
  stale/incomplete rejection, scratch,
  Phase 7A, Phase 8A mask-only, a single-level objective, equal-weight,
  frozen/full transfer contracts, exact compute accounting, launch-order
  invariance, resume binding, leakage mutation evidence, validation-only
  selection, test-lock negatives, aggregation, piece uncertainty, and
  diagnostics. It does not require SSL to beat scratch.

## Phase 9 — Dilemmadata adapter and theory supervision

- Goal: add local key, harmony, cadence, phrase, and note-theory targets.
- Dependencies: canonical/graph/model foundations.
- Phase 9A: completed evidence audit and raw/target contract on branch
  `phase/9a-dilemmadata-evidence-contract`. The exact v1.0 snapshot has 1,633
  primary records and 2,880,723 note rows across two TSV dialects. Phase 9A
  uses an acceptance-backed separate clean-checkout comparison. Its narrow
  MIDI note-event multiset fingerprint is conservative split-grouping evidence,
  not full input identity. Phase 9A adds no production adapter, heads, losses,
  or training runtime.
- Phase 9B.1: production raw adapter and SSL-ready corpus path implemented.
  It supplies two pinned streaming dialect parsers, target-blind canonical
  conversion, structured quarantine, target-independent cache identity,
  transitive group-safe splits, and an official Phase 8B one-step real-record
  smoke. Tie, zero-duration grace, meter/bar, and required-default policies are
  explicit and theory-independent. Blocking remediation makes policy config
  and discovered records failure-closed, distinguishes key-signature
  conflicts, repeats the second cache build from source, and gates readiness on
  a committed deterministic production manifest.
- Phase 9B.2A: production source-native target sidecars and exact alignment
  implemented as an external 22-task registry extension. Raw cache/graph/split
  identity stays unchanged; an independent target-neutral raw oracle binds
  every ordered source row to its exact canonical note before target parsing;
  no heads or losses are added.
- Phase 9B.2B: implemented the first safe supervised Dilemmadata path. Only AN
  and DLC chord quality/inversion receive distinct categorical heads; five PU
  and 13 open-string tasks remain headless. Immutable sidecar caching,
  source-entry-normalized loss/evaluation, encoder-only transfer, official
  Hydra presets, and a fixed RTX 3090 scratch-versus-SSL plan are present.
  Bounded tests establish plumbing only; the long independent pilot remains a
  post-merge execution task.
- Phase 9B.2C: implemented a committed, bounded RTX 3090 scratch runner and an
  independent source-free evidence verifier. The gate exercises all four
  active heads, CUDA+AMP updates, atomic checkpoint reload, and official
  validation while test remains closed. The PR stays draft until independently
  executed hardware evidence passes; this phase does not run the long pilot.
- Tests: target deletion/replacement/reordering leaves canonical, graph, graph
  fingerprint, and model-input fingerprint unchanged; candidate
  multiple-analysis groups stay grouped and are rechecked using the Phase 9B
  canonical/model-input fingerprint; raw pitch/onset/duration/meter/tie/voice
  changes alter raw evidence or quarantine, and cache/load/spawn paths preserve
  exact parity.
- Non-goals: Phase 8/CUDA lifecycle changes, preference critic, and quality
  scoring.
- Acceptance: harmony/key/cadence/phrase/Roman-numeral columns remain targets,
  raw-MIDI inference requires none of staff, voice, spelling, `step`, `alter`,
  or `tpc`, and no release split separates a transitive source component.

### Phase 9C-A — executable one-seed SSL → Dilemmadata pilot

- Status: implemented as an executable production control plane; bounded
  fixture acceptance is local, while independent RTX profile and the actual
  one-seed pilot remain pending.
- Goal: compare scratch, Phase 7A control, Phase 8A mask-only, and Phase 8B
  multilevel-equal under seed 17, paired schedules/initialization, and 12
  observed encoder forwards per logical SSL update.
- Outputs: `plan/profile/run/resume/aggregate/select/verify`, target-blind
  three-source SSL mixture with exact-assignment composition of the two existing
  split manifests, an immutable raw-structural eligibility view shared by all
  SSL variants without repartitioning, encoder-only transfer, scratch/pretrained frozen probes and
  full fine-tunes, fixed-update `last.pt` comparison on the 71-record validation,
  component bootstrap,
  immutable test lock, resumable cells, bundle verifier, plots/tables, and a
  standalone exact-clean-HEAD RTX 3090 runner.
- Optional: onset/beat/bar/track latent ablations are runner-supported but
  excluded from the primary preset.
- Non-goals: production execution inside this change, test evaluation,
  multi-seed inference, PDMX/Phase 10, critic/quality score, or superiority
  claims.
- Next gate: independent RTX profile; production batch size and budgets are
  selected explicitly from that artifact before a separate pilot `run`.

### Phase 9C-B — diagnostic onset-BiGRU decoder matrix

- Status: implementation and both blocking profile remediations are complete
  locally. After the schedule fix, the rerun at `9de8f34` passed both MLP
  cells and trained/checkpointed `scratch_onset_bigru`, then failed only because
  evaluation ignored its top-level decoder contract and rebuilt an MLP before
  strict loading. Checkpoint reconstruction is now typed and decoder/state
  consistent; the SSL encoder export is explicit and structurally checked in
  preflight. A fresh-root profile rerun and the explicit one-seed production
  matrix remain pending.
- Goal: isolate whether the independent Dilemmadata MLP readout bottlenecks
  useful sequential SSL information.
- Outputs: unchanged `decoder.kind=mlp`, optional raw-only
  `decoder.kind=onset_bigru`, onset/beat/bar residual context without changing
  candidate semantics, four-cell scratch/SSL × MLP/BiGRU matrix, fixed-update
  `last.pt` comparison, complete imbalance diagnostics, profile/run/resume/
  aggregate/verify, immutable evidence hashes, and exact RTX 3090 wrapper.
- Non-goals: cache/split/target/class-weight changes, new SSL objectives,
  attention/Transformer decoder, multi-seed or test evaluation, PDMX, and
  Phase 10.
- Next gate: code review, then independent RTX profile, then an explicit four-
  cell seed-17 run. No scientific claim is authorized from one seed.

### Phase 9C-C — one-seed scratch-vs-SSL MLP convergence diagnostic

- Status: control plane and bounded CPU acceptance implemented on a stacked
  branch; production RTX execution and scientific interpretation are pending.
- Goal: determine whether the Phase 9C-B scratch/SSL MLP comparison was stopped
  too early at 3,000 updates.
- Outputs: exactly two MLP full-fine-tune cells, one continuous 9,000-applied-
  update epoch, 100-update train telemetry, atomic 1,000-update mid-epoch
  checkpoints/resume, fixed validation milestones 0/1,000/3,000/6,000/9,000,
  convergence facts/deltas, immutable bundle, independent verifier and
  run/resume/verify RTX 3090 wrapper.
- Non-goals: BiGRU, frozen probes, new SSL objectives or pretraining, additional
  seeds, model/data/cache/split/class-weight changes, test evaluation,
  automatic plateau decisions, production execution in CI, PDMX or Phase 10.
- Next gate: Required CI and draft-PR review, then an exact-head fresh-root RTX
  run. Interpret convergence only after the verified evidence bundle exists.

## Phase 10 — PDMX adapter and large-scale SSL cache

- Goal: support scalable role-agnostic public-domain score pretraining and
  future actual-score completion through a raw-MIDI-compatible projection.
- Dependencies: SSL and canonical cache contracts.
- Outputs: PDMX adapter, filters, windowed/versioned cache, and full-scale
  rerun/evaluation entry points for the accepted Phase 7–8 SSL objectives on
  the PDMX raw-compatible corpus.
- Tests: timing conversion, invalid-score filtering, cache compatibility, and
  reproducible scaled SSL evaluation configuration.
- Non-goals: using ratings as absolute quality labels.
- Acceptance: a small licensed subset preprocesses reproducibly and optional
  notation/role metadata can be removed without changing mandatory inputs;
  accepted Phase 7–8 objectives are rerun and evaluated at full scale before
  scaled SSL or Phase 11 objective conclusions.

## Phase 11 — UGMAE-inspired adaptive and structural objectives

- Goal: add adaptive masking and optional structure consistency.
- Dependencies: stable SSL baseline.
- Outputs: adaptive policies, structural/consistency losses, and ablatable
  coherent onset/beat/bar, pitch-only-with-visible-rhythm, and track/span masks.
- Tests: probability bounds, deterministic evaluation, ablation toggles.
- Non-goals: preference deployment.
- Acceptance: objectives train without collapsing mandatory graph structure.

## Phase 12 — Preference critic and real generator outputs

- Goal: learn aspect and pairwise preference scores from real candidates.
- Dependencies: trained shared encoder and grouped preference data.
- Outputs: aspect heads, preference head, calibrated pairwise losses, and
  optional separately identified likelihood/fragility signals from accepted
  probabilistic experiments.
- Tests: pair-swap invariance, group-aware sampling, one-batch ranking overfit.
- Non-goals: universal genre-independent MOS.
- Acceptance: held-out prompt-group ranking beats defined baselines; SSL
  reconstruction loss is not treated as a quality score.

## Phase 13 — Audio-aesthetic teacher labels and MIDI surrogate

- Goal: approximate renderer-based aesthetic signals without mandatory rendering.
- Dependencies: preference critic and controlled renderer provenance.
- Outputs: teacher-label pipeline and symbolic surrogate head.
- Tests: provenance, cache identity, teacher/student agreement.
- Non-goals: treating audio aesthetics as music theory.
- Acceptance: surrogate evaluation is reported separately and reproducibly.

## Phase 14 — Raw-MIDI inference and GRPO integration

- Goal: expose deployable scoring and reward APIs.
- Dependencies: validated critic checkpoint.
- Outputs: MIDI inference CLI/API, structured output, policy integration hooks.
- Tests: unlabeled type-0/type-1 MIDI, missing metadata, batch ranking.
- Non-goals: changing model training objectives silently.
- Acceptance: inference requires no gold theory, chord track,
  melody/accompaniment/bass role, voice/staff label, or semantic segmentation.

## Phase 15 — Ablations, calibration, and human evaluation

- Goal: support defensible research conclusions and deployment thresholds.
- Dependencies: all claimed components.
- Outputs: architecture/data ablations, calibration, robustness, human studies,
  and the required harmonic comparisons: no supervision, HookTheory-only,
  POP909-CL-only, combined supervision, label-only versus pitch-class-set
  heads, SSL without PLL, probabilistic PLL, PLL plus preference critic,
  track/metadata perturbations, and melody-only versus combined-score versus
  heterogeneous raw-MIDI evaluation.
- Tests: reproducible evaluation manifests and leakage audits.
- Non-goals: adding unablated features.
- Acceptance: every major claim has an ablation and uncertainty report; PLL is
  normalized and bias-audited rather than presented as complete aesthetic
  quality, and a blind raw-MIDI set verifies role-agnostic inference.
