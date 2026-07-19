# Music Critic V2 Engineering Roadmap

The scientific detail in `IMPLEMENTATION_PLAN.md` remains authoritative. This
document is the phase execution checklist.

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

- Status: In progress
- Sequence: Phase 2A.1 and Phase 2B.0 are completed; Phase 2B.1 is in review.
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

- Status: In review
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
  rejects mismatched structure rows. Phase 2B.1 remains In review.
- Non-goals: MIDI rendering, chord-note synthesis, section alignment, deferred
  chord-field interpretation, graphs, datasets, models, or training.

The graph phase, model phases, and training phases remain pending. Phase 2B.1
must remain in review until its implementation is separately accepted.

## Phase 3 — Raw graph builder

- Goal: construct inference-safe heterogeneous graphs.
- Dependencies: Phases 1–2.
- Outputs: `song/track/bar/beat/onset/note` graph and versioned metadata.
- Tests: edge validity, temporal order, sustained activity, no target leakage.
- Non-goals: learned encoders.
- Acceptance: equivalent raw schemas produce consistent graph structure.

## Phase 4 — POP909 adapter

- Goal: validate track-aware canonical and graph paths on multitrack pop data.
- Dependencies: Phases 1–3.
- Outputs: POP909 parsing, alignment diagnostics, role targets, manifests.
- Tests: tempo/annotation alignment and version grouping.
- Non-goals: large-scale training.
- Acceptance: leakage-safe POP909 graphs pass validation.

## Phase 5 — Multi-source dataset and collator

- Goal: batch heterogeneous task availability across datasets.
- Dependencies: adapter and graph phases.
- Outputs: datasets, samplers, collator, task routing.
- Tests: masks, empty tasks, dataset balancing, deterministic sampling.
- Non-goals: advanced SSL.
- Acceptance: one mixed batch routes only available targets.

## Phase 6 — Baseline local GNN, hierarchy, and bar Transformer

- Goal: implement the minimum hybrid encoder.
- Dependencies: Phase 5.
- Outputs: feature encoder, local GNN, pooling, bar Transformer, fusion.
- Tests: shapes, empty node types, checkpoint round trip, one-batch overfit.
- Non-goals: GraphMAE2/Hi-GMAE/UGMAE extensions.
- Acceptance: a small raw graph batch trains end to end.

## Phase 7 — GraphMAE2-style SSL

- Goal: add masked observable-feature representation learning.
- Dependencies: Phase 6.
- Outputs: masking views, remasked decoder, latent prediction losses.
- Tests: no masked-value leakage, deterministic views, stop-gradient behavior.
- Non-goals: quality scoring.
- Acceptance: tiny reconstruction overfit and stable held-out metrics.

## Phase 8 — Hi-GMAE-style hierarchical masking

- Goal: mask coherent descendants and learn multi-level representations.
- Dependencies: Phase 7.
- Outputs: hierarchy-aware masks and multi-level objectives.
- Tests: descendant masks, non-degenerate views, level-specific losses.
- Non-goals: theory corpus integration.
- Acceptance: hierarchical masking works on variable graph sizes.

## Phase 9 — Dilemmadata adapter and theory supervision

- Goal: add local key, harmony, cadence, phrase, and note-theory targets.
- Dependencies: canonical/graph/model foundations.
- Outputs: adapter, annotation views, theory heads and masked losses.
- Tests: alternative-analysis grouping, span compression, no-label loss zero.
- Non-goals: preference critic.
- Acceptance: theory heads overfit a tiny masked batch without raw leakage.

## Phase 10 — PDMX adapter and large-scale SSL cache

- Goal: support scalable public-domain score pretraining.
- Dependencies: SSL and canonical cache contracts.
- Outputs: PDMX adapter, filters, windowed/versioned cache.
- Tests: timing conversion, invalid-score filtering, cache compatibility.
- Non-goals: using ratings as absolute quality labels.
- Acceptance: a small licensed subset preprocesses reproducibly.

## Phase 11 — UGMAE-inspired adaptive and structural objectives

- Goal: add adaptive masking and optional structure consistency.
- Dependencies: stable SSL baseline.
- Outputs: adaptive policies and structural/consistency losses.
- Tests: probability bounds, deterministic evaluation, ablation toggles.
- Non-goals: preference deployment.
- Acceptance: objectives train without collapsing mandatory graph structure.

## Phase 12 — Preference critic and real generator outputs

- Goal: learn aspect and pairwise preference scores from real candidates.
- Dependencies: trained shared encoder and grouped preference data.
- Outputs: aspect heads, preference head, calibrated pairwise losses.
- Tests: pair-swap invariance, group-aware sampling, one-batch ranking overfit.
- Non-goals: universal genre-independent MOS.
- Acceptance: held-out prompt-group ranking beats defined baselines.

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
- Acceptance: inference requires no gold theory or semantic segmentation.

## Phase 15 — Ablations, calibration, and human evaluation

- Goal: support defensible research conclusions and deployment thresholds.
- Dependencies: all claimed components.
- Outputs: architecture/data ablations, calibration, robustness, human studies.
- Tests: reproducible evaluation manifests and leakage audits.
- Non-goals: adding unablated features.
- Acceptance: every major claim has an ablation and uncertainty report.
