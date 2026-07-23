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

- Status: Pending.
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

## Phase 5 — Multi-source dataset and collator

- Goal: batch heterogeneous task availability across datasets.
- Dependencies: adapter and graph phases.
- Outputs: datasets, samplers, collator, task routing, a common harmonic target
  ontology, dataset-specific annotation views, availability masks, and
  per-target provenance for mixed HookTheory/POP909-CL batches.
- Tests: masks, empty/ambiguous tasks, no unavailable-as-negative conversion,
  lineage-safe grouping, dataset balancing, deterministic sampling.
- Non-goals: advanced SSL.
- Acceptance: one mixed batch routes only available targets and preserves
  source/lineage grouping.

## Phase 6 — Baseline local GNN, hierarchy, and bar Transformer

- Goal: implement the minimum hybrid encoder.
- Dependencies: Phase 5.
- Outputs: feature encoder, local GNN, pooling, bar Transformer, fusion, and
  auxiliary boundary/root/quality/pitch-class-set/bass-inversion/no-chord
  heads.
- Tests: shapes, empty node types, checkpoint round trip, one-batch overfit.
- Non-goals: GraphMAE2/Hi-GMAE/UGMAE extensions.
- Acceptance: a small raw graph batch trains end to end with masked harmonic
  routing; Phase 6 is not described as a quality critic.

## Phase 7 — GraphMAE2-style SSL

- Goal: add masked observable-feature representation learning.
- Dependencies: Phase 6.
- Outputs: masking views, remasked representation decoder, latent prediction
  losses, and a design gate before any normalized probabilistic
  masked-note/pitch-set decoder or deterministic PLL protocol.
- Tests: no masked-value leakage, deterministic views, stop-gradient behavior.
- Non-goals: quality scoring.
- Acceptance: tiny reconstruction overfit and stable held-out metrics;
  reconstruction loss is reported separately from masked conditional
  likelihood, and no final probability factorization is assumed.

## Phase 8 — Hi-GMAE-style hierarchical masking

- Goal: mask coherent descendants and learn multi-level representations.
- Dependencies: Phase 7.
- Outputs: hierarchy-aware onset/beat/bar-span masks, pitch-only masks with
  visible rhythm, track/span masks, and multi-level objectives.
- Tests: descendant masks, non-degenerate views, level-specific losses.
- Non-goals: theory corpus integration.
- Acceptance: hierarchical masking works on variable graph sizes and exact
  objective families remain independently ablatable.

## Phase 9 — Dilemmadata adapter and theory supervision

- Goal: add local key, harmony, cadence, phrase, and note-theory targets.
- Dependencies: canonical/graph/model foundations.
- Outputs: adapter, annotation views, theory heads, masked losses, and a
  raw-MIDI-compatible projection that does not assume staff, voice, spelling,
  `step`, `alter`, or `tpc` at inference.
- Tests: alternative-analysis grouping, span compression, no-label loss zero.
- Non-goals: preference critic.
- Acceptance: theory heads overfit a tiny masked batch without raw leakage;
  harmony/key/cadence/phrase/Roman-numeral columns remain targets.

## Phase 10 — PDMX adapter and large-scale SSL cache

- Goal: support scalable role-agnostic public-domain score pretraining and
  future actual-score completion through a raw-MIDI-compatible projection.
- Dependencies: SSL and canonical cache contracts.
- Outputs: PDMX adapter, filters, windowed/versioned cache.
- Tests: timing conversion, invalid-score filtering, cache compatibility.
- Non-goals: using ratings as absolute quality labels.
- Acceptance: a small licensed subset preprocesses reproducibly and optional
  notation/role metadata can be removed without changing mandatory inputs.

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
