# Architecture Decision Log

This log is append-only.

## 2026-07-16 — ADR-001: Separate clean repository

- Status: Accepted
- Context: V1 contains HookTheory-specific data, graph, teacher, corruption,
  and observer assumptions.
- Decision: Build V2 in a separate repository rather than refactoring V1 in
  place.
- Consequences: Migration must be explicit; V1 remains available for audit and
  comparison without constraining V2 packaging.

## 2026-07-16 — ADR-002: Legacy is read-only and non-runtime

- Status: Accepted
- Context: The legacy worktree already contains valuable experiments and
  uncommitted state.
- Decision: Never modify or import the legacy repository from V2.
- Consequences: Adapt concepts selectively and ensure V2 runs without V1.

## 2026-07-16 — ADR-003: Package name has no `_v2` suffix

- Status: Accepted
- Context: The whole repository is the V2 system.
- Decision: Use the import package `music_critic`.
- Consequences: Package paths remain concise and do not perpetuate migration
  naming in the long-term API.

## 2026-07-16 — ADR-004: Datasets stay outside Git

- Status: Accepted
- Context: Symbolic corpora and rendered artifacts are large and may have
  separate licenses.
- Decision: Ignore datasets, outputs, audio, MIDI, caches, and checkpoints.
- Consequences: Tests use only tiny synthetic fixtures.

## 2026-07-16 — ADR-005: Raw MIDI inference is mandatory

- Status: Accepted
- Context: V1 teacher inputs require annotations absent from generated MIDI.
- Decision: Mandatory V2 inputs and graph structure must be reproducible from
  unlabeled MIDI.
- Consequences: Gold semantic annotations cannot be required at inference.

## 2026-07-16 — ADR-006: Theory annotations are targets

- Status: Accepted
- Context: V1 encodes scale degree, chord theory, key, and section labels as
  node inputs.
- Decision: Theory annotations are auxiliary targets unless a later decision
  explicitly changes a narrowly scoped experiment.
- Consequences: Missing labels require masks; train/inference paths stay aligned.

## 2026-07-16 — ADR-007: Exact rational timing begins in Phase 1

- Status: Accepted
- Context: V1 uses float beats and epsilon-based grouping.
- Decision: Canonical V2 timing will use exact quarter-note rationals.
- Consequences: Phase 0 documents the contract but implements no timing class.

## 2026-07-16 — ADR-008: Bootstrap contains no model implementation

- Status: Accepted
- Context: Data and interface decisions must precede model code.
- Decision: Phase 0 contains only packaging, documentation, audit, and tests.
- Consequences: Torch, PyG, Hydra, MIDI, and audio libraries are not runtime
  dependencies.

## 2026-07-16 — ADR-009: Phase 1 schema is immutable and standard-library-only

- Status: Accepted
- Context: Canonical records must be safe to share across adapters,
  serialization, windowing, graph construction, and tests without hidden
  mutation or dependency coupling.
- Decision: Schema `2.0.0` uses frozen, slotted dataclasses and tuple-valued
  collections. The schema, timing, validation, and serialization modules use
  only the Python standard library.
- Consequences: Adapters may use mutable builders internally but return deeply
  immutable canonical records. JSON arrays map to tuples and no tensor or MIDI
  library type appears in the public schema.

## 2026-07-16 — ADR-010: Canonical IDs are stable prefixed strings

- Status: Accepted
- Context: Integer array positions are not stable under sorting, windowing,
  adapter conversion, or serialization.
- Decision: Entity IDs use fixed type prefixes and deterministic string local
  IDs. IDs are globally unique within a piece and are never rewritten by
  sorting or ordinary window selection.
- Consequences: All references and target alignments are explicit. Newly
  synthesized or clipped entities need a new deterministic ID and provenance
  link rather than reusing an unrelated index.

## 2026-07-16 — ADR-011: Raw records exclude semantic theory and role labels

- Status: Accepted
- Context: Raw MIDI inference cannot supply gold harmony, local key, cadence,
  phrase, section, scale-degree, non-chord-tone, or semantic track-role labels.
- Decision: Raw note and track records contain observations only. Theory and
  role supervision is represented by typed `TargetArray` records with entity
  IDs, values, masks, confidence, per-entry source, and per-entry provenance.
- Consequences: Missing entries are null with `mask=false`, never implicit
  negative classes. Categorical, scalar, multi-label, and distribution targets
  have explicit serialized encodings.

## 2026-07-16 — ADR-012: Canonical JSON is strict and deterministic

- Status: Accepted
- Context: Silent field passthrough and best-effort version loading make caches
  ambiguous and prevent reproducible round trips.
- Decision: Strict readers and writers accept exactly schema version `2.0.0`,
  reject unknown or missing fields, require normalized rational objects, and
  serialize every field explicitly and deterministically.
- Consequences: Compatibility is never inferred from a matching major version.
  Future schema changes require a new version, ADR, migration path, and tests.
  A generic `dataclasses.asdict()` result is not the public contract.

## 2026-07-16 — ADR-013: Musical time and event semantics are explicit

- Status: Accepted
- Context: Pickups, tempo/meter changes, sustained notes, grace notes, and
  percussion must survive canonicalization without float equality or
  dataset-specific conventions.
- Decision: Time is normalized immutable rational quarter-note units starting
  at zero, including pickups. Pickups use actual duration plus a metric offset.
  Notes are half-open intervals, remain unsplit across bars, and may overlap.
  Zero duration is allowed only for grace notes. Canonical tracks are
  homogeneous for percussion. Same-onset event application order is meter,
  tempo, then key signature.
- Consequences: Adapters insert explicit provenance-bearing defaults when
  initial tempo or meter is absent, split mixed pitched/percussion source tracks
  deterministically, and never depend on negative pickup time.

## 2026-07-16 — ADR-014: Validation separates invalid data from diagnostics

- Status: Accepted
- Context: Callers need complete structured diagnostics while still being able
  to reject unsafe canonical data.
- Decision: `validate_piece` returns a deterministic `ValidationReport`.
  `validate_or_raise` raises `CanonicalValidationError` containing that report
  only when errors exist. Errors cover contract, reference, timing, range,
  target, and provenance violations; warnings cover valid but noteworthy source
  conditions.
- Consequences: Warnings never invalidate a piece, persisted `QualityFlag`
  records are distinct from computed validation issues, and callers can report
  all failures in one pass.

## 2026-07-16 — ADR-015: Targets preserve alternative annotation views

- Status: Accepted
- Context: Dilemmadata and other corpora may provide multiple legitimate
  analyses of the same entities. One target array per task cannot represent
  this disagreement without discarding information.
- Decision: Every `TargetArray` has a globally unique `target_id` and an
  optional stable `annotation_view_id`. Target uniqueness is enforced on
  `(task, annotation_view_id)`, while the same aligned entity may appear in
  different views.
- Consequences: Alternative analyses remain separate records and remain grouped
  with the same piece/source group. They are not converted into probability
  distributions unless the source explicitly supplies a distribution.

## 2026-07-16 — ADR-016: Available target confidence may be unknown

- Status: Accepted
- Context: Many human and dataset labels are available without a calibrated
  numeric confidence estimate.
- Decision: For `mask=true`, value, source, and provenance are required while
  confidence may be null. Non-null confidence must be finite and in `[0,1]`.
  For `mask=false`, value, confidence, source, and provenance are all null.
- Consequences: Null confidence means unknown numeric confidence only; it is
  neither missing supervision nor an implicit value of zero or one.
  `LOW_CONFIDENCE_TARGET` applies only to non-null confidence below `0.5`.

## 2026-07-16 — ADR-017: Observable modes and adapter diagnostics remain extensible

- Status: Accepted
- Context: Restricting key-signature mode to major/minor discards source
  observations, while a closed quality-flag vocabulary would require a schema
  migration for routine adapter diagnostics.
- Decision: Key-signature mode includes the common diatonic modes plus `other`
  and `unknown`, with source-specific notation retained in `raw_value`.
  `QualityFlag.code` is an open stable lowercase dotted identifier validated by
  syntax; `ValidationCode` remains closed.
- Consequences: Modal key-signature metadata remains observable rather than
  becoming a local-key label. Adapters may add namespaced diagnostics without
  changing schema version `2.0.0`.

## 2026-07-16 — ADR-018: Schema 2.0.0 limits spelling alterations to semitones

- Status: Accepted
- Context: The integer `spelling_alter` field cannot represent quarter-tone or
  other microtonal notation faithfully.
- Decision: Keep `spelling_alter: int | None` for schema `2.0.0`. Unsupported
  microtonal source notation is preserved in provenance, accompanied by a
  namespaced quality flag, and is never silently rounded.
- Consequences: Microtonal spelling is an explicit accepted limitation requiring
  a future versioned extension if first-class support is needed.

## 2026-07-16 — ADR-019: Trailing silence excludes structural coverage

- Status: Accepted
- Context: Bars and beats normally cover the full piece duration, making a
  structural-end definition of trailing silence unreachable.
- Decision: `PIECE_TRAILING_SILENCE` compares piece duration with the latest end
  of positive-duration notes or observation annotations. Structural events,
  target-alignment spans, point annotations, and zero-duration grace notes do
  not extend sounding/observation content.
- Consequences: The warning is exact and reachable. Percussion counts as
  sounding content; structural-only positive-duration pieces emit both empty
  piece and trailing-silence warnings.

## 2026-07-16 — ADR-020: Semantic values and annotation views validate deterministically

- Status: Accepted
- Context: JSON runtime-type failures and correctly typed but semantically
  invalid values need distinct diagnostics. Annotation views also require
  deterministic lexical rules, and the canonical fixture should exercise
  multiple valid analyses directly.
- Decision: Reserve `JSON_TYPE_INVALID` for JSON values whose runtime type cannot
  satisfy the declared schema type. Use `FIELD_VALUE_INVALID` as the fallback
  error for declared semantic constraints without a more specific code,
  including key-signature, spelling, provenance timestamp/checksum, open-string,
  and programmatic enum/Literal violations. Non-null `annotation_view_id` values
  must be non-empty, already trimmed, free of ASCII control characters, and are
  compared case-sensitively; view-specific violations use
  `TARGET_VIEW_INVALID`.
- Consequences: `validate_piece` checks programmatically constructed records as
  strictly as decoded records. The canonical example contains default and
  alternative chord-quality views plus the track-role target, providing the
  normative three-target round-trip fixture for Phase 1B.

## 2026-07-19 — ADR-021: Target families share alignment spans; meter changes may end bars

- Status: Accepted
- Context: HookTheory local-key and chord entities each align several target
  tasks to one source span. Exact source meter changes can also occur before a
  nominal bar completes, while the canonical piece must preserve the event and
  still validate without moving or padding it.
- Decision: An annotation-span target task may equal its target-alignment
  annotation type or extend it with a dotted subtask suffix. A shortened
  non-pickup bar is valid when it ends exactly at the next meter-event onset;
  piece duration itself is also a valid terminal meter boundary.
- Consequences: Related theory targets share stable source-span IDs without
  duplicating annotations. Exact mid-bar meter changes create diagnosed
  incomplete bars while preserving contiguous bar/beat coverage and schema
  version `2.0.0`.

## 2026-07-19 — ADR-022: HookTheory timing, tempo, and pitch use upstream metric semantics

- Status: Accepted; Phase 2B.1 is Accepted and Completed at implementation
  `3898b168063094b87e5ca5d88aae0317c1562c3f`.
- Context: The first adapter implementation treated every TheoryTab beat as one
  quarter note, every BPM as quarter-note BPM, and melody octave zero as MIDI
  72. Structural validation did not establish those musical meanings.
- Evidence: The raw/simplified crosswalk contains 27,216 exact paired meter
  regions and no value mismatches. Pinned Sheet Sage
  `bbdd7b7b6a5fb845828f82790acdceb03a197779` defines compound meters through
  three secondary pulses and converts notes with active-scale intervals. The
  complete melody crosswalk pairs 1,211,093 notes with zero pitch-class or
  relative-octave mismatches. Refined alignment has no compound-meter interval;
  72 eligible user-alignment intervals select felt-pulse tempo with 0.39%
  median relative error, versus 50.04% for quarter-BPM and 200.07% for
  raw-beat-BPM. Sheet Sage `Note.as_midi_pitch()` establishes MIDI 60 for
  relative octave zero. The one distributed MIDI match is postprocessed and is
  not used as raw source truth.
- Decision — meter labels (`observed_corpus_semantics`): numerator is
  `numBeats`; denominator is 4 for `beatUnit=1` and 8 for `beatUnit=3`.
- Decision — time coordinates (`upstream_semantics`): raw beat 1 maps to qn 0;
  `qn_per_raw_beat` is 1 for `beatUnit=1` and 1/2 for `beatUnit=3`. Changes are
  integrated piecewise, including event ends and `endBeat`.
- Decision — tempo (`upstream_semantics`, supported by corpus alignment): BPM
  is quarter-note BPM in simple meter and compound felt-pulse BPM in
  denominator-8 meter. Therefore `us_per_qn=60_000_000/bpm` in simple meter
  and `40_000_000/bpm` in compound meter. A tempo at a meter-change onset uses
  the new meter.
- Decision — pitch class (`observed_corpus_semantics` and
  `upstream_semantics`): derive a degree from the active key's immutable scale
  steps, then apply `bb`, `b`, natural, `#`, or `##` accidental offset.
- Decision — absolute octave (`upstream_semantics`): canonical MIDI pitch is
  `60 + 12*raw_octave + tonic_pc + active_scale_degree_offset + accidental`.
  MIDI 72 remains documented only as `legacy_compatibility`.
- Consequences: durations crossing meter changes remain exact, 12/8 retains 12
  half-qn canonical beats per complete bar, unsupported scales omit dependent
  notes rather than assuming major, and production provenance names the
  upstream scale-degree method. No schema-version change or runtime Sheet Sage
  dependency is introduced.

## 2026-07-20 — ADR-023: Close Phase 2B.1 with explicit legacy-drift waiver

- Status: Accepted.
- Context: Final compound-meter controls confirm the production mapping without
  changing production code. The external read-only legacy checkout retains the
  pinned HEAD but its staged worktree no longer matches the status captured in
  `docs/legacy_snapshot.json`.
- Decision: Accept and complete Phase 2B.1 at implementation
  `3898b168063094b87e5ca5d88aae0317c1562c3f`. Preserve the timing derivation as
  raw beat -> four tertiary units -> primary meter pulse -> canonical qn.
  Apply closure resolution C to legacy drift: keep the failing check visible,
  publish bounded recorded/current blob evidence in
  `docs/LEGACY_DRIFT_REPORT.md`, and neither refresh the snapshot nor modify the
  external checkout without owner classification.
- Consequences: `raw beat - 1` remains a valid raw-to-simplified source-beat
  comparison, not a universal qn conversion. Phase 2B.1 can merge while the
  external waiver remains explicit; a future owner action must choose snapshot
  refresh or manual legacy restoration.

## 2026-07-20 — ADR-024: Canonical MIDI export is an exact diagnostic boundary

- Status: Accepted for Phase 2B.2 review.
- Context: Listening to canonical HookTheory conversion and round-tripping it
  through the generic MIDI adapter are useful diagnostics, but a MIDI created
  from canonical records cannot independently validate the raw-source mapping.
  SMF PPQ is also bounded to 32767 while exact source decimals can require a
  much larger denominator LCM.
- Decision: Add an output-only `music_critic.exporters` package using the
  already-declared low-level `mido` dependency. Validate input, choose the LCM
  of every rendered canonical time when it fits, and require explicit caller
  opt-in before half-up PPQ quantization at a documented fallback. Preserve
  canonical tempo, meter, and non-null melody performance fields; otherwise
  use explicit defaults. Generate clicks only from canonical beats and expose
  theory only as optional marker text. Keep simplified-source comparison in an
  audit script that does not import or call the HookTheory adapter.
- Consequences: Exact representable events round-trip without float equality;
  excessive-LCM events carry a rational error bound. Rendering remains absent
  from the data, graph, model, training, and inference dependency paths. Chord
  voicing, audio synthesis, and unsupported harmony semantics remain deferred.

## 2026-07-20 — ADR-025: MIDI review acceptance and ambiguity are independent diagnostics

- Status: Accepted for Phase 2B.2 review.
- Context: An independent comparison cannot use an exporter-reported error as
  its own tolerance. Standard MIDI also cannot uniquely preserve every
  canonical identity: same-pitch overlapping notes share note-off semantics,
  and simultaneous programs on one channel share channel state.
- Decision: Derive the comparison endpoint bound only from parsed MIDI PPQ as
  `1/(2*PPQ)`, directly measure exact onset/offset/duration error, require zero
  observed error for exact renders, and use the exporter maximum only as a
  bounded consistency check. Audit same-track/effective-channel/pitch interval
  overlaps and same-channel simultaneous program conflicts in separate
  diagnostics. Preserve canonical channel/program values with no allocator;
  render findings unchanged, reserve channel 9 for percussion/click, and make
  audio disagreement non-fatal alignment evidence.
- Consequences: HookTheory golden guarantees cover melody pitch/timing, tempo,
  meter, and piece duration. Generic exact representable timing/pitch/tempo/meter
  remain supported, but full `CanonicalPiece` identity and timbre are not
  promised for ambiguity groups, unrepresentable data, targets, provenance, or
  annotations. The full corpus has 1,802 same-pitch overlap pairs in 102 clips,
  1,627 nested pairs, and zero channel/program conflict pairs.

## 2026-07-20 — ADR-026: Derived MIDI duration uses a full-tick audit bound

- Status: Accepted for Phase 2B.2 review remediation.
- Context: ADR-025 correctly bounds each independently rounded MIDI endpoint by
  half a tick, but its wording did not distinguish a note duration calculated
  as `offset - onset`. Opposite endpoint rounding errors can accumulate to one
  full tick in that derived duration.
- Decision: Keep the independently derived single-endpoint bound at
  `1/(2*PPQ)` for note onsets/offsets, tempo/meter onsets, and terminal piece
  duration. Use `1/PPQ` only for derived note-duration error. Exact mode uses
  zero for both acceptance bounds. Continue comparing the exporter-reported
  pointwise maximum only against the maximum observed endpoint error, never
  against duration and never as the audit tolerance.
- Consequences: Correctly quantized notes with opposing endpoint errors are no
  longer rejected, while endpoints remain half-tick bounded and exact renders
  admit no nonzero note endpoint/duration, tempo/meter-onset, or piece-duration
  error. The production exporter and its report contract remain unchanged.

## 2026-07-20 — ADR-027: Meter equality and meter acceptance are distinct

- Status: Accepted for Phase 2B.2 review remediation.
- Context: The canonical-meter audit already applied the endpoint bound, but
  the simplified-source aggregate and CLI still treated non-exact meter onset
  as a mismatch even when quantization was within that bound.
- Decision: Preserve `meter_regions_exact` for exact event count, onset,
  numerator, and denominator identity. Add `meter_regions_accepted`, requiring
  equal count and exact numerator/denominator while allowing onset error up to
  the active endpoint acceptance bound: zero in exact mode and `1/(2*PPQ)` in
  quantized mode. Use accepted meter regions for symbolic acceptance,
  `meter_mismatch_clips`, and CLI exit; report exact and quantization-accepted
  counts separately.
- Consequences: Valid half-tick meter-onset quantization no longer fails the
  independent audit. Structural meter differences and exact-mode onset drift
  still fail. Simplified-source and canonical-JSON meter comparisons remain
  separate evidence paths; the production exporter is unchanged.

## 2026-07-20 — ADR-028: Accept and close Phase 2B.2 canonical MIDI renderer

- Status: Accepted.
- Decision: Accept and complete Phase 2B.2 at implementation HEAD
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`. The accepted behavior comprises
  the generic `CanonicalPiece` MIDI exporter; rational PPQ selection; explicit
  opt-in quantization; direct canonical tempo and meter export; melody-note
  export; optional canonical-beat click and target markers; independent
  simplified-source comparison; separate endpoint and derived-duration bounds;
  exact-versus-accepted meter reporting; report-only overlap and
  channel/program ambiguity diagnostics; and a reproducible HookTheory
  listening/review package.
- Explicit non-goals: HookTheory chord-note synthesis, automatic chord voicing,
  SoundFont or audio rendering, channel-allocation policy changes, graph
  construction, SSL or preference training, and treating renderer output as
  independent dataset truth.
- Consequences: Phase 3 may rely on validated canonical data and generic MIDI
  diagnostics, but it must not treat diagnostic MIDI output or synthesized
  target-derived content as raw model input. Audio-alignment disagreement
  remains diagnostic evidence rather than an exporter failure, and generic MIDI
  round trips retain the documented ambiguity and representational limits.

## 2026-07-22 — ADR-029: Phase 3A graph is a versioned raw-only heterograph

- Status: Accepted.
- Context: HookTheory supplies rich supervisory targets while generic MIDI does
  not. A shared encoder graph must therefore be invariant to every target,
  annotation view, split, source group, and provenance field. Polyphonic note
  cliques would also make dense passages grow quadratically.
- Decision: Graph schema `1.0.0` contains exactly `song`, `track`, `bar`,
  `beat`, `onset`, and `note`, with the containment, chronological, reverse,
  and sustained relations recorded in `docs/ARCHITECTURE.md`. Exact canonical
  onset determines note/bar and onset/bar/beat ownership. Positive-duration
  notes connect to every beat start in `[onset, offset)`; grace notes do not.
  Beat and onset nodes are unconditional raw candidate slots. Feature registry
  `1.0.0` declares separate categorical, continuous, and availability tensors,
  all marked raw-inference-safe. Builder `1.0.0` ignores targets, annotations,
  dataset/split/group/source identity, provenance, confidence, and quality
  flags. Each PyG `HeteroData` stores canonical schema, graph schema, feature
  registry, and builder versions. Exact allowlists cover graph, node-store, and
  edge-store attributes; deterministic JSON serialization and fingerprinting
  validate the graph before encoding. Program/channel absence uses dedicated
  non-colliding unknown categories, while known out-of-vocabulary categorical
  observations are rejected. Exact rational time controls structure and is
  converted to `float32` only when continuous feature tensors are materialized.
  The builder validates canonical input by default and exposes
  `assume_valid=True` only as an explicit validated-input fast path.
- Dependency boundary: PyTorch and PyG imports are isolated to
  `music_critic.graph`; they are nevertheless current global package
  dependencies. `music_critic.data` remains importable without importing them,
  and adapters/exporters retain their existing `mido` boundary. Optional
  compiled PyG extensions are not required for Phase 3A.
- Consequences: HookTheory target-visible/hidden and generic MIDI pieces share
  one model-facing schema (schema parity, not general data parity).
  Simultaneous-note context flows through onset/beat intermediaries instead of
  cliques. Construction is output-sensitive in containment, chronological, and
  note/beat incidence; long sustains can still emit many `active_at` edges.
  Float feature timing has less precision than exact canonical structure.
  Semantic nodes, target routing, graph batching/caching, GNNs, SSL, masking,
  and corruption training remain later phases and require explicit version
  decisions if they alter this base contract.

## 2026-07-23 — ADR-030: POP909 adaptation requires official evidence and masked views

- Status: Accepted for Phase 4A.
- Context: The installed `data/pop909-cl` tree is an unversioned flattened
  processed MIDI mirror with no annotations or documentation, and its track
  structure differs from the official corpus. The pinned official repository
  has complete annotation assets, but its algorithmic audio/MIDI views do not
  align exactly and its alternative MIDI versions do not retain the complete
  primary track-name contract.
- Decision: Define the Phase 4B supervised source contract from official
  POP909 repository commit
  `d83e6edba6872a704f5d3b8b32f5cb540088dae6` and its recorded hashes. Preserve
  all raw tracks. Expose primary `MELODY`, `BRIDGE`, and `PIANO` semantics only
  as masked track-level targets resolved by unique exact names; never use role
  labels in raw graph input and never infer missing roles from order. Preserve
  annotation decimal seconds and raw labels, keep audio/MIDI views separate,
  and require an explicit versioned tolerance for any derived alignment.
  Treat beat/chord/key annotations as algorithmic auxiliary targets with
  unknown confidence, not human gold. Use one `pop909:<three-digit-song-id>`
  group for the primary, all annotations, and every version, without assigning
  splits in Phase 4A. Retain official song `043` as an explicit conversion
  failure until a general mid-bar-meter rule is accepted and tested.
- Consequences: The local processed mirror remains usable only through the
  generic unlabeled-MIDI path unless independent provenance is supplied.
  Alternative roles and missing annotations are masked rather than negative or
  guessed. Phase 4B must preserve raw-label provenance, exact timing, group
  integrity, and raw-graph leakage invariance; it cannot special-case one song
  merely to obtain 100% conversion.

## 2026-07-23 — ADR-031: POP909-CL supersedes original POP909 for production Phase 4B

- Status: Accepted; explicitly supersedes ADR-030 for the production Phase 4B
  corpus and adapter contract. ADR-030 remains in this append-only log as the
  incorrect prior decision.
- Context: ADR-030 misidentified the installed `data/pop909-cl` extraction as
  an unproven flattened original-POP909 mirror. Complete path/hash comparison
  proves that all 909 relevant local MIDI files are byte-identical to
  `POP909_processed` at POP909-CL commit
  `be9094392903c471a930519e1c0bacf8b6be5d62`. POP909-CL embeds corrected chord
  blocks in the MIDI rather than external sidecars. The original audit remains
  scientifically useful, but its external labels, roles, alternatives, and
  song-043 failure are not production CL facts.
- Decision: Make `pop909_cl` the primary Phase 4B corpus and use
  `pop909-cl:<song-id>` for its source group. Retain original POP909 as
  `pop909_original` with `pop909-original:<song-id>`. If both are later used,
  matching IDs share `pop909-lineage:<song-id>` and one split. Resolve the
  combined musical score from the documented channel-0 instrument using
  measured channel evidence. Treat the documented channel-1 chord instrument
  as target-only: it cannot enter canonical raw tracks/notes, statistics,
  graph structure/features, serialization, fingerprints, or inference input.
  Preserve chord blocks losslessly at exact ticks/PPQN before applying the
  upstream root/quality/bass normalization; retain unsupported, ambiguous,
  overlapping, and implicit no-chord evidence. Record target source as human
  with provenance details `human_corrected` and `expert_reviewed`, without
  claiming infallible gold or fabricated numeric confidence. Preserve MIDI
  time/key signatures as source meta-events. Song `172`, not original song
  `043`, is the unresolved production meter case and remains quarantined until
  a general partial-bar policy is accepted.
- Consequences: Complete-file generic-adapter warnings are unsafe diagnostics,
  not score-quality measurements. AppleDouble extraction noise is excluded
  from the CL content fingerprint. Missing chord instruments yield unavailable
  targets rather than negatives; missing/ambiguous instruments are structured
  failures and are never repaired from pitch range, names, or order. Phase 4B
  may implement only the score projection and masked target contract; this ADR
  adds no production adapter or canonical meter special case.

## 2026-07-23 — ADR-032: POP909-CL target semantics separate observation from derivation

- Status: Accepted for Phase 4A; refines ADR-031 without changing the selected
  corpus or raw/target leakage boundary.
- Context: The first remediated audit assigned source `human` to normalized
  chord fields, treated uncovered time after the final chord as `N`, retained
  pairing anomalies only as counts, and made expected target absence plus the
  known song-172 meter case fail one undifferentiated readiness flag. Those
  semantics overstate both annotation coverage and direct human provenance.
- Decision: Raw channel-1 chord blocks use source `human` with
  `human_corrected` and `expert_reviewed` details. Root, quality, inversion,
  and inferred leading/internal `N` use source `derived` with explicit chains
  through the pinned upstream normalizer or gap-event construction. The
  upstream-compatible `N` contract has leading/internal spans only; trailing
  uncovered time is masked/unannotated. Directly observed boundary and bass
  remain available. Ambiguous root/inversion are unavailable single-label
  targets, ambiguous quality is available only when all candidates agree, and
  unsupported root/quality/inversion are unavailable. Pairing anomalies retain
  exact event and affected-region evidence. Missing chord targets for `367`
  and `658` are expected masked availability, and `172` is the documented
  quarantine. Strict output separates `evidence_contract_ready` from
  `production_adapter_ready`.
- Consequences: Phase 4A evidence can be ready while production remains
  unimplemented. The manifest pins 947 derived `N` spans, 151 trailing masked
  spans, field-specific availability counts, and the exact anomaly-evidence
  fingerprint. Phase 4B must implement this contract without adding chord
  evidence to raw inputs or special-casing song `172`.

## 2026-07-23 — ADR-033: Phase 4B MVP retains the documented song-172 quarantine

- Status: Accepted; closes the Phase 4A readiness question left open by
  ADR-031 and ADR-032.
- Context: The adapter contract already permits retaining song `172` as a
  documented quarantine, but readiness metadata still named a pending general
  partial-bar-meter policy as a production blocker. That made an optional
  future enhancement appear mandatory for the Phase 4B MVP.
- Decision: Lock the Phase 4B MVP score policy to accept the 908 generic
  score-only conversions and quarantine song `172` under the observed
  `midi_adapter.meter_change_inside_bar` condition. A general partial-bar meter
  policy requires a later recorded decision and is not an MVP dependency. The
  strict audit retains `evidence_contract_ready=true`, reports
  `production_adapter_ready=false`, and names only
  `phase_4b_production_adapter_not_implemented` as a production blocker.
- Consequences: Phase 4B can implement the evidence-backed adapter without a
  meter-semantics expansion or a song-specific repair. Production acceptance
  is 908/909 for the MVP, with `172` preserved as explicit provenance-bearing
  quarantine evidence. No adapter, graph, model, or meter code is added by
  this decision.

## 2026-07-23 — ADR-034: Harmonic annotations are target-only semantics, not accompaniment quality

- Status: Accepted.
- Context: HookTheory supplies melody-conditioned chord annotations while
  POP909-CL supplies expert-reviewed/human-corrected chord blocks describing
  harmony in a channel-0 combined score. Both can supervise shared harmonic
  concepts, but neither target representation should be mistaken for raw input,
  actual performed/score accompaniment, or a quality judgment. Arbitrary MIDI
  also cannot be assumed to carry reliable semantic track roles.
- Decision: Treat HookTheory and POP909-CL chord annotations as target-only
  auxiliary harmonic supervision. Direct annotations may produce explicitly
  provenance-linked derived harmonic targets, including pitch-class/set
  representations. Bass and inversion are separate target families with
  independent availability masks; a joint or factorized head is a future
  ablation that must preserve both masks. Any target-derived note realization
  is forbidden in raw canonical tracks/notes, graph features/topology,
  raw-input serialization, graph serialization, raw-input cache identity,
  graph fingerprints, and inference. Derived targets may be serialized in
  separate target/annotation/diagnostic artifacts with provenance, but those
  artifacts cannot define raw/graph identity or enter inference. A derived
  realization is a target-only diagnostic or experimental view, not actual
  accompaniment ground truth. Chord prediction is an auxiliary semantic task
  and classifier confidence is not a harmony quality metric. Role-agnostic
  probabilistic completion and normalized PLL remain future
  design-and-ablation questions. Production inference requires neither melody,
  accompaniment, chord, bass, voice, nor staff roles.
- Consequences: HookTheory melody-only graphs and POP909-CL channel-0
  combined-score graphs may train shared harmonic heads through
  dataset-specific masks, annotation views, and per-target provenance. Missing
  or ambiguous labels remain unavailable rather than negative. Representation
  reconstruction, masked conditional likelihood, actual accompaniment
  likelihood, and the preference/quality critic remain separate objectives.
  Phases 7–8 validate SSL mechanics on bounded pre-PDMX data; Phase 10 must
  enable full-scale rerun and evaluation of their accepted objectives on the
  PDMX raw-compatible corpus before scaled SSL conclusions.
  This decision changes no schema, adapter, graph, audit, model, or inference
  implementation; the complete contract and deferred questions are in
  `docs/HARMONIC_SUPERVISION.md`.

## 2026-07-25 — ADR-035: Phase 4B production POP909-CL adapter is accepted

- Status: Accepted and Completed.
- Context: ADR-031 through ADR-034 establish the pinned corpus identity,
  channel-0/raw versus channel-1/target boundary, field-specific masks,
  provenance, missing-target policy, and song-172 MVP quarantine. Schema
  `2.0.0` can represent the six target families, but it forbids annotation
  spans beyond raw piece duration and has no lineage or structured
  candidate-evidence field.
- Decision — API and versions: implement adapter
  `music_critic.adapters.pop909_cl` version `1.0.0` and production corpus
  manifest version `1.0.0`. Public entry points are
  `discover_pop909_cl_corpus`, `convert_pop909_cl_file`, and
  `iter_pop909_cl_corpus`, configured by `Pop909ClCorpusIdentity` and
  `Pop909ClAdapterConfig`. Results are typed as `Pop909ClAccepted`,
  `Pop909ClExpectedTargetAbsence`, or `Pop909ClQuarantine`; typed corpus,
  adapter, and conversion errors preserve failure categories.
- Decision — identity and grouping: require the pinned 909-file content
  fingerprint, upstream repository/commit, MIT identity and license hash,
  original relative path and per-file SHA-256. Use
  `pop909-cl:<song-id>` as canonical source group and retain
  `pop909-lineage:<song-id>` in the production corpus record and provenance.
  Assign no split.
- Decision — raw/target boundary: route instruments only by channel-bearing
  events. Convert a temporary score-only channel-0 plus conductor/meta
  projection through the unchanged public generic MIDI API. Channel 1 never
  affects raw tracks/notes/statistics, raw identity, duration, graph content,
  serialization, or inference. `include_targets=False` removes all target
  spans, arrays, and target-only provenance while preserving raw content.
- Decision — target representation: use alignment type
  `pop909_cl.chord`, annotation view `pop909_cl.channel_1`, and stable task IDs
  `pop909_cl.chord.boundary`, `.root`, `.quality`, `.bass`, `.inversion`, and
  `.no_chord`. Direct boundary/bass target provenance points to a per-block
  human annotation record. Root/quality/inversion point to derived normalizer
  records; inferred `N` uses a separate gap derivation. Bass and inversion
  masks remain independent. Structured production evidence retains every
  candidate and exact diagnostic outside the fixed canonical schema.
- Decision — missing and quarantine: `367` and `658` use one full-piece target
  alignment plus six one-entry arrays whose mask/value/confidence/source/
  provenance are respectively false/null/null/null/null. `172` is quarantined
  only if the generic adapter actually reports
  `midi_adapter.meter_change_inside_bar`; any other failure or unexpected
  success is fatal.
- Decision — target intervals beyond raw duration: retain exact original
  onset/end ticks and PPQN in `Pop909ClChordBlock` and provenance. Intersect
  only the canonical target-alignment span with the raw piece interval when
  required by schema bounds. Never extend raw duration from target evidence.
- Acceptance: the fresh streaming pass reproduced 909 logical files, 908
  validator-clean accepted pieces, only `172` quarantined, 907 chord
  instruments, expected masked absence for `367`/`658`, 116,055 chord blocks,
  root/inversion 109,668, quality 109,800, boundary/bass 116,055, 5,801
  ambiguous, 586 unsupported, 947 derived `N`, 151 trailing masked spans, and
  anomaly fingerprint
  `d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`.
  This value is the historical Phase 4A/Phase 4B-v1 evidence fingerprint:
  its anomaly rows used the earlier source-path representation. It remains
  part of the historical acceptance record and is not the current portable
  POP909-CL `2.0.0` production contract.
  All 908 accepted visible/hidden pieces validated, round-tripped
  deterministically, retained equal raw content, and produced equal raw graph
  fingerprints.
- Consequences: Phase 4B is production-ready without changing canonical schema
  `2.0.0`, graph schema `1.0.0`, generic MIDI public API, or meter semantics.
  The Phase 4A audit remains an independent evidence oracle and historical
  readiness record. Phase 5 ontology/collation, models, SSL, training, PLL,
  preference/quality, splits, partial-bar support, other corpus adapters, and
  chord rendering remain deferred.

## 2026-07-26 — ADR-036: Phase 5A preserves source-native target semantics

- Status: Accepted.
- Context: HookTheory emits functional, melody-conditioned theory annotations,
  while POP909-CL emits absolute, score-conditioned chord-recognition
  evidence. Similar field names conceal different semantics. A future mixed
  collator also needs target/entity indices without weakening the Phase 3A
  raw-only graph allowlists.
- Decision: Introduce target ontology `1.0.0` in `music_critic.tasks` with all
  12 HookTheory and six POP909-CL stable task IDs, required masks/provenance,
  source view and supervision context, exact per-task alignment policy, and
  deterministic serialization/fingerprint. Declare no `exact_shared` or
  accepted `derived_lossless_subset` mapping in this version. Classify
  functional root versus absolute root, extent versus quality, ordinal versus
  semitone inversion, presence versus boundary, and presence/rest versus `N`
  as incompatible. Defer absolute-root and pitch-class-set rendering until
  applied/borrowed/decorations semantics have a versioned lossless rule.
  Define immutable sample/batch sidecar shapes and group/lineage validation,
  but leave dataset, tensorizer, sampler, collator, and splits to Phase 5B.
- Evidence: A deterministic audit converts 18 usable real-source HookTheory
  golden excerpts and reads only the accepted POP909-CL production manifest
  aggregates. No full HookTheory scan or repeated 909-file acceptance is
  required. Tests prove actual adapter structures match the registry, masked
  entries remain null, ambiguous/unsupported mappings remain masked, lineage
  cannot cross splits, and target/group/provenance changes do not enter or
  change raw graphs.
- Consequences: Future mixed batches distinguish absent families from masked
  entries and store values, masks, entity/sample indices, confidence,
  provenance, and diagnostics outside the PyG batch. HookTheory and POP909-CL
  may share an encoder, but shared model heads require a later explicit routing
  decision rather than an ontology-name shortcut. Bass and inversion remain
  independent. Applied harmony, borrowed crosswalks, chord rendering, final
  splits, model/loss/SSL/PLL/critic work, and production collation are
  unchanged deferred scope.

### 2026-07-26 pre-merge clarification

- Grouping: canonical provenance lineage is authoritative. A supplied lineage
  override is a non-empty equality assertion; absent provenance lineage falls
  back explicitly to `piece.source_group_id`. Duplicate assignment rows and
  conflicting source/lineage identities for one dataset piece are errors.
  Deterministic ordering hashes atomic transitive components connected by
  either source or lineage, then uses stable piece order inside each component.
- Alignment: notes use exact identity. Onsets use point time and beats/bars use
  their start anchors under half-open containment. Equal values addressing one
  typed candidate merge; conflicts are masked with
  `multisource.alignment_conflict`. Entity node type is explicit, so there is
  no implicit node priority. Boundary events require exact-time candidates;
  unmatched events remain present with index `-1`, a false index mask, and
  null node type, with no nearest-neighbor snapping.
- Boundary objective: `pop909_cl.chord.boundary` is positive-unlabeled event
  detection. Only observed span starts are positive. Non-boundary candidates
  are unlabeled, and ontology `1.0.0` has no `absent` class or derived-negative
  policy.
- Containers: sample target values, masks, entity IDs, confidence, task
  availability, batch leading dimensions, node-type/index consistency,
  non-empty metadata, sample-index ranges, sorted task sidecars, and raw-only
  graph separation are constructor invariants. These decisions constrain the
  future Phase 5B implementation without implementing it.

### 2026-07-26 final pre-merge clarification

- PyG batches: validate the actual `Batch.from_data_list` representation
  against the Phase 3A exact allowlists. Global attributes are unchanged.
  Node stores may add only PyG `batch` and `ptr`; edge stores add nothing.
  Production metadata and raw-only truth are checked per source graph, and
  combined shapes/dtypes, offsets, endpoints, reverse relations, cross-graph
  isolation, and reconstructed source graphs must all validate. Any unknown
  store field is invalid, independent of a dangerous-name denylist.
- Split safety: the atomic source/lineage component builder is shared by
  validation and deterministic ordering. All non-null splits in a transitive
  component must agree, including components connected through `split=None`.
  A dataset piece has exactly one assignment; every repeated identity is an
  error.
- PU objectives: unobserved boundary events and no-chord coverage may not
  become negative examples in Phase 5B. Phase 6 either records and uses a
  separately accepted PU-compatible objective for each task or leaves that
  task disabled.

## 2026-07-26 — ADR-037: Phase 5B.1 uses exact typed alignment and versioned sidecar encodings

- Status: Accepted.
- Context: Phase 5A fixed source-native semantics and raw-only PyG allowlists
  but deliberately deferred executable alignment, encodings, offsets, and
  collation. A mixed batch must distinguish source availability from whether a
  raw graph entity can be aligned, without reconstructing exact time from
  float features or leaking supervision into graph stores.
- Decision: Bind each prepared sample to its validated canonical piece and raw
  graph. Align notes by exact canonical ID; align onset points and beat/bar
  start anchors with exact `RationalTime` and half-open containment; align
  boundary events only at exact time. Expand every allowed typed match with no
  node-type priority. Retain available-but-unaligned and masked source entries
  separately, merge equal values, and mask conflicting values with
  `multisource.alignment_conflict`.
- Decision: Introduce target encoding registry `1.0.0`. Closed categorical
  values use ontology-order `torch.long [N]` and unavailable sentinel `-1`.
  Closed multilabel values use `torch.bool [N, C]`; an all-false row under
  false availability is a sentinel, not a negative. Open strings remain
  lossless CPU values with `model_ready=false`; per-batch or per-worker dynamic
  vocabularies and Python-hash IDs are forbidden.
- Decision: Keep source availability and entity alignment as independent
  boolean masks. Future supervision eligibility requires both plus model
  readiness, but selects no concrete loss. Nullable confidence remains
  nullable; partial confidence uses a separate mask. POP909-CL boundary and
  no-chord contain annotated positives only; Phase 6 must separately accept a
  PU-compatible objective or disable each task.
- Decision: Collate raw graphs with normal PyG `Batch`. Translate a local
  target index using the explicit node type and
  `ptr[sample_index]`, then verify that the corresponding `batch` value equals
  the sample index. Values, masks, sample/dataset identity, confidence,
  provenance, and diagnostics remain immutable CPU/tensor sidecars and never
  enter global, node, or edge PyG stores.
- Evidence: Bounded HookTheory, POP909-CL, and synthetic raw-only fixtures cover
  all 18 tasks, note/span/boundary alignment, half-open boundaries, duplicate
  merge/conflict, unaligned and masked rows, encodings, offsets, leakage,
  malformed tensors/batches, deterministic statistics, and repeated
  collation. A lightweight dozens-of-graphs benchmark is separate from corpus
  acceptance. The final semantic remediation patches ontology to `1.0.1`
  without changing adapter targets or production manifests.
- Consequences: Phase 5B.2 owns corpus `Dataset`, indexing, mixture sampler,
  worker-safe `DataLoader`, and split consumption. Phase 6 owns models, heads,
  objectives, and the PU decision. No HookTheory/POP909 semantic crosswalk,
  chord renderer, target-derived notes, adapter change, or graph-schema change
  is authorized by this ADR.

### 2026-07-26 pre-merge remediation

- Alignment is indexed and output-sensitive. One immutable `AlignmentIndex`
  is built per canonical piece with O(1) note/annotation and exact-time
  mappings plus sorted rational candidate arrays. Half-open spans use bisect;
  because index construction sorts temporal candidates, strict complexity is
  `O(P + C log C + T log C + R + F*C)`. For the fixed task registry, `F*C`
  is linear in candidate count.
  Instrumentation, rather than a timing threshold, guards against repeated
  full index construction or source-entry candidate scans.
- Encoding registry `1.0.0` describes value representation and the semantic
  regimes `fully_supervised`, `positive_unlabeled`, and
  `deferred_open_vocabulary`; it does not select CE, BCE, focal, or PU loss.
  POP909-CL boundary and no-chord remain distinct positive-unlabeled tasks with
  no synthetic negatives. Phase 6 must separately choose a PU-compatible
  objective or disable each task.
- Production preparation owns canonical-to-graph proof:
  `prepare_multisource_sample` builds the Phase 3A graph and records a complete
  fingerprint; the external-graph factory compares against a fresh projection
  and has no bypass; collation recomputes the fingerprint to catch later
  categorical, continuous, or topology mutation. Target-only audit projection
  has no graph, and no binding is added to PyG stores.
- `model_encodable_row_count` means only that a registry representation
  exists. `supervision_eligible_row_count` is the exact sum of
  `availability_mask & entity_index_mask & model_ready`. Masked,
  available-but-unaligned, conflict, and deferred-open-vocabulary counts are
  independent explicit statistics, checked per task and in aggregate.
- The raw-only benchmark remains graph baseline evidence. A separate
  small/medium/large target-heavy benchmark reports index construction, target
  lookup, emitted rows, full collation, and operation counts; it is not a
  default CI or corpus-acceptance job.

### 2026-07-26 final semantic remediation

- Patch target ontology `1.0.0` to `1.0.1`. Stable task IDs, adapters,
  production target values, vocabularies, masks, and provenance remain
  unchanged; the patch corrects the declared semantics of
  `pop909_cl.chord.no_chord`.
- `pop909_cl.chord.boundary` and `pop909_cl.chord.no_chord` are distinct
  positive-unlabeled tasks. Boundary is observed event detection with only
  `present` positives. No-chord is observed coverage detection with the
  one-class vocabulary `("N",)` and only explicit leading/internal positive
  spans. Absence of an observed boundary/span, chord spans, uncovered
  candidates, and absent annotations create no synthetic negative.
- A one-class no-chord representation is not a trainable fully-supervised
  classification task by itself. `supervision_eligibility_mask` means only
  that a represented row may be routed to a future task-specific objective;
  it does not imply ordinary classification. Phase 6 must separately accept a
  PU-compatible objective for boundary and no-chord or leave the corresponding
  task disabled.
- No `not_N` class or derived-negative policy is introduced. Explicit
  negative no-chord evidence would be a future versioned ontology/adapter
  experiment with its own evidence contract. Phase 5B.1 selects no concrete
  CE, BCE, focal, or PU loss.

## 2026-07-26 — ADR-038: Phase 5B.2 uses an offline canonical cache and target-blind deterministic loading

- Status: Accepted for draft pre-merge implementation.
- Context: Re-running production adapters every training epoch is expensive,
  while loading every canonical piece or graph into memory is not scalable.
  Dataset splitting and source balancing must preserve proven source/lineage
  closure and cannot infer labels or scientific configuration.
- Decision: Introduce portable corpus index and canonical cache contracts
  `1.0.0`. Cache identity binds source content, adapter version/config,
  canonical schema, target ontology semantics, and cache version. Artifact
  identity additionally binds deterministic canonical payload SHA-256. Writes
  use temporary files plus atomic rename; partial content is invalid, stale
  namespaces are retained, and PyG/pickle graph or tensor caches are forbidden.
- Decision: Use an offline HookTheory single-pass record stream and the
  existing POP909-CL discovery/production adapter. Each accepted
  `CanonicalPiece` is serialized and released; quarantine remains structured
  report evidence and never enters an index. Only
  `HookTheoryAdapterError` is a quarantinable HookTheory record failure, under
  stable category `hooktheory.record_conversion_invalid`; unexpected runtime,
  programming, and resource failures abort without a successful index/report.
  Builder limits are `None` or positive non-bool integers. Production
  adapters, target values/masks, and manifests are unchanged.
- Decision: Use a lazy map-style Dataset whose constructor reads metadata only.
  One indexed item verifies and validates one artifact and calls
  `prepare_multisource_sample`. Canonical source group, prepared
  dataset/piece/source/lineage identity, and recomputed target availability
  must equal the indexed sidecars. These comparisons do not add identity or
  target fields to PyG stores. Spawn restoration reinstates the private
  binding token and repeats graph fingerprint verification rather than
  weakening the binding contract.
- Decision: Require one versioned external `SplitManifest` for the exact
  complete multi-corpus index set. `MultiCorpusDataset` validates it once
  globally before deriving per-dataset views. The manifest covers every piece
  and binds source group, lineage group, cross-dataset transitive atomic
  component, seed/policy/config, and unique dataset/index fingerprints.
  Missing, extra, duplicate, stale, differently manifested, or independently
  validated constituents are rejected. Source split is only a diagnostic
  suggestion. No production ratios or seed are selected; fuzzy duplicate
  discovery remains out of scope.
- Decision: Bind each derived view to the global manifest fingerprint, split,
  corpus index fingerprint, and exact ordered record identities. A versioned
  composition fingerprint additionally binds all constituent fingerprints and
  memberships. Compose only views of one split. Allocate explicit positive
  dataset weights with deterministic largest remainder, then use local torch
  generators for the epoch schedule and no-repeat-before-exhaustion local
  cycles. Same seed/epoch/contracts replay exactly; `set_epoch` changes the
  schedule. Epoch evidence carries global manifest/view/composition identity,
  and its schedule fingerprint hashes resolved `(dataset_id, piece_id)`
  identities plus sampler version, seed, epoch, weights, and quotas rather than
  temporary integer offsets. Mid-epoch resume is deferred.
- Decision: Seed Python and torch from the PyTorch worker seed in a top-level
  spawn-picklable initializer and retain the Phase 5B.1 collator unchanged.
  NumPy is not imported because it is not a project dependency. Split,
  sampling, and worker diagnostics never enter PyG stores. Worker parity
  compares every raw graph field, target tensor and CPU sidecar, identity,
  diagnostic, and deterministic statistic; no CPU field is intentionally
  excluded.
- Consequences: Targets and availability counts are audit metadata only and
  never affect split assignment, quotas, or record choice. Production split,
  mixture weights, models, losses, PU objectives, SSL/corruptions, PDMX, and
  any graph-cache optimization require later evidence-backed decisions.

## 2026-07-26 — ADR-039: Phase 6A preserves local evidence in the first trainable baseline

- Status: Accepted for draft pre-merge implementation.
- Context: Phase 5B supplies validated raw-only heterogeneous batches and
  source-native target sidecars. The first learned phase must prove trainable
  raw graph encoding and auxiliary supervision without prematurely adding
  hierarchy, SSL, likelihood, or critic semantics. Dense pieces and global
  averages must not erase isolated-note evidence.
- Decision: Introduce model, encoder-output, loss, reconstruction, and
  checkpoint contracts `1.0.0`. One `LocalBaselineConfig` controls a
  feature-only baseline and a local relation-aware baseline. Both encode every
  Phase 3A feature column and availability mask for all six node types. The
  local GNN handles every ordered Phase 3A relation with a distinct projection,
  sum aggregation, self/residual path, LayerNorm, GELU, and dropout. Feature,
  optional layer, and final skip-fused embeddings retain one row and exact
  batch membership for every original node.
- Decision: Instantiate only the ten HookTheory and four POP909-CL tasks whose
  versioned encodings are model-ready and `fully_supervised`. Keep source heads
  separate. Open mode/borrowed and positive-unlabeled boundary/no-chord have no
  head or ordinary CE/BCE. There is no shared pitch-class-set head. Task rows
  gather by explicit node type, global entity index, and sample index.
- Decision: Use unreduced CE for closed categorical targets and unreduced
  BCE-with-logits for closed multi-label targets. Eligibility additionally
  requires `fully_supervised`. Mean rows inside task/node-type/sample, mean
  active groups inside task, then take the configurable weighted mean of active
  tasks. Empty tasks add no artificial target. Retain every row loss.
- Decision: Reconstruct one visible inference-safe field per node type only to
  verify gradient and bounded overfit plumbing. This is not masking, SSL,
  likelihood, anomaly, corruption, or quality scoring. Bind checkpoints to
  model/config, canonical/graph/feature, ontology/encoding, and ordered-head
  metadata before loading.
- Consequences: Phase 6A can compare feature-only and local message passing and
  diagnose one-note sensitivity without claiming musical preference or corpus
  feasibility. Phase 6B owns hierarchy pooling, bar+track Transformer, song
  embedding, and top-down fusion; it must not make mean-only aggregation the
  final evidence path. A future critic must compare global context with
  retained local or top-k worst evidence. SSL begins in Phase 7. Shared
  pitch-class semantics remain blocked on a versioned lossless
  renderer/crosswalk.

## 2026-07-26 — ADR-040: Phase 6A prediction is candidate-first and checkpoint application is failure-atomic

- Status: Accepted as pre-merge remediation of draft PR #9.
- Context: Target-routed logits made raw-only output empty and allowed target
  sidecars to determine which predictions existed. Row-wise host
  materialization also made forward/loss work scale in Python with supervision
  rows. The first trainable baseline needs an inference path defined entirely
  by raw graph candidates and a checkpoint failure boundary that cannot leave
  partially changed training state.
- Decision: Patch the model/output and loss contracts to `1.1.0`, introduce
  candidate prediction `1.0.0`, patch `BatchTarget` to `1.1.0` with validated
  tensor node-type codes, and patch checkpoint contract to `1.1.0`. Each of
  the 14 active heads enumerates every allowed raw-graph candidate before any
  target access. Targets join to those identities only for loss. Replacing,
  deleting, masking, or adding targets cannot alter candidate identities or
  eval logits; raw-only batches emit candidate logits and no harmonic loss.
- Decision: Candidate routing, supervision join, and
  task/node-type/sample reductions use tensor operations. Python work is
  bounded by the fixed task/node-type families; model forward/loss performs
  no per-row host conversion or row list processing.
- Decision: Single-note evidence changes one validator-clean canonical note
  pitch while preserving its stable ID, rebuilds and validates both production
  graphs, requires different fingerprints and identical topology, and reports
  exact raw-feature and local-embedding changes. Oversmoothing is computed
  separately for every `(sample, node_type, scale)`. Membership is validated
  and scanned once per node type to build contiguous `S+1` boundaries; other
  scales must have identical sample IDs. Basic `start:end` views replace
  boolean feature indexing, so production creates no `N_group x D` group
  copies. For normalized rows `u_i`, subtract
  `sum_i ||u_i||²` from `||sum_i u_i||²`, then divide by `N*(N-1)`. Subtracting
  `N` is invalid for zero rows because PyTorch normalization leaves them zero.
  The report records the exact pre-normalization `zero_norm_count`, and remains
  unavailable for fewer than two nodes. Boundary construction uses
  `O(sum_t N_t + T*S)` time and `O(T*S)` CPU metadata; cosine work uses
  `O(K*sum_t N_t*D)` time and `O(D)` temporary accumulator memory per group;
  report traversal and storage are honestly `O(K*T*S)` time and memory.
- Decision: Checkpoint load validates metadata, exact model keys/shapes/dtypes,
  and optimizer groups/state tensors before mutation; an application failure
  restores complete model and optimizer states. Save uses a same-directory
  temporary followed by atomic replace.
- Consequences: Target sidecars now select loss rows, never prediction
  existence. Model parameters, active tasks, ontology `1.0.1`, encoding
  registry `1.0.0`, adapters, production manifests, and Phase 6A scientific
  scope are unchanged. The oversmoothing correction restores the already
  stated exact diagnostic semantics; no separate versioned diagnostic-policy
  contract exists, so model/output/loss versions do not change. Phase 6B and
  Phase 7 remain unstarted at this remediation point.

## 2026-07-26 — ADR-041: Phase 6B adds raw-owned coarse context while retaining local evidence

- Status: Accepted for draft pre-merge implementation.
- Context: Accepted Phase 6A provides target-independent raw candidates,
  one-row-per-node local representations, auxiliary source-native heads, and
  failure-atomic checkpoints. Phase 6B must add deterministic musical
  hierarchy and longer-range context without turning target annotations into
  topology, hiding isolated evidence behind a mean, or introducing critic/SSL
  semantics.
- Decision: Introduce separate hierarchy-pooling, coarse-token-sequence,
  hierarchical-encoder-output, top-down-fusion, hierarchical-model/output, and
  hierarchical-checkpoint contracts at `1.0.0`. Keep all Phase 6A contracts
  and public behavior unchanged. Exact ownership is derived only from raw
  beat/onset/note-to-bar, note-to-track, and bar/track-to-song forward/reverse
  edges. Every child has one owner; malformed order, cardinality, transpose,
  range, membership, or sample ownership raises `HierarchyContractError`
  instead of being repaired.
- Decision: Pool bar own+beat/onset/note and track own+note families through
  sparse mean, maximum, log-count, explicit availability, learned projection,
  and an explicit parent residual. Do not construct dense membership or
  child-by-parent tensors. Empty families remain explicitly unavailable.
- Decision: Form one padded `[SONG] + bars + tracks` sequence per sample with
  distinct type embeddings, runtime sinusoidal ordinal positions, and a
  key-padding mask. Apply a batch-first pre-norm Transformer. Samples never
  share an attention sequence. The contextual SONG row is representation
  evidence, not a quality score.
- Decision: Fuse contextual bar+track+song into notes, bar+song into
  onsets/beats, contextual parent+song into bars/tracks, and contextual song
  into song through node-type-specific gated residuals. Do not invent track
  ownership for onset or beat. Retain the entire Phase 6A multi-scale output
  and run the unchanged 14 heads, loss join, and reconstruction over fused raw
  candidate rows.
- Decision: Bind hierarchical checkpoints to all six Phase 6B contracts and
  every inherited Phase 6A/graph/feature/ontology/encoding/configuration/head
  contract. Validate before mutation and restore full model/optimizer state
  after any application-time failure. Compare feature-only, local GNN, and
  hierarchy+Transformer on identical bounded data, and diagnose hierarchical
  single-note propagation plus unrelated-sample isolation without a quality
  threshold.
- Consequences: Phase 6B supplies coarse context alongside directly
  inspectable local evidence while preserving 237 tiny and 79 isolated
  raw-only candidate rows. It does not change ontology, encoding, adapters,
  manifests, corpus contracts, Phase 6A checkpoint compatibility, or
  production target semantics. SSL, corruption/remasking, latent prediction,
  PLL, PU objectives, preference learning, and critic/quality scoring remain
  later work; Phase 7 has not started.

## 2026-07-26 — ADR-042: Phase 6B ownership is structured and coarse packing is tensorized

- Status: Accepted as final pre-merge remediation of draft PR #10.
- Context: PyG mapping reads can create missing stores, the initial
  `HierarchyOwnership` constructor did not prove equality with the raw graph,
  and coarse sequence construction scanned membership rows through
  device-to-host `.item()` calls. These implementation gaps weakened failure
  categories and made synchronization count grow with coarse rows even though
  the intended Phase 6B serialized outputs were already correct.
- Decision: Check graph type, mandatory node stores, all six
  ownership/containment relation pairs, and `edge_index` attributes before
  indexing. Give input/store/attribute/tensor/missing/duplicate/reordered/range/
  reverse/cross-sample failures distinct stable `HierarchyContractError`
  categories, and guarantee validation failure does not change store types or
  attribute-key sets.
- Decision: Any externally supplied ownership is validated for exact ordered
  keys, complete membership, sample count, dtype/rank/shape/device, one owner
  per child, ranges, sample agreement, and exact raw forward/reverse equality.
  The standard model extracts raw ownership once before Phase 6A encoding,
  then uses a private local-row/device consistency handoff rather than a second
  raw scan.
- Decision: Compute bar/track counts with `bincount`, cumulative family starts
  with `cumsum`, and ordinals/positions/indexed placement on device. Production
  has no Python scan or per-row `.item()`/`.tolist()`/`.cpu()`. One
  `max(lengths).item()` synchronization per batch remains necessary to allocate
  the padded `B x max(L) x D` output and is independent of coarse-row count.
- Consequences: Canonical `[SONG]+bars+tracks` order, type/position embeddings,
  padding, gradients, eval determinism, six public/serialized Phase 6B
  contracts, and all Phase 6A semantics remain unchanged, so the six versions
  remain `1.0.0`. No graph/canonical/ontology/encoding/adapter/manifest/corpus
  semantics change. Phase 7 remains unstarted.

## 2026-07-27 — ADR-043: Phase 6C owns reproducible execution, not new learning semantics

- Status: Accepted for draft pre-merge implementation.
- Context: Phase 6A/6B provide trainable feature-only, local-GNN, and
  hierarchical baselines, while Phase 5B.2 provides versioned caches, global
  splits, lazy datasets, deterministic quota sampling, and worker-safe
  collation. A separate execution contract is needed before SSL work so
  one-batch optimization, ordinary supervised epochs, device movement, and
  resume evidence do not become ad hoc scripts.
- Decision: Use Hydra structured groups with explicit deterministic fields and
  persist the fully resolved application configuration. Select existing
  model/data/loss paths only; Phase 6C does not add or reinterpret heads,
  targets, reconstruction, graph inputs, or corpus artifacts.
- Decision: Make `move_multisource_batch` the official non-mutating device
  boundary. Move raw-graph tensors and model-facing target tensors together,
  keep provenance/diagnostics/strings/statistics as CPU sidecars, preserve
  tuple-valued PyG metadata, and never insert a target into the graph.
  Validate device, shape, task order, and graph binding through fixed registry
  tensor operations rather than replaying per-row Python validation on CUDA.
- Decision: Treat one-batch loss decrease and bit-exact checkpoint reload as
  optimization-plumbing evidence only. Retain harmonic, reconstruction, and
  total losses separately. A batch without eligible harmonic rows may optimize
  reconstruction only under an objective with explicit nonzero reconstruction
  weight; missing supervision never becomes a negative.
- Decision: Keep LR `0.02` and joint harmonic plus visible reconstruction only
  in the one-batch plumbing preset. The production supervised baseline starts
  at LR `3e-4`, harmonic weight `1`, and reconstruction weight `0`. Joint
  visible reconstruction is a separately named ablation. Explicit task
  weights may address only existing fully-supervised active heads;
  positive-unlabeled and deferred open-vocabulary tasks remain disabled.
- Decision: Validation membership is immutable across epochs. The default
  visits the complete validation view exactly once without replacement; an
  optional bounded subset is selected once and fingerprinted. Best-checkpoint
  selection uses only this fixed validation evidence.
- Decision: Aggregate task loss as an epoch numerator divided by its exact
  eligible-row denominator, then compute the epoch harmonic objective from
  explicitly weighted task means. Do not average batch means. Emit the same
  task/count accounting per dataset. Reduce each batch to dataset/task-or-field
  scalars on device, use at most one packed host transfer, and immediately fold
  into CPU aggregates. Retain no device tensor from earlier batches;
  persistent GPU metric memory is zero and CPU aggregate memory is
  `O(dataset_count * task_or_field_count)`. Floating-point aggregation is
  batch/order independent within the documented numerical tolerance, not
  claimed bit-exact under arbitrary reduction orders.
- Decision: Training checkpoints `1.0.0` bind the existing model contract,
  fully resolved configuration fingerprint, and corpus/index/split/
  composition fingerprints. Store model, optimizer, scheduler, AMP scaler,
  next epoch, best validation metric, and Python/CPU-torch/CUDA-torch RNG.
  Prevalidate the complete payload and auxiliary application where possible;
  any live application failure rolls back model, optimizer, scheduler, scaler,
  and all RNG state bit-exactly. An atomic per-epoch metric journal and
  checkpoint `committed_metric_rows` make `last.pt` and `metrics.jsonl`
  recoverable across either write-order crash window. Resume is supported only
  at deterministic epoch boundaries; mid-epoch resume remains explicitly
  deferred. Require the checkpoint loader to reject
  `next_epoch > configured epochs` during prevalidation, before any live-state
  mutation.
- Decision: A fresh run rejects an output directory containing managed
  training artifacts. Explicit `experiment.overwrite_output=true` removes the
  complete known managed set but never unknown user files. Resume cannot use
  overwrite; it validates a versioned run manifest and existing evidence
  fingerprints before journal recovery or any artifact write. An incompatible
  resume leaves both live state and artifacts unchanged.
- Decision: Epoch evidence records `learning_rate_used` before optimizer
  steps and `next_learning_rate` after scheduler advancement. Do not emit the
  prior ambiguous `learning_rate` field.
- Decision: Validate raw graph/target semantics on CPU before transfer. Normal
  CUDA training performs no full gradient-evidence scan. Engine/device
  hot-path functions contain no tensor-to-Python conversion, joint
  reconstruction avoids per-feature-family data-dependent host predicates,
  and metric transfers are explicitly counted at their single packed
  per-batch site. Retained device tensors/bytes are measured from accumulator
  state. Gradient evidence is restricted to one-batch or explicit diagnostic
  mode.
- Decision: Split planning remains target-blind and delegates to
  `plan_group_hash_split`, followed by the existing complete global
  source/lineage validation. CUDA acceptance is optional in CPU CI and must
  skip explicitly; hardware identity and VRAM are reported only from an actual
  CUDA run.
- Consequences: Device-transfer and training-checkpoint contracts begin at
  `1.0.0`. Phase 6A/6B model/output/loss/checkpoint versions, ontology,
  encoding, adapters, production manifests, canonical/graph semantics, and
  corpus contracts do not change. Phase 7 and SSL have not started.

## 2026-07-27 — ADR-044: POP909-CL separates record, raw-input, lineage, graph, and target identity

- Status: Accepted as a Phase 6C pre-merge full-corpus blocker remediation.
- Context: POP909-CL source records 543 and 553 have different source-file
  SHA-256 values but byte-identical score-only projections. Reusing the generic
  MIDI content-addressed piece ID therefore collapsed two corpus records into
  one `(dataset_id, piece_id)`. Weakening duplicate validation, deleting one
  record, or allowing the identical raw input across splits would either lose
  observed target evidence or create leakage.
- Decision: A POP909-CL canonical/source-record identity is
  `piece:pop909-cl-<song-id>`. It is deterministic, path/order/split
  independent, and target independent. The generic MIDI adapter retains its
  existing `piece:midi-<payload-sha256>` default but accepts a validated
  explicit ID before conversion so all canonical entity references are
  consistent.
- Decision: `source_group_id` means target-independent raw-input equivalence
  for POP909-CL and is
  `pop909-cl-score:<score-projection-sha256>`. It participates in the existing
  transitive split closure. `lineage_group_id` remains independently
  `pop909-lineage:<song-id>`; it is not replaced with a raw hash.
- Decision: Canonical serialization preserves the record-specific piece ID.
  Raw graph serialization preserves exact canonical entity references.
  `graph_fingerprint` excludes only the record-specific song entity ID, so
  identical model inputs have identical fingerprints without changing graph
  schema, features, topology, or model-facing tensors. Target bundles retain
  their own identity and never enter raw grouping or graph fingerprints.
- Decision: Keep both 543 and 553 as separate samples in one split-atomic
  component. Their target bundle fingerprints differ: boundary and no-chord
  values/masks agree; bass values differ with equal availability masks; root,
  quality, and inversion values and availability masks differ.
  The bounded evidence classification is “multiple observed target views for
  one score input.” It does not assert an alternative-harmonization semantic
  relation that the source contract does not establish.
- Decision: Preserve strict duplicate `(dataset_id, piece_id)` rejection.
  Diagnostics list every duplicate cluster deterministically with dataset,
  piece, cluster size, source identity, and relative path; absolute temporary
  paths are excluded.
- Consequences: POP909-CL adapter and corpus/production manifest versions
  patch-bump to `1.0.1`, which invalidates old cache hits without deleting old
  immutable artifacts. Corpus index/split structures, canonical schema, graph
  schema/features/topology, ontology, encoding, model, output, loss, and
  checkpoint contracts do not change. Phase 7/SSL remains unstarted.

## 2026-07-27 — ADR-045: Strict graph identity and numerical model-input equivalence are separate

- Status: Accepted as the post-merge correction to ADR-044. The ADR-044
  statement that `graph_fingerprint` excludes song identity is superseded.
- Context: Canonical piece ↔ raw graph binding, external graph validation,
  collator mutation detection, and integrity diagnostics require the complete
  deterministic graph serialization, including every entity ID. Reusing that
  fingerprint for cross-record numerical equivalence weakened those exact
  identity boundaries.
- Decision: Restore `graph_fingerprint` as SHA-256 of the complete validated
  `dumps_graph` serialization. A song entity-ID-only change changes the strict
  fingerprint and fails sample/collator binding.
- Decision: Introduce `model_input_fingerprint@1.0.0` for validated numerical
  model input. It commits to schema/feature/builder versions, `raw_only`,
  ordered feature names, feature and availability tensors, candidate slots,
  ordered edge types, and topology. It excludes entity IDs and all
  targets/provenance/path/group/split sidecars. It is evidence only and cannot
  authorize canonical/graph binding.
- Decision: POP909-CL score-projection `source_group_id` is the authoritative
  raw-equivalence and split-closure identity. Record `piece_id`, independent
  song lineage, strict graph fingerprint, model-input fingerprint, and target
  bundle fingerprint remain distinct contracts.
- Decision: Runtime POP909-CL identity/grouping API is a breaking change from
  `1.0.0`; runtime adapter and corpus/production manifest therefore become
  `2.0.0`. The unchanged target-extraction semantics remain version `1.0.0`;
  ontology `source_adapter=music_critic.adapters.pop909_cl@1.0.0` explicitly
  names that target-semantic contract, not the runtime adapter version.
- Consequences: Corpus index/split structures, canonical schema, graph
  schema/features/topology, ontology `1.0.1`, encoding, model, loss, output,
  and checkpoint contracts remain unchanged. Existing cache artifacts are not
  deleted. Phase 7/SSL remains unstarted.

## 2026-07-27 — ADR-046: POP909-CL anomaly evidence is installation-portable

- Status: Accepted as the final evidence-only remediation for the `2.0.0`
  POP909-CL hotfix.
- Context: Historical Phase 4A/v1 anomaly rows retained the former
  source-path representation and produced
  `d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`.
  That remains valid historical evidence, but an absolute or installation-root
  component is not portable corpus evidence.
- Decision: The current `POP909_CL_ANOMALY_FINGERPRINT` is
  `603ca5eb9fa248ef3e718b0f5d6ddce166b310860473e89e7e35be0a1158662b`.
  It hashes the same eight semantic anomaly rows with corpus-relative paths,
  independent of the supported direct or nested installation layout.
  The old value is exposed only as the explicitly historical
  `POP909_CL_ANOMALY_FINGERPRINT_V1`.
- Decision: Production acceptance independently binds its calculated anomaly
  fingerprint to both the public current constant and the production
  manifest. Public-contract and manifest-contract mismatches have separate
  stable categories and either makes `ready=false`.
- Consequences: This closes evidence linkage only. Record identity, raw-input
  grouping, lineage, strict graph/model-input fingerprints, targets, cache
  keys, split assignments, and all `2.0.0` versions remain unchanged. Existing
  immutable cache artifacts are retained; Phase 7/SSL remains unstarted.

## 2026-07-28 — ADR-047: Supervised evaluation is candidate-first, train-prior-bound, and timing-isolated

- Status: Accepted for Phase 6D-A implementation.
- Context: Phase 6C reports row-weighted loss but cannot show whether an
  imbalanced supervised head beats a majority or empirical-prior baseline.
  Adding held-out metrics must not expose targets to candidate generation,
  combine source-incompatible heads, use test for selection, or weaken
  checkpoint/resume determinism. Real wall-clock observations are inherently
  nondeterministic and therefore cannot be embedded in the byte-exact
  deterministic epoch journal.
- Decision: Reconstruct a fresh accepted baseline from its complete model
  contract and load model weights only while preserving caller RNG. Phase 6C
  data fingerprints are checked against fixed validation evidence; older
  model-only checkpoints state that historical data binding is unavailable.
  Current index/cache/split/composition/membership and ontology/encoding
  evidence is always retained.
- Decision: Call `predict(raw_graph_batch)` before any target-sidecar join.
  Stream only available, aligned, model-ready rows after the join. Conflict
  rows remain unavailable. Key every accumulator by exact dataset and task,
  admit a task only for its ontology-declared source adapter, and never average
  HookTheory and POP909-CL heads together.
- Decision: Build majority, empirical-prior, prevalence, and fixed
  0.5-threshold evidence from the train split only, in a separate
  provenance-bearing artifact. Validation/test labels cannot choose priors or
  thresholds. Metrics with no defined denominator are `null` with a stable
  category/reason rather than a fabricated zero.
- Decision: Test evaluation is fail-closed without explicit acknowledgement
  and never participates in checkpoint selection. Evaluation artifacts are
  deterministic; accumulators retain only fixed registry-sized counts and
  exact likelihood sums.
- Decision: Detailed performance timing is an explicitly enabled bounded
  profiler, never the normal hot path. Normal training records only epoch
  train/validation wall time and throughput in
  `epoch_performance.jsonl`. That nondeterministic sidecar does not participate
  in checkpoint compatibility or the crash-consistent deterministic
  `metrics.jsonl` journal. No CUDA synchronization or unbounded per-batch
  timing history is added.
- Consequences: Evaluation, artifact, train-prior, and profiler contracts begin
  at `1.0.0`. Model, graph, adapter, target, ontology, encoding, cache,
  manifest, corpus, loss, and checkpoint semantics do not change. Raw
  unlabeled inference remains possible; Phase 7/SSL remains unstarted.

## 2026-07-29 — ADR-048: F1 undefined semantics and profiler timing boundaries are explicit

- Status: Accepted as Phase 6D-A remediation.
- Context: Deriving F1 only after requiring separately defined precision and
  recall excluded supported-but-unpredicted classes and unsupported classes
  with false positives. Their lawful zero F1 values disappeared from
  macro-F1. The first profiler artifact also mixed exclusive work with
  repeated alignment and complete DataLoader traversal, so its stage values
  were not one additive or comparable decomposition.
- Decision: Categorical and multilabel per-class F1 is computed directly as
  `2TP/(2TP+FP+FN)`. It is `null` only when the denominator is zero, exactly
  when eligible truth and predictions both omit the class. A class with true
  support but no predicted positives and a class with false-positive
  predictions but no true support both have defined F1 zero. Precision and
  recall keep their independent denominator rules. Macro-F1 averages all and
  only defined per-class F1 values, including zero.
- Decision: Preserve full `(dataset_id, task_id)` metrics as primary evidence.
  A versioned macro-summary view groups only by `(dataset_id, encoding_kind)`
  and computes an unweighted arithmetic mean over defined task metric values.
  It records included/undefined task IDs and counts. Dataset sources or
  categorical/multilabel encodings never cross groups. NLL/BCE and any metric
  whose probability or label-set dimension is task-specific are omitted with
  a structured scientific reason.
- Decision: The serial `workers=0` preparation pass is a result-flow chain:
  canonical read, graph construction, target projection/alignment/
  tensorization, then metadata/statistics assembly using the already
  tensorized targets. Each stage declares per-sample or per-batch units.
  Prepared training compute, prepared validation, full loader traversal, and
  loader-plus-training end-to-end throughput are separate passes and are never
  summed into that decomposition. The loader timers begin before
  `iter(loader)`. With `workers>0`, overlapping startup, IPC, prefetch, sample
  preparation, and collation receive structured unavailable component
  attribution rather than being assigned to collation. RSS is a cumulative
  process high-water mark.
- Decision: An optional production-read-only mode validates explicitly passed
  absolute index/cache/split paths and reads only a capped deterministic,
  fingerprinted subset per dataset. It performs no cache writes, canonical
  corpus scan, or checkpoint read. Detailed profiling remains outside normal
  training and deterministic checkpoint/journal state.
- Consequences: Evaluation and evaluation-artifact contracts advance to
  `1.1.0`; profiler advances to `1.1.0`; the new macro-summary sub-contract
  begins at `1.0.0`. The unchanged train-prior contract stays `1.0.0`.
  Models, checkpoints, adapters, canonical/graph contracts, ontology, target
  semantics, caches, split manifests, and Phase 7 remain unchanged.

## 2026-07-29 — ADR-049: Fixed-validation membership preserves Phase 6C bytes

- Status: Accepted as a blocking Phase 6D-A compatibility hotfix.
- Context: Phase 6C ranked validation identities and fingerprinted membership
  with compact sorted UTF-8 JSON and no terminal newline. Phase 6D-A
  evaluation reused its global canonical fingerprint, which deliberately adds
  a newline. The payload objects looked identical but their SHA-256 values did
  not, so valid Phase 6C checkpoints trained with a partial fixed validation
  view failed `validation_membership_fingerprint` verification. Documentation
  for evaluation `1.1.0` incorrectly claimed exact Phase 6C parity.
- Decision: Training and evaluation import one neutral, versioned
  `fixed_validation_membership_v1` implementation. Ranking hashes the exact
  payload `{policy, seed, identity}` and membership hashes the exact payload
  `{policy, seed, subset_limit, full_view_count, selected_identities}` using
  compact sorted UTF-8 JSON without a terminal newline. Selection remains
  seed-dependent, without replacement, and is emitted in canonical view order.
  `limit=0` and `limit=len(view)` both select the complete view while retaining
  their distinct historical `subset_limit` payload value.
- Decision: Existing Phase 6C checkpoint bytes and metadata are authoritative
  compatibility oracles. No fingerprint substitution, mismatch override, or
  checkpoint migration is permitted. Global evaluation
  `canonical_fingerprint` remains newline-bearing because existing evaluation
  artifacts depend on it. Index, split-manifest, train/validation composition,
  and membership checks remain fail-closed.
- Consequences: Evaluation and evaluation-artifact contracts advance from
  `1.1.0` to `1.1.1`. Fixed-validation membership begins its neutral shared
  contract at `1.0.0`. Macro-summary `1.0.0`, profiler `1.1.0`, train-prior
  `1.0.0`, training checkpoint, model, target, cache, and split contracts do
  not change. Existing Phase 6C checkpoints remain valid without mutation.
  Phase 7 remains unstarted.

## 2026-07-29 — ADR-050: Phase 7A is deterministic GraphMAE2-inspired representation SSL

- Status: Accepted. Final Phase 7A acceptance remediation is implemented in
  draft PR #15. Bounded execution is recorded in
  `PHASE7A_SSL_BASELINE.md`; head-relative Required CI is operational evidence
  recorded in the PR comment, not a pending architectural decision. Its
  initial prepared-input security boundary and current contract-version
  consequences are superseded by ADR-051.
- Context: The Phase 6B hierarchical encoder can consume ordinary raw-only
  MIDI graphs, but visible-feature reconstruction does not prove masked
  representation learning. The first SSL increment must prevent redundant
  pitch, peer-relative-pitch, and owner-track aggregate leakage, remain
  independent of target sidecars and loader ordering, and preserve every
  Phase 6 supervised/no-mask contract. It must not conflate representation
  reconstruction with musical likelihood, preference, or quality.
- Decision: Describe the implementation as GraphMAE2-inspired, not a faithful
  GraphMAE2 reproduction. Phase 7A uses target mode
  `shared_stop_gradient_full_view`: the shared hierarchical encoder produces
  full-view note/bar/song targets under eval/no-grad, while the online path
  uses a model-side masked feature overlay. Targets are detached. No EMA target
  encoder is implemented; EMA remains a future controlled ablation.
- Decision: The only Phase 7A field group is `note_pitch_group`. Resolve note
  `pitch`, `pitch_class`, `octave`, and `track_relative_pitch` through the
  versioned raw feature registry and mask both value and availability
  contributions. Through exact raw note ownership, collateral-mask
  `track_relative_pitch` and availability on every unselected note peer in an
  affected owner track, plus owner-track `mean_pitch`, `pitch_std`, `min_pitch`,
  and `max_pitch` with availability. Peer-note and owner-track collateral
  fields are leakage closure, not reconstruction targets. Duration, velocity,
  timing, topology, bars, and whole tracks remain visible.
- Decision: Use immutable per-sample
  `uniform_note_without_replacement@1.0.0` MaskPlans with canonical SHA-256
  seeds and fingerprints. Train masks depend on epoch; validation masks
  canonicalize epoch to zero. Plans and decoder views do not use Python
  `hash()`, global RNG, target/annotation content, batch order, or worker
  scheduling. Mask overlays neither mutate nor cache raw graphs. The
  no-overlay contribution order and Phase 6 state dict remain unchanged.
- Decision: Construct every MaskPlan and prepared binding from the fully
  validated CPU `SSLBatch` before device transfer. Prepared binding `1.0.0`
  binds ordered identities, raw structure and ownership, stage, canonicalized
  epoch, seed, and exact plan fingerprints, and rejects externally substituted
  bindings failure-closed. It is a runtime sidecar, not graph/cache state.
  CPU and CUDA share this path; prepared accelerator forward performs no
  graph-tensor `.cpu()`, `.tolist()`, or `.item()` plan construction. Report
  CPU plan preparation separately from transfer and compute.
- Decision: Decode selected online note representations through deterministic
  latent re-mask views and a row-wise contextual representation decoder. Mode
  `online_owner_track_bar_song_temporal_neighbors` derives context only from
  masked-online owner-track, available owner-bar, song, and previous/next
  in-track note representations; adding it after re-masking prevents
  fully-remasked predictions from depending on one constant mask token alone.
  Predict detached full-view bar and song latents with separate
  projector/predictors. Use versioned `1-cosine` with `eps=1e-8`,
  `sum_count_mean`, retained and counted zero-norm rows, and explicit
  unavailable components. Anti-collapse diagnostics `1.1.0` separately merge
  target and prediction rows for note, bar, and song across the complete
  stage. For each it exposes row count, dimension, the exact variance formula,
  mean L2 norm, zero-norm count, and global mean off-diagonal cosine, with
  structured unavailability for fewer than two rows. `O(D)` sufficient
  statistics retain no embedding history or `N x N` matrix and make the result
  invariant to batch partition/order and worker count. Epoch artifacts use
  `anti_collapse_aggregate`, not the rejected
  `anti_collapse_last_batch` snapshot.
- Decision: Bound final mechanics acceptance to a deterministic multi-piece,
  multi-note canonical fixture with disjoint train/validation identities and
  explicit multitrack/multibar cases. Its varied pitches and rhythms make the
  configured `0.30` mask select multiple primary rows and exercise nonzero
  peer-note and owner-track collateral masking. Counts and fingerprints are
  execution evidence, not architecture constants.
- Decision: Resolve an unset optimizer learning rate to `3e-4` for the
  Phase 7A one-batch experiment. Explicit optimizer overrides remain
  authoritative, and the generic supervised one-batch preset is unchanged.
  This bounded profile keeps the shared target encoder pitch-sensitive enough
  for the fixed counterfactual margin gate across supported CPU execution
  environments.
- Decision: After bounded one-batch fitting, use pitch-mutation contract
  `1.0.0` and fixed `midi_axis_reflection_v1` policy
  (`mutated_pitch = 127 - source_pitch`), then rebuild the canonical piece,
  raw graph, and dependent raw features under the same fixed MaskPlan. Bind
  each alternative to actual runtime-source fingerprints and a
  selection-specific mutation fingerprint. Require a positive difference
  between `cos(prediction, correct_target)` and
  `cos(prediction, mutated_target)`, and report correct-to-mutated target
  distance. This diagnoses pitch-sensitive representation mechanics and is not
  a label, cross-entropy, normalized likelihood, or PLL.
- Decision: Evaluate fixed, disjoint validation once before any optimizer step
  and after every training epoch. Record train/validation loss and exact
  stage-wide note/bar/song diagnostics; choose the best checkpoint only by
  fixed-validation loss. Deterministic reruns bind the memberships, prepared
  plan fingerprints, initial validation baseline, and metric rows.
- Decision: Keep a simple one-view/no-remask mode and a main
  three-view/`0.20`-remask mode. No superiority claim follows from including
  the multi-view mode. Production cache training uses a dedicated raw-only
  dataset/collator around `load_cached_piece` and `build_raw_graph`, never
  supervised target projection, while reusing group-safe training and fixed
  validation membership. Reports distinguish four scopes: one-batch plumbing,
  bounded held-out/non-collapse evidence, named production-cache execution,
  and production/full-corpus training. The first two establish deterministic
  mechanics only, and cache reads alone establish neither production-training
  nor full-corpus effectiveness. SSL checkpoint/resume is failure-atomic and
  epoch-boundary-only. Pretrained export strictly transfers local encoder,
  hierarchy pooling, Transformer, and fusion parameters without overwriting
  supervised heads.
- Decision: Hierarchical/coarse-to-fine masks belong to Phase 8. Full-scale
  effectiveness requires the Phase 10 PDMX raw-compatible corpus projection
  and rerun. Phase 7A implements no normalized masked-note/pitch-set
  likelihood, perplexity, PLL, critic, reward, preference, aesthetic, or
  quality score.
- Consequences: New Phase 7A SSL, MaskPlan/policy, overlay, maskable-field
  registry, decoder, objective/diagnostic, model/output/target,
  checkpoint/journal, encoder-export, and run/report contracts begin at
  `1.0.0`. The maskable-field registry fingerprint is
  `97836b2adb610529994ae609e89913eb6b21ad0f07d4bf695c911251d5f8ac85`.
  Canonical, graph, adapter, ontology, encoding, split, corpus/cache, Phase 6
  model/output, and Phase 6 checkpoint contracts do not change.
- Consequences: The first acceptance remediation added prepared MaskPlan
  binding `1.0.0` and
  advances SSL contract/model/output, anti-collapse diagnostics, checkpoint,
  epoch journal, metric row, run manifest, training report, and performance
  row to `1.1.0`. MaskPlan/policy, maskable-field registry, representation
  target/objective/loss, decoder, encoder export, and every Phase 6 contract
  remained `1.0.0` at their existing semantics. ADR-051 records the subsequent
  security-remediation versions.
- Consequences: The first real supervised baseline supplies later-ablation
  context only: strong HookTheory tonic/scale-degree and POP909-CL root/bass
  signals, weak or collapsed remaining heads, all-negative HookTheory
  multilabel output with `F1=0` at threshold `0.5`, and POP909-CL validation
  limited to 18 independent pieces. Scientific evaluation hardening and the
  ambiguous `test_not_used_for_checkpoint_selection` field remain registered
  backlog items; neither blocks Phase 7A mechanics or establishes downstream
  improvement.

## 2026-07-29 — ADR-051: Prepared Phase 7A execution requires complete process-local attestation

- Status: Accepted. This is the final Phase 7A security/contract remediation
  in draft PR #15; it does not begin Phase 8 or authorize production training.
- Context: Prepared Phase 7A execution previously attested only a subset of
  the validated raw model input while the SSL model could invoke the Phase 6
  encoder with a caller-controlled `_prevalidated_input=True` boolean. After
  preparation or transfer, mutation/replacement of features, availability,
  candidates, `raw_only`, or store attributes could therefore bypass the
  complete raw-graph validator. The fast path must remain free of graph-sized
  accelerator-to-host materialization while failing closed over the entire
  numerical and structural input.
- Decision: Prepared binding `1.1.0` holds a private, process-local descriptor
  over the exact graph presented to the model. It keeps strong references to
  the graph and every global/node/edge store and binds their object identities
  and types, ordered `node_types` and `edge_types`, and every exact store
  attribute set. For the current model-facing schema, it covers all 65 graph
  tensors: global `raw_only`; `x_cat`, `x_cat_available`, `x_cont`,
  `x_cont_available`, `batch`, and `ptr` for each mandatory node store;
  beat/onset `candidate_slot`; and `edge_index` for every mandatory relation.
  Every tensor, plus the compact selected-note-index tensor, is held by strong
  reference and attested by identity, `_version`, shape, dtype, and device.
- Decision: A typed immutable hash covers all non-tensor metadata used by the
  validated graph contract, including schema/registry/builder values,
  feature-name collections, `num_nodes`, and every `entity_id` collection.
  Entity identifiers are therefore inside, not outside, the prepared-input
  boundary. Adding or removing any global/node/edge attribute—including
  target, theory, split, provenance, diagnostics, or an unknown field—changes
  the exact surface and is rejected before encoder computation.
- Decision: Transfer first re-attests the complete source graph, then
  deep-copies the stores and moves tensor attributes. It compares transferred
  store types, ordered node/edge types, exact attribute sets, non-tensor
  metadata, and tensor shape/dtype/device surfaces, then discards the CPU
  descriptor in favor of a fresh descriptor with strong references and
  identities for the destination graph. No accelerator tensor value is read
  on the host.
- Decision: Runtime graph/store/tensor identities, strong references, version
  counters, device metadata, HMAC secrets, and capability tokens are
  intentionally excluded from `to_dict()`, deterministic binding
  fingerprints, checkpoints, reports, graph serialization, and caches. The
  portable binding fingerprint continues to derive only from immutable
  preparation-stage CPU evidence, so process-local attestation strengthens
  runtime authorization without making deterministic artifacts
  process-dependent.
- Decision: Remove the public boolean bypass. Normal Phase 6 `forward` and
  `encode` calls always run the existing full raw-graph validator. The private
  prepared encoder path accepts only an opaque process-local, HMAC-backed
  capability issued for one exact batch, graph, binding, runtime descriptor,
  and mask rate. The full-view target path and the masked-online path each
  issue and re-attest a fresh capability immediately before encoder
  computation. A foreign graph, forged binding/token, in-place tensor
  mutation, tensor replacement, or surface mutation fails with a structured
  contract error before either encoder runs.
- Decision: Preserve the diagnostic implementation but state its performance
  boundary precisely. Mergeable anti-collapse accumulators retain `O(D)`
  state and no embedding history or pairwise matrix. Current
  `_StreamingEmbeddingStatistics.from_values` nevertheless materializes a
  float64 `N x D` `values64` tensor and a normalized `N x D` working
  temporary. There is no `O(D)` peak-temporary-memory guarantee. Real CUDA
  cost has not been measured, and a separate profiler/optimization gate on an
  RTX 3090 is required before production SSL.
- Consequences: Prepared binding advances to `1.1.0`. SSL contract,
  model/output, checkpoint, epoch journal, metric row, run manifest, training
  report, and performance row advance to `1.2.0`. Anti-collapse diagnostics
  remain `1.1.0`. MaskPlan/policy/overlay, maskable registry, decoder,
  representation target/objective/loss, pitch mutation, and encoder export
  remain `1.0.0`.
- Consequences: Model/checkpoint-derived fingerprints change with the
  incompatible prepared execution boundary; the accepted model fingerprint is
  `7a1ece2b44dc6b52aef6f7c7532238d4716b1a45c38b8ca66957225a24b76774`.
  Canonical, graph schema, feature registry, cache, split, and every Phase 6
  numerical/state-dict contract remain unchanged. Raw unlabeled MIDI inference
  continues through the ordinary fully validated Phase 6 path.

## 2026-07-30 — ADR-052: Runtime CUDA devices resolve to a concrete index before transfer

- Status: Accepted for the blocking post-merge Phase 7A hotfix. This decision
  does not begin Phase 8 or authorize production training.
- Context: Independent RTX 3090 execution after PR #15 merged found two SSL
  CUDA+AMP failures. Transfer received abstract `torch.device("cuda")`, PyTorch
  placed tensors on concrete `cuda:0`, and the strict validator correctly
  observed that those device objects are unequal. CPU CI skipped the real-CUDA
  paths and could not expose this distinction.
- Decision: Use one runtime-device resolver for SSL, supervised training, and
  evaluation. Canonicalize CPU to bare `cpu`, resolve bare CUDA through
  `torch.cuda.current_device()`, and preserve every explicit `cuda:N`. Any CUDA
  request when CUDA is unavailable is a structured contract failure.
- Decision: Preserve exact device validation. Comparing only `device.type` is
  forbidden because a tensor expected on `cuda:1` must reject actual
  `cuda:0`. Resolution and validation inspect only device metadata; they do
  not call `.cpu()`, `.item()`, `.tolist()`, or otherwise materialize tensor
  values on the host.
- Decision: SSL mismatch category remains
  `ssl.data.device_transfer_tensor_mismatch`. Evidence adds one stable
  location—`global:<attribute>`, `node:<node-type>:<attribute>`,
  `edge:<source>|<relation>|<destination>:<attribute>`, or
  `binding:<field>`—and concrete expected/actual devices without object
  `repr`.
- Consequences: Device-transfer contract advances from `1.0.0` to `1.0.1`.
  Umbrella SSL and SSL training-report contracts advance from `1.2.0` to
  `1.2.1`; reports now expose concrete `cuda:N`. Prepared binding remains
  `1.1.0`, while SSL model/output, checkpoint/journal/metric-row,
  run-manifest/performance-row, objectives, masking, decoder, registry,
  fixture, and encoder-export versions remain unchanged.
- Consequences: New umbrella-SSL metadata changes derived model-contract and
  checkpoint-binding fingerprints. Historical Phase 7A `1.2.0` evidence is
  retained rather than regenerated. Exact metadata makes old bounded SSL
  checkpoints non-resumable under `1.2.1`; no migration is added in this
  narrow hotfix. Graph/canonical schema, ontology, encoding, datasets, caches,
  model architecture, numerical objectives, and production artifacts do not
  change.

## 2026-07-30 — ADR-053: CUDA indices and AMP representation objectives fail closed

- Status: Accepted as remediation of draft PR #17. This decision extends
  ADR-052, does not begin Phase 8, and does not authorize production training.
- Context: An independent RTX 3090 run at hotfix head `fb54e85` confirmed that
  bare `cuda` resolves to `cuda:0`; the original
  `ssl.data.device_transfer_tensor_mismatch` disappeared, and graph plus
  prepared-binding tensors passed exact `cuda:0` assertions. The remaining
  failures exposed four separate issues: explicit CUDA indices were not
  checked against visible devices; subsystem allowlists and a string-based AMP
  check rejected valid `cuda:N`; decoder predictions under AMP were FP16 while
  stop-gradient targets were FP32; and two CUDA acceptance assertions mutated
  unavailable raw values or compared JSON lists directly with live tuples.
- Decision: Runtime resolution `1.0.1` validates both explicit and current CUDA
  indices against `torch.cuda.device_count()` before any tensor transfer.
  `runtime.device.cuda_index_out_of_range` carries the requested device and
  visible count; a bare-current failure also records the resolved index.
  Training, SSL, and evaluation accept `cpu`, `cuda`, `cuda:N`, and `auto`
  through the shared resolver. AMP eligibility depends on
  `resolved_device.type == "cuda"`, never the unresolved input string.
- Decision: Representation loss `1.0.1` requires equal shape and concrete
  device and floating inputs. Any FP16/BF16/FP32 pair computes row-wise cosine,
  differentiable empty and non-empty numerators, means, and reductions in FP32
  with autocast disabled. Only a matching FP64 pair remains FP64; incompatible
  combinations fail. The target is detached before an out-of-place cast, and
  the prediction cast retains gradient flow. Multi-view loss and combined SSL
  objective advance to `1.0.1`. Immediate and streaming anti-collapse
  diagnostics apply the same numerical normalization and advance to `1.1.1`.
- Decision: CUDA velocity perturbation selects only available sample-zero
  values, preserves unavailable placeholders exactly, and reruns raw-graph
  validation before model execution. JSON-loaded membership evidence is
  compared with live evidence only after canonical JSON normalization, while
  fingerprint, selected count, dataset counts, subset limit, full-view count,
  ordered identities, and byte-identical metric journals remain separate
  exact assertions. Production membership selection, fingerprints, resume
  binding, raw validation, and placeholder policy are not weakened.
- Consequences: Device transfer advances to `1.0.2`; umbrella SSL advances to
  `1.2.2`. SSL training report remains `1.2.1`, and SSL model/output,
  checkpoint/journal/metric-row, run-manifest/performance-row, prepared
  binding, representation target, decoder, masking, graph/canonical schema,
  ontology, encoding, dataset, cache, and model-architecture contracts remain
  unchanged. PR #17 stays draft until the exact final commit passes Required
  CI and an independent RTX 3090 rerun records passed/skipped counts plus
  bounded-smoke peak allocated/reserved VRAM.

## 2026-07-30 — ADR-054: Leakage safety, reconstruction sensitivity, and target preference are separate evidence

- Status: Accepted as continued remediation of draft PR #17. This decision
  supersedes only ADR-050's bounded positive-margin acceptance gate. It does
  not change the masking/model objective, begin Phase 8, or authorize
  production training.
- Context: Independent RTX 3090 execution at exact head `145ee10` produced
  `195 passed, 1 failed, 1 skipped` for the complete SSL suite,
  `15 passed, 1 skipped` for the training CUDA suite, and a passing prepared
  CUDA AMP test. The sole failure was the bounded two-step CUDA AMP smoke.
  Raw stores remained bit-exact, runtime-source binding passed, MaskPlan and
  prepared-binding fingerprint stayed fixed, masked-online embeddings and
  predictions stayed bit-exact, the hidden full-view target and reconstruction
  loss changed, metrics were finite, and target distance was positive. The
  only rejected field was a correct-minus-mutated cosine margin of
  `-0.04540175199508667` from an FP16 prediction with an FP16-derived floor of
  `0.0078125`.
- Decision: A signed correct-target-preference margin after two optimizer
  steps is not a no-leakage invariant and is not required to prove that a
  controlled pitch mutation forms an effective reconstruction challenge. A
  nearly untrained model may prefer either target without implying leakage or
  broken CUDA/AMP plumbing. Preserve the signed value; never relabel a
  negative margin as leakage, hide it, clamp it, or adjust a tolerance to
  change its sign.
- Decision: Introduce independent
  `no_leakage_mutation_evidence@1.0.0` and
  `pitch_sensitive_reconstruction_evidence@1.0.0` objects. Each is a separate
  dictionary with its own domain-separated canonical SHA-256 fingerprint.
  The report must not alias one mutable object under both keys.
- Decision: No-leakage `passed` requires an applicable mutation; bit-exact raw
  graph stores; valid runtime-source binding; fixed MaskPlan; fixed prepared
  binding fingerprint; strict `torch.equal` masked-online embeddings and
  predictions; an actually changed hidden full-view target; and finite
  metrics. Reconstruction-loss change, target-distance magnitude, margin, and
  correct-target preference do not participate.
- Decision: Pitch-sensitive reconstruction `passed` requires an applicable
  mutation, changed hidden full-view target, positive correct-to-mutated target
  distance, changed reconstruction loss, and finite metrics. Raw-online
  leakage flags and correct-target preference do not participate.
- Decision: Compute decoder-view averaging, both target cosines, target L2
  distance, signed margin, and numerical floors in FP32 under explicitly
  disabled autocast regardless of FP16/BF16/FP32 source dtype. Record
  `source_dtype`, `diagnostic_compute_dtype="float32"`, and
  `margin_floor=8*finfo(float32).eps`. Reject nonfinite sources, diagnostics,
  or reconstruction losses before fingerprinting/JSON serialization.
- Decision: Retain correct-target preference as a finite diagnostic with
  `correct_target_preference_observed`, signed
  `correct_minus_mutated_margin`, both cosines,
  `preference_status=observed|not_observed`, and
  `preference_is_acceptance_criterion=false`. A scientific claim that a model
  prefers the correct target requires a genuinely trained checkpoint and
  held-out evaluation, not a bounded two-step smoke.
- Consequences: SSL training report advances from `1.2.1` to `1.2.2`; both
  new evidence contracts begin at `1.0.0`. Umbrella SSL remains `1.2.2`;
  model/output, checkpoint/journal/metric-row, run-manifest/performance-row,
  representation objective/loss, anti-collapse diagnostics, pitch mutation,
  prepared binding, masking, graph/canonical schema, ontology, encoding,
  dataset, cache, and model architecture contracts remain unchanged.
  Historical Phase 7A positive-margin values and fingerprints remain
  historical diagnostics. At this decision point PR #17 remained draft
  pending Required CI and independent RTX 3090 evidence; it was subsequently
  accepted and merged at main
  `5afec305cfa62ab2c200c5b1e7270ae35cd8a102`.

## 2026-07-30 — ADR-055: Phase 8A adds start-anchored hierarchy views without new objectives

- Status: Accepted for implementation on the Phase 8A draft branch. Merge
  still requires maintainer review and both final-head Required workflow runs.
  This ADR does not start Phase 8B or authorize production training.
- Context: Accepted Phase 7A masks independent note-pitch rows. Musical events
  also have raw onset, beat, bar, and track ownership that can define coherent
  views without adding theory labels or changing the graph. The increment must
  preserve Phase 7A artifacts/checkpoints and its complete prepared-input
  security boundary.
- Decision: Define exactly five Phase 8A policy names. Control
  `independent_note_pitch` dispatches directly to the unchanged Phase 7A
  `uniform_note_without_replacement` builder. New
  `onset_pitch_descendants` follows `onset -> starts_note -> note`;
  `beat_pitch_descendants` follows
  `beat -> contains_onset -> onset -> starts_note -> note`;
  `contiguous_bar_pitch_span` follows
  `bar -> contains_onset -> onset -> starts_note -> note` over one contiguous
  bounded bar range; and `track_bar_pitch_span` intersects that range with one
  raw `track -> contains_note -> note` ownership set. No melody/chord/bass/
  voice role is accepted.
- Decision: Span semantics are start-anchored. A note beginning before a
  selected span is not primary merely because it remains active inside it.
  `active_at`/`has_active_note` remain visible encoder topology and are never
  traversed by the planner.
- Decision: Every new policy is pitch-only. Primary rows mask note `pitch`,
  `pitch_class`, `octave`, and `track_relative_pitch`, including availability.
  The unchanged Phase 7A closure also masks relative pitch/availability on
  unselected affected-track peers and mean/std/min/max pitch/availability on
  affected tracks. Rhythm, velocity, membership, and topology remain visible;
  collateral rows are not reconstruction targets.
- Decision: The fail-closed audit pins all 68 raw feature identities and raw
  registry fingerprint
  `567a5fdbb0d132010af4716c5988686c2bdf998cf6f1b2eec897f8af3ca8c0e2`.
  It serializes the four primary fields, four unique owner-track collateral
  fields, and the exact ordered 60-field visible remainder; the peer-note
  relative-pitch field is already one of the primary identities but applies
  to a different row population. The resulting leakage-audit fingerprint is
  `27fc135b61649e5b892036dd0aacc92f679493ff671320c8235d33396a7c9949`.
  The eight unique note/track fields above are the only current exact pitch
  values, duplicates, or aggregates. Same-track simultaneous-note topology
  can expose relative rank because canonical ordering uses a pitch tie-break;
  it is deliberately visible and is not described as pitch-information-free.
- Decision: Build one target-blind CPU sparse hierarchy index. It rejects
  duplicate/missing note-onset, onset-beat/bar, beat-bar, and note-track
  ownership, disagreement between an onset's direct bar and its owning beat's
  bar, duplicate relevant edges, cross-sample endpoints, disagreement between
  direct and composed start-bar ownership, and malformed bar continuity.
  Its portable structure fingerprint contains local counts/endpoints, never
  batch position, feature values, entity IDs, targets, provenance, or
  diagnostics.
- Decision: For onset/beat policies, apply a versioned linear deterministic
  permutation, visit units once, deduplicate descendants, and compare valid
  prefixes immediately before/after crossing the requested hidden-note
  budget. For spans, enforce
  `1 <= min_span_bars <= max_span_bars <= 8`, enumerate bounded contiguous
  candidates, and use stable seed evidence for equal-distance ties.
  Track/bar enumeration is sparse around occupied cells rather than dense
  tracks-by-bars. Every available hierarchy plan masks at least one note and
  leaves at least one pitched note visible.
- Decision: An impossible or disabled policy produces a versioned structured
  reason and never silently falls back. A versioned configuration records all
  ordered weights and span bounds. Mixture resolution explicitly records the
  eligible set, deterministic renormalized weights, resolution seed, selected
  policy, and report-level realized frequencies.
- Decision: Introduce a distinct portable
  `PreparedHierarchyMaskBinding@1.0.0` envelope and
  `Phase8AHierarchySSLForwardOutput@1.0.0` instead of changing the existing
  Phase 7A binding/output shapes. The hierarchy binding reuses the exact
  `PreparedMaskBinding@1.1.0` graph/store/tensor runtime-evidence kernel, HMAC,
  opaque token, transfer renewal, and private prepared encoder, and
  additionally binds configuration and ordered resolution fingerprints.
  There is no parallel validator or public bypass. The portable Phase 7A
  `to_dict()`/fingerprint is the compatibility boundary even though the
  internal shared dataclass has hierarchy-only optional fields used by its
  strict subclass. An independent-only configuration delegates to the old
  preparation function, preserving old plan, overlay, binding, and numerical
  artifacts exactly.
- Decision: Phase 8A adds no model parameter, head, objective, Hydra root
  field, checkpoint metadata field, or training-engine artifact. Existing
  Phase 7A note/bar/song objectives are bounded integration smoke only.
  Phase 8B owns future onset/beat/bar/track objectives, held-out comparison,
  and ablations.
- Decision: Phase 8A is mechanics evidence, not likelihood, PLL, critic,
  quality, representation-improvement, or scaled-effectiveness evidence.
  No HookTheory/POP909 full scan, PDMX projection, production cache rebuild, or
  production/full-corpus SSL training is authorized.
- Consequences: New hierarchical plan, policy, configuration, mixture,
  unit/descendant evidence, unavailable reason, prepared hierarchy binding,
  hierarchy output, profile, leakage-audit, fixture, acceptance, and benchmark
  contracts start at `1.0.0`. The policy contract fingerprint is
  `b188e90a60d3ec6184dfdb3233ef37b1a0ea133cd5957a10fad3eddf58d77ccd`.
  Existing MaskPlan/policy/overlay, `PreparedMaskBinding@1.1.0`,
  `SSLForwardOutput@1.2.0`, SSL model/checkpoint/config, graph/canonical/cache/
  split, target, and Phase 6 contracts do not change.
- Consequences: Planner/index work is
  `O(nodes + relevant edges + emitted candidate/mask entries)` for the
  contract-fixed span bound. SplitMix64/Fisher-Yates unit permutation and
  linear scans avoid comparison sorts on node-sized planner inputs. No dense
  node-unit/note-note matrix, clique, full `O(B²)` spans, or all-note-pairs
  loop is constructed.
  Benchmark
  retained JSON bytes are not Python/CUDA/total peak-memory evidence, and no
  GPU performance claim is made.

## 2026-07-30 — ADR-056: Phase 8A span choice uses a bounded near-optimal seed-dependent pool

- Status: Accepted for the final pre-merge remediation of draft PR #16.
  Merge still requires both Required workflow runs and independent
  exact-final RTX 3090 CUDA/AMP evidence. This decision does not start Phase
  8B or authorize production training. It supersedes only ADR-055's
  exact-closest span-selection sentence and its initial versions for the
  contracts listed below.
- Context: ADR-055 selected the exact-closest span and used the seed only to
  break equal-error ties. A graph with one unique closest candidate therefore
  chose the same actual span across epochs even though a seed was bound. Phase
  8A requires deterministic, seed-dependent view diversity without permitting
  unbounded budget error, dense candidate structures, or accelerator-to-host
  planning.
- Decision: Replace only the span selector with
  `bounded_near_optimal_seed_rank_v1`. First scan all valid candidates for the
  best absolute hidden-note budget error. In a second pass admit candidates
  through `best_error + span_budget_error_slack`, retain the canonically
  smallest configured number ordered by error, track, start, end, and exact
  descendants, then choose one retained candidate by domain-separated stable
  seed SHA-256 rank with the canonical key as collision fallback.
- Decision: Policy configuration binds integer
  `span_selection_pool_size` in `[1, 8]` and
  `span_budget_error_slack` in `[0, 8]`. Defaults are pool size `4` and slack
  `1`. Pool size `1` is the canonical exact-closest control; slack `0`
  restricts admission to exact-best error. The pool is canonical and therefore
  independent of candidate enumeration order.
- Decision: Selection evidence records total valid candidates, best error,
  tolerance-qualified candidates, retained-pool count, configured pool/slack,
  selected error/start/end/track, selected descendant count, realized mask
  rate, and method identifier. Plan validation binds these fields to the
  configuration and canonical graph-aware recomputation.
- Decision: Bounded-fixture audit uses seed `42`, mask rate `0.30`, and train
  epochs `0..63`. Both span policies produce actual selection diversity for
  every one of the four train identities and every selected error is within
  `best + 1`. In the crafted unique-closest single-bar track case, six valid
  candidates contain one error-0 candidate and five error-1 candidates; the
  default pool selects four actual spans with error distribution
  `0:14, 1:50` and exact replay. Pool size `1` and slack `0` each retain the
  unique closest selection. This audit justifies bounded defaults, not masking
  quality or corpus representativeness.
- Decision: Existing candidate construction retains `O(C+S)` candidates and
  descendant entries. The bounded selector uses `O(C*K)=O(C)` time because
  `K <= 8`, plus `O(K)` selection scratch. It performs no full/unbounded
  candidate sort, dense tracks-by-bars representation, or pairwise note
  construction.
- Decision: Preserve the independent Phase 7A control exactly and reprove it
  on CPU and, when available, explicit indexed-CUDA with AMP. The optional
  command emits `Phase8ACudaAmpHardwareEvidence@1.0.0` separately from
  portable CPU acceptance. All five policies and the mixture must bind
  concrete `cuda:0`, complete finite forward/loss/gradient execution, and
  record peak allocated/reserved VRAM. CUDA absence is an honest skip, never
  substituted CPU evidence. Independent exact-final RTX 3090 execution remains
  a pre-merge gate.
- Decision: Streaming anti-collapse diagnostics are not redesigned here.
  Retained state is `O(D)`, but current `from_values` creates temporary
  float64 `N x D` values and normalized working tensors. No `O(D)` peak
  temporary-memory claim is permitted; real CUDA cost remains unmeasured, and
  production SSL requires a separate RTX 3090 profiler/optimization gate.
- Consequences: Hierarchical plan, policy, configuration, selection evidence,
  prepared hierarchy profile, prepared hierarchy envelope, bounded acceptance,
  and benchmark contracts advance to `1.1.0`. Mixture, unavailable reason,
  `Phase8AHierarchySSLForwardOutput`, fixture, and leakage audit remain
  `1.0.0`. The hardware-evidence artifact begins at `1.0.0`.
- Consequences: Accepted post-hotfix semantics remain fixed: device transfer
  `1.0.2`, representation/multi-view/objective `1.0.1`, umbrella SSL `1.2.2`,
  training report `1.2.2`, `PreparedMaskBinding@1.1.0`, and independent
  no-leakage/pitch-sensitive-reconstruction evidence `1.0.0`. Correct-target
  preference remains a signed non-gating diagnostic. Graph, feature,
  canonical/cache/split, Phase 6 numerical, model, checkpoint, and
  independent-control contracts do not change.
- Consequences: Phase 8B objectives, Dilemmadata, PDMX, PLL, adaptive masking,
  preference or critic learning, production training, and
  representation-quality claims remain out of scope.

## 2026-07-30 — ADR-057: Phase 8A span pools rank the complete tolerance set

- Status: Accepted for the remaining pre-merge remediation of draft PR #16.
  This supersedes ADR-056 only where it retained a canonical prefix before
  applying a seed rank. The PR remains draft; Phase 8B and production
  training remain unauthorized.
- Context: ADR-056 admitted every candidate through `best + slack` but then
  retained the canonically first `K` by error/track/start/end/descendants.
  With more than `K` equally near-optimal candidates, later tracks and bars
  could never enter the pool under any seed or epoch. Diversity inside that
  prefix did not establish coverage of the full tolerance-qualified set.
- Decision: `bounded_near_optimal_seed_rank_v2` first scans for best budget
  error and preserves the same `candidate_error <= best + configured_slack`
  admission rule. For every admitted candidate it computes
  `stable_seed_sha256_pool_membership_v1`, binding dataset/piece identity,
  canonical epoch, encoder view, global seed, policy/version, configuration
  fingerprint, and full canonical candidate identity. A bounded streaming
  selector retains the `K` smallest ranks. Canonical candidate identity is
  only the collision fallback.
- Decision: Final choice uses the distinct
  `stable_seed_sha256_final_choice_v1` domain over the retained pool. The
  membership hash is not reused. Enumeration order, Python hash, process RNG,
  global random state, and unordered iteration cannot affect membership or
  choice. Validation still canonicalizes epoch to zero.
- Decision: `span_selection_pool_size` now unambiguously means seed-ranked
  retained-pool size over the complete tolerance set. `K=1` is a seed-ranked
  singleton, not a canonical exact-closest control. Slack `0` remains the
  exact-best admission control. Primary/visible-note rules, start-anchored
  descendants, track/sample boundaries, and `active_at` exclusion do not
  change.
- Decision: Selection evidence binds tolerance and retained counts, both rank
  method identifiers, the complete selected canonical identity, selected
  error, and existing budget/realized-rate evidence. The bounded acceptance
  additionally reports selected start-bar min/max, distinct selected tracks,
  and obsolete-canonical-prefix escape count.
- Decision: The positional-bias oracle has 36 tolerance-qualified candidates
  over three tracks and bars `0..11`, with one unique error-0 candidate and
  35 error-1 candidates. Across train epochs `0..255`, all 36 enter a retained
  pool and all 36 are selected; start bars span `0..11`, all three tracks are
  selected, 224 choices escape the obsolete four-candidate prefix, and error
  histogram is `0:7, 1:249`. Replay and reverse/permuted enumeration are
  bit-exact. This is deterministic reachability/mechanics evidence, not a
  claim of unbiased or uniform random sampling.
- Decision: Candidate generation retains its existing `O(C+S)` sparse
  candidates/descendants. Best-error scan plus bounded top-K insertion costs
  `O(C*K)=O(C)` for contract-fixed `K <= 8`, with `O(K)` selector scratch and
  no full sort, dense track/bar grid, or `O(B²)` span materialization.
- Decision: Use a minor rather than patch bump because the meaning of the
  serialized pool-size field and the plan/evidence selection semantics change
  incompatibly. Hierarchical plan, policy, configuration, selection evidence,
  prepared hierarchy profile/envelope, CPU acceptance, and benchmark advance
  from `1.1.0` to `1.2.0`. CUDA hardware evidence advances from `1.0.0` to
  `1.1.0` because it binds those portable contracts. Mixture, unavailable
  reason, hierarchy output, fixture, and leakage audit remain `1.0.0`.
- Consequences: Policy fingerprint becomes
  `2d39eb5e1ddf6ad53c626a18b364d0ffae0896663008a4e1422215c0c20fbdb1`;
  default configuration becomes
  `e38651e00726ce9681dc015634c5d1f48f11586d07e0faf3187e20bda9ffee67`.
  Phase 7A, device-transfer, raw graph, feature registry, canonical/cache/
  split, Phase 6 numerical, model architecture, and checkpoint contracts do
  not change. Any RTX result for an earlier head is intermediate evidence;
  exact-final RTX 3090 execution remains a separate pre-merge gate.

## 2026-07-30 — ADR-058: Phase 8A CUDA evidence separates exact semantics from bounded backend numerics

- Status: Accepted for the final pre-merge remediation of draft PR #16. The
  PR remains draft; Phase 8B and production training remain unauthorized.
- Context: An independent RTX 3090 run at intermediate SHA `00ba0f38` found
  two evidence-path defects. Direct
  `python scripts/accept_phase8a_cuda_amp.py` could not resolve an import from
  the `scripts` package, so no hardware artifact was emitted. Separately, the
  all-policy SSL run treated CPU FP32 and CUDA FP32 embeddings as nearly
  bit-exact and exceeded `rtol=1e-4`, `atol=1e-5` by a maximum absolute
  difference of `3.632158041000366e-05`; all preceding semantic invariants
  were exact.
- Decision: Move reusable CPU and CUDA acceptance implementations under
  `music_critic.ssl` and keep both documented `scripts/accept_phase8a_*.py`
  entrypoints as thin wrappers. Direct root execution is normative; module
  execution remains supported. No wrapper mutates `sys.path`. Exact-final
  host preflight validates the portable report path, exact HEAD, hotfix
  ancestry, and clean tree before CUDA resolution or output creation.
- Decision: Preserve bit-exact gates for hierarchy plans and selection,
  prepared bindings, overlay/masks, selected indices, raw graph/topology,
  target/provenance blindness, leakage mutations, same-device replay, and the
  independent Phase 7A-versus-Phase 8A CUDA control. No tolerance applies to
  those contracts.
- Decision: Treat CPU FP32 versus CUDA FP32 embeddings, predictions, targets,
  and required losses only as bounded numerical-parity diagnostics. Fix
  `rtol=1e-3`, `atol=5e-5`, and minimum cosine similarity `0.999` before the
  final hardware rerun. Record exact shape/dtype/finite status, total tensor
  and element counts, max absolute/relative error, cosine, and total-objective
  difference. Report per policy and per encoder node type. Tests prove a
  value just inside the elementwise boundary passes, a value just outside
  fails, and a large direction change fails the cosine gate. Thresholds are
  not searched or widened from observed runs.
- Decision: Cross-backend closeness is not evidence that CPU and CUDA execute
  identical floating operations. The final artifact must be regenerated on
  the exact final SHA; the targeted success at `00ba0f38` is intermediate
  evidence only, and its failed CLI produced no hardware acceptance.
- Consequences: Serialized hardware evidence changes incompatibly and
  advances from `Phase8ACudaAmpHardwareEvidence@1.1.0` to `1.2.0`.
  Hierarchical mask/selection/prepared contracts, portable CPU acceptance,
  benchmark, Phase 7A, device transfer, graph/feature/data, model/checkpoint,
  and Phase 6 numerical contracts do not change.

## 2026-07-30 — ADR-059: Exact-final CUDA preflight is deterministic in shallow CI checkouts

- Status: Accepted as Required-CI remediation for draft PR #16. Phase 8B and
  production training remain unauthorized.
- Context: Both Required runs for SHA `25ac6c7` failed the same negative
  subprocess test under GitHub Actions `fetch-depth: 1`. The test deliberately
  dirtied the checkout, but exact-final preflight attempted
  `git merge-base --is-ancestor` first. Because the accepted-hotfix ancestor
  was absent from the shallow object set, Git returned status 128 and leaked a
  raw `CalledProcessError` instead of the required structured dirty-tree
  rejection.
- Decision: After portable-report readability, exact-final preflight checks
  exact HEAD and then dirtiness before the ancestry proof. A dirty checkout
  always raises `phase8a.cuda.source_tree_dirty` without consulting ancestry.
  A clean checkout must still prove that the accepted hotfix is an ancestor;
  a missing or unavailable proof raises the structured
  `phase8a.cuda.hotfix_ancestor_missing_or_unavailable` error.
- Consequences: The exact-final gate is not weakened and no CUDA resolution or
  hardware-evidence output occurs on either failure path. Independent hardware
  execution must fetch enough Git history to prove ancestry. Unit tests cover
  dirty-before-ancestry ordering and clean shallow-history failure. No
  serialized Phase 8A contract, fingerprint, model/checkpoint version, graph
  schema, or Phase 6 numerical output changes.

## 2026-07-30 — ADR-060: CUDA raw-graph parity reuses the common store surface and portable report hashes are runtime-local

- Status: Accepted as narrow Phase 8A GPU-only remediation in draft PR #16.
  Phase 8B and production training remain unauthorized; independent
  exact-final RTX 3090 evidence is still pending.
- Context: The independent RTX 3090 run on `4da0988` produced two mutually
  byte-identical CPU reports, then all seven CUDA failures stopped in
  `_graphs_cross_device_bit_exact` with
  `NameError: name '_store_items' is not defined`. Commit `25ac6c7` moved the
  acceptance implementations into `music_critic.ssl` and added this parity
  function, but imported only three of its common CPU-acceptance dependencies.
  `_store_items` and the subsequent `_metadata_signature` dependency remained
  defined in `phase8a_acceptance.py`. CPU CI skipped every call to the
  CUDA-gated parity function.
- Decision: Import and reuse the existing common private store enumerator
  rather than create a second implementation. The enumerator reads the
  existing global store and existing `node_items()`/`edge_items()` only,
  discriminates node and edge identities, and does not create stores. Raw
  cross-device equality requires exact ordered node/edge type surfaces, exact
  ordered attribute surfaces, tensor shape/dtype/value equality after the
  explicit evidence-only CPU transfer, and exact non-tensor metadata
  signatures. A mismatched key surface rejects before an injected
  target/provenance-like value is read.
- Decision: Execute this helper on ordinary CPU graph pairs in Required CI.
  Regressions cover equivalent batches, value/dtype/shape changes,
  missing/extra node and edge attributes, global mutation, reordered
  attributes and stores, target non-access, input-surface non-mutation, and
  failure-atomic artifact creation/replacement/temporary cleanup. The parity
  gate, all five policies, mixture, standalone CUDA CLI, AMP objective and
  gradient finiteness, and complete-success-only hardware artifact remain
  enabled.
- Decision: `portable` CPU acceptance means hardware-independent content, not
  byte identity across arbitrary Python/PyTorch/CPU-kernel environments. The
  report deliberately includes bounded FP32 loss observations. Fresh processes
  in one compatible runtime must be byte-identical; cross-host acceptance
  compares versioned contracts and deterministic fixture/model/policy/plan/
  overlay/binding fingerprints, while hardware evidence binds the SHA-256 of
  the exact host-local report consumed. Local
  `2d107944c38d8ee465d73f2f71f07b224451f5a31213e86e0049dbaf3958c8f4`
  and independent-host
  `076bb56126dd1ba262014b553a5009e93bd464dac99564531b01fcea09f941b1`
  therefore are not claimed globally equal.
- Consequences: This restores an already specified exact semantic/security
  gate and clarifies its evidence boundary. No public or serialized schema,
  contract version, acceptance/model/checkpoint fingerprint, graph/model/
  ontology contract, or Phase 6 numerical output changes. The failed
  `4da0988` run emitted no hardware report and cannot be used as successful
  CUDA evidence.
