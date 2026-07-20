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
