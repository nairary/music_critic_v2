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

## 2026-08-14 — ADR-061: Phase 8A exact replay owns a scoped deterministic evidence runtime

- Status: Accepted as pre-merge remediation in draft PR #16. This does not
  accept Phase 8A: both Required workflows and independent exact-final RTX
  3090 evidence remain mandatory. The PR stays draft; Phase 8B and production
  training remain unauthorized.
- Context: Independent execution at
  `4c71990f0df715afee9040908da4c99b17f9d99d` produced two byte-identical
  host-local CPU reports at SHA-256
  `076bb56126dd1ba262014b553a5009e93bd464dac99564531b01fcea09f941b1`.
  The isolated two-file CUDA invocation then failed all five policies only at
  first-versus-repeated same-device output equality
  (`5 failed, 62 passed, 2 warnings`). Full SSL immediately afterward passed
  (`357 passed, 1 skipped, 8 warnings`), and standalone CUDA acceptance also
  passed and emitted hardware fingerprint
  `1d7f904afa538c71111c33f72c78a90b4739814ee3d75343fd4438bfe57c5a5d`.
  Overall orchestration correctly failed, so that artifact is insufficient and
  is invalid for every later head.
- Context: The standalone builder entered a private deterministic context,
  but the optional five-policy pytest performed its repeated forwards
  directly. Separately, training tests invoked run-scoped configuration that
  set process-global Torch deterministic and cuDNN flags without test-scoped
  restoration. The later full suite could therefore make the direct CUDA test
  pass as an accidental order effect. The scientific/model operation was not
  shown to require an equality relaxation.
- Decision: Introduce public package contract
  `DeterministicCudaEvidenceRuntime@1.0.0` and use it for standalone Phase 8A
  CUDA acceptance, the five-policy CUDA pytest, and all repeated same-device
  hierarchy-output evidence. It validates an existing
  `CUBLAS_WORKSPACE_CONFIG` against `:4096:8|:16:8`, installs `:4096:8` when
  absent, enables Torch deterministic algorithms with `warn_only=false`,
  disables cuDNN benchmarking, and enables cuDNN determinism before CUDA
  model/device work. A nondeterministic operation raises; it is not converted
  into a warning, skip, xfail, or tolerance.
- Decision: The context snapshots and failure-atomically restores the CPU RNG,
  all CUDA RNG states, deterministic-algorithm enabled/warning state, both
  cuDNN flags, and the prior CUBLAS environment presence/value. Normal,
  exception, nested, and repeated paths are covered by CPU-accessible tests;
  mocked CUDA RNG proves the same restoration surface without pretending to
  be hardware evidence. The pytest boundary independently restores Torch,
  cuDNN, and CUBLAS flags after every test so one training test cannot
  authorize a later evidence test. Production training configuration remains
  unchanged.
- Decision: Keep exact same-device replay as a strict gate. Add bounded
  `Phase8AOutputDifferenceDiagnostic@1.0.0`, which recursively compares the
  hierarchy-output dataclass and reports the first path, all retained paths up
  to a fixed maximum of 64, total/group counts, shape/dtype/device,
  differing-element count, max absolute/relative difference, and finite
  FP16/FP32 ULP distance. Embeddings, predictions, targets, loss tensors, and
  other fields are grouped separately. Diagnostics retain only bounded
  scalar/string/tuple evidence and never modify or retain production outputs.
- Decision: Preserve the existing cross-backend boundary. CPU FP32 versus CUDA
  FP32 remains bounded by fixed `rtol=1e-3`, `atol=5e-5`, and cosine floor
  `0.999`; no bit-equality promise is introduced across backends. Deterministic
  mask planning is a separate seed/structure contract. Ordinary production
  training has no Phase 8A promise of bit identity across arbitrary processes,
  hosts, Torch versions, or backend kernels without equivalent configuration.
- Decision: CUDA regressions invoke fresh subprocesses for the isolated
  five-policy node, the targeted two-file suite, complete `tests/ssl` followed
  by the targeted suite, and reversed module order. A sentinel prevents the
  full-suite child from recursively launching itself. CUDA absence remains an
  honest skip and never counts as hardware evidence.
- Consequences: No masking, model, objective, graph/canonical, dataset/cache,
  ontology, checkpoint, or successful serialized hardware-report schema
  changes. `Phase8ACudaAmpHardwareEvidence` remains `1.2.0`; the new runtime
  and diagnostic contracts are evidence-only `1.0.0`. The next exact head
  requires new Required runs plus host-local CPU replay, isolated/targeted/
  full/post-full/reverse-order CUDA pytest evidence and standalone RTX 3090
  acceptance with exact counts before any successful final evidence comment.

## 2026-08-14 — ADR-062: Phase 8B.1 adds exact-identity independently weighted hierarchy recovery objectives

- Status: Accepted for implementation in the Phase 8B.1 draft PR. This ADR
  authorizes bounded mechanics only; it does not authorize merge, Phase 8B.2,
  production/full-corpus SSL training, Phase 9, PLL, preference critic, or
  quality scoring.
- Context: Phase 7A supplies unchanged note reconstruction and bar/song latent
  integration objectives. Phase 8A supplies deterministic leakage-closed
  hierarchy masks, exact prepared bindings, and a shared deterministic CUDA
  evidence runtime, but intentionally adds no objective. Phase 8B.1 needs
  independently ablatable recovery signals at onset, beat, bar, and track
  levels without converting descendants into duplicate latent targets or
  allowing target/theory sidecars into SSL inputs.
- Decision: Define a seven-entry registry containing the three existing
  Phase 7A families and four new families: `onset_latent`, `beat_latent`,
  `hierarchy_bar_latent`, and `track_latent`. Keep the new hierarchy-bar name
  separate from `phase7a_bar_latent`; the former uses contextual coarse bars,
  while the latter retains the accepted fused-bar integration semantics.
- Decision: Use retained fused contextual/local rows for onset and beat, and
  contextual coarse rows for hierarchy bar and track. Each new family owns a
  separate `LatentProjectorPredictor`. Its masked online rows predict a
  detached, no-EMA full-view target from the shared encoder under the existing
  full-view evaluation behavior and cosine recovery formula.
- Decision: Align exclusively through exact Phase 8A plan identities and
  attested node pointers. Onset and beat policies select their units once;
  contiguous spans select bars once; track/bar spans select the track once and
  their bars once. Canonically sort and deduplicate `(sample, local, global)`
  rows and reject any row outside its sample interval. Do not use temporal
  snapping, nearest neighbours, theory labels, targets, provenance, dataset/
  split identity, derived topology, missing-store creation, dense membership,
  or cross-sample alignment.
- Decision: Record numerator, eligible denominator, mean, availability,
  unavailable reason, configured weight, active state, and zero norms for each
  family. Denominator zero means unavailable with no fabricated mean. Weight
  zero bypasses the new head and gradient path. Aggregate the fixed sum of
  weight times each available mean without normalization by active weights or
  redistribution around unavailable families. A bounded multi-policy step
  uses its fixed scheduled-pass divisor, not an availability-dependent one.
- Decision: `phase7a_control` constructs the exact old
  `MaskedGraphSSLModel`; it has no Phase 8B heads, state keys, or output
  wrapper. Other modes use the additive multilevel model and one immutable
  objective config. Hydra exposes control, four single-family modes, and equal
  weight. The default Phase 7A root keeps Phase 8B inactive.
- Decision: Continue using the existing failure-atomic SSL checkpoint
  container. Additive model metadata binds the complete objective registry,
  fixed weights, fingerprints, target mode, aggregation, and head count. An
  explicit transfer validates every old key/shape/dtype before mutation,
  loads all old components, preserves and enumerates every separately
  initialized `phase8b_latent_heads.*` tensor, and reports the source SHA-256.
  Incompatible new objective fingerprints reject before live state changes.
- Decision: All new objective, eligibility, prepared-objective, loss,
  model/output, metric, checkpoint-transfer, and bounded-comparison contracts
  begin at `1.0.0`. Existing Phase 7A/8A, graph, canonical, data/cache/target,
  and checkpoint-container versions do not change.
- Decision: The bounded comparison uses the accepted Phase 8A synthetic
  fixture, fixed train/held-out membership, seed, initialization discipline,
  five-policy schedule, optimizer, and step count. It reports initial/final
  train and held-out family losses, denominators, O(D) anti-collapse
  diagnostics, and gradient coverage for Phase 7A control, Phase 8A hierarchy
  masks with old objectives, four single-family variants, and equal weight.
  Metrics retain detached CPU scalar/O(D) state and no prediction/CUDA tensors.
- Consequences: Phase 8B.1 can prove exact routing, ablation, gradients,
  checkpoint compatibility, and bounded overfit mechanics. It cannot prove
  representation quality, musical quality, likelihood, downstream gains, or
  scientific model preference. Those claims require separately authorized
  Phase 8B.2/scaled evaluation after the raw-compatible corpus boundary.

## 2026-08-14 — ADR-063: Explicit Phase 8B configs control the official SSL engine fail-closed

- Status: Accepted as integration remediation in draft PR #18. The PR remains
  draft and unmerged; this decision does not authorize Phase 8B.2, Phase 9,
  PDMX/full-corpus training, PLL, preference critic, or quality scoring.
- Context: ADR-062 introduced the registry, Hydra objective group, builder,
  multilevel forward, checkpoint transfer, and standalone bounded comparison.
  The official `music_critic.ssl.run`/`ssl.engine` implementation nevertheless
  still called `build_ssl_model()`, prepared independent Phase 7A masks, and
  invoked the Phase 7A forward unconditionally. An explicit Phase 8B Hydra
  config could therefore be present but ignored. The bounded runner was not a
  supported production training path.
- Decision: Preserve the literal old engine branch whenever
  `phase8b_objective` is absent/null. Elide the new inactive masking key from
  its plain config so accepted Phase 7A resolved artifacts, model/state, loss,
  checkpoint payload, and report fields remain bit-exact.
- Decision: An explicit objective must have a separately materialized and
  fingerprinted `phase8b_masking` config. Onset, beat, bar, and track schedule
  only onset descendants, beat descendants, contiguous bars, and track/bar,
  respectively. Equal weight schedules those four in canonical order. The
  Phase 7A control schedules independent note masking. The explicit mask-only
  control pairs Phase 7A objectives with the four hierarchy policies. Reject
  every incompatible mode/policy pair; never substitute a policy.
- Decision: Build explicit runs only with
  `build_phase8b_model_from_config()`. New objective modes require the exact
  additive model, a prepared hierarchy binding, prepared objective binding,
  and `forward_multilevel`. Mask-only requires the literal old model and
  `forward_hierarchy`; explicit Phase 7A control uses the old public forward.
  Validate the pair and model/forward contract before the first optimizer
  step, with no silent fallback.
- Decision: Use the official path for one-batch and multi-epoch training,
  fixed epoch-zero validation, best/last checkpoints, ordered journal, and
  exact epoch-boundary resume. Aggregate family numerators and eligible
  denominators with `Phase8BObjectiveAccumulator`; keep zero denominators
  unavailable and never renormalize another family.
- Decision: Bind the registry, complete objective config and active weights,
  masking config, Phase 8A mixture, execution mode, and concrete model class
  in resolved config, run manifest, report, and checkpoint metadata. Any
  objective mode/weight, policy/span, model, or active-weight change rejects
  before checkpoint application. Old checkpoints are not implicit Phase 8B
  resume inputs; the explicit transfer API starts a new run.
- Decision: Reports separately count optimizer steps, forwards, scheduled
  policy passes, objective evaluations, eligible entities, retained tensors,
  and primary/collateral masked entities. Single/control variants use one
  scheduled forward per batch while equal/mask-only use four. Neither the
  official bounded runs nor the earlier standalone comparison are described
  as compute matched or as effectiveness/model-selection evidence; Phase 8B.2
  owns that scientific decision.
- Consequences: Real CLI subprocesses now prove routing for null Phase 7A,
  each single family, equal weight, and mask-only, plus weight binding,
  fail-before-output incompatibility, one-batch optimization, checkpoint
  bindings, and exact two-epoch stop/resume. Optional onset/equal CUDA+AMP
  smoke uses the same official engine and may skip honestly without hardware.

## 2026-08-14 — ADR-064: Phase 8B.1 aggregates scheduled views per family before weighting

- Status: Accepted as cross-policy aggregation remediation in draft PR #18.
  The PR remains draft and unmerged; this decision does not authorize Phase
  8B.2, production/full-corpus training, PDMX, PLL, critic, or quality scoring.
- Context: ADR-062 specified a fixed family-weighted sum and ADR-063 connected
  explicit Phase 8B configs to the official engine. The engine nevertheless
  optimized the average of complete per-policy totals. A family present in
  two views therefore received its configured weight twice before the
  policy-count divisor, while reporting independently accumulated that
  family's numerator and denominator. Hierarchy-bar appears in both
  contiguous-bar and track/bar views, so optimizer and report semantics
  diverged. ADR-062's scheduled-pass-divisor sentence is superseded by this
  decision.
- Decision: For each CPU batch, run every scheduled policy view, then compute
  `N_f=sum_v N_(f,v)`, `D_f=sum_v D_(f,v)`, and `mean_f=N_f/D_f` for each
  active family with `D_f>0`. Optimize and report
  `sum_f weight_f*mean_f`, applying each available family's configured weight
  exactly once. Never divide by policy count, active-weight sum, or available-
  family count.
- Decision: Preserve each prediction made in a distinct view as one
  observation in that family's numerator and denominator even when the raw
  entity identity repeats. Do not deduplicate across views. Denominator zero
  remains unavailable with `numerator=None` and `mean=None`; do not fabricate
  zero or rescale another family.
- Decision: Use the same family-global formula for differentiable batch
  training totals, stage/epoch aggregates, validation, best-checkpoint
  selection, and resume journals. Reports expose family numerators,
  denominators, means, view-pass counts, one-or-zero family-weight application
  counts, optimizer/reported totals, and consistency evidence.
- Decision: Pack the available family numerators and optimizer total into at
  most one metrics D2H transfer per CPU batch. Reports retain no graph,
  prediction, or CUDA tensor. Separately count CPU batches, optimizer steps,
  forwards, scheduled policy passes, family-view passes, and eligible
  prediction rows. Four-view modes remain more compute and are not described
  as compute matched.
- Decision: Move the registry/config/family-loss/objective/model/output/metric,
  official engine/masking/report/manifest, checkpoint-binding/transfer-report,
  and bounded-comparison contracts that bind this semantic boundary to
  `1.1.0`. Bind the exact aggregation string and revised fingerprints in
  resolved config, manifests, model metadata, reports, and checkpoints. Reject
  Phase 8B remediation checkpoints created under the old rule. Leave the null
  Phase 7A route and its checkpoint container unchanged.
- Consequences: The independent equal-weight oracle is
  `bar=(6+15)/(3+5)=2.625` and total `6.875`; the superseded pass average is
  `2.3125`. Tests cover bar weight once, eligibility mutation isolation,
  policy-order invariance, unavailable-family non-rescaling, single-policy and
  Phase 7 controls, mask-only old-family aggregation, validation/best/resume,
  and optional CUDA AMP transfer/finite-gradient evidence. All prior
  783,207-byte bounded artifacts and fingerprints are invalid evidence and
  were replaced by fresh byte-identical family-global artifacts for the exact
  remediation tree.

## 2026-08-14 — ADR-065: Phase 8B.1 AMP training requires real-update evidence

- Status: Accepted as CUDA/AMP zero-update remediation in draft PR #18. The
  PR remains draft and unmerged. This does not authorize Phase 8B.2,
  production/full-corpus training, PDMX, PLL, critic, or quality scoring.
- Context: Independent RTX 3090 acceptance on exact head
  `b41dd410e757db1f595880074c106c67327fb13e` showed successful CUDA execution
  and two nominal optimizer steps but identical initial/final onset and
  equal-weight losses, with zero non-zero-gradient counts for every online
  encoder component. The Phase 8A mask-only control trained normally on the
  same machine. The old CUDA smoke asserted only device/AMP/scaler routing,
  and the engine counted a `scaler.step()` attempt as an optimizer step even
  when public GradScaler overflow handling skipped the update.
- Context: A deterministic CPU autocast-FP16 oracle reproduced the numerical
  mechanism. The broad encoder autocast also covered each new Phase 8B
  projector/predictor, including LayerNorm, and scale `65536` produced
  non-finite head/encoder gradients. Equal-weight execution additionally had
  no explicit online/full-view dtype boundary. Dynamic scaling could back off,
  but two bounded steps were consumed by skipped updates while accounting
  claimed success.
- Decision: Keep only new `phase8b_latent_heads.*` projection, GELU,
  LayerNorm, cosine normalization, and reduction in FP32 inside an explicit
  disabled-autocast island. Preserve the differentiable cast back to the
  online encoder and stop-gradient only the full-view target. Do not change
  Phase 7A heads or the null-config route. Start the public Phase 8B GradScaler
  at `16384`; retain normal public growth/backoff behavior.
- Decision: Count `optimizer_step_attempt_count`,
  `optimizer_step_applied_count`, and `optimizer_step_skipped_count`
  separately. Infer an AMP skip only from the public `get_scale()` decrease
  across `step()`/`update()`; retain `optimizer_step_count` as an alias for
  applied steps. Record the first applied step's packed finite/non-zero
  gradient counts and exact changed-parameter/element counts for the online
  encoder, each Phase 8B head, and Phase 7A control paths. Record optimizer
  membership and public scaler transitions without private PyTorch APIs.
- Decision: An official bounded run fails closed unless at least one step is
  applied, active Phase 8B heads and the online encoder have finite non-zero
  gradients and exact parameter changes, inactive Phase 8B heads have neither,
  the initial/final input fixture matches, model state changes, final loss is
  finite, and bounded loss decreases. Phase 8A mask-only additionally requires
  finite non-zero updates in its encoder/fusion/hierarchy/transformer,
  decoder, and old projector paths.
- Decision: Add an independent CUDA runner that executes FP32 and AMP on the
  same seed, fixture, policies, and initialization for onset, beat, bar,
  track, equal, and mask-only. Require exact identities/denominators/family
  structure and initial/final loss parity within documented `rtol=0.02,
  atol=0.02`; do not require FP32/FP16 bit identity. Save complete subprocess
  logs, official reports, counters, gradients, parameter updates, losses,
  scaler evidence, CUDA peaks, environment, and an archive.
- Decision: Advance affected Phase 8B objective/model/engine/report/checkpoint
  and bounded-comparison contracts to `1.2.0`; latent prediction advances to
  `1.1.0`; new optimizer and CUDA-training-acceptance evidence begin at
  `1.0.0`. Masking remains `1.1.0`; eligibility/prepared binding/batch
  aggregate remain `1.0.0`; graph/canonical/ontology and Phase 7 contracts do
  not change. Bind the FP32 AMP compute rule, scaler initial scale, and
  optimizer-evidence contract into Phase 8B checkpoints so head
  `b41dd410...` artifacts reject fail-closed.
- Consequences: The RTX artifact from `b41dd410...` is invalid as successful
  training evidence. It remains valid negative evidence proving CUDA routing
  and exposing the zero-update defect. Local CPU/FP16 and contract tests can
  prove the code path and failure semantics, but successful CUDA remediation
  remains unclaimed until the independent RTX 3090 runner passes on the exact
  final head.

## 2026-08-15 — ADR-066: Phase 8B.2A separates compute-matched comparison, validation selection, and held-out test access

- Status: Accepted for implementation on the Phase 8B.2A draft branch. This
  authorizes reproducible comparison mechanics and bounded CPU acceptance,
  not full-corpus/PDMX training or a scientific superiority claim.
- Context: Phase 8B.1 proved routing, aggregation, gradients, checkpointing,
  and resume, but its natural schedules use one policy view for controls and
  single-level objectives and four for mask-only/equal-weight. Those losses
  and bounded decreases are not comparable representation-quality evidence.
- Decision: Version the complete binding as
  `Phase8B2ComparisonProtocol@1.0.0`. Treat `natural_schedule` as a secondary
  compute-unmatched diagnostic and `encoder_forward_matched` as primary. The
  primary branch fixes raw sample exposures, applied/skipped optimizer-update
  budget, and encoder forwards. Repeat a single policy through independently
  seeded views of the same raw batch; preserve the family-global loss without
  hidden normalization.
- Decision: Count actual encoder invocations, including detached full-view
  passes. The default matched budget is 12 calls per logical update: controls
  use six views at two calls each; latent objectives use four views at three
  calls each. A nominal equal view count is not compute matching.
- Decision: Pair exact encoder initialization, SSL/downstream sample schedules,
  and memberships. Derive independent named model/data/mask/downstream/
  bootstrap seed domains so launch permutation has no effect. Fingerprint
  actual initial and transferred encoder states and reject incompatible resume
  or aggregation.
- Decision: Compare `frozen_probe` and `full_finetune` against
  `supervised_scratch` using the current hierarchical supervised architecture,
  fresh heads/optimizer, and the same downstream budget. Frozen exports are
  excluded from the optimizer and remain bit-exact; full fine-tuning loads
  only representation state failure-atomically. SSL decoders/heads/optimizer
  never transfer.
- Decision: Keep current fully supervised model-ready, source-native heads
  only. Candidate-first validation and train priors remain authoritative.
  Select by declared HookTheory/POP909-CL macro endpoints, mean dataset rank,
  then lower NLL, lower compute, and lexical variant ID. Diagnostics do not
  select.
- Decision: The independent statistical unit is a piece. Use deterministic
  paired piece bootstrap, per-seed mean/median, between-seed SD, and deltas
  against scratch and Phase 7A; emit unavailable reasons rather than row-level
  uncertainty. Bounded fixtures do not produce significance claims.
- Decision: Test remains locked until a matching validation-selection artifact
  chooses exactly one checkpoint per scope, explicit acknowledgement is given,
  a new output and pre-inference test membership are bound, and an experiment
  identity is consumed once. Unauthorized test artifacts invalidate an
  aggregate.
- Decision: Reuse official SSL/training/evaluation/checkpoint engines. Bind the
  optional repeated-view schedule into Phase 8B checkpoints and transfer into
  supervised checkpoints. Create immutable `1.0.0` comparison artifacts and
  reject dirty production worktrees, stale/incomplete/duplicate/mixed bundles.
- Consequences: Phase 8B.2A can establish fair mechanics and a production-ready
  protocol without running long training. Phase 10 may supply PDMX raw caches
  to the same binding. PDMX, Dilemmadata, PLL, preference/quality scoring,
  curriculum masking, EMA, theory-as-input, schema/ontology changes, and any
  effectiveness claim remain outside this decision.

## 2026-08-15 — ADR-067: Phase 8B.2A completion requires an attested, resumable official-engine DAG

- Status: Accepted on the existing Phase 8B.2A draft branch. This refines
  ADR-066 after commit `7365286` proved to contain control-plane primitives but
  no executable end-to-end experiment.
- Decision: Advance affected Phase 8B.2A orchestration, protocol, schedule,
  selection, statistics, artifact, seed, transfer, and test-lock contracts to
  `1.1.0`; advance evaluation output to `1.3.0` for piece sufficient
  statistics. Unrelated graph, canonical, ontology, encoding, adapter, and
  corpus contracts do not change.
- Decision: `music_critic.experiments.phase8b2.run` owns `plan`, `run`,
  `resume`, `aggregate`, and `select`. `run` executes official SSL, training,
  and candidate-first evaluation modules as list-argv Python subprocesses. A
  cell is reusable only after its complete manifest, artifact SHA-256s, and
  protocol/runtime binding validate. All writes use staging and atomic
  publication; stale/incomplete cells require operator inspection and are not
  overwritten.
- Decision: Resolve schedules from the official target-free sampler before
  training and record dataset ID, piece ID, sample position, logical update,
  and batch position. Derive index/cache/split/train/validation/test identities
  from official metadata, treating configured fingerprints only as assertions.
  Every observed SSL/downstream sequence must match the attested schedule
  bit-exactly. All variants are preflighted before long training.
- Decision: Execute a real logical-update budget with multiple batches per
  epoch, interval validation, and exact final validation. Attempted, applied,
  skipped, raw-sample, policy-view, and instrumented encoder-call counts are
  evidence. An unavailable objective or AMP overflow invalidates a scientific
  cell; no replacement sample silently changes its schedule.
- Decision: Fixed validation membership is independent of downstream training
  order and must match the comparison protocol, training checkpoint/report,
  and standalone evaluation. Per-piece evaluation persists mergeable CPU-only
  counts. Corpus endpoints are recomputed after every independent-piece
  bootstrap draw; exact AP remains descriptive because score-row retention is
  outside this contract.
- Decision: Downstream identity is `(seed, variant_id, transfer_mode)`.
  Selection aggregates complete paired seeds for each
  `(variant_id, transfer_mode)` before mean-dataset-rank, NLL, compute, and
  lexical-configuration ranking. Test authorization is limited to the complete
  selected seed-checkpoint manifest, one single-use experiment per seed.
- Consequences: The bounded CLI executes 8 SSL cells, 8 encoder exports, 18
  downstream cells, and 18 validation cells across two seeds, plus preflight,
  aggregation, selection, and final reporting. Those fixtures prove mechanics,
  resume, and binding—not scientific superiority. Production training is not
  part of remediation, and full-scale PDMX evidence remains owned by Phase 10.

## 2026-08-15 — ADR-068: Phase 8B.2A data attestation uses a source-neutral semantic projection

- Status: Accepted as blocking pre-merge remediation in existing draft PR #19.
- Context: The `1.1.0` production planner correctly resolved sample slots from
  metadata without loading target/canonical payloads, but represented its data
  evidence with null fingerprints/membership and a placeholder composition.
  SSL/downstream publication compared those values with complete official-
  engine dictionaries; SSL therefore failed on every real `index_paths` run,
  and downstream could raise `KeyError` for `train_dataset_counts`. The bounded
  fixture materialized an engine runtime while planning and masked the defect.
- Decision: Advance the affected comparison protocol, artifact, plan, schedule,
  data-attestation, actual-schedule, matrix-runner, cell-manifest and test-lock
  contracts to `1.2.0`. Introduce
  `Phase8B2DataSemanticProjection@1.0.0`. Unchanged compute, selection,
  statistics, diagnostics, transfer, seed, evaluation, graph, model, canonical,
  corpus, encoding and ontology contracts retain their versions.
- Decision: Metadata planning and official SSL/downstream evidence use the same
  projection function and shape: ordered dataset/index and cache identities,
  split fingerprint, normalized train dataset counts/size, fixed-validation
  membership/counts and mixture weights. Production values must match the
  protocol binding. Exact schedule slots continue to come only from indices,
  split metadata and the official deterministic sampler; target and canonical
  payload reads are forbidden for schedule resolution.
- Decision: Runtime binding separately verifies the semantic projection, exact
  observed sample-schedule fingerprint and optimizer/update/encoder-forward
  accounting. Every mismatch raises a stable `Phase8B2ContractError`; no
  incomplete or stale cell is atomically published.
- Decision: Replace the ambiguous `test_accessed=false` claim. Planning may
  resolve test membership metadata for a future single-use lock and records
  `test_membership_metadata_resolved=true`. Until unlock it must also record
  `test_inference_performed=false`, `test_targets_accessed=false`, and
  `test_metrics_accessed=false`. Serialize only the test membership
  fingerprint, counts and split binding, never the complete test piece list.
- Consequences: A synthetic on-disk HookTheory/POP909-CL mini-DAG now exercises
  the real production-format path through SSL, export, frozen/full/scratch and
  validation evaluation. The unchanged bounded 52-cell DAG remains mechanics
  evidence only. No Phase 9, production/PDMX training, held-out inference, or
  scientific-effectiveness claim is authorized.

## 2026-08-15 — ADR-069: CUDA runtime evidence uses a logical integer device index

- Status: Accepted as blocking CUDA pre-merge remediation in existing draft
  PR #19. The PR remains draft and unmerged; successful hardware remediation
  requires an independent exact-head RTX 3090 rerun.
- Context: A real-corpus RTX 3090 smoke on
  `91c2d0c536cbe35fe40d83e0ad09a4c5200a3d97` resolved `cuda:0` correctly and
  read both production caches, but the first Phase 8B.2A preflight failed
  before model forward. `phase8b_engine._prepare` passed the concrete
  `torch.device("cuda:0")` to `torch.cuda.reset_peak_memory_stats`; the
  installed CUDA/PyTorch runtime rejected that argument as `Invalid device
  argument`. The same type mismatch existed in Phase 7A SSL, supervised
  training, Phase 8A hardware acceptance, and Phase 8B.2 environment evidence.
- Decision: Keep concrete `torch.device` objects for tensor/module placement.
  Add `CudaRuntimeDeviceIndex@1.0.0`, which calls the canonical resolver and
  returns only a validated logical integer CUDA index. It preserves explicit
  `cuda:0`/`cuda:1`, resolves abstract CUDA through the current logical device,
  honors `CUDA_VISIBLE_DEVICES` through PyTorch's visible count, rejects CPU
  with `runtime.device.cuda_operation_requires_cuda`, and preserves stable
  unavailable/out-of-range categories. CUDA discovery probe failures are also
  structured by runtime-device resolution `1.0.2`.
- Decision: Every runtime-device call to CUDA reset, allocated/reserved peaks,
  synchronization, device name, and device properties receives that explicit
  integer. Fixed device-zero acceptance calls remain explicit integers. Do not
  disable VRAM evidence, catch the failure as unavailable evidence, fall back
  to CPU, remove reset, or rely on an implicit current device.
- Decision: Advance only affected evidence/execution contracts: SSL training
  report `1.2.3`; Phase 8B engine and report `1.2.1`; Phase 8A CUDA AMP hardware
  evidence `1.2.1`; Phase 8B.2 artifact evidence `1.2.1`; and runtime-device
  resolution `1.0.2`. Device transfer, comparison protocol, graph, canonical,
  ontology, model, objective, schedule, data, checkpoint, and scientific
  contracts retain their versions and semantics.
- Consequences: CPU/mock regressions enforce the integer boundary for all three
  official training paths and all affected evidence APIs. Optional real-CUDA
  tests execute reset, allocation, peak/name evidence and invalid-index
  rejection. The official production-format CUDA mini-DAG executes one
  `phase7a_control` seed, one SSL update, frozen/full/scratch training, and
  three validation evaluations with 8/8 scientific cells, 8/8 runtime
  bindings, 3/3 checkpoint-to-evaluation bindings, nonzero VRAM, and no test
  inference/target/metric access. CPU results cannot establish that the RTX
  blocker is fixed.

## 2026-08-15 — ADR-070: The independent RTX gate is a bounded, artifact-preserving operator workflow

- Status: Accepted as final command/evidence remediation for draft PR #19.
  Independent exact-final RTX 3090 success remains pending, so the PR stays
  draft and unmerged.
- Context: The published preflight required an entirely empty porcelain
  status even though the operator checkout deliberately preserves untracked
  Phase 8A/8B evidence. The published one-seed `production_pilot` also
  violated its three-seed minimum. In addition, Phase 8B.2 has a structured
  `device` field rather than a `device=cuda` Hydra group.
- Decision: Define the short hardware run as a production-format real-corpus
  bounded smoke. It uses `comparison=bounded_acceptance`, one
  `phase7a_control` variant, seed 17, one SSL and downstream update, explicit
  `device.name=cuda:0`, and FP16 AMP. It is mechanics/hardware evidence only;
  never label it a production pilot or scientific comparison. Keep
  `production_pilot` at a minimum of three seeds. Place `_self_` before the
  comparison default so registered Hydra presets retain their declared names,
  seed sets, and budgets.
- Decision: Reject tracked unstaged and staged changes independently with
  `git diff --quiet` and `git diff --cached --quiet`. Allow and print untracked
  files, list preserved output roots diagnostically, and never delete, move,
  clean, reset, or reuse an evidence/output root. Fetch the named branch,
  require its head to equal the operator-supplied exact SHA, and detach it.
  Scope `set -euo pipefail` to the gate script's subshell.
- Decision: A prior plan may supply only two index paths, two cache roots, and
  the global split path. Re-resolve and validate the bounded plan/config at the
  exact head. Verify the complete final bundle, CUDA logical index/positive
  peaks, 1/1 update accounting, absence of CPU fallback and test access, and
  both corpus IDs in train/validation evidence. Archive logs, resolved
  configs, attestations, reports, and payload hashes; exclude caches,
  checkpoints, and corpus payloads and write an archive checksum sidecar.
- Consequences: A failed command returns nonzero while leaving tmux/SSH and all
  artifacts intact. CPU suites can validate the workflow and verifier but do
  not establish successful CUDA remediation. Phase 9, test inference,
  production-scale comparison, and scientific-effectiveness claims remain
  unauthorized.

## 2026-08-15 — ADR-071: The RTX bounded smoke fixes validation at 128 pieces

- Status: Accepted as the final boundedness correction for draft PR #19;
  independent exact-head RTX 3090 success remains pending.
- Context: The official one-seed runner bounded optimizer updates but omitted
  `comparison.validation_samples`. Its zero default selects the complete
  validation split, allowing each of the three downstream evaluation cells to
  traverse an arbitrarily large validation corpus.
- Decision: Set `comparison.validation_samples=128` explicitly and record 128
  in the invocation artifact. Treat the selected validation membership as a
  cross-stage runtime binding: it must contain exactly 128 identities and both
  HookTheory and POP909-CL, while one fingerprint must agree across the plan,
  projected schedules, SSL/downstream training reports, evaluation metrics,
  and checkpoint evidence. Downstream validation epoch size and evaluation
  maximum samples must each resolve to 128. Dataset-name agreement without
  exact count and fingerprint agreement is not acceptance evidence.
- Decision: Preserve `bounded_acceptance`, one variant, one seed, one SSL
  update, one downstream update, locked test access, `cuda:0`, FP16 AMP,
  artifact-preserving preflight, fresh output roots, and the payload-excluding
  evidence archive. Do not change `production_pilot` or its three-seed minimum.
- Consequences: The command is a fixed-cost production-format real-corpus
  hardware smoke, not a production pilot or scientific comparison. A missing,
  zero, off-by-one, or fingerprint-divergent validation binding fails closed.
  CPU regression results cannot establish successful RTX 3090 execution.

## 2026-08-15 — ADR-072: Indexed CUDA memory statistics require an initialized scoped device lifecycle

- Status: Accepted as blocking CUDA lifecycle remediation in existing draft
  PR #19. Independent success remains pending at the new exact head.
- Context: The first preflight worker of the independent bounded smoke failed
  again at head `aa5fe538d45499f84cbf5ee8de99f7514ff111ce`, before model
  forward. Replacing `torch.device("cuda:0")` with integer zero was
  insufficient: `reset_peak_memory_stats(0)` still raised
  `RuntimeError: Invalid device argument` in the fresh process.
- Hardware probe: On an NVIDIA GeForce RTX 3090 with one visible device,
  PyTorch `2.13.0+cu130`, CUDA runtime 13.0, and
  `initialized_before=false` in every fresh process: bare implicit reset
  passed and initialized CUDA with peaks 0/0; explicit integer zero and
  `torch.device("cuda:0")` both failed; `set_device(0)` then indexed reset,
  `torch.cuda.init()` then indexed reset, and scoped device context plus init
  then indexed reset all passed with peaks 0/0. A dummy allocation also
  passed but left `peak_reserved=2097152` bytes and is rejected as contaminating
  measurement evidence.
- Decision: Add `CudaMemoryStatisticsLifecycle@1.0.0`. Resolve the concrete
  CUDA device with `CudaRuntimeDeviceIndex@1.0.0`, enter
  `torch.cuda.device(index)`, call the idempotent public `torch.cuda.init()`,
  and only then call `reset_peak_memory_stats(index)`. The indexed reset stays
  explicit; the scoped context restores the previous current device. Do not
  use implicit reset, permanent `set_device`, dummy allocation, CPU fallback,
  skipped reset, or fabricated peaks.
- Decision: Distinguish
  `runtime.cuda_memory_statistics.initialization_failed` from
  `runtime.cuda_memory_statistics.reset_failed`. Evidence binds lifecycle
  contract version, logical index, `initialized_before`, and
  `initialized_after`. Route Phase 7A SSL, Phase 8B SSL, supervised training,
  Phase 8A/8B CUDA acceptance, and Phase 8B.2 workers through the one helper;
  source audit forbids every other direct reset call.
- Decision: Advance only affected CUDA evidence/execution contracts: SSL
  training report to `1.2.4`; Phase 8B engine/report to `1.2.2`; Phase 8A CUDA
  AMP hardware evidence to `1.2.2`; and Phase 8B.2 artifact evidence to
  `1.2.2`. Phase 8B.2 preflight worker, matrix runner, and cell manifest
  advance to `1.2.1`. Runtime-device resolution, logical index, transfer, comparison,
  graph, model, objective, data, schedule, ontology, checkpoint, and
  evaluation contracts retain their versions and semantics.
- Consequences: The earlier device-object remediation and the later
  integer-only remediation are both insufficient hardware evidence. CPU/mock
  and optional fresh-process CUDA regressions validate lifecycle ordering,
  isolation, restoration, idempotence, structured failures, and non-mutation,
  but only a new independent exact-head RTX 3090 run can close the gate.

## 2026-08-16 — ADR-073: Dilemmadata enters V2 through an evidence-first raw/target boundary

- Status: Accepted for Phase 9A only. This decision authorizes audit tooling,
  the v1.0 evidence manifest, synthetic fixtures, and the Phase 9B adapter
  contract. It does not implement or authorize a production adapter, target
  encodings, theory heads, losses, supervised training/evaluation, CUDA
  lifecycle changes, or Phase 8 SSL changes.
- Integration note: Phase 9A was rebased onto `origin/main` merge
  `1e31e2e1c71b4c1d4b93a9b2a61af53dd81c02f7` after PR #19 merged. The
  provisional number is finalized as `ADR-073`, following Phase 8B.2A
  `ADR-072`; shared-document resolution preserves both accepted histories.
- Context: The official Dilemmadata v1.0 release at commit
  `d60ee75b4a9495e932a4a7be39381578be17e222` is a processed score-derived
  symbolic dataset, not raw MIDI and not one TSV dialect. The audited snapshot
  contains 353 AN joint records and 1,280 DLC records with 2,880,723 note rows
  and 14 primary header shapes. All 2,743 regular files match a clean checkout.
- Context: Every primary record provides exact rational note onset/duration,
  MIDI-compatible pitch, and one corroborating integer resolution without
  theory fields. The release does not provide tempo, velocity, channel,
  program, or an explicit percussion field in primary arrays. Per-note
  meter/measure evidence, 74,773 tied-continuation rows, and 23,314
  zero-duration grace candidates require production adapter validation.
- Context: A bounded-memory target-independent MIDI-compatible note-event
  multiset grouping fingerprint and score/metadata overlap
  reveal 1,507 transitive source components. Five components conflict with
  release split hints. Thirty narrow-multiset groups contain different target
  fingerprints, so conservative record-level splitting would leak candidate
  multiple analyses.
- Decision: Treat score-derived note/timing/pitch and optional
  spelling/part/staff/voice/tie/meter fields as raw observations only when the
  release processing boundary proves they precede annotation merge. Treat all
  key, harmony, cadence, phrase, section, note-degree, validation-gate,
  alternative-analysis, analyst, confidence, and label-provenance fields as
  target sidecars or diagnostics. Source voice identity is not semantic voice
  role.
- Decision: Target-independent exact rational onset/duration and MIDI pitch are
  accepted as Phase 9A evidence; a production-complete `CanonicalPiece` is not
  yet accepted. The bounded-memory multiset fingerprint over those three fields
  is conservative split evidence, not raw canonical, complete-input, or
  model-input identity. Phase
  9B must implement exact tie/grace/meter/bar and required-default policies
  with provenance or structured quarantine. It must not use theory to create
  notes, topology, features, IDs, timing, or fingerprints.
- Decision: Preserve source-native target values and make `available`, `masked`,
  `missing`, and `unsupported` mutually exclusive and exhaustive. Missing or
  false gates are masked; malformed non-empty gates are unsupported. Ambiguity
  is a separate family-local diagnostic that overlaps only available. Row-level
  `alt_label` remains a provenance/diagnostic sidecar and does not make every
  family ambiguous. Preserve producer/version/lineage and unknown
  confidence. Do not infer a universal root/quality/inversion/applied-harmony
  ontology or unobserved target span end in Phase 9A. Exact alignment never
  uses floats or nearest-neighbour snapping.
- Decision: Assign final splits only after transitive closure over the narrow
  MIDI note-event multiset grouping fingerprint, identical AN score bytes, and
  explicit merged-summary links. Phase 9B must redefine and recheck exact
  alternative-analysis identity using its versioned canonical/model-input
  fingerprint.
  Composer/title similarity alone does not join samples. Conflicting release
  hints are warnings and Phase 9B production blockers; they never split a
  component.
- Decision: Version Phase 9A audit/manifest semantics at `1.1.0`; the grouping
  fingerprint, target-state, and upstream-comparison subcontracts are `1.0.0`.
  The official evidence run compares the installed release with a separate
  clean checkout at exact commit `d60ee75b4a9495e932a4a7be39381578be17e222`
  and records performed/exact-match/commit/matching-file evidence plus stable
  mismatch categories. Reports use
  canonical JSON, corpus-relative paths, bounded vocabularies, separate
  raw/target fingerprints, structured quarantine, and a semantic fingerprint
  free of runtime/platform values. A complete `--limit` run is impossible by
  definition and is marked incomplete evidence.
- Consequences: Phase 9B has an exact bounded scope: two streaming parsers,
  target-blind canonical conversion, group-safe split identity, source-native
  target sidecars, exact alignment, quarantine, and leakage mutation tests.
  Theory heads and training remain a later separately authorized increment.
  The Phase 9A corpus scan has zero structural record quarantines but retains
  five split conflicts as explicit production blockers.

## 2026-08-16 — ADR-074: Phase 9B.1 binds Dilemmadata cache identity to the raw projection

- Status: Accepted for the Phase 9B.1 production raw adapter and SSL-ready
  corpus path only.
- Context: A Dilemmadata primary TSV physically co-locates score-derived raw
  observations and theory annotations. Binding canonical provenance or cache
  keys to the complete file SHA-256 would make deletion, replacement, or
  reordering of theory columns change raw artifacts even though conversion is
  forbidden to read those columns. Conversely, the Phase 9A narrow
  onset/duration/pitch multiset intentionally omits tie, meter, spelling, and
  source voice and is insufficient as a full adapter-input identity.
- Decision: Retain full-file SHA-256 only as external physical inventory/index
  evidence. Define `DilemmadataRawProjection@1.0.0` over every normalized raw
  field used by conversion and use it through
  `CorpusCacheInputIdentity@1.0.0` for Dilemmadata cache keys. Preserve the
  existing generic cache/index versions and legacy HookTheory/POP909-CL key
  semantics when the optional projection identity is absent.
- Decision: Convert every accepted record to one source-neutral pitched track
  with empty targets/annotations. Merge ties only across one exact contiguous
  same-pitch/source-voice predecessor; retain source-zero duration as grace;
  derive meter, pickups, incomplete bars, and beats on exact rational measure
  evidence; quarantine contradictions. Insert the existing default tempo and
  schema-required `is_percussion=false` only with explicit provenance and a
  non-claim quality warning. Key-signature mode remains unknown.
- Decision: Preserve separate record, physical, raw-projection, narrow
  grouping, source/lineage, canonical, graph, and model-input identities. Final
  splits use transitive closure over narrow grouping, identical AN score bytes,
  and explicit overlap links; release split hints are diagnostic only.
- Decision: The production gate streams all 1,633 pinned records, emits exactly
  one accepted/quarantined outcome per record, validates cache rebuild and
  group-safe split evidence, and routes exactly two real AN plus two real DLC
  singleton components through one existing official Phase 8B optimizer step.
- Consequences: Theory-column mutation cannot change raw canonical, graph,
  model input, or cache artifact identity, while raw-field mutation must change
  evidence or quarantine. Phase 9B.2 theory sidecars/alignment, target
  encodings, theory heads/losses, supervised evaluation, PDMX/Phase 10, and
  effectiveness claims are not implemented or authorized by this decision.

## 2026-08-17 — ADR-075: Phase 9B.1 fails closed on config, discovery binding, and manifest drift

- Status: Accepted as blocking remediation of the existing Phase 9B.1 draft;
  Phase 9B.2 remains unauthorized.
- Context: The initial adapter serialized versioned policy names without
  rejecting unsupported runtime values. A frozen discovered-record dataclass
  could still be copied with `dataclasses.replace`, allowing record, path,
  dataset, grouping, lineage, or split identity to diverge from discovery. The
  acceptance runner's second build loaded canonical cache artifacts instead of
  proving a second source conversion, and the full-corpus result was not a
  committed machine-checkable contract. Simultaneous key-signature conflict
  also used the meter-conflict category.
- Decision: Patch `DilemmadataAdapter` to `1.0.1`. Every policy field accepts
  only its implemented identifier. Add
  `DilemmadataDiscoveryRecordBinding@1.0.0`, a deterministic seal over corpus,
  record, path, raw/grouping, source/lineage, split, resolution, score,
  physical-source, and discovery-statistic identities. Verify it before any
  canonical construction and reject drift as
  `dilemmadata.record_binding_mismatch`. Rebind only the external physical SHA
  after a target-only byte mutation whose raw projection still matches.
- Decision: Add `dilemmadata.key_signature_conflict`. Keep the raw projection,
  canonical schema, graph/model input, grouping, ontology, targets, and
  HookTheory/POP909-CL contracts unchanged.
- Decision: Advance the acceptance report and production manifest to `1.1.0`.
  The second build must independently repeat pinned discovery, record binding,
  conversion, validation, and cache insertion. Require 0/719 then 719/0
  hit/miss counts, byte-identical indices, identical full quarantine and
  conversion-semantic projections, and unchanged immutable-artifact snapshots.
  Gate `ready=true` on exact equality with the committed compact manifest.
- Decision: The manifest contains contract versions, pinned identity, exact
  outcome/category and accepted graph totals, grouping/cache/split evidence,
  semantic acceptance fingerprint, and SSL composition/mechanics invariants.
  It excludes corpus contents, absolute paths, duration, RSS, caches,
  checkpoints, and the concrete scientific loss value.
- Consequences: The adapter-version patch intentionally invalidates old
  Dilemmadata cache keys because adapter version is already part of generic
  cache identity. Existing artifacts are not rewritten or deleted. Raw
  projection `1.0.0` is unchanged, so this is not a raw-data semantic change.
  Theory sidecars, targets, heads/losses, supervised/scientific training,
  PDMX, and Phase 10 remain out of scope.

## 2026-08-17 — ADR-076: Dilemmadata theory is an external source-native registry extension

- Status: Accepted for Phase 9B.2A target sidecars and exact alignment only.
- Context: Adding Dilemmadata tasks directly to the core HookTheory/POP909-CL
  ontology would change the ontology fingerprint embedded in the already
  accepted raw cache/index identity. AN and DLC also use source-specific label
  grammars, and the raw adapter did not preserve which source rows were merged
  into one canonical tied note.
- Decision: Keep the core ontology and encoding serialization unchanged. Add a
  complete explicit 22-task Dilemmadata registry extension, selected only when
  an external `TargetBundle@1.0.0` is attached. Namespace AN and DLC separately;
  do not create borrowed harmony or semantic voice roles; do not crosswalk AN,
  DLC, HookTheory, or POP909-CL without separate lossless evidence.
- Decision: Freeze only the full-scan AN/DLC quality, inversion, cadence, and
  positive-event vocabularies. Preserve all other source strings on CPU with no
  runtime vocabulary or hash IDs. Missing, masked, unsupported, ambiguous,
  conflict, unaligned, and deferred states never become negative labels.
- Decision: Add target-neutral raw alignment evidence over source-row ordinal,
  exact onset, tie state, and canonical note ID. Raw adapter, projection,
  canonical, cache, graph, and model-input versions do not change. A note target
  is available only if every row merged into it agrees. Point events require an
  exact onset; spans are exact half-open; no tolerance, snapping, or node-type
  priority is permitted.
- Decision: Retain every source record as a separate analysis view in its
  existing split-atomic component. Preserve `alt_label` and analyst/reviewer
  metadata only as target provenance/diagnostics. Attach targets after raw-cache
  loading through the existing alignment/tensorizer/collator path.
- Consequences: Phase 9B.2A can audit and batch deterministic source-native
  supervision without invalidating Phase 9B.1 artifacts. Nine encodings are
  mechanically model-ready and 13 remain open/deferred, but no new head, loss,
  supervised evaluation, training run, or effectiveness claim is authorized.
  Phase 9B.2B requires a separate review after this change merges.

## 2026-08-17 — ADR-077: Dilemmadata target alignment requires independent raw-origin verification

- Status: Accepted as blocking remediation of Phase 9B.2A in draft PR #22.
- Context: `DilemmadataRawTargetAlignmentEvidence@1.0.0` checked its record and
  canonical identities, row structure, and self-fingerprint. A caller could
  nevertheless alter an onset, tie state, canonical note ID, or ordered row
  semantics and recompute a consistent fingerprint, redirecting target
  supervision without proving that the evidence came from the pinned raw
  source.
- Decision: Advance raw alignment evidence and the target adapter to `1.1.0`,
  and the audit report/manifest to `1.1.0`. Keep the self-fingerprint only as a
  corruption check. Before reading theory or analyst/reviewer metadata, rebuild
  the evidence independently by running the same raw parser, exact tie merger,
  and canonical builder from the pinned record. Require byte-exact canonical
  serialization and equality of the complete ordered evidence, including row
  ordinal, physical line, exact `RationalTime`, tie-continuation state, and
  canonical note ID. Reject every difference as
  `dilemmadata.target.alignment_binding_mismatch`.
- Decision: Give the raw oracle a closed dialect-specific value-field contract
  and do not materialize theory, gate, alternative-label, or metadata values in
  the raw parser. Do not use float conversion, tolerance, nearest-note
  matching, snapping, or heuristic renumbering.
- Consequences: A consistently re-fingerprinted forged evidence object fails
  closed, while valid tie-merged evidence and target-only mutation remain
  supported. The target sidecar serialization stays `1.0.0`; its aggregate
  fingerprint changes because adapter provenance advances. Raw adapter
  `1.0.1`, raw projection `1.0.0`, grouping, canonical piece, cache key/artifact,
  split, graph, and model-input contracts and fingerprints remain unchanged.
  Phase 9B.2B remains unauthorized.

## 2026-08-17 — ADR-078: Dilemmadata supervision is cached, candidate-first, and source-entry normalized

- Status: Accepted for Phase 9B.2B implementation and bounded plumbing
  evidence; independent RTX results are pending.
- Context: Nine Dilemmadata encodings are mechanically model-ready, but five
  describe positive events without proven negatives and 13 retain open source
  strings. Exact chord spans can expand one annotation into many candidates,
  so ordinary row-mean CE would overweight long/dense chords. Re-running raw
  conversion and the independent alignment oracle per epoch is also both
  expensive and an unsafe source/runtime coupling.
- Decision: Activate only AN/DLC chord quality and inversion, using four
  distinct heads and source-native class indices. Five PU and 13 open-string
  families have no ordinary CE head/loss. Build `TargetBundle` offline into an
  immutable SHA-addressed JSON cache bound to raw, canonical, source,
  alignment, registry and target identities. Load it lazily after raw cache
  verification. Advance `BatchTarget` to `1.2.0` with exact source-entry
  identity.
- Decision: Produce logits solely from raw hierarchical encoder outputs, join
  targets afterward, and reduce CE as rows per source entry, entries per task,
  then a fixed task-weight sum without hidden active-task renormalization.
  Use the same unit for evaluation and component bootstrap. Priors/optional
  class weights are train-only; model selection is validation-only and test is
  locked.
- Decision: Permit scratch, Phase 7A SSL and Phase 8B multilevel SSL
  initialization through failure-atomic encoder-only transfer. Heads and
  optimizer are always fresh. Fix the RTX 3090 comparison protocol before test
  access at seeds 17/29/43 with equal schedules/budgets and full fine-tuning.
- Consequences: The existing 14 heads/checkpoints remain unchanged and
  transferable through explicit encoder exports. Raw adapter `1.0.1`, raw
  projection/cache identity `1.0.0`, canonical, grouping, split, graph and
  model-input bytes/versions remain unchanged. Bounded overfit is plumbing
  evidence only; long training, Phase 9C, PDMX, critic score, PLL and Phase 10
  remain outside this change.

## 2026-08-17 — ADR-079: Hardware readiness requires sealed executable evidence

- Status: Accepted for Phase 9B.2C implementation; independent RTX 3090
  execution remains pending.
- Context: Phase 9B.2B established fixture and bounded CPU plumbing plus a
  future long-run plan, but did not prove that its four-head supervised path
  executes end to end on the intended production caches and GPU. A useful
  hardware gate must be committed, reproducible at an exact clean Git head,
  independently verifiable, source-free at runtime, and unable to open test.
- Decision: Add `DilemmadataSupervisedSmoke@1.0.0` and sealed bundle `1.0.0`.
  Pin the accepted production raw/target indices, split and model fingerprint;
  require exact RTX 3090 `cuda:0`, float16 AMP with GradScaler, seed-17 scratch
  initialization, fixed four-task weights, AdamW `3e-4`, and 10--20 updates.
  Require train-only target-assisted coverage selection, label-blind
  validation membership, candidate-first/source-entry evidence, finite
  encoder/four-head gradients and updates, failure-atomic checkpoint reload
  parity, official validation, test closure, VRAM evidence and complete CUDA
  cleanup. Guard source conversion/alignment calls fail closed.
- Decision: Publish only a new uniquely named run by atomic rename, containing
  a sealed regular-file directory, deterministic tar and SHA sidecar. A
  separate source-free verifier checks semantic bindings, checkpoint state,
  evaluation, internal hashes, archive safety, exact head and current GPU.
  Keep the PR draft until an independent RTX 3090 run passes this verifier.
- Consequences: This is bounded mechanics evidence only. It changes no
  Phase 9B.2B raw, target-cache, BatchTarget, model, loss, evaluation, split,
  graph or model-input contract/version/fingerprint. No long training,
  scratch-versus-SSL conclusion, Phase 9C, PDMX, or Phase 10 is authorized.

## 2026-08-17 — ADR-080: Target semantics and physical index identity are separate smoke bindings

- Status: Accepted as the minimal Phase 9B.2C RTX unblock in draft PR #24.
- Context: Independent local and RTX production builds contain the same raw
  index, 719 records, metadata index and aggregate `TargetBundle` projection,
  but their self-consistent target-index fingerprints are respectively
  `76feee8d128cc3c5dd1a5b261599df89ef241baa21d82b3c24202a11218beea4`
  and `02fcf7eb03adda2962ade7223924e0fe44483e4900097bd33f50bf93b68d862a`.
  Treating either physical observation as universal blocks execution without
  improving corruption or semantic-mutation detection.
- Decision: Advance smoke/evidence bundle contracts to `1.1.0`. Pin production
  semantics to raw index, metadata index, record count 719, aggregate bundle
  fingerprint and current target adapter/cache/registry contracts. Before
  training, run the existing source-free full-cache checker over the index
  self-fingerprint, coverage and every record, artifact SHA and decoded bundle
  identity/fingerprint. Accept any index that passes those checks and the
  stable semantic projection.
- Decision: Record the exact observed target-index fingerprint in the run
  report, checkpoint data bindings, reload and validation evidence. Require an
  exact match for resume/evaluation; never weaken artifact, record, raw,
  metadata or bundle checks. Keep both observed physical fingerprints in
  provenance and defer the broader cross-host portability root-cause audit.
- Consequences: The existing 719 target artifacts are reusable without rebuild.
  Phase 9B.2B target-cache/adapter/registry versions and raw index/cache,
  grouping, split, graph and model-input semantics remain unchanged. The PR
  remains draft until the bounded RTX smoke passes; Phase 9C remains out of
  scope.

## 2026-08-17 — ADR-081: Leakage invariance and CUDA replay are separate gates

- Status: Accepted as blocking Phase 9B.2C remediation in draft PR #24.
- Context: RTX attempt
  `b7254151ef3b4f11eb55b13d33d02b35d114ee3c` passed semantic target-cache
  validation, then compared byte-exact logits from two independent CUDA+AMP
  forwards and failed before training. Parallel GNN/scatter replay can differ
  numerically without reading targets, so this conflated leakage with kernel
  replay determinism.
- Decision: Advance the Dilemmadata model contract to `1.1.0` and expose typed
  `predict(raw_graph)` plus `supervise(encoded, predictions, target_batches,
  class_weights)`. `forward` reuses `supervise`, preserving one join/loss
  implementation. Leakage evidence predicts once, joins original and mutated
  targets to that same object, and requires candidate identities plus every
  prediction tensor object, storage, layout, and value to remain exact while
  target and supervision/loss evidence changes.
- Decision: Advance smoke/bundle to `1.2.0` and add CUDA replay diagnostic
  `1.0.0`. Independent forwards require exact candidate identities, finite
  logits, FP32 max-absolute/max-relative/cosine evidence, elementwise
  `atol=0.005`, `rtol=0.005`, and cosine >= `0.9999`. Checkpoint model tensors
  reload bit-exactly; independent reload logits use this diagnostic and are
  not leakage evidence.
- Consequences: The failed SHA is not hardware-training success and the PR
  remains draft pending a new RTX run. The semantic target-index unblock is
  preserved. No target artifacts are rebuilt; head/loss, raw index/cache,
  target cache, grouping, split, graph and model-input contracts are unchanged.
  Phase 9C and long training remain out of scope.

## 2026-08-17 — ADR-082: Dilemmadata AMP reuses the Phase 8B FP32 and scaler policy

- Status: Accepted as blocking Phase 9B.2C remediation in draft PR #24;
  independent RTX 3090 execution remains pending.
- Context: Exact RTX attempt
  `cd87a3436f6db9ecadbab64dfb229ef039c465bf` passed production cache,
  semantic-index, leakage, and replay gates. Its finite first loss yielded a
  non-finite gradient at `task_heads.heads.task_03.3.weight`. The smoke failed
  immediately after `GradScaler.unscale_`, before public overflow handling
  could skip the attempt and reduce scale; no update or checkpoint occurred.
- Decision: Advance `DilemmadataHierarchicalModel` to `1.2.0` and add the
  opt-in `DilemmadataFp32HeadLossBoundary@1.0.0`. Permit encoder float16
  autocast, cast each Dilemmadata head input differentiably to FP32 on-device,
  and execute head logits, CE, source-entry reduction, and total loss in FP32.
  Do not alter generic heads by default or detach/transfer the gradient path.
- Decision: Advance smoke/bundle to `1.3.0` and add
  `DilemmadataAmpPolicy@1.0.0`, reusing Phase 8B's public scale-transition
  oracle with explicit initial scale `16384`, growth factor `2.0`, backoff
  factor `0.5`, and growth interval `2000`. Record attempted/applied/skipped
  attempts; a scale decrease is a skipped overflow and does not move the
  scheduler. Record bounded offending names/scales, accept gradient/update
  evidence only from finite applied attempts, require at least one applied
  update and final recovery, require finite model/optimizer state and changes
  in the encoder and all four heads, and checkpoint/restore the exact scaler
  configuration and state.
- Consequences: Persistent overflow and zero applied updates fail closed; the
  failed SHA remains negative hardware evidence and no hardware training
  success is claimed. Head/loss mathematical contracts, target artifacts,
  raw/cache/grouping/split/graph/model-input versions and fingerprints remain
  unchanged. No cache rebuild, corpus audit, long training, Phase 9C, PDMX, or
  Phase 10 is authorized.

## 2026-08-17 — ADR-083: CUDA allocator residue is not tensor-retention evidence

- Status: Accepted as the final Phase 9B.2C lifecycle remediation in draft PR
  #24; independent RTX artifact publication remains pending.
- Context: RTX attempt `20ba52b7e6fcf961702517d7ed9e467ea57eeea7`
  completed training, exact checkpoint/model/scaler reload, bounded reload
  logits, and validation. All tracked prediction weakrefs were dead, but the
  final gate rejected 67,108,864 allocated bytes in the still-live CUDA
  process. CUDA runtime/workspace/caching-allocator residue is not itself proof
  of a retained prediction tensor.
- Decision: Advance smoke/bundle to `1.4.0` and add
  `DilemmadataCudaLifecycleEvidence@1.0.0`. Preserve exact zero retained tracked
  prediction weakrefs. Do not claim zero globally live CUDA tensors without a
  safe bounded enumeration. Record allocator end/peak allocated and reserved
  bytes and require end allocated/reserved not to exceed their peaks.
- Decision: After one warmup, execute three identical no-grad validation
  predictions. Delete outputs, collect, synchronize, empty the cache, and
  synchronize between measurements. Record allocated/reserved sequences,
  tracked/retained weakref counts for every pass, and maximum allocated growth;
  require zero growth and zero retained prediction weakrefs. Treat subprocess
  exit as the standalone CUDA-context release boundary.
- Consequences: Constant allocator residue may pass but monotonic growth,
  retained prediction tensors, malformed lifecycle evidence, and the old
  unproved `retained_cuda_tensor_count=0` claim fail closed. The failed SHA did
  not publish hardware evidence. Model, target, raw/cache/grouping/split/graph/
  model-input contracts and fingerprints remain unchanged; no rebuild, audit,
  long training, Phase 9C, PDMX, or Phase 10 is authorized.

## 2026-08-17 — ADR-084: Phase 9C-A is a one-seed validation-only production pilot

- Status: Accepted for executable control-plane implementation; independent
  RTX profile and production pilot execution remain pending.
- Context: Phase 8B.2 supplies compute-matched SSL mechanics and Phase 9B.2
  supplies safe four-head Dilemmadata supervision, but the earlier comparison
  contracts did not execute the requested three-source SSL → Dilemmadata
  one-seed matrix or fix its normalized validation selection rule.
- Decision: Add `Phase9CProtocol@1.0.0` and artifact/plan/profile/selection/test-
  lock contracts `1.0.0`. Fix seed 17, primary variants scratch, Phase 7A,
  Phase 8A mask-only and Phase 8B multilevel-equal, with optional single-level
  variants excluded by default. Pair initialization and sample schedules and
  require 12 observed encoder forwards per SSL logical update.
- Decision: SSL uses only train raw graphs from HookTheory, POP909-CL, and
  Dilemmadata under equal source weights and deterministic no-replacement
  cycles. Downstream uses the complete 577/71 train/validation records and
  keeps all 71 test records locked. Select by mean `NLL/log(class_count)` over
  the four tasks, with macro-F1/NLL/epoch/identity tie breakers fixed before
  results. Component bootstrap expresses validation-sample uncertainty only.
- Decision: Production budgets and batch size are unset until a per-candidate
  subprocess RTX profile. Profile never starts production automatically.
  Cells stage and publish atomically, completed cells are immutable, and the
  final regular-file bundle is independently SHA-256 verified.
- Consequences: This adds no new data, graph, model, target, head, loss, or
  checkpoint semantics. Bounded fixture results are mechanics evidence only.
  Test evaluation, multi-seed claims, PDMX/Phase 10, PLL, and critic/quality
  work remain unauthorized.

## 2026-08-17 — ADR-085: Phase 9C-A composes existing splits and compares fixed-budget final checkpoints

- Status: Accepted; narrows and corrects ADR-084 before production execution.
- Context: The initial control plane required a prebuilt three-source SSL
  manifest and evaluated downstream `best.pt`, while its prose implied a
  normalized-NLL selection rule that the training path did not implement.
- Decision: Deterministically compose the existing HookTheory+POP909-CL and
  Dilemmadata manifests without repartitioning. Preserve every source
  assignment exactly, validate the result against all three indices, and reject
  any Dilemmadata validation/test membership in SSL train.
- Decision: Train every downstream configuration for the same number of fully
  applied optimizer updates and compare only the resulting `last.pt` on the
  complete validation split. Normalized NLL is a final-checkpoint comparison
  metric, not a between-epoch checkpoint-selection policy. Test remains locked.
- Consequences: No data/model/target/checkpoint-container contract changes and
  no production execution. A skipped update, assignment drift, manifest/index
  mismatch, destination conflict, or non-`last.pt` comparison fails closed.

## 2026-08-18 — ADR-086: Phase 9C-A binds a raw-only structural SSL eligibility view

- Status: Accepted as the minimal production-runtime remediation before a new
  independent RTX profile. Production execution remains pending.
- Context: Exact head `2ee853f7dc025b4dedb51817e878682182140604`
  composed Hydra successfully, then failed in the initial validation pass of
  `ssl/phase8a_mask_only`. The exact exception fingerprint resolves to
  HookTheory `piece:hooktheory-ANmpQBZngyM`, whose raw graph has zero note
  nodes. A fixed hierarchy-policy pass cannot mask that graph while leaving a
  pitched note visible. Silent policy fallback, replacement sampling, masking
  all notes, or treating the record as an applied objective would violate the
  accepted Phase 8A/8B.2 contracts.
- Decision: Preserve every corpus-index record and every source split
  assignment exactly. Materialize a separate immutable
  `phase9c-ssl-eligibility@1.0.0` raw-only identity view, bound to all index and
  split-manifest fingerprints. A record is eligible for the common paired SSL
  population only when it contains at least two raw notes occupying at least
  two canonical bars. This is the structural intersection needed by the
  scheduled independent, onset, beat, contiguous-bar, and track/bar policies;
  it reads no target sidecar and uses no label, mask, theory, provenance, or
  confidence value.
- Decision: Apply the same eligibility identities to train sampling, fixed SSL
  validation membership, metadata schedule attestation, and every SSL variant.
  Equal source weights and deterministic no-replacement cycles operate over
  each source's eligible train view. Excluded records retain their original
  train/validation assignment and remain available to unrelated data paths;
  no record is repartitioned and Dilemmadata validation/test remain excluded
  from SSL train.
- Decision: Expose the view only through the new `data=phase9c_mixed` Hydra
  group. Existing Phase 7/8 `data=mixed` resolved configs and checkpoint data
  fingerprints remain unchanged. Advance only `Phase9CProtocol` and
  `Phase9CPlan` to `1.1.0`; the new eligibility artifact starts at `1.0.0`.
- Consequences: The original failed candidate root and report remain negative
  evidence. A local full HookTheory+POP909-CL scan found 19,143/2,328 eligible
  and 1,850/249 excluded train/validation HookTheory records, plus 701/101
  eligible and zero excluded POP909-CL train/validation records. Production
  Dilemmadata counts and the final three-source fingerprint are materialized
  on the RTX host before candidate subprocesses. This changes only the
  raw-structural SSL-eligible population required to make the predeclared
  masking schedule executable; downstream Dilemmadata membership, budgets,
  models, objectives, checkpoints, selection, and test lock are unchanged.

## 2026-08-19 — ADR-087: Post-pilot class balancing is a sealed, opt-in downstream diagnostic

- Status: Accepted as a narrow Phase 9C-A follow-up after the completed
  one-seed validation pilot; it does not revise that pilot or authorize test
  evaluation.
- Context: The fixed-budget validation artifacts show frozen probes with one
  argmax class per task and severe within-task imbalance. Equal weighting of
  the four tasks does not balance classes inside a task, so this is an
  optimization/readout diagnostic before making a claim about SSL information.
- Decision: Materialize a fingerprinted class-weight artifact only from the
  sealed Dilemmadata train-prior artifact. Use
  `inverse_sqrt_frequency_supported`: a zero-support class has weight zero and
  every positive-support class receives inverse-square-root count weighting,
  scaled so the mean weight of an observed train source entry is one. Reject a
  non-train, invalid, incomplete, or fingerprint-mismatched prior/artifact.
- Decision: The artifact is explicit and opt-in through the Dilemmadata
  training configuration. It affects categorical training CE only; validation
  continues to report the existing unweighted source-entry metrics, and the
  Phase 9C test lock remains unchanged.
- Decision: For the class-balanced diagnostic only, use AMP loss scale one
  without growth. This changes no weighted-loss value or optimizer update: it
  prevents a rare-class gradient from overflowing FP16 after loss scaling and
  before the existing fail-closed clipping check. The unweighted protocol
  retains its existing AMP behavior.
- Consequences: Any class-balanced result uses a fresh output root and is a
  diagnostic comparison, not a replacement for the immutable unweighted
  fixed-budget pilot. Splits, datasets, model structure, SSL protocol,
  budgets, checkpoint policy, and production artifacts remain unchanged.

## 2026-08-19 — ADR-088: Class-balanced diagnostic reuses only hash-bound SSL exports

- Status: Accepted for the pending class-balanced downstream diagnostic.
- Context: Repeating the three completed SSL cells would add substantial GPU
  time while answering no new class-balancing question. Reuse is safe only if
  it cannot silently substitute another data view, seed, budget, encoder
  export, or SSL checkpoint.
- Decision: The diagnostic may omit new SSL/export cells only when it binds a
  completed pilot root with identical data projection, seed, primary variants,
  SSL update budget, and batch size. The plan records SHA-256 values for every
  reused encoder export and source SSL checkpoint; the runner rechecks them
  before each downstream command. Any mismatch or missing artifact fails
  closed.
- Consequences: The diagnostic still starts fresh Dilemmadata heads, optimizer,
  scheduler, scaler, train prior, class-weight artifact, downstream fixed-
  budget `last.pt`, validation, and bootstrap artifacts under a new root. It
  does not resume or mutate the completed pilot, and it does not unlock test.

## 2026-08-21 — ADR-089: Phase 9C-B isolates an optional raw-only onset sequence decoder

- Status: Accepted as a one-seed diagnostic; independent RTX profile and the
  four production cells remain pending.
- Context: The class-balanced Phase 9C-A diagnostic did not establish a useful
  SSL advantage. Its supervised decoder predicts every onset/beat/bar candidate
  independently, so weak downstream results cannot distinguish an encoder
  limitation from a temporal-readout bottleneck.
- Decision: Preserve the accepted MLP path bit-exact and add an optional one-
  layer bidirectional GRU over raw onset rows only. Each direction has
  `hidden_dim / 2` units. Gated residual fusion preserves the local onset;
  deterministic mean pooling through raw ownership provides separate gated
  residual context to beat and bar with learned availability states. No target,
  sidecar, label, provenance, confidence, float-time sorting, or synthetic onset
  participates.
- Decision: Compare scratch/SSL × MLP/onset-BiGRU under seed 17, the existing
  supported inverse-square-root class weights, one exact metadata-only batch
  schedule, equal attempted/applied budgets, complete validation, and final
  `last.pt`. Require an explicit immutable SSL checkpoint, SHA-256, and source
  kind; transfer only existing encoder prefixes and fingerprint fresh decoder/
  head tensors before and after transfer. Test stays locked.
- Consequences: Onset-BiGRU decoder, Phase 9C-B protocol/plan/bundle start at
  `1.0.0`; Dilemmadata evaluation advances to `1.1.0` for normalized NLL,
  entropy, distributions, support, accuracy aliases, and aggregate diagnostics.
  Canonical/raw graph/cache/split/target/class-weight/head/loss and MLP model
  contracts do not change. A one-seed outcome remains descriptive only.

## 2026-08-21 — ADR-090: Downstream planning and runtime share one exact schedule builder

- Status: Accepted for the Phase 9C-B profile remediation.
- Context: The independent profile at `5ac4a30` completed three optimizer
  updates in `scratch_mlp` and then failed closed on
  `training.phase8b2.actual_sample_schedule_mismatch`. Planner and engine had
  equal schedule payload semantics but different canonical byte encoders: the
  Phase 9C-B generic encoder omitted the trailing newline required by the
  Phase 8B.2 schedule contract. Profile planning also sliced a production
  schedule rather than constructing the exact profile epoch size.
- Decision: Production dataset-view construction is shared between planning
  and runtime, including the Dilemmadata target-sidecar index requirement.
  Both use one decoder-neutral builder around the real
  `DeterministicQuotaSampler` and one normalized downstream fingerprint
  function. Profile and production schedules are built separately with their
  exact seed, first epoch, epoch count, steps per epoch, epoch size, batch size,
  and identity representation.
- Consequences: The observed-versus-declared check remains fail closed without
  an allowlist or bypass. Phase 9C-B protocol and plan advance to `1.0.1`;
  decoder, bundle, data, split, cache, target, class-weight, SSL, scientific
  budget, and evaluation contracts remain unchanged. The failed root cannot be
  resumed and the profile must use a fresh output root at the remediated SHA.

## 2026-08-22 — ADR-091: Checkpoint contracts reconstruct decoder type explicitly

- Status: Accepted for the Phase 9C-B profile remediation.
- Context: The profile at `9de8f34` trained and checkpointed the scratch
  onset-BiGRU cell, but evaluation rebuilt only `model_contract.config`.
  Because the writer stores a non-default decoder separately at top level, the
  evaluator silently instantiated MLP and strict loading rejected valid
  `sequence_decoder.*` tensors. Planning also allowed an omitted encoder
  export to alias the full SSL checkpoint.
- Decision: One typed Dilemmadata reconstruction helper consumes the complete
  model contract and state inventory. No top-level decoder means default MLP;
  onset-BiGRU requires the exact decoder contract version, structure and raw
  semantics. Contract/state cross-kind pairs fail closed, followed by an
  unchanged strict state load. Decoder type is never inferred from tensor
  names. Phase 9C-B also requires and structurally validates a distinct,
  hash-bound encoder-only export during plan preflight.
- Consequences: Legacy/default MLP checkpoint logits remain unchanged; valid
  onset-BiGRU checkpoints become evaluable. Missing, substituted or malformed
  decoder contracts and full-checkpoint-as-export substitutions are rejected
  before evaluation or CUDA cell execution. Model architecture, weights,
  data, splits, caches and scientific budgets are unchanged.

## 2026-08-22 — ADR-092: Convergence uses applied-update checkpoints and post-training milestones

- Status: Accepted for the Phase 9C-C one-seed diagnostic; production RTX
  execution remains pending.
- Context: The verified Phase 9C-B matrix compared final checkpoints after
  3,000 updates but one epoch produced only one aggregate train point and one
  final validation point. It cannot distinguish a stable disadvantage from an
  early stopping artifact. Treating later observations as separate epochs or
  runs would change the sampler and optimizer trajectory.
- Decision: Compare only scratch MLP and SSL MLP under the exact paired seed-17
  conditions for one continuous epoch and exactly 9,000 applied updates.
  Record scalar update telemetry every 100. Save a separate Phase 9C-C atomic
  checkpoint every 1,000 with model, optimizer, scaler, RNG and deterministic
  sampler position. AMP-skipped attempts do not advance the schedule and the
  same batch is retried; repeated overflow fails closed.
- Decision: Evaluate immutable checkpoints at updates 0, 1,000, 3,000, 6,000
  and 9,000 only after continuous training, using the unchanged strict official
  validation loader. Bind checkpoint SHA, model state and validation membership
  in every milestone. Validation cannot select, stop or mutate training. The
  convergence report contains values and predeclared deltas but no automatic
  plateau verdict, superiority claim or significance claim.
- Consequences: The generic Phase 6C epoch checkpoint and Phase 9C-B behavior
  remain unchanged. No model, decoder, SSL objective, class weight, target,
  cache, split, graph, head, loss or scientific data budget changes. One-seed
  results remain descriptive and test stays locked.
