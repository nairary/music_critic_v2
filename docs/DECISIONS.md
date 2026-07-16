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
