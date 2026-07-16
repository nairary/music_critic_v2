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
