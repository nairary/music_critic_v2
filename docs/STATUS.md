# Music Critic V2 Status

## Current phase

- Date: 2026-07-16
- Current phase: Phase 1A — canonical schema API and JSON contract
- State: completed; current handoff task
- Previous phase: Phase 0 — clean repository bootstrap and legacy audit,
  completed
- Next phase: Phase 1B — implement the accepted schema, validation, and
  serialization API with tests

## Phase 1A results

- Fixed `SCHEMA_VERSION` as `2.0.0`.
- Finalized the exact future public API for:
  - `music_critic.data.timing`;
  - `music_critic.data.schema`;
  - `music_critic.data.validation`;
  - `music_critic.data.serialization`.
- Defined complete fields for rational timing, pieces, metadata, tracks, notes,
  bars, beats, tempo, meter, key signatures, annotations, targets, provenance,
  quality flags, validation issues/reports, and validation exceptions.
- Accepted frozen dataclasses with tuple collections and no non-standard
  dependency.
- Accepted stable prefixed string entity IDs and canonical collection ordering.
- Defined pickup, tempo/meter change, cross-bar note, overlap, grace-note, and
  percussion semantics.
- Defined explicit unavailable-versus-empty behavior.
- Defined categorical, scalar, multi-label, and distribution target encodings,
  including masks, confidence, per-entry source, and provenance.
- Defined strict unknown-field rejection, exact-version compatibility, and
  deterministic field-by-field JSON serialization.
- Defined validation error and warning codes and their severity boundary.
- Added a complete synthetic two-track canonical JSON example with tempo,
  meter, pickup bars and beats, pitched and percussion notes, an unavailable
  optional field, target-only track roles, a partially masked theory target,
  and provenance.

## Files changed in Phase 1A

- `docs/DATA_CONTRACT.md`
- `docs/DECISIONS.md`
- `docs/ROADMAP.md`
- `docs/STATUS.md`

No production Python files, tests, dependency declarations, generated data, or
legacy files were changed.

## Verification

- All authoritative repository documents were read before editing.
- The legacy checkout was not inspected because no V1 behavior was needed to
  settle the V2 contract.
- Documentation-only scope was confirmed by the Git diff.
- The embedded canonical JSON example parsed successfully with the Python
  standard library.
- Structural checks for required JSON keys, normalized rationals, entity
  references/order, aligned target lengths, and mask/null semantics passed.
- `git diff --check` passed.
- No implementation tests are added or required for this design-only phase.

## Phase 0 retained status

- Legacy path:
  `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic`
- Legacy commit: `2d8281f31cc9ad9c8fecaf332da0c61e0e949415`
- Legacy branch: `sections`
- Legacy initial state: dirty; exact porcelain entries remain stored in
  `legacy_snapshot.json`.
- Phase 0 verification result: `7 passed in 0.03s`; compile and legacy snapshot
  checks passed.

## Blockers and remaining ambiguities

None for Phase 1B implementation. Future schema changes must be handled through
an explicit schema version and ADR rather than reinterpretation of `2.0.0`.

Repository licensing remains an organizational question for future publication
or distribution. Developers still need the `dev` extra or another pytest
installation to run test targets.
