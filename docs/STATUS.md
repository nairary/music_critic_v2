# Music Critic V2 Status

## Current phase

- Date: 2026-07-16
- Current phase: Phase 1A — canonical schema API and JSON contract
- State: completed after review revision; current handoff task
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
- Added globally unique target IDs and alternative annotation views, with
  uniqueness on `(task, annotation_view_id)` rather than task alone.
- Allowed available labels to carry unknown numeric confidence without becoming
  missing or zero-confidence targets.
- Corrected trailing-silence validation to use positive-duration sounding notes
  and observation annotations rather than structural bar/beat coverage.
- Expanded observable key-signature modes without treating them as local-key
  theory labels.
- Made persisted quality-flag identifiers open, namespaced, lowercase stable
  strings while keeping validation codes closed.
- Recorded the schema `2.0.0` integer-semitone spelling limitation and required
  provenance/quality-flag preservation for unsupported microtonal notation.
- Defined strict unknown-field rejection, exact-version compatibility, and
  deterministic field-by-field JSON serialization.
- Defined validation error and warning codes and their severity boundary.
- Added a complete synthetic two-track canonical JSON example with tempo,
  meter, pickup bars and beats, pitched and percussion notes, an unavailable
  optional field, target-only track roles, a partially masked theory target, an
  available target with unknown confidence, namespaced quality flags, and
  provenance.

## Review findings addressed

- Multiple legitimate analyses can now coexist as separate annotation views.
- Duplicate aligned entities remain invalid within one target array but are
  valid across distinct views.
- Unknown numeric confidence is distinct from missing supervision.
- `PIECE_TRAILING_SILENCE` is reachable and has defined empty, percussion,
  grace-note, point-annotation, and structural-only behavior.
- Modal key-signature observations are preserved.
- Adapter diagnostics no longer require routine schema migrations.
- Unsupported microtonal spelling cannot be silently rounded.

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
- Revised canonical JSON verification passed: 52 normalized rational checks,
  two target arrays, aligned lengths, required target IDs/views, unique
  `(task, annotation_view_id)` pairs, unknown available confidence, masked null
  semantics, namespaced quality flags, references, and reachable trailing
  silence all passed.
- `git diff --check`: passed.
- System `/usr/bin/python -m pytest -q`: could not start because that
  interpreter has no pytest module.
- `python -m pytest -q` using the existing pytest-capable environment:
  `7 passed in 0.03s`.
- `python -m compileall src`: passed.
- No new tests or dependencies were added.

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

None for Phase 1B implementation after the review fixes. Future schema changes
must be handled through an explicit schema version and ADR rather than
reinterpretation of `2.0.0`.

Repository licensing remains an organizational question for future publication
or distribution. Developers still need the `dev` extra or another pytest
installation to run test targets.
