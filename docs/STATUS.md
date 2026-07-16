# Music Critic V2 Status

## Current phase

- Date: 2026-07-17
- Current phase: Phase 1B — schema implementation and tests
- Completed task: Phase 1B.2 — canonical validation
- State: Phase 1B remains in progress
- Previous task: Phase 1B.1 — canonical timing and schema types, completed
- Next task: Phase 1B.3 — canonical serialization

## Phase 1B.2 results

- Implemented the complete standard-library validation API:
  `ValidationCode`, `ValidationIssue`, `ValidationReport`,
  `CanonicalValidationError`, `validate_piece`, and `validate_or_raise`.
- Validation collects all detectable issues, preserves deterministic RFC 6901
  paths and issue ordering, never mutates or repairs the input, and raises only
  when errors exist.
- Implemented schema-version, open-string, enum, finite-value, exact timing,
  duration, source-index, MIDI range, spelling, and piece-bound checks.
- Implemented lexical entity IDs, exact prefixes, global uniqueness, raw
  references, target references, provenance references, and quality-flag
  references without relying on collection indices.
- Implemented canonical collection-order checks for tracks, notes, bars, beats,
  events, annotations, targets, provenance, and quality flags.
- Implemented tempo and meter maps, required initial events, duplicate-onset
  checks, meter-at-bar-start checks, and the mid-bar tempo warning.
- Implemented complete bar, pickup, incomplete-final-bar, metric-offset,
  nominal-meter-duration, contiguous-coverage, beat-containment, denominator
  unit grid, pickup downbeat, and compound-meter behavior.
- Implemented note, percussion, grace-note, cross-boundary, and same-pitch
  overlap behavior.
- Implemented observation and target-alignment annotation rules.
- Implemented categorical, scalar, multi-label, and distribution targets,
  aligned lengths, masks, unknown available confidence, source/provenance
  requirements, entity alignment, alternative annotation views, and explicit
  negative labels.
- Implemented provenance presence, parent validity, parent-before-child order,
  iterative cycle detection, detail-key/value rules, timestamps, checksums, and
  unreferenced-record diagnostics.
- Implemented quality-flag syntax, severity, message, entity, and provenance
  validation.
- Implemented and tested all documented warnings:
  `EMPTY_PIECE`, `EMPTY_TRACK`, `SOURCE_RESOLUTION_UNAVAILABLE`,
  `INCOMPLETE_FINAL_BAR`, `OVERLAPPING_SAME_PITCH_NOTES`,
  `MID_BAR_TEMPO_CHANGE`, `LOW_CONFIDENCE_TARGET`,
  `UNREFERENCED_PROVENANCE`, `EMPTY_OBSERVATION`, and
  `PIECE_TRAILING_SILENCE`.
- Added a clean two-track dataclass fixture with pitched and percussion notes,
  complete bars/beats, provenance, targets, and an alternative annotation view.
- Fixed the Phase 1B.1 test portability gap by replacing the hard-coded
  subprocess working directory and added the explicit `TargetValue` alias
  assertion.

## Phase 1B.2 review fixes

- Corrected note collection ordering to derive track ranks from the documented
  canonical track sort key, independently of the current track tuple order.
  Track and note ordering errors are now reported independently.
- Corrected provenance ordering to select the lexicographically smallest
  currently ready provenance ID one node at a time. This implements the
  accepted parent-before-child ordering with `provenance_id` tie-breaking
  without breadth-first layer bias.
- Added cached typed views for top-level collections and exact
  `ValidationIssue` deduplication. Distinct codes, messages, severities, paths,
  or entity IDs remain separate diagnostics.
- Replaced quadratic all-pairs same-pitch overlap checks with grouping by
  `(track_id, pitch)`, deterministic interval sorting, and a linear scan per
  group after sorting. At most one overlap warning is emitted per later
  overlapping note.
- Added regressions for unsorted tracks/notes, provenance tie-breaking,
  malformed collection diagnostics, touching/nested/chained/cross-track/
  cross-pitch/grace-note overlaps, unsorted note input, percussion, and a
  2,000-note overlap group with bounded warning output.

## Phase 1B.2 review-fix files changed

- `src/music_critic/data/validation.py`
- `tests/data/test_validation.py`
- `docs/ROADMAP.md`
- `docs/STATUS.md`

No accepted public validation semantics or schema fields changed.

## Phase 1B.2 files created or changed

- `src/music_critic/data/validation.py`
- `src/music_critic/data/__init__.py`
- `tests/data/conftest.py`
- `tests/data/test_validation.py`
- `tests/data/test_schema.py`
- `docs/ROADMAP.md`
- `docs/STATUS.md`

`src/music_critic/data/schema.py`, `src/music_critic/data/timing.py`, and
`docs/DECISIONS.md` were not changed. No new architectural decision was
required.

## Phase 1B.2 verification

The system `/usr/bin/python` does not have pytest installed. Pytest commands
were run with the existing pytest-capable interpreter
`/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python`.
That interpreter is only the environment used for these checks and is not a
Music Critic V2 project requirement.

- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_timing.py -q`:
  `28 passed in 0.02s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_schema.py -q`:
  `13 passed in 0.09s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_validation.py -q`:
  `110 passed in 0.18s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest -q`:
  `158 passed in 0.29s`.
- `python -m compileall src`: passed.
- The explicit `PYTHONPATH=src` import-isolation command passed with:
  `standard-library import contract passed`.

The following codes are deliberately declared for Phase 1B.3 serialization and
are not emitted by `validate_piece`:

- `JSON_UNKNOWN_FIELD`
- `JSON_MISSING_FIELD`
- `JSON_TYPE_INVALID`
- `RATIONAL_INVALID`
- `RATIONAL_NOT_NORMALIZED`

Every other `ValidationCode` has executable `validate_piece` coverage.

## Phase 1B.2 scope and compatibility audit

- Validation uses only the Python standard library.
- No external dependency or dependency declaration changed.
- No `serialization.py` was created and serialization implementation did not
  start; Phase 1B.3 remains pending.
- No adapter, MIDI, graph, dataset, model, training, evaluation, or inference
  code was added.
- No accepted schema field or timing behavior changed.
- No legacy file was modified or used as a runtime dependency.
- The repository-wide heavy-import restriction remains unchanged.

## Completed Phase 1B.1 history

- Implemented `RationalTime` as a frozen, slotted, normalized exact rational
  value type in quarter-note units.
- Implemented all accepted Phase 1 schema aliases and exact frozen, slotted
  canonical dataclasses without constructor validation.
- Added explicit stable `music_critic.data` exports.
- Extracted the complete three-target canonical JSON fixture and kept it equal
  to the accepted `DATA_CONTRACT.md` example.
- Added timing and schema tests covering exact arithmetic, types, field order,
  immutability, raw/theory separation, target views/masks, and lightweight
  imports.
- Phase 1B.1 completed in commit `0ca7b95` with no dependency, legacy,
  validation, serialization, adapter, or graph changes.

## Blockers and remaining risks

No Phase 1B.3 implementation blocker is known. Phase 1B.3 must preserve the
validator's distinction between programmatic semantic errors and JSON decoding
shape/type errors, and must continue to reject unknown fields and unsupported
schema versions without mutating input data.
