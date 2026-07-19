# Music Critic V2 Status

## Current phase

- Date: 2026-07-19
- Completed phase: Phase 1 — canonical data schema and serialization
- Phase 1A: Completed
- Phase 1B.1: Completed
- Phase 1B.2: Completed
- Phase 1B.3: Completed
- Phase 1 merge SHA: `37edf76889730980aa6ce9e9ec981e362c3480a9`
- Current branch: `phase/2a-generic-midi-adapter-mvp`
- Current phase: Phase 2 — generic MIDI and HookTheory adapters
- Phase 2 state: In progress
- Current task: Phase 2A.1 — generic MIDI adapter MVP

## Phase 2 migration status

- The HookTheory migration contract is documented in
  `docs/HOOKTHEORY_MIGRATION.md` from the reverse-engineered legacy pipeline.
- HookTheory melody pitch uses the accepted legacy derived reconstruction
  formula anchored at MIDI 72, with algorithmic provenance method
  `hooktheory_sd_octave_to_midi_v1`.
- Applied harmony is deferred from the first HookTheory adapter.
- The HookTheory adapter has not been implemented.
- Phase 2B.0 legacy audit/golden fixtures and Phase 2B.1 adapter implementation
  remain pending.
- No graph, dataset, model, SSL, training, preference, quality, inference, or
  GRPO work has started.

## Phase 2A.1 generic MIDI result

- Public API: `MidiAdapterConfig`, `MidiAdapterError`, and `load_midi_piece`
  from `music_critic.adapters`.
- Added the sole runtime dependency `mido>=1.3,<2`. The accepted Phase 1 data
  layer remains standard-library-only and importing `music_critic.data` does
  not import `mido`.
- Supported input: Standard MIDI type 0 and type 1 files with PPQN timing,
  multiple source tracks, multiple channels per source track, empty source
  tracks, note-on/off and velocity-zero note-off, tempo/meter/key metadata,
  names, instruments, programs, percussion channel 9, and empty/no-note files.
- Timing remains exact: absolute source ticks are integers and canonical onset,
  duration, bar, and beat positions use `RationalTime` without float conversion,
  rounding, epsilon comparison, or note splitting at bar/tempo/meter changes.
- Canonical track identity is `(source_track_index, MIDI channel)`. Note pairing
  is FIFO per `(source_track_index, channel, pitch)` and never crosses source
  tracks, channels, or pitches. Unmatched note-offs and dangling note-ons are
  diagnosed without invented notes; real same-tick pairs are preserved as
  grace-like zero-duration notes.
- Tempo defaults to `500000` microseconds per quarter at tick 0 when absent or
  first observed later. Meter defaults to `4/4` at tick 0 under the same policy.
  Defaults use `kind=default` provenance; observed source events use
  `kind=source`, the accepted observed equivalent.
- Global metadata events use deterministic `(tick, source track, message)`
  ordering. Exact duplicates are removed and conflicting same-tick values keep
  the first deterministic value plus a namespaced quality flag.
- Generic MIDI emits `annotations=()` and `targets=()`. Every successful
  conversion passes `validate_piece` and both string/file JSON round trips
  preserve exact equality. No canonical cache is written by default.
- Rejected input: MIDI type 2, SMPTE/non-PPQN timing, non-positive PPQN,
  unreadable/corrupt files, and meter changes inside an active bar.
- Intentionally unsupported: MIDI 2.0, proprietary sequencer/SysEx semantics,
  lyric alignment, sustain-pedal reconstruction, voice/role/pickup inference,
  chord or key detection from notes, section detection, and aesthetic scoring.

## Phase 2A.1 verification

All commands used the project-local Python 3.13.5 interpreter at
`.venv/bin/python`.

- `tests/data/test_timing.py`: `28 passed in 0.04s`.
- `tests/data/test_schema.py`: `13 passed in 0.13s`.
- `tests/data/test_validation.py`: `110 passed in 0.23s`.
- `tests/data/test_serialization.py`: `94 passed in 0.30s`.
- `tests/adapters/test_midi.py`: `50 passed in 0.22s`.
- Full suite: `302 passed in 0.60s`.
- `.venv/bin/python -m compileall src scripts`: passed.
- Data-layer import isolation: `data import isolation passed`.
- Adapter public imports: `adapter imports passed`.
- Dependency import: `mido imported`.
- `git diff --check`: passed.
- Synthetic smoke root: `/tmp/music-critic-midi-smoke.W2edj4`.
- Synthetic smoke: `files_seen=3`, `attempted=3`, `converted=3`, `failed=0`,
  `warnings=10`, `notes=3`, `tracks=5`, `type_0=2`, `type_1=1`.
- `MUSIC_DATA_ROOT` was unset. `real-data smoke test not run: no MIDI files found
  under MUSIC_DATA_ROOT`.

## Phase 2A.1 scope confirmation

- Phase 1 production code, Phase 1 data tests, the accepted schema/data
  contract, and the normative fixture were not modified.
- The Phase 0 repository-contract test was updated to allow `mido` only inside
  `music_critic.adapters`; its bans remain active everywhere else, and the
  adapter/document packages are now required repository structure.
- The read-only legacy repository remains at
  `2d8281f31cc9ad9c8fecaf332da0c61e0e949415` with the same pre-existing dirty
  status observed before this task. No legacy file was modified.
- HookTheory remains documentation-only. No graph, dataset, model, SSL,
  training, preference, quality, inference, or GRPO code was added.

## Final Phase 1 result

- Accepted and implemented canonical schema version `2.0.0` with an exact,
  explicit public `music_critic.data` API.
- Implemented normalized exact quarter-note timing with frozen, slotted
  `RationalTime` values and no float-equality timing contract.
- Implemented deeply immutable frozen canonical records. Collection fields are
  tuples, optional observations preserve `None` versus empty values, and raw
  note/track records contain no theory-label or semantic-role leakage.
- Implemented complete deterministic validation with structured errors and
  warnings, exact RFC 6901 paths, reference and ordering checks, target masks,
  confidence and provenance, exact musical timing semantics, and warning-only
  valid pieces.
- Implemented strict field-by-field decoding and validated deterministic JSON
  encoding. Unknown, missing, type, rational, version, and semantic failures
  retain their accepted error-code boundaries.
- Compact and indented JSON are deterministic; file output is UTF-8 with exactly
  one terminal newline, and public operations do not mutate canonical records
  or caller-owned mappings and lists.
- The normative `tests/fixtures/data/canonical_piece_v2.json` mapping decodes,
  validates with warnings only, re-encodes exactly, and remains equal through
  `dumps_piece` and `loads_piece`. Rational fields and immutable collections
  retain their exact Python types; masks, unknown confidence, provenance, and
  alternative annotation views are preserved.
- At Phase 1 completion the data layer used only the Python standard library and
  project runtime dependencies were empty. Phase 2A.1 preserves that data-layer
  isolation while adding `mido` only for adapters.
- No adapter, MIDI parser, graph, dataset, model, training, evaluation, or
  inference implementation was added in Phase 1.

The final float-decoding review fix in commit `396a2b5` was accepted. Huge
positive or negative integers supplied for float-valued mapping fields now
produce `VALUE_NOT_FINITE` at the exact path through
`CanonicalValidationError`; raw `OverflowError` cannot escape and inputs are
not clamped or mutated.

## Phase 1 commit history

- Phase 1A contract review and closure: `241d0e5`, `30ba3f9`, merged by
  `7ca1ce0`.
- Phase 1B.1 timing and schema types: `0ca7b95`.
- Phase 1B.2 validation: `b5c31c6`, with review fixes in `2c16d72`.
- Phase 1B.3 serialization: `1dd4e00`, with accepted float-decoding fix in
  `396a2b5`.
