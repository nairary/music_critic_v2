# Music Critic V2 Status

## Current phase

- Date: 2026-07-17
- Current phase: Phase 1B — schema implementation and tests
- Current task: Phase 1B.3 — canonical serialization, implementation completed
  and pending final review
- State: Phase 1 and Phase 1B remain in progress until final review
- Next action: final Phase 1 review
- Phase 2 remains pending

## Phase 1B.3 float-decoding review fix

- Final review confirmed that `_expect_float_number` allowed `OverflowError` to
  escape when a direct caller-owned mapping supplied an integer too large for
  Python float conversion, such as `10**10000`.
- Float-valued decoding now catches only that conversion overflow, emits
  `VALUE_NOT_FINITE` at the exact RFC 6901 field path, returns no placeholder
  value, and lets `piece_from_dict` raise `CanonicalValidationError` with the
  deterministic report. Values are never clamped or replaced with a finite
  boundary value.
- Added positive and negative huge-integer regressions for note source seconds,
  beat strength, target confidence, and distribution probabilities. Every case
  verifies the exact issue path and that the caller-owned mapping is unchanged.
- Added JSON exponent coverage showing that `1e9999` decoded as a float infinity
  continues through the existing semantic validation path and produces
  `VALUE_NOT_FINITE`.
- Corrected the key-signature IDs in the modal semantic-handoff tests from the
  unrelated `key:` prefix to the accepted `keysig:` prefix, isolating mode
  string and runtime-type behavior.
- Changed only `src/music_critic/data/serialization.py`,
  `tests/data/test_serialization.py`, and this status file. `docs/ROADMAP.md`
  already had the required Phase 1B.3 pending-review state and did not change.

## Phase 1B.3 results

- Added the standard-library-only canonical serialization API:
  `JsonObject`, `piece_to_dict`, `piece_from_dict`, `dumps_piece`,
  `loads_piece`, `dump_piece`, and `load_piece`.
- Writers validate through `validate_or_raise` before producing output. Warnings
  remain loadable and serializable; errors prevent encoding before a target file
  is opened or overwritten.
- Encoding is explicit for every schema record and `RationalTime`; it emits all
  accepted fields, converts immutable tuples to JSON arrays, represents rational
  time as exact `num`/`den` objects, and maps provenance detail pairs to JSON
  objects. No generic dataclass mapping contract is used.
- Decoding is explicit and field-by-field. Every canonical object requires its
  exact field set, JSON arrays must be lists at the mapping boundary, booleans
  are not integers, nested canonical objects must be mappings with string keys,
  and caller-owned mappings and lists are not mutated.
- Structural decoding safely collects independent issues and returns a
  deterministic `ValidationReport` with RFC 6901 paths. Unknown fields use
  `JSON_UNKNOWN_FIELD`, missing fields use `JSON_MISSING_FIELD`, runtime shape
  and type errors use `JSON_TYPE_INVALID`, and rational form errors use
  `RATIONAL_INVALID` or `RATIONAL_NOT_NORMALIZED`.
- Correctly typed semantic errors are handed to the accepted validator and keep
  their existing codes, including MIDI-range, target, reference, enum, finite
  value, and collection-order errors.
- Only schema version `2.0.0` is accepted. Other strings use
  `SCHEMA_VERSION_UNSUPPORTED`; missing and non-string versions retain their
  structural decoder codes. No compatibility migration is attempted.
- Target values decode according to `value_type`; unavailable entries remain
  `None`, immutable aligned fields become tuples, and missing labels are never
  interpreted as negative labels.
- Provenance detail objects decode to lexicographically sorted immutable pairs.
  Nested detail collections are rejected structurally, while non-finite values
  supplied through a direct mapping reach semantic validation.
- `dumps_piece` uses the exact deterministic JSON options from the accepted
  contract, preserves Unicode, sorts object keys, supports compact and indented
  output, and adds no terminal newline.
- `dump_piece` writes UTF-8 with `newline="\n"` and exactly one terminal newline;
  it does not create parent directories. `load_piece` reads UTF-8 through the
  same strict decoder.
- Malformed JSON, trailing data, and `NaN`/`Infinity` constants raise
  `json.JSONDecodeError`; invalid UTF-8 bytes propagate `UnicodeDecodeError`.
- The normative `canonical_piece_v2.json` fixture decodes to `CanonicalPiece`,
  retains its warning-only validity, round-trips to the exact fixture mapping,
  and is stable through deterministic JSON serialization.

## Phase 1B.3 files created or changed

- Created `src/music_critic/data/serialization.py`.
- Updated `src/music_critic/data/__init__.py` with the seven accepted public
  serialization exports while preserving all existing exports.
- Created `tests/data/test_serialization.py` with public API, fixture,
  determinism, file I/O, structural error, rational, version, target,
  provenance, syntax, encoding, mutation, and semantic-handoff coverage.
- Updated `tests/data/test_schema.py` only to synchronize its exact
  `music_critic.data.__all__` assertion with the seven accepted Phase 1B.3
  exports and remove the now-contradictory old absence check for
  `piece_to_dict`. The exact-export assertion remains strict; no unrelated
  schema test changed.
- Updated `docs/ROADMAP.md` and this status file.

`src/music_critic/data/schema.py`, `src/music_critic/data/timing.py`,
`src/music_critic/data/validation.py`, `docs/DATA_CONTRACT.md`, the normative
fixture, dependency declarations, and `docs/DECISIONS.md` were not changed. No
new architectural decision was required.

## Phase 1B.3 verification

No V2-local virtual environment was available. Pytest commands used the
existing pytest-capable interpreter
`/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python`.
That interpreter is only the environment used for checks and is not a V2
runtime dependency.

- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_timing.py -q`:
  `28 passed in 0.02s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_schema.py -q`:
  `13 passed in 0.10s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_validation.py -q`:
  `110 passed in 0.21s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_serialization.py -q`:
  `94 passed in 0.22s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest -q`:
  `252 passed in 0.43s`.
- `python -m compileall src`: passed.
- The explicit `PYTHONPATH=src` import-isolation check passed with
  `standard-library import contract passed`.
- `git diff --check` and the staged diff check: passed.

The additional `python scripts/check_legacy_unchanged.py` check did not pass:
the external read-only legacy checkout currently differs from the recorded
snapshot across pre-existing documentation, notebooks, scripts, and artifacts.
Phase 1B.3 did not modify, stage, restore, or import runtime code from that
checkout.

## Scope and remaining risks

- No dependency or dependency declaration changed.
- No adapter, MIDI parser, graph, dataset, model, training, evaluation,
  inference, or Phase 2 code was added.
- Raw unlabeled MIDI inference remains an architectural requirement; no gold
  semantic labels or segmentation were introduced into canonical loading.
- Exact rational timing, missing-label masks, provenance, target availability,
  confidence, and alternative annotation views remain unchanged.
- The only external verification risk is the independently changed legacy
  checkout reported above. It must not be repaired from the V2 task.
- The remaining V2 action is the final Phase 1 review; Phase 1 and Phase 1B are
  intentionally not marked complete yet.

## Compact completed history

### Phase 1B.2 — canonical validation

- Implemented deterministic structured validation, semantic and reference
  checks, canonical ordering, provenance, target masks/views, warnings, and
  scalable same-pitch overlap detection.
- Completed in commits `b5c31c6` and `2c16d72`; serialization-only decoder codes
  were reserved for Phase 1B.3.

### Phase 1B.1 — canonical timing and schema types

- Implemented normalized frozen `RationalTime`, immutable canonical schema
  records, stable public exports, and the accepted three-target normative
  fixture.
- Completed in commit `0ca7b95` without dependencies, adapters, graphs, or
  legacy runtime coupling.
