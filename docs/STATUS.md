# Music Critic V2 Status

## Current phase

- Date: 2026-07-16
- Current phase: Phase 1B — schema implementation and tests
- Completed task: Phase 1B.1 — canonical timing and schema types
- State: Phase 1B remains in progress
- Previous task: Phase 1A — canonical schema API and JSON contract, completed
- Next task: Phase 1B.2 — canonical validation

## Phase 1B.1 results

- Implemented `RationalTime` as a frozen, slotted, normalized exact rational
  value type in quarter-note units.
- Implemented all accepted Phase 1 schema aliases and the exact canonical
  dataclasses with contract field names, order, and annotations.
- Kept schema records validation-free so invalid programmatic records remain
  constructible for the future one-pass validator.
- Added explicit stable exports from `music_critic.data` without wildcard
  imports or unfinished APIs.
- Extracted the complete canonical JSON example into the normative test fixture.
- Added timing tests for construction, normalization, ordering, arithmetic,
  unsupported operands, exact `Fraction` conversion, large integers,
  immutability, and slots.
- Added schema tests for aliases, exact field names/types/order, frozen slotted
  records, tuple collections, unavailable-versus-empty values, invalid object
  construction, public exports, target views/masks, raw/theory separation,
  fixture/document equality, and lightweight imports.

## Files created or changed

- `src/music_critic/data/__init__.py`
- `src/music_critic/data/timing.py`
- `src/music_critic/data/schema.py`
- `tests/data/test_timing.py`
- `tests/data/test_schema.py`
- `tests/fixtures/data/canonical_piece_v2.json`
- `docs/ROADMAP.md`
- `docs/STATUS.md`

`docs/DECISIONS.md` was not changed because Phase 1B.1 implemented the accepted
contract without exposing a new architectural decision.

## Verification

The system `/usr/bin/python -m pytest ...` commands could not start because the
system interpreter has no pytest module. The required pytest checks were
therefore run with the existing pytest-capable interpreter:

`/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python`

- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_timing.py -q`:
  `28 passed in 0.03s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest tests/data/test_schema.py -q`:
  `13 passed in 0.07s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m pytest -q`:
  `48 passed in 0.10s`.
- `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic/.venv/bin/python -m compileall src`:
  passed.
- `git diff --check`: passed.
- The explicit `PYTHONPATH=src` import-isolation command passed with:
  `standard-library import contract passed`.
- The fixture/document consistency test passed; the parsed fixture is identical
  to the complete JSON example in `docs/DATA_CONTRACT.md`.
- The supplementary `python scripts/check_legacy_unchanged.py` check reports
  that the current legacy worktree differs from the Phase 0 captured snapshot.
  The legacy porcelain status was already in that state before Phase 1B.1 and
  remained unchanged by this task.

## Scope and compatibility audit

- The implementation uses only the Python standard library.
- No dependency declaration changed.
- No `validation.py` or `serialization.py` was created.
- No adapter, MIDI, graph, dataset, model, training, or inference code was
  added.
- No legacy file was inspected for implementation logic, reused, or modified.
- Raw notes, tracks, and metadata contain no theory or semantic-role fields.
- Missing target labels remain represented by masks and null aligned values.
- The existing repository-wide heavy-import restriction remains unchanged. It
  must be narrowed before Phase 2 or model phases introduce legitimate optional
  imports, but not during Phase 1B.1.

## Blockers and remaining ambiguities

No Phase 1B.2 implementation blocker. Validation must implement the already
accepted contract without moving validation behavior into schema dataclass
constructors. The Phase 0 legacy snapshot should be reviewed separately because
its verifier no longer matches the legacy worktree's pre-existing status.
