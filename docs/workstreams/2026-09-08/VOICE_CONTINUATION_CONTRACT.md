# Voice continuation candidate and target contract

2026-09-08. Status: isolated CPU-tested feasibility implementation. This is
separate from the random-versus-group SSL pilot. No graph relation, encoder,
head, loss, corpus adapter, cache schema, checkpoint, or production input is
changed. No real-data coverage or model-quality result is established.

Implementation: [`voice_continuation.py`](../../../src/music_critic/data/voice_continuation.py).
Fixtures: [`test_voice_continuation.py`](../../../tests/data/test_voice_continuation.py).
Contract version: `voice-continuation-candidates-targets@1.0.0`.

## Meaning of the target

For a source note, predict the uniquely identified next note at a strictly
later onset in the same **source-annotated voice and explicit source scope**
within this piece. The scope is a source part/staff/other documented namespace,
not an inferred MIDI track role. Voice identity does not mean melody, bass,
accompaniment or any other semantic voice function.

This first contract uses canonical sounding notes: tied rows that an existing
adapter merged into one canonical note are not separate continuation targets.
Sustained overlap is allowed, because next onset need not follow the source
note's offset. Grace notes follow exact onset order; coincident grace/ordinary
notes in the same voice are ambiguous. Simultaneous notes sharing a voice ID
are not resolved by pitch rank, duration, source row order or a heuristic.

`VoiceAnnotationView` explicitly binds piece ID, annotation-view ID, source
annotations and provenance records. A `VoiceAnnotation` has a canonical note
ID, `(source_scope, source_voice_id)`, source kind, provenance reference and
optional numeric confidence. Supported source kinds are `dataset`, `human`
and `synthetic`; algorithmic, derived and pseudo-label assignments reject.
The caller remains responsible for authenticating the asserted annotation
source: a type check cannot prove that an external claim is human annotation.

`CanonicalNote.voice` and `CanonicalNote.staff` are never read by this module.
An empty sidecar produces entirely unavailable targets even if scalar voice
metadata exists. This avoids silently treating a repeated local voice number
as a globally unique identity, or `track_id` as a voice label.

## Candidate construction and inference boundary

`build_voice_candidates(piece, policy)` projects only canonical note ID, exact
onset, pitch and duration. Every note can be an anchor/destination, including
notes in different MIDI tracks. A candidate must have a strictly later onset
within the inclusive exact onset-distance horizon. Candidates are ordered by
`(onset_qn, pitch, duration_qn, note_id)` and truncated to the predeclared
per-note cap. No annotation value, mask, confidence, spelling, staff, voice,
track summary, provenance, target span or semantic boundary affects the result.

The explicit default mechanics are an 8-quarter-note onset horizon, at most
16 candidates per note and at most 4,096 notes per piece. These defaults are
engineering bounds, not selected scientific hyperparameters. Oversized pieces
reject rather than silently dropping source notes. Candidate construction uses
sorted exact times and bounded slices, with `O(N log N + N K)` work and no
quadratic all-pairs allocation. Target mapping is an intentionally bounded
reference implementation and may scan the piece per anchor; it is not a
corpus preprocessing implementation.

The policy uses raw pitch only to break equal-onset candidate ties. It is not
a nearest-pitch gold voice heuristic, and crossing voices retain their source
assignment. This tie break can exclude a true successor under a tight cap;
candidate misses are reported rather than repaired using labels.

The resulting immutable candidate object contains one piece ID and local note
IDs. Alignment rejects a different sidecar piece ID, foreign annotation note
IDs, duplicates, unresolved provenance, malformed ordering and forged candidate
rows that do not match the raw policy. No cross-piece batch edge construction
is present. Source loaders must retain trustworthy piece IDs; matching strings
alone cannot authenticate source bytes.

## Availability and confidence

`map_voice_continuation_targets(candidates, view)` returns a separate target
row for each candidate anchor. Available rows contain `next_note_id`, its
zero-based index in that anchor's candidate list, `mask=True`, evidence and
confidence. A target is unavailable for each of these explicit reasons:

| Reason | Meaning |
| --- | --- |
| `missing_annotation` | The anchor has no scoped source voice assignment. |
| `ambiguous_source_onset` | More than one note shares the anchor's voice and onset. |
| `ambiguous_successor_onset` | The next same-voice onset contains multiple notes. |
| `incomplete_annotation_interval` | Any note from anchor onset through proposed successor onset has unknown voice. |
| `no_observed_successor` | No later same-voice note is observed in this piece. |
| `successor_outside_candidates` | A unique annotated successor is excluded by raw horizon or cap. |

The interval completeness gate is deliberately conservative across all scopes:
an unannotated intervening note might belong to this voice and invalidate a
claimed next-note link. Unknown notes at either endpoint onset also mask the
link. Unknown notes strictly after the next onset do not affect the earlier
link. Coverage losses must be measured before a real experiment; widening the
gate using source scope requires a separately reviewed annotation contract.

Every unavailable row has null next-note ID, candidate index and confidence,
and empty evidence. It is excluded from supervised loss. There is no invented
negative edge, no `end_of_voice` class and no end-of-piece-as-negative rule.
The reason field is diagnostic only and must not enter model inputs.

Available rows retain all interval annotation records used to establish the
claim, including their original sources, confidences and provenance references.
Derived target confidence is the minimum of those numeric source confidences
only when all are known; otherwise it stays null. This is an explicit
conservative aggregation convention, not a calibrated probability or invented
confidence of one. A confidence of zero remains an available source annotation
with zero confidence; any future loss-weight policy must be separately explicit.

The existing canonical `TargetArray` can hold note-aligned categorical strings,
but has no edge alignment or dependent next-note reference validation. The
existing model ontology/collator also supplies no variable candidate-index task
for voice continuation. Therefore this deliverable remains an isolated typed
sidecar. It does not silently introduce a new canonical target or trainable head.

## Actual annotation feasibility in the current snapshot

This is a code/document audit; no real corpus records were opened or counted.

| Source | Confirmed implementation evidence | Voice-target limitation |
| --- | --- | --- |
| Dilemmadata AN | `adapters/dilemmadata.py` parses optional `s_part_id` and `s_voice_id` into `source_voice_key=(part, voice)`. | Canonical note output retains integer `voice` but AN has `staff=None`; original part scope is not a canonical note field. The single source-neutral track cannot reconstruct it. |
| Dilemmadata DLC | The same parser constructs `source_voice_key=(staff, voice)` and preserves scalar staff/voice where parseable. | Source notation IDs need validation as continuing voices; per-note source confidence and completeness are not provided by the canonical note. Missing fields remain unknown. |
| Dilemmadata ties/repairs | `_merge_notes` / `_merge_notes_with_repairs` merge source rows; `DilemmadataSourceRowBinding` retains row-to-note IDs and `repair_mask_scope`. | A future importer must use these bindings, preserve scope and mask conflicting/cross-voice repair evidence. Never select the first tied row's voice as new gold. |
| Generic MIDI | `adapters/midi.py` creates notes with `voice=None`. | No available voice supervision. MIDI track/channel labels cannot fill it. |
| HookTheory | `adapters/hooktheory.py` creates notes with `voice=None`. | Melody-only content is not an explicit voice annotation source for this task. |
| POP909-CL | The adapter builds its score projection through the generic MIDI adapter. | Score/chord channel separation is harmonic evidence, not source voice continuation gold. |
| PDMX supported JSON projection | `adapters/pdmx.py` records `voice_policy=unavailable_no_inference`; existing tests prove injected `voice`/`staff` fields do not change canonical notes. | The current accepted projection has no voice annotations. Do not infer them from tracks or claim upstream full-score coverage from fixtures. |

Primary local code references: `data/schema.py:CanonicalNote`,
`adapters/dilemmadata.py:_RawObservation`, `:_parse_raw_file`,
`:_merge_notes`, `:_merge_notes_with_repairs`, and
`:DilemmadataSourceRowBinding`; `adapters/midi.py`, `adapters/hooktheory.py`,
`adapters/pop909_cl.py`, `adapters/pdmx.py`. Relevant accepted constraints are
`DATA_CONTRACT.md` CanonicalNote/TargetArray, `ARCHITECTURE.md` raw inference
and next-in-track definitions, `DECISIONS.md` ADR-073 and ADR-WS3-001, plus
`PDMX_PIPELINE_CONTRACT.md`. Source voice identity is not semantic voice role.

There is **no production annotation importer** for this sidecar in this step.
AN part scope loss is a specific compatibility blocker for reconstruction from
canonical notes alone. Original source rows plus verified row bindings are the
candidate route; neither their coverage nor musical voice semantics has been
measured here. DLC scalar staff/voice makes a potential binding possible, but
is not enough to assert clean, complete supervision without the same audit.

## CPU verification and next separate gate

Fixtures hand-author two voices crossing pitch rank, multiple voices in one
track, same-number voices in separate source scopes and one voice crossing
tracks. Tests also cover exact fractional timing, sustained overlap, raw-only
candidate invariance under metadata removal/permutation, missing annotation,
same-onset ambiguity, cap/horizon misses, unknown confidence, piece boundaries,
provenance, heuristic-label rejection, forged topology and empty pieces.

Command (existing dependency environment, no installation):

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/music-critic-pop-eval-20260908/bin/python -m pytest \
  tests/data/test_voice_continuation.py -q
```

Initial result: **29 passed in 0.40s**. Expanded dependency-light canonical
regression command used the same environment and
`-m pytest tests/data/test_voice_continuation.py tests/data/test_schema.py tests/data/test_serialization.py tests/data/test_timing.py tests/data/test_validation.py -q`:
**278 passed in 0.78s**. A broader `-m pytest tests/data -q` attempt stopped at
collection with **1 error, 2 warnings in 0.90s**: existing
`test_validation_membership.py` imports training configuration, and this runtime
lacks optional `hydra` (`ModuleNotFoundError: No module named 'hydra'`). No
dependencies were installed during that attempt.

The complete `tests/data` follow-up passed using the existing declared
dependency overlay, read-only; no dependency installation was performed:

```sh
PYTHONPATH=src:/private/tmp/music-critic-fusion-20260908-deps \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/music-critic-pop-eval-20260908/bin/python -m pytest tests/data -q
```

Final result: **292 passed, 2 warnings in 1.17s**. Both warnings are existing
Torch `torch.jit.script` deprecation warnings. The prior `hydra` collection
block is resolved by the existing overlay. The full data-suite log is retained
at `outputs/ssl_group/verification/data.log`; this does not claim a full-project
test-suite pass.

These are ordinary synthetic contract tests, not a substantial experiment or
evidence that voice learning improves the model. No standalone `EXP-VOICE-*`
record is warranted for these tests. No legacy files or logic were inspected,
copied or reused.

Next voice work is a separately registered bounded `EXP-VOICE-*` TRAIN source
annotation audit through the coordinator's resource queue: pin approved source
release/files/hashes, split and row-to-canonical binding; establish scope,
source semantics, missingness and tie-repair exclusions; report annotation
coverage, ambiguous onsets, candidate recall/misses and confidence availability
per source. No executable real-data command is claimed here because its source
manifest and importer are not yet bound. No GPU is needed for that audit.
Voice edges/heads remain excluded from the first group-SSL GPU package.
