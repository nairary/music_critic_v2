# HookTheory Adapter Migration Contract

Status: **ACCEPTED FOR FUTURE PHASE 2B IMPLEMENTATION**.

This document records the migration contract reverse-engineered from the legacy
HookTheory pipeline at commit
`2d8281f31cc9ad9c8fecaf332da0c61e0e949415`. The legacy source is the primary
executable specification because the selected source format has no complete
official field contract. Phase 2A.1 documents these rules but does not implement
a HookTheory adapter.

## Audited legacy sources

- `src/data/preprocess_hooktheory.py`
- `src/data/canonicalize_hooktheory.py`
- `src/data/build_preprocess_song_timelines.py`
- `src/data/encode_teacher_features.py`
- `src/data/render_encoded_song_to_midi.py`
- `src/dataloader/theory_helpers.py`
- `tests/test_canonicalize_hooktheory.py`
- `docs/hooktheory_processed.txt`
- `docs/hooktheory_selected_field_types_documentation.txt`
- `docs/FIELDS_DECODE.txt`

The documentation files currently appear under `docs/music_critic_v1/` in the
legacy worktree because of pre-existing, uncommitted moves. That external state
was inspected read-only and was not changed.

## Source ingestion and identity

The main symbolic payload is `record["json"]`. A record without a dictionary in
that field is invalid for symbolic conversion. The raw top-level dump may be a
JSON object or a JSON fragment without its outer braces; the future reader must
support the legacy wrapping behavior while preserving parse diagnostics.

The source split is normalized case-insensitively after trimming, with `valid`
mapped to `val`. HookTheoryStructure rows are joined within the corresponding
split using the stem of `audio_path` as the symbolic clip identifier.

`ori_uid` is the canonical `source_group_id`. Every clip sharing an `ori_uid`
must remain in one train, validation, or test partition. A missing `ori_uid`
must be diagnosed and must not be replaced with a value that could allow clips
from one original song to leak across splits.

## Coordinate systems

HookTheory symbolic beat coordinates are 1-based. For every symbolic event, V2
onset is computed exactly as:

```text
canonical_onset_qn = exact(raw_beat) - 1
```

The conversion must use exact decimal/rational parsing. It must not round
through binary floating point or use epsilon comparisons. Melody, chord, key,
tempo, and meter coordinates are symbolic beats.

HookTheoryStructure section coordinates (`segment_start`, `segment_end`, and
`duration`) are audio seconds. They are a distinct coordinate system. Section
seconds must not be converted directly to beat, bar, phrase, or section targets
without a separately specified and tested audio-to-symbolic alignment
procedure. Raw coordinates and structured alignment diagnostics must be
preserved.

Multiple key, tempo, and meter regions are meaningful and must not be collapsed
to the first region. The future adapter must preserve all valid regions in
canonical order.

## Melody records and derived pitch

Selected melody-note records contain:

- `sd`;
- `octave`;
- `beat`;
- `duration`;
- `isRest` (normalized by the legacy pipeline to `is_rest`).

This representation contains no directly observed absolute MIDI pitch. A
non-rest pitch is reconstructed algorithmically from the active tonic and the
legacy scale-degree chromatic table:

```text
midi_pitch =
    72
    + 12 * hooktheory_octave
    + tonic_pitch_class
    + scale_degree_chromatic_offset
```

The anchor `72` is part of the accepted migration rule. The future adapter must
not invent another octave anchor without a concrete real-data counterexample
and a recorded decision. The reconstructed MIDI pitch is derived, not directly
observed, and its provenance method must be exactly:

```text
hooktheory_sd_octave_to_midi_v1
```

Applied harmony does not participate in this reconstruction. If required
inputs are missing or the reconstructed pitch is outside `0..127`, the adapter
must emit a structured diagnostic and omit the note. It must never clamp the
pitch. Rest records do not create notes.

Raw scale-degree values must remain available in provenance or diagnostics.
Legacy encoded fields such as `sd_id` and `octave_id` are model-era encodings,
not raw V2 features and not authoritative source values.

## Chord normalization

Raw roots have these accepted semantics:

| Raw value | Canonical interpretation |
|---|---|
| `1..7` | functional degrees `0..6` |
| `8` | special `bVII` representation |
| `0` | rest/empty marker; never tonic |

The raw root must also respect `isRest`; a rest chord has no functional root.
Chord type values are `5`, `7`, `9`, `11`, and `13`, representing tertian
extent in the legacy format.

The future adapter must preserve inversion, adds, omits, alterations,
suspensions, borrowed information, and the raw `alternate` value. It must keep
raw values even when normalization fails.

`borrowed` is heterogeneous. Confirmed forms include:

- `null` or an empty string;
- a mode-name string;
- a pitch-class list;
- a stringified list;
- an unknown value.

Unknown forms produce diagnostics; they must not be silently coerced into a
known mode.

## Applied harmony decision

Applied harmony is explicitly deferred from the first HookTheory adapter:

- it is not implemented;
- it does not participate in melody-pitch reconstruction;
- it does not participate in chord pitch-class reconstruction;
- it is not a raw V2 feature;
- it is not a supervised target in the first adapter;
- its raw occurrence may be preserved only in diagnostics or provenance for
  possible later work.

Any later applied-harmony implementation requires its own verified semantic
contract, fixtures, and recorded architectural decision.

## Canonical target and provenance policy

HookTheory theory values are auxiliary annotations/targets, never mandatory raw
encoder inputs. Missing labels use masks and are not negative labels. Raw
values, conversion methods, derived-value status, confidence, and structured
diagnostics must be preserved.

Legacy `sd_id`, `root_id`, `type_id`, `inversion_id`, `applied_id`, borrowed
IDs, and similar encoded IDs are not raw V2 features. The future adapter starts
from pre-encoding values whenever possible and must not expose those IDs as
ordinary MIDI-observable evidence.

## Unresolved issues

The following remain open for the Phase 2B.0 audit and golden-fixture slice:

- exact HookTheory `beatUnit` semantics;
- mapping from `beatUnit` to canonical meter denominator;
- `alternate` semantics;
- `pedal` semantics;
- reliable alignment from audio-section seconds to symbolic clip beats.

Unresolved fields must remain raw and diagnostic. They must not be guessed
silently.

## Phase 2B implementation gate

Before adapter implementation begins, Phase 2B.0 must establish bounded real
examples and golden fixtures covering coordinate origin, pitch reconstruction,
root `0` and `8`, borrowed variants, multiple regions, split/group identity,
and malformed values. Phase 2A.1 does not satisfy that gate and contains no
HookTheory production code.
