# Music Critic V2 Data Contract

Status: **PROPOSED**. No schema classes or serializers are implemented in
Phase 0.

## Fixed decisions

- Canonical musical timing is exact.
- Canonical time is measured in quarter-note-relative units.
- JSON rationals use `{"num": integer, "den": positive_integer}`.
- `split` is optional for inference pieces.
- Raw track roles are not assumed known.
- Supervised track roles live in targets.
- Theory labels are not mandatory raw features.
- Missing labels are unavailable targets, not negative labels.
- Negative durations are invalid.
- Zero duration is valid only for grace notes.
- Provenance and confidence are retained.
- Every aligned target includes explicit entity IDs.
- The initial proposed schema version is `2.0.0`, pending Phase 1.

## Input and target separation

### Raw observations

Raw observations are values present in ordinary MIDI or deterministically
derived from it: note pitch, onset, duration, velocity, channel/program,
percussion flag, track identity, tempo/meter events, bars, beats, onsets, and
raw-derived statistics.

### Optional observations

Notation-only metadata such as spelling, staff, voice, articulation, dynamics,
track names, and key signatures is optional. Absence must be explicit and the
public inference API cannot require it.

### Targets

Theory, segmentation, preference, and quality annotations are targets. They
must not become mandatory encoder features or mandatory graph structure.

## Proposed rational representation

```json
{
  "onset_qn": {"num": 3, "den": 2},
  "duration_qn": {"num": 1, "den": 4}
}
```

Denominators must be positive. Values should be normalized during Phase 1
serialization.

## Proposed TargetArray

```text
task
alignment_type
entity_ids
values
mask
confidence
source
provenance
```

- `task`: stable task identifier;
- `alignment_type`: note, track, beat, bar boundary, piece, or another declared
  entity level;
- `entity_ids`: explicit canonical IDs;
- `values`: task values;
- `mask`: availability for each value;
- `confidence`: per-value confidence;
- `source`: human, dataset, inferred, pseudo-label, or another declared source;
- `provenance`: source-specific trace information.

All aligned arrays must have compatible lengths. Values with `mask=false` are
unavailable and must not contribute to supervised loss.

## Proposed top-level responsibilities

A future canonical piece will preserve identity, dataset/source grouping,
optional split, source resolution, metadata, exact timing events, tracks,
notes, bars, beats, annotations, target arrays, provenance, and quality flags.

Phase 1 will define the exact Python API, validation behavior, and serialized
field set. Until then, every section in this document remains `PROPOSED`.
