# HookTheory Adapter Migration Contract

Status: **ACCEPTED**. Phase 2B.0 and Phase 2B.1 are completed. Accepted Phase
2B.0 implementation: `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`.
Accepted Phase 2B.1 implementation:
`3898b168063094b87e5ca5d88aae0317c1562c3f`.

The real-data inventory, runtime domains, hashes, join statistics, leakage
findings, and bounded golden evidence are recorded in
[`HOOKTHEORY_FIELD_AUDIT.md`](HOOKTHEORY_FIELD_AUDIT.md). That Phase 2B.0 audit
is the evidence source when it is more specific than this migration summary.

This accepted contract separates three evidence sources: observed m-a-p
artifacts; upstream Sheet Sage TheoryTab at commit
`bbdd7b7b6a5fb845828f82790acdceb03a197779`; and the legacy Music Critic pipeline
at commit `2d8281f31cc9ad9c8fecaf332da0c61e0e949415` where explicit V1
compatibility is intended. Inferred/project decisions and unresolved items are
labeled separately. No HookTheory adapter is implemented here.

## Evidence hierarchy

Claims are classified as observed corpus, upstream Sheet Sage, Music Critic V1
compatibility, inferred/project decision, or unresolved. The pinned upstream
files inspected are `sheetsage/theory/theorytab.py`, `theorytab_test.py`,
`lead_sheet.py`, and `internal.py`. A V1 behavior is never presented as a corpus
observation or upstream invariant.

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

The split unit is atomic: all clips with one non-null `ori_uid` move together
or are excluded together. A policy must never select individual clips from a
cross-split group. Null-`ori_uid` clips remain individually identified; the
adapter must not fabricate group identity.

`data/HookTheory/Hooktheory.json` is a crosswalk source, classified as the
upstream Sheet Sage simplified alternate schema. Its meter and melody
representations are semantically crosswalked against the raw TheoryTab record;
key and harmony remain inventoried for availability and shape only. None of
these fields silently replaces raw data. The audit found 26,175
matches, three raw-only missing-payload records, no simplified-only records,
and no split or nested-identifier mismatches.

## Coordinate systems

HookTheory symbolic beat coordinates are 1-based source coordinates. They are
converted exactly through a piecewise meter timeline:

```text
raw beat 1 = canonical qn 0
qn_per_raw_beat = 1 when beatUnit=1
qn_per_raw_beat = 1/2 when beatUnit=3
```

The compound result follows Sheet Sage's three coordinate levels. In
`LeadSheet.from_theorytab()`, one raw TheoryTab beat becomes four tertiary
subdivision units. A simple primary pulse contains `2 * 2 = 4` tertiary units,
so one raw beat is one primary quarter-note pulse. A compound primary pulse
contains `3 * 4 = 12` tertiary units, so one raw beat is `4/12` of a dotted
quarter pulse: one eighth note, or one-half qn.

Simplified HookTheory melody `onset`/`offset`, meter `beat`, and alignment beat
coordinates remain zero-based source-beat coordinates. They support the
raw-to-simplified identity check `raw beat - 1 == simplified beat`, but they do
not become canonical qn merely because one was subtracted. Canonical qn still
requires the active meter conversion above.

The conversion must use exact decimal/rational parsing. It must not round
through binary floating point or use epsilon comparisons. Melody, chord, key,
tempo, and meter coordinates are symbolic beats.

HookTheoryStructure section coordinates (`segment_start`, `segment_end`, and
`duration`) are audio seconds. They are a distinct coordinate system. Section
seconds must not be converted directly to beat, bar, phrase, or section targets
without a separately specified and tested audio-to-symbolic alignment
procedure. Raw coordinates and structured alignment diagnostics must be
preserved.

Both ends of every duration and `endBeat` use the same piecewise mapping;
durations cannot be scaled only by the meter at onset. Multiple key, tempo, and meter regions are meaningful and must not be collapsed
to the first region. The production adapter preserves all valid regions in
canonical order.

## Meter mapping

The raw-to-simplified corpus comparison covers all 26,175 matched records. It
compares `raw beat - 1` with simplified `beat`, `raw numBeats` with simplified
`beats_per_bar`, and raw `beatUnit` values `1` and `3` with simplified
`beat_unit` values `4` and `8`, respectively.

All 27,216 paired regions match exactly. One additional raw region is absent
from the simplified record for clip `nvgy-WaRgkA`, producing one count mismatch
and one missing-simplified region but no value mismatch. This is simplified
coverage loss, not contradictory mapping evidence. The accepted canonical
mapping is therefore:

```text
numerator = raw numBeats
denominator = 4 if raw beatUnit == 1 else 8
```

The adapter must still preserve and diagnose invalid or unsupported raw meter
values rather than applying this expression blindly outside the audited domain.

## Tempo mapping

Three formulas were audited: quarter-note BPM (`60_000_000/bpm`), raw-beat BPM
(`60_000_000/(bpm*qn_per_raw_beat)`), and compound felt-pulse BPM
(`40_000_000/bpm`). They coincide in simple meter. No refined compound-meter
interval exists; across 72 eligible user-alignment intervals the respective
median absolute relative errors are 50.04%, 200.07%, and 0.39%. Pinned Sheet
Sage also defines compound primary pulses as groups of three source beats.
Production therefore uses quarter-note BPM for `beatUnit=1` and felt-pulse BPM
for `beatUnit=3`, classified as upstream semantics with corpus alignment
support. A tempo at the exact onset of a meter change uses the new meter.

## Melody records and derived pitch

Selected melody-note records contain:

- `sd`;
- `octave`;
- `beat`;
- `duration`;
- `isRest` (normalized by the legacy pipeline to `is_rest`).

This representation contains no directly observed absolute MIDI pitch. A
non-rest pitch is reconstructed from the exact active key using pinned Sheet
Sage scale intervals and accidentals:

```text
midi_pitch =
    60
    + 12 * hooktheory_octave
    + tonic_pitch_class
    + active_scale_degree_offset
    + accidental_offset
```

The MIDI-60 anchor is upstream semantics from Sheet Sage
`Note.as_midi_pitch()` with default `octave_0=4`. MIDI 72 remains only a
documented V1 compatibility convention. The production provenance method is:

```text
hooktheory_scale_degree_to_midi_upstream
```

Applied harmony does not participate in this reconstruction. If required
inputs are missing or the reconstructed pitch is outside `0..127`, the adapter
must emit a structured diagnostic and omit the note. It must never clamp the
pitch. Rest records do not create notes.

Raw scale-degree values must remain available in provenance or diagnostics.
Legacy encoded fields such as `sd_id` and `octave_id` are model-era encodings,
not raw V2 features and not authoritative source values.

## Chord normalization

Raw roots have this accepted normalization:

| Raw value | Canonical interpretation |
|---|---|
| `1..7` | functional degrees `0..6` |
| `8` | synthetic V1 compatibility mapping to `bVII`; not observed in corpus |
| `0` | rest/empty marker; never tonic |

The raw root must also respect `isRest`; a rest chord has no functional root.
Chord type values are `5`, `7`, `9`, `11`, and `13`, representing tertian
extent in the legacy format.

The production adapter preserves inversion, adds, omits, alterations,
suspensions, borrowed information, and the raw `alternate` value. It must keep
raw values even when normalization fails.

`borrowed` is heterogeneous. Observed corpus forms include:

- `null` or an empty string;
- a mode-name string;
- a pitch-class list;
- an unknown value.

Stringified lists are supported by V1 compatibility logic but were not observed
in the audited corpus; unexpected runtime types were also not observed. Unknown
forms produce diagnostics and must not be silently coerced into a known mode.

## Applied harmony decision

Applied harmony is partially available upstream: Sheet Sage can reinterpret a
chord relative to an applied target. It is nevertheless explicitly deferred
from the first HookTheory adapter:

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
IDs, and similar encoded IDs are not raw V2 features. The production adapter
starts from pre-encoding values and does not expose those IDs as ordinary
MIDI-observable evidence.

## Future harmonic-supervision use

The accepted migration and Phase 2 results above are unchanged. For later
dataset/model phases, HookTheory chord annotations are direct source
annotations that may supervise the same auxiliary harmonic families as
POP909-CL where semantics align. The complete cross-dataset contract is in
[`HARMONIC_SUPERVISION.md`](HARMONIC_SUPERVISION.md).

The current raw observation remains the adapter's isolated melody. Future
derived harmonic targets may include chord boundary/presence, root, quality, a
12-dimensional pitch-class set, bass/inversion when available, span/duration,
and supported decorations or functional fields. They require dataset-specific
availability masks, annotation views, and per-target provenance. One annotated
chord sequence is one observed melody-conditioned harmonization; single-label
cross-entropy must not be interpreted as evidence that no other harmonization
is valid.

Deriving a pitch-class set or target-only diagnostic rendering is not by itself
leakage. Any derived value must use source `derived`, reference the direct
HookTheory annotation, and record the deterministic normalizer/renderer
version. A realization such as `C major -> C3 E3 G3` adds project-selected
voicing, octave, register, doubling, order, rhythm, and duration. It is not
actual performed/score accompaniment and is not independent human ground
truth.

No target-derived notes may enter raw canonical tracks/notes, graph
features/topology, serialization, fingerprints, caches, or inference, even if
track names or roles are removed. This documentation change adds no renderer,
new target arrays, adapter behavior, or model code.

## Unresolved issues

The Phase 2B.0 audit resolves the canonical meter fraction over the audited
domain. It also confirms that `alternate` is either empty or `_`, and that
every audited `pedal` is null. The following remain open for adapter
implementation:

- `alternate` semantics;
- `pedal` semantics;
- reliable alignment from audio-section seconds to symbolic clip beats.

Real-data categories not observed in the audited source are raw root `8`, a
stringified borrowed pitch-class list, an unexpected borrowed runtime type, a
derived out-of-range melody pitch, non-null `pedal`, exact duplicate regions,
duplicate structure clip IDs, unmatched structure rows, and missing structure
`ori_uid`. The legacy root-8-to-bVII rule remains only synthetic compatibility
behavior, has no real-data golden case, and must not be represented as observed
or upstream-supported.

Unresolved fields must remain raw and diagnostic. They must not be guessed
silently.

## Phase 2B implementation gate

Phase 2B.0 establishes bounded real examples for every observed category listed
in the field audit; root `8` is covered only by a synthetic compatibility unit
test and an explicit corpus-wide zero count. Review accepted this contract and
Phase 2B.1 may now implement it on its dedicated branch.

## Phase 2B.1 production implementation

Status: **Accepted and Completed**. Accepted implementation:
`3898b168063094b87e5ca5d88aae0317c1562c3f`.
Closure: `6111d3d062e02897e3f8ebdca7e4388f80ef434e`; merged to `main` by
`b1df77737f641b705e3c48724b2741c7a022a2e4`.

`music_critic.adapters.hooktheory` exposes `HookTheoryAdapterConfig`,
`HookTheoryAdapterError`, `convert_hooktheory_record`, and
`load_hooktheory_piece`. It consumes only the raw merged m-a-p record plus an
optional structure row. Production conversion does not read Hooktheory.json,
HTCanon, Sheet Sage, or legacy modules.

Exact piecewise raw timing, scale-aware MIDI-60 pitch, simple-quarter versus
compound-felt-pulse BPM conversion, accepted meter mapping, metric grids,
grouping, provenance, and diagnostics are canonical raw content. A supplied
structure row is validated by clip stem and compatible split before it may
affect grouping. Theory annotations remain confined
to these target tasks:

- `theory.melody.scale_degree`;
- `theory.local_key.tonic_pc`;
- `theory.local_key.mode`;
- `theory.chord.presence`;
- `theory.chord.root_degree`;
- `theory.chord.extent`;
- `theory.chord.inversion`;
- `theory.chord.adds`;
- `theory.chord.omits`;
- `theory.chord.alterations`;
- `theory.chord.suspensions`;
- `theory.chord.borrowed`.

Target hiding returns identical non-target canonical content and diagnostics,
with empty annotations and targets and no annotation-only provenance. Structure
seconds remain unaligned and create no section spans or targets. Applied,
alternate, and pedal values remain diagnostic-only.

The read-only corpus smoke converted all 26,175 usable records into valid
canonical pieces with zero unexpected failures; it skipped exactly the three
known missing-payload records. A 32-clip deterministic spread passed exact JSON
round trips and target-visible/hidden equivalence.
