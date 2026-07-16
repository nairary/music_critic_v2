# Music Critic V2 Data Contract

Status: **ACCEPTED FOR PHASE 1 IMPLEMENTATION**.

This document is the normative Phase 1 contract for the future
`music_critic.data` schema layer. It defines the public Python API and canonical
JSON representation. Implementations must not add, remove, rename, or reinterpret
public fields without a schema-version decision recorded in `docs/DECISIONS.md`.

The schema layer uses only the Python standard library. In particular, it has no
runtime dependency on NumPy, PyTorch, PyG, MIDI libraries, Hydra, or the legacy
repository.

## 1. Fixed constants and public modules

```python
SCHEMA_VERSION = "2.0.0"
```

The future public modules are:

```text
music_critic.data.timing
music_critic.data.schema
music_critic.data.validation
music_critic.data.serialization
```

No production modules are created by Phase 1A.

### `music_critic.data.timing`

```python
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from typing import Any, Mapping


@total_ordering
@dataclass(frozen=True, slots=True)
class RationalTime:
    num: int
    den: int = 1

    def __post_init__(self) -> None: ...
    def __lt__(self, other: RationalTime) -> bool: ...
    def __add__(self, other: RationalTime) -> RationalTime: ...
    def __sub__(self, other: RationalTime) -> RationalTime: ...
    def __neg__(self) -> RationalTime: ...
    def __mul__(self, factor: int) -> RationalTime: ...
    def __truediv__(self, divisor: int) -> RationalTime: ...
    def to_fraction(self) -> Fraction: ...

    @classmethod
    def from_fraction(cls, value: Fraction) -> RationalTime: ...
```

Construction rejects booleans, non-integer numerators or denominators, and
non-positive denominators. Construction normalizes by the greatest common
divisor, keeps the denominator positive, and canonicalizes zero as `0/1`.
Comparison is numeric by cross multiplication; dataclass tuple ordering must not
be used.

`RationalTime` is measured in quarter-note units (`qn`). It is not seconds,
ticks, or a meter-relative beat number.

### `music_critic.data.schema`

The module exports `SCHEMA_VERSION`, the aliases below, and these dataclasses:

```text
PieceMetadata
CanonicalTrack
CanonicalNote
CanonicalBar
CanonicalBeat
TempoEvent
MeterEvent
KeySignatureEvent
AnnotationSpan
TargetArray
ProvenanceRecord
QualityFlag
CanonicalPiece
```

`RationalTime` is imported from `music_critic.data.timing`.

```python
from typing import Literal, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
ProvenanceDetail: TypeAlias = tuple[str, JsonScalar]

Split: TypeAlias = str | None
SourceFormat: TypeAlias = Literal[
    "midi", "musicxml", "json", "jsonl", "tsv", "synthetic", "other"
]
KeySignatureMode: TypeAlias = Literal[
    "major",
    "minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "locrian",
    "other",
    "unknown",
]
AnnotationLayer: TypeAlias = Literal["observation", "target_alignment"]
AlignmentType: TypeAlias = Literal[
    "piece",
    "track",
    "note",
    "bar",
    "beat",
    "bar_boundary",
    "beat_boundary",
    "annotation_span",
]
TargetValueType: TypeAlias = Literal[
    "categorical", "scalar", "multi_label", "distribution"
]
TargetSource: TypeAlias = Literal[
    "human",
    "dataset",
    "algorithm",
    "pseudo_label",
    "derived",
    "synthetic",
]
TargetValue: TypeAlias = (
    str
    | int
    | float
    | tuple[str, ...]
    | tuple[float, ...]
)
ProvenanceKind: TypeAlias = Literal[
    "source", "conversion", "annotation", "derivation", "default", "synthetic"
]
IssueSeverity: TypeAlias = Literal["error", "warning"]
QualitySeverity: TypeAlias = Literal["info", "warning"]
QualityFlagCode: TypeAlias = str
```

Open stable identifiers include `dataset_name`, `source_group_id`, `task`,
`annotation_type`, and provenance `source`. Unless a more specific syntax is
declared, each must:

- be a string;
- be non-empty after `strip()`;
- already equal its stripped form;
- contain no ASCII control character in `U+0000..U+001F` or `U+007F`;
- be compared case-sensitively.

Task identifiers should be dotted namespaces, for example
`theory.chord_quality` or `track.role`. Entity IDs and quality-flag codes use
their own stricter syntax.

### `music_critic.data.validation`

The module exports `ValidationIssue`, `ValidationReport`,
`CanonicalValidationError`, `validate_piece`, and `validate_or_raise`.

```python
def validate_piece(piece: CanonicalPiece) -> ValidationReport: ...

def validate_or_raise(piece: CanonicalPiece) -> None: ...
```

`validate_piece` never raises for a well-formed Python `CanonicalPiece`; it
returns all detected validation issues. `validate_or_raise` raises
`CanonicalValidationError` containing that same report when at least one error
exists. Warnings do not cause an exception. Validation applies the complete
semantic-value rules to programmatically constructed dataclasses as well as
records produced by JSON decoding.

### `music_critic.data.serialization`

```python
from os import PathLike
from typing import Any, Mapping

JsonObject = dict[str, Any]


def piece_to_dict(piece: CanonicalPiece) -> JsonObject: ...
def piece_from_dict(data: Mapping[str, Any]) -> CanonicalPiece: ...

def dumps_piece(
    piece: CanonicalPiece,
    *,
    indent: int | None = None,
) -> str: ...

def loads_piece(payload: str | bytes | bytearray) -> CanonicalPiece: ...

def dump_piece(
    piece: CanonicalPiece,
    path: str | PathLike[str],
    *,
    indent: int | None = 2,
) -> None: ...

def load_piece(path: str | PathLike[str]) -> CanonicalPiece: ...
```

Serialization is field-by-field and explicit. `dataclasses.asdict()` may be an
internal aid but is not the public mapping contract and cannot determine field
names, ordering, rational encoding, tuple conversion, validation, or version
behavior.

`piece_to_dict`, `dumps_piece`, and `dump_piece` call `validate_or_raise` before
emitting data. JSON syntax errors use `json.JSONDecodeError`. JSON shape,
unknown-field, missing-field, unsupported-version, and semantic schema errors
use `CanonicalValidationError` with a `ValidationReport`.
`piece_from_dict`, `loads_piece`, and `load_piece` perform strict shape/version
decoding, construct the immutable records, call `validate_or_raise`, and return
only a valid piece. `validate_piece` remains available for programmatically
constructed records and adapter diagnostics before serialization.

## 2. Immutability and collection policy

All schema records are `@dataclass(frozen=True, slots=True)`.

All collection-valued fields in the immutable Python model are tuples. JSON
uses arrays for those tuples. JSON objects used for provenance details are
represented in Python as sorted tuples of `(key, JsonScalar)` pairs, preventing
mutable dictionaries from being embedded in frozen records.

The Python model is therefore immutable down to its supported field values.
Adapters may use mutable builder objects internally, but must freeze and
canonically order all data before returning a `CanonicalPiece`.

Optional fields are not omitted from canonical JSON:

- `null` means unavailable, not observed, or not supplied;
- `""` means an observed empty string;
- `[]` means an observed empty collection;
- an absent key is invalid canonical JSON.

This distinction applies equally to Python: `None` is unavailable, while an
empty tuple or empty string is an available empty value.

The required/optional split is:

| Record | Required non-null fields | Optional nullable fields |
|---|---|---|
| `PieceMetadata` | `source_format` | all remaining fields |
| `CanonicalTrack` | `track_id`, `is_percussion` | all remaining fields |
| `CanonicalNote` | `note_id`, `track_id`, `pitch`, `onset_qn`, `duration_qn`, `is_percussion`, `is_grace` | all remaining fields |
| `CanonicalBar` | all except `display_number`, `provenance_id` | `display_number`, `provenance_id` |
| `CanonicalBeat` | all except `strength`, `provenance_id` | `strength`, `provenance_id` |
| `TempoEvent` | all except `provenance_id` | `provenance_id` |
| `MeterEvent` | all except `provenance_id` | `provenance_id` |
| `KeySignatureEvent` | all except `raw_value`, `provenance_id` | `raw_value`, `provenance_id` |
| `AnnotationSpan` | `annotation_id`, `annotation_type`, `layer`, `start_qn`, `end_qn` | `track_id`, `value`, `provenance_id` |
| `TargetArray` | every top-level field except `annotation_view_id`; entry nullability is controlled by `mask` | `annotation_view_id` may be null; `class_labels` may be null by value type; aligned entries may be null only as specified below |
| `ProvenanceRecord` | `provenance_id`, `kind`, `source`, `parents`, `details` | `record_id`, `uri`, `version`, `checksum_sha256`, `created_at` |
| `QualityFlag` | all except `provenance_id` | `provenance_id` |
| `CanonicalPiece` | every top-level field key | `split`, `source_path`, `source_resolution` may be null |

Collection fields never use `null`. They use tuples in Python and arrays in
JSON. Empty collections are valid unless a specific invariant below requires
content. `tempo_events`, `meter_events`, and `provenance` must each be non-empty.

## 3. Exact dataclass definitions

Every field shown below is present in the Python dataclass and serialized JSON.
Types containing `None` are optional observations but remain required JSON
keys. No additional public fields are part of schema `2.0.0`.

### `PieceMetadata`

```python
@dataclass(frozen=True, slots=True)
class PieceMetadata:
    source_format: SourceFormat
    title: str | None
    creators: tuple[str, ...] | None
    collection: str | None
    movement_title: str | None
    movement_number: str | None
    genres: tuple[str, ...] | None
    copyright: str | None
    language: str | None
```

`source_format` is required and non-null. All other values are optional
observations. Theory labels, semantic roles, local keys, chords, cadences,
phrases, and section functions are forbidden in metadata.

### `CanonicalTrack`

```python
@dataclass(frozen=True, slots=True)
class CanonicalTrack:
    track_id: str
    source_track_index: int | None
    name: str | None
    instrument_name: str | None
    program: int | None
    channel: int | None
    is_percussion: bool
    provenance_id: str | None
```

`track_id` and `is_percussion` are required observations. MIDI-like `program`
is in `[0, 127]` when available and `channel` is in `[0, 15]` when available.
`source_track_index` is zero-based and non-negative when available. It is not
the entity ID and may be shared when one source track is deterministically split
into pitched and percussion canonical tracks.

No role field exists. Melody, bass, accompaniment, drums, and other semantic
roles belong only in `TargetArray`.

### `CanonicalNote`

```python
@dataclass(frozen=True, slots=True)
class CanonicalNote:
    note_id: str
    track_id: str
    pitch: int
    onset_qn: RationalTime
    duration_qn: RationalTime
    velocity: int | None
    channel: int | None
    program: int | None
    is_percussion: bool
    is_grace: bool
    spelling_step: str | None
    spelling_alter: int | None
    staff: int | None
    voice: int | None
    articulations: tuple[str, ...] | None
    dynamic: str | None
    source_onset_ticks: int | None
    source_duration_ticks: int | None
    source_onset_seconds: float | None
    source_duration_seconds: float | None
    provenance_id: str | None
```

`pitch` is the preserved MIDI-compatible pitch number in `[0, 127]`, including
percussion note numbers. `velocity`, when available, is in `[0, 127]`.
`spelling_step` is one of `A` through `G` when available. `spelling_alter` is an
integer semitone alteration and is meaningful only when spelling is available.
`staff` and `voice` are non-negative source identifiers when available.

Schema `2.0.0` supports integer-semitone pitch spelling alterations only.
Quarter-tone and other microtonal spelling alterations are outside this
contract. An adapter encountering unsupported source notation must preserve the
original representation in provenance details, emit an appropriate namespaced
quality flag, and leave `spelling_alter=None`; it must not silently round the
alteration. `spelling_alter` must be `None` when `spelling_step` is `None`.

Theory fields such as scale degree, chord membership, Roman numeral, local key,
non-chord-tone class, or voice function are forbidden.

### `CanonicalBar`

```python
@dataclass(frozen=True, slots=True)
class CanonicalBar:
    bar_id: str
    index: int
    start_qn: RationalTime
    duration_qn: RationalTime
    meter_event_id: str
    metric_offset_qn: RationalTime
    is_pickup: bool
    is_incomplete: bool
    display_number: str | None
    provenance_id: str | None
```

`index` is the zero-based chronological bar ordinal, regardless of a displayed
measure number. `metric_offset_qn` is the position inside the nominal meter at
which the represented bar begins. It is zero for ordinary bars.

For a one-quarter-note pickup in 4/4:

```text
start_qn          = 0/1
duration_qn       = 1/1
metric_offset_qn  = 3/1
is_pickup         = true
is_incomplete     = true
```

The first full bar then starts at `1/1` with index `1`. Canonical time never
uses negative pickup coordinates. A shortened final bar has
`is_incomplete=true`, `is_pickup=false`, and normally a zero metric offset.

### `CanonicalBeat`

```python
@dataclass(frozen=True, slots=True)
class CanonicalBeat:
    beat_id: str
    bar_id: str
    meter_event_id: str
    index_in_bar: int
    start_qn: RationalTime
    duration_qn: RationalTime
    position_in_bar_qn: RationalTime
    is_downbeat: bool
    strength: float | None
    provenance_id: str | None
```

Canonical beats are denominator-unit meter positions. In `6/8`, the canonical
grid has six half-quarter-note positions; compound-beat grouping is a later
derived graph feature. `position_in_bar_qn` includes the bar's metric offset, so
the sole beat of the 4/4 pickup above has `index_in_bar=3` and
`position_in_bar_qn=3/1`. `strength`, when supplied, is finite and in `[0, 1]`.

### `TempoEvent`

```python
@dataclass(frozen=True, slots=True)
class TempoEvent:
    tempo_event_id: str
    onset_qn: RationalTime
    microseconds_per_quarter: int
    provenance_id: str | None
```

Tempo is stored exactly as a positive integer number of microseconds per
quarter note. BPM is derived as `60_000_000 / microseconds_per_quarter` and is
not serialized. Tempo changes take effect at their onset and do not alter
quarter-note positions.

### `MeterEvent`

```python
@dataclass(frozen=True, slots=True)
class MeterEvent:
    meter_event_id: str
    onset_qn: RationalTime
    numerator: int
    denominator: int
    provenance_id: str | None
```

The numerator is positive. The denominator is a positive power of two. A meter
event takes effect at its onset, which must be the start of a canonical bar.
The nominal bar duration is `numerator * 4 / denominator` quarter notes.

### `KeySignatureEvent`

```python
@dataclass(frozen=True, slots=True)
class KeySignatureEvent:
    key_signature_event_id: str
    onset_qn: RationalTime
    fifths: int
    mode: KeySignatureMode
    raw_value: str | None
    provenance_id: str | None
```

`fifths` is in `[-7, 7]`. A key signature is optional observable notation or
MIDI metadata, not a local-key analysis and not guaranteed to describe sounding
tonality. Local key remains a target. Observed modal source information is
preserved using the declared mode values. Unsupported or source-specific modes
use `"other"` while retaining the original notation in `raw_value`;
`"unknown"` means that a key-signature event was observed but its mode was not
available. When `mode="other"`, `raw_value` must be non-null and non-empty after
stripping. Pieces without an observed key signature use an empty
`key_signature_events` tuple.

### `AnnotationSpan`

```python
@dataclass(frozen=True, slots=True)
class AnnotationSpan:
    annotation_id: str
    annotation_type: str
    layer: AnnotationLayer
    start_qn: RationalTime
    end_qn: RationalTime
    track_id: str | None
    value: str | None
    provenance_id: str | None
```

Positive spans use half-open intervals `[start_qn, end_qn)`. Equal start and end
represent a point annotation.

`layer="observation"` is limited to source-observable material. Its
`annotation_type` must begin with one of:

```text
text.
performance.
notation.
other.
```

Examples are `text.lyric`, `text.rehearsal`, and `performance.direction`.
Prefixes `theory.`, `harmony.`, `key.`, `cadence.`, `phrase.`, `section.`, and
`role.` are forbidden for observation spans.

`layer="target_alignment"` creates a stable span entity to which a
`TargetArray` may align; its `annotation_type` is the target task namespace and
its `value` must be `None`. Harmony, tonal region, phrase, cadence, section, and
other theory labels are stored in targets, never in `AnnotationSpan.value`.

### `TargetArray`

```python
@dataclass(frozen=True, slots=True)
class TargetArray:
    target_id: str
    task: str
    annotation_view_id: str | None
    alignment_type: AlignmentType
    entity_ids: tuple[str, ...]
    value_type: TargetValueType
    class_labels: tuple[str, ...] | None
    values: tuple[TargetValue | None, ...]
    mask: tuple[bool, ...]
    confidence: tuple[float | None, ...]
    source: tuple[TargetSource | None, ...]
    provenance: tuple[str | None, ...]
```

The following fields have exactly the same length:

```text
entity_ids
values
mask
confidence
source
provenance
```

For each position:

- `mask=false` means unavailable and requires `value=null`,
  `confidence=null`, `source=null`, and `provenance=null`;
- `mask=true` means an observed target and requires a non-null value,
  a non-null source, and a valid provenance reference;
- for `mask=true`, confidence may be null when the source supplied no numeric
  confidence estimate; otherwise it must be finite and in `[0, 1]`;
- a masked value never means the negative class;
- an actual negative label, such as `no_cadence`, is valid only when explicitly
  stored with `mask=true`.

For an available target, `confidence=null` means only that numeric confidence
is unknown. It does not mean that the target is missing, that confidence is
zero, or that confidence is one.

Target encodings are:

| `value_type` | Python value when available | JSON value | `class_labels` |
|---|---|---|---|
| `categorical` | `str` | string | optional closed vocabulary; if present, values must belong to it |
| `scalar` | finite `int` or `float` | JSON number | must be `null` |
| `multi_label` | `tuple[str, ...]` | array of unique strings in canonical class-label order | required and defines the closed vocabulary |
| `distribution` | `tuple[float, ...]` | array of finite probabilities | required; same length and order as values, each probability in `[0,1]`, sum within `1e-9` of `1.0` |

Boolean targets use categorical labels such as `"false"` and `"true"` rather
than JSON booleans. This prevents confusion between target values and masks.

`alignment_type` determines the required entity prefix:

| Alignment | Referenced IDs |
|---|---|
| `piece` | exactly the containing `piece_id` |
| `track` | `track:*` |
| `note` | `note:*` |
| `bar` | `bar:*` |
| `beat` | `beat:*` |
| `bar_boundary` | `bar:*`; the boundary is that bar's `start_qn` |
| `beat_boundary` | `beat:*`; the boundary is that beat's `start_qn` |
| `annotation_span` | `span:*` with `layer="target_alignment"` |

`target_id` is a globally unique `target:*` entity ID.
`annotation_view_id` is an open stable identifier, not an entity ID, and does
not require an entity prefix. `None` means that the dataset supplies one default
annotation view. A non-null value:

- must be a string;
- must be non-empty after `strip()`;
- must already equal its stripped form;
- must contain no ASCII control character in `U+0000..U+001F` or `U+007F`;
- is case-sensitive.

Valid examples:

```text
dcml
augmentednet
analysis.primary
analysis.alternative
annotator.alice
```

Invalid examples:

```text
""
"   "
" analysis.primary "
"analysis\nprimary"
```

`entity_ids` may select a subset of the aligned collection, but may not contain
duplicates within one target array. Multiple target arrays may share a task
only when their `annotation_view_id` values differ. Uniqueness is enforced on
`(task, annotation_view_id)`. The same aligned entity ID may appear in
different annotation views.

Alternative valid analyses remain separate target arrays. They must not be
collapsed into a `distribution` target unless the source explicitly supplies a
probability distribution.

### `ProvenanceRecord`

```python
@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    provenance_id: str
    kind: ProvenanceKind
    source: str
    record_id: str | None
    uri: str | None
    version: str | None
    checksum_sha256: str | None
    created_at: str | None
    parents: tuple[str, ...]
    details: tuple[ProvenanceDetail, ...]
```

`source` identifies the dataset, person, tool, or conversion stage.
It follows the open stable-identifier rules above.
`created_at`, when present, is an RFC 3339 timestamp in the profile
`YYYY-MM-DDTHH:MM:SS[.fraction](Z|±HH:MM)`, with a valid calendar date/time,
seconds in `[00,59]`, and an explicit offset.
`checksum_sha256`, when present, is 64 lowercase hexadecimal characters.
`parents` reference earlier `prov:*` records and must form an acyclic graph.
`details` keys are non-empty, unique, and lexicographically sorted. Details are
for trace data such as source divisions, adapter options, or an unavailable
field reason; they must not be used to hide theory labels from `TargetArray`.

### `QualityFlag`

```python
@dataclass(frozen=True, slots=True)
class QualityFlag:
    code: QualityFlagCode
    severity: QualitySeverity
    message: str
    entity_ids: tuple[str, ...]
    provenance_id: str | None
```

Quality flags are persisted source/conversion facts. They are not validation
errors and their severity is only `info` or `warning`.

`QualityFlag.code` is an open, stable, lowercase dotted identifier matching:

```text
^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$
```

It therefore contains at least two non-empty lowercase ASCII namespace
components. Examples:

```text
canonical.default_tempo_inserted
canonical.default_meter_inserted
adapter.midi.mixed_percussion_track_split
adapter.pop909.alignment_uncertain
adapter.pdmx.unsupported_event_dropped
```

Adapters may define new stable codes without changing the schema version.
Invalid syntax is a validation error. `ValidationCode` remains a closed
`Literal` because validation behavior is part of the strict schema contract.

### `CanonicalPiece`

```python
@dataclass(frozen=True, slots=True)
class CanonicalPiece:
    schema_version: str
    piece_id: str
    dataset_name: str
    source_group_id: str
    split: str | None
    source_path: str | None
    source_resolution: int | None
    duration_qn: RationalTime
    metadata: PieceMetadata
    tracks: tuple[CanonicalTrack, ...]
    notes: tuple[CanonicalNote, ...]
    bars: tuple[CanonicalBar, ...]
    beats: tuple[CanonicalBeat, ...]
    tempo_events: tuple[TempoEvent, ...]
    meter_events: tuple[MeterEvent, ...]
    key_signature_events: tuple[KeySignatureEvent, ...]
    annotations: tuple[AnnotationSpan, ...]
    targets: tuple[TargetArray, ...]
    provenance: tuple[ProvenanceRecord, ...]
    quality_flags: tuple[QualityFlag, ...]
```

`schema_version` is exactly `SCHEMA_VERSION`. `dataset_name` and
`source_group_id` are required non-empty grouping strings. `split` is exactly
`str | None`; the schema does not constrain split vocabulary. Inference pieces
normally use `None`. `source_resolution`, when present, is a positive integer
number of source ticks or divisions per quarter note.

`duration_qn` is non-negative and at least every note offset, bar end, beat end,
event onset, and annotation end.

### `ValidationIssue`, `ValidationReport`, and
`CanonicalValidationError`

```python
ValidationCode: TypeAlias = Literal[
    "SCHEMA_VERSION_UNSUPPORTED",
    "JSON_UNKNOWN_FIELD",
    "JSON_MISSING_FIELD",
    "JSON_TYPE_INVALID",
    "FIELD_VALUE_INVALID",
    "RATIONAL_INVALID",
    "RATIONAL_NOT_NORMALIZED",
    "ENTITY_ID_INVALID",
    "ENTITY_ID_PREFIX_INVALID",
    "ENTITY_ID_DUPLICATE",
    "ENTITY_REFERENCE_INVALID",
    "COLLECTION_ORDER_INVALID",
    "VALUE_NOT_FINITE",
    "TIME_NEGATIVE",
    "DURATION_NEGATIVE",
    "ZERO_DURATION_NON_GRACE",
    "PITCH_OUT_OF_RANGE",
    "VELOCITY_OUT_OF_RANGE",
    "CHANNEL_OUT_OF_RANGE",
    "PROGRAM_OUT_OF_RANGE",
    "SOURCE_INDEX_INVALID",
    "PERCUSSION_MISMATCH",
    "PIECE_DURATION_TOO_SHORT",
    "TEMPO_INVALID",
    "TEMPO_INITIAL_MISSING",
    "TEMPO_DUPLICATE_ONSET",
    "METER_INVALID",
    "METER_INITIAL_MISSING",
    "METER_DUPLICATE_ONSET",
    "METER_NOT_AT_BAR_START",
    "BAR_INVALID",
    "BAR_COVERAGE_INVALID",
    "BAR_METER_MISMATCH",
    "BEAT_INVALID",
    "BEAT_GRID_INVALID",
    "ANNOTATION_INVALID",
    "TARGET_VIEW_INVALID",
    "TARGET_VIEW_DUPLICATE",
    "TARGET_LENGTH_MISMATCH",
    "TARGET_ENTITY_DUPLICATE",
    "TARGET_ALIGNMENT_INVALID",
    "TARGET_ENTITY_INVALID",
    "TARGET_VALUE_INVALID",
    "TARGET_MASK_INVALID",
    "TARGET_CONFIDENCE_INVALID",
    "TARGET_SOURCE_INVALID",
    "TARGET_PROVENANCE_INVALID",
    "QUALITY_FLAG_CODE_INVALID",
    "PROVENANCE_DETAIL_INVALID",
    "PROVENANCE_MISSING",
    "PROVENANCE_PARENT_INVALID",
    "PROVENANCE_CYCLE",
    "EMPTY_PIECE",
    "EMPTY_TRACK",
    "SOURCE_RESOLUTION_UNAVAILABLE",
    "INCOMPLETE_FINAL_BAR",
    "OVERLAPPING_SAME_PITCH_NOTES",
    "MID_BAR_TEMPO_CHANGE",
    "LOW_CONFIDENCE_TARGET",
    "UNREFERENCED_PROVENANCE",
    "EMPTY_OBSERVATION",
    "PIECE_TRAILING_SILENCE",
]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    severity: IssueSeverity
    message: str
    path: str
    entity_id: str | None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]: ...

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]: ...

    @property
    def is_valid(self) -> bool: ...


class CanonicalValidationError(ValueError):
    report: ValidationReport

    def __init__(self, report: ValidationReport) -> None: ...
```

`path` is an RFC 6901 JSON Pointer into the canonical serialized form, such as
`/notes/3/duration_qn`.

## 4. Entity-ID contract

Entity IDs are not arbitrary strings. Every ID has:

```text
<prefix>:<local-id>
```

The complete Phase 1 prefixes are:

| Entity | Prefix |
|---|---|
| piece | `piece:` |
| track | `track:` |
| note | `note:` |
| bar | `bar:` |
| beat | `beat:` |
| tempo event | `tempo:` |
| meter event | `meter:` |
| key-signature event | `keysig:` |
| annotation span | `span:` |
| target array | `target:` |
| provenance record | `prov:` |

The local portion matches:

```text
[A-Za-z0-9][A-Za-z0-9._~-]*
```

IDs are globally unique within a piece, including across entity types.
Adapters create them deterministically from stable source identifiers, not from
the entity's current array index. Synthetic adapters use deterministic
zero-padded counters.

Serialization and canonical sorting never rewrite IDs. Windowing retains every
included entity's original ID; future window identifiers are separate metadata,
not replacements for entity IDs. Adapter conversion must preserve an existing
canonical ID when converting between canonical-compatible representations.
Clipped or genuinely synthesized entities require new deterministic IDs and a
provenance parent pointing to the source entity or conversion record.

## 5. Canonical ordering

Collections are already sorted in a valid `CanonicalPiece`; non-canonical order
is an error rather than an invitation for a serializer to mutate the model.
Sort keys are:

| Collection | Sort key |
|---|---|
| tracks | `(source_track_index is None, source_track_index, track_id)` |
| notes | `(onset_qn, canonical track order, pitch, duration_qn, note_id)` |
| bars | `(start_qn, index, bar_id)` |
| beats | `(start_qn, bar_id, index_in_bar, beat_id)` |
| tempo events | `(onset_qn, tempo_event_id)` |
| meter events | `(onset_qn, meter_event_id)` |
| key-signature events | `(onset_qn, key_signature_event_id)` |
| annotations | `(start_qn, end_qn, annotation_id)` |
| targets | `(task, annotation_view_id is not None, annotation_view_id or "", target_id)`; null views sort first |
| provenance | topological parent-before-child order, with `provenance_id` breaking ties |
| quality flags | `(code, entity_ids, message)` |

At most one tempo, meter, or key-signature event of its type may occur at one
onset. When different event types share an onset, consumers apply them in this
deterministic order:

```text
meter -> tempo -> key signature
```

All take effect inclusively at that onset. The ordering defines deterministic
processing and serialization; it does not imply that tempo changes meter or
that key signature is a theory label.

## 6. Timing and musical semantics

### Origin and quarter-note units

Piece time begins at `0/1` at the first represented musical instant, including
the beginning of a pickup. All onsets, starts, and event times are non-negative.
One quarter note is `1/1`; an eighth note is `1/2`; a triplet eighth is `1/3`.
Float equality is never part of canonical timing.

At least one tempo event and one meter event must exist at `0/1`. When source
metadata is missing, an adapter inserts an explicit default event with
`ProvenanceKind="default"` and an appropriate `QualityFlag`; the schema never
silently assumes a default.

### Tempo and meter changes

Tempo changes may occur mid-bar and are valid; validation emits
`MID_BAR_TEMPO_CHANGE` as a warning because downstream renderers must handle the
piecewise map carefully. Meter changes must occur at a canonical bar start.

Bars and beats reference the effective `meter_event_id`. A pickup is represented
by actual duration plus `metric_offset_qn`, not negative time or a synthetic
pre-zero bar.

### Notes, overlap, and bar boundaries

Non-grace notes occupy half-open intervals:

```text
[onset_qn, onset_qn + duration_qn)
```

Notes crossing bar, meter, or tempo boundaries remain one `CanonicalNote`; they
are never split merely for containment. Later graph construction may connect
the one stable note ID to every overlapped beat or bar.

Polyphony and overlap are represented by independent note records. Notes may
share onset, pitch, track, or duration. Same-pitch overlaps on one track are
valid but produce the warning `OVERLAPPING_SAME_PITCH_NOTES`; they are never
merged automatically.

Negative duration is an error. Zero duration is valid only when
`is_grace=true`. Grace notes are point events at their onset and do not create a
negative or artificial sounding interval. A grace note may also have a positive
source-measured duration.

### Percussion

Canonical tracks are homogeneous with respect to percussion. Every note's
`is_percussion` must equal its track's value. If one source track mixes MIDI
percussion and pitched channels, the adapter creates deterministic logical
canonical tracks that may share `source_track_index`.

Percussion pitch preserves the source MIDI drum-note number. Pitch spelling is
normally unavailable, program may be unavailable, and MIDI channel 9 is
evidence but is not the sole definition of percussion. Semantic `drums` role
supervision is a target, not a raw track field.

## 7. Provenance model

`CanonicalPiece.provenance` is the piece-local provenance graph. Raw entities
and events optionally reference one record with `provenance_id`. Every available
target entry requires an aligned provenance reference. Conversion records
reference their source records through `parents`.

The minimum recommended chain is:

```text
source record -> conversion record -> annotation/derivation record
```

Provenance preserves source identity, conversion version, checksums when
available, target origin, confidence context, and explicit defaults. Confidence
does not replace provenance, and provenance does not replace target masks.

### Trailing-silence content boundary

The sounding/observation content end used by `PIECE_TRAILING_SILENCE` is the
latest of:

- the offset of every positive-duration note, including percussion notes and
  positive-duration grace notes;
- the `end_qn` of every positive-duration
  `AnnotationSpan(layer="observation")`.

Zero-duration grace notes and point annotations do not extend this boundary.
Bars, beats, tempo events, meter events, key-signature events, targets, and
`target_alignment` spans are structural or supervisory and do not count as
sounding/observation content.

When no qualifying content exists, the content end is `0/1`. Therefore:

- an empty or structural-only piece with positive `duration_qn` emits both
  `EMPTY_PIECE` and `PIECE_TRAILING_SILENCE`;
- an empty or structural-only piece with zero duration emits `EMPTY_PIECE` but
  not `PIECE_TRAILING_SILENCE`;
- a piece containing only zero-duration grace notes or point annotations follows
  the same rule;
- percussion notes count exactly like pitched notes.

## 8. Validation policy

### Errors

These codes always have `severity="error"`:

| Code | Exact condition |
|---|---|
| `SCHEMA_VERSION_UNSUPPORTED` | `schema_version` is not exactly `2.0.0` |
| `JSON_UNKNOWN_FIELD` | any decoded object contains an undeclared key |
| `JSON_MISSING_FIELD` | any declared serialized field key is absent |
| `JSON_TYPE_INVALID` | during JSON decoding, a value's runtime type cannot satisfy its declared schema type; JSON booleans do not count as integers |
| `FIELD_VALUE_INVALID` | a value has an admissible runtime type but violates a declared semantic constraint for which no more specific validation code exists |
| `RATIONAL_INVALID` | rational keys/types are wrong, denominator is non-positive, or construction otherwise fails |
| `RATIONAL_NOT_NORMALIZED` | a JSON rational is reducible, has a negative denominator, or encodes zero with a denominator other than one |
| `ENTITY_ID_INVALID` | an entity ID does not match the required lexical form |
| `ENTITY_ID_PREFIX_INVALID` | an entity field or target alignment uses the wrong type prefix |
| `ENTITY_ID_DUPLICATE` | two piece entities, including target arrays, share an ID |
| `ENTITY_REFERENCE_INVALID` | a non-target reference points to no entity of the required type |
| `COLLECTION_ORDER_INVALID` | a collection differs from its canonical order |
| `VALUE_NOT_FINITE` | any schema float is NaN or positive/negative infinity |
| `TIME_NEGATIVE` | an onset, start, end, metric offset, or source time that must be non-negative is negative |
| `DURATION_NEGATIVE` | a note, bar, beat, or source duration is negative |
| `ZERO_DURATION_NON_GRACE` | a note has zero duration and `is_grace=false` |
| `PITCH_OUT_OF_RANGE` | note pitch is outside `[0,127]` |
| `VELOCITY_OUT_OF_RANGE` | available velocity is outside `[0,127]` |
| `CHANNEL_OUT_OF_RANGE` | available channel is outside `[0,15]` |
| `PROGRAM_OUT_OF_RANGE` | available program is outside `[0,127]` |
| `SOURCE_INDEX_INVALID` | available source track index, tick, staff, voice, or similar integer constrained non-negative is negative |
| `PERCUSSION_MISMATCH` | a note's percussion flag differs from its referenced track |
| `PIECE_DURATION_TOO_SHORT` | piece duration is before any note offset, bar/beat end, event onset, or annotation end |
| `TEMPO_INVALID` | tempo onset is negative or microseconds per quarter is not a positive integer |
| `TEMPO_INITIAL_MISSING` | no tempo event exists at `0/1` |
| `TEMPO_DUPLICATE_ONSET` | two tempo events share an onset |
| `METER_INVALID` | meter onset/numerator is invalid or denominator is not a positive power of two |
| `METER_INITIAL_MISSING` | no meter event exists at `0/1` |
| `METER_DUPLICATE_ONSET` | two meter events share an onset |
| `METER_NOT_AT_BAR_START` | in a piece with bars, a meter event onset is not a bar start |
| `BAR_INVALID` | bar index/start/duration/metric offset/pickup flags are internally inconsistent |
| `BAR_COVERAGE_INVALID` | non-empty bars do not form contiguous, non-overlapping coverage from `0/1` through piece duration |
| `BAR_METER_MISMATCH` | a complete bar differs from effective nominal duration, or an incomplete bar exceeds it |
| `BEAT_INVALID` | beat duration/index/position/downbeat state is internally inconsistent or lies outside its referenced bar |
| `BEAT_GRID_INVALID` | for non-empty bars, beats do not form the effective denominator-unit grid over each actual bar extent |
| `ANNOTATION_INVALID` | span ordering, layer/value constraint, type prefix, track reference, or piece bounds are invalid |
| `TARGET_VIEW_INVALID` | a programmatic non-null `annotation_view_id` is not a string, or a string view is empty after trimming, untrimmed, or contains an ASCII control character |
| `TARGET_VIEW_DUPLICATE` | two target arrays share the same `(task, annotation_view_id)` pair |
| `TARGET_LENGTH_MISMATCH` | aligned target fields have different lengths |
| `TARGET_ENTITY_DUPLICATE` | one target array repeats an entity ID |
| `TARGET_ALIGNMENT_INVALID` | alignment type and entity prefix/count rules disagree |
| `TARGET_ENTITY_INVALID` | a target entity does not exist in the containing piece |
| `TARGET_VALUE_INVALID` | value type, class vocabulary, scalar, multi-label, or distribution rules fail |
| `TARGET_MASK_INVALID` | a masked entry is non-null or an available entry is null |
| `TARGET_CONFIDENCE_INVALID` | non-null available confidence is non-finite/outside `[0,1]`, or unavailable confidence is non-null |
| `TARGET_SOURCE_INVALID` | available source is absent/unsupported, or unavailable source is non-null |
| `TARGET_PROVENANCE_INVALID` | available provenance is absent/dangling, or unavailable provenance is non-null |
| `QUALITY_FLAG_CODE_INVALID` | a quality flag code does not match the lowercase dotted namespace syntax |
| `PROVENANCE_DETAIL_INVALID` | detail keys are empty, duplicate, unsorted, or values are not finite JSON scalars |
| `PROVENANCE_MISSING` | the piece has no provenance records |
| `PROVENANCE_PARENT_INVALID` | a parent is missing, duplicated, self-referential, or not earlier in canonical topological order |
| `PROVENANCE_CYCLE` | provenance parent links contain a cycle |

`FIELD_VALUE_INVALID` is the fallback for semantic-value constraints, not a
replacement for specific codes such as `TEMPO_INVALID`, `METER_INVALID`,
`TARGET_VALUE_INVALID`, `QUALITY_FLAG_CODE_INVALID`, or `ENTITY_ID_INVALID`.
It includes at least:

- `KeySignatureEvent.fifths` outside `[-7,7]`;
- a `KeySignatureEvent.mode` outside `KeySignatureMode`;
- `mode="other"` with `raw_value` absent or empty after stripping;
- `CanonicalNote.spelling_step` outside `A` through `G`;
- non-null `spelling_alter` when `spelling_step` is null;
- unsupported microtonal spelling placed directly into `spelling_alter` or
  silently rounded there instead of being preserved in provenance;
- an open stable identifier that is empty/whitespace-only, untrimmed, or
  contains an ASCII control character, except where a more specific code such
  as `TARGET_VIEW_INVALID` applies;
- an invalid RFC 3339 `ProvenanceRecord.created_at`;
- a `checksum_sha256` that is not exactly 64 lowercase hexadecimal characters;
- an empty or otherwise invalid `ProvenanceRecord.source`;
- an unsupported enum or `Literal` value in a programmatically constructed
  canonical record when no more specific validation code applies.

A decoded JSON value with the wrong runtime type uses `JSON_TYPE_INVALID`
instead. For example, a numeric `KeySignatureEvent.mode` fails JSON type
validation, while the string `"aeolian"` fails semantic field-value validation.
Likewise, a non-string JSON `annotation_view_id` uses `JSON_TYPE_INVALID`;
programmatic or lexical view violations use `TARGET_VIEW_INVALID`.

### Warnings

These codes always have `severity="warning"`:

| Code | Exact condition |
|---|---|
| `EMPTY_PIECE` | the piece contains no notes |
| `EMPTY_TRACK` | a canonical track has no notes |
| `SOURCE_RESOLUTION_UNAVAILABLE` | `source_resolution` is `None` |
| `INCOMPLETE_FINAL_BAR` | the final bar is shortened, incomplete, and not a pickup |
| `OVERLAPPING_SAME_PITCH_NOTES` | positive-duration notes of the same pitch overlap on one track |
| `MID_BAR_TEMPO_CHANGE` | a tempo event occurs strictly inside a bar |
| `LOW_CONFIDENCE_TARGET` | an available target entry has non-null confidence below `0.5` |
| `UNREFERENCED_PROVENANCE` | a provenance record is neither referenced by a record/target/flag nor the parent of another record |
| `EMPTY_OBSERVATION` | an observation annotation has the available value `""` |
| `PIECE_TRAILING_SILENCE` | piece duration is greater than the sounding/observation content end defined above |

The following are explicitly valid and do not themselves cause warnings:

- absent optional metadata represented by `null`;
- an empty optional collection represented by `[]`;
- notes crossing bar, tempo, or meter boundaries;
- simultaneous notes or different-pitch overlaps;
- empty target arrays whose aligned length is zero;
- `mask=false` entries with null values;
- available target entries with `confidence=null`;
- a key signature that is absent or differs from a target local key;
- percussion notes outside common General MIDI drum mappings.

Validation issues are sorted deterministically by `(path, severity, code,
entity_id or "", message)`.

## 9. Unknown fields and version compatibility

Canonical loaders reject unknown fields at every object level and reject missing
declared fields. Unknown values are not retained. Dataset-specific trace data
must use declared provenance `details`; schema evolution must add a versioned
field rather than relying on silent passthrough.

The strict Phase 1 reader accepts exactly schema version `2.0.0`. It does not
assume that another version with the same major number is compatible. Future
additive or breaking changes require:

1. a new schema version;
2. an ADR;
3. an explicit migration function or loader path;
4. round-trip and compatibility tests.

Writers always emit exactly `2.0.0`. There is no permissive or
best-effort public loader in Phase 1.

## 10. Deterministic JSON rules

The canonical mapping uses these exact field names and nesting from the
dataclasses above. Rationals are always:

```json
{"num": 3, "den": 2}
```

`num` is an integer and `den` is a positive integer. Strict input requires the
pair already be normalized and requires exactly those two keys.

Python tuples become JSON arrays. Provenance `details` become JSON objects.
All dataclass keys, including null and empty values, are emitted.

`dumps_piece` uses:

```python
json.dumps(
    piece_to_dict(piece),
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    indent=indent,
    separators=None if indent is not None else (",", ":"),
)
```

`dump_piece` writes UTF-8 with `newline="\n"` and appends exactly one terminal
newline. `loads_piece` accepts UTF-8 JSON text only. Determinism is defined as
byte-identical output for the same valid immutable piece and the same `indent`.

## 11. Complete canonical JSON example

This synthetic piece has a one-quarter-note pickup in 4/4, a pitched track, a
percussion track, a sustained note crossing the pickup boundary, explicit tempo
and meter, bars and beats, unavailable optional metadata, track roles only in a
target, a partially masked chord-quality target, and an available theory label
whose numeric confidence is unknown. Its last sounding note ends at `4/1` while
the piece ends at `5/1`, so the revised trailing-silence rule is reachable. No
raw track or note field contains theory. The example contains two separate valid
analyses for `theory.chord_quality`: the default view and
`analysis.alternative`.

```json
{
  "annotations": [
    {
      "annotation_id": "span:lyric-000",
      "annotation_type": "text.lyric",
      "end_qn": {"den": 1, "num": 1},
      "layer": "observation",
      "provenance_id": "prov:source",
      "start_qn": {"den": 1, "num": 0},
      "track_id": "track:melody",
      "value": "la"
    }
  ],
  "bars": [
    {
      "bar_id": "bar:000",
      "display_number": "0",
      "duration_qn": {"den": 1, "num": 1},
      "index": 0,
      "is_incomplete": true,
      "is_pickup": true,
      "meter_event_id": "meter:000",
      "metric_offset_qn": {"den": 1, "num": 3},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 0}
    },
    {
      "bar_id": "bar:001",
      "display_number": "1",
      "duration_qn": {"den": 1, "num": 4},
      "index": 1,
      "is_incomplete": false,
      "is_pickup": false,
      "meter_event_id": "meter:000",
      "metric_offset_qn": {"den": 1, "num": 0},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 1}
    }
  ],
  "beats": [
    {
      "bar_id": "bar:000",
      "beat_id": "beat:000",
      "duration_qn": {"den": 1, "num": 1},
      "index_in_bar": 3,
      "is_downbeat": false,
      "meter_event_id": "meter:000",
      "position_in_bar_qn": {"den": 1, "num": 3},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 0},
      "strength": 0.5
    },
    {
      "bar_id": "bar:001",
      "beat_id": "beat:001",
      "duration_qn": {"den": 1, "num": 1},
      "index_in_bar": 0,
      "is_downbeat": true,
      "meter_event_id": "meter:000",
      "position_in_bar_qn": {"den": 1, "num": 0},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 1},
      "strength": 1.0
    },
    {
      "bar_id": "bar:001",
      "beat_id": "beat:002",
      "duration_qn": {"den": 1, "num": 1},
      "index_in_bar": 1,
      "is_downbeat": false,
      "meter_event_id": "meter:000",
      "position_in_bar_qn": {"den": 1, "num": 1},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 2},
      "strength": 0.5
    },
    {
      "bar_id": "bar:001",
      "beat_id": "beat:003",
      "duration_qn": {"den": 1, "num": 1},
      "index_in_bar": 2,
      "is_downbeat": false,
      "meter_event_id": "meter:000",
      "position_in_bar_qn": {"den": 1, "num": 2},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 3},
      "strength": 0.75
    },
    {
      "bar_id": "bar:001",
      "beat_id": "beat:004",
      "duration_qn": {"den": 1, "num": 1},
      "index_in_bar": 3,
      "is_downbeat": false,
      "meter_event_id": "meter:000",
      "position_in_bar_qn": {"den": 1, "num": 3},
      "provenance_id": "prov:conversion",
      "start_qn": {"den": 1, "num": 4},
      "strength": 0.5
    }
  ],
  "dataset_name": "synthetic",
  "duration_qn": {"den": 1, "num": 5},
  "key_signature_events": [],
  "metadata": {
    "collection": "",
    "copyright": null,
    "creators": ["Music Critic V2"],
    "genres": [],
    "language": null,
    "movement_number": null,
    "movement_title": null,
    "source_format": "synthetic",
    "title": null
  },
  "meter_events": [
    {
      "denominator": 4,
      "meter_event_id": "meter:000",
      "numerator": 4,
      "onset_qn": {"den": 1, "num": 0},
      "provenance_id": "prov:source"
    }
  ],
  "notes": [
    {
      "articulations": [],
      "channel": 0,
      "duration_qn": {"den": 1, "num": 2},
      "dynamic": null,
      "is_grace": false,
      "is_percussion": false,
      "note_id": "note:melody-000",
      "onset_qn": {"den": 1, "num": 0},
      "pitch": 67,
      "program": 0,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 960,
      "source_onset_seconds": null,
      "source_onset_ticks": 0,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:melody",
      "velocity": 88,
      "voice": null
    },
    {
      "articulations": [],
      "channel": 0,
      "duration_qn": {"den": 1, "num": 1},
      "dynamic": null,
      "is_grace": false,
      "is_percussion": false,
      "note_id": "note:melody-001",
      "onset_qn": {"den": 1, "num": 1},
      "pitch": 72,
      "program": 0,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 480,
      "source_onset_seconds": null,
      "source_onset_ticks": 480,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:melody",
      "velocity": 92,
      "voice": null
    },
    {
      "articulations": [],
      "channel": 9,
      "duration_qn": {"den": 1, "num": 1},
      "dynamic": null,
      "is_grace": false,
      "is_percussion": true,
      "note_id": "note:drums-000",
      "onset_qn": {"den": 1, "num": 1},
      "pitch": 36,
      "program": null,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 480,
      "source_onset_seconds": null,
      "source_onset_ticks": 480,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:drums",
      "velocity": 100,
      "voice": null
    },
    {
      "articulations": [],
      "channel": 0,
      "duration_qn": {"den": 1, "num": 0},
      "dynamic": null,
      "is_grace": true,
      "is_percussion": false,
      "note_id": "note:melody-002",
      "onset_qn": {"den": 1, "num": 2},
      "pitch": 74,
      "program": 0,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 0,
      "source_onset_seconds": null,
      "source_onset_ticks": 960,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:melody",
      "velocity": 76,
      "voice": null
    },
    {
      "articulations": [],
      "channel": 0,
      "duration_qn": {"den": 1, "num": 2},
      "dynamic": null,
      "is_grace": false,
      "is_percussion": false,
      "note_id": "note:melody-003",
      "onset_qn": {"den": 1, "num": 2},
      "pitch": 76,
      "program": 0,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 960,
      "source_onset_seconds": null,
      "source_onset_ticks": 960,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:melody",
      "velocity": 90,
      "voice": null
    },
    {
      "articulations": [],
      "channel": 9,
      "duration_qn": {"den": 2, "num": 1},
      "dynamic": null,
      "is_grace": false,
      "is_percussion": true,
      "note_id": "note:drums-001",
      "onset_qn": {"den": 1, "num": 3},
      "pitch": 38,
      "program": null,
      "provenance_id": "prov:source",
      "source_duration_seconds": null,
      "source_duration_ticks": 240,
      "source_onset_seconds": null,
      "source_onset_ticks": 1440,
      "spelling_alter": null,
      "spelling_step": null,
      "staff": null,
      "track_id": "track:drums",
      "velocity": 96,
      "voice": null
    }
  ],
  "piece_id": "piece:synthetic-two-track",
  "provenance": [
    {
      "checksum_sha256": null,
      "created_at": "2026-07-16T00:00:00+03:00",
      "details": {
        "description": "hand-authored Phase 1A contract fixture",
        "resolution": 480
      },
      "kind": "synthetic",
      "parents": [],
      "provenance_id": "prov:source",
      "record_id": "synthetic-two-track",
      "source": "music_critic_v2",
      "uri": null,
      "version": "2.0.0"
    },
    {
      "checksum_sha256": null,
      "created_at": "2026-07-16T00:00:00+03:00",
      "details": {
        "method": "exact rational construction"
      },
      "kind": "conversion",
      "parents": ["prov:source"],
      "provenance_id": "prov:conversion",
      "record_id": null,
      "source": "synthetic_adapter",
      "uri": null,
      "version": "1"
    },
    {
      "checksum_sha256": null,
      "created_at": "2026-07-16T00:00:00+03:00",
      "details": {
        "annotator": "contract example"
      },
      "kind": "annotation",
      "parents": ["prov:source"],
      "provenance_id": "prov:theory",
      "record_id": null,
      "source": "human",
      "uri": null,
      "version": null
    }
  ],
  "quality_flags": [
    {
      "code": "canonical.synthetic_source",
      "entity_ids": ["piece:synthetic-two-track"],
      "message": "This piece is a deterministic documentation fixture.",
      "provenance_id": "prov:source",
      "severity": "info"
    }
  ],
  "schema_version": "2.0.0",
  "source_group_id": "synthetic-two-track",
  "source_path": null,
  "source_resolution": 480,
  "split": null,
  "targets": [
    {
      "alignment_type": "beat",
      "annotation_view_id": null,
      "class_labels": ["major", "minor", "dominant_seventh", "no_chord"],
      "confidence": [null, 0.95, null, 0.8, null],
      "entity_ids": [
        "beat:000",
        "beat:001",
        "beat:002",
        "beat:003",
        "beat:004"
      ],
      "mask": [true, true, false, true, false],
      "provenance": [
        "prov:theory",
        "prov:theory",
        null,
        "prov:theory",
        null
      ],
      "source": ["human", "human", null, "human", null],
      "target_id": "target:theory-chord-quality-default",
      "task": "theory.chord_quality",
      "value_type": "categorical",
      "values": ["major", "major", null, "dominant_seventh", null]
    },
    {
      "alignment_type": "beat",
      "annotation_view_id": "analysis.alternative",
      "class_labels": ["major", "minor", "dominant_seventh", "no_chord"],
      "confidence": [0.9, 0.85, 0.7, 0.75, null],
      "entity_ids": [
        "beat:000",
        "beat:001",
        "beat:002",
        "beat:003",
        "beat:004"
      ],
      "mask": [true, true, true, true, false],
      "provenance": [
        "prov:theory",
        "prov:theory",
        "prov:theory",
        "prov:theory",
        null
      ],
      "source": ["human", "human", "human", "human", null],
      "target_id": "target:theory-chord-quality-alternative",
      "task": "theory.chord_quality",
      "value_type": "categorical",
      "values": ["major", "minor", "major", "dominant_seventh", null]
    },
    {
      "alignment_type": "track",
      "annotation_view_id": null,
      "class_labels": [
        "melody",
        "secondary_melody",
        "bass",
        "accompaniment",
        "drums",
        "texture",
        "other"
      ],
      "confidence": [1.0, 1.0],
      "entity_ids": ["track:melody", "track:drums"],
      "mask": [true, true],
      "provenance": ["prov:theory", "prov:theory"],
      "source": ["synthetic", "synthetic"],
      "target_id": "target:track-role-default",
      "task": "track.role",
      "value_type": "categorical",
      "values": ["melody", "drums"]
    }
  ],
  "tempo_events": [
    {
      "microseconds_per_quarter": 500000,
      "onset_qn": {"den": 1, "num": 0},
      "provenance_id": "prov:source",
      "tempo_event_id": "tempo:000"
    }
  ],
  "tracks": [
    {
      "channel": 0,
      "instrument_name": "Acoustic Grand Piano",
      "is_percussion": false,
      "name": "",
      "program": 0,
      "provenance_id": "prov:source",
      "source_track_index": 0,
      "track_id": "track:melody"
    },
    {
      "channel": 9,
      "instrument_name": null,
      "is_percussion": true,
      "name": null,
      "program": null,
      "provenance_id": "prov:source",
      "source_track_index": 1,
      "track_id": "track:drums"
    }
  ]
}
```

## 12. Phase 1 implementation acceptance derived from this contract

Phase 1B must implement this API without inventing fields and add tests for:

- normalized rational construction, arithmetic, ordering, and JSON;
- exact round trip of the revised three-target synthetic example;
- malformed JSON, unknown fields, missing fields, and version mismatch;
- `FIELD_VALUE_INVALID` for programmatically constructed records and the
  distinction from `JSON_TYPE_INVALID`;
- ID syntax, uniqueness, ordering, and references;
- pickup bars, denominator-unit beats, meter/tempo changes, cross-bar notes,
  overlaps, percussion splitting assumptions, and grace notes;
- target IDs, default and alternative annotation views, view uniqueness, and
  separation of alternative analyses;
- two target arrays sharing a task with different views, rejection of duplicate
  `(task, annotation_view_id)`, and whitespace-only/untrimmed/control-character
  view IDs;
- all target value types, masks, aligned lengths, known/unknown confidence,
  source, and provenance;
- modal key-signature observations;
- invalid key-signature fifths and `"other"` mode without `raw_value`;
- invalid pitch-spelling fields and unsupported microtonal spelling placement;
- invalid provenance RFC 3339 timestamps and SHA-256 checksums;
- namespaced quality-flag validation;
- reachable sounding/observation-based trailing-silence warnings;
- unsupported microtonal spelling preservation without silent rounding;
- deterministic serialization bytes;
- error/warning classification and exception report retention.

MIDI parsing, graph construction, tensor conversion, window implementation, and
model code remain outside Phase 1A.
