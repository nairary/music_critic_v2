"""Phase 9E-B3 AnalysisGNN-derived multi-task dataset contract.

This module is deliberately target-only.  It defines corpus selection, task,
vocabulary, entity, split, metric, and TEST-lock contracts without importing a
model or changing the Phase 9E-B2 raw graph path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import ast
from bisect import bisect_right
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from music_critic.adapters.dilemmadata import (
    DilemmadataAccepted,
    DilemmadataCorpusDiscovery,
    DilemmadataCorpusRecord,
)
from music_critic.data import CanonicalPiece, RationalTime
from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    ANALYSISGNN_REPOSITORY,
    canonical_json,
    fingerprint,
)


FULL_RAW_UNIVERSE_ID = "dilemmadata-full-raw-v1"
PAPER_CANDIDATE_UNIVERSE_ID = "analysisgnn-paper-candidate-an-dlc-v1"
PINNED_CODE_REGISTRY_ID = "analysisgnn-pinned-code-reference-v1"
PRODUCTION_REGISTRY_ID = "analysisgnn-corrected-multitask-v1"
ENTITY_REGISTRY_VERSION = "analysisgnn-multitask-entities-v1"
SPLIT_CONTRACT_VERSION = "analysisgnn-source-component-split-v1"
METRIC_CONTRACT_VERSION = "analysisgnn-paper-compatible-metrics-v1"
DATASET_MANIFEST_VERSION = "phase9eb3-analysisgnn-multitask-dataset-v1"
TARGET_SIDECAR_VERSION = "analysisgnn-source-native-target-sidecar-v1"
SOURCE_COMPONENT_VERSION = "dilemmadata-source-component-v1"
ASSIGNMENT_ALGORITHM = "sha256-canonical-greedy-record-quota-v1"
ASSIGNMENT_NAMESPACE = "music-critic-v2.phase9eb3.analysisgnn"
ASSIGNMENT_SEED = "e115182fb29b74bdcb6bf3547ed427d967580947"
EXPECTED_FULL_COUNTS = {"an_joint": 353, "dlc": 1280, "total": 1633}
EXPECTED_PAPER_COUNTS = {"an_joint": 353, "dlc": 1266, "total": 1619}


class AnalysisGNNMultitaskContractError(ValueError):
    """A B3 artifact or API request violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class VocabularySpec:
    vocabulary_id: str
    labels: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...] = ()
    external_reference: str = f"{ANALYSISGNN_REPOSITORY}@{ANALYSISGNN_COMMIT}"
    notes: str = ""

    @property
    def class_count(self) -> int:
        return len(self.labels)

    def __post_init__(self) -> None:
        if not self.vocabulary_id or len(self.labels) != len(set(self.labels)):
            raise AnalysisGNNMultitaskContractError(
                "vocabulary IDs and labels must be non-empty and unique"
            )
        alias_keys = tuple(source for source, _target in self.aliases)
        if len(alias_keys) != len(set(alias_keys)):
            raise AnalysisGNNMultitaskContractError("duplicate vocabulary alias")
        if any(target not in self.labels for _source, target in self.aliases):
            raise AnalysisGNNMultitaskContractError(
                "vocabulary aliases must resolve to canonical labels"
            )

    def normalize(self, value: str) -> str | None:
        aliases = dict(self.aliases)
        canonical = aliases.get(value, value)
        return canonical if canonical in self.labels else None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    schema_version: str
    task_family: str
    prediction_level: Literal[
        "note", "onset", "harmonic_event", "beat", "measure", "piece"
    ]
    entity_type: str
    source_dialects: tuple[str, ...]
    source_fields: tuple[str, ...]
    vocabulary_id: str
    class_count: int
    missing_policy: str
    mask_policy: str
    metric_ids: tuple[str, ...]
    joint_metric_group: str | None
    external_reference: str


@dataclass(frozen=True, slots=True)
class SourceTargetRow:
    ordinal: int
    line: int
    onset_qn: RationalTime
    canonical_note_id: str | None
    repair_mask_scope: str
    values: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TargetState:
    available: bool
    masked: bool
    missing_reason: str | None
    source_value: str | None
    canonical_value: str | None
    source_entity_id: str
    canonical_entity_id: str
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    record_id: str
    dialect: str
    source_component_id: str
    split: Literal["train", "validation", "test"]
    assignment_algorithm: str = ASSIGNMENT_ALGORITHM
    assignment_namespace: str = ASSIGNMENT_NAMESPACE


OVERLAP_EXCLUSION_NAMES = (
    "ABC_n01op18-1_01",
    "ABC_n01op18-1_03",
    "ABC_n06op18-6_03",
    "ABC_n07op59-1_01",
    "ABC_n08op59-2_03",
    "ABC_n10op74_03",
    "ABC_n10op74_04",
    "ABC_n11op95_03",
    "ABC_n12op127_02",
    "ABC_n16op135_02",
    "beethoven_piano_sonatas_01-1",
    "beethoven_piano_sonatas_07-1",
    "beethoven_piano_sonatas_10-1",
    "beethoven_piano_sonatas_23-1",
)
OVERLAP_AN_PEERS = {
    "ABC_n01op18-1_01": "an:test:abc-op18-no1-1",
    "ABC_n01op18-1_03": "an:test:abc-op18-no1-3",
    "ABC_n06op18-6_03": "an:test:abc-op18-no6-3",
    "ABC_n07op59-1_01": "an:test:abc-op59-no1-1",
    "ABC_n08op59-2_03": "an:test:abc-op59-no2-3",
    "ABC_n10op74_03": "an:test:abc-op74-3",
    "ABC_n10op74_04": "an:test:abc-op74-4",
    "ABC_n11op95_03": "an:test:abc-op95-3",
    "ABC_n12op127_02": "an:test:abc-op127-2",
    "ABC_n16op135_02": "an:test:abc-op135-2",
    "beethoven_piano_sonatas_01-1": "an:test:bps-01-op002-no1-1",
    "beethoven_piano_sonatas_07-1": "an:test:bps-07-op010-no3-1",
    "beethoven_piano_sonatas_10-1": "an:test:bps-10-op014-no2-1",
    "beethoven_piano_sonatas_23-1": (
        "an:test:bps-23-op057-appassionata-1"
    ),
}


_KEYS = tuple(
    "A A# Ab Abb B B# Bb Bbb C C# C## Cb D D# Db Dbb E E# Eb Ebb "
    "F F# F## Fb G G# G## Gb a a# ab b b# bb bbb c c# cb d d# db "
    "e e# eb f f# f## g g# gb".split()
)
_TONE_FUNCTIONS = tuple(
    "A A# A## Ab Abb B B# Bb Bbb Bbbb C C# C## C### Cb Cbb D D# "
    "D## Db Dbb E E# E## E### Eb Ebb F F# F## Fb Fbb G G# G## Gb Gbb".split()
)
_DEGREES = tuple(
    "-1 -2 -3 -4 -5 -6 -7 1 2 3 4 5 6 7 #1 #2 #3 #4 #5 #6 #7".split()
)
_QUALITIES = (
    "major triad",
    "minor triad",
    "diminished triad",
    "augmented triad",
    "minor seventh chord",
    "major seventh chord",
    "dominant seventh chord",
    "incomplete dominant-seventh chord",
    "diminished seventh chord",
    "half-diminished seventh chord",
    "augmented sixth",
    "German augmented sixth chord",
    "French augmented sixth chord",
    "Italian augmented sixth chord",
    "minor-augmented tetrachord",
    "augmented seventh chord",
    "augmented major tetrachord",
)
_ROMAN_NUMERALS = tuple(
    """I V7 V i viio7 IV ii viio vi iv VI ii7 ii%7 Cad v N Ger7 V9 iii III
    iio vii%7 vi7 VII iv7 IV7 iio7 I7 It bVI VI7 III+ I+ V+ i7 bVII Fr7
    III7 vii iii7 vi%7 II7 II V+7 v7 N7 iiio III+7 vii7 #iio7 Ger bVII7
    Fr bVI7 IV+ VII7 #ivo7 bV #io7 iiio7 bvi bvii #vio7 #vii%7 #vo7 ii%
    vio vo V79 VI+7 #io #ivo vii% #V ii9 bIII VI+ v%7 #vi%7 #iii ivo7
    #iio bviio7 #vio biii vio7 #vo IV9 iii%7 #iv%7 bIV7 iv9 bvii7 vii+
    #iv io ivo #iiio7 vo7 iv%7 #i I+7 N+ #vi7 #vii #vii7 bII bV7 #iiio
    #vi #vii% vii%9 #iv7 VII+ bi bVI+ bbVII II+ I9 ii%9 #III N+7 bbvii
    iii9 bv biv bi7 bii7 biio bI bviio #VII bvio7 #i%7 bV+ biv7 bvio ii+7
    i%7 bvi%7 bI+ #VI%7 bv7 bii%7 io7 #I7 bV+7 bIV biii7 bI7 #iii7 II+7
    #VII+ #IV7 #VI7 biio7 #v #v%7 #ii #v7 #i7 bvii%7 #II #VI+ #I #ii7
    bVII+ bvi7 #IV #viio #V7 ##vio #viio7 bIII7 #III7 bIII+ #V+ bIII+7
    #ii%7 bi+7 #VII7 IV+7 #VI bii""".split()
)
_PCSETS = tuple(
    """0,1,5,8 0,2,5,9 0,2,6 0,2,6,8 0,2,6,9 0,3,5,8 0,3,5,9
    0,3,6 0,3,6,8 0,3,6,9 0,3,7 0,3,7,8 0,3,7,10 0,3,8 0,3,9
    0,4,5,9 0,4,6,10 0,4,7 0,4,7,9 0,4,7,10 0,4,7,11 0,4,9
    0,4,10 0,5,8 0,5,9 0,6,8 0,6,9 1,2,6,9 1,3,6,10 1,3,7
    1,3,7,9 1,3,7,10 1,4,6,9 1,4,6,10 1,4,7 1,4,7,9 1,4,7,10
    1,4,8 1,4,8,9 1,4,8,11 1,4,9 1,4,10 1,5,6,10 1,5,7,11
    1,5,8 1,5,8,10 1,5,8,11 1,5,10 1,5,11 1,6,9 1,6,10 1,7,9
    1,7,10 2,3,7,10 2,4,7,11 2,4,8 2,4,8,10 2,4,8,11 2,5,7,10
    2,5,7,11 2,5,8 2,5,8,10 2,5,8,11 2,5,9 2,5,9,10 2,5,10
    2,5,11 2,6,7,11 2,6,9 2,6,9,11 2,6,11 2,7,10 2,7,11 2,8,10
    2,8,11 3,4,8,11 3,5,9 3,5,9,11 3,6,8,11 3,6,9 3,6,9,11
    3,6,10 3,6,10,11 3,6,11 3,7,10 3,8,11 3,9,11 4,6,10 4,7,10
    4,7,11 4,8,11 5,7,11 5,8,11""".split()
)
_NOTE_DEGREES = tuple(
    f"{accidental}{degree}"
    for degree in "1234567"
    for accidental in ("bbb", "bb", "b", "", "#", "##", "###")
)


VOCABULARIES = (
    VocabularySpec("analysisgnn.key-v1", _KEYS),
    VocabularySpec("analysisgnn.tone-function-v1", _TONE_FUNCTIONS),
    VocabularySpec("analysisgnn.degree-v1", _DEGREES),
    VocabularySpec(
        "analysisgnn.quality-corrected-v1",
        _QUALITIES,
        aliases=(
            ("+7", "augmented seventh chord"),
            ("+M7", "augmented major tetrachord"),
            ("augmented seventh", "augmented seventh chord"),
        ),
        notes=(
            "The pinned 16-entry literal contains missing=None while the head is 15; "
            "missing is masked, and two semantically distinct DLC +7/+M7 qualities "
            "are restored instead of being collapsed into augmented triad."
        ),
    ),
    VocabularySpec("analysisgnn.inversion-v1", ("0", "1", "2", "3")),
    VocabularySpec(
        "analysisgnn.roman-numeral-corrected-v1",
        _ROMAN_NUMERALS,
        notes=(
            "The pinned 184-entry Python literal is repaired by splitting the "
            "accidentally concatenated #VIIbvio7 token; missing=none is masked, "
            "leaving 184 semantic classes from 185 intended entries."
        ),
    ),
    VocabularySpec("analysisgnn.pitch-class-set-v1", _PCSETS),
    VocabularySpec("analysisgnn.boolean-v1", ("false", "true")),
    VocabularySpec(
        "analysisgnn.cadence-dlc-v1", ("DC", "EC", "HC", "IAC", "PAC", "PC")
    ),
    VocabularySpec("analysisgnn.metrical-strength-v1", tuple(map(str, range(45)))),
    VocabularySpec("analysisgnn.note-degree-v1", _NOTE_DEGREES),
)
VOCABULARY_BY_ID = {row.vocabulary_id: row for row in VOCABULARIES}


def _task(
    task_id: str,
    family: str,
    level: Literal["note", "onset", "harmonic_event", "beat", "measure", "piece"],
    dialects: tuple[str, ...],
    fields: tuple[str, ...],
    vocabulary_id: str,
    *,
    joint: str | None = None,
    metrics: tuple[str, ...] = ("accuracy",),
) -> TaskSpec:
    vocabulary = VOCABULARY_BY_ID[vocabulary_id]
    return TaskSpec(
        task_id=task_id,
        schema_version="1.0.0",
        task_family=family,
        prediction_level=level,
        entity_type=level,
        source_dialects=dialects,
        source_fields=fields,
        vocabulary_id=vocabulary_id,
        class_count=vocabulary.class_count,
        missing_policy="missing_and_unknown_are_distinct_and_never_class_zero",
        mask_policy="loss_and_metrics_require_available_true_and_masked_false",
        metric_ids=metrics,
        joint_metric_group=joint,
        external_reference=f"{ANALYSISGNN_REPOSITORY}@{ANALYSISGNN_COMMIT}",
    )


_BOTH = ("an_joint", "dlc")
PRODUCTION_TASKS = (
    _task("local_key", "local_key", "harmonic_event", _BOTH, ("a_localKey",), "analysisgnn.key-v1", joint="roman_numeral_joint"),
    _task("tonicized_key", "tonicized_key", "harmonic_event", _BOTH, ("a_tonicizedKey", "a_degree2"), "analysisgnn.key-v1"),
    _task("root", "root", "harmonic_event", _BOTH, ("a_root",), "analysisgnn.tone-function-v1"),
    _task("bass", "bass", "harmonic_event", _BOTH, ("a_bass",), "analysisgnn.tone-function-v1"),
    _task("primary_degree", "scale_degree", "harmonic_event", _BOTH, ("a_degree1",), "analysisgnn.degree-v1", joint="roman_numeral_joint"),
    _task("secondary_degree", "scale_degree", "harmonic_event", _BOTH, ("a_degree2",), "analysisgnn.degree-v1", joint="roman_numeral_joint"),
    _task("quality", "quality", "harmonic_event", _BOTH, ("a_quality", "chord_type"), "analysisgnn.quality-corrected-v1", joint="roman_numeral_joint"),
    _task("inversion", "inversion", "harmonic_event", _BOTH, ("a_inversion", "figbass"), "analysisgnn.inversion-v1", joint="roman_numeral_joint"),
    _task("roman_numeral", "roman_numeral", "harmonic_event", _BOTH, ("a_simpleNumeral",), "analysisgnn.roman-numeral-corrected-v1"),
    _task("pitch_class_set", "pitch_class_set", "harmonic_event", _BOTH, ("a_pcset", "chord_tones"), "analysisgnn.pitch-class-set-v1"),
    _task("harmonic_rhythm", "harmonic_rhythm", "harmonic_event", _BOTH, ("a_isOnset",), "analysisgnn.boolean-v1"),
    _task("cadence", "cadence", "onset", ("dlc",), ("cadence_type",), "analysisgnn.cadence-dlc-v1", metrics=("accuracy", "macro_f1")),
    _task("phrase", "phrase", "onset", ("dlc",), ("a_phraseend",), "analysisgnn.boolean-v1", metrics=("accuracy", "macro_f1")),
    _task("section", "section", "onset", ("dlc",), ("section_start",), "analysisgnn.boolean-v1", metrics=("accuracy", "macro_f1")),
    _task("pedal", "organ_point", "harmonic_event", ("dlc",), ("pedal",), "analysisgnn.boolean-v1"),
    _task("metrical_strength", "metrical_strength", "note", _BOTH, ("downbeat",), "analysisgnn.metrical-strength-v1"),
    _task("note_degree", "note_degree", "note", _BOTH, ("note_degree",), "analysisgnn.note-degree-v1"),
    _task("chord_tone", "chord_tone_non_chord_tone", "note", _BOTH, ("tpc_is_in_label", "a_pitchNames", "s_step", "s_alter"), "analysisgnn.boolean-v1"),
    _task("is_root", "is_root", "note", _BOTH, ("tpc_is_root", "a_root", "s_step", "s_alter"), "analysisgnn.boolean-v1"),
    _task("is_bass", "is_bass", "note", _BOTH, ("tpc_is_bass", "a_bass", "s_step", "s_alter"), "analysisgnn.boolean-v1"),
)
TASK_BY_ID = {row.task_id: row for row in PRODUCTION_TASKS}


PINNED_CODE_HEADS = (
    ("cadence", 4, 4), ("localkey", 50, 50), ("tonkey", 50, 50),
    ("quality", 15, 16), ("inversion", 4, 4), ("root", 38, 38),
    ("bass", 38, 38), ("degree1", 22, 22), ("degree2", 22, 22),
    ("hrythm", 2, 2), ("pcset", 94, 94), ("romanNumeral", 185, 184),
    ("section", 2, 2), ("phrase", 2, 2), ("organ_point", 2, 2),
    ("tpc_in_label", 2, 2), ("tpc_is_root", 2, 2),
    ("tpc_is_bass", 2, 2), ("downbeat", 45, 45),
    ("note_degree", 49, 49), ("staff", 4, 4),
)
CODE_TO_CANONICAL = {
    "cadence": "cadence", "localkey": "local_key", "tonkey": "tonicized_key",
    "quality": "quality", "inversion": "inversion", "root": "root",
    "bass": "bass", "degree1": "primary_degree", "degree2": "secondary_degree",
    "hrythm": "harmonic_rhythm", "pcset": "pitch_class_set",
    "romanNumeral": "roman_numeral", "section": "section", "phrase": "phrase",
    "organ_point": "pedal", "tpc_in_label": "chord_tone",
    "tpc_is_root": "is_root", "tpc_is_bass": "is_bass",
    "downbeat": "metrical_strength", "note_degree": "note_degree", "staff": "staff",
}


def _task_status(code_name: str, official: int, literal: int) -> str:
    if code_name == "staff":
        return "code_only"
    if code_name in {"organ_point", "downbeat"}:
        return "alias_normalized"
    if official != literal:
        return "corrected_literal"
    if code_name in {"cadence", "root", "bass", "degree1", "degree2", "pcset"}:
        return "corrected_missing_mask"
    return "exact_match"


def pinned_code_reference_registry() -> dict[str, object]:
    rows = []
    for code_name, official, literal in PINNED_CODE_HEADS:
        canonical = CODE_TO_CANONICAL[code_name]
        task = TASK_BY_ID.get(canonical)
        rows.append(
            {
                "availability": "unavailable" if task is None else "available",
                "canonical_task_id": canonical,
                "mask_policy": "pinned code may coerce missing/out-of-range to class 0",
                "metric": None if task is None else list(task.metric_ids),
                "notes": (
                    "auxiliary staff head; not one of the paper's 20 analytical properties"
                    if code_name == "staff"
                    else "immutable evidence row; corrected production semantics are separate"
                ),
                "observed_literal_count": literal,
                "official_code_class_count": official,
                "source_dialects": [] if task is None else list(task.source_dialects),
                "source_fields": [] if task is None else list(task.source_fields),
                "status": _task_status(code_name, official, literal),
                "task_level": None if task is None else task.prediction_level,
                "task_name_in_code": code_name,
                "task_name_in_paper": None if code_name == "staff" else canonical,
                "corrected_class_count": None if task is None else task.class_count,
            }
        )
    payload: dict[str, object] = {
        "external_commit": ANALYSISGNN_COMMIT,
        "external_repository": ANALYSISGNN_REPOSITORY,
        "head_count": len(rows),
        "paper_task_count": 20,
        "registry_id": PINNED_CODE_REGISTRY_ID,
        "rows": rows,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def production_task_registry() -> dict[str, object]:
    payload: dict[str, object] = {
        "aliases": {
            "downbeat": "metrical_strength",
            "organ_point": "pedal",
            "pedal": "pedal",
        },
        "exact_official_reproduction": False,
        "paper_compatible_metric_subset": [
            "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
        ],
        "production_head_count": len(PRODUCTION_TASKS),
        "registry_id": PRODUCTION_REGISTRY_ID,
        "tasks": [asdict(row) for row in PRODUCTION_TASKS],
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def vocabularies_payload() -> dict[str, object]:
    rows = [
        {
            **asdict(row),
            "aliases": [list(alias) for alias in row.aliases],
            "class_count": row.class_count,
            "classes": [
                {"class_id": class_id, "label": label}
                for class_id, label in enumerate(row.labels)
            ],
            "labels": list(row.labels),
        }
        for row in VOCABULARIES
    ]
    payload: dict[str, object] = {"registry_id": PRODUCTION_REGISTRY_ID, "vocabularies": rows}
    payload["fingerprint"] = fingerprint(payload)
    return payload


def get_vocabulary(vocabulary_id: str) -> VocabularySpec:
    try:
        return VOCABULARY_BY_ID[vocabulary_id]
    except KeyError as exc:
        raise AnalysisGNNMultitaskContractError(
            f"unknown vocabulary ID: {vocabulary_id}"
        ) from exc


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASK_BY_ID[task_id]
    except KeyError as exc:
        raise AnalysisGNNMultitaskContractError(f"unknown task ID: {task_id}") from exc


def validate_static_contract() -> None:
    if len(PRODUCTION_TASKS) != 20 or len(TASK_BY_ID) != len(PRODUCTION_TASKS):
        raise AnalysisGNNMultitaskContractError("production task IDs are not unique")
    if len(VOCABULARIES) != len(VOCABULARY_BY_ID):
        raise AnalysisGNNMultitaskContractError("vocabulary IDs are not unique")
    for task in PRODUCTION_TASKS:
        if task.class_count != get_vocabulary(task.vocabulary_id).class_count:
            raise AnalysisGNNMultitaskContractError(
                f"task/vocabulary class count mismatch: {task.task_id}"
            )
    if len(_QUALITIES) != 17 or len(_ROMAN_NUMERALS) != 184:
        raise AnalysisGNNMultitaskContractError("corrected vocabulary lock changed")


validate_static_contract()


def _counts(records: Sequence[DilemmadataCorpusRecord]) -> dict[str, int]:
    counts = Counter(row.dialect for row in records)
    return {"an_joint": counts["an_joint"], "dlc": counts["dlc"], "total": len(records)}


def full_raw_manifest(discovery: DilemmadataCorpusDiscovery) -> dict[str, object]:
    records = tuple(sorted(discovery.records, key=lambda row: row.record_id))
    counts = _counts(records)
    if counts != EXPECTED_FULL_COUNTS:
        raise AnalysisGNNMultitaskContractError(
            f"full raw universe differs: expected {EXPECTED_FULL_COUNTS}, observed {counts}"
        )
    payload: dict[str, object] = {
        "cadence_external_corpus_available": False,
        "cadence_external_corpus_included": False,
        "corpus_content_fingerprint": discovery.content_fingerprint,
        "counts": counts,
        "record_ids": [row.record_id for row in records],
        "records_fingerprint": fingerprint(
            [[row.record_id, row.dialect, row.raw_projection_sha256] for row in records]
        ),
        "universe_id": FULL_RAW_UNIVERSE_ID,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def overlap_exclusions(
    records: Sequence[DilemmadataCorpusRecord],
) -> tuple[dict[str, object], ...]:
    by_name = {
        f"{row.collection}_{row.piece_name}": row
        for row in records
        if row.dialect == "dlc"
    }
    by_id = {row.record_id: row for row in records}
    rows = []
    for name in OVERLAP_EXCLUSION_NAMES:
        if name not in by_name:
            raise AnalysisGNNMultitaskContractError(f"overlap record missing: {name}")
        dlc = by_name[name]
        an = by_id.get(OVERLAP_AN_PEERS[name])
        if an is None or an.dialect != "an_joint":
            raise AnalysisGNNMultitaskContractError(f"AN overlap peer missing: {name}")
        if dlc.source_group_id != an.source_group_id:
            raise AnalysisGNNMultitaskContractError(
                f"overlap pair has different source component: {name}"
            )
        rows.append(
            {
                "dialect": dlc.dialect,
                "exclusion_reason": "DLC record duplicates an AN source component",
                "external_reference_commit": ANALYSISGNN_COMMIT,
                "matched_an_record_id": an.record_id,
                "record_id": dlc.record_id,
                "source_component_id": dlc.source_group_id,
            }
        )
    return tuple(rows)


def paper_candidate_records(
    records: Sequence[DilemmadataCorpusRecord],
) -> tuple[DilemmadataCorpusRecord, ...]:
    excluded = {row["record_id"] for row in overlap_exclusions(records)}
    selected = tuple(sorted((row for row in records if row.record_id not in excluded), key=lambda row: row.record_id))
    counts = _counts(selected)
    if counts != EXPECTED_PAPER_COUNTS:
        raise AnalysisGNNMultitaskContractError(
            f"paper-candidate universe differs: expected {EXPECTED_PAPER_COUNTS}, observed {counts}"
        )
    if any("monteverdi_madrigals_5-04d" in row.record_id for row in records):
        raise AnalysisGNNMultitaskContractError(
            "unexpected Monteverdi record entered the pinned snapshot"
        )
    return selected


def paper_candidate_manifest(
    records: Sequence[DilemmadataCorpusRecord],
) -> dict[str, object]:
    selected = paper_candidate_records(records)
    payload: dict[str, object] = {
        "attests_official_graph_build_success": False,
        "cadence_external_corpus_available": False,
        "cadence_external_corpus_included": False,
        "counts": _counts(selected),
        "exclusion_count": len(OVERLAP_EXCLUSION_NAMES),
        "record_ids": [row.record_id for row in selected],
        "records_fingerprint": fingerprint(
            [[row.record_id, row.dialect, row.source_group_id] for row in selected]
        ),
        "universe_id": PAPER_CANDIDATE_UNIVERSE_ID,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def source_component_rows(
    records: Sequence[DilemmadataCorpusRecord],
    *,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_group_id].append(record)
    rows = []
    for component_id, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: row.record_id)
        evidence = []
        for record in ordered:
            values = {} if metadata is None else dict(metadata.get(record.record_id, {}))
            evidence.append(
                {
                    "collection": record.collection,
                    "composer": values.get("composer"),
                    "movement": values.get("movement"),
                    "normalized_source_basename": record.piece_name.casefold().replace("_", "-"),
                    "record_id": record.record_id,
                    "score_identity": record.score_sha256,
                    "title": values.get("title"),
                }
            )
        rows.append(
            {
                "component_provenance": (
                    "Phase 9E-B2 deterministic raw grouping plus score identity and "
                    "processing/merged_summary.tsv AN/DLC overlap evidence"
                ),
                "dialects": sorted({row.dialect for row in ordered}),
                "manual_override": False,
                "manual_override_evidence": None,
                "manual_override_reason": None,
                "record_ids": [row.record_id for row in ordered],
                "source_component_id": component_id,
                "source_evidence": evidence,
                "version": SOURCE_COMPONENT_VERSION,
            }
        )
    return tuple(rows)


def _record_availability_from_rows(
    dialect: str, rows: Sequence[Mapping[str, str]]
) -> dict[str, bool]:
    missing = {"", "<NA>", "NA", "NaN", "nan", "None", "null"}
    true = {"1", "True", "true", "TRUE"}

    def present(field: str, *, gate: str | None = None) -> bool:
        return any(
            row.get(field, "").strip() not in missing
            and (gate is None or row.get(gate, "").strip() in true)
            for row in rows
        )

    values = {
        task.task_id: False for task in PRODUCTION_TASKS
    }
    chord_gate = "valid_chord_label"
    for task in PRODUCTION_TASKS:
        if dialect not in task.source_dialects:
            continue
        if task.task_id == "cadence":
            values[task.task_id] = present("cadence_type", gate="valid_cadence_label")
        elif task.task_id == "phrase":
            values[task.task_id] = any(
                row.get("a_phraseend", "").strip() in true
                and row.get("valid_phrase_label", "").strip() in true
                for row in rows
            )
        elif task.task_id == "section":
            values[task.task_id] = any(
                row.get("section_start", "").strip() in true
                and row.get("valid_section_start_label", "").strip() in true
                for row in rows
            )
        elif task.task_id == "pedal":
            values[task.task_id] = any(
                row.get("valid_pedal_point_label", "").strip() in true for row in rows
            )
        elif task.prediction_level == "note" and task.task_id == "metrical_strength":
            values[task.task_id] = present("downbeat")
        elif task.prediction_level == "note" and task.task_id == "note_degree":
            values[task.task_id] = present("note_degree", gate=chord_gate)
        elif task.prediction_level == "note":
            direct = task.source_fields[0]
            values[task.task_id] = (
                present(direct, gate=chord_gate)
                if direct in rows[0]
                else present("a_pitchNames", gate=chord_gate)
            ) if rows else False
        elif task.task_id == "tonicized_key":
            values[task.task_id] = present("a_degree2", gate=chord_gate)
        elif task.task_id == "inversion" and dialect == "dlc":
            values[task.task_id] = present("figbass", gate=chord_gate)
        elif task.task_id == "pitch_class_set" and dialect == "dlc":
            values[task.task_id] = present("chord_tones", gate=chord_gate)
        else:
            values[task.task_id] = present(task.source_fields[0], gate=chord_gate)
    return values


def read_source_rows(record: DilemmadataCorpusRecord) -> tuple[dict[str, str], ...]:
    with record.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AnalysisGNNMultitaskContractError(
                f"invalid source header: {record.record_id}"
            )
        return tuple(dict(row) for row in reader)


def record_task_availability(record: DilemmadataCorpusRecord) -> dict[str, bool]:
    return _record_availability_from_rows(record.dialect, read_source_rows(record))


def stable_split_assignments(
    records: Sequence[DilemmadataCorpusRecord],
    availability: Mapping[str, Mapping[str, bool]],
) -> tuple[SplitAssignment, ...]:
    """Assign whole source components without reading any class value."""

    ordered_records = tuple(sorted(records, key=lambda row: row.record_id))
    if len(ordered_records) != len({row.record_id for row in ordered_records}):
        raise AnalysisGNNMultitaskContractError("duplicate record in split input")
    if not {row.record_id for row in ordered_records} <= set(availability):
        raise AnalysisGNNMultitaskContractError("availability does not cover split input")
    groups: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    for record in ordered_records:
        groups[record.source_group_id].append(record)
    target = {"train": 1295, "validation": 162, "test": 162}
    counts = Counter()
    dialect_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    assigned: dict[str, str] = {}
    group_rows = []
    for component_id, members in groups.items():
        members = sorted(members, key=lambda row: row.record_id)
        group_tasks = {
            task_id
            for member in members
            for task_id, available in availability[member.record_id].items()
            if available
        }
        token = fingerprint(
            {
                "component": component_id,
                "namespace": ASSIGNMENT_NAMESPACE,
                "seed": ASSIGNMENT_SEED,
            }
        )
        group_rows.append((token, component_id, members, group_tasks))
    # The hash order is label-independent. Availability affects only the explicit
    # coverage tie-break below, never the group order or a class-value objective.
    group_rows.sort(key=lambda item: (item[0], item[1]))
    splits = ("train", "validation", "test")
    for _token, component_id, members, group_tasks in group_rows:
        size = len(members)
        dialects = Counter(member.dialect for member in members)

        def score(split: str) -> tuple[float, float, float, str]:
            projected = counts[split] + size
            quota_penalty = abs(projected - target[split]) / target[split]
            overflow = max(0, projected - target[split]) / target[split]
            coverage_gain = sum(task_counts[split][task] == 0 for task in group_tasks)
            coverage_gain += sum(
                dialect_counts[split][dialect] == 0 for dialect in dialects
            )
            return (overflow, quota_penalty, -float(coverage_gain), split)

        chosen = min(splits, key=score)
        assigned[component_id] = chosen
        counts[chosen] += size
        dialect_counts[chosen].update(dialects)
        task_counts[chosen].update(group_tasks)
    rows = tuple(
        SplitAssignment(
            record_id=record.record_id,
            dialect=record.dialect,
            source_component_id=record.source_group_id,
            split=assigned[record.source_group_id],  # type: ignore[arg-type]
        )
        for record in ordered_records
    )
    validate_split(rows)
    return rows


def validate_split(rows: Sequence[SplitAssignment]) -> None:
    if len(rows) != len({row.record_id for row in rows}):
        raise AnalysisGNNMultitaskContractError("duplicate split record")
    component_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        component_splits[row.source_component_id].add(row.split)
    if any(len(splits) != 1 for splits in component_splits.values()):
        raise AnalysisGNNMultitaskContractError("split component overlap")


def split_summary(
    rows: Sequence[SplitAssignment],
    availability: Mapping[str, Mapping[str, bool]],
) -> dict[str, object]:
    validate_split(rows)
    record_counts = Counter(row.split for row in rows)
    component_sets = {
        split: {row.source_component_id for row in rows if row.split == split}
        for split in ("train", "validation", "test")
    }
    leakage = {
        "train_validation": sorted(component_sets["train"] & component_sets["validation"]),
        "train_test": sorted(component_sets["train"] & component_sets["test"]),
        "validation_test": sorted(component_sets["validation"] & component_sets["test"]),
    }
    task_counts = {
        split: {
            task.task_id: sum(
                availability[row.record_id][task.task_id]
                for row in rows if row.split == split
            )
            for task in PRODUCTION_TASKS
        }
        for split in ("train", "validation", "test")
    }
    payload: dict[str, object] = {
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "assignment_namespace": ASSIGNMENT_NAMESPACE,
        "assignment_seed": ASSIGNMENT_SEED,
        "component_counts": {split: len(values) for split, values in component_sets.items()},
        "component_leakage": leakage,
        "dialect_counts": {
            split: dict(Counter(row.dialect for row in rows if row.split == split))
            for split in ("train", "validation", "test")
        },
        "objective": "record-ratio first with deterministic dialect/task-availability coverage",
        "record_counts": dict(record_counts),
        "target_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "task_structural_availability": task_counts,
        "test_assignment_frozen": True,
        "test_metrics_computed": False,
        "test_targets_used_for_model_evaluation": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def test_lock_manifest(assignments: Sequence[SplitAssignment]) -> dict[str, object]:
    frozen = sorted(
        [row.record_id, row.source_component_id]
        for row in assignments if row.split == "test"
    )
    payload: dict[str, object] = {
        "explicit_unlock_required": True,
        "test_assignment_fingerprint": fingerprint(frozen),
        "test_assignment_frozen": True,
        "test_metrics_computed": False,
        "test_targets_used_for_model_evaluation": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def require_test_evaluation_unlock(*, explicit_allow: bool = False) -> None:
    if not explicit_allow:
        raise AnalysisGNNMultitaskContractError(
            "TEST evaluation is locked; pass an explicit evaluation authorization"
        )


def metric_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "joint_metrics": [
            {
                "components": [
                    "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
                ],
                "entity_type": "harmonic_event",
                "metric_id": "roman_numeral_joint_accuracy",
                "row_policy": (
                    "include only a shared harmonic_event_id whose five component "
                    "masks are available and whose values are vocabulary-valid"
                ),
                "undefined_payload": {
                    "accuracy": None,
                    "available": False,
                    "support": 0,
                    "undefined_reason": "no rows satisfy the joint component contract",
                },
            }
        ],
        "paper_compatible_not_exact_official": True,
        "test_evaluation_locked": True,
        "version": METRIC_CONTRACT_VERSION,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def load_fingerprinted_payload(
    path: str | Path,
    *,
    expected_id_key: str,
    expected_id: str,
) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get(expected_id_key) != expected_id:
        raise AnalysisGNNMultitaskContractError("unknown artifact version/registry")
    observed = value.pop("fingerprint", None)
    expected = fingerprint(value)
    value["fingerprint"] = observed
    if observed != expected:
        raise AnalysisGNNMultitaskContractError("artifact fingerprint mismatch")
    return value


def load_dataset_universe(path: str | Path, *, universe_id: str) -> dict[str, object]:
    if universe_id not in {FULL_RAW_UNIVERSE_ID, PAPER_CANDIDATE_UNIVERSE_ID}:
        raise AnalysisGNNMultitaskContractError("unknown dataset universe version")
    return load_fingerprinted_payload(
        path,
        expected_id_key="universe_id",
        expected_id=universe_id,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_assignments(
    path: str | Path,
    *,
    manifest_record_ids: Iterable[str],
    expected_sha256: str,
) -> tuple[SplitAssignment, ...]:
    source = Path(path)
    if _file_sha256(source) != expected_sha256:
        raise AnalysisGNNMultitaskContractError("split assignment fingerprint mismatch")
    rows: list[SplitAssignment] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = SplitAssignment(**json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise AnalysisGNNMultitaskContractError(
                    "split assignment schema invalid"
                ) from exc
            if (
                row.assignment_algorithm != ASSIGNMENT_ALGORITHM
                or row.assignment_namespace != ASSIGNMENT_NAMESPACE
            ):
                raise AnalysisGNNMultitaskContractError(
                    "unknown split assignment algorithm/namespace"
                )
            rows.append(row)
    record_ids = tuple(sorted(set(manifest_record_ids)))
    if tuple(sorted(row.record_id for row in rows)) != record_ids:
        raise AnalysisGNNMultitaskContractError("split contains record outside manifest")
    validate_split(rows)
    return tuple(rows)


def get_entity_mappings(
    sidecar: Mapping[str, object], *, entity_id: str | None = None
) -> tuple[Mapping[str, object], ...]:
    if sidecar.get("schema_version") != TARGET_SIDECAR_VERSION:
        raise AnalysisGNNMultitaskContractError("unknown entity sidecar version")
    relations = sidecar.get("relations")
    if not isinstance(relations, list):
        raise AnalysisGNNMultitaskContractError(
            "expanded sidecar is required to load entity mappings"
        )
    rows = tuple(row for row in relations if isinstance(row, dict))
    if len(rows) != len(relations):
        raise AnalysisGNNMultitaskContractError("entity relation schema invalid")
    if entity_id is None:
        return rows
    return tuple(row for row in rows if row.get("source_entity_id") == entity_id)


def validate_loaded_registry(
    task_registry: Mapping[str, object], vocabularies: Mapping[str, object]
) -> None:
    if task_registry.get("registry_id") != PRODUCTION_REGISTRY_ID:
        raise AnalysisGNNMultitaskContractError("unknown task registry")
    task_rows = task_registry.get("tasks")
    vocabulary_rows = vocabularies.get("vocabularies")
    if not isinstance(task_rows, list) or not isinstance(vocabulary_rows, list):
        raise AnalysisGNNMultitaskContractError("registry rows are absent")
    task_ids = [row.get("task_id") for row in task_rows if isinstance(row, dict)]
    vocabulary_ids = [
        row.get("vocabulary_id") for row in vocabulary_rows if isinstance(row, dict)
    ]
    if len(task_ids) != len(task_rows) or len(task_ids) != len(set(task_ids)):
        raise AnalysisGNNMultitaskContractError("duplicate task ID")
    if len(vocabulary_ids) != len(vocabulary_rows) or len(vocabulary_ids) != len(set(vocabulary_ids)):
        raise AnalysisGNNMultitaskContractError("duplicate vocabulary ID")
    by_id = {row["vocabulary_id"]: row for row in vocabulary_rows}
    for task in task_rows:
        vocabulary = by_id.get(task["vocabulary_id"])
        if vocabulary is None:
            raise AnalysisGNNMultitaskContractError("unknown task vocabulary")
        classes = vocabulary.get("classes")
        if not isinstance(classes, list):
            raise AnalysisGNNMultitaskContractError("vocabulary classes absent")
        class_ids = [row.get("class_id") for row in classes if isinstance(row, dict)]
        if class_ids != list(range(len(classes))) or len(class_ids) != len(set(class_ids)):
            raise AnalysisGNNMultitaskContractError("duplicate/non-contiguous class ID")
        if task["class_count"] != len(classes):
            raise AnalysisGNNMultitaskContractError("vocabulary length mismatch")


_MISSING = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
_TRUE = frozenset({"1", "True", "true", "TRUE"})
_FALSE = frozenset({"0", "False", "false", "FALSE"})
_HARMONIC_TASK_IDS = (
    "local_key",
    "tonicized_key",
    "root",
    "bass",
    "primary_degree",
    "secondary_degree",
    "quality",
    "inversion",
    "roman_numeral",
    "pitch_class_set",
    "harmonic_rhythm",
    "pedal",
)
_NOTE_TASK_IDS = (
    "metrical_strength", "note_degree", "chord_tone", "is_root", "is_bass"
)
_ONSET_TASK_IDS = ("cadence", "phrase", "section")


def _time_payload(value: RationalTime) -> dict[str, int]:
    return {"den": value.den, "num": value.num}


def _entity_id(kind: str, payload: object) -> str:
    return f"{kind}:analysisgnn:{fingerprint(payload)[:32]}"


def _fast_fingerprint(value: object) -> str:
    """Canonical SHA-256 for large sidecars without a redundant tree walk."""

    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bool_label(value: str) -> str | None:
    normalized = value.strip()
    if normalized in _TRUE:
        return "true"
    if normalized in _FALSE:
        return "false"
    return None


def _pcset_label(value: str) -> str:
    normalized = value.strip()
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (tuple, list)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in parsed
    ):
        return ",".join(map(str, parsed))
    return normalized.replace("(", "").replace(")", "").replace(" ", "")


def _spelling(row: Mapping[str, str], dialect: str) -> str | None:
    step_field = "s_step" if dialect == "an_joint" else "step"
    alter_field = "s_alter" if dialect == "an_joint" else "alter"
    step = row.get(step_field, "").strip()
    alter = row.get(alter_field, "").strip()
    if not step or alter in _MISSING:
        return None
    try:
        amount = int(float(alter))
    except ValueError:
        return None
    accidental = "#" * amount if amount >= 0 else "b" * -amount
    return f"{step}{accidental}"


def _pitch_names(value: str) -> frozenset[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, str):
        return frozenset({parsed})
    if isinstance(parsed, (tuple, list, set)):
        return frozenset(str(item) for item in parsed)
    return frozenset(
        item.strip(" '\"") for item in value.strip("()[]{}").split(",") if item.strip()
    )


def _field_and_value(task_id: str, row: Mapping[str, str], dialect: str) -> tuple[str, str]:
    task = get_task(task_id)
    if (
        task_id == "quality"
        and dialect == "dlc"
        and row.get("chord_type", "").strip() in {"+7", "+M7"}
    ):
        field = "chord_type"
    elif task_id == "pitch_class_set":
        field = "a_pcset" if "a_pcset" in row else "chord_tones"
    elif task_id == "cadence":
        field = "cadence_type"
    elif task_id == "phrase":
        field = "a_phraseend"
    elif task_id == "section":
        field = "section_start"
    elif task_id in {"chord_tone", "is_root", "is_bass"} and dialect == "an_joint":
        field = task.source_fields[1]
    else:
        field = task.source_fields[0]
    return field, row.get(field, "").strip()


def _gate(task_id: str, row: Mapping[str, str]) -> tuple[bool, str | None]:
    if task_id == "metrical_strength":
        return True, None
    gate = {
        "cadence": "valid_cadence_label",
        "phrase": "valid_phrase_label",
        "section": "valid_section_start_label",
        "pedal": "valid_pedal_point_label",
    }.get(task_id, "valid_chord_label")
    raw = row.get(gate, "").strip()
    if raw in _TRUE:
        if task_id in {"phrase", "section"}:
            field = "a_phraseend" if task_id == "phrase" else "section_start"
            if row.get(field, "").strip() not in _TRUE:
                return False, "positive_unlabeled_absence"
        if task_id == "tonicized_key" and row.get("a_degree2", "").strip() in _MISSING:
            return False, "secondary_degree_missing"
        if task_id == "inversion" and "figbass" in row and row.get("figbass", "").strip() in _MISSING:
            return False, "source_figbass_missing"
        return True, None
    if raw in _FALSE or raw in _MISSING:
        return False, f"{gate}_false_or_missing"
    return False, f"{gate}_invalid"


def _canonical_value(task_id: str, value: str, row: Mapping[str, str], dialect: str) -> str | None:
    if task_id in {"harmonic_rhythm", "phrase", "section"}:
        candidate = _bool_label(value)
    elif task_id == "pedal":
        candidate = "false" if value in _MISSING else "true"
    elif task_id in {"chord_tone", "is_root", "is_bass"}:
        if dialect == "dlc":
            candidate = _bool_label(value)
        else:
            spelling = _spelling(row, dialect)
            if spelling is None:
                candidate = None
            elif task_id == "chord_tone":
                candidate = str(spelling in _pitch_names(row.get("a_pitchNames", ""))).lower()
            else:
                comparison = row.get("a_root" if task_id == "is_root" else "a_bass", "").strip()
                candidate = str(spelling == comparison).lower() if comparison not in _MISSING else None
    elif task_id == "pitch_class_set":
        candidate = _pcset_label(value)
    else:
        candidate = value
    if candidate is None:
        return None
    return get_vocabulary(get_task(task_id).vocabulary_id).normalize(candidate)


def _target_state(
    task_id: str,
    row: SourceTargetRow,
    dialect: str,
    *,
    source_entity_id: str,
    canonical_entity_id: str,
) -> TargetState:
    task = get_task(task_id)
    if dialect not in task.source_dialects:
        return TargetState(
            available=False,
            masked=True,
            missing_reason="unsupported_dialect",
            source_value=None,
            canonical_value=None,
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            provenance={"dialect": dialect, "source_fields": list(task.source_fields)},
        )
    if row.repair_mask_scope == "all" or (
        row.repair_mask_scope == "note" and task.prediction_level == "note"
    ):
        return TargetState(
            available=False,
            masked=True,
            missing_reason=f"raw_repair_mask_scope_{row.repair_mask_scope}",
            source_value=None,
            canonical_value=None,
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            provenance={"repair_lineage_applied": True, "source_row_ordinal": row.ordinal},
        )
    gate, gate_reason = _gate(task_id, row.values)
    if not gate:
        return TargetState(
            available=False,
            masked=True,
            missing_reason=gate_reason,
            source_value=None,
            canonical_value=None,
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            provenance={"repair_lineage_applied": False, "source_row_ordinal": row.ordinal},
        )
    field, raw = _field_and_value(task_id, row.values, dialect)
    # A valid pedal gate makes blank/None an explicit negative; other blanks are missing.
    if raw in _MISSING and task_id != "pedal":
        return TargetState(
            available=False,
            masked=True,
            missing_reason="source_value_missing",
            source_value=None,
            canonical_value=None,
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            provenance={"source_field": field, "source_row_ordinal": row.ordinal},
        )
    canonical = _canonical_value(task_id, raw, row.values, dialect)
    if canonical is None:
        return TargetState(
            available=False,
            masked=True,
            missing_reason="unknown_value_not_in_pinned_production_vocabulary",
            source_value=raw or None,
            canonical_value=None,
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            provenance={"source_field": field, "source_row_ordinal": row.ordinal},
        )
    return TargetState(
        available=True,
        masked=False,
        missing_reason=None,
        source_value=raw,
        canonical_value=canonical,
        source_entity_id=source_entity_id,
        canonical_entity_id=canonical_entity_id,
        provenance={
            "dialect": dialect,
            "source_field": field,
            "source_row_ordinal": row.ordinal,
            "target_sidecar_version": TARGET_SIDECAR_VERSION,
        },
    )


def source_target_rows(accepted: DilemmadataAccepted) -> tuple[SourceTargetRow, ...]:
    record = accepted.record
    with record.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        raw_rows = tuple(dict(row) for row in reader)
    bindings = accepted.alignment_evidence.rows
    if len(raw_rows) != len(bindings):
        raise AnalysisGNNMultitaskContractError("source-row binding length mismatch")
    return tuple(
        SourceTargetRow(
            ordinal=binding.ordinal,
            line=binding.line,
            onset_qn=binding.onset_qn,
            canonical_note_id=binding.canonical_note_id,
            repair_mask_scope=binding.repair_mask_scope,
            values=raw,
        )
        for raw, binding in zip(raw_rows, bindings, strict=True)
    )


def _interval_index(
    piece: CanonicalPiece, kind: Literal["beat", "measure"]
) -> tuple[tuple[RationalTime, ...], tuple[tuple[RationalTime, str], ...]]:
    values = piece.beats if kind == "beat" else piece.bars
    rows = tuple(
        (
            value.start_qn + value.duration_qn,
            value.beat_id if kind == "beat" else value.bar_id,
        )
        for value in values
    )
    return tuple(value.start_qn for value in values), rows


def _interval_entity_id(
    onset: RationalTime,
    index: tuple[tuple[RationalTime, ...], tuple[tuple[RationalTime, str], ...]],
) -> str | None:
    starts, rows = index
    offset = bisect_right(starts, onset) - 1
    if offset < 0:
        return None
    end, entity_id = rows[offset]
    return entity_id if onset < end else None


def materialize_target_sidecar(accepted: DilemmadataAccepted) -> dict[str, object]:
    """Materialize shared entities and independent per-head masks for one record."""

    record = accepted.record
    piece = accepted.piece
    rows = source_target_rows(accepted)
    beat_index = _interval_index(piece, "beat")
    measure_index = _interval_index(piece, "measure")
    note_binding: dict[str, SourceTargetRow] = {}
    onset_rows: dict[RationalTime, list[SourceTargetRow]] = defaultdict(list)
    entity_rows: list[dict[str, object]] = []
    relations: list[dict[str, object]] = []
    harmonic_by_ordinal: dict[int, str] = {}
    source_identity_field = (
        "a_annotationNumber" if record.dialect == "an_joint" else "unfolded_harmony_index"
    )
    harmonic_groups: dict[
        tuple[str, RationalTime, int | None], list[SourceTargetRow]
    ] = defaultdict(list)
    for row in rows:
        onset_rows[row.onset_qn].append(row)
        table_identity = row.values.get(source_identity_field, "").strip() or "missing"
        # A missing annotation identity cannot merge unrelated source rows.
        fallback_ordinal = row.ordinal if table_identity == "missing" else None
        harmonic_groups[(table_identity, row.onset_qn, fallback_ordinal)].append(row)
        if row.canonical_note_id is not None and row.canonical_note_id not in note_binding:
            note_binding[row.canonical_note_id] = row
    for (table_identity, onset, _fallback), candidates in sorted(
        harmonic_groups.items(),
        key=lambda item: (item[0][1], item[0][0], -1 if item[0][2] is None else item[0][2]),
    ):
        row = min(candidates, key=lambda item: item.ordinal)
        source_entity_id = (
            f"{record.record_id}:{source_identity_field}:{table_identity}:row:{row.ordinal}"
        )
        harmonic_id = _entity_id(
            "harmonic-event",
            [
                record.record_id,
                record.dialect,
                source_identity_field,
                table_identity,
                row.ordinal,
                _time_payload(onset),
            ],
        )
        for candidate in candidates:
            harmonic_by_ordinal[candidate.ordinal] = harmonic_id
        targets = {
            task_id: asdict(
                _target_state(
                    task_id,
                    row,
                    record.dialect,
                    source_entity_id=source_entity_id,
                    canonical_entity_id=harmonic_id,
                )
            )
            for task_id in _HARMONIC_TASK_IDS
        }
        beat_id = _interval_entity_id(onset, beat_index)
        measure_id = _interval_entity_id(onset, measure_index)
        entity_rows.append(
            {
                "canonical_entity_id": harmonic_id,
                "entity_type": "harmonic_event",
                "onset_qn": _time_payload(onset),
                "source_entity_id": source_entity_id,
                "source_row_ordinal": row.ordinal,
                "source_row_ordinals": [item.ordinal for item in candidates],
                "targets": targets,
            }
        )
        for target_type, target_id in (("beat", beat_id), ("measure", measure_id)):
            if target_id is not None:
                relations.append(
                    {
                        "relation": f"harmonic_event_to_{target_type}",
                        "source_entity_id": harmonic_id,
                        "target_entity_id": target_id,
                    }
                )
    onset_ids: dict[RationalTime, str] = {}
    for onset, candidates in sorted(onset_rows.items()):
        onset_id = _entity_id("onset", [record.record_id, _time_payload(onset)])
        onset_ids[onset] = onset_id
        representative = min(candidates, key=lambda row: row.ordinal)
        source_id = f"{record.record_id}:onset:{onset.num}/{onset.den}"
        targets = {
            task_id: asdict(
                _target_state(
                    task_id,
                    representative,
                    record.dialect,
                    source_entity_id=source_id,
                    canonical_entity_id=onset_id,
                )
            )
            for task_id in _ONSET_TASK_IDS
        }
        entity_rows.append(
            {
                "canonical_entity_id": onset_id,
                "entity_type": "onset",
                "onset_qn": _time_payload(onset),
                "source_entity_id": source_id,
                "targets": targets,
            }
        )
        beat_id = _interval_entity_id(onset, beat_index)
        if beat_id is not None:
            relations.append(
                {
                    "relation": "onset_to_beat",
                    "source_entity_id": onset_id,
                    "target_entity_id": beat_id,
                }
            )
    note_by_id = {note.note_id: note for note in piece.notes}
    for note_id, note in sorted(note_by_id.items()):
        row = note_binding.get(note_id)
        source_id = f"{record.record_id}:canonical-note:{note_id}"
        if row is None:
            targets = {
                task_id: asdict(
                    TargetState(
                        available=False,
                        masked=True,
                        missing_reason="canonical_note_has_no_unambiguous_source_row",
                        source_value=None,
                        canonical_value=None,
                        source_entity_id=source_id,
                        canonical_entity_id=note_id,
                        provenance={"repair_lineage_applied": True},
                    )
                )
                for task_id in _NOTE_TASK_IDS
            }
        else:
            targets = {
                task_id: asdict(
                    _target_state(
                        task_id,
                        row,
                        record.dialect,
                        source_entity_id=source_id,
                        canonical_entity_id=note_id,
                    )
                )
                for task_id in _NOTE_TASK_IDS
            }
        entity_rows.append(
            {
                "canonical_entity_id": note_id,
                "entity_type": "note",
                "onset_qn": _time_payload(note.onset_qn),
                "source_entity_id": source_id,
                "targets": targets,
            }
        )
        onset_id = onset_ids[note.onset_qn]
        relations.append(
            {
                "relation": "note_to_onset",
                "source_entity_id": note_id,
                "target_entity_id": onset_id,
            }
        )
        if row is not None:
            relations.append(
                {
                    "relation": "note_to_harmonic_event",
                    "source_entity_id": note_id,
                    "target_entity_id": harmonic_by_ordinal[row.ordinal],
                }
            )
    for beat in piece.beats:
        relations.append(
            {
                "relation": "beat_to_measure",
                "source_entity_id": beat.beat_id,
                "target_entity_id": beat.bar_id,
            }
        )
    entity_rows.sort(key=lambda row: (str(row["entity_type"]), str(row["canonical_entity_id"])))
    relations.sort(key=lambda row: (str(row["relation"]), str(row["source_entity_id"]), str(row["target_entity_id"])))
    seen = [str(row["canonical_entity_id"]) for row in entity_rows]
    if len(seen) != len(set(seen)):
        raise AnalysisGNNMultitaskContractError("duplicate canonical entity ID")
    payload: dict[str, object] = {
        "dialect": record.dialect,
        "entities": entity_rows,
        "entity_counts": dict(Counter(str(row["entity_type"]) for row in entity_rows)),
        "record_id": record.record_id,
        "relations": relations,
        "relation_counts": dict(Counter(str(row["relation"]) for row in relations)),
        "repair_evidence_fingerprint": (
            None if accepted.repair_evidence is None else accepted.repair_evidence.fingerprint
        ),
        "schema_version": TARGET_SIDECAR_VERSION,
        "source_component_id": record.source_group_id,
    }
    payload["fingerprint"] = _fast_fingerprint(payload)
    return payload


def sidecar_contract_counts(sidecar: Mapping[str, object]) -> dict[str, object]:
    entities = sidecar["entities"]
    if not isinstance(entities, list):
        raise AnalysisGNNMultitaskContractError("sidecar entity rows absent")
    states: dict[str, Counter[str]] = defaultdict(Counter)
    joint_support = 0
    joint_components = {
        "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
    }
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        targets = entity.get("targets")
        if not isinstance(targets, dict):
            continue
        for task_id, raw_state in targets.items():
            if not isinstance(raw_state, dict):
                continue
            state = "available" if raw_state.get("available") is True else str(raw_state.get("missing_reason"))
            states[str(task_id)][state] += 1
        if entity.get("entity_type") == "harmonic_event" and joint_components <= set(targets):
            joint_support += int(
                all(
                    isinstance(targets[task_id], dict)
                    and targets[task_id].get("available") is True
                    and targets[task_id].get("masked") is False
                    for task_id in joint_components
                )
            )
    return {
        "joint_structural_support": joint_support,
        "task_states": {task_id: dict(counts) for task_id, counts in sorted(states.items())},
    }


def materialize_target_sidecar_descriptor(
    accepted: DilemmadataAccepted,
) -> dict[str, object]:
    """Stream a complete logical sidecar into content-addressed audit evidence.

    The descriptor records counts plus independent hashes of every entity,
    relation, and task-state row.  It is the corpus-audit encoding; callers that
    need the expanded rows use :func:`materialize_target_sidecar`.
    """

    record = accepted.record
    piece = accepted.piece
    rows = source_target_rows(accepted)
    beat_index = _interval_index(piece, "beat")
    measure_index = _interval_index(piece, "measure")
    source_identity_field = (
        "a_annotationNumber" if record.dialect == "an_joint" else "unfolded_harmony_index"
    )
    harmonic_groups: dict[
        tuple[str, RationalTime, int | None], list[SourceTargetRow]
    ] = defaultdict(list)
    onset_rows: dict[RationalTime, list[SourceTargetRow]] = defaultdict(list)
    note_binding: dict[str, SourceTargetRow] = {}
    for row in rows:
        onset_rows[row.onset_qn].append(row)
        table_identity = row.values.get(source_identity_field, "").strip() or "missing"
        fallback = row.ordinal if table_identity == "missing" else None
        harmonic_groups[(table_identity, row.onset_qn, fallback)].append(row)
        if row.canonical_note_id is not None and row.canonical_note_id not in note_binding:
            note_binding[row.canonical_note_id] = row

    entity_digest = sha256(b"analysisgnn-b3-entities-v1\0")
    relation_digest = sha256(b"analysisgnn-b3-relations-v1\0")
    target_digest = sha256(b"analysisgnn-b3-target-states-v1\0")
    entity_counts = Counter()
    relation_counts = Counter()
    task_states: dict[str, Counter[str]] = defaultdict(Counter)
    joint_support = 0
    harmonic_by_ordinal: dict[int, str] = {}
    onset_ids: dict[RationalTime, str] = {}

    def update(digest, value: object) -> None:
        digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")

    def observe_target(task_id: str, state: TargetState) -> None:
        label = "available" if state.available else str(state.missing_reason)
        task_states[task_id][label] += 1
        update(target_digest, {"task_id": task_id, **asdict(state)})

    def observe_relation(relation: str, source: str, target: str | None) -> None:
        if target is None:
            return
        relation_counts[relation] += 1
        update(
            relation_digest,
            {
                "relation": relation,
                "source_entity_id": source,
                "target_entity_id": target,
            },
        )

    joint_components = (
        "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
    )
    for (table_identity, onset, _fallback), candidates in sorted(
        harmonic_groups.items(),
        key=lambda item: (item[0][1], item[0][0], -1 if item[0][2] is None else item[0][2]),
    ):
        row = min(candidates, key=lambda item: item.ordinal)
        source_id = (
            f"{record.record_id}:{source_identity_field}:{table_identity}:row:{row.ordinal}"
        )
        entity_id = _entity_id(
            "harmonic-event",
            [
                record.record_id,
                record.dialect,
                source_identity_field,
                table_identity,
                row.ordinal,
                _time_payload(onset),
            ],
        )
        for candidate in candidates:
            harmonic_by_ordinal[candidate.ordinal] = entity_id
        entity_counts["harmonic_event"] += 1
        update(
            entity_digest,
            {
                "canonical_entity_id": entity_id,
                "entity_type": "harmonic_event",
                "onset_qn": _time_payload(onset),
                "source_entity_id": source_id,
                "source_row_ordinal": row.ordinal,
                "source_row_ordinals": [item.ordinal for item in candidates],
            },
        )
        available: dict[str, bool] = {}
        for task_id in _HARMONIC_TASK_IDS:
            state = _target_state(
                task_id,
                row,
                record.dialect,
                source_entity_id=source_id,
                canonical_entity_id=entity_id,
            )
            observe_target(task_id, state)
            available[task_id] = state.available and not state.masked
        joint_support += int(all(available[task_id] for task_id in joint_components))
        observe_relation(
            "harmonic_event_to_beat",
            entity_id,
            _interval_entity_id(onset, beat_index),
        )
        observe_relation(
            "harmonic_event_to_measure",
            entity_id,
            _interval_entity_id(onset, measure_index),
        )

    for onset, candidates in sorted(onset_rows.items()):
        entity_id = _entity_id("onset", [record.record_id, _time_payload(onset)])
        onset_ids[onset] = entity_id
        source_id = f"{record.record_id}:onset:{onset.num}/{onset.den}"
        representative = min(candidates, key=lambda item: item.ordinal)
        entity_counts["onset"] += 1
        update(
            entity_digest,
            {
                "canonical_entity_id": entity_id,
                "entity_type": "onset",
                "onset_qn": _time_payload(onset),
                "source_entity_id": source_id,
            },
        )
        for task_id in _ONSET_TASK_IDS:
            observe_target(
                task_id,
                _target_state(
                    task_id,
                    representative,
                    record.dialect,
                    source_entity_id=source_id,
                    canonical_entity_id=entity_id,
                ),
            )
        observe_relation(
            "onset_to_beat",
            entity_id,
            _interval_entity_id(onset, beat_index),
        )

    for note in sorted(piece.notes, key=lambda item: item.note_id):
        row = note_binding.get(note.note_id)
        source_id = f"{record.record_id}:canonical-note:{note.note_id}"
        entity_counts["note"] += 1
        update(
            entity_digest,
            {
                "canonical_entity_id": note.note_id,
                "entity_type": "note",
                "onset_qn": _time_payload(note.onset_qn),
                "source_entity_id": source_id,
            },
        )
        for task_id in _NOTE_TASK_IDS:
            if row is None:
                state = TargetState(
                    available=False,
                    masked=True,
                    missing_reason="canonical_note_has_no_unambiguous_source_row",
                    source_value=None,
                    canonical_value=None,
                    source_entity_id=source_id,
                    canonical_entity_id=note.note_id,
                    provenance={"repair_lineage_applied": True},
                )
            else:
                state = _target_state(
                    task_id,
                    row,
                    record.dialect,
                    source_entity_id=source_id,
                    canonical_entity_id=note.note_id,
                )
            observe_target(task_id, state)
        observe_relation("note_to_onset", note.note_id, onset_ids.get(note.onset_qn))
        if row is not None:
            observe_relation(
                "note_to_harmonic_event",
                note.note_id,
                harmonic_by_ordinal.get(row.ordinal),
            )
    for beat in sorted(piece.beats, key=lambda item: item.beat_id):
        observe_relation("beat_to_measure", beat.beat_id, beat.bar_id)

    semantic = {
        "dialect": record.dialect,
        "entities_fingerprint": entity_digest.hexdigest(),
        "entity_counts": dict(entity_counts),
        "joint_structural_support": joint_support,
        "record_id": record.record_id,
        "relations_fingerprint": relation_digest.hexdigest(),
        "relation_counts": dict(relation_counts),
        "repair_evidence_fingerprint": (
            None if accepted.repair_evidence is None else accepted.repair_evidence.fingerprint
        ),
        "schema_version": TARGET_SIDECAR_VERSION,
        "source_component_id": record.source_group_id,
        "target_states_fingerprint": target_digest.hexdigest(),
        "task_states": {
            task_id: dict(counts) for task_id, counts in sorted(task_states.items())
        },
    }
    return {**semantic, "fingerprint": _fast_fingerprint(semantic)}
