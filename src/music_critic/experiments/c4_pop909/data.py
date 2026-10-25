"""Fingerprint-bound POP909-CL inputs and independent source-native targets.

This training boundary never admits TEST payloads. Raw score channel 0 alone
creates graph nodes; channel 1 provides targets after raw candidates exist.
Root/bass supervise pitch-class marginals of the unchanged spelled C3 heads.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig, Pop909ClCorpusRecord, Pop909ClQuarantine,
    convert_pop909_cl_file, pop909_cl_piece_id, pop909_cl_raw_input_group_id,
    pop909_lineage_group_id,
)
from music_critic.adapters.pop909_cl_frozen_source import read_raw_source
from music_critic.data import RationalTime
from music_critic.data.raw_time_readout_slots import build_readout_slots
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.temporal_slices.graph import (
    TemporalGraphBatch, build_temporal_batch, _build_temporal_batch_from_validated_pieces,
)
from music_critic.experiments.analysisgnn.transposition import valid_shift_for_midi
from music_critic.experiments.pop909_cl_frozen.evaluation import align_event_start
from music_critic.experiments.pop909_cl_frozen.projection import quality_gold_class_id
from music_critic.tasks import loads_corpus_index, loads_split_manifest, validate_split_manifest

DATA_SCHEMA = "C4Pop909NativeData@1.0.0"
TARGET_SCHEMA = "C4Pop909NativeTargets@1.0.0"
SUPERVISED_HEADS = ("root", "bass", "quality", "inversion", "pitch_class_set")
_INTERVALS = {
    "M": (0, 4, 7), "m": (0, 3, 7), "o": (0, 3, 6), "+": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "D7": (0, 4, 7, 10),
    "M7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "/o7": (0, 3, 6, 10),
    "o7": (0, 3, 6, 9), "mM7": (0, 3, 7, 11), "+7": (0, 4, 8, 10),
}


@dataclass(frozen=True)
class PreparedRecord:
    record_id: str
    split: str
    graph: TemporalGraphBatch
    target_rows: tuple[dict, ...]
    source_binding: dict
    valid_shifts: tuple[int, ...]


def _gate_split(split: str) -> None:
    if isinstance(split, str) and split.lower() == "test":
        raise ValueError("c4_pop909.test_access_forbidden")
    if split not in ("train", "validation"):
        raise ValueError("c4_pop909.train_or_validation_required")


def _file(path: Path) -> tuple[bytes, dict]:
    payload = Path(path).read_bytes()
    return payload, {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}


def load_population(index_path: Path, split_path: Path, companion_indices=()) -> dict:
    """Validate complete cross-index split closure using metadata only.

    TEST identities may occur in this metadata manifest but cannot pass any
    payload reader. Companion indices are necessary when the global manifest
    includes other corpora: split validation must see its entire grouping set.
    """
    indices, identities = [], []
    for path in (index_path, *companion_indices):
        payload, identity = _file(Path(path))
        index = loads_corpus_index(payload.decode("utf-8"))
        indices.append(index)
        identities.append({**identity, "dataset_id": index.header.dataset_id,
                           "index_fingerprint": index.header.index_fingerprint})
    payload, split_identity = _file(Path(split_path))
    split = loads_split_manifest(payload.decode("utf-8"))
    validate_split_manifest(split, tuple(indices))
    pop = [index for index in indices if index.header.dataset_id == "pop909_cl"]
    if len(pop) != 1:
        raise ValueError("c4_pop909.one_pop_index_required")
    assignments = {row.piece_id: row for row in split.assignments
                   if row.dataset_id == "pop909_cl"}
    records = []
    for row in sorted(pop[0].records, key=lambda item: item.piece_id):
        assignment = assignments[row.piece_id]
        relative = PurePosixPath(row.source_relative_path)
        projection = row.source_group_id.removeprefix("pop909-cl-score:")
        if (not row.source_relative_path or relative.is_absolute() or ".." in relative.parts
                or "\\" in row.source_relative_path
                or not re.fullmatch(r"[0-9]{3}", row.source_identity)
                or row.source_identity == "172"
                or row.piece_id != pop909_cl_piece_id(row.source_identity)
                or row.source_group_id != pop909_cl_raw_input_group_id(projection)
                or row.lineage_group_id != pop909_lineage_group_id(row.source_identity)
                or assignment.split not in ("train", "validation", "test")):
            raise ValueError("c4_pop909.invalid_source_identity_or_split")
        records.append(dict(record_id=row.piece_id, piece_id=row.piece_id,
            song_id=row.source_identity, split=assignment.split,
            source_relative_path=row.source_relative_path, source_sha256=row.source_sha256,
            score_projection_sha256=projection, source_group_id=row.source_group_id,
            lineage_group_id=row.lineage_group_id,
            component_fingerprint=assignment.component_fingerprint))
    if not records or set(assignments) != {row["record_id"] for row in records}:
        raise ValueError("c4_pop909.complete_population_required")
    body = dict(schema_version=DATA_SCHEMA, records=records,
        index_fingerprint=pop[0].header.index_fingerprint,
        split_fingerprint=split.manifest_fingerprint,
        indices=sorted(identities, key=lambda row: (row["dataset_id"], row["sha256"])),
        split_artifact=split_identity,
        split_counts=dict(Counter(row["split"] for row in records)),
        target_schema=TARGET_SCHEMA, test_payload_access=False)
    return {**body, "binding_fingerprint": fingerprint(body)}


def _bound_row(population: dict, record_id: str, split: str) -> dict:
    _gate_split(split)  # Before any metadata or source-path resolution.
    body = {key: value for key, value in population.items() if key != "binding_fingerprint"}
    if (population.get("schema_version") != DATA_SCHEMA
            or population.get("binding_fingerprint") != fingerprint(body)):
        raise ValueError("c4_pop909.population_binding_mismatch")
    rows = [row for row in population["records"] if row["record_id"] == record_id]
    if len(rows) != 1 or rows[0]["split"] != split:
        raise ValueError("c4_pop909.record_not_in_requested_split")
    return rows[0]


def read_source_record(population: dict, record_id: str, source_root: Path, *, split: str):
    """Return raw piece, native evidence and binding without allocating a graph."""
    row = _bound_row(population, record_id, split)
    root = Path(source_root).resolve()
    path = (root / row["source_relative_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("c4_pop909.source_path_escapes_root")
    source = Pop909ClCorpusRecord(row["song_id"], path, row["source_relative_path"],
        path.name, row["source_sha256"], row["source_group_id"], row["lineage_group_id"])
    # Both readers independently authenticate bytes. The first never extracts
    # chord labels, and provides the only piece allowed to create model inputs.
    piece = read_raw_source(source, split=split,
                            expected_projection_sha256=row["score_projection_sha256"])
    converted = convert_pop909_cl_file(source, config=Pop909ClAdapterConfig(include_targets=False))
    if (isinstance(converted, Pop909ClQuarantine)
            or converted.score_projection_sha256 != row["score_projection_sha256"]
            or converted.piece.piece_id != row["piece_id"]):
        raise ValueError("c4_pop909.target_source_projection_mismatch")
    return piece, converted.chord_evidence, deepcopy(row)


def _state(value=None, *, source_value=None, reason="source_unavailable", provenance=None):
    available = value is not None
    return dict(available=available, masked=not available, value=value,
                source_value=source_value, reason=None if available else reason,
                provenance=provenance or {})


def build_target_rows(piece, evidence) -> tuple[dict, ...]:
    """Bind source events to existing exact slices and half-open raw beats.

    Native semitone inversion becomes an ordinal only for an exact compatible
    template. Root/quality ambiguity masks remain independent. Bass is the
    observed native lowest pitch class, including root-ambiguous blocks.
    """
    slots = dict(build_readout_slots(piece).time_slots)
    pcset_labels = get_vocabulary(TASK_BY_ID["pitch_class_set"].vocabulary_id).labels
    rows = []
    for block in evidence.blocks:
        time = RationalTime(block.onset_tick, block.ppqn)
        beat_index, alignment_reason = align_event_start(time, piece.beats)
        selected = block.selected_candidate
        qualities = {candidate.quality for candidate in block.candidates}
        if (block.root_available and selected is None
                or block.inversion_available and selected is None
                or block.quality_available and len(qualities) != 1):
            raise ValueError("c4_pop909.inconsistent_native_mask")
        native_quality = next(iter(qualities)) if block.quality_available else None
        quality_id = quality_gold_class_id(native_quality) if native_quality is not None else None
        inversion = None
        if block.inversion_available and selected is not None:
            intervals = _INTERVALS.get(selected.quality, ())
            if selected.inversion_semitones in intervals:
                inversion = intervals.index(selected.inversion_semitones)
        pcset = ",".join(str(pc) for pc in sorted(set(block.pitch_class_set)))
        pcset_id = pcset_labels.index(pcset) if pcset in pcset_labels else None
        provenance = dict(source_block_id=block.block_id, source_sha256=block.source_sha256,
            source_channel=block.source_channel, normalization_status=block.normalization_status,
            pairing_anomaly_ids=list(block.pairing_anomaly_ids),
            target_schema=TARGET_SCHEMA)
        targets = {
            "root": _state(selected.root_pc if block.root_available else None,
                source_value=selected.root if selected and block.root_available else None,
                reason=block.normalization_status, provenance=provenance),
            "bass": _state(block.bass_pitch_class, source_value=block.lowest_source_pitch,
                provenance=provenance),
            "quality": _state(quality_id, source_value=native_quality,
                reason="outside_frozen_subset9" if native_quality is not None else block.normalization_status,
                provenance=provenance),
            "inversion": _state(inversion,
                source_value=selected.inversion_semitones if selected and block.inversion_available else None,
                reason="unavailable_or_incompatible_native_inversion", provenance=provenance),
            "pitch_class_set": _state(pcset_id, source_value=pcset,
                reason="outside_existing_pcset_vocabulary", provenance=provenance),
        }
        rows.append(dict(event_id=block.block_id,
            onset_qn={"num": time.num, "den": time.den},
            slice_id=slots.get(time.to_fraction()),
            beat_id=None if beat_index is None else piece.beats[beat_index].beat_id,
            beat_alignment_reason=alignment_reason, targets=targets))
    if evidence.status == "expected_missing" and rows:
        raise ValueError("c4_pop909.missing_source_with_events")
    return tuple(rows)


def eligible_shifts(piece) -> tuple[int, ...]:
    """Historical signed representatives in all twelve pitch classes; raw only."""
    pitches = tuple(note.pitch for note in piece.notes)
    return tuple(shift for shift in range(12) if valid_shift_for_midi(pitches, shift))


def transpose_target_rows(rows, shift_pc: int) -> tuple[dict, ...]:
    if type(shift_pc) is not int or not 0 <= shift_pc < 12:
        raise ValueError("c4_pop909.invalid_transposition")
    result = deepcopy(tuple(rows))
    pcset_labels = get_vocabulary(TASK_BY_ID["pitch_class_set"].vocabulary_id).labels
    for row in result:
        for head in ("root", "bass"):
            state = row["targets"][head]
            if state["available"]:
                state["value"] = (state["value"] + shift_pc) % 12
        state = row["targets"]["pitch_class_set"]
        if state["available"]:
            original = pcset_labels[state["value"]]
            shifted = ",".join(map(str, sorted({(int(pc) + shift_pc) % 12 for pc in original.split(",")})))
            if shifted in pcset_labels:
                state["value"] = pcset_labels.index(shifted)
            else:
                state.update(value=None, available=False, masked=True,
                             reason="transposed_pcset_outside_existing_vocabulary")
        row["applied_shift_pc"] = shift_pc
    return result


def prepare_record(piece, evidence, source_binding: dict) -> PreparedRecord:
    """Public canonical-object path: validate input even when caller claims trust."""
    return _prepare_record(piece, evidence, source_binding, validated_source=False)


def _prepare_record(piece, evidence, source_binding: dict, *, validated_source: bool) -> PreparedRecord:
    _gate_split(source_binding["split"])
    if (piece.piece_id != source_binding["piece_id"]
            or piece.source_group_id != source_binding["source_group_id"]
            or piece.split not in (None, source_binding["split"])
            or piece.targets or piece.annotations):
        raise ValueError("c4_pop909.raw_piece_binding_mismatch")
    if any(block.source_sha256 != source_binding["source_sha256"]
           or block.source_channel != 1 for block in evidence.blocks):
        raise ValueError("c4_pop909.target_source_binding_mismatch")
    build = _build_temporal_batch_from_validated_pieces if validated_source else build_temporal_batch
    graph = build((piece,))
    rows = build_target_rows(piece, evidence)
    binding = {**deepcopy(source_binding), "target_schema": TARGET_SCHEMA,
               "target_fingerprint": fingerprint(rows), "source_status": evidence.status}
    return PreparedRecord(source_binding["record_id"], source_binding["split"],
                          graph, rows, binding, eligible_shifts(piece))


def read_record(population: dict, record_id: str, source_root: Path, *, split: str) -> PreparedRecord:
    # read_raw_source authenticates bytes and load_midi_piece runs validate_piece
    # before returning. Repeating that canonical validation is unnecessary here;
    # raw and temporal graph validation remain enabled in the private builder.
    return _prepare_record(*read_source_record(population, record_id, source_root, split=split),
                           validated_source=True)
