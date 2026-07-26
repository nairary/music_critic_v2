from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import mido
import pytest

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    Pop909ClCorpusRecord,
    convert_hooktheory_record,
    convert_pop909_cl_file,
)
from music_critic.graph import build_raw_graph, graph_fingerprint, validate_raw_graph
from music_critic.tasks import (
    CROSSWALKS,
    TARGET_FAMILIES,
    TARGET_ONTOLOGY_VERSION,
    BatchTarget,
    GroupAssignment,
    MultiSourceBatch,
    MultiSourceContractError,
    SampleTarget,
    build_multisource_sample,
    deterministic_group_order,
    dumps_ontology_contract,
    ontology_contract_fingerprint,
    validate_group_assignments,
)


HOOK_TASKS = {
    "theory.chord.adds",
    "theory.chord.alterations",
    "theory.chord.borrowed",
    "theory.chord.extent",
    "theory.chord.inversion",
    "theory.chord.omits",
    "theory.chord.presence",
    "theory.chord.root_degree",
    "theory.chord.suspensions",
    "theory.local_key.mode",
    "theory.local_key.tonic_pc",
    "theory.melody.scale_degree",
}
POP_TASKS = {
    "pop909_cl.chord.bass",
    "pop909_cl.chord.boundary",
    "pop909_cl.chord.inversion",
    "pop909_cl.chord.no_chord",
    "pop909_cl.chord.quality",
    "pop909_cl.chord.root",
}


def _hook_piece(*, root: int = 1):
    record = {
        "hash": "clip",
        "split": "train",
        "json": {
            "endBeat": 5,
            "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
            "tempos": [{"beat": 1, "bpm": 120}],
            "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
            "notes": [
                {
                    "beat": 1,
                    "duration": 1,
                    "sd": "1",
                    "octave": 0,
                    "isRest": False,
                }
            ],
            "chords": [
                {
                    "beat": 1,
                    "duration": 2,
                    "root": root,
                    "type": 5,
                    "inversion": 0,
                    "adds": [],
                    "omits": [],
                    "alterations": [],
                    "suspensions": [],
                    "borrowed": None,
                    "isRest": False,
                    "applied": 0,
                    "alternate": "",
                    "pedal": None,
                }
            ],
        },
    }
    return convert_hooktheory_record(
        "clip",
        record,
        config=HookTheoryAdapterConfig(dataset_name="hooktheory"),
        structure_row={
            "audio_path": "audio/clip.mp3",
            "ori_uid": "hook-source",
        },
        source_path="4_merged.json",
    )


def _pop_piece(
    tmp_path: Path,
    *,
    pitches: tuple[int, ...] = (60, 64, 67),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "001.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.MetaMessage("end_of_track", time=1_920),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", channel=0, program=0, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=1_920),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    chord_track = mido.MidiTrack(
        [mido.Message("program_change", channel=1, program=0, time=0)]
    )
    for pitch in pitches:
        chord_track.append(
            mido.Message("note_on", channel=1, note=pitch, velocity=70, time=0)
        )
    for index, pitch in enumerate(pitches):
        chord_track.append(
            mido.Message(
                "note_off",
                channel=1,
                note=pitch,
                velocity=0,
                time=1_920 if index == 0 else 0,
            )
        )
    chord_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(chord_track)
    midi.save(path)
    record = Pop909ClCorpusRecord(
        song_id="001",
        path=path,
        relative_path="POP909_processed/001.mid",
        corpus_relative_path="001.mid",
        sha256=sha256(path.read_bytes()).hexdigest(),
        source_group_id="pop909-cl:001",
        lineage_group_id="pop909-lineage:001",
    )
    result = convert_pop909_cl_file(record)
    assert result.status == "accepted"
    return result.piece


def test_registry_version_ids_value_spaces_and_serialization_are_stable() -> None:
    assert TARGET_ONTOLOGY_VERSION == "1.0.0"
    assert {spec.task_id for spec in TARGET_FAMILIES} == HOOK_TASKS | POP_TASKS
    assert len(TARGET_FAMILIES) == len({spec.task_id for spec in TARGET_FAMILIES})
    assert all(spec.availability_mask_required for spec in TARGET_FAMILIES)
    assert all(spec.provenance_required for spec in TARGET_FAMILIES)
    assert all(spec.alignment_policy.storage == "sidecar_only" for spec in TARGET_FAMILIES)
    assert {
        spec.source_alignment_type for spec in TARGET_FAMILIES
    } == {"note", "annotation_span"}
    for spec in TARGET_FAMILIES:
        if spec.vocabulary is not None:
            assert spec.vocabulary
            assert len(spec.vocabulary) == len(set(spec.vocabulary))
        else:
            assert spec.open_vocabulary
    payload = dumps_ontology_contract()
    assert payload == dumps_ontology_contract()
    assert ontology_contract_fingerprint() == sha256(payload.encode()).hexdigest()
    assert ontology_contract_fingerprint() == (
        "7e130f2dc811eece370002a44a05f48369b1f1eb481a0952d5f2052d1dcfe042"
    )


def test_actual_adapter_target_structures_match_registry(tmp_path: Path) -> None:
    hook = build_multisource_sample(
        _hook_piece(), build_raw_graph(_hook_piece())
    )
    pop_piece = _pop_piece(tmp_path)
    pop = build_multisource_sample(pop_piece, build_raw_graph(pop_piece))
    assert {target.task_id for target in hook.target_bundle} == HOOK_TASKS
    assert {target.task_id for target in pop.target_bundle} == POP_TASKS
    assert hook.dataset_id == "hooktheory"
    assert hook.lineage_group_id == "hook-source"
    assert pop.dataset_id == "pop909_cl"
    assert pop.source_group_id == "pop909-cl:001"
    assert pop.lineage_group_id == "pop909-lineage:001"


def test_incompatible_targets_have_no_automatic_mapping() -> None:
    task_ids = {spec.task_id for spec in TARGET_FAMILIES}
    assert all(
        task_id in task_ids
        for item in CROSSWALKS
        for task_id in (item.left_task_id, item.right_task_id)
        if task_id is not None
    )
    paired = {
        item.crosswalk_id: item.status
        for item in CROSSWALKS
        if item.left_task_id is not None and item.right_task_id is not None
    }
    assert paired["hooktheory_root_degree__pop909_cl_absolute_root"] == "incompatible"
    assert paired["hooktheory_extent__pop909_cl_quality"] == "incompatible"
    assert paired["hooktheory_ordinal_inversion__pop909_cl_semitones"] == "incompatible"
    assert paired["hooktheory_presence__pop909_cl_boundary"] == "incompatible"
    assert paired["hooktheory_presence__pop909_cl_no_chord"] == "incompatible"
    assert "exact_shared" not in paired.values()
    assert "derived_lossless_subset" not in paired.values()
    for item in CROSSWALKS:
        if item.status == "derived_lossless_subset":
            assert item.prerequisites and item.algorithm


def test_independent_masks_preserve_ambiguous_unsupported_and_missing(
    tmp_path: Path,
) -> None:
    hook = build_multisource_sample(
        _hook_piece(root=0), build_raw_graph(_hook_piece(root=0))
    )
    by_task = {target.task_id: target for target in hook.target_bundle}
    assert by_task["theory.chord.root_degree"].availability_mask == (False,)
    assert by_task["theory.chord.presence"].availability_mask == (True,)
    assert by_task["theory.chord.root_degree"].values == (None,)

    ambiguous_piece = _pop_piece(tmp_path / "ambiguous", pitches=(60, 63, 66, 69))
    ambiguous = {
        target.task_id: target
        for target in build_multisource_sample(
            ambiguous_piece, build_raw_graph(ambiguous_piece)
        ).target_bundle
    }
    assert ambiguous["pop909_cl.chord.boundary"].availability_mask == (True,)
    assert ambiguous["pop909_cl.chord.bass"].availability_mask == (True,)
    assert ambiguous["pop909_cl.chord.quality"].availability_mask == (True,)
    assert ambiguous["pop909_cl.chord.root"].availability_mask == (False,)
    assert ambiguous["pop909_cl.chord.inversion"].availability_mask == (False,)

    unsupported_piece = _pop_piece(tmp_path / "unsupported", pitches=(60, 61))
    unsupported = {
        target.task_id: target
        for target in build_multisource_sample(
            unsupported_piece, build_raw_graph(unsupported_piece)
        ).target_bundle
    }
    assert unsupported["pop909_cl.chord.boundary"].availability_mask == (True,)
    assert unsupported["pop909_cl.chord.bass"].availability_mask == (True,)
    for task_id in (
        "pop909_cl.chord.root",
        "pop909_cl.chord.quality",
        "pop909_cl.chord.inversion",
    ):
        assert unsupported[task_id].availability_mask == (False,)
        assert unsupported[task_id].values == (None,)


def test_unavailable_entries_cannot_be_converted_to_negative_labels() -> None:
    with pytest.raises(MultiSourceContractError, match="entirely null"):
        SampleTarget(
            task_id="theory.chord.presence",
            annotation_view_id=None,
            alignment_type="annotation_span",
            entity_ids=("span:1",),
            values=("false",),
            availability_mask=(False,),
            confidence=(None,),
            source=(None,),
            provenance_ids=(None,),
        )
    sample = build_multisource_sample(
        _hook_piece(root=0), build_raw_graph(_hook_piece(root=0))
    )
    availability = {item.task_id: item for item in sample.target_availability}
    assert availability["pop909_cl.chord.no_chord"].family_present is False
    assert availability["pop909_cl.chord.no_chord"].available_count == 0
    assert availability["theory.chord.root_degree"].family_present is True
    assert availability["theory.chord.root_degree"].masked_count == 1

    piece = _hook_piece()
    raw_only_piece = replace(piece, annotations=(), targets=())
    raw_only = build_multisource_sample(
        raw_only_piece, build_raw_graph(raw_only_piece)
    )
    assert raw_only.target_bundle == ()
    assert raw_only.target_provenance_sidecar == ()
    assert all(
        not item.family_present and item.available_count == item.masked_count == 0
        for item in raw_only.target_availability
    )


def test_empty_batch_family_and_mixed_batch_api_contract(tmp_path: Path) -> None:
    hook_piece = _hook_piece()
    pop_piece = _pop_piece(tmp_path)
    hook = build_multisource_sample(hook_piece, build_raw_graph(hook_piece))
    pop = build_multisource_sample(pop_piece, build_raw_graph(pop_piece))
    empty = BatchTarget(
        task_id="pop909_cl.chord.no_chord",
        values=(),
        availability_mask=(),
        entity_indices=(),
        sample_indices=(),
        confidence=None,
        entry_count=0,
        provenance_cpu=(),
        diagnostics_cpu=(),
    )
    batch = MultiSourceBatch(
        raw_graph_batch=object(),
        target_batches=(empty,),
        dataset_ids=(hook.dataset_id, pop.dataset_id),
        piece_ids=(hook.piece_id, pop.piece_id),
        source_group_ids=(hook.source_group_id, pop.source_group_id),
        lineage_group_ids=(hook.lineage_group_id, pop.lineage_group_id),
        diagnostics_cpu=(hook.diagnostics, pop.diagnostics),
    )
    assert batch.target_batches[0].entry_count == 0
    assert batch.dataset_ids == ("hooktheory", "pop909_cl")


def test_grouping_is_lineage_safe_and_order_is_deterministic() -> None:
    safe = (
        GroupAssignment("pop909_cl", "cl-001", "pop909-cl:001", "pop909-lineage:001", "train"),
        GroupAssignment("pop909_original", "original-001", "pop909-original:001", "pop909-lineage:001", "train"),
        GroupAssignment("hooktheory", "clip-a", "song-a", "song-a", None),
    )
    validate_group_assignments(safe)
    assert deterministic_group_order(safe, seed=17) == deterministic_group_order(
        tuple(reversed(safe)), seed=17
    )
    unsafe = (
        safe[0],
        replace(safe[1], split="val"),
    )
    with pytest.raises(MultiSourceContractError, match="lineage_group_id"):
        validate_group_assignments(unsafe)


def test_target_sidecars_do_not_change_raw_graph_or_enter_pyg_stores() -> None:
    piece = _hook_piece()
    graph = build_raw_graph(piece)
    fingerprint = graph_fingerprint(graph)
    presence = next(
        target for target in piece.targets if target.task == "theory.chord.presence"
    )
    changed_presence = replace(presence, values=("false",))
    changed_piece = replace(
        piece,
        targets=tuple(
            changed_presence if target is presence else target
            for target in piece.targets
        ),
        split="test",
        source_group_id="sentinel-group",
    )
    changed_graph = build_raw_graph(changed_piece)
    assert graph_fingerprint(changed_graph) == fingerprint
    validate_raw_graph(changed_graph)
    forbidden = {
        "target",
        "targets",
        "provenance",
        "source_group_id",
        "lineage_group_id",
        "split",
        "dataset_id",
        "availability_mask",
    }
    assert not (forbidden & set(changed_graph._global_store.keys()))
    for store in (*changed_graph.node_stores, *changed_graph.edge_stores):
        assert not (forbidden & set(store.keys()))
