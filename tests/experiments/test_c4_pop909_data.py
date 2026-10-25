"""Synthetic POP909-CL data-boundary tests; no real corpus reads."""
from dataclasses import replace
from hashlib import sha256

import pytest

from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig, Pop909ClCorpusRecord, convert_pop909_cl_file,
)
from music_critic.adapters.pop909_cl_synthetic import build_pop909_cl_smoke_midi
from music_critic.experiments.c4_pop909 import data
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.tasks import (
    CanonicalCorpusInput, CorpusCacheConfig, cache_canonical_corpus,
    create_split_manifest, dump_corpus_index, dump_split_manifest, loads_corpus_index,
)


def synthetic_source(tmp_path, song_id="001"):
    path = tmp_path / f"{song_id}.mid"
    build_pop909_cl_smoke_midi().save(path)
    source = Pop909ClCorpusRecord(song_id, path, path.name, path.name,
        sha256(path.read_bytes()).hexdigest(), "pending", f"pop909-lineage:{song_id}")
    converted = convert_pop909_cl_file(source, config=Pop909ClAdapterConfig(include_targets=False))
    piece = replace(converted.piece, split="train")
    binding = dict(record_id=piece.piece_id, piece_id=piece.piece_id, split="train",
        source_sha256=source.sha256, source_group_id=piece.source_group_id,
        component_fingerprint="synthetic-component", song_id=song_id)
    return piece, converted.chord_evidence, binding


def test_native_targets_are_separate_and_exact_beat_fallback(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    prepared = data.prepare_record(piece, evidence, binding)
    assert prepared.record_id == piece.piece_id
    assert prepared.valid_shifts == tuple(range(12))
    assert not piece.targets and not piece.annotations
    row, = prepared.target_rows
    assert row["onset_qn"] == {"num": 1, "den": 3}
    assert row["slice_id"] is None  # Annotation must not create a raw slice.
    assert row["beat_id"] == piece.beats[0].beat_id
    assert row["targets"]["root"]["value"] == 0
    assert row["targets"]["bass"]["value"] == 0
    assert row["targets"]["inversion"]["value"] == 0
    assert set(row["targets"]) == set(data.SUPERVISED_HEADS)


def test_transposition_rotates_pc_targets_without_mutating_identity(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    prepared = data.prepare_record(piece, evidence, binding)
    shifted, = data.transpose_target_rows(prepared.target_rows, 2)
    original, = prepared.target_rows
    assert original["targets"]["root"]["value"] == 0
    assert shifted["targets"]["root"]["value"] == 2
    assert shifted["targets"]["bass"]["value"] == 2
    assert shifted["targets"]["quality"] == original["targets"]["quality"]
    assert shifted["targets"]["inversion"] == original["targets"]["inversion"]
    labels = get_vocabulary(TASK_BY_ID["pitch_class_set"].vocabulary_id).labels
    assert labels[shifted["targets"]["pitch_class_set"]["value"]] == "2,6,9"
    assert shifted["onset_qn"] == original["onset_qn"]


def test_test_guard_precedes_population_or_source_payload(tmp_path):
    with pytest.raises(ValueError, match="test_access_forbidden"):
        data.read_record({}, "piece:pop909-cl-999", tmp_path / "missing", split="test")


def test_independent_native_masks_do_not_invent_root_or_inversion(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    block, = evidence.blocks
    ambiguous = replace(block, selected_candidate=None, root_available=False,
                        inversion_available=False, normalization_status="ambiguous")
    prepared = data.prepare_record(piece, replace(evidence, blocks=(ambiguous,)), binding)
    targets = prepared.target_rows[0]["targets"]
    assert targets["root"]["value"] is None and not targets["root"]["available"]
    assert targets["inversion"]["value"] is None and not targets["inversion"]["available"]
    assert targets["bass"]["available"] and targets["quality"]["available"]
    assert targets["pitch_class_set"]["available"]


def test_native_inversion_is_ordinal_not_semitone_class(tmp_path):
    piece, evidence, _ = synthetic_source(tmp_path)
    block, = evidence.blocks
    candidate = replace(block.selected_candidate, bass_pc=4, bass="E", inversion_semitones=4)
    block = replace(block, candidates=(candidate,), selected_candidate=candidate,
                    bass_pitch_class=4, lowest_source_pitch=64)
    row, = data.build_target_rows(piece, replace(evidence, blocks=(block,)))
    assert row["targets"]["bass"]["value"] == 4
    assert row["targets"]["inversion"]["value"] == 1
    assert row["targets"]["inversion"]["source_value"] == 4


def test_expected_missing_source_has_no_negative_rows_or_graph_change(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    full = data.prepare_record(piece, evidence, binding)
    missing = data.prepare_record(piece, replace(evidence, status="expected_missing", blocks=()), binding)
    assert missing.target_rows == ()
    assert missing.graph.fingerprint() == full.graph.fingerprint()
    assert missing.source_binding["source_status"] == "expected_missing"
    assert missing.source_binding["target_fingerprint"] != full.source_binding["target_fingerprint"]


def test_exact_slice_then_half_open_beat_and_outside_extent(tmp_path):
    piece, evidence, _ = synthetic_source(tmp_path)
    block, = evidence.blocks
    boundary = replace(block, onset_tick=480)
    row, = data.build_target_rows(piece, replace(evidence, blocks=(boundary,)))
    assert row["slice_id"] == "rawtime:1_1"
    assert row["beat_id"] == piece.beats[1].beat_id
    outside = replace(block, onset_tick=1921)
    row, = data.build_target_rows(piece, replace(evidence, blocks=(outside,)))
    assert row["slice_id"] is None and row["beat_id"] is None
    assert row["beat_alignment_reason"] == "missing_containing_beat"


def test_raw_midi_range_gates_shifts(tmp_path):
    piece, _, _ = synthetic_source(tmp_path)
    edge_notes = (replace(piece.notes[0], pitch=0), replace(piece.notes[1], pitch=127))
    assert data.eligible_shifts(replace(piece, notes=edge_notes)) == (0,)


@pytest.fixture
def synthetic_population(tmp_path):
    accepted = []
    source_root = tmp_path / "source"
    source_root.mkdir()
    for number, split in enumerate(("train", "validation", "test"), 1):
        song_id = f"{number:03d}"
        midi = build_pop909_cl_smoke_midi()
        for track in midi.tracks:
            for message in track:
                if message.type in ("note_on", "note_off") and message.channel == 0:
                    message.note += number
        path = source_root / f"{song_id}.mid"
        midi.save(path)
        source = Pop909ClCorpusRecord(song_id, path, path.name, path.name,
            sha256(path.read_bytes()).hexdigest(), "pending", f"pop909-lineage:{song_id}")
        converted = convert_pop909_cl_file(source)
        accepted.append((converted, split))
    index, _ = cache_canonical_corpus([
        CanonicalCorpusInput(piece=row.piece, lineage_group_id=row.record.lineage_group_id,
            source_identity=row.record.song_id, source_relative_path=row.record.relative_path,
            source_sha256=row.record.sha256) for row, _ in accepted],
        cache_config=CorpusCacheConfig(tmp_path / "cache"), dataset_id="pop909_cl",
        adapter_name="synthetic_c4", adapter_version="2.0.0", adapter_config={},
        source_identity="synthetic:c4", source_fingerprint=sha256(b"synthetic:c4").hexdigest())
    splits = create_split_manifest((index,), {
        ("pop909_cl", row.piece.piece_id): split for row, split in accepted}, seed=17)
    index_path, split_path = tmp_path / "pop.index.json", tmp_path / "split.json"
    dump_corpus_index(index, index_path)
    dump_split_manifest(splits, split_path)
    return data.load_population(index_path, split_path), source_root


def test_population_preserves_all_metadata_and_loads_only_requested_split(synthetic_population):
    population, root = synthetic_population
    assert population["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    prepared = data.read_record(population, "piece:pop909-cl-001", root, split="train")
    assert prepared.split == "train" and len(prepared.target_rows) == 1
    with pytest.raises(ValueError, match="record_not_in_requested_split"):
        data.read_record(population, "piece:pop909-cl-003", root / "missing", split="train")
    with pytest.raises(ValueError, match="record_not_in_requested_split"):
        data.read_record(population, "piece:pop909-cl-002", root / "missing", split="train")
    assert data.read_record(population, "piece:pop909-cl-002", root, split="validation").split == "validation"


def test_population_and_source_hashes_are_authenticated_before_conversion(synthetic_population):
    population, root = synthetic_population
    from copy import deepcopy
    modified = deepcopy(population)
    modified["records"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="population_binding_mismatch"):
        data.read_record(modified, "piece:pop909-cl-001", root, split="validation")
    (root / "001.mid").write_bytes(b"tampered")
    with pytest.raises(Exception) as caught:
        data.read_record(population, "piece:pop909-cl-001", root, split="train")
    assert caught.value.category == "pop909_frozen.source_fingerprint_mismatch"


def test_preparation_rejects_targets_from_another_source(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    block, = evidence.blocks
    unrelated = replace(evidence, blocks=(replace(block, source_sha256="f" * 64),))
    with pytest.raises(ValueError, match="target_source_binding_mismatch"):
        data.prepare_record(piece, unrelated, binding)


def test_all_twelve_shifts_preserve_masks_and_original_evidence(tmp_path):
    piece, evidence, binding = synthetic_source(tmp_path)
    rows = data.prepare_record(piece, evidence, binding).target_rows
    labels = get_vocabulary(TASK_BY_ID["pitch_class_set"].vocabulary_id).labels
    expected_sets = ("0,4,7", "1,5,8", "2,6,9", "3,7,10", "4,8,11", "0,5,9",
                     "1,6,10", "2,7,11", "0,3,8", "1,4,9", "2,5,10", "3,6,11")
    for shift, expected in enumerate(expected_sets):
        row, = data.transpose_target_rows(rows, shift)
        assert row["targets"]["root"]["value"] == shift
        assert row["targets"]["bass"]["value"] == shift
        assert labels[row["targets"]["pitch_class_set"]["value"]] == expected
        assert all(target["available"] for target in row["targets"].values())
        assert row["targets"]["root"]["source_value"] == "C"
    assert "applied_shift_pc" not in rows[0]


def test_native_quality_oov_and_pcset_oov_are_independent(tmp_path):
    piece, evidence, _ = synthetic_source(tmp_path)
    block, = evidence.blocks
    candidate = replace(block.selected_candidate, quality="sus2")
    block = replace(block, candidates=(candidate,), selected_candidate=candidate,
                    pitch_class_set=(0, 2, 7))
    row, = data.build_target_rows(piece, replace(evidence, blocks=(block,)))
    assert row["targets"]["root"]["available"]
    assert row["targets"]["bass"]["available"]
    assert not row["targets"]["quality"]["available"]
    assert row["targets"]["quality"]["reason"] == "outside_frozen_subset9"
    assert not row["targets"]["pitch_class_set"]["available"]
    assert row["targets"]["pitch_class_set"]["value"] is None
    shifted, = data.transpose_target_rows((row,), 2)
    assert not shifted["targets"]["pitch_class_set"]["available"]
    assert shifted["targets"]["root"]["value"] == 2


def test_global_split_requires_companion_index_and_preserves_grouping(synthetic_population):
    population, root = synthetic_population
    index_path = root.parent / "pop.index.json"
    pop = loads_corpus_index(index_path.read_text())
    piece, _, row = data.read_source_record(population, "piece:pop909-cl-001", root, split="train")
    other_piece = replace(piece, dataset_name="synthetic_companion", piece_id="piece:companion",
                          source_group_id=row["lineage_group_id"])
    other, _ = cache_canonical_corpus([
        CanonicalCorpusInput(piece=other_piece, lineage_group_id=row["lineage_group_id"],
            source_identity="companion", source_relative_path="companion.json",
            source_sha256=sha256(b"companion").hexdigest())],
        cache_config=CorpusCacheConfig(root.parent / "other-cache"), dataset_id="synthetic_companion",
        adapter_name="synthetic_c4_companion", adapter_version="1.0.0", adapter_config={},
        source_identity="synthetic:companion", source_fingerprint=sha256(b"companion").hexdigest())
    assignments = {("pop909_cl", row["piece_id"]): row["split"] for row in population["records"]}
    assignments["synthetic_companion", "piece:companion"] = "train"
    split = create_split_manifest((pop, other), assignments, seed=17)
    split_path, other_path = root.parent / "global.json", root.parent / "other.index.json"
    dump_split_manifest(split, split_path)
    dump_corpus_index(other, other_path)
    with pytest.raises(ValueError):
        data.load_population(index_path, split_path)
    complete = data.load_population(index_path, split_path, companion_indices=(other_path,))
    assert complete["split_counts"] == population["split_counts"]
    assert len(complete["indices"]) == 2
    assert data.read_record(complete, "piece:pop909-cl-001", root, split="train").record_id == piece.piece_id


def test_resolved_source_cannot_escape_declared_root(synthetic_population, tmp_path):
    population, root = synthetic_population
    original = root / "001.mid"
    outside = tmp_path / "outside.mid"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="source_path_escapes_root"):
        data.read_record(population, "piece:pop909-cl-001", root, split="train")


def test_trusted_graph_path_is_identical_and_public_path_rejects_invalid_piece(tmp_path):
    import torch
    from music_critic.experiments.analysisgnn.temporal_slices.graph import (
        build_temporal_batch, _build_temporal_batch_from_validated_pieces,
    )
    from music_critic.graph import GraphBuildError
    piece, evidence, binding = synthetic_source(tmp_path)
    public = build_temporal_batch((piece,))
    trusted = _build_temporal_batch_from_validated_pieces((piece,))
    assert public.fingerprint() == trusted.fingerprint()
    assert public.keys == trusted.keys and public.pointers == trusted.pointers
    assert torch.equal(public.slice_features, trusted.slice_features)
    assert all(torch.equal(public.edges[edge], trusted.edges[edge]) for edge in public.edges)
    invalid = replace(piece, notes=(replace(piece.notes[0], pitch=200), *piece.notes[1:]))
    with pytest.raises(GraphBuildError):
        data.prepare_record(invalid, evidence, binding)


def test_authenticated_source_validation_precedes_trusted_graph(synthetic_population, monkeypatch):
    population, root = synthetic_population
    calls = []
    raw = data.read_raw_source
    convert = data.convert_pop909_cl_file
    build = data._build_temporal_batch_from_validated_pieces
    def raw_read(*args, **kwargs):
        result = raw(*args, **kwargs)
        calls.append("raw_validated")
        return result
    def targets_read(*args, **kwargs):
        result = convert(*args, **kwargs)
        calls.append("targets_authenticated")
        return result
    def trusted(pieces):
        assert calls == ["raw_validated", "targets_authenticated"]
        calls.append("trusted_graph")
        return build(pieces)
    monkeypatch.setattr(data, "read_raw_source", raw_read)
    monkeypatch.setattr(data, "convert_pop909_cl_file", targets_read)
    monkeypatch.setattr(data, "_build_temporal_batch_from_validated_pieces", trusted)
    data.read_record(population, "piece:pop909-cl-001", root, split="train")
    assert calls == ["raw_validated", "targets_authenticated", "trusted_graph"]
