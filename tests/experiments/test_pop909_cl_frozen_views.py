from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import mido
import pytest
import torch
from torch_geometric.data import Batch

from music_critic.adapters import (
    MidiAdapterConfig,
    Pop909ClAdapterConfig,
    convert_pop909_cl_file,
    load_midi_piece,
)
from music_critic.data import dumps_piece
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.pop909_cl_frozen.views import (
    REMOVED_FIELDS,
    VIEW_IDS,
    VIEW_SCHEMA,
    build_view,
    view_contract,
)
from music_critic.graph import RAW_FEATURE_REGISTRY, build_raw_graph, graph_fingerprint
from tests.adapters.test_pop909_cl import _write_midi
from tests.graph.test_builder import _write_multitrack_midi


def test_metadata_removal_changes_only_declared_canonical_fields_and_rebuilds_closure(
    tmp_path: Path,
) -> None:
    piece = load_midi_piece(
        _write_multitrack_midi(tmp_path / "synthetic.mid"),
        config=MidiAdapterConfig(dataset_name="synthetic", split="training"),
    )
    before = dumps_piece(piece)
    native = build_view(piece, "native")
    removed = build_view(piece, "metadata_removed")
    assert dumps_piece(piece) == before
    assert native.piece == piece and native.piece is not piece
    assert native.graph_fingerprint == graph_fingerprint(build_raw_graph(piece))
    assert removed.graph_fingerprint == graph_fingerprint(build_raw_graph(removed.piece))
    assert native.graph_fingerprint != removed.graph_fingerprint

    for field in fields(piece):
        if field.name not in {"notes", "tracks"}:
            assert getattr(removed.piece, field.name) == getattr(piece, field.name)
    assert any(track.is_percussion for track in piece.tracks)
    for original, actual in zip(piece.tracks, removed.piece.tracks, strict=True):
        assert actual == replace(original, program=None, channel=None)
    for original, actual in zip(piece.notes, removed.piece.notes, strict=True):
        assert actual == replace(original, program=None, channel=None, velocity=None)

    changed = {
        "note": {"velocity", "program", "channel"},
        "track": {"program", "channel", "mean_velocity"},
    }
    for node_type in native.graph.node_types:
        old, new = native.graph[node_type], removed.graph[node_type]
        assert old.entity_id == new.entity_id
        if node_type in {"onset", "beat"}:
            assert torch.equal(old.candidate_slot, new.candidate_slot)
        for kind, prefix in (("categorical", "cat"), ("continuous", "cont")):
            for index, spec in enumerate(RAW_FEATURE_REGISTRY.for_node(node_type, kind)):
                old_values, new_values = getattr(old, f"x_{prefix}")[:, index], getattr(new, f"x_{prefix}")[:, index]
                old_mask, new_mask = getattr(old, f"x_{prefix}_available")[:, index], getattr(new, f"x_{prefix}_available")[:, index]
                if spec.name in changed.get(node_type, set()):
                    assert not new_mask.any()
                    expected_value = spec.unknown_id if kind == "categorical" else 0
                    assert torch.all(new_values == expected_value)
                else:
                    assert torch.equal(old_values, new_values)
                    assert torch.equal(old_mask, new_mask)
    for edge_type in native.graph.edge_types:
        assert torch.equal(native.graph[edge_type].edge_index, removed.graph[edge_type].edge_index)


def test_view_identity_is_deterministic_and_policy_separated_for_already_missing_fields(
    tmp_path: Path,
) -> None:
    piece = convert_pop909_cl_file(_write_midi(tmp_path / "synthetic.mid")).piece
    first = build_view(piece, "metadata_removed")
    second = build_view(piece, "metadata_removed")
    assert first.piece == second.piece
    assert first.view_fingerprint == second.view_fingerprint
    native_absent = build_view(first.piece, "native")
    removed_absent = build_view(first.piece, "metadata_removed")
    assert native_absent.graph_fingerprint == removed_absent.graph_fingerprint
    assert native_absent.model_input_fingerprint == removed_absent.model_input_fingerprint
    assert native_absent.view_fingerprint != removed_absent.view_fingerprint
    assert VIEW_IDS == ("native", "metadata_removed")
    contract = view_contract()
    assert contract["schema"] == VIEW_SCHEMA
    assert tuple(contract["metadata_removed_fields"]) == REMOVED_FIELDS
    contract["metadata_removed_fields"].append("tempo")
    assert view_contract()["metadata_removed_fields"] == list(REMOVED_FIELDS)
    with pytest.raises(ValueError, match="unsupported frozen input view"):
        build_view(piece, "drop_all_metadata")
    with pytest.raises(TypeError, match="CanonicalPiece"):
        build_view(object())


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_annotation_channel_and_target_mutations_leave_inputs_candidates_and_eval_logits_identical(
    tmp_path: Path, view_id: str,
) -> None:
    records = [
        _write_midi(tmp_path / "original.mid", chord_first=(60, 64, 67)),
        _write_midi(tmp_path / "changed.mid", chord_first=(62, 65, 69)),
    ]
    deleted = _write_midi(tmp_path / "deleted.mid")
    midi = mido.MidiFile(deleted.path)
    midi.tracks[2] = mido.MidiTrack([
        mido.MetaMessage("track_name", name="annotation_changed", time=0),
        mido.Message("program_change", channel=1, program=50, time=0),
        mido.MetaMessage("end_of_track", time=3_840),
    ])
    midi.save(deleted.path)
    records.append(replace(deleted, sha256=sha256(deleted.path.read_bytes()).hexdigest()))
    visible = [convert_pop909_cl_file(record) for record in records]
    hidden = convert_pop909_cl_file(records[0], config=Pop909ClAdapterConfig(include_targets=False))
    assert visible[0].piece.targets != visible[1].piece.targets
    assert hidden.piece.targets == ()
    assert visible[2].chord_evidence.blocks == ()
    pieces = [row.piece for row in visible] + [hidden.piece]
    assert all(len(piece.notes) == 2 for piece in pieces)
    assert all(note.channel == 0 for piece in pieces for note in piece.notes)
    views = [build_view(piece, view_id) for piece in pieces]
    assert len({view.graph_fingerprint for view in views}) == 1
    assert len({view.model_input_fingerprint for view in views}) == 1
    assert len({view.view_fingerprint for view in views}) == 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(402)
        model = CorrectedAnalysisGNNModel().eval()
    with torch.inference_mode():
        predictions = [model.predict(Batch.from_data_list([view.graph])).predictions for view in views]
    for result in predictions[1:]:
        for baseline, actual in zip(predictions[0], result, strict=True):
            assert actual.task_id == baseline.task_id
            assert torch.equal(actual.global_entity_indices, baseline.global_entity_indices)
            assert torch.equal(actual.candidate_node_type_codes, baseline.candidate_node_type_codes)
            assert torch.equal(actual.sample_indices, baseline.sample_indices)
            assert torch.equal(actual.logits, baseline.logits)
