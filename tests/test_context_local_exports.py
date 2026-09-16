"""Source-free cache-boundary and native-alignment oracles; no corpus/weights.

``synthetic_sidecar`` is also usable by the local dispatcher fixtures. Its
labels are arbitrary vocabulary members, never scientific evidence.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
import torch

from music_critic.data import AnnotationSpan, RationalTime, TargetArray, piece_to_dict
from music_critic.experiments import context_local_exports as module
from music_critic.experiments.analysisgnn.corrected_training import (
    align_target_sidecars_after_prediction,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.context_diagnostics import FileBinding, Member, fingerprint
from music_critic.experiments.context_diagnostics_pairs import build_context_pair_suite
from music_critic.experiments.context_local_reference import (
    ALLOWED_TASKS, TASK_LEVELS, build_reference_view,
)


def _rehash(sidecar):
    sidecar["fingerprint"] = fingerprint({k: v for k, v in sidecar.items() if k != "fingerprint"})
    return sidecar


def _recount(sidecar):
    sidecar["entity_counts"] = dict(Counter(e["entity_type"] for e in sidecar["entities"]))
    sidecar["relation_counts"] = dict(Counter(r["relation"] for r in sidecar["relations"]))
    return _rehash(sidecar)


def synthetic_sidecar(piece, record_id="fixture", dialect="dlc"):
    """Expanded native schema with all task states, exact joins and own hash.

    One harmonic observation per raw beat, one onset per exact note onset, and
    one note observation per canonical note. Every task has a separate state;
    test mutations introduce missing labels/OOV without changing other tasks.
    """
    entities, relations = [], []

    def entity(identity, level, time):
        source = f"{record_id}:synthetic:{identity}"
        states = {}
        for task, spec in TASK_BY_ID.items():
            if spec.prediction_level != level:
                continue
            label = get_vocabulary(spec.vocabulary_id).labels[0]
            states[task] = {
                "available": True, "masked": False, "missing_reason": None,
                "source_value": label, "canonical_value": label,
                "source_entity_id": source, "canonical_entity_id": identity,
                "provenance": {"source_free_fixture": True, "dialect": dialect},
            }
        row = {"canonical_entity_id": identity, "entity_type": level,
               "onset_qn": {"num": time.num, "den": time.den},
               "source_entity_id": source, "targets": states}
        entities.append(row)
        return row

    def relation(name, source, target):
        relations.append({"relation": name, "source_entity_id": source, "target_entity_id": target})

    def owning_beat(time):
        return next((beat for beat in piece.beats
                     if beat.start_qn <= time < beat.start_qn + beat.duration_qn), None)

    harmonic_ids = {}
    for i, beat in enumerate(piece.beats):
        identity = f"harmonic-event:fixture:{i}"
        row = entity(identity, "harmonic_event", beat.start_qn)
        row.update(source_row_ordinal=i, source_row_ordinals=[i])
        harmonic_ids[beat.beat_id] = identity
        relation("harmonic_event_to_beat", identity, beat.beat_id)
        relation("harmonic_event_to_measure", identity, beat.bar_id)
        relation("beat_to_measure", beat.beat_id, beat.bar_id)
    onset_ids = {}
    for time in sorted({note.onset_qn for note in piece.notes}):
        identity = f"sidecar-onset:fixture:{time.num}_{time.den}"
        onset_ids[time] = identity
        entity(identity, "onset", time)
        beat = owning_beat(time)
        if beat is not None:
            relation("onset_to_beat", identity, beat.beat_id)
    for note in piece.notes:
        entity(note.note_id, "note", note.onset_qn)
        relation("note_to_onset", note.note_id, onset_ids[note.onset_qn])
        beat = owning_beat(note.onset_qn)
        if beat is not None:
            relation("note_to_harmonic_event", note.note_id, harmonic_ids[beat.beat_id])
    entities.sort(key=lambda row: (row["entity_type"], row["canonical_entity_id"]))
    relations.sort(key=lambda row: (row["relation"], row["source_entity_id"], row["target_entity_id"]))
    return _recount({
        "schema_version": "analysisgnn-source-native-target-sidecar-v1",
        "record_id": record_id, "dialect": dialect, "source_component_id": piece.source_group_id,
        "repair_evidence_fingerprint": None, "entities": entities, "relations": relations,
    })


@pytest.fixture
def piece():
    return replace(build_context_pair_suite()[0].left, dataset_name="dilemmadata",
                   piece_id="piece:dilemmadata-fixture", source_group_id="component:fixture", split="train")


def _binding(path, payload):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
    path.write_bytes(raw)
    return FileBinding(str(path), sha256(raw).hexdigest(), len(raw))


def _export(tmp_path, piece, *, sidecar=None, stored_piece=None, metadata_changes=None):
    raw = _binding(tmp_path / "raw.json", piece_to_dict(piece if stored_piece is None else stored_piece))
    targets = _binding(tmp_path / "targets.json", synthetic_sidecar(piece) if sidecar is None else sidecar)
    metadata_payload = {"record_id": "fixture", "split": piece.split,
                        "piece_sha256": raw.sha256, "sidecar_sha256": targets.sha256}
    metadata_payload.update(metadata_changes or {})
    metadata = _binding(tmp_path / "metadata.json", metadata_payload)
    return module.SelectedExport("fixture", Member(piece.piece_id, piece.source_group_id, piece.split),
                                 "dlc", raw, targets, metadata)


def _candidate_ids(view, tasks=ALLOWED_TASKS):
    return {task: set(view.evidence["candidate_ids_by_level"][TASK_LEVELS[task]]) for task in tasks}


def test_portable_binding_resolves_inside_root_and_verifies_bytes(tmp_path):
    (tmp_path / "cache").mkdir()
    expected = _binding(tmp_path / "cache/piece.json", {"finite": 1})
    bound = module.resolve_binding({"locator": "cache/piece.json", "sha256": expected.sha256,
                                    "size_bytes": expected.size_bytes}, tmp_path)
    assert bound == expected
    assert module.checked_json(bound) == {"finite": 1}


@pytest.mark.parametrize("locator", ["", "/outside.json", "../outside.json", "a/../b", "./a",
                                    "a//b", "a/", "a\\b", "https:thing", "C:/file"])
def test_nonportable_or_traversal_locator_rejected_without_read(tmp_path, monkeypatch, locator):
    monkeypatch.setattr(FileBinding, "read", lambda *_a, **_k: pytest.fail("payload read"))
    with pytest.raises(ValueError, match="portable_locator_required"):
        module.resolve_binding({"locator": locator, "sha256": "a" * 64, "size_bytes": 1}, tmp_path)


def test_symlink_escape_rejected_and_internal_symlink_resolves(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = _binding(tmp_path / "outside.json", {})
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    declaration = {"locator": "escape/outside.json", "sha256": outside.sha256, "size_bytes": outside.size_bytes}
    with pytest.raises(ValueError, match="locator_escaped_input_root"):
        module.resolve_binding(declaration, root)
    inside = _binding(root / "inside.json", {"ok": True})
    (root / "alias.json").symlink_to(root / "inside.json")
    assert module.resolve_binding({**declaration, "locator": "alias.json", "sha256": inside.sha256,
                                   "size_bytes": inside.size_bytes}, root) == inside


@pytest.mark.parametrize("change,reason", [({"size_bytes": True}, "positive_size_required"),
                                           ({"size_bytes": 0}, "positive_size_required"),
                                           ({"sha256": "unknown"}, "sha256_required"),
                                           ({"extra": "unbound"}, "explicit_file_binding_required")])
def test_explicit_binding_fields_are_required(tmp_path, change, reason):
    with pytest.raises(ValueError, match=reason):
        module.resolve_binding({"locator": "x", "sha256": "a" * 64, "size_bytes": 1, **change}, tmp_path)


@pytest.mark.parametrize("raw,reason", [(b'{"x":1,"x":2}', "duplicate_json_key"),
                                       (b'{"outer":{"x":1,"x":2}}', "duplicate_json_key"),
                                       (b'{"x":NaN}', "nonfinite_json"),
                                       (b'{"x":Infinity}', "nonfinite_json"),
                                       (b'{"x":-Infinity}', "nonfinite_json"),
                                       (b'{"nested":[1e999]}', "nonfinite_json"),
                                       (b'{"nested":[-1e999]}', "nonfinite_json")])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path, raw, reason):
    with pytest.raises(ValueError, match=reason):
        module.checked_json(_binding(tmp_path / "invalid.json", raw))


@pytest.mark.parametrize("corruption", ["sha", "short", "long", "budget"])
def test_bound_size_and_sha_are_checked_before_json(tmp_path, corruption):
    bound = _binding(tmp_path / "cache.json", {"verified": True})
    if corruption == "sha":
        bound = replace(bound, sha256="0" * 64)
    elif corruption in {"short", "long"}:
        bound = replace(bound, size_bytes=bound.size_bytes + (-1 if corruption == "short" else 1))
    with pytest.raises(ValueError, match="file_budget" if corruption == "budget" else "file_binding_mismatch"):
        module.checked_json(bound, max_bytes=bound.size_bytes - 1 if corruption == "budget" else 1024)


@pytest.mark.parametrize("stored_split", [None, "train"])
def test_raw_read_stamps_official_split_and_does_not_read_targets(tmp_path, monkeypatch, piece, stored_split):
    export = _export(tmp_path, piece, stored_piece=replace(piece, split=stored_split))
    reads, native_read = [], FileBinding.read

    def observed(binding, **kwargs):
        assert binding != export.targets, "raw stage opened semantic payload"
        reads.append(binding.path)
        return native_read(binding, **kwargs)

    monkeypatch.setattr(FileBinding, "read", observed)
    assert module.read_raw(export) == piece
    assert reads == [export.metadata.path, export.raw.path]


@pytest.mark.parametrize("field,value", [("record_id", "other"), ("split", "validation"),
                                         ("piece_sha256", "0" * 64), ("sidecar_sha256", "0" * 64),
                                         ("extra", "undeclared")])
def test_native_cache_metadata_mismatch_precedes_raw_read(tmp_path, monkeypatch, piece, field, value):
    export = _export(tmp_path, piece, metadata_changes={field: value})
    native_read = FileBinding.read

    def observed(binding, **kwargs):
        assert binding == export.metadata, "mismatched metadata reached payload"
        return native_read(binding, **kwargs)

    monkeypatch.setattr(FileBinding, "read", observed)
    with pytest.raises(ValueError, match="native_cache_metadata_mismatch"):
        module.read_raw(export)


@pytest.mark.parametrize("field,value", [("split", "validation"), ("split", "test"),
                                         ("piece_id", "piece:other"), ("source_group_id", "other"),
                                         ("dataset_name", "other")])
def test_raw_identity_and_existing_split_must_match_official_membership(tmp_path, piece, field, value):
    export = _export(tmp_path, piece, stored_piece=replace(piece, **{field: value}))
    with pytest.raises(ValueError, match="raw_official_identity_mismatch"):
        module.read_raw(export)


@pytest.mark.parametrize("reader", [module.read_raw, module.read_targets])
@pytest.mark.parametrize("split", ["test", "TEST", None, "unknown"])
def test_test_or_undefined_split_rejected_before_any_file_read(tmp_path, monkeypatch, piece, reader, split):
    export = _export(tmp_path, piece)
    export = replace(export, member=replace(export.member, split=split))
    monkeypatch.setattr(FileBinding, "read", lambda *_a, **_k: pytest.fail("TEST payload read"))
    with pytest.raises(ValueError, match="TRAINVAL_only_TEST_closed"):
        reader(export)


@pytest.mark.parametrize("semantic", ["annotation", "target"])
def test_raw_export_rejects_embedded_semantic_content(tmp_path, piece, semantic):
    if semantic == "annotation":
        stored = replace(piece, annotations=(AnnotationSpan(
            "span:gold", "chord", "target_alignment", RationalTime(0), RationalTime(1),
            None, None, None),))
    else:
        stored = replace(piece, targets=(TargetArray(
            "target:gold", "quality", None, "beat", (piece.beats[0].beat_id,), "categorical",
            ("D7",), ("D7",), (True,), (None,), ("synthetic",), (piece.provenance[0].provenance_id,)),))
    export = _export(tmp_path, piece, stored_piece=stored)
    with pytest.raises(ValueError, match="raw_export_contains_semantic_targets"):
        module.read_raw(export)


def test_complete_synthetic_sidecar_passes_expanded_native_contract(tmp_path, piece):
    sidecar = synthetic_sidecar(piece)
    assert module.read_targets(_export(tmp_path, piece, sidecar=sidecar)) == sidecar
    for entity in sidecar["entities"]:
        assert set(entity["targets"]) == {task for task, spec in TASK_BY_ID.items()
                                          if spec.prediction_level == entity["entity_type"]}
        assert all(state["available"] and not state["masked"] for state in entity["targets"].values())


@pytest.mark.parametrize("descriptor", [{"schema_version": "analysisgnn-source-native-target-sidecar-descriptor-v1"},
                                         {"schema_version": "analysisgnn-source-native-target-sidecar-v1",
                                          "record_id": "fixture", "dialect": "dlc",
                                          "source_component_id": "component:fixture"}])
def test_descriptor_cannot_substitute_for_expanded_entities(tmp_path, piece, descriptor):
    _rehash(descriptor)
    with pytest.raises(ValueError, match="expanded_native_sidecar_required|expanded_sidecar_budget"):
        module.read_targets(_export(tmp_path, piece, sidecar=descriptor))


@pytest.mark.parametrize("corruption,reason", [
    ("self_hash", "native_sidecar_fingerprint_mismatch"),
    ("record_id", "target_official_identity_mismatch"),
    ("component", "target_official_identity_mismatch"),
    ("dialect", "target_official_identity_mismatch"),
    ("entity_count", "entity_count_mismatch"), ("relation_count", "relation_count_mismatch"),
    ("duplicate_entity", "duplicate_or_missing_target_entity"),
    ("duplicate_join", "ambiguous_duplicate_join"),
    ("mask_equal", "independent_boolean_masks_required"),
    ("mask_integer", "independent_boolean_masks_required"),
    ("masked_gold", "masked_gold_must_be_null_with_reason"),
    ("missing_reason", "masked_gold_must_be_null_with_reason"),
    ("empty_provenance", "target_state_provenance_mismatch"),
    ("wrong_provenance_id", "target_state_provenance_mismatch"),
    ("source_entity", "source_entity_provenance_required"),
    ("wrong_level", "target_level_or_task_mismatch"),
    ("float_time", "exact_canonical_time_required"), ("unreduced_time", "unreduced_time"),
])
def test_expanded_sidecar_rejects_inconsistent_counts_masks_provenance_and_joins(tmp_path, piece, corruption, reason):
    sidecar = synthetic_sidecar(piece)
    entity = next(row for row in sidecar["entities"] if row["entity_type"] == "harmonic_event")
    state = entity["targets"]["quality"]
    if corruption == "self_hash":
        sidecar["fingerprint"] = "0" * 64
    elif corruption in {"record_id", "dialect"}:
        sidecar[corruption] = "other"
    elif corruption == "component":
        sidecar["source_component_id"] = "other"
    elif corruption == "entity_count":
        sidecar["entity_counts"]["harmonic_event"] += 1
    elif corruption == "relation_count":
        sidecar["relation_counts"]["harmonic_event_to_beat"] += 1
    elif corruption == "duplicate_entity":
        sidecar["entities"].append(deepcopy(entity))
    elif corruption == "duplicate_join":
        join = next(r for r in sidecar["relations"] if r["relation"] == "harmonic_event_to_beat")
        sidecar["relations"].append({**join, "target_entity_id": "beat:ambiguous"})
    elif corruption == "mask_equal":
        state["masked"] = True
    elif corruption == "mask_integer":
        state["available"] = 1
    elif corruption in {"masked_gold", "missing_reason"}:
        state.update(available=False, masked=True, missing_reason="source_unavailable")
        if corruption == "missing_reason":
            state.update(canonical_value=None, missing_reason=None)
    elif corruption == "empty_provenance":
        state["provenance"] = {}
    elif corruption == "wrong_provenance_id":
        state["source_entity_id"] = "other"
    elif corruption == "source_entity":
        entity["source_entity_id"] = ""
    elif corruption == "wrong_level":
        entity["targets"]["cadence"] = deepcopy(state)
    elif corruption == "float_time":
        entity["onset_qn"]["num"] = 0.0
    elif corruption == "unreduced_time":
        entity["onset_qn"] = {"num": 0, "den": 2}
    if corruption != "self_hash":
        _rehash(sidecar)
    with pytest.raises(ValueError, match=reason):
        module.read_targets(_export(tmp_path, piece, sidecar=sidecar))


def test_native_train_layout_matches_common_aligner_all_task_masks_and_counts(piece):
    sidecar = synthetic_sidecar(piece)
    view = build_reference_view(piece, ALLOWED_TASKS)
    layout = module.training_candidate_layout(view)
    assert view.output is None and len(layout.predictions) == 18
    assert all(not hasattr(head, "logits") for head in layout.predictions)
    native = align_target_sidecars_after_prediction(layout, view.batch, [sidecar])
    rows, evidence = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    assert evidence["alignment_diagnostics"] == []
    for task in ALLOWED_TASKS:
        expected = [entity for entity in sidecar["entities"] if task in entity["targets"]]
        assert len(rows[task]) == sidecar["entity_counts"][TASK_BY_ID[task].prediction_level]
        assert tuple(row.observation_id for row in rows[task]) == tuple(e["canonical_entity_id"] for e in expected)
        assert [row.available for row in rows[task]] == native.heads[task].valid_mask.tolist()
        assert all(row.value == 0 and row.reason is None and row.confidence is None for row in rows[task])
        assert {row.candidate_id for row in rows[task]} == _candidate_ids(view)[task]
        assert all(json.loads(row.provenance)["sidecar_fingerprint"] == sidecar["fingerprint"] for row in rows[task])


def test_common_eligibility_keeps_missing_oov_and_unaligned_observations_distinct(tmp_path, piece):
    sidecar = synthetic_sidecar(piece)
    harmonic = [e for e in sidecar["entities"] if e["entity_type"] == "harmonic_event"]
    harmonic[0]["targets"]["quality"].update(available=False, masked=True, canonical_value=None,
                                            missing_reason="source_field_absent")
    harmonic[1]["targets"]["root"]["canonical_value"] = "fixture:out-of-vocabulary"
    missing_id = harmonic[2]["canonical_entity_id"]
    sidecar["relations"] = [r for r in sidecar["relations"]
                            if not (r["relation"] == "harmonic_event_to_beat" and r["source_entity_id"] == missing_id)]
    _recount(sidecar)
    module.read_targets(_export(tmp_path, piece, sidecar=sidecar))
    view = build_reference_view(piece, ALLOWED_TASKS)
    layout = module.training_candidate_layout(view)
    native = align_target_sidecars_after_prediction(layout, view.batch, [sidecar])
    rows, _ = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    lookup = {task: {r.observation_id: r for r in values} for task, values in rows.items()}
    assert lookup["quality"][harmonic[0]["canonical_entity_id"]].reason == "source_field_absent"
    assert lookup["root"][harmonic[1]["canonical_entity_id"]].reason == "unknown_target"
    assert lookup["quality"][harmonic[1]["canonical_entity_id"]].available
    assert lookup["root"][harmonic[0]["canonical_entity_id"]].available
    assert lookup["primary_degree"][missing_id].candidate_id == "unaligned:" + missing_id
    assert lookup["primary_degree"][missing_id].reason == "exact_alignment_failed"
    for task in ALLOWED_TASKS:
        assert [r.available for r in rows[task]] == native.heads[task].valid_mask.tolist()
        assert all(r.value is None for r in rows[task] if not r.available)


def test_multiple_harmonic_source_observations_same_beat_are_not_collapsed(tmp_path, piece):
    sidecar = synthetic_sidecar(piece)
    original = next(e for e in sidecar["entities"] if e["entity_type"] == "harmonic_event")
    extra = deepcopy(original)
    extra.update(canonical_entity_id="harmonic:second-source-event", source_entity_id="source:second-event",
                 onset_qn={"num": 1, "den": 3}, source_row_ordinal=99, source_row_ordinals=[99])
    for state in extra["targets"].values():
        state.update(canonical_entity_id=extra["canonical_entity_id"], source_entity_id=extra["source_entity_id"])
    extra["targets"]["quality"]["canonical_value"] = get_vocabulary(TASK_BY_ID["quality"].vocabulary_id).labels[1]
    sidecar["entities"].append(extra)
    sidecar["relations"].append({"relation": "harmonic_event_to_beat",
                                 "source_entity_id": extra["canonical_entity_id"],
                                 "target_entity_id": piece.beats[0].beat_id})
    _recount(sidecar)
    module.read_targets(_export(tmp_path, piece, sidecar=sidecar))
    view = build_reference_view(piece, ALLOWED_TASKS)
    rows, _ = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    same_beat = [r for r in rows["quality"] if r.candidate_id == piece.beats[0].beat_id]
    assert {r.observation_id for r in same_beat} == {original["canonical_entity_id"], extra["canonical_entity_id"]}
    assert {r.value for r in same_beat} == {0, 1}
    assert all(r.available for r in same_beat)
    assert len(rows["cadence"]) == len({n.onset_qn for n in piece.notes})
    assert len(rows["metrical_strength"]) == len(piece.notes)


def test_exact_onset_and_note_identity_joins_ignore_coarser_beat_relation(piece):
    fractional = replace(piece.notes[0], note_id="note:one-third", onset_qn=RationalTime(1, 3),
                         duration_qn=RationalTime(1, 3))
    piece = replace(piece, notes=tuple(sorted(piece.notes + (fractional,), key=lambda n: n.onset_qn)))
    sidecar = synthetic_sidecar(piece)
    onset = next(e for e in sidecar["entities"] if e["entity_type"] == "onset" and e["onset_qn"] == {"num": 1, "den": 3})
    for relation in sidecar["relations"]:
        if relation["relation"] == "onset_to_beat" and relation["source_entity_id"] == onset["canonical_entity_id"]:
            relation["target_entity_id"] = piece.beats[-1].beat_id
    _rehash(sidecar)
    view = build_reference_view(piece, ALLOWED_TASKS)
    rows, _ = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    onset_row = next(r for r in rows["cadence"] if r.observation_id == onset["canonical_entity_id"])
    note_row = next(r for r in rows["metrical_strength"] if r.observation_id == fractional.note_id)
    assert onset_row.available and onset_row.candidate_id == "onset:1_3"
    assert note_row.available and note_row.candidate_id == fractional.note_id


@pytest.mark.parametrize("level,task", [("onset", "cadence"), ("note", "metrical_strength")])
def test_unmatched_exact_onset_or_note_id_is_masked_without_nearest_join(piece, level, task):
    sidecar = synthetic_sidecar(piece)
    entity = next(e for e in sidecar["entities"] if e["entity_type"] == level)
    if level == "onset":
        entity["onset_qn"] = {"num": 1, "den": 999}
    else:
        entity["canonical_entity_id"] = "note:absent-from-raw"
        for state in entity["targets"].values():
            state["canonical_entity_id"] = entity["canonical_entity_id"]
    _rehash(sidecar)
    view = build_reference_view(piece, ALLOWED_TASKS)
    rows, _ = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    row = next(r for r in rows[task] if r.observation_id == entity["canonical_entity_id"])
    assert not row.available and row.value is None and row.reason == "exact_alignment_failed"
    assert row.candidate_id == "unaligned:" + row.observation_id


def test_graph_endpoint_rows_are_explicitly_masked_for_common_local_eligibility(piece):
    endpoint = replace(piece.notes[0], note_id="note:endpoint", onset_qn=piece.duration_qn,
                       duration_qn=RationalTime(0), is_grace=True)
    piece = replace(piece, notes=piece.notes + (endpoint,))
    view = build_reference_view(piece, ALLOWED_TASKS)
    sidecar = synthetic_sidecar(piece)
    native = align_target_sidecars_after_prediction(module.training_candidate_layout(view), view.batch, [sidecar])
    rows, _ = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    for task, identity in [("metrical_strength", endpoint.note_id),
                           ("cadence", f"onset:{piece.duration_qn.num}_{piece.duration_qn.den}")]:
        row = next(r for r in rows[task] if r.candidate_id == identity)
        native_index = native.heads[task].entity_ids.index(row.observation_id)
        assert bool(native.heads[task].valid_mask[native_index])
        assert not row.available and row.value is None and row.reason == "raw_candidate_at_piece_end"
        assert identity not in _candidate_ids(view)[task]
        assert all(r.available for r in rows[task] if r.candidate_id != identity)


def test_prediction_layout_permutation_preserves_identity_based_alignment(piece):
    view = build_reference_view(piece, ALLOWED_TASKS)
    sidecar = synthetic_sidecar(piece)
    expected = module.aligned_target_rows(view, sidecar, ALLOWED_TASKS, _candidate_ids(view))
    layout = module.training_candidate_layout(view)
    permuted = SimpleNamespace(predictions=tuple(SimpleNamespace(
        **{**vars(head), "global_entity_indices": torch.flip(head.global_entity_indices, dims=(0,))})
        for head in layout.predictions))
    result = module.aligned_target_rows(replace(view, output=permuted), sidecar, ALLOWED_TASKS, _candidate_ids(view))
    assert result == expected
