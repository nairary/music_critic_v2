from __future__ import annotations

from dataclasses import replace

from music_critic.data import CanonicalPiece, ProvenanceRecord, QualityFlag
from music_critic.graph import build_raw_graph, dumps_graph


def _serialized(piece: CanonicalPiece) -> str:
    return dumps_graph(build_raw_graph(piece, raw_only=True))


def test_removing_or_changing_every_target_does_not_change_inputs_or_topology(
    canonical_piece: CanonicalPiece,
) -> None:
    baseline = _serialized(canonical_piece)
    assert canonical_piece.targets
    for index, target in enumerate(canonical_piece.targets):
        without = replace(
            canonical_piece,
            targets=canonical_piece.targets[:index] + canonical_piece.targets[index + 1 :],
        )
        assert _serialized(without) == baseline

        assert target.class_labels
        replacement_label = target.class_labels[-1]
        sentinel = replace(
            target,
            values=tuple(
                replacement_label if available else None
                for available in target.mask
            ),
            confidence=tuple(
                0.51 if available else None for available in target.mask
            ),
        )
        changed = replace(
            canonical_piece,
            targets=canonical_piece.targets[:index]
            + (sentinel,)
            + canonical_piece.targets[index + 1 :],
        )
        assert _serialized(changed) == baseline


def test_gold_alignment_annotations_never_change_raw_topology(
    canonical_piece: CanonicalPiece,
) -> None:
    baseline = _serialized(canonical_piece)
    assert _serialized(replace(canonical_piece, annotations=())) == baseline
    if canonical_piece.annotations:
        annotation = replace(
            canonical_piece.annotations[0],
            annotation_id="span:sentinel",
            annotation_type="theory.gold.sentinel",
            layer="target_alignment",
            value=None,
        )
        assert _serialized(replace(canonical_piece, annotations=(annotation,))) == baseline


def test_train_test_and_provenance_fields_never_enter_graph(
    canonical_piece: CanonicalPiece,
) -> None:
    baseline = _serialized(canonical_piece)
    sentinel_provenance = ProvenanceRecord(
        provenance_id="prov:train-test-sentinel",
        kind="annotation",
        source="secret_split_manifest",
        record_id="heldout-test-record",
        uri="private://test",
        version="999",
        checksum_sha256=None,
        created_at=None,
        parents=("prov:source",),
        details=(("split", "test"),),
    )
    sentinel_flag = QualityFlag(
        code="sentinel.test_split",
        severity="warning",
        message="held-out test provenance",
        entity_ids=(canonical_piece.piece_id,),
        provenance_id="prov:train-test-sentinel",
    )
    changed = replace(
        canonical_piece,
        dataset_name="secret-test-dataset",
        source_group_id="test-only-group",
        split="test",
        source_path="/private/test-piece.mid",
        source_resolution=999,
        provenance=(*canonical_piece.provenance, sentinel_provenance),
        quality_flags=(*canonical_piece.quality_flags, sentinel_flag),
    )
    assert _serialized(changed) == baseline
