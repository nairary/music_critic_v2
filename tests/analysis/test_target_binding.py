"""Synthetic exact-lineage tests; these do not assess musical interpretation."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import pytest

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence,
    DilemmadataSourceRowBinding,
    _alignment_evidence_fingerprint,
)
from music_critic.data import RationalTime, dumps_piece, load_piece
from music_critic.analysis.target_binding import TargetBindingError, bind_source_events


RID = "dlc:fixture:target-binding"
SHA = "a" * 64


def fixture():
    base = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    prototype = next(note for note in base.notes if not note.is_percussion)
    note = replace(prototype, note_id="note:target-0", pitch=60,
                   onset_qn=RationalTime(0), duration_qn=RationalTime(2),
                   staff=1, voice=1)
    piece = replace(base, notes=(note,), targets=(), annotations=())
    evidence = DilemmadataRawTargetAlignmentEvidence(
        version="1.1.0", record_id=RID, piece_id=piece.piece_id,
        record_binding_sha256="b" * 64, raw_projection_sha256="c" * 64,
        canonical_piece_sha256=sha256(dumps_piece(piece).encode()).hexdigest(),
        raw_source_fields=("pitch",),
        rows=(DilemmadataSourceRowBinding(0, 2, note.onset_qn, note.note_id, False),),
        fingerprint="",
    )
    evidence = replace(evidence, fingerprint=_alignment_evidence_fingerprint(evidence))
    packet = {
        "source_version_id": f"{RID}@sha256:{SHA}", "current_split": "train",
        "source_table": {"sha256": SHA},
        "events": [{"id": "source:l2", "source_version_id": f"{RID}@sha256:{SHA}",
                    "source_row_id": f"{RID}#line=2", "source_row_number": 2,
                    "start": "0/1", "duration": "2/1", "end": "2/1", "pitch": 60,
                    "source_staff": "1", "source_voice": "1", "part_id": "source:part:1",
                    "is_note_onset": True, "time_unit": "quarter_note"}],
    }
    return piece, asdict(evidence), packet


def bind(piece, evidence, packet, **kwargs):
    return bind_source_events(piece, evidence, packet, record_id=RID, source_sha256=SHA,
                              current_split="train", **kwargs)


def test_source_row_binds_exact_note_without_changing_raw_piece():
    piece, evidence, packet = fixture()
    before = dumps_piece(piece)
    result = bind(piece, evidence, packet)
    assert len(result) == 1
    target = result[0]
    assert target.source_event_id == "source:l2"
    assert target.canonical_note_id == "note:target-0"
    assert (target.canonical_start_qn, target.canonical_end_qn) == ("0/1", "2/1")
    assert target.binding_available and target.mask_reason is None
    assert target.canonical_track_id == piece.notes[0].track_id
    assert target.source_part_id == "source:part:1"
    assert dumps_piece(piece) == before


def reseal(piece, payload):
    fields = dict(payload)
    fields["canonical_piece_sha256"] = sha256(dumps_piece(piece).encode()).hexdigest()
    fields["raw_source_fields"] = tuple(fields["raw_source_fields"])
    fields["rows"] = tuple(DilemmadataSourceRowBinding(
        **{**row, "onset_qn": RationalTime(**row["onset_qn"])}) for row in fields["rows"])
    evidence = DilemmadataRawTargetAlignmentEvidence(**fields)
    return asdict(replace(evidence, fingerprint=_alignment_evidence_fingerprint(evidence)))


def test_tie_continuation_keeps_its_subspan_on_same_midi_note():
    piece, evidence, packet = fixture()
    packet["events"][0].update(duration="1/1", end="1/1")
    packet["events"].append({**packet["events"][0], "id": "source:l3",
                             "source_row_number": 3, "source_row_id": f"{RID}#line=3",
                             "start": "1/1", "end": "2/1", "is_note_onset": False})
    evidence["rows"] = (*evidence["rows"], {**evidence["rows"][0], "ordinal": 1, "line": 3,
                                           "onset_qn": {"num": 1, "den": 1}, "tie_continuation": True})
    evidence = reseal(piece, evidence)
    first, continuation = bind(piece, evidence, packet)
    assert first.canonical_note_id == continuation.canonical_note_id == "note:target-0"
    assert (first.canonical_start_qn, first.canonical_end_qn) == ("0/1", "1/1")
    assert (continuation.canonical_start_qn, continuation.canonical_end_qn) == ("1/1", "2/1")
    assert not first.tie_continuation and continuation.tie_continuation
    assert first.binding_available and continuation.binding_available


def test_pickup_offset_is_exact_and_required_explicitly():
    piece, evidence, packet = fixture()
    packet["events"][0].update(start="-1/3", end="5/3")
    with pytest.raises(TargetBindingError, match="onset mismatch"):
        bind(piece, evidence, packet)
    (result,) = bind(piece, evidence, packet, source_to_canonical_offset_qn="1/3")
    assert result.source_start_qn == "-1/3"
    assert result.canonical_start_qn == "0/1"
    assert result.canonical_end_qn == "2/1"


@pytest.mark.parametrize("missing_note", [False, True])
def test_repair_mask_is_unavailable_instead_of_negative_or_invented_note(missing_note):
    piece, evidence, packet = fixture()
    evidence.update(version="1.2.0", raw_repair_evidence_fingerprint="d" * 64)
    evidence["rows"][0]["repair_mask_scope"] = "note"
    if missing_note:
        evidence["rows"][0]["canonical_note_id"] = None
    (result,) = bind(piece, reseal(piece, evidence), packet)
    assert not result.binding_available
    assert result.mask_reason == "raw_repair_mask:note"
    assert result.canonical_note_id == (None if missing_note else "note:target-0")


@pytest.mark.parametrize("change, message", [
    (lambda e: e.update(pitch=61), "pitch mismatch"),
    (lambda e: e.update(source_staff="2"), "staff/voice mismatch"),
    (lambda e: e.update(source_voice="2"), "staff/voice mismatch"),
    (lambda e: e.update(duration="3/1", end="3/1"), "outside canonical note"),
    (lambda e: e.update(duration="1/1"), "duration/end mismatch"),
    (lambda e: e.update(start="0.0"), "reduced rational"),
    (lambda e: e.update(duration=None), "exact rational"),
    (lambda e: e.update(time_unit="measured_seconds"), "quarter-note"),
    (lambda e: e.update(source_row_number=3), "source line"),
    (lambda e: e.update(source_row_id="foreign#line=2"), "lineage mismatch"),
    (lambda e: e.update(is_note_onset=False), "tie-onset"),
])
def test_no_fuzzy_fallback_for_corrupt_source_geometry_or_identity(change, message):
    piece, evidence, packet = fixture()
    change(packet["events"][0])
    with pytest.raises(TargetBindingError, match=message):
        bind(piece, evidence, packet)


def test_duplicate_source_lines_are_not_extra_training_examples():
    piece, evidence, packet = fixture()
    packet["events"].append({**packet["events"][0], "id": "other-id-same-row"})
    with pytest.raises(TargetBindingError, match="source line"):
        bind(piece, evidence, packet)


@pytest.mark.parametrize("corruption", ["piece", "fingerprint", "source_sha", "record", "test", "validation"])
def test_changed_evidence_and_non_train_fail_closed(corruption):
    piece, evidence, packet = fixture()
    if corruption == "piece":
        piece = replace(piece, notes=(replace(piece.notes[0], pitch=61),))
    elif corruption == "fingerprint":
        evidence["fingerprint"] = "e" * 64
    elif corruption == "source_sha":
        packet["source_table"]["sha256"] = "e" * 64
    elif corruption == "record":
        evidence["record_id"] = "dlc:foreign:version"
    else:
        packet["current_split"] = corruption
    with pytest.raises(TargetBindingError):
        bind(piece, evidence, packet)


def test_grace_point_stays_point_and_is_not_made_into_sounding_duration():
    piece, evidence, packet = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], duration_qn=RationalTime(0), is_grace=True),))
    packet["events"][0].update(duration="0/1", end="0/1")
    (result,) = bind(piece, reseal(piece, evidence), packet)
    assert result.canonical_start_qn == result.canonical_end_qn == "0/1"


def test_verified_remediated_orphan_tie_remains_accounted_but_masked():
    piece, evidence, packet = fixture()
    repair_fingerprint = 'd' * 64
    provenance = replace(piece.provenance[0], provenance_id='prov:remediated-fixture',
                         source='music_critic.dilemmadata_adapter', kind='conversion',
                         parents=('prov:source',),
                         details=(('raw_repair_evidence_fingerprint', repair_fingerprint),))
    piece = replace(piece, provenance=(*piece.provenance[:2], provenance, *piece.provenance[2:]))
    evidence.update(version='1.2.0', raw_repair_evidence_fingerprint=repair_fingerprint)
    evidence['rows'][0]['tie_continuation'] = True
    packet['events'][0]['is_note_onset'] = False
    (binding,) = bind(piece, reseal(piece, evidence), packet)
    assert binding.canonical_note_id == piece.notes[0].note_id
    assert binding.canonical_start_qn == '0/1'
    assert binding.tie_continuation
    assert not binding.binding_available
    assert binding.mask_reason == 'remediated_continuation_at_canonical_attack'


@pytest.mark.parametrize('version', ['1.1.0', '1.2.0'])
def test_unexplained_continuation_at_note_attack_still_fails(version):
    piece, evidence, packet = fixture()
    evidence['version'] = version
    if version == '1.2.0':
        evidence['raw_repair_evidence_fingerprint'] = 'd' * 64
    evidence['rows'][0]['tie_continuation'] = True
    packet['events'][0]['is_note_onset'] = False
    with pytest.raises(TargetBindingError, match='placement mismatch'):
        bind(piece, reseal(piece, evidence), packet)


@pytest.mark.parametrize('attack,staff', [(False, 1), (True, 1), (False, 2), (True, 2)])
def test_verified_remediated_identity_change_only_allows_masked_continuation(attack, staff):
    piece, evidence, packet = fixture()
    provenance = replace(piece.provenance[0], provenance_id='prov:remediated-fixture',
                         source='music_critic.dilemmadata_adapter', kind='conversion', parents=('prov:source',),
                         details=(('raw_repair_evidence_fingerprint', 'd' * 64),))
    piece = replace(piece, provenance=(*piece.provenance[:2], provenance, *piece.provenance[2:]))
    evidence.update(version='1.2.0', raw_repair_evidence_fingerprint='d' * 64)
    evidence['rows'][0].update(tie_continuation=not attack, onset_qn={'num': 1, 'den': 1})
    packet['events'][0].update(is_note_onset=attack, start='1/1', duration='1/1',
                               source_voice='2', source_staff=str(staff))
    if attack:
        with pytest.raises(TargetBindingError):
            bind(piece, reseal(piece, evidence), packet)
    else:
        (binding,) = bind(piece, reseal(piece, evidence), packet)
        assert binding.canonical_note_id == piece.notes[0].note_id
        assert binding.source_voice == 2 and piece.notes[0].voice == 1
        assert binding.source_staff == staff and piece.notes[0].staff == 1
        assert binding.canonical_track_id == piece.notes[0].track_id
        assert not binding.binding_available
        assert binding.mask_reason == ('remediated_continuation_voice_change' if staff == 1
                                       else 'remediated_continuation_staff_change')


@pytest.mark.parametrize('corruption', ['old_version', 'missing_provenance', 'foreign_fingerprint',
                                       'pitch', 'onset', 'containment'])
def test_cross_staff_continuation_keeps_strict_repair_and_geometry_guards(corruption):
    piece, evidence, packet = fixture()
    provenance = replace(piece.provenance[0], provenance_id='prov:remediated-fixture',
                         source='music_critic.dilemmadata_adapter', kind='conversion', parents=('prov:source',),
                         details=(('raw_repair_evidence_fingerprint', 'd' * 64),))
    if corruption != 'missing_provenance':
        piece = replace(piece, provenance=(*piece.provenance[:2], provenance, *piece.provenance[2:]))
    if corruption != 'old_version':
        evidence.update(version='1.2.0', raw_repair_evidence_fingerprint='d' * 64)
    if corruption == 'foreign_fingerprint':
        evidence['raw_repair_evidence_fingerprint'] = 'e' * 64
    evidence['rows'][0].update(tie_continuation=True, onset_qn={'num': 1, 'den': 1})
    event = packet['events'][0]
    event.update(is_note_onset=False, start='1/1', duration='1/1', source_staff='2')
    if corruption == 'pitch': event['pitch'] = 61
    if corruption == 'onset': event.update(start='1/2', duration='3/2')
    if corruption == 'containment': event.update(duration='2/1', end='3/1')
    with pytest.raises(TargetBindingError):
        bind(piece, reseal(piece, evidence), packet)
