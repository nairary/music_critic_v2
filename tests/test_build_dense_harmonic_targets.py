"""Publication and interval admission checks using synthetic notes only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/build_dense_harmonic_targets.py'
    spec = importlib.util.spec_from_file_location('dense_target_release', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture():
    from tests.tasks.test_dense_harmonic_targets import fixture as dense_fixture
    piece, packet = dense_fixture()
    assignment = dict(
        assignment_id='a0', case_id='synthetic', note_id=piece.notes[0].note_id, role='suspension',
        start_qn={'num': 1, 'den': 1}, end_qn={'num': 3, 'den': 1},
        related_note_ids=[], source_harmony_ids=['h0', 'h1'],
        source_version_id=packet['source_version_id'], origin='astra_individual_context',
        canonical_piece_id=piece.piece_id, source_event_ids=['l2'],
        target_available=True, target_value='present',
        status='admitted_weak', evidence='Synthetic mechanical test, not musical evidence',
        confidence=None, human_gold=False,
    )
    return piece, packet, assignment


def test_context_preserves_subinterval_and_numeric_index():
    piece, packet, assignment = fixture()
    m = module()
    rows = m.validate_context_assignments(piece, packet, [assignment])
    numeric = m.numeric_context(rows, ['passing_tone', 'suspension'])
    assert numeric['canonical_note_indices'] == [0]
    assert numeric['values'] == [1]
    assert numeric['availability_mask'] == [True]
    assert numeric['intervals'] == [[assignment['start_qn'], assignment['end_qn']]]


def test_candidate_is_retained_but_not_a_training_positive():
    piece, packet, assignment = fixture()
    assignment.update(origin='algorithmic_candidate', status='candidate', target_available=False, target_value=None)
    rows = module().validate_context_assignments(piece, packet, [assignment])
    numeric = module().numeric_context(rows, ['suspension'])
    assert numeric['values'] == [-1]
    assert numeric['availability_mask'] == [False]


@pytest.mark.parametrize('change', ['negative', 'outside', 'gap', 'wrong_note', 'wrong_related',
                                   'wrong_version', 'gold', 'confidence', 'candidate_admitted',
                                   'duplicate', 'fraction_not_reduced', 'wrong_piece', 'missing_source',
                                   'source_masked', 'source_gap', 'status_mask_conflict'])
def test_invalid_context_cannot_become_target(change):
    piece, packet, assignment = fixture()
    if change == 'negative': assignment['start_qn']['num'] = -1
    if change == 'outside': assignment['end_qn']['num'] = 5
    if change == 'gap': packet['harmonies'][1]['start_qn'] = {'num': 5, 'den': 2}
    if change == 'wrong_note': assignment['note_id'] = 'missing'
    if change == 'wrong_related': assignment['related_note_ids'] = ['missing']
    if change == 'wrong_version': assignment['source_version_id'] = 'other'
    if change == 'gold': assignment['human_gold'] = True
    if change == 'confidence': assignment['confidence'] = 0.9
    if change == 'candidate_admitted': assignment['origin'] = 'algorithmic_candidate'
    if change == 'fraction_not_reduced': assignment['end_qn'] = {'num': 6, 'den': 2}
    if change == 'wrong_piece': assignment['canonical_piece_id'] = 'piece:wrong'
    if change == 'missing_source': assignment['source_event_ids'] = ['missing']
    if change == 'source_masked': packet['note_segments'][0]['binding_available'] = False
    if change == 'source_gap': packet['note_segments'][0]['end_qn'] = {'num': 2, 'den': 1}
    if change == 'status_mask_conflict': assignment['target_available'] = False
    values = [assignment, deepcopy(assignment)] if change == 'duplicate' else [assignment]
    with pytest.raises(ValueError):
        module().validate_context_assignments(piece, packet, values)


def test_hash_authentication_precedes_json_decode(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('not JSON')
    with pytest.raises(ValueError, match='hash'):
        module().bound_json(dict(path=str(path), sha256='0' * 64, size_bytes=path.stat().st_size))


def test_refuses_existing_release_before_reading_music(tmp_path):
    with pytest.raises(FileExistsError):
        module().materialize_release({}, tmp_path)


def test_full_release_authenticates_reopens_and_preserves_raw_piece(tmp_path):
    import json
    from music_critic.data import dumps_piece
    from music_critic.tasks.dense_harmonic_targets import loads_dense_harmonic_targets

    piece, packet, assignment = fixture()
    m = module()

    def save(name, value):
        path = tmp_path / name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return m.binding(path)

    canonical = save('piece.json', dumps_piece(piece))
    before = Path(canonical['path']).read_bytes()
    packet.update(record_id='synthetic-record', case_id='synthetic', source_sha256='a' * 64,
                  source_version_id='synthetic-record@sha256:' + 'a' * 64, canonical_piece=canonical)
    assignment['source_version_id'] = packet['source_version_id']
    metadata = save('dilemma-data-manifest.json', {'records': [dict(
        record_id='synthetic-record', split='train', source_component_id='synthetic-component', piece=canonical)]})
    admission = save('admission.json', dict(record_id='synthetic-record', current_split='train',
                    known_family_heldout_conflict=False, metadata_bindings=[metadata],
                    source_expected=dict(record_id='synthetic-record', sha256='a' * 64,
                                         component_id='synthetic-component')))
    authored = save('authored.json', dict(schema='DenseContextRoleAssignments@1',
                    model_requested='gpt-6-astra', reasoning_effort_requested='ultra',
                    role_vocabulary=['suspension', 'neighbor_tone'], assignments=[assignment]))
    previous = save('old.json', {'assignments': []})
    manifest = dict(schema='DenseHarmonicTargetInputs@1', experiment_id='SYNTHETIC',
                    context_assignments=authored, works=[dict(case_id='synthetic',
                    record_id='synthetic-record', admission=admission, canonical_piece=canonical,
                    source_packet=save('packet.json', packet), previous_event_assignments=previous)])
    result = m.materialize_release(manifest, tmp_path / 'release')
    assert result['canonical_notes'] == 1 and result['numeric_batch_rows'] == 3
    assert result['numeric_available_by_field']['role'] == 2
    assert result['context_role_vocabulary'] == ['suspension', 'neighbor_tone']
    assert result['works'][0]['context_admitted_count'] == 1
    bundle = loads_dense_harmonic_targets(m.bound_bytes(result['works'][0]['dense_targets']), piece=piece)
    assert len(bundle['rows']) == 3
    numeric = m.bound_json(result['works'][0]['dense_numeric'])
    assert numeric['availability_masks']['role'] == [True, True, False]
    assert numeric['values']['role'] == [0, 0, -1]
    assert numeric['canonical_note_indices'] == [0, 0, 0]
    assert Path(canonical['path']).read_bytes() == before
    assert m.bound_json(result['works'][0]['previous_event_assignments_copy']) == {'assignments': []}
    with pytest.raises(FileExistsError):
        m.materialize_release(manifest, tmp_path / 'release')
    wrong = deepcopy(packet)
    wrong['record_id'] = 'other-record'
    changed_manifest = deepcopy(manifest)
    changed_manifest['works'][0]['source_packet'] = save('wrong-record.json', wrong)
    with pytest.raises(ValueError, match='identity'):
        m.materialize_release(changed_manifest, tmp_path / 'wrong-record-release')
    # A changed split snapshot must block even otherwise valid musical bytes.
    Path(metadata['path']).write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        m.materialize_release(manifest, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()
