"""Synthetic source-to-target compiler checks; no musical truth fixtures."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/build_relational_midi_targets.py'
    spec = importlib.util.spec_from_file_location('build_relational_midi_targets', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def binding(event, note, start, end):
    return dict(source_event_id=event, canonical_note_id=note,
                source_start_qn=start, source_end_qn=end,
                canonical_start_qn=start, canonical_end_qn=end,
                binding_available=True, mask_reason=None)


def assignment():
    return dict(assignment_id='a1', source_event_ids=['e1'],
                related_source_event_ids=['e2'],
                interval=dict(start='1/1', end='2/1', time_unit='quarter_note'),
                source_version_id='version1', flag='anticipation',
                available=True, target_value='present', source='pseudo_label',
                confidence=None, human_gold=False, origin_status='admitted_weak')


def test_only_assigned_note_gets_flag_not_related_context():
    rows = [binding('e1', 'note:n1', '0', '2'), binding('e2', 'note:n2', '2', '3')]
    result = module().resolve_role_assignment(assignment(), rows)
    assert result['note_id'] == 'note:n1'
    assert result['related_note_ids'] == ['note:n2']
    assert (result['start_qn'], result['end_qn']) == ('1/1', '2/1')


def test_two_tie_segments_join_one_canonical_note_without_new_attack():
    rows = [binding('e1', 'note:n1', '0', '2'), binding('e2', 'note:n1', '2', '4')]
    a = assignment(); a.update(source_event_ids=['e1', 'e2'], related_source_event_ids=[])
    a['interval'].update(start='1/1', end='3/1')
    result = module().resolve_role_assignment(a, rows)
    assert result['note_id'] == 'note:n1'
    assert (result['start_qn'], result['end_qn']) == ('1/1', '3/1')


def test_unobserved_gap_is_not_filled_by_same_note_id():
    rows = [binding('e1', 'note:n1', '0', '1'), binding('e2', 'note:n1', '2', '4')]
    a = assignment(); a.update(source_event_ids=['e1', 'e2'], related_source_event_ids=[])
    a['interval'].update(start='0', end='3')
    with pytest.raises(ValueError, match='coverage'):
        module().resolve_role_assignment(a, rows)


@pytest.mark.parametrize('change', ['missing', 'masked', 'two_notes', 'seconds', 'duplicate'])
def test_invalid_binding_or_clock_cannot_create_available_target(change):
    a = assignment(); rows = [binding('e1', 'note:n1', '0', '2'), binding('e2', 'note:n2', '2', '3')]
    if change == 'missing': a['source_event_ids'] = ['absent']
    if change == 'masked': rows[0]['binding_available'] = False
    if change == 'two_notes': a['source_event_ids'] = ['e1', 'e2']
    if change == 'seconds': a['interval']['time_unit'] = 'measured_seconds'
    if change == 'duplicate': rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError): module().resolve_role_assignment(a, rows)


def test_source_offset_applies_to_both_endpoints_exactly():
    rows = [binding('e1', 'note:n1', '0', '2'), binding('e2', 'note:n2', '2', '3')]
    rows[0].update(canonical_start_qn='1/3', canonical_end_qn='7/3')
    rows[1].update(canonical_start_qn='7/3', canonical_end_qn='10/3')
    result = module().resolve_role_assignment(assignment(), rows)
    assert (result['start_qn'], result['end_qn']) == ('4/3', '7/3')


def test_hash_checked_before_decode(tmp_path):
    path = tmp_path / 'invalid.json'; path.write_text('not JSON')
    with pytest.raises(ValueError, match='SHA-256'):
        module().read_bound_json({'path': str(path), 'sha256': '0' * 64, 'size_bytes': path.stat().st_size})


def synthetic_release_inputs(tmp_path, *, canonical_split=None):
    """Authenticate only a tiny synthetic source, alignment and TRAIN piece."""
    from music_critic.data import dumps_piece

    fixture_path = Path(__file__).parent / 'analysis/test_target_binding.py'
    spec = importlib.util.spec_from_file_location('target_binding_fixture', fixture_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    piece, evidence, packet = helpers.fixture()
    piece = replace(piece, split=canonical_split)
    evidence = helpers.reseal(piece, evidence)
    compiler = module()

    def save(name, value):
        path = tmp_path / name
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')
        return compiler.file_binding(path)

    table = save('source.tsv', 'pitch\tonset\tduration\n60\t0\t2\n')
    version = f"{helpers.RID}@sha256:{table['sha256']}"
    packet.update(source_version_id=version, source_table=table)
    packet['events'][0]['source_version_id'] = version
    metadata = save('split.json', {helpers.RID: 'train'})
    admission = save('admission.json', {
        'current_split': 'train', 'known_family_heldout_conflict': False,
        'record_id': helpers.RID, 'metadata_bindings': [metadata],
    })
    authored = assignment()
    authored.update(
        case_id='synthetic-work', source_event_ids=['source:l2'],
        related_source_event_ids=[], source_version_id=version,
        relation_id='relation:synthetic-anticipation',
        annotation_provenance={
            'model_requested': 'gpt-6-astra', 'reasoning_effort_requested': 'ultra',
        },
    )
    manifest = {
        'schema': 'RelationalMIDITargetInputs@1', 'experiment_id': 'SYNTHETIC-ONLY',
        'source_assignments': save('roles.json', {
            'schema': 'RelationalSourceRoleTargets@1', 'assignments': [authored],
        }),
        'works': [{
            'case_id': 'synthetic-work', 'record_id': helpers.RID,
            'source_sha256': table['sha256'], 'admission': admission,
            'source_events': save('events.json', packet),
            'canonical_piece': save('canonical.json', dumps_piece(piece)),
            'raw_alignment': save('alignment.json', evidence),
        }],
    }
    return compiler, manifest, piece, Path(table['path'])


@pytest.mark.parametrize('canonical_split', [None, 'train'])
def test_materialize_release_reopens_usable_targets_and_never_overwrites(tmp_path, canonical_split):
    from music_critic.data import RationalTime
    from music_critic.tasks.multisource import loads_target_bundle
    from music_critic.tasks.relational_registry import RELATIONAL_FLAGS
    from music_critic.tasks.relational_targets import loads_relational_assignments

    compiler, manifest, piece, _ = synthetic_release_inputs(tmp_path, canonical_split=canonical_split)
    canonical_path = Path(manifest['works'][0]['canonical_piece']['path'])
    original_canonical = canonical_path.read_bytes()
    output = tmp_path / 'release'
    release = compiler.materialize_release(manifest, output)
    assert json.loads((output / 'index.json').read_text()) == release
    assert release['total_assignments'] == release['total_available_note_flags'] == 1
    work = release['works'][0]
    assert work['archived_piece_split'] == canonical_split
    assert work['split'] == 'train'
    assert work['canonical_piece'] == manifest['works'][0]['canonical_piece']
    assert work['admission'] == manifest['works'][0]['admission']
    bundle = loads_target_bundle(compiler.read_bound_bytes(work['target_bundle']))
    assert (bundle.dataset_id, bundle.piece_id) == (piece.dataset_name, piece.piece_id)
    assert {target.task_id for target in bundle.targets} == {
        f'theory.relations.{flag}' for flag in RELATIONAL_FLAGS
    }
    for target in bundle.targets:
        positive = target.task_id == 'theory.relations.anticipation'
        assert target.entity_ids == ('note:target-0',)
        assert target.values == (('present',) if positive else (None,))
        assert target.availability_mask == (positive,)
        assert target.source == (('pseudo_label',) if positive else (None,))
        assert target.confidence == (None,)
        assert (target.provenance_ids[0] is not None) is positive
    (role,) = loads_relational_assignments(compiler.read_bound_bytes(work['event_assignments']))
    assert role.note_id == 'note:target-0'
    assert (role.start_qn, role.end_qn) == (RationalTime(1), RationalTime(2))
    assert role.source_event_ids == ('source:l2',)
    assert role.model_requested == 'gpt-6-astra' and role.reasoning_effort == 'ultra'
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError, match='already exists'):
        compiler.materialize_release(manifest, output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert canonical_path.read_bytes() == original_canonical


def test_materialize_release_rejects_corrupt_source_before_publication(tmp_path):
    compiler, manifest, _, source_table = synthetic_release_inputs(tmp_path)
    source_table.write_text(source_table.read_text().replace('60', '61'), encoding='utf-8')
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='SHA-256 mismatch: .*source.tsv'):
        compiler.materialize_release(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize('canonical_split', ['validation', 'test'])
def test_materialize_release_never_relabels_embedded_heldout_split(tmp_path, canonical_split):
    compiler, manifest, _, _ = synthetic_release_inputs(tmp_path, canonical_split=canonical_split)
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='embedded canonical split conflicts with TRAIN admission'):
        compiler.materialize_release(manifest, output)
    assert not output.exists()
