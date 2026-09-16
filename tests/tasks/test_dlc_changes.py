import csv
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.tasks.dlc_changes import encode_presence, export_train_targets, parse_changes, source_events


@pytest.mark.parametrize(('value', 'tokens'), [
    ('64', ['6', '4']), ('^4^2', ['^4', '^2']), ('139', ['13', '9']),
    ('+b9-3v#9', ['+b9', '-3', 'v#9']), ('^#8', ['^#8']),
    ('##7^bb3', ['##7', '^bb3']), ('', []), ('44', ['4', '4']),
])
def test_lexical_atoms_not_temporal_sequence(value, tokens):
    assert [a['token'] for a in parse_changes(value)] == tokens


@pytest.mark.parametrize('value', ['64?', ' 64', '6 4', 'sus4', '(4)', 'b#3', '^', None, '<NA>', 'NaN'])
def test_full_consumption_and_missing(value):
    with pytest.raises(ValueError):
        parse_changes(value)


def row(**overrides):
    return dict({'unfolded_harmony_index': '1', 'changes': '64', 'chord': 'V(64)',
                 'valid_chord_label': 'True', 'quarterbeats_playthrough': '1/3',
                 'a_isOnset': 'True'}, **overrides)


def test_dedup_exact_time_and_no_fabricated_alignment():
    events = source_events([row(), row(quarterbeats_playthrough='2/3'),
                            row(unfolded_harmony_index='2')], 'dlc:fixture:a')
    assert len(events) == 2
    assert events[0]['source_row_numbers'] == [2, 3]
    assert events[0]['first_observed_note_onset_qn'] == '1/3'
    assert events[0]['canonical_alignment_status'] == 'not_performed'
    assert events[0]['target_mask']
    assert encode_presence(events[0], ['4', '6', '+9']) == ([1, 1, 0], [True]*3)


def test_known_empty_versus_missing_and_invalid():
    empty = source_events([row(changes='', chord='V')], 'dlc:fixture:a')[0]
    assert encode_presence(empty, ['4']) == ([0], [True])
    for override in ({'changes': None}, {'valid_chord_label': 'False'}, {'chord': ''},
                     {'unfolded_harmony_index': ''}):
        masked = source_events([row(**override)], 'dlc:fixture:a')[0]
        assert encode_presence(masked, ['4']) == ([0], [False])
    absent = row(); del absent['changes']
    assert not source_events([absent], 'dlc:fixture:a')[0]['target_mask']


def test_conflicts_never_become_negatives():
    event = source_events([row(), row(changes='4')], 'dlc:fixture:a')[0]
    assert 'conflicting_changes' in event['mask_reasons']
    assert encode_presence(event, ['4', '6']) == ([0, 0], [False, False])
    event = source_events([row(), row(chord='I')], 'dlc:fixture:a')[0]
    assert 'conflicting_chord_context' in event['mask_reasons']


def test_shorthand_expansion_and_unknown_vocabulary():
    event = source_events([row(changes='b3', chord='Ger6')], 'dlc:fixture:a')[0]
    assert event['target_mask'] and event['positive_tokens'] == ['b3']
    assert encode_presence(event, ['4']) == ([0], [False])


def fixture_manifest(tmp_path):
    source = tmp_path / 'pitch_arrays/DLC/fixture/a.tsv'
    source.parent.mkdir(parents=True)
    rows = [row(), row(), row(unfolded_harmony_index='2', changes='', chord='I'),
            row(unfolded_harmony_index='3', changes='+99?', valid_chord_label='False')]
    with source.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    manifest = {'root': str(tmp_path), 'records': [{'record_id': 'dlc:fixture:a',
        'component_id': 'fixture', 'split': 'train',
        'relative_path': str(source.relative_to(tmp_path)),
        'sha256': sha256(source.read_bytes()).hexdigest(), 'size_bytes': source.stat().st_size}]}
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    return path, manifest, source


def test_export_deterministic_dedup_support_and_no_overwrite(tmp_path):
    path, _, _ = fixture_manifest(tmp_path)
    out = tmp_path / 'one'
    receipt = export_train_targets(path, out)
    assert receipt['counts'] == {'events': 3, 'note_rows': 4, 'unmasked_events': 2,
                                'nonempty_unmasked_events': 1, 'empty_unmasked_events': 1,
                                'masked_events': 1, 'files': 1}
    vocabulary = json.loads((out / 'vocabulary.json').read_text())
    assert vocabulary['classes'] == ['4', '6']
    assert vocabulary['event_support'] == {'4': 1, '6': 1}
    assert export_train_targets(path, tmp_path / 'two') == receipt
    with pytest.raises(FileExistsError):
        export_train_targets(path, out)


def test_test_split_rejected_before_payload_access(tmp_path, monkeypatch):
    path, manifest, _ = fixture_manifest(tmp_path)
    manifest['records'].append(dict(manifest['records'][0], record_id='dlc:fixture:test', split='test'))
    path.write_text(json.dumps(manifest))
    original = Path.read_bytes
    def read(p):
        assert p == path, 'source must not be opened'
        return original(p)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(ValueError, match='TRAIN'):
        export_train_targets(path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('bad_path', ['../outside.tsv', '/outside.tsv'])
def test_path_safety(tmp_path, bad_path):
    path, manifest, _ = fixture_manifest(tmp_path)
    manifest['records'][0]['relative_path'] = bad_path
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe'):
        export_train_targets(path, tmp_path / 'out')


def test_hash_failure_has_no_completed_receipt(tmp_path):
    path, _, source = fixture_manifest(tmp_path)
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='binding mismatch'):
        export_train_targets(path, tmp_path / 'out')
    assert not (tmp_path / 'out/receipt.json').exists()
