"""Synthetic admission/identity/whole-file materialization, not musical quality."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.data import load_piece, dumps_piece, RationalTime as Q
from scripts.materialize_corpus400 import materialize_one, run


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def active_record(tmp_path, kind='corpus400_materialization'):
    from tests.experiments.test_registry import _record
    from music_critic.experiments.registry import seal_record, canonical_json_bytes
    record = _record(experiment_id='EXP-FIXTURE-MATERIALIZE-001', status='planned')
    record.update(phase='tonal-relations-T08', kind=kind)
    manifest = tmp_path / 'manifest.jsonl'; raw = manifest.read_bytes()
    record['protocol'] = {'admission_manifest': {'path': manifest.name, 'sha256': sha256(raw).hexdigest(), 'size_bytes': len(raw)}}
    path = tmp_path / (record['experiment_id'] + '.json'); path.write_bytes(canonical_json_bytes(seal_record(record)) + b'\n')
    return path


def source(tmp_path):
    piece = load_piece(ROOT / 'tests/fixtures/data/canonical_piece_v2.json')
    note = next(n for n in piece.notes if not n.is_percussion)
    piece = replace(piece, split='train', targets=(), annotations=(), quality_flags=(),
                    notes=(replace(note, onset_qn=Q(0), duration_qn=Q(1)),
                           replace(note, note_id='note:late', onset_qn=Q(4), duration_qn=Q(1))))
    raw = (dumps_piece(piece) + '\n').encode(); path = tmp_path / 'canonical.json'; path.write_bytes(raw)
    item = {'candidate_id': 'fixture:1', 'work_family_id': 'family:1', 'primary_genre': 'pop',
            'genre_evidence': ['fixture'], 'piece_id': piece.piece_id,
            'source': {'kind': 'canonical', 'binding': {'path': str(path), 'sha256': sha256(raw).hexdigest(), 'size_bytes': len(raw)}},
            'admission': {'status': 'admitted', 'current_split': 'train',
                          'known_heldout_family_conflict': False, 'evidence': ['fixture']}}
    return piece, item


def test_entire_piece_export_and_packet_keep_late_event_without_semantic_claim(tmp_path):
    piece, item = source(tmp_path)
    result = materialize_one(item, tmp_path / 'out')
    packet = json.loads(Path(result['inspection_packet']['path']).read_text())
    assert packet['note_count'] == len(piece.notes) == 2
    assert packet['note_rows'][-1][0] == 'note:late'
    assert packet['note_rows'][0][packet['note_columns'].index('spelling_step')] == piece.notes[0].spelling_step
    assert packet['cropped'] is False and packet['source_extent']['status'] == 'unverified'
    assert result['annotation_complete'] is False
    assert result['midi_event_coverage_complete'] is True
    assert Path(result['midi']['path']).read_bytes().startswith(b'MThd')
    receipt = json.loads(Path(result['midi_event_map']['path']).read_text())
    assert {n for route in receipt['routes'] for n in route['note_ids']} == {n.note_id for n in piece.notes}
    assert receipt['target_markers'] == 0


@pytest.mark.parametrize('change', [
    lambda x: x['admission'].update(current_split='test'),
    lambda x: x['admission'].update(known_heldout_family_conflict=True),
    lambda x: x['admission'].update(evidence=[]),
    lambda x: x.update(primary_genre='unknown'),
])
def test_rejected_admission_does_not_open_source_or_make_output(tmp_path, change):
    _, item = source(tmp_path); change(item)
    Path(item['source']['binding']['path']).unlink()
    with pytest.raises(ValueError):
        materialize_one(item, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_hash_mismatch_retains_receipt_and_does_not_export(tmp_path):
    _, item = source(tmp_path); item['source']['binding']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash/size'):
        materialize_one(item, tmp_path / 'out')
    assert (tmp_path / 'out' / 'admission.json').is_file()
    assert not (tmp_path / 'out' / 'score.mid').exists()


def test_existing_materialization_is_immutable(tmp_path):
    _, item = source(tmp_path); materialize_one(item, tmp_path / 'out')
    before = (tmp_path / 'out' / 'score.mid').read_bytes()
    with pytest.raises(FileExistsError):
        materialize_one(item, tmp_path / 'out')
    assert (tmp_path / 'out' / 'score.mid').read_bytes() == before


def test_catalog_scope_cannot_read_payload(tmp_path):
    _, item = source(tmp_path)
    manifest = tmp_path / 'manifest.jsonl'; manifest.write_text(json.dumps(item) + '\n')
    record = active_record(tmp_path, kind='metadata_candidate_selection')
    with pytest.raises(ValueError, match='scope'):
        run(manifest, tmp_path / 'out', record)
    assert not (tmp_path / 'out').exists()


def test_duplicate_work_family_rejected_before_payload(tmp_path):
    _, item = source(tmp_path); other = {**item, 'candidate_id': 'fixture:2'}
    manifest = tmp_path / 'manifest.jsonl'; manifest.write_text('\n'.join(map(json.dumps, [item, other])))
    record = active_record(tmp_path)
    with pytest.raises(ValueError, match='duplicate family'):
        run(manifest, tmp_path / 'out', record)
    assert not (tmp_path / 'out').exists()


def test_pdmx_materializes_full_raw_notes_without_inventing_theory(tmp_path):
    from tests.adapters.test_pdmx import _source
    _, item = source(tmp_path)
    path = tmp_path / 'raw.json'; path.write_text(json.dumps(_source()))
    item.update(source={'kind': 'pdmx', 'binding': {'path': str(path)}}, source_id='fixture-A')
    result = materialize_one(item, tmp_path / 'out')
    assert result['notes'] == 1 and result['native_theory'] is None
    assert result['annotation_complete'] is False
    assert (tmp_path / 'out' / 'source-binding.json').exists()
    extras = json.loads(Path(result['source_analytical_extras']['path']).read_text())
    assert extras['resolution'] == _source()['resolution']
    assert extras['tracks'][0]['track_id'] == 'track:000000'
    assert extras['tracks'][0]['chords'] == _source()['tracks'][0].get('chords', [])


def test_midi_materializes_without_analytic_targets(tmp_path):
    from scripts.export_genre_midi import export_preview_midi
    piece, item = source(tmp_path)
    payload, _ = export_preview_midi(piece)
    path = tmp_path / 'raw.mid'; path.write_bytes(payload)
    item.update(source={'kind': 'midi', 'binding': {'path': str(path)}},
                piece_id='piece:external-fixture',dataset_id='fixture_midi')
    result = materialize_one(item, tmp_path / 'out')
    assert result['notes'] == 2 and result['piece_id'] == 'piece:external-fixture'
    assert result['native_theory'] is None


def test_catalog_id_gets_safe_midi_piece_id(tmp_path):
    from scripts.export_genre_midi import export_preview_midi
    piece, item = source(tmp_path)
    path = tmp_path / 'raw.mid'; path.write_bytes(export_preview_midi(piece)[0])
    item.pop('piece_id'); item.update(candidate_id='candidate:abcdef', source={'kind': 'midi', 'binding': {'path': str(path)}})
    result = materialize_one(item, tmp_path / 'out')
    assert result['piece_id'].startswith('piece:corpus400-')


def test_invalid_midi_records_failure_and_continues(tmp_path):
    _, item = source(tmp_path)
    bad = {**item, 'candidate_id': 'candidate:bad', 'work_family_id': 'family:bad',
           'source': {'kind': 'midi', 'binding': {'path': str(tmp_path / 'bad.mid')}}}
    (tmp_path / 'bad.mid').write_bytes(b'not-midi')
    manifest = tmp_path / 'manifest.jsonl'; manifest.write_text('\n'.join(map(json.dumps, [bad, item])))
    result = run(manifest, tmp_path / 'out', active_record(tmp_path))
    assert result['success'] == result['failed'] == 1
    assert (tmp_path / 'out' / 'active-experiment-record.json').exists()


def test_unsealed_record_cannot_authorize_read(tmp_path):
    _, item = source(tmp_path)
    manifest = tmp_path / 'manifest.jsonl'; manifest.write_text(json.dumps(item))
    record = active_record(tmp_path)
    bad = json.loads(record.read_text()); bad['title'] = 'tampered'; record.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        run(manifest, tmp_path / 'out', record)
    assert not (tmp_path / 'out').exists()


def test_changed_admission_manifest_rejected_before_music(tmp_path):
    _, item = source(tmp_path)
    manifest = tmp_path / 'manifest.jsonl'; manifest.write_text(json.dumps(item))
    record = active_record(tmp_path)
    manifest.write_text(json.dumps(item) + '\n')
    with pytest.raises(ValueError, match='hash/size'):
        run(manifest, tmp_path / 'out', record)
    assert not (tmp_path / 'out').exists()


def test_percussion_only_is_not_usable_melodic_source(tmp_path):
    piece, item = source(tmp_path)
    percussion = next(t for t in piece.tracks if t.is_percussion)
    piece = replace(piece, notes=tuple(replace(n, is_percussion=True, track_id=percussion.track_id,
                                              program=percussion.program, channel=9) for n in piece.notes))
    raw = (dumps_piece(piece) + '\n').encode()
    path = Path(item['source']['binding']['path']); path.write_bytes(raw)
    item['source']['binding'].update(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    with pytest.raises(ValueError, match='usable'):
        materialize_one(item, tmp_path / 'out')


def test_stale_theory_canonical_binding_rejected(tmp_path):
    from tests.tasks.test_event_theory_enrichment import fixture
    from scripts.materialize_corpus400 import save_json
    piece, bundle = fixture()
    _, item = source(tmp_path)
    raw = (dumps_piece(piece) + '\n').encode(); path = tmp_path / 'piece2.json'; path.write_bytes(raw)
    item['piece_id'] = piece.piece_id
    item['source']['binding'] = {'path': str(path), 'sha256': sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    bundle['canonical_piece'] = {'path': str(path), 'sha256': '0' * 64, 'size_bytes': len(raw)}
    item['native_theory'] = save_json(tmp_path / 'theory.json', bundle)
    with pytest.raises(ValueError, match='canonical binding'):
        materialize_one(item, tmp_path / 'out')


def test_direct_cli_help_works_with_src_only_pythonpath():
    import os
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, 'scripts/materialize_corpus400.py', '--help'],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout and '--experiment-record' in result.stdout
