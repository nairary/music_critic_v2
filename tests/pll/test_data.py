"""Synthetic local manifests; no external datasets are accessed."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import pytest

from music_critic.data import load_piece
from music_critic.data.serialization import dumps_piece
from music_critic.pll.data import DataManifest, MaskSampler, seal_manifest


def manifest_fixture(tmp_path, *, empty=False):
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    piece = replace(piece, split='train', notes=() if empty else piece.notes)
    raw = dumps_piece(piece).encode()
    (tmp_path/'piece.json').write_bytes(raw)
    record = dict(record_id='record1', source=piece.dataset_name, work_id='component1',
                  source_group_id=piece.source_group_id, piece_id=piece.piece_id, split='train',
                  path='piece.json', sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    payload = seal_manifest(dict(schema='PLLDataManifest@1', records=[record],
        source_weights={'synthetic':1.}, split_provenance='synthetic fixture identities',
        group_before_window=True, cross_source_components_declared=True))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(payload))
    return path, payload


def test_hash_bound_train_data_and_exact_sampler_resume(tmp_path):
    path,_=manifest_fixture(tmp_path)
    data=DataManifest.load(path)
    sampler=MaskSampler(data,seed=17)
    first=sampler.draw()
    state=json.loads(json.dumps(sampler.state_dict()))
    expected=[sampler.draw() for _ in range(5)]
    restored=MaskSampler(data,seed=17)
    restored.load_state_dict(state)
    assert [restored.draw() for _ in range(5)] == expected
    assert first.example.context.query.start in {b.start_qn for b in data.load_piece('record1').bars}


def test_test_entry_is_rejected_before_opening_any_piece(tmp_path):
    path,payload=manifest_fixture(tmp_path)
    payload['records'][0]['split']='test'
    payload['records'][0]['path']='not-present.json'
    path.write_text(json.dumps(seal_manifest(payload)))
    with pytest.raises(ValueError,match='TEST_closed'):
        DataManifest.load(path)


def test_piece_hash_and_canonical_split_are_both_enforced(tmp_path):
    path,payload=manifest_fixture(tmp_path)
    data=DataManifest.load(path)
    (tmp_path/'piece.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes_changed'):
        data.load_piece('record1')
    path,payload=manifest_fixture(tmp_path)
    raw=(tmp_path/'piece.json').read_text().replace('"train"','"validation"')
    (tmp_path/'piece.json').write_text(raw)
    payload['records'][0].update(sha256=sha256(raw.encode()).hexdigest(),size_bytes=len(raw.encode()))
    path.write_text(json.dumps(seal_manifest(payload)))
    with pytest.raises(ValueError,match='split_mismatch'):
        DataManifest.load(path).load_piece('record1')


def test_sampler_draw_is_independent_of_hidden_note_content(tmp_path):
    from music_critic.data.serialization import loads_piece
    path,payload=manifest_fixture(tmp_path)
    expected=MaskSampler(DataManifest.load(path),seed=19).draw().example.context.query
    p=loads_piece((tmp_path/'piece.json').read_text())
    raw=dumps_piece(replace(p,notes=(),targets=(),annotations=())).encode()
    (tmp_path/'piece.json').write_bytes(raw)
    payload['records'][0].update(sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    path.write_text(json.dumps(seal_manifest(payload)))
    other=MaskSampler(DataManifest.load(path),seed=19).draw()
    assert other.example.context.query==expected
    assert other.example.targets==()


@pytest.mark.parametrize('change',['group','path','identity','fingerprint','source_weights'])
def test_manifest_binding_failures(tmp_path,change):
    path,payload=manifest_fixture(tmp_path)
    if change=='group':
        second={**payload['records'][0],'record_id':'second','work_id':'different'}
        payload['records'].append(second)
    elif change=='path':
        payload['records'][0]['path']='../piece.json'
    elif change=='identity':
        payload['records'][0]['piece_id']='wrong'
    elif change=='source_weights':
        payload['source_weights']={'wrong':1.}
    else:
        payload['fingerprint']='0'*64
    path.write_text(json.dumps(payload if change=='fingerprint' else seal_manifest(payload)))
    with pytest.raises(ValueError):
        data=DataManifest.load(path)
        data.load_piece('record1')


def test_sampler_state_cannot_resume_different_protocol(tmp_path):
    path,_=manifest_fixture(tmp_path)
    data=DataManifest.load(path)
    state=MaskSampler(data,seed=1).state_dict()
    with pytest.raises(ValueError,match='state_mismatch'):
        MaskSampler(data,seed=2).load_state_dict(state)
