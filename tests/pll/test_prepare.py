"""Synthetic source preparation; never opens project corpora."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.data import load_piece, dumps_piece
from music_critic.pll.data import DataManifest
from music_critic.pll.prepare import SourceCandidate, PrepareConfig, prepare_sources


def source(tmp_path, **changes):
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    piece = replace(piece, split='train', targets=(), annotations=())
    raw = dumps_piece(piece).encode()
    path = tmp_path / 'source.json'
    path.write_bytes(raw)
    candidate = SourceCandidate(record_id='synthetic:one', source=piece.dataset_name,
        piece_id=piece.piece_id, source_group_id=piece.source_group_id,
        work_id='known-fixture-component', split='train', path=str(path),
        sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    return replace(candidate, **changes)


def test_prepare_train_only_and_preserves_exact_events(tmp_path):
    train = source(tmp_path)
    closed = replace(train, record_id='closed', split='test', path='/must-not-open',
                     work_id='held-out-work',source_group_id='held-out-group')
    out = tmp_path / 'out'
    result = prepare_sources((train, closed), out, config=PrepareConfig(window_bars=2),
        source_weights={'synthetic': 1.}, split_provenance='synthetic assignment')
    manifest = DataManifest.load(out / 'manifest.json')
    prepared = manifest.load_piece(manifest.records[0].record_id)
    original = load_piece(train.path)
    assert [(n.onset_qn,n.duration_qn,n.pitch) for n in prepared.notes] == [
        (n.onset_qn,n.duration_qn,n.pitch) for n in original.notes]
    assert not prepared.targets and not prepared.annotations
    assert result['train_payloads_read'] == 1
    assert result['held_out_payloads_read'] == 0
    assert result['metadata_excluded_nontrain'] == 1
    assert manifest.records[0].work_id == train.work_id
    with pytest.raises(FileExistsError):
        prepare_sources((train,), out, config=PrepareConfig(window_bars=2),
            source_weights={'synthetic':1.},split_provenance='fixture')


def test_group_split_conflict_is_rejected_before_payload_or_output(tmp_path):
    train=source(tmp_path,path='/must-not-open')
    closed=replace(train,record_id='heldout',split='test')
    with pytest.raises(ValueError,match='group_crosses_split'):
        prepare_sources((train,closed),tmp_path/'out',source_weights={'synthetic':1.},
                        split_provenance='fixture')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('damage',['hash','embedded_split','identity'])
def test_bad_source_fails_with_immutable_read_receipt(tmp_path,damage):
    c=source(tmp_path)
    if damage=='hash': c=replace(c,sha256='0'*64)
    else:
        piece=load_piece(c.path)
        piece=replace(piece,quality_flags=(),**({'split':'test'} if damage=='embedded_split' else {'piece_id':'piece:other'}))
        raw=dumps_piece(piece).encode()
        Path(c.path).write_bytes(raw)
        c=replace(c,sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    with pytest.raises(ValueError,match={'hash':'hash_mismatch','embedded_split':'embedded_nontrain',
                                       'identity':'identity_mismatch'}[damage]):
        prepare_sources((c,),tmp_path/'out',source_weights={'synthetic':1.},split_provenance='fixture')
    receipt=json.loads((tmp_path/'out/prepare-receipt.json').read_text())
    assert receipt['status']=='failed'
    assert not receipt['accesses'][0]['verified']
    assert receipt['held_out_payloads_read']==int(damage=='embedded_split')
    assert not (tmp_path/'out/manifest.json').exists()


def test_clipped_windows_are_reported_and_never_written(tmp_path):
    from music_critic.data import RationalTime as Q
    c=source(tmp_path)
    piece=load_piece(c.path)
    crossing=replace(piece.notes[0],duration_qn=Q(2))
    raw=dumps_piece(replace(piece,notes=(crossing,*piece.notes[1:]))).encode()
    Path(c.path).write_bytes(raw)
    c=replace(c,sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    with pytest.raises(ValueError,match='all_requested_sources_required'):
        prepare_sources((c,),tmp_path/'out',config=PrepareConfig(window_bars=1),
            source_weights={'synthetic':1.},split_provenance='fixture')
    receipt=json.loads((tmp_path/'out/prepare-receipt.json').read_text())
    assert {r['reason'] for r in receipt['exclusions']}=={'clipped_notes_forbidden'}
    assert not tuple((tmp_path/'out/pieces').iterdir())


def test_equal_observations_union_known_work_ids(tmp_path):
    c=source(tmp_path)
    other=replace(c,record_id='synthetic:two',work_id='known-other-component')
    report=prepare_sources((c,other),tmp_path/'out',config=PrepareConfig(window_bars=2),
                          source_weights={'synthetic':1.},split_provenance='fixture')
    manifest=DataManifest.load(tmp_path/'out/manifest.json')
    assert len({r.work_id for r in manifest.records})==1
    assert report['cross_source_near_duplicate_coverage']=='unknown'


def test_full_visible_resource_bound_covers_every_masked_subset(tmp_path,monkeypatch):
    from music_critic.pll import prepare
    from music_critic.pll.masking import split_masked_piece,build_observed_tiles
    from music_critic.pll.contracts import MaskQuery
    from music_critic.pll.graph import build_tile_graph
    counts=[]
    def counted(tile,profile):
        g=build_tile_graph(tile,profile)
        counts.append((g.slice_features.shape[0],sum(g.base[e].edge_index.shape[1] for e in g.base.edge_types)
                       +sum(v.shape[1] for v in g.edges.values())))
        return g
    monkeypatch.setattr(prepare,'build_tile_graph',counted)
    c=source(tmp_path)
    config=PrepareConfig(window_bars=2)
    prepare_sources((c,),tmp_path/'out',config=config,source_weights={'synthetic':1.},split_provenance='fixture')
    full=counts[0]
    p=load_piece(c.path)
    for first in range(len(p.bars)):
        for last in range(first,len(p.bars)):
            for tracks in [(t.track_id,) for t in p.tracks]+[tuple(t.track_id for t in p.tracks)]:
                a,b=p.bars[first],p.bars[last]
                ex=split_masked_piece(p,MaskQuery(a.start_qn,b.start_qn+b.duration_qn,tracks))
                for tile in build_observed_tiles(ex.context):
                    counted(tile,config.context)
                    assert all(masked<=unmasked for masked,unmasked in zip(counts[-1],full))


def test_pop_annotation_channel_never_enters_output(tmp_path):
    c=source(tmp_path)
    p=load_piece(c.path)
    p=replace(p,dataset_name='pop909_cl',notes=tuple(replace(n,channel=1) for n in p.notes),
              tracks=tuple(replace(t,channel=1) for t in p.tracks))
    raw=dumps_piece(p).encode()
    Path(c.path).write_bytes(raw)
    c=replace(c,source='pop909_cl',sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    # The synthetic altered fixture uses channel1, forbidden for POP909-CL.
    with pytest.raises(ValueError,match='annotation_channel_forbidden'):
        prepare_sources((c,),tmp_path/'out',source_weights={'pop909_cl':1.},split_provenance='fixture')


def test_archive_failure_preserves_unknown_read_receipt(tmp_path):
    import importlib.util,io,tarfile
    script=Path(__file__).parents[2]/'scripts/prepare_pll_sources.py'
    spec=importlib.util.spec_from_file_location('prepare_pll_cli',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'archive';root.mkdir()
    raw=b'not canonical'
    part=root/'part.tar.gz'
    with tarfile.open(part,'w:gz') as tf:
        info=tarfile.TarInfo('records/x/piece.json');info.size=len(raw)
        tf.addfile(info,io.BytesIO(raw))
    manifest={'records':[{'record_id':'synthetic:x','source_component_id':'group:x','split':'train',
        'piece':{'path':'/srv/source/records/x/piece.json','sha256':sha256(raw).hexdigest(),'size_bytes':len(raw)}}]}
    manifest_raw=json.dumps(manifest).encode();(root/'data-manifest.json').write_bytes(manifest_raw)
    part_raw=part.read_bytes()
    receipt={'destination':'/srv/source','relocated_manifest_sha256':sha256(manifest_raw).hexdigest(),
        'test_payload_packaged':False,'parts':[{'path':part.name,'sha256':sha256(part_raw).hexdigest(),'size_bytes':len(part_raw)}]}
    (root/'receipt.json').write_text(json.dumps(receipt))
    stage=tmp_path/'out/stage'
    with pytest.raises(ValueError):module.dilemma_candidates(root,stage,17,1)
    audit=json.loads((stage.parent/'archive-access.json').read_text())
    assert audit['status']=='failed'
    assert len(audit['accesses'])==1
    assert audit['test_payloads_read'] is None
    assert audit['unverified_accesses']==1


def test_native_heldout_hint_excludes_entire_group_before_payload_stat(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace as NS
    script=Path(__file__).parents[2]/'scripts/prepare_pll_sources.py'
    spec=importlib.util.spec_from_file_location('prepare_pll_cli',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assignments=[];indexes={}
    for source_name in ('hooktheory','pop909_cl'):
        records=[]
        for suffix,hint,component in [('heldout','test','bad'),('train_alias','train','bad'),('ok','train','good')]:
            if source_name=='pop909_cl': hint='train'
            piece_id=source_name+':'+suffix
            records.append(NS(piece_id=piece_id,source_group_id=component,lineage_group_id=component,
                suggested_split=hint,canonical_relative_path=suffix+'.json',canonical_sha256='0'*64))
            assignments.append(NS(dataset_id=source_name,piece_id=piece_id,source_group_id=component,
                lineage_group_id=component,component_fingerprint=component,split='train'))
        indexes[source_name]=NS(header=NS(index_fingerprint=source_name,adapter_version='fixture'),records=records)
        cache=tmp_path/source_name/'multisource-cache-v1.0.0';cache.mkdir(parents=True)
        (cache/'ok.json').write_text('not opened; metadata stat only')
    monkeypatch.setattr(module,'load_split_manifest',lambda _:NS(assignments=assignments,
        index_fingerprints=[(s,s) for s in indexes]))
    monkeypatch.setattr(module,'load_corpus_index',lambda p,**_:indexes[p.name.removesuffix('.index.json')])
    monkeypatch.setattr(module,'binding',lambda p:{'path':str(p),'sha256':'0'*64,'size_bytes':1})
    candidates,metadata=module.candidates_from_indexes(tmp_path,17,256)
    assert {c.record_id for c in candidates}=={'hooktheory:ok','pop909_cl:ok'}
    assert all(row.get('native_heldout_components_excluded_before_payload',1)==1 for row in metadata)
