"""Synthetic immutable-release selection and numeric export tests."""
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from music_critic.data import dumps_piece
from tests.tasks.test_event_theory_enrichment import fixture


def module():
    path=Path(__file__).parents[1]/"scripts/finalize_event_theory_release.py"
    spec=importlib.util.spec_from_file_location("finalize_enrichment",path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_complete_numeric_release_is_hash_bound_and_write_once(tmp_path):
    m=module();piece,bundle=fixture()
    raw=tmp_path/"canonical.json";raw.write_text(dumps_piece(piece))
    bind=lambda p:dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
    bundle["canonical_piece"]=bind(raw)
    path=tmp_path/"bundle.json.gz";path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    index=tmp_path/"index.jsonl"
    index.write_text(json.dumps(dict(status="completed",piece_id=piece.piece_id,bundle=bind(path),
        canonical_piece=bind(raw),work_family=['fixture','work']))+'\n')
    out=tmp_path/"release"
    second=tmp_path/'replacement.jsonl'
    second.write_text(json.dumps(dict(piece_id=piece.piece_id,bundle=bind(path),canonical_piece=bind(raw)))+'\n')
    summary=m.finalize([index,second],out)
    assert summary['pieces']==1 and summary['notes']==1 and summary['context_rows']==2
    rows=[json.loads(l) for l in (out/'index.jsonl').read_text().splitlines()]
    assert rows[0]['source_group_id']==piece.source_group_id
    assert rows[0]['source_metadata']['work_family']==['fixture','work']
    assert summary['superseded_records']==1
    numeric=json.loads(gzip.decompress(Path(rows[0]['numeric']['path']).read_bytes()))
    assert numeric['note_contexts']['masks']['native.member']==[True,False]
    with pytest.raises(FileExistsError):m.finalize([index],out)
