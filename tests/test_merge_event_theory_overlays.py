"""An authored layer is bound to one immutable native piece."""
import importlib.util
import json
from pathlib import Path

import pytest

from music_critic.data import dumps_piece
from tests.tasks.test_event_theory_enrichment import fixture


def module():
    path=Path(__file__).parents[1]/'scripts/merge_event_theory_overlays.py'
    spec=importlib.util.spec_from_file_location('merge_overlays',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_merge_exports_numeric_ready_target_and_preserves_base(tmp_path):
    m=module();piece,bundle=fixture()
    raw=tmp_path/'piece.json';raw.write_text(dumps_piece(piece))
    bundle['canonical_piece']=m.binding(raw)
    base=tmp_path/'base.json';base.write_text(json.dumps(bundle))
    base_index=tmp_path/'base.jsonl'
    base_index.write_text(json.dumps(dict(piece_id=piece.piece_id,bundle=m.binding(base),
                                         canonical_piece=m.binding(raw)))+'\n')
    overlay=dict(schema='EventTheoryInterpretations@1',piece_id=piece.piece_id,
        base_sha256=m.binding(base)['sha256'],model_requested='gpt-6-astra',
        reasoning_effort_requested='ultra',human_gold=False,note_fields=[dict(context_id='c0',fields={
        'analysis.role':dict(value='goal',available=True,origin='astra_individual_context',confidence=None,
                            evidence=['fixture'],unknown_reason=None)})])
    path=tmp_path/'overlay.json';path.write_text(json.dumps(overlay))
    overlay_index=tmp_path/'overlay.jsonl'
    overlay_index.write_text(json.dumps(dict(piece_id=piece.piece_id,overlay=m.binding(path)))+'\n')
    summary=m.merge_indexes([base_index],[overlay_index],tmp_path/'merged')
    assert summary['pieces']==1 and summary['available_authored_fields']==1
    assert json.loads(base.read_text())==bundle
    with pytest.raises(FileExistsError):m.merge_indexes([base_index],[overlay_index],tmp_path/'merged')
