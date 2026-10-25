from copy import deepcopy
from pathlib import Path
import pytest
from scripts import hcm1_admission as admission


def setup(monkeypatch, tmp_path):
    rows = {}
    for mode, identifier in admission.IDS.items():
        path = tmp_path / (identifier + '.json');path.write_text(mode)
        rows[identifier] = dict(experiment_id=identifier,record_fingerprint=mode,
            status='completed',hcm_diagnostic=dict(mode=mode,parent='parent',code='current',cases='fixed'),
            results=dict(native_admitted=True,full_state_exact=True,resume_verified=True,parent_update=10000))
    monkeypatch.setattr(admission,'load_record',lambda p:deepcopy(rows[Path(p).stem]))
    monkeypatch.setattr(admission,'binding',lambda a:dict(mode=a.mode,parent=a.parent_sha256,code='current',cases=str(a.cases)))
    return rows


def check(root, mode='train'):
    return admission.verify_prerequisites(mode,root,parent_sha256='parent',cases='fixed')


def test_completed_exact_prerequisites_produce_byte_receipts(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    assert check(tmp_path,'gate')==[]
    assert len(check(tmp_path,'evaluate'))==1
    rows=check(tmp_path);assert len(rows)==2 and all(len(r['sha256'])==64 for r in rows)


@pytest.mark.parametrize('mode', ['gate','evaluate'])
@pytest.mark.parametrize('field,value',[('parent','different'),('code','stale'),('cases','changed')])
def test_stale_prerequisite_is_rejected(monkeypatch,tmp_path,mode,field,value):
    rows=setup(monkeypatch,tmp_path)
    rows[admission.IDS[mode]]['hcm_diagnostic'][field]=value
    with pytest.raises(ValueError,match='binding mismatch'):check(tmp_path)


def test_complete_state_and_parent_progress_are_required(monkeypatch,tmp_path):
    rows=setup(monkeypatch,tmp_path)
    rows[admission.IDS['gate']]['results']['full_state_exact']=False
    with pytest.raises(ValueError,match='complete state'):check(tmp_path)
    rows[admission.IDS['gate']]['results']['full_state_exact']=True
    rows[admission.IDS['evaluate']]['results']['parent_update']=9999
    with pytest.raises(ValueError,match='parent progress'):check(tmp_path)


def test_nonterminal_prerequisite_is_rejected(monkeypatch,tmp_path):
    rows=setup(monkeypatch,tmp_path);rows[admission.IDS['evaluate']]['status']='running'
    with pytest.raises(ValueError,match='binding mismatch'):check(tmp_path)
