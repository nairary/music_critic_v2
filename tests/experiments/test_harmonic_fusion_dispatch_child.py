import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def child():
    path=Path(__file__).resolve().parents[2]/"scripts/run_harmonic_fusion_pinned_request.py"
    spec=importlib.util.spec_from_file_location("fusion_child",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind,entry",[("reference","run_harmonic_fusion_reference_gate.py"),("pilot","run_harmonic_fusion_real_pilot.py")])
def test_child_pins_exact_request_and_uses_public_cli(tmp_path,monkeypatch,child,kind,entry):
    request=tmp_path/"request.json";request.write_bytes(b'{}\n')
    observed=[];monkeypatch.setattr(child.runpy,"run_path",lambda path,run_name:observed.append((path,run_name,list(sys.argv))))
    monkeypatch.setattr(sys,"argv",["before"])
    child.main([kind,str(request),str(tmp_path/"run")])
    assert Path(observed[0][0]).name==entry
    assert observed[0][1]=="__main__"
    assert observed[0][2][-3:]==["--execute","--output-root",str(tmp_path/"run")]
    assert observed[0][2][6]=="3"


@pytest.mark.parametrize("args",[[],["bad","a","b"],["reference","relative.json","/tmp/run"]])
def test_child_rejects_bad_arguments_before_reading(tmp_path,child,args):
    with pytest.raises(ValueError):child.main(args)
