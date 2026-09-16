"""TorchVersion serialization regressions; CUDA observations are explicit fixtures."""
import json

import pytest
import torch
from torch.torch_version import TorchVersion

from music_critic.experiments.analysisgnn import corrected_training, harmonic_fusion_resume
from music_critic.experiments.analysisgnn import harmonic_fusion_dispatch as setup
from music_critic.experiments.analysisgnn import harmonic_fusion_pilot as pilot
from music_critic.experiments.analysisgnn.contracts import fingerprint


@pytest.fixture
def runtime(monkeypatch):
    observed={"python":"fixture-python", "platform":"fixture-platform",
        "torch":TorchVersion("2.2.2+cu118"), "numpy":"1.26.4",
        "packages":{"numpy":"1.26.4","torch":"2.2.2+cu118","torch-geometric":"2.6.1"},
        "torch_cuda_build":"11.8", "cudnn_version":8700, "cuda_available":True,
        "cuda_device_count":1, "gpu":"fixture RTX 3090", "cublas_workspace_config":":4096:8"}
    monkeypatch.setattr(corrected_training,"environment_fingerprint",lambda:{**observed,"fingerprint":fingerprint(observed)})
    monkeypatch.setattr(torch,"get_num_threads",lambda:1)
    monkeypatch.setattr(torch,"get_num_interop_threads",lambda:1)
    monkeypatch.setattr(torch,"set_num_threads",lambda value:None)
    monkeypatch.setattr(torch,"set_num_interop_threads",lambda value:None)
    monkeypatch.setattr(torch,"are_deterministic_algorithms_enabled",lambda:True)
    monkeypatch.setattr(torch,"use_deterministic_algorithms",lambda value:None)
    for obj,key,value in ((torch.backends.cudnn,"benchmark",False),
                          (torch.backends.cudnn,"deterministic",True),
                          (torch.backends.cudnn,"allow_tf32",False),
                          (torch.backends.cuda.matmul,"allow_tf32",False)):
        monkeypatch.setattr(obj,key,value)
    for key in pilot.THREAD_ENV:monkeypatch.setenv(key,"1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",setup.GPU_UUID)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    return observed


def test_torchversion_normalizes_without_changing_canonical_runtime_fingerprint(runtime):
    observed=pilot.runtime_observations("cuda")
    assert type(runtime["torch"]) is TorchVersion
    assert type(observed["torch"]) is str and observed["torch"]=="2.2.2+cu118"
    assert fingerprint(observed)==fingerprint({**observed,"torch":runtime["torch"]})
    pilot.validate_runtime_policy(observed)


def test_capture_runtime_serializes_admissible_identity_with_torch22_metadata(tmp_path,monkeypatch,runtime):
    calls=[]
    monkeypatch.setattr(harmonic_fusion_resume,"prepare_training_resume_runtime",
        lambda device,**kwargs:calls.append((device,kwargs)))
    def forbidden(*args,**kwargs):pytest.fail("fixture capture must not access CUDA")
    monkeypatch.setattr(torch.cuda,"init",forbidden)
    path=tmp_path/"runtime.json"
    pin=setup.capture_runtime(path)
    captured=pilot.checked_json(pin)
    pilot.validate_runtime_policy(captured)
    assert captured["torch"]=="2.2.2+cu118"
    assert fingerprint(captured)==fingerprint(pilot.runtime_observations("cuda"))
    assert calls==[("cuda",{"expected_cuda_device_count":1})]


@pytest.mark.parametrize("invalid",[None,"",22])
def test_version_normalization_never_fabricates_missing_identity(runtime,invalid):
    runtime["torch"]=invalid
    observed=pilot.runtime_observations("cuda")
    assert observed["torch"] is invalid
    with pytest.raises(ValueError,match="runtime_identity_required"):
        pilot.validate_runtime_policy(observed)
