"""End-to-end inference on a generated synthetic MIDI, never corpus files."""
import json
import mido
import torch
from music_critic.critic.contracts import FeatureBinding
from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
from music_critic.critic.model import MultiscaleAspectCritic
from music_critic.critic.inference import CriticInference
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel


def test_one_midi_needs_no_reference_or_labels_and_exports_exact_coordinates(tmp_path):
    midi=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack(); midi.tracks.append(track)
    track.extend([mido.Message('note_on',note=60,velocity=80,time=0),
                  mido.Message('note_off',note=60,time=480),
                  mido.Message('note_on',note=64,velocity=80,time=0),
                  mido.Message('note_off',note=64,time=480)])
    path=tmp_path/'synthetic.mid'; midi.save(path)
    adapter=BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('synthetic','synthetic-random-init',C2_ADAPTER_VERSION))
    predictor=CriticInference(adapter,MultiscaleAspectCritic(),
        feature_binding=adapter.binding,critic_id='synthetic-untrained')
    result=predictor.score_midi(path)
    json.dumps(result,allow_nan=False)
    assert result['schema']=='CriticInferenceReport@1.0.0'
    assert result['overall'] is None  # A reference bank is mandatory for a scaled overall.
    assert result['musical_error_location']['available'] is False
    assert result['local']['melody/note'][0]['start_qn']==[0,1]
    assert result['local']['melody/note'][0]['end_qn']==[1,1]
    assert set(result['aspects'])=={'harmony','rhythm','melody','structure'}
    assert all(p.grad is None for p in adapter.model.parameters())


def test_bundles_and_reference_scales_are_bound_to_actual_predictor(tmp_path):
    from dataclasses import replace
    from fractions import Fraction
    import pytest
    from music_critic.critic.inference import save_critic_bundle,load_critic_bundle,predictor_identity
    from music_critic.critic.scoring import TrainReferenceBank,ReferenceRow
    from music_critic.critic.contracts import SUPPORT_POLICY
    adapter=BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('synthetic','synthetic-random-init',C2_ADAPTER_VERSION))
    critic=MultiscaleAspectCritic(width=8)
    name='synthetic-selected'
    identity=predictor_identity(critic,adapter.binding,name)
    bank=TrainReferenceBank('reference',(ReferenceRow('work',{'harmony':0.},'original'),),
        '4/1',SUPPORT_POLICY,{'original':1.},predictor_id=identity)
    predictor=CriticInference(adapter,critic,feature_binding=adapter.binding,critic_id=name,reference=bank)
    path=tmp_path/'critic.pt'
    save_critic_bundle(path,critic,adapter.binding,critic_id=name)
    rng=torch.get_rng_state().clone()
    restored,restored_name=load_critic_bundle(path,expected_binding=adapter.binding)
    assert torch.equal(rng,torch.get_rng_state())
    assert restored_name==name
    assert predictor_identity(restored,adapter.binding,name)==identity
    with pytest.raises(FileExistsError): save_critic_bundle(path,critic,adapter.binding,critic_id=name)
    with pytest.raises(ValueError,match='binding'):
        load_critic_bundle(path,expected_binding=replace(adapter.binding,source_id='changed'))
    with torch.no_grad(): next(critic.parameters()).add_(1.)
    with pytest.raises(ValueError,match='changed'):
        predictor.score_piece(None)  # Reject stale scales before reading any raw piece.
    with pytest.raises(ValueError,match='reference bank'):
        CriticInference(adapter,critic,feature_binding=adapter.binding,critic_id=name,reference=bank)


def test_inference_bundle_can_be_loaded_inside_inference_mode(tmp_path):
    from music_critic.critic.inference import save_critic_bundle,load_critic_bundle
    adapter=BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('synthetic','synthetic-random-init',C2_ADAPTER_VERSION))
    path=tmp_path/'bundle.pt'
    save_critic_bundle(path,MultiscaleAspectCritic(width=8),adapter.binding,critic_id='synthetic')
    with torch.inference_mode():
        head,ident=load_critic_bundle(path,expected_binding=adapter.binding)
        predictor=CriticInference(adapter,head,feature_binding=adapter.binding,critic_id=ident)
    assert predictor.predictor_id
    assert not any(p.is_inference() for p in head.parameters())
