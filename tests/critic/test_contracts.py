"""Synthetic public contract checks for critic v1."""
from fractions import Fraction as F
import pytest
import torch
from music_critic.critic.contracts import FeatureBinding, RawNote, RawInterval


def test_bindings_reject_unknown_and_exact_intervals():
    with pytest.raises(ValueError):
        FeatureBinding(source_id='', checkpoint_id='x', adapter_version='v', dtype='float32')
    note = RawNote('n', 't', F(1, 3), F(2, 3), 60, False)
    assert note.end - note.start == F(1, 3)
    with pytest.raises(ValueError):
        RawInterval('b', F(2), F(1))


@pytest.mark.parametrize('kwargs',[{'source_id':42},{'source_id':'   '},{'model_state_id':42}])
def test_feature_binding_metadata_rejects_malformed_values(kwargs):
    values=dict(source_id='source',checkpoint_id='checkpoint',adapter_version='v1')
    values.update(kwargs)
    with pytest.raises(ValueError): FeatureBinding(**values)


def test_raw_geometry_flags_and_identity_are_not_coerced():
    with pytest.raises(ValueError): RawInterval('',F(0),F(1))
    with pytest.raises(ValueError): RawInterval('b',F(0),F(1),complete='false')
    with pytest.raises(ValueError): RawNote('n','t',F(0),F(1),60,is_drum='false')
