from dataclasses import replace

import pytest

from music_critic.data import RationalTime as R
from music_critic.pll.contracts import CodecProfile, ContextProfile, Event, MaskQuery, ModelConfig


def test_exact_event_key_and_grace_contract():
    assert Event(R(1, 3), 0, 0, 60, R(2, 3)).key == (R(1, 3), 0, 0, 60, R(2, 3))
    Event(R(0), 0, 1, 60, R(0))
    with pytest.raises(ValueError):
        Event(R(0), 0, 0, 60, R(0))
    with pytest.raises((TypeError, ValueError)):
        Event(0.1, 0, 0, 60, R(1))


def test_query_requires_known_nonempty_geometry_and_distinct_tracks():
    MaskQuery(R(0), R(4), ('piano',))
    for args in [(R(4), R(4), ('piano',)), (R(0), R(4), ()), (R(0), R(4), ('piano', 'piano'))]:
        with pytest.raises(ValueError):
            MaskQuery(*args)


def test_profiles_fail_closed_and_config_identity_is_stable():
    for build in [lambda: CodecProfile(max_tracks=17), lambda: CodecProfile(max_denominator=65536),
                  lambda: ContextProfile(core_bars=0), lambda: ModelConfig(width=31)]:
        with pytest.raises(ValueError):
            build()
    a = ModelConfig()
    assert a.fingerprint() == ModelConfig().fingerprint()
    assert a.fingerprint() != replace(a, relation_components=2).fingerprint()
