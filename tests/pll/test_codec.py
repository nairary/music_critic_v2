"""Synthetic exact serialization and prefix-grammar contracts."""

import pytest

from music_critic.data.timing import RationalTime as R
from music_critic.pll.codec import EOS, NOTE, EventCodec
from music_critic.pll.contracts import CodecProfile, Event


def test_empty_answer_has_one_eos_and_no_continuation():
    codec = EventCodec(CodecProfile(max_denominator=4, max_numerator=10))
    assert codec.encode((), span=R(4), tracks=(0,)) == (EOS,)
    assert codec.decode((EOS,), span=R(4), tracks=(0,)) == ()
    assert codec.allowed((), span=R(4), tracks=(0,)) == (NOTE, EOS)
    assert codec.allowed((EOS,), span=R(4), tracks=(0,)) == ()
    with pytest.raises(ValueError, match="EOS|continuation|token"):
        codec.decode((EOS, EOS), span=R(4), tracks=(0,))


def test_hand_worked_integer_encoding_and_prefix_trace():
    codec = EventCodec(CodecProfile(max_events=2, max_denominator=4, max_numerator=10))
    event = Event(R(0), 0, 0, 60, R(1))
    # NOTE, denominator1, numerator0, track0, pitched, C, MIDI octave5,
    # duration denominator1/numerator1, EOS. Values are independently written.
    expected = (0, 2, 11, 2, 10, 266, 282, 286, 303, 2, 11, 2, 11, 1)
    assert codec.encode((event,), span=R(4), tracks=(0,)) == expected
    assert codec.decode(expected, span=R(4), tracks=(0,)) == (event,)
    trace = codec.trace(expected, span=R(4), tracks=(0,))
    for cut in range(len(expected) + 1):
        assert codec.trace_prefix(expected[:cut], span=R(4), tracks=(0,)) == trace[:cut]
    assert [step.field_index for step in trace] == list(range(13)) + [0]
    assert [step.event_index for step in trace] == [0] * 13 + [1]
    assert codec.event_token_groups(expected, span=R(4), tracks=(0,)) == (expected[:-1], (EOS,))


def test_roundtrip_uncommon_fractions_grace_drums_duplicates_and_long_tail():
    codec = EventCodec(CodecProfile(max_denominator=263, max_numerator=65537))
    events = (
        Event(R(257, 263), 1, 0, 127, R(65537, 257)),
        Event(R(0), 0, 3, 35, R(0)),
        Event(R(0), 0, 1, 71, R(0)),
        Event(R(0), 0, 1, 72, R(3, 7)),
        Event(R(0), 0, 1, 72, R(3, 7)),
    )
    tokens = codec.encode(events, span=R(1), tracks=(1, 0))
    assert codec.decode(tokens, span=R(1), tracks=(0, 1)) == tuple(sorted(events, key=lambda e: e.key))
    assert codec.encode(reversed(events), span=R(1), tracks=(0, 1)) == tokens
    assert codec.encode(codec.decode(tokens, span=R(1), tracks=(0, 1)), span=R(1), tracks=(0, 1)) == tokens
    trace = codec.trace(tokens, span=R(1), tracks=(0, 1))
    assert "drum" in {step.field for step in trace}
    assert max(step.field_index for step in trace) > 12
    assert all(token in step.allowed for token, step in zip(tokens, trace))


def test_pitch_class_order_is_not_midi_pitch_order():
    from music_critic.pll.codec import PC_BASE, OCTAVE_BASE
    codec = EventCodec(CodecProfile(max_denominator=1, max_numerator=2))
    low = Event(R(0), 0, 0, 71, R(1))  # B4
    high = Event(R(0), 0, 0, 72, R(1))  # C5
    tokens = codec.encode((low, high), span=R(1), tracks=(0,))
    trace = codec.trace(tokens, span=R(1), tracks=(0,))
    pcs = [step for step in trace if step.field == 'pitch_class']
    assert PC_BASE in pcs[1].allowed  # C5 remains legal after B4.
    octaves = [step for step in trace if step.field == 'octave']
    assert OCTAVE_BASE + 5 not in octaves[1].allowed
    assert OCTAVE_BASE + 6 in octaves[1].allowed
    assert codec.decode(tokens, span=R(1), tracks=(0,)) == (low, high)


@pytest.mark.parametrize('mutation', ['leading_zero', 'unreduced', 'zero_alias', 'out_of_span', 'unfinished', 'bos', 'bool'])
def test_rejects_noncanonical_or_unsupported_streams(mutation):
    from music_critic.pll.codec import BOS, BYTE_BASE, LEN_BASE
    codec = EventCodec(CodecProfile(max_denominator=4, max_numerator=300))
    tokens = list(codec.encode((Event(R(0), 0, 0, 60, R(1)),), span=R(1), tracks=(0,)))
    if mutation == 'leading_zero':
        tokens[1:3] = [LEN_BASE + 1, BYTE_BASE, BYTE_BASE + 1]
    elif mutation == 'unreduced':
        tokens[-4:] = [BYTE_BASE + 2, LEN_BASE, BYTE_BASE + 2, EOS]
    elif mutation == 'zero_alias':
        tokens[2] = BYTE_BASE + 2
    elif mutation == 'out_of_span':
        tokens[4] = BYTE_BASE + 1
    elif mutation == 'unfinished':
        tokens.pop()
    elif mutation == 'bos':
        tokens[0] = BOS
    else:
        tokens[0] = False
    with pytest.raises(ValueError):
        codec.decode(tokens, span=R(1), tracks=(0,))


def test_canonical_order_and_cap_are_enforced_in_grammar():
    codec = EventCodec(CodecProfile(max_events=2, max_denominator=3, max_numerator=5))
    first = Event(R(1, 2), 1, 2, 70, R(3))
    later = Event(R(1, 2), 1, 2, 70, R(4))
    tokens = codec.encode((first, later), span=R(1), tracks=(0, 1))
    trace = codec.trace(tokens, span=R(1), tracks=(0, 1))
    assert trace[-1].allowed == (EOS,)
    # Reversing whole event groups is invalid, including the duration tie break.
    groups = codec.event_token_groups(tokens, span=R(1), tracks=(0, 1))
    with pytest.raises(ValueError):
        codec.decode(groups[1] + groups[0] + (EOS,), span=R(1), tracks=(0, 1))
    with pytest.raises(ValueError, match='count'):
        codec.encode((first, first, first), span=R(1), tracks=(0, 1))
    # With a repeated event, grammar reaches a complete valid duplicate.
    duplicate = codec.encode((first, first), span=R(1), tracks=(0, 1))
    assert codec.decode(duplicate, span=R(1), tracks=(0, 1)) == (first, first)


def test_no_silent_rounding_or_target_clipping():
    codec = EventCodec(CodecProfile(max_denominator=5, max_numerator=5))
    for event in (Event(R(1, 7), 0, 0, 60, R(1)), Event(R(0), 0, 0, 60, R(6)),
                  Event(R(1), 0, 0, 60, R(1)), Event(R(0), 1, 0, 60, R(1))):
        with pytest.raises(ValueError):
            codec.encode((event,), span=R(1), tracks=(0,))


@pytest.mark.parametrize('span,tracks', [(R(0), (0,)), (R(1), ()), (R(1), (0, 0)), (R(1), (16,)), (R(1), (True,))])
def test_invalid_public_queries_fail_closed(span, tracks):
    codec = EventCodec()
    with pytest.raises(ValueError):
        codec.allowed((), span=span, tracks=tracks)


def test_exhaustive_tiny_profile_support_and_normalized_completion_mass():
    """All multisets for a one-event profile, independent finite event oracle.

    Equality of every grammar edge with the oracle trie proves no dead ends,
    missing completions or alternate encodings for this exhaustive tiny case.
    """
    from fractions import Fraction

    codec = EventCodec(CodecProfile(max_events=1, max_denominator=1, max_numerator=1))
    streams = {(EOS,)}
    for kind in range(4):
        for pitch in range(128):
            for duration in ((0, 1) if kind % 2 else (1,)):
                event = Event(R(0), 0, kind, pitch, R(duration))
                streams.add(codec.encode((event,), span=R(1), tracks=(0,)))
    assert len(streams) == 769  # 1 empty + 256 ordinary + 512 grace.
    trie = {}
    for stream in streams:
        for cut in range(len(stream)):
            trie.setdefault(stream[:cut], set()).add(stream[cut])
    probability = {(): Fraction(1)}
    complete_mass = Fraction(0)
    for prefix in sorted(trie, key=len):
        expected = trie[prefix]
        allowed = codec.allowed(prefix, span=R(1), tracks=(0,))
        assert set(allowed) == expected
        for token in allowed:
            next_prefix = (*prefix, token)
            mass = probability[prefix] / len(allowed)
            if token == EOS:
                complete_mass += mass
                assert codec.allowed(next_prefix, span=R(1), tracks=(0,)) == ()
            else:
                probability[next_prefix] = mass
    assert complete_mass == 1
    assert len({codec.decode(stream, span=R(1), tracks=(0,)) for stream in streams}) == len(streams)


def test_every_small_exact_event_is_allowed_after_a_fixed_predecessor():
    """Independent rational comparisons cover duration and pitch tie breaks."""
    codec = EventCodec(CodecProfile(max_events=2, max_denominator=3, max_numerator=2))
    first = Event(R(1, 2), 0, 1, 126, R(2, 3))
    first_tokens = codec.encode((first,), span=R(1), tracks=(0,))[:-1]
    values = {R(0), R(1, 3), R(1, 2), R(2, 3), R(1), R(2)}
    # A bounded edge-focused universe: zero, thirds, halves, tails, and the
    # MIDI boundary, all four kinds and a later track. Full pitch coverage is
    # already exhaustive above.
    for onset in (R(0), R(1, 3), R(1, 2), R(2, 3)):
        for track in (0, 1):
            for kind in range(4):
                for pitch in (0, 125, 126, 127):
                    for duration in values:
                        if duration == R(0) and kind % 2 == 0:
                            continue
                        candidate = Event(onset, track, kind, pitch, duration)
                        single = codec.encode((candidate,), span=R(1), tracks=(0, 1))
                        stream = first_tokens + single
                        if candidate.key >= first.key:
                            assert codec.decode(stream, span=R(1), tracks=(0, 1)) == (first, candidate)
                        else:
                            with pytest.raises(ValueError):
                                codec.decode(stream, span=R(1), tracks=(0, 1))


def test_integer_prefix_feasibility_matches_brute_force_reduced_fractions():
    from math import gcd
    from music_critic.pll.codec import BYTE_BASE, LEN_BASE

    codec = EventCodec(CodecProfile(max_denominator=17, max_numerator=300))
    # After NOTE and a denominator, exact span excludes all n/d >= 2/7.
    for denominator in range(1, 18):
        values = [n for n in range(301) if gcd(n, denominator) == 1 and n * 7 < 2 * denominator]
        den_prefix = (NOTE, LEN_BASE, BYTE_BASE + denominator)
        if not values:
            with pytest.raises(ValueError):
                codec.allowed(den_prefix, span=R(2, 7), tracks=(0,))
            continue
        assert codec.allowed(den_prefix, span=R(2, 7), tracks=(0,)) == (LEN_BASE,)
        assert codec.allowed((*den_prefix, LEN_BASE), span=R(2, 7), tracks=(0,)) == tuple(BYTE_BASE + n for n in values)


def test_full_uint64_numerator_uses_exact_large_integer_arithmetic():
    maximum = 2**64 - 1
    codec = EventCodec(CodecProfile(max_denominator=2))
    event = Event(R(maximum, 2), 15, 0, 127, R(maximum, 2))
    tokens = codec.encode((event,), span=R(maximum + 1), tracks=(15,))
    assert codec.decode(tokens, span=R(maximum + 1), tracks=(15,)) == (event,)
    trace = codec.trace(tokens, span=R(maximum + 1), tracks=(15,))
    assert sum(step.field == 'onset_num_byte' for step in trace) == 8
    assert sum(step.field == 'duration_num_byte' for step in trace) == 8


def test_tiny_query_still_has_zero_onset_and_no_invalid_denominator():
    from music_critic.pll.codec import BYTE_BASE, LEN_BASE
    codec = EventCodec(CodecProfile(max_denominator=65535))
    # No positive rational in this profile fits, but canonical zero does.
    span = R(1, 10**30)
    assert codec.allowed((NOTE,), span=span, tracks=(0,)) == (LEN_BASE,)
    assert codec.allowed((NOTE, LEN_BASE), span=span, tracks=(0,)) == (BYTE_BASE + 1,)
    event = Event(R(0), 0, 0, 60, R(1))
    tokens = codec.encode((event,), span=span, tracks=(0,))
    assert codec.decode(tokens, span=span, tracks=(0,)) == (event,)
