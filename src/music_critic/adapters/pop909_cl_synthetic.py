"""Tiny generated MIDI fixture for the frozen POP909-CL evaluation smoke.

MIDI construction stays at the adapter boundary. This helper performs no file
access and does not discover or read a corpus.
"""

from __future__ import annotations

import mido


def build_pop909_cl_smoke_midi() -> mido.MidiFile:
    """Return the fixed channel-0 score and target-only channel-1 fixture."""
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=1920),
    ]))
    for channel, pitches in ((0, (48, 60, 64, 67, 72)), (1, (60, 64, 67))):
        track = mido.MidiTrack([mido.Message("program_change", channel=channel, program=0, time=0)])
        for index, pitch in enumerate(pitches):
            track.append(mido.Message("note_on", channel=channel, note=pitch, velocity=80,
                                      time=160 if channel == 1 and index == 0 else 0))
        for index, pitch in enumerate(pitches):
            track.append(mido.Message("note_off", channel=channel, note=pitch, velocity=0,
                                      time=(480 if channel == 1 else 1920) if index == 0 else 0))
        midi.tracks.append(track)
    return midi
