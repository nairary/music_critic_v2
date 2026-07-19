"""Dataset adapters that convert source formats into canonical pieces."""

from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
    load_hooktheory_piece,
)
from music_critic.adapters.midi import (
    MidiAdapterConfig,
    MidiAdapterError,
    load_midi_piece,
)

__all__ = [
    "HookTheoryAdapterConfig",
    "HookTheoryAdapterError",
    "MidiAdapterConfig",
    "MidiAdapterError",
    "convert_hooktheory_record",
    "load_hooktheory_piece",
    "load_midi_piece",
]
