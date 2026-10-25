#!/usr/bin/env python3
"""Render diagnostic note events, not the original performance or arrangement.

This optional, output-only utility uses NumPy from the caller's environment. It
does not load MIDI, project records, targets, a model, a SoundFont, or an audio
device. Input is a JSON list of note events or {"events": [...],
"duration_seconds": 30}. Every event has pitch (MIDI), onset_seconds,
duration_seconds, and optional velocity (default 80). Seconds are an audio-export
boundary only; they must not replace exact source timing in canonical data.

Use ONE externally selected gain for every variant in a listening comparison.
There is deliberately no per-file loudness or peak normalization. The bounded
additive keyboard-like timbre is a listening diagnostic, not an acoustic piano
model, original timbre, expert quality judgment, or training target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import wave

import numpy as np


SYNTH_VERSION = "DiagnosticAdditiveKeyboard@1.0.0"
HARMONICS = ((1, 1.0), (2, 0.28), (3, 0.16), (4, 0.08), (5, 0.035))
RELEASE_SECONDS = 0.08
ATTACK_SECONDS = 0.003


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and sort without mutating caller-owned notes."""
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be a list or tuple")
    result = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"event {index} must be an object")
        pitch = event.get("pitch")
        velocity = event.get("velocity", 80)
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127:
            raise ValueError(f"event {index}: pitch must be a MIDI integer 0..127")
        if isinstance(velocity, bool) or not isinstance(velocity, int) or not 0 <= velocity <= 127:
            raise ValueError(f"event {index}: velocity must be an integer 0..127")
        onset = _number(event.get("onset_seconds"), "onset_seconds")
        duration = _number(event.get("duration_seconds"), "duration_seconds")
        if onset < 0 or duration < 0:
            raise ValueError(f"event {index}: onset and duration must be nonnegative")
        result.append(dict(pitch=pitch, velocity=velocity,
                           onset_seconds=onset, duration_seconds=duration))
    return sorted(result, key=lambda e: (e["onset_seconds"], e["pitch"],
                                        e["duration_seconds"], e["velocity"]))


def synthesize_preview(
    events: Sequence[Mapping[str, Any]],
    *,
    duration_seconds: float = 30.0,
    sample_rate: int = 22050,
    gain: float = 0.04,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mono float64 samples and pre-quantization diagnostics.

    Notes have independent phase/envelopes, including overlapping equal pitches.
    A 3 ms attack and 80 ms release prevent note boundary clicks. Harmonics above
    45% of the sample rate are omitted. A fundamental above that limit is an
    error, not a silently absent or aliased note. Clip-end fade is 20 ms.
    """
    duration = _number(duration_seconds, "duration_seconds")
    gain = _number(gain, "gain")
    if not 0 < duration <= 300:
        raise ValueError("duration_seconds must be in (0, 300]")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise ValueError("sample_rate must be an integer in 8000..96000")
    if not 0 < gain <= 1:
        raise ValueError("gain must be in (0, 1]")
    notes = validate_events(events)
    frame_count = int(math.floor(duration * sample_rate + 0.5))
    samples = np.zeros(frame_count, dtype=np.float64)
    rendered = 0
    for event in notes:
        onset = event["onset_seconds"]
        note_duration = event["duration_seconds"]
        if note_duration == 0 or event["velocity"] == 0 or onset >= duration:
            continue
        frequency = 440.0 * 2.0 ** ((event["pitch"] - 69) / 12.0)
        if frequency >= 0.45 * sample_rate:
            raise ValueError("sample_rate too low for note fundamental; use a higher rate")
        start = int(math.floor(onset * sample_rate + 0.5))
        count = min(frame_count - start,
                    int(math.ceil((note_duration + RELEASE_SECONDS) * sample_rate)))
        if count <= 0:
            continue
        time = np.arange(count, dtype=np.float64) / sample_rate
        voice = np.zeros(count, dtype=np.float64)
        for harmonic, amplitude in HARMONICS:
            if frequency * harmonic >= 0.45 * sample_rate:
                continue
            decay = np.exp(-time * (0.65 + 0.30 * (harmonic - 1)))
            voice += amplitude * decay * np.sin(2.0 * np.pi * frequency * harmonic * time)
        envelope = np.minimum(time / ATTACK_SECONDS, 1.0)
        envelope *= np.clip((note_duration + RELEASE_SECONDS - time) / RELEASE_SECONDS, 0.0, 1.0)
        samples[start:start + count] += gain * (event["velocity"] / 127.0) * envelope * voice
        rendered += 1
    fade_frames = min(frame_count, int(round(0.02 * sample_rate)))
    if fade_frames:
        samples[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames)
    peak = float(np.max(np.abs(samples))) if frame_count else 0.0
    clipped = int(np.count_nonzero(np.abs(samples) > 1.0))
    report = dict(synth_version=SYNTH_VERSION,
                  label="Diagnostic synthetic keyboard; not the source arrangement or performance",
                  sample_rate=sample_rate, channels=1, sample_width_bytes=2,
                  duration_seconds=frame_count / sample_rate, frame_count=frame_count,
                  input_event_count=len(notes), rendered_event_count=rendered,
                  gain=gain, peak_amplitude_before_clipping=peak,
                  clipping_sample_count=clipped, per_file_normalization=False)
    return samples, report


def render_preview(
    events: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    duration_seconds: float = 30.0,
    sample_rate: int = 22050,
    gain: float = 0.04,
) -> dict[str, Any]:
    """Write little-endian PCM16 WAV. Refuse clipping and existing outputs."""
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing audio: {path}")
    samples, report = synthesize_preview(events, duration_seconds=duration_seconds,
                                        sample_rate=sample_rate, gain=gain)
    if report["clipping_sample_count"]:
        raise ValueError(f"Clipping would affect {report['clipping_sample_count']} samples "
                         f"(peak={report['peak_amplitude_before_clipping']:.6f}); "
                         "lower the common gain for ALL comparison variants")
    pcm = np.rint(samples * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier renders even under concurrent callers.
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm)
    report["wav_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["wav_byte_size"] = path.stat().st_size
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--gain", type=float, default=0.04)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if isinstance(payload, dict):
        events = payload["events"]
        duration = payload.get("duration_seconds", 30.0)
    else:
        events, duration = payload, 30.0
    if args.duration_seconds is not None:
        duration = args.duration_seconds
    report = render_preview(events, args.output, duration_seconds=duration,
                            sample_rate=args.sample_rate, gain=args.gain)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
