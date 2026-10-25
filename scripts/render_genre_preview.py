#!/usr/bin/env python3
"""Deterministic GM-family diagnostic audio, not a sampled/original performance.

`program` is a zero-based GM program; `is_percussion` must be an observed MIDI
drum-channel fact, never an inferred genre or track role. Raw symbolic timing
stays exact elsewhere; seconds are only this output boundary. Use the SAME gain
for all variants of a work. No loudness normalization is performed.

The browser helper addGenreNote(samples, sr, event, gain) uses the same synthesis
parameters and noise hash as Python. Call finishGenreSamples(samples, sr) ONCE
after adding every note, for the identical 20ms clip-end fade. Browser playback
must independently reject clipping after summation, as render_preview does.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import wave

import numpy as np


SYNTH_VERSION = "DiagnosticGMFamilies@1.0.0"
FAMILIES = {
    "keys": dict(harmonics=[[1, 1], [2, .28], [3, .16], [4, .08], [5, .035]], attack=.003, release=.08, decay=.65, partial_decay=.30),
    "guitar": dict(harmonics=[[1, .9], [2, .45], [3, .28], [4, .14], [5, .08], [6, .04]], attack=.002, release=.045, decay=1.8, partial_decay=.70),
    "bass": dict(harmonics=[[1, 1.2], [2, .25], [3, .10], [4, .04]], attack=.006, release=.06, decay=.35, partial_decay=.18),
    "organ": dict(harmonics=[[1, .8], [2, .45], [3, .22], [4, .15]], attack=.006, release=.04, decay=.025, partial_decay=.01),
    "pad": dict(harmonics=[[1, .8], [2, .20], [3, .12], [5, .04]], attack=.045, release=.14, decay=.08, partial_decay=.04),
    "wind": dict(harmonics=[[1, .9], [2, .30], [3, .18], [4, .07]], attack=.015, release=.07, decay=.15, partial_decay=.07),
    "synth": dict(harmonics=[[k, .85 / k] for k in range(1, 9)], attack=.008, release=.08, decay=.10, partial_decay=.08),
}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _midi_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 127:
        raise ValueError(f"{name} must be an integer 0..127")
    return value


def family_for_program(program: int) -> str:
    """A timbre approximation only; it does not label a note's musical role."""
    _midi_int(program, "program")
    if program < 16:
        return "keys"
    if program < 24:
        return "organ"
    if program < 32:
        return "guitar"
    if program < 40:
        return "bass"
    if program < 56 or 88 <= program < 96:
        return "pad"
    if program < 80:
        return "wind"
    return "synth"


def validate_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be a list or tuple")
    result = []
    for item in events:
        if not isinstance(item, Mapping):
            raise ValueError("event must be an object")
        event = dict(item)
        for field, default in (("pitch", None), ("velocity", 80), ("program", 0)):
            event[field] = _midi_int(event.get(field, default), field)
        event["is_percussion"] = event.get("is_percussion", False)
        if not isinstance(event["is_percussion"], bool):
            raise ValueError("is_percussion must be a boolean MIDI-channel fact")
        event["id"] = event.get("id", "")
        if not isinstance(event["id"], str):
            raise ValueError("id must be a string")
        for field in ("onset_seconds", "duration_seconds"):
            event[field] = _number(event.get(field), field)
            if event[field] < 0:
                raise ValueError(f"{field} must be nonnegative")
        result.append(event)
    return sorted(result, key=lambda e: (e["onset_seconds"], e["id"], e["pitch"], e["program"], e["is_percussion"], e["duration_seconds"], e["velocity"]))


def _seed(event: Mapping[str, Any]) -> int:
    value = 2166136261
    for byte in event["id"].encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return (value ^ (event["pitch"] * 2654435761)) & 0xFFFFFFFF


def _noise(count: int, seed: int) -> np.ndarray:
    # Uint32 wraps intentionally; the JS implementation uses Math.imul/>>> 0.
    values = np.arange(count, dtype=np.uint32) + np.uint32(seed)
    values = ((values >> 16) ^ values) * np.uint32(0x45D9F3B)
    values = ((values >> 16) ^ values) * np.uint32(0x45D9F3B)
    values = (values >> 16) ^ values
    return values.astype(np.float64) / 2147483648.0 - 1.0


def _drum_kind(pitch: int) -> tuple[str, float]:
    if pitch in (35, 36):
        return "kick", .45
    if pitch in (38, 40):
        return "snare", .24
    if pitch in (42, 44, 46):
        return "hat", .45 if pitch == 46 else (.15 if pitch == 44 else .11)
    if pitch in (41, 43, 45, 47, 48, 50):
        return "tom", .35
    if pitch in (49, 51, 52, 53, 55, 57, 59):
        return "cymbal", .75
    return "percussion", .18


def _drum_voice(time: np.ndarray, event: Mapping[str, Any], sample_rate: int) -> np.ndarray:
    kind, span = _drum_kind(event["pitch"])
    noise = _noise(len(time) + 1, _seed(event))
    high = (noise[1:] - noise[:-1]) * .5
    attack = np.minimum(time / .001, 1.0)
    tail = np.clip((span - time) / .02, 0, 1)
    if kind == "kick":
        phase = 2 * np.pi * (48 * time + 105 / 35 * (1 - np.exp(-35 * time)))
        signal = 1.35 * np.sin(phase) * np.exp(-10 * time) + .15 * high * np.exp(-90 * time)
    elif kind == "snare":
        signal = .9 * high * np.exp(-18 * time) + .35 * np.sin(2 * np.pi * 180 * time) * np.exp(-22 * time)
    elif kind == "hat":
        signal = .48 * high * np.exp(-6 / span * time)
    elif kind == "tom":
        frequency = 95 * 2 ** ((event["pitch"] - 41) / 12)
        signal = np.sin(2 * np.pi * frequency * time) * np.exp(-12 * time) + .06 * high * np.exp(-50 * time)
    elif kind == "cymbal":
        frequency = min(3200, .35 * sample_rate)
        signal = (.40 * high + .08 * np.sin(2 * np.pi * frequency * time)) * np.exp(-7 * time)
    else:
        frequency = 300 + event["pitch"] * 9
        signal = (.55 * high + .28 * np.sin(2 * np.pi * frequency * time)) * np.exp(-24 * time)
    return signal * attack * tail


def synthesize_preview(events: Sequence[Mapping[str, Any]], *, duration_seconds: float = 30.0, sample_rate: int = 22050, gain: float = .04, duration_limit_seconds: float = 300.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Allow an explicit longer clip without exceeding the original frame budget."""
    duration = _number(duration_seconds, "duration_seconds")
    duration_limit = _number(duration_limit_seconds, "duration_limit_seconds")
    gain = _number(gain, "gain")
    if not 0 < duration_limit <= 3600:
        raise ValueError("duration_limit_seconds must be in (0, 3600]")
    if not 0 < duration <= duration_limit:
        raise ValueError(f"duration_seconds must be in (0, {duration_limit:g}]")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise ValueError("sample_rate must be an integer in 8000..96000")
    if not 0 < gain <= 1:
        raise ValueError("gain must be in (0, 1]")
    notes = validate_events(events)
    count = math.floor(duration * sample_rate + .5)
    if count > 28_800_000:  # Original maximum: 300 seconds at 96000Hz.
        raise ValueError("Preview exceeds the 28,800,000 frame budget")
    samples = np.zeros(count, dtype=np.float64)
    rendered = Counter()
    for event in notes:
        if event["duration_seconds"] == 0 or event["velocity"] == 0 or event["onset_seconds"] >= duration:
            continue
        percussion = event["is_percussion"]
        family = "drums" if percussion else family_for_program(event["program"])
        parameters = FAMILIES.get(family)
        span = _drum_kind(event["pitch"])[1] if percussion else event["duration_seconds"] + parameters["release"]
        start = math.floor(event["onset_seconds"] * sample_rate + .5)
        length = min(count - start, math.ceil(span * sample_rate))
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float64) / sample_rate
        if percussion:
            voice = _drum_voice(time, event, sample_rate)
        else:
            frequency = 440 * 2 ** ((event["pitch"] - 69) / 12)
            if frequency >= .45 * sample_rate:
                raise ValueError("sample_rate too low for note fundamental; use a higher rate")
            voice = np.zeros(length, dtype=np.float64)
            for harmonic, amplitude in parameters["harmonics"]:
                if frequency * harmonic < .45 * sample_rate:
                    decay = parameters["decay"] + parameters["partial_decay"] * (harmonic - 1)
                    voice += amplitude * np.exp(-decay * time) * np.sin(2 * np.pi * frequency * harmonic * time)
            voice *= np.minimum(time / parameters["attack"], 1) * np.clip((span - time) / parameters["release"], 0, 1)
        samples[start:start + length] += voice * gain * event["velocity"] / 127
        rendered[family] += 1
    fade = min(count, math.floor(.02 * sample_rate + .5))
    if fade:
        samples[-fade:] *= np.linspace(1, 0, fade)
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite rendered samples")
    report = dict(synth_version=SYNTH_VERSION, label="Diagnostic synthetic GM families and drums; not original timbres or performance", sample_rate=sample_rate, channels=1, sample_width_bytes=2, duration_seconds=count / sample_rate, frame_count=count, input_event_count=len(notes), rendered_event_count=sum(rendered.values()), rendered_family_counts=dict(sorted(rendered.items())), gain=gain, peak_amplitude_before_clipping=float(np.max(np.abs(samples))), clipping_sample_count=int(np.count_nonzero(np.abs(samples) > 1)), per_file_normalization=False, percussion_envelope="deterministic one-shot; positive MIDI duration triggers natural drum decay")
    return samples, report


def render_preview(events: Sequence[Mapping[str, Any]], output_path: str | Path, *, duration_seconds: float = 30.0, sample_rate: int = 22050, gain: float = .04, duration_limit_seconds: float = 300.0) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing audio: {path}")
    samples, report = synthesize_preview(events, duration_seconds=duration_seconds, sample_rate=sample_rate, gain=gain, duration_limit_seconds=duration_limit_seconds)
    if report["clipping_sample_count"]:
        raise ValueError("Clipping; lower the common gain for ALL comparison variants")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    report.update(wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), wav_byte_size=path.stat().st_size)
    return report


def browser_synth_js() -> str:
    """Standalone JS declarations; add notes, finish once, then check peak."""
    return "const GENRE_SYNTH_FAMILIES=" + json.dumps(FAMILIES, separators=(",", ":")) + ";\n" + r'''
function genreFamily(program) {
  if (program < 16) return "keys";
  if (program < 24) return "organ";
  if (program < 32) return "guitar";
  if (program < 40) return "bass";
  if (program < 56 || (program >= 88 && program < 96)) return "pad";
  if (program < 80) return "wind";
  return "synth";
}
function genreDrum(p) {
  if ([35,36].includes(p)) return ["kick",.45];
  if ([38,40].includes(p)) return ["snare",.24];
  if ([42,44,46].includes(p)) return ["hat",p===46?.45:p===44?.15:.11];
  if ([41,43,45,47,48,50].includes(p)) return ["tom",.35];
  if ([49,51,52,53,55,57,59].includes(p)) return ["cymbal",.75];
  return ["percussion",.18];
}
function genreNoise(seed, i) {
  let x=(seed+i)>>>0;
  x=Math.imul((x>>>16)^x,0x45d9f3b)>>>0;
  x=Math.imul((x>>>16)^x,0x45d9f3b)>>>0;
  return (((x>>>16)^x)>>>0)/2147483648-1;
}
function addGenreNote(samples, sr, event, gain) {
  const p=event.pitch, velocity=event.velocity??80, program=event.program??0;
  const onset=event.onset_seconds, duration=event.duration_seconds;
  const percussion=event.is_percussion??false, id=event.id??"";
  if (![p,velocity,program].every(x=>Number.isInteger(x)&&x>=0&&x<=127) ||
      typeof percussion!=="boolean" || typeof id!=="string" ||
      ![onset,duration].every(x=>typeof x==="number"&&Number.isFinite(x)&&x>=0) ||
      !Number.isInteger(sr)||sr<8000||sr>96000 || typeof gain!=="number"||!Number.isFinite(gain)||gain<=0||gain>1)
    throw new Error("Invalid genre preview event or rendering parameters");
  if (!duration || !velocity || onset>=samples.length/sr) return false;
  const pars=GENRE_SYNTH_FAMILIES[genreFamily(program)], drum=genreDrum(p);
  const span=percussion?drum[1]:duration+pars.release;
  const start=Math.floor(onset*sr+.5), count=Math.min(samples.length-start,Math.ceil(span*sr));
  if (count<=0) return false;
  const freq=440*Math.pow(2,(p-69)/12);
  if (!percussion && freq>=.45*sr) throw new Error("sample_rate too low for note fundamental");
  let seed=2166136261;
  for (const byte of new TextEncoder().encode(id)) seed=Math.imul(seed^byte,16777619)>>>0;
  seed=(seed^Math.imul(p,2654435761))>>>0;
  for(let i=0;i<count;i++) {
    const t=i/sr;
    let value=0;
    if(percussion) {
      const high=(genreNoise(seed,i+1)-genreNoise(seed,i))*.5;
      if(drum[0]==="kick") value=1.35*Math.sin(2*Math.PI*(48*t+105/35*(1-Math.exp(-35*t))))*Math.exp(-10*t)+.15*high*Math.exp(-90*t);
      else if(drum[0]==="snare") value=.9*high*Math.exp(-18*t)+.35*Math.sin(2*Math.PI*180*t)*Math.exp(-22*t);
      else if(drum[0]==="hat") value=.48*high*Math.exp(-6/span*t);
      else if(drum[0]==="tom") value=Math.sin(2*Math.PI*(95*Math.pow(2,(p-41)/12))*t)*Math.exp(-12*t)+.06*high*Math.exp(-50*t);
      else if(drum[0]==="cymbal") value=(.40*high+.08*Math.sin(2*Math.PI*Math.min(3200,.35*sr)*t))*Math.exp(-7*t);
      else value=(.55*high+.28*Math.sin(2*Math.PI*(300+p*9)*t))*Math.exp(-24*t);
      value*=Math.min(t/.001,1)*Math.max(0,Math.min(1,(span-t)/.02));
    } else {
      for(const [harmonic,amplitude] of pars.harmonics)
        if(freq*harmonic<.45*sr) value+=amplitude*Math.exp(-(pars.decay+pars.partial_decay*(harmonic-1))*t)*Math.sin(2*Math.PI*freq*harmonic*t);
      value*=Math.min(t/pars.attack,1)*Math.max(0,Math.min(1,(span-t)/pars.release));
    }
    samples[start+i]+=value*gain*velocity/127;
  }
  return true;
}
function finishGenreSamples(samples,sr) {
  const fade=Math.min(samples.length,Math.floor(.02*sr+.5));
  for(let i=0;i<fade;i++) samples[samples.length-fade+i]*=fade===1?1:1-i/(fade-1);
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gain", type=float, default=.04)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    events = payload["events"] if isinstance(payload, dict) else payload
    duration = args.duration_seconds if args.duration_seconds is not None else payload.get("duration_seconds", 30) if isinstance(payload, dict) else 30
    print(json.dumps(render_preview(events, args.output, gain=args.gain, sample_rate=args.sample_rate, duration_seconds=duration), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
