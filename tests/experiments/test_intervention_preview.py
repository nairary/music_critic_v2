"""Synthetic-only audio export tests; no corpus access or acoustic-quality claim."""
import importlib.util
from pathlib import Path
import wave

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[2] / "scripts/render_intervention_preview.py"
SPEC = importlib.util.spec_from_file_location("intervention_preview", PATH)
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


def note(**extra):
    return dict(pitch=69, onset_seconds=0.2, duration_seconds=0.3, **extra)


def test_wav_silence_duration_format_and_no_clipping(tmp_path):
    path = tmp_path / "example.wav"
    report = preview.render_preview([note()], path, duration_seconds=1.0)
    with wave.open(str(path), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) == (1, 2, 22050, 22050)
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert not samples[:4410].any()
    assert not samples[int(0.58 * 22050) + 1:].any()
    assert samples[4500:8000].any()
    assert report["clipping_sample_count"] == 0
    assert report["per_file_normalization"] is False
    assert report["peak_amplitude_before_clipping"] > 0


def test_frequency_and_duration_are_audible():
    samples, _ = preview.synthesize_preview([dict(pitch=69, onset_seconds=0, duration_seconds=1)], duration_seconds=1.2)
    segment = samples[:22050]
    peak_bin = int(np.argmax(np.abs(np.fft.rfft(segment))))
    assert abs(peak_bin - 440) <= 1
    assert not samples[24000:].any()


def test_common_gain_preserves_linear_amplitude_and_overlap():
    low, _ = preview.synthesize_preview([note()], duration_seconds=1, gain=0.02)
    high, _ = preview.synthesize_preview([note()], duration_seconds=1, gain=0.04)
    overlap, _ = preview.synthesize_preview([note(), note()], duration_seconds=1, gain=0.02)
    np.testing.assert_array_equal(high, low * 2)
    np.testing.assert_array_equal(overlap, high)


def test_deterministic_wav_and_no_input_mutation(tmp_path):
    notes = [note(), dict(pitch=60, onset_seconds=0.1, duration_seconds=0.1)]
    before = [dict(n) for n in notes]
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    preview.render_preview(notes, first, duration_seconds=1)
    preview.render_preview(notes[::-1], second, duration_seconds=1)
    assert first.read_bytes() == second.read_bytes()
    assert notes == before
    with pytest.raises(FileExistsError):
        preview.render_preview(notes, first, duration_seconds=1)


def test_clipping_is_reported_and_refused_before_output(tmp_path):
    notes = [note(velocity=127)] * 30
    _, report = preview.synthesize_preview(notes, duration_seconds=1, gain=1)
    assert report["clipping_sample_count"] > 0
    assert report["peak_amplitude_before_clipping"] > 1
    path = tmp_path / "clipped.wav"
    with pytest.raises(ValueError, match="ALL comparison variants"):
        preview.render_preview(notes, path, duration_seconds=1, gain=1)
    assert not path.exists()


@pytest.mark.parametrize("change", [dict(pitch=128), dict(pitch=True), dict(velocity=-1), dict(onset_seconds=-1), dict(duration_seconds=-1), dict(duration_seconds=float("nan"))])
def test_invalid_event_rejected(change):
    value = note()
    value.update(change)
    with pytest.raises(ValueError):
        preview.synthesize_preview([value], duration_seconds=1)


def test_grace_zero_velocity_and_outside_clip_are_silent():
    samples, report = preview.synthesize_preview([
        dict(pitch=60, onset_seconds=0, duration_seconds=0),
        dict(pitch=60, onset_seconds=0, duration_seconds=1, velocity=0),
        dict(pitch=60, onset_seconds=2, duration_seconds=1),
    ], duration_seconds=1)
    assert not samples.any()
    assert report["rendered_event_count"] == 0


def test_unrepresentable_fundamental_not_silently_aliased():
    with pytest.raises(ValueError, match="fundamental"):
        preview.synthesize_preview([dict(pitch=127, onset_seconds=0, duration_seconds=1)])
