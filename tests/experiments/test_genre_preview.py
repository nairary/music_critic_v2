"""Synthetic-only diagnostic audio tests: no corpus or musical-quality claims."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("genre_preview", ROOT / "scripts/render_genre_preview.py")
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


def note(**kwargs):
    return dict(id="synthetic-note", pitch=60, onset_seconds=.05, duration_seconds=.25, velocity=100, program=0, is_percussion=False, **kwargs)


def changed(**kwargs):
    result = note()
    result.update(kwargs)
    return result


def test_gm_families_produce_distinct_signals_without_inferred_roles():
    voices = []
    for program, family in ((0, "keys"), (17, "organ"), (25, "guitar"), (33, "bass"), (49, "pad"), (65, "wind"), (81, "synth")):
        samples, report = preview.synthesize_preview([changed(program=program)], duration_seconds=.6)
        assert report["rendered_family_counts"] == {family: 1}
        assert samples.any()
        assert np.isfinite(samples).all()
        voices.append(samples)
    assert len({v.tobytes() for v in voices}) == 7
    assert preview.family_for_program(90) == "pad"


def test_kick_snare_hihat_tom_cymbal_are_distinct_and_not_pitched_keyboard():
    voices = []
    for pitch in (36, 38, 42, 46, 45, 49, 75):
        drum, report = preview.synthesize_preview([changed(pitch=pitch, is_percussion=True)], duration_seconds=.9)
        piano, _ = preview.synthesize_preview([changed(pitch=pitch)], duration_seconds=.9)
        assert report["rendered_family_counts"] == {"drums": 1}
        assert not np.array_equal(drum, piano)
        voices.append(drum)
    assert len({v.tobytes() for v in voices}) == 7


def test_one_shot_duration_is_explicit_and_note_off_does_not_choke_snare():
    short, report = preview.synthesize_preview([changed(pitch=38, is_percussion=True, duration_seconds=.001)], duration_seconds=.5)
    long, _ = preview.synthesize_preview([changed(pitch=38, is_percussion=True, duration_seconds=2)], duration_seconds=.5)
    np.testing.assert_array_equal(short, long)
    assert "one-shot" in report["percussion_envelope"]
    assert short[int(.10 * 22050):int(.20 * 22050)].any()
    assert not short[int(.30 * 22050):].any()


def test_timing_shift_moves_percussion_signal_without_changing_noise():
    left, _ = preview.synthesize_preview([changed(pitch=42, is_percussion=True, onset_seconds=.05)], duration_seconds=.6)
    right, _ = preview.synthesize_preview([changed(pitch=42, is_percussion=True, onset_seconds=.15)], duration_seconds=.6)
    np.testing.assert_array_equal(left[:5000], right[2205:7205])
    assert not left[:1103].any()


def test_velocity_gain_and_overlap_are_linear_without_normalization():
    event = changed(pitch=36, is_percussion=True, velocity=40)
    base, _ = preview.synthesize_preview([event], duration_seconds=1, gain=.01)
    velocity, _ = preview.synthesize_preview([dict(event, velocity=80)], duration_seconds=1, gain=.01)
    gain, report = preview.synthesize_preview([event], duration_seconds=1, gain=.02)
    overlap, _ = preview.synthesize_preview([event, event], duration_seconds=1, gain=.01)
    np.testing.assert_array_equal(velocity, base * 2)
    np.testing.assert_array_equal(gain, base * 2)
    np.testing.assert_array_equal(overlap, base * 2)
    assert report["per_file_normalization"] is False


def test_deterministic_wav_order_and_input_preservation(tmp_path):
    events = [note(), changed(id="удар-二", pitch=38, is_percussion=True)]
    before = copy.deepcopy(events)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    report = preview.render_preview(events, a, duration_seconds=.75)
    preview.render_preview(events[::-1], b, duration_seconds=.75)
    assert a.read_bytes() == b.read_bytes()
    assert events == before
    assert report["wav_byte_size"] == a.stat().st_size
    with wave.open(str(a), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 22050, 16538)
    with pytest.raises(FileExistsError):
        preview.render_preview(events, a, duration_seconds=.75)


def test_explicit_duration_limit_renders_full_long_clip_and_late_note(tmp_path):
    # Same duration as the rejected work, but a synthetic sparse event fixture.
    path = tmp_path / "long.wav"
    report = preview.render_preview(
        [changed(onset_seconds=301.125)], path,
        duration_seconds=454.4, duration_limit_seconds=600, sample_rate=8000,
    )
    assert report["frame_count"] == 3_635_200
    assert report["duration_seconds"] == 454.4
    assert report["rendered_event_count"] == 1
    assert report["clipping_sample_count"] == 0
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnframes()) == (8000, 3_635_200)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert not pcm[:2_409_000].any()
    assert pcm[2_409_000:2_417_000].any()


@pytest.mark.parametrize("limit", [0, -1, 3600.1, float("nan"), float("inf"), True, "600"])
def test_duration_limit_must_be_finite_positive_and_at_most_one_hour(limit):
    with pytest.raises(ValueError, match="duration_limit_seconds"):
        preview.synthesize_preview([], duration_seconds=.01, duration_limit_seconds=limit)


def test_sample_budget_rejects_before_pcm_allocation(monkeypatch):
    allocated = []
    original_zeros = preview.np.zeros

    def allocation_probe(*args, **kwargs):
        allocated.append(args)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(preview.np, "zeros", allocation_probe)
    # Exactly one frame beyond the existing 300s * 96000 frame budget.
    with pytest.raises(ValueError, match="frame budget"):
        preview.synthesize_preview(
            [], duration_seconds=300 + 1 / 96000,
            sample_rate=96000, duration_limit_seconds=600,
        )
    assert not allocated


def test_maximum_duration_and_exact_frame_budget_are_inclusive(monkeypatch):
    class ReachedAllocation(Exception):
        pass

    def allocation_probe(count, **kwargs):
        assert count == 28_800_000
        raise ReachedAllocation

    # Probe the allocation boundary without creating a 230MB synthetic buffer.
    monkeypatch.setattr(preview.np, "zeros", allocation_probe)
    with pytest.raises(ReachedAllocation):
        preview.synthesize_preview([], duration_seconds=3600, sample_rate=8000, duration_limit_seconds=3600)


def test_explicit_duration_limit_can_also_reduce_the_allowed_clip():
    with pytest.raises(ValueError, match="duration_seconds"):
        preview.synthesize_preview([], duration_seconds=1, duration_limit_seconds=.5)


def test_default_duration_limit_still_rejects_long_clips_before_writing(tmp_path):
    path = tmp_path / "long-default.wav"
    with pytest.raises(ValueError, match=r"duration_seconds must be in \(0, 300\]"):
        preview.render_preview([], path, duration_seconds=454.4, sample_rate=8000)
    assert not path.exists()


def test_explicit_duration_limit_preserves_short_clip_samples_report_and_wav(tmp_path):
    events = [note(), changed(id="snare", pitch=38, is_percussion=True)]
    default, report = preview.synthesize_preview(events, duration_seconds=.75)
    extended, extended_report = preview.synthesize_preview(
        events, duration_seconds=.75, duration_limit_seconds=600,
    )
    np.testing.assert_array_equal(default, extended)
    assert report == extended_report
    first, second = tmp_path / "default.wav", tmp_path / "extended.wav"
    preview.render_preview(events, first, duration_seconds=.75)
    preview.render_preview(events, second, duration_seconds=.75, duration_limit_seconds=600)
    assert first.read_bytes() == second.read_bytes()


def test_clipping_rejected_before_wav_creation(tmp_path):
    events = [changed(pitch=36, is_percussion=True, velocity=127)] * 20
    _, report = preview.synthesize_preview(events, duration_seconds=.75, gain=1)
    assert report["clipping_sample_count"] > 0
    path = tmp_path / "reject.wav"
    with pytest.raises(ValueError, match="ALL comparison variants"):
        preview.render_preview(events, path, duration_seconds=.75, gain=1)
    assert not path.exists()


@pytest.mark.parametrize("field,value", [("pitch", 128), ("velocity", True), ("program", -1), ("program", None), ("is_percussion", "false"), ("id", 12), ("onset_seconds", -1), ("duration_seconds", float("nan")), ("duration_seconds", float("inf"))])
def test_bad_events_fail_explicitly(field, value):
    with pytest.raises(ValueError):
        preview.synthesize_preview([changed(**{field: value})], duration_seconds=.5)


def test_silent_events_and_high_pitch_guard():
    events = [changed(duration_seconds=0), changed(velocity=0), changed(onset_seconds=1), changed(duration_seconds=0, is_percussion=True)]
    samples, report = preview.synthesize_preview(events, duration_seconds=.5)
    assert not samples.any()
    assert report["rendered_event_count"] == 0
    with pytest.raises(ValueError, match="fundamental"):
        preview.synthesize_preview([changed(pitch=127)])
    # A MIDI drum number is not a tonal note frequency.
    samples, _ = preview.synthesize_preview([changed(pitch=127, is_percussion=True)], duration_seconds=.5)
    assert samples.any()


def test_note_release_and_clip_end_are_smooth_and_bounded():
    samples, _ = preview.synthesize_preview([changed(onset_seconds=0, duration_seconds=.1)], duration_seconds=.3)
    assert samples[0] == 0
    assert not samples[int(.181 * 22050):].any()
    samples, _ = preview.synthesize_preview([changed(onset_seconds=0, duration_seconds=1)], duration_seconds=.3)
    assert samples[-1] == 0


def node_binary():
    available = shutil.which("node")
    if available:
        return available
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    if bundled.exists():
        return str(bundled)
    pytest.skip("Node runtime absent: cross-language check unavailable")


def test_browser_and_python_render_same_multitimbral_samples(tmp_path):
    # Every tonal and percussion branch, including non-ASCII stable IDs.
    events = [changed(id=f"нота-{i}", program=p, onset_seconds=i * .01) for i, p in enumerate((0, 17, 25, 33, 49, 65, 81, 90))]
    events += [changed(id=f"удар-二-{i}", pitch=p, is_percussion=True, onset_seconds=i * .012) for i, p in enumerate((36, 38, 42, 44, 46, 45, 49, 75))]
    events = preview.validate_events(events)
    samples, _ = preview.synthesize_preview(events, duration_seconds=.9, sample_rate=16000, gain=.015)
    js = preview.browser_synth_js() + "\nconst data=" + json.dumps(events) + ";\n"
    js += "const samples=new Float64Array(14400); for(const e of data)addGenreNote(samples,16000,e,.015); finishGenreSamples(samples,16000); process.stdout.write(Buffer.from(samples.buffer));"
    script = tmp_path / "check.js"
    script.write_text(js)
    result = subprocess.run([node_binary(), str(script)], check=True, capture_output=True)
    browser = np.frombuffer(result.stdout, dtype="<f8")
    np.testing.assert_allclose(browser, samples, rtol=1e-10, atol=1e-12)


def test_browser_noops_and_input_validation(tmp_path):
    js = preview.browser_synth_js() + "\nconst samples=new Float64Array(8000);\n"
    js += "const e={pitch:60,onset_seconds:0,duration_seconds:0}; if(addGenreNote(samples,8000,e,.04)!==false)throw Error('no-op');"
    js += "for(const change of [{program:-1},{is_percussion:'false'},{pitch:128}]) {let rejected=false; try{addGenreNote(samples,8000,{...e,...change},.04);}catch(err){rejected=true;} if(!rejected)throw Error('validation');}"
    script = tmp_path / "invalid.js"
    script.write_text(js)
    subprocess.run([node_binary(), str(script)], check=True, capture_output=True)
