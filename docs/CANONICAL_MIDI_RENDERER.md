# Canonical MIDI Renderer

Status: **Phase 2B.2 ready for review**.

## Scope and trust boundary

The production path is:

```text
raw source -> adapter -> validated CanonicalPiece -> MIDI exporter -> format-1 SMF
```

The exported MIDI proves that a canonical piece can be rendered consistently.
It does not independently establish that a source adapter interpreted its raw
input correctly. The separate audit script reads simplified HookTheory evidence
and never imports or calls the production HookTheory adapter.

The renderer is diagnostic infrastructure. It is not used by graph building,
training, inference, or any model input path.

## Public API

`music_critic.exporters` exports:

```python
MidiRenderConfig
MidiRenderReport
MidiRenderError
piece_to_midi_bytes(piece, *, config=...) -> tuple[bytes, MidiRenderReport]
write_piece_midi(piece, path, *, config=...) -> MidiRenderReport
```

`MidiRenderConfig.require_exact_timing` defaults to `True`. The default render
includes the canonical-beat click track and target markers. Canonical velocity,
program, and channel values are preserved when present; otherwise melody
defaults are velocity 96, program 0, and channel 0. Percussion source notes are
not copied into the melody track. Clicks use channel index 9, MIDI notes 37 and
31, and velocity 80.

## Exact timing and PPQ

The exporter validates the piece before rendering and collects denominators
from piece duration, note boundaries, tempo and meter onsets, bar boundaries,
beat boundaries, click ends, and enabled marker positions. It uses their least
common multiple when that value is no greater than the SMF PPQ ceiling 32767.

An explicit PPQ must represent every rendered time exactly while
`require_exact_timing=True`. If the automatic LCM exceeds 32767, exact mode
raises `MidiRenderError`. With exactness disabled, the exporter uses the
configured fallback (960 by default), rounds non-negative ticks half-up, and
reports the exact maximum error in qn. It never silently quantizes. Positive
notes that collapse to zero ticks are rejected.

## MIDI structure

- Track 0 is the conductor track: name, every canonical time signature, every
  canonical microseconds-per-quarter event, optional key/chord markers, and an
  end-of-track event at canonical duration.
- Subsequent non-percussion canonical tracks carry program and note events.
  Note-offs precede new note-ons at a shared boundary.
- The optional final `Canonical Click` track contains exactly one bounded
  percussion click per `CanonicalBeat`.
- Key and chord targets become marker text only. The renderer does not create
  chord notes or interpret applied, alternate, or pedal harmony.

## HookTheory commands

One clip:

```bash
python scripts/render_hooktheory_midi.py \
  --raw-path data/HookTheory/Hooktheory_Raw.json/4_merged.json \
  --clip-id CLIP_ID \
  --output-dir artifacts/hooktheory_midi
```

Golden manifest, including the one explicitly bounded quantized case:

```bash
python scripts/render_hooktheory_midi.py \
  --manifest tests/fixtures/hooktheory/golden_manifest.json \
  --structure-root data/HookTheory \
  --allow-quantization \
  --output-dir artifacts/hooktheory_midi
```

`--deterministic-sample` chooses stable lexicographic representatives for
major, minor, every observed scale mode, 6/8, 9/8, 12/8, multiple meters,
multiple tempos, fractional timing, and a shared `ori_uid`. `--hide-targets`
provides the target-hiding rendering condition. Generated MIDI and listening
artifacts remain outside version control.

The renderer writes exact canonical JSON, MIDI, and a per-clip report, plus
batch and listening manifests. Missing payloads are reported as expected skips.

## Independent comparison

`scripts/compare_hooktheory_midi_rendering.py` reads `Hooktheory.json` directly.
It reconstructs reference pitch as `60 + 12 * octave + pitch_class`, maps
simplified source-beat boundaries through simplified meter regions, and compares
them with MIDI events. Strict symbolic equality is reported separately from
acceptance within a renderer-declared quantization bound.

Audio-aligned metrics use monotonic refined alignment when available, otherwise
user alignment, normalized to local beat zero. Audio comparison is restricted
to constant-meter, constant-tempo, non-swing eligible clips and reports timing
deviation rather than requiring performance alignment to equal symbolic tempo.
Per-clip counts, pairing, pitch/timing errors, tempo/meter status, quality flags,
and bounded examples accompany aggregate median, p90, and p95 metrics.

## Current golden result

The 2026-07-20 opt-in run selected all 19 golden cases: 18 rendered and the
required missing payload was skipped. Seventeen renders were strictly exact.
`ANmplRlZmyM` requires exact PPQ 500000000000000, so its opt-in PPQ-960 render
reported maximum error `29/1500000000000000` qn. The independent comparison
accepted all 18: 17 strictly exact, one within its declared bound, zero pitch
mismatches, zero note-count mismatches, and zero meter disagreements.
