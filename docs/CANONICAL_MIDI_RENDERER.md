# Canonical MIDI Renderer

Phase status: **Accepted and Completed**.

Accepted implementation:
`97eda0d8fdb7c884bd3d22f0027fb872b2034399`.

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

There is deliberately no MIDI channel allocator. Canonical channel/program
values are emitted without moving notes or resolving conflicts between tracks;
channel 9 remains reserved for percussion/click use. Simultaneous different
programs on one effective channel are reported as a timbre ambiguity, but the
piece is still rendered and no timbre guarantee is made for that interval.

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

Permanent review-package form (the production raw and simplified paths are the
defaults):

```bash
python scripts/render_hooktheory_midi.py \
  --manifest tests/fixtures/hooktheory/golden_manifest.json \
  --output-dir artifacts/hooktheory_midi_review \
  --allow-quantization
```

Full-corpus ambiguity audit without corpus MIDI rendering:

```bash
python scripts/audit_hooktheory_midi_ambiguities.py \
  --output artifacts/hooktheory_midi_review/corpus-ambiguity-report.json
```

`--deterministic-sample` chooses stable lexicographic representatives for
major, minor, every observed scale mode, 6/8, 9/8, 12/8, multiple meters,
multiple tempos, fractional timing, and a shared `ori_uid`. `--hide-targets`
provides the target-hiding rendering condition. Generated MIDI and listening
artifacts remain outside version control.

The renderer writes exact canonical JSON, MIDI, and a per-clip report, plus
batch and listening manifests. The same command also writes
`comparison-report.json`, `audio-disagreement-clips.json`, and
`ambiguity-report.json`; each listening entry includes meter/tempo regions,
mode, duration, note count, PPQ/exactness/error, ambiguity counts, audio status,
onset p95, diagnostic focus, and artifact paths. Missing payloads are reported
as expected skips. `artifacts/` and generated MIDI remain ignored by Git.

## Independent comparison

`scripts/compare_hooktheory_midi_rendering.py` reads `Hooktheory.json` directly.
It reconstructs reference pitch as `60 + 12 * octave + pitch_class`, maps
simplified source-beat boundaries through simplified meter regions, and compares
them with MIDI events. Strict symbolic equality is reported separately from
acceptance. The audit derives two bounds independently from the parsed MIDI:
each onset/offset, tempo onset, meter onset, or terminal piece-duration endpoint
may differ by at most `1 / (2 * PPQ)` qn, while note duration (the difference
of two independently rounded endpoints) may differ by at most `1 / PPQ` qn. It
never uses
`MidiRenderReport.maximum_quantization_error_qn` as the acceptance tolerance.
It directly measures onset, offset, and duration errors as exact `Fraction`
values. Exact renders must observe zero note endpoint/duration, tempo/meter
onset, and piece-duration error; PPQ-derived tolerance never admits nonzero
exact-mode error. Quantized renders use the separate endpoint and duration
bounds with zero technical slack in the current all-rational implementation.
The exporter-reported maximum concerns one converted time point and is only
cross-checked: it must not exceed the PPQ bound or under-report observed
endpoint error. Any independent or report cross-check violation makes the
comparison command exit nonzero.

Simplified-source meter identity has two explicit results. `meter_regions_exact`
requires equal event count, onset, numerator, and denominator.
`meter_regions_accepted` still requires equal event count and exact
numerator/denominator, but accepts each onset within the active endpoint bound:
zero in exact mode or `1 / (2 * PPQ)` in quantized mode. Symbolic acceptance,
`meter_mismatch_clips`, and CLI success use the accepted result; exact and
quantization-accepted meter counts remain separate diagnostics.

Audio-aligned metrics use monotonic refined alignment when available, otherwise
user alignment, normalized to local beat zero. Audio comparison is restricted
to constant-meter, constant-tempo, non-swing eligible clips and reports timing
deviation rather than requiring performance alignment to equal symbolic tempo.
Per-clip counts, pairing, pitch/timing errors, tempo/meter status, quality flags,
and bounded examples accompany aggregate median, p90, and p95 metrics.

Audio results are explicitly partitioned into symbolic-accepted,
audio-agreeing, audio-disagreeing, and audio-ineligible clips. Disagreement
details are also written separately with alignment source, aligned-note count,
onset and duration median/p90/p95, meter, tempo, and quality flags. Audio
disagreement is evidence about source alignment/tempo assumptions, not an
exporter error.

## Guarantee boundary and ambiguity audit

For the HookTheory golden comparison, the semantic guarantee covers melody
pitch, onset, duration, canonical tempo, canonical meter, and piece duration.
For a generic `CanonicalPiece`, exact representable timing, pitch, tempo, and
meter are rendered faithfully, but SMF does not promise full
`CanonicalPiece` identity. In particular, same-pitch overlapping notes on one
canonical track/effective channel can be ambiguous under MIDI note-off pairing;
program conflicts can make timbre ambiguous; and provenance, targets,
annotations, and otherwise unrepresentable data do not round-trip as canonical
identity.

`scripts/audit_hooktheory_midi_ambiguities.py` independently scans canonical
notes without changing exporter behavior. It classifies strict interval
overlaps for the same canonical track, effective channel, and pitch (including
nested pairs), and simultaneous different effective programs on the same MIDI
channel. Adjacent notes, different tracks, and different channels are not
same-pitch overlap findings. The exporter continues to render every finding;
it does not reject, shift channels, or rewrite programs.

The 2026-07-20 streaming corpus audit covered all 26,175 usable clips and
1,228,022 canonical notes. It found 1,802 same-pitch overlap pairs in 102 clips,
including 1,627 nested pairs. It found zero channel/program conflict clips and
zero conflict pairs. Only bounded examples are retained; no corpus-wide MIDI
batch is created.

## Current golden result

The 2026-07-20 opt-in run selected all 19 golden cases: 18 rendered and the
required missing payload was skipped. Seventeen renders were strictly exact.
`ANmplRlZmyM` requires exact PPQ 500000000000000, so its opt-in PPQ-960 render
reported maximum error `29/1500000000000000` qn. The independent comparison
accepted all 18: 17 strictly exact, one within its independent PPQ-derived
bound, zero pitch mismatches, zero note-count mismatches, zero meter
disagreements, and zero independent/report cross-check violations. The review
set contains no same-pitch overlaps or program conflicts. Audio classification
is seven agreeing, nine disagreeing, and two ineligible clips; the nine
disagreements are retained in `audio-disagreement-clips.json` and do not fail
rendering.
