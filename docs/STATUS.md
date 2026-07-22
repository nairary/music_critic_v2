# Music Critic V2 Status

## Current phase

- Date: 2026-07-22
- Completed phase: Phase 1 — canonical data schema and serialization
- Phase 1A: Completed
- Phase 1B.1: Completed
- Phase 1B.2: Completed
- Phase 1B.3: Completed
- Phase 1 merge SHA: `37edf76889730980aa6ce9e9ec981e362c3480a9`
- Phase 2: Accepted and Completed
- Phase 2A.1: Accepted and Completed
- Accepted Phase 2A.1 implementation SHA:
  `32d68e8cb446d9b5dd57bfea1d28b94ccce46274`
- Phase 2B.0: Accepted and Completed
- Accepted Phase 2B.0 implementation SHA:
  `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`
- Phase 2B.1: Accepted and Completed
- Accepted Phase 2B.1 implementation SHA:
  `3898b168063094b87e5ca5d88aae0317c1562c3f`
- Phase 2B.1 closure SHA:
  `6111d3d062e02897e3f8ebdca7e4388f80ef434e`
- Phase 2B.1 merge SHA:
  `b1df77737f641b705e3c48724b2741c7a022a2e4`
- Phase 2B.2: Accepted and Completed
- Phase 2B.2 starting SHA: `3d814a2e2db7434ee6c666619dc287e5eb101101`
- Phase 2B.2 initial implementation SHA:
  `f3799765b74b17cc3a493430dc11f2a64a781b74`
- Accepted Phase 2B.2 implementation HEAD:
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`
- Phase 2B.2 closure SHA:
  `bb94e2972f94a4e092331ebd240781263656dea1`
- Phase 2B.2 merge SHA:
  `1d8a5ecf217ebd466018a1f845eedfab7e1f7828`
- Phase 3A: Completed
- Phase 3A branch: `phase/3a-raw-graph-contract`
- Next phase: Phase 4 — POP909 adapter

## Phase 3A raw graph result

- Public API from `music_critic.graph`: `build_raw_graph`,
  `validate_raw_graph`, `graph_to_dict`, `dumps_graph`, `dump_graph`, and
  `graph_fingerprint`, plus the feature/relation/version registries.
- Every build returns PyG `HeteroData` with mandatory `song`, `track`, `bar`,
  `beat`, `onset`, and `note` stores. Canonical schema `2.0.0`, graph schema
  `1.0.0`, feature registry `1.0.0`, and graph builder `1.0.0` are stored on
  each graph.
- Exact onset-based containment, chronological/reverse relations, and sparse
  note-to-beat sustained activity are deterministic. Beat and onset candidate
  slots are raw unconditional positions for future direct theory heads.
- Separate categorical, continuous, and availability tensors contain only raw
  MIDI-observable or deterministic raw-derived fields. Targets, theory/gold
  annotations, dataset/source grouping, split, source path, provenance,
  confidence, and quality flags are not read for features or topology.
- Simultaneous notes share onset/beat intermediaries; no pairwise simultaneous
  clique is built. Regression tests bound edge growth linearly on dense
  same-onset fixtures.
- PyTorch and PyG are explicit graph runtime dependencies. The canonical data
  layer remains standard-library-only, and optional compiled PyG extensions are
  not required.
- `scripts/benchmark_graph_builder.py` reports per-type node/edge counts,
  contract versions, and min/mean/max construction time for canonical JSON.
- Non-goals remain GNNs, SSL objectives, masking/corruptions, semantic nodes,
  graph caches/collation, models, training, preference, and scoring inference.

## Phase 3A verification

- Focused graph suite: `17 passed`.
- Full default repository suite: `452 passed, 9 skipped`; all skips remain
  explicitly gated local-corpus integrations. PyG emits two upstream
  `torch.jit.script` deprecation warnings; there are no test failures.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed with no output.
- Five-repeat canonical fixture benchmark: 20 nodes and 108 directed edges;
  construction mean `0.003632 s`, minimum `0.003575 s`, and maximum
  `0.003689 s` on the implementation environment.
- Installed verification runtime: Python 3.13, CPU-only PyTorch `2.13.0`, and
  PyG `2.8.0.post1`. Declared compatibility remains bounded by
  `torch>=2.8,<3` and `torch-geometric>=2.7,<3`.

## Phase 2 migration status

- The HookTheory migration contract in `docs/HOOKTHEORY_MIGRATION.md` is
  **Accepted**. Evidence is classified as observed corpus,
  upstream Sheet Sage, V1 compatibility, project decision, or unresolved.
- HookTheory remediation uses piecewise meter-aware qn timing, compound
  felt-pulse tempo, active-scale pitch, and the upstream MIDI-60 anchor with
  provenance method `hooktheory_scale_degree_to_midi_upstream`. MIDI 72 is
  legacy compatibility history only.
- Applied harmony is deferred from the first HookTheory adapter.
- The Phase 2B.1 HookTheory adapter is **Accepted and Completed** at
  `3898b168063094b87e5ca5d88aae0317c1562c3f`.
- Phase 2B.0 is **Accepted and Completed** at implementation SHA
  `9bfcd45d7d3ae7e404a88dc8c0a040aa23c49e7e`.
- Phase 2B.2 is **Accepted and Completed** at implementation HEAD
  `97eda0d8fdb7c884bd3d22f0027fb872b2034399`; the accepted chain includes the
  initial implementation and every review remediation.
- At Phase 2 closure, no graph, dataset, model, SSL, training, preference,
  quality, inference, or GRPO work had started; Phase 3A is now completed as
  recorded above.

## Phase 2B.2 canonical MIDI renderer result

- Public API from `music_critic.exporters`: `MidiRenderConfig`,
  `MidiRenderReport`, `MidiRenderError`, `piece_to_midi_bytes`, and
  `write_piece_midi`. It is generic and imports neither HookTheory, Sheet Sage,
  nor the legacy repository.
- Existing `mido>=1.3,<2` is reused; no dependency changed. The canonical data
  layer remains isolated from `mido` and the exporter.
- Validated canonical qn timing selects the denominator LCM up to PPQ 32767.
  Explicit or fallback quantization is forbidden unless
  `require_exact_timing=False`; enabled quantization uses deterministic half-up
  rounding and reports its exact rational maximum error.
- Format-1 MIDI contains a conductor track, non-percussion canonical melody
  track(s), and an optional final percussion click track. Canonical tempo and
  time-signature values are written directly. Clicks derive from
  `CanonicalBeat`; key/chord targets become optional marker text only. No chord
  notes are synthesized.
- `scripts/render_hooktheory_midi.py` supports one clip, golden manifests,
  target hiding, click/marker toggles, explicit PPQ, explicit quantization, and
  deterministic samples covering every observed mode, 6/8, 9/8, 12/8,
  multiple meters/tempos, fractional timing, and shared `ori_uid`. It writes
  exact canonical JSON, MIDI, per-clip reports, a batch manifest, and a
  listening manifest, plus independent comparison, audio-disagreement, and
  ambiguity reports in one reproducible review package.
- The real golden batch selected 19 cases, rendered all 18 usable cases, and
  reported the required missing payload as an expected skip. Seventeen cases
  are strictly exact. `ANmplRlZmyM` requires PPQ 500000000000000; the explicit
  PPQ-960 fallback reports maximum error `29/1500000000000000` qn.
- The independent simplified-source audit imports no production HookTheory
  adapter. It derives the single-endpoint bound `1/(2*PPQ)` and derived-duration
  bound `1/PPQ` from each parsed MIDI instead of trusting the exporter report as
  a tolerance. Exact mode permits no nonzero note endpoint/duration,
  tempo/meter-onset, or piece-duration error; the reported pointwise maximum is
  only cross-checked against observed endpoints. Across 1,383
  rendered/reference notes it reports 18/18 accepted clips, 17 strictly exact
  clips, one independently quantization-bounded clip, zero pitch mismatches,
  zero note-count mismatches, zero meter disagreements, and zero audit/report
  cross-check violations.
- Simplified meter reporting now separates exact identity from acceptance.
  Exact requires identical count, onset, numerator, and denominator; accepted
  requires identical count/signature and onset within zero in exact mode or the
  half-tick endpoint bound in quantized mode. Aggregate mismatch and CLI exit
  use acceptance while retaining exact and quantization-accepted counts.
- Eligible constant-meter/constant-tempo/non-swing audio comparison covers
  1,236 notes. Onset absolute error is median 0.0328056 s, p90 0.96854 s, p95
  1.667565 s; duration absolute error is median 0.00120975 s, p90 0.013095 s,
  p95 0.04021 s. Nine clips exceed the report's 50 ms onset-p95 diagnostic;
  seven agree, nine disagree, and two are ineligible. Disagreement details are
  a separate artifact and remain alignment/tempo evidence, not exporter errors.
- A streaming ambiguity audit covers all 26,175 usable records and 1,228,022
  notes without corpus-wide MIDI rendering. It finds 1,802 same-pitch overlap
  pairs across 102 clips, including 1,627 nested pairs, and zero simultaneous
  different-program conflicts on one channel. The exporter reports these
  ambiguities without rejecting, shifting, or rewriting notes/channels/programs.
- The guarantee is deliberately split: the HookTheory semantic comparison
  covers pitch, onset, duration, tempo, meter, and piece duration; generic MIDI
  rendering preserves representable pitch/timing/tempo/meter but does not
  promise full canonical identity for overlaps, program conflicts,
  unrepresentable data, provenance, targets, or annotations.
- Generated listening artifacts are outside Git at
  `/tmp/music-critic-v2-phase2b2-remediation/listening-manifest.json`. No generated
  MIDI or canonical batch output is tracked.
- Non-goals remain chord voicing and deferred harmony interpretation, audio or
  SoundFont rendering, graphs, datasets, models, training, inference, and
  Phase 3.

## Phase 2B.2 verification

- Exporter unit tests remain `20 passed` (including all nine observed scale
  families); the production exporter API and event architecture are unchanged.
- Focused independent-comparison tests: `13 passed in 0.07s`; renderer CLI:
  `2 passed in 0.12s`; ambiguity/conflict audit: `2 passed in 0.05s`.
- Opt-in real golden renderer/round-trip/review-package plus full-corpus
  ambiguity integration: `3 passed`; 18 renders/reloads, one
  required missing-payload skip, every required report, and all 26,175 usable
  canonical clips audited without corpus MIDI rendering.
- Full default repository suite: `435 passed, 9 skipped in 1.04s`; every skip
  is an explicitly gated local-corpus integration.
- Full suite with every HookTheory, semantic-crosswalk, renderer, and real-MIDI
  integration enabled: `444 passed in 383.95s`.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed with no output.
- Production dependency/import scan: passed through repository-contract and
  import-isolation tests. `mido` is allowed only in adapters/exporters; the data
  layer imports neither it nor rendering, and production rendering imports no
  HookTheory or legacy module.
- Absolute-path scan found only the pre-existing, deliberate legacy-check and
  legacy-contract references; no new production absolute path was introduced.
- The external legacy snapshot check remains exit 1 under the documented
  ADR-023 resolution-C waiver. Its current 29-entry staged state is unchanged
  by Phase 2B.2 and remains detailed in `docs/LEGACY_DRIFT_REPORT.md`.

## Phase 2B.1 production HookTheory adapter result

- Public API from `music_critic.adapters`: `HookTheoryAdapterConfig`,
  `HookTheoryAdapterError`, `convert_hooktheory_record`, and
  `load_hooktheory_piece`.
- Production input is only
  `data/HookTheory/Hooktheory_Raw.json/4_merged.json`, with optional
  `HookTheoryStructure.<split>.jsonl` group metadata. The adapter does not read
  the simplified crosswalk, HTCanon, Sheet Sage, or the legacy repository.
- The incremental production parser supports complete top-level objects and
  legacy fragments, preserves decimal lexemes, detects duplicate requested
  IDs, and has bounded memory use.
- Melody and chord timing use exact `Fraction(str(value))` arithmetic and a
  piecewise timeline: one qn per `beatUnit=1` raw beat and one-half qn per
  `beatUnit=3` raw beat, including spans crossing changes and `endBeat`.
  Sounding pitch uses active scale steps, true accidentals, and MIDI 60 for
  relative octave zero; rests and malformed/unresolved notes create no note.
- Tempo uses exact quarter-pulse BPM in simple meter and three-eighth
  felt-pulse BPM in compound meter, with final half-up rounding. Bars and
  denominator-unit beats preserve exact meter changes and incomplete
  boundaries without padding duration. Structure metadata must match clip stem
  and split before `ori_uid` may affect grouping.
- Local keys and chord spans are target-alignment annotations only. The 12
  target tasks are melody scale degree; local-key tonic and mode; and chord
  presence, root degree, extent, inversion, adds, omits, alterations,
  suspensions, and borrowed value. Applied, alternate, pedal, and section
  semantics remain deferred.
- `include_targets=False` removes annotations, targets, and annotation-only
  provenance without changing identity, grouping, split, duration, tracks,
  notes, tempo, meter, bars, beats, diagnostics, or their IDs/timing.
- Full-corpus smoke: 26,178 raw records, three missing payloads, 26,175 usable
  records attempted, 26,175 valid pieces, and zero unexpected failures.
  Remediated totals are 1,228,022 notes, 302,619 bars, 1,229,208 beats, 26,315 tempo
  events, 27,171 meter events, 476,347 target-alignment spans, and 314,100
  target arrays. All 32 deterministic spread samples passed serialization and
  target-visible/hidden comparisons. A second full-corpus hidden-target pass
  produced the same raw-content and quality-flag totals with zero annotations,
  zero targets, and zero unexpected failures.
- Old to remediated metric totals: notes 1,228,022 -> 1,228,022 (0);
  bars 304,230 -> 302,619 (-1,611); beats 1,242,480 -> 1,229,208
  (-13,272); tempo events 26,315 -> 26,315 (0); meter events 27,171 ->
  27,171 (0). Visible annotations and target arrays remain 476,347 and
  314,100; hidden mode remains zero for both.
- Quality-flag totals: alternate unresolved 14; applied deferred 19,540;
  borrowed unknown string 1; non-rest root zero 6; invalid chord timing 4;
  default tempo 3; duration extended 23; negative rest
  root anomaly 20; invalid note duration 296; invalid note timing 23;
  structure alignment unresolved 11,515; unmatched structure 14,660; and
  invalid tempo 3.
- All 19 Phase 2B.0 golden cases pass against the raw production source: 18
  usable cases convert and the missing-payload case raises the required adapter
  error.
- Semantic audit: 27,216/27,216 paired meter regions match; 1,211,093 melody
  pairs have zero pitch-class and zero relative-octave mismatches. Candidate
  pitch conversion changes 1,227,982 of 1,228,022 production sounding-note
  pitches. Candidate timing changes 6,443 note and 2,009 chord intervals,
  1,611 bars, and 13,272 beats. Compound tempo hypothesis C has 0.39% median
  error across 72 eligible user-alignment intervals, versus 50.04% for A and
  200.07% for B.
- Closure regressions confirm one complete 12/8 bar is 6 qn with 12 half-qn
  canonical beats, one complete 6/8 bar is 3 qn with six beats, compound 12/8
  at 120 BPM renders 6 qn in 1,999,998 microseconds after required integer
  tempo rounding, and simple 4/4 at 120 BPM renders 4 qn in 2,000,000
  microseconds. Production code was unchanged by closure.

## Phase 2B.1 verification

All Python commands used the project-local Python 3.13.5 interpreter.

- HookTheory parser unit tests: `10 passed`.
- HookTheory adapter unit tests after closure regressions: `53 passed`.
- HookTheory validation regression tests: `112 passed`.
- HookTheory semantic-audit and golden-fixture audit tests: `10 passed`.
- Opt-in real golden adapter integration: `1 passed` (all 19 manifest cases;
  18 conversions and one required missing-payload error).
- Opt-in corpus semantic crosswalk integration: `1 passed in 121.24s`.
- Data-layer tests: `247 passed`.
- MIDI adapter tests: `62 passed, 2 skipped`; the skips are gated real-corpus
  integrations.
- Full default repository suite after closure regressions: `397 passed, 6
  skipped`; all skips are explicitly gated real-corpus integrations. The final
  full suite with both HookTheory corpus integrations enabled: `399 passed, 4
  skipped in 131.13s`.
- Full target-visible corpus smoke: 26,175 valid pieces, zero unexpected
  failures, `32/32` serialization round trips, and `32/32` target-hiding
  comparisons.
- Full target-hidden corpus smoke: 26,175 valid pieces, zero unexpected
  failures, zero annotations, and zero targets.
- `python -m compileall src scripts tests`: passed.
- `git show --check --oneline
  3898b168063094b87e5ca5d88aae0317c1562c3f`: passed and printed
  `3898b16 Remediate HookTheory timing and pitch semantics`.
- `git diff --check
  47812f6cea2d8183b3543798ba1a252bb1380f85..HEAD`: passed with no output.
- Closure commit verification `git show --check --oneline HEAD` passed on the
  phase branch and printed `6111d3d Close Phase 2B.1 HookTheory adapter`;
  `git diff --check main..HEAD` passed with no output. After merging,
  `git show --check --oneline HEAD` printed
  `b1df777 merge: complete Phase 2B.1 HookTheory adapter`, and the base-to-main
  diff check again passed with no output.
- Production dependency/import scan: passed; the HookTheory production adapter
  imports only the standard library, its private production JSON reader, and
  `music_critic.data`.
- Added-line and new-file forbidden absolute-path scan: passed.
- Legacy unchanged check: remains intentionally failing under the explicit
  resolution-C waiver in `docs/LEGACY_DRIFT_REPORT.md`. The report records all
  staged added, removed/renamed, and modified paths with recorded/current Git
  blob hashes. Phase 2B.1 did not modify the external checkout or refresh the
  snapshot.

## Phase 2B.0 HookTheory audit result

- `scripts/audit_hooktheory_legacy.py` is a deterministic, read-only,
  standard-library audit CLI for complete JSON objects, legacy top-level
  fragments, and JSONL. It preserves decimal lexemes with `Decimal`, inventories
  and hashes sources, profiles bounded field evidence, runs named corpus-wide
  anomaly/duplicate/pitch/meter checks, crosswalks the simplified schema,
  audits structure joins, and reports `ori_uid` leakage.
- The raw merged source has 26,178 records: train 21,233, val 2,184, and test
  2,761. Three train records have no `json` payload. Existing processed and
  canonical full outputs each contain the remaining 26,175 records.
- Primary hashes: raw merged
  `8ab601050d0b8c8752c3b6bf190d63edefa5fce07735ce823bca6a3922dff833`,
  processed full
  `18421660eada680a223666f8e9af6b193900d91292b2ea7148e5c0687d2d42fe`,
  and canonical full
  `2b78e7d90bd81bd6a9d9ce946bc1ebff259d6967dcda1ad7b139bfbc5a5d8dc8`.
  The upstream simplified source hash is
  `5e7457df5640170337c6e320d32fe90d6355b5ab96f15dbd3567180a05be9c08`.
  The complete source/processed hash inventory is in
  `docs/HOOKTHEORY_FIELD_AUDIT.md` and the fixture manifest.
- Structure joins by normalized split plus `audio_path` stem match all 11,515
  structure rows: train 9,498, val 927, and test 1,090. Symbolic-only counts are
  train 11,735, val 1,257, and test 1,671; there are no structure-only or
  duplicate structure IDs and no missing structure `ori_uid` values.
- There are 2,714 original-song groups with multiple clips. Twenty-three
  `ori_uid` values cross split boundaries and are explicit leakage findings
  that must be resolved atomically before training.
- The pinned upstream Sheet Sage evidence commit is
  `bbdd7b7b6a5fb845828f82790acdceb03a197779`. The simplified-schema crosswalk
  has 26,175 matches, three raw-only missing-payload records, no
  simplified-only records, and no identifier/split mismatches.
- The crosswalk semantically compares meter regions for every matched record:
  27,217 raw regions, 27,216 simplified regions, 27,216 compared regions, and
  27,216 exact matches. It reports zero missing-raw regions, one
  missing-simplified region, one record count mismatch, and zero value
  mismatches. The bounded coverage discrepancy is clip `nvgy-WaRgkA`; key,
  melody, and harmony are inventoried but were not corpus-wide semantically
  compared.
- Nineteen bounded cases cover major/minor/modal examples, integer and
  fractional timing, first-beat conversion, rests and derived pitches,
  multiple key/tempo/meter regions, root-zero rest and malformed non-rest zero,
  chord types/inversions/decorations, borrowed null/empty/mode/list/unknown
  forms, applied raw evidence, matched and unmatched symbolic structure,
  shared `ori_uid`, a missing payload, `beatUnit=3`, `numBeats=8`, negative
  roots, null note beats/octaves, alternate `_`, and `bb1`.
- Not observed and not fabricated: raw root `8`, stringified borrowed lists,
  unexpected borrowed runtime types, derived out-of-range pitch, non-null
  pedal, exact duplicate regions, duplicate structure IDs, structure-only rows,
  or missing structure `ori_uid`.
- The semantic meter crosswalk accepts canonical numerator `numBeats`, with
  denominator 4 for `beatUnit=1` and 8 for `beatUnit=3`; the one omitted
  simplified region is coverage loss rather than a value counterexample. Still
  unresolved or intentionally deferred: `alternate`, non-null `pedal`, applied
  harmony, and audio-seconds-to-symbolic alignment. Structure timestamps remain
  audio seconds with `section_alignment_status=unresolved_audio_seconds`.
- Phase 2B.0 intentionally added no production adapter or canonical conversion
  entry point. Its evidence gate preceded the Phase 2B.1 implementation above;
  it also added no graph, dataset, model, SSL, training, preference, evaluation,
  inference, or GRPO work.

## Phase 2B.0 remediation verification

All Python commands used the project-local Python 3.13.5 interpreter.

- Corpus-wide audit CLI: passed; the final report was written outside the
  repository under `/tmp` and asserted all named counts, pitch-accounting
  totals, and crosswalk totals.
- Static audit and golden-fixture tests: `17 passed`.
- Opt-in raw/simplified/processed/canonical/structure integration, including a
  full `build_report` count assertion: `2 passed`.
- Full default suite: `331 passed, 4 skipped`; the skips are explicitly gated
  local real-data tests.
- `compileall src scripts tests`: passed.
- `git diff --check`: passed.
- The repository `make check` wrapper was unavailable because `make` is not
  installed; its two commands were run directly and passed as reported above.
- The legacy snapshot checker reports that the external read-only legacy
  worktree's pre-existing staged/dirty state differs from the recorded
  snapshot. The legacy commit remains pinned and this task did not modify,
  format, stage, reset, clean, or restore any legacy file.

## Phase 2A.1 generic MIDI result

- Public API: `MidiAdapterConfig`, `MidiAdapterError`, and `load_midi_piece`
  from `music_critic.adapters`.
- Added the sole runtime dependency `mido>=1.3,<2`. The accepted Phase 1 data
  layer remains standard-library-only and importing `music_critic.data` does
  not import `mido`.
- Supported input: Standard MIDI type 0 and type 1 files with PPQN timing,
  multiple source tracks, multiple channels per source track, empty source
  tracks, note-on/off and velocity-zero note-off, tempo/meter/key metadata,
  names, instruments, programs, percussion channel 9, and empty/no-note files.
- Timing remains exact: absolute source ticks are integers and canonical onset,
  duration, bar, and beat positions use `RationalTime` without float conversion,
  rounding, epsilon comparison, or note splitting at bar/tempo/meter changes.
- Canonical track identity is `(source_track_index, MIDI channel)`. Note pairing
  is FIFO per `(source_track_index, channel, pitch)` and never crosses source
  tracks, channels, or pitches. Unmatched note-offs and dangling note-ons are
  diagnosed without invented notes; real same-tick pairs are preserved as
  grace-like zero-duration notes.
- Tempo defaults to `500000` microseconds per quarter at tick 0 when absent or
  first observed later. Meter defaults to `4/4` at tick 0 under the same policy.
  Defaults use `kind=default` provenance; observed source events use
  `kind=source`, the accepted observed equivalent.
- Global metadata events use deterministic `(tick, source track, message)`
  ordering. Exact duplicates are removed and conflicting same-tick values keep
  the first deterministic value plus a namespaced quality flag.
- Generic MIDI emits `annotations=()` and `targets=()`. Every successful
  conversion passes `validate_piece` and both string/file JSON round trips
  preserve exact equality. No canonical cache is written by default.
- Rejected input: MIDI type 2, SMPTE/non-PPQN timing, non-positive PPQN,
  unreadable/corrupt files, and meter changes inside an active bar.
- Intentionally unsupported: MIDI 2.0, proprietary sequencer/SysEx semantics,
  lyric alignment, sustain-pedal reconstruction, voice/role/pickup inference,
  chord or key detection from notes, section detection, and aesthetic scoring.

## Phase 2A.1 remediation review

- Finding: an observed time signature with `numerator=0` and a positive source
  duration could enter metric-grid construction with a zero nominal bar length,
  preventing the bar loop from advancing.
- Fix: every selected observed/default meter is now validated before boundary
  checking, canonical meter creation, or metric-grid construction. Numerator
  and denominator must be positive integers and denominator must be a power of
  two. Invalid source values raise `MidiAdapterError` with source path, event
  tick, raw numerator/denominator, and the reason; they are never clamped,
  normalized, or replaced with `4/4`.
- Meter-boundary checking now uses one exact rational divisibility calculation
  per meter region instead of iterating across intervening bars.
- Metric-grid materialization has a deterministic combined bar-and-beat limit
  of `1,000,000` records. The adapter computes the exact record count per meter
  interval with integer/rational arithmetic before allocating any bar or beat.
  A rejection reports the source path, active meter, interval, estimated count,
  and configured limit.
- A serialized meter with denominator `2**127` and positive duration is rejected
  by the safety policy without iterative materialization. Ordinary power-of-two
  denominators including `2`, `4`, `8`, and `16` remain supported.
- Smoke discovery remains recursive and case-insensitive, excludes symlinks
  resolving outside the requested root, and now supports `first` and `spread`
  sampling. `first` remains the default. `spread` uses deterministic evenly
  spaced ceiling indices, includes both endpoints when selecting more than one
  file, and never duplicates a selected path.

## Phase 2A.1 verification

All commands used the project-local Python 3.13.5 interpreter at
`.venv/bin/python`.

- `tests/data/test_timing.py`: `28 passed`.
- `tests/data/test_schema.py`: `13 passed`.
- `tests/data/test_validation.py`: `110 passed`.
- `tests/data/test_serialization.py`: `94 passed`.
- `tests/adapters/test_midi.py`: `62 passed`.
- Full suite without the opt-in real-data variable: `314 passed, 2 skipped`;
  both skips are the explicitly gated local real-data cases.
- Explicit real-data integration with
  `MUSIC_CRITIC_RUN_REAL_MIDI_TESTS=1`: `2 passed`. This strictly converted,
  validated, and JSON-round-tripped 20 spread-selected POP909 files and 20
  spread-selected PDMX files without skipping any selected file.
- `.venv/bin/python -m compileall src scripts tests/integration`: passed.
- Data-layer import isolation: `data import isolation passed`.
- Adapter public imports: `adapter imports passed`.
- `git diff --check`: passed.
- Synthetic smoke root: `/tmp/music-critic-midi-smoke.W2edj4`.
- Synthetic smoke: `files_seen=3`, `attempted=3`, `converted=3`, `failed=0`,
  `warnings=10`, `notes=3`, `tracks=5`, `type_0=2`, `type_1=1`.

## Real-MIDI validation

Both source datasets were read recursively and remained unmodified.

- POP909 root:
  `/home/str/music-critic-v2/data/pop909-cl/POP909_processed/POP909_processed`.
- POP909 recursive discovery: `files_seen=909`.
- POP909 100-file spread smoke: `attempted=100`, `converted=100`, `failed=0`,
  `warnings=14475`, `notes=209228`, `tracks=300`, `type_0=0`, `type_1=100`.
- POP909 selected-path coverage: `selected_parent_dirs=1`,
  `selected_min_depth=1`, `selected_max_depth=1`.
- PDMX root: `/home/str/music-critic-v2/data/pdmx/mid`.
- PDMX recursive discovery: `files_seen=254035` across the complete branched
  MIDI tree.
- PDMX 100-file spread smoke: `attempted=100`, `converted=99`, `failed=1`,
  `warnings=378`, `notes=47459`, `tracks=246`, `type_0=0`, `type_1=99`.
- PDMX selected-path coverage: `selected_parent_dirs=100`,
  `selected_min_depth=3`, `selected_max_depth=3`.

Failure triage for both 100-file diagnostic runs:

- unreadable/corrupt MIDI: `0`;
- MIDI type 2: `0`;
- SMPTE/non-PPQN: `0`;
- invalid meter values: `0`;
- meter change inside a bar: `1`, represented by
  `2/31/QmcmH3b8xr1N9KSEu5zS4HG7f6Beq1fENiy3bdZ9D3FXrE.mid` at tick `8970`
  under active meter `75/4`;
- metric-grid safety rejection: `0`;
- canonical validation failure: `0`;
- serialization round-trip failure: `0`;
- unexpected exception: `0`.

The one diagnostic PDMX failure is an explicitly unsupported MVP condition;
the adapter contract was not broadened merely to force 100% conversion. No
hang, uncontrolled memory growth, parser bug, validation escape, or
serialization mismatch was observed.

The mid-bar meter-change rejection and its single PDMX diagnostic failure are
accepted for the Phase 2A.1 MVP. POP909's warning total requires later
warning-code analysis before making training-data quality decisions, but it is
not a Phase 2A.1 merge blocker.

## Phase 2A.1 scope confirmation

- Phase 1 production code, Phase 1 data tests, the accepted schema/data
  contract, and the normative fixture were not modified.
- Project dependencies were not modified by the remediation.
- The Phase 0 repository-contract test was updated to allow `mido` only inside
  `music_critic.adapters`; its bans remain active everywhere else, and the
  adapter/document packages are now required repository structure.
- The read-only legacy repository remains at
  `2d8281f31cc9ad9c8fecaf332da0c61e0e949415` with the same pre-existing dirty
  status observed before this task. No legacy file was modified.
- HookTheory remains documentation-only. No graph, dataset, model, SSL,
  training, preference, quality, inference, or GRPO code was added.
- Phase 2A.1 is accepted and Completed. The later Phase 2B.0 evidence and
  remediation work described above does not alter the accepted MIDI adapter.

## Final Phase 1 result

- Accepted and implemented canonical schema version `2.0.0` with an exact,
  explicit public `music_critic.data` API.
- Implemented normalized exact quarter-note timing with frozen, slotted
  `RationalTime` values and no float-equality timing contract.
- Implemented deeply immutable frozen canonical records. Collection fields are
  tuples, optional observations preserve `None` versus empty values, and raw
  note/track records contain no theory-label or semantic-role leakage.
- Implemented complete deterministic validation with structured errors and
  warnings, exact RFC 6901 paths, reference and ordering checks, target masks,
  confidence and provenance, exact musical timing semantics, and warning-only
  valid pieces.
- Implemented strict field-by-field decoding and validated deterministic JSON
  encoding. Unknown, missing, type, rational, version, and semantic failures
  retain their accepted error-code boundaries.
- Compact and indented JSON are deterministic; file output is UTF-8 with exactly
  one terminal newline, and public operations do not mutate canonical records
  or caller-owned mappings and lists.
- The normative `tests/fixtures/data/canonical_piece_v2.json` mapping decodes,
  validates with warnings only, re-encodes exactly, and remains equal through
  `dumps_piece` and `loads_piece`. Rational fields and immutable collections
  retain their exact Python types; masks, unknown confidence, provenance, and
  alternative annotation views are preserved.
- At Phase 1 completion the data layer used only the Python standard library and
  project runtime dependencies were empty. Phase 2A.1 preserves that data-layer
  isolation while adding `mido` only for adapters.
- No adapter, MIDI parser, graph, dataset, model, training, evaluation, or
  inference implementation was added in Phase 1.

The final float-decoding review fix in commit `396a2b5` was accepted. Huge
positive or negative integers supplied for float-valued mapping fields now
produce `VALUE_NOT_FINITE` at the exact path through
`CanonicalValidationError`; raw `OverflowError` cannot escape and inputs are
not clamped or mutated.

## Phase 1 commit history

- Phase 1A contract review and closure: `241d0e5`, `30ba3f9`, merged by
  `7ca1ce0`.
- Phase 1B.1 timing and schema types: `0ca7b95`.
- Phase 1B.2 validation: `b5c31c6`, with review fixes in `2c16d72`.
- Phase 1B.3 serialization: `1dd4e00`, with accepted float-decoding fix in
  `396a2b5`.
