# Future targets beyond harmonic classification

Status: **user-requested direction and source-readiness proposal, 2026-09-08**.
This is not an adopted architecture, a new training run or a claim of corpus
coverage. Current C0/C2, their 18 heads and the three isolated research pilots
remain unchanged. Evidence below is current adapter code plus retained field
audits; no real corpus scan, TEST access or checkpoint loading was performed.

The requested backbone should represent melody, voices, rhythm, musical form
and arrangement as well as harmony. Additional heads alone do not establish
that these capabilities have been learned. Define an observable task and its
target/evaluation source for each aspect first.

## Targets do not always require new manual annotation

- Source annotation: voice identity, melody role or section function, with
  scope, availability, confidence and provenance retained.
- Self-supervision: hide an observed pitch, rhythmic span or part and predict
  the hidden original or its representation. These targets come from the
  music itself; a class name or expert quality rating is not required.
- Derived descriptors: intervals, onset patterns, density, register and defined
  syncopation measures can be computed from raw notes and meter where available.
  They are useful proxy tasks, not expert labels of musical correctness.

Reconstruction of one observed melody/arrangement must not turn all alternative
valid versions into quality negatives. Degradations remain evaluation-first.

## Verified source readiness

| Source | Relevant evidence | Limitation and next binding |
| --- | --- | --- |
| Dilemmadata AN | Pitch, exact timing, tie/onset evidence, optional `s_part_id/s_voice_id`. | Part/voice scope is used during parsing; canonical scalar voice alone loses AN part scope. Build voice targets from original rows and verified note bindings, not global voice integers. |
| Dilemmadata DLC | Pitch, timing, optional staff/voice, meter and existing harmonic annotations. | Notational voices are not semantic melody/bass roles. Audit coverage, stable scope, simultaneous notes, ties and repairs. |
| HookTheory | Already isolated raw melody, exact pitch/timing, note-aligned `theory.melody.scale_degree`, harmonic targets. | Useful for melodic context, not supervised extraction of melody from polyphony. Section intervals remain unaligned audio seconds; no section targets are emitted. |
| Production POP909-CL | Combined musical score on channel0, target-only harmonic annotation on channel1. | Melody/accompaniment roles are not preserved as separate musical tracks. Channel1 must not become an accompaniment input. |
| Original POP909 | Retained audit and upstream release identify `MELODY`, `BRIDGE` (secondary melody), `PIANO` in primary files. | Original-only production role adapter remains future work. `BRIDGE` is a track role, not a song-form bridge label. Ambiguous alternative versions are masked. |
| PDMX | Accepted adapter preserves raw notes, exact metrical timing, tracks and optional instrument/program/percussion/velocity. | Current path does not infer voice/staff/roles and ignores optional notation/lyrics/expressive objects. No expert theory/form/arrangement-quality targets are implied. Corpus readiness and usable coverage still require separate evidence. |

Code anchors: `adapters/dilemmadata.py` source parsing, tie/repair and source-row
bindings; `adapters/hooktheory.py` melody conversion, scale-degree targets and
`_structure_flags`; `adapters/pop909_cl.py` score-instrument contract. See
[original POP audit](POP909_ORIGINAL_FIELD_AUDIT.md),
[future original adapter](POP909_ORIGINAL_ADAPTER_CONTRACT.md),
[HookTheory alignment audit](HOOKTHEORY_FIELD_AUDIT.md),
[PDMX input contract](PDMX_PIPELINE_CONTRACT.md), and
[harmonic input/target boundary](HARMONIC_SUPERVISION.md).

## Candidate tasks and acceptance questions

| Aspect | First useful objective | Required answer source | What must be checked |
| --- | --- | --- | --- |
| Melodic context | Reconstruct a hidden span's intervals and timing; identify transformed motif recurrence later. | Original observed notes; source melody annotations where necessary. | Remove answer-revealing derived features. Evaluate on unseen works; exact recurrence proxies do not establish semantic motif equivalence. |
| Voice continuity | Predict a link to the next note of the same voice. | Audited source-scoped notation voices and canonical row bindings. | Crossings, rests, ties, chords within a notational voice and multiple voices per track. Unknown relations remain masked; no gold voice-derived input edges for this task. |
| Melody/part roles | Predict lead/secondary/accompaniment roles for notes or temporal part segments. | Original POP909 role mapping; later other audited sources. | Hide semantic names, fixed track order and other direct role identifiers. Roles may change in time; an instrument can serve several roles. |
| Rhythm | Recover hidden onset/duration patterns; predict a missing part's rhythmic activity from other parts. | Original notes, rests and reliable meter where present. | Timing-derived features and graph topology may reveal hidden values. This is a new mask audit, not a claim that current pitch masking already supports rhythm masking. |
| Rhythmic descriptors | Estimate explicitly defined syncopation, onset density or inter-part coincidence for hidden regions. | Deterministic definitions over notes/meter. | Distinguish descriptor imitation from musical understanding. Reading an exposed meter/beat index is not evidence of contextual rhythm analysis. Score-only data cannot supply missing performance microtiming. |
| Form | Segment boundaries, repeated-section relations, then intro/verse/chorus/etc. | Coverage-audited boundaries and symbolic-aligned semantic section labels. | Boundary detection, repetition and section-function classification are separate tasks. Audio seconds require alignment; a short clip may omit the whole-song context. |
| Arrangement | Recover missing part/texture/activity conditioned on the others; later role and register interaction. | Actual multitrack music and verified optional roles. | Piano accompaniment is useful texture data but limited evidence for broad instrumentation. Instrument program is not arrangement role or quality. MIDI does not establish timbral/mix quality. |
| Contextual non-chord tones | Passing/neighbor/suspension/anticipation roles and preparation/resolution links. | Audited voice plus harmonic context, ideally expert role annotations. | Chord membership and a sus chord symbol do not uniquely determine these roles. Rule-derived labels must remain explicitly weak/pseudo labels. |

## Form is the least ready supervised family

Current Dilemma `phrase`/`section` are deferred with
`missing_negative_supervision`; absent markers cannot be relabelled as known
non-boundaries. Existing markers also do not define a shared intro/verse/chorus
taxonomy. Obtain audited fully annotated spans or explicitly design a separate
partial-label objective before enabling them.

HookTheory structure fields are a candidate source, but the current adapter
uses verified grouping metadata and deliberately leaves audio-second intervals
unaligned. Their semantics and coverage must be audited before promising section
role supervision. Repetition can be studied with SSL/proxy objectives sooner,
but an A/B repetition pattern does not itself label which section is a chorus.

## Proposed order and shared controls

1. Finish the existing isolated pilots. Task03 provides synthetic-tested voice
   candidates/targets; it has no production voice importer or trained voice
   head. Group pitch masking remains separate from the later rhythm objective.
2. Perform a registered TRAIN-only source-coverage audit for voice continuity
   and the proposed rhythmic objectives: independent pieces, valid target
   relations, uncertainty/repair exclusions, distributions and source overlap.
   Choose pilot size from measured support and resource cost, not a fixed
   arbitrary number of pieces.
3. Prepare a separate original-POP909 role adapter if role/extraction tasks are
   selected. Preserve shared `pop909-lineage:<song-id>` grouping with POP909-CL;
   derivatives and alternative versions cannot cross partitions.
4. Audit form semantics/alignment separately. New external section annotations
   or a reviewed small evaluation set may be necessary. Do not advertise named
   form recognition based only on unlabelled reconstruction.

Use source-aware target adapters with per-row masks. Not every dataset needs
all heads or a shared raw class ID. Supervised and SSL draws must retain their
own provenance and comparable accounting. Report each aspect separately plus
retention of the existing harmonic tasks; a harmonic primary_score alone
cannot measure progress in melody, rhythm, form or arrangement.

Hierarchical windows and revised pooling are a separate architecture ablation
after the target/evaluation contracts are concrete. Do not combine voice,
rhythmic SSL, role labels and shifted attention in one uninterpretable pilot.

## Primary external references checked

- [Dilemmadata source description](https://github.com/johentsch/dilemmadata):
  notation voice/staff and harmonic/metrical/structural fields. Current local
  adapters and masks determine what our model actually receives.
- [Original POP909 release](https://github.com/music-x-lab/POP909-Dataset):
  primary melody, secondary melody and piano accompaniment tracks.
- [Sheet Sage HookTheory release](https://github.com/chrisdonahue/sheetsage#hooktheory-dataset):
  melody/harmony annotations and alignment metadata; not full accompaniment.

No legacy code was inspected or reused. No model/schema change, dependency,
new substantial experiment or scientific decision is introduced by this note.
