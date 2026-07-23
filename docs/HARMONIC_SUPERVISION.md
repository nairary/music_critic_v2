# Harmonic Supervision Contract

Status: **ACCEPTED DOCUMENTATION CONTRACT**. This document records future
training and evaluation boundaries. It does not implement Phase 4B, a target
ontology, model heads, rendering, probabilistic decoding, scoring, or
inference.

## 1. Terminology and task boundary

Music Critic V2 separates four tasks that must not be described as equivalent:

1. **Harmonic-semantic recognition** predicts which harmony is present in an
   observed score, or which harmonic analyses are supported at a candidate
   position.
2. **Melody-conditioned harmonization** predicts harmonies compatible with a
   melody. A direct source annotation is one observed harmonization, not proof
   that it is the only correct one.
3. **Actual accompaniment generation or likelihood** models the performed or
   score notes, voicing, register, rhythm, texture, and transitions of an
   accompaniment in context.
4. A **preference/quality critic** estimates musical aspects or comparative
   preference. It requires quality/preference evidence and evaluation beyond
   chord recognition or masked reconstruction.

The following terms are normative in project documentation:

- **raw observation**: source material available to the deployable encoder, or
  a deterministic projection reproducible from arbitrary unlabeled MIDI;
- **direct source annotation**: a label or annotation event supplied by the
  source rather than inferred by V2;
- **derived harmonic target**: a deterministic or set-valued transformation of
  a direct source annotation, with explicit provenance and method version;
- **target-only diagnostic rendering**: a realization created from a target
  for inspection or a separately declared experiment, never raw input;
- **actual performed/score accompaniment**: notes and texture that belong to
  the musical performance or score itself, not an annotation rendering;
- **auxiliary harmonic supervision**: masked semantic targets used to train
  interpretable heads and the shared representation;
- **masked conditional likelihood**: a normalized conditional probability for
  held-out observed musical values under an explicit masking protocol;
- **pseudo-log-likelihood (PLL)**: an aggregate of masked conditional
  log-probabilities over events or spans;
- **preference/quality critic**: a model trained and evaluated against
  preference, aspect, or calibrated quality evidence.

## 2. Raw-observation and target boundary

The shared encoder consumes only a raw observation. Harmonic annotations and
everything deterministically derived from them remain target-only.

Safe data flows are:

```text
HookTheory melody-only raw graph
    -> shared encoder
    -> auxiliary harmonic predictions

POP909-CL channel-0 combined-score raw graph
    -> shared encoder
    -> auxiliary harmonic predictions
```

HookTheory direct chord annotations and POP909-CL channel-1 chord blocks may be
normalized into compatible target families. Dataset-specific availability
masks and `annotation_view_id` values retain differences in coverage,
ambiguity, and interpretation. Missing or ambiguous supervision is never
converted to a negative class.

The following flow is forbidden:

```text
target-derived chord notes or annotation blocks
    -> raw canonical tracks/notes
    -> graph features or topology
    -> shared encoder
```

The prohibition still applies if a target-derived track is renamed, its
channel/program is changed, its semantic role is omitted, or its notes are
presented as ordinary note nodes. Pitch content, onset, duration, voicing, and
regular block structure can identify the answer. Consequently target-derived
information must not affect raw canonical musical content, raw statistics,
graph features, graph topology, serialization, cache identity, fingerprints,
or any production inference path.

Deriving a target representation is not itself leakage. Leakage occurs when
the derived target crosses this boundary into model input or input identity.

## 3. HookTheory and POP909-CL comparison

| Property | HookTheory | POP909-CL |
|---|---|---|
| Raw observation | The current adapter exposes the annotated clip's isolated melody as one raw musical track. | Channel 0 is the combined polyphonic musical score. It is not melody-only. |
| Direct source annotation | Chord/theory annotations associated with the melody. | Expert-reviewed, human-corrected chord blocks embedded on target-only channel 1. |
| Learning question | Which harmony is compatible with this melody, and what theory semantics describe it? | Which harmony is present in the combined score? |
| Main limitation | One annotation is one observed harmonization; multiple harmonizations of the same melody may be valid. No actual performed accompaniment is supplied. | Channel 1 is an annotation instrument, not an ordinary performed accompaniment part. It describes harmony already represented by the combined score. |
| Ambiguity policy | Preserve source views and later use set-valued or multi-candidate targets where specified; single-label cross-entropy must not be interpreted as proof of one uniquely correct chord. | Preserve the Phase 4A field-specific supported, ambiguous, unsupported, missing, and no-chord masks and provenance. |

The datasets can train shared harmonic heads where their target meanings and
alignment are compatible. Dataset identity alone never determines target
availability; the actual per-target mask does.

The Phase 4A/4B POP909-CL leakage contract is unchanged. In particular, the
channel-1 block may supervise recognition from channel 0 but may not become an
encoder input. POP909-CL must not be described as an accompaniment-generation
corpus merely because its annotation is encoded as MIDI notes.

## 4. Harmonic target representation and provenance

Future common harmonic target families may include:

- chord presence, boundary, and span/duration;
- root;
- quality;
- a 12-dimensional pitch-class multi-hot or equivalent set-valued target;
- raw pitch multiset where the source directly provides it;
- bass pitch class and inversion when available;
- no-chord state;
- decorations, applied/borrowed, and functional fields only after their
  semantics are supported;
- multi-candidate or set-valued targets for ambiguity.

HookTheory chord annotations may and should be used analogously to POP909-CL
chord targets, including derived pitch-class/set targets. This is a future
extension of the accepted HookTheory target use, not an implementation in this
documentation change.

Every target entry retains availability, source, provenance, and confidence
according to the canonical contract. A direct source annotation points to its
source annotation provenance. A derived harmonic target uses source `derived`,
references the direct annotation as a provenance parent, and records the
deterministic normalizer or renderer version. Alternative legitimate analyses
remain separate annotation views unless the source supplies a probability
distribution.

A conversion such as:

```text
C major annotation -> C3 E3 G3
```

adds project choices for octave, voicing, doubling, register, note order,
rhythm, and duration. Such a realization is not independent human evidence and
is not actual performed/score accompaniment. With explicit `derived`
provenance it may later serve as a target-only diagnostic rendering, a
canonical pitch-set view, a controlled visualization, or a decoder target in
an explicitly scoped experiment. It may never be silently called real
accompaniment ground truth.

## 5. Harmonic semantics versus actual accompaniment

Future modeling has two distinct levels.

### Level 1: harmonic semantics and planning

Shared auxiliary heads may predict boundary, root, quality, pitch-class set,
bass/inversion, and no-chord, followed later by local key, Roman numeral, and
cadence. HookTheory and POP909-CL are complementary sources; Dilemmadata may
add theory supervision in its later roadmap phase.

This level asks: **what harmony is present or suitable in this context?** It
does not assign a normalized probability to a particular accompaniment
voicing, and classifier confidence is not a harmony-quality score.

### Level 2: actual note and voicing likelihood

A future probabilistic or energy-based objective may evaluate the observed
notes and texture through masked pitch prediction, masked pitch-set
prediction, beat/bar-span completion, leave-one-track-out completion,
normalized PLL, and controlled harmonic corruptions.

Candidate sources of actual polyphony are:

- PDMX through a raw-MIDI-compatible projection;
- original POP909 only after a separate alignment and lineage audit;
- the POP909-CL channel-0 combined score;
- other future inference-safe raw MIDI corpora.

This level asks: **how plausible are these particular notes, voicing, and
transitions in context?** Neither HookTheory chord rendering nor POP909-CL
channel-1 annotation blocks constitute actual voicing supervision.

## 6. Arbitrary-MIDI inference contract

Production inference accepts MIDI without required semantic roles such as
melody, accompaniment, bass, chords, voice, or staff. The shared encoder must
not require those labels. Track roles may be auxiliary targets, but never
mandatory encoder inputs.

Future training and evaluation policy must cover:

- track permutation and removal of track names;
- metadata, program, and channel dropout;
- track merging and, where justified, track splitting;
- single-track polyphonic and multi-track inputs;
- missing or unreliable metadata;
- a separate blind raw-MIDI evaluation set.

For the future Dilemmadata adapter, pitch, onset, duration, and compatible
meter fields may enter a raw-compatible projection. Staff, voice, note
spelling, `step`, `alter`, and `tpc` must not automatically be treated as
available in arbitrary MIDI. Harmony, key, cadence, phrase, Roman numeral, and
other analysis columns are targets. The adapter and its exact projection
remain Phase 9 work.

## 7. Future masked likelihood and PLL direction

A conventional chord classifier does not compute the probability of a
specific accompaniment. Its confidence must not be renamed harmonic quality.
Likewise, GraphMAE reconstruction loss is a representation-learning objective,
not automatically an aesthetic score.

A true PLL experiment requires a decoder that emits normalized probabilistic
distributions over observed musical values. Evaluation masking must be
deterministic and inference-safe. Scores must be normalized at least by the
number of predicted events, and evaluation must separately measure dependence
on length, density, genre, and complexity.

A conceptual harmonic-note PLL is:

```text
PLL_harmony(X) = mean_i log p(pitch_i | X without masked event/span i)
```

This expression is a research direction, not an implemented decoder,
production API, or final probability factorization. A design gate before the
probabilistic decoder must specify event/span units, conditioning context,
normalization, masking, treatment of simultaneous notes, and calibration.

High likelihood can reflect commonness or banality. Low likelihood can reflect
high-quality but unusual music. PLL is therefore not a complete aesthetic
assessment. A future preference/quality critic should test combinations of
note/harmonic PLL, theory representations, preference learning, controlled
corruptions, fragility or quality-drop measures, calibration, and human
evaluation.

## 8. Required future ablations

At minimum, Phase 15 must compare:

1. no chord supervision;
2. HookTheory-only chord supervision;
3. POP909-CL-only chord supervision;
4. combined HookTheory and POP909-CL chord supervision;
5. label-only harmonic heads;
6. pitch-class-set harmonic heads;
7. SSL reconstruction without PLL;
8. a probabilistic PLL variant;
9. PLL plus the preference/quality critic;
10. sensitivity to track permutation, merging, and metadata removal;
11. melody-only, combined-score, and heterogeneous raw-MIDI evaluation.

Exact hierarchical and adaptive objectives must also be compared rather than
assumed equivalent: coherent onset/beat/bar-span masks, pitch-only masks with
rhythm visible, and track/span masks can answer different questions.

## 9. Limitations

- HookTheory provides one annotated harmonization per clip and no actual
  performed accompaniment.
- POP909-CL channel 1 provides expert-reviewed harmonic annotation, not a
  performed accompaniment track and not infallible gold.
- Dataset-specific vocabularies and ambiguity policies may not admit a
  lossless one-to-one common class vocabulary.
- Pitch-class sets discard voicing, register, doubling, order, rhythm, and
  expressive performance.
- A semantic chord error, an unlikely voicing, and a low human preference are
  different observations and require separate metrics.
- Role-agnostic robustness and likelihood/quality calibration remain empirical
  claims that require blind evaluation.

## 10. Explicitly deferred implementation questions

This documentation decision does not choose or implement:

- the final common harmonic vocabulary or target tensor layouts;
- HookTheory pitch-class-set derivation or a chord renderer;
- POP909-CL Phase 4B adapter behavior beyond its accepted contract;
- handling of HookTheory applied harmony or unresolved decorations;
- a Dilemmadata or PDMX adapter;
- a probabilistic decoder, event factorization, energy model, or PLL API;
- exact masking/corruption mixtures or likelihood normalization beyond the
  minimum requirements above;
- a quality/preference model, training run, checkpoint, or inference path.

These decisions belong to their named roadmap phases and require tests,
provenance, leakage checks, and ablations before implementation claims are
made.
