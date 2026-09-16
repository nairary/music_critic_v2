# Raw-time readout preparation

Phase 9E diagnostic preparation; this is an isolated candidate/routing contract,
not a change to the trained C2, fusion or MMoE models. The existing eighteen
heads, loss, graph, sidecars, training stream and checkpoints are unchanged.

## Raw candidates

`music_critic.data.raw_time_candidates` reads only note `onset_qn` and
`duration_qn`, and beat `start_qn` and `duration_qn`. It returns exact rational
times; duplicate times from different raw origins are one temporal position.
Note offsets are onset plus duration. Beat boundaries include starts and ends,
including the final end. Grace and percussion notes are retained. No rounding,
clipping, interpolation, label filtering or semantic segmentation is used.

Four fixed alternatives are available: attacks; attacks plus offsets; attacks
plus beat boundaries; and the union of all three sources. Serialized
`RawTimeSources@1.0.0` keeps the three source sets separately, sorted by exact
time and bound to the canonical piece hash. They can be regenerated from
unlabeled canonical MIDI/score input. No chord boundary is an input.

## Coverage and preserved support

The source-eligible denominator is `available AND NOT masked AND known class`,
independent of old output alignment. Report old-valid retention separately.
Cadence coverage must include eligible rows without a historical raw onset.
The five-component joint denominator is reconstructed from the source-eligible
per-head entity intersection, rather than the old valid-only joint rows.

The five note heads remain indexed by note identity; simultaneous notes must
not be merged into one temporal prediction. The twelve harmonic heads and
onset cadence are the temporal coverage subjects. Repeated label rows retain
their weight/counting identity in this diagnostic; no relabeling or deduplication
of training examples is implied.

A raw-only exact-time grid need not contain every annotated moment: a tied
segment can have an internal source position absent from the merged MIDI note.
A source-eligible label missing the proposed grid must be reported explicitly.
It cannot be dropped silently or treated as a negative example.

The isolated routing extension preserves two candidate namespaces, constructed
before reading targets: fine times and existing raw beat identities. An
available known harmonic target uses its exact fine slot when present;
otherwise it may retain its existing valid beat slot. Cadence cannot borrow a
beat slot through this fallback. Source masks and vocabulary checks precede
alignment. Unknown beat IDs remain unmapped. A fallback is coarse coverage,
not a claim to recover the exact annotated event time.

Different labels at the same exact time still share a fine prediction slot.
Conflicting labels within a fallback beat may also remain. Both are counted
explicitly, with no analyst selection, majority relabeling or silent masking.

## What remains before training

The generator and routing code do not supply a learned vector at a note offset
or a silent beat boundary. A future readout must construct a raw-time-specific
representation using the position and the relevant sounding/starting/ending
notes, together with existing context. Copying the identical beat vector into
several slots without a mechanism that distinguishes their times does not fix
the information bottleneck.

That readout, its gradients, batching, prediction-to-target contract, loss
normalization and checkpoint schema require a separate opt-in implementation
and tests. Candidate count is not training time or GPU memory. These coverage
diagnostics establish neither a macro-F1 ceiling nor an expected quality gain.
