# POP909-CL frozen corrected-model evaluation

2026-09-07 extension: [POP909_CL_TRANSFER](POP909_CL_TRANSFER.md) describes the
separate explicit TRAIN/VALIDATION entrypoint. This document's original v1
API, execution guard and report remain synthetic-only.

WS4, 2026-09-05. Implementation gate for future transfer evaluation, based on
`f0e0886161c9f67786011508006db61c803c1752`. This additive protocol evaluates
root and quality only. It neither reproduces AnalysisGNN's paper evaluator nor
measures music quality, aesthetic preference, PLL, or all 18 corrected heads.
Synthetic CPU acceptance is not measured zero-shot accuracy or corpus readiness.

## Execution boundary

Implementation: `music_critic.experiments.pop909_cl_frozen`.
`Pop909ClFrozenEvaluationProtocol@1.0.0` and
`Pop909ClFrozenEvaluationReport@1.0.0` are independent of existing evaluation
and training contracts. No shared registry, graph, adapter, trainer, frozen
C0/C2 profile, experiment record, or ledger changes.

The CLI defaults to help without opening a source or checkpoint. The only
execution option in this wave is:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/music-critic-v2-integration-venv/bin/python \
  -m music_critic.experiments.pop909_cl_frozen.run --smoke
```

This creates a temporary synthetic MIDI and randomly initialized corrected
checkpoint, performs CPU inference in eval mode, writes JSON to stdout, then
removes its temporary files. `--output /absolute/new/report.json` writes a new
file with exclusive creation. No implicit source/checkpoint discovery or
production paths exist. `--split test` (including uppercase TEST) fails before
source/cache/checkpoint callbacks or temporary fixture construction. Train and
validation are also disabled: future real-data execution requires a separately
registered protocol and explicit inputs.

`evaluate_from_readers(split=..., checkpoint_reader=..., source_reader=...)`
places the guard before callbacks. `evaluate_frozen` places it before input
iteration. Its low-level providers must be declared synthetic; arbitrary
caller-provided objects do not establish real-data provenance. A preloaded
piece explicitly labelled TEST is rejected before graph build/inference.
The standalone loader cannot infer a split from bytes; the guarded execution
entry point owns access authorization.

## Model-only checkpoints

`load_model_checkpoint(path, expected_sha256=..., expected_size_bytes=...)`
reads at most expected size plus one, hashes those exact bytes, and passes the
same bytes to `torch.load(map_location="cpu", weights_only=True)`. There is no
unrestricted pickle fallback or global safe-global registration. It constructs
a fresh fixed corrected model on CPU, checks the contract, complete state
keys, shapes, exact dtypes, dense CPU tensors, finiteness, and state fingerprint,
then loads strictly and returns a frozen eval model plus artifact evidence.
Caller Python, NumPy, CPU Torch and already-initialized CUDA RNG states are
restored on success and failure; checkpoint RNG is never applied. No optimizer,
scheduler, scaler, sampler, training resume API, or optimizer construction is used.

Supported envelopes:

- `CorrectedAnalysisGNNModelOnlyCheckpoint@1.0.0` has exactly
  `schema_version`, `model_contract`, `model_contract_fingerprint`,
  `model_state`, `model_state_fingerprint`.
  `create_model_only_payload(model)` returns a detached snapshot of primitive
  metadata and CPU tensors. The complete contract has its internal fingerprint
  recalculated and exact canonical JSON bytes compared, including scalar types.
- Restricted-compatible `CorrectedAnalysisGNNCheckpoint@1.0.0` envelopes use
  current `model_contract_fingerprint`, `model_state`, and
  `model_state_fingerprint`. Extra training metadata is ignored. When the full
  contract payload is absent, evidence says
  `reconstructed_current_contract_fingerprint_only`; it does not claim to have
  verified a historical payload. A supplied payload must pass the full check.

**Historical compatibility gap:** actual C0/C2 files were not read. Training
envelopes can contain NumPy RNG arrays or environment objects that restricted
loading rejects. Supporting a synthetic primitive envelope does not prove those
historical files will load. Any future export/remediation requires separate
authorization and evidence; this implementation never retries unsafe loading.

Evaluation rechecks supplied checkpoint evidence against the actual model
contract/state before source iteration. An in-memory model without artifact
evidence is explicitly reported with `checkpoint_evidence=null`.

## Root and quality projection

The local projection locks the current ordered root/quality task and vocabulary
registry to SHA-256
`c76437a04c6b2b985831d4c249b894dd49eb4a0e97045612521bf76e4184db52`.
Every contract includes that complete binding, spelling-to-PC table and quality
templates. Registry drift fails closed and requires a new reviewed protocol.

Root: softmax over all 37 spelled classes, sum the probabilities of every
enharmonic class into PCs 0..11, then argmax12. An exact tie chooses the lowest
PC ID. Argmax37 followed by PC conversion is forbidden.

Quality: choose the argmax over all 17 corrected classes, with lowest corrected
class ID breaking an exact tie. Only gold coverage is limited to subset9:

| POP label | Root-relative semitones | Corrected label | Corrected ID |
| --- | --- | --- | --- |
| M | 0,4,7 | major triad | 0 |
| m | 0,3,7 | minor triad | 1 |
| o | 0,3,6 | diminished triad | 2 |
| + | 0,4,8 | augmented triad | 3 |
| D7 | 0,4,7,10 | dominant seventh chord | 6 |
| M7 | 0,4,7,11 | major seventh chord | 5 |
| m7 | 0,3,7,10 | minor seventh chord | 4 |
| /o7 | 0,3,6,10 | half-diminished seventh chord | 9 |
| o7 | 0,3,6,9 | diminished seventh chord | 8 |

Gold sus2, sus4, mM7 and +7 are explicitly unsupported in v1 and counted.
Other unrecognized strings fail rather than silently expanding coverage.
Predictions outside subset9 remain errors, false positives in their full17
class and false negatives in the true class. There is no subset9
renormalization, 17-to-15 paper mapping, or tuning from evaluation metrics.

## Raw views and target alignment

Native is the primary view; metadata_removed is secondary, fixed before
metrics. Each view clones the canonical input and rebuilds its graph through
public `build_raw_graph`. Metadata removal changes exactly:

```text
note.velocity, note.program, note.channel, track.program, track.channel -> None
```

Rebuilding closes derived values and availability, including
`track.mean_velocity`. All other fields stay unchanged, including tempo, meter,
track IDs, source_track_index, is_percussion, pitch, onset, duration and
topology-driving timing. No arbitrary tensor columns are zeroed. Canonical
sources are not mutated; strict graph, numerical model-input and policy-bound
view fingerprints are reported separately.

POP909-CL channel 0 already contains combined melody and accompaniment.
Channel 1 contains annotation only. The smoke calls public score projection
and MIDI loading before prediction, never putting channel-1 notes into a raw
piece. Source chord extraction is deferred until both views have predictions.
The general evaluator's target callback runs once per piece after both views;
source-side annotations already present in a supplied canonical object remain
sidecars and never enter raw graphs, candidate identities, or logits.

`targets_from_chord_evidence` uses original block `onset_tick/ppqn`, without
clipping it to canonical duration. For each source chord block, exact
`RationalTime` containment finds a unique raw beat `[start, start+duration)`.
One block is one observation. No nearest onset, span average, or chord-duration
weighting is used. Missing/ambiguous containing beats are unavailable. Derived
no-chord gaps and trailing unannotated spans are counted separately; they are
not synthetic chord events. Missing chord instruments count as missing target
pieces/groups with zero events, never negative targets.

Multiple events in one beat share its prediction and remain separate. The
report counts beats with multiple events, events in them and conflicting
available source labels independently for root and quality. This is a beat
resolution limit, not a rule for choosing among conflicting annotations.

## Metrics and deterministic evidence

Root and quality masks are independent; their intersection is used only for
joint accuracy. Each task includes accuracy, observed-true-class macro-F1,
included class IDs, full class order, per-class support/predicted count/TP/FP/FN,
per-class F1 and full confusion. A supported class with no correct predictions
has F1 zero. An absent true class with false-positive predictions has per-class
F1 zero, but is outside the declared observed-true-class macro. All zero
denominators use `value=null` with a specific reason. There is no mean of the
two F1 values and no `corrected_primary_macro_score`.

Coverage names event, piece and group populations and eligible counts.
Label unavailability (missing/ambiguous/unsupported) and alignment
unavailability are separate axes with event/piece/group counts; they can
overlap. `excluded_events` is a disjoint count using label reason first, then
alignment reason. `unaligned_events` counts every alignment failure;
`otherwise_label_eligible_unaligned_events` is the subset with usable labels.
These statistics must not be summed across overlapping axes.

Inference currently processes one raw piece at a time. Integer confusion
aggregation and canonical sorting make the complete report invariant to
piece/event iteration order; there is no floating batch aggregation dependency.
Reports bind protocol, views, registry, model, artifact evidence when available,
candidate IDs/logit fingerprints, source-event target fingerprints, counting
policies, caller-declared synthetic provenance and access observations. Runtime
timestamps and temporary paths are excluded. The top-level fingerprint hashes
compact sorted-key UTF-8 JSON with one trailing newline, excluding its own
`report_fingerprint` field. Nested contracts retain their named hash semantics.

No experiment record is required for these ordinary synthetic contract tests.
The next scientific phase is a separately registered TRAIN/VALIDATION transfer
evaluation after verifying the required checkpoint artifacts and historical
envelope compatibility. TEST stays closed.
