# Multiscale aspect critic v1

The implementation is a frozen-backbone critic prototype with global aspect
ranking and local synthetic-intervention evidence. It is not a trained or
human-validated music-quality model. Its [specification](workstreams/2026-09-21/critic-wayfinder/spec.md)
sets engineering, native-runtime and scientific completion as separate gates.

## Implemented behavior

`src/music_critic/critic/` provides:

- `BackboneFeatureAdapter`: strict main C2/temporal adapters, one encoder pass,
  detached latest states plus local notes and earlier track/SONG context.
  `forward_with_base` adds access without changing temporal state-dict keys or
  ordinary outputs. C2 slice states are raw sounding-note means, explicitly
  distinguished from pretrained temporal states.
- Exact raw note/event, slice, beat, bar and track geometry. Incomplete bars
  do not form full 2/4/8-bar spans; releases are outside the sounding half-open
  interval, and grace events are separate from sustained sound.
- `MultiscaleAspectCritic`: shared track-event BiGRU, slice BiGRU, bar BiGRU;
  sequential gated top-down refinement and upward repooling; aspect attention
  and four global scores. Default width128,64 hidden units in each direction.
- Local harmony/slice, melody/note+event, rhythm/event+beat and structure/boundary
  outputs. `musical_error_location` remains unavailable with a stated reason.
- `InterventionTargetBuilder`/`VerifiedPatch`: independently checked actual
  differences, masks, exact old/new support and generation-side provenance.
  Eligibility `support_ids` do not become an all-positive target. No-op edits
  cannot become weak preference pairs. Incomplete diffs provide no negatives.
- `CriticLoss`: work/aspect-balanced ranking plus class/type/work/aspect-balanced
  local BCE; default local coefficient0.25. `lambda_local=0` supports the three
  global-only controls on the same pair sidecars.
- `evaluate_local_interventions`: separate localization precision/recall and
  exact interval-union IoU for slices/beats. Point events have no invented extent.
- `FrozenFeatureCache`: per-view hashes, actual backbone weight binding, exact
  geometry/stages and integrity checks. Safe weights-only deserialization; new
  heads can train on features extracted/loaded even under caller inference mode.
- `CriticTrainer`: work→aspect→pair sampling and paired side swaps, AdamW over
  the new model, finite checks/gradient clipping, limits on attempts/updates,
  unsupported-batch skips and resumable optimizer/sampler/RNG/selection state.
  Head-dev selection rejects known train-work/component overlap. Diagnostics on
  train data are explicit and ineligible for checkpoint selection.
- `AspectScorePolicy`: work-balanced TRAIN-reference percentiles, explicit
  weights/coverage and common-support comparisons. Reference scales bind actual
  critic weights and encoder features, horizon and support policy.
- `CriticInference`: one unlabelled MIDI/canonical piece to a JSON-ready report,
  exact rational coordinates, bound inference bundles and explicit unavailable
  values. Raw scores are available without a reference bank; normalized overall
  is not. No renderer or original/reference MIDI is needed for prediction.

The original theory heads remain intact and frozen. Isolated C3 worktree
variants require their own admitted adapter; they are not accepted just because
some state-dict keys look similar. No new dependency was added.

## Public interfaces

The following snippets illustrate interfaces, not runnable experiments or
permission to process corpus/TEST data. Bind the selected trained backbone and
its source/checkpoint provenance before using them.

```python
from music_critic.critic import (
    BackboneFeatureAdapter, FeatureBinding, MultiscaleAspectCritic,
    CriticInference, load_critic_bundle, save_critic_bundle,
)
from music_critic.critic.features import TEMPORAL_ADAPTER_VERSION
from music_critic.critic.cache import FrozenFeatureCache

# backbone is an already loaded main TemporalSliceModel with verified provenance.
adapter = BackboneFeatureAdapter(backbone, FeatureBinding(
    source_id=source_snapshot_id,
    checkpoint_id=checkpoint_sha256,
    adapter_version=TEMPORAL_ADAPTER_VERSION,
))
pyramid = adapter.extract((original, variant))  # CanonicalPiece objects, no labels argument
cache = FrozenFeatureCache(feature_cache_directory)
keys = cache.put(pyramid)
first = cache.get(keys[0], adapter.binding, pyramid.samples[0].raw_hash)

# After a separately registered training/selection run:
save_critic_bundle(new_bundle_path, trained_critic, adapter.binding,
                   critic_id=selected_critic_id)
head, critic_id = load_critic_bundle(new_bundle_path, expected_binding=adapter.binding)
predictor = CriticInference(adapter, head, feature_binding=adapter.binding,
                           critic_id=critic_id)  # no normalized overall without reference
report = predictor.score_midi(midi_path)
predictor.write_report(report, new_report_path)
```

Backbone IDs alone are not trusted as weight identity: the adapter hashes actual
weights and invalidates normal in-place changes, replacements and state loads.
Do not mutate `.data` or manually corrupt version counters to bypass guards.
Reconstruct the adapter after changing source/weights/dtype/device outside its
supported `.to(...)` path. Cached tensors are owned by the cache/read result;
trainer guards reject in-place bank mutation after binding.

For training, `CachedPair` holds one detached pyramid per side and its
`PairTargets`. Build targets with `InterventionTargetBuilder` and explicit
`VerifiedPatch`; never put that sidecar inside `forward` input. `CriticTrainConfig`
requires seed, learning rate, batch size, update/attempt limits and clipping.
`CriticTrainer(..., bindings=...)` requires source, split and target-version
identities. Model initialization seed is caller-owned and should be recorded.

Use `step()` for one bounded update and `save(new_path)` / `load(path)` for
continuation. The trainer does not decide cloud cost or launch an unbounded loop.
Use fresh retained checkpoint paths; publication/retention follows
[EXPERIMENT_LOGGING](EXPERIMENT_LOGGING.md). CPU fixture replay is checked; native
GPU resume/memory must still be admitted on the selected environment.

`evaluate(head_dev_bank)` returns pair metrics; `selection.consider(result,
update=...)` applies the fixed macro-aspect criterion, pair-loss tie-break and
earlier-update tie-break. Keep local metrics separate from checkpoint selection.
Set `CachedPair.split='head_dev'` for development evaluation and bind known
components; unknown component/pretraining overlap remains a documented gap.

For scaled overall, construct a `TrainReferenceBank` with its work-balanced rows,
original/variant type weights, fixed rational-QN horizon and support policy.
Obtain `predictor_id` with `critic.inference.predictor_identity(head, binding,
critic_id)`; this identity is also in unscaled inference reports. Pass the bank
as `reference=` to `CriticInference`. Changing model weights or backbone binding
requires a new reference bank. A different horizon is rejected, not silently
compared. Scores are relative to this reference, not calibrated human ratings.

Controls live in `critic.baselines`: `RawSummaryCritic`, `BeatProbe`, and
`MultilevelPoolingCritic`. They share the score/output contract and can train
with `CriticLoss(lambda_local=0)` on the same weak pair labels.

## Tests and remaining execution gates

Run the new synthetic suite with the repository environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest tests/critic -q
```

The suite includes full-context fixture fitting, exact CPU continuation, frozen
encoder preservation, single generated-MIDI inference, all span-scale gradients,
no cross-sample mixing, mask/alignment failure cases and reference/bundle binding.
The tests use generated or documentation fixtures, never corpus checkpoints or
real music-quality evidence. Test snapshots/results are recorded in
[implementation closure](workstreams/2026-09-21/critic-wayfinder/implementation.md).

Before a real pilot: select/admit checkpoint/source, inventory role-eligible
TRAIN works/operators, freeze work/component splits, bound feature-bank size,
register the experiment, choose optimizer/stopping settings and admit native
memory/throughput/cost. Existing20works are development examples. The earlier
1000 RUB ceiling is not a measured cost for this richer model. No paid run,
real-data expansion, TEST access, changes training or PLL follows automatically.

## Selected cloud C3 compatibility and campaign

`music_critic.critic.c3_backbone.load_c3_relationgate_adapter` admits the exact
2026-09-18cloud x3+relation-gate source, with caller-pinned checkpoint SHA256/size
and default expected update80000. It checks envelope/model integrity and the
full original architecture/source contract, freezes weights, and returns the
ordinary multilevel adapter. It does not restore the old optimizer or admit the
separate CHANGES backbone.

`scripts/prepare_critic_sources.py` builds hash-bound TRAIN clips; explicit
generation-only inferred roles remain outside raw predictor input.
`scripts/run_critic_campaign.py` provides metadata readiness, frozen feature
preparation, native runtime probe and training, bound to a sealed experiment
record and config fingerprint. Shared feature caches retain source/data/weight
identities. Immutable checkpoints and history coexist with atomic best/latest
pointers. A terminal run needs a new attempt for continuation.

`scripts/run_critic_cloud.py` owns the bounded process, export attempt and own-VM
stop; deployment also requires an independent systemd deadline. See the
[cloud protocol](experiments/CRITIC_C3G_CLOUD_PROTOCOL_RU.md) for actual admission,
run IDs, cost limits and claim boundaries.
