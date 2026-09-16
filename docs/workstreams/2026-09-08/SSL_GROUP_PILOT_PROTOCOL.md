# Task 03 — random note versus raw beat pitch masking

2026-09-08. Engineering implementation with review-002 corrections and a separately specified
scientific pilot. Worktree: `/Users/aavalsamakin/.codex/worktrees/582d/music-critic-v2`.
Bootstrap manifest SHA-256:
`9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf`.
All 752 source files were verified before installation and after copy. The
source is HEAD `da8c9dcd999a4b522f87f9a20bf3506ad26497b7` plus the coordinator's
recorded dirty snapshot. Inherited changes are not this task's implementation.

## Implemented scope

The opt-in adapter is `src/music_critic/ssl/group_masking_pilot.py`;
`group_masking_fixture.py` provides only generated engineering batches, and
`group_masking_cli.py` / `scripts/run_group_masking_pilot.py` provide the
executable two-arm fixture gate. Configuration:
`configs/ssl_group/random_vs_beat.json`.

This reuses existing Phase 7A losses and Phase 8A policies through WS2. It adds
no encoder, EMA, latent loss, Phase 8B multilevel head, voice edge/head,
harmonic fusion, modifier, preference or degradation training. `multilevel.py`
already has detached hierarchy targets; it is not needed to isolate masking.
The current model's full-view latent target is shared and stop-gradient.
Representation reconstruction is not PLL or music-quality scoring.

The two policies are `independent_note_pitch` and `beat_pitch_descendants`.
Independent masking dispatches exactly to the old Phase 7A binding. Beat units
follow raw beat→onset→note **start** membership; sounding notes started earlier
are not hidden merely because they overlap the beat. No gold chord/phrase
boundary is consumed. Both policies must be available on every scheduled
batch before either can update. Unavailable beat groups fail explicitly;
there is no fallback or arm-specific sample replacement.

Requested density is 0.30 in both arms. The raw beat policy has indivisible
units, so realized density need not match the random arm. This is the
predeclared policy comparison, not an exactly density-matched causal claim.
Each update reports both plans, hidden notes/eligible notes, requested/realized
density, hidden/candidate units, collateral note/track counts and plan hashes.
Both arms use the same raw SSL and theory replay schedule and exact graph,
sidecar and actual class-weight tensor bindings. Resume schema 1.1 binds compact
batch draw identities, per-piece note counts, node/edge counts and per-head
source target populations together with a content digest of the complete
inputs. Schema 1.0 checkpoints remain retained historical artifacts and cannot
resume under the changed accounting contract.

## Feature audit and retained information

Existing `hierarchy_leakage.py` classifies the pinned 68 raw registry fields.
For selected notes the overlay masks pitch, pitch_class, octave and
track_relative_pitch, including availability contributions. For unselected
peers in affected tracks it masks track_relative_pitch; for owner tracks it
masks mean_pitch, pitch_std, min_pitch and max_pitch, including availability.
Collateral rows are not reconstruction targets. Raw graphs remain unchanged.

The other 60 registry fields, timing, duration, velocity, ownership and topology
remain visible. Simultaneous same-track temporal ordering can expose relative
pitch rank. Therefore the claim is closure of the audited exact pitch feature
dependencies, not hidden full notes or feature-complete masking. Tests exercise
both actual WS2 policies with closed-slot mutations and coherent canonical pitch
rebuilds; online embeddings remain exact while full targets change.

## C2 artifact bridge and ownership

The new `load_model_only_theory_warmstart` entrypoint in `theory_warmstart.py`
uses the existing restricted CPU loader (`weights_only=True`) and then the
same complete WS2 state validator/mapping. It requires the full model-only
schema and exact expected tensor-state fingerprint. Artifact hashes, size,
contract and every tensor are checked before destination mutation. Historical
training-envelope acceptance is not extended. The old synthetic entrypoint
continues rejecting non-synthetic envelopes.

The returned result explicitly carries verified artifact evidence. The adapter
checks its SHA, size and state against the supplied C2 binding and verifies the
current theory model still has the transferred source weights. The CLI compares
**full initial harness fingerprints** across arms, including fresh SSL state,
and constructs independently owned arms under the same initialization seed.

Bound C2 artifact from `EXP-POP-TRANSFER-EXPORT-001`:

- locator: `outputs/pop909_transfer/20260908/export-001/c2-model-only.pt`;
- SHA-256: `a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797`;
- size: 14,844,299 bytes;
- tensor state: `82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39`.

This task verified the registered bytes of `result.json` and
`c2-provenance.json` before using their identities. It did not deserialize the
real checkpoint. The future loader verifies its bytes at execution. The C2
training identity comes from the registered parent evidence; the model-only
payload alone cannot authenticate training history or checkpoint-selection
update. Existing limitations of C2 provenance remain.

| Role | State and optimizer ownership |
| --- | --- |
| Transferred/shared | local encoder, pooling, Transformer and fusion; 470 parameter tensors / 3,060,224 scalars; identical module/Parameter objects across the two objective branches within one arm |
| Retained theory | separate BiGRU and all 18 original heads; 100 tensors / 601,712 scalars |
| Fresh SSL | feature mask token, contextual representation decoder, bar/song projectors/predictors; 35 tensors / 183,296 scalars |
| Inactive | SSL container's inherited unused task/reconstruction heads; 81 tensors / 324,357 scalars; frozen and excluded |

The optimizer owns exactly 605 unique active tensors (3,845,232 scalars), not a
concatenation of both branches. Exact per-group keys/fingerprints and parameter
names are saved in each arm binding. Adam is fresh, LR 3e-4, weight decay zero,
no scheduler, AMP or gradient clipping. The combined objective is
`0.7 * existing_SSL + 0.3 * existing_corrected_theory`; corrected theory retains
its historical primary plus 0.25 auxiliary aggregation. These coefficients are
predeclared pilot settings, not tuned evidence. Missing replay targets fail the
paired pilot instead of silently changing its objective. The reusable harness
continues supporting missing targets outside this stricter runner.

## Schedule, compute and resume

The adapter requires a complete bounded ordered tuple of actual batch hashes.
`step_from_factory` denies every split except TRAIN before invoking a reader.
It verifies graph bytes, ordered identities, sidecars and weight tensors against
the next scheduled entry. It does not establish that a caller's TRAIN assertion
is authoritative; a production data factory must attest full split/component
closure before use. No factory scans/discovers a corpus here.

A Transformer module hook shared by both branches measures actual backbone
passes. Mixed updates require exactly two SSL passes (full target, masked
online) plus one theory pass. Decoder remask views do not add encoder passes.
Theory-only and SSL-only paths count one and two passes respectively and clear
stale gradients before each step. Each row retains head losses/available counts
and note/bar/song variance, norms, zero norms and pairwise-cosine diagnostics.
These are per-batch training diagnostics, not held-out aggregate metrics.

Mask epoch and Torch dropout domain use the zero-based applied update. Torch
RNG is scoped, separately seeded by a fixed SHA-256 domain identical across
arms, and restored after each step. This is this adapter's declared RNG policy;
it is not a continuation of C2's historical RNG or optimizer. Schedule, source,
policy, complete fresh initial state, parameters, optimizer, numerical runtime,
completed prefix and rows are bound into a versioned full resume envelope.

Restricted resume validates artifact SHA/size, payload, complete tensors,
shared aliases, finite Adam slots, nonnegative second moments, per-parameter
step/gradient history and a strict successful-row schema before application.
Draw counts, both mask policies, note densities/units, head populations and
aggregates, active paths, forward counts, loss aggregation and diagnostics must
agree with the config and schedule-bound compact evidence. Valid, masked and
unsupported rows remain distinct; their per-head sum equals the bound source
target population. This establishes structural and arithmetic consistency.
It cannot recover exact selections, target classifications or measured losses
from opaque digests; those claims require the original artifact or replay. A zero-update resume must match initial weights exactly.
Actual numerical runtime is rechecked before update, before and after Adam,
before checkpoint publication and before resume application. CUDA bindings
include `CUBLAS_WORKSPACE_CONFIG`; matmul precision and current autocast
enabled/dtype state are also explicit. The check does not initialize CUDA.
Model tensors and all Adam slots must remain finite after a normally returned
Adam call before an update is confirmed. Finite gradients alone are insufficient
because their squared FP32 moments can overflow. No clipping or loss change is
introduced. The original 1.0 dropout seed domain remains pinned explicitly.

Checkpoint paths are exclusively published and never overwritten. A failed
update poisons that instance; recreate and resume the last successful checkpoint.
A durable `attempts.jsonl` receipt precedes each factory/update and records
progress, completed encoder forwards, failure stage and optimizer outcome.
`not_applied` means the optimizer was never entered, `applied` means the call
returned normally, and `unknown` means an entered call raised or did not return.
An optimizer can mutate state and then raise, so `unknown` is not zero work or
rollback. Even `applied` is not a confirmed successful update until post-step
validation and accounting pass. `execution.jsonl` contains returned successful
rows; failure reports separately count confirmed updates, observed forwards and
optimizer outcomes, including work missing from that successful-row journal.
Flush/fsync receipts improve retained evidence but cannot make a process kill
or hardware failure atomic with device execution; external timeout logs and
any last unknown receipt must remain part of the coordinator evidence.
CPU exact replay is tested. CUDA uses the existing deterministic evidence runtime
and still needs independent confirmation on the actual host. No cross-host or
cross-version bitwise promise is made. No post-selection checkpoint promotion
is implemented.

## Next executable CUDA engineering gate

Planned record: `EXP-SSL-GROUP-CUDA-001`. It is **not launched**. The gate uses
the bound C2 weights and generated batches only. It performs two updates per
arm plus a one-update interruption/resume replay for each arm: eight actual
updates, 24 encoder forwards, 16 SSL draws and 16 theory draws in total. The
paired comparison itself accounts for four updates; replay overhead is separate.
One process, one GPU `cuda:0`, one CPU thread, FP32, no preprocessing, a 600-second
pre-step wall check and fixed tiny graphs. This is not a hard timeout: a single
step, checkpoint load/save or stalled process can exceed 600 seconds. The
coordinator must supply an external hard timeout/resource wrapper and preserve
its termination outcome alongside the attempt journal. Actual timings and peak allocated/reserved
VRAM are recorded; no throughput or corpus-memory conclusion follows.

From the reviewed worktree on the assigned host, with existing project
dependencies and the immutable C2 export copied/located under its verified hash:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_pilot.py \
  --config configs/ssl_group/random_vs_beat.json \
  --mode cuda-fixture \
  --source-checkpoint <explicit-root>/outputs/pop909_transfer/20260908/export-001/c2-model-only.pt \
  --experiment-record docs/experiments/records/EXP-SSL-GROUP-CUDA-001.json \
  --output outputs/ssl_group/EXP-SSL-GROUP-CUDA-001
```

The explicit artifact root is the only host-specific substitution. The CLI
requires a matching planned engineering record with exact config/source/code
fingerprints before opening the checkpoint or initializing CUDA. It retains
plan, durable attempt receipts, successful execution JSONL, arm ownership/bindings, immutable checkpoints and either
result or failure. The coordinator must retain logs and finalize every launched
record before another substantial run; the CLI does not mutate tracked records.
If integration changes source, review and reseal the still-planned record before
execution; after a failed launch use a new attempt ID and output root.

## Predeclared first scientific comparison, blocked on bindings

Planned record: `EXP-SSL-GROUP-PILOT-001`, TRAIN/VALIDATION only, TEST closed.
First use the existing verified Dilemmadata TRAIN projection once its bindings
are supplied. No PDMX dependency is needed. Two arms start from the same C2,
seed17, 200 applied updates per arm, batch2 SSL plus batch2 theory replay,
identical predeclared schedules and replay coefficients. No augmentation is
introduced; bind the existing raw projection explicitly. Limit each batch to
8,192 notes, 20,000 nodes and 200,000 edges; no retained corpus cache or workers.
Bounded graph hashes are computed on CPU before transfer. Rejected samples must
be resolved by a shared pre-run eligibility manifest, never replaced per arm.

Primary endpoints are the unchanged 18 Dilemmadata theory tasks on one fixed
VALIDATION membership, relative to original C2 and the random-mask control,
with per-task coverage, event accuracy and observed-class macro-F1. Also report
fixed-mask reconstruction and aggregate variance/collapse diagnostics on the
same held-out raw pieces under **both** policies. Weighted training SSL loss
alone cannot select a winner. All 18 tasks and rare/absent classes remain
reported; no POP labels, TEST, or dropped difficult classes tune this pilot.

Selection is descriptive at the fixed 200-update endpoint: no best-checkpoint
search or architecture promotion from one seed. A separately bound, identical
100-update theory-only recovery schedule for each endpoint should measure skill
recovery, with the same endpoint metrics and added compute reported separately.
That recovery-stage transition/evaluator is not implemented by this fixture CLI.
Neither arm is presented as broadly superior without matched repeated evidence.

Concrete missing bindings in this package: raw TRAIN index/source artifacts;
full component/split identity; paired policy eligibility; ordered SSL and replay
schedule; replay sidecar manifest and frozen class-weight artifact; fixed
VALIDATION membership; original-C2 and endpoint evaluator integration and
aggregation; the common recovery schedule/stage binding. These are explicit nulls
in the config, not claims that project artifacts do not exist elsewhere. The
in-memory adapter's replay tests do not make this a production data factory.

## PDMX readiness and voice separation

The existing source/window/cache pipeline is synthetic-tested foundation.
Production `SourceManifest` requires release identity/hash and license-artifact
hash; it does not authenticate declarations. Exact source hashes and supplied
lineage group records before filtering/windows. Complete split metadata is
required; cross-corpus linkage is still represented as unknown. The production
report deliberately returns `production_ready=False` and no measured throughput
or corpus distribution. Only a synthetic PDMX configuration is supplied.

Missing release/license authentication, permitted source selection, duplicate/
lineage completeness, cross-corpus overlap evidence, full split bindings,
source-schema/resource acceptance and objective eligibility prevent readiness
claims. No download, whole-corpus scan, cache or preprocessing was performed.

Voice continuation is the separate [candidate/target contract](VOICE_CONTINUATION_CONTRACT.md).
Crossing/polyphonic fixtures prove mapping mechanics, not annotation coverage.
AN part/voice scope is lost in canonical scalar voice metadata; a trustworthy
annotation importer must preserve source scope, ties/repair evidence and masks.
No voice relation is fed into either SSL arm. Legacy was not inspected or reused.
