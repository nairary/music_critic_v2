# Multiscale aspect critic v1 — implementation specification

22 September 2026. **Implemented and synthetically checked**;
[closure and test results](implementation.md). Not runtime-admitted and not
evidence of music-quality accuracy. The user accepts
the multilevel/aspect direction and adding localization. Numerical defaults below
are explicit implementation choices, not measured optima or a training budget.

Current main remains Phase9E; this is the requested bounded critic prototype
from Phase12. It does not declare the remaining roadmap phases complete.
Source reference: HEAD `287bf856643f9e555352e1661a7bf0563772178a`, with pre-existing
dirty/untracked work. Record the actual source tree fingerprint at implementation
and execution; this HEAD alone does not identify all inspected files.

## Problem and outcome

Existing theory classifiers do not provide four musical quality scores or
validated error locations. A beat-only scalar readout discards the requested
joint use of note/event, vertical and larger contextual representations.

Input: one ordinary raw MIDI/canonical piece, without original/reference,
intervention metadata, roles, voices, gold harmony or semantic segmentation.
Output: harmony, rhythm, melody and local-structure relative scores, explicit
weighted overall, and aspect-specific local intervention evidence. The last
output locates synthetic alteration signatures; it is not automatically a
validated musical-error map. Human musical-error labels remain unavailable.

Out of v1: backbone unfreezing, changes training, PLL, whole-form quality,
arrangement quality training, audio teacher, renderer dependence, GRPO and
new cloud runs. Preserve existing theory heads and checkpoint compatibility.

Authoritative contracts: [architecture](../../../ARCHITECTURE.md),
[data](../../../DATA_CONTRACT.md), [implementation plan](../../../IMPLEMENTATION_PLAN.md),
[roadmap](../../../ROADMAP.md), [logging](../../../EXPERIMENT_LOGGING.md),
[decisions](../../../DECISIONS.md). Background and verified source paths:
[multiscale design](../../../research/MULTISCALE_ASPECT_CRITIC_DESIGN_20260922_RU.md).

## Module ownership and public seams

Implement isolated modules under `src/music_critic/critic/`, with matching tests
under `tests/critic/`. Names below define responsibility, not a requirement to
create one file per class. Keep real-data drivers in the experiment layer.

| Public seam | Contract |
|---|---|
| `BackboneFeatureAdapter.extract(raw_batch)` | Returns a detached `CriticFeaturePyramid`; no targets argument; encoder eval mode and no encoder gradients |
| `CriticFeaturePyramid` | Per-level vectors, sample ownership, raw identities, rational times/durations, raw relations, availability and source/checkpoint/stage bindings |
| `MultiscaleAspectCritic.forward(pyramid)` | One input view independently yields contextual states, four global raw scores and local evidence logits/availability |
| `InterventionTargetBuilder.build(original, variant, verified_patch)` | Target-sidecar only; exact observable edits and support, separate pair-aspect mask and local-label mask; cannot add predictor nodes |
| `CriticLoss(output_pair, target_sidecar)` | Masked global ranking and local intervention classification, separate metrics/support counts; no theory labels |
| `AspectScorePolicy.apply(outputs, reference, weights)` | Versioned relative scales, declared scope, nonnegative explicit weights, overall and coverage |

Never treat IDs, filenames, provenance, original/edited flags, patch coordinates,
operator IDs or target masks as numeric model features. Raw identity is permitted
only for joins and alignment. Changing the target sidecar cannot change raw
features/topology or forward predictions in eval mode.

## Feature extraction and raw geometry

1. Support current main C2 and temporal sources through separately versioned
   adapters. C2 has no pretrained slice state: construct its slice input from
   notes/raw geometry and record that distinction. An isolated C3 worktree is
   not silently accepted as the main schema; admit its source separately.
2. Preserve local note features and the latest contextual/temporal note state;
   keep base track/SONG context with its earlier-stage tag. Temporal extraction
   retains the base output from the same forward, without a second base encode.
   Additive access must preserve existing forward outputs and state-dict keys.
3. Attack ownership and sounding overlap are different relations. Preserve raw
   onset→beat/bar ownership; note→bar owns the attack. Construct exact overlap
   incidence for sustained notes across metric intervals. Sounding is `[onset,end)`;
   releases and zero-duration/grace events have separate event channels.
4. Track events group simultaneous attacks as a set. Sort events by exact time,
   never interpret pitch-sorted simultaneous notes as a temporal voice. A merged
   raw track remains a polyphonic sequence. No gold parts/voices required.
5. Slice slots derive only from raw onset, offset and beat boundaries. Preserve
   repeated sounding/PC sets and their different metric/event context.
6. Cache each original and variant separately. Bind raw-content hash, source
   snapshot, checkpoint, adapter/schema, feature stage and dtype. Unknown/mismatched
   bindings fail closed. Cache frozen features, never newly learned context outputs.

## Concrete context/readout defaults

Common width128; three one-layer bidirectional GRUs, hidden64 per direction.
All new projections/GRUs/gates/readouts initialize independently and train jointly.
Existing backbone/theory parameters stay frozen and outside the optimizer.

- Project available inputs to width128. Set aggregation uses masked mean/max,
  explicit availability and log-count, followed by a learned projection. Empty
  sets have defined zero aggregates with availability=false, not learned targets.
- Shared track-event BiGRU processes each sample×raw-track independently.
  Preserve individual note vectors alongside the event context.
- Slice input combines starts/holds/releases, attached track-event context,
  existing slice/onset/beat features when present and raw metric timing. A second
  BiGRU processes the slice sequence separately for each sample.
- Pool refined slices into bars by exact covered duration; event contributions
  are a separately normalized channel so sustained notes are not counted as
  attacks. Combine with existing bar context; run the third BiGRU over bars.
  Form sliding complete 2/4/8-bar spans only where they exist, with availability.
- One cross-scale block has a fixed schedule: bar/span→slice, updated
  slice→note/track-event, then repool refined states upward and global context.
  Raw incidence determines edges. Normalize incoming messages per relation;
  use a sigmoid linear gate on receiver/message plus a learned relation embedding
  and a residual update. This is sequential, not two-hop propagation in one
  synchronous round. No unbounded all-to-all attention in v1.
- Each aspect has one learned query per available level for masked dot-product
  attention pooling. A masked softmax gate combines the level vectors; an MLP
  `128→32→1` yields its raw global score. All aspects may access all available
  levels. Attention weights are not musical-error probabilities.
- Local aspect heads `128→32→1` read post-context states: harmony on slices,
  melody on notes/events, rhythm on events and raw beat intervals, local structure
  on internal bar boundaries from adjacent contextual bars. A missing candidate
  is unavailable, not a fake note or an extrapolated label. Whole-form is unavailable.

Absent level masks exclude it from attention/gate denominators. Empty samples
return unavailable scores without NaNs. Rhythm requires usable events/metric
support; harmony requires pitched sounding support; melody requires pitched
events; local structure requires at least two adjacent supported bars. These
are computability gates, not evidence of competence. Report support and horizon.

## Localization targets and claim boundary

`local_intervention_evidence` means support for an artificial change detectable
from the input; logits/sigmoid values are not calibrated musical-error probability.
Expose `musical_error_location` as unavailable in v1, with reason `no_validated_targets`.
Do not present an untrained second head or attention heatmap as that output.

The existing generator's `support_ids` describe eligibility, not necessarily
actual edits. Build targets from verified before/after field differences and
patches; its `quality_preference=unreviewed` remains part of target provenance.

| Edit | Positive localization support in the edited view |
|---|---|
| Pitch edit | Actually changed surviving notes; sounding slots with positive overlap with those notes, within the labelled aspect |
| Time/duration edit | Union of old/new exact sounding support plus old/new attack positions; project onto independently built raw candidates |
| Deletion | Old support projected onto existing raw metric intervals; no deleted-note candidate or invented input node |
| Bar swap | Internal raw bar boundaries at the entrance, join and exit of the swap where such boundaries exist; this is intervention-boundary supervision |

Zero-duration targets project to existing event candidates only. No candidate
means masked target. Pitch-edit negative slice labels require complete sounding
incidence; unavailable joins remain masked. Contextual consequences outside the
patch are not labelled musically correct or incorrect.

With an exhaustive verified diff, unchanged candidates are legitimate negatives
for **intervention presence only**. Corresponding originals have no applied
intervention and supply negative presence examples. They are not globally
flawless music. Unknown/incomplete patch coverage cannot supply negative labels.
Unlabelled aspects are masked. Arrangement dropout remains diagnostic in v1.

This distinguishes detecting synthetic edits from identifying musical harm.
Later benign edits and independently labelled musical spans must evaluate that
gap; no high edit-localization metric establishes musical-error accuracy.

## Training, scoring and selection contracts

First split by work/known component, then generate variants. Keep all siblings
together. Separate head-held-out works from novel-to-backbone works and record
unknown pretraining overlap. Existing20works are development/feasibility only.
Operators needing melody/accompaniment roles apply only where generation has
explicit usable role provenance; do not assume every raw TRAIN work is eligible.
Those roles remain generation-side and never become model input.

Global pair loss per available aspect is `softplus(s_edited-s_original)`.
Local loss is BCE-with-logits on observed intervention-presence labels. Average
local loss first within available positive/negative classes and candidate type,
then work/aspect; omit an absent class, do not fabricate it. Average pair loss
by work/aspect as well. Default `L=L_pair+0.25*L_local`; lambda is versioned and
configurable, with zero as a required control. This default is not an empirical
claim. If neither term has support, skip the optimizer update and record it.

Swap pair order with targets; the predictor never receives pair identity or
the reference. Shared context changes may affect all aspects even when only
one aspect has direct pair supervision. Optimizer owns all new modules; freeze
encoder mode as well as weights. Implement deterministic seeded sampling,
finite-gradient checks/clipping and resumable critic-only training on feature
banks. Learning rate/batch size/step limits are explicit run configuration,
not inferred from old backbone schedules. Persist optimizer, sampler, RNG,
bindings and selection progress; fail closed on mismatched resume.

Select on predeclared head-dev macro-aspect pair accuracy, averaged by work;
ties within fixed tolerance count one-half. Tie-break by lower pair loss, then
earlier update. An aspect without support is unavailable and explicitly omitted;
the declared selection aspect set cannot silently change mid-run. Report local
precision/recall and interval overlap on the appropriate raw support separately.
Raw-summary, beat-only and multilevel-pooling controls use the same split/labels.
Do not select by train loss or edit-location accuracy alone.

After selection, freeze critic and reference scales. Per-aspect percentile is
relative to a versioned work-balanced TRAIN reference bank, with the same horizon
and support policy. Empirical CDF uses midpoint tie handling; the run manifest
specifies included original/variant types and weights. No bank → raw scores only,
normalized overall unavailable. Reference values are not human-quality calibration.
Weights default equal; reject negative/nonfinite/all-zero weights. Compute weighted
overall on available aspects and expose weighted coverage. Pair comparison uses
common aspects/horizon; no silent renormalization across unequal support.

## Observable engineering acceptance

1. One ordinary unlabelled MIDI/fixture can be scored without its original,
   generation metadata or source roles. Changing sidecar labels cannot change
   predictor input or eval-mode output; missing targets produce no negative loss.
2. Original/variant embeddings differ when raw content differs; cached versus
   fresh features agree within a declared tolerance. Binding changes invalidate
   caches/resume. No old state-dict keys or theory outputs change under the same
   eval input merely by enabling feature export.
3. No cross-sample recurrence/messages under batching. Padding/empty samples
   cannot change another sample's output. Simultaneous-note set order and raw
   track-row permutation at fixed features do not alter the readout; this is not
   a claim of end-to-end invariance of the pretrained track-positional encoder.
4. Fixtures cover held notes crossing bars, release at onset, grace events,
   delayed notes, deleted events, incomplete diff and bar swaps. Deleted notes
   never reappear as predictor candidates. All interval joins use exact time.
5. Every enabled new branch receives finite gradients on a suitable synthetic
   batch; encoder parameters remain byte-identical with no gradients and no
   train-mode dropout/state changes. Target-masked local positions have no direct
   local-loss gradient. Completely unsupported batches do not step optimizer.
6. A tiny controlled synthetic pair/localization task can be fitted, and reverse
   pair presentation with corresponding labels preserves loss. This tests
   mechanics only, not real musical judgment. Resume reproduces the next samples,
   losses and updates within documented device determinism limits.
7. Missing aspects/reference bank, zero support, invalid weights and horizon
   mismatch have explicit behavior; no NaN or apparently complete score.
8. Output labels distinguish aspect ranking, synthetic intervention evidence and
   unavailable validated musical-error locations. Training never upgrades these
   claims automatically based on synthetic accuracy.

## Delivery sequence and readiness

Implement in behavioral slices: pyramid/export and cache → context/global
readout → local heads/targets/loss → score policy and trainer/resume → one-MIDI
inference/export plus controls. Tests accompany each slice. Existing preview
integration may consume the exported values and exact spans; rendering is not
an inference prerequisite. Follow repository implementation/review workflows.

**Engineering implementation is complete** under this spec; see the closure above.
Routine layer packaging can vary while preserving these observable contracts.
There is no claim that the architecture will outperform simpler controls.

**Native/data execution still requires:** actual checkpoint/source binding,
eligible work/operator inventory and split, a bounded feature-bank expansion,
explicit optimizer/stopping configuration, hardware memory/throughput admission
and a registered run with a concrete compute cap. Existing1000 RUB pilot ceiling
is not proof the richer model fits; reported7300 RUB is not a live account check.

**Scientific completion requires:** held-out work/family evidence, separate
localization evaluation, controls and independent human judgments. Human labels
are not a prerequisite for synthetic implementation or the initial weak pilot.
Keep final human evaluation separate from any later human calibration/training.
Follow the existing experiment registry before real-data processing/evaluation;
this specification creates no run, spends no money and does not unlock TEST.

Legacy is not inspected/reused. The initial specification was documentation-only;
the subsequent authorized implementation and synthetic tests are recorded in
[implementation closure](implementation.md). No real-data experiment was launched.
