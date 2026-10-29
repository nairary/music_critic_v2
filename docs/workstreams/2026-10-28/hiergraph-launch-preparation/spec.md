# HIERGRAPH E1–E5 launch preparation

User requests complete preparation, then readiness report before training.
Opt-in package experiments/hierarchical_graph_analyzer, accepted release003
projection and frozen C3 features. No scientific training, TEST, publication,
new paid resources or legacy execution. Work only harmonic-event-pilot; main
readonly. Existing declared dependencies may be explicitly installed in isolated
environment. Preserve unrelated WIP. Root owns integration/trainer/config/docs.

## Shared interfaces v1

A raw model sample is one tile with unique core ownership and halo plus full-work
coarse context. Exact Fraction times/physical note IDs stay in metadata. Tensor
raw dict contains no targets, availability or analytical view IDs:
- x float[N,266]: frozenC3 state(s)+raw features, finite.
- node_type long[N]:0..8=note,onset,beat,bar,track,song,slice,fragment,global_bar.
- edge_index long[2,E],edge_type long[E], raw schema <=64.
- node_time float[N,2]: relative to tile extent; global context may be outside[0,1].
- time_index long[T]: raw boundary states; repeated indices allowed.
- time_positions float[T]: strictly increasing[0,1], includes0/1.
- bank_indices long[B,2]: raw-grid interval endpoint indices; start<end.
- pool_mask bool[N]: raw eligible note-fragments and time states for E5.
Initially no batch dimension; trainer streams deterministic work/tile samples.

HAnalyzer(kind,head_sizes,input_width=266,hidden=128,rounds=3,max_events=64,
edge_types=64,node_types=9). kind E1..E5. forward(raw) returns spans float[K,2]
(differentiable normalized start<end within[0,1]),object_logits float[K],
field_logits dict[name,float[K,C]],states float[K,hidden]; optional boundary_logits
and assignment diagnostics allowed. E1 may emit B bank candidates, all arms
share decoded raw-duration budget (initial4 per quarter, capped bycapacity).
E2 segmentation with differentiable boundary learning; E3 anchor-free endpoints;
E4 learned event-query cross-attention; E5 sigmoid graph assignments, normalized
pergroup, followed by independent interval head (not min/max pooled note times).
Shared typed raw updater and downstream H/refiner across all arms. Preserve raw
residual channels and nonchord notes. E5 S is latent, not semantic membership.
Physical held notes can participate in multiple fragments/H. No goldcountsetsK.

Targets dict: ids list[str],spans float[M,2] normalized,fields
{name:{values long[M],mask bool[M]}}, optional known_negative_spans[L,2] emptybydefault.
Exacttime/family/work/view/scope/coverage are target metadata. Match only compatible
view/scope, never corpusIDlookup. Separate target-independent prediction from
assignment. Training matching allows lowIoU pairs for localization; metrics use
predeclared overlapthreshold. Unknown/unmatched never automatically background.
compute_loss(pred,targets,weights=None) returns dict loss/components/support/matches;
field-wise maskedCE, matched boundaryloss and licensed objectnessBCE. No-label
loss differentiablezero. Positive-only objectness is uncalibrated. Any budget or
dedup regularizer is explicit modelprior, not negative labels.
Metrics: knownHrecovery with one-to-one matching and raw-duration budget,
jointH+F,conditionalmacroF1,boundaryerror,coverage/capacity; familymacroprimary.
Precision only in proven complete scopes. Evaluation thresholds fixedbeforetraining.

Data agent owns data.py/overlay.py, scripts/prepare_hiergraph_overlay.py, associated
tests and registered CPU export. Model agent owns model.py and modeltests.
Supervision agent owns matching/losses/metrics and tests. Root owns init, raw
cache/C3/runtime/trainer/configs/docs and finalintegration. Coordinateinterfacechanges.

## Readiness checks

Target-blindforward, exactboundaries/heldnotes, empty/point/padding, cross-tile
ownership, permutations, maskedgradzero, H/Fgrad→E5S, commonheads/budget,
checkpoint/config/data hashes and deterministic resume. Genuine PyTorch runtime;
no module stubs. Native CPU/CUDA synthetic tests and registered bounded real
forward/backward gate without optimizer updates. Record measured resources,
sourcebindings and limitations; do not declare neuralquality from mechanics.

Refresh all67 TRAIN targets with15acceptedrelease003bundles, unchanged44/10/13
family split; fit-only vocabulary, support-gated Hheads and rawcoverage. First
comparison frozenC3/H/F; P/feedback/NBF/Query2box/liveC3 later. Transposition must
transform rawnotes plusabsolute targets; never shift cachedC3vectors. Configure
it explicitly and identically in all arms before training.

## Final numerical preparation contract — 2026-10-28

The first pilot uses seed17 for each E1–E5, then independently planned paired
17/42/73 replications. It uses the same frozen C3 cache, 42 support-gated H heads,
hidden128, three raw and H update rounds, max64 learned event slots and identical
work→tile→view sampling. E1 scores its declared raw span bank; E3 uses every raw
anchor. Prediction decoding is capped at min(64,4×core_duration_qn); known-H
recovery uses IoU≥0.5 and one-to-one maximum-cardinality matching. Constructor
packages have different parameter/compute costs; they are not compute-matched.

Targets are start-owned by a unique raw core and must fit the halo. Matching
uses all halo predictions during training; only decoded evaluation filters by
core start. Missing raw tiles and owned H outside the halo remain unrecovered
in the primary denominator. Zero-duration H is a separate unsupported-domain
count. Raw note points remain input nodes. Complete-scope precision is absent
unless a source explicitly licenses complete annotation; unmatched predictions
are never automatic negatives. Positive objectness is uncalibrated.

E2 additionally uses positive-only boundary BCE at exact canonical H endpoints
found on the raw grid. Indices are computed by Fraction equality outside forward;
unknown/off-grid locations are not negative. This explicitly repairs the zero-cut
case where interval loss cannot create the first discrete cut. E2 gradient tests
include that case. Each other constructor learns its geometry through matched
localization. This is a comparison of constructor packages, including this
mechanism-specific supervision, rather than an isolated same-loss operator swap.

Loss is matched smooth-L1 endpoints plus known-positive/explicit-negative
objectness and mean of available field CE terms. E2 adds its boundary term.
Field averaging gives each eligible output equal weight; the 30 active PC bins
therefore have greater combined weight than one function head. Inspect H.function
and semantic-group metrics separately before tuning this weighting. No quality
claim follows from passing this preparation.

All first-pilot conditions use augmentation=none. A full transposition orbit is
an explicitly separate future matched ablation: shift raw pitches and supported
absolute labels together, mask unsupported transforms, recompute C3 states,
retain source family splits and do not count copies as independent works.
Cached C3 vectors must never be pitch-shifted directly.

Native C3 cannot encode some raw source measure geometry after sanitization.
Only the known canonical validator rejection is an explicit geometry exclusion;
other graph errors, encoder failures and nonfinite states fail the attempt.
Exact core/halo bounds and reasons are retained. Missing owned global-bar states
are reported, not fabricated. This is a launch coverage limitation, not a repair
to the dataset or proof of support for every MIDI.

Training is opt-in and requires a fresh planned record bound to exact mode,
config/source/target/cache digests, condition/seed, output and integer wall cap.
Checkpoint selection is earliest maximum inner-selection family-macro known-H
recovery. Evaluate exploratory dev only on that selected checkpoint; diagnostic
oracle-H interpretation is separate. Persist each completed update before the
next, full optimizer/RNG state, attempted/applied/skipped counts and exposure hash.
A new attempt or retry never overwrites terminal evidence. No training is launched
by writing this specification.

RawGraphProfile admits up to64 observed tracks for direct C3 graph encoding.
It is a separate validated raw profile; shared PLL ContextProfile/Codec/Event
remain unchanged at16. Native cache004 includes all67 works.
