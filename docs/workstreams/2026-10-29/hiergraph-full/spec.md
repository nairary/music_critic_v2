# HIERGRAPH full E3 baseline — implementation contract, 2026-10-29

User requests complete hierarchical model with H revision; NBFNet/Query2box
are deferred. Phase T08, isolated harmonic-event-pilot checkout, HEAD
287bf856643f9e555352e1661a7bf0563772178a plus existing WIP. Original main checkout,
legacy and completed E1–E5 source/results remain unchanged. No commit/push.
Implementation and synthetic/native engineering readiness are distinct from a
new real-data training/evaluation campaign and scientific quality.

## Behavior

Ordinary canonical MIDI-derived observations -> frozen C3 per-node features
-> trainable existing E3 -> global H and physical note-context h -> process P,
participants and directed relation candidates -> iterative bottom-up and
P->H/h->note feedback -> updated H/h/P spans, memberships and typed properties.
Every event retains identity and links to individual observations; no whole-work
mean is the sole route to distant events. Windows are local feature-computation
units, not a mandatory boundary on global H/P geometry. H and P spans can cross
original tile/halo bounds after global refinement. Retain lower raw observations
and uncertainty; f/F are independent typed attributes, not exclusive across fields.

## Interfaces and ownership

- full_data: pure whole-work raw assembly from sanitized canonical piece and
  existing C3 tile payloads; separate target views from native NumericTargets@3.
  Physical note IDs, exact rational origin/duration and source geometry remain
  routing/evidence. No analytic view, target field, mask or confidence in forward.
- full_model: FullGraphAnalyzer with constructor=HAnalyzer(E3), shared global
  attention over H, note states and h/P messages; explicit span refinement and
  refresh of h geometric participation candidates. Optional soft predicted
  property feedback uses prior-round distributions, never target labels.
- full_losses/full_metrics: match H/P outside forward, align h by physical note
  and matched H plus intervals, supervise process members and directed endpoints
  only when known. Many-to-many relation endpoint sets never become invented
  pairwise positive edges. Unsupported candidate shapes remain coverage misses.
- full_runtime and run_hiergraph_full.py: deterministic whole-work/view samples,
  applied optimizer updates, exact immutable checkpoint bindings and weight-only
  E3 warmstart, structured raw-only inference, explicit oracle diagnostics.

Raw work schema: tiles(tuple raw_dict), tile_spans/core_spans(Tx2 normalized
whole-work geometry), owned_time_indices(tuple Long exact-core-owned local time anchors),
note_index_by_tile(tuple Long localnote->globalnote), notes
(spans Nx2,pitch,track_index,is_percussion,is_grace), bar_spans Bx2, note_ids,
origin/duration Fraction, coverage. Labels are a distinct object. Raw schema
validation rejects unknown target-like keys. Capacity rejection is explicit;
never drop/truncate a work or targets silently. Nonzero origin is supported.

Prediction schema: H/P standard spans/object_logits/field_logits/states;
h spans/note_index/H_index/field_logits/states; participants P_index,
endpoint_level(H/h),endpoint_index,existence_logits (optional supported role
heads); relations typed logits with explicit directed source/target endpoint
sets. Inference serializes predicted rationalized coordinates separately from
exact observed timing; no claim of recovered original Fraction boundaries.

P begins with bounded raw bar-window proposals at multiple scales. Endpoint
sets for addressed relations are raw/prediction-derived. Singleton relation
candidates are initial baseline; unrepresented set-valued relations retain
candidate coverage exclusions. No role names are fabricated from untyped P
membership. Heads/vocabularies bind fit-only availability; unknown remains masked.

## Acceptance

1. Changing/removing target labels cannot alter forward inputs or candidates.
2. Stable physical identity for held/grace/percussion notes and unique owner
   across halos; exact canonical metadata retained; no gold-count candidate cap.
3. Distant H/P affects local states through individual global nodes; no universal
   averaging bottleneck. Context values retain source/event addressing.
4. At least two rounds update H/P geometry and h participation; process/property
   losses have nonzero gradients to H constructor and lower trainable states.
   Non-chord notes stay present. Technical reverse messages are not reverse
   semantic labels.
5. H/P are proposed without gold intervals, including crossing tile boundaries.
6. Partial masks, absent classes, view alternatives and multi-endpoint relations
   remain explicit; all-unknown draws perform no optimizer step.
7. Inference runs from raw observations/cache without numeric target shards.
8. CPU fixture training, checkpoint/resume parity, CLI and serialization verified.
9. Standards and Spec independent code reviews, scope inventories and relevant
   regression tests. No implementation pass constitutes model-quality evidence.

## Initial configuration and launch boundary

Frozen C3, width128, E3 warmstart selected40k optional; three global rounds,
no augmentation. Existing E3 remains trainable. Global attention operates on
individual H/P with chunking/configured bounds. Resource caps are explicit and
measured in a separately recorded native gate before any duration estimate.
Selection metric/budget for full real-data training must be bound to a fresh
ExperimentRecord; old 80k condition is not resumed or overwritten. Global TEST
remains closed. No new paid resources/LLM calls or corpus annotation.

## Implementation evidence

Implementation and synthetic integration are complete; see [implementation.md](implementation.md)
for current scope, limits, verification and separate deployment gate. No real-data
full-model training/evaluation, new GPU run or new musical-quality result is
established by this implementation. Existing terminal records are unchanged.
