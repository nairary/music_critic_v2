# Phase 8B.2A compute-matched comparison and downstream transfer protocol

## Status and claim boundary

Phase 8B.2A implements `Phase8B2ComparisonProtocol@1.0.0`, deterministic
matrix planning, official-engine schedule/transfer bindings, validation-only
selection, held-out test locking, piece-level statistical aggregation, bounded
anti-collapse diagnostics, immutable artifact helpers, and CPU acceptance. It
does not contain production training results and makes no claim that one SSL
variant is better than another or better than supervised scratch.

No PDMX or Dilemmadata adapter, PLL, pseudo-likelihood, fragility/preference/
quality score, curriculum masking, EMA teacher, theory input, synthetic chord
note, graph-schema change, or ontology change is part of this phase. Phase 10
may supply PDMX indices/caches to this same protocol without changing its
comparison semantics.

## Variants and analyses

The exact SSL variants are:

1. `phase7a_control`;
2. `phase8a_mask_only`;
3. `onset_latent`;
4. `beat_latent`;
5. `hierarchy_bar_latent`;
6. `track_latent`;
7. `multilevel_equal`.

`natural_schedule` retains one view for Phase 7A and each single-level
objective and four views for mask-only and equal-weight. It pairs sample order,
optimizer-step budget, initialization, and seeds, but is explicitly a
secondary compute-unmatched diagnostic.

`encoder_forward_matched` is primary. Its default budget is 12 actual encoder
forwards per logical update. The Phase 7A and Phase 8A controls perform two
encoder calls per policy view, so they receive six deterministic views. The
Phase 8B.1 latent objectives perform three calls per view—the existing online
and detached target calls plus the latent-family full-view call—so they receive
four views. Repeated views use independent seed domains over the same raw
batch. The existing family-global numerator/eligible-denominator aggregation
remains unchanged; the comparison path performs no hidden loss
renormalization.

Every cell reports logical updates, policy views, encoder forwards, raw samples,
nodes, edges, eligible objective rows, applied/skipped updates, wall time, and
CUDA allocated/reserved peaks when a real CUDA device is used. Matched-mode
validation fails closed if forward, raw-exposure, or update budgets differ.

## Paired schedules and leakage boundary

Each base seed derives independent named domains for model initialization, SSL
data order, SSL mask planning, downstream initialization, downstream data
order, and piece bootstrap. Derivation depends only on the base seed, domain,
and explicit coordinates—not launch order. Plans and cells are canonically
sorted, so permuting variant or seed launch order leaves their artifacts
unchanged.

The official SSL data boundary remains `SSLBatch`: raw graph, dataset/piece
identity sidecars used for deterministic scheduling, and aggregate raw graph
counts. It has no target tensor, provenance, confidence, or evaluation label.
The Phase 8B.2A leakage contract rejects supervision-shaped model inputs and
requires target changed/removed/replaced mutation evidence to preserve plans,
logits, losses, gradients, checkpoints, and transferred encoder fingerprints
exactly.

Actual runs bind initial encoder state, raw sample schedule, fixed validation
membership, downstream schedule, and final transferred encoder fingerprints.
The Phase 8B.1 checkpoint binding now includes the optional comparison
schedule. Changing its protocol fingerprint, view seeds, data/model seed, or
sample schedule invalidates resume before checkpoint application.

## Downstream transfer

The primary downstream architecture is the existing hierarchical supervised
baseline. Feature-only and local-GNN scratch remain configurable, but a
pretrained hierarchy export requires the hierarchical model contract.

- `frozen_probe` failure-atomically loads only the accepted representation
  prefixes, freezes all loaded encoder parameters, excludes them from AdamW,
  and verifies the final encoder state bit-exactly.
- `full_finetune` failure-atomically loads the same encoder parameters and
  leaves them trainable.
- `supervised_scratch` builds the same architecture and fresh heads from the
  paired downstream initialization domain without an SSL export.

All task heads and optimizers are newly created. SSL decoders, latent heads,
optimizer, scheduler, scaler, and RNG state are never transferred. Encoder
exports and their SHA-256, source SSL checkpoint SHA-256, protocol fingerprint,
loaded/fresh parameter manifests, optimizer membership, and before/after
encoder fingerprints are recorded.

Only the current `ACTIVE_TASK_IDS` fully supervised model-ready heads are
eligible. Source-native HookTheory and POP909-CL heads remain isolated by
dataset, task family, and encoding kind. The protocol excludes
`theory.local_key.mode`, `theory.chord.borrowed`,
`pop909_cl.chord.boundary`, `pop909_cl.chord.no_chord`, and every PU/open-
vocabulary target.

## Evaluation and selection

Downstream validation uses the existing candidate-first evaluator and train-
only priors. Evaluation contract `1.2.0` adds all-negative multilabel
prediction counts and exact per-label average precision. AP is computed from
CPU scalar score groups, with ties handled as one threshold; no prediction
tensor is retained across batches and no batch-average approximation is used.

Metrics stay keyed by dataset and task. Categorical tasks retain eligible
rows, support, accuracy, balanced accuracy, macro/micro F1, NLL, train-prior
baseline, and model-minus-baseline values. Multilabel tasks retain per-label
support, micro/macro F1 and undefined reasons, prevalence baseline, exact
match, BCE/NLL, and all-negative counts. No cross-dataset or cross-encoding
global score is emitted.

The versioned primary endpoints are the declared HookTheory and POP909-CL
dataset macro summaries. Selection is validation-only, using mean dataset rank
and then, in order, lower validation NLL, lower encoder-forward count, and
lexicographic `variant_id`. Anti-collapse diagnostics never participate.

## Held-out test lock

Test is unavailable to ordinary comparison/evaluation configuration. A test
authorization is created before inference only when all of these hold:

- a valid validation-selection artifact exists and says test was unused;
- its protocol fingerprint matches;
- it selects exactly one checkpoint/variant per declared scope;
- acknowledgement is explicit;
- the output directory does not exist;
- the test membership fingerprint is already known and recorded;
- the locked experiment identity has not been consumed.

An exclusive sibling marker consumes the identity immediately before the new
output directory is created. Reuse, pre-existing output, missing selection,
wrong protocol, missing acknowledgement, multiple selections, or missing test
membership fails before inference. Unit acceptance exercises negative paths
without unlocking a real test split.

## Statistics and diagnostics

Production presets require at least three paired seeds; the paper preset uses
five. Statistical summaries contain per-seed mean/median, between-seed SD,
paired deltas against `supervised_scratch` and `phase7a_control`, and a
deterministic paired bootstrap over independent piece IDs. Fewer than two
pieces or seeds yields a structured unavailable reason. Bounded synthetic
acceptance never emits a significance claim or scientific p-value.

Transferred-encoder diagnostics report representation variance, effective
rank from singular values, adjacent-row oversmoothing cosine, zero norms, and
single-note perturbation deltas for note/onset/beat/bar/song. They form no
N-by-N matrix and retain no prediction tensor after a batch.

## Artifacts and aggregation

The artifact contract is `1.0.0`. A complete experiment has:

- `comparison_protocol.json`;
- `run_manifest.json`;
- `ssl_training_metrics.jsonl`;
- `ssl_checkpoint_evidence.json`;
- `transfer_evidence.json`;
- `downstream_metrics.json`;
- `validation_selection.json`;
- `statistical_summary.json`;
- `compute_accounting.json`;
- optional `test_metrics.json`;
- `final_comparison_report.json`.

JSON/JSONL creation is atomic and immutable. Production evidence records exact
clean git SHA, environment, Python/PyTorch/PyG/CUDA versions, concrete device,
all seed domains, data/cache/split/membership identities, protocol and contract
fingerprints, checkpoint SHA-256, and compute counters. Aggregation rejects
incomplete or duplicate cells, stale artifact fingerprints, mixed protocols,
data bindings, initial encoders, natural/matched modes, and unauthorized test
access.

SSL and downstream work reuse the official epoch-boundary checkpoint engines.
Those checkpoints bind the protocol/transfer runtime, full optimizer/
scheduler/scaler/RNG state, and metric journal. A changed protocol cannot
resume; existing failure-atomic load and journal recovery remain authoritative.

## CLI and presets

Dry-run is the default and performs no writes or training:

```bash
PYTHONPATH=src python -m music_critic.experiments.phase8b2.run \
  comparison=bounded_acceptance action=plan
```

Registered presets are `bounded_acceptance`, `production_pilot`,
`production_paper`, and `natural_schedule_diagnostic`. Pilot uses three seeds;
paper uses five; natural schedule is never relabelled compute matched.
Production paths are explicit and absolute; none are hard-coded:

```bash
PYTHONPATH=src python -m music_critic.experiments.phase8b2.run \
  comparison=production_pilot action=plan \
  data.index_paths='[/data/hook.index.json,/data/pop.index.json]' \
  data.cache_roots='[/data/hook-cache,/data/pop-cache]' \
  data.split_manifest=/data/global.split.json \
  data.index_fingerprints='{hooktheory:SHA,pop909_cl:SHA}' \
  data.cache_fingerprints='{hooktheory:SHA,pop909_cl:SHA}'
```

The plan emits the complete SSL/downstream matrix, fingerprints, seed domains,
budgets, artifact schema, and official SSL/training/evaluation overrides. Cell
execution therefore continues through `music_critic.ssl.run`,
`music_critic.training.run`, and candidate-first
`music_critic.evaluation.run`; there is no parallel trainer or evaluator.

Read-only production smoke is available only for explicitly configured paths,
at most three train/validation pieces per dataset/split. Planning never scans
cache directories. Missing paths produce a structured skip. Test is never read
and caches are never created or modified.

## Bounded acceptance

CPU acceptance covers two paired seeds, scratch, Phase 7A, Phase 8A mask-only,
one single-level objective, equal-weight, frozen/full transfer, launch-order
invariance, exact forward/sample/update accounting, protocol/resume mismatch,
target mutation evidence, aggregation rejection, validation-only selection,
test-lock negatives, piece bootstrap, diagnostics, and official-engine matched
execution. It asserts mechanics and variant distinction only; SSL need not beat
scratch on the tiny fixture.

CUDA/AMP acceptance is optional and must use explicit `cuda:0`. A skip is not
CUDA evidence. No production corpus, PDMX, long-training, effectiveness, or
quality claim was produced in Phase 8B.2A.
