# Phase 8B.2A compute-matched comparison and downstream transfer protocol

## Status and claim boundary

Commit `7365286eb4df5ed8090aaf07964a33c95db2ed4d` implemented the original
control-plane primitives, but not an executable end-to-end experiment. The
remediated Phase 8B.2A implements `Phase8B2ComparisonProtocol@1.2.0`: the
official CLI now executes and resumes the complete SSL → export → downstream
→ fixed-validation → aggregation → selection → immutable-report DAG. It uses
the official SSL, training, and candidate-first evaluation engines in isolated
Python subprocesses. It does not contain production training results and makes
no claim that one SSL variant is better than another or better than supervised
scratch.

The CUDA pre-merge remediation does not change that scientific protocol. It
routes runtime CUDA statistics and device evidence through the shared
`CudaRuntimeDeviceIndex@1.0.0` logical-integer boundary and the initialized
`CudaMemoryStatisticsLifecycle@1.0.0` boundary. It advances only the Phase
8B.2 artifact evidence contract to `1.2.2` plus the affected engine and
hardware-report contracts.

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
CUDA allocated/reserved peaks when a real CUDA device is used. CUDA memory
statistics receive the explicit logical integer index resolved from the
concrete runtime device; they never depend on an implicit current device.
Matched-mode validation fails closed if forward, raw-exposure, or update
budgets differ.

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

Before training, the runner resolves the actual target-free sample schedule
with the official sampler. `actual_sample_schedule.json` records dataset ID,
piece ID, sample position, logical update, and batch position. Caller-provided
index/cache/split fingerprints are only expected values: official metadata is
read and attested independently, and stale values or index/cache path mismatches
fail before training. Every variant for a seed must reproduce the same observed
`(dataset_id, piece_id)` sequence bit-exactly.

Production metadata planning and engine validation communicate through
`Phase8B2DataSemanticProjection@1.0.0`. The projection has one source-neutral
shape for bounded and on-disk paths: ordered dataset/index and cache
identities, split fingerprint, normalized train dataset counts and size,
fixed-validation membership/counts, and mixture weights. Production sample
slots are still resolved only from index/split metadata and the official
sampler; neither targets nor canonical payloads participate in schedule
resolution. Engine reports are projected through the same function before
comparison, so internal `DataRuntime` dictionary layouts and null placeholders
are not scientific evidence. Index, cache, split, composition, validation,
mixture, observed schedule, or compute mismatches fail with stable Phase 8B.2A
contract categories before a cell is published.

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

Downstream validation uses one fingerprinted fixed view and the existing
candidate-first evaluator with train-only priors. The full validation split is
the default; a positive limit selects a deterministic subset without
replacement. The comparison protocol, training checkpoint, training report,
and standalone evaluation must agree on validation membership. Evaluation
contract `1.3.0` adds CPU-only per-piece sufficient statistics while retaining
all-negative multilabel prediction counts and exact descriptive per-label AP.

Metrics stay keyed by dataset and task. Categorical tasks retain eligible
rows, support, accuracy, balanced accuracy, macro/micro F1, NLL, train-prior
baseline, and model-minus-baseline values. Multilabel tasks retain per-label
support, micro/macro F1 and undefined reasons, prevalence baseline, exact
match, BCE/NLL, and all-negative counts. No cross-dataset or cross-encoding
global score is emitted.

The versioned primary endpoints are the declared HookTheory and POP909-CL
dataset macro summaries. A downstream cell is identified by
`(seed, variant_id, transfer_mode)`. Selection first aggregates the complete
paired-seed evidence for each `(variant_id, transfer_mode)` configuration, then
ranks configurations by mean dataset rank and, in order, lower validation NLL,
lower encoder-forward count, and lexical configuration ID. It never selects a
fortunate individual seed. Anti-collapse diagnostics never participate.

## Held-out test lock and access terminology

Test is unavailable to ordinary comparison/evaluation configuration. A test
authorization is created before inference only when all of these hold:

- a valid validation-selection artifact exists and says test was unused;
- its protocol fingerprint matches;
- it selects exactly one configuration and one checkpoint for every declared
  seed;
- acknowledgement is explicit;
- the output directory does not exist;
- the test membership fingerprint is already known and recorded;
- the locked experiment identity has not been consumed.

An exclusive sibling marker consumes the identity immediately before the new
output directory is created. Reuse, pre-existing output, missing selection,
wrong protocol, missing acknowledgement, multiple selections, or missing test
membership fails before inference. Unit acceptance exercises negative paths
without unlocking a real test split.

Planning resolves only test membership metadata needed by the lock. Plans and
final reports state all four facts independently:

- `test_membership_metadata_resolved=true`;
- `test_inference_performed=false`;
- `test_targets_accessed=false`;
- `test_metrics_accessed=false`.

The membership artifact stores the fingerprint, count, per-dataset counts and
split binding, not full test piece identities. Before a single-use unlock,
model forward and all test target/metric reads remain forbidden.

## Statistics and diagnostics

Production presets require at least three paired seeds; the paper preset uses
five. Evaluation stores categorical confusion/correct/eligible/NLL counts and
multilabel TP/FP/FN/TN/support/BCE counts per independent piece, task, dataset,
and encoding. Each paired bootstrap replicate resamples pieces and recomputes
the corpus endpoint from merged counts; averaging arbitrary per-piece macro-F1
is forbidden. Exact bootstrap AP would require retaining prediction-score rows,
so AP is explicitly descriptive and outside inferential claims. Bounded
acceptance never emits a significance claim or scientific p-value.

Transferred-encoder diagnostics report representation variance, effective
rank from singular values, adjacent-row oversmoothing cosine, zero norms, and
single-note perturbation deltas for note/onset/beat/bar/song. They form no
N-by-N matrix and retain no prediction tensor after a batch.

## Artifacts and aggregation

The protocol, plan, schedule, data-attestation, and test-lock contracts remain
`1.2.0`. Matrix-runner and cell-manifest contracts plus the preflight worker
evidence advance to `1.2.1` because CUDA preflight now publishes and validates
the lifecycle binding. Artifact evidence advances to `1.2.2` to bind
`CudaRuntimeDeviceIndex@1.0.0`, the logical CUDA index, and
`CudaMemoryStatisticsLifecycle@1.0.0`; the normalized data semantic projection
remains `1.0.0`. A complete
experiment has:

- `comparison_protocol.json`;
- `actual_sample_schedule.json`;
- `run_manifest.json`;
- `ssl_training_metrics.jsonl`;
- `ssl_checkpoint_evidence.json`;
- `transfer_evidence.json`;
- `downstream_metrics.json`;
- `piece_statistics.json`;
- `validation_selection.json`;
- `statistical_summary.json`;
- `compute_accounting.json`;
- optional `test_metrics.json`;
- `final_comparison_report.json`.

Each cell runs with list-form argv and `shell=false`, captures stdout, stderr,
exit code, runtime binding evidence, and SHA-256s in a cell manifest, and is
published from staging by atomic rename. Resume skips only a complete cell
whose manifest, hashes, and protocol binding verify. Failed, incomplete, stale,
or mixed-protocol outputs are never overwritten. Dependent stages stop on the
first invalid cell, and the final report is impossible until every required
cell is verified.

JSON/JSONL creation is atomic and immutable. Production evidence records exact
clean git SHA, environment, Python/PyTorch/PyG/CUDA versions, concrete device,
logical CUDA index and its boundary version, all seed domains,
data/cache/split/membership identities, protocol and contract
fingerprints, checkpoint SHA-256, and compute counters. Aggregation rejects
incomplete or duplicate cells, stale artifact fingerprints, mixed protocols,
data bindings, initial encoders, natural/matched modes, and unauthorized test
access.

SSL and downstream work reuse the official epoch-boundary checkpoint engines.
Those checkpoints bind the protocol/transfer runtime, full optimizer/
scheduler/scaler/RNG state, and metric journal. A changed protocol cannot
resume; existing failure-atomic load and journal recovery remain authoritative.

## CLI and presets

Planning is the default and performs no writes or training:

```bash
.venv/bin/python -m music_critic.experiments.phase8b2.run \
  action=plan comparison=bounded_acceptance
```

Executable actions are `plan`, `run`, `resume`, `aggregate`, and `select`.
`run` consumes the exact precomputed plan without manual engine overrides:

```bash
.venv/bin/python -m music_critic.experiments.phase8b2.run \
  action=run comparison=bounded_acceptance \
  output_root=/absolute/path/to/phase8b2-bounded
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
  data.split_manifest=/data/global.split.json
```

Optional `data.*_fingerprint` overrides are expected-value assertions, never
sources of truth; the runner derives authoritative identities from metadata.

The independent RTX 3090 pre-merge gate is a **production-format real-corpus
bounded smoke**, not a production pilot or a scientific comparison. Its
official implementation is
`scripts/run_phase8b2a_rtx3090_bounded_smoke.sh`; the operator supplies the
exact final commit, and the script uses `comparison=bounded_acceptance`, only
`phase7a_control`, seed 17, one SSL update, one downstream update, `cuda:0`,
FP16 AMP, and `comparison.validation_samples=128` with the production
index/cache/split paths. The validation view is one fixed deterministic subset
of exactly 128 pieces containing HookTheory and POP909-CL. It runs in a
subshell, preserves existing output/evidence roots, allows untracked files,
rejects tracked or staged changes, and creates a unique output root.

The previously published one-seed command using
`comparison=production_pilot comparison.seeds='[17]'` is **invalid**. It
violates `minimum_production_seeds=3` and must not be retried or treated as
pilot evidence. Likewise, `test -z "$(git status --porcelain)"` is invalid for
the independent GPU checkout because preserved untracked evidence is allowed;
the gate rejects only `git diff --quiet` or `git diff --cached --quiet`
failures and prints the untracked-file diagnostic. `device=cuda` is also not a
Phase 8B.2 Hydra group override; the concrete field is
`device.name=cuda:0`.

Any earlier runner invocation that omitted
`comparison.validation_samples=128` is also **invalid** as a bounded hardware
smoke: the zero default means the complete validation split, so the three
downstream evaluations are unbounded by sample count.

The script extracts only `runtime_paths` from the preserved failed plan by
default. It never reuses the failed plan's comparison preset, seeds, output
root, or other configuration. `PHASE8B2_SOURCE_PLAN` may identify another
preserved plan with the same production paths. A successful run is verified
for 8/8 cells, 8/8 runtime bindings, 3/3 checkpoint-to-evaluation bindings,
locked test access, positive logical-device-zero CUDA peaks, exact 1/1 SSL
update accounting, CUDA/AMP runtime reports without CPU fallback, and both
datasets in the planned and observed train/validation evidence. It also
requires exactly 128 selected validation identities and one consistent
membership fingerprint across the plan, SSL/downstream schedules and training
reports, and all evaluation metrics/checkpoint evidence. Downstream
`validation_epoch_size` and evaluation `max_evaluation_samples` must both
equal 128; matching dataset IDs without the exact count and fingerprint is
insufficient. Every CUDA worker report must also bind lifecycle contract
`1.0.0`, logical index zero, and `initialized_after=true` before its indexed
reset. The resulting archive contains logs and attestations, never
caches, checkpoints, or corpus payloads. Bounded results remain non-scientific
mechanics evidence.

The plan emits complete cell manifests, actual raw-sample and encoder-forward
budgets, validation-pass estimates, output paths, fingerprints, seed domains,
and artifact schema. Execution invokes `music_critic.ssl.run`,
`music_critic.training.run`, and candidate-first `music_critic.evaluation.run`;
there is no parallel trainer or evaluator.

Read-only production smoke is available only for explicitly configured paths,
at most three train/validation pieces per dataset/split. Planning never scans
cache directories. Missing paths produce a structured skip. Planning resolves
test membership metadata but performs no test model forward and reads no test
targets or metrics; caches are never created or modified.

## Bounded acceptance

CPU acceptance is a real CLI matrix: two paired seeds; Phase 7A, Phase 8A
mask-only, onset latent, and multilevel equal; frozen probe and full fine-tune;
paired scratch; 8 SSL cells, 8 encoder exports, 18 downstream cells, and 18
validation evaluations. It exercises preflight, interruption/resume, launch-
order permutation, actual schedule parity, checkpoint-to-evaluation
verification, sufficient-statistics aggregation, multi-seed selection,
immutable rerun, stale/incomplete rejection, resolved test-membership metadata,
and zero test inference/target/metric access. It asserts mechanics only; SSL
need not beat scratch on the tiny fixture.

CUDA/AMP acceptance is optional and must use explicit `cuda:0`. In addition to
the bounded matrix, the official production-format regression uses on-disk
HookTheory/POP909-CL-style cache/index/split artifacts, one seed,
`phase7a_control`, one SSL update, frozen/full/scratch downstream modes, and
three validation evaluations. It requires 8/8 scientific cells, 8/8 runtime
bindings, 3/3 checkpoint-to-evaluation bindings, positive allocated/reserved
VRAM, and no test inference, target, or metric access. A CPU skip is not CUDA
evidence. No production corpus, PDMX, long-training, effectiveness, or quality
claim was produced in Phase 8B.2A.
