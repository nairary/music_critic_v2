# Phase 9C-C MLP convergence diagnostic

## Scope

This diagnostic asks one narrow question: did the seed-17 Phase 9C-B scratch-
MLP versus SSL-MLP comparison stop too early at 3,000 updates?

It declares only `scratch_mlp` and `ssl_mlp`. Both are MLP full fine-tunes with
batch size two, identical fresh supervised heads, one identical deterministic
Dilemmadata schedule, the unchanged four tasks/class weights/optimizer/LR and
the same multilevel-equal encoder export for the SSL cell. BiGRU, frozen probes,
new SSL objectives, extra seeds and test evaluation are outside scope.

## Continuous budget and evidence

Production is exactly one epoch and 9,000 applied optimizer updates. Telemetry
is written every 100 applied updates to each cell's
`train_telemetry.jsonl`. It contains only Python scalar/count/string evidence:
rolling objective and per-task losses, learning rate, scaler scale, finite
gradient norms, applied/attempted/skipped counts and the consumed schedule
prefix fingerprint.

Atomic checkpoints are written every 1,000 applied updates. Each contains the
strict model contract and state, optimizer, scaler, scheduler-null binding,
Python/Torch/CUDA RNG, cell/plan/data/schedule bindings, applied position and
committed telemetry. Resume reconstructs and advances the same epoch-zero
loader before restoring RNG. A scaler-decrease skip retries the same batch and
does not advance applied count; persistent skips fail closed.

Validation is performed from checkpoints at updates 0, 1,000, 3,000, 6,000 and
9,000 after uninterrupted training. The official evaluator uses `strict=True`
checkpoint loading and the same complete validation membership. Evaluation is
never an epoch, training input, checkpoint selection or stopping condition.

## Bundle

The output root contains:

- `experiment_plan.json` and `protocol.json`;
- `cells/{scratch_mlp,ssl_mlp}/train_telemetry.jsonl`;
- cell checkpoints `update-0.pt` and `update-1000.pt` through
  `update-9000.pt`;
- per-checkpoint validation reports plus `validation_milestones.json`;
- `convergence_report.json`, `manifest.json` and `payload.sha256`.

The convergence report preserves milestone values, 1,000→3,000,
3,000→6,000 and 6,000→9,000 changes, SSL-minus-scratch gaps, descriptive best
milestones/final-minus-best values, moving train averages and update accounting.
It deliberately has no plateau verdict and claims neither superiority nor
statistical significance.

## RTX 3090 commands

Use an existing Phase 9C-B-style config JSON that points to the unchanged raw
index/cache, target index/cache, split, class weights, train priors, SSL
checkpoint and the distinct versioned encoder export, including every declared
SHA-256. Phase 9C-C requires explicit `*_sha256` values for the raw index,
target index, split manifest, class-weight artifact and train-prior artifact in
addition to the SSL checkpoint/export hashes. Substitute the implementation
SHA exactly and use a new output root.

```bash
scripts/run_phase9cc_rtx3090_convergence.sh \
  run \
  <EXACT_PHASE9CC_SHA> \
  <CONFIG_JSON> \
  outputs/phase9cc-seed17-<UTC_TIMESTAMP>
```

Resume the same root after an interruption:

```bash
scripts/run_phase9cc_rtx3090_convergence.sh \
  resume \
  <EXACT_PHASE9CC_SHA> \
  <CONFIG_JSON> \
  outputs/phase9cc-seed17-<UTC_TIMESTAMP>
```

Run independent verification without CUDA training:

```bash
scripts/run_phase9cc_rtx3090_convergence.sh \
  verify \
  <EXACT_PHASE9CC_SHA> \
  <CONFIG_JSON> \
  outputs/phase9cc-seed17-<UTC_TIMESTAMP>
```

The wrapper requires a clean exact HEAD and, for `run`/`resume`, an RTX 3090
at logical `cuda:0`. It never rebuilds caches. A failure records the root and
full log path and returns nonzero. The completion marker and evidence path are
printed only after the independent verifier passes.
