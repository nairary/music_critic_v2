# Phase 9C-A: one-seed SSL → Dilemmadata production pilot

## Status and claim

Phase 9C-A implements an executable, immutable experiment bundle and RTX 3090
runner. The committed bounded fixture proves pipeline mechanics only. The
production pilot has not been run in this repository task.

The only allowed result is a one-seed exploratory validation comparison. It
may report observed validation differences from scratch and measured compute
or VRAM. It may not claim test quality, statistical or generalization
superiority, a final SSL benefit, paper-level significance, PDMX-scale
evidence, or a complete music critic.

## Fixed experiment matrix

The base seed is exactly `17`. The default `one_seed_primary_pilot` contains:

| Variant | SSL objective | Frozen probe | Full fine-tune |
|---|---|---:|---:|
| `scratch` | none | yes, random frozen encoder | yes, supervised baseline |
| `phase7a_control` | Phase 7A GraphMAE2 control | yes | yes |
| `phase8a_mask_only` | Phase 8A masks with old objectives | yes | yes |
| `multilevel_equal` | Phase 8B equal multilevel objectives | yes | yes |

`onset_latent`, `beat_latent`, `hierarchy_bar_latent`, and `track_latent` are
registered only by `one_seed_full_ablation`. No primary preset adds them
implicitly.

All variants bind the same initial encoder seed/fingerprint, downstream-head
seed/fingerprint, downstream data order, fixed validation membership,
optimizer/scheduler settings, downstream budget, raw SSL sample schedule, and
12 observed encoder forwards per logical SSL update.

## Data isolation and mixture

SSL reads raw graphs from train splits only:

- HookTheory train: weight `1/3`;
- POP909-CL train: weight `1/3`;
- Dilemmadata train: weight `1/3`.

The source-balanced schedule chooses the dataset independently of the variant.
Each dataset uses a deterministic shuffled cycle without replacement, and a
domain-separated reshuffle on every new cycle. Artifacts retain requested and
normalized weights, actual dataset counts, unique record counts, repeat
counts, cycle counts, slot identities, and schedule fingerprints.

The SSL path does not load Dilemmadata target bundles, target values or masks,
theory/provenance columns, validation records, or test records. It does not
import adapters or alignment oracles in the worker. Target-based selection is
forbidden.

Before planning, the CLI deterministically composes the existing
HookTheory+POP909-CL split manifest and the existing Dilemmadata split manifest
into the configured common SSL manifest. It never repartitions a record. Every
assignment field must remain exactly equal to its source-manifest assignment;
the result is validated against all three exact index fingerprints. Composition
fails if any Dilemmadata validation/test record appears in SSL train, if source
manifests overlap or omit an index, or if an existing destination has different
bytes.

## Dilemmadata downstream boundary

The production plan validates the existing component-closed split:

- train: 577 records / 565 components;
- validation: 71 records / 71 components;
- test: 71 records / 71 components.

Training consumes the complete train split. Validation traverses all 71
records without replacement. The test lock records only its membership
fingerprint and count; it serializes no full test identity list and permits no
test batch, inference, target read, metric, or unlock in any Phase 9C-A action.

Only these four CE tasks are enabled:

- `dilemmadata.an.chord.inversion`;
- `dilemmadata.an.chord.quality`;
- `dilemmadata.dlc.chord.inversion`;
- `dilemmadata.dlc.chord.quality`.

PU and open-string tasks have no head or loss. Reduction remains candidate
rows mean → source entries mean → fixed equal task sum, without active-task
renormalization. Encoder CUDA autocast may use float16; head inputs/logits, CE,
source-entry reduction, and total loss stay FP32. GradScaler begins at `16384`,
and the scheduler advances only after an applied update.

Encoder transfer is failure-atomic and transfers only the accepted local
encoder, hierarchy pooling/Transformer, and fusion prefixes. SSL decoders,
masking and multilevel heads, supervised heads, optimizers, schedulers, and
scalers never transfer. Every downstream cell records loaded tensor count,
source/loaded encoder fingerprints, fresh-head fingerprint, and fresh
optimizer/scheduler/scaler evidence. Frozen encoders must stay bit-exact;
fine-tuned encoders must receive finite gradients and change.

## Fixed-budget comparison and bootstrap

Every downstream variant is trained for the same declared and applied optimizer
update count. The comparison checkpoint is exclusively `last.pt` after that
budget; skipped updates, a missing checkpoint, or any count mismatch fail
closed. The complete validation split compares those final checkpoints. There
is no normalized-NLL checkpoint selection between epochs and `best.pt` is not
used by Phase 9C-A comparison.

The comparison metric was fixed before results:

```text
mean over four tasks of (source-entry NLL / log(class_count))
```

Lower is better. Phase 9C-A reports the final configurations without a separate
between-configuration winner-selection procedure.

Component bootstrap compares each SSL variant with scratch in the same
transfer mode. Production presets require at least 1,000 replicates. These
intervals measure validation-sample uncertainty only; they do not measure
optimization-seed uncertainty and are not a final significance claim.

## Actions and presets

The official CLI is:

```bash
.venv/bin/python -m music_critic.experiments.phase9c.run ACTION \
  --config CONFIG.json --output-root OUTPUT
```

Actions are `plan`, `profile`, `run`, `resume`, `aggregate`, `select`, and
`verify`. Presets are `bounded_acceptance`, `rtx_profile`,
`one_seed_primary_pilot`, and `one_seed_full_ablation`.

`profile` runs every candidate batch size in a fresh subprocess. A production
candidate executes short SSL cells, frozen probe, full fine-tune, and complete
validation traversal and reports allocated/reserved VRAM, samples/sec,
encoder-forwards/sec, epoch time, and projected pilot time. OOM cleanup is the
candidate-process exit boundary. A recommendation never mutates the later
production config.

The generated official SSL Hydra command appends the complete three-source map
with `+data.mixture_weights={...}` so the structured `DataConfig` accepts the
new `dilemmadata` key while retaining exact equal weights. Focused acceptance
composes the generated command through `music_critic.ssl.run --cfg job`.

Profile is fail-closed. If no candidate passes, `profile_report.json` records
`status=no_candidate_passed`, the CLI and RTX wrapper return nonzero, and the
wrapper never prints `phase9c.rtx.profile.complete`. Candidate roots and the
aggregate profile report remain in the requested output root for diagnosis.

Production epochs, steps, and batch size have no blind default. The production
config must state them explicitly and bind a separately completed immutable
profile report. `run` never follows `profile` automatically.

## Production configuration

An RTX configuration JSON supplies these fields:

```json
{
  "preset": "one_seed_primary_pilot",
  "ssl_updates": 1000,
  "downstream_epochs": 20,
  "downstream_steps_per_epoch": 100,
  "batch_size": 3,
  "bootstrap_replicates": 2000,
  "profile_report_path": "/absolute/evidence/phase9c-profile/profile_report.json",
  "profile_batch_candidates": [1, 2, 3, 4, 6, 8],
  "data": {
    "ssl_index_paths": [
      "/absolute/cache/hooktheory.index.json",
      "/absolute/cache/pop909_cl.index.json",
      "/absolute/cache/dilemmadata.index.json"
    ],
    "ssl_cache_roots": [
      "/absolute/cache/hooktheory",
      "/absolute/cache/pop909_cl",
      "/absolute/cache/dilemmadata"
    ],
    "ssl_source_split_manifests": [
      "/absolute/cache/hooktheory-pop909_cl.split.json",
      "/absolute/cache/dilemmadata.split.json"
    ],
    "ssl_split_manifest": "/absolute/cache/all-three.split.json",
    "downstream_raw_index": "/absolute/cache/dilemmadata.index.json",
    "downstream_raw_cache_root": "/absolute/cache/dilemmadata",
    "target_cache_index": "/absolute/cache/dilemmadata-target.index.json",
    "target_cache_root": "/absolute/cache/dilemmadata-target",
    "downstream_split_manifest": "/absolute/cache/dilemmadata.split.json"
  }
}
```

The numeric values above illustrate the fields and are not accepted production
recommendations. Select them only after the independent profile.

Exact profile command:

```bash
scripts/run_phase9c_rtx3090_pilot.sh profile EXACT_CLEAN_SHA \
  /absolute/phase9c-profile.json /absolute/evidence/phase9c-profile
```

After reviewing the report and writing explicit production budgets, exact run
command:

```bash
scripts/run_phase9c_rtx3090_pilot.sh run EXACT_CLEAN_SHA \
  /absolute/phase9c-primary.json /absolute/evidence/phase9c-primary
```

The script requires a clean exact HEAD and `NVIDIA GeForce RTX 3090` at
`cuda:0`, writes one log, preserves failed roots, never closes an interactive
shell, creates a regular-file evidence tar plus SHA-256 sidecar after success,
and runs the source-free verifier. It never starts `run` after `profile`.

## Bundle and resume

The bundle contains `experiment_plan.json`, `protocol.json`,
`data_semantic_projection.json`, `profile_report.json`, per-cell resolved
configs, SSL/downstream JSONL metrics, epoch performance, compute accounting,
checkpoint manifests, fixed-budget bindings, transfer and validation reports,
bootstrap and comparison
reports, final JSON/CSV/Markdown comparisons, curve data, PNG plots,
claim boundaries, and a SHA-256 artifact manifest.

Each unfinished cell lives under `.staging`; publication is one atomic rename.
A completed cell is immutable. Resume requires exact protocol and cell
fingerprints and rejects drift in index, split, mixture, target binding, seed,
budget, schedule, model, or variant. Rerunning a completed matrix verifies it
without rewriting artifacts. Variant enumeration order is absent from cell
identity.

## Bounded acceptance

The committed test fixture runs all seven actions across three short SSL
variants, initial/SSL encoder exports, scratch/pretrained frozen and full
downstream paths, fixed validation, interruption/resume, aggregation,
comparison, and verification. It checks 12-forward matching, AMP accounting,
fresh transfer state, frozen/fine-tune behavior, FP32 boundaries, test closure,
component bootstrap, manifest corruption, unsafe tar members, and zero retained
prediction/no-growth evidence. It does not use production caches or establish
model quality.
