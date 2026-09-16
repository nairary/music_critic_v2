# Bounded real-data fusion pilot — increment 003

Date: 2026-09-13. Isolated worktree `a0bd/music-critic-v2`.
Phase: finish the accepted fusion mechanics with a registered reference-path
dispatcher. This document supersedes the launch-blocker wording of the older
protocol for the **original sequential path**. Earlier handoff documents,
patches, smoke results and terminal records remain historical immutable evidence.

## Implemented behavior

`scripts/run_harmonic_fusion_real_pilot.py` admits a byte-pinned request and,
only with `--execute`, runs the two arms sequentially on one CUDA device.
The production configuration must equal the existing pilot JSON exactly:
512 applied updates per arm, B2, seed17, LR0.0003, fresh64-update warmup and
cosine, weight decay0.0005, full identity VALIDATION at0/256/512, checkpoint64.
There is no120k/early-stop/extension/optimized-worker option. TEST requests
fail before payload I/O. The underlying Python seam permits tiny synthetic
tests; the production CLI exposes no reduced scientific budget or fixture mode.

Both arms load the same SHA/size/model-state-bound corrected C2 warmstart,
then own fresh optimizers, schedulers and stochastic state. The disabled arm
continues all base parameters; the enabled arm additionally learns zero-initialized
fusion matrices. Both consume the same full-orbit seed17 draw0..1023 prefix,
with the historical C2 forward transform, including its stale track-pitch
summaries. Raw canonical singleton preparation/collation is rebuilt per read.
There is no retained decoded/graph cache, worker pool or optimized C2 path.
All18-head losses, target masks, target joins after prediction, global clipping
and disabled FP32 scaler come from the existing accepted implementation.

The new full-validation callback derives all162 records/147components from
the complete frozen assignment, rather than accepting a sampled ID list.
It checks each raw/sidecar record/component identity, emits durable per-record
progress and preserves the existing18-head arithmetic, confusions, component
predictions, native TRAIN seen-tuple inventory and prediction-consistency
denominator. An exact primary-score tie selects the earliest of0/256/512.
Absent or empty TRAIN tuple inventory is rejected before evaluation. The new
metric contract exposes both the verified native B4 source SHA and the derived
tuple-set fingerprint; missing inventory cannot silently label every target
unseen. Historical evaluator APIs and scientific records are unchanged.
Selected-best and final full checkpoints and model-only exports are separate.
The control export retains the old corrected model format; the fusion export
retains its strict enabled format and learned matrices. Reports include
observed base/fusion weight changes without demanding quality improvement.

## Required data contract and current verified index

`FrozenFusionData` accepts `HarmonicFusionDilemmaData@1.0.0` with four metadata
file pins: `split_assignments`, `shift_eligibility`, `class_weights`, and
`joint_tuple_counts`. Every pin is `{path, sha256, size_bytes}` with an explicit
absolute regular-file locator, SHA256 and positive integer size. A record has
`record_id`, `split`, `source_component_id`, plus `piece`, `sidecar`, `metadata`
pins. The supplied coordinator index calls the canonical file `raw`; map that
field to `piece` without altering its bytes. Include all VALIDATION records
and at least every TRAIN record in the fixed1024-draw prefix; declaring all
1295TRAIN is also supported.

The loader independently enforces the checked-in frozen split hash/counts,
component disjointness, complete shift-eligibility rows, B5H table fingerprint,
B5B class-weight fingerprint and B4 native TRAIN tuple-file hash. All three
per-record files are checked before parsing; cache metadata, canonical piece
identity, sidecar self-fingerprint and record/component binding must agree.
It reuses canonical deserialization and raw-only production graph preparation.
Missing cache files are rejected; no corpus discovery or cache population is
performed. Historical conversion authenticity remains the coordinator's
export/audit responsibility, not something inferred from caller-provided
self-consistent hashes.

The read-only compatibility check of
`EXP-RESEARCH-DILEMMA-BINDINGS-001` authenticated its compact source bindings,
1,233,467-byte cache index and nine metadata files. The1457cache rows exactly
match1295TRAIN/1209components and162VALIDATION/147components, with zeroTEST
rows. No canonical/sidecar payload or checkpoint was opened locally. Detailed
hashes and explicit known remote pins are retained in
`outputs/fusion/handoff-003/dilemma-data-compatibility-001.json`.

The coordinator subsequently identified the checked-in B5B policy file
`tests/fixtures/analysisgnn/phase9eb5b_training_policy.json`:303658bytes,
SHA `12b8b812e138af45e0ca2f7926bf2de3a53872368f165b3ead24b6aa9140dd34`.
Those local bytes were independently verified; their TRAIN-only weight payload
fingerprint is `87266d8a60574b16a6e66d4081b11144f680f55e6f5a7b203ff722f64d817ee8`.

Still needed for the operational manifest:

- The absolute deployed path of that byte-verified `class_weights` file.
- Explicit remote native B4 `joint_tuple_counts` path/SHA/size; file SHA must be
  `32cd9a4396bdb68c2077b68ca0efd0cd31d400e7c3385ca000d9b1206e38686b`.
- Eligibility has a verified remote pin but no local copy; its full table is
  checked when the admitted dispatcher runs against that explicit target path.

No complete operational data manifest is fabricated while the tuple-file pin
and deployed weight locator are missing. InitialC2 checkpoint/model-state and target runtime must likewise
be explicitly supplied and verified, not inferred from directory names.

## Request admission and exact invocation

The request schema is `HarmonicFusionPilotRequest@1.0.0`. Required keys:

| Key | Contract |
| --- | --- |
| `comparison_id`, `attempt_id` | Nonempty stable IDs; a continuation uses a new attempt ID. |
| `config` | Pin of the unchanged `configs/experiments/harmonic_fusion_pilot.json`. |
| `data_manifest` | Pin of the completed data manifest described above. |
| `source_checkpoint`, `source_model_state_fingerprint` | Corrected C2 checkpoint pin and expected base tensor fingerprint. |
| `source_manifest` | Pin of a reviewed `{files:[{path,sha256,size_bytes}]}` manifest. Its paths are repository-relative; every runtime Python file under `src/` and `scripts/`, `pyproject.toml` and pilot config must be present. |
| `runtime` | Exact `runtime_observations("cuda")` payload: Python/packages/platform/CUDA/GPU plus inherited native/Torch thread getters, deterministic/CUBLAS/cuDNN/TF32 policy. One visible GPU, all threads1, deterministic algorithms, no cuDNN benchmarking. Preserve explicitly accepted TF32 flags. |
| `gates` | Three entries `cuda_synthetic`, `cuda_fresh_process`, `real_train_reference`, each containing pinned `record` and `report` artifacts. |
| `experiment_records` | Separate pinned canonical planned records for `control` and `fusion`, each512updates/1024draws/seed17 and explicit TRAIN+VALIDATION/closedTEST policy. |
| `parent` | Null for the first attempt; otherwise `{journal:pin,experiment_records:{control:pin,fusion:pin}}` for the immutable predecessor. |
| `max_wall_seconds` | Exactly2400 for the whole pair, including admitted setup. |

The common stable binding saved in both planned records at
`protocol.fusion_pilot_bindings` contains `comparison_id`, data/config/source
manifest SHA, source checkpoint SHA/size/model-state fingerprint,
`runtime_fingerprint=fingerprint(runtime)`, `reference_path="sequential_historical_C2"`,
and the record's `arm`. Runtime source and data fingerprints plus the exact
draw schedule are also bound into every full training checkpoint.

Each completed gate must be an actual terminal valid `engineering_gate` record
registering the exact SHA/size of its primary report. Its protocol binding
contains `role` and, for the two CUDA synthetic gates, reviewed source manifest
SHA plus runtime fingerprint. The real-TRAIN reference gate additionally binds
the data manifest, source checkpoint SHA/model-state and reference path.
No existing terminal record may be edited to acquire these fields: bind them
in a new planned record before execution when existing evidence is insufficient.

Synthetic admission requires the native complete report schemas and budgets,
both arms, successful full-state trajectories, learned-fusion/export evidence
or distinct fresh worker processes, and matching observed package/runtime
fields. An unrelated one-step CUDA report or a bare `passed=true` is rejected.
The reference smoke report schema is `HarmonicFusionReferenceTrainSmoke@1.0.0`;
it must report positive actual B2 update/draw counts, closed TEST, realTRAIN
access and true `reference_smoke_passed`, `source_hashes_verified`,
`raw_only_prediction_verified`, `target_alignment_verified`, with its actual
environment. This attests a real reference-path smoke. **Optimized-C2
equivalence is required only if an optimized path is proposed**, which this
dispatcher cannot select.

With coordinator-created request bytes and their verified SHA/size, the
metadata admission command is:

```bash
PYTHONPATH=src python3 scripts/run_harmonic_fusion_real_pilot.py \
  --request "$FUSION_REQUEST" \
  --request-sha256 "$FUSION_REQUEST_SHA256" \
  --request-size-bytes "$FUSION_REQUEST_SIZE_BYTES"
```

It checks source, registration, gate and metadata contracts; it performs no
graph build, checkpoint tensor load or CUDA initialization. The coordinator's
actual execution adds the sealed runtime environment and a fresh output root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
python3 scripts/run_harmonic_fusion_real_pilot.py \
  --request "$FUSION_REQUEST" \
  --request-sha256 "$FUSION_REQUEST_SHA256" \
  --request-size-bytes "$FUSION_REQUEST_SIZE_BYTES" \
  --execute --output-root "$FUSION_NEW_ATTEMPT_ROOT"
```

Use the request's exactCUBLAS value if it is the other accepted `:16:8` value.
These are parameterized commands, not claims that the missing remote
artifacts/gates have been supplied. `--help` is locally executable without
data. No real-data CPU or GPU execution is authorized in this worktree.

## Durable attempts, selection and accounting

Each exclusive output root owns resolved config/bindings, a hash-chained
append-only fsynced journal, immutable checkpoint files, per-arm summaries,
best/final exports, comparison and SHA/size manifest. Success `report.json`
is published last after preparation of its manifest. Exceptions produce a
failure receipt where storage permits, retaining the original exception even
if receipt storage fails. A hard kill can leave an incomplete last journal
line; recovery uses only the complete verified prefix, never silently repairs
an internal corrupt row and never rewrites its parent files.

Before each validation boundary the dispatcher saves a checkpoint with the
current validation pending. It saves the completed boundary separately after
the callback succeeds. Earlier missing validation points fail closed. The
checkpoint descriptor's cursor/history/pending marker and the selected-best
checkpoint update are checked against the deserialized full state.

A continuation requires a byte-bound parent journal registered in both
terminal parent records, child `parent_ids` links, the same common bindings,
and a new attempt/output root. It restores the last recorded complete
checkpoint, not an orphan file discovered on disk. Prior successful updates
beyond that checkpoint remain spent compute even when replayed. Per-attempt
attempted/applied updates, B2 draw slots and actual TRAIN/VALIDATION reader
calls remain distinct from the scientific cursor and cumulative costs.
An update remains explicitly in flight across nested completed reader events;
unknown work after a hard kill remains unknown in all descendants. Complete
scientific cursor equality does not imply zero wasted or unknown compute.
Rejected checkpoint descriptors contribute zero new applied updates; successful
work in the preceding arm cannot be charged again if the next arm fails to
initialize. Initial resolved-metadata storage errors also retain a zero-work
failure receipt where storage permits. Rejections during request admission or
CUDA/runtime setup happen before an attempt journal exists and must be recorded
by the coordinator's enclosing supervisor.

The2400-second deadline is checked at Python operation boundaries. It cannot
interrupt a blocked native call; the coordinator still owns the external
GPU-slot/resource supervisor. No CPU/GPU throughput or memory-fit claim is
made from these unit tests. The dispatcher emits terminal attempt evidence;
the coordinator must finalize the corresponding experiment records and
ledger before a scientific decision or another substantial run.

Coordinator metadata-only sizing `EXP-PILOT-RAW-SIZING-001` covers the same
1295TRAIN/162VALIDATION records. Verified maxima are11686notes/18141raw-node
upper bound for TRAIN and7512/13557 for VALIDATION. These are raw counts,
not graph edge counts or GPU memory predictions. The requested external wrapper
limits are16GiB sampled whole-device memory, at least8GiB available host memory
and at most256MiB swap growth; the native reference gate must test admission
under those limits before a scientific arm. Overages explicitly fail; no large
record is filtered, cropped or substituted and the fixed orbit draws stay exact.
The dispatcher reports process peak RSS/CUDA allocated/reserved bytes but does
not implement that external whole-device/host supervisor. Metadata verification:
`outputs/fusion/handoff-003/coordinator-raw-sizing-verification.json`.

## Fresh-process synthetic readiness

The coordinator's canonical `EXP-FUSION-CUDA-SYNTHETIC-002` is now
**completed/valid** for reviewed handoff002, with24actual updates/48draws,
matching uninterrupted/restored trajectories in both arms,216886 changed fusion
scalars, fusion deltaL2 `0.4369781669384113`,12nonzero harmonic heads and
6optimizer steps in all12heads. Registered report SHA:
`4359bde5da37438984b6a6e6229f7c8e8c21e8097fb77ff0b7f925cb1c169002`.
Record fingerprint:
`425dd84c8bd9cf9d5cac95f5ca5a14a6505108d1b1c26177a2fd11079e85575f`.
Reported runner time is10.275480526965111seconds; supervisor sampling observed
1417MiB whole-device peak. This is coordinator CUDA evidence, independently
checked here only through canonical record/report metadata and hashes.

It proves the earlier same-process fresh-object path. It neither tests a
distinct Python process nor binds the new003 source. Its immutable protocol
has no `fusion_pilot_bindings`, so the strict003 admission does **not** accept
it as a current-source gate automatically. The coordinator must seal valid
current-source evidence under a new record before executable admission; reuse
through a future explicit compatibility review would require its own recorded
contract. No terminal002 bytes were edited or re-registered here. Its source
manifest SHA is `5840a2f95acd065223232ae0e2a61c8cbb4ea3fdbd87cf31fd93574ab7e394c7`.
The earlier001 failure is retained by the coordinator; the old local planned
copy in this historical snapshot remains unchanged. Metadata verification:
`outputs/fusion/handoff-003/coordinator-cuda002-metadata-verification.json`.

`prepare_training_resume_runtime` explicitly initializes an expected CUDA
inventory before strict loading in a fresh process. CPU readiness never
queries CUDA; missing/mismatched reservations and unverifiable generators
fail closed. The full checkpoint schema remains1.1 and the old loader does
not initialize CUDA implicitly.

The separate CLI keeps the fixed24-update/48-draw synthetic budget:6full plus
3prefix+3in a separate Python child for each arm. It checks whole model,
optimizer, schedule, scaler, sampler, RNG and validation history equality,
distinct worker PIDs and matching parent/child environments. It complements
the earlier active-learning/export gate rather than replacing it.

`EXP-FUSION-CUDA-FRESH-PROCESS-001` is newly **planned/unlaunched**; source and
runtime admission bindings remain to be sealed by the coordinator. The old
planned gate and all terminal records remain unchanged. The new planned record
links completed002 as historical parent evidence; that record is owned by the
coordinator's registry, not copied into this worktree.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src python3 scripts/run_harmonic_fusion_fresh_process_gate.py \
  --smoke --device cuda --expected-cuda-device-count 1 \
  --experiment-record docs/experiments/records/EXP-FUSION-CUDA-FRESH-PROCESS-001.json \
  --output-root outputs/fusion/EXP-FUSION-CUDA-FRESH-PROCESS-001
```

The retained local CPU gate in
`outputs/fusion/review003-fresh-process-cpu-001/` observed24updates/48draws,
12fixture validation reads and exact state equality in both arms across
distinct workers, in8.715487374996883seconds. Its16manifest entries and two
linked child manifests were verified. This is fixture mechanics, not CUDA
evidence or a musical result.

## Verification and package

Final combined CPU acceptance: **534 passed,0failed/0skipped/0deselected,
1existing PyG deprecation warning,106.87seconds**. This includes old fusion,
corrected C2 and frozen POP regressions plus new data, full-validation,
process-resume, attempt-recovery and real-request admission coverage. The
earlier523-test and532-test logs remain as intermediate evidence; only
`pytest-003.log` reflects the final code, including rejected-resume and
initialization/storage failure accounting fixes. Source/artifact verification is recorded
in `outputs/fusion/handoff-003/handoff-verification.json`, with exact command
and logs in `outputs/fusion/review003-acceptance/`. Focused development runs
include35data tests,32full-validation helper tests,16fresh-process readiness tests and dispatcher
admission/attempt tests. The final runtime is the coordinator-provided
read-only Python3.12.14/Torch2.8.0/PyG2.7.0/NumPy1.26.4 environment with Hydra;
the removed September8environment was not assumed to exist. A separate
temporary environment using already-declared dependencies was prepared before
the shared runtime arrived; no project dependencies changed.

`handoff-003` contains an incremental patch against the22verified stage002
files and a cumulative task-owned patch against the original installed source
snapshot, both checked by application and reconstructed-file hashes. Its source
candidate includes runtime source/config/tests and `pyproject.toml`; independent
coordinator review remains pending. Earlier package files are preserved. Main checkout is read-only;
no commit, push, merge, legacy access, SSH, real payload/checkpoint loading,
real-data experiment or GPU execution was performed by this task.
Legacy logic was neither inspected nor adapted; only existing V2 frozen
contracts, training, serialization and evaluation logic were reused. Static
review found no new device-argument autocast API dependency; the existing
Torch2.2-compatible disabled scaler fallback is retained. Current003 native
Torch2.2/CUDA execution remains a separate uncompleted target check.

Integration must respect each patch's base. The cumulative patch is for the
original snapshot only: it contains the historical planned001 record and must
not overwrite the coordinator's now-terminal001 record. Prefer the003
incremental patch against accepted002 and merge current status/decision/ledger
changes with the coordinator's newer registry state. The local ledger contains
48records; it is not the complete current coordinator ledger.

`outputs/fusion/handoff-003/request-preparation.json` gives the exact request
shape, verified local source/config/weight pins and outstanding values. It is
explicitly unsealed and is not an executable request. Only a completed byte-pinned
`HarmonicFusionPilotRequest@1.0.0` passing the CLI's default admission may run.

Remaining launch conditions: the tuple-file pin and deployed weight locator,
current reviewed source/runtime sealing, source-compatible target
fusion/fresh-process evidence and the
actual TRAIN reference smoke, explicit warmstart pin, registered paired
requests and coordinator slot. ScientificP512 results and baseline promotion
remain absent.
