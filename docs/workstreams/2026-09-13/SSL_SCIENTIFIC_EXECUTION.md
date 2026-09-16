# Bounded WS2 Dilemmadata execution — 2026-09-13

This implements the fixed endpoint comparison in the
[September 8 protocol](../2026-09-08/SSL_GROUP_PILOT_PROTOCOL.md). Model, loss,
pitch-masking semantics and the accepted review-002 update engine stay intact.
The new entrypoint is `scripts/run_group_masking_scientific.py`.
Execution belongs to the coordinator; this task uses synthetic CPU tests only.

## Fixed question and stages

Both arms start from the same strictly verified C2 model-only state and the
same freshly seeded SSL state. Compare independent_note_pitch against
beat_pitch_descendants for 200 mixed updates per arm, batch 2 raw SSL and
batch 2 theory replay, seed17, mask rate0.30, Adam LR0.0003, SSL coefficient0.7
and unchanged corrected-theory coefficient0.3. Evaluate the initial C2 and
both fixed endpoints. No best-checkpoint selection is performed.

Each endpoint then starts 100 theory-only updates (lambda_SSL0, lambda_theory1)
under one identical recovery schedule and the same LR/seed. Preserve every
endpoint model tensor, including SSL modules, and explicitly reset Adam to a
fresh empty state. This is a documented stage transition, not another C2
warmstart or a silent reuse of historical C2 optimizer state. Evaluate again.
The result reports per-head forgetting, recovery gain and remaining gap against
original C2, plus beat-minus-random endpoint and recovered contrasts. Missing
support yields null; no relative-recovery percentage or significance claim.

Logical TRAIN budget is 600 confirmed updates: 400 mixed and 200 recovery.
At full batch2 this is 800 SSL draws and 1,200 theory draws. Mixed updates use
three encoder forwards; recovery uses one: 1,400 TRAIN forwards, separately
from validation, preparation, interruption replay and verification replay.
Checkpoints default to every50 confirmed updates and every stage boundary.
All published artifacts are immutable; this avoids a full checkpoint per step.

## Data authority and preparation

The primary request uses the coordinator-verified historical cache index,
complete original split assignments and source components. Index records bind
record_id, split, source_component_id and raw canonical piece, expanded target
sidecar and metadata paths/SHA256/sizes. Read canonical raw data and build the
existing raw graph sequentially; targets remain separate and join only after
raw prediction. No whole-corpus graph retention or speculative optimized reader.
The source conversion was not repeated by the coordinator audit; byte-verified
historical cache consistency does not authenticate the original conversion.

The data audit supplied only compact metadata here. Its index SHA256 is
5ad21b1f9bad7cfeac2a098866724b5d80d08361d0e5d3188b5fb298086cc6ee
(1,233,467bytes), with 1,295 TRAIN and162 VALIDATION records. The official split
also contains162 TEST metadata records; no TEST payload reader is exposed.
An original record_id containing the word test may legitimately belong to
TRAIN under this frozen split; membership, never filename text, is authoritative.

Preparation streams the permitted TRAIN records and records structural
eligibility for both policies before compiling a shared deterministic schedule.
Rejected training examples and reasons remain explicit; no per-arm replacement
or runtime fallback changes a sealed schedule. Preparation and execution have
separate planned records: finalize preparation, hash its prepared-binding.json,
then bind those bytes in the execution plan and a new planned run record.
The optional native-source request also follows the old reference resolver and
adapter; it is separate from the verified-cache provenance path.

`preflight-size` and `size` are cached-only CPU actions under a separate planned
engineering record with `operation=size`. The request may set class_weights and
validation_limits to null only in this mode. It streams TRAIN/VALIDATION raw
pieces and metadata, builds raw graphs, and reports actual note/node/edge counts;
it does not read target sidecars/class weights or create/load a model. This graph
sizing API is distinct from the coordinator's earlier raw JSON inventory, which
did not construct graphs. It still needs external RAM/time supervision.

The coordinator's completed compact `EXP-PILOT-RAW-SIZING-001` receipt was read
locally: 1,295 TRAIN/162 VALIDATION records, zero model forwards/updates, declared
zero TEST payload reads. Its VALIDATION maxima are 7,512 notes and 13,557
conservative raw nodes. Proposed explicit per-piece admission limits are 16,384
notes, 32,768 nodes and 524,288 edges. The edge limit and GPU acceptance are
conservative proposals, not measured graph/VRAM maxima. Oversized VALIDATION
must reject the run, never drop a piece. TRAIN batch caps stay unchanged.

## Full held-out evaluation

Theory evaluation includes the complete frozen VALIDATION membership. Training
eligibility never silently removes held-out records. Raw records are streamed
one at a time with separate, explicitly configured validation note/node/edge
bounds. Those bounds are admission limits, not measured hardware acceptance;
if a record exceeds them, resolve the resource contract before execution.

Use the existing 18-head corrected validation accumulator. Valid targets whose
class was unsupported in TRAIN remain in event/class metrics; the original
training loss separately masks those unsupported classes. Preserve missing,
unsupported, unavailable and zero support distinctions. Report all heads and
per-class support, confusion-derived metrics, accuracy and observed-class macro-F1.
Joint tuple novelty is unavailable unless an independently bound TRAIN tuple
inventory is supplied; no full-TRAIN label scan is invented to fill it. An absent
inventory yields null seen/unseen accuracies, supports and unique tuple count,
with reason train_seen_tuple_inventory_unavailable. A supplied empty inventory
is distinct and means all joint targets are unseen relative to that inventory.

Evaluate fixed epoch-zero masks under both policies on the same held-out raw
membership. Structural mask unavailability is reported with coverage/reason and
must remain identical across checkpoints; it does not remove a record from
theory evaluation. Nonfinite output or runtime drift always fails the evaluation.
Aggregate SSL numerators/counts and existing mergeable anti-collapse sufficient
statistics; never average per-batch variance or pairwise cosine. No reconstruction
loss is represented as a music-quality score. Evaluation preserves parameters,
gradients, module modes and caller RNG, including interrupted attempts.

## CLI and admission

Start from `configs/ssl_group/scientific_pilot.template.json`. It intentionally
contains null data/prepared/resume bindings and cannot launch as-is. Plans bind
all scientific settings; the real CLI has no tiny-budget or synthetic bypass.
Use the runner API only for tiny synthetic mechanics tests.

A matching planned record must have phase WS2-SSL-GROUP and kind engineering_gate
for size/prepare or scientific_comparison for run/resume. Its protocol must contain
operation, implementation_fingerprint, scientific_plan_sha256, data_request,
prepared_data, source_checkpoint, resume, pilot_config and recovery_config exactly
matching the plan and current implementation. data_request and prepared_data
are objects with sha256 and size_bytes. The C2 parent must appear in parent_ids.
VALIDATION is explicitly allowed and every TEST flag is false. A resume adds
its prior experiment parent and externally verified workflow-manifest hash/size.
The CLI validates the registry record and these bindings before payload work.

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  preflight-size --plan <size-plan.json> --experiment-record <planned-size-record.json> \
  --data-request <verified-cached-size-request.json>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  size --plan <size-plan.json> --experiment-record <planned-size-record.json> \
  --data-request <verified-cached-size-request.json> --output <fresh-sizing-output>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  preflight-prepare --plan <prepare-plan.json> --experiment-record <planned-prepare-record.json> \
  --data-request <verified-data-request.json>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  prepare --plan <prepare-plan.json> --experiment-record <planned-prepare-record.json> \
  --data-request <verified-data-request.json> --output <fresh-preparation-output>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  preflight-run --plan <run-plan.json> --experiment-record <planned-run-record.json> \
  --data-request <verified-data-request.json> --prepared-data <prepared-binding.json>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  run --plan <run-plan.json> --experiment-record <planned-run-record.json> \
  --data-request <verified-data-request.json> --prepared-data <prepared-binding.json> \
  --source-checkpoint <verified-C2-model-only.pt> --device cuda:0 --output <fresh-run-output>

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_group_masking_scientific.py \
  resume --plan <resume-plan.json> --experiment-record <new-planned-resume-record.json> \
  --data-request <verified-data-request.json> --prepared-data <prepared-binding.json> \
  --source-checkpoint <verified-C2-model-only.pt> --resume-manifest <prior-workflow-manifest.json> \
  --device cuda:0 --output <fresh-resume-output>
```

The command must run under the coordinator's external wall-time/RAM/VRAM/disk
resource wrapper; neither the fixture pre-step wall600 check nor this workflow
claims a hard timeout. Record exact interpreter/dependencies/runtime/device.
Historical Torch2.2.2 is below the repository's declared torch>=2.8 requirement;
CUDA admission must verify a supported isolated environment, not assume historical
runtime acceptance transfers to this implementation. No dependencies are fetched
by these commands. A separately tested autocast getter compatibility helper is
included for the coordinator-requested historical-runtime fixture retry; it does
not alter these dependency declarations or prove broader Torch2.2 compatibility.

## Interruption and evidence

Durable started/progress/failure receipts precede update work and count observed
completed encoder forwards. Adam not_applied/applied/unknown is separate from a
fully confirmed update and a written successful row. A returned Adam with invalid
state is applied but unconfirmed; Adam that mutates then raises is unknown.
Finite gradients alone do not prove finite optimizer slots. All prior review-002
runtime/state checks remain active during pilot, recovery, save and resume.

Resume uses a fresh attempt output and externally hashed workflow manifest plus
confined, bounded, rehashed state/evaluation artifacts. It checks the same source,
data, config, runtime, schedule prefixes and recovery parent, then continues
from the last confirmed saved boundary. Completed metrics are verified against
their bound model states. Actual repeated work and evaluation replay are counted
separately; an immutable manifest cannot report work occurring after it was
published, and interruption does not imply rollback. Retain full attempt logs and
finalize every launched record before retrying. The CLI never edits tracked records.

## Inputs still owned by coordinator

- Verified cache root/index, official complete split and source-components
  descriptors (paths, hashes, sizes); compact audit descriptors exist, actual
  payload reads remain coordinator-owned.
- The frozen TRAIN class-weight artifact in existing B5B payload format with
  path/SHA256/size and TRAIN-only provenance.
- Revalidated C2 model-only locator/SHA256/size/full state fingerprint and parent.
- Explicit per-piece VALIDATION graph budgets and independently accepted CUDA
  runtime/resource wrapper; no training eligibility cutoff substitutes for these.
- Prepared common eligibility/schedule/batch/validation binding bytes and their
  completed preparation record, followed by a resealed planned execution record.

No real payload, SSH, GPU, TEST, voice/rhythm/window task, PDMX, model promotion,
commit, push or merge is part of this implementation task.
