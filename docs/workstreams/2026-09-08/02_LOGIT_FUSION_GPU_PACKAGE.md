# Harmonic fusion: bounded paired pilot and coordinator queue package

Date: 2026-09-08. Phase: isolated post-corrected-Phase-9E fusion integration.
This package fixes the first pilot design before either arm produces results.
It is a protocol and queue request, not a launched experiment or a positive
performance gate. TEST remains closed. Legacy inspection/reuse is unnecessary.

Implementation scope is the new bounded trainer, complete resume contract,
fusion-aware evaluator and their synthetic tests. Historical C0/C2 model,
runner, split, vocabulary, sidecars and transposition remain the reference.
The coordinator owns substantial runs, source binding, records and GPU access.

## Gate status from current saved evidence

| Gate | Evidence and actual scope | Status for this pilot |
| --- | --- | --- |
| Historical C2 CUDA mechanics/restoration | `EXP-9EP1-C2-CUDA-EQUIVALENCE-001` and independent `EXP-9EP1-GPU-EVIDENCE-007`: 108 applied updates, 216 draws, 18 heads, original and optimized direct0/1/2/4, fresh-object checkpoint restoration. Synthetic historical stale-track-feature fixture. | Completed synthetic prerequisite; does not authenticate real TRAIN C2 payloads or production throughput. |
| Integrated performance | `EXP-9EP1-GPU-EVIDENCE-015`: C0 identity-only, one fixed-order B2 comparison; workers4 1.43733456348544 updates/s, original0 0.9491713760697515 updates/s. | Useful bounded planning prior. Original controller remains failed; post-hoc acceptance is explicit. No real C2 acceptance. |
| Real TRAIN historical C2 equivalence | No completed record in the supplied source snapshot closes this gate. | **Missing; scientific launch blocked.** |
| Fusion training/resume and evaluation | New task-owned implementation and source-free CPU acceptance. | CPU readiness must be reported separately from target CUDA execution. |
| New trainer on target runtime | Neither prior gate contains this trainer or fusion weights. | **Queued synthetic CUDA gate required.** |
| Historical C2 checkpoint export | `EXP-POP-TRANSFER-EXPORT-001` records a strict C2 model-only export with matching source/export state fingerprints. | Concrete candidate exists; rehash actual checkpoint bytes before launch. Model-only export supplies no historical optimizer/RNG resume. |

Primary scope references are
[C2 restoration review](../../experiments/PERFORMANCE_REPLAY_C2_RESTORATION_REVIEW.md)
and [integrated post-hoc review](../../experiments/PERFORMANCE_REPLAY_INTEGRATED_POSTHOC_REVIEW.md).
They explicitly deny real C2 equivalence and production promotion. The
implemented `performance_c2_train_schedule.py` declares 12 shift-coverage
updates plus a separate 100-update seed-17 orbit prefix, 112 jobs / 224 draws
for future sizing. Its metadata-only descriptor is not a completed run.
The existing `performance_prefetch_replay.build_prefetch_schedule` rejects
nonidentity production jobs with `prefetch_replay.identity_schedule_required`.
There is no honest `--profile C2` invocation of that existing CLI.

The coordinator's missing real TRAIN gate must bind those two cases, the
actual raw/sidecar bytes and historical transposition; compare original versus
the exact proposed optimized preparation/update, uninterrupted and restored;
retain per-step logits, alignments, 18 losses, gradients, model, optimizer,
schedule and RNG equality; and measure reservations for any workers. AN/DLC
and full-vocabulary support must be stated as observed, with missing support
explicit. A new source-bound executor/reservation package is required before
this can become an executable real-data queue entry. This task does not
invent its command or claim that a synthetic rerun would close it.

## Fixed numerical pilot

| Field | Both arms |
| --- | --- |
| Applied updates `P` | 512 per arm; 1,024 total |
| Draws | B2; 1,024 per arm; 2,048 total; no gradient accumulation |
| Initial state | Same preselected C2 model-only bytes and base tensor fingerprint |
| Optimizer | Fresh AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.0005`; all base parameters trainable; only fusion adds parameters |
| LR | Peak `0.0003`, linear warmup 64 applied updates, cosine decay over the remaining 448, terminal multiplier zero; no inherited 120k scheduler |
| Gradient/precision | FP32 logits/loss, finite guards, clip global norm 1.0; no AMP/scaler skipping |
| Seeds | 17; identical fresh model/dropout RNG and sampler seed domains; loader/sampler work must not consume dropout RNG |
| TRAIN | Same historical C2 full-orbit seed-17 stream starting at new experiment draw 0, same pieces/shifts/batch boundaries; commit two draws only on successful update |
| Validation | Complete frozen identity Dilemmadata VALIDATION at updates 0, 256 and 512 in each arm; no TEST loader |
| Persistence | Full training checkpoint every 64 applied updates and at completion; immutable checkpoint names; final and selected best separate |
| Selection | Existing eight-head primary macro-F1 on those three validation points; exact tie selects earliest update; selection separate for each arm |
| Stopping | Fixed 512 updates; no quality-triggered early stop, extension, arm-only retry or LR change |

The fixed comparison is continued C2 without fusion versus equally continued
C2 with zero-initialized fusion. Historical C2 metrics are a starting-point
description, never the control's final result. This is fresh continuation,
not exact historical resume. Resume within this new experiment must restore
its model, optimizer, scheduler, RNG and committed sampler cursor exactly.
Both arms retain B5B masks, TRAIN-supported class weights and
`L_primary + 0.25 * L_auxiliary` over the same 18 heads. No WS1 complete pitch
transformation, different source mixture, consistency loss or cadence tuning
enters this comparison.

The LR is a conservative, predeclared fresh-continuation choice (6% of the
historical 0.005 peak), not an empirically optimal value. Warmup occupies one
64-update checkpoint interval. Exact multiplier indexing is defined and
tested by the new trainer; the resolved schedule is saved with each attempt.

### Why 512 and what the timing does not prove

Request a **40-minute exclusive paired slot**: 20 minutes provisionally for
updates and 20 for startup, six complete validation passes and checkpoint
I/O. Use the slower measured original0 rate only as a planning prior; apply
a provisional 10% extra-time allowance to both arms, then round down to a
whole number of 256-update validation intervals:

```text
P = 256 * floor((1200 seconds * 0.9491713760697515 updates/second)
                / (2 arms * 1.10 * 256)) = 512
```

At that prior the pair's update-only estimate with allowance is about
1,187 seconds. Neither C2 runtime nor validation runtime nor fusion overhead
has been measured by this task. The 10% allowance is an assumption; the
216,886 added parameters (+5.9227%, total 3,878,822) do not imply any runtime
ratio. A completed fresh C2 gate must show that this exact budget fits the
slot before launch. If it does not fit, close the preparation attempt and
preregister a replacement resource/config decision before either scientific
arm runs; do not adapt `P` to one arm's metrics or silently truncate the pair.

One target GPU is used sequentially, never two parallel model jobs. Use B2
with parent Torch intra/inter-op limits of one and all four inherited native
thread limits set to one before imports. Use zero workers for the initial
new-trainer gate. Workers or retained caches cannot be enabled by this
document: an exact source/schedule reservation and C2 equivalence acceptance
must select that runtime explicitly. Record actual parent/worker getters,
peak CUDA allocated/reserved bytes, sampled owned RSS/headroom/swap, startup,
data preparation, model update, validation, checkpoint and whole-arm times.
Retained-cache accounting is not a complete RSS bound.

The coordinator enforces the slot and existing host safety limits. OOM,
nonfinite update, source/hash drift, scheduler/cursor mismatch or resource
timeout stops the scientific pair. Preserve the interrupted attempt and all
successful-prefix counts, report the pair as incomplete, and allocate new
linked IDs for retries. A restart that passes full strict resume may finish
the same immutable planned budget only under the coordinator's recorded
attempt protocol. Do not compare unequal successful-update counts as a
completed matched pilot. No quality result triggers automatic continuation.

## Concrete starting-checkpoint locator

The selected candidate is the already exported C2 artifact from
`EXP-POP-TRANSFER-EXPORT-001`, fixed before fusion outcomes:

| Binding | Value |
| --- | --- |
| Portable locator | `outputs/pop909_transfer/20260908/export-001/c2-model-only.pt` |
| Current read-only evidence root | `/Users/aavalsamakin/Projects/music-critic-v2` |
| Checkpoint SHA-256 / bytes | `a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797` / 14,844,299 |
| Model state fingerprint | `82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39` |
| Corrected base contract | `c12380dc6a9e4923bf80e843d010698fa7dd8fdefb62e9d3c0262d0c1ff32748` |
| Provenance locator | `outputs/pop909_transfer/20260908/export-001/c2-provenance.json` |
| Provenance SHA-256 / bytes | `bdb054ca9133650a14ece1a5b95d17faf85dca8594d013eb8d85b32f9a3a2eab` / 400,689 |
| Historical source checkpoint SHA-256 / bytes | `fb5247920ca000ed4ca8d6266bbdb24b2dddcef230231d6ad723bd4faba6fc8a` / 44,500,549 |

This task verified the provenance JSON's bytes against the terminal record
before reading it. It did not read or deserialize checkpoint tensors. The
provenance records identical source/export model-state fingerprints and
strict export verification. The source was named `best-validation.ckpt`,
but `selection_update_fields_extracted=false`: do not invent a verified
historical selection step. The source's full contract was reconstructed from
the current matching fingerprint; that historical provenance limit persists.
The explicit candidate selection here is the supplied SHA-bound C2 export,
not a newly recomputed historical-best claim. Rehash and strictly load the
export under its own contract on the target before constructing either arm.

## Evaluation and interpretation fixed before results

Preserve identity VALIDATION reports at start, final and selected best. Keep
the established eight-head macro-F1 primary, the full 18-head breakdown,
supported-class confusion/support, and component-level predictions. Keep
harmonic joint correctness and consistency distinct: correctness compares
the same supported target tuple; consistency measures stated compatibility
relations among predictions on an explicitly checkable subset. Unknown or
uncheckable relations remain masked and the report names the denominator.
No new consistency score may silently replace the established joint metric.

Cadence and metrical strength are separate mandatory monitors, with all
available classes (including IAC/PAC/HC and rare metrical classes), support,
confusions and missing denominators. Primary improvement cannot erase either
regression. Fusion does not promise to repair cadence. Any unseen-tuple
analysis must distinguish native TRAIN unseen from augmentation-exposure
unseen and bind the actual exposure inventory.

POP is evaluated only after Dilemmadata checkpoint selection, with the same
frozen native primary / metadata_removed secondary views for both arms.
Root37 probability marginalization, quality full17 predictions/subset9 gold,
independent masks and exact source-event-to-beat alignment remain unchanged.
Fusion-aware loading must preserve nonzero learned fusion weights and full
class order. POP has already informed external diagnosis; subsequent results
are development/transfer evidence, not a new independent zero-shot claim.
POP labels never set this pilot's LR, duration, checkpoint or view.

A positive one-seed result warrants a separately planned repetition; a null
or negative result is equally retained. Matching updates/draws and base
passes does not match FLOPs or wall time. Attribute the measured added
compute separately; isolating cross-head exchange from added parameter
capacity needs a future capacity control. No automatic baseline promotion.

## Queue entries and launch bindings

The immediately executable next GPU work is the new trainer's **synthetic**
target-runtime acceptance, not the missing real TRAIN C2 gate. Before running,
the coordinator allocates `EXP-FUSION-CUDA-SYNTHETIC-001` as a planned
`engineering_gate`, pins the reviewed source snapshot and command, chooses a
fresh ignored output root, and reserves one GPU / 10 minutes / one CPU thread
per native library. It records and then finalizes the result, including any
failure. The tiny smoke uses its explicitly separate synthetic budget;
`harmonic_fusion_pilot.json` is not passed to it.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src python3 scripts/run_harmonic_fusion_pilot.py \
  --smoke --device cuda \
  --experiment-record docs/experiments/records/EXP-FUSION-CUDA-SYNTHETIC-001.json \
  --output-root outputs/fusion/EXP-FUSION-CUDA-SYNTHETIC-001
```

`EXP-FUSION-CUDA-SYNTHETIC-001` is allocated as a **planned** record in this
worktree; it has not run. Its full budget is six uninterrupted synthetic
updates per arm plus three-and-three interrupted/resumed updates per arm:
24 optimizer updates / 48 B2 draws, with initial checks separate. The exact
final source hashes remain pending independent review and must be sealed
before dispatch. The implemented smoke and output contract must be checked
with `--help` and CPU acceptance. Only coordinator queue dispatch may execute
CUDA. Do not rerun the already completed historical C2 synthetic fixture
solely to rename its ID as fusion evidence.

After the real TRAIN C2 gate and this CUDA gate pass, allocate separate
`EXP-FUSION-CONTROL-001` and `EXP-FUSION-PILOT-001` planned records in one
comparison group, linked to the selected C2 source/export and gate records.
They share the frozen numeric config, source/checkpoint/split/membership/
class-weight/transposition/schedule hashes, source code manifest, runtime
policy, and exact initial common-parameter state. The new request must bind
those actual artifacts and full successful gate evidence; a bare boolean
`gate_passed=true` is not an acceptance artifact. Missing hashes/locators
remain blockers. No guessed path or stale live-controller PID is a binding.

The implemented Python boundary is `FusionTrainingSession`: construct each
arm from the same strictly loaded model-only C2, use `FusionTrainingConfig`
from the fixed JSON, and supply the explicit TRAIN record reader, frozen
class weights, full-orbit table, VALIDATION callback and full-checkpoint
callback. This boundary and resume are source-free CPU tested. A registered
corpus dispatcher that authenticates actual source/metadata/gate bindings
before those callbacks is still a separate gate-dependent integration step.
There is deliberately no invented real-data `--request` command here.
The missing dispatcher and real TRAIN C2 gate are concrete launch blockers;
the ready executable next queue entry is the bounded synthetic CUDA command.
The 40-minute scientific bound covers the pair, including both validations
and setup; it does not allocate 40 minutes independently to each arm.
The coordinator seals SHA/size, actual updates/draws/skips, observed split
access and resource accounting, finalizes launched records, regenerates the
ledger and updates STATUS. Full tensors, predictions and logs remain ignored.

## Local configuration verification

`tests/experiments/test_harmonic_fusion_pilot_config.py`: **2 passed, 2
upstream Torch JIT deprecation warnings in 0.77 seconds**. The checks load the
actual pilot JSON through `FusionTrainingConfig`, recompute its 512-update
resource bound from the compact original0 timing, verify matching draw and
validation/checkpoint intervals and closed TEST, and check the complete LR
curve through its terminal zero. This is configuration acceptance only.
Local runtime was Python 3.12, Torch 2.13.0, PyG 2.8.0.post1; target CUDA
compatibility is separately unverified. No checkpoint tensors or corpus
payloads were read, and no scientific or GPU run was launched by this task.
