# Executable fusion dispatch handoff 004

This completes handoff003 with target setup, the mandatory real-TRAIN reference
gate, terminal registration and a first P512 request. It supersedes003's
unfilled-input and mandatory-repeat-synthetic wording. No target execution is
performed in the isolated worktree; the coordinator owns deployment, native
checks, the GPU queue and experiment finalization.

## Fixed scope

- Reference: one actual TRAIN update per arm, two updates/four draws total,
  exact unchanged P512 optimizer/config and identical orbit positions0/1.
  No VALIDATION, TEST, checkpoint export or music-quality decision.
- First scientific pair:512updates/arm, B2, seed17, fresh optimizer/warmup64,
  full identity VALIDATION0/256/512, checkpoint64, fixed unchanged full records
  and schedule. The control receives the same additional training as fusion.
- Reuse completed CUDA002 only after verifying the record/report and historical
  source manifest. The227-file runtime closure has226byte-identical files; the
  remaining resume file has an exact old AST after removing its single uncalled
  runtime helper and export. Newly added dispatcher/reference/preparation files
  are checked by the complete current source manifest and real reference gate.
  This does not relabel002 as current-source execution.
- An initial pair needs CUDA mechanics plus the real-TRAIN reference gate. A
  continuation with a parent still requires completed fresh-process CUDA
  evidence. The unlaunched planned fresh-process record is not in the initial
  queue. No additional experiment beyond the required gate/pair is introduced.

## Verified concrete inputs

The coordinator supplied server root
`/home/humtech/music_critic_research_20260913`, existing Python
`environment/bin/python`, Torch2.2.2+cu118/PyG2.6.1 and reserved GPU UUID
`GPU-1d7c6d9f-751f-7ea8-a951-d1b8a7991b42`. Use that existing environment;
the setup does not install or change dependencies. Native004 acceptance is
still required: local CPU tests use Torch2.8/PyG2.7.

The setup implementation contains explicit locators and verified pins for:

- `root/c2-model-only.pt`:14844299bytes,
  SHA `a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797`,
  model state `82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39`.
  Full pins were checked against the registered export provenance JSON,
  without opening checkpoint tensors here.
- `root/evidence/EXP-RESEARCH-DILEMMA-BINDINGS-001/cache-bindings.jsonl`:
  SHA `5ad21b1f9bad7cfeac2a098866724b5d80d08361d0e5d3188b5fb298086cc6ee`,
  1233467bytes, all1295TRAIN/162VALIDATION and zeroTEST entries.
- Deployed `tests/fixtures/analysisgnn/phase9eb5b_training_policy.json`:
  SHA `12b8b812e138af45e0ca2f7926bf2de3a53872368f165b3ead24b6aa9140dd34`,303658bytes.
- Native B4 `/home/humtech/music_critic_v2/outputs/phase9eb4/analysisgnn-class-balance-671097b/joint_tuple_counts.jsonl`:
  SHA `32cd9a4396bdb68c2077b68ca0efd0cd31d400e7c3385ca000d9b1206e38686b`,3031726bytes,
  confirmed by the coordinator's remote hash/size check. Only TRAIN tuples
  become the novelty inventory; no missing inventory becomes all-unseen.
- The already authenticated B3 split and B5A shift-eligibility locators/pins
  from the metadata binding audit. Setup rechecks all metadata and the entire
  frozen population through `FrozenFusionData`; it does not populate caches.

## Deploy and check

The ignored handoff directory carries README, patches, source archive,
per-file manifests, commands/logs and authenticated `cuda002-history/` JSON
plus the old resume source used only for AST comparison. No imported code is
executed by this compatibility check. Coordinator deployment destinations:

```bash
R=/home/humtech/music_critic_research_20260913
S="$R/fusion-004-source"
H="$R/fusion-004-handoff"
PY="$R/environment/bin/python"
```

Extract the reviewed `source.tar.gz` into a new `$S` and the handoff into a
new `$H`; verify archive SHA from README and per-file manifests first. These
are new deployment destinations, not claims they already exist. Run the
handoff's acceptance command from `$S` with `$PY` substituted for its local
interpreter before dispatch. The cumulative patch is for the original
snapshot; do not replay historical planned records onto newer terminal
coordinator records. Incremental004 applies to accepted003. Integrating current
status/decision/ledger changes remains a coordinator merge.

```bash
cd "$S"
export PYTHONPATH="$S/src"
export CUDA_VISIBLE_DEVICES=GPU-1d7c6d9f-751f-7ea8-a951-d1b8a7991b42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" scripts/prepare_harmonic_fusion_dispatch.py runtime \
  --output "$R/fusion-004-runtime.json"

"$PY" scripts/prepare_harmonic_fusion_dispatch.py reference \
  --server-root "$R" --repository-root "$S" --handoff-root "$H" \
  --runtime "$R/fusion-004-runtime.json" \
  --output-root "$R/fusion-004-reference-plan" \
  --records-root "$S/docs/experiments/records" \
  --comparison-id fusion-p512-20260913 \
  --experiment-id EXP-FUSION-TRAIN-REFERENCE-001

"$PY" scripts/experiment_registry.py render \
  --records docs/experiments/records --output docs/EXPERIMENT_LEDGER.md
```

Runtime capture is an explicit reserved-device metadata probe with zero model
construction/forwards/updates. Reference setup hashes the checkpoint bytes but
does not deserialize tensors; source and data manifests are sealed and the
request must pass public metadata admission before publication. It writes the
canonical planned record under `docs/experiments/records` and a separate
immutable planned copy in the plan root. Synchronize this compact planned
record into the coordinator's tracked registry before the worker launch.
No `null` placeholders remain in a successfully generated request.

## Native reference gate and registration

Use the already deployed, independently accepted resource supervisor. Its
existing CLI was checked against source here; host floor8192MiB and maximum
swap growth256MiB are enforced defaults. It uses the accepted Linux ownership
wrapper and cleans owned process groups before reaping.

```bash
REF="$R/evidence/EXP-FUSION-TRAIN-REFERENCE-001"
mkdir "$REF"
"$PY" "$R/supervisor-source/scripts/research_pilot_resource_wrapper.py" \
  --ownership-wrapper "$R/context-r2-source/scripts/context_diagnostics_resource_wrapper.py" \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES" --wall-seconds 600 --memory-mib 16384 \
  --receipt "$REF/resource-receipt.json" --stdout "$REF/stdout.log" --stderr "$REF/stderr.log" \
  -- "$PY" "$H/run-pinned-request.py" reference \
     "$R/fusion-004-reference-plan/reference-request.json" "$REF/run"

"$PY" "$H/finalize-reference.py" \
  --request "$R/fusion-004-reference-plan/reference-request.json" \
  --record "$S/docs/experiments/records/EXP-FUSION-TRAIN-REFERENCE-001.json" \
  --report "$REF/run/report.json" --resource-receipt "$REF/resource-receipt.json"

"$PY" scripts/experiment_registry.py render \
  --records docs/experiments/records --output docs/EXPERIMENT_LEDGER.md
```

Finalization is allowed only for the expected still-planned record and verified
native report, manifest/journal and successful bound supervisor receipt. It
registers the exact evidence and observed compute; terminal records are never
overwritten. Failed/aborted attempts remain evidence and require coordinator
terminal registration before a linked new attempt. No resource overage causes
record truncation or substitution. Synchronize the completed reference record
and evidence before creating the scientific plans.

The reference session deliberately has no validation lifecycle: it is a
separate non-resumable one-step engineering protocol, calls unchanged native
`FusionTrainingSession.step`, and cannot be exported as a scientific full
checkpoint. Hooks validate raw-only B2 forward and native target alignment;
AdamW post-step accounting retains applied work even if later code fails.
The no-VALIDATION execution scope refers to per-record payloads, forwards and
metrics. Admission still reads the frozen split/eligibility/weight/B4 metadata;
the pre-existing B4 file includes TRAIN/VALIDATION aggregate rows and only its
TRAIN rows are retained as the native tuple inventory. This metadata access is
not a validation evaluation and does not enter the model input or smoke loss.

## Generate and dispatch the scientific pair

```bash
"$PY" scripts/prepare_harmonic_fusion_dispatch.py pair \
  --reference-request "$R/fusion-004-reference-plan/reference-request.json" \
  --reference-record "$S/docs/experiments/records/EXP-FUSION-TRAIN-REFERENCE-001.json" \
  --reference-report "$REF/run/report.json" \
  --repository-root "$S" --records-root "$S/docs/experiments/records" \
  --output-root "$R/fusion-004-pair-plan" --attempt-id fusion-p512-attempt-001 \
  --control-id EXP-FUSION-P512-CONTROL-001 --fusion-id EXP-FUSION-P512-ENABLED-001

"$PY" scripts/experiment_registry.py render \
  --records docs/experiments/records --output docs/EXPERIMENT_LEDGER.md

PAIR="$R/evidence/fusion-p512-attempt-001"
mkdir "$PAIR"
"$PY" "$R/supervisor-source/scripts/research_pilot_resource_wrapper.py" \
  --ownership-wrapper "$R/context-r2-source/scripts/context_diagnostics_resource_wrapper.py" \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES" --wall-seconds 2400 --memory-mib 16384 \
  --receipt "$PAIR/resource-receipt.json" --stdout "$PAIR/stdout.log" --stderr "$PAIR/stderr.log" \
  -- "$PY" "$H/run-pinned-request.py" pilot \
     "$R/fusion-004-pair-plan/pilot-request.json" "$PAIR/run"
```

Synchronize both generated planned records into the coordinator's tracked
registry before that last worker command. The pair builder refuses failed,
unregistered, mismatched or mutated reference evidence and runs full admission
before emitting its executable request. A successful pair report does not
itself finalize both scientific records; the coordinator must register the
actual outcomes, resources, full validation histories and immutable artifacts
before a scientific decision. Earlier failure tails remain spent or unknown
work, even if a later separately gated continuation replays them.

## Local validation and boundaries

Final local CPU acceptance: **701 passed,0failed/0skipped/0deselected,
1 existing PyG deprecation warning,117.07seconds**. Compileall, diff check,
the48-record local registry and deterministic ledger check also pass.

The source tests include real tiny CPU updates, exact report admission,
checkpoint recovery costs, metadata construction and full setup→terminal
reference→paired-request flow. Fixture gate metadata is explicitly fabricated
for unit checks and never registered as actual CUDA evidence. Exact final test
counts/timing, patch SHA and source-manifest SHA are in the handoff README and
`handoff-verification.json`. Old534-test handoff003 remains immutable.

No legacy inspection/adaptation, main checkout edits, SSH, GPU execution, real
payload/checkpoint tensor reads, TEST, new loss, optimizer changes or120k run
occurred here. Next phase is coordinator native acceptance, the registered
two-update reference gate and the already agreed P512 pair.
