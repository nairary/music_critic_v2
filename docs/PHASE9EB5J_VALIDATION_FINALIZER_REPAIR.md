# Phase 9E-B5J — all-shift VALIDATION finalizer repair

## Incident

C2 completed all `120000` optimizer updates and its identity-only VALIDATION at
update `120000`, then failed before publication of `run_summary.json`:

```text
analysisgnn.full_orbit.validation_eligibility_incomplete: 162
```

The failed code populated the 162 frozen VALIDATION record IDs but looked up
allowed shifts in the Phase 9E-B5A `record_shift_eligibility.jsonl` artifact.
That artifact is intentionally TRAIN-only, so every VALIDATION shift list was
empty. Training, the update-120000 checkpoint, and all 25 identity VALIDATION
reports were unaffected.

## Repair boundary

The repair derives allowed VALIDATION shifts directly from each frozen
VALIDATION raw graph and sidecar using the already-audited B5G executable
criteria:

1. the directed raw graph transform must execute without a MIDI-range failure;
2. transformed targets must stay in the frozen vocabularies;
3. target round trip, masks, and entity identities must remain valid;
4. identity shift `0` must be valid for every one of the 162 records.

TRAIN-only B5A eligibility is not consulted. TEST assignments are not loaded as
records, TEST targets are not read, and no optimizer update is executed by the
repair.

The original B5H and B5I runners remain byte-identical. Two wrappers replace
only their final all-shift diagnostic entrypoint:

- `scripts/run_phase9eb5j_finalize_c2.py`
- `scripts/run_phase9eb5j_analysisgnn_c0_orbit_matched.py`

This preserves the C2 checkpoint's immutable model, optimizer, scheduler,
sampler, runtime, and profile contracts.

## Finalize the completed C2 checkpoint

From the repository root, on the repaired commit:

```bash
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON" -u scripts/run_phase9eb5j_finalize_c2.py \
  --full \
  --device cuda \
  --output-root outputs/phase9eb5h \
  --resume \
    outputs/phase9eb5h/c2-seed17-full-orbit-u120000/last.ckpt
```

Because the checkpoint contains `applied_update=120000`, the TRAIN loop is
skipped. The command derives VALIDATION eligibility, runs the repaired
all-shift diagnostic, and publishes:

```text
outputs/phase9eb5h/c2-seed17-full-orbit-u120000/all_shift_validation.json
outputs/phase9eb5h/c2-seed17-full-orbit-u120000/run_summary.json
```

## Launch the C0-120K-MATCHED control

Run preflight and CUDA smoke first:

```bash
"$PYTHON" -u scripts/run_phase9eb5j_analysisgnn_c0_orbit_matched.py \
  --preflight \
  --output-root outputs/phase9eb5i

"$PYTHON" -u scripts/run_phase9eb5j_analysisgnn_c0_orbit_matched.py \
  --smoke \
  --device cuda \
  | tee outputs/phase9eb5i/cuda-smoke.json
```

Then start the fresh seed-17 control:

```bash
mkdir -p outputs/phase9eb5i
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -u scripts/run_phase9eb5j_analysisgnn_c0_orbit_matched.py \
    --full \
    --device cuda \
    --output-root outputs/phase9eb5i \
  > outputs/phase9eb5i/c0-120k.console.log 2>&1 &

echo $! | tee outputs/phase9eb5i/c0-120k.pid
```

The control still uses exactly the C2 240000-draw schedule and optimizer budget,
but applies shift `0` to every TRAIN graph and sidecar. The repaired finalizer
runs only after update `120000`.

## Non-goals

No dataset, split, model, loss, class weights, TRAIN sampler, initialization,
optimizer, scheduler, checkpoint, C2 training history, C0 treatment, or TEST
policy is changed. The repair makes no superiority or statistical-significance
claim.
