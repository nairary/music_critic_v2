# Phase 9E-B5I — C0 120k orbit-matched control

## Purpose

Phase 9E-B5I adds the no-transposition control needed to compare the completed
Phase 9E-B5H C2 full-orbit experiment against the same optimization and sampling
budget. This is a fresh seed-17 run. It does not resume the historical 10,000-
update C0 checkpoint and does not modify the C2 implementation.

The control consumes the exact deterministic C2 sequence of eligible
`(record_id, shift_pc)` rows. The scheduled `shift_pc` remains part of sampling
identity and logging, but the graph and target sidecar are always passed to the
trainer with `applied_shift_pc=0`. Therefore C0 and C2 have the same record order,
record multiplicity, batch boundaries, initialization seed, optimizer,
scheduler, update budget, validation cadence, checkpoint cadence, and final
all-shift diagnostic. The intended treatment difference is TRAIN
transposition.

## Frozen runtime

- profile label: `C0-120K-MATCHED`
- profile ID: `music-critic-v2-corrected-no-transposition-orbit-matched-v1`
- seed: `17`
- device: CUDA
- batch size: `2`
- applied optimizer updates: `120000`
- TRAIN draws: `240000`
- schedule: exact C2 full-orbit table and epoch permutations
- applied TRAIN shift: always `0`
- peak learning rate: `0.005`
- warmup: `6000` applied updates
- scheduler: linear warmup followed by cosine decay
- precision: FP32
- primary validation: identity-only every `5000` updates
- checkpoint: every `500` updates
- final diagnostic: the same all-eligible-shift VALIDATION diagnostic as C2
- TEST loaders, targets, metrics, and checkpoint selection: disabled

## Files

- contract and sampler: `src/music_critic/experiments/analysisgnn/orbit_matched_control.py`
- executable runner: `scripts/run_phase9eb5i_analysisgnn_c0_orbit_matched.py`
- focused tests: `tests/experiments/test_phase9eb5i_orbit_matched_control.py`

## GPU-server launch

Run from the repository root after fetching the Phase 9E-B5I branch.

```bash
source outputs/phase9eb1/environment/venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

First run the production preflight. It reads only the already accepted
non-TEST artifacts and proves that the C0 source schedule is byte-equivalent to
the C2 schedule fingerprints while all applied shifts are identity.

```bash
python -u scripts/run_phase9eb5i_analysisgnn_c0_orbit_matched.py \
  --preflight \
  --output-root outputs/phase9eb5i
```

Then run the CUDA smoke:

```bash
mkdir -p outputs/phase9eb5i
python -u scripts/run_phase9eb5i_analysisgnn_c0_orbit_matched.py \
  --smoke \
  --device cuda \
  | tee outputs/phase9eb5i/cuda-smoke.json
```

Start the fresh full run:

```bash
mkdir -p outputs/phase9eb5i
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -u scripts/run_phase9eb5i_analysisgnn_c0_orbit_matched.py \
    --full \
    --device cuda \
    --output-root outputs/phase9eb5i \
  > outputs/phase9eb5i/c0-120k.console.log 2>&1 &

echo $! | tee outputs/phase9eb5i/c0-120k.pid
```

Monitor the process and the structured progress log:

```bash
PID=$(cat outputs/phase9eb5i/c0-120k.pid)
ps -fp "$PID"
tail -f outputs/phase9eb5i/c0-120k.console.log
```

The run directory is:

```text
outputs/phase9eb5i/c0-seed17-orbit-matched-u120000/
```

Important artifacts are `training_metrics.jsonl`, `validation_metrics.jsonl`,
`last.ckpt`, `best-validation.ckpt`, `all_shift_validation.json`, and
`run_summary.json`.

## Resume after interruption

Resume only from the `last.ckpt` located under the same output root:

```bash
python -u scripts/run_phase9eb5i_analysisgnn_c0_orbit_matched.py \
  --full \
  --device cuda \
  --output-root outputs/phase9eb5i \
  --resume \
    outputs/phase9eb5i/c0-seed17-orbit-matched-u120000/last.ckpt
```

Resume is fail-closed. It verifies the immutable runtime config, model and
optimizer state, sampler table and RNG domain, checkpoint interval, JSONL
ledger prefix, exact C2 record/shift schedule prefix, and identity-only applied
shift history. Rows written after the last atomic checkpoint are discarded
before training continues.

## Comparison boundary

Use the identity-only `corrected_primary_macro_score` curves and their best and
final values as the primary C0-versus-C2 comparison. The final all-shift
VALIDATION reports are robustness diagnostics and do not replace primary
checkpoint selection. The first comparison is one seed only and does not by
itself establish statistical significance.

No dataset, split, vocabulary, target mask, class weight, model architecture,
loss, C2 runtime, raw cache, TEST policy, or legacy code is changed by this
phase. No training result or improvement claim is committed.
