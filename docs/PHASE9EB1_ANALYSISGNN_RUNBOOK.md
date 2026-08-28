# Phase 9E-B1 AnalysisGNN reconstruction runbook

## Claim boundary

This phase separates two evidence tracks. Public run `rhsjiz03` is a
historical checkpoint/result attestation, not an independent reproduction.
The pinned environment and 719-record common arm are reconstructions with
declared substitutions, not exact official reproductions. No Phase 9E-B1 code
changes the Music Critic model, decoder, SSL, V2 BiGRU/Transformer, or Phase
9E-A artifacts. Data, graphs, logs, checkpoints, and predictions remain in
ignored output roots.

## Historical attestation

Pinned identities:

- AnalysisGNN: `e115182fb29b74bdcb6bf3547ed427d967580947`
- public run: `melkisedeath/AnalysisGNN/rhsjiz03`
- run-recorded unpublished source: `7738a282abe5090d44627759786dfa31b71e1a43`
- artifact: `melkisedeath/AnalysisGNN/model-rhsjiz03:v0`
- original checkpoint: `epoch=98-step=8910.ckpt`
- bytes: `289662455`
- SHA-256: `a557d0046e2c03c19514e1351a3cd0f2b49c31b991c370307345a7f1c6a65f31`

Expected ignored evidence layout:

```text
outputs/phase9eb1/historical/
  config.yaml
  requirements.txt
  wandb-summary.json
  output.log
  model-rhsjiz03-v0/model.ckpt
```

Acquire the public bytes with the pinned W&B client (anonymous access works
while the project remains public):

```bash
outputs/phase9eb1/environment/venv/bin/python - <<'PY'
from pathlib import Path
import wandb

root = Path("outputs/phase9eb1/historical")
root.mkdir(parents=True, exist_ok=False)
api = wandb.Api()
run = api.run("melkisedeath/AnalysisGNN/rhsjiz03")
for name in ("config.yaml", "requirements.txt", "wandb-summary.json", "output.log"):
    run.file(name).download(root=str(root), replace=False)
api.artifact("melkisedeath/AnalysisGNN/model-rhsjiz03:v0").download(
    root=str(root / "model-rhsjiz03-v0")
)
PY
```

Verify all bytes without checkpoint deserialization:

```bash
.venv/bin/python scripts/run_phase9eb1_analysisgnn.py attest \
  --evidence-root outputs/phase9eb1/historical \
  --output outputs/phase9eb1/historical/attestation.json
```

Committed config, runtime, metric, and digest evidence lives in
`configs/phase9eb1/`.

## Pinned reconstruction environment

The environment requires Python 3.12.8, PyTorch 2.2.2 CUDA 11.8 wheels, PyG
2.6.1, Lightning 2.5.0.post0, and the explicitly substituted GraphMuse commit
`c36eedba811a24c0addf96bdd3d1df449cf753c1`. A C compiler available as `cc`
is required, and the target environment directory must not already exist.

```bash
scripts/prepare_phase9eb1_environment.sh \
  /usr/bin/python3.12 outputs/phase9eb1/environment
```

The script verifies both Git revisions, checks/applies the three minimal
patches, installs both repositories without implicit dependencies, and saves
the resolved freeze and applied diff. Compare against
`reconstruction-lock.txt`; all differences are in
`patch_substitution_manifest.json`.

The historical reconstruction must retain the public configuration (including
weight decay `0.005`, 21 heads, batch 240, SWA, and 100 epochs) and be labelled
`pinned_paper_reconstruction`. It must not substitute the common dataset for
the public corpus. Do not start it before the real-graph smoke gate below has
been reviewed.

## Pre-training RTX acceptance order

Preparation validates the pinned Dilemmadata release, rebuilds target-neutral
pieces and source-native sidecars, and binds the unchanged Phase 9E-A common
projection. It fails unless there are exactly 108 AN + 611 DLC records and the
source-first split is 577/71/71.

Run these commands in order: environment capture, data preparation, the real
TRAIN graph on CPU, and the same deterministically selected TRAIN graph on
CUDA.

```bash
phase9eb1_python=outputs/phase9eb1/environment/venv/bin/python
export CUDA_VISIBLE_DEVICES=0
export MUSIC_CRITIC_DILEMMADATA_ROOT=/absolute/path/to/johentsch-dilemmadata-d60ee75
PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
  "$phase9eb1_python" scripts/run_phase9eb1_analysisgnn.py environment \
  --output outputs/phase9eb1/smoke/environment.json
PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
  "$phase9eb1_python" scripts/run_phase9eb1_analysisgnn.py prepare-data \
  --corpus-root "$MUSIC_CRITIC_DILEMMADATA_ROOT" \
  --output-root outputs/phase9eb1/common-data
PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
  "$phase9eb1_python" scripts/run_phase9eb1_analysisgnn.py real-graph-smoke \
  --cache-root outputs/phase9eb1/common-data \
  --manifest outputs/phase9eb1/common-data/manifest.json \
  --device cpu --output outputs/phase9eb1/smoke/real-train-cpu.json
PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
  "$phase9eb1_python" scripts/run_phase9eb1_analysisgnn.py real-graph-smoke \
  --cache-root outputs/phase9eb1/common-data \
  --manifest outputs/phase9eb1/common-data/manifest.json \
  --device cuda --output outputs/phase9eb1/smoke/real-train-cuda.json
```

Both smoke artifacts must report `split=train`, `transposition=P1`, finite
loss/gradients, logits `[N,50]` and `[N,4]`, and `optimizer_step=false`.
Verify that selection and graph construction are identical before proceeding:

```bash
"$phase9eb1_python" - <<'PY'
from pathlib import Path
import json

root = Path("outputs/phase9eb1/smoke")
cpu = json.loads((root / "real-train-cpu.json").read_text())
cuda = json.loads((root / "real-train-cuda.json").read_text())
for artifact in (cpu, cuda):
    assert artifact["acceptance"] is True
    assert artifact["split"] == "train"
    assert artifact["optimizer_step"] is False
for key in ("record_id", "piece_id", "graph_sha256", "node_counts", "edge_counts"):
    assert cpu[key] == cuda[key], key
print(cpu["record_id"], cpu["graph_sha256"])
PY
```

**STOP. Inspect and retain all three smoke artifacts before starting any
training command.**

The existing `smoke` command uses a synthetic graph and remains a model-only
diagnostic. Even without `--allow-model-only-stub`, it is not pinned GraphMuse
graph acceptance and cannot replace either `real-graph-smoke` artifact.

The ignored manifest binds every piece, source group, split, raw projection,
target bundle, and common projection. Graph fingerprints are recorded per
view. Training has 6,924 views (577 × 12); validation/test use only P1.

## Historical reconstruction after smoke review

After the STOP gate has been reviewed, the exact RTX command for the historical
arm is:

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export MUSIC_CRITIC_ANALYSISGNN_PUBLIC_DATA_ROOT=/absolute/path/to/struttura
export PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse
outputs/phase9eb1/environment/venv/bin/analysisgnn-train \
  --do_train --do_eval --gpus 0 \
  --raw_dir "$MUSIC_CRITIC_ANALYSISGNN_PUBLIC_DATA_ROOT" \
  --model HybridGNN --num_layers 3 --hidden_channels 256 --out_channels 128 \
  --dropout 0.3 --lr 0.005 --weight_decay 0.005 \
  --batch_size 240 --num_workers 5 --subgraph_size 500 --num_epochs 100 \
  --main_tasks all,cadence,rna --feature_type simple --mt_strategy wloss \
  --use_jk --logit_fusion --add_beats --add_measures \
  --use_swa --use_transpositions --use_wandb
```

The official public corpus revision/split is not published in the run config;
the operator must record the supplied corpus inventory and fingerprint as a
substitution. Without that exact input, this command is a pinned paper
reconstruction only.

## Three scratch seeds

Run these commands only after the real CPU/CUDA smoke artifacts pass review.
Do not open test during preparation, smoke, or validation selection:

```bash
for phase9eb1_seed in 17 23 42; do
  PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
    outputs/phase9eb1/environment/venv/bin/python \
    scripts/run_phase9eb1_analysisgnn.py train \
    --cache-root outputs/phase9eb1/common-data \
    --manifest outputs/phase9eb1/common-data/manifest.json \
    --output-root outputs/phase9eb1/common-runs \
    --dependency-lock outputs/phase9eb1/environment/resolved-freeze.txt \
    --seed "$phase9eb1_seed" --device cuda
done
export RTX_OPERATOR_ID=operator-or-job-identity
outputs/phase9eb1/environment/venv/bin/python \
  scripts/run_phase9eb1_analysisgnn.py unlock-test \
  --manifest outputs/phase9eb1/common-data/manifest.json \
  --runs-root outputs/phase9eb1/common-runs \
  --output outputs/phase9eb1/common-runs/test-unlock.json \
  --authorized-by "$RTX_OPERATOR_ID" --authorize-locked-test
for phase9eb1_seed in 17 23 42; do
  PYTHONPATH=outputs/phase9eb1/environment/sources/graphmuse \
    outputs/phase9eb1/environment/venv/bin/python \
    scripts/run_phase9eb1_analysisgnn.py evaluate-test \
    --cache-root outputs/phase9eb1/common-data \
    --manifest outputs/phase9eb1/common-data/manifest.json \
    --runs-root outputs/phase9eb1/common-runs \
    --unlock outputs/phase9eb1/common-runs/test-unlock.json \
    --seed "$phase9eb1_seed" --device cuda
done
outputs/phase9eb1/environment/venv/bin/python \
  scripts/run_phase9eb1_analysisgnn.py summarize \
  --runs-root outputs/phase9eb1/common-runs \
  --output outputs/phase9eb1/common-runs/three-seed-summary.json
```

Each seed uses exactly 10,000 applied updates; CE smoothing `0.1`,
`ignore_index=-1`, no class weights, learned two-task uncertainty, AdamW
`0.005/0.0005`, 500-update warmup, then cosine. Validation every 500 selects
minimum mean normalized quality/inversion NLL. `train` leaves test locked. The
explicit unlock is created only after all three selected checkpoint hashes are
frozen; `evaluate-test` then opens test once per seed on CUDA.

One candidate update contains one complete source/transposition graph. The
seeded shuffled source-view order, source group, graph hash, and applied or
skipped outcome are written to `batch_schedule.jsonl`. This is an explicit
common-arm substitution for the public run's sampled-subgraph batch 240.

Per-seed outputs include configs/bindings, architecture, update logs,
validation, graph fingerprints, checkpoint SHA-256, per-entry
logits/predictions/masks, all required metrics, joint accuracy, confusion
matrices/support, majority baseline, normalized mean NLL, grouped bootstrap,
and a file-hash manifest. Summary requires
exactly seeds 17/23/42 and reports mean ± sample standard deviation.

## Current host observation

This host has Python 3.13.5, PyTorch 2.13.0+cpu, PyG 2.8.0.post1, no CUDA, no
C compiler, and no installed GraphMuse/Lightning/Partitura. The one preliminary
model-only CPU forward/backward produced finite gradients and the required
logit shapes with a recorded import stub. Static review then removed one extra
shape-preserving hierarchy projection to match the checkpoint's two-linear
layout. The local one-smoke cap forbids rerunning it, so that preliminary output
is diagnostic only. Final pinned-GraphMuse CPU graph smoke, GPU smoke,
historical reconstruction, all training/checkpoint selection, and locked-test
evaluation are `NOT RUN locally`; no synthetic results replace them.

## RTX host inputs and expected artifacts

Required environment variables are `CUDA_VISIBLE_DEVICES=0`,
`MUSIC_CRITIC_ANALYSISGNN_PUBLIC_DATA_ROOT` for the historical arm,
`MUSIC_CRITIC_DILEMMADATA_ROOT` for the pinned Dilemmadata release,
`PYTHONPATH` pointing to the pinned GraphMuse checkout, and `WANDB_MODE=offline`
unless the operator explicitly authorizes a private logging destination.
`RTX_OPERATOR_ID` is required only for locked-test authorization. No API key is
required for common training.

Set the authorization identity before the unlock command, for example:

```bash
export RTX_OPERATOR_ID=operator-or-job-identity
```

Required inputs are the official AnalysisGNN/GraphMuse source revisions, the
three patch files, Python 3.12/CUDA 11.8 environment, the public historical
corpus for reconstruction, Dilemmadata commit
`d60ee75b4a9495e932a4a7be39381578be17e222`, and the locally
replayable 719-record source data. The W&B historical checkpoint is attestation
input only and is forbidden as common-arm initialization.

Expected remote artifacts are: environment report/freeze; real TRAIN-graph CPU
and CUDA smoke with matching record/graph fingerprints;
historical reconstruction config, substitutions, logs, selected checkpoint and
evaluation; common dataset/split/graph manifests; three fresh seed logs and
validation-selected checkpoint hashes; a three-checkpoint locked-test unlock;
per-entry test logits/predictions/masks; all task/joint metrics,
confusion/support and bootstrap; three-seed mean ± std; and file SHA-256
manifests. None of GPU smoke, training, selection, or test evaluation was run
locally.
