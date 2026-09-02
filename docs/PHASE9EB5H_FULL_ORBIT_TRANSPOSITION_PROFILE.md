# Phase 9E-B5H full-orbit transposition profile

## Outcome

Repository status is:

```text
inverse_contract_valid=true
full_orbit_profile_valid=true
ready_for_full_orbit_training=true
full_orbit_training_run=false
```

C2 has immutable ID
`music-critic-v2-corrected-full-orbit-transposition-v1`. It is a new
from-scratch experiment and does not rename or resume C1. C0 remains the
selected baseline. C1 remains the completed fixed-compute stochastic profile
with `experimental_deferred` status.

## TRAIN orbit

The frozen piece-disjoint 1,295/162/162 split is applied before expansion.
Only TRAIN is expanded. The stable table is sorted by `(record_id, shift_pc)`
and contains each eligible pair exactly once:

```text
base TRAIN records           1295
nominal record-shift pairs  15540
eligible pairs              15389
excluded pairs                151
identity pairs               1295
identity fraction        0.08415101696016636
```

Valid-shift-set sizes are `12:1231`, `11:31`, `10:8`, `9:12`, `8:3`,
`7:8`, `6:1`, and `2:1`. Every excluded row retains its record/shift,
structured reason, source identity fingerprint, B5A source-file SHA-256 and
eligibility-row fingerprint.

Epoch order is a deterministic no-replacement permutation in domain
`sha256_B5H_full_orbit_epoch_permutation_v1`. A new epoch starts only after
all 15,389 pairs have been consumed. The sealed 240,000-draw schedule has 15
complete epochs and a deterministic 9,165-draw partial epoch. Shift zero has
no extra weight.

Graphs are not materialized. The runtime loads the canonical raw-only record,
applies canonical directed forward to a detached view, and uses `shift_pc` for
semantic targets after logits. Twelve views remain variants of one source
record, not twelve independent musical works.

## Sealed training contract

```text
seed                       17
batch size                  2
maximum applied updates 120000
TRAIN draws             240000
warmup updates            6000
peak learning rate       0.005
scheduler     linear warmup -> cosine decay
precision          FP32 baseline
early stopping             false
```

Model architecture, encoder, 18 active heads, losses, class weights,
vocabularies, routing, split, sidecars and raw cache are unchanged. The seed-17
initial state fingerprint remains
`0c18ba4a3b092f1ddfc2c88a09a0435c9881ef34750bf2012f93b41e715414f3`.

Evidence fingerprints are:

- C2 profile: `2de42ac93cfb6eb63399a912a69943886a5864ca27c6e367d2e6595b9044b3cd`;
- orbit table: `133983af065f28faab2258e8e2a1de057c87e34cdf214e494fa19a1e76987661`;
- partial final epoch: `2bec321d1b80a674e57ab5cbc9849b00e53ad0dd4f317c015b87c1a3fe1b6396`;
- compact fixture: `20797004bbbc59afd6d5e699df5220b215c96c8601aa7c49e807768afb344b29`.

Primary checkpoint selection uses complete identity-only VALIDATION. The
separate all-shift diagnostic reports per-shift primary score, loss and
corrected joint accuracy, five semantic head groups, macro over shifts,
worst-shift score, identity score and their gap. It never replaces primary
identity validation.

## Commands

Source-free/repository checks:

```bash
.venv/bin/python scripts/audit_phase9eb5g_directed_inverse.py --check
.venv/bin/python scripts/audit_phase9eb5h_full_orbit_profile.py --check
.venv/bin/python scripts/run_phase9eb5h_analysisgnn_full_orbit.py \
  --smoke --device cpu --seed 17
```

The later RTX run is intentionally not executed by this PR:

```bash
cd "$HOME/music_critic_v2"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

.venv/bin/python scripts/run_phase9eb5h_analysisgnn_full_orbit.py \
  --preflight --device cuda --seed 17 \
  --output-root outputs/phase9eb5h

.venv/bin/python scripts/run_phase9eb5h_analysisgnn_full_orbit.py \
  --full --device cuda --seed 17 \
  --output-root outputs/phase9eb5h
```

TEST loaders, targets and metrics remain disabled. Checkpoints, training logs,
datasets, caches, outputs, generated MIDI and rendered audio remain outside
Git.
