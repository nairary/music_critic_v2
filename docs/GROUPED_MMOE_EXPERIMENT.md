# Two-group MMoE with attention: authorized80k experiment

2026-09-15. User explicitly requested implementation and one from-scratch
80000-update experiment with two expert groups, following the completed
attention80k run. The redundant no-fusion control was cancelled separately.

## Architecture

Use raw contextual128-dimensional embeddings after the existing hierarchy and
onset BiGRU. One bank serves12beat tasks. A second bank shares expert weights
between five note tasks and the onset cadence task; note/onset rows are
processed separately. Each bank contains four independent dense experts:
Linear128→256, GELU, Linear256→128. Experts have no extra dropout or residual.
Every one of18tasks owns Linear128→4 followed by softmax, computed per event;
its convex mixture of expert outputs feeds the existing task MLP classifier.
The unchanged attention-fusion block then refines the12beat logits.

No top-k routing, capacity/drop policy, balancing auxiliary loss, task labels
as inputs, predicted-label sampling, additional temporal attention, new targets
or new dependencies. Raw candidate enumeration/identities/masks remain the
existing contract. The backbone and all task heads continue training jointly.
Two banks add536648parameters; attention adds85894 to baseline3661936;
total4284478. This tests added specialization/capacity together, not a
parameter-matched proof of the MMoE mechanism alone.

## Training and comparison

One seed17 model, from scratch. Construct baseline and attention before the
expert banks, preserving their initial weights relative to historical
attention. Reset the training RNG domain to17001 after construction. The
expert banks introduce no additional dropout; equal RNG seeds nevertheless
are not a general guarantee of identical numeric trajectories across hardware.

Use the historical C2 orbit, batch2, original120k scheduler horizon,
6000warmup, peak LR0.005, AdamW/global clip1.0 and unchanged masked weighted
`mean(primary)+0.25mean(auxiliary)` loss. Use pinned Torch2.2.2+cu118 on RTX3090,
four ordered workers,8GiB total compact cache, existing bounded transport and
16GiB GPU/host-RAM/swap guards. Training supervisor has a24h bound. Do not
restart the cancelled baseline or automatically continue past80k.

Primary reference is completed attention80k, with historical C2/linear as
descriptive secondary references at matching updates. Validation every5000
on the same162works, eight-head mean macro-F1 plus all18head/class metrics,
harmonic joint and unseen-tuple scores. Full checkpoints every1000; final and
best saved separately. Model and full-state checkpoints have distinct
GroupedMMoEAttention versioned schemas and strict hash/contract loaders.

Record per-task mean mixture weights, mean routing entropy and top1 counts
at first two updates and every1000. These summaries describe the one sampled
TRAIN batch/all raw candidates, not the whole dataset or task-label frequency;
they cannot alone prove musical expert specialization. No balancing loss is
added based on these diagnostics during the run.

## Admission and evidence

`EXP-MMOE-GATE-001`:12TRAIN batches across all12shifts, one reference and one
accelerated model,24total applied updates. Check raw graphs, sidecars, losses,
model/optimizer/scheduler/RNG equality in the native GPU environment. The gate
must complete and its terminal record be sealed before training starts. Gate
weights are discarded; training is initialized freshly.

`EXP-MMOE-C2-80K-001`: the single authorized80k run. Launch through a detached
queue plus the existing ownership-aware resource supervisor. Every terminal
outcome is recorded, including aborted/failed runs with durable accounting.
Source/templates/data/environment are hash-bound in the records. Full evidence
lives in the new ignored/local and configured external campaign directories.
TEST is closed. No baseline promotion, multi-seed significance, automatic
architecture changes or further runs follow from this launch.

Focused tests cover mixtures, group sharing/isolation, all18candidate metadata,
empty candidates, gradients, common baseline/attention initialization, optimizer
membership, checkpoint type/hash rejection and exact next-step full-state
resume. Existing heads/baseline and accelerated-runner tests remain required.
No legacy modules are read, copied, imported or changed.
