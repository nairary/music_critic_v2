# Task-attention fusion: authorized controlled comparison

2026-09-14. User selected80000from-scratch updates per arm, followed by a
separately reviewed MMoE stage. This first comparison contains no MMoE.

The upstream reference is `manoskary/analysisgnn` commit
`e115182fb29b74bdcb6bf3547ed427d967580947`, `analysisgnn/models/analysis.py`,
CrossTaskTransformer and forward_clf. We independently implement its mechanism:
task logits → Linear(C,64) → ReLU → LayerNorm; stack task tokens per entity;
MultiheadAttention(64,4,dropout0.1); LayerNorm(tokens+attention); per-task
Linear(64,C). Initial weights are ordinary PyTorch initialization; there is no
raw-logit residual, identity initialization, task detach or extra raw-head loss.

This adapts the mechanism to twelve already aligned beat heads, keeping the
six note/onset heads unchanged. It is not an all-task note-level reproduction
of AnalysisGNN. Existing encoder, head MLPs, graph/transposition semantics,
class dictionaries, data split, masks, group loss weights and scheduler remain
the corrected V2 definitions. No upstream runtime dependency or legacy reuse.

Both arms initialize an identical baseline from seed17; dropout stream17001
is reset after constructing optional modules. Attention consumes extra random
numbers during forward; common initialization/stream seeds do not imply equal
dropout masks throughout both models. Each arm draws the same C2 historical
orbit from update1, with batch2, fp32, four ordered workers,8GiB aggregate
compact cache and bounded transport. Both use the same pinned Python3.12.8 /
Torch2.2.2+cu118 runtime. Original120k scheduler horizon,6000warmup, peakLR0.005,
AdamW and global clip1.0 are preserved. Equal updates are not equal FLOPs;
parameters and wall time are reported separately.

Sequence on the single GPU: attention reference/accelerated equivalence gate,
control reference/accelerated equivalence gate, attention80000, control80000.
Each gate uses12batches covering12shifts,24total applied updates. Failed gates
stop the sequence. Training instances are fresh, never continued from gate
weights. Every stage has an immutable experiment ID and terminal record;
supervised maximum wall24h per training arm, existing GPU/RAM/swap guards.

Validation every5000updates on the same162pieces; full checkpoints every1000.
Primary score is eight-head mean macro-F1; also retain all18head/class metrics,
joint correctness, unseen tuples, and timing. Final and best are separate.
The full state has `TaskAttentionCampaignState@1.0.0`; model-only artifacts use
`TaskAttentionModelOnlyCheckpoint@1.0.0`. Strict loaders distinguish control,
attention and legacy linear fusion; no unsafe pickle fallback. Resume requires
an independently registered continuation and restores all state and draw offset.

Acceptance tests cover upstream-formula equality, cross-task gradients,
cross-piece isolation, entity alignment, empty beats, identical base/RNG
initialization, unique optimizer membership, strict checkpoint roundtrip and
exact next-step restoration, plus old linear-fusion regressions. Native gates
compare raw batches, sidecars, losses, model/optimizer/scheduler and RNG states.

TEST, new data, architectural changes outside fusion, baseline promotion and
automatic MMoE training are excluded. After the paired outcome is reviewed,
specify MMoE placement/size and its matched control in a new protocol.
