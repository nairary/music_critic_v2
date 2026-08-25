# Phase 9C-D onset-BiGRU continuation

Phase 9C-D continues only `scratch_onset_bigru` and `ssl_onset_bigru` from the
verified Phase 9C-B update-3,000 full downstream checkpoints to update 15,000.
It reuses the Phase 9C-C optimizer loop, telemetry, atomic checkpoint and
mid-epoch resume boundary. The Phase 9C-B checkpoint adapter restores the
complete model, optimizer, scaler, scheduler-null state and RNG; it never
reloads an SSL encoder export.

Production is fixed to seed 17, batch size 2, AdamW at 0.0003, float16 AMP,
the existing four FP32 supervised heads/losses, one epoch-zero deterministic
schedule and validation milestones 3,000/6,000/9,000/12,000/15,000. The first
3,000 batches must reproduce Phase 9C-B, and the full 15,000 schedule must
equal the completed MLP continuation schedule.

Parent checkpoint paths are resolved from each verified Phase 9C-B
`cell_report.json` and must occur with the same SHA-256 in its bundle manifest.
Both parent validations, decoder contracts, membership, logits and state are
checked before either optimizer can step. The MLP reference is read-only and
bound by Git SHA, manifest, report and schedule fingerprints.

Outputs are a fresh immutable root containing BiGRU convergence and decoder
comparison reports, atomic checkpoints, telemetry, milestones, manifest and
payload digest. Reports are descriptive only: test stays locked and no
automatic plateau, superiority or significance verdict exists.
