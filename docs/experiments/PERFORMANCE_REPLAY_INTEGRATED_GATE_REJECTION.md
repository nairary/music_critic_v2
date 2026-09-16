# Integrated replay: retained final-controller rejection

`EXP-9EP1-PREFETCH-INTEGRATED-001` retains its original **completed** producer
record and **failed** controller result. The controller stopped at its final
`unchanged_FP32_runtime` assertion on 2026-09-07 at 04:22:13 UTC. The source
summary records `cudnn_allow_tf32=true`; the controller expected `false`.
No source artifact, runtime flag or historical status has been changed, and
no additional GPU run was launched.

Independent import `EXP-9EP1-GPU-EVIDENCE-014` verifies the exact 55-file archive,
all 29 source artifacts, lifecycle, source256 fingerprint `9d6…`, receipt,
60-test launch gate and owned-process closure. It preserves 1,200 complete
measured rows across three arms and the reported 240 warmup updates as durable
execution evidence. Numerical/performance acceptance remains false in this
import; it does not bypass the rejected runtime expectation.

The initial offline verifier failed on `parent_runtime_cudnn_allow_tf32`.
Its script, command, output and error are immutable in `verification-attempt-001`.
A narrowly scoped negative-result adapter in `verification-attempt-002`
accepts the archive's structural integrity while retaining that exact error,
`analysis=null`, `integrated_replay_verified=false` and
`production_speedup_accepted=false`. It never marks the old controller passed.

The retained 1,266 resource samples contain no policy violations: sampled
owned RSS is at most 13,396,729,856 bytes, effective headroom at least
45,667,319,808 bytes, and swap-I/O delta is zero. All 255 desktop observations
are accepted. Closure reports all observed owned processes exited without
cleanup signals. These are sampled observations, not universal memory bounds.

`EXP-9EP1-GPU-EVIDENCE-015` is separately planned as an explicit post-hoc review.
It must document the historical/runtime-source basis of any corrected
expectation and verify every remaining row, numerical fingerprint, cache,
resource, callback and timing check before accepting a bounded comparison.
There is no separate current-run pre-TF32 snapshot; neither measured pre/post
continuity nor execution with every TF32 option disabled may be claimed.
TEST/VALIDATION observations in the producer remain unknown, separate from
its closed declared policy; this import only reads saved metadata.

Archive SHA-256: `1108cf331ad8dac54b17c3d7cbd18316b1bc62f7dcc62b524ca2e62e68bdb32a`
(946,107 bytes). Source record fingerprint:
`68c12538ffacd28a18269d784ebd0249d994146edf3684eeeb7736a9e73d24df`.
Negative verification SHA-256:
`10f44b99e9c1acc41edf44686478b8d3ca78172dc605137258beee53dea97c9c`.
