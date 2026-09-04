# Experiment evidence ledger

Generated deterministically from schema 1.0.0 records. Do not edit by hand.

| Experiment | Phase | Kind | Status | Title | Decision use | Observed access (VALIDATION; TEST inputs/targets/metrics/selection) | Access evidence | Declared policy (VALIDATION; TEST inputs/targets/metrics/selection) | Artifacts | Fingerprint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| EXP-9EB5D-C0-001 | 9E-B5D | scientific_training | completed | C0 seed-17 no-transposition 10k screen | Select the current corrected AnalysisGNN baseline using VALIDATION only. | yes; no/no/no/no | verified_primary_evidence | allowed; forbidden/forbidden/forbidden/forbidden | 3 | 2d7c7586414052e1fe34cfbe7b7ce7159f3d627004196cf4d97bbfdf01884daa |
| EXP-9EB5D-C1-001 | 9E-B5D | scientific_training | completed | C1 seed-17 stochastic-transposition 10k screen | Accept or reject C1 as the current corrected AnalysisGNN baseline using VALIDATION only. | yes; no/no/no/no | verified_primary_evidence | allowed; forbidden/forbidden/forbidden/forbidden | 3 | 3ec74d210b69b39d659f23dd1fb08c9d54d070b29f4092cc585178d140dcf5c5 |
| EXP-9EB5H-C2-001 | 9E-B5H | scientific_training | completed | C2 seed-17 full-orbit 120k run | Decide whether a compute-matched C0-120k control is warranted; do not select a baseline from an unmatched comparison. | yes; no/no/no/no | verified_primary_evidence | allowed; forbidden/forbidden/forbidden/forbidden | 2 | 39ca91372e80d759a8565b2108c07ec9e0bb56f71044acdc6f50f65b98dd5195 |
| EXP-9EB5K-C0-120K-001 | 9E-B5K | scientific_training_control | running | C0 seed-17 compute-matched 120k control attempt 001 | Interpret the C2 result causally and reconsider the selected corrected baseline only after a matched VALIDATION comparison. | unknown; unknown/unknown/unknown/unknown | pending_primary_evidence | allowed; forbidden/forbidden/forbidden/forbidden | 0 | 987c9a89b311f470a84ae5e143da100cf40fa165a1ac82f1fcfe096aba866b2e |
| PLAN-9EB5K-C0-120K | 9E-B5K | scientific_training_control | planned | Planned C0 seed-17 compute-matched 120k control | Interpret the C2 result causally and reconsider the selected corrected baseline only after a matched VALIDATION comparison. | unknown; unknown/unknown/unknown/unknown | not_applicable_planning_record | allowed; forbidden/forbidden/forbidden/forbidden | 1 | e951278003b952b8e38df3715a3f242a5b74d1a953e12bf86378a696f6d5e4f0 |

Registry fingerprint: `f51f4f947a5da57a502352d951a7c2a06684889f151b09ace6f7e7ac59664e15`
