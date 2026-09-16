# Integrated replay: post-hoc engineering review

Experiment `EXP-9EP1-GPU-EVIDENCE-015`, 2026-09-07. Parent negative import `EXP-9EP1-GPU-EVIDENCE-014` remains immutable; source `EXP-9EP1-PREFETCH-INTEGRATED-001` completed, while its V14 controller remains **failed**.

The actual three-arm run supports a bounded engineering result: four CPU workers with the compact cache and optimized update achieved **1.5143× steady throughput** and **1.4145× throughput over the directly measured whole arm** relative to original preparation and update. All five recorded state/RNG/loss/schedule fingerprints agree across the three arms. This is an explicitly registered post-hoc acceptance of those measurements, not a passed original controller gate.

## Direct timing comparison

Each arm applied 480 B2 updates: 80 discarded warmup updates and 400 measured updates, using the same 960 record draws, including the same measured suffix. The fixed order was compact-cache workers4 → compact-cache workers0 → original workers0. The initial model, optimizer, scheduler, scaler and RNG snapshot was common.

| Arm | Measured 400 updates, s | Measured updates/s | Direct whole arm, s | Steady throughput / original | Whole-arm throughput / original |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact cache, workers4, optimized update | 278.292897 | 1.437335 | 377.015950 | 1.514305 | 1.414511 |
| Compact cache, workers0, optimized update | 335.281900 | 1.193026 | 442.329977 | 1.256913 | 1.205645 |
| Original preparation and update, workers0 | 421.420209 | 0.949171 | 533.293085 | 1 | 1 |

Workers4 therefore reduced measured elapsed time by **33.96%** and direct whole-arm time by **29.30%** versus original. Relative to the compact-cache workers0 arm, workers4 improved steady throughput by 20.48% and whole-arm throughput by 17.32%.

Whole-arm elapsed includes cold setup, startup, warmup, measured updates and cleanup. Ratios use those directly recorded elapsed times; nested stage durations are not summed. The three-arm replay took 1352.962561 s; the registered wrapper took 1353.563858 s; the owned-execution supervisor took 1355.866986 s, including launch, imports, registration, evidence output and confirmed closure. Controller dispatch to terminal state was 1356.947438 s. These are distinct nested scopes, not independent runs.

## Integrity and numerical checks

The unchanged archive contains 55 regular files: 29 bound source artifacts, 25 controller files and the terminal source record. SHA/size, safe paths, controller manifests, receipt, lifecycle and the complete 256-file source map were checked. The source fingerprint is `68c12538ffacd28a18269d784ebd0249d994146edf3684eeeb7736a9e73d24df`; the runtime source-map fingerprint is `9d6dc4df7356924caaae62692dc99199c0995c1260550d0e2b272228296f49c3`.

All **1200 measured rows** were checked independently: finite loss and gradient norm, ordered B2 jobs, exact loss and schedule trace hashes, graph counts, queue cursors, reservation bounds and per-worker runtime/source identities. Warmup stage coverage is 80 updates per arm; actual totals are **1440 applied updates and 2880 draws**. Initial state, final full state, final RNG, loss trace and measured schedule fingerprints agree in all three arms. Model/optimizer/scheduler/scaler/RNG equality uses producer-recorded recursive hashes, not checkpoint reload or independent tensor reconstruction. The four producer callbacks, closed owned subtree, source bindings, resource streams and the remaining controller postflight checks passed offline.

## Cache and memory scope

Workers4 use four private 2 GiB caches under an 8 GiB total retained-cache policy. Across 960 record lookups they reported 76 hits, 884 misses, 190 evictions and 8,553,047,973 accounted bytes at the end. The 694 retained entries are summed across private caches and may duplicate records. Workers0 uses one 8 GiB cache: 270 hits, 690 misses, no evictions, all 690 scheduled TRAIN records retained, 8,420,704,345 accounted bytes. Original workers0 has no prepared cache. This measures the complete chosen configurations; it does not isolate worker count from cache partitioning.

The maximum measured worker transport payload was 53,900,749 B, below R=477,102,080 B. The queue retained at most five reservations under B=2,385,510,400 B. Every cache, queue and worker observation passed its bound checks. Caches started empty and were cleared at cleanup; filesystem/OS-page-cache coldness is not established.

There were 1266 resource samples and 255 accepted desktop observations. Maximum sampled owned RSS sum was 13,396,729,856 B (12.477 GiB), minimum effective headroom 45,667,319,808 B (42.531 GiB), swap I/O delta zero, with no observed OOM-counter change or resource-policy violation. RSS sums may count shared pages repeatedly and sampling can miss peaks; cache and reservation accounting excludes other working data, IPC copies and model/runtime memory. No live OS or GPU check was performed by this offline diagnostic.

## Why the original controller failed

V14 asserted `cudnn_allow_tf32=False`; the final recorded current environment reports **True**, with `matmul_allow_tf32=False`. Accepted historical THROUGHPUT002 and SCREEN002 primary summaries also report that combination. Their execution manifests and the current manifest bind the same runtime helper SHA `293b5e6c61013f28b372696d1c1e79a838aa0edab575a5f516c440dd245a2edb`; inspection of the matching local helper found no assignment to `allow_tf32`. Historical summary and execution artifact hashes were verified separately in `runtime-primary-anchors.json`.

No current-process pre-run TF32 snapshot exists in this evidence. These facts support the explicitly recorded correction of the postflight expectation, but do **not** prove measured pre/post runtime continuity or TF32-disabled execution. Original state `failed`, error `RuntimeError('unchanged_FP32_runtime')`, missing `integrated-acceptance.json`, negative014 and its failed verifier attempt are preserved. `integrated_replay_verified` and original-controller-pass remain false; only `posthoc_engineering_verified` is true under015.

## Limits and next action

This is one finite run in one fixed order, C0 identity shifts only. It establishes numerical equivalence under the recorded path and a useful measured speedup; it does not establish nonidentity C2 equivalence, musical quality, restart/checkpoint equivalence, or a full-120k/multi-repeat speed guarantee. It does not infer observed TEST/VALIDATION access from declared policy: those observations remain unknown and policy remains closed.

Keep the accepted performance configuration available for the next authorized integration step. Before using it for scientific C2 training, complete the outstanding nonidentity C2 equivalence coverage and record the runtime policy explicitly. No additional training or inference was run for015.

## Evidence pins

- Archive: `1108cf331ad8dac54b17c3d7cbd18316b1bc62f7dcc62b524ca2e62e68bdb32a`, 946107 B; same bytes as014.
- Local post-hoc adapter: `119f0e9a5e04e2089e1bf93fffd2b508786e7431ccdce4d66ecab9594f0e7ec0`.
- Full verification: `d84ea7535cfd6ef5a763464ea115782b375c53d66fbed663f8a944883da498a6`, 31300 B, `outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-015/verification-attempt-001/verification.json`.
- Historical runtime anchors: `d037bf1d4f74e4cfb86556221105baf99d51d5371a13ca6eca674684d7cdc7b3`, 4228 B.

Validation: the single actual local015 adapter invocation exited 0 and passed every remaining check; no new test framework, GPU execution, imported-code execution, checkpoint deserialization or registry mutation was used. Legacy logic was neither inspected nor reused.
