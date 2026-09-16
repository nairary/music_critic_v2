# RES004: compact-cache reservation sizing

`EXP-9EP1-PREFETCH-RESERVATIONS-004` completed on 2026-09-07. Independent offline validation under `EXP-9EP1-GPU-EVIDENCE-013` confirms that all 480 compact-preparation B2 payloads fit the retained reservation. All 960 draws, 480 jobs and scheduled source bindings match accepted RES003 exactly. This is serial CPU sizing with zero model updates and no worker pool; it does not establish training speed or four-worker memory safety.

| Quantity | Verified result |
| --- | ---: |
| Serialized jobs / record draws | 480 / 960 |
| Warmup / future measured jobs | 80 / 400 |
| Scheduled TRAIN records | 690 of 1,295 |
| Sizing / controller wall time | 561.062156 / 563.684145 s |
| Largest serialized payload | 53,900,761 bytes = 51.404 MiB |
| Retained per-job reservation, R | 477,102,080 bytes = 455 MiB |
| Largest payload / R | 11.298% |
| Global admitted reservation budget, B / pending jobs, Q | 2,385,510,400 bytes / 5 |
| Cumulative serialized bytes | 8,064,454,869 bytes |
| Cache hits / misses / evictions | 122 / 838 / 683 |
| Final cache entries / accounted bytes | 155 / 2,140,521,313 bytes |
| Per-process retained-cache cap | 2,147,483,648 bytes = 2 GiB |

The cache policy declares 8 GiB divided among four preparing processes. This sizing run uses one process with a 2 GiB share; it does not start four workers. Reported counters satisfy `hits + misses = 960` and `misses - evictions - oversized_bypasses = entries`, with zero oversized bypasses. The hit fraction is 12.708%. The bound constructor starts an empty process-local cache in the fresh child, and the final counters satisfy the cold-cache admission balance; there is no separately recorded initial cache snapshot.

Compared with original-preparation RES003, 468 of 480 recorded payload lengths differ by 0–395 bytes, adding 134,281 bytes cumulatively. Serialization memoization can change lengths even for equal values. These records alone do not prove tensor equality or a particular cause for each byte difference. The reservation base binds the compact algorithm and exact source/jobs; the preparer instance additionally binds cache capacity. Capacity/history invariance of serialized size is explicitly false, and every future serialization must still fit R. Cumulative serialized bytes and accounted cache bytes are not RSS.

The archive contains 42 regular files: the terminal source record, 12 sizing artifacts, 23 controller files and six selected project test files. Validation checked its externally supplied SHA/size, exact safe inventory, every record/artifact/controller binding, all 480 ordered measurements, job and typed schedule hashes, the recomputed reservation manifest, and planned → running → completed lifecycle. Complete scheduled bindings match the accepted full TRAIN index; the older RES002 prefix of 240 draws also remains unchanged.

Initial and final source maps both contain the accepted 256 files: 251 unchanged files and exactly five approved successors, with fingerprint `9d6dc4df7356924caaae62692dc99199c0995c1260550d0e2b272228296f49c3`. The target gate contains 302 passing XML test cases, no failures/errors/skips, and a log without warnings. Its preserved original receipt has identical test, runtime, log and XML fields; the accepted receipt adds only `source_count` and an explicit `metadata_completion` descriptor. The tests were not rerun for that metadata completion. The existing bootstrap remains byte-bound to the prior accepted import-order fix.

There are 526 resource samples spanning 562.203 seconds. Maximum sampled owned-process RSS is 3,390,922,752 bytes (3.158 GiB), exceeding the 2 GiB retained-cache cap as expected for a process that also holds decoded batches and runtime state. Minimum effective/host-available headroom is 54,969,856,000 bytes (51.195 GiB). Swap I/O and sampled ancestor OOM/OOM-kill counter deltas are zero. The closure reports one owned child, PID 507922, confirmed exit and no cleanup signals. These are sampled observations, not a true memory peak or a bound for the future worker pool.

All 107 desktop-policy observations are accepted, with at most 288 MiB admitted desktop memory and no owned GPU compute row. Global utilization reaches 30%; the 5% limit applies only to preflight, so continuous GPU idleness is not claimed. Runtime evidence reports Python 3.12.8, Torch 2.2.2+cu118, PyG 2.6.1, one intra-op and 28 inter-op threads in the serial sizing process. This is not worker telemetry.

The current B5A descriptor remains SHA `3ab036e5f6c58104ccbdfc41d259e02e3f7a9108142c62ff99bfab8001fb6195`, 6,594,901 bytes. The accepted controller/export checks bind those bytes before and after execution, but raw B5A is absent from the archive and is not rehashed by this offline import. Payload objects, live processes and the full sampler permutation are likewise not reconstructed. TEST/VALIDATION access observations stay `null`, separately from the preserved closed access policy. No imported code, model or checkpoint was executed during validation.

After this import is sealed, proceed to the authorized single three-arm series: four workers with compact cache, zero workers with compact cache, then the original zero-worker path. It must check actual serialization bounds, full update equivalence, observed worker/runtime and aggregate memory, and measured throughput. RES004 supplies the compact reservation evidence; neither its wall time nor its cache hit fraction establishes a production speedup.

Evidence anchors:

- Source record fingerprint: `9f418f5bd23237b4b2c636907a7f9ba36ee7e898019d7bec4ffbabcf0fd6ae37`.
- Archive: `outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-013/gpu-evidence-v13.tar.gz`; SHA `6bb092b81a22174135245a5dacbf0c250f3bc134710631cbcf0d38b7d77520da`, 341,419 bytes.
- Independent verification: `outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-013/verification-attempt-001/verification.json`; SHA `698dadfa599212c81b98a06b3cc76ea09bad7d146b347a2ae460c8064d34f44b`.
- Accepted receipt SHA: `8e6d49cd3614ef918204ebf1bd740bbc6ba521ab41c829a576e47c90a5b1f4c0`.
- Reservation-manifest fingerprint: `29ede49da3fd77053a5583f9eac497450889b15ff92a4c78fbdb8d9e04dbbf1f`.
- Final preparer source fingerprint: `ec322951bdde4c88ac4fe04be38818084fd7ddb40cc751441efeb8db1db6b635`.
