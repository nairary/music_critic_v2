# RES003: longer original-preparation reservation sizing

`EXP-9EP1-PREFETCH-RESERVATIONS-003` completed successfully on 2026-09-07. Independent offline validation under `EXP-9EP1-GPU-EVIDENCE-012` confirms that all 480 serialized B2 jobs fit the unchanged reservation policy. This result covers the original CPU preparation path; it does not establish compact-cache integration or GPU throughput.

| Quantity | Verified result |
| --- | ---: |
| Serialized jobs / record draws | 480 / 960 |
| Warmup / future measured jobs | 80 / 400 |
| Scheduled TRAIN records | 690 of the accepted 1,295-record index |
| Sizing wall time | 546.901234 s |
| Controller elapsed time | 549.653775 s |
| Largest serialized payload | 53,900,513 bytes = 51.404 MiB |
| Retained reservation per job, R | 477,102,080 bytes = 455 MiB |
| Largest payload / R | 11.297% |
| Global admitted reservation budget, B | 2,385,510,400 bytes = 2,275 MiB |
| Maximum pending jobs, Q | 5 |
| Cumulative serialized bytes | 8,064,320,588 bytes |

The cumulative byte count adds separate serializations; it is not resident memory. The job reservation integers remain unchanged. The run uses 800 future measured draws and a 160-draw warmup prefix. The first 240 draws and first 120 jobs match RES002 exactly, while the warmup boundary moves from 20 to 80 jobs. The new population and reservation fingerprints were recomputed rather than inherited from RES002.

The archive contains 36 regular files: the immutable terminal source record, 12 sizing artifacts and 23 V12 controller files. The offline check validated the outer SHA/size, exact path allowlist, duplicate/link/traversal rejection, every source artifact and controller manifest binding, all 480 ordered payload rows, ASCII job hashes, typed schedule hash, reservation arithmetic, complete scheduled bindings against the accepted TRAIN index, and the planned → running → completed lifecycle. The declared protocol performs zero model updates and starts no worker pool. The bound target gate reports 32 passing tests with no failures, errors or skips.

The source's initial and final 256-file maps equal the accepted `98b5d0d0b8f2a150731743ae0a8400fc6ad222ab834421a968fee6303c782e26` map. The controller and exporter bind the current B5A eligibility descriptor: SHA `3ab036e5f6c58104ccbdfc41d259e02e3f7a9108142c62ff99bfab8001fb6195`, 6,594,901 bytes. B5A data are absent from the archive, so this offline import does not independently hash their raw bytes or establish their historical identity. The later source deployment is outside this completed run's evidence.

There are 513 resource samples spanning 548.553 seconds. Maximum sampled owned-process RSS is 740,106,240 bytes (705.820 MiB); minimum effective and host-available headroom is 57,775,665,152 bytes (53.808 GiB). Swap I/O page-counter change and sampled ancestor OOM/OOM-kill counter changes are zero. The preserved closure reports one owned sizing process, no cleanup signals and confirmed subtree exit; this offline check does not query live processes.

All 105 desktop-policy samples are accepted. The largest observed admitted desktop allocation is 288 MiB, and no owned GPU compute row appears. Global utilization reaches 9%; the 5% threshold applies only to preflight, so this does not imply a policy failure or continuous GPU idleness. Resource and GPU observations are sampled, not true peaks or universal capacity guarantees. Runtime evidence reports Python 3.12.8, Torch 2.2.2+cu118, PyG 2.6.1, one intra-op thread and 28 inter-op threads; the last value is an observation of this serial sizing process, not worker telemetry.

Payload byte counts and hashes were checked as recorded evidence without deserializing or regenerating payloads. The full sampler permutation was not regenerated independently. TEST/VALIDATION access observations remain `null`, with the original closed access policy preserved. No model, checkpoint or imported source code was executed during validation.

Next, use a separately bound compact-preparation sizing result before the authorized single three-arm series: four workers with cache, zero workers with cache, then the original zero-worker path. RES003 supplies the original-preparation reference schedule and sizes; it does not authorize treating a changed preparation implementation as equivalent.

Evidence anchors:

- Source record fingerprint: `8cfbe975f07f32ea3ea230be46baf60b9039bd136d6286763a7eef8847dfd6b9`.
- Archive: `outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-012/gpu-evidence-v12.tar.gz`; SHA `537e1f8a1334ec87798be6d26acdb7e48588de16757e720cb39b7db9068295cf`, 305,210 bytes.
- Independent report: `outputs/phase9ep1/imports/EXP-9EP1-GPU-EVIDENCE-012/verification-attempt-001/verification.json`; SHA `daba66dbef8104d1151ab318a446ea6b43e1c9a6c6e8994b36882ebf5d33c073`.
- Reservation-manifest fingerprint: `d2229d8f9a0523cf09de2a89a184621f0b636f17a4c0e479767e6604642a3d68`.
- Final preparer source fingerprint: `d27209c83d917861b5907e711855a97b8258c451de45be7ec134e9c26290ca7f`.
