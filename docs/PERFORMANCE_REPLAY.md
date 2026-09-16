# C0 performance replay — Phase 9E-P1

## Latest user scope — 2026-09-07

The latest user instruction adds an entirely unoptimized baseline. The next
GPU comparison is **one series: integrated workers4, integrated workers0,
then original baseline workers0**, superseding the two-arm and three-repeat
continuations. Each arm uses B2, 400 measured plus 80 discarded warmup updates;
the original 120k scheduler remains unchanged. Total requested work is
1,440 updates / 2,880 draws. The first two arms combine compact prepared cache,
direct collation and optimized update, with one 8 GiB total accounted cache
budget: 4 x 2 GiB versus 1 x 8 GiB. Record cache partitioning consequences;
their comparison does not isolate workers from cache topology.

The third arm must use original loading/collation/identity preparation,
original class weights and the existing reference `attempt_applied_update`,
without prepared cache, direct-collation fast path or worker pool. Reuse the
baseline path already present in `performance_replay.py`; do not rewrite
reference mathematics. The existing prefetch worker-zero arm still calls the
optimized update and cannot serve as this baseline. Keep one initial state,
ordered data stream, source-population policy and direct timing boundary for
all three arms. Do not retain unused prepared caches in the original arm.
No additional no-cache worker pair or three repetitions are requested.

The integrated path is implemented, independently reviewed and covered by
focused equivalence/lifecycle tests. Its single GPU comparison is completed;
explicit post-hoc review015 accepts a bounded speed/equivalence result while
the original controller failure remains preserved. Old defaults are unchanged. Do not replicate a full cache in
every worker or retain an unused parent cache. Record actual aggregate RSS,
cache hits/misses/evictions, and directly timed whole-arm duration alongside
the steady-state window. New preparation strategy/limits require explicit
source and reservation bindings before real execution.

One fixed-order series gives a bounded practical comparison, not the earlier
three-repeat acceptance or a full-120k duration guarantee. Compare workers4
against integrated workers0 for marginal worker/topology effects, and each
optimized arm against original baseline for total measured package gain.
Do not multiply earlier cache/update/prefetch ratios. Completed sizing remains immutable and is reused
only where its exact input and payload contracts still apply.

## Accepted bounded result — 2026-09-07

The one authorized three-arm series is complete: workers4 / compact workers0 /
original workers0 achieved 1.437335 / 1.193026 / 0.949171 measured updates/s,
with directly timed whole arms of 377.016 / 442.330 / 533.293 seconds.
Workers4 gives 1.5143x steady throughput and 29.30% less whole-arm time against
the original reference. This is one cold prepared-cache run in fixed order,
not a full-120k duration estimate or a multiplication of historical ratios.

All1200 measured rows, state/RNG/loss/schedule fingerprints, cache/queue/resource
limits and remaining postflight checks passed explicit post-hoc review015.
The original V14 controller remains failed: its expectation that cuDNN TF32 was
disallowed contradicted historical and current final runtime snapshots and
was not set by the frozen helper. Negative014 and the failed assertion/test
fixture are preserved. No current pre-run TF32 snapshot exists; no pre/post
continuity or all-TF32-disabled execution is claimed. No training was repeated.

Retain workers4 with8GiB total compact-cache accounting as the measured candidate.
Do not reuse the erroneous V14 false-TF32 expectation for a future controller.
Record actual runtime policy explicitly before subsequent work; nonidentity C2
and scientific-runner adoption remain separate steps. No further GPU work is
scheduled by this completed comparison. Detailed immutable results are in
[the post-hoc review](experiments/PERFORMANCE_REPLAY_INTEGRATED_POSTHOC_REVIEW.md).

## Integrated implementation and target gate — 2026-09-07

The opt-in integrated path is implemented and independently reviewed. It keeps
the original defaults. A single explicit invocation selects compact preparation,
8GiB aggregate entry accounting, worker order4/0 and the additional original
reference arm. The three arms have distinct identifiers; the original control
uses raw class weights and the existing reference update, not the optimized
worker-zero step. All arms restore the same initial training state before
warmup and measured execution while keeping the80-job warmup cursor.

Prepared caches begin empty in every arm, are lazy and process-local, and
remain across the warmup-to-measurement boundary. Worker4 partitions the entry
budget into4x2GiB; direct0 receives8GiB; original0 has no prepared cache. No
population-wide prepared-cache prewarm is performed. Whole-arm time is a direct
elapsed interval from fresh setup through worker closure and cache release.
The measured window and the broader supervised-execution interval remain
separate. Cache hit/miss/eviction/bypass counters and sampled RSS are retained.

The local combined gate passes145tests; the target passes302tests with no
failures/skips in98.64s. The isolated current256 source map is
`9d6dc4df7356924caaae62692dc99199c0995c1260550d0e2b272228296f49c3`;
exactly five opt-in files replaced their preserved prior bytes. Canonical
repositories and the frozen training artifacts are unchanged.

Original-preparer RES003/import012 are completed and immutable. Their480jobs
cover690TRAINrecords, with a51.404MiB maximum payload and705.82MiB sampled
RSS. These are original-preparation observations, not compact-cache readiness.
Compact sizing004 completed at03:52:54UTC with0updates and0poolworkers;
import013 independently accepted all480jobs, the currentsource and302targettests.
The single2GiBcache reported122hits/838misses, maximumsampledRSS3.158GiB
and minimumheadroom51.195GiB. This is neither a four-worker memory nor speed result.

The three-arm GPU comparison `EXP-9EP1-PREFETCH-INTEGRATED-001` started
at03:59:36UTC after60additional targetcontroller/producer tests passed in3.66s.
It uses exactly80warmup+400measured updates for each of integrated4,integrated0
and original0. Import014 subsequently retained the completed source and failed controller;
post-hoc015 completed the independent numerical/timing review on identical bytes. Whole-arm and supervised elapsed timers are direct measurements,
not sums of overlapping phase timers. No extra calibration or repeat is scheduled.

Synthetic checks found equal payload values with different pickle sizes across
cache histories (including70340 versus69745bytes in one repeated-record case).
No alias normalization or false byte-independence claim is introduced. Sizing
reports its actual history and policy; every transported payload must still
fit its declared reservation before it is sent. Cache entry accounting and
serialized reservations are not physical RAM bounds; the existing RSS,
headroom and IPC guards remain independent.

## Explicit resumption — 2026-09-07 MSK

The user reported recovery and requested continuation, superseding the pause
below. RECOVERY001 and the bounded desktop preflight are complete. The V11
target gate passed 114 tests; SCREEN002 ran from 2026-09-06 23:51:58 through
2026-09-07 00:04:32 UTC. Independent import011 accepted its exact four-arm
result: four workers improve the warm window by 32.95%, while the sum of
reported setup, warmup, cleanup and measured phases improves by 11.91%.
Fresh longer sizing and three repeats were proposed after that screen; the
latest user scope above replaces that continuation. This is not production
training acceptance. See [worker screen review](experiments/PERFORMANCE_REPLAY_WORKER_SCREEN_REVIEW.md).
Its versioned desktop policy and explicit optional producer callback
preserve numerical execution; old receipts remain historical evidence. A
transient network failure gets one bounded probe, then the configured
ten-minute heartbeat. Report setup/cache/worker costs separately from warm
throughput; the existing 2.9x already includes compact cache and update changes.

The local `performance_c2_train_schedule.py` module prepares the subsequent
12-job shift coverage and independent 100-update historical orbit prefix.
It authenticates declared TRAIN metadata and selects only eligible shifts;
it does not read raw payloads, measure reservations or execute updates.
Root tests: 37 passed, two upstream Torch warnings. A production caller must
bind externally accepted metadata/descriptor hashes and use the production
re-verifier; internal descriptor consistency alone does not authenticate data.
The dedicated C2 sizing and real observation/resume runner remain pending.

## User pause — 2026-09-06 16:44 UTC

All work is paused until an explicit new user message resumes it. The
scientific coordinator paused automation `music-critic-v2`; the saved local
configuration confirms `PAUSED`. Do not reconnect, probe, transfer files, run
tests or experiments, or continue local implementation while paused. Server
availability alone does not resume authorization. The earlier continuation
instructions below describe the deferred plan only. The preserved stopping
point and incident evidence are in `STATUS.md` and
`outputs/phase9ep1/coordination.json`; no completed evidence is changed.

This is an engineering experiment, authorized on 2026-09-06. It does not
change the canonical C0/C2 runs, selected baseline, model architecture, graph
features, targets, losses, or scientific training budget. Legacy code is not
used. TEST and VALIDATION inputs/targets are closed in this replay.

## Acceptance and scope

First compare the frozen implementation with an explicitly selected optimized
implementation at batch 2. The model, optimizer, LR schedule, clipping, seed,
dropout RNG, ordered draws, scheduled shifts and applied identity shifts stay
the same. Require exact same-runtime state and loss/RNG fingerprints, plus
unit checks of logits, gradients, alignment and checkpoint/resume. A numerical
mismatch is a failed equivalence gate, not a reason to relax a fingerprint.

The desired production outcome is at least a 20% median records/s improvement
over three interleaved repeats on the same idle GPU, with all equivalence gates
passing and no memory/resource regression that threatens the largest TRAIN
graphs. This is a predeclared engineering target, not a promised speedup.
Synthetic CPU results establish only fixture mechanics and local performance.

Historical modules remain byte-unchanged. Opt-in performance modules own the
new preparation and update paths. Full GPU graph validation, per-sample GRU,
loss equations and every-update finite/clip guards remain enabled initially.
Do not mix the newer feature-complete transposition view into this comparison.
No mutation of canonical run outputs, checkpoints or source caches is allowed.

## Measurements

Record source hashes (including dirty source files), environment, exact argv,
phase/UTC/monotonic timestamps, record schedule, data binding seal, graph sizes,
loss/RNG/final training-state fingerprints and update outcomes. Use fresh ignored
output directories and a planned experiment record before each substantial run.
Finalize completed, failed, invalid or aborted runs and regenerate the ledger.

Measure preparation, collation, identity transform, H2D, forward, alignment,
loss, backward, finite/norm/clip diagnostics, optimizer and logging separately.
Detailed CPU/CUDA traces are a separate diagnostic pass; unprofiled throughput
is the decision metric. Nested/overlapping stage times must not be added as if
they partitioned the critical path. CUDA events may include host launch gaps.
Synchronize at window boundaries, not after every stage. Report validation and
checkpoint overhead separately from TRAIN throughput; replay measures neither
validation quality nor amortized full-run duration.

## Equal-draw pilot

Use 800 identical measured draws: B2=400, B4=200, B8=100 applied updates.
Use the same declared warm-up draw count, then restore the common model,
optimizer, scheduler, scaler and RNG snapshot before timing. Warm preparation
caches separately with exactly the measured draw population and disclose that
policy; materialization cost is reported outside the steady-state window.
Rotate arm order across three repetitions. Never select/reorder works for speed.

Report records/s, nodes/s, edges/s, p50/p95 step time, peak allocated/reserved
VRAM, process CPU time/utilization and RSS, and actual coarse-token padding.
Capture external utilization during the separate GPU replay when available;
missing utilization observations stay unavailable. Test largest TRAIN graphs
separately before accepting a production batch configuration. OOM is retained
as a failed arm. A fixture is never evidence for production memory capacity.

B4/B8 use a new throughput-only protocol. Equal draws do not preserve optimizer
steps, gradient noise, batch boundaries or LR trajectory. No quality conclusion
or scientific-seed substitution is permitted.

## Optimization order

1. Seal split membership and warm source bytes; bounded immutable CPU graph
   cache. Check source identities on use and full content seals at run/resume
   boundaries. Stat checks are not cryptographic integrity proofs.
2. Use CPU candidate metadata after predictions exist; coalesce finite and
   per-parameter norm transfers without changing reduction order; keep class
   weights resident on the device.
3. Direct collation and identity fast path with owned batch tensors and
   consumer-mutation tests.
4. Add bounded ordered prefetch only if remaining timing justifies it. Current
   canonical cache writers use shared temporary names, so concurrent cold-cache
   writes are forbidden. Sampler advances only after applied updates.
5. Consider pinned/asynchronous H2D and GRU packing as later measured variants.
   They are not silently enabled. Never remove Dropout-consuming branches just
   because their outputs appear unused.

## Operational boundary

No profiler is attached to the live canonical process. Production replay needs
an independently verified idle CUDA device, sealed warm TRAIN cache and exact
source/schedule bindings. The local Mac has no CUDA; server access and corpus
availability must be verified rather than inferred from the user's earlier
process snapshot. Existing C0 completion status is not changed by this work.

## First accepted engineering evidence

`EXP-9EP1-PREFLIGHT-001` verifies two identical server source checkouts and the
actual older Torch 2.2.2 runtime of live C0. A separate checkout now contains the
opt-in replay; the same runtime passes 94 synthetic CPU tests (5 CUDA skips).

`EXP-9EP1-CPU-FIXTURE-001` completed three rotated repeats, 800 draws per arm.
Median fixture records/s are 20.1267 (reference B2), 40.4289 (optimized B2),
70.9255 (optimized B4), and 111.7123 (optimized B8). Every paired B2 full-state,
loss-trace, RNG and schedule fingerprint is identical. This approximately
2.009x CPU-fixture improvement does not establish real-graph CUDA throughput.
The entire run took 278.95 seconds including all arms and warm-up.

The queued GPU stages first establish fixture CUDA equivalence, then collect a
separate 16-draw diagnostic trace, then run the 800-draw production benchmark.
They retain C0's actual Torch runtime. Bounded cache state is reset and prewarmed
in the same record order before every arm; hit/miss/eviction statistics and
accounted cache bytes are retained. Process RSS is a lifetime high-water mark;
reserved CUDA allocator memory can include earlier arms. Neither quantity is
claimed to be a fresh-process isolated peak.

No production B4/B8 configuration is accepted merely because this selected
draw stream fits. Full-corpus worst-case size/OOM checks remain a separate gate.

## First GPU queue and linked remediation

`EXP-9EP1-GPU-FIXTURE-001` passed exact B2 CUDA state/loss/RNG/schedule
equivalence. Its short synthetic window is not a production throughput claim.
`EXP-9EP1-GPU-DIAGNOSTIC-001` failed during prewarm because the opt-in loader
mistook the intentionally neutral `CanonicalPiece.split=None` for missing
TRAIN authorization. The existing contract puts that authorization in the
frozen B3 assignment and cache metadata, both checked before payload decoding.

The reviewed correction accepts neutral or explicit TRAIN canonical pieces,
preserves the original piece and rejects explicit non-TRAIN values. The
regression fixture now reflects the real adapter's neutral split. All 123
profile tests pass in the same GPU runtime, including the CUDA-specific gates.
The first queue's original terminal records/artifacts and controller manifest
were verified and retained by `EXP-9EP1-GPU-EVIDENCE-001`; negative evidence
and unknown parent access observations are immutable.

Retries use the optional `--parent-experiment-id` to bind a validated sealed
terminal parent before the child is planned. `EXP-9EP1-GPU-DIAGNOSTIC-002`
retries diagnostic-001 with this correction; the first production throughput
attempt follows only after it passes. Always resolve the current controller
and connection from ignored `outputs/phase9ep1/coordination.json`, not from a
historical PID or queue directory. Original source caches remain read-only.

## Negative production result and update isolation

`EXP-9EP1-GPU-THROUGHPUT-001` passed exact equivalence in all three B2 pairs
but missed the speed target: median reference 1.649032 versus prepared-cache
1.192079 records/s (0.722896x). The original terminal result remains completed
negative evidence. `EXP-9EP1-GPU-EVIDENCE-002` verified its full artifacts and
those of diagnostic-002. See the detailed GPU review in `docs/experiments/`.

The opt-in `optimized_data_strategy=frozen` keeps reference loading,
collation, identity transformation and H2D in both arms, changing only update.
It never constructs or prewarms a prepared graph cache. Both arms prewarm the
same frozen source population and restore all training state after warmup.
The distinct cache/prewarm policy and each arm's data/update strategies are
recorded. The default `prepared_cache` retains the earlier experiment path.
Explicit strategy fields in a registered protocol must match returned fields;
missing or conflicting bindings fail the attempt rather than sealing success.

`EXP-9EP1-GPU-UPDATE-ONLY-001`, linked to throughput-001, is an unprofiled
diagnostic isolation: B2, 800 measured and 160 warmup draws per arm, one repeat.
Its planned totals are 1,600 measured draws, 320 warmup draws and 960 applied
updates including warmup. It cannot satisfy the three-repeat acceptance by
itself. The original >=20% target and memory/equivalence requirements remain.

The diagnostic is now complete: **1.856601 → 2.185625 records/s**, exact B2,
**17.72%** higher throughput in one pair. Verified immutable import:
`EXP-9EP1-GPU-EVIDENCE-003`. It does not meet the three-repeat acceptance.
Frozen loading remains about 267 seconds per 800 draws and dominates the
optimized measured window. The next cache candidate retains original verified
sidecar bytes while preserving fresh consumer dictionaries and graph ownership.

The next prepared-cache package protocol must construct no unused candidate
cache for baseline arms, release previous arm caches outside measurement, and
prewarm the same source population. Only optimized arms retain a fresh bounded
LRU. Record this policy explicitly and test optimized-first order as well as
baseline-first. The older default policy described above is historical; its
terminal records and artifacts must not be rewritten when this change lands.

This policy is now implemented and passed 157 tests on the actual CUDA
runtime. `EXP-9EP1-GPU-THROUGHPUT-002` started at 11:46:58 UTC: three rotated
B2 comparisons of the compact-cache-plus-update package, with unchanged
800 measured / 160 warmup draws and 8 GiB accounted cap. Its declared policy
is `global_frozen_prewarm_baseline_no_cache_optimized_fresh_bounded_LRU`.
The next import is planned as `EXP-9EP1-GPU-EVIDENCE-004`.

## Next authorized pipeline experiment: ordered CPU prefetch

The user explicitly requested CPU workers preparing batches while the parent
executes GPU updates. Develop this locally while the current remote replay
remains unchanged. Screen 0/1/2/4 workers on the same frozen-data plus optimized
update implementation, separating startup from steady-state measurement;
then use a new registered three-repeat gate for the selected configuration.
Do not change compact-cache strategy simultaneously or attribute cross-run
differences to prefetch alone.

Workers may prepare sealed warm TRAIN graphs, historical CPU transpositions
and collation. CUDA/model/optimizer remain in the parent. Fixed draw IDs,
record IDs, shifts and B2 boundaries must return in strict schedule order.
Only an applied update commits the sampler position; speculative jobs do not
consume committed state. Resume discards pending jobs and rebuilds them from
the committed cursor and sealed schedule/source/config identity. Worker
startup must preserve parent dropout RNG, including in the zero-worker path.

The runtime-observation variant adds a versioned ready acknowledgment
from every spawned worker before dispatch. It records actual Torch intra-op
and inter-op getters, PID/worker identity, source/runtime identity, inherited
native thread environment and the environment after historical setters.
Production workers require all four native environment limits to be one
before spawn; the launcher must set them before Python imports. Startup
readback is not a measurement of already-created native pools or continuous
telemetry. Direct worker0 has no worker observations. Separate runtime
artifacts do not enter model traces, serialized batch payloads or saved cursors.
This variant changes preparation/implementation fingerprints and therefore
requires new exact-job reservations. It was kept outside the earlier v8
memory run; the later deployed variant and refresh are accepted below.

Bound pending and in-flight jobs and serialized-payload reservations across
all workers/queues. No per-worker 8 GiB caches. Accounted buffers do not bound
worker peak RSS or IPC copies; measure these separately and require credible
reservations before real-data launches. Limit nested CPU threads, propagate
worker errors, detect dead workers, and terminate only owned children during
cleanup. Order/state/RNG/resume and CUDA equivalence tests precede substantial
runs. A pure memory-index helper does not substitute for a registered complete
TRAIN index or measured worst-case memory gate.

The integrated local replay uses a direct zero-worker control with no pickle
encoding/decoding. A completed registered reservation gate must authenticate
the exact schedule, source bindings and serialized sizes before production
workers can start. The provisional per-job reservation remains unchanged
after sizing. Queue reservations cover executing and ready jobs until commit;
they do not include all decoded objects, IPC copies or process memory.

Predeclare the first screen as B2, 200 measured and 40 warmup draws per arm,
one repeat, workers 0/1/2/4. The existing production plan requires at least 100
measured updates; do not weaken that guard for a short screen. All arms use
the original frozen preparation and optimized update. Warmup restores the
common full training snapshot while retaining speculative CPU preparation;
this measures steady state, not cold-start latency. Record startup/prewarm
separately and retain worker CPU/RSS/thread observations and transport bytes.
The measured queue stage includes wait, decoding and dispatch; it cannot
attribute pure IPC overhead independently.

The local integrated test gate passed 337 tests with five CUDA skips. Before
real-data screening, run the full-model logits/state/RNG/serialized-resume
fixture explicitly on the target CUDA runtime, then the registered full TRAIN
size index and exact scheduled payload-size gate. Native thread limits must
be set before starting Python (OMP/MKL/OPENBLAS/NUMEXPR); requested limits are
not a substitute for the recorded thread observations. A separate newly sized
800 + 160 draw schedule is required for the selected configuration's three
repeat gate. Fresh-process extreme B2 memory checks and historical C2-path
equivalence remain separate acceptance requirements.

## Telemetry deployment and accepted reservation refresh — 2026-09-06

`EXP-9EP1-PREFETCH-RESERVATIONS-002` and independent `EVIDENCE-009` are
completed. The two-module telemetry overlay preserves the other 254 source
files, original loader/index bindings and the 120-job screen schedule.
All 120 reported payloads fit the unchanged reservation policy; maximum
41,565,463 bytes and total 1,874,422,579 bytes match the earlier sizing.
This sequential measurement performs no model updates and starts no worker
pool. See the sealed
[reservation refresh review](experiments/PERFORMANCE_REPLAY_RESERVATION_REFRESH_REVIEW.md)
for source fingerprints, original-runtime observations and evidence limits.

The target project gate passed 312 unchanged tests after an explicit
PyG-before-pytest bootstrap resolved a test import-order side effect. The first
302-pass/10-fail attempt is preserved. The complete TRAIN index, synthetic C2
restoration and finite native B2 memory evidence remain accepted through
imports005,007,008; do not repeat these prerequisites without a new reason.

The immediate next substantial run is the one-repeat `PREFETCH-SCREEN-001`
defined above. A narrow controller reuses the reviewed V9 process supervisor
and independently monitors host/cgroup headroom, sampled owned RSS and swap
I/O. Its 44 local tests pass; the initial target 43/1 result is solely a
missing approved proposal fixture, to be delivered before a separate retry.
The screen remains unlaunched after SSH became unreachable at 16:26 UTC.
Resume with a checked fixture transfer, those same 44 tests and fresh host
preflight; no additional preparation experiment is introduced. The
authoritative live-state snapshot is `outputs/phase9ep1/coordination.json`.
