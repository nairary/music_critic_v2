# Ordered prefetch CUDA fixture and complete TRAIN index — 2026-09-06

`EXP-9EP1-PREFETCH-CUDA-FIXTURE-001` and
`EXP-9EP1-TRAIN-MEMORY-INDEX-001` completed in the isolated RTX 3090 checkout.
The queue ended at 13:40:33 UTC. Independent offline import is
`EXP-9EP1-GPU-EVIDENCE-005`; original terminal records and artifacts are retained
unchanged. No new scientific training, VALIDATION evaluation or TEST selection
follows from this engineering result.

## CUDA fixture and serialized resume

The actual accepted runtime is Python 3.12.8, Torch 2.2.2+cu118 and PyG 2.6.1.
The preceding twelve-module target gate passed 414 tests, with 40 warnings and
no skips in 88.17 seconds. Its receipt binds the complete 254-file executable
source mapping and the controller.

The registered fixture compares direct worker zero with 1, 2 and 4 CPU workers,
all using original preparation and the optimized parent update. Four arms each
apply one warmup and three measured B2 updates: 16 updates and 32 draws.
The separate uninterrupted/resumed checks add 24 updates and 48 draws. The
combined accounting is **40 applied updates and 80 draws**, while the primary
registry window budget remains 24 measured draws.

All four paths agree on saved training-state, RNG, loss-trace and logit
fingerprints. The resumed paths serialize/load their own synthetic checkpoints
and match uninterrupted execution. Nonzero-worker resume actually discards pending
work: pending counts are 1, 2 and 2 for workers 1, 2 and 4 respectively. Worker
zero prepares directly without queue serialization. This is identity-path
fixture evidence, not historical nonzero C2 equivalence or production speed.

**Review found a restoration coverage gap.** The historical v5 helper calls
`_restore` on the same model/Adam/scheduler/scaler objects that already hold the
correct prefix state. It does not replace or perturb them first. A restore
that does nothing could therefore escape this test. The completed source result
is retained as executed: checkpoint byte roundtrip, queue rebuild/cursor and
exact continuation were exercised. Effective restoration of training state
into a fresh runtime is not established. A separate corrected gate must require
different state before restore and exact complete state after it, with negative
tests that omit model/Adam/RNG restoration. No new run is inferred here.

The importer rechecks 12 measured metric rows, plan/cursor/count/stage bindings,
all four checkpoint byte hashes, and reported exact/resume relationships. It
does not deserialize checkpoints or independently reconstruct model tensors and
logits. Those claims also rely on the reviewed execution code and target tests.

## Full TRAIN index

All **1,295 TRAIN records / 1,209 components** have canonical scalar size rows.
Initial and final source bindings agree; full row membership, per-record byte
bindings, frozen split hash, common versions and derived counts were checked.
Eight deterministic maximum axes select five distinct records and **25 ordered
B2 cases** for the separate fresh-process memory gate.

| Quantity | Complete TRAIN result |
|---|---:|
| Sum of compact prepared-entry accounting | 16,245,576,971 bytes |
| Sum of decoded prepared-entry accounting | 34,767,811,358 bytes |
| Largest compact entry | 76,903,888 bytes |
| Maximum notes in one record | 11,686 |
| Maximum nodes | 18,141 |
| Maximum edges | 140,148 |
| Maximum coarse sequence length | 624 |

The sums are representation accounting, not process RAM requirements or model
activation peaks. They explain why the previous 8 GiB/1,024-entry cache cannot
retain all TRAIN records. Whether a larger cache or multiprocessing gives the
best sustained result still requires a measured comparison and memory evidence.
All indexed graphs have one track; this maximum has many ties and its selected
record follows the declared lexical tie-break, not a speed-based choice.

Some record IDs contain an upstream `an:test:` namespace. Their authorization
here comes from the frozen V2 TRAIN assignment and verified loader bindings;
that text prefix is not the V2 TEST split. The importer itself opens supplied
evidence artifacts, with checkpoints read only for byte hashing, not corpus data.
Historical runner access fields remain explicitly unknown in their records;
the guarded TRAIN source path does not constitute a system-wide I/O trace.

## Evidence and remaining work

Archive SHA-256:
`83ebfbef76e50df18fcbd8b3bb46f3388f29bed605ad1c990ce56e76281cfbbc`
(150,121,383 bytes). The safe archive contains 57 regular files. Verification
covers **37 run artifacts, 17 controller-manifest entries and all 254 source
bindings**. The stdlib-only verifier passed 24 synthetic tests; root rerun:
24 passed in 1.65 seconds. No imported code was executed.

Next, freeze an index-derived provisional job reservation and measure the exact
120 serialized jobs for the 200-measured/40-warmup-draw screen. Use a common
five-job global queue so one delivered batch can coexist with four preparing
or ready batches. Reservations remain unchanged after measurement; they are
not a bound on decoded objects, IPC copies or aggregate RSS.

Before final pipeline acceptance, close the fresh-runtime restoration gap.
Before the production worker screen, inspect finite fresh-process B2 memory
measurements and current host headroom. Compare 0/1/2/4 workers for one screen
repeat, then size a new 800+160-draw schedule for the selected three-repeat
comparison. Historical C2 nonzero-transposition correctness/resume remains a
separate gate. Canonical C0/C2 and the scientific comparison are unchanged.
