# Server recovery and desktop launch policy — 2026-09-07 MSK

`EXP-9EP1-RECOVERY-001` completed a registered read-only host inspection.
The server remained in the same boot since 2026-09-05 19:10:43 UTC; uptime
at the snapshot was 95,142.77 seconds. Thus no intervening reboot explains
the earlier SSH outage.

The explicitly UTC kernel window 2026-09-06 16:20–16:30 contains one TUN
interface event and no OOM, NVIDIA Xid, or panic entry. NetworkManager records
VPN route changes at 16:23:27, activation of a TUN interface at 16:23:56,
L2TP/IPsec connection and permission failures, PPP hangup at 16:24:43, and a
VPN connection timeout at 16:25:41. These observations support a coincident
network incident, but do not establish the complete cause of the SSH failure.
No network configuration was changed by the performance task.

At 21:36:25 UTC the host had 56,927,780,864 bytes available RAM out of
66,108,309,504, 5,136,384 bytes swap used, zero recent memory/CPU pressure
averages, and 498,637,615,104 bytes disk availability. No known repository
workload was found. All 256 receipt-pinned source hashes matched. Python
3.12.8, Torch 2.2.2+cu118, and PyG 2.6.1 matched package metadata; this was
not a CUDA model test. Cgroup limits and swap-I/O deltas are checked separately
by the existing launch/resource monitor.

The original launch guard rejected a new desktop Nautilus GPU context. The
coordinator authorized a narrowly versioned desktop policy for the unlaunched
screen. V9/V10 source and receipts remain unchanged, as do producer, model,
loader, and the accepted RESERVATIONS002 source/schedule/payload bindings.
The unlaunched EVIDENCE010 was retained as aborted; SCREEN002 / EVIDENCE011
identify the new launch policy. SCREEN001 never launched.

`EXP-9EP1-DESKTOP-PREFLIGHT-001` then registered five observations one second
apart: GPU utilization was 0% in all five, GPU reported free memory 23,407 MiB,
Nautilus held 28 MiB and the admitted remote-desktop service held 260 MiB.
Nautilus PID, executable, UID, start ticks and boot identity were checked.
The 288 MiB sum describes admitted desktop compute contexts, not all GPU
memory: total reported GPU memory use was 717 MiB. This short observation
supports an explicit desktop exception, not a guarantee of constant future
GUI activity. The new screen must retain background identity/memory monitoring
alongside the existing CPU-memory, ownership and exact-equivalence checks.

Full primary evidence is ignored under the two experiment output directories;
compact terminal records carry hashes and claim limits. No model update,
TEST/VALIDATION access, canonical checkout mutation or legacy reuse occurred
in these diagnostics. Worker throughput remains to be measured.

Subsequent full call-chain review found two additional calls to the original
GPU guard inside `run_prefetch_replay`. The coordinator authorized an explicit
optional policy callable at those boundaries, with the old default retained.
This requires one successor orchestration source hash and a new execution
receipt, rather than claiming unchanged full source256. Numerical functions,
loader/preparer code and RESERVATIONS002 bindings remain unchanged; their
compatibility must be checked before launch. No screen was launched with the
incomplete controller-only adjustment.
