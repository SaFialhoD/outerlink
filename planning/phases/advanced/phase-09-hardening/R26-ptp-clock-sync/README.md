# R26: Hardware Clock Sync via PTP

**Phase:** 9 — Hardening
**Status:** RESEARCH COMPLETE
**Priority:** HIGH
**Depends On:** P8 (Performance phase working), R17 (Topology-Aware Scheduling)

## Summary

Use ConnectX-5's hardware PTP (Precision Time Protocol) timestamping to synchronize clocks across nodes to sub-microsecond precision. Enables coordinated GPU kernel launches ("all GPUs start at T+0") and precise distributed profiling.

## What This Enables

- Sub-microsecond clock alignment across all cluster nodes
- Coordinated kernel launches for pipeline parallelism
- Accurate distributed profiling and latency measurement
- Foundation for cooperative kernel splitting (R25)

## Key Findings from Research

| Topic | Finding |
|-------|---------|
| ConnectX-5 PTP accuracy | <100ns typical offset with hardware timestamping over L2 |
| Implementation approach | linuxptp (ptp4l + phc2sys) — battle-tested, handles all edge cases |
| Grandmaster selection | BMCA with topology-driven Priority1 assignment |
| GPU clock integration | Three clock domains (PHC, system, GPU). GPU timer drifts, needs periodic calibration |
| Coordinated launches | Host-side PTP scheduling (~5-20us jitter) or hybrid with GPU spin-wait (~1-2us) |
| Rust PTP option | statime crate exists but less mature; evaluate later |
| Kernel launch jitter | ~5-20us dominates over PTP sync error (~100ns) |

## Key Decisions

1. **Use linuxptp** (ptp4l + phc2sys) as PTP daemon, not custom Rust implementation
2. **L2 transport** for lowest jitter on dedicated RDMA network
3. **Host-side PTP scheduling** for coordinated launches initially; GPU spin-wait for R25
4. **Topology-driven grandmaster selection** — coordinator node gets Priority1=100

## Implementation Phases (5-8 weeks total)

1. **PTP Infrastructure** (2-3 weeks) — ptp4l/phc2sys management, health monitoring, grandmaster selection
2. **Clock API & Profiling** (1-2 weeks) — Rust ClockSync module, GPU-to-PTP calibration, distributed timestamps
3. **Coordinated Launches** (1-2 weeks) — schedule_at() API, launch jitter measurement, optional GPU spin
4. **Hardening** (1 week) — Failover testing, drift verification, performance impact measurement

## Success Criteria

| Metric | Target |
|--------|--------|
| Cross-node clock offset | <1 microsecond sustained |
| Coordinated launch jitter (host-side) | <20 microseconds |
| Coordinated launch jitter (GPU spin) | <5 microseconds |
| GPU-to-PTP calibration accuracy | <2 microseconds |
| PTP convergence time (cold start) | <2 minutes |

## Folder Contents

- `research/01-ptp-protocol-and-hardware.md` — IEEE 1588 PTP, hardware timestamping, ConnectX-5 PHC capabilities
- `research/02-linux-ptp-stack.md` — linuxptp (ptp4l/phc2sys), configuration, BMCA, servo tuning, Rust options
- `research/03-gpu-clock-integration.md` — GPU clock domains, CUDA events, %globaltimer, coordinated launch strategies
- `preplan.md` — Scope, dependencies, decisions, risks, implementation phases, success criteria
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments

## Related Topics

- R25 Cooperative Kernel Splitting (needs synchronized clocks for cross-GPU kernel coordination)
- R30 Persistent Kernels (timing coordination for network-fed data handoff)
- R17 Topology-Aware Scheduling (provides node roles for grandmaster selection, uses PTP for latency measurement)
- R20 NCCL Backend (optional PTP-synced profiling of collective operations)
