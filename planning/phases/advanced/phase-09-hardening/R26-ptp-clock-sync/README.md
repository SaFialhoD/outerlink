# R26: Hardware Clock Sync via PTP

**Phase:** 9 — Hardening
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** P8 (Performance phase working)

## Summary

Use ConnectX-5's hardware PTP (Precision Time Protocol) timestamping to synchronize clocks across nodes to sub-microsecond precision. Enables coordinated GPU kernel launches ("all GPUs start at T+0") and precise distributed profiling.

## What This Enables

- Sub-μs clock alignment across all cluster nodes
- Coordinated kernel launches for pipeline parallelism
- Accurate distributed profiling and latency measurement
- Foundation for cooperative kernel splitting (R25)

## Key Questions

- PTP grandmaster selection: which node is the clock source?
- Hardware vs software timestamping on ConnectX-5?
- Integration with CUDA events timing?
- Drift compensation for long-running workloads?

## Folder Contents

- `research/` — PTP protocol, ConnectX PTP support, linuxptp
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R25 Cooperative Kernel Splitting (needs synchronized clocks)
- R30 Persistent Kernels (timing coordination)
- R17 Topology-Aware Scheduling (latency measurement uses PTP)
