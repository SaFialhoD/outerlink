# R16: BlueField DPU Offload

**Phase:** 10 — Compute Distribution
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P8 (Performance phase working)

## Summary
Offload OuterLink's transport logic to BlueField DPU's ARM cores. The DPU runs routing decisions, compression, and memory management at the network edge — the host CPU becomes almost unnecessary for data movement.

## What This Enables
- True zero-CPU data movement
- Routing decisions at line rate on the NIC
- Compression/decompression without host CPU
- Host CPU freed entirely for application work

## Key Questions
- BlueField programming model: DOCA SDK, raw DPDK, or custom?
- ARM core performance sufficient for transport logic?
- How to split work between host and DPU?
- BlueField-2 vs BlueField-3 capabilities?

## Folder Contents
- `research/` — BlueField architecture, DOCA SDK, ARM programming
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R14 Transport Compression (compression offloaded to DPU)
- R17 Topology-Aware Scheduling (routing on DPU)
- R9 Multi-Transport (DPU manages transport selection)
