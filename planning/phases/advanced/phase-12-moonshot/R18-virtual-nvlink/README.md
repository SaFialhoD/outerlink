# R18: Virtual NVLink Emulation

**Phase:** 12 — Moonshot
**Status:** NOT STARTED
**Priority:** LOW (highest complexity, highest reward)
**Depends On:** R19 (Network Page Faults), R25 (Cooperative Kernel Splitting)

## Summary
The holy grail: emulate NVLink semantics over the network. Atomic operations across remote GPUs, cache coherency protocol between VRAM pools, unified virtual address space. All GPUs behave as one NVLink-connected system.

## What This Enables
- All GPUs behave as one giant GPU
- Existing NVLink-optimized code works over the network
- Atomic operations across remote GPUs
- True unified address space (not just memory pooling)

## Key Questions
- Coherency protocol: directory-based, snooping, or hybrid?
- Can we emulate NVLink atomics over RDMA atomics?
- Scalability: does coherency work past 4 GPUs?
- Performance gap: NVLink = 900 GB/s, network = 12-22 GB/s

## Folder Contents
- `research/` — NVLink protocol, cache coherency, RDMA atomics
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R19 Network Page Faults (page faults are part of coherency)
- R25 Cooperative Kernel Splitting (kernels span multiple GPUs)
- R26 PTP Clock Sync (synchronized operations)
