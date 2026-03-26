# R20: NCCL Backend

**Phase:** 7 — Memory Intelligence
**Status:** PRE-PLAN COMPLETE
**Priority:** CRITICAL
**Depends On:** P6 (Core Transport working)

## Summary

Register OuterLink as a custom NCCL transport backend (`libnccl-net-outerlink.so`). This gives instant compatibility with every ML framework that uses NCCL: PyTorch, TensorFlow, JAX, DeepSpeed, Megatron-LM. Zero code changes needed in user applications — just set `NCCL_NET=outerlink` and distributed training works.

## What This Enables

- PyTorch `torch.distributed` works over OuterLink out of the box
- AllReduce, AllGather, Broadcast, ReduceScatter over OuterLink transport
- Leverages all OuterLink optimizations (OpenDMA, compression, multi-path)
- THE way 90% of ML users will interact with OuterLink

## Key Questions (Answered in Research)

- **Which NCCL versions to support?** Target v8 API (NCCL 2.19+) as primary, with shim layers for v9/v10/v11. See `research/01-nccl-net-plugin-api.md`.
- **NCCL net plugin API — what's the exact interface?** 19 function pointers in ncclNet_v8_t struct. Fully documented in `research/01-nccl-net-plugin-api.md`.
- **Can we expose multi-transport (ConnectX + USB4) through NCCL?** Yes — report each transport as a separate NCCL device. NCCL distributes channels across devices. See `research/03-nccl-topology-and-collectives.md`.
- **Performance parity with native NCCL over InfiniBand?** Achievable with RDMA/OpenDMA paths. Plugin overhead must be <2us per operation. See preplan.md validation criteria.

## Folder Contents

- `research/01-nccl-net-plugin-api.md` — Complete API surface (v4-v11), function signatures, lifecycle
- `research/02-existing-nccl-plugins.md` — Survey of 5 existing plugins with lessons learned
- `research/03-nccl-topology-and-collectives.md` — How NCCL uses transport internally
- `side-docs/` — Notes, experiments
- `preplan.md` — Scope, dependencies, 5 decisions, risks, 6 implementation phases
- `plan.md` — TO BE CREATED (next step after decisions are made)
- `progress.md` — Lifecycle tracker

## Related Topics

- R14 Transport Compression (compressed collectives)
- R29 RDMA Multicast (hardware-accelerated broadcast)
- R17 Topology-Aware Scheduling (NCCL ring/tree topology)
