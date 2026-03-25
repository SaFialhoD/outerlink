# R20: NCCL Backend

**Phase:** 7 — Memory Intelligence
**Status:** NOT STARTED
**Priority:** CRITICAL
**Depends On:** P6 (Core Transport working)

## Summary

Register OuterLink as a custom NCCL transport backend (`libnccl-net-outerlink.so`). This gives instant compatibility with every ML framework that uses NCCL: PyTorch, TensorFlow, JAX, DeepSpeed, Megatron-LM. Zero code changes needed in user applications — just set `NCCL_NET=outerlink` and distributed training works.

## What This Enables

- PyTorch `torch.distributed` works over OuterLink out of the box
- AllReduce, AllGather, Broadcast, ReduceScatter over OuterLink transport
- Leverages all OuterLink optimizations (OpenDMA, compression, multi-path)
- THE way 90% of ML users will interact with OuterLink

## Key Questions

- Which NCCL versions to support? (2.18+? 2.21+?)
- NCCL net plugin API — what's the exact interface?
- Can we expose multi-transport (ConnectX + USB4) through NCCL?
- Performance parity with native NCCL over InfiniBand?

## Folder Contents

- `research/` — NCCL plugin API docs, existing plugin examples
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R14 Transport Compression (compressed collectives)
- R29 RDMA Multicast (hardware-accelerated broadcast)
- R17 Topology-Aware Scheduling (NCCL ring/tree topology)
