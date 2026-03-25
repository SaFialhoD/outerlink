# R23: Heterogeneous GPU Mixing

**Phase:** 10 — Compute Distribution
**Status:** PRE-PLAN COMPLETE
**Priority:** MEDIUM
**Depends On:** P10 (Multi-Node working)

## Summary

Support mixing different GPU generations, architectures, and VRAM sizes in the same pool. RTX 3060 + RTX 4070 + RTX 5090 all contribute. OuterLink schedules work based on each GPU's compute capability, memory size, and bandwidth.

## What This Enables

- No need for identical hardware across nodes
- Upgrade incrementally — add new GPUs without replacing old ones
- Memory-heavy work on big-VRAM cards, compute-heavy on fast-shader cards
- Maximum utilization of all available hardware
- VRAM aggregation across generations (e.g., 24+24+12+12 = 72 GB unified pool)

## Key Findings from Research

- **Performance asymmetry is massive:** RTX 5090 is ~8x faster than RTX 3060 in FP32 TFLOPS. Naive scheduling destroys performance; proportional scheduling is mandatory.
- **CUDA binary compatibility via PTX:** PTX provides forward compatibility across GPU generations via JIT compilation. Pre-warming JIT caches at module load avoids latency spikes.
- **Workload-aware scheduling is critical:** Compute-bound, memory-bound, and capacity-bound workloads need different GPUs. No single metric captures GPU fitness for all tasks.
- **Hybrid normalization recommended:** Static TFLOPS + startup benchmarks + runtime adaptation. Gavel research shows 1.4-3.5x improvement from heterogeneity-aware policies.
- **Driver compatibility matters:** Forward compatibility packages do NOT work on GeForce. All nodes need compatible driver versions.
- **Minimum CC 7.5 (Turing) recommended:** Covers all RTX GPUs, has Tensor Cores, excludes only very old hardware.

## Key Decisions Needed

1. Minimum compute capability (CC 7.5 vs 8.6)
2. Benchmark suite specification for GPU profiling
3. Kernel dispatch strategy (pre-warm JIT vs on-demand)
4. Scheduler integration API (R23 scores + R17 topology scores)
5. Driver version policy (strict vs moderate)

## Folder Contents

- `research/01-gpu-capability-landscape.md` — GPU hardware diversity, specs, CC features
- `research/02-heterogeneous-scheduling.md` — Scheduling algorithms, load balancing, binary compatibility
- `research/03-practical-mixing-scenarios.md` — Real-world use cases, LLM inference, training, multi-user
- `preplan.md` — Scope, decisions, risks, dependencies, effort estimate
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments (empty)

## Estimated Effort

15-24 days after P10 multi-node and kernel interception are working.

## Related Topics

- R17 Topology-Aware Scheduling (GPU capabilities in topology map)
- R24 Time-Sliced Sharing (heterogeneous quotas)
- R13 CUDA Graph Interception (place graph nodes by GPU capability)
- R10 Memory Tiering (GPU bandwidth affects tier placement)
- R20 NCCL Backend (heterogeneous device reporting)
