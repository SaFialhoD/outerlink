# R13: CUDA Graph Interception

**Phase:** 10 — Compute Distribution
**Status:** RESEARCH COMPLETE / PRE-PLAN READY
**Priority:** HIGH
**Depends On:** P6 (Core Transport working), R3 (CUDA Interception)

## Summary
Intercept CUDA Graph captures to analyze the full computation DAG before execution. Split independent branches across multiple GPUs, schedule data transfers between dependent nodes, and optimize the entire distributed execution plan at compile time rather than runtime.

## What This Enables
- Automatic distributed execution of CUDA graphs
- Optimal work placement based on full DAG dependency analysis
- Transfer scheduling overlapped with compute (proactive, not reactive)
- Perfect memory access prediction for R11 prefetching
- Foundation for cooperative kernel splitting (R25)
- 100x reduction in coordinator messages vs eager per-kernel interception

## Key Questions — Answered

| Question | Answer | Details |
|----------|--------|---------|
| CUDA Graph API interception points? | ~78 Driver API functions | See research/01-cuda-graph-api.md |
| Can we modify a captured graph? | Yes — clone, remove nodes, add edges, rebuild | `cuGraphClone` + manipulation APIs |
| Data-dependent control flow? | Conditional nodes (CUDA 12.3+): IF/WHILE/SWITCH | Treat as atomic unit in Phase 1 |
| Overhead of graph analysis? | < 1ms for 1000-node graphs | HEFT is O(V^2 * K), well within budget |

## Research Findings

- **Primary interception path:** `cuStreamBeginCapture` / `cuStreamEndCapture` (stream capture is dominant in ML frameworks)
- **Recommended partitioning:** HEFT algorithm — handles heterogeneous GPUs, accounts for communication cost, preserves acyclicity
- **Recommended splitting strategy:** Per-GPU subgraphs via `cuGraphClone` + node removal + communication insertion
- **Execution model:** Coordinator-executor — one client coordinates, each GPU runs its subgraph independently
- **Key academic work:** Mustard (ICS 2025) validates multi-GPU CUDA graph partitioning and device-side execution

## Folder Contents
- `research/01-cuda-graph-api.md` — Full CUDA Graph Driver API surface (~78 functions, 14 node types)
- `research/02-graph-analysis-and-splitting.md` — DAG analysis, HEFT partitioning, splitting strategies
- `research/03-distributed-graph-execution.md` — Execution model, synchronization, dynamic shapes, integration
- `preplan.md` — Pre-plan with unknowns, dependencies, risks, milestones
- `plan.md` — TO BE CREATED
- `side-docs/` — Notes, experiments
- `progress.md` — Lifecycle tracker

## Implementation Milestones (from preplan)

1. Transparent pass-through (intercept all 78 APIs, forward unchanged)
2. Shadow graph construction (build internal DAG during capture)
3. DAG analysis (topological sort, critical path, parallelism profile)
4. Single-GPU validation (analyze but execute on one GPU, compare results)
5. Same-PC partitioning (split across GPUs using CUDA IPC)
6. Cross-PC partitioning (extend to network transport)
7. Performance optimization (prefetch, caching, profiling feedback)

## Related Topics
- R8 Kernel Parameter Introspection (precise memory dependency tracking)
- R10 Memory Hierarchy (memory region locations)
- R11 Speculative Prefetching (graph provides perfect access prediction)
- R17 Topology-Aware Scheduling (GPU placement, cost model)
- R20 NCCL Backend (collective operations within graphs)
- R25 Cooperative Kernel Splitting (splits individual kernels within graphs)
- R30 Persistent Kernels (alternative execution model)
