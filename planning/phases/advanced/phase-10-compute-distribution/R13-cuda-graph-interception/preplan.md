# R13: CUDA Graph Interception — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH

## Purpose

Define WHAT needs to be planned for implementing CUDA Graph interception and distributed execution in OuterLink, before writing the detailed implementation plan.

---

## Scope Definition

### What R13 Covers

1. **Interception of all ~78 CUDA Graph Driver API functions** — transparent hooking via the existing LD_PRELOAD mechanism (R3)
2. **Shadow graph construction** — building an internal representation of the captured CUDA graph as the application constructs it
3. **DAG analysis** — topological sort, critical path analysis, parallelism detection, data dependency tracking
4. **Graph partitioning** — assigning graph nodes to GPUs using HEFT or similar algorithms
5. **Subgraph construction** — building per-GPU CUDA graphs from the partition assignment
6. **Communication insertion** — adding transfer nodes at partition boundaries
7. **Distributed execution coordination** — launching, synchronizing, and collecting results from per-GPU subgraphs
8. **Graph update fast-path** — handling `cuGraphExecUpdate` without re-analyzing topology

### What R13 Does NOT Cover (Adjacent Topics)

| Topic | Covered By | R13 Interaction |
|-------|-----------|-----------------|
| Kernel argument memory analysis | R8 (Kernel Param Introspection) | R13 uses R8's output for data dependency tracking |
| Memory page management | R10 (Memory Hierarchy) | R13 queries R10 for memory region locations |
| Data prefetching | R11 (Speculative Prefetching) | R13 feeds perfect access predictions to R11 |
| GPU placement decisions | R17 (Topology-Aware Scheduling) | R13 uses R17's topology map for HEFT cost model |
| NCCL collective handling | R20 (NCCL Backend) | R13 identifies NCCL nodes and delegates to R20 |
| Splitting individual kernels | R25 (Cooperative Kernel Splitting) | Complementary — R13 splits at graph level, R25 at kernel level |
| Transport layer | P6 (Core Transport) | R13 uses whatever transport is active for cross-GPU transfers |

---

## Key Technical Unknowns

### Unknown 1: Kernel Memory Access Without R8

**Risk:** HIGH
**Question:** How do we track data dependencies through kernel nodes without knowing which arguments are memory pointers?
**Research finding:** Conservative approach assumes all predecessor outputs are needed. This works but over-constrains partitioning.
**Required before planning:** Decide whether R8 is a hard prerequisite or if we ship with the conservative approach first.
**Proposed resolution:** Implement in two tiers — conservative (no R8) first, precise (with R8) as optimization.

### Unknown 2: NCCL Kernel Identification

**Risk:** MEDIUM
**Question:** How do we reliably identify NCCL collective kernels within a captured graph?
**Research finding:** NCCL kernels have distinctive names (e.g., `ncclKernel_AllReduce_RING_LL_Sum_float`). We can match kernel function pointers against a known list built by intercepting NCCL initialization.
**Required before planning:** Validate this approach works by capturing a PyTorch DDP training graph and examining kernel names.
**Proposed resolution:** Prototype NCCL kernel detection as a standalone test before integrating into graph analysis.

### Unknown 3: Device-Side Graph Features

**Risk:** LOW (for now)
**Question:** What percentage of real workloads use device-side graph launch or device-side parameter updates?
**Research finding:** These features are CUDA 12.0+ / 12.4+ and not yet widely adopted in ML frameworks as of 2026. PyTorch and TensorFlow do not use them.
**Required before planning:** None — plan for the common case (host-side launch), document device-side as a future limitation.
**Proposed resolution:** Detect device-side flags during interception and fall back to single-GPU execution for those graphs.

### Unknown 4: Graph Re-Capture Cost Amortization

**Risk:** MEDIUM
**Question:** How often do real workloads re-capture graphs, and is our analysis overhead acceptable?
**Research finding:** Fixed-shape training: capture once. Variable-shape: capture per unique shape (PyTorch caches). NLP: many shapes, frequent re-capture.
**Required before planning:** Profile analysis overhead target (< 1ms for 1000-node graphs) and validate with benchmarks.
**Proposed resolution:** Cache partition plans by topology hash. Only re-analyze when topology changes. Only update parameters when tensor pointers change.

### Unknown 5: Conditional Node Partitioning

**Risk:** LOW
**Question:** How do we partition graphs containing conditional nodes (IF/WHILE/SWITCH)?
**Research finding:** Conditional bodies must be analyzed as potential execution paths. The conditional node itself must stay on one GPU (it contains the condition kernel). Bodies can potentially be on different GPUs, but synchronization becomes complex.
**Required before planning:** Decide whether to support conditional node splitting or treat the entire conditional subtree as an atomic unit.
**Proposed resolution:** Phase 1 — treat conditional nodes as atomic (keep on one GPU). Phase 2 — investigate body splitting if profiling shows it matters.

---

## Dependencies

### Hard Dependencies (Must Exist Before R13 Implementation)

| Dependency | Status | What R13 Needs From It |
|------------|--------|----------------------|
| R3 CUDA Interception | Complete | LD_PRELOAD framework, function hooking infrastructure |
| P6 Core Transport | Planned | Network send/receive for cross-PC transfers |
| Basic memory tracking | Part of R10 | Know which GPU owns which memory region |

### Soft Dependencies (Enhances R13 but Not Required)

| Dependency | Status | Enhancement |
|------------|--------|-------------|
| R8 Kernel Param Introspection | Research phase | Precise data dependency tracking (vs conservative) |
| R11 Speculative Prefetching | Research phase | Prefetch data before graph launch |
| R17 Topology-Aware Scheduling | Research phase | Better cost model for HEFT partitioning |
| R20 NCCL Backend | Research phase | Handle collective operations in graphs |

### R13 Enables (Downstream)

| Topic | How R13 Helps |
|-------|--------------|
| R25 Cooperative Kernel Splitting | Graph analysis identifies which kernels to split |
| R30 Persistent Kernels | Graph mode vs persistent mode comparison |
| General workload optimization | Graph-level "compile time" optimization for all patterns |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Graph splitting overhead > benefit for small graphs | HIGH | MEDIUM | Auto-detect: if graph < threshold, skip splitting |
| CUDA graph API changes in future CUDA versions | LOW | HIGH | Abstract behind our own graph representation |
| Frameworks move to device-side features we cannot intercept | MEDIUM | HIGH | Monitor framework adoption, implement device-side hooks if needed |
| Memory overhead of shadow graph for very large graphs | LOW | LOW | Lazy analysis, stream shadow graph nodes |
| Cross-GPU synchronization latency dominates for fine-grained graphs | MEDIUM | HIGH | Coarsen partition (fewer, larger chunks per GPU) |

---

## What Needs to Be in the Detailed Plan

### Architecture Section
- Shadow graph data structures (Rust structs)
- Integration points with existing interception layer (R3)
- Coordinator-executor communication protocol
- Thread model (analysis on which thread? Async?)

### Implementation Milestones

1. **Milestone 1: Transparent pass-through** — Intercept all ~78 graph APIs, forward to driver, validate nothing breaks
2. **Milestone 2: Shadow graph construction** — Build internal DAG representation during capture/construction
3. **Milestone 3: DAG analysis** — Topological sort, critical path, parallelism profile
4. **Milestone 4: Single-GPU validation** — Run analysis but execute on single GPU, compare results
5. **Milestone 5: Same-PC partitioning** — Split graph across GPUs on same PC using CUDA IPC
6. **Milestone 6: Cross-PC partitioning** — Extend to network transport
7. **Milestone 7: Performance optimization** — Prefetch integration, partition caching, profiling feedback

### Testing Strategy
- Unit tests for DAG analysis algorithms (with synthetic graphs)
- Integration tests with real CUDA graph captures (PyTorch models)
- Performance benchmarks: analysis overhead, distribution speedup
- Correctness: compare distributed graph output vs single-GPU output

### Acceptance Criteria
- All ~78 graph API functions intercepted without breaking any CUDA application
- Shadow graph matches `cuGraphDebugDotPrint` output for validation
- Analysis completes in < 1ms for 1000-node graphs
- Distributed execution produces bit-identical results to single-GPU execution
- Speedup > 1.5x on 2-GPU setup for graphs with parallelism width >= 4

---

## Estimated Effort

| Component | Estimated Complexity | Notes |
|-----------|---------------------|-------|
| API interception (78 functions) | MEDIUM | Boilerplate-heavy but straightforward, follows R3 pattern |
| Shadow graph construction | MEDIUM | Data structure design critical, implementation straightforward |
| DAG analysis algorithms | LOW-MEDIUM | Well-known algorithms, clean implementation |
| HEFT partitioning | MEDIUM | Core algorithmic work, needs good cost model |
| Subgraph construction | HIGH | Graph manipulation APIs are tricky, edge cases with node types |
| Communication insertion | HIGH | Must handle all transport mechanisms, synchronization is subtle |
| Coordinator-executor protocol | HIGH | Distributed systems complexity, error handling |
| Graph update fast-path | MEDIUM | Topology comparison, parameter propagation |
| Testing and validation | HIGH | Requires real ML workloads, bit-exact comparison |

**Total estimated scope:** Large feature. Recommend splitting into 3-4 planning sub-phases aligned with the milestones above.

---

## Open Questions for Planning

1. Should R13 be a single plan or broken into sub-plans (e.g., R13a: Interception + Analysis, R13b: Partitioning, R13c: Distributed Execution)?
2. Do we need a prototype/proof-of-concept before the full plan? Specifically: capture a PyTorch graph, analyze it, validate the shadow graph matches.
3. What is the minimum viable version? Suggestion: Milestones 1-4 (analysis without distribution) as v1, Milestones 5-7 as v2.
4. Should the partition planner be pluggable (allow different algorithms)? HEFT is the default but dagP or custom heuristics may be better for specific workloads.

---

## Related Documents

- research/01-cuda-graph-api.md — Full API surface documentation
- research/02-graph-analysis-and-splitting.md — Partitioning algorithms and strategies
- research/03-distributed-graph-execution.md — Execution model and synchronization
- R3: CUDA Interception Strategies — Foundation interception architecture
- planning/pre-planning/02-FINAL-PREPLAN.md — Overall project pre-plan
