# R17: Topology-Aware Scheduling — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Depends On:** R9 (Multi-Transport), R10 (Memory Tiering), P10 (Multi-Node working)

## Purpose

Define the scope, dependencies, decisions, risks, and implementation phases for R17 Topology-Aware Scheduling before creating the detailed plan. This document answers "WHAT needs to be planned" so the detailed plan can answer "HOW to build it."

---

## 1. Scope Definition

### 1.1 What R17 IS

- **Topology discovery engine:** auto-discover all devices (RDMA NICs, GPUs, USB4 tunnels, OCuLink) and represent the cluster as a weighted graph with latency and bandwidth attributes per link
- **Intra-node locality mapper:** use hwloc (or sysfs fallback) to determine GPU-NIC proximity, NUMA affinity, and PCIe topology
- **Active link prober:** measure actual RTT and bandwidth between every node pair, continuously updated
- **Routing table:** pre-computed per-(src, dst) route list, recomputed on topology changes
- **Per-transfer path selector:** choose single best link or multi-path stripe based on transfer size and priority
- **Multi-path striping engine:** split large transfers across multiple links proportional to bandwidth
- **Congestion-aware routing:** track per-link utilization and shift traffic to less-loaded paths
- **Failure detection and failover:** detect link/node failures via heartbeat + RDMA/USB4 events, reroute in-flight transfers
- **Data placement advisor:** recommend where new allocations should live and when existing pages should migrate, based on affinity and topology
- **NCCL topology integration:** generate NCCL-compatible topology XML for R20

### 1.2 What R17 IS NOT

- **Not a transport implementation:** R17 selects paths and makes routing decisions. The actual data transport (RDMA verbs, TCP sockets, USB4 tunnels) is implemented in R9 Multi-Transport and the existing transport layer.
- **Not a replacement for R10:** R17 advises tier placement (which remote node is best). R10 handles eviction policy, page tables, and tier mechanics.
- **Not a replacement for R11:** R17 provides bandwidth availability info to R11. R11 handles prefetch pattern prediction and scheduling.
- **Not a cluster scheduler:** R17 routes data transfers within an OuterLink cluster. It does not schedule GPU compute jobs (that's the ML framework's job).
- **Not a full SDN controller:** R17 makes application-level routing decisions. It does not program physical switches or modify network forwarding rules.

### 1.3 Boundary Conditions

| Boundary | Inside R17 | Outside R17 |
|---|---|---|
| Device discovery | Enumerate RDMA, PCIe, USB4, NIC | Install/configure drivers |
| Topology graph | Build, maintain, query | Visual dashboard (future) |
| Routing | Path selection, multi-path striping | Packet-level framing, checksums |
| Failure detection | Heartbeat, async events, failover | Physical link repair |
| Data placement | Recommend node + tier | Execute migration (R10), prefetch (R11) |
| NCCL integration | Generate topology XML | NCCL plugin implementation (R20) |
| Congestion control | Per-link utilization tracking | End-to-end flow control (transport layer) |

---

## 2. Dependencies

### 2.1 Upstream (R17 requires)

| Dependency | What R17 Needs From It | Status |
|---|---|---|
| **R9 Multi-Transport** | Transport abstraction layer: send/receive over RDMA, USB4, TCP via uniform API | Not started |
| **R10 Memory Tiering** | Page table with per-page metadata (access counts, tier, location), tier migration API | Pre-planned |
| **P10 Multi-Node** | Basic cluster membership: nodes can discover and communicate with each other | Planned |
| **CUDA Interception** (P1-P3) | LD_PRELOAD hooks for cuMemAlloc (placement decisions), cuLaunchKernel (access tracking) | Planned |
| **libibverbs** | System library for RDMA device enumeration and port queries | Available (system package) |
| **hwloc** | System library for hardware topology discovery | Available (system package) |

### 2.2 Downstream (depends on R17)

| Dependent | What It Needs From R17 |
|---|---|
| **R11 Speculative Prefetching** | Available bandwidth per link (for prefetch budgeting), best route for prefetch transfers |
| **R20 NCCL Backend** | NCCL-compatible topology XML, per-link bandwidth/latency for NCCL channel configuration |
| **R23 Heterogeneous GPU Mixing** | GPU capability info in topology graph (SM count, VRAM size, compute class) |
| **R14 Transport Compression** | Link bandwidth info (compression more valuable on slower links) |
| **R10 Memory Tiering** | Best-connected remote node for each tier (Tier 1 = best remote VRAM, Tier 3 = best remote DRAM) |

### 2.3 Soft Dependencies

| Dependency | Impact If Unavailable |
|---|---|
| **hwloc library** | Fall back to raw sysfs parsing (Linux only, more code, same result) |
| **USB4 kernel support** | USB4 links not discoverable; must be manually configured |
| **RDMA async events** | Cannot detect link failures via hardware; rely solely on heartbeat timeout |
| **OCuLink devices** | OCuLink links appear as PCIe devices in sysfs; should work via standard PCIe discovery |

---

## 3. Key Decisions That Need To Be Made

### 3.1 Discovery Library

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **A: hwloc (primary) + libibverbs** | Portable, unified API, well-maintained, GPU/NIC/NUMA in one call | External dependency, Rust bindings may be incomplete | Phase 1 — use this |
| **B: Raw sysfs + libibverbs** | No external dependency, full control | Linux only, complex string parsing, fragile | Fallback only |
| **C: Custom discovery from scratch** | Maximum control | Massive implementation effort, reinventing hwloc | Do not do |

**Proposed decision:** Option A. Use `hwloc2` Rust crate for intra-node topology, `rdma-sys` or raw FFI for libibverbs port queries, Thunderbolt sysfs for USB4. Fall back to Option B if hwloc is unavailable.

### 3.2 Topology Graph Library

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **A: Custom graph (HashMap-based)** | Zero dependency, tailored to our needs | Must implement graph algorithms ourselves | Phase 1 — start here |
| **B: petgraph crate** | Full graph algorithm suite, well-tested | Additional dependency, generic (not optimized for our use case) | Evaluate for Phase 2 |

**Proposed decision:** Option A for Phase 1. Our graph is small (2-8 nodes, ~20 edges) and the algorithms are simple (Dijkstra, Floyd-Warshall). Custom implementation is clearer and avoids mapping our domain to petgraph's generic types.

### 3.3 Routing Table Architecture

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **A: Static table, recomputed on change** | Simple, predictable, O(1) lookup | Stale during topology changes (brief) | Phase 1 — start here |
| **B: Dynamic per-transfer computation** | Always optimal | O(E log V) per transfer (~1 us, but at high rate adds up) | Phase 2 for congestion-aware |
| **C: Hybrid (static + congestion overlay)** | Static base + dynamic adjustment | More complex but best performance | Phase 2 |

**Proposed decision:** Option A for Phase 1. Recompute the full routing table on any topology change (takes <1 us for 8 nodes). Add congestion overlay (Option C) in Phase 2.

### 3.4 Multi-Path Striping Granularity

| Option | Granularity | Overhead | Reordering | Recommendation |
|---|---|---|---|---|
| **A: Per-page (64KB)** | Fine | High (per-page routing decision) | Complex reassembly | Phase 2 |
| **B: Per-chunk (split transfer into N chunks)** | Coarse | Low (one decision per link) | Simple (ordered within chunk) | Phase 1 — start here |
| **C: Adaptive (pages for small, chunks for large)** | Mixed | Medium | Medium | Phase 2 |

**Proposed decision:** Option B for Phase 1. For a multi-path transfer, split the page batch into N contiguous chunks (one per link), proportional to link bandwidth. Each chunk is transferred in order on its link. Receiver waits for all chunks.

### 3.5 Failure Detection Method

| Option | Detection Time | Complexity | False Positive Rate | Recommendation |
|---|---|---|---|---|
| **A: Fixed-timeout heartbeat** | 3-5 seconds | Low | Medium (depends on timeout) | Phase 1 — start here |
| **B: Phi accrual detector** | Adaptive (1-10 sec) | Medium | Very low | Phase 2 |
| **C: Hybrid (RDMA events + heartbeat)** | <2 seconds | Medium | Low | Phase 1 with RDMA events |

**Proposed decision:** Option C. Use RDMA async events (IBV_EVENT_PORT_ERR) and TCP connection resets for fast detection (<1 second), plus fixed-timeout heartbeat (3 second timeout) as backup. Implement phi accrual in Phase 2.

### 3.6 Data Placement Integration Depth

| Option | Scope | Complexity | Recommendation |
|---|---|---|---|
| **A: Passive advisor** | R17 provides topology data; R10 makes placement decisions | Low | Phase 1 — start here |
| **B: Active placement** | R17 intercepts cuMemAlloc and makes placement decisions | Medium | Phase 2 |
| **C: Predictive placement** | R17 uses R11 profiles to pre-place data before access | High | Phase 3 |

**Proposed decision:** Option A for Phase 1. R17 provides a `best_node_for(gpu_id, size)` API that R10 can query. R10 makes the final decision. Active placement (Option B) in Phase 2 when we have confidence in the scoring model.

---

## 4. Risks and Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Discovery overhead at scale** — hwloc + libibverbs + probing takes too long at startup | Medium | Low | For 8 nodes, total discovery is ~5 seconds. Acceptable. Cache results, probe in background. |
| 2 | **Stale routing table** — topology changes between recomputations cause suboptimal routing | Medium | Medium | Recompute takes <1 us; trigger on any event. Worst case: one transfer on suboptimal path. |
| 3 | **Multi-path striping overhead exceeds benefit** — coordination cost of splitting across links wastes more time than it saves | Medium | Low | Only stripe transfers >1 MB. Below that threshold, single best path. Benchmark the crossover. |
| 4 | **False positive failure detection** — temporary network hiccup causes node to be marked dead | High | Medium | Require 3 consecutive heartbeat misses before declaring failure. Confirmation probes before removing from topology. |
| 5 | **USB4 bandwidth unpredictable** — DisplayPort tunnels consume variable bandwidth | Medium | High | Probe USB4 bandwidth periodically. Use conservative estimates. Mark USB4 as lower priority than RDMA for latency-sensitive transfers. |
| 6 | **NCCL topology XML incompatibility** — OuterLink-generated XML confuses NCCL | High | Medium | Study NCCL's XML parser closely. Test with multiple NCCL versions. Fall back to letting NCCL auto-detect (disabling our custom XML). |
| 7 | **hwloc Rust bindings incomplete** — `hwloc2` crate missing features we need | Low | Medium | Write raw FFI bindings for missing functions. hwloc C API is stable and well-documented. |
| 8 | **Congestion feedback loop** — rapid rerouting causes oscillation (all traffic shifts to same link repeatedly) | Medium | Low | Dampen routing changes: require utilization difference >20% before shifting. Exponential backoff on route changes. |

---

## 5. Implementation Phases

### Phase 1: Discovery and Static Routing (3-4 weeks)

**Goal:** Working topology discovery, static routing table, single-path transfer selection.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| hwloc integration: enumerate CPUs, NUMA, PCIe, GPUs | 3-4 days | `IntraNodeTopology` struct with all local hardware |
| libibverbs integration: enumerate RDMA devices, query ports | 3-4 days | `RdmaDeviceInfo` with bandwidth/latency per port |
| sysfs USB4/Thunderbolt detection | 2-3 days | `Usb4LinkInfo` with tunnel type and bandwidth |
| Network interface enumeration (TCP/Ethernet) | 1-2 days | `EthernetInfo` with link speed |
| Topology graph data structure | 3-4 days | `TopologyGraph` with nodes, links, weighted edges |
| Active RTT probing (RDMA and TCP) | 3-4 days | Per-link latency measurements |
| Active bandwidth probing | 2-3 days | Per-link throughput measurements |
| Heartbeat + fixed-timeout failure detection | 3-4 days | Node liveness monitoring, link failure events |
| Static routing table (all-pairs shortest path) | 2-3 days | Pre-computed route table, recomputed on topology change |
| Per-transfer path selector (size-based) | 2-3 days | Select best single link based on transfer size |
| Integration test: route transfers over correct links | 3-4 days | Verify path selection matches expected behavior |

**Exit criteria:** Topology is auto-discovered at startup. Transfers route over the best available link. Node/link failures detected within 5 seconds and routes updated.

### Phase 2: Multi-Path and Congestion (3-4 weeks)

**Goal:** Multi-path striping, congestion-aware routing, NCCL integration.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| Multi-path chunk-based striping engine | 1 week | Split large transfers across multiple links |
| Per-link utilization tracking | 2-3 days | Real-time congestion map |
| Congestion-aware route selection | 3-4 days | Weight routes by available bandwidth |
| NCCL topology XML generation | 1 week | Generate XML compatible with NCCL 2.18+ |
| Phi accrual failure detector | 3-4 days | Adaptive failure detection with low false positive rate |
| Priority-based routing (demand > prefetch > migration) | 2-3 days | Transfer priority classes mapped to link preferences |
| Placement advisor API (`best_node_for`) | 3-4 days | R10 can query optimal placement for new allocations |
| Integration test: striping across ConnectX-5 + USB4 | 3-4 days | Verify aggregate bandwidth >150 Gbps on large transfers |

**Exit criteria:** Multi-path striping works and improves throughput by >40% over single-link for large transfers. NCCL uses OuterLink-provided topology.

### Phase 3: Smart Placement and Optimization (2-3 weeks)

**Goal:** Active data placement, phase-aware optimization, predictive placement.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| Active placement in cuMemAlloc interception | 1 week | New allocations placed on optimal node |
| Cost-benefit migration model (ARMS-style) | 3-4 days | Migrate only when benefit > 2x cost |
| Read-only page replication | 3-4 days | Broadcast weights to all nodes at iteration start |
| Phase-aware placement profiles | 3-4 days | Different placement policies for forward/backward/optimizer |
| Integration with R11 for predictive placement | 3-4 days | Use prefetch profiles to pre-place data |
| Performance tuning and benchmarking | 3-4 days | Optimize crossover thresholds, damping parameters |

**Exit criteria:** Data placement reduces remote access rate by >50%. Phase-aware placement correctly identifies training phases. Migration decisions don't cause thrashing.

### Phase 4: Future Enhancements (not in initial plan)

- RL-based placement optimization
- Cross-node computation migration (move kernel to data)
- Topology-aware collective algorithm selection (not just NCCL XML)
- Live topology visualization dashboard
- Predictive link failure (detect degradation trends)

**Total estimated effort: 8-11 weeks** for Phases 1-3.

---

## 6. Open Questions

### Must Answer Before Detailed Planning

1. **Is the `hwloc2` Rust crate sufficient for our needs?** Specifically: does it expose GPU NUMA nodes, PCIe link speed, and I/O device enumeration? If not, how much raw FFI do we need?

2. **What NCCL topology XML versions are we targeting?** NCCL 2.18+ changed the XML format. We need to decide the minimum supported NCCL version and test XML generation against it.

3. **What is the actual overhead of 5-second heartbeat probing at 8 nodes?** Each node sends 7 heartbeats/second (one per peer). At ~100 bytes each, that's ~700 bytes/second — negligible. But the processing overhead of 7 RTT measurements per second needs benchmarking.

4. **How does USB4 bandwidth allocation interact with DisplayPort tunnels on Pedro's hardware?** If USB4 is used for a monitor, the available data bandwidth drops significantly. Need actual measurements.

### Can Answer During Implementation

5. What is the optimal stripe-threshold transfer size? (Benchmark: 512 KB, 1 MB, 4 MB.)

6. Should the topology graph be in shared memory for multi-process access on the same node?

7. How should R17 handle nodes with asymmetric hardware (e.g., one node has OCuLink, others don't)?

8. What is the maximum number of concurrent RDMA queue pairs before ConnectX-5 performance degrades?

---

## 7. Success Criteria

### Quantitative

| Metric | Target (Phase 1) | Target (Phase 2) | Target (Phase 3) |
|---|---|---|---|
| **Topology discovery time** (startup) | <10 seconds | <10 seconds | <10 seconds |
| **Routing decision latency** | <1 us (table lookup) | <5 us (with congestion check) | <5 us |
| **Link failure detection** | <5 seconds | <3 seconds | <2 seconds |
| **Failover time** (failure to reroute) | <10 seconds | <5 seconds | <3 seconds |
| **Multi-path aggregate bandwidth** | N/A | >80% of theoretical sum | >85% of theoretical sum |
| **Routing optimality** (chosen path vs best path) | >90% correct | >95% correct | >98% correct |
| **Remote access reduction** (via placement) | N/A | N/A | >50% fewer remote accesses |
| **Migration thrashing rate** | N/A | N/A | <5% of migrations reversed within 10s |
| **Bandwidth utilization** (all links utilized) | >60% of each link | >70% of each link | >80% of each link |

### Qualitative

- Topology discovery is fully automatic (zero manual configuration for supported link types)
- Routing adapts to topology changes without manual intervention
- Multi-path striping is transparent to callers (hidden behind transfer API)
- Failure handling is graceful (no data loss, transfers retry on alternate paths)
- NCCL sees OuterLink's topology correctly and builds optimal collectives
- Observable: topology graph, link utilization, routing decisions all exposed as metrics
- No performance regression: R17 never makes things slower than a naive "always use first available link" approach

---

## 8. Testing Strategy

### Unit Tests

- TopologyGraph: add/remove nodes/links, query paths, verify Dijkstra results
- Routing table: recomputation correctness, route ordering by priority
- Bandwidth calculator: `ibv_query_port` attribute decoding, link speed computation
- Congestion tracker: utilization updates, penalty calculation
- Cost-benefit migration: verify migration decision for known scenarios

### Integration Tests

- 2-node RDMA: discover topology, verify correct link characterization
- 2-node RDMA + TCP: verify both links discovered, large transfer uses RDMA
- Link failure simulation: kill one link, verify failover to alternate
- Multi-path striping: verify chunk distribution proportional to bandwidth
- NCCL XML: generate XML, verify NCCL parses it successfully

### Benchmarks

- Single-link vs multi-link aggregate bandwidth
- Routing decision overhead under load (1000 decisions/second)
- Failover latency (time from link failure to successful transfer on alternate)
- Topology discovery startup time at various cluster sizes (2, 4, 8 nodes)
- Placement quality: compare affinity-based vs random placement on training throughput

---

## Related Documents

- [R17 README](./README.md) — summary and folder contents
- [research/01-topology-discovery.md](./research/01-topology-discovery.md) — discovery methods and tools
- [research/02-routing-algorithms.md](./research/02-routing-algorithms.md) — routing and path selection
- [research/03-data-placement.md](./research/03-data-placement.md) — placement optimization
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — upstream dependency
- [R11 Speculative Prefetching](../R11-speculative-prefetching/preplan.md) — downstream dependency (bandwidth budgeting)
- [R20 NCCL Backend](../../phase-07-memory-intelligence/R20-nccl-backend/README.md) — downstream dependency (topology XML)
- [CONSOLIDATION](../../../research/CONSOLIDATION-all-research.md) — project-wide research consolidation
