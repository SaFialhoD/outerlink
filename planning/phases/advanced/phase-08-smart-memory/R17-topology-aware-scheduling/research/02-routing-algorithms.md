# R17 Research: Routing Algorithms and Path Selection

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Survey routing algorithms applicable to OuterLink's heterogeneous multi-transport cluster: how to select the best path for each transfer, how to stripe across multiple links, how to handle congestion and failures, and how real-world systems (MPTCP, data center networks, RDMA multipath) solve these problems.

## TL;DR — What Works and What OuterLink Should Use

| Technique | What It Does | OuterLink Applicability |
|---|---|---|
| Weighted shortest path (Dijkstra) | Route based on latency or bandwidth | Per-transfer path selection: small transfers prefer low-latency, large prefer high-bandwidth |
| Weighted ECMP | Split traffic proportional to link capacity | Stripe across ConnectX-5 (100G) + USB4 (80G) at ~56%/44% ratio |
| MPTCP-style subflow management | Multiple concurrent streams over different paths | Reference for how to split a single large transfer across links |
| Congestion-aware rerouting (CONGA-style) | Avoid saturated links | Monitor link utilization, shift new transfers to less loaded paths |
| MP-RDMA (66 bytes extra per connection) | Multipath RDMA with minimal state | Direct model for ConnectX-5 multi-path if dual-port or multi-NIC |
| Static routing table + dynamic overlay | Pre-computed routes updated on topology change | Best balance of simplicity and adaptability for small clusters |

**The single biggest lesson:** For OuterLink's small cluster (2-8 nodes), full routing protocol complexity (BGP, OSPF) is overkill. A static routing table recomputed on topology changes, combined with per-transfer size-based path selection, covers 95% of use cases. Dynamic congestion-aware rerouting is a Phase 2 optimization.

---

## 1. The Routing Problem for OuterLink

### 1.1 What Makes This Different from Traditional Routing

Traditional IP routing finds one path between source and destination. OuterLink's problem is harder:

| Dimension | IP Routing | OuterLink Routing |
|---|---|---|
| Link types | Homogeneous (Ethernet) | Heterogeneous (RDMA 100G, USB4 80G, OCuLink 256G, TCP 25G) |
| Optimization metric | Usually latency only | Latency AND bandwidth (depends on transfer size) |
| Multi-path | Optional (ECMP) | Essential (aggregate bandwidth) |
| Transfer granularity | Per-packet | Per-page (64KB) or per-transfer (variable) |
| Topology size | Millions of nodes | 2-8 nodes (small enough for exact algorithms) |
| Link failure | Rare | Common (USB4 hotplug, NIC errors) |

### 1.2 Transfer Size Determines Optimal Path

The routing decision depends heavily on transfer size:

| Transfer Size | Optimal Strategy | Why |
|---|---|---|
| < 4 KB | Lowest latency link | Transfer time dominated by RTT, not bandwidth |
| 4 KB - 1 MB | Lowest latency with sufficient bandwidth | Both matter |
| 1 MB - 100 MB | Highest bandwidth single link | Bandwidth dominates; use the fattest pipe |
| > 100 MB | Multi-path striping | Saturate all available links |

**Crossover point:** A 64KB page (OuterLink's page size) takes:
- ConnectX-5 RDMA (12.5 GB/s): ~5.1 us transfer + ~2 us RTT = ~7.1 us total
- USB4 (8 GB/s est): ~8 us transfer + ~7 us RTT = ~15 us total
- TCP 25GbE (2.8 GB/s): ~23 us transfer + ~50 us RTT = ~73 us total

For single pages, always use the lowest-latency link (RDMA). For bulk transfers (many pages), striping across RDMA + USB4 is worthwhile.

---

## 2. Path Selection Algorithms

### 2.1 Weighted Dijkstra (Single Best Path)

Standard Dijkstra with configurable weight function:

**For latency-optimized routing:**
`weight(link) = link.measured_rtt_us`

**For bandwidth-optimized routing:**
`weight(link) = 1.0 / link.measured_bw_gbps`

**For balanced routing (combines both):**
```
weight(link) = alpha * (link.rtt_us / max_rtt) + (1 - alpha) * (1.0 / link.bw_gbps)
```
Where alpha = 0.7 for latency-sensitive, 0.3 for bandwidth-sensitive.

**Complexity:** O(E log V) where E = edges (links), V = vertices (nodes). For 8 nodes with ~20 links, this runs in <1 us. Pre-compute and cache.

### 2.2 Widest Path (Maximum Bottleneck Bandwidth)

For large transfers, we want the path with the highest minimum-bandwidth edge:

**Modified Dijkstra:** Instead of minimizing sum of weights, maximize the minimum weight along the path.
```
path_bandwidth(path) = min(link.bw for link in path)
best_path = argmax over all paths(path_bandwidth(path))
```

For direct links (1 hop), this trivially selects the highest-bandwidth link. For multi-hop (future), it correctly handles bottleneck edges.

### 2.3 K-Shortest Paths (for Multi-Path Enumeration)

Yen's algorithm finds the K shortest paths between source and destination. OuterLink uses this to enumerate ALL viable paths for multi-path striping.

For a 2-node cluster with ConnectX-5 + USB4:
- Path 1: ConnectX-5 (100 Gbps, 2 us)
- Path 2: USB4 (80 Gbps, 7 us)
- Path 3: TCP/25GbE (25 Gbps, 50 us)

All three are viable; the routing layer decides which to use based on transfer characteristics.

---

## 3. Multi-Path Striping

### 3.1 Why Striping Matters for OuterLink

Single-link bandwidth caps:
- ConnectX-5: ~12.5 GB/s (100 Gbps)
- USB4: ~8 GB/s (80 Gbps estimated)
- OCuLink: ~8 GB/s (PCIe 4.0 x4)

Striped: ConnectX-5 + USB4 = ~20 GB/s (160 Gbps). This is a 60% improvement over single-link for large transfers.

### 3.2 Weighted Round-Robin Striping

Assign pages to links proportional to link bandwidth:

```
total_bw = sum(link.bw for link in active_links)
for each link:
    link.weight = link.bw / total_bw
    link.pages_per_round = round(link.weight * batch_size)
```

**Example with ConnectX-5 (100G) + USB4 (80G):**
- Total: 180 Gbps
- ConnectX-5 weight: 100/180 = 55.6% -> 56 pages per 100-page batch
- USB4 weight: 80/180 = 44.4% -> 44 pages per 100-page batch

### 3.3 Chunk-Based Striping (Better Than Round-Robin)

Round-robin per-page creates reordering overhead. Better: split the transfer into contiguous chunks proportional to link speed.

```
For a 1000-page transfer over ConnectX-5 (100G) + USB4 (80G):
  ConnectX-5 chunk: pages 0-555    (556 pages)
  USB4 chunk:       pages 556-999  (444 pages)
```

Each chunk is sent as a contiguous stream on its link. Receiver reassembles after both chunks complete. This eliminates head-of-line blocking and simplifies flow control.

### 3.4 Completion Synchronization

Striped transfers complete when the SLOWEST chunk finishes. To minimize tail latency:

1. **Slightly overweight the faster link:** Give ConnectX-5 58% instead of 56%, so it finishes slightly before USB4
2. **Work-stealing at the tail:** When a fast link finishes its chunk, it can steal remaining pages from the slow link's queue
3. **Target: all links finish within 5% of each other**

### 3.5 MPTCP Reference Implementation

MPTCP (Multipath TCP, RFC 8684) is the closest real-world reference:
- Establishes multiple TCP subflows over different network paths
- Congestion control linked across subflows (LIA, OLIA, BALIA algorithms)
- Moves traffic away from congested subflows automatically
- In fat-tree data centers: MPTCP with 8 subflows achieves ~90% bisection bandwidth vs ~50% for single-path TCP with ECMP

**Key lessons for OuterLink:**
- Start all subflows simultaneously (don't wait for one to fail)
- Coupled congestion control ensures fairness across paths
- Adaptive subflow count: MPTCP_OPN adjusts active subflows based on network state, reducing flow completion time by up to 50%

### 3.6 MP-RDMA Reference

MP-RDMA (NSDI 2018) extends RDMA to support multiple paths:
- Adds only 66 bytes per connection state (very lightweight)
- Multi-path ACK-clocking: maintains per-path congestion windows
- Out-of-order aware path selection: avoids reordering penalties
- Achieves 2-4x higher throughput under 0.5-10% link loss
- Improves network utilization by up to 47%

**Key lesson:** RDMA multi-path is feasible with minimal overhead. OuterLink can implement per-path RDMA queue pairs with independent flow control.

### 3.7 Virtuoso: Software Multi-Path RDMA

Virtuoso creates a logical multi-path RDMA connection by:
- Mapping one logical connection to multiple QPs (one per path)
- Manipulating source IP addresses for ECMP-friendly load balancing
- Providing a transparent API so applications don't need to manage paths

**Key lesson:** The API should hide multi-path complexity. The caller says "transfer 100 pages to node 2" and the routing layer handles striping.

---

## 4. Congestion-Aware Routing

### 4.1 Why Congestion Matters

Multiple GPU jobs may share links. If node A is doing a large allreduce over ConnectX-5 while node B requests a page transfer, the page transfer should use USB4 instead of waiting for ConnectX-5.

### 4.2 Link Utilization Tracking

Each node tracks outbound utilization per link:

```rust
struct LinkState {
    link_id: LinkId,
    current_bw_used_gbps: f64,     // Bytes in-flight / time window
    max_bw_gbps: f64,              // From topology graph
    available_bw_gbps: f64,        // max - current
    queue_depth: u32,              // Pending transfers
    last_updated: Instant,
}
```

**Update frequency:** Every completed transfer updates `current_bw_used_gbps` using a sliding window (1 second).

### 4.3 Congestion-Aware Path Selection

Modify the path selection weight to include utilization:

```
effective_weight(link) = base_weight(link) * (1.0 + congestion_penalty(link))
congestion_penalty(link) = (link.utilization / link.capacity) ^ 2
```

At 50% utilization: penalty = 0.25 (mild preference for less-loaded link)
At 90% utilization: penalty = 0.81 (strong preference for alternative)
At 100% utilization: penalty = 1.0 (avoid unless no alternative)

### 4.4 CONGA-Style Distributed Congestion Feedback

In data center networks, CONGA (Distributed Congestion-Aware Load Balancing) works by:
- Each ToR switch maintains end-to-end congestion info for all paths
- Uses "flowlets" (bursts of packets separated by idle periods) as routing units
- Reroutes flowlets to least-congested path

**OuterLink adaptation:** Each node maintains a congestion map for all links. Congestion info is piggybacked on data transfers (zero extra messages). When starting a new transfer, consult the local congestion map to pick the least-loaded path.

### 4.5 Backpressure Mechanism

If all links to a destination are saturated:
1. Queue the transfer request (bounded queue, default 1024 entries)
2. Apply backpressure to the caller (prefetch scheduler, demand fetch queue)
3. When any link drops below 80% utilization, drain queued transfers
4. If queue is full, reject lowest-priority transfers (prefetches before demand fetches)

---

## 5. Handling Asymmetric Links

### 5.1 The Asymmetry Problem

OuterLink's links have vastly different characteristics:

| Link | Bandwidth | Latency | Full-Duplex | Reliability |
|---|---|---|---|---|
| ConnectX-5 RDMA | 100 Gbps | ~2 us | Yes | Very high |
| USB4 | 80 Gbps | ~5-10 us | Yes | High |
| OCuLink | 256 Gbps | <1 us | Yes | Very high |
| TCP/25GbE | 25 Gbps | ~50 us | Yes | High |

### 5.2 Link Selection by Transfer Characteristics

| Characteristic | Best Link | Rationale |
|---|---|---|
| Single 64KB page (latency-critical) | OCuLink > RDMA > USB4 > TCP | Minimize total time = RTT + transfer |
| Bulk transfer (>10 MB) | All links striped | Maximize aggregate bandwidth |
| Control message (<1 KB) | RDMA (lowest RTT) | Minimize latency |
| Background migration | TCP or USB4 | Preserve RDMA/OCuLink for demand traffic |
| NCCL collective | RDMA (NCCL expects it) | Compatibility with NCCL's transport expectations |

### 5.3 Priority Classes

Assign transfer requests to priority classes that map to link preferences:

| Priority | Class | Examples | Preferred Links |
|---|---|---|---|
| 0 (highest) | Demand fetch | GPU stalled waiting for page | OCuLink, RDMA |
| 1 | Interactive | cuMemcpy triggered by app | RDMA, OCuLink |
| 2 | Prefetch | R11 speculative prefetch | Any available |
| 3 | Migration | R10 tier migration, rebalancing | USB4, TCP (preserve fast links) |
| 4 (lowest) | Background | Dedup sync, metadata updates | TCP |

---

## 6. Routing Table Design

### 6.1 Static Table with Dynamic Refresh

For a small cluster (2-8 nodes), a full routing table is feasible:

```rust
struct RoutingTable {
    version: u64,                                       // Incremented on topology change
    entries: HashMap<(NodeId, NodeId), Vec<Route>>,     // (src, dst) -> sorted routes
    last_computed: Instant,
}

struct Route {
    path: Vec<LinkId>,          // Links to traverse (usually 1 for direct)
    total_latency_us: f64,
    bottleneck_bw_gbps: f64,
    priority_class: PriorityClass,  // Which priority classes should use this route
}
```

### 6.2 Table Recomputation

Recompute the routing table when:
1. **Topology change:** Node join/leave, link failure/recovery
2. **Significant metric change:** Link bandwidth drops >20% or RTT increases >50%
3. **Periodic:** Every 60 seconds regardless (catch gradual degradation)

**Recomputation cost:** For 8 nodes with ~20 links, computing all-pairs shortest paths (Floyd-Warshall) takes <1 us. This is negligible.

### 6.3 Per-Transfer Route Selection

```rust
fn select_route(
    src: NodeId,
    dst: NodeId,
    transfer_size: usize,
    priority: PriorityClass,
    routing_table: &RoutingTable,
    congestion_map: &CongestionMap,
) -> RouteDecision {
    let routes = routing_table.entries.get(&(src, dst));

    if transfer_size > STRIPE_THRESHOLD {  // e.g., 1 MB
        // Multi-path striping
        let viable = routes.filter(|r| congestion_map.available(r) > 0.2);
        RouteDecision::Stripe(viable, compute_weights(viable))
    } else {
        // Single best path
        let best = routes
            .filter(|r| r.priority_class <= priority)
            .min_by(|r| route_cost(r, transfer_size, congestion_map));
        RouteDecision::Single(best)
    }
}
```

---

## 7. Failure Handling and Rerouting

### 7.1 Link Failure Detection

| Detection Method | Time to Detect | Source |
|---|---|---|
| RDMA async event (IBV_EVENT_PORT_ERR) | <100 ms | Hardware/driver notification |
| RDMA completion error (IBV_WC_RETRY_EXC_ERR) | <1 second | Transfer-level failure |
| Heartbeat timeout | 3-5 seconds | Application-level probing |
| USB4 tunnel teardown event | <500 ms | Kernel event notification |
| TCP connection reset | <1 second | OS notification |

### 7.2 Failover Sequence

1. **Detect:** Link failure event received
2. **Mark:** Set link health to 0.0 in topology graph
3. **Recompute:** Trigger routing table recomputation (<1 us)
4. **Retry:** Re-queue all in-flight transfers on failed link to next-best route
5. **Notify:** Inform R11 prefetcher and R10 tier manager of topology change
6. **Probe:** Begin periodic probing of failed link for recovery

**Target failover time:** <2 seconds from link failure to first successful transfer on alternate path.

### 7.3 In-Flight Transfer Recovery

When a link fails mid-transfer:
- RDMA: The QP enters error state. All pending work requests complete with error.
- TCP: Socket receives RST or timeout.

**Recovery:** The transfer layer must track which pages were successfully transferred. Failed pages are re-queued on the alternate route. Successfully transferred pages are not retransmitted.

```rust
struct TransferTracker {
    transfer_id: u64,
    pages: Vec<PageTransferState>,  // Pending, InFlight(link_id), Completed, Failed
}
```

### 7.4 Graceful Degradation

| Links Available | Aggregate BW | Strategy |
|---|---|---|
| All (RDMA + USB4 + TCP) | ~200+ Gbps | Full striping, all features |
| RDMA + USB4 | ~180 Gbps | Normal operation minus TCP background |
| RDMA only | ~100 Gbps | Single-path, no striping benefit |
| USB4 only | ~80 Gbps | Reduced throughput, increase prefetch distance |
| TCP only | ~25 Gbps | Survival mode: increase cache aggressiveness, reduce prefetch |
| None | 0 | Node isolated, operate on local data only |

---

## 8. Integration Points

### 8.1 R10 Memory Tiering

R10's tier migration system asks R17: "What is the best path to move page X from node A to node B?" R17 returns the selected route and estimated transfer time.

### 8.2 R11 Speculative Prefetching

R11's prefetch scheduler asks R17: "How much bandwidth is available for prefetches to node B?" R17 returns available bandwidth after subtracting demand traffic. R11 adjusts prefetch aggressiveness accordingly.

### 8.3 R20 NCCL Backend

R20 exposes each OuterLink transport as a NCCL "channel." R17 provides the bandwidth/latency data that R20 reports to NCCL's topology engine. NCCL uses this to build optimal ring/tree topologies for collectives.

### 8.4 R14 Transport Compression

Compressed transfers have smaller effective size, which changes the routing decision. A 1 MB tensor that compresses to 200 KB should be routed as a small transfer (single best path) rather than striped.

---

## Open Questions

### Must Answer Before Detailed Planning

1. **What is the actual overhead of maintaining per-link congestion state?** At 100+ concurrent transfers per second, updating congestion counters must not become a bottleneck. Atomic counters should suffice.

2. **Should striping be at the page level (64KB) or at the transfer level (batch of pages)?** Page-level gives finer control but more overhead. Transfer-level (chunk-based) is simpler and likely sufficient.

3. **How does RDMA QP error state recovery work in practice?** When a QP enters error state, can it be reset and reused, or must a new QP be created? This affects failover latency.

4. **What is the actual USB4 bandwidth available for data when DisplayPort is active?** Pedro's setup may use USB4 for both displays and data. Need to measure the bandwidth partition.

### Can Answer During Implementation

5. At what transfer size does multi-path striping break even vs single-path? (Benchmark on actual hardware.)

6. How quickly can the routing table be propagated to all nodes after a topology change? (Expected: <100 ms for 8-node cluster.)

7. Should the congestion map use local measurements only, or incorporate feedback from remote nodes? (Start local, add remote in Phase 2.)

---

## Related Documents

- [R17 README](../README.md) — summary and folder contents
- [research/01-topology-discovery.md](./01-topology-discovery.md) — how topology is discovered
- [research/03-data-placement.md](./03-data-placement.md) — where to place data based on routing costs
- [R11 Speculative Prefetching preplan](../../R11-speculative-prefetching/preplan.md) — bandwidth budgeting integration
- [R20 NCCL Backend](../../../phase-07-memory-intelligence/R20-nccl-backend/README.md) — NCCL topology integration
- [CONSOLIDATION](../../../../research/CONSOLIDATION-all-research.md) — transport decisions and link characteristics
