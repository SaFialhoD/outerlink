# R17: Topology-Aware Scheduling --- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Depends On:** R9 (Multi-Transport), R10 (Memory Tiering), P10 (Multi-Node)
**Supersedes:** [preplan.md](./preplan.md) (v1)

## Purpose

Second-round refinement of R17's pre-plan. This version adds exact Rust struct definitions, a step-by-step discovery protocol, a concrete routing algorithm with cost function formula, a multi-path striping algorithm, a data placement scoring formula, and a dynamic topology update protocol. All designs integrate cross-topic findings from R10 v2, R11, R14, R20 v2, R23, R26, R28, and R29.

---

## 1. Exact Rust Structs

### 1.1 TopologyGraph

The central data structure. A weighted directed multigraph where nodes are physical machines and edges are network links. Small enough (2-8 nodes, ~20 edges) to fit entirely in memory on every node.

```rust
/// Version-stamped topology graph. Every node holds a local copy.
/// On any topology change, version increments and the graph is re-propagated.
struct TopologyGraph {
    /// Monotonically increasing version. Resolves conflicts during propagation.
    version: u64,
    /// Cluster nodes keyed by NodeId.
    nodes: HashMap<NodeId, NodeInfo>,
    /// All links (edges). Multiple links between the same pair are separate entries.
    links: Vec<LinkInfo>,
    /// Adjacency list: NodeId -> list of LinkId (indices into `links`).
    adjacency: HashMap<NodeId, Vec<usize>>,
    /// Pre-computed routing table. Invalidated and recomputed on version change.
    routing_table: RoutingTable,
    /// When this graph was last recomputed from measurements.
    last_recomputed: Instant,
}
```

### 1.2 NodeInfo

Includes GPU capability profiles (R23 integration) and per-node resource state.

```rust
struct NodeInfo {
    id: NodeId,
    hostname: String,
    /// hwloc-discovered intra-node topology.
    gpus: Vec<GpuProfile>,
    nics: Vec<NicInfo>,
    numa_nodes: Vec<NumaNodeInfo>,
    /// Total resources for placement scoring.
    total_vram_bytes: u64,
    free_vram_bytes: u64,
    total_dram_bytes: u64,
    free_dram_bytes: u64,
    /// PTP-synchronized clock offset from reference node (R26 integration).
    /// Used for accurate one-way delay measurement.
    ptp_offset_ns: i64,
    /// Node join timestamp. Older nodes are preferred for home-node assignment.
    joined_at: Instant,
    /// Heartbeat state.
    last_heartbeat: Instant,
    heartbeat_misses: u32,
    /// Phi accrual suspicion level (Phase 2). 0.0 = alive, >8.0 = suspected dead.
    phi_suspicion: f64,
}

/// GPU capability profile (R23 integration).
/// Integrated into topology so placement scorer knows GPU strengths.
struct GpuProfile {
    gpu_id: GpuId,
    /// PCI BDF address (for hwloc locality and NCCL XML).
    pci_bdf: PciBdf,
    /// SM architecture version (e.g., 86 for Ampere GA102).
    sm_version: u32,
    /// Compute capability (e.g., 8.6).
    compute_capability: (u32, u32),
    /// Peak single-precision TFLOPS.
    fp32_tflops: f32,
    /// Peak half-precision TFLOPS (relevant for training).
    fp16_tflops: f32,
    /// Total VRAM in bytes.
    vram_bytes: u64,
    /// VRAM bandwidth in GB/s (e.g., 936 for RTX 3090).
    vram_bandwidth_gbps: f32,
    /// PCIe generation and width of the GPU's slot.
    pcie_gen: u8,
    pcie_width: u8,
    /// NUMA node this GPU is closest to.
    numa_node: u32,
}

struct NicInfo {
    /// RDMA device name (e.g., "mlx5_0").
    device_name: String,
    /// PCI BDF address.
    pci_bdf: PciBdf,
    /// NUMA node for NUMA-aware pinned buffer allocation.
    numa_node: u32,
    /// Port number (1-based, ConnectX-5 typically has 1 port).
    port: u8,
    /// Theoretical bandwidth from ibv_query_port (Gbps).
    theoretical_bw_gbps: f64,
    /// Link layer: Ethernet (RoCE) or InfiniBand.
    link_layer: LinkLayer,
    /// GID table entries for RoCE addressing.
    gid_count: u32,
}

struct NumaNodeInfo {
    numa_id: u32,
    /// Total memory on this NUMA node.
    memory_bytes: u64,
    /// NUMA distance to every other NUMA node (from /sys/devices/system/node/).
    distances: Vec<u32>,
    /// CPUs on this NUMA node (for thread pinning).
    cpus: Vec<u32>,
}
```

### 1.3 LinkInfo

Each physical link between two nodes. Multiple links between the same pair are separate entries (e.g., ConnectX-5 + USB4).

```rust
struct LinkInfo {
    /// Unique link identifier (index into TopologyGraph.links).
    id: usize,
    /// Source node.
    src: NodeId,
    /// Destination node.
    dst: NodeId,
    /// Transport type.
    link_type: LinkType,
    /// Theoretical max bandwidth (from device enumeration), Gbps.
    theoretical_bw_gbps: f64,
    /// Measured bandwidth (from active probing), Gbps.
    /// Updated every 60 seconds and on-demand.
    measured_bw_gbps: f64,
    /// Smoothed RTT (exponential moving average, alpha=0.125), microseconds.
    /// Updated from heartbeat probes every 5 seconds.
    smoothed_rtt_us: f64,
    /// RTT variance (for jitter estimation), microseconds.
    rtt_var_us: f64,
    /// PTP-calibrated one-way delay (R26), microseconds.
    /// More accurate than RTT/2 for asymmetric links.
    one_way_delay_us: f64,
    /// Link health score: 0.0 (dead) to 1.0 (perfect).
    /// health = (measured_bw / theoretical_bw) * 0.7 + (baseline_rtt / smoothed_rtt) * 0.3
    health: f64,
    /// Current utilization (bytes in-flight / capacity), range 0.0 to 1.0.
    utilization: f64,
    /// Available bandwidth = measured_bw * (1.0 - utilization), Gbps.
    available_bw_gbps: f64,
    /// MTU for this link.
    mtu: u32,
    /// Whether R14 compression should be used on this link.
    /// Decision: compress if measured_bw < 50 Gbps (R14 integration).
    compress: bool,
    /// USB4 specific: current tunnel bandwidth allocation.
    /// None for non-USB4 links.
    usb4_allocated_bw_gbps: Option<f64>,
    /// Baseline RTT recorded at link discovery (for health calculation).
    baseline_rtt_us: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LinkType {
    /// ConnectX-5 RoCE v2 (100 Gbps).
    RdmaRoCE,
    /// InfiniBand (future).
    RdmaIB,
    /// USB4/Thunderbolt tunnel.
    Usb4,
    /// OCuLink PCIe 4.0 x4.
    OcuLink,
    /// Plain TCP over Ethernet.
    TcpEthernet,
}
```

### 1.4 Route and RoutingTable

```rust
/// Pre-computed routing table. All-pairs, recomputed on topology change.
struct RoutingTable {
    /// Topology version this table was computed from.
    version: u64,
    /// (src, dst) -> sorted list of routes. Best route first.
    entries: HashMap<(NodeId, NodeId), Vec<Route>>,
    /// When this table was last recomputed.
    computed_at: Instant,
}

/// A route from src to dst over a specific link (or chain of links for multi-hop).
struct Route {
    /// Links to traverse. For direct connections, this is a single LinkId.
    /// For multi-hop (future): ordered list of links.
    links: Vec<usize>,
    /// Estimated total latency (sum of one-way delays + per-hop overhead).
    total_latency_us: f64,
    /// Bottleneck bandwidth (minimum link bandwidth along the path), Gbps.
    bottleneck_bw_gbps: f64,
    /// Which priority classes should prefer this route.
    /// Lower-priority traffic avoids high-bandwidth links to preserve them.
    min_priority: TransferPriority,
    /// Estimated cost (from weighted Dijkstra). Lower is better.
    cost: f64,
}

/// Transfer priority classes. Lower value = higher priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TransferPriority {
    /// GPU stalled waiting for a page (R19 demand fault).
    DemandFetch = 0,
    /// Application-triggered cuMemcpy.
    Interactive = 1,
    /// R11 speculative prefetch.
    Prefetch = 2,
    /// R10 tier migration, rebalancing.
    Migration = 3,
    /// Dedup sync, metadata, heartbeats.
    Background = 4,
}
```

### 1.5 PlacementDecision

Returned by the placement scorer when R10 asks "where should this allocation go?"

```rust
/// Result of the data placement scoring algorithm.
struct PlacementDecision {
    /// Recommended node for the allocation.
    target_node: NodeId,
    /// Score breakdown for observability.
    score: PlacementScore,
    /// Alternative nodes if the first choice fails (sorted by score).
    alternatives: Vec<(NodeId, PlacementScore)>,
    /// Whether to replicate instead of single-place.
    /// True for read-only data accessed by multiple GPUs.
    replicate: bool,
    /// If replicate=true, which nodes should receive copies.
    replica_targets: Vec<NodeId>,
}

/// Breakdown of the placement score for a single (node, allocation) pair.
struct PlacementScore {
    /// Total score. Higher is better.
    total: f64,
    /// Subscores for observability and tuning.
    access_affinity: f64,
    migration_cost: f64,
    tier_capacity: f64,
    link_bandwidth: f64,
    gpu_capability: f64,
}
```

---

## 2. Discovery Protocol

Step-by-step protocol for building the topology graph from scratch at node join, and updating it incrementally.

### 2.1 Node Join Sequence

```
PHASE 1: LOCAL DISCOVERY (node-local, no network)
============================================================
1. hwloc_topology_load()
   - Enumerate CPUs, NUMA nodes, caches
   - Enumerate PCIe devices (GPUs, NICs)
   - Record GPU <-> NIC locality (common PCIe ancestor depth)
   - Record NUMA distances

2. ibv_get_device_list() + ibv_query_port()
   - For each RDMA device: name, GUID, port count
   - For each port: state, active_width, active_speed, link_layer, GID table
   - Compute theoretical_bw_gbps = lanes(active_width) * rate_gbps(active_speed)

3. Thunderbolt/USB4 sysfs scan
   - Read /sys/bus/thunderbolt/devices/*/
   - For each device: unique_id, device_name, security level
   - Subscribe to KOBJ_CHANGE for tunnel bandwidth changes

4. Network interface scan
   - Enumerate interfaces via getifaddrs()
   - For each: name, IP addresses, link speed (ethtool IOCTL)

5. GPU capability profiling (R23 integration)
   - cuDeviceGetAttribute() for each GPU: SM count, memory, compute capability
   - Record TFLOPS, VRAM bandwidth, PCIe gen/width

6. Build NodeInfo struct with all discovered hardware
   -> Output: NodeInfo for this node

PHASE 2: CLUSTER JOIN (network)
============================================================
7. Connect to seed node (or discover via mDNS/broadcast)
   - Send JOIN_REQUEST containing our NodeInfo

8. Seed node responds with:
   - Current TopologyGraph (all nodes + links)
   - List of all peer addresses to probe

9. Broadcast our NodeInfo to all existing nodes
   - Each existing node adds us to their local graph

PHASE 3: ACTIVE PROBING (network, per-link)
============================================================
10. For each peer node, for each link type available between us:
    a. RTT probing:
       - Send 10 RTT probes (RDMA WRITE+IMM for RDMA links, TCP echo for TCP)
       - Record min, median, p99
       - Initialize smoothed_rtt_us = median
       - Initialize baseline_rtt_us = min
    b. Bandwidth probing:
       - Send 100 x 1MB RDMA WRITE bursts (one link at a time, not concurrent)
       - Compute measured_bw_gbps = total_bytes / elapsed_time
    c. PTP offset measurement (R26 integration):
       - Exchange PTP timestamps with peer
       - Compute ptp_offset_ns for one-way delay calibration
       - one_way_delay_us = (rtt_us / 2) adjusted by ptp_offset

11. For each discovered link, create LinkInfo struct:
    - link_type from discovery method
    - theoretical_bw from device enumeration
    - measured_bw from active probe
    - smoothed_rtt from RTT probe
    - health = (measured_bw / theoretical_bw) * 0.7
              + (baseline_rtt / smoothed_rtt) * 0.3
    - compress = (measured_bw_gbps < 50.0) [R14 decision]

PHASE 4: GRAPH BUILD + ROUTE COMPUTATION
============================================================
12. Insert all discovered LinkInfo into TopologyGraph
13. Increment TopologyGraph.version
14. Recompute RoutingTable via weighted Dijkstra (Section 3)
15. Generate NCCL topology XML for R20:
    - Each transport appears as a separate device with accurate PCI path (R20 v2)
    - Include bandwidth and latency attributes per link
16. Propagate updated TopologyGraph to all nodes (gossip or direct push)

Total time: ~5 seconds for a 4-node cluster (dominated by bandwidth probing ~1s/link)
```

### 2.2 Steady-State Probing

```
Every 1 second:   Heartbeat to all peers
                   - Piggyback: sequence_number, node_load_metrics, link_utilization
                   - Combine with RTT measurement:
                     smoothed_rtt = 0.875 * smoothed_rtt + 0.125 * measured_rtt
                     rtt_var = 0.75 * rtt_var + 0.25 * |measured_rtt - smoothed_rtt|

Every 60 seconds:  Bandwidth re-probe on all links
                   - Only if link utilization < 30% (don't probe congested links)
                   - Update measured_bw_gbps, recalculate health

On event:          USB4 tunnel change (KOBJ_CHANGE) -> re-probe affected link
                   RDMA async event (IBV_EVENT_PORT_ERR) -> mark link dead
                   Heartbeat miss (3 consecutive) -> suspect node
                   Heartbeat miss (5 consecutive) -> declare node dead
```

---

## 3. Routing Algorithm

### 3.1 Weighted Dijkstra with Congestion Penalty

The cost function for routing. Each link has a cost that combines latency, inverse bandwidth, and congestion:

```
cost(link) = w_lat * normalized_latency(link)
           + w_bw  * normalized_inv_bandwidth(link)
           + w_cong * congestion_penalty(link)

Where:
  normalized_latency(link)     = link.smoothed_rtt_us / max_rtt_across_all_links
  normalized_inv_bandwidth(link) = (1.0 / link.measured_bw_gbps) / max_inv_bw
  congestion_penalty(link)     = (link.utilization) ^ 2

Default weights by transfer priority:
  DemandFetch:  w_lat=0.7, w_bw=0.2, w_cong=0.1   (latency-dominated)
  Interactive:  w_lat=0.5, w_bw=0.3, w_cong=0.2
  Prefetch:     w_lat=0.2, w_bw=0.5, w_cong=0.3   (bandwidth-dominated)
  Migration:    w_lat=0.1, w_bw=0.3, w_cong=0.6   (avoid congestion)
  Background:   w_lat=0.0, w_bw=0.2, w_cong=0.8   (use whatever is free)
```

### 3.2 Routing Table Computation

```rust
fn compute_routing_table(graph: &TopologyGraph) -> RoutingTable {
    let mut entries = HashMap::new();

    for &src in graph.nodes.keys() {
        for &dst in graph.nodes.keys() {
            if src == dst { continue; }

            let mut routes = Vec::new();

            // For each direct link from src to dst, create a Route.
            // (Multi-hop routing is future work; current clusters are fully connected.)
            for &link_idx in &graph.adjacency[&src] {
                let link = &graph.links[link_idx];
                if link.dst != dst { continue; }
                if link.health < 0.1 { continue; } // Skip dead links

                // Compute cost for each priority class
                for priority in [DemandFetch, Interactive, Prefetch, Migration, Background] {
                    let (w_lat, w_bw, w_cong) = weight_for_priority(priority);
                    let cost = w_lat * (link.smoothed_rtt_us / max_rtt)
                             + w_bw * ((1.0 / link.measured_bw_gbps) / max_inv_bw)
                             + w_cong * (link.utilization.powi(2));

                    routes.push(Route {
                        links: vec![link_idx],
                        total_latency_us: link.one_way_delay_us,
                        bottleneck_bw_gbps: link.measured_bw_gbps,
                        min_priority: priority,
                        cost,
                    });
                }
            }

            // Sort routes by cost (lowest first) within each priority class.
            routes.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
            entries.insert((src, dst), routes);
        }
    }

    RoutingTable {
        version: graph.version,
        entries,
        computed_at: Instant::now(),
    }
}
```

### 3.3 Per-Transfer Route Selection

```rust
fn select_route(
    src: NodeId,
    dst: NodeId,
    transfer_size_bytes: usize,
    priority: TransferPriority,
    table: &RoutingTable,
    graph: &TopologyGraph,
) -> TransferPlan {
    let routes = &table.entries[&(src, dst)];

    // Filter routes appropriate for this priority class.
    let viable: Vec<&Route> = routes.iter()
        .filter(|r| r.min_priority == priority)
        .filter(|r| {
            let link = &graph.links[r.links[0]];
            link.health >= 0.3 && link.available_bw_gbps > 0.1
        })
        .collect();

    if viable.is_empty() {
        // Fallback: accept any route regardless of priority preference.
        let fallback = routes.iter()
            .filter(|r| graph.links[r.links[0]].health >= 0.1)
            .next();
        return TransferPlan::Single(fallback.expect("no route to destination"));
    }

    const STRIPE_THRESHOLD: usize = 1 * 1024 * 1024; // 1 MB

    if transfer_size_bytes > STRIPE_THRESHOLD && viable.len() > 1 {
        // Multi-path striping (Section 4).
        TransferPlan::Stripe(compute_stripe_plan(&viable, transfer_size_bytes, graph))
    } else {
        // Single best route.
        TransferPlan::Single(viable[0])
    }
}
```

### 3.4 Recomputation Triggers

The routing table is recomputed when:

1. **Topology change:** Node join/leave, link added/removed. Immediate.
2. **Significant metric shift:** Any link's measured bandwidth changes by >20% or RTT changes by >50%. Checked every heartbeat cycle.
3. **Periodic refresh:** Every 60 seconds regardless, to catch gradual degradation.
4. **Congestion penalty update:** Utilization values are updated in-place without full recompute. Only the `cost` field of affected routes is recalculated (O(affected_links) not O(V*E)).

Recomputation cost: <1 us for 8 nodes with ~20 links (Floyd-Warshall or per-pair Dijkstra).

---

## 4. Multi-Path Striping Algorithm

### 4.1 Chunk Distribution

Split a transfer into contiguous chunks proportional to each link's available bandwidth. This avoids per-page reordering overhead.

```rust
struct StripePlan {
    /// One entry per link participating in the stripe.
    chunks: Vec<StripeChunk>,
    /// Total pages across all chunks.
    total_pages: usize,
}

struct StripeChunk {
    /// Which link to send this chunk over.
    link_idx: usize,
    /// Contiguous range of pages: [start_page, start_page + page_count).
    start_page: usize,
    page_count: usize,
    /// Expected completion time for this chunk (for tail-latency estimation).
    estimated_time_us: f64,
}

fn compute_stripe_plan(
    routes: &[&Route],
    transfer_size_bytes: usize,
    graph: &TopologyGraph,
) -> StripePlan {
    let page_size: usize = 64 * 1024; // 64KB pages (R10 v2)
    let total_pages = (transfer_size_bytes + page_size - 1) / page_size;

    // Collect available bandwidth per link.
    let mut link_bws: Vec<(usize, f64)> = routes.iter()
        .map(|r| {
            let link = &graph.links[r.links[0]];
            (r.links[0], link.available_bw_gbps.max(0.1)) // Floor at 0.1 to avoid zero
        })
        .collect();

    let total_bw: f64 = link_bws.iter().map(|(_, bw)| bw).sum();

    // Distribute pages proportional to available bandwidth.
    let mut chunks = Vec::new();
    let mut pages_assigned: usize = 0;

    for (i, (link_idx, bw)) in link_bws.iter().enumerate() {
        let fraction = bw / total_bw;
        let page_count = if i == link_bws.len() - 1 {
            // Last link gets the remainder to avoid rounding loss.
            total_pages - pages_assigned
        } else {
            (total_pages as f64 * fraction).round() as usize
        };

        if page_count == 0 { continue; }

        let link = &graph.links[*link_idx];
        let chunk_bytes = page_count * page_size;
        let transfer_time_us = (chunk_bytes as f64 * 8.0)
                             / (link.available_bw_gbps * 1e3); // Gbps -> us
        let estimated_time_us = transfer_time_us + link.smoothed_rtt_us;

        chunks.push(StripeChunk {
            link_idx: *link_idx,
            start_page: pages_assigned,
            page_count,
            estimated_time_us,
        });

        pages_assigned += page_count;
    }

    // Tail-latency optimization: if the slowest chunk takes >10% longer than
    // the fastest, steal pages from the slow link and give them to the fast one.
    balance_chunks(&mut chunks, graph);

    StripePlan { chunks, total_pages }
}

/// Iteratively rebalance until all chunks finish within 5% of each other.
fn balance_chunks(chunks: &mut Vec<StripeChunk>, graph: &TopologyGraph) {
    const MAX_ITERATIONS: usize = 5;
    const TOLERANCE: f64 = 0.05; // 5%

    for _ in 0..MAX_ITERATIONS {
        let max_time = chunks.iter().map(|c| c.estimated_time_us)
            .fold(0.0f64, f64::max);
        let min_time = chunks.iter().map(|c| c.estimated_time_us)
            .fold(f64::MAX, f64::min);

        if (max_time - min_time) / max_time < TOLERANCE { break; }

        // Find slowest and fastest chunks.
        let slow_idx = chunks.iter().position(|c| c.estimated_time_us == max_time).unwrap();
        let fast_idx = chunks.iter().position(|c| c.estimated_time_us == min_time).unwrap();

        // Move 1 page from slow to fast.
        if chunks[slow_idx].page_count > 1 {
            chunks[slow_idx].page_count -= 1;
            chunks[fast_idx].page_count += 1;

            // Recalculate estimated times.
            for chunk in chunks.iter_mut() {
                let link = &graph.links[chunk.link_idx];
                let chunk_bytes = chunk.page_count * 64 * 1024;
                let transfer_time_us = (chunk_bytes as f64 * 8.0)
                                     / (link.available_bw_gbps * 1e3);
                chunk.estimated_time_us = transfer_time_us + link.smoothed_rtt_us;
            }
        } else {
            break;
        }
    }
}
```

### 4.2 Striping Across ConnectX-5 + USB4 (Example)

```
Transfer: 1000 pages (64 MB) from Node A to Node B
Links:
  ConnectX-5 RDMA: available_bw = 90 Gbps (100 Gbps - 10% utilization)
  USB4:            available_bw = 60 Gbps (80 Gbps - 25% utilization for DP tunnel)

Total available: 150 Gbps

Distribution:
  ConnectX-5: 1000 * (90/150) = 600 pages (pages 0-599)
  USB4:       1000 * (60/150) = 400 pages (pages 600-999)

Expected times:
  ConnectX-5: 600 * 64KB * 8 / 90 Gbps = 3413 us + 2 us RTT = ~3415 us
  USB4:       400 * 64KB * 8 / 60 Gbps = 3413 us + 7 us RTT = ~3420 us
  (Well balanced: within 0.15%)

Single-link time (ConnectX-5 only):
  1000 * 64KB * 8 / 90 Gbps = 5689 us

Speedup from striping: 5689 / 3420 = 1.66x
```

### 4.3 Scatter-Gather Fragment Routing (R28 Integration)

For scatter-gather operations where a transfer consists of non-contiguous fragments:

```
Input: list of (page_id, source_node) pairs for a gather operation
For each source_node:
  1. Group all pages needed from that source
  2. Select route(s) to that source using select_route()
  3. If group size > STRIPE_THRESHOLD, stripe across available links
  4. Each fragment within a link's chunk is a single RDMA READ
```

This integrates with R28's scatter-gather routing: R17 decides which link carries which fragments, R28 handles the RDMA scatter-gather lists.

---

## 5. Data Placement Scorer

### 5.1 Scoring Formula

When R10 asks "where should this allocation go?" or "which node should this page migrate to?":

```
score(node, allocation) = w_aff  * access_affinity(node, allocation)
                        + w_cost * (1.0 - migration_cost(node, allocation))
                        + w_cap  * tier_capacity(node)
                        + w_bw   * link_bandwidth(node, requesting_gpu)
                        + w_gpu  * gpu_capability(node, workload_type)

Default weights:
  w_aff  = 0.35   (locality is most important)
  w_cost = 0.10   (migration cost is a tiebreaker)
  w_cap  = 0.15   (don't overload nodes)
  w_bw   = 0.25   (bandwidth to requester matters heavily)
  w_gpu  = 0.15   (GPU capability match, R23 integration)
```

### 5.2 Sub-Score Definitions

```
access_affinity(node, alloc):
  If alloc is new (cuMemAlloc): 1.0 if requesting GPU is on this node, else 0.0
  If alloc is existing page with history:
    = access_count_from_this_node / total_access_count
    Requires R10 PTE per-GPU access tracking.

migration_cost(node, alloc):
  = 1.0 - (transfer_time_us / max_acceptable_transfer_time_us)
  transfer_time_us = alloc.size_bytes / best_link_bandwidth_to(node) + best_link_rtt_to(node)
  max_acceptable = 1000 us (1 ms)
  Clamped to [0.0, 1.0].

tier_capacity(node):
  = node.free_vram_bytes / node.total_vram_bytes
  Range: 0.0 (full) to 1.0 (empty).
  If requesting DRAM tier: use free_dram_bytes / total_dram_bytes instead.

link_bandwidth(node, requesting_gpu):
  = best_link_bw_gbps(requesting_gpu.node, node) / max_link_bw_in_cluster
  Where best_link_bw = highest measured_bw_gbps of any link between the two nodes.
  Range: 0.0 to 1.0.

gpu_capability(node, workload_type):
  Based on workload_type (inferred from kernel analysis or hint):
    Training:   score = node.gpu.fp16_tflops / max_fp16_in_cluster
    Inference:  score = node.gpu.fp32_tflops / max_fp32_in_cluster
    Memory-bound: score = node.gpu.vram_bandwidth_gbps / max_vram_bw_in_cluster
  Default (unknown workload): average of all three.
```

### 5.3 Replication Decision

```
If the highest-scoring node's access_affinity < 0.6
   AND multiple nodes have access_affinity > 0.2
   AND the allocation is read-only (from R12 dedup status or kernel analysis):
     -> replicate = true
     -> replica_targets = all nodes with access_affinity > 0.2
Otherwise:
     -> replicate = false, place on highest-scoring node.
```

### 5.4 Migration Cost-Benefit Gate (ARMS-Style)

Before executing any migration recommended by the placement scorer:

```
benefit = hotness * hot_age * latency_reduction
cost    = page_size / link_bandwidth + disruption_penalty

hotness         = page.accesses_per_second (from R10 PTE access_count / time)
hot_age         = min(time_since_page_became_hot, 100ms) / 100ms   [0.0 to 1.0]
latency_reduction = current_access_latency - target_access_latency (us)
disruption_penalty = 10 us (page table update + brief unavailability)

Migrate if: benefit > 2.0 * cost   (require 2x benefit to cover cost + uncertainty)
```

---

## 6. Dynamic Topology Updates

### 6.1 Node Join Protocol

```
Time 0: New node N sends JOIN_REQUEST to any known seed node.
        Payload: N's NodeInfo (hardware profile).

Time ~1ms: Seed node S receives JOIN_REQUEST.
        S adds N to its local graph (no links yet).
        S broadcasts NODE_JOINED(N.NodeInfo) to all other nodes.
        S sends TOPOLOGY_SNAPSHOT(full graph) to N.

Time ~5ms: All nodes receive NODE_JOINED.
        Each node adds N to its local graph.
        Each node starts probing N (RTT + bandwidth).

Time ~1s: RTT probes complete (10 samples per peer per link).
        Each node creates LinkInfo entries for links to N.
        Nodes exchange LINK_DISCOVERED messages.

Time ~3s: Bandwidth probes complete (100 x 1MB per link).
        LinkInfo entries updated with measured_bw.

Time ~4s: All nodes have full link information for N.
        graph.version incremented.
        RoutingTable recomputed at each node.
        NCCL topology XML regenerated (R20 integration).

Time ~5s: N is fully integrated. Transfers can route through N.
```

### 6.2 Node Leave Protocol

**Graceful leave:**
```
1. Departing node D sends LEAVE_INTENT to all peers.
2. All nodes mark D's links as health=0.0.
3. RoutingTable recomputed immediately (routes bypass D).
4. R10 migration engine moves pages off D (if time allows).
5. D sends LEAVE_CONFIRMED.
6. All nodes remove D from graph. Version incremented.
```

**Crash detection:**
```
1. Heartbeat timeout (3 consecutive misses = 3 seconds).
2. Suspecting node sends SUSPECT(D) to all other nodes.
3. All nodes probe D directly (confirmation probes, 2 at 500ms intervals).
4. If no response after confirmation: declare D dead.
5. All nodes mark D's links as health=0.0.
6. RoutingTable recomputed.
7. R15 (Fault Tolerance) handles data recovery for pages homed on D.
8. D removed from graph after data recovery completes.
```

### 6.3 Graph Version Propagation

```
Every topology change increments graph.version (u64).

Propagation method:
  - Small cluster (<=8 nodes): direct broadcast. On change, the node that
    detected the change sends TOPOLOGY_UPDATE to every other node.
  - TOPOLOGY_UPDATE contains: version, changed_nodes, changed_links.
  - Receiving node merges the update into its local graph.
  - If receiving node's version is newer, it sends its version back (conflict resolution).
  - If versions diverge: higher version wins. Ties broken by node_id of the updater.

Consistency guarantee:
  - Eventually consistent. All nodes converge to the same graph within one
    round-trip time (~2-5 us RDMA, ~50 us TCP) of the change propagating.
  - During the convergence window, different nodes may route slightly differently.
    This is acceptable: at worst, a transfer uses a suboptimal but functional path.
```

### 6.4 Link Degradation Handling

When a link's measured metrics shift significantly (but the link is not dead):

```
On heartbeat:
  new_rtt = measure()
  if |new_rtt - smoothed_rtt| > 0.5 * smoothed_rtt:
    // Significant RTT change
    smoothed_rtt = 0.875 * smoothed_rtt + 0.125 * new_rtt
    trigger_route_recompute_for_affected_pairs()

On bandwidth re-probe:
  new_bw = measure()
  if |new_bw - measured_bw| > 0.2 * measured_bw:
    // >20% bandwidth change
    measured_bw = new_bw
    health = recalculate()
    if health < 0.5:
      log_warning("Link {} degraded to health {}", link.id, health)
    trigger_route_recompute_for_affected_pairs()

Dampening:
  - Don't recompute routes more than once per 100ms for the same link.
  - Require utilization difference >20% before shifting traffic between links
    (prevents oscillation: all traffic shifts to same link repeatedly).
  - Exponential backoff on route changes for the same (src, dst) pair:
    1st change: immediate
    2nd change within 1s: wait 100ms
    3rd change within 1s: wait 500ms
```

---

## 7. Integration Points (Cross-Topic)

### 7.1 R10 v2 (Virtual Memory Manager)

- **R17 -> R10:** `best_node_for(gpu_id, alloc_size)` returns PlacementDecision. R10 calls this during cuMemAlloc interception and tier migration.
- **R10 -> R17:** R10's `PageTable` provides access_count per page for affinity scoring. R10's `MigrationEngine` uses R17's routing to transfer pages.

### 7.2 R11 (Speculative Prefetching)

- **R17 -> R11:** `available_bandwidth(src, dst, priority)` returns Gbps available for prefetch traffic after subtracting demand. R11 adjusts prefetch aggressiveness.
- **R11 -> R17:** R11's prefetch source selection uses `nearest_node_with_page(page_id)` from R17 to find closest copy.

### 7.3 R14 (Transport Compression)

- **R17 -> R14:** LinkInfo.compress flag (set during discovery: compress if measured_bw < 50 Gbps). R14 checks this before compressing.
- **R14 -> R17:** Compressed transfer effective size changes routing: 1 MB tensor compressed to 200 KB routes as small transfer (single path, latency-optimized).

### 7.4 R20 v2 (NCCL Backend)

- **R17 -> R20:** Generates NCCL topology XML with each OuterLink transport as a separate device, accurate PCI BDF paths from hwloc, and per-link bandwidth/latency.
- **R20 -> R17:** NCCL collective traffic patterns feed back into R17's utilization tracking.

### 7.5 R23 (Heterogeneous GPU Mixing)

- **R23 -> R17:** GpuProfile (TFLOPS, VRAM, bandwidth) integrated into NodeInfo.
- **R17 -> R23:** PlacementDecision includes gpu_capability sub-score that accounts for GPU strength match to workload.

### 7.6 R26 (PTP Clock Synchronization)

- **R26 -> R17:** PTP-measured clock offset enables accurate one-way delay calculation. LinkInfo.one_way_delay_us is more accurate than RTT/2 for asymmetric links.

### 7.7 R28 (Scatter-Gather Routing)

- **R17 -> R28:** R17 decides which link carries which fragments. R28 builds RDMA scatter-gather lists based on R17's routing decisions.
- **R28 -> R17:** Fragment completion events update R17's utilization tracking.

### 7.8 R29 (RDMA Multicast)

- **R17 -> R29:** Topology graph provides multicast group membership (which nodes need the same data). R29 uses RDMA multicast on links that support it.
- **R29 -> R17:** Multicast traffic counts toward link utilization in congestion tracking.

---

## 8. Open Questions Carried Forward

### Must Answer Before Implementation

1. **hwloc2 Rust crate completeness** -- Does it expose GPU NUMA nodes, PCIe link speed, and I/O device enumeration? If not, how much raw FFI do we need?

2. **USB4 bandwidth query API** -- Can we programmatically query current allocated bandwidth per tunnel type, or only react to KOBJ_CHANGE events?

3. **ConnectX-5 RDMA async event catalog** -- Which IBV_EVENT_* values indicate link health issues vs transient errors? Need an exhaustive mapping.

4. **NCCL topology XML version** -- Which NCCL versions accept our generated XML? Minimum NCCL 2.18+ target.

### Resolved from v1

5. ~~Heartbeat overhead at 8 nodes~~ -- 7 heartbeats/sec * 100 bytes = 700 B/s. Negligible. Confirmed.

6. ~~Routing table architecture~~ -- Static table with congestion overlay (Section 3). Recompute on topology change.

7. ~~Multi-path striping granularity~~ -- Chunk-based (Section 4). Per-page striping rejected for Phase 1.

---

## Related Documents

- [R17 preplan v1](./preplan.md) -- superseded by this document
- [research/01-topology-discovery.md](./research/01-topology-discovery.md) -- discovery methods
- [research/02-routing-algorithms.md](./research/02-routing-algorithms.md) -- routing algorithms
- [research/03-data-placement.md](./research/03-data-placement.md) -- placement strategies
- R10 v2 -- PageTable trait, MigrationEngine, 64KB pages, Robin Hood hash table
- R11 -- Prefetch source selection uses topology data
- R14 -- Per-link compression decision
- R20 v2 -- NCCL topology XML with accurate PCI paths
- R23 -- GPU capability profiles in topology graph
- R26 -- PTP-measured RTT for link latency
- R28 -- Scatter-gather fragment routing
- R29 -- Multicast group membership
