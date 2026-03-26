# R17 Research: Topology Discovery and Representation

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Survey methods for discovering cluster topology (devices, links, speeds, latencies, NUMA distances) and representing it as a queryable graph model. This covers RDMA device enumeration, PCIe topology via sysfs, USB4/Thunderbolt tunnel discovery, active link probing, topology graph representation, dynamic topology changes, and NCCL's topology detection as a reference implementation.

## TL;DR — What Works and What OuterLink Should Use

| Discovery Method | What It Reveals | OuterLink Applicability |
|---|---|---|
| `ibv_get_device_list` + `ibv_query_port` | RDMA devices, link speed, width, layer | Enumerate ConnectX-5 NICs, compute bandwidth per port |
| sysfs PCIe hierarchy | PCIe topology tree, bridge/switch structure | Map GPU-to-NIC locality, find shared PCIe switches |
| `/sys/devices/system/node/` NUMA | NUMA distances, CPU-to-device affinity | Optimize pinned memory allocation, avoid cross-NUMA penalties |
| hwloc library | Unified view: CPUs, caches, NUMA, PCIe, GPUs | Single API for full intra-node topology discovery |
| Thunderbolt sysfs + kernel events | USB4 routers, tunnels, bandwidth allocation | Detect USB4 links, track tunnel creation/teardown |
| Active RTT probing | Actual link latencies under load | Ground-truth latency map between all node pairs |
| NCCL `ncclTopoGetSystem` | PCI path analysis, GPU interconnect graph | Reference for how to build topology XML; align our graph with NCCL's |

**The single biggest lesson:** No single discovery method covers all link types. OuterLink needs a multi-source discovery pipeline: enumerate devices via libibverbs and sysfs, probe latencies actively, and fuse everything into a unified weighted graph.

---

## 1. RDMA Device Enumeration (libibverbs)

### 1.1 Device Discovery

The `ibv_get_device_list()` function returns a NULL-terminated array of all RDMA devices available on the system. Each device is opened with `ibv_open_device()` to get a context for further queries.

**Discovery flow:**
1. `ibv_get_device_list(&num_devices)` — enumerate all RDMA devices
2. `ibv_get_device_name(device)` — get device name (e.g., `mlx5_0`)
3. `ibv_get_device_guid(device)` — get 64-bit globally unique identifier
4. `ibv_open_device(device)` — get context handle
5. `ibv_query_device(context, &device_attr)` — get device-level attributes
6. `ibv_query_port(context, port_num, &port_attr)` — get per-port attributes

### 1.2 Port Attributes for Link Characterization

`ibv_query_port()` returns `struct ibv_port_attr` with fields critical for topology:

| Field | What It Tells Us | Values |
|---|---|---|
| `state` | Port operational state | `IBV_PORT_ACTIVE` = usable |
| `active_width` | Number of active lanes | 1=1x, 2=4x, 4=8x, 8=12x |
| `active_speed` | Per-lane signaling rate | 1=SDR(2.5G), 2=DDR(5G), 4=QDR(10G), 16=FDR(14G), 32=EDR(25G) |
| `link_layer` | Protocol on wire | `IBV_LINK_LAYER_INFINIBAND` or `IBV_LINK_LAYER_ETHERNET` |
| `max_mtu` / `active_mtu` | Maximum transfer unit | Typically 4096 bytes for IB |
| `gid_tbl_len` | Number of GID entries | Used for RoCE addressing |

### 1.3 Bandwidth Calculation

**Formula:** `bandwidth_gbps = lanes(active_width) * rate_gbps(active_speed)`

For ConnectX-5 at 100GbE (Ethernet mode with RoCE):
- `active_width` = 2 (meaning 4x)
- `active_speed` = 32 (meaning EDR, 25 Gb/s per lane)
- **Result: 4 * 25 = 100 Gb/s**

Practical throughput: ~12.4 GB/s unidirectional (99.2 Gb/s), measured with `ib_write_bw`.

Helper functions `ibv_rate_to_mbps()` and `mbps_to_ibv_rate()` convert between enum values and megabits.

### 1.4 Multi-Port and Multi-Device Discovery

A single ConnectX-5 card may expose 1 or 2 ports. Each port can connect to a different network. OuterLink must enumerate ALL ports across ALL devices to find every available RDMA path.

**OuterLink relevance:** On Pedro's setup, each node has one ConnectX-5 with typically one 100GbE port. But the discovery must be general for future multi-NIC or dual-port configurations.

### 1.5 Subnet Discovery Tools (Reference)

For InfiniBand fabrics, `ibnetdiscover` performs full subnet topology discovery, revealing switches, HCAs, and their interconnections. `ibdiagnet` validates link speeds and widths match expectations. These are diagnostic tools, not programmatic APIs, but useful for validating OuterLink's discovery.

---

## 2. PCIe Topology via sysfs

### 2.1 Hierarchy in /sys/devices

The Linux kernel materializes the PCIe topology as a directory hierarchy under `/sys/devices/pci<domain>:<bus>/`. Each device appears as a subdirectory named by its BDF (Bus:Device.Function) address, e.g., `0000:41:00.0`.

**Key principle:** Parent-child relationships in the directory tree mirror physical PCIe switch/bridge relationships. Two devices sharing a parent bridge share a PCIe switch and can communicate with lower latency.

### 2.2 Critical sysfs Files per Device

| Path | Content | Use |
|---|---|---|
| `/sys/bus/pci/devices/<BDF>/class` | PCI class code | Identify GPUs (0x030000), NICs (0x020000) |
| `/sys/bus/pci/devices/<BDF>/vendor` | Vendor ID | NVIDIA = 0x10de, Mellanox = 0x15b3 |
| `/sys/bus/pci/devices/<BDF>/numa_node` | NUMA affinity | Place pinned memory on correct NUMA node |
| `/sys/bus/pci/devices/<BDF>/max_link_speed` | Max PCIe speed | e.g., "16.0 GT/s" for PCIe 4.0 |
| `/sys/bus/pci/devices/<BDF>/max_link_width` | Max PCIe width | e.g., "16" for x16 |
| `/sys/bus/pci/devices/<BDF>/current_link_speed` | Current speed | May be lower than max (power saving) |
| `/sys/bus/pci/devices/<BDF>/current_link_width` | Current width | May be lower than max (degraded) |

### 2.3 Determining GPU-NIC Locality

The critical question: are the GPU and NIC on the same PCIe root complex / NUMA node?

**Method 1 — NUMA node comparison:**
```
# If both return the same value, they share a NUMA node
cat /sys/bus/pci/devices/0000:41:00.0/numa_node  # GPU
cat /sys/bus/pci/devices/0000:81:00.0/numa_node  # NIC
```

**Method 2 — Common ancestor in PCIe tree:**
Use `lspci -tv` or walk the sysfs hierarchy to find the common parent bridge. Fewer hops between GPU and NIC = lower DMA latency.

**Method 3 — hwloc (recommended):**
hwloc provides a unified API that resolves both NUMA and PCIe locality in a single call.

### 2.4 PCIe Bandwidth by Generation

| PCIe Gen | Per-Lane Rate | x16 Bandwidth | x4 Bandwidth |
|---|---|---|---|
| 3.0 | 8 GT/s (~1 GB/s) | ~16 GB/s | ~4 GB/s |
| 4.0 | 16 GT/s (~2 GB/s) | ~32 GB/s | ~8 GB/s |
| 5.0 | 32 GT/s (~4 GB/s) | ~64 GB/s | ~16 GB/s |

OuterLink's GPUs (RTX 3090) use PCIe 4.0 x16 = ~32 GB/s (256 Gb/s). ConnectX-5 uses PCIe 3.0 x16 = ~16 GB/s (128 Gb/s). The NIC is the bottleneck for GPU-to-NIC transfers, not the GPU's PCIe slot.

---

## 3. NUMA Distance Matrices

### 3.1 What NUMA Distance Means

NUMA (Non-Uniform Memory Access) means different memory regions have different access latencies from different CPUs. The kernel exposes this as a distance matrix:

```
/sys/devices/system/node/node0/distance -> "10 21"
/sys/devices/system/node/node1/distance -> "21 10"
```

A distance of 10 = local access. Higher values = more hops through interconnect (QPI/UPI on Intel, Infinity Fabric on AMD).

### 3.2 Impact on OuterLink

Cross-NUMA memory access adds ~50-100ns latency penalty. For host-staged transfers:
- If pinned memory and NIC are on different NUMA nodes, every DMA to/from the staging buffer crosses the interconnect
- For a 1 MB transfer at 12 GB/s, NUMA penalty is negligible
- For small transfers (<64 KB), NUMA penalty can increase latency by 30-50%

**Decision:** OuterLink's pinned staging buffers MUST be allocated on the same NUMA node as the NIC performing the DMA. Use `numa_alloc_onnode()` or `set_mempolicy()`.

### 3.3 NUMA-Aware Allocation in Rust

The `libnuma` crate or direct syscalls (`mbind`, `set_mempolicy`) can enforce NUMA placement. CUDA's `cudaHostAlloc` does NOT guarantee NUMA affinity — OuterLink should use `mmap` + `mbind` + `cudaHostRegister` instead.

---

## 4. hwloc: Unified Hardware Locality

### 4.1 What hwloc Provides

hwloc (Hardware Locality) from the Open MPI project discovers the complete hardware topology in a single unified model:
- CPU packages, cores, threads
- Cache hierarchy (L1/L2/L3)
- NUMA nodes with memory sizes
- PCIe devices with full tree structure
- I/O devices: GPUs, NICs, NVMe drives
- OS device objects with attributes (GPU memory, NIC speed)

### 4.2 Why hwloc Over Raw sysfs

| Aspect | Raw sysfs | hwloc |
|---|---|---|
| Portability | Linux only | Linux, Solaris, Windows, macOS, FreeBSD |
| API complexity | String parsing, tree walking | C API with typed objects |
| GPU memory | Not directly exposed | Exposed via `GPUMemory` subtype |
| NUMA distances | Separate parsing | Integrated with topology tree |
| PCIe locality | Manual tree walking | `hwloc_get_common_ancestor_obj()` |

### 4.3 Key API for OuterLink

```c
// Discover topology
hwloc_topology_t topo;
hwloc_topology_init(&topo);
hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
hwloc_topology_load(topo);

// Find GPU's NUMA node
hwloc_obj_t gpu = hwloc_get_pcidev_by_busid(topo, domain, bus, dev, func);
hwloc_obj_t numa = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_NUMANODE, gpu);

// Find common ancestor of GPU and NIC (fewer hops = better locality)
hwloc_obj_t ancestor = hwloc_get_common_ancestor_obj(topo, gpu_obj, nic_obj);
```

### 4.4 Rust Bindings

The `hwloc2` crate provides Rust bindings to hwloc. It covers topology loading, object iteration, and bitmap operations. OuterLink should use this for intra-node topology discovery.

**Recommendation:** Use hwloc as the primary intra-node discovery method. Fall back to raw sysfs only if hwloc is not installed (rare on HPC systems).

---

## 5. USB4/Thunderbolt Tunnel Discovery

### 5.1 Kernel Driver Architecture

The Linux kernel's unified `thunderbolt` driver handles both Thunderbolt 3 and USB4 devices. It supports two connection manager modes:
- **Firmware CM:** Traditional PCs with Thunderbolt 3 (firmware manages tunnels)
- **Software CM:** Newer USB4 devices, Apple systems (kernel manages tunnels)

### 5.2 Discovery Path

1. **Device enumeration:** Devices appear in `/sys/bus/thunderbolt/devices/` as router objects
2. **Tunnel discovery:** `tb_discover_tunnels()` finds tunnels created by boot firmware
3. **Tunnel types:** PCIe tunneling, DisplayPort tunneling, USB 3.x tunneling, Ethernet/Network tunneling
4. **Security:** PCIe tunneling may require user authorization (security level `user` or `secure`)

### 5.3 Sysfs Topology for USB4

Each Thunderbolt/USB4 device exposes:
- `/sys/bus/thunderbolt/devices/<device>/device_name` — device identification
- `/sys/bus/thunderbolt/devices/<device>/unique_id` — UUID for persistent identification
- `/sys/bus/thunderbolt/devices/domainX/security` — security level for the domain

### 5.4 Bandwidth Detection

USB4 bandwidth is shared across tunnel types. The kernel sends `KOBJ_CHANGE` events with `TUNNEL_EVENT` environment variables when bandwidth allocation changes.

For OuterLink's USB4 links:
- USB4 40Gbps: 5 GB/s theoretical (PCIe tunneling gets a portion)
- USB4 80Gbps (USB4 v2): ~10 GB/s theoretical
- Actual available bandwidth depends on concurrent tunnel usage (DisplayPort, USB 3.x)

**Challenge:** USB4 bandwidth is not fixed — it's dynamically allocated across tunnels. OuterLink must monitor tunnel events and re-query available bandwidth.

### 5.5 Intel tbtools for Debugging

Intel's `tbtools` package provides:
- `tbtunnels` — dump all active tunnels
- `tbman` — visual topology viewer
- `tbmonitor` — event monitoring

These are useful for development/debugging but not for programmatic discovery.

---

## 6. Active Link Probing

### 6.1 Why Probing is Necessary

Device enumeration reveals theoretical capabilities, but actual performance depends on:
- Cable quality and length
- Switch congestion
- Other traffic on shared links
- Thermal throttling
- Driver/firmware issues

Active probing measures ground-truth latency and bandwidth between every pair of nodes.

### 6.2 RTT (Latency) Measurement

**Method:** Send small probe messages and measure round-trip time.

| Probe Type | What It Measures | Implementation |
|---|---|---|
| RDMA WRITE + IMM | RDMA path latency | Post RDMA WRITE with immediate data, time completion |
| TCP ping | TCP path latency | Send 1-byte TCP message, wait for echo |
| ICMP ping | IP-level latency | Standard `ping`, but doesn't test RDMA path |
| ibv_rc_pingpong | Verbs-level latency | libibverbs test tool, ~1.5 us for ConnectX-5 |

**Sampling strategy:**
- At node join: 10 RTT probes to every existing node, record min/median/p99
- Steady-state: 1 probe per link every 5 seconds (heartbeat + RTT combined)
- On suspicion: burst 5 probes to confirm degradation

**Smoothed RTT:** Use exponential moving average (EMA) with alpha=0.125 (TCP-style):
`sRTT = (1 - alpha) * sRTT + alpha * measured_RTT`

### 6.3 Bandwidth Estimation

**Method:** Send a burst of large messages and measure throughput.

- Use RDMA WRITE of 1 MB blocks, time 100 consecutive writes
- Run for each link type independently (don't probe ConnectX and USB4 simultaneously)
- Run at node join and periodically (every 60 seconds) to detect degradation

**Estimated bandwidth vs theoretical:**

| Link Type | Theoretical | Typical Measured | Ratio |
|---|---|---|---|
| ConnectX-5 100GbE (RDMA) | 12.5 GB/s | 11.5-12.2 GB/s | ~95% |
| ConnectX-5 100GbE (TCP) | 12.5 GB/s | 8-10 GB/s | ~70-80% |
| USB4 80Gbps | 10 GB/s | 6-8 GB/s | ~65-80% |
| OCuLink PCIe 4.0 x4 | 8 GB/s | 7-7.5 GB/s | ~90% |
| TCP/25GbE | 3.1 GB/s | 2.5-2.9 GB/s | ~80-90% |

### 6.4 Link Health Scoring

Combine RTT and bandwidth measurements into a single health score per link:

```
health = (measured_bw / theoretical_bw) * 0.7 + (baseline_rtt / measured_rtt) * 0.3
```

Score range: 0.0 (dead) to 1.0 (perfect). Links below 0.5 should be flagged for avoidance.

---

## 7. Topology Representation (Graph Model)

### 7.1 Data Model

The topology is a weighted, directed multigraph:
- **Nodes:** Physical machines in the cluster
- **Edges:** Network links between machines (multiple edges possible between same pair)
- **Edge weights:** Bandwidth (GB/s), latency (us), link type, health score

```rust
struct TopologyGraph {
    nodes: HashMap<NodeId, NodeInfo>,
    edges: Vec<Link>,
    adjacency: HashMap<NodeId, Vec<LinkId>>,
}

struct NodeInfo {
    id: NodeId,
    hostname: String,
    gpus: Vec<GpuInfo>,           // GPU count, VRAM, PCIe gen
    numa_topology: NumaInfo,       // NUMA nodes, distances
    total_vram_bytes: u64,
    total_dram_bytes: u64,
}

struct Link {
    id: LinkId,
    src: NodeId,
    dst: NodeId,
    link_type: LinkType,           // RDMA, USB4, OCuLink, TCP
    theoretical_bw_gbps: f64,      // From device enumeration
    measured_bw_gbps: f64,         // From active probing
    measured_rtt_us: f64,          // From active probing
    health: f64,                   // 0.0 to 1.0
    mtu: u32,
    bidirectional: bool,           // Most links are symmetric
}

enum LinkType {
    RdmaRoCE,       // ConnectX-5 RoCE v2
    RdmaIB,         // InfiniBand (future)
    Usb4,           // USB4/Thunderbolt
    OcuLink,        // OCuLink PCIe tunnel
    TcpEthernet,    // Plain TCP over Ethernet
}
```

### 7.2 Adjacency Matrix for Fast Lookups

For a small cluster (2-8 nodes), an adjacency matrix is efficient:

```
          Node0   Node1   Node2   Node3
Node0     [  0,    100,     80,     25  ]   // bandwidth in Gbps
Node1     [100,      0,    100,     80  ]
Node2     [ 80,    100,      0,     25  ]
Node3     [ 25,     80,     25,      0  ]
```

For multi-link scenarios, use a matrix per link type, or store the best link's metrics.

### 7.3 Multi-Link Representation

Two nodes connected by both ConnectX-5 (100Gbps) and USB4 (80Gbps) have an aggregate capacity of ~180Gbps. The graph must represent both links separately for the routing layer to make per-transfer decisions.

### 7.4 Graph Operations Needed

| Operation | Use Case | Algorithm |
|---|---|---|
| Shortest path (latency) | Route latency-sensitive small transfers | Dijkstra with latency weights |
| Widest path (bandwidth) | Route bandwidth-sensitive large transfers | Modified Dijkstra maximizing min-edge bandwidth |
| All paths | Multi-path striping | K-shortest paths or DFS enumeration |
| Min-cut | Detect bottlenecks | Max-flow / min-cut |
| Connected components | Detect partitions | BFS/DFS |

---

## 8. Dynamic Topology Changes

### 8.1 Events That Change Topology

| Event | Detection Method | Response Time Target |
|---|---|---|
| Node joins cluster | Membership protocol (heartbeat/gossip) | <1 second to start probing |
| Node leaves (graceful) | Membership protocol | <500 ms to update graph |
| Node crashes | Heartbeat timeout | <5 seconds (configurable) |
| Link degrades | RTT/bandwidth probe detects change | <10 seconds to re-score |
| Link fails | Probe timeout or RDMA error event | <2 seconds to remove edge |
| USB4 device hotplug | Thunderbolt kernel event (KOBJ_CHANGE) | <1 second to enumerate |

### 8.2 Failure Detection

**Heartbeat design for OuterLink:**
- Each node sends heartbeat to every peer every 1 second
- Heartbeat includes: timestamp, sequence number, node load metrics
- Timeout: 3 missed heartbeats (3 seconds) = node suspected
- Confirmation: 2 additional probes at 500ms intervals
- Total failure detection time: ~4 seconds

**Phi Accrual Failure Detector** (Cassandra-style):
- Instead of binary alive/dead, compute suspicion level (phi) on continuous scale
- Based on statistical distribution of heartbeat inter-arrival times
- Threshold phi > 8 (roughly 1 in 10,000 chance of false positive)
- Adapts automatically to network jitter

**Recommendation:** Start with fixed-timeout (simpler), implement phi accrual in Phase 2 when false positive rate data is available.

### 8.3 Topology Version and Propagation

Every topology change increments a global version number. Changed topology is propagated via:
1. Direct notification to affected nodes (link failure)
2. Gossip protocol for eventual consistency (background)
3. Each node maintains its own copy; version number resolves conflicts

---

## 9. NCCL's Topology Detection (Reference)

### 9.1 How NCCL Discovers Topology

NCCL builds an XML-based topology representation through:

1. **XML Loading:** Check `/var/run/nvidia-topologyd/virtualTopology.xml` or `NCCL_TOPO_FILE` env var
2. **GPU Detection:** Each rank detects its own GPU via bus ID, calls `ncclTopoFillGpu`
3. **PCI Path Analysis:** Reads sysfs to walk PCIe hierarchy, determines GPU-to-NIC proximity
4. **XML Fusion:** All ranks on a node exchange their XML fragments via `bootstrapIntraNodeAllGather`
5. **System Build:** `ncclTopoGetSystemFromXml()` constructs the full topology graph

### 9.2 NCCL's PCI Distance Levels

| Level | Meaning | Example |
|---|---|---|
| PIX | Same PCI switch | GPU and NIC on same PLX switch |
| PXB | Multiple PCI switches | GPU behind one switch, NIC behind another, same root complex |
| PHB | Same NUMA node | Different root complexes on same CPU socket |
| SYS | Cross NUMA | Different CPU sockets, QPI/UPI crossing required |

### 9.3 NCCL Topology XML Format

```xml
<system version="1">
  <cpu numaid="0" affinity="0-15">
    <pci busid="0000:41:00.0" class="0x030000" vendor="0x10de">
      <gpu dev="0" gdr="1" sm="86" mem="24576" />
    </pci>
    <pci busid="0000:81:00.0" class="0x020000" vendor="0x15b3">
      <net name="mlx5_0" port="1" speed="100000" />
    </pci>
  </cpu>
</system>
```

### 9.4 Alignment with OuterLink

OuterLink's topology graph must align with NCCL's view for the R20 NCCL backend:
- When NCCL asks for topology, OuterLink should generate a compatible XML
- NCCL's ring/tree algorithm selection depends on topology
- OuterLink can set `NCCL_TOPO_FILE` to provide a custom topology that includes cross-node OuterLink transports

**Key integration point:** OuterLink exposes each transport as a "device" to NCCL (from R20 research). The topology graph in R17 must provide the distance/bandwidth data that R20 uses to populate NCCL's topology XML.

---

## 10. OuterLink Discovery Pipeline (Proposed)

### 10.1 Startup Sequence

```
1. hwloc: Discover intra-node topology (CPUs, NUMA, PCIe, GPUs)
2. libibverbs: Enumerate RDMA devices, query port attributes, compute theoretical BW
3. sysfs/thunderbolt: Detect USB4/OCuLink devices
4. Network interfaces: Enumerate all network-capable interfaces
5. Membership: Connect to cluster, exchange node topology summaries
6. Active probe: Measure RTT and bandwidth to every peer
7. Build graph: Construct TopologyGraph with all discovered info
8. NCCL XML: Generate NCCL-compatible topology XML for R20
```

### 10.2 Periodic Refresh

```
Every 5 seconds:   Heartbeat + RTT measurement to all peers
Every 60 seconds:  Bandwidth re-probe on all links
On event:          USB4 hotplug, RDMA error, membership change
                   -> Immediate re-probe of affected link
```

### 10.3 Cost of Discovery

| Operation | Time | CPU | Network |
|---|---|---|---|
| hwloc topology load | ~10 ms | Minimal | None |
| ibv device enumeration | ~1 ms | Minimal | None |
| sysfs reading | ~5 ms | Minimal | None |
| RTT probe (10 samples per peer) | ~50 ms per peer | Minimal | ~10 KB per peer |
| Bandwidth probe (1 MB x 100) | ~1 second per link | Low | ~100 MB per link |
| Total initial discovery (4-node cluster) | ~5 seconds | Low | ~400 MB |

---

## Open Questions

### Must Answer Before Detailed Planning

1. **Does hwloc detect OCuLink devices properly?** OCuLink is PCIe-based, so it should appear in the PCIe hierarchy, but need to verify hwloc exposes it with correct bandwidth attributes.

2. **Can we read USB4 available bandwidth programmatically?** The kernel exposes tunnel events, but is there an API to query current allocated bandwidth per tunnel type?

3. **How does RDMA error reporting work for link degradation?** ConnectX-5 may report errors via the async event queue (`ibv_get_async_event`) — need to catalog which events indicate link health issues vs transient errors.

4. **What is the actual RTT between Pedro's machines?** Need baseline measurements to calibrate the probing system. Expected: ~2 us for ConnectX-5 RDMA, ~5-10 us for USB4, ~50 us for TCP.

### Can Answer During Implementation

5. Is hwloc's Rust binding (`hwloc2` crate) feature-complete enough, or do we need raw FFI?

6. What is the overhead of continuous 5-second heartbeat probing at scale (16+ nodes)?

7. Should the topology graph be stored in shared memory (mmap) for multiple OuterLink processes on the same node to access?

---

## Related Documents

- [R17 README](../README.md) — summary and folder contents
- [research/02-routing-algorithms.md](./02-routing-algorithms.md) — how to route over this topology
- [research/03-data-placement.md](./03-data-placement.md) — where to place data based on topology
- [R20 NCCL Backend](../../../phase-07-memory-intelligence/R20-nccl-backend/README.md) — NCCL topology integration
- [R10 Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — memory hierarchy this topology feeds into
- [CONSOLIDATION](../../../../research/CONSOLIDATION-all-research.md) — transport decisions (ConnectX-5, BAR1, etc.)
