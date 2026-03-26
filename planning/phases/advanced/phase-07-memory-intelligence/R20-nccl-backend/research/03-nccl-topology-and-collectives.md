# R20 Research: NCCL Topology Detection and Collectives

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Understand how NCCL uses the transport layer internally — topology detection, algorithm selection (ring/tree), collective decomposition into point-to-point operations, and channel management. This knowledge is critical for OuterLink to report accurate properties and achieve optimal performance through the NCCL plugin.

---

## 1. NCCL Topology Detection

### How NCCL Discovers Hardware

NCCL builds an internal topology graph (`ncclTopoSystem`) during initialization:

1. **GPU enumeration:** Discovers all CUDA-visible GPUs, their PCI paths, NVLink connections, and NVSwitch presence
2. **NIC enumeration:** Queries each network plugin for devices and their properties (PCI path, speed, latency)
3. **PCI topology:** Uses PCI paths to determine which NICs are closest to which GPUs (same socket, same PCIe switch, etc.)
4. **Topology XML:** Can load/dump topology from XML files (`/var/run/nvidia-topologyd/virtualTopology.xml`)

### Why PCI Path Matters

The `pciPath` field in `ncclNetProperties` is how NCCL knows where a network device sits in the PCI hierarchy. NCCL uses this to:

- **Minimize PCI hops:** Route traffic through the NIC closest to the GPU doing the work
- **Rail locality:** Keep traffic "rail-local" (NVLink-connected GPUs share the same NIC)
- **NUMA awareness:** Prefer NICs on the same NUMA node as the GPU

### Implications for OuterLink

OuterLink's NCCL plugin must report accurate PCI paths:

| Transport | PCI Path to Report | Rationale |
|-----------|-------------------|-----------|
| ConnectX-5 NIC | Actual NIC PCI path (e.g., `/sys/bus/pci/devices/0000:41:00.0`) | Standard — NCCL handles the rest |
| USB4 port | Controller PCI path | NCCL uses this for proximity decisions |
| TCP (fallback) | A synthetic or host-bridge path | Low priority, will be treated as "far" |
| OpenDMA | BAR1 target GPU's PCI path | The DMA engine targets a specific GPU |

**Critical decision:** When OuterLink has multiple transport paths to the same remote GPU (e.g., ConnectX + USB4), should these be reported as separate devices with different PCI paths? Yes — NCCL's topology engine will select the best one per GPU pair.

### Topology XML Override

NCCL supports loading custom topology via `NCCL_TOPO_FILE`. OuterLink could generate a topology XML that describes its multi-transport network, giving NCCL a complete picture. This is an advanced optimization for later phases.

### Cross-DC Topology (NCCL 2.27+)

NCCL introduced "fabric ID" for cross-data-center awareness. The fabric ID captures topology information and connectivity between devices. OuterLink could use this to describe its multi-hop topologies (e.g., local GPU -> ConnectX -> remote ConnectX -> remote GPU).

---

## 2. Algorithm Selection

### Three Algorithm Families

NCCL implements three families of collective algorithms:

| Algorithm | Pattern | Best For | How It Uses Transport |
|-----------|---------|----------|----------------------|
| **Ring** | Circular pipeline | Large messages, high bandwidth utilization | Each GPU sends to successor, receives from predecessor. 2(k-1) steps for k GPUs |
| **Tree** | Double binary tree | Small-medium messages, lower latency | Parent-child communication. Log(k) depth |
| **PAT** | Pairwise | AllToAll, specific patterns | Direct peer-to-peer between all pairs |

### Ring Algorithm Details

For AllReduce on k GPUs with ring:
1. **ReduceScatter phase:** Each GPU reduces local data with received data, keeps one segment. k-1 steps.
2. **AllGather phase:** Each GPU forwards its reduced segment to all others. k-1 steps.
3. Total: 2(k-1) steps. Each step transfers `data_size / k` bytes.

**Bandwidth utilization:** Ring achieves `(k-1)/k` of peak bidirectional bandwidth — near-optimal for large messages.

**Transport implications:** The plugin handles individual point-to-point send/recv pairs. NCCL manages the ring ordering. OuterLink sees a sequence of `isend`/`irecv` calls between specific peer pairs.

### Tree Algorithm Details

NCCL uses a **double binary tree** structure:
- Two overlapping trees where no node is an internal node in both
- At most one node appears as a leaf in both trees
- Halves bandwidth requirement compared to single tree while achieving tree latency

**Transport implications:** Tree has fewer hops (log k) but less bandwidth utilization. Better for small messages where latency dominates. OuterLink's plugin reports latency in properties, affecting NCCL's ring-vs-tree decision threshold.

### Algorithm Selection Logic

NCCL selects algorithm based on:
- **Message size:** Small = tree, large = ring
- **Topology:** NVSwitch presence enables NVLink SHARP (NVLSTREE)
- **Environment:** `NCCL_ALGO` can force specific algorithm
- **Tuner plugin:** Custom tuning plugin can override selection
- **Transport properties:** Speed and latency from plugin influence thresholds

**The ring-to-tree crossover point** depends on latency. Lower transport latency shifts the crossover toward smaller messages (tree is used less). OuterLink should report accurate latency values:

| OuterLink Transport | Expected Latency | Effect on Algorithm Choice |
|--------------------|-------------------|---------------------------|
| TCP (Phase 1) | ~100-500 us | Tree preferred for wider message range |
| RDMA (Phase 2) | ~2-5 us | Ring preferred except for very small messages |
| OpenDMA (Phase 5) | ~1-2 us | Ring dominant |

---

## 3. Protocol Selection

Within each algorithm, NCCL selects a communication protocol:

| Protocol | Mechanism | Bandwidth | Latency | When Used |
|----------|-----------|-----------|---------|-----------|
| **Simple** | Standard copy | Highest | Higher | Large messages |
| **LL (Low Latency)** | 4B data + 4B flag per 8B | ~50% of Simple | Lower | Small messages |
| **LL128** | 120B data + 8B flag per 128B | ~94% of Simple | Low | Medium messages |

**Transport implications:** LL and LL128 embed completion flags in the data stream to reduce latency. If OuterLink's transport supports this pattern (flag bytes interspersed with data), it can benefit from LL128. This is transparent to the plugin — NCCL handles protocol selection and data formatting before calling `isend`.

---

## 4. Channel Architecture

### What Channels Are

Channels are dedicated, persistent execution contexts on each GPU. They:
- Bind to specific GPU SMs (Streaming Multiprocessors)
- Act as parallel pathways for collective operations
- Each channel has its own ring/tree connections to peers
- Each channel has independent send/recv comm objects in the plugin

### Channel Count and Transport

- Default channel count depends on GPU type and NIC count
- More channels = more parallelism but also more connections
- Each channel creates separate `sendComm`/`recvComm` objects via the plugin
- A 4-GPU ring with 8 channels means 8 independent ring connections per GPU pair

**OuterLink impact:** If NCCL uses 8 channels, OuterLink's plugin must handle 8 concurrent connection pairs per peer. Each pair runs independent async operations. This maps naturally to OuterLink's multi-connection transport.

### Channel-Transport Binding

NCCL assigns channels to network devices (NICs). With multi-NIC systems:
- Channels are distributed across NICs (rail-local assignment)
- Each NIC handles a subset of channels

**OuterLink opportunity:** If OuterLink reports ConnectX as device 0 and USB4 as device 1, NCCL may assign channels across both, using both transports simultaneously. This is exactly how multi-transport should work.

---

## 5. Collective Operation Decomposition

### How NCCL Decomposes Collectives

| Collective | Ring Decomposition | Tree Decomposition |
|-----------|-------------------|-------------------|
| AllReduce | ReduceScatter + AllGather | Reduce-to-root + Broadcast |
| AllGather | k-1 rotations around ring | Broadcast from each node |
| ReduceScatter | k-1 reduce-and-rotate steps | Reduce to subtree roots |
| Broadcast | Ring rotation from root | Tree propagation from root |
| AllToAll | Peer-to-peer between all pairs | (No tree version) |

### What the Plugin Sees

The plugin does NOT see collective operations. It only sees:
- `isend(sendComm, data, size, tag, ...)` — Send a chunk to a specific peer
- `irecv(recvComm, n, data, sizes, tags, ...)` — Receive chunk(s) from a specific peer
- `test(request, ...)` — Poll for completion

NCCL handles all the algorithmic logic (ring ordering, tree parent/child, etc.) internally. The plugin is a point-to-point transport layer.

**This is good for OuterLink:** The plugin implementation is straightforward — map NCCL's send/recv to OuterLink's transport layer. No collective logic needed in the plugin.

### Message Sizes at the Plugin Level

For an AllReduce of N bytes on k GPUs:
- Ring: Each `isend`/`irecv` moves `N/k` bytes per step, across `2(k-1)` steps
- Tree: Each step moves N bytes but fewer steps (log k)

With NCCL's chunking, large messages are further split into smaller chunks that pipeline through the algorithm. Typical chunk sizes: 128KB to 4MB depending on message size and protocol.

**OuterLink impact:** The plugin sees many small-to-medium transfers (128KB-4MB), not huge bulk transfers. Transport optimization should focus on this range.

---

## 6. Grouped Operations and Multi-Receive

### Grouped Receives

NCCL's `irecv` supports grouped receives where multiple buffers are received in a single call:
- `n` parameter specifies the number of buffers
- `maxRecvs` in properties tells NCCL how many the plugin supports
- Allows the plugin to batch-post multiple receives for efficiency

### Point-to-Point Grouped Operations

`ncclGroupStart`/`ncclGroupEnd` allows applications to batch multiple point-to-point operations:
- Each transfer is assigned to a separate channel when possible
- Enables parallel independent sends and receives

### Multi-Request API (v11)

NCCL 2.28+ introduces Multi-Request Net API:
- Plugin reports `maxMultiRequestSize` in properties
- NCCL batches multiple send/recv requests together
- Allows the plugin to optimize for request batching

**OuterLink opportunity:** OuterLink's transport already supports batched operations. The multi-request API maps naturally to OuterLink's batched transfer submission.

---

## 7. Performance Expectations

### What NCCL Expects from a Transport

Based on NCCL's internal benchmarking and tuning:

| Metric | InfiniBand (baseline) | TCP (fallback) | OuterLink Target |
|--------|----------------------|----------------|------------------|
| Bandwidth | 100-400 Gbps | 10-25 Gbps | 100+ Gbps (ConnectX) |
| Latency | 1-5 us | 50-500 us | 2-5 us (RDMA), <100 us (TCP) |
| regMr time | <1 ms | N/A | <1 ms |
| isend overhead | <1 us | ~10 us | <5 us |
| Max concurrent ops | 8-64 | 8 | 8-64 |

### NCCL Performance Testing

NCCL provides `nccl-tests` (https://github.com/NVIDIA/nccl-tests) for benchmarking:
- `all_reduce_perf` — AllReduce bandwidth and latency
- `all_gather_perf` — AllGather performance
- `broadcast_perf` — Broadcast performance
- `sendrecv_perf` — Raw point-to-point performance

**OuterLink validation:** Run nccl-tests with `NCCL_NET_PLUGIN=outerlink` to measure actual performance through the plugin path.

---

## 8. Mapping to OuterLink's Transport Architecture

### Phase 1: Host-Staged (TCP + io_uring)

```
NCCL isend(data_ptr) -> Plugin copies to pinned host buffer
  -> OuterLink TCP transport sends over network
  -> Remote receives into pinned host buffer
  -> Plugin copies to destination (NCCL calls irecv buffer)
```

- Report `ptrSupport = NCCL_PTR_HOST` (no GPU direct)
- NCCL handles GPU-to-host copy before calling isend
- `regMr` pins host memory for efficient DMA
- `speed` = actual TCP bandwidth, `latency` = measured RTT

### Phase 2: UCX / RDMA (ConnectX-5)

```
NCCL isend(data_ptr) -> Plugin registers with ConnectX
  -> RDMA send over InfiniBand/RoCE
  -> Remote NIC writes directly to registered buffer
  -> irecv completes when RDMA write finishes
```

- Report `ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA` (GPU direct with GDR)
- `regMr` registers with RDMA NIC (IB verbs `ibv_reg_mr`)
- `iflush` ensures GPU sees RDMA-written data
- `speed` = 100000 (100Gbps), `latency` = 2.0

### Phase 5: OpenDMA (PCIe BAR1)

```
NCCL isend(gpu_ptr) -> Plugin initiates DMA from GPU VRAM
  -> ConnectX DMA engine reads from BAR1
  -> Wire transfer
  -> Remote ConnectX DMA engine writes to remote GPU BAR1
  -> irecv completes when DMA finishes
```

- Report `ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA` (true zero-copy)
- `regMr` maps BAR1 region
- `speed` = 100000+, `latency` = 1.0-2.0
- This is the ultimate performance path

### Multi-Transport (ConnectX + USB4)

```
NCCL creates channels
  -> Channels 0-3 assigned to device 0 (ConnectX, 100Gbps)
  -> Channels 4-5 assigned to device 1 (USB4, 80Gbps)
  -> Both transports active simultaneously
  -> Combined bandwidth: 180 Gbps
```

- Report ConnectX and USB4 as separate NCCL devices
- Each with correct PCI path, speed, latency
- NCCL's topology engine assigns channels optimally

---

## 9. Edge Cases and Pitfalls

### Completion Ordering

NCCL expects completions in order per request. If `isend` returns request A, then request B, calling `test(A)` must complete before or simultaneously with `test(B)` for the same comm. OuterLink's transport must preserve this ordering.

### GPU Memory Visibility

After RDMA writes to GPU memory, the GPU may not immediately see the data (GPU caches, PCIe ordering). `iflush` exists for this purpose. The plugin must ensure:
- For RDMA writes to GPU: issue a fence/flush operation
- For host-staged: no flush needed (CPU copy ensures visibility)
- For OpenDMA via BAR1: PCIe write ordering should handle this, but verify

### Connection Establishment Timeout

NCCL retries `connect`/`accept` in a loop. If the remote side is slow to start, this can spin for a while. NCCL has a `NCCL_CONNECT_TIMEOUT` that defaults to some value. The plugin should handle this gracefully.

### Buffer Size Limits

NCCL may pass very large buffers to `regMr`. The plugin must handle registration of buffers up to several GB. With OuterLink's virtualized VRAM, this requires coordination with the VRAM manager.

---

## Related Documents

- [01-nccl-net-plugin-api.md](./01-nccl-net-plugin-api.md) — API surface
- [02-existing-nccl-plugins.md](./02-existing-nccl-plugins.md) — Plugin survey
- R14 Transport Compression — Compressing collective data in-flight
- R17 Topology-Aware Scheduling — Feeding OuterLink topology to NCCL
- R29 RDMA Multicast — Hardware-accelerated broadcast via CollNet

## Open Questions

1. **Channel-to-transport mapping:** When NCCL assigns channels to devices, does it respect the speed ratio? E.g., will it assign 5 channels to 100Gbps ConnectX and 4 channels to 80Gbps USB4? Or equal distribution?

2. **Ring ordering with mixed transports:** If a ring spans nodes connected by different transports (node A-B via ConnectX, node B-C via USB4), does the slower link become the bottleneck? NCCL should auto-detect this via speed properties, but needs verification.

3. **Topology XML generation:** Can OuterLink generate a `NCCL_TOPO_FILE` that describes its multi-transport network? What format does NCCL expect? This could be a powerful optimization.

4. **Tuner plugin integration:** NCCL supports tuner plugins for algorithm/protocol selection. Should OuterLink provide its own tuner plugin alongside the net plugin, to optimize algorithm selection for its specific transport characteristics?

5. **NCCL's LL128 with RDMA:** LL128 protocol embeds flags in data. Does this work over RDMA where the receiver can't inspect individual cache lines as they arrive? Need to check if NCCL handles this transparently.

6. **Compression integration with NCCL:** R14 plans transport compression. Where in the NCCL data flow should compression happen? Before isend (in plugin)? Or does NCCL have its own compression hooks? If in the plugin, the compressed size differs from the reported size — how does `test` report sizes?

## Sources

- Demystifying NCCL (arXiv): https://arxiv.org/html/2507.04786v1
- NVIDIA Blog: NCCL Cross-DC Deep Dive: https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness
- NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL Tests: https://github.com/NVIDIA/nccl-tests
- NCCL 2.28 Device API Blog: https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- NCCL 2.27 Blog: https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27
