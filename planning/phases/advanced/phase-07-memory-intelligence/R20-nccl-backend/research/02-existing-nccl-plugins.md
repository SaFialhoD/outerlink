# R20 Research: Existing NCCL Plugins

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Survey existing NCCL net plugins to understand implementation patterns, architectural decisions, and lessons learned. These plugins serve as reference implementations for OuterLink's NCCL backend.

---

## 1. Plugin Landscape

| Plugin | Maintainer | Transport | Production Use | Open Source |
|--------|-----------|-----------|---------------|-------------|
| nccl-rdma-sharp-plugins | NVIDIA/Mellanox | IB verbs, UCX, SHARP | HPC-X, DGX | Yes (GitHub) |
| aws-ofi-nccl | AWS | libfabric (EFA) | AWS EC2 GPU instances | Yes (GitHub) |
| Spectrum-X NCCL Plugin | NVIDIA | Spectrum-X fabric | NVIDIA networking | Proprietary |
| nccl-mesh-plugin | autoscriptlabs | Direct RDMA mesh | Consumer GPU clusters | Yes (GitHub) |
| (NCCL built-in) | NVIDIA | TCP sockets, IB verbs | Default fallback | In NCCL source |

---

## 2. nccl-rdma-sharp-plugins (NVIDIA/Mellanox)

**Repository:** https://github.com/Mellanox/nccl-rdma-sharp-plugins

### Architecture

This is NVIDIA's flagship plugin, shipped with HPC-X. It implements two separate NCCL APIs:

1. **ncclNet (Point-to-Point):** Two transport backends:
   - **IB verbs (default):** Direct InfiniBand verbs for RDMA send/recv
   - **UCX:** Uses UCX framework for transport abstraction

2. **ncclCollNet (Collective):** SHARP in-network reduction
   - Offloads AllReduce partial sums to InfiniBand switches
   - Reduces inter-node traffic by computing reductions in the network fabric

### Key Design Decisions

- **Separate transports behind one plugin:** The plugin selects between IB verbs and UCX based on `NCCL_PLUGIN_P2P` environment variable
- **GPUDirect RDMA:** Supports direct GPU-to-NIC transfers via `nv_peer_mem` kernel module. When GPU pointers are passed (with `NCCL_PTR_CUDA` support), the plugin registers GPU memory directly with the NIC
- **Memory registration:** Uses IB verbs `ibv_reg_mr` for host memory, `ibv_reg_mr` with `nv_peer_mem` for GPU memory. Maintains registration caches for performance
- **Connection management:** Uses Queue Pairs (QPs) for IB connections. Supports multiple QPs per connection in round-robin for load distribution
- **Build system:** GNU autotools (autogen.sh, configure, make)

### UCX Transport Modes

The UCX backend within this plugin supports multiple transport modes:

| Mode | Layer | Mechanism | Best For |
|------|-------|-----------|----------|
| UCP rendezvous | Protocol | High-level message exchange | General use |
| UCT read-based | Transport | RDMA read operations | Large messages |
| UCT write-based | Transport | RDMA write with PUT_ZCOPY | Adaptive routing, LL/LL128 |
| UCP put-based | Protocol | UCX put operations | Alternative write path |

Selected via `NCCL_UCX_TLS` environment variable.

### Lessons for OuterLink

- **Multi-transport in one plugin works.** OuterLink can similarly expose TCP, RDMA, and OpenDMA through a single plugin, selecting transport based on environment or auto-detection.
- **Registration caching is critical.** The plugin maintains caches to avoid re-registering the same memory regions.
- **CollNet is optional but powerful.** SHARP reduces inter-node traffic significantly for AllReduce-heavy workloads (most ML training).
- **The IB verbs path is heavily optimized.** Direct verbs calls outperform UCX abstraction in many cases. OuterLink should consider a direct path for its primary transport.

---

## 3. aws-ofi-nccl (AWS)

**Repository:** https://github.com/aws/aws-ofi-nccl

### Architecture

Maps NCCL's connection-oriented transport API to libfabric's connectionless reliable interface. Designed primarily for AWS Elastic Fabric Adapter (EFA) but works with any libfabric provider.

### Key Design Decisions

- **Connectionless mapping:** NCCL assumes connection-oriented semantics (connect/accept), but libfabric/EFA uses connectionless reliable datagrams (FI_EP_RDM). The plugin bridges this gap by creating virtual connections over connectionless transport.
- **Tagged messaging:** Uses libfabric's `FI_TAGGED` interface for message matching, mapping NCCL's tags to libfabric tags.
- **Provider requirements:** Requires `FI_EP_RDM` (reliable datagram) endpoints with `FI_TAGGED` and `FI_MSG` capabilities. For GPUDirect: also requires `FI_HMEM`.
- **No NCCL build dependency:** The plugin does not require NCCL headers at build time. It copies the versioned struct definitions it supports into its own source tree.
- **Broad NCCL compatibility:** Supports NCCL v2.17.1 (v6 API) through latest. Same binary works across multiple NCCL versions.

### Performance Tuning

- **Channel count:** Must match the NIC count. On p4d/p4de (4 EFA NICs), use 8 channels minimum. Too many channels starve the NIC.
- **GPUDirect RDMA:** Supported when libfabric provider offers `FI_HMEM`. Falls back to host-staged if not available.
- **Multi-NIC awareness:** Plugin detects multiple NICs and reports them as separate devices to NCCL, allowing NCCL's topology engine to use them optimally.

### Lessons for OuterLink

- **Connectionless-to-connection mapping is viable.** If OuterLink's transport has different semantics than NCCL's model, an adaptation layer works fine.
- **No NCCL build dependency.** Copy the header definitions you need. This simplifies distribution.
- **Version compatibility through shim layers.** Implement the latest version, provide thin wrappers for older versions. This pattern is proven at scale (millions of AWS GPU instances).
- **Multi-NIC reporting matters.** OuterLink should report ConnectX and USB4 as separate devices, letting NCCL's topology engine decide how to use them.

---

## 4. nccl-mesh-plugin (autoscriptlabs)

**Repository:** https://github.com/autoscriptlabs/nccl-mesh-plugin

### Architecture

Enables NCCL over direct-connect RDMA mesh topologies (no switch required). Designed for consumer/prosumer GPU clusters where each node pair has a dedicated RDMA link on a different subnet.

### Key Design Decisions

- **Subnet-per-link model:** Standard NCCL plugins assume switched fabric (all nodes on same subnet) or TCP. This plugin handles the case where each direct link has its own subnet.
- **Cost reduction:** Eliminates $15,000-50,000 managed InfiniBand switches by using direct RDMA cables.
- **Tested at scale:** 3x DGX Spark workstations with 100Gbps direct RDMA links, running DeepSpeed ZeRO-3.

### Lessons for OuterLink

- **Custom topology plugins are practical.** Even small teams build working NCCL plugins for non-standard topologies.
- **OuterLink's multi-transport model is more complex but follows the same principle.** Like mesh-plugin handles per-link subnets, OuterLink handles per-link transport types (ConnectX RDMA, USB4, TCP).
- **Consumer hardware RDMA is a real use case.** This validates OuterLink's target market — making RDMA/high-speed transport accessible beyond data centers.

---

## 5. Spectrum-X NCCL Plugin (NVIDIA)

### Architecture

NVIDIA's plugin for Spectrum-X Ethernet fabric. Not open source but documented.

### Key Features

- **Network failure recovery:** Detects and recovers from link failures during training
- **Dynamic load balancing:** Distributes traffic across available paths
- **Topology specification:** Mechanism to explicitly describe network topology to NCCL
- **Transport parameter tuning:** Configurable per-path transport parameters

### Lessons for OuterLink

- **Failure recovery is table stakes.** OuterLink's NCCL plugin must handle transport failures gracefully (connection loss, device removal).
- **Topology specification API exists.** OuterLink can feed its multi-transport topology to NCCL through the properties API.
- **Per-path tuning matters.** Different OuterLink transports have wildly different characteristics (TCP: 10Gbps/100us vs OpenDMA: 100Gbps/2us). The plugin must report accurate per-device properties.

---

## 6. NCCL Built-in Transports (Reference)

NCCL ships with built-in transport implementations that serve as the baseline:

### Socket Transport (`net_socket.cc`)
- Pure TCP sockets
- Always available as fallback
- No RDMA, no zero-copy
- Simple but slow for large-scale training

### IB Transport (`net_ib.cc`)
- InfiniBand verbs
- GPUDirect RDMA support
- Queue Pair management
- This is effectively what nccl-rdma-sharp-plugins replaces/extends

### Key Architecture Patterns from Built-in Transports

- **Proxy thread model:** NCCL uses proxy threads to handle network I/O. The plugin's async operations (isend/irecv) queue work that proxy threads process.
- **Request pool:** Pre-allocated request objects to avoid allocation in the data path.
- **Inline data:** Small messages may be inlined in the control path for lower latency.

---

## 7. Comparative Analysis for OuterLink

### What OuterLink's Plugin Must Do Differently

| Aspect | Existing Plugins | OuterLink Plugin |
|--------|-----------------|------------------|
| Transport count | 1-2 per plugin | 3+ (TCP, RDMA, USB4, OpenDMA) |
| Device model | Physical NICs | Virtual + physical devices |
| Memory model | Real GPU/host memory | Virtualized VRAM pool |
| Topology | Static (NIC PCI paths) | Dynamic multi-transport |
| Registration | Standard regMr | Must coordinate with VRAM manager |
| Failure handling | Connection-level | Transport-level with failover |

### Recommended Architecture

Based on the survey, OuterLink's NCCL plugin should:

1. **Report each transport as a separate NCCL device** — ConnectX NIC = device 0 (100Gbps/2us), USB4 port = device 1 (80Gbps/5us), TCP = device 2 (10Gbps/100us). NCCL's topology engine handles the rest.

2. **Implement v8 as the primary API** with shim layers for v9-v11, following the aws-ofi-nccl pattern.

3. **Use OuterLink's existing transport layer** as the backend. The plugin is a thin adapter between NCCL's API and OuterLink's transport, NOT a new transport implementation.

4. **Start with NCCL_PTR_HOST** (host-staged). Add `NCCL_PTR_CUDA` when OpenDMA is ready. This matches Phase 1 vs Phase 5 progression.

5. **Implement registration caching** following nccl-rdma-sharp-plugins pattern.

6. **Skip CollNet initially.** Point-to-point ncclNet is sufficient. Add CollNet for SHARP-like acceleration in a later phase.

---

## Related Documents

- [01-nccl-net-plugin-api.md](./01-nccl-net-plugin-api.md) — API surface details
- [03-nccl-topology-and-collectives.md](./03-nccl-topology-and-collectives.md) — NCCL internals
- R14 Transport Compression — Compressed collectives
- R29 RDMA Multicast — Hardware broadcast for CollNet

## Open Questions

1. **Language for shim layer:** The v8->v9->v10->v11 shim layer involves trivial type conversions (int to size_t, adding/removing params). Is this best done in C (minimal, follows NCCL example) or in Rust with cbindgen-generated headers?

2. **aws-ofi-nccl's no-NCCL-dependency approach:** Should OuterLink copy NCCL headers into its tree (like aws-ofi-nccl) or depend on NCCL at build time? Copying is simpler for distribution.

3. **Multi-transport device reporting:** If we report ConnectX as device 0 and USB4 as device 1, does NCCL try to use both? How does NCCL handle devices with vastly different bandwidths? Need to test.

4. **Plugin loading priority:** If the system also has nccl-rdma-sharp-plugins installed, does `NCCL_NET_PLUGIN=outerlink` guarantee our plugin is loaded? Need to verify NCCL's plugin precedence logic.

## Sources

- nccl-rdma-sharp-plugins: https://github.com/Mellanox/nccl-rdma-sharp-plugins
- NVIDIA NCCL-RDMA-SHARP docs: https://docs.nvidia.com/networking/display/hpcvx225/NCCL-RDMA-SHARP-Plugins
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- AWS EFA + NCCL guide: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html
- nccl-mesh-plugin: https://github.com/autoscriptlabs/nccl-mesh-plugin
- Spectrum-X NCCL Plugin: https://docs.nvidia.com/networking/display/hpcvx225/spectrum-x-nccl-plugin
- UCX NCCL plugin presentation (UCF 2024): https://ucfconsortium.org/wp-content/uploads/2024/12/2024_2_UCX-network-plugin-for-NCCL.pdf
- NCCL InfiniBand transport (DeepWiki): https://deepwiki.com/NVIDIA/nccl/5.3-infiniband-and-rdma-transport
