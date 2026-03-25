# R29 Research: RDMA Multicast Fundamentals (IB/RoCE)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Document how RDMA multicast works on InfiniBand and RoCE v2 fabrics, ConnectX-5 hardware capabilities and limits, the UD QP requirement, IGMP management for RoCE, and performance comparison of multicast vs N unicasts. This is the foundation for OuterLink's one-to-many broadcast capability.

---

## 1. InfiniBand Multicast

### Multicast Groups and MGIDs

In InfiniBand, multicast groups are identified by a Multicast GID (MGID) — a 128-bit identifier following a specific format defined in the IB specification (Volume 1, Section 3.5.11).

**How it works:**
1. A node requests to join a multicast group by sending a SubnAdm request to the Subnet Manager (SM)
2. The SM assigns a Multicast LID (MLID) to the group and configures switch multicast forwarding tables
3. IB switches perform cut-through replication: one incoming packet is replicated to multiple output ports
4. Packet delivery to all group members happens at wire speed with sub-microsecond overhead

**Key properties:**
- The SM manages all group membership and switch forwarding tables
- Multicast routing is optimized by the SM (spanning tree within the fabric)
- Switches replicate at the hardware level — no store-and-forward penalty
- Maximum group size is limited by switch port count and fabric topology, not by protocol

### IB Multicast Performance

InfiniBand multicast achieves near-line-rate delivery to all group members simultaneously:
- One sender transmits one packet
- Each switch along the path replicates the packet to all relevant output ports
- All receivers get the packet within ~1us of each other (fabric propagation delay)
- Total bandwidth consumed at the sender: 1x (same as unicast)
- Total bandwidth consumed at each switch output port: 1x per receiver port

This is fundamentally different from N unicasts, which consume N x bandwidth at the sender.

---

## 2. RoCE v2 Multicast

### IP Multicast Mapping

RoCE v2 runs over UDP/IP, so multicast uses standard IP multicast:
- **IPv4 multicast range:** 224.0.0.0 - 239.255.255.255
- **MAC mapping:** IPv4 multicast maps to MAC 01:00:5e:xx:xx:xx (lower 23 bits of IP)
- **UDP port:** 4791 (standard RoCE v2 port)
- **No Subnet Manager:** RoCE uses standard Ethernet/IP multicast infrastructure

### IGMP for Group Management

Instead of the IB Subnet Manager, RoCE v2 uses IGMP (Internet Group Management Protocol):
- **IGMPv2/v3:** Hosts send IGMP Join/Leave messages to the local router/switch
- **IGMP snooping:** Layer-2 switches inspect IGMP messages to learn which ports need multicast traffic
- Switches that support IGMP snooping only forward multicast frames to ports that have joined the group
- Without IGMP snooping, multicast is flooded to all ports (degrades to broadcast)

### RoCE v2 Multicast API

The libibverbs API for multicast on RoCE:

```c
// Create UD QP (only UD supports multicast)
struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
// qp_init_attr.qp_type = IBV_QPT_UD

// Join multicast group
struct rdma_cm_id *cm_id;
rdma_create_id(ec, &cm_id, NULL, RDMA_PS_UDP);

struct sockaddr_in mcast_addr = {
    .sin_family = AF_INET,
    .sin_addr.s_addr = inet_addr("239.0.0.1"),
    .sin_port = htons(4791),
};
rdma_join_multicast(cm_id, (struct sockaddr *)&mcast_addr, NULL);

// Alternatively, using raw verbs:
union ibv_gid mgid;
// Fill mgid with multicast GID
ibv_attach_mcast(qp, &mgid, mlid);
```

### Critical Limitation: RoCE Multicast Reliability

**RoCE v2 multicast is inherently unreliable on Ethernet:**
- Ethernet switches lack the congestion control needed for lossless multicast
- Only UD (Unreliable Datagram) QPs support multicast — no RC (Reliable Connection) multicast exists
- Packet loss under congestion is expected and can be significant
- The IB specification's claim of lossless RoCE (Annex A17.9.1) applies to unicast only
- No switch-level flow control for multicast traffic (PFC/ECN don't help multicast)

This is the single biggest challenge for OuterLink's multicast adoption on RoCE v2 networks.

---

## 3. ConnectX-5 Multicast Capabilities

### Hardware Limits

Queried via `ibv_query_device()`:

| Parameter | ConnectX-5 Value | Description |
|-----------|-----------------|-------------|
| `max_mcast_grp` | **2,097,152** (2M) | Maximum multicast groups |
| `max_mcast_qp_attach` | **240** | Maximum QPs attached per multicast group |
| `max_total_mcast_qp_attach` | **503,316,480** (~503M) | Total QP-to-group attachments |

**Comparison across generations:**

| Parameter | ConnectX-3 (mlx4) | Connect-IB (mlx5) | ConnectX-5 (mlx5) |
|-----------|-------------------|-------------------|-------------------|
| `max_mcast_grp` | 8,192 | 2,097,152 | **2,097,152** |
| `max_mcast_qp_attach` | 248 | 48 | **240** |
| `max_total_mcast_qp_attach` | 2,031,616 | 100,663,296 | **503,316,480** |

### What These Numbers Mean for OuterLink

- **2M multicast groups:** Far more than needed. Even with per-tensor multicast groups, OuterLink would use hundreds at most.
- **240 QPs per group:** Each node typically uses 1 QP per multicast group. With 240 QPs per group, OuterLink supports clusters up to 240 nodes per multicast group — far beyond the 2-8 node target.
- **503M total attachments:** Not a constraint for any realistic deployment.

### ConnectX-5 RoCE v2 Configuration

```
# Enable RoCE v2 on ConnectX-5
# Firmware 16.20.1000+ required

# GID table: 128 entries per port
# Each GID entry can be RoCE v1 or v2
# Multicast GIDs are separate from unicast GIDs
```

---

## 4. UD QP: The Only Option for Multicast

### Why UD (Unreliable Datagram)?

RDMA multicast is defined only for UD QP transport:
- **RC (Reliable Connection):** Point-to-point, connection-oriented. Cannot multicast by definition.
- **UC (Unreliable Connection):** Point-to-point, no multicast support.
- **UD (Unreliable Datagram):** Connectionless, supports multicast. No reliability guarantees.
- **RD (Reliable Datagram):** Connectionless, could theoretically support reliable multicast, but no hardware implements it.

### UD QP Properties

| Property | Value |
|----------|-------|
| Max message size | MTU (typically 4096 bytes for IB, 1024-4096 for RoCE) |
| Reliability | None — packets can be lost, duplicated, or reordered |
| Connection setup | None — send to any destination by address |
| Multicast support | Yes — `ibv_attach_mcast()` to receive |
| RDMA operations | Only SEND/RECV — no RDMA READ/WRITE |

### Critical Constraint: No RDMA WRITE with Multicast

UD QPs only support SEND/RECV operations. This means:
- **No RDMA WRITE to multiple destinations** — the most natural multicast pattern is unavailable
- Sender must use `ibv_post_send` with `IBV_WR_SEND` opcode
- All receivers must pre-post `ibv_post_recv` buffers
- Receiver flow control is entirely application-managed

### MTU Limitation

UD messages are limited to a single MTU:
- **IB:** 256, 512, 1024, 2048, or 4096 bytes
- **RoCE v2:** Typically 1024 or 4096 bytes (Ethernet MTU - headers)
- For large transfers (model weights = GBs), data must be fragmented into MTU-sized chunks

A 1GB model weight broadcast at 4096-byte MTU requires 262,144 UD SEND operations. This is significant overhead.

---

## 5. IGMP for RoCE Multicast Group Management

### IGMP Snooping Requirements

For RoCE v2 multicast to work efficiently, the network switches must support IGMP snooping:

```
Host joins multicast:
    1. Host sends IGMP Membership Report (Join) for 239.0.0.1
    2. Switch records: port X is member of 239.0.0.1
    3. Future multicast to 239.0.0.1 goes only to port X

Host leaves multicast:
    1. Host sends IGMP Leave for 239.0.0.1
    2. Switch removes port X from 239.0.0.1 forwarding table
    3. (After IGMP query timeout if no other members on this port)
```

### Switch Requirements

| Feature | Required? | Notes |
|---------|-----------|-------|
| IGMP snooping | Required | Without it, multicast floods all ports |
| IGMP querier | Required | One device must be the IGMP querier |
| PFC (Priority Flow Control) | Recommended | Reduces packet loss for RoCE |
| ECN (Explicit Congestion Notification) | Recommended | Congestion feedback |
| Multicast flow control | Ideal but rare | Only a few switches support this |

### Consumer Switch Considerations

Pedro's cluster uses consumer/prosumer networking. Most managed switches support IGMP snooping, but:
- Unmanaged switches do NOT — multicast degrades to broadcast
- Consumer switches may have limited multicast forwarding table size (64-256 groups)
- IGMP querier must be explicitly configured (router or switch)

---

## 6. Performance: Multicast vs N Unicasts

### Bandwidth Analysis

For broadcasting B bytes to N nodes:

| Method | Sender Bandwidth | Total Network | Latency |
|--------|-----------------|---------------|---------|
| N unicasts (sequential) | B per send, N sends | N x B | N x t_transfer |
| N unicasts (parallel) | N x B simultaneously | N x B | 1 x t_transfer (if link not saturated) |
| Multicast | B (one send) | B at sender, replicated by switches | 1 x t_transfer |

**Example: 1GB model weights to 7 nodes (8-node cluster)**

| Method | Sender Bandwidth Used | Wall Time at 100Gbps |
|--------|----------------------|---------------------|
| 7 sequential unicasts | 1GB x 7 = 7GB | 7 x 80ms = 560ms |
| 7 parallel unicasts | 7GB simultaneous (saturates link) | ~560ms (link bottleneck) |
| Multicast | 1GB | ~80ms |
| Tree broadcast (NCCL) | 1GB at root, forwarded per level | ~160ms (2 levels for 8 nodes) |

Multicast is 7x faster than sequential unicast and ~2x faster than tree broadcast for 8 nodes.

### Latency Analysis

- **UD SEND latency:** ~2-5us per message (including UD header overhead)
- **Multicast replication:** Near-zero additional latency (switch hardware)
- **MTU fragmentation overhead:** For 1GB at 4KB MTU = 262,144 messages at ~3us each = ~786ms
- **With message pipelining:** Overlap is possible, but UD flow control adds complexity

### Practical Throughput

Measured multicast throughput on InfiniBand (from research papers):
- **Small messages (< 4KB):** Multicast achieves near-unicast latency with 1/N sender bandwidth
- **Large messages (> 1MB):** Multicast throughput limited by receiver processing (UD has no flow control)
- **Congestion under load:** RoCE multicast drops packets when receivers can't keep up

### Key Insight for OuterLink

Multicast's bandwidth advantage is clear (1/N sender bandwidth). But the practical challenges on RoCE are significant:
1. UD-only means no RDMA WRITE, no reliability, MTU-limited messages
2. Large transfers require application-level fragmentation and reassembly
3. Packet loss requires application-level retransmission
4. Real throughput depends heavily on switch IGMP snooping quality

---

## 7. Verdict for OuterLink

### Multicast: Promising but Requires Reliability Layer

The raw performance potential is excellent — 7x bandwidth savings for 8-node broadcast. But the practical implementation challenges are substantial:

**Plan A: RDMA Multicast + Application Reliability**
- Use UD multicast for bulk data distribution
- Build NACK-based reliability on top (see research doc 02)
- Fragment large transfers into MTU-sized messages
- Best for: model weight broadcast, read-only data distribution

**Plan B: Tree Broadcast (NCCL-Style)**
- Use existing RC RDMA WRITE infrastructure
- Each node forwards to children in a binary tree
- O(log N) latency, full bandwidth, inherently reliable
- Best for: when multicast reliability overhead exceeds tree overhead

**Plan C: Hybrid**
- Use multicast for initial bulk distribution (fast, most data arrives)
- Use unicast RC RDMA WRITE for repair of any lost packets
- Combines multicast speed with RC reliability

**Recommendation:** Plan C (Hybrid) is the right approach for OuterLink. Multicast handles the happy path at 1/N bandwidth, unicast handles the rare retransmission.

---

## Related Documents

- [02-reliable-multicast.md](./02-reliable-multicast.md) — Making multicast reliable
- [03-multicast-for-ml.md](./03-multicast-for-ml.md) — ML-specific multicast use cases
- [R20 NCCL Backend](../../../R20-nccl-backend/) — Broadcast collective implementation
- [R17 Topology-Aware Scheduling](../../../R17-topology-scheduling/) — Network topology for multicast routing

## Open Questions

- [ ] Do Pedro's switches support IGMP snooping? What model are they?
- [ ] Actual packet loss rate for RoCE v2 multicast on Pedro's network under load
- [ ] Can we use jumbo frames (9000 MTU) with RoCE multicast to reduce fragmentation?
- [ ] Does ConnectX-5 firmware support multicast loopback (sender also receives)?
