# R29 Research: Reliable Multicast Approaches

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

RDMA multicast over UD QPs is inherently unreliable — packets can be lost with no built-in recovery. This document evaluates approaches for building reliable delivery on top of unreliable multicast, from application-level NACK protocols to hardware-assisted solutions. The goal is to identify the right reliability strategy for OuterLink's model weight broadcast use case.

---

## 1. The Reliability Problem

### Why RDMA Multicast is Unreliable

1. **UD QPs have no sequence numbers or ACKs** — the NIC doesn't track what was delivered
2. **No flow control** — sender can overwhelm receivers (receiver buffer overflow)
3. **Ethernet switches drop under congestion** — unlike IB switches, no multicast-specific flow control
4. **No retransmission mechanism** — lost packets are silently gone
5. **RoCE v2 PFC/ECN only helps unicast** — congestion control doesn't apply to multicast traffic

### Loss Scenarios

| Scenario | Cause | Frequency |
|----------|-------|-----------|
| Receiver buffer overflow | Receiver can't post recv WRs fast enough | Common under burst |
| Switch queue overflow | Multicast replication exhausts switch buffers | Common on cheap switches |
| Link-level errors | CRC failures, cable issues | Rare on short copper |
| IGMP table miss | Switch hasn't learned membership yet | Rare, startup only |

### What "Reliable" Means for OuterLink

For model weight broadcast, we need **exactly-once, complete delivery**: every byte of the model weights must arrive at every node. Missing even one 4KB chunk means a corrupted model. But we don't need ordering guarantees (data is reassembled by offset) or real-time constraints (100ms extra latency is acceptable).

---

## 2. NACK-Based Reliability

### How It Works

Instead of receivers ACKing every packet (ACK implosion at scale), receivers only send NACKs for missing packets:

```
Sender:
    Assign sequence numbers to each multicast message
    Broadcast messages via UD multicast
    Buffer sent messages for potential retransmission

Receiver:
    Track received sequence numbers
    Detect gaps (missing sequence numbers)
    Send NACK (unicast) to sender for missing sequences

Sender (on receiving NACK):
    Retransmit requested messages via unicast to the requesting receiver
    (or re-multicast if multiple receivers lost the same packet)
```

### Advantages

- **Low overhead in happy path:** If no packets are lost, zero feedback traffic
- **Scalable:** Feedback only from receivers that lost data (not all receivers)
- **Simple sender logic:** Just buffer and retransmit on demand

### Challenges

- **NACK loss:** If the NACK itself is lost, the receiver never gets repair. Solution: periodic NACK retransmission.
- **Late NACK detection:** Receiver may not know a packet is missing until it sees a later sequence number. Solution: sender periodically sends "I've sent up to sequence N" heartbeats.
- **Buffer management:** Sender must buffer all sent data until all receivers confirm receipt. For 1GB broadcast, that's 1GB of sender buffer.
- **NACK implosion (mitigated):** If all receivers lose the same packet (switch drop), all send NACKs simultaneously. Solution: random NACK backoff + NACK suppression (receiver waits to see if someone else NACKed first).

### Implementation Complexity: Medium

A basic NACK protocol for OuterLink:
- Sequence numbering: 64-bit sequence per MTU-sized message
- NACK format: (transfer_id, missing_sequence_start, missing_sequence_count)
- Retransmission: unicast RC RDMA WRITE for repair (reliable, no further loss risk)
- Completion: sender sends "transfer complete" with total sequence count; receivers verify completeness

---

## 3. Forward Error Correction (FEC)

### How It Works

Sender transmits redundant (parity) packets along with data packets. Receivers can reconstruct lost packets from the parity without requesting retransmission.

```
Sender:
    Split data into groups of K packets
    Compute M parity packets per group (Reed-Solomon or XOR)
    Multicast all K+M packets

Receiver:
    Receive any K out of K+M packets per group
    Reconstruct the full group (even if M packets were lost)
```

### FEC Parameters for OuterLink

| Configuration | Data Packets (K) | Parity Packets (M) | Overhead | Tolerates Loss |
|---------------|------------------|---------------------|----------|----------------|
| FEC(8,2) | 8 | 2 | 25% | Up to 20% loss |
| FEC(16,4) | 16 | 4 | 25% | Up to 20% loss |
| FEC(32,4) | 32 | 4 | 12.5% | Up to 11% loss |
| FEC(64,8) | 64 | 8 | 12.5% | Up to 11% loss |

### Advantages

- **Zero feedback in happy path AND moderate loss:** No NACKs needed if loss < M/K
- **Constant overhead:** Known bandwidth cost regardless of loss pattern
- **Latency hiding:** No retransmission round-trip for recoverable losses

### Challenges

- **Encoding/decoding cost:** Reed-Solomon over GF(2^8) requires matrix multiplication. Intel ISA-L does this at 10+ GB/s per core (fast enough for 100Gbps wire).
- **Fixed overhead even when no loss:** 12.5-25% bandwidth wasted if network is perfect
- **Insufficient for burst loss:** If more than M packets lost in a group, FEC fails and we need fallback to NACK/retransmit
- **Group boundary problem:** All packets in a group must arrive before decoding. Increases buffer requirements and latency.

### FEC + NACK Hybrid

The PGM protocol and academic research converge on the same answer: use FEC to handle common small losses, NACK for rare large losses.

```
Normal operation:
    FEC handles 1-5% loss silently

Burst loss (> FEC capacity):
    Receiver sends NACK for unrecoverable groups
    Sender retransmits via unicast
```

This is the approach used by NORM (NACK-Oriented Reliable Multicast, RFC 5740) and is proven at scale.

---

## 4. PGM (Pragmatic General Multicast)

### Overview

PGM (RFC 3208) is a NACK-based reliable multicast protocol designed for IP multicast over best-effort networks. It's the most mature reliable multicast standard.

### Key Mechanisms

1. **Sequenced data packets (ODATA):** Sender multicasts numbered data packets
2. **Selective NACKs:** Receivers unicast NACKs for missing sequences
3. **NAK forwarding:** Network elements (routers) forward NAKs hop-by-hop toward the source
4. **NAK suppression:** If a router already has a pending NAK for a sequence, it suppresses duplicates
5. **NAK elimination:** Routers can serve repairs from local caches (DLRs — Designated Local Repairers)
6. **FEC integration:** Optional FEC groups with parity packets

### Scalability

PGM scales well due to the hierarchical NAK handling:
- Without router support: NACK suppression + FEC handle moderate scale
- With PGM-aware routers: NAK suppression and local repair at each hop
- Tested at thousands of receivers in production deployments

### PGM for OuterLink?

PGM operates at the IP/UDP level, not the RDMA level. Adapting PGM concepts to RDMA:
- **ODATA → UD SEND multicast** with sequence numbers
- **NAK → Unicast RDMA SEND** to sender
- **Repair → Unicast RC RDMA WRITE** (reliable, direct to receiver's buffer)
- **NAK suppression → Application-level** (random backoff per node)
- **DLR → Any node that has the data** can repair via RDMA WRITE

We don't need full PGM protocol compliance. We adopt PGM's proven mechanisms into our RDMA context.

---

## 5. NVIDIA SHARP: Hardware Multicast Reduction

### What SHARP Does

NVIDIA's Scalable Hierarchical Aggregation and Reduction Protocol (SHARP) offloads collective operations to InfiniBand switch ASICs:
- **AllReduce in-network:** Switches sum/reduce data as it passes through, halving round-trips
- **Broadcast via switch tree:** Data sent once, replicated by switches with guaranteed delivery
- **2x bandwidth improvement** for AllReduce operations
- **7x latency reduction** for MPI AllReduce at scale

### SHARP Generations

| Generation | Switch | Key Feature |
|-----------|--------|-------------|
| SHARPv1 | EDR (100Gbps) | In-network reduction for MPI |
| SHARPv2 | HDR (200Gbps) | Extended for AI/NCCL workloads |
| SHARPv3 | NDR (400Gbps) | Multi-tenant, cloud-native |

### SHARP for OuterLink?

**Not directly applicable:**
- SHARP requires NVIDIA Quantum InfiniBand switches (not Ethernet/RoCE)
- Pedro's cluster uses Ethernet with ConnectX-5 in RoCE mode
- SHARP is proprietary — requires NVIDIA switch silicon

**But the concept is relevant:** SHARP proves that in-network multicast with reliability is the performance optimum. OuterLink aims for the same result via software on commodity hardware.

### NVLink SHARP (NVSwitch)

Third-generation NVSwitch includes SHARP for NVLink-connected GPUs within a single server. This is irrelevant for cross-node OuterLink (we use network, not NVLink), but shows NVIDIA's direction.

---

## 6. AWS SRD (Scalable Reliable Datagram)

### What SRD Is

Amazon's custom network protocol for EFA (Elastic Fabric Adapter):
- Connectionless, datagram-based (like UD)
- Built-in reliability via packet-level retransmission
- Multi-path load balancing across network paths
- Low latency (~5us) with reliability
- Used by NCCL for AllReduce on AWS P4d/P5 instances

### How SRD Achieves Reliability

```
SRD header:
    Contains sequence number, transfer ID, and path ID

Sender:
    Transmits datagrams with sequence numbers
    Tracks per-destination delivery status
    Retransmits on timeout or NACK

Receiver:
    Detects out-of-order and missing packets
    Sends selective ACKs back to sender

Network (EFA/Nitro):
    Multi-path spraying across network paths
    Per-packet load balancing (no head-of-line blocking)
```

### SRD for OuterLink?

**Not directly usable** (SRD is proprietary to AWS's Nitro network cards). But SRD's design validates the approach:
- Connectionless datagrams with application-level reliability work at datacenter scale
- Packet-level sequence tracking is sufficient (no need for complex connection state)
- Multi-path spraying + reliability can coexist

OuterLink can implement similar concepts using UD multicast + NACK reliability.

---

## 7. Comparison of Approaches

### For OuterLink's Use Case (Model Weight Broadcast, 2-8 Nodes, RoCE v2)

| Approach | Bandwidth Overhead | Latency Overhead | Complexity | Reliability |
|----------|-------------------|-----------------|------------|-------------|
| **Pure NACK** | 0% (happy path) | 1 RTT per lost packet | Medium | 100% (with retransmit) |
| **Pure FEC** | 12-25% always | 0 (within FEC capacity) | Medium | < 100% (burst loss) |
| **NACK + FEC hybrid** | 12-25% + rare retransmit | Near-zero for most loss | High | 100% |
| **Tree broadcast (no multicast)** | 0% | O(log N) stages | Low | 100% (RC) |
| **SHARP** | 0% | Minimal | N/A | 100% (hardware) |
| **SRD-style** | ~5% (ACK traffic) | ~1 RTT for loss | High | 100% |

### Recommended: Tiered Approach

**Tier 1 (Implement First): Tree Broadcast**
- Use existing RC RDMA WRITE infrastructure
- Binary tree: root sends to 2 children, each forwards to 2 more
- 8 nodes: 3 stages x transfer_time = 3x latency vs multicast, but 100% reliable
- Zero new infrastructure needed

**Tier 2 (Implement Second): Multicast + NACK**
- UD multicast for bulk data
- NACK-based repair via unicast RC RDMA WRITE
- Sender buffers data until all receivers confirm
- Completes transfer in 1x transfer_time + rare retransmission overhead

**Tier 3 (If Needed): Multicast + FEC + NACK**
- Add FEC to reduce NACK frequency
- Only worthwhile if NACK rate is high (> 5% packet loss)
- Adds encoding/decoding overhead

---

## 8. Verdict for OuterLink

### Primary Strategy: Multicast + NACK with Tree Fallback

```
For each broadcast operation:
    1. Try multicast (UD SEND to multicast group)
    2. Receivers track sequence numbers
    3. After sender signals "complete", receivers NACK any missing sequences
    4. Sender repairs via unicast RC RDMA WRITE (reliable)
    5. If multicast fails completely (no IGMP, switch issues):
       Fall back to tree broadcast via RC RDMA WRITE
```

### Why This Works for OuterLink

- **2-8 nodes:** NACK implosion is not a concern at this scale
- **Model weights are idempotent:** Re-sending a chunk is harmless
- **Latency tolerance:** 100ms extra for repair is acceptable for model loading
- **Existing infrastructure reuse:** RC RDMA WRITE for repair leverages existing transport

### What We Don't Need

- PGM router support (we have 2-8 nodes, no hierarchical routing needed)
- SHARP (requires IB switches we don't have)
- SRD (requires AWS EFA hardware)
- Complex FEC (NACK is sufficient at our scale)

---

## Related Documents

- [01-rdma-multicast-fundamentals.md](./01-rdma-multicast-fundamentals.md) — Multicast basics
- [03-multicast-for-ml.md](./03-multicast-for-ml.md) — ML use cases
- [R15 Fault Tolerance](../../R15-fault-tolerance/) — Erasure coding concepts (related to FEC)
- [R20 NCCL Backend](../../../R20-nccl-backend/) — Broadcast collective implementation

## Open Questions

- [ ] What's the actual packet loss rate on Pedro's network for UD multicast traffic?
- [ ] How much sender buffer is acceptable? (1GB for model weights is significant)
- [ ] Should NACK repair go to the original sender or to any node that has the data? (peer repair)
- [ ] Can we overlap multicast send with NACK processing? (pipelined reliability)
