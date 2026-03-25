# R29: RDMA Multicast — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)

## Purpose

Define the scope, dependencies, decisions, and implementation roadmap for adding RDMA multicast to OuterLink. This enables one-to-many broadcast at hardware line rate — one send reaches N receivers. Primary use case: distributing model weights to all GPUs simultaneously instead of N sequential or tree-based transfers.

---

## Scope Definition

### In Scope

1. **Multicast group management** — Create, join, leave multicast groups via IGMP (RoCE v2) or SA (IB)
2. **UD multicast send/receive** — Send data to multicast group, receive on all members
3. **Application-level reliability** — NACK-based protocol for lost packet recovery
4. **Message fragmentation/reassembly** — Break large transfers (GBs) into MTU-sized UD messages
5. **Multicast-accelerated Broadcast** — NCCL Broadcast collective via multicast
6. **Tree broadcast fallback** — When multicast is unavailable or unreliable, use tree-based RC RDMA WRITE
7. **Sender-side retransmission buffer** — Buffer sent data for NACK-based repair
8. **Unicast repair path** — RC RDMA WRITE for retransmitting lost packets (reliable)

### Out of Scope (Handled Elsewhere)

- NCCL AllReduce optimization — R20 handles collectives; we provide multicast as a transport option
- Topology discovery — R17 provides network topology for multicast routing decisions
- OpenDMA integration — Phase 2 optimization, not initial scope
- FEC encoding — Only add if NACK-based repair proves insufficient
- In-network reduction (SHARP) — Requires IB switches we don't have
- Multi-path multicast — Single multicast group per operation initially

### Boundary: Multicast vs Tree

Multicast and tree broadcast serve the same purpose (one-to-many). OuterLink provides both and selects automatically:
- **Multicast:** Used when IGMP snooping is confirmed working and loss rate < 5%
- **Tree:** Used as fallback when multicast is unavailable or loss rate is unacceptable

---

## Dependencies

### Upstream (Must Exist Before R29)

| Dependency | Component | Why Needed |
|-----------|-----------|------------|
| **Transport Layer** | RC RDMA WRITE | Unicast repair path for lost packets |
| **Transport Layer** | UD QP support | Multicast requires UD QPs |
| **R10** | Page Table | Know which pages to broadcast |
| **P5/OpenDMA** | BAR1 Access | Phase 2: direct multicast to GPU VRAM |
| **Network** | IGMP Snooping | Switch must support IGMP for efficient RoCE multicast |

### Downstream (Uses R29)

| Consumer | How It Uses R29 |
|----------|----------------|
| **R20** | NCCL Broadcast collective uses multicast for one-to-all |
| **R12** | Memory deduplication broadcasts shared read-only pages |
| **Cluster Manager** | Model weight distribution at startup |

### Parallel (No Dependency)

- R28 Scatter-Gather DMA — Independent, but receivers can use scatter to place multicast data
- R15 Fault Tolerance — Independent, but multicast could distribute parity fragments
- R26 PTP Clock Sync — No interaction

---

## Key Decisions Required

### Decision 1: Multicast Transport or Application-Level Broadcast

**Context:** We could implement multicast at the RDMA UD level (true hardware multicast) or use application-level broadcast (sender sends N unicast RDMA WRITEs via RC QPs). The latter is simpler but slower.

**Options:**
- **A:** True RDMA UD multicast + reliability layer (Plan A — best performance)
- **B:** Application-level broadcast via tree of RC RDMA WRITEs (Plan B — simpler, reliable)
- **C:** Both, with automatic selection based on network capability

**Leaning:** Option C — implement both, auto-select. Tree is the safe fallback.

### Decision 2: Reliability Mechanism

**Context:** UD multicast loses packets. How to ensure complete delivery?

**Options:**
- **A:** NACK-only — receivers report missing sequences, sender retransmits via unicast
- **B:** FEC + NACK — FEC handles small losses, NACK for burst losses
- **C:** ACK-based — receivers ACK each message (ACK implosion risk, but only 2-8 nodes)

**Leaning:** Option A for Phase 1 (NACK-only). At 2-8 nodes, NACK implosion isn't a concern. FEC adds complexity for marginal benefit at this scale. Evaluate loss rates before adding FEC.

### Decision 3: Multicast Group Lifecycle

**Context:** When are multicast groups created and destroyed?

**Options:**
- **A:** Persistent groups — created when nodes join cluster, destroyed when they leave
- **B:** Ephemeral groups — created per-operation, destroyed after transfer completes
- **C:** Hybrid — persistent "all-nodes" group + ephemeral per-subset groups

**Leaning:** Option C — one persistent group for "broadcast to all" (model weights), ephemeral groups for subset operations.

### Decision 4: Message Fragmentation Strategy

**Context:** UD messages are limited to MTU (4096 bytes typically). Large transfers (GBs) need fragmentation.

**Options:**
- **A:** Fixed-size fragments (MTU-sized) with sequence numbers
- **B:** Variable-size fragments matching page boundaries (64KB pages split into MTU chunks)
- **C:** Coalesced fragments (pack multiple small logical messages into one UD SEND)

**Leaning:** Option A — simple fixed-size fragmentation. Each UD SEND carries one MTU-sized payload with a header containing (transfer_id, sequence_number, total_sequences).

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pedro's switches don't support IGMP snooping | MEDIUM | HIGH | Tree broadcast fallback; verify switch capabilities before implementing |
| Packet loss rate too high on RoCE multicast | MEDIUM | HIGH | If > 5% loss: disable multicast, use tree; measure before committing |
| UD MTU fragmentation overhead too high | LOW | MEDIUM | Test with jumbo frames (9000 MTU); worst case: 4KB MTU still works |
| NACK storm from correlated loss | LOW | MEDIUM | Random backoff on NACKs; at 2-8 nodes, not a real concern |
| GRH overhead for BAR1 targeting | LOW | LOW | Phase 2 concern; host staging works for Phase 1 |
| IGMP join latency at startup | LOW | LOW | Pre-join groups before broadcast starts |

---

## Implementation Phases

### Phase 1: Tree Broadcast (Foundation)

**Deliverables:**
- Binary tree broadcast using RC RDMA WRITE
- Root node sends to 2 children, each forwards to 2 more children
- Handles arbitrary data sizes (chunked at sender, reassembled at receiver)
- Tree topology derived from R17 node ordering

**Acceptance criteria:**
- Broadcast 1GB to 8 nodes via tree in < 35 seconds (3 levels x 11.2s per level)
- 100% reliability (RC guarantees delivery)
- Works on any network (no IGMP requirement)

### Phase 2: UD Multicast Infrastructure

**Deliverables:**
- Multicast group creation/join/leave via IGMP (RoCE v2)
- UD QP creation and configuration for multicast
- Basic UD SEND/RECV for multicast messages
- IGMP snooping verification tool (detect if switch supports it)

**Acceptance criteria:**
- Can send/receive UD multicast messages between all cluster nodes
- IGMP snooping confirmed working (multicast only goes to members, not flooded)
- Measure baseline packet loss rate

### Phase 3: Reliable Multicast Protocol

**Deliverables:**
- Message fragmentation (MTU-sized chunks with sequence numbers)
- Reassembly buffer on receiver (tracks received sequences)
- NACK generation (detect missing sequences, send NACK to sender)
- NACK processing (sender retransmits via unicast RC RDMA WRITE)
- Transfer completion protocol (sender signals "done", receivers verify completeness)
- Sender retransmission buffer management

**Acceptance criteria:**
- Reliable transfer of 1GB to 8 nodes via multicast + NACK in < 15 seconds
- Zero data corruption (byte-level verification)
- Handles 0-5% packet loss without degradation
- Falls back to tree broadcast if loss > 10%

### Phase 4: NCCL Integration

**Deliverables:**
- NCCL Broadcast collective using multicast (R20 backend)
- Automatic algorithm selection (multicast vs tree based on message size and loss rate)
- CollNet interface implementation for multicast AllGather (optional)

**Acceptance criteria:**
- NCCL Broadcast uses multicast for messages > 4MB on clusters > 2 nodes
- Performance matches or exceeds NCCL tree broadcast
- Transparent fallback to tree if multicast is unavailable

### Phase 5: OpenDMA Integration (Future)

**Deliverables:**
- UD RECV buffers in BAR1-registered GPU VRAM (skip host staging)
- GRH handling for BAR1 targets
- Unicast repair via OpenDMA RDMA WRITE

**Acceptance criteria:**
- Multicast data arrives directly in GPU VRAM (no host copy on happy path)
- Latency reduction vs host-staged multicast

---

## Estimated Effort

| Phase | Complexity | Estimated Time | Risk |
|-------|-----------|---------------|------|
| Phase 1: Tree Broadcast | Low | 1-2 weeks | Low |
| Phase 2: UD Multicast Infra | Medium | 2-3 weeks | Medium |
| Phase 3: Reliable Multicast | High | 3-4 weeks | Medium |
| Phase 4: NCCL Integration | Medium | 2-3 weeks | Low |
| Phase 5: OpenDMA Integration | High | 2-3 weeks | High |
| **Total** | | **10-15 weeks** | |

---

## Success Metrics

| Metric | Baseline (Tree Only) | Target (With Multicast) |
|--------|---------------------|------------------------|
| 1GB broadcast to 8 nodes | ~33.6s (tree, 3 levels) | ~12s (multicast) |
| 140GB model load to 8 nodes | ~394s (tree) | ~140s (multicast) |
| Sender bandwidth for broadcast | 100% per level | 100% (one send) |
| NCCL Broadcast latency (1MB, 8 nodes) | ~2.4ms (tree) | ~0.8ms (multicast) |
| Network overhead for 8-node broadcast | 3x data (tree levels) | 1x data + ~1% NACK |

---

## Network Requirements Checklist

Before implementing multicast, verify Pedro's network:

- [ ] Switch model and firmware version
- [ ] IGMP snooping support and configuration
- [ ] IGMP querier configuration
- [ ] Multicast forwarding table size
- [ ] RoCE v2 configuration on all nodes
- [ ] Jumbo frame support (9000 MTU preferred)
- [ ] Baseline packet loss rate for UD traffic

---

## Related Documents

- [research/01-rdma-multicast-fundamentals.md](./research/01-rdma-multicast-fundamentals.md) — IB/RoCE multicast mechanics
- [research/02-reliable-multicast.md](./research/02-reliable-multicast.md) — Reliability approaches
- [research/03-multicast-for-ml.md](./research/03-multicast-for-ml.md) — ML use cases and performance analysis
- [R20 NCCL Backend](../../R20-nccl-backend/) — Broadcast collective implementation
- [R17 Topology-Aware Scheduling](../../R17-topology-scheduling/) — Network topology for tree/multicast routing

## Open Questions

- [ ] Should multicast be a transport-layer feature (available to all transfers) or specifically a collective operation (NCCL only)?
- [ ] How to handle heterogeneous clusters where some nodes have slower NICs? (multicast sends at slowest receiver rate?)
- [ ] Can we use multicast for the ReduceScatter phase if we decompose it creatively?
- [ ] What's the optimal sender retransmission buffer size? (Trade off memory vs retransmission capability)
