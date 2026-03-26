# R29: RDMA Multicast — Pre-Plan v2

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of RDMA multicast planning. v1 established scope, reliability strategies, and phases. v2 adds exact Rust structs, concrete multicast group management algorithms, integration with R10/R12/R19/R20, performance crossover models, and resolved open questions from v1.

---

## 1. Rust Data Structures

### Core Types

```rust
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// A multicast group in OuterLink.
/// Maps to one IP multicast address (RoCE v2) and one UD QP per member.
pub struct MulticastGroup {
    /// Unique group identifier within OuterLink
    pub group_id: MulticastGroupId,
    /// IP multicast address (e.g., 239.0.0.1)
    pub mcast_addr: Ipv4Addr,
    /// Multicast GID for ibv_attach_mcast
    pub mgid: Gid,
    /// Group lifecycle type
    pub lifecycle: GroupLifecycle,
    /// Current members (node_id -> member state)
    pub members: HashMap<NodeId, MemberState>,
    /// The root/sender node (for broadcast groups)
    pub root: Option<NodeId>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Statistics
    pub stats: MulticastGroupStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MulticastGroupId(pub u32);

#[derive(Clone, Debug)]
pub enum GroupLifecycle {
    /// Persistent group — created when cluster forms, destroyed when cluster tears down.
    /// Used for "all-nodes broadcast" (model weights, config).
    Persistent,
    /// Ephemeral group — created for one operation, destroyed after completion.
    /// Used for subset operations (e.g., broadcast to specific GPU subset).
    Ephemeral {
        /// Transfer that created this group
        transfer_id: u64,
    },
}

#[derive(Clone, Debug)]
pub struct MemberState {
    pub node_id: NodeId,
    /// UD QP number for this member's multicast receive
    pub ud_qpn: u32,
    /// Join timestamp
    pub joined_at: Instant,
    /// Whether this member has confirmed receipt of current transfer
    pub transfer_complete: bool,
    /// Loss rate observed for this member (rolling average)
    pub observed_loss_rate: f64,
}

/// Multicast group manager — handles creation, join, leave, and cleanup.
pub struct MulticastManager {
    /// All active groups
    groups: RwLock<HashMap<MulticastGroupId, Arc<Mutex<MulticastGroup>>>>,
    /// IP multicast address allocator (239.0.0.1 - 239.0.0.255 range)
    addr_allocator: Mutex<MulticastAddrAllocator>,
    /// UD QP pool — pre-created UD QPs for multicast receive
    ud_qp_pool: UdQpPool,
    /// The persistent "all-nodes" group (created at startup)
    all_nodes_group: Option<MulticastGroupId>,
    /// IGMP snooping verified on this network
    igmp_verified: bool,
    /// Observed baseline loss rate (from verification)
    baseline_loss_rate: f64,
    /// Configuration
    config: MulticastConfig,
}

pub struct MulticastConfig {
    /// Maximum multicast groups to create
    pub max_groups: u32,                     // default: 64
    /// IP multicast range start
    pub mcast_range_start: Ipv4Addr,         // default: 239.0.0.1
    /// IP multicast range end
    pub mcast_range_end: Ipv4Addr,           // default: 239.0.0.254
    /// Loss rate threshold above which to disable multicast
    pub max_acceptable_loss_rate: f64,       // default: 0.05 (5%)
    /// NACK backoff range (random delay before sending NACK)
    pub nack_backoff_min: Duration,          // default: 10us
    pub nack_backoff_max: Duration,          // default: 100us
    /// Sender retransmission buffer size limit
    pub max_retransmit_buffer: u64,          // default: 2GB
    /// MTU for UD messages
    pub mtu: u32,                            // default: 4096
    /// Heartbeat interval (sender announces highest sequence)
    pub heartbeat_interval: Duration,        // default: 1ms
}
```

### Reliability Protocol Types

```rust
/// Header prepended to every multicast UD SEND payload.
/// Total overhead: 24 bytes per MTU-sized message.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C, packed)]
pub struct MulticastHeader {
    /// Transfer identifier (unique per broadcast operation)
    pub transfer_id: u64,
    /// Sequence number within this transfer (0-indexed)
    pub sequence: u64,
    /// Total sequences in this transfer (known upfront)
    pub total_sequences: u64,
}

/// NACK message — sent via unicast from receiver to sender.
/// Small: 32 bytes covers up to ~4 billion missing sequences via ranges.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NackMessage {
    pub transfer_id: u64,
    pub sender_node: NodeId,
    /// Missing sequence ranges (start, count)
    pub missing_ranges: SmallVec<[SequenceRange; 8]>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SequenceRange {
    pub start: u64,
    pub count: u64,
}

/// Transfer completion — sender announces end, receivers verify.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferComplete {
    pub transfer_id: u64,
    pub total_sequences: u64,
    /// CRC64 of the entire payload for end-to-end verification
    pub payload_crc: u64,
}

/// Receiver ACK — sent after receiver verifies completeness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiverAck {
    pub transfer_id: u64,
    pub node_id: NodeId,
    pub status: ReceiverStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReceiverStatus {
    /// All sequences received, CRC matches
    Complete,
    /// Still missing sequences after retransmission rounds
    Incomplete { missing: Vec<SequenceRange> },
    /// Gave up — requesting full tree fallback
    RequestTreeFallback,
}

/// Sender-side retransmission buffer.
/// Holds sent data until all receivers ACK.
pub struct RetransmitBuffer {
    /// transfer_id -> sent data chunks
    transfers: HashMap<u64, TransferData>,
    /// Total bytes currently buffered
    total_bytes: u64,
    /// Maximum bytes allowed (config.max_retransmit_buffer)
    max_bytes: u64,
}

pub struct TransferData {
    /// The actual data, indexed by sequence number
    /// Each entry is one MTU-sized payload
    chunks: Vec<Option<Bytes>>,
    /// Bitmap of which receivers have ACKed
    receiver_acks: HashMap<NodeId, bool>,
    /// Transfer start time (for timeout)
    started_at: Instant,
}
```

### Receiver-Side Reassembly

```rust
/// Receiver-side reassembly buffer for one multicast transfer.
pub struct ReassemblyBuffer {
    pub transfer_id: u64,
    /// Expected total sequences
    pub total_sequences: u64,
    /// Received sequence bitmap (for gap detection)
    pub received: BitVec,
    /// Reassembled data destination (host pinned or VRAM via staging)
    pub dest_buffer: DestinationBuffer,
    /// Number of sequences received so far
    pub received_count: u64,
    /// Last sequence number seen (for gap detection)
    pub highest_seen: u64,
    /// NACK state
    pub nack_state: NackState,
}

pub struct NackState {
    /// Sequences we've already NACKed (avoid re-NACKing)
    pub nacked: BitVec,
    /// Time of last NACK sent (for backoff)
    pub last_nack_sent: Option<Instant>,
    /// Number of NACK rounds for this transfer
    pub nack_rounds: u32,
}

pub enum DestinationBuffer {
    /// Host pinned memory (Phase 1 — standard path)
    HostPinned {
        addr: *mut u8,
        size: u64,
        lkey: u32,
    },
    /// BAR1 GPU VRAM (Phase 2 — OpenDMA path)
    Bar1Vram {
        bar1_offset: u64,
        size: u64,
        lkey: u32,
        grh_offset: u32, // 40 bytes for UD GRH
    },
}
```

---

## 2. Algorithms

### 2.1 Multicast Group Management

```rust
impl MulticastManager {
    /// Create a new multicast group.
    /// For persistent groups: called once at cluster startup.
    /// For ephemeral groups: called per-operation.
    pub fn create_group(
        &self,
        lifecycle: GroupLifecycle,
        root: NodeId,
        members: &[NodeId],
    ) -> Result<MulticastGroupId> {
        let group_id = MulticastGroupId(self.next_id());
        let mcast_addr = self.addr_allocator.lock().allocate()?;

        // 1. Create UD QP on root for sending
        let root_qp = self.ud_qp_pool.acquire()?;

        // 2. For each member: create UD QP, attach to multicast group
        let mut member_states = HashMap::new();
        for &node_id in members {
            let member_qp = self.create_member_qp(node_id, &mcast_addr)?;
            // ibv_attach_mcast(member_qp, &mgid, 0) on the member node
            // This is an RPC to the member node's OuterLink daemon
            self.rpc_attach_mcast(node_id, member_qp, &mcast_addr)?;

            member_states.insert(node_id, MemberState {
                node_id,
                ud_qpn: member_qp,
                joined_at: Instant::now(),
                transfer_complete: false,
                observed_loss_rate: 0.0,
            });
        }

        let group = MulticastGroup {
            group_id,
            mcast_addr,
            mgid: ip_to_gid(mcast_addr),
            lifecycle,
            members: member_states,
            root: Some(root),
            created_at: Instant::now(),
            stats: MulticastGroupStats::default(),
        };

        self.groups.write().insert(group_id, Arc::new(Mutex::new(group)));
        Ok(group_id)
    }

    /// Destroy an ephemeral group after transfer completes.
    pub fn destroy_group(&self, group_id: MulticastGroupId) -> Result<()> {
        let group = self.groups.write().remove(&group_id)
            .ok_or(Error::GroupNotFound)?;
        let group = group.lock();

        // Detach all members from multicast group
        for (node_id, member) in &group.members {
            self.rpc_detach_mcast(*node_id, member.ud_qpn, &group.mgid)?;
            self.ud_qp_pool.release(member.ud_qpn);
        }

        // Release IP multicast address
        self.addr_allocator.lock().release(group.mcast_addr);
        Ok(())
    }

    /// Verify IGMP snooping works on the network.
    /// Called once at startup. Sends test multicast and checks
    /// that non-members DON'T receive it.
    pub fn verify_igmp_snooping(&mut self) -> Result<bool> {
        // 1. Create test group with only node 0 as member
        // 2. Send test multicast from node 1
        // 3. Check that node 0 receives it
        // 4. Check that node 2 (non-member) does NOT receive it
        // 5. Measure baseline loss rate with 1000 test packets
        let (received_by_member, received_by_nonmember, loss_rate) =
            self.run_igmp_test()?;

        self.igmp_verified = received_by_member && !received_by_nonmember;
        self.baseline_loss_rate = loss_rate;

        if !self.igmp_verified {
            log::warn!(
                "IGMP snooping NOT working. Multicast disabled, using tree fallback."
            );
        }
        if loss_rate > self.config.max_acceptable_loss_rate {
            log::warn!(
                "Multicast loss rate {:.1}% exceeds threshold {:.1}%. Using tree fallback.",
                loss_rate * 100.0,
                self.config.max_acceptable_loss_rate * 100.0,
            );
        }

        Ok(self.igmp_verified && loss_rate <= self.config.max_acceptable_loss_rate)
    }
}
```

### 2.2 Reliable Multicast Send Protocol

```rust
/// Broadcast data from root to all group members with reliability.
///
/// Protocol:
///   1. Root fragments data into MTU-sized chunks with sequence headers
///   2. Root multicasts all chunks via UD SEND
///   3. Root sends TransferComplete message
///   4. Receivers detect gaps, send NACKs via unicast
///   5. Root retransmits NACKed sequences via unicast RC RDMA WRITE
///   6. Repeat NACK/retransmit until all receivers ACK complete
///   7. If any receiver requests tree fallback, switch to tree for that receiver
pub async fn reliable_multicast_send(
    group: &MulticastGroup,
    data: &[u8],
    retransmit_buf: &RetransmitBuffer,
    config: &MulticastConfig,
) -> Result<BroadcastStats> {
    let mtu_payload = config.mtu as usize - std::mem::size_of::<MulticastHeader>();
    let total_sequences = (data.len() + mtu_payload - 1) / mtu_payload;
    let transfer_id = next_transfer_id();

    // Phase 1: Multicast all chunks
    for seq in 0..total_sequences {
        let offset = seq * mtu_payload;
        let chunk_len = std::cmp::min(mtu_payload, data.len() - offset);
        let chunk = &data[offset..offset + chunk_len];

        let header = MulticastHeader {
            transfer_id,
            sequence: seq as u64,
            total_sequences: total_sequences as u64,
        };

        // Buffer for retransmission
        retransmit_buf.store(transfer_id, seq as u64, chunk);

        // UD SEND to multicast group
        ud_multicast_send(&group.mgid, &header, chunk)?;

        // Rate-limit to prevent receiver overflow
        // At 100Gbps with 4KB MTU: ~3M packets/sec max
        // Throttle to ~80% to give receivers breathing room
        if seq % 1000 == 0 {
            tokio::task::yield_now().await;
        }
    }

    // Phase 2: Send heartbeat with final sequence
    send_transfer_complete(group, transfer_id, total_sequences as u64, data)?;

    // Phase 3: NACK/retransmit loop
    let mut stats = BroadcastStats::new(total_sequences);
    let mut pending_receivers: HashSet<NodeId> =
        group.members.keys().cloned().collect();
    let mut nack_round = 0u32;
    let max_nack_rounds = 10;

    while !pending_receivers.is_empty() && nack_round < max_nack_rounds {
        // Wait for NACKs or ACKs (with timeout)
        let deadline = Instant::now() + Duration::from_millis(100);

        while Instant::now() < deadline && !pending_receivers.is_empty() {
            match recv_feedback(deadline - Instant::now()).await {
                Some(Feedback::Ack(ack)) => {
                    if ack.transfer_id == transfer_id {
                        match ack.status {
                            ReceiverStatus::Complete => {
                                pending_receivers.remove(&ack.node_id);
                                stats.receivers_complete += 1;
                            }
                            ReceiverStatus::Incomplete { ref missing } => {
                                // Retransmit missing via unicast RC RDMA WRITE
                                for range in missing {
                                    retransmit_via_unicast(
                                        ack.node_id,
                                        transfer_id,
                                        range,
                                        retransmit_buf,
                                    ).await?;
                                    stats.retransmitted_sequences +=
                                        range.count as usize;
                                }
                            }
                            ReceiverStatus::RequestTreeFallback => {
                                // This receiver gave up on multicast
                                tree_send_to_node(ack.node_id, data).await?;
                                pending_receivers.remove(&ack.node_id);
                                stats.tree_fallbacks += 1;
                            }
                        }
                    }
                }
                Some(Feedback::Nack(nack)) => {
                    if nack.transfer_id == transfer_id {
                        for range in &nack.missing_ranges {
                            retransmit_via_unicast(
                                nack.sender_node,
                                transfer_id,
                                range,
                                retransmit_buf,
                            ).await?;
                            stats.retransmitted_sequences += range.count as usize;
                        }
                    }
                }
                None => break, // timeout
            }
        }

        nack_round += 1;
    }

    // Cleanup retransmit buffer
    retransmit_buf.remove(transfer_id);

    // Any still-pending receivers: fall back to tree
    for node_id in &pending_receivers {
        tree_send_to_node(*node_id, data).await?;
        stats.tree_fallbacks += 1;
    }

    Ok(stats)
}
```

### 2.3 Receiver-Side Reassembly

```rust
/// Receiver event loop for one multicast transfer.
pub async fn reliable_multicast_recv(
    group_id: MulticastGroupId,
    expected_size: u64,
    config: &MulticastConfig,
) -> Result<Vec<u8>> {
    let mtu_payload = config.mtu as usize - std::mem::size_of::<MulticastHeader>();
    let total_sequences = (expected_size as usize + mtu_payload - 1) / mtu_payload;

    let mut reassembly = ReassemblyBuffer::new(total_sequences, expected_size);

    loop {
        match ud_multicast_recv(Duration::from_millis(50)).await {
            Some((header, payload)) => {
                if reassembly.received.get(header.sequence as usize) == Some(true) {
                    continue; // duplicate, skip
                }

                // Place data at correct offset
                let offset = header.sequence as usize * mtu_payload;
                reassembly.write_chunk(offset, &payload);
                reassembly.received.set(header.sequence as usize, true);
                reassembly.received_count += 1;
                reassembly.highest_seen =
                    std::cmp::max(reassembly.highest_seen, header.sequence);

                if reassembly.received_count == total_sequences as u64 {
                    // All received — send ACK
                    send_receiver_ack(ReceiverStatus::Complete).await?;
                    return Ok(reassembly.into_data());
                }
            }
            None => {
                // Timeout — check for gaps
            }
        }

        // Gap detection: if we've seen sequence N but are missing
        // sequences < N, generate NACK
        if reassembly.has_gaps() && reassembly.should_nack(config) {
            let missing = reassembly.compute_missing_ranges();
            let nack = NackMessage {
                transfer_id: reassembly.transfer_id,
                sender_node: local_node_id(),
                missing_ranges: missing,
            };

            // Random backoff to avoid NACK implosion
            let backoff = rand_duration(
                config.nack_backoff_min,
                config.nack_backoff_max,
            );
            tokio::time::sleep(backoff).await;

            send_nack_to_root(&nack).await?;
            reassembly.nack_state.last_nack_sent = Some(Instant::now());
            reassembly.nack_state.nack_rounds += 1;
        }

        // Too many NACK rounds — request tree fallback
        if reassembly.nack_state.nack_rounds > 10 {
            send_receiver_ack(ReceiverStatus::RequestTreeFallback).await?;
            // Wait for tree delivery
            return recv_tree_fallback(expected_size).await;
        }
    }
}
```

### 2.4 Tree Broadcast Fallback

```rust
/// Binary tree broadcast using RC RDMA WRITE.
/// Fallback when multicast is unavailable or loss is too high.
///
/// Tree structure for 8 nodes (root = 0):
///        0
///       / \
///      1   2
///     / \ / \
///    3  4 5  6
///   /
///  7
///
/// Depth = ceil(log2(8)) = 3 levels
/// Each level: parent RDMA WRITEs to both children in parallel
pub async fn tree_broadcast(
    root: NodeId,
    nodes: &[NodeId],
    data: &[u8],
) -> Result<()> {
    let tree = build_binary_tree(root, nodes);

    // BFS level-by-level broadcast
    for level in tree.levels() {
        let mut sends = Vec::new();
        for parent in level {
            for child in parent.children() {
                // RDMA WRITE from parent to child (RC, reliable)
                sends.push(rdma_write_reliable(parent.id, child.id, data));
            }
        }
        // All sends in this level execute in parallel
        futures::future::try_join_all(sends).await?;
    }

    Ok(())
}
```

---

## 3. Integration with Other Topics

### 3.1 R10 (Memory Tiering) Integration

R10's page table tells multicast what pages to broadcast:

```rust
/// Broadcast all pages of a tensor to all nodes via multicast.
/// Uses R10's bulk_lookup to get the page list, then either:
///   - If pages are contiguous: multicast the contiguous buffer directly
///   - If pages are fragmented: gather first (R28), then multicast
pub async fn broadcast_tensor(
    tensor_id: TensorId,
    page_table: &dyn PageTable,
    mcast_mgr: &MulticastManager,
) -> Result<()> {
    // R10: get physical pages
    let ptes = page_table.bulk_lookup(tensor_id);
    let addrs: Vec<u64> = ptes.iter().map(|p| p.phys_addr).collect();

    // R28: analyze fragmentation
    let analysis = analyze_fragments(&addrs);

    // Gather to contiguous buffer if fragmented (R28 software pre-pack)
    let contiguous_data = match analysis.recommendation {
        TransferMethod::SingleSge => {
            // Already contiguous — read directly
            read_contiguous_vram(analysis.runs[0].start_addr, analysis.total_bytes)
        }
        _ => {
            // Fragmented — use R28 gather to staging
            let staging = staging_pool.acquire()?;
            gpu_gather_to_staging(&analysis.runs, &staging)?;
            staging.as_bytes()
        }
    };

    // Get or create multicast group
    let group_id = mcast_mgr.all_nodes_group
        .unwrap_or_else(|| mcast_mgr.create_group(
            GroupLifecycle::Persistent,
            local_node_id(),
            &cluster_node_ids(),
        ).unwrap());

    let group = mcast_mgr.get_group(group_id)?;

    // Broadcast via reliable multicast
    reliable_multicast_send(&group, &contiguous_data, &retransmit_buf, &config).await
}
```

### 3.2 R12 (Memory Deduplication) Integration

R12 identifies duplicate pages across nodes. Multicast is the ideal transport for seeding all nodes with shared read-only data (model weights, embeddings):

```rust
/// R12 calls this when it identifies a set of pages that are
/// identical across N nodes. Instead of N unicast sends,
/// one multicast seeds all nodes simultaneously.
///
/// Data flow:
///   R12 detects: tensor T is identical on nodes [0, 1, 2, 3]
///   R12 picks canonical source: node 0
///   R12 calls multicast_seed(node_0, [1,2,3], tensor_T_data)
///   -> One multicast send delivers to all 3 receivers
///   -> 3x bandwidth savings vs 3 unicast sends
pub async fn multicast_seed_dedup_pages(
    source: NodeId,
    targets: &[NodeId],
    data: &[u8],
    mcast_mgr: &MulticastManager,
) -> Result<()> {
    if targets.len() == 1 {
        // Only one target — unicast is simpler
        return rdma_write_reliable(source, targets[0], data).await;
    }

    // Create ephemeral multicast group for this dedup operation
    let group_id = mcast_mgr.create_group(
        GroupLifecycle::Ephemeral { transfer_id: next_transfer_id() },
        source,
        targets,
    )?;

    let group = mcast_mgr.get_group(group_id)?;
    let result = reliable_multicast_send(&group, data, &retransmit_buf, &config).await;

    // Destroy ephemeral group
    mcast_mgr.destroy_group(group_id)?;
    result.map(|_| ())
}
```

### 3.3 R19 (Network Page Faults / Coherency) Integration

R19 v2 defines 13 coherency message types. Multicast enables bulk invalidation -- one message invalidates a page across all nodes that have it cached:

```rust
/// R19 coherency message types that benefit from multicast.
///
/// Instead of sending N unicast invalidation messages (one per caching node),
/// multicast sends ONE message that reaches all caching nodes.
///
/// R19 message types that map to multicast:
///   - INVALIDATE: page modified, all cached copies must be invalidated
///   - DOWNGRADE: page ownership changing, demote from exclusive to shared
///   - FLUSH: force dirty pages to be written back
///   - BARRIER: synchronization point across all nodes
///
/// R19 message types that remain unicast:
///   - FETCH: one node requesting a specific page (point-to-point)
///   - GRANT: owner granting access to requester (point-to-point)
///   - ACK: acknowledgment of specific operation (point-to-point)
pub async fn multicast_coherency_invalidate(
    page_id: PageId,
    caching_nodes: &[NodeId],
    mcast_mgr: &MulticastManager,
) -> Result<()> {
    // Coherency messages are small (< 100 bytes).
    // Use the persistent all-nodes group to avoid ephemeral group overhead.
    // Non-caching nodes receive and ignore the invalidation (checked by page_id).
    let group_id = mcast_mgr.all_nodes_group
        .ok_or(Error::NoMulticastGroup)?;
    let group = mcast_mgr.get_group(group_id)?;

    let invalidation = CoherencyMessage::Invalidate {
        page_id,
        invalidation_epoch: current_epoch(),
    };

    // Coherency messages are small — single UD SEND, no fragmentation
    // But they MUST be reliable. Use unicast RC as the delivery
    // mechanism and multicast only as a fast-path hint.
    //
    // Strategy: multicast the invalidation (fast delivery to all),
    // then unicast confirm to each caching node (reliable guarantee).
    ud_multicast_send_small(&group.mgid, &invalidation)?;

    // Reliable confirmation via unicast to each caching node
    for &node_id in caching_nodes {
        unicast_invalidate_confirm(node_id, &invalidation).await?;
    }

    Ok(())
}
```

**Why multicast + unicast confirm:** Coherency messages are latency-sensitive and must be 100% reliable. Multicast delivers the hint in ~1us to all nodes (fast path). The unicast confirmation ensures reliability. Nodes that received the multicast already processed the invalidation before the unicast arrives, so the unicast is a no-op ACK. Nodes that missed the multicast process it from the unicast. This gives best-case latency of multicast with worst-case reliability of unicast.

### 3.4 R20 (NCCL Backend) Integration

R20's NCCL plugin uses multicast for the Broadcast collective:

```rust
/// NCCL Broadcast collective implementation using multicast.
///
/// NCCL calls: ncclBroadcast(sendbuf, recvbuf, count, datatype, root, comm)
///
/// Decision logic:
///   - count * sizeof(datatype) > 4MB AND num_nodes > 2 AND multicast_available:
///       Use multicast path
///   - Otherwise:
///       Use standard tree/ring path
pub async fn nccl_broadcast_multicast(
    sendbuf: *const u8,
    recvbuf: *mut u8,
    byte_count: usize,
    root: i32,
    comm: &NcclComm,
    mcast_mgr: &MulticastManager,
) -> Result<()> {
    let use_multicast = byte_count > 4 * 1024 * 1024
        && comm.num_nodes() > 2
        && mcast_mgr.igmp_verified
        && mcast_mgr.baseline_loss_rate <= mcast_mgr.config.max_acceptable_loss_rate;

    if use_multicast {
        let data = unsafe { std::slice::from_raw_parts(sendbuf, byte_count) };
        let group_id = mcast_mgr.all_nodes_group
            .ok_or(Error::NoMulticastGroup)?;
        let group = mcast_mgr.get_group(group_id)?;

        reliable_multicast_send(&group, data, &retransmit_buf, &config).await?;
    } else {
        // Fallback to NCCL standard tree broadcast
        nccl_tree_broadcast(sendbuf, recvbuf, byte_count, root, comm).await?;
    }

    Ok(())
}

/// NCCL AllGather could use N simultaneous multicasts.
/// Each node multicasts its chunk to all others.
/// This is a Phase 2 optimization — only if loss rates are acceptable.
pub async fn nccl_allgather_multicast(
    sendbuf: *const u8,
    recvbuf: *mut u8,
    sendcount: usize,
    comm: &NcclComm,
    mcast_mgr: &MulticastManager,
) -> Result<()> {
    // Each node creates/uses an ephemeral multicast group
    // and sends its chunk simultaneously.
    // Receivers assemble from N multicast streams.
    // Complexity: N simultaneous multicast groups.
    // Only worth it if loss rate < 1% (N groups = N * loss overhead).

    let chunk_size = sendcount;
    let my_rank = comm.rank();

    // Send my chunk via multicast
    let my_data = unsafe { std::slice::from_raw_parts(sendbuf, chunk_size) };
    let group = mcast_mgr.get_or_create_rank_group(my_rank)?;
    reliable_multicast_send(&group, my_data, &retransmit_buf, &config).await?;

    // Receive all other chunks via multicast
    for rank in 0..comm.num_nodes() {
        if rank == my_rank { continue; }
        let recv_group = mcast_mgr.get_rank_group(rank)?;
        let chunk = reliable_multicast_recv(
            recv_group.group_id,
            chunk_size as u64,
            &config,
        ).await?;
        // Place in correct offset in recvbuf
        let offset = rank * chunk_size;
        unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.as_ptr(),
                recvbuf.add(offset),
                chunk_size,
            );
        }
    }

    Ok(())
}
```

### 3.5 R15 (Fault Tolerance / Checkpointing) Integration

R15 can distribute checkpoint data via multicast:

```
Checkpoint broadcast scenario:
  - Node 0 creates a checkpoint of its GPU state
  - Checkpoint must be replicated to N-1 nodes for fault tolerance
  - Multicast delivers the checkpoint to all nodes simultaneously
  - Same as model weight broadcast, but triggered periodically during training

R15 calls multicast_seed(source=0, targets=[1..N], data=checkpoint_data)
Uses the same reliable_multicast_send infrastructure as R12 dedup seeding.
```

---

## 4. Performance Model

### 4.1 Multicast vs Tree vs Sequential Unicast

```
Definitions:
  B = data size (bytes)
  N = number of receivers
  W = wire bandwidth (12.5 GB/s for 100Gbps)
  t_net = B / W  (transfer time for data at wire speed)
  L = loss rate (fraction of packets lost)
  t_nack = round-trip latency for NACK/retransmit (~50us)

Sequential unicast:
  t_seq = N * t_net
  For 140GB to 7 nodes: 7 * 11.2s = 78.4s

Tree broadcast (binary, ceil(log2(N)) levels):
  t_tree = ceil(log2(N)) * t_net
  For 140GB to 7 nodes: 3 * 11.2s = 33.6s

Multicast (with NACK reliability):
  t_mcast = t_net + L * total_sequences * t_nack
  Where total_sequences = B / (MTU - header_size)
  For 140GB at 4KB MTU: total_sequences = 36,700,160
  At 0% loss: 11.2s
  At 0.1% loss: 11.2s + 36,700 * 50us = 11.2s + 1.8s = 13.0s
  At 1% loss: 11.2s + 367,001 * 50us = 11.2s + 18.4s = 29.6s
  At 5% loss: 11.2s + 1,835,008 * 50us = 11.2s + 91.8s = 103s (WORSE than tree)
```

### 4.2 Crossover Analysis: When Multicast Wins

| Nodes (N) | Loss Rate (L) | Multicast | Tree | Winner |
|-----------|---------------|-----------|------|--------|
| 2 | 0% | 11.2s | 11.2s | Tie |
| 4 | 0% | 11.2s | 22.4s | Multicast (2x) |
| 4 | 1% | 29.6s | 22.4s | Tree |
| 4 | 0.1% | 13.0s | 22.4s | Multicast (1.7x) |
| 8 | 0% | 11.2s | 33.6s | Multicast (3x) |
| 8 | 0.1% | 13.0s | 33.6s | Multicast (2.6x) |
| 8 | 1% | 29.6s | 33.6s | Multicast (1.1x) |
| 8 | 2% | 48.0s | 33.6s | Tree |

**Key finding:** Multicast wins when loss rate < ~1.5% for 8 nodes. Above that, tree is better. This validates v1's 5% threshold as conservative -- the actual crossover for large transfers is around 1.5-2%.

**Revised threshold:** Use multicast when observed loss rate < 2%. Disable multicast (fall back to tree) when loss rate >= 2%.

### 4.3 Small Message Crossover

For small messages (NCCL collective calls):

```
UD SEND overhead per message: ~3us (including GRH, CQE)
RC RDMA WRITE overhead per message: ~1.5us
Tree broadcast small message (8 nodes, 3 levels): 3 * 1.5us = 4.5us
Multicast small message (8 nodes): 3us + reliability overhead

For messages < 64KB, single UD SEND = one multicast message.
Reliability overhead = 0 in happy path.
Multicast latency: ~3us
Tree latency: ~4.5us
Multicast wins by ~1.5us for small messages... BUT:
  - UD SEND has GRH overhead (40 bytes)
  - UD flow control is manual
  - NCCL LL protocol already achieves ~5us
  - Marginal improvement doesn't justify complexity
```

**Conclusion:** Multicast is NOT worth it for small messages (< 64KB). Use tree/ring for these.

### 4.4 Retransmission Buffer Sizing

```
Model weight broadcast: 140GB
Retransmission buffer = all sent data until all ACKs received
At 100Gbps, 140GB takes 11.2s to send
Receiver detection lag: ~100ms (heartbeat interval)
So retransmit buffer must hold ~100ms worth of data at wire speed = 1.25GB
Plus repair buffer for any lost sequences

Practical sizing:
  - Sliding window: buffer last 2GB of sent data
  - Older data is discarded (if a receiver needs it, it requests tree fallback)
  - config.max_retransmit_buffer = 2GB

For smaller broadcasts (< 2GB): buffer entire transfer.
For larger broadcasts (> 2GB): sliding window of 2GB.
```

---

## 5. Resolved Open Questions from v1

### v1 Q1: "Should multicast be transport-layer or specifically a collective operation?"
**Resolved: Both, layered.** The transport layer provides `reliable_multicast_send()` and `reliable_multicast_recv()` as primitives. These are used by:
- R20 NCCL backend for Broadcast/AllGather collectives
- R12 dedup for page seeding
- R19 coherency for bulk invalidation
- Cluster manager for model weight distribution

The transport layer handles reliability; consumers just call `send` with a group handle.

### v1 Q2: "How to handle heterogeneous clusters where some nodes have slower NICs?"
**Resolved: Send at the slowest receiver's rate.** The sender measures each receiver's `observed_loss_rate` (tracked in `MemberState`). If one receiver has a higher loss rate (indicating it can't keep up), the sender throttles to that receiver's effective rate. In practice, Pedro's cluster is homogeneous (all ConnectX-5 100Gbps), so this is a future concern only.

### v1 Q3: "Can we use multicast for ReduceScatter?"
**Resolved: No.** ReduceScatter sends different data to different destinations. This is inherently point-to-point. Multicast sends the same data to all destinations. There is no creative decomposition that makes ReduceScatter use multicast. AllGather (the complement) does benefit from multicast, which is why R20 integration focuses on Broadcast and AllGather only.

### v1 Q4: "Optimal sender retransmission buffer size?"
**Resolved:** 2GB sliding window for large transfers, full buffer for transfers < 2GB. See Section 4.4.

### Research Q: "Do Pedro's switches support IGMP snooping?"
**Partially resolved:** The `verify_igmp_snooping()` function tests this at runtime. If IGMP snooping is not working, OuterLink automatically falls back to tree broadcast. The v2 design does not require IGMP to be present -- it is an optimization, and tree is the guaranteed fallback.

### Research Q: "Can we use jumbo frames with RoCE multicast?"
**Resolved: Yes, if the switch supports it.** Jumbo frames (9000 MTU) reduce fragmentation overhead by ~2.25x (9000 vs 4096 payload). The `MulticastConfig::mtu` field is configurable. At startup, OuterLink probes for the maximum supported MTU and uses it. With jumbo frames, a 140GB broadcast requires ~16.3M UD SENDs instead of ~36.7M -- significant reduction in per-packet overhead.

### Research Q: "Multicast loopback (sender receives own multicast)?"
**Resolved: ConnectX-5 supports it via `ibv_attach_mcast` on the sender's QP.** But for OuterLink, the sender already has the data, so loopback is disabled. The sender does NOT join as a receiver -- only other nodes do.

---

## 6. Implementation Phases (Refined from v1)

### Phase 1: Tree Broadcast Foundation (1-2 weeks)

**Deliverables:**
- `tree_broadcast()` function using RC RDMA WRITE
- Binary tree construction from node list
- Chunked large data transfer (pipeline across tree levels)
- This is the guaranteed fallback -- must work first

**Acceptance criteria:**
- Broadcast 1GB to 7 nodes in < 35 seconds (3 levels)
- 100% reliable (RC guarantees)
- Works on any network (no IGMP needed)

### Phase 2: Multicast Infrastructure (2-3 weeks)

**Deliverables:**
- `MulticastManager` with create/join/leave/destroy
- `MulticastAddrAllocator` for IP multicast address management
- UD QP pool for multicast
- `verify_igmp_snooping()` test
- Basic UD SEND/RECV for multicast messages
- Persistent "all-nodes" group created at startup

**Acceptance criteria:**
- Can send/receive UD multicast between all cluster nodes
- IGMP snooping verification passes (or correctly falls back)
- Baseline loss rate measured and logged

### Phase 3: Reliable Multicast Protocol (3-4 weeks)

**Deliverables:**
- `MulticastHeader` with sequence numbering
- `ReassemblyBuffer` on receiver
- `NackMessage` generation and processing
- `RetransmitBuffer` on sender (2GB sliding window)
- Unicast RC RDMA WRITE retransmission path
- `TransferComplete` / `ReceiverAck` protocol
- Automatic fallback to tree when loss > 2%

**Acceptance criteria:**
- 1GB broadcast to 7 nodes via multicast in < 15 seconds
- Zero data corruption (CRC64 end-to-end verification)
- Handles 0-2% loss without falling back to tree
- Falls back to tree when loss > 2%

### Phase 4: NCCL and R12/R19 Integration (2-3 weeks)

**Deliverables:**
- `nccl_broadcast_multicast()` in R20 backend
- `multicast_seed_dedup_pages()` for R12
- `multicast_coherency_invalidate()` for R19
- Automatic algorithm selection in NCCL backend

**Acceptance criteria:**
- NCCL Broadcast uses multicast for messages > 4MB on clusters > 2 nodes
- R12 dedup seeding uses multicast when targets > 1 node
- R19 invalidation uses multicast hint + unicast confirm

### Phase 5: OpenDMA Integration (2-3 weeks, future)

**Deliverables:**
- UD RECV buffers registered on BAR1 GPU VRAM
- GRH offset handling for BAR1 targets
- Unicast repair via OpenDMA RDMA WRITE

**Acceptance criteria:**
- Multicast data arrives in GPU VRAM without host staging (happy path)
- GRH correctly handled (first 40 bytes skipped)

---

## 7. Success Metrics (Refined from v1)

| Metric | Baseline (Tree) | Target (Multicast) | Condition |
|--------|-----------------|-------------------|-----------|
| 1GB broadcast to 7 nodes | ~33.6s | < 15s | Loss < 2% |
| 140GB model load to 7 nodes | ~33.6s | ~12s | Loss < 0.1% |
| NCCL Broadcast 100MB, 8 nodes | ~800us (tree) | ~300us (multicast) | Loss < 1% |
| Sender bandwidth for 8-node broadcast | 3x data (tree) | 1x data | All cases |
| Network overhead | 0% | < 2% (NACKs + headers) | Loss < 2% |
| Fallback reliability | 100% (RC) | 100% (tree fallback) | All cases |

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-rdma-multicast-fundamentals.md](./research/01-rdma-multicast-fundamentals.md) -- IB/RoCE multicast mechanics
- [research/02-reliable-multicast.md](./research/02-reliable-multicast.md) -- Reliability approaches
- [research/03-multicast-for-ml.md](./research/03-multicast-for-ml.md) -- ML use cases
- R10 Memory Tiering -- PageTable bulk_lookup for broadcast data identification
- R12 Memory Deduplication -- Multicast seeding of deduplicated pages
- R15 Fault Tolerance -- Checkpoint distribution via multicast
- R19 Network Page Faults -- Bulk invalidation via multicast
- R20 NCCL Backend -- Broadcast and AllGather collectives

## Open Questions

- [ ] Measure actual UD multicast loss rate on Pedro's network (blocked on hardware access and IGMP verification)
- [ ] Benchmark jumbo frame support on Pedro's switches (9000 MTU vs 4096 MTU)
- [ ] Evaluate whether NCCL's CollNet interface is sufficient for our multicast backend or if a custom plugin path is needed
