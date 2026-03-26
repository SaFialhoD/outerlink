//! RDMA Multicast support for one-to-many GPU data distribution.
//!
//! Implements multicast group management, reliable multicast protocol with
//! NACK-based recovery, tree-based fallback, and receiver-side reassembly.
//!
//! # Architecture
//!
//! When broadcasting data (model weights, checkpoint data, deduplicated pages)
//! to multiple nodes, multicast sends one copy on the wire that reaches all
//! receivers simultaneously, instead of N unicast sends.
//!
//! The reliability layer adds sequence numbers, NACK-based gap detection, and
//! retransmission via unicast RC RDMA WRITE. When multicast is unavailable or
//! loss exceeds the configured threshold, the system falls back to binary tree
//! broadcast using reliable RC connections.
//!
//! # Protocol
//!
//! 1. Sender fragments data into MTU-sized chunks with sequence headers
//! 2. Sender multicasts all chunks via UD SEND
//! 3. Sender sends TransferComplete message
//! 4. Receivers detect gaps, send NACKs via unicast
//! 5. Sender retransmits NACKed sequences via unicast RC RDMA WRITE
//! 6. Repeat NACK/retransmit until all receivers ACK complete
//! 7. If any receiver requests tree fallback, switch to tree for that receiver

use bytes::Bytes;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Node and Group identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the OuterLink cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Unique identifier for a multicast group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MulticastGroupId(pub u32);

/// InfiniBand Global Identifier (GID) for multicast group addressing.
/// In RoCE v2, this maps from the IPv4 multicast address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gid {
    /// Raw 128-bit GID value.
    pub raw: [u8; 16],
}

impl Gid {
    /// Convert an IPv4 multicast address to an InfiniBand multicast GID.
    ///
    /// RoCE v2 mapping: prefix `ff0e::ffff:` + IPv4 address.
    /// RFC 4391 Section 4 defines the mapping.
    pub fn from_ipv4_multicast(addr: Ipv4Addr) -> Self {
        let octets = addr.octets();
        let mut raw = [0u8; 16];
        // IPv4-mapped multicast GID prefix
        raw[0] = 0xff;
        raw[1] = 0x0e;
        // bytes 2-11 are zero
        raw[10] = 0xff;
        raw[11] = 0xff;
        raw[12] = octets[0];
        raw[13] = octets[1];
        raw[14] = octets[2];
        raw[15] = octets[3];
        Gid { raw }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the multicast subsystem.
#[derive(Clone, Debug)]
pub struct MulticastConfig {
    /// Maximum multicast groups to create.
    pub max_groups: u32,
    /// IP multicast range start (inclusive).
    pub mcast_range_start: Ipv4Addr,
    /// IP multicast range end (inclusive).
    pub mcast_range_end: Ipv4Addr,
    /// Loss rate threshold above which to disable multicast and fall back to tree.
    pub max_acceptable_loss_rate: f64,
    /// Minimum NACK backoff duration (random delay before sending NACK).
    pub nack_backoff_min: Duration,
    /// Maximum NACK backoff duration.
    pub nack_backoff_max: Duration,
    /// Sender retransmission buffer size limit in bytes.
    pub max_retransmit_buffer: u64,
    /// MTU for UD messages in bytes.
    pub mtu: u32,
    /// Heartbeat interval (sender announces highest sequence).
    pub heartbeat_interval: Duration,
    /// Maximum NACK rounds before requesting tree fallback.
    pub max_nack_rounds: u32,
}

impl Default for MulticastConfig {
    fn default() -> Self {
        Self {
            max_groups: 64,
            mcast_range_start: Ipv4Addr::new(239, 0, 0, 1),
            mcast_range_end: Ipv4Addr::new(239, 0, 0, 254),
            max_acceptable_loss_rate: 0.05, // 5% — crossover point from perf model (preplan spec)
            nack_backoff_min: Duration::from_micros(10),
            nack_backoff_max: Duration::from_micros(100),
            max_retransmit_buffer: 2 * 1024 * 1024 * 1024, // 2GB
            mtu: 4096,
            heartbeat_interval: Duration::from_millis(1),
            max_nack_rounds: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Group lifecycle and member state
// ---------------------------------------------------------------------------

/// Lifecycle type for a multicast group.
#[derive(Clone, Debug, PartialEq)]
pub enum GroupLifecycle {
    /// Persistent group -- created when cluster forms, destroyed on teardown.
    /// Used for "all-nodes broadcast" (model weights, config).
    Persistent,
    /// Ephemeral group -- created for one operation, destroyed after completion.
    /// Used for subset operations (e.g., broadcast to specific GPU subset).
    Ephemeral {
        /// Transfer that created this group.
        transfer_id: u64,
    },
}

/// State of a single member within a multicast group.
#[derive(Clone, Debug)]
pub struct MemberState {
    /// Node identifier for this member.
    pub node_id: NodeId,
    /// UD QP number for this member's multicast receive.
    pub ud_qpn: u32,
    /// Join timestamp.
    pub joined_at: Instant,
    /// Whether this member has confirmed receipt of the current transfer.
    pub transfer_complete: bool,
    /// Rolling average loss rate observed for this member.
    pub observed_loss_rate: f64,
}

/// A multicast group in OuterLink.
/// Maps to one IP multicast address (RoCE v2) and one UD QP per member.
pub struct MulticastGroup {
    /// Unique group identifier within OuterLink.
    pub group_id: MulticastGroupId,
    /// IP multicast address (e.g., 239.0.0.1).
    pub mcast_addr: Ipv4Addr,
    /// Multicast GID for ibv_attach_mcast.
    pub mgid: Gid,
    /// Group lifecycle type.
    pub lifecycle: GroupLifecycle,
    /// Current members (node_id -> member state).
    pub members: HashMap<NodeId, MemberState>,
    /// The root/sender node (for broadcast groups).
    pub root: Option<NodeId>,
    /// Creation timestamp.
    pub created_at: Instant,
    /// Statistics for this group.
    pub stats: MulticastGroupStats,
}

/// Statistics tracked per multicast group.
#[derive(Clone, Debug, Default)]
pub struct MulticastGroupStats {
    /// Total transfers completed via this group.
    pub transfers_completed: u64,
    /// Total bytes sent via multicast through this group.
    pub bytes_multicast: u64,
    /// Total bytes sent via tree fallback through this group.
    pub bytes_tree_fallback: u64,
    /// Total sequences retransmitted via unicast.
    pub sequences_retransmitted: u64,
    /// Total NACK messages received.
    pub nacks_received: u64,
    /// Total tree fallback events.
    pub tree_fallback_count: u64,
}

// ---------------------------------------------------------------------------
// Reliability protocol types
// ---------------------------------------------------------------------------

/// Header prepended to every multicast UD SEND payload.
/// Total overhead: 24 bytes per MTU-sized message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct MulticastHeader {
    /// Transfer identifier (unique per broadcast operation).
    pub transfer_id: u64,
    /// Sequence number within this transfer (0-indexed).
    pub sequence: u64,
    /// Total sequences in this transfer (known upfront).
    pub total_sequences: u64,
}

impl MulticastHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = 24;
}

/// A range of sequence numbers (start inclusive, count elements).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceRange {
    /// First missing sequence number.
    pub start: u64,
    /// Number of consecutive missing sequences.
    pub count: u64,
}

impl SequenceRange {
    /// Create a new sequence range.
    pub fn new(start: u64, count: u64) -> Self {
        Self { start, count }
    }

    /// Returns true if this range contains the given sequence number.
    pub fn contains(&self, seq: u64) -> bool {
        seq >= self.start && seq < self.start + self.count
    }

    /// Iterator over all sequence numbers in this range.
    pub fn iter(&self) -> impl Iterator<Item = u64> {
        self.start..self.start + self.count
    }
}

/// NACK message -- sent via unicast from receiver to sender.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NackMessage {
    /// Transfer this NACK refers to.
    pub transfer_id: u64,
    /// Node sending the NACK.
    pub sender_node: NodeId,
    /// Missing sequence ranges.
    pub missing_ranges: Vec<SequenceRange>,
}

/// Transfer completion -- sender announces end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferComplete {
    /// Transfer that completed.
    pub transfer_id: u64,
    /// Total number of sequences in the transfer.
    pub total_sequences: u64,
    /// CRC64 of the entire payload for end-to-end verification.
    pub payload_crc: u64,
}

/// Receiver ACK -- sent after receiver verifies completeness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiverAck {
    /// Transfer being acknowledged.
    pub transfer_id: u64,
    /// Node sending the ACK.
    pub node_id: NodeId,
    /// Status of reception.
    pub status: ReceiverStatus,
}

/// Status of a receiver's reception of a multicast transfer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReceiverStatus {
    /// All sequences received, CRC matches.
    Complete,
    /// Still missing sequences after retransmission rounds.
    Incomplete { missing: Vec<SequenceRange> },
    /// Gave up -- requesting full tree fallback.
    RequestTreeFallback,
}

// ---------------------------------------------------------------------------
// Sender-side retransmission buffer
// ---------------------------------------------------------------------------

/// Sender-side retransmission buffer.
/// Holds sent data until all receivers ACK.
pub struct RetransmitBuffer {
    /// transfer_id -> sent data chunks.
    transfers: DashMap<u64, TransferData>,
    /// Total bytes currently buffered across all transfers.
    total_bytes: AtomicU64,
    /// Maximum bytes allowed (from config).
    max_bytes: u64,
}

/// Data for one in-flight transfer in the retransmit buffer.
#[allow(dead_code)] // started_at used for timeout tracking with RDMA hardware
pub struct TransferData {
    /// The actual data, indexed by sequence number.
    /// Each entry is one MTU-sized payload.
    chunks: Vec<Option<Bytes>>,
    /// Bitmap of which receivers have ACKed.
    receiver_acks: HashMap<NodeId, bool>,
    /// Transfer start time (for timeout).
    started_at: Instant,
}

impl RetransmitBuffer {
    /// Create a new retransmit buffer with the given capacity.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            transfers: DashMap::new(),
            total_bytes: AtomicU64::new(0),
            max_bytes,
        }
    }

    /// Store a chunk for potential retransmission.
    /// Returns false if the buffer is full and the chunk was not stored.
    ///
    /// Uses a compare-exchange loop to atomically reserve space before
    /// inserting, preventing TOCTOU races where concurrent callers both
    /// pass the capacity check.
    pub fn store(&self, transfer_id: u64, sequence: u64, data: &[u8]) -> bool {
        let chunk_bytes = data.len() as u64;

        // Atomically reserve space: CAS loop ensures no two threads can
        // both pass the capacity check for the same bytes.
        loop {
            let current = self.total_bytes.load(Ordering::Relaxed);
            if current + chunk_bytes > self.max_bytes {
                return false;
            }
            if self
                .total_bytes
                .compare_exchange(current, current + chunk_bytes, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        self.transfers
            .entry(transfer_id)
            .or_insert_with(|| TransferData {
                chunks: Vec::new(),
                receiver_acks: HashMap::new(),
                started_at: Instant::now(),
            })
            .value_mut()
            .store_chunk(sequence, data);

        true
    }

    /// Retrieve a chunk for retransmission.
    pub fn get_chunk(&self, transfer_id: u64, sequence: u64) -> Option<Bytes> {
        self.transfers.get(&transfer_id).and_then(|td| {
            td.chunks
                .get(sequence as usize)
                .and_then(|c| c.clone())
        })
    }

    /// Record that a receiver has ACKed a transfer.
    /// Returns true if all receivers have now ACKed.
    pub fn record_ack(
        &self,
        transfer_id: u64,
        node_id: NodeId,
        receivers: &[NodeId],
    ) -> bool {
        if let Some(mut td) = self.transfers.get_mut(&transfer_id) {
            td.receiver_acks.insert(node_id, true);
            receivers.iter().all(|n| td.receiver_acks.get(n) == Some(&true))
        } else {
            false
        }
    }

    /// Remove a completed transfer and free its buffer space.
    pub fn remove(&self, transfer_id: u64) {
        if let Some((_, td)) = self.transfers.remove(&transfer_id) {
            let freed: u64 = td
                .chunks
                .iter()
                .filter_map(|c| c.as_ref().map(|b| b.len() as u64))
                .sum();
            self.total_bytes.fetch_sub(freed, Ordering::Relaxed);
        }
    }

    /// Current total bytes buffered.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Number of active transfers in the buffer.
    pub fn active_transfers(&self) -> usize {
        self.transfers.len()
    }
}

impl TransferData {
    fn store_chunk(&mut self, sequence: u64, data: &[u8]) {
        let idx = sequence as usize;
        if idx >= self.chunks.len() {
            self.chunks.resize(idx + 1, None);
        }
        self.chunks[idx] = Some(Bytes::copy_from_slice(data));
    }
}

// ---------------------------------------------------------------------------
// Receiver-side reassembly
// ---------------------------------------------------------------------------

/// Receiver-side reassembly buffer for one multicast transfer.
pub struct ReassemblyBuffer {
    /// Transfer being reassembled.
    pub transfer_id: u64,
    /// Expected total sequences.
    pub total_sequences: u64,
    /// Bitmap of received sequences (index = sequence number).
    received: Vec<bool>,
    /// Reassembled data buffer.
    data: Vec<u8>,
    /// Number of sequences received so far.
    pub received_count: u64,
    /// Highest sequence number seen (for gap detection).
    pub highest_seen: u64,
    /// NACK state tracking.
    pub nack_state: NackState,
    /// MTU payload size (excluding header).
    mtu_payload: usize,
}

/// NACK tracking state for a receiver.
pub struct NackState {
    /// Sequences we've already NACKed (avoid re-NACKing too aggressively).
    pub nacked: Vec<bool>,
    /// Time of last NACK sent (for backoff).
    pub last_nack_sent: Option<Instant>,
    /// Number of NACK rounds for this transfer.
    pub nack_rounds: u32,
}

impl ReassemblyBuffer {
    /// Create a new reassembly buffer for a transfer.
    ///
    /// `total_sequences` is the number of MTU-sized chunks expected.
    /// `expected_size` is the total byte count of the original data.
    /// `mtu_payload` is the usable payload per chunk (MTU - header size).
    pub fn new(total_sequences: u64, expected_size: u64, mtu_payload: usize) -> Self {
        Self {
            transfer_id: 0,
            total_sequences,
            received: vec![false; total_sequences as usize],
            data: vec![0u8; expected_size as usize],
            received_count: 0,
            highest_seen: 0,
            nack_state: NackState {
                nacked: vec![false; total_sequences as usize],
                last_nack_sent: None,
                nack_rounds: 0,
            },
            mtu_payload,
        }
    }

    /// Write a chunk into the correct position in the reassembly buffer.
    /// Returns true if this was a new (non-duplicate) chunk.
    pub fn write_chunk(&mut self, header: &MulticastHeader, payload: &[u8]) -> bool {
        let seq = header.sequence as usize;
        if seq >= self.total_sequences as usize {
            return false;
        }
        if self.received[seq] {
            return false; // duplicate
        }

        let offset = seq * self.mtu_payload;
        let end = std::cmp::min(offset + payload.len(), self.data.len());
        let copy_len = end - offset;
        self.data[offset..offset + copy_len].copy_from_slice(&payload[..copy_len]);

        self.received[seq] = true;
        self.received_count += 1;
        if header.sequence > self.highest_seen {
            self.highest_seen = header.sequence;
        }

        true
    }

    /// Check if all sequences have been received.
    pub fn is_complete(&self) -> bool {
        self.received_count == self.total_sequences
    }

    /// Check if there are gaps in the received sequence numbers.
    pub fn has_gaps(&self) -> bool {
        if self.highest_seen == 0 && self.received_count <= 1 {
            return false;
        }
        // There are gaps if we've seen sequence N but haven't received
        // all sequences 0..=N
        self.received_count < self.highest_seen + 1
    }

    /// Whether the receiver should send a NACK now.
    /// Respects backoff timing.
    pub fn should_nack(&self, config: &MulticastConfig) -> bool {
        if !self.has_gaps() {
            return false;
        }
        if self.nack_state.nack_rounds >= config.max_nack_rounds {
            return false; // will request tree fallback instead
        }
        match self.nack_state.last_nack_sent {
            None => true,
            Some(last) => last.elapsed() >= config.nack_backoff_max,
        }
    }

    /// Whether the receiver should request tree fallback.
    pub fn should_request_tree_fallback(&self, config: &MulticastConfig) -> bool {
        self.nack_state.nack_rounds >= config.max_nack_rounds
    }

    /// Compute the ranges of missing sequence numbers up to `highest_seen`.
    ///
    /// Use `compute_all_missing_ranges()` after receiving `TransferComplete`
    /// to also detect trailing sequences that were never seen.
    pub fn compute_missing_ranges(&self) -> Vec<SequenceRange> {
        self.compute_missing_up_to((self.highest_seen + 1).min(self.total_sequences) as usize)
    }

    /// Compute all missing ranges across the full transfer.
    ///
    /// Call this after receiving `TransferComplete` when the receiver knows
    /// the final sequence count. This catches trailing sequences that were
    /// never seen (above `highest_seen`).
    pub fn compute_all_missing_ranges(&self) -> Vec<SequenceRange> {
        self.compute_missing_up_to(self.total_sequences as usize)
    }

    fn compute_missing_up_to(&self, limit: usize) -> Vec<SequenceRange> {
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < limit {
            if !self.received[i] {
                let start = i as u64;
                let mut count = 0u64;
                while i < limit && !self.received[i] {
                    count += 1;
                    i += 1;
                }
                ranges.push(SequenceRange::new(start, count));
            } else {
                i += 1;
            }
        }
        ranges
    }

    /// Mark sequences as NACKed to avoid duplicate NACKs.
    pub fn mark_nacked(&mut self, ranges: &[SequenceRange]) {
        for range in ranges {
            for seq in range.iter() {
                if (seq as usize) < self.nack_state.nacked.len() {
                    self.nack_state.nacked[seq as usize] = true;
                }
            }
        }
        self.nack_state.last_nack_sent = Some(Instant::now());
        self.nack_state.nack_rounds += 1;
    }

    /// Consume the reassembly buffer and return the assembled data.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Get a reference to the assembled data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the fraction of sequences received (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.total_sequences == 0 {
            return 1.0;
        }
        self.received_count as f64 / self.total_sequences as f64
    }
}

// ---------------------------------------------------------------------------
// Multicast address allocator
// ---------------------------------------------------------------------------

/// Allocator for IPv4 multicast addresses within a configured range.
/// Manages the 239.0.0.x range to avoid collisions between groups.
pub struct MulticastAddrAllocator {
    /// Start of the allocatable range.
    range_start: Ipv4Addr,
    /// End of the allocatable range (inclusive).
    range_end: Ipv4Addr,
    /// Currently allocated addresses.
    allocated: Vec<Ipv4Addr>,
    /// Next candidate (last octet offset from range_start).
    next_candidate: u8,
}

/// Errors specific to multicast operations.
#[derive(Debug, thiserror::Error)]
pub enum MulticastError {
    /// No more multicast addresses available in the configured range.
    #[error("multicast address pool exhausted")]
    AddressPoolExhausted,
    /// The specified multicast group was not found.
    #[error("multicast group not found: {0:?}")]
    GroupNotFound(MulticastGroupId),
    /// Maximum number of groups reached.
    #[error("maximum multicast groups ({0}) reached")]
    MaxGroupsReached(u32),
    /// Address was not allocated (double-free).
    #[error("multicast address {0} was not allocated")]
    AddressNotAllocated(Ipv4Addr),
    /// The retransmit buffer is full.
    #[error("retransmit buffer full ({0} bytes used of {1} max)")]
    RetransmitBufferFull(u64, u64),
    /// Transfer not found in retransmit buffer.
    #[error("transfer {0} not found in retransmit buffer")]
    TransferNotFound(u64),
    /// IGMP snooping verification is required but not performed.
    #[error("IGMP snooping not verified")]
    IgmpNotVerified,
    /// Multicast is disabled (loss rate too high).
    #[error("multicast disabled: loss rate {0:.1}% exceeds threshold {1:.1}%")]
    MulticastDisabled(f64, f64),
}

impl MulticastAddrAllocator {
    /// Create a new allocator for the given range.
    pub fn new(range_start: Ipv4Addr, range_end: Ipv4Addr) -> Self {
        Self {
            range_start,
            range_end,
            allocated: Vec::new(),
            next_candidate: 0,
        }
    }

    /// Allocate the next available multicast address.
    ///
    /// Uses u16 arithmetic internally to prevent u8 overflow when
    /// computing candidate addresses near the top of the range.
    pub fn allocate(&mut self) -> Result<Ipv4Addr, MulticastError> {
        let start_last = self.range_start.octets()[3] as u16;
        let end_last = self.range_end.octets()[3] as u16;
        let range_size = end_last - start_last + 1;

        // Linear scan for next available
        for _ in 0..range_size {
            let candidate_last = (start_last + self.next_candidate as u16) as u8;
            let octets = self.range_start.octets();
            let candidate = Ipv4Addr::new(octets[0], octets[1], octets[2], candidate_last);

            self.next_candidate = ((self.next_candidate as u16 + 1) % range_size) as u8;

            if !self.allocated.contains(&candidate) {
                self.allocated.push(candidate);
                return Ok(candidate);
            }
        }

        Err(MulticastError::AddressPoolExhausted)
    }

    /// Release a previously allocated address back to the pool.
    pub fn release(&mut self, addr: Ipv4Addr) -> Result<(), MulticastError> {
        if let Some(pos) = self.allocated.iter().position(|a| *a == addr) {
            self.allocated.remove(pos);
            Ok(())
        } else {
            Err(MulticastError::AddressNotAllocated(addr))
        }
    }

    /// Number of currently allocated addresses.
    pub fn allocated_count(&self) -> usize {
        self.allocated.len()
    }

    /// Total capacity of the address pool.
    pub fn capacity(&self) -> usize {
        let start_last = self.range_start.octets()[3];
        let end_last = self.range_end.octets()[3];
        (end_last - start_last + 1) as usize
    }

    /// Number of addresses still available.
    pub fn available(&self) -> usize {
        self.capacity() - self.allocated_count()
    }
}

// ---------------------------------------------------------------------------
// Binary tree broadcast fallback
// ---------------------------------------------------------------------------

/// A node in the binary broadcast tree.
#[derive(Clone, Debug)]
pub struct TreeNode {
    /// Node identifier.
    pub id: NodeId,
    /// Left child (if any).
    pub left: Option<Box<TreeNode>>,
    /// Right child (if any).
    pub right: Option<Box<TreeNode>>,
}

/// Binary broadcast tree for tree-based fallback.
/// Used when multicast is unavailable or loss rate is too high.
#[derive(Clone, Debug)]
pub struct BroadcastTree {
    /// Root of the tree.
    pub root: TreeNode,
    /// Total number of nodes in the tree.
    pub node_count: usize,
    /// Depth of the tree (number of levels).
    pub depth: usize,
}

impl BroadcastTree {
    /// Build a binary broadcast tree from a root and list of member nodes.
    ///
    /// The tree is constructed level by level:
    /// ```text
    ///        0
    ///       / \
    ///      1   2
    ///     / \ / \
    ///    3  4 5  6
    ///   /
    ///  7
    /// ```
    ///
    /// Depth = ceil(log2(N)) levels.
    /// Each level: parent sends to both children in parallel.
    pub fn build(root_id: NodeId, members: &[NodeId]) -> Self {
        let all_nodes: Vec<NodeId> = std::iter::once(root_id)
            .chain(members.iter().copied().filter(|n| *n != root_id))
            .collect();

        let node_count = all_nodes.len();
        let depth = if node_count <= 1 {
            0
        } else {
            (node_count as f64).log2().ceil() as usize
        };

        let root = Self::build_subtree(&all_nodes, 0);

        BroadcastTree {
            root,
            node_count,
            depth,
        }
    }

    fn build_subtree(nodes: &[NodeId], index: usize) -> TreeNode {
        let left = {
            let left_idx = 2 * index + 1;
            if left_idx < nodes.len() {
                Some(Box::new(Self::build_subtree(nodes, left_idx)))
            } else {
                None
            }
        };

        let right = {
            let right_idx = 2 * index + 2;
            if right_idx < nodes.len() {
                Some(Box::new(Self::build_subtree(nodes, right_idx)))
            } else {
                None
            }
        };

        TreeNode {
            id: nodes[index],
            left,
            right,
        }
    }

    /// Get the nodes at each level of the tree (BFS order).
    /// Returns a Vec of levels, each containing (parent_id, child_ids) pairs.
    pub fn levels(&self) -> Vec<Vec<(NodeId, Vec<NodeId>)>> {
        let mut result = Vec::new();
        let mut current_level = vec![&self.root];

        while !current_level.is_empty() {
            let mut level_pairs = Vec::new();
            let mut next_level = Vec::new();

            for node in &current_level {
                let mut children = Vec::new();
                if let Some(ref left) = node.left {
                    children.push(left.id);
                    next_level.push(left.as_ref());
                }
                if let Some(ref right) = node.right {
                    children.push(right.id);
                    next_level.push(right.as_ref());
                }
                if !children.is_empty() {
                    level_pairs.push((node.id, children));
                }
            }

            if !level_pairs.is_empty() {
                result.push(level_pairs);
            }
            current_level = next_level;
        }

        result
    }

    /// Collect all node IDs in BFS order.
    pub fn all_nodes_bfs(&self) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(&self.root);

        while let Some(node) = queue.pop_front() {
            result.push(node.id);
            if let Some(ref left) = node.left {
                queue.push_back(left);
            }
            if let Some(ref right) = node.right {
                queue.push_back(right);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Broadcast statistics
// ---------------------------------------------------------------------------

/// Statistics for a single broadcast operation.
#[derive(Clone, Debug, Default)]
pub struct BroadcastStats {
    /// Total sequences in the transfer.
    pub total_sequences: usize,
    /// Number of receivers that completed successfully.
    pub receivers_complete: usize,
    /// Number of sequences retransmitted via unicast.
    pub retransmitted_sequences: usize,
    /// Number of receivers that fell back to tree delivery.
    pub tree_fallbacks: usize,
    /// Total time for the broadcast.
    pub elapsed: Duration,
}

impl BroadcastStats {
    /// Create stats for a new broadcast.
    pub fn new(total_sequences: usize) -> Self {
        Self {
            total_sequences,
            ..Default::default()
        }
    }

    /// Compute the effective loss rate (retransmitted / total * receivers).
    pub fn effective_loss_rate(&self, num_receivers: usize) -> f64 {
        if num_receivers == 0 || self.total_sequences == 0 {
            return 0.0;
        }
        self.retransmitted_sequences as f64
            / (self.total_sequences as f64 * num_receivers as f64)
    }
}

// ---------------------------------------------------------------------------
// Global statistics
// ---------------------------------------------------------------------------

/// Global multicast subsystem statistics.
#[derive(Debug, Default)]
pub struct MulticastStats {
    /// Total broadcasts initiated.
    pub total_broadcasts: AtomicU64,
    /// Total broadcasts completed via multicast.
    pub multicast_broadcasts: AtomicU64,
    /// Total broadcasts completed via tree fallback.
    pub tree_broadcasts: AtomicU64,
    /// Total bytes sent via multicast.
    pub bytes_multicast: AtomicU64,
    /// Total bytes sent via tree.
    pub bytes_tree: AtomicU64,
    /// Total retransmissions.
    pub total_retransmissions: AtomicU64,
}

impl MulticastStats {
    /// Record a completed multicast broadcast.
    pub fn record_multicast(&self, bytes: u64, retransmissions: u64) {
        self.total_broadcasts.fetch_add(1, Ordering::Relaxed);
        self.multicast_broadcasts.fetch_add(1, Ordering::Relaxed);
        self.bytes_multicast.fetch_add(bytes, Ordering::Relaxed);
        self.total_retransmissions
            .fetch_add(retransmissions, Ordering::Relaxed);
    }

    /// Record a completed tree broadcast.
    pub fn record_tree(&self, bytes: u64) {
        self.total_broadcasts.fetch_add(1, Ordering::Relaxed);
        self.tree_broadcasts.fetch_add(1, Ordering::Relaxed);
        self.bytes_tree.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Snapshot of current stats.
    pub fn snapshot(&self) -> MulticastStatsSnapshot {
        MulticastStatsSnapshot {
            total_broadcasts: self.total_broadcasts.load(Ordering::Relaxed),
            multicast_broadcasts: self.multicast_broadcasts.load(Ordering::Relaxed),
            tree_broadcasts: self.tree_broadcasts.load(Ordering::Relaxed),
            bytes_multicast: self.bytes_multicast.load(Ordering::Relaxed),
            bytes_tree: self.bytes_tree.load(Ordering::Relaxed),
            total_retransmissions: self.total_retransmissions.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of multicast statistics.
#[derive(Clone, Debug)]
pub struct MulticastStatsSnapshot {
    pub total_broadcasts: u64,
    pub multicast_broadcasts: u64,
    pub tree_broadcasts: u64,
    pub bytes_multicast: u64,
    pub bytes_tree: u64,
    pub total_retransmissions: u64,
}

// ---------------------------------------------------------------------------
// Multicast Manager
// ---------------------------------------------------------------------------

/// Transfer ID generator.
static NEXT_TRANSFER_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique transfer ID.
pub fn next_transfer_id() -> u64 {
    NEXT_TRANSFER_ID.fetch_add(1, Ordering::Relaxed)
}

/// Group ID generator (u32 to match MulticastGroupId).
static NEXT_GROUP_ID: AtomicU32 = AtomicU32::new(1);

/// Multicast group manager -- handles creation, join, leave, and cleanup.
pub struct MulticastManager {
    /// All active groups.
    groups: DashMap<MulticastGroupId, Arc<Mutex<MulticastGroup>>>,
    /// IP multicast address allocator.
    addr_allocator: Mutex<MulticastAddrAllocator>,
    /// The persistent "all-nodes" group (created at startup).
    all_nodes_group: RwLock<Option<MulticastGroupId>>,
    /// IGMP snooping verified on this network.
    igmp_verified: RwLock<bool>,
    /// Observed baseline loss rate (from verification).
    baseline_loss_rate: RwLock<f64>,
    /// Configuration.
    config: MulticastConfig,
    /// Global statistics.
    stats: MulticastStats,
}

impl MulticastManager {
    /// Create a new multicast manager with the given configuration.
    pub fn new(config: MulticastConfig) -> Self {
        let allocator = MulticastAddrAllocator::new(
            config.mcast_range_start,
            config.mcast_range_end,
        );
        Self {
            groups: DashMap::new(),
            addr_allocator: Mutex::new(allocator),
            all_nodes_group: RwLock::new(None),
            igmp_verified: RwLock::new(false),
            baseline_loss_rate: RwLock::new(0.0),
            config,
            stats: MulticastStats::default(),
        }
    }

    /// Create a new multicast group.
    ///
    /// For persistent groups: called once at cluster startup.
    /// For ephemeral groups: called per-operation.
    pub fn create_group(
        &self,
        lifecycle: GroupLifecycle,
        root: NodeId,
        members: &[NodeId],
    ) -> Result<MulticastGroupId, MulticastError> {
        if self.groups.len() as u32 >= self.config.max_groups {
            return Err(MulticastError::MaxGroupsReached(self.config.max_groups));
        }

        let mcast_addr = self
            .addr_allocator
            .lock()
            .map_err(|_| MulticastError::AddressPoolExhausted)?
            .allocate()?;

        let group_id = MulticastGroupId(
            NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed) as u32,
        );
        let mgid = Gid::from_ipv4_multicast(mcast_addr);

        let mut member_states = HashMap::new();
        for &node_id in members {
            // In production, this would call ibv_attach_mcast on each member's
            // UD QP via RPC to the member node's OuterLink daemon.
            // TODO: requires RDMA hardware — stub QPN allocation
            let ud_qpn = node_id.0 * 1000; // placeholder QPN
            member_states.insert(
                node_id,
                MemberState {
                    node_id,
                    ud_qpn,
                    joined_at: Instant::now(),
                    transfer_complete: false,
                    observed_loss_rate: 0.0,
                },
            );
        }

        let group = MulticastGroup {
            group_id,
            mcast_addr,
            mgid,
            lifecycle,
            members: member_states,
            root: Some(root),
            created_at: Instant::now(),
            stats: MulticastGroupStats::default(),
        };

        self.groups
            .insert(group_id, Arc::new(Mutex::new(group)));

        Ok(group_id)
    }

    /// Destroy a multicast group, releasing its resources.
    pub fn destroy_group(
        &self,
        group_id: MulticastGroupId,
    ) -> Result<(), MulticastError> {
        let (_, group_arc) = self
            .groups
            .remove(&group_id)
            .ok_or(MulticastError::GroupNotFound(group_id))?;

        let group = group_arc
            .lock()
            .map_err(|_| MulticastError::GroupNotFound(group_id))?;

        // Release multicast address back to the pool.
        // In production, would also call ibv_detach_mcast on each member.
        // TODO: requires RDMA hardware — RPC detach calls
        if let Ok(mut allocator) = self.addr_allocator.lock() {
            let _ = allocator.release(group.mcast_addr);
        }

        Ok(())
    }

    /// Get a reference to a group by ID.
    pub fn get_group(
        &self,
        group_id: MulticastGroupId,
    ) -> Result<Arc<Mutex<MulticastGroup>>, MulticastError> {
        self.groups
            .get(&group_id)
            .map(|r| r.value().clone())
            .ok_or(MulticastError::GroupNotFound(group_id))
    }

    /// Add a member to an existing group.
    pub fn add_member(
        &self,
        group_id: MulticastGroupId,
        node_id: NodeId,
    ) -> Result<(), MulticastError> {
        let group_arc = self
            .groups
            .get(&group_id)
            .map(|r| r.value().clone())
            .ok_or(MulticastError::GroupNotFound(group_id))?;

        let mut group = group_arc
            .lock()
            .map_err(|_| MulticastError::GroupNotFound(group_id))?;

        // TODO: requires RDMA hardware — ibv_attach_mcast RPC
        let ud_qpn = node_id.0 * 1000;
        group.members.insert(
            node_id,
            MemberState {
                node_id,
                ud_qpn,
                joined_at: Instant::now(),
                transfer_complete: false,
                observed_loss_rate: 0.0,
            },
        );

        Ok(())
    }

    /// Remove a member from a group.
    pub fn remove_member(
        &self,
        group_id: MulticastGroupId,
        node_id: NodeId,
    ) -> Result<(), MulticastError> {
        let group_arc = self
            .groups
            .get(&group_id)
            .map(|r| r.value().clone())
            .ok_or(MulticastError::GroupNotFound(group_id))?;

        let mut group = group_arc
            .lock()
            .map_err(|_| MulticastError::GroupNotFound(group_id))?;

        // TODO: requires RDMA hardware — ibv_detach_mcast RPC
        group.members.remove(&node_id);
        Ok(())
    }

    /// Create the persistent "all-nodes" group at cluster startup.
    pub fn create_all_nodes_group(
        &self,
        root: NodeId,
        members: &[NodeId],
    ) -> Result<MulticastGroupId, MulticastError> {
        let group_id = self.create_group(
            GroupLifecycle::Persistent,
            root,
            members,
        )?;

        if let Ok(mut all_nodes) = self.all_nodes_group.write() {
            *all_nodes = Some(group_id);
        }

        Ok(group_id)
    }

    /// Get the persistent all-nodes group ID, if created.
    pub fn all_nodes_group(&self) -> Option<MulticastGroupId> {
        self.all_nodes_group
            .read()
            .ok()
            .and_then(|g| *g)
    }

    /// Record IGMP snooping verification result.
    pub fn set_igmp_verified(&self, verified: bool, loss_rate: f64) {
        if let Ok(mut v) = self.igmp_verified.write() {
            *v = verified;
        }
        if let Ok(mut lr) = self.baseline_loss_rate.write() {
            *lr = loss_rate;
        }
    }

    /// Whether IGMP snooping has been verified.
    pub fn is_igmp_verified(&self) -> bool {
        self.igmp_verified.read().map(|v| *v).unwrap_or(false)
    }

    /// Current baseline loss rate.
    pub fn baseline_loss_rate(&self) -> f64 {
        self.baseline_loss_rate.read().map(|v| *v).unwrap_or(1.0)
    }

    /// Whether multicast is currently enabled and usable.
    /// Requires IGMP verified and loss rate within threshold.
    pub fn is_multicast_available(&self) -> bool {
        self.is_igmp_verified()
            && self.baseline_loss_rate() <= self.config.max_acceptable_loss_rate
    }

    /// Decide whether to use multicast or tree for a given transfer.
    ///
    /// Decision criteria:
    /// - Data size > 4MB
    /// - Number of receivers > 2
    /// - Multicast available (IGMP verified, loss rate acceptable)
    pub fn should_use_multicast(&self, data_size: u64, num_receivers: usize) -> bool {
        const MIN_MULTICAST_SIZE: u64 = 4 * 1024 * 1024; // 4MB
        const MIN_MULTICAST_RECEIVERS: usize = 2;

        data_size > MIN_MULTICAST_SIZE
            && num_receivers > MIN_MULTICAST_RECEIVERS
            && self.is_multicast_available()
    }

    /// Number of active groups.
    pub fn active_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get a reference to the global statistics.
    pub fn global_stats(&self) -> &MulticastStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &MulticastConfig {
        &self.config
    }

    /// Compute the payload size per chunk (MTU minus header).
    pub fn mtu_payload_size(&self) -> usize {
        self.config.mtu as usize - MulticastHeader::SIZE
    }

    /// Calculate how many sequences are needed for a given data size.
    pub fn compute_total_sequences(&self, data_size: u64) -> u64 {
        let payload = self.mtu_payload_size() as u64;
        if payload == 0 {
            return 0;
        }
        (data_size + payload - 1) / payload
    }

    /// Fragment data into MTU-sized chunks with headers.
    ///
    /// Returns a vector of (header, payload) pairs ready for UD SEND.
    /// In production, these would be posted to the UD QP via ibv_post_send.
    /// TODO: requires RDMA hardware — actual UD SEND posting
    pub fn fragment_data(
        &self,
        transfer_id: u64,
        data: &[u8],
    ) -> Vec<(MulticastHeader, Bytes)> {
        let mtu_payload = self.mtu_payload_size();
        if mtu_payload == 0 {
            return Vec::new();
        }

        let total_sequences = self.compute_total_sequences(data.len() as u64);
        let mut fragments = Vec::with_capacity(total_sequences as usize);

        for seq in 0..total_sequences {
            let offset = seq as usize * mtu_payload;
            let chunk_len = std::cmp::min(mtu_payload, data.len() - offset);
            let chunk = Bytes::copy_from_slice(&data[offset..offset + chunk_len]);

            let header = MulticastHeader {
                transfer_id,
                sequence: seq,
                total_sequences,
            };

            fragments.push((header, chunk));
        }

        fragments
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MulticastConfig tests --

    #[test]
    fn test_config_defaults() {
        let config = MulticastConfig::default();
        assert_eq!(config.max_groups, 64);
        assert_eq!(config.mtu, 4096);
        assert_eq!(config.max_nack_rounds, 10);
        assert_eq!(config.mcast_range_start, Ipv4Addr::new(239, 0, 0, 1));
        assert_eq!(config.mcast_range_end, Ipv4Addr::new(239, 0, 0, 254));
        assert!((config.max_acceptable_loss_rate - 0.05).abs() < f64::EPSILON);
    }

    // -- Gid tests --

    #[test]
    fn test_gid_from_ipv4_multicast() {
        let addr = Ipv4Addr::new(239, 0, 0, 1);
        let gid = Gid::from_ipv4_multicast(addr);
        assert_eq!(gid.raw[0], 0xff);
        assert_eq!(gid.raw[1], 0x0e);
        assert_eq!(gid.raw[10], 0xff);
        assert_eq!(gid.raw[11], 0xff);
        assert_eq!(gid.raw[12], 239);
        assert_eq!(gid.raw[13], 0);
        assert_eq!(gid.raw[14], 0);
        assert_eq!(gid.raw[15], 1);
        // Middle bytes should be zero
        for i in 2..10 {
            assert_eq!(gid.raw[i], 0);
        }
    }

    #[test]
    fn test_gid_different_addresses_produce_different_gids() {
        let gid1 = Gid::from_ipv4_multicast(Ipv4Addr::new(239, 0, 0, 1));
        let gid2 = Gid::from_ipv4_multicast(Ipv4Addr::new(239, 0, 0, 2));
        assert_ne!(gid1, gid2);
    }

    // -- SequenceRange tests --

    #[test]
    fn test_sequence_range_contains() {
        let range = SequenceRange::new(5, 3);
        assert!(!range.contains(4));
        assert!(range.contains(5));
        assert!(range.contains(6));
        assert!(range.contains(7));
        assert!(!range.contains(8));
    }

    #[test]
    fn test_sequence_range_iter() {
        let range = SequenceRange::new(10, 4);
        let seqs: Vec<u64> = range.iter().collect();
        assert_eq!(seqs, vec![10, 11, 12, 13]);
    }

    #[test]
    fn test_sequence_range_empty() {
        let range = SequenceRange::new(0, 0);
        assert!(!range.contains(0));
        assert_eq!(range.iter().count(), 0);
    }

    // -- MulticastAddrAllocator tests --

    #[test]
    fn test_allocator_basic() {
        let mut alloc = MulticastAddrAllocator::new(
            Ipv4Addr::new(239, 0, 0, 1),
            Ipv4Addr::new(239, 0, 0, 5),
        );
        assert_eq!(alloc.capacity(), 5);
        assert_eq!(alloc.available(), 5);

        let addr1 = alloc.allocate().unwrap();
        assert_eq!(addr1, Ipv4Addr::new(239, 0, 0, 1));
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.available(), 4);

        let addr2 = alloc.allocate().unwrap();
        assert_eq!(addr2, Ipv4Addr::new(239, 0, 0, 2));
    }

    #[test]
    fn test_allocator_exhaustion() {
        let mut alloc = MulticastAddrAllocator::new(
            Ipv4Addr::new(239, 0, 0, 1),
            Ipv4Addr::new(239, 0, 0, 3),
        );

        let _ = alloc.allocate().unwrap();
        let _ = alloc.allocate().unwrap();
        let _ = alloc.allocate().unwrap();

        let result = alloc.allocate();
        assert!(matches!(result, Err(MulticastError::AddressPoolExhausted)));
    }

    #[test]
    fn test_allocator_release_and_reallocate() {
        let mut alloc = MulticastAddrAllocator::new(
            Ipv4Addr::new(239, 0, 0, 1),
            Ipv4Addr::new(239, 0, 0, 2),
        );

        let addr1 = alloc.allocate().unwrap();
        let addr2 = alloc.allocate().unwrap();
        assert_eq!(alloc.available(), 0);

        alloc.release(addr1).unwrap();
        assert_eq!(alloc.available(), 1);

        // Can allocate again after release
        let addr3 = alloc.allocate().unwrap();
        assert_eq!(addr3, addr1); // wraps around and finds the released one

        // Double release is an error
        let result = alloc.release(addr2);
        assert!(result.is_ok());
        let result = alloc.release(addr2);
        assert!(matches!(
            result,
            Err(MulticastError::AddressNotAllocated(_))
        ));
    }

    // -- RetransmitBuffer tests --

    #[test]
    fn test_retransmit_buffer_store_and_retrieve() {
        let buf = RetransmitBuffer::new(1024 * 1024);
        let data = b"hello multicast";

        assert!(buf.store(1, 0, data));
        assert_eq!(buf.total_bytes(), data.len() as u64);
        assert_eq!(buf.active_transfers(), 1);

        let chunk = buf.get_chunk(1, 0).unwrap();
        assert_eq!(&chunk[..], data);

        // Non-existent chunk
        assert!(buf.get_chunk(1, 999).is_none());
        assert!(buf.get_chunk(999, 0).is_none());
    }

    #[test]
    fn test_retransmit_buffer_capacity_limit() {
        let buf = RetransmitBuffer::new(100); // tiny buffer
        let data = vec![0u8; 60];

        assert!(buf.store(1, 0, &data));   // 60 bytes: fits
        assert!(!buf.store(1, 1, &data));  // 60+60=120 > 100: rejected
        assert_eq!(buf.total_bytes(), 60); // only first stored
    }

    #[test]
    fn test_retransmit_buffer_remove_frees_space() {
        let buf = RetransmitBuffer::new(1024);
        let data = vec![0u8; 100];

        buf.store(1, 0, &data);
        buf.store(1, 1, &data);
        assert_eq!(buf.total_bytes(), 200);

        buf.remove(1);
        assert_eq!(buf.total_bytes(), 0);
        assert_eq!(buf.active_transfers(), 0);
    }

    #[test]
    fn test_retransmit_buffer_ack_tracking() {
        let buf = RetransmitBuffer::new(1024);
        buf.store(1, 0, b"data");

        let receivers = vec![NodeId(1), NodeId(2), NodeId(3)];

        // Not all acked yet
        assert!(!buf.record_ack(1, NodeId(1), &receivers));
        assert!(!buf.record_ack(1, NodeId(2), &receivers));

        // All acked
        assert!(buf.record_ack(1, NodeId(3), &receivers));
    }

    // -- ReassemblyBuffer tests --

    #[test]
    fn test_reassembly_basic_complete() {
        let mtu_payload = 100;
        let data_size = 250u64;
        let total_seq = 3u64; // ceil(250/100)

        let mut buf = ReassemblyBuffer::new(total_seq, data_size, mtu_payload);
        buf.transfer_id = 42;

        assert!(!buf.is_complete());
        assert_eq!(buf.progress(), 0.0);

        // Write all 3 chunks
        let h0 = MulticastHeader {
            transfer_id: 42,
            sequence: 0,
            total_sequences: 3,
        };
        let h1 = MulticastHeader {
            transfer_id: 42,
            sequence: 1,
            total_sequences: 3,
        };
        let h2 = MulticastHeader {
            transfer_id: 42,
            sequence: 2,
            total_sequences: 3,
        };

        assert!(buf.write_chunk(&h0, &[1u8; 100]));
        assert!(buf.write_chunk(&h1, &[2u8; 100]));
        assert!(buf.write_chunk(&h2, &[3u8; 50])); // last chunk is smaller

        assert!(buf.is_complete());
        assert!((buf.progress() - 1.0).abs() < f64::EPSILON);

        let data = buf.into_data();
        assert_eq!(data.len(), 250);
        assert_eq!(&data[0..100], &[1u8; 100]);
        assert_eq!(&data[100..200], &[2u8; 100]);
        assert_eq!(&data[200..250], &[3u8; 50]);
    }

    #[test]
    fn test_reassembly_duplicate_detection() {
        let mut buf = ReassemblyBuffer::new(3, 300, 100);
        let h = MulticastHeader {
            transfer_id: 1,
            sequence: 0,
            total_sequences: 3,
        };

        assert!(buf.write_chunk(&h, &[1u8; 100]));  // new
        assert!(!buf.write_chunk(&h, &[1u8; 100])); // duplicate
        assert_eq!(buf.received_count, 1);
    }

    #[test]
    fn test_reassembly_out_of_order() {
        let mut buf = ReassemblyBuffer::new(3, 300, 100);

        // Receive chunks out of order: 2, 0, 1
        let h2 = MulticastHeader {
            transfer_id: 1,
            sequence: 2,
            total_sequences: 3,
        };
        let h0 = MulticastHeader {
            transfer_id: 1,
            sequence: 0,
            total_sequences: 3,
        };
        let h1 = MulticastHeader {
            transfer_id: 1,
            sequence: 1,
            total_sequences: 3,
        };

        assert!(buf.write_chunk(&h2, &[3u8; 100]));
        assert!(!buf.is_complete());
        assert!(buf.has_gaps());

        assert!(buf.write_chunk(&h0, &[1u8; 100]));
        assert!(buf.has_gaps()); // still missing seq 1

        assert!(buf.write_chunk(&h1, &[2u8; 100]));
        assert!(buf.is_complete());
        assert!(!buf.has_gaps());

        let data = buf.into_data();
        assert_eq!(&data[0..100], &[1u8; 100]);
        assert_eq!(&data[100..200], &[2u8; 100]);
        assert_eq!(&data[200..300], &[3u8; 100]);
    }

    #[test]
    fn test_reassembly_gap_detection_and_nack() {
        let config = MulticastConfig::default();
        let mut buf = ReassemblyBuffer::new(10, 1000, 100);

        // Receive sequences 0, 1, 4, 5 (missing 2, 3)
        for seq in [0u64, 1, 4, 5] {
            let h = MulticastHeader {
                transfer_id: 1,
                sequence: seq,
                total_sequences: 10,
            };
            buf.write_chunk(&h, &[seq as u8; 100]);
        }

        assert!(buf.has_gaps());
        assert!(buf.should_nack(&config));

        let missing = buf.compute_missing_ranges();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].start, 2);
        assert_eq!(missing[0].count, 2);

        // Mark as NACKed
        buf.mark_nacked(&missing);
        assert_eq!(buf.nack_state.nack_rounds, 1);
        assert!(buf.nack_state.last_nack_sent.is_some());
    }

    #[test]
    fn test_reassembly_multiple_gap_ranges() {
        let mut buf = ReassemblyBuffer::new(10, 1000, 100);

        // Receive 0, 3, 7 (gaps: 1-2, 4-6)
        for seq in [0u64, 3, 7] {
            let h = MulticastHeader {
                transfer_id: 1,
                sequence: seq,
                total_sequences: 10,
            };
            buf.write_chunk(&h, &[0u8; 100]);
        }

        let missing = buf.compute_missing_ranges();
        assert_eq!(missing.len(), 2);
        assert_eq!(missing[0], SequenceRange::new(1, 2));
        assert_eq!(missing[1], SequenceRange::new(4, 3));
    }

    #[test]
    fn test_reassembly_tree_fallback_trigger() {
        let config = MulticastConfig {
            max_nack_rounds: 3,
            ..MulticastConfig::default()
        };
        let mut buf = ReassemblyBuffer::new(5, 500, 100);

        // Simulate 3 NACK rounds
        buf.nack_state.nack_rounds = 3;
        assert!(buf.should_request_tree_fallback(&config));
        assert!(!buf.should_nack(&config));
    }

    #[test]
    fn test_reassembly_out_of_bounds_sequence() {
        let mut buf = ReassemblyBuffer::new(3, 300, 100);
        let h = MulticastHeader {
            transfer_id: 1,
            sequence: 99, // way out of bounds
            total_sequences: 3,
        };
        assert!(!buf.write_chunk(&h, &[0u8; 100]));
        assert_eq!(buf.received_count, 0);
    }

    // -- BroadcastTree tests --

    #[test]
    fn test_broadcast_tree_single_node() {
        let tree = BroadcastTree::build(NodeId(0), &[]);
        assert_eq!(tree.node_count, 1);
        assert_eq!(tree.depth, 0);
        assert_eq!(tree.root.id, NodeId(0));
        assert!(tree.root.left.is_none());
        assert!(tree.root.right.is_none());
    }

    #[test]
    fn test_broadcast_tree_eight_nodes() {
        let members: Vec<NodeId> = (0..8).map(NodeId).collect();
        let tree = BroadcastTree::build(NodeId(0), &members);

        assert_eq!(tree.node_count, 8);
        assert_eq!(tree.depth, 3); // ceil(log2(8)) = 3

        // Verify BFS order
        let bfs = tree.all_nodes_bfs();
        assert_eq!(bfs.len(), 8);
        assert_eq!(bfs[0], NodeId(0)); // root

        // Verify levels
        let levels = tree.levels();
        assert_eq!(levels.len(), 3); // 3 levels of parent->child pairs

        // Level 0: root sends to children
        assert_eq!(levels[0].len(), 1); // just root
        assert_eq!(levels[0][0].0, NodeId(0));
        assert_eq!(levels[0][0].1.len(), 2);
    }

    #[test]
    fn test_broadcast_tree_three_nodes() {
        let members = vec![NodeId(0), NodeId(1), NodeId(2)];
        let tree = BroadcastTree::build(NodeId(0), &members);

        assert_eq!(tree.node_count, 3);
        assert_eq!(tree.root.id, NodeId(0));
        assert_eq!(tree.root.left.as_ref().unwrap().id, NodeId(1));
        assert_eq!(tree.root.right.as_ref().unwrap().id, NodeId(2));
    }

    #[test]
    fn test_broadcast_tree_root_dedup() {
        // If root is also in members list, it should not be duplicated
        let members = vec![NodeId(0), NodeId(1), NodeId(2)];
        let tree = BroadcastTree::build(NodeId(0), &members);

        let bfs = tree.all_nodes_bfs();
        assert_eq!(bfs.len(), 3);
    }

    // -- MulticastManager tests --

    #[test]
    fn test_manager_create_and_destroy_group() {
        let config = MulticastConfig::default();
        let mgr = MulticastManager::new(config);

        let members = vec![NodeId(1), NodeId(2), NodeId(3)];
        let group_id = mgr.create_group(
            GroupLifecycle::Persistent,
            NodeId(0),
            &members,
        ).unwrap();

        assert_eq!(mgr.active_groups(), 1);

        let group = mgr.get_group(group_id).unwrap();
        let group = group.lock().unwrap();
        assert_eq!(group.members.len(), 3);
        assert_eq!(group.root, Some(NodeId(0)));
        assert!(matches!(group.lifecycle, GroupLifecycle::Persistent));
        drop(group);

        mgr.destroy_group(group_id).unwrap();
        assert_eq!(mgr.active_groups(), 0);
    }

    #[test]
    fn test_manager_ephemeral_group() {
        let config = MulticastConfig::default();
        let mgr = MulticastManager::new(config);

        let members = vec![NodeId(1), NodeId(2)];
        let group_id = mgr.create_group(
            GroupLifecycle::Ephemeral { transfer_id: 42 },
            NodeId(0),
            &members,
        ).unwrap();

        let group = mgr.get_group(group_id).unwrap();
        let group = group.lock().unwrap();
        assert_eq!(
            group.lifecycle,
            GroupLifecycle::Ephemeral { transfer_id: 42 }
        );
    }

    #[test]
    fn test_manager_max_groups_limit() {
        let config = MulticastConfig {
            max_groups: 2,
            ..MulticastConfig::default()
        };
        let mgr = MulticastManager::new(config);

        let members = vec![NodeId(1)];
        mgr.create_group(GroupLifecycle::Persistent, NodeId(0), &members)
            .unwrap();
        mgr.create_group(GroupLifecycle::Persistent, NodeId(0), &members)
            .unwrap();

        let result = mgr.create_group(
            GroupLifecycle::Persistent,
            NodeId(0),
            &members,
        );
        assert!(matches!(result, Err(MulticastError::MaxGroupsReached(2))));
    }

    #[test]
    fn test_manager_destroy_nonexistent_group() {
        let mgr = MulticastManager::new(MulticastConfig::default());
        let result = mgr.destroy_group(MulticastGroupId(999));
        assert!(matches!(
            result,
            Err(MulticastError::GroupNotFound(MulticastGroupId(999)))
        ));
    }

    #[test]
    fn test_manager_add_remove_member() {
        let mgr = MulticastManager::new(MulticastConfig::default());
        let members = vec![NodeId(1)];
        let group_id = mgr.create_group(
            GroupLifecycle::Persistent,
            NodeId(0),
            &members,
        ).unwrap();

        mgr.add_member(group_id, NodeId(2)).unwrap();
        {
            let group = mgr.get_group(group_id).unwrap();
            let group = group.lock().unwrap();
            assert_eq!(group.members.len(), 2);
            assert!(group.members.contains_key(&NodeId(2)));
        }

        mgr.remove_member(group_id, NodeId(2)).unwrap();
        {
            let group = mgr.get_group(group_id).unwrap();
            let group = group.lock().unwrap();
            assert_eq!(group.members.len(), 1);
            assert!(!group.members.contains_key(&NodeId(2)));
        }
    }

    #[test]
    fn test_manager_all_nodes_group() {
        let mgr = MulticastManager::new(MulticastConfig::default());
        assert!(mgr.all_nodes_group().is_none());

        let members = vec![NodeId(1), NodeId(2), NodeId(3)];
        let group_id = mgr.create_all_nodes_group(NodeId(0), &members).unwrap();

        assert_eq!(mgr.all_nodes_group(), Some(group_id));
    }

    #[test]
    fn test_manager_multicast_availability() {
        let mgr = MulticastManager::new(MulticastConfig::default());

        // Not available initially
        assert!(!mgr.is_multicast_available());

        // Set IGMP verified with low loss
        mgr.set_igmp_verified(true, 0.001);
        assert!(mgr.is_multicast_available());

        // Set high loss rate (above 5% threshold)
        mgr.set_igmp_verified(true, 0.08); // 8% > 5% threshold
        assert!(!mgr.is_multicast_available());

        // Set IGMP not verified
        mgr.set_igmp_verified(false, 0.0);
        assert!(!mgr.is_multicast_available());
    }

    #[test]
    fn test_manager_should_use_multicast_decision() {
        let mgr = MulticastManager::new(MulticastConfig::default());
        mgr.set_igmp_verified(true, 0.001);

        // Small data: no
        assert!(!mgr.should_use_multicast(1000, 4));

        // Too few receivers: no
        assert!(!mgr.should_use_multicast(10_000_000, 2));

        // Large data, many receivers, multicast available: yes
        assert!(mgr.should_use_multicast(10_000_000, 4));

        // Large data, many receivers, but multicast unavailable: no
        mgr.set_igmp_verified(false, 0.0);
        assert!(!mgr.should_use_multicast(10_000_000, 4));
    }

    #[test]
    fn test_manager_fragment_data() {
        let mgr = MulticastManager::new(MulticastConfig {
            mtu: 4096,
            ..MulticastConfig::default()
        });

        let payload_size = mgr.mtu_payload_size();
        assert_eq!(payload_size, 4096 - MulticastHeader::SIZE);

        // Create data that's exactly 2.5 chunks
        let data_size = payload_size * 2 + payload_size / 2;
        let data = vec![0xABu8; data_size];

        let tid = next_transfer_id();
        let fragments = mgr.fragment_data(tid, &data);
        assert_eq!(fragments.len(), 3);

        assert_eq!(fragments[0].0.sequence, 0);
        assert_eq!(fragments[0].0.total_sequences, 3);
        assert_eq!(fragments[0].1.len(), payload_size);

        assert_eq!(fragments[1].0.sequence, 1);
        assert_eq!(fragments[1].1.len(), payload_size);

        assert_eq!(fragments[2].0.sequence, 2);
        assert_eq!(fragments[2].1.len(), payload_size / 2);
    }

    // -- BroadcastStats tests --

    #[test]
    fn test_broadcast_stats_loss_rate() {
        let mut stats = BroadcastStats::new(1000);
        stats.retransmitted_sequences = 50;

        // 50 retransmissions across 4 receivers = 50 / (1000 * 4) = 1.25%
        let loss = stats.effective_loss_rate(4);
        assert!((loss - 0.0125).abs() < 0.0001);

        // Edge case: no receivers
        assert_eq!(stats.effective_loss_rate(0), 0.0);
    }

    // -- MulticastStats tests --

    #[test]
    fn test_global_stats_tracking() {
        let stats = MulticastStats::default();

        stats.record_multicast(1_000_000, 5);
        stats.record_multicast(2_000_000, 10);
        stats.record_tree(500_000);

        let snap = stats.snapshot();
        assert_eq!(snap.total_broadcasts, 3);
        assert_eq!(snap.multicast_broadcasts, 2);
        assert_eq!(snap.tree_broadcasts, 1);
        assert_eq!(snap.bytes_multicast, 3_000_000);
        assert_eq!(snap.bytes_tree, 500_000);
        assert_eq!(snap.total_retransmissions, 15);
    }

    // -- MulticastHeader tests --

    #[test]
    fn test_header_size() {
        assert_eq!(MulticastHeader::SIZE, 24);
        // Verify the struct size matches the constant
        assert_eq!(
            std::mem::size_of::<MulticastHeader>(),
            MulticastHeader::SIZE
        );
    }

    // -- Integration-style test --

    #[test]
    fn test_end_to_end_fragment_and_reassemble() {
        let config = MulticastConfig {
            mtu: 128, // small MTU for testing
            ..MulticastConfig::default()
        };
        let mgr = MulticastManager::new(config.clone());
        let payload_size = mgr.mtu_payload_size();

        // Original data
        let original: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let tid = next_transfer_id();
        let fragments = mgr.fragment_data(tid, &original);

        let total_seq = fragments.len() as u64;
        let mut reassembly = ReassemblyBuffer::new(
            total_seq,
            original.len() as u64,
            payload_size,
        );

        // Feed all fragments to reassembly
        for (header, payload) in &fragments {
            reassembly.write_chunk(header, payload);
        }

        assert!(reassembly.is_complete());
        let result = reassembly.into_data();
        assert_eq!(result, original);
    }

    #[test]
    fn test_end_to_end_with_simulated_loss_and_retransmit() {
        let config = MulticastConfig {
            mtu: 128,
            ..MulticastConfig::default()
        };
        let mgr = MulticastManager::new(config.clone());
        let payload_size = mgr.mtu_payload_size();
        let retransmit_buf = RetransmitBuffer::new(1024 * 1024);

        // Original data
        let original: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();
        let tid = next_transfer_id();
        let fragments = mgr.fragment_data(tid, &original);
        let total_seq = fragments.len() as u64;

        // Store all fragments in retransmit buffer
        for (header, payload) in &fragments {
            retransmit_buf.store(tid, header.sequence, payload);
        }

        // Receiver gets all fragments EXCEPT seq 1 (simulated loss)
        let mut reassembly = ReassemblyBuffer::new(
            total_seq,
            original.len() as u64,
            payload_size,
        );
        for (header, payload) in &fragments {
            if header.sequence != 1 {
                reassembly.write_chunk(header, payload);
            }
        }

        assert!(!reassembly.is_complete());
        assert!(reassembly.has_gaps());

        // Compute missing ranges
        let missing = reassembly.compute_missing_ranges();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].start, 1);
        assert_eq!(missing[0].count, 1);

        // Retransmit the missing chunk
        let retransmitted = retransmit_buf.get_chunk(tid, 1).unwrap();
        let repair_header = MulticastHeader {
            transfer_id: tid,
            sequence: 1,
            total_sequences: total_seq,
        };
        reassembly.write_chunk(&repair_header, &retransmitted);

        assert!(reassembly.is_complete());
        let result = reassembly.into_data();
        assert_eq!(result, original);
    }
}
