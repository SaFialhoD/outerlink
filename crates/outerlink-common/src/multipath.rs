//! Multi-Path Bandwidth Aggregation for OuterLink.
//!
//! Provides types and logic for splitting large transfers across multiple
//! network interfaces to aggregate bandwidth, inspired by FuseLink (OSDI 2025).
//! This module is purely computational -- no socket I/O.

use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Transfer ID generation
// ---------------------------------------------------------------------------

static NEXT_TRANSFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_transfer_id() -> u64 {
    NEXT_TRANSFER_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// NetworkInterface
// ---------------------------------------------------------------------------

/// Describes a single network interface available for transfers.
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// OS-level interface name (e.g. "eth0", "enp5s0").
    pub name: String,
    /// MAC address as a colon-separated hex string.
    pub mac_address: String,
    /// IPv4 or IPv6 address assigned to the interface.
    pub ip_address: String,
    /// Maximum link bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// NUMA node the NIC is attached to (for locality-aware scheduling).
    pub numa_node: u32,
    /// Whether this NIC supports RDMA (e.g. RoCE / InfiniBand).
    pub is_rdma_capable: bool,
    /// Whether the link is currently up.
    pub link_up: bool,
}

impl NetworkInterface {
    /// An interface is usable if its link is up AND it reports non-zero bandwidth.
    pub fn is_usable(&self) -> bool {
        self.link_up && self.bandwidth_bps > 0
    }
}

// ---------------------------------------------------------------------------
// InterfacePool
// ---------------------------------------------------------------------------

/// A collection of network interfaces available for multi-path transfers.
#[derive(Debug, Clone)]
pub struct InterfacePool {
    pub interfaces: Vec<NetworkInterface>,
}

impl InterfacePool {
    /// Returns only interfaces that are usable (link up + bandwidth > 0).
    pub fn usable_interfaces(&self) -> Vec<&NetworkInterface> {
        self.interfaces.iter().filter(|i| i.is_usable()).collect()
    }

    /// Sum of bandwidth across all usable interfaces.
    pub fn total_bandwidth_bps(&self) -> u64 {
        self.usable_interfaces().iter().map(|i| i.bandwidth_bps).sum()
    }

    /// Returns only RDMA-capable usable interfaces.
    pub fn rdma_interfaces(&self) -> Vec<&NetworkInterface> {
        self.usable_interfaces()
            .into_iter()
            .filter(|i| i.is_rdma_capable)
            .collect()
    }

    /// Total number of interfaces (including non-usable).
    pub fn count(&self) -> usize {
        self.interfaces.len()
    }
}

// ---------------------------------------------------------------------------
// SplitStrategy
// ---------------------------------------------------------------------------

/// Strategy for distributing a transfer across multiple interfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Equal-sized chunks rotated across interfaces.
    RoundRobin,
    /// Chunk sizes proportional to each interface's bandwidth.
    WeightedBandwidth,
    /// Minimise latency by picking the lowest-latency interface first.
    LatencyOptimal,
    /// Send the full payload on every interface for redundancy.
    Redundant,
}

impl SplitStrategy {
    /// Human-readable description of the strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::RoundRobin => "Equal-sized chunks rotated across interfaces",
            Self::WeightedBandwidth => "Chunk sizes proportional to interface bandwidth",
            Self::LatencyOptimal => "Route via lowest-latency interface first",
            Self::Redundant => "Send full payload on every interface for redundancy",
        }
    }
}

// ---------------------------------------------------------------------------
// SplitChunk
// ---------------------------------------------------------------------------

/// A single chunk of a split transfer assigned to one interface.
#[derive(Debug, Clone)]
pub struct SplitChunk {
    /// Unique identifier within the transfer.
    pub chunk_id: u32,
    /// Sequence number for reassembly ordering.
    pub sequence_number: u32,
    /// Byte offset into the original transfer.
    pub offset: u64,
    /// Length in bytes.
    pub length: u64,
    /// Name of the interface this chunk is assigned to.
    pub interface_name: String,
    /// Whether the chunk has been sent.
    pub sent: bool,
    /// Whether an acknowledgement has been received.
    pub acked: bool,
}

impl SplitChunk {
    /// Chunk has not yet been sent.
    pub fn is_pending(&self) -> bool {
        !self.sent
    }

    /// Chunk has been sent AND acknowledged.
    pub fn is_done(&self) -> bool {
        self.sent && self.acked
    }
}

// ---------------------------------------------------------------------------
// TransferSplit
// ---------------------------------------------------------------------------

/// The result of splitting a transfer: metadata plus a list of chunks.
#[derive(Debug, Clone)]
pub struct TransferSplit {
    pub transfer_id: u64,
    pub total_bytes: u64,
    pub chunks: Vec<SplitChunk>,
}

impl TransferSplit {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// A split is complete when every byte of `total_bytes` has been assigned
    /// to a chunk (no gaps, no shortfall).
    pub fn is_complete(&self) -> bool {
        self.total_assigned_bytes() == self.total_bytes
    }

    /// Sum of all chunk lengths.
    pub fn total_assigned_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.length).sum()
    }
}

// ---------------------------------------------------------------------------
// compute_split
// ---------------------------------------------------------------------------

/// Divide `total_bytes` across the given interfaces according to `strategy`.
///
/// * `min_chunk_bytes` prevents excessively small chunks (default 1 MB).
/// * Only usable interfaces are considered. If none are usable, returns an
///   empty split.
/// * For `Redundant`, every usable interface gets a full copy.
pub fn compute_split(
    total_bytes: u64,
    interfaces: &[NetworkInterface],
    strategy: &SplitStrategy,
    min_chunk_bytes: u64,
) -> TransferSplit {
    let usable: Vec<&NetworkInterface> = interfaces.iter().filter(|i| i.is_usable()).collect();

    let transfer_id = next_transfer_id();

    if usable.is_empty() || total_bytes == 0 {
        return TransferSplit {
            transfer_id,
            total_bytes,
            chunks: Vec::new(),
        };
    }

    let chunks = match strategy {
        SplitStrategy::Redundant => compute_redundant(total_bytes, &usable),
        SplitStrategy::RoundRobin => {
            compute_round_robin(total_bytes, &usable, min_chunk_bytes)
        }
        SplitStrategy::WeightedBandwidth => {
            compute_weighted(total_bytes, &usable, min_chunk_bytes)
        }
        SplitStrategy::LatencyOptimal => {
            // LatencyOptimal currently acts like weighted; a future version
            // will incorporate measured RTT. For now, treat identically to
            // WeightedBandwidth so that there is always a valid split.
            compute_weighted(total_bytes, &usable, min_chunk_bytes)
        }
    };

    TransferSplit {
        transfer_id,
        total_bytes,
        chunks,
    }
}

/// Redundant: one full-copy chunk per usable interface.
fn compute_redundant(total_bytes: u64, usable: &[&NetworkInterface]) -> Vec<SplitChunk> {
    usable
        .iter()
        .enumerate()
        .map(|(i, iface)| SplitChunk {
            chunk_id: i as u32,
            sequence_number: 0, // all are sequence 0 (full copy)
            offset: 0,
            length: total_bytes,
            interface_name: iface.name.clone(),
            sent: false,
            acked: false,
        })
        .collect()
}

/// RoundRobin: equal-sized chunks (last may be smaller).
fn compute_round_robin(
    total_bytes: u64,
    usable: &[&NetworkInterface],
    min_chunk_bytes: u64,
) -> Vec<SplitChunk> {
    let n = usable.len() as u64;
    // Target chunk size: total / n, but at least min_chunk_bytes.
    let raw_chunk = total_bytes / n;
    let chunk_size = if raw_chunk < min_chunk_bytes {
        // If enforcing min_chunk_bytes would exceed total_bytes, just use one
        // chunk per interface up to what fits.
        min_chunk_bytes
    } else {
        raw_chunk
    };

    let mut chunks = Vec::new();
    let mut offset: u64 = 0;
    let mut seq: u32 = 0;

    while offset < total_bytes {
        let iface = &usable[(seq as usize) % usable.len()];
        let remaining = total_bytes - offset;
        let length = remaining.min(chunk_size);

        chunks.push(SplitChunk {
            chunk_id: seq,
            sequence_number: seq,
            offset,
            length,
            interface_name: iface.name.clone(),
            sent: false,
            acked: false,
        });

        offset += length;
        seq += 1;
    }

    chunks
}

/// WeightedBandwidth: chunk sizes proportional to interface bandwidth.
fn compute_weighted(
    total_bytes: u64,
    usable: &[&NetworkInterface],
    min_chunk_bytes: u64,
) -> Vec<SplitChunk> {
    let total_bw: u64 = usable.iter().map(|i| i.bandwidth_bps).sum();
    if total_bw == 0 {
        // All usable interfaces have zero bandwidth -- degenerate case.
        return Vec::new();
    }

    // Compute raw proportional bytes per interface.
    let mut alloc: Vec<u64> = usable
        .iter()
        .map(|i| {
            // Use u128 to avoid overflow on large bandwidth * large total_bytes.
            ((i.bandwidth_bps as u128 * total_bytes as u128) / total_bw as u128) as u64
        })
        .collect();

    // Enforce min_chunk_bytes: interfaces whose allocation would be too small
    // get zero (excluded), and their bytes are redistributed.
    let mut excluded = vec![false; usable.len()];
    loop {
        let mut changed = false;
        for idx in 0..alloc.len() {
            if !excluded[idx] && alloc[idx] > 0 && alloc[idx] < min_chunk_bytes {
                excluded[idx] = true;
                alloc[idx] = 0;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Recompute among remaining.
        let remaining_bw: u64 = usable
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded[*i])
            .map(|(_, iface)| iface.bandwidth_bps)
            .sum();
        if remaining_bw == 0 {
            break;
        }
        for idx in 0..alloc.len() {
            if !excluded[idx] {
                alloc[idx] = ((usable[idx].bandwidth_bps as u128 * total_bytes as u128)
                    / remaining_bw as u128) as u64;
            }
        }
    }

    // Fix rounding: assign any remaining bytes to the highest-bandwidth
    // non-excluded interface.
    let assigned: u64 = alloc.iter().sum();
    let remainder = total_bytes.saturating_sub(assigned);
    if remainder > 0 {
        // Find the interface with highest bandwidth that is not excluded.
        if let Some(best) = usable
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded[*i])
            .max_by_key(|(_, iface)| iface.bandwidth_bps)
            .map(|(i, _)| i)
        {
            alloc[best] += remainder;
        }
    }

    // Build chunks.
    let mut chunks = Vec::new();
    let mut offset: u64 = 0;
    let mut seq: u32 = 0;

    for (idx, bytes) in alloc.iter().enumerate() {
        if *bytes == 0 {
            continue;
        }
        chunks.push(SplitChunk {
            chunk_id: seq,
            sequence_number: seq,
            offset,
            length: *bytes,
            interface_name: usable[idx].name.clone(),
            sent: false,
            acked: false,
        });
        offset += bytes;
        seq += 1;
    }

    chunks
}

// ---------------------------------------------------------------------------
// ReassemblyBuffer
// ---------------------------------------------------------------------------

/// Tracks which chunks of a transfer have been received on the remote side.
#[derive(Debug, Clone)]
pub struct ReassemblyBuffer {
    pub transfer_id: u64,
    pub expected_chunks: u32,
    received: Vec<bool>,
}

impl ReassemblyBuffer {
    /// Create a new buffer expecting `expected_chunks` chunks.
    pub fn new(transfer_id: u64, expected_chunks: u32) -> Self {
        Self {
            transfer_id,
            expected_chunks,
            received: vec![false; expected_chunks as usize],
        }
    }

    /// Record that chunk with the given sequence number has been received.
    /// Returns `false` if `seq` is out of range.
    pub fn record_chunk(&mut self, seq: u32) -> bool {
        if (seq as usize) < self.received.len() {
            self.received[seq as usize] = true;
            true
        } else {
            false
        }
    }

    /// All expected chunks have been received.
    pub fn is_complete(&self) -> bool {
        self.received.iter().all(|&r| r)
    }

    /// Returns sequence numbers of chunks not yet received.
    pub fn missing_chunks(&self) -> Vec<u32> {
        self.received
            .iter()
            .enumerate()
            .filter(|(_, &r)| !r)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Completion as a percentage (0.0 -- 100.0).
    pub fn completion_percent(&self) -> f64 {
        if self.expected_chunks == 0 {
            return 100.0;
        }
        let received_count = self.received.iter().filter(|&&r| r).count();
        (received_count as f64 / self.expected_chunks as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// MultipathConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for multi-path bandwidth aggregation.
#[derive(Debug, Clone)]
pub struct MultipathConfig {
    /// Whether multi-path splitting is enabled at all.
    pub enabled: bool,
    /// Minimum chunk size in bytes (prevents excessively small chunks).
    pub min_chunk_bytes: u64,
    /// Strategy used to distribute data across interfaces.
    pub strategy: SplitStrategy,
    /// Transfers smaller than this threshold use a single interface.
    pub split_threshold_bytes: u64,
    /// If true, transfers below `split_threshold_bytes` are sent redundantly
    /// on all interfaces for reliability.
    pub enable_redundant_small_transfers: bool,
}

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_chunk_bytes: 1_048_576,       // 1 MB
            strategy: SplitStrategy::WeightedBandwidth,
            split_threshold_bytes: 1_048_576, // 1 MB
            enable_redundant_small_transfers: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn make_iface(name: &str, bw_gbps: u64, rdma: bool, up: bool) -> NetworkInterface {
        NetworkInterface {
            name: name.to_string(),
            mac_address: "00:11:22:33:44:55".to_string(),
            ip_address: "10.0.0.1".to_string(),
            bandwidth_bps: bw_gbps * 1_000_000_000,
            numa_node: 0,
            is_rdma_capable: rdma,
            link_up: up,
        }
    }

    const MB: u64 = 1_048_576;

    // ===== NetworkInterface =====

    #[test]
    fn interface_usable_when_up_and_has_bandwidth() {
        let iface = make_iface("eth0", 100, false, true);
        assert!(iface.is_usable());
    }

    #[test]
    fn interface_not_usable_when_link_down() {
        let iface = make_iface("eth0", 100, false, false);
        assert!(!iface.is_usable());
    }

    #[test]
    fn interface_not_usable_when_zero_bandwidth() {
        let mut iface = make_iface("eth0", 0, false, true);
        iface.bandwidth_bps = 0;
        assert!(!iface.is_usable());
    }

    // ===== InterfacePool =====

    #[test]
    fn pool_usable_filters_down_interfaces() {
        let pool = InterfacePool {
            interfaces: vec![
                make_iface("eth0", 100, false, true),
                make_iface("eth1", 25, false, false), // link down
                make_iface("eth2", 50, true, true),
            ],
        };
        assert_eq!(pool.usable_interfaces().len(), 2);
    }

    #[test]
    fn pool_total_bandwidth_sums_usable_only() {
        let pool = InterfacePool {
            interfaces: vec![
                make_iface("eth0", 100, false, true),
                make_iface("eth1", 25, false, false), // down
                make_iface("eth2", 50, true, true),
            ],
        };
        assert_eq!(pool.total_bandwidth_bps(), 150_000_000_000);
    }

    #[test]
    fn pool_rdma_interfaces_filters_correctly() {
        let pool = InterfacePool {
            interfaces: vec![
                make_iface("eth0", 100, false, true),
                make_iface("mlx0", 100, true, true),
                make_iface("mlx1", 100, true, false), // down
            ],
        };
        let rdma = pool.rdma_interfaces();
        assert_eq!(rdma.len(), 1);
        assert_eq!(rdma[0].name, "mlx0");
    }

    #[test]
    fn pool_count_includes_all() {
        let pool = InterfacePool {
            interfaces: vec![
                make_iface("eth0", 100, false, true),
                make_iface("eth1", 25, false, false),
            ],
        };
        assert_eq!(pool.count(), 2);
    }

    // ===== SplitStrategy =====

    #[test]
    fn strategy_descriptions_are_non_empty() {
        assert!(!SplitStrategy::RoundRobin.description().is_empty());
        assert!(!SplitStrategy::WeightedBandwidth.description().is_empty());
        assert!(!SplitStrategy::LatencyOptimal.description().is_empty());
        assert!(!SplitStrategy::Redundant.description().is_empty());
    }

    // ===== SplitChunk state transitions =====

    #[test]
    fn chunk_starts_pending() {
        let chunk = SplitChunk {
            chunk_id: 0,
            sequence_number: 0,
            offset: 0,
            length: 1024,
            interface_name: "eth0".into(),
            sent: false,
            acked: false,
        };
        assert!(chunk.is_pending());
        assert!(!chunk.is_done());
    }

    #[test]
    fn chunk_sent_but_not_acked_is_not_done() {
        let chunk = SplitChunk {
            chunk_id: 0,
            sequence_number: 0,
            offset: 0,
            length: 1024,
            interface_name: "eth0".into(),
            sent: true,
            acked: false,
        };
        assert!(!chunk.is_pending());
        assert!(!chunk.is_done());
    }

    #[test]
    fn chunk_sent_and_acked_is_done() {
        let chunk = SplitChunk {
            chunk_id: 0,
            sequence_number: 0,
            offset: 0,
            length: 1024,
            interface_name: "eth0".into(),
            sent: true,
            acked: true,
        };
        assert!(!chunk.is_pending());
        assert!(chunk.is_done());
    }

    // ===== compute_split -- WeightedBandwidth =====

    #[test]
    fn weighted_split_proportional_to_bandwidth() {
        // 75% / 25% bandwidth split.
        let ifaces = vec![
            make_iface("fast", 75, false, true),
            make_iface("slow", 25, false, true),
        ];
        let split = compute_split(100 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);

        assert!(split.is_complete());
        assert_eq!(split.chunk_count(), 2);

        let fast_chunk = split.chunks.iter().find(|c| c.interface_name == "fast").unwrap();
        let slow_chunk = split.chunks.iter().find(|c| c.interface_name == "slow").unwrap();

        // fast should get ~75 MB, slow ~25 MB.
        assert_eq!(fast_chunk.length, 75 * MB);
        assert_eq!(slow_chunk.length, 25 * MB);
    }

    #[test]
    fn weighted_split_handles_rounding() {
        // 3 interfaces with equal bandwidth, 10 MB total -> each gets ~3.33 MB
        // but rounding means one gets the remainder.
        let ifaces = vec![
            make_iface("a", 100, false, true),
            make_iface("b", 100, false, true),
            make_iface("c", 100, false, true),
        ];
        let split = compute_split(10 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);

        assert!(split.is_complete());
        assert_eq!(split.total_assigned_bytes(), 10 * MB);
    }

    #[test]
    fn weighted_split_excludes_tiny_chunks() {
        // Two interfaces: 99% and 1% bandwidth. With 2 MB total, the slow
        // interface would get ~20 KB which is below min_chunk_bytes (1 MB).
        // It should be excluded, and fast gets everything.
        let ifaces = vec![
            make_iface("fast", 99, false, true),
            make_iface("slow", 1, false, true),
        ];
        let split = compute_split(2 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);

        assert!(split.is_complete());
        assert_eq!(split.chunk_count(), 1);
        assert_eq!(split.chunks[0].interface_name, "fast");
    }

    #[test]
    fn weighted_single_interface_passthrough() {
        let ifaces = vec![make_iface("eth0", 100, false, true)];
        let split = compute_split(50 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);

        assert!(split.is_complete());
        assert_eq!(split.chunk_count(), 1);
        assert_eq!(split.chunks[0].length, 50 * MB);
    }

    // ===== compute_split -- RoundRobin =====

    #[test]
    fn round_robin_equal_chunks() {
        let ifaces = vec![
            make_iface("a", 100, false, true),
            make_iface("b", 100, false, true),
        ];
        let split = compute_split(10 * MB, &ifaces, &SplitStrategy::RoundRobin, MB);

        assert!(split.is_complete());
        // Each interface should get 5 MB.
        let a_bytes: u64 = split
            .chunks
            .iter()
            .filter(|c| c.interface_name == "a")
            .map(|c| c.length)
            .sum();
        let b_bytes: u64 = split
            .chunks
            .iter()
            .filter(|c| c.interface_name == "b")
            .map(|c| c.length)
            .sum();
        assert_eq!(a_bytes, 5 * MB);
        assert_eq!(b_bytes, 5 * MB);
    }

    #[test]
    fn round_robin_last_chunk_may_be_smaller() {
        let ifaces = vec![
            make_iface("a", 100, false, true),
            make_iface("b", 100, false, true),
            make_iface("c", 100, false, true),
        ];
        // 10 MB across 3 interfaces -> chunk_size = 10485760/3 = 3495253.
        // 3 chunks of 3495253 = 10485759, remainder = 1 byte as the 4th chunk.
        let total = 10 * MB;
        let split = compute_split(total, &ifaces, &SplitStrategy::RoundRobin, MB);

        assert!(split.is_complete());
        assert_eq!(split.total_assigned_bytes(), total);
        // The last chunk should be smaller than the others.
        let last = split.chunks.last().unwrap();
        let first = &split.chunks[0];
        assert!(last.length < first.length, "last chunk should be smaller than first");
    }

    #[test]
    fn round_robin_sequence_numbers_are_sequential() {
        let ifaces = vec![
            make_iface("a", 100, false, true),
            make_iface("b", 100, false, true),
        ];
        let split = compute_split(4 * MB, &ifaces, &SplitStrategy::RoundRobin, MB);

        for (i, chunk) in split.chunks.iter().enumerate() {
            assert_eq!(chunk.sequence_number, i as u32);
        }
    }

    // ===== compute_split -- Redundant =====

    #[test]
    fn redundant_sends_full_copy_on_each_interface() {
        let ifaces = vec![
            make_iface("a", 100, false, true),
            make_iface("b", 50, false, true),
        ];
        let split = compute_split(10 * MB, &ifaces, &SplitStrategy::Redundant, MB);

        // Two chunks, each with full data.
        assert_eq!(split.chunk_count(), 2);
        for chunk in &split.chunks {
            assert_eq!(chunk.length, 10 * MB);
            assert_eq!(chunk.offset, 0);
        }
    }

    // ===== compute_split -- edge cases =====

    #[test]
    fn split_with_no_usable_interfaces_returns_empty() {
        let ifaces = vec![make_iface("down", 100, false, false)];
        let split = compute_split(10 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);
        assert_eq!(split.chunk_count(), 0);
    }

    #[test]
    fn split_with_zero_bytes_returns_empty() {
        let ifaces = vec![make_iface("eth0", 100, false, true)];
        let split = compute_split(0, &ifaces, &SplitStrategy::WeightedBandwidth, MB);
        assert_eq!(split.chunk_count(), 0);
    }

    #[test]
    fn split_with_empty_interface_list_returns_empty() {
        let split = compute_split(10 * MB, &[], &SplitStrategy::RoundRobin, MB);
        assert_eq!(split.chunk_count(), 0);
    }

    #[test]
    fn split_offsets_are_contiguous_and_cover_total() {
        let ifaces = vec![
            make_iface("a", 60, false, true),
            make_iface("b", 40, false, true),
        ];
        let split = compute_split(100 * MB, &ifaces, &SplitStrategy::WeightedBandwidth, MB);

        // Verify chunks are contiguous.
        let mut expected_offset = 0u64;
        for chunk in &split.chunks {
            assert_eq!(chunk.offset, expected_offset);
            expected_offset += chunk.length;
        }
        assert_eq!(expected_offset, 100 * MB);
    }

    #[test]
    fn transfer_ids_are_unique() {
        let ifaces = vec![make_iface("eth0", 100, false, true)];
        let s1 = compute_split(MB, &ifaces, &SplitStrategy::RoundRobin, MB);
        let s2 = compute_split(MB, &ifaces, &SplitStrategy::RoundRobin, MB);
        assert_ne!(s1.transfer_id, s2.transfer_id);
    }

    // ===== ReassemblyBuffer =====

    #[test]
    fn reassembly_starts_empty() {
        let buf = ReassemblyBuffer::new(1, 5);
        assert!(!buf.is_complete());
        assert_eq!(buf.missing_chunks().len(), 5);
        assert!((buf.completion_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reassembly_tracks_received_chunks() {
        let mut buf = ReassemblyBuffer::new(1, 3);
        assert!(buf.record_chunk(0));
        assert!(buf.record_chunk(2));

        assert!(!buf.is_complete());
        assert_eq!(buf.missing_chunks(), vec![1]);
        assert!((buf.completion_percent() - 66.666_666_666_666_66).abs() < 0.01);
    }

    #[test]
    fn reassembly_complete_when_all_received() {
        let mut buf = ReassemblyBuffer::new(1, 2);
        buf.record_chunk(0);
        buf.record_chunk(1);
        assert!(buf.is_complete());
        assert!((buf.completion_percent() - 100.0).abs() < f64::EPSILON);
        assert!(buf.missing_chunks().is_empty());
    }

    #[test]
    fn reassembly_out_of_range_returns_false() {
        let mut buf = ReassemblyBuffer::new(1, 2);
        assert!(!buf.record_chunk(5));
    }

    #[test]
    fn reassembly_zero_chunks_is_complete() {
        let buf = ReassemblyBuffer::new(1, 0);
        assert!(buf.is_complete());
        assert!((buf.completion_percent() - 100.0).abs() < f64::EPSILON);
    }

    // ===== MultipathConfig =====

    #[test]
    fn config_defaults_are_correct() {
        let cfg = MultipathConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.min_chunk_bytes, 1_048_576);
        assert_eq!(cfg.strategy, SplitStrategy::WeightedBandwidth);
        assert_eq!(cfg.split_threshold_bytes, 1_048_576);
        assert!(!cfg.enable_redundant_small_transfers);
    }

    // ===== TransferSplit =====

    #[test]
    fn transfer_split_is_complete_matches_total() {
        let split = TransferSplit {
            transfer_id: 1,
            total_bytes: 100,
            chunks: vec![
                SplitChunk {
                    chunk_id: 0,
                    sequence_number: 0,
                    offset: 0,
                    length: 60,
                    interface_name: "a".into(),
                    sent: false,
                    acked: false,
                },
                SplitChunk {
                    chunk_id: 1,
                    sequence_number: 1,
                    offset: 60,
                    length: 40,
                    interface_name: "b".into(),
                    sent: false,
                    acked: false,
                },
            ],
        };
        assert!(split.is_complete());
        assert_eq!(split.chunk_count(), 2);
        assert_eq!(split.total_assigned_bytes(), 100);
    }

    #[test]
    fn transfer_split_incomplete_when_bytes_mismatch() {
        let split = TransferSplit {
            transfer_id: 1,
            total_bytes: 100,
            chunks: vec![SplitChunk {
                chunk_id: 0,
                sequence_number: 0,
                offset: 0,
                length: 50,
                interface_name: "a".into(),
                sent: false,
                acked: false,
            }],
        };
        assert!(!split.is_complete());
    }
}
