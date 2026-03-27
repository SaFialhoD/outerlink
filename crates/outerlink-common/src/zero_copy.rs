//! Zero-copy transfer path types and routing logic.
//!
//! Defines the transfer modes available in OuterLink's data plane, from
//! standard kernel-buffered copies through MSG_ZEROCOPY, io_uring registered
//! buffers, and full OpenDMA RDMA bypass.  This module contains NO actual
//! socket, io_uring, or DMA code -- only types, configuration, and the
//! routing function that selects the best transfer mode for a given transfer.
//!
//! # Transfer Mode Progression
//!
//! | Phase | Mode              | Kernel copies | CPU involved? |
//! |-------|-------------------|---------------|---------------|
//! | 1.0   | Standard          | 4             | yes           |
//! | 1.5   | MsgZeroCopy       | 2             | yes           |
//! | 1.7   | IoUringRegistered | 1             | yes           |
//! | 5.0   | OpenDma           | 0             | no            |
//!
//! # Safety notes
//!
//! - `vmsplice` is intentionally excluded: it lacks completion notification
//!   and risks data corruption when the sender reuses buffers.
//! - `tokio-uring` is excluded: it is not stable and its futures are `!Send`.
//! - MSG_ZEROCOPY break-even is ~10 KB; we default the threshold to 16 KB to
//!   leave headroom.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TransferMode
// ---------------------------------------------------------------------------

/// The data-plane transfer mode for a single bulk transfer.
///
/// Each variant corresponds to a phase in OuterLink's transport roadmap.
/// Higher variants require more infrastructure but yield fewer copies and
/// lower CPU usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferMode {
    /// Standard TCP send/recv with kernel-managed buffers.
    /// Data is copied: app -> kernel send buf -> wire -> kernel recv buf -> app.
    Standard,

    /// Linux `MSG_ZEROCOPY` flag on `sendmsg(2)`.  The kernel pins the
    /// user-space send buffer and DMAs directly from it, eliminating the
    /// send-side kernel copy.  Requires source buffer to be pinned (CUDA
    /// host-pinned or mlock'd).  Break-even is ~10 KB; we use 16 KB default.
    MsgZeroCopy,

    /// `io_uring` with pre-registered, pre-pinned buffers.  Both send and
    /// receive avoid extra copies because the kernel already knows the buffer
    /// physical addresses.  Requires both source and destination buffers to
    /// be registered.
    IoUringRegistered,

    /// Full RDMA zero-copy via OpenDMA.  The NIC DMA engines read/write GPU
    /// VRAM directly through PCIe BAR1 -- zero kernel copies, zero CPU
    /// involvement.  Requires OpenDMA kernel module on both nodes.
    OpenDma,
}

impl TransferMode {
    /// Total end-to-end copies in the full host-staged pipeline
    /// (kernel network-stack + CUDA D2H/H2D). Per R39 spec:
    ///
    /// Standard:          4 (ksend-copy + krecv-copy + cudaD2H + cudaH2D)
    /// MsgZeroCopy:       3 (krecv-copy + cudaD2H + cudaH2D; send side DMA-direct)
    /// IoUringRegistered: 2 (cudaD2H + cudaH2D; both NIC copies eliminated)
    /// OpenDma:           0 (NIC DMA engines read/write GPU VRAM directly)
    pub fn copies_required(&self) -> u32 {
        match self {
            Self::Standard => 4,
            Self::MsgZeroCopy => 3,
            Self::IoUringRegistered => 2,
            Self::OpenDma => 0,
        }
    }

    /// Whether the host CPU is on the data path for this mode.
    ///
    /// Only `OpenDma` fully bypasses the CPU; all other modes require the
    /// CPU to initiate or complete the transfer.
    pub fn cpu_involved(&self) -> bool {
        !matches!(self, Self::OpenDma)
    }
}

// ---------------------------------------------------------------------------
// ZeroCopyConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the zero-copy transfer path selection.
///
/// These are set once at startup (from config file / CLI) and consulted by
/// [`select_transfer_mode`] on every bulk transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyConfig {
    /// Minimum payload size (bytes) before we consider zero-copy modes.
    /// Below this threshold, standard memcpy is faster due to completion
    /// notification overhead.  Default: 16384 (16 KB).
    pub min_zerocopy_bytes: usize,

    /// Enable `MSG_ZEROCOPY` on send sockets.  Requires Linux >= 4.14 and
    /// pinned source buffers.
    pub enable_msg_zerocopy: bool,

    /// Enable `io_uring` with registered buffers.  Requires Linux >= 5.6
    /// and both source and destination buffers to be pre-registered.
    pub enable_iouring: bool,

    /// Maximum number of buffers that can be registered with `io_uring`.
    /// Each registered buffer occupies a slot in the kernel's fixed-buffer
    /// table.  Default: 1024.
    pub max_registered_buffers: u32,

    /// Timeout (ms) waiting for a MSG_ZEROCOPY completion notification
    /// before falling back to Standard mode.  Default: 5000 ms.
    pub completion_timeout_ms: u64,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            min_zerocopy_bytes: 16_384,
            enable_msg_zerocopy: true,
            enable_iouring: false,
            max_registered_buffers: 1024,
            completion_timeout_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// TransferPath
// ---------------------------------------------------------------------------

/// Describes a concrete transfer: the requested mode plus the buffer state.
///
/// [`effective_mode`](TransferPath::effective_mode) may downgrade the mode
/// if the buffer pinning requirements are not met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferPath {
    /// The requested (or selected) transfer mode.
    pub mode: TransferMode,

    /// Whether the source buffer is pinned (CUDA host-pinned or mlock'd).
    pub src_pinned: bool,

    /// Whether the destination buffer is pinned.
    pub dst_pinned: bool,

    /// Transfer size in bytes.
    pub bytes: u64,
}

impl TransferPath {
    /// The actual mode that will be used, after downgrading based on buffer
    /// pinning requirements.
    ///
    /// Downgrade rules:
    /// - `OpenDma` requires both pinned -> downgrades to `IoUringRegistered`
    ///   if possible, else `MsgZeroCopy` if src pinned, else `Standard`.
    /// - `IoUringRegistered` requires both pinned -> downgrades to
    ///   `MsgZeroCopy` if src pinned, else `Standard`.
    /// - `MsgZeroCopy` requires src pinned -> downgrades to `Standard`.
    /// - `Standard` never downgrades.
    pub fn effective_mode(&self) -> TransferMode {
        match self.mode {
            TransferMode::OpenDma => {
                if self.src_pinned && self.dst_pinned {
                    TransferMode::OpenDma
                } else if self.src_pinned {
                    TransferMode::MsgZeroCopy
                } else {
                    TransferMode::Standard
                }
            }
            TransferMode::IoUringRegistered => {
                if self.src_pinned && self.dst_pinned {
                    TransferMode::IoUringRegistered
                } else if self.src_pinned {
                    TransferMode::MsgZeroCopy
                } else {
                    TransferMode::Standard
                }
            }
            TransferMode::MsgZeroCopy => {
                if self.src_pinned {
                    TransferMode::MsgZeroCopy
                } else {
                    TransferMode::Standard
                }
            }
            TransferMode::Standard => TransferMode::Standard,
        }
    }
}

// ---------------------------------------------------------------------------
// select_transfer_mode
// ---------------------------------------------------------------------------

/// Select the best transfer mode for the given conditions.
///
/// This is the main routing function called on every bulk transfer.  It
/// considers the configuration, payload size, buffer pinning, and whether
/// RDMA is available to pick the highest-performance mode that is safe to
/// use.
///
/// # Precedence (highest to lowest)
///
/// 1. **OpenDma** -- if `rdma_available` and both buffers pinned
/// 2. **IoUringRegistered** -- if enabled and both buffers pinned
/// 3. **MsgZeroCopy** -- if enabled and source buffer pinned
/// 4. **Standard** -- always available as fallback
///
/// Any mode is skipped if the payload is below `min_zerocopy_bytes`
/// (except Standard, which is always valid).
pub fn select_transfer_mode(
    config: &ZeroCopyConfig,
    bytes: usize,
    src_pinned: bool,
    dst_pinned: bool,
    rdma_available: bool,
) -> TransferMode {
    // Small transfers: standard is faster due to completion overhead.
    if bytes < config.min_zerocopy_bytes {
        return TransferMode::Standard;
    }

    // Neither side pinned: can't use any zero-copy mode.
    if !src_pinned && !dst_pinned {
        return TransferMode::Standard;
    }

    // Try modes from best to worst.
    if rdma_available && src_pinned && dst_pinned {
        return TransferMode::OpenDma;
    }

    if config.enable_iouring && src_pinned && dst_pinned {
        return TransferMode::IoUringRegistered;
    }

    if config.enable_msg_zerocopy && src_pinned {
        return TransferMode::MsgZeroCopy;
    }

    TransferMode::Standard
}

// ---------------------------------------------------------------------------
// CompletionTracker
// ---------------------------------------------------------------------------

/// Tracks in-flight MSG_ZEROCOPY completions.
///
/// MSG_ZEROCOPY requires polling `recvmsg(MSG_ERRQUEUE)` to learn when the
/// kernel is done with a buffer.  This tracker enforces a maximum number of
/// in-flight (pending) completions to prevent unbounded memory pinning.
#[derive(Debug)]
pub struct CompletionTracker {
    /// Number of sends whose completion notification has not yet arrived.
    pending_completions: u32,

    /// Maximum allowed pending completions before we block new submissions.
    max_pending: u32,

    /// Lifetime count of completed transfers.
    total_completed: u64,

    /// Lifetime count of bytes transferred (completed only).
    total_bytes: u64,
}

impl CompletionTracker {
    /// Create a new tracker with the given maximum pending count.
    pub fn new(max_pending: u32) -> Self {
        Self {
            pending_completions: 0,
            max_pending,
            total_completed: 0,
            total_bytes: 0,
        }
    }

    /// Whether a new send can be submitted without exceeding the pending
    /// limit.
    pub fn can_submit(&self) -> bool {
        self.pending_completions < self.max_pending
    }

    /// Record a new send submission.  Returns `false` if the tracker is
    /// full (caller should wait for completions first).
    pub fn submit(&mut self) -> bool {
        if !self.can_submit() {
            return false;
        }
        self.pending_completions += 1;
        true
    }

    /// Record that a completion notification arrived for `bytes` transferred.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if called with no pending completions.
    pub fn complete(&mut self, bytes: u64) {
        debug_assert!(
            self.pending_completions > 0,
            "complete() called with no pending completions"
        );
        self.pending_completions = self.pending_completions.saturating_sub(1);
        self.total_completed += 1;
        self.total_bytes += bytes;
    }

    /// Whether all submitted sends have been completed.
    pub fn is_drained(&self) -> bool {
        self.pending_completions == 0
    }

    /// Current number of pending (in-flight) completions.
    pub fn pending(&self) -> u32 {
        self.pending_completions
    }

    /// Lifetime count of completed transfers.
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Lifetime bytes transferred.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

// ---------------------------------------------------------------------------
// SocketTuning
// ---------------------------------------------------------------------------

/// Socket-level tuning parameters optimized for high-bandwidth LAN transfers.
///
/// These values are designed for 100 GbE (ConnectX-5/6) links where the
/// default kernel socket buffers (128-256 KB) are far too small to fill the
/// bandwidth-delay product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketTuning {
    /// `SO_SNDBUF` size in bytes.  64 MB keeps the TCP window open on
    /// 100 GbE links with ~1 ms RTT (BDP = 12.5 GB/s * 0.001 = 12.5 MB,
    /// so 64 MB provides ample headroom for bursts).
    pub send_buffer_bytes: usize,

    /// `SO_RCVBUF` size in bytes.  Matched to send buffer to avoid
    /// asymmetric window scaling issues.
    pub recv_buffer_bytes: usize,

    /// `TCP_NODELAY` -- disable Nagle's algorithm.  Always true for
    /// latency-sensitive GPU transfers; we handle our own batching.
    pub tcp_nodelay: bool,

    /// `TCP_QUICKACK` -- disable delayed ACKs.  True for LAN where the
    /// 40 ms delayed-ACK timer wastes bandwidth.  May be false for WAN
    /// to reduce ACK traffic.
    pub tcp_quickack: bool,

    /// `TCP_CORK` -- accumulate small writes into full MSS segments.
    /// False by default; we send large aligned buffers and don't benefit
    /// from corking.
    pub cork: bool,
}

impl Default for SocketTuning {
    fn default() -> Self {
        Self {
            send_buffer_bytes: 64 * 1024 * 1024, // 64 MB
            recv_buffer_bytes: 64 * 1024 * 1024, // 64 MB
            tcp_nodelay: true,
            tcp_quickack: true,
            cork: false,
        }
    }
}

// ---------------------------------------------------------------------------
// NicTuning
// ---------------------------------------------------------------------------

/// NIC-level tuning parameters for ConnectX-5/6 and similar high-perf NICs.
///
/// These settings are typically applied via `ethtool` at system startup.
/// This struct records the expected/desired state so the OuterLink daemon
/// can verify or log mismatches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NicTuning {
    /// Ring buffer size (rx/tx descriptors).  8192 is the ConnectX-5 max
    /// and prevents ring overflows at 100 Gbps line rate.
    pub ring_buffer_size: u32,

    /// Maximum Transmission Unit in bytes.  9000 = jumbo frames, which
    /// cut per-packet overhead by ~6x vs 1500-byte standard frames.
    /// Requires all switches in the path to support jumbo.
    pub mtu: u32,

    /// TCP Segmentation Offload -- the NIC splits large sends into MSS
    /// segments, saving CPU cycles.
    pub tso_enabled: bool,

    /// Generic Receive Offload -- the NIC coalesces small received frames
    /// into large buffers before handing them to the kernel.
    pub gro_enabled: bool,

    /// Large Receive Offload -- disabled because LRO is incompatible with
    /// IP forwarding/routing and can cause issues with RDMA.  GRO is the
    /// modern replacement.
    pub lro_disabled: bool,

    /// NUMA node the NIC is attached to.  For optimal performance, memory
    /// allocations and CPU affinity should match this node.  `None` means
    /// the node is unknown or the system is non-NUMA.
    pub numa_node: Option<u32>,
}

impl Default for NicTuning {
    fn default() -> Self {
        Self {
            ring_buffer_size: 8192,
            mtu: 9000,
            tso_enabled: true,
            gro_enabled: true,
            lro_disabled: true,
            numa_node: None,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- TransferMode -------------------------------------------------------

    #[test]
    fn transfer_mode_copies_standard() {
        assert_eq!(TransferMode::Standard.copies_required(), 4);
    }

    #[test]
    fn transfer_mode_copies_msg_zerocopy() {
        assert_eq!(TransferMode::MsgZeroCopy.copies_required(), 3);
    }

    #[test]
    fn transfer_mode_copies_iouring() {
        assert_eq!(TransferMode::IoUringRegistered.copies_required(), 2);
    }

    #[test]
    fn transfer_mode_copies_opendma() {
        assert_eq!(TransferMode::OpenDma.copies_required(), 0);
    }

    #[test]
    fn transfer_mode_cpu_involved() {
        assert!(TransferMode::Standard.cpu_involved());
        assert!(TransferMode::MsgZeroCopy.cpu_involved());
        assert!(TransferMode::IoUringRegistered.cpu_involved());
        assert!(!TransferMode::OpenDma.cpu_involved());
    }

    // -- ZeroCopyConfig defaults --------------------------------------------

    #[test]
    fn config_default_min_bytes() {
        let cfg = ZeroCopyConfig::default();
        assert_eq!(cfg.min_zerocopy_bytes, 16_384);
    }

    #[test]
    fn config_default_flags() {
        let cfg = ZeroCopyConfig::default();
        assert!(cfg.enable_msg_zerocopy);
        assert!(!cfg.enable_iouring);
        assert_eq!(cfg.max_registered_buffers, 1024);
        assert_eq!(cfg.completion_timeout_ms, 5000);
    }

    // -- select_transfer_mode routing ---------------------------------------

    #[test]
    fn select_small_payload_always_standard() {
        let cfg = ZeroCopyConfig::default();
        // Even with everything enabled and pinned, small payloads use Standard.
        assert_eq!(
            select_transfer_mode(&cfg, 100, true, true, true),
            TransferMode::Standard,
        );
    }

    #[test]
    fn select_below_threshold_boundary() {
        let cfg = ZeroCopyConfig::default();
        // Exactly at threshold - 1 should still be Standard.
        assert_eq!(
            select_transfer_mode(&cfg, 16_383, true, true, false),
            TransferMode::Standard,
        );
    }

    #[test]
    fn select_at_threshold_with_msg_zerocopy() {
        let cfg = ZeroCopyConfig::default();
        assert_eq!(
            select_transfer_mode(&cfg, 16_384, true, false, false),
            TransferMode::MsgZeroCopy,
        );
    }

    #[test]
    fn select_neither_pinned_always_standard() {
        let cfg = ZeroCopyConfig {
            enable_msg_zerocopy: true,
            enable_iouring: true,
            ..Default::default()
        };
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, false, false, true),
            TransferMode::Standard,
        );
    }

    #[test]
    fn select_rdma_both_pinned() {
        let cfg = ZeroCopyConfig::default();
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, true, true, true),
            TransferMode::OpenDma,
        );
    }

    #[test]
    fn select_iouring_both_pinned() {
        let cfg = ZeroCopyConfig {
            enable_iouring: true,
            ..Default::default()
        };
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, true, true, false),
            TransferMode::IoUringRegistered,
        );
    }

    #[test]
    fn select_msg_zerocopy_src_pinned_only() {
        let cfg = ZeroCopyConfig {
            enable_iouring: true,
            ..Default::default()
        };
        // dst not pinned -> can't use io_uring, falls to MsgZeroCopy.
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, true, false, false),
            TransferMode::MsgZeroCopy,
        );
    }

    #[test]
    fn select_dst_pinned_only_standard() {
        let cfg = ZeroCopyConfig::default();
        // Only dst pinned, src not -> can't use MsgZeroCopy either.
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, false, true, false),
            TransferMode::Standard,
        );
    }

    #[test]
    fn select_all_disabled_large_pinned() {
        let cfg = ZeroCopyConfig {
            enable_msg_zerocopy: false,
            enable_iouring: false,
            ..Default::default()
        };
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, true, true, false),
            TransferMode::Standard,
        );
    }

    #[test]
    fn select_rdma_overrides_iouring() {
        let cfg = ZeroCopyConfig {
            enable_iouring: true,
            ..Default::default()
        };
        // RDMA available + io_uring enabled -> RDMA wins.
        assert_eq!(
            select_transfer_mode(&cfg, 1_000_000, true, true, true),
            TransferMode::OpenDma,
        );
    }

    // -- CompletionTracker --------------------------------------------------

    #[test]
    fn tracker_new_is_drained() {
        let t = CompletionTracker::new(10);
        assert!(t.is_drained());
        assert!(t.can_submit());
        assert_eq!(t.pending(), 0);
    }

    #[test]
    fn tracker_submit_and_complete() {
        let mut t = CompletionTracker::new(2);
        assert!(t.submit());
        assert!(t.submit());
        assert!(!t.can_submit());
        assert!(!t.submit()); // full

        t.complete(1024);
        assert!(t.can_submit());
        assert_eq!(t.pending(), 1);
        assert_eq!(t.total_completed(), 1);
        assert_eq!(t.total_bytes(), 1024);

        t.complete(2048);
        assert!(t.is_drained());
        assert_eq!(t.total_completed(), 2);
        assert_eq!(t.total_bytes(), 3072);
    }

    #[test]
    fn tracker_submit_returns_false_when_full() {
        let mut t = CompletionTracker::new(1);
        assert!(t.submit());
        assert!(!t.submit());
    }

    // -- SocketTuning defaults ----------------------------------------------

    #[test]
    fn socket_tuning_defaults() {
        let s = SocketTuning::default();
        assert_eq!(s.send_buffer_bytes, 64 * 1024 * 1024);
        assert_eq!(s.recv_buffer_bytes, 64 * 1024 * 1024);
        assert!(s.tcp_nodelay);
        assert!(s.tcp_quickack);
        assert!(!s.cork);
    }

    // -- NicTuning defaults -------------------------------------------------

    #[test]
    fn nic_tuning_defaults() {
        let n = NicTuning::default();
        assert_eq!(n.ring_buffer_size, 8192);
        assert_eq!(n.mtu, 9000);
        assert!(n.tso_enabled);
        assert!(n.gro_enabled);
        assert!(n.lro_disabled);
        assert_eq!(n.numa_node, None);
    }

    // -- TransferPath effective_mode ----------------------------------------

    #[test]
    fn effective_mode_opendma_both_pinned() {
        let p = TransferPath {
            mode: TransferMode::OpenDma,
            src_pinned: true,
            dst_pinned: true,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::OpenDma);
    }

    #[test]
    fn effective_mode_opendma_src_only_downgrades() {
        let p = TransferPath {
            mode: TransferMode::OpenDma,
            src_pinned: true,
            dst_pinned: false,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::MsgZeroCopy);
    }

    #[test]
    fn effective_mode_opendma_neither_pinned() {
        let p = TransferPath {
            mode: TransferMode::OpenDma,
            src_pinned: false,
            dst_pinned: false,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::Standard);
    }

    #[test]
    fn effective_mode_iouring_src_only_downgrades() {
        let p = TransferPath {
            mode: TransferMode::IoUringRegistered,
            src_pinned: true,
            dst_pinned: false,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::MsgZeroCopy);
    }

    #[test]
    fn effective_mode_iouring_neither_downgrades() {
        let p = TransferPath {
            mode: TransferMode::IoUringRegistered,
            src_pinned: false,
            dst_pinned: false,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::Standard);
    }

    #[test]
    fn effective_mode_msg_zerocopy_not_pinned() {
        let p = TransferPath {
            mode: TransferMode::MsgZeroCopy,
            src_pinned: false,
            dst_pinned: true,
            bytes: 1_000_000,
        };
        assert_eq!(p.effective_mode(), TransferMode::Standard);
    }

    #[test]
    fn effective_mode_standard_never_downgrades() {
        let p = TransferPath {
            mode: TransferMode::Standard,
            src_pinned: false,
            dst_pinned: false,
            bytes: 100,
        };
        assert_eq!(p.effective_mode(), TransferMode::Standard);
    }
}
