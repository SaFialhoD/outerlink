//! RDMA transport types for OuterLink.
//!
//! Pure type definitions for ibverbs-based RDMA connection management,
//! queue pair lifecycle, and memory registration. No ibverbs calls --
//! these are data structures exchanged between nodes during connection
//! setup and used by the future RDMA transport backend.

use serde::{Deserialize, Serialize};

// ── MTU sizes ───────────────────────────────────────────────────────

/// InfiniBand MTU sizes supported by the RDMA transport.
///
/// These correspond to the `ibv_mtu` enum in libibverbs and determine
/// the maximum payload per RDMA packet on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MtuSize {
    Mtu256,
    Mtu512,
    Mtu1024,
    Mtu2048,
    Mtu4096,
}

impl MtuSize {
    /// Returns the MTU size in bytes.
    pub fn bytes(&self) -> u32 {
        match self {
            MtuSize::Mtu256 => 256,
            MtuSize::Mtu512 => 512,
            MtuSize::Mtu1024 => 1024,
            MtuSize::Mtu2048 => 2048,
            MtuSize::Mtu4096 => 4096,
        }
    }

    /// Construct from a byte size. Returns `None` for non-standard sizes.
    pub fn from_bytes(n: u32) -> Option<Self> {
        match n {
            256 => Some(MtuSize::Mtu256),
            512 => Some(MtuSize::Mtu512),
            1024 => Some(MtuSize::Mtu1024),
            2048 => Some(MtuSize::Mtu2048),
            4096 => Some(MtuSize::Mtu4096),
            _ => None,
        }
    }
}

// ── Queue Pair state machine ────────────────────────────────────────

/// InfiniBand Queue Pair states.
///
/// QPs transition through these states during connection setup:
/// Reset -> Init -> ReadyToReceive -> ReadyToSend.
/// Any state can transition to Error on failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueuePairState {
    /// QP just created or reset. Cannot send or receive.
    Reset,
    /// QP initialized with port and access flags. Cannot send or receive yet.
    Init,
    /// QP can receive but not send. Remote QP info has been applied.
    ReadyToReceive,
    /// QP is fully connected. Can send and receive.
    ReadyToSend,
    /// QP encountered an error. Must be reset to recover.
    Error,
}

impl QueuePairState {
    /// Whether the QP can post send work requests in this state.
    pub fn can_send(&self) -> bool {
        matches!(self, QueuePairState::ReadyToSend)
    }

    /// Whether the QP can post receive work requests in this state.
    ///
    /// Both ReadyToReceive and ReadyToSend allow posting receives.
    pub fn can_receive(&self) -> bool {
        matches!(
            self,
            QueuePairState::ReadyToReceive | QueuePairState::ReadyToSend
        )
    }

    /// Returns the next state in the normal QP lifecycle, or `None`
    /// if the QP is already at the terminal state (ReadyToSend) or
    /// in an error state.
    pub fn next_state(&self) -> Option<Self> {
        match self {
            QueuePairState::Reset => Some(QueuePairState::Init),
            QueuePairState::Init => Some(QueuePairState::ReadyToReceive),
            QueuePairState::ReadyToReceive => Some(QueuePairState::ReadyToSend),
            QueuePairState::ReadyToSend => None,
            QueuePairState::Error => None,
        }
    }
}

// ── RDMA device info ────────────────────────────────────────────────

/// Capabilities of an RDMA NIC (HCA) discovered at startup.
///
/// Populated from `ibv_query_device` / `ibv_query_port` results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdmaDeviceInfo {
    /// Device name as reported by the kernel (e.g. "mlx5_0").
    pub device_name: String,
    /// Physical port number (1-based, typically 1).
    pub port: u8,
    /// GID table index to use for RoCEv2.
    pub gid_index: u32,
    /// Maximum number of Queue Pairs the device supports.
    pub max_qp: u32,
    /// Maximum number of Completion Queues.
    pub max_cq: u32,
    /// Maximum number of Memory Regions.
    pub max_mr: u32,
    /// Maximum inline data size for send work requests (bytes).
    pub max_inline_data: u32,
    /// Active MTU on the port.
    pub active_mtu: MtuSize,
}

// ── Queue Pair config ───────────────────────────────────────────────

/// Configuration for creating a Queue Pair.
///
/// Defaults are tuned for OuterLink's GPU memory transfer workload:
/// moderate queue depth, small SGE count (most transfers are single-buffer),
/// and CQ moderation to reduce interrupt overhead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuePairConfig {
    /// Maximum outstanding send work requests.
    pub max_send_wr: u32,
    /// Maximum outstanding receive work requests.
    pub max_recv_wr: u32,
    /// Maximum scatter-gather entries per send WR.
    pub max_send_sge: u32,
    /// Maximum scatter-gather entries per receive WR.
    pub max_recv_sge: u32,
    /// Maximum inline data bytes per send WR.
    pub max_inline_data: u32,
    /// If true, every send WR generates a completion. If false, only
    /// WRs with the IBV_SEND_SIGNALED flag generate completions.
    pub sq_sig_all: bool,
    /// Number of completions to coalesce before generating an event.
    pub cq_moderation_count: u16,
    /// Maximum time in microseconds before generating a CQ event.
    pub cq_moderation_period_us: u16,
}

impl Default for QueuePairConfig {
    fn default() -> Self {
        Self {
            max_send_wr: 256,
            max_recv_wr: 256,
            max_send_sge: 4,
            max_recv_sge: 4,
            max_inline_data: 220,
            sq_sig_all: false,
            cq_moderation_count: 100,
            cq_moderation_period_us: 50,
        }
    }
}

// ── Memory region info ──────────────────────────────────────────────

/// Descriptor for a registered RDMA memory region.
///
/// Exchanged between nodes so that one side can perform RDMA read/write
/// operations targeting the other side's registered memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRegionInfo {
    /// Virtual address of the registered buffer.
    pub addr: u64,
    /// Length of the registered region in bytes.
    pub length: u64,
    /// Local key (used by the registering side for local access).
    pub lkey: u32,
    /// Remote key (shared with the remote side for RDMA access).
    pub rkey: u32,
    /// Whether this region backs GPU VRAM (true) or host RAM (false).
    pub is_gpu_memory: bool,
}

// ── RDMA connection ─────────────────────────────────────────────────

/// Represents one RDMA connection (QP pair) between two nodes.
///
/// Both sides exchange their QP number, LID, and GID during connection
/// setup so that each can transition its QP to ReadyToSend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdmaConnection {
    /// Local Queue Pair number.
    pub local_qp_num: u32,
    /// Remote Queue Pair number.
    pub remote_qp_num: u32,
    /// Local LID (InfiniBand) or 0 for RoCE.
    pub local_lid: u16,
    /// Remote LID (InfiniBand) or 0 for RoCE.
    pub remote_lid: u16,
    /// Local GID as a hex string (for RoCEv2).
    pub local_gid: String,
    /// Remote GID as a hex string (for RoCEv2).
    pub remote_gid: String,
    /// Current QP state.
    pub state: QueuePairState,
    /// Negotiated MTU for this connection.
    pub mtu: MtuSize,
}

impl RdmaConnection {
    /// Whether this connection is ready to transfer data.
    pub fn is_ready(&self) -> bool {
        self.state == QueuePairState::ReadyToSend
    }

    /// Whether the connection is in an error state.
    pub fn is_error(&self) -> bool {
        self.state == QueuePairState::Error
    }
}

// ── RDMA pool config ────────────────────────────────────────────────

/// Configuration for the pool of RDMA connections to a single peer.
///
/// Multiple QPs per peer allow concurrent transfers without head-of-line
/// blocking, and shared receive queues reduce memory consumption.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdmaPoolConfig {
    /// Number of QP connections to maintain per peer node.
    pub connections_per_peer: u32,
    /// Completion Queue depth (shared across QPs to this peer).
    pub cq_size: u32,
    /// Maximum in-flight send work requests across all QPs to this peer.
    pub max_outstanding_sends: u32,
    /// Maximum in-flight receive work requests across all QPs to this peer.
    pub max_outstanding_recvs: u32,
    /// Transfers smaller than this (bytes) use inline send.
    pub inline_threshold: u32,
    /// Whether to use a Shared Receive Queue across QPs.
    pub use_shared_receive_queue: bool,
}

impl Default for RdmaPoolConfig {
    fn default() -> Self {
        Self {
            connections_per_peer: 4,
            cq_size: 4096,
            max_outstanding_sends: 1024,
            max_outstanding_recvs: 1024,
            inline_threshold: 220,
            use_shared_receive_queue: true,
        }
    }
}

// ── Work request type ───────────────────────────────────────────────

/// RDMA work request (operation) types.
///
/// Two-sided operations (Send/Recv) require cooperation from the remote
/// side. One-sided operations (RdmaWrite/RdmaRead) access remote memory
/// directly using the remote key, without remote CPU involvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkRequestType {
    /// Two-sided send. Remote must have a matching Recv posted.
    Send,
    /// Two-sided receive. Matches a remote Send.
    Recv,
    /// One-sided RDMA write to remote memory. No remote CPU involvement.
    RdmaWrite,
    /// One-sided RDMA read from remote memory. No remote CPU involvement.
    RdmaRead,
}

impl WorkRequestType {
    /// Returns `true` for one-sided RDMA operations (Write, Read)
    /// that bypass the remote CPU entirely.
    pub fn is_one_sided(&self) -> bool {
        matches!(self, WorkRequestType::RdmaWrite | WorkRequestType::RdmaRead)
    }
}

// ── Completion status ───────────────────────────────────────────────

/// Outcome of a completed RDMA work request.
///
/// Polled from a Completion Queue after a work request finishes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionStatus {
    /// Work request completed successfully.
    Success,
    /// Local HCA error (e.g. protection domain mismatch, invalid lkey).
    LocalError(String),
    /// Remote side reported an error (e.g. invalid rkey, access violation).
    RemoteError(String),
    /// Work request timed out waiting for completion.
    Timeout,
    /// Work request was flushed due to QP entering error state.
    Flushed,
}

impl CompletionStatus {
    /// Whether the completion indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, CompletionStatus::Success)
    }

    /// Whether the completion indicates any kind of error.
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MtuSize tests ───────────────────────────────────────────

    #[test]
    fn mtu_bytes_round_trip() {
        for mtu in [
            MtuSize::Mtu256,
            MtuSize::Mtu512,
            MtuSize::Mtu1024,
            MtuSize::Mtu2048,
            MtuSize::Mtu4096,
        ] {
            let bytes = mtu.bytes();
            assert_eq!(MtuSize::from_bytes(bytes), Some(mtu));
        }
    }

    #[test]
    fn mtu_bytes_values() {
        assert_eq!(MtuSize::Mtu256.bytes(), 256);
        assert_eq!(MtuSize::Mtu512.bytes(), 512);
        assert_eq!(MtuSize::Mtu1024.bytes(), 1024);
        assert_eq!(MtuSize::Mtu2048.bytes(), 2048);
        assert_eq!(MtuSize::Mtu4096.bytes(), 4096);
    }

    #[test]
    fn mtu_from_bytes_invalid() {
        assert_eq!(MtuSize::from_bytes(0), None);
        assert_eq!(MtuSize::from_bytes(100), None);
        assert_eq!(MtuSize::from_bytes(8192), None);
        assert_eq!(MtuSize::from_bytes(1025), None);
    }

    #[test]
    fn mtu_all_are_powers_of_two() {
        for mtu in [
            MtuSize::Mtu256,
            MtuSize::Mtu512,
            MtuSize::Mtu1024,
            MtuSize::Mtu2048,
            MtuSize::Mtu4096,
        ] {
            let b = mtu.bytes();
            assert!(b.is_power_of_two(), "{b} is not a power of two");
        }
    }

    // ── QueuePairState tests ────────────────────────────────────

    #[test]
    fn qp_state_can_send() {
        assert!(!QueuePairState::Reset.can_send());
        assert!(!QueuePairState::Init.can_send());
        assert!(!QueuePairState::ReadyToReceive.can_send());
        assert!(QueuePairState::ReadyToSend.can_send());
        assert!(!QueuePairState::Error.can_send());
    }

    #[test]
    fn qp_state_can_receive() {
        assert!(!QueuePairState::Reset.can_receive());
        assert!(!QueuePairState::Init.can_receive());
        assert!(QueuePairState::ReadyToReceive.can_receive());
        assert!(QueuePairState::ReadyToSend.can_receive());
        assert!(!QueuePairState::Error.can_receive());
    }

    #[test]
    fn qp_state_next_state_lifecycle() {
        let mut state = QueuePairState::Reset;
        let expected = [
            QueuePairState::Init,
            QueuePairState::ReadyToReceive,
            QueuePairState::ReadyToSend,
        ];
        for expected_next in &expected {
            let next = state.next_state().expect("should have a next state");
            assert_eq!(next, *expected_next);
            state = next;
        }
        // ReadyToSend is terminal
        assert_eq!(state.next_state(), None);
    }

    #[test]
    fn qp_state_error_is_terminal() {
        assert_eq!(QueuePairState::Error.next_state(), None);
    }

    #[test]
    fn qp_state_error_cannot_send_or_receive() {
        assert!(!QueuePairState::Error.can_send());
        assert!(!QueuePairState::Error.can_receive());
    }

    // ── QueuePairConfig tests ───────────────────────────────────

    #[test]
    fn qp_config_defaults() {
        let cfg = QueuePairConfig::default();
        assert_eq!(cfg.max_send_wr, 256);
        assert_eq!(cfg.max_recv_wr, 256);
        assert_eq!(cfg.max_send_sge, 4);
        assert_eq!(cfg.max_recv_sge, 4);
        assert_eq!(cfg.max_inline_data, 220);
        assert!(!cfg.sq_sig_all);
        assert_eq!(cfg.cq_moderation_count, 100);
        assert_eq!(cfg.cq_moderation_period_us, 50);
    }

    #[test]
    fn qp_config_custom() {
        let cfg = QueuePairConfig {
            max_send_wr: 512,
            max_recv_wr: 512,
            max_send_sge: 8,
            max_recv_sge: 8,
            max_inline_data: 512,
            sq_sig_all: true,
            cq_moderation_count: 128,
            cq_moderation_period_us: 100,
        };
        assert_eq!(cfg.max_send_wr, 512);
        assert!(cfg.sq_sig_all);
    }

    // ── MemoryRegionInfo tests ──────────────────────────────────

    #[test]
    fn memory_region_info_gpu_flag() {
        let gpu_mr = MemoryRegionInfo {
            addr: 0x7f00_0000_0000,
            length: 1024 * 1024,
            lkey: 0x1234,
            rkey: 0x5678,
            is_gpu_memory: true,
        };
        assert!(gpu_mr.is_gpu_memory);

        let host_mr = MemoryRegionInfo {
            addr: 0x7f00_0000_0000,
            length: 4096,
            lkey: 0xAAAA,
            rkey: 0xBBBB,
            is_gpu_memory: false,
        };
        assert!(!host_mr.is_gpu_memory);
    }

    #[test]
    fn memory_region_info_equality() {
        let mr1 = MemoryRegionInfo {
            addr: 100,
            length: 200,
            lkey: 1,
            rkey: 2,
            is_gpu_memory: false,
        };
        let mr2 = mr1;
        assert_eq!(mr1, mr2);

        let mr3 = MemoryRegionInfo {
            rkey: 99,
            ..mr1
        };
        assert_ne!(mr1, mr3);
    }

    // ── RdmaConnection tests ────────────────────────────────────

    fn make_connection(state: QueuePairState) -> RdmaConnection {
        RdmaConnection {
            local_qp_num: 100,
            remote_qp_num: 200,
            local_lid: 1,
            remote_lid: 2,
            local_gid: "fe80::1".to_string(),
            remote_gid: "fe80::2".to_string(),
            state,
            mtu: MtuSize::Mtu4096,
        }
    }

    #[test]
    fn connection_is_ready_only_when_rts() {
        assert!(!make_connection(QueuePairState::Reset).is_ready());
        assert!(!make_connection(QueuePairState::Init).is_ready());
        assert!(!make_connection(QueuePairState::ReadyToReceive).is_ready());
        assert!(make_connection(QueuePairState::ReadyToSend).is_ready());
        assert!(!make_connection(QueuePairState::Error).is_ready());
    }

    #[test]
    fn connection_is_error() {
        assert!(make_connection(QueuePairState::Error).is_error());
        assert!(!make_connection(QueuePairState::ReadyToSend).is_error());
    }

    #[test]
    fn connection_stores_gid_strings() {
        let conn = make_connection(QueuePairState::ReadyToSend);
        assert_eq!(conn.local_gid, "fe80::1");
        assert_eq!(conn.remote_gid, "fe80::2");
    }

    // ── RdmaPoolConfig tests ────────────────────────────────────

    #[test]
    fn pool_config_defaults() {
        let cfg = RdmaPoolConfig::default();
        assert_eq!(cfg.connections_per_peer, 4);
        assert_eq!(cfg.cq_size, 4096);
        assert_eq!(cfg.max_outstanding_sends, 1024);
        assert_eq!(cfg.max_outstanding_recvs, 1024);
        assert_eq!(cfg.inline_threshold, 220);
        assert!(cfg.use_shared_receive_queue);
    }

    #[test]
    fn pool_config_custom() {
        let cfg = RdmaPoolConfig {
            connections_per_peer: 8,
            cq_size: 8192,
            max_outstanding_sends: 2048,
            max_outstanding_recvs: 2048,
            inline_threshold: 512,
            use_shared_receive_queue: false,
        };
        assert_eq!(cfg.connections_per_peer, 8);
        assert!(!cfg.use_shared_receive_queue);
    }

    // ── WorkRequestType tests ───────────────────────────────────

    #[test]
    fn work_request_is_one_sided() {
        assert!(!WorkRequestType::Send.is_one_sided());
        assert!(!WorkRequestType::Recv.is_one_sided());
        assert!(WorkRequestType::RdmaWrite.is_one_sided());
        assert!(WorkRequestType::RdmaRead.is_one_sided());
    }

    #[test]
    fn work_request_two_sided_are_not_one_sided() {
        // Explicit inverse check for clarity
        for wr in [WorkRequestType::Send, WorkRequestType::Recv] {
            assert!(
                !wr.is_one_sided(),
                "{wr:?} should not be one-sided"
            );
        }
    }

    // ── CompletionStatus tests ──────────────────────────────────

    #[test]
    fn completion_success() {
        let s = CompletionStatus::Success;
        assert!(s.is_success());
        assert!(!s.is_error());
    }

    #[test]
    fn completion_errors() {
        let cases = [
            CompletionStatus::LocalError("bad lkey".into()),
            CompletionStatus::RemoteError("access violation".into()),
            CompletionStatus::Timeout,
            CompletionStatus::Flushed,
        ];
        for c in &cases {
            assert!(!c.is_success(), "{c:?} should not be success");
            assert!(c.is_error(), "{c:?} should be error");
        }
    }

    #[test]
    fn completion_error_messages_preserved() {
        let local = CompletionStatus::LocalError("pd mismatch".into());
        let remote = CompletionStatus::RemoteError("invalid rkey".into());
        match local {
            CompletionStatus::LocalError(msg) => assert_eq!(msg, "pd mismatch"),
            _ => panic!("wrong variant"),
        }
        match remote {
            CompletionStatus::RemoteError(msg) => assert_eq!(msg, "invalid rkey"),
            _ => panic!("wrong variant"),
        }
    }

    // ── RdmaDeviceInfo tests ────────────────────────────────────

    #[test]
    fn device_info_construction() {
        let info = RdmaDeviceInfo {
            device_name: "mlx5_0".to_string(),
            port: 1,
            gid_index: 3,
            max_qp: 65536,
            max_cq: 65536,
            max_mr: 16777216,
            max_inline_data: 220,
            active_mtu: MtuSize::Mtu4096,
        };
        assert_eq!(info.device_name, "mlx5_0");
        assert_eq!(info.port, 1);
        assert_eq!(info.active_mtu.bytes(), 4096);
    }

    #[test]
    fn device_info_clone_eq() {
        let info = RdmaDeviceInfo {
            device_name: "mlx5_1".to_string(),
            port: 2,
            gid_index: 0,
            max_qp: 1024,
            max_cq: 1024,
            max_mr: 1024,
            max_inline_data: 64,
            active_mtu: MtuSize::Mtu1024,
        };
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    // ── Serde round-trip tests ──────────────────────────────────

    #[test]
    fn serde_mtu_round_trip() {
        let mtu = MtuSize::Mtu4096;
        let bytes = bincode::serialize(&mtu).unwrap();
        let back: MtuSize = bincode::deserialize(&bytes).unwrap();
        assert_eq!(mtu, back);
    }

    #[test]
    fn serde_qp_config_round_trip() {
        let cfg = QueuePairConfig::default();
        let bytes = bincode::serialize(&cfg).unwrap();
        let back: QueuePairConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn serde_memory_region_round_trip() {
        let mr = MemoryRegionInfo {
            addr: 0xDEAD_BEEF_0000,
            length: 1 << 30, // 1 GiB
            lkey: 42,
            rkey: 43,
            is_gpu_memory: true,
        };
        let bytes = bincode::serialize(&mr).unwrap();
        let back: MemoryRegionInfo = bincode::deserialize(&bytes).unwrap();
        assert_eq!(mr, back);
    }

    #[test]
    fn serde_connection_round_trip() {
        let conn = make_connection(QueuePairState::ReadyToSend);
        let bytes = bincode::serialize(&conn).unwrap();
        let back: RdmaConnection = bincode::deserialize(&bytes).unwrap();
        assert_eq!(conn, back);
    }

    #[test]
    fn serde_completion_status_round_trip() {
        let statuses = [
            CompletionStatus::Success,
            CompletionStatus::LocalError("err".into()),
            CompletionStatus::RemoteError("remote err".into()),
            CompletionStatus::Timeout,
            CompletionStatus::Flushed,
        ];
        for s in &statuses {
            let bytes = bincode::serialize(s).unwrap();
            let back: CompletionStatus = bincode::deserialize(&bytes).unwrap();
            assert_eq!(s, &back);
        }
    }

    #[test]
    fn serde_pool_config_round_trip() {
        let cfg = RdmaPoolConfig::default();
        let bytes = bincode::serialize(&cfg).unwrap();
        let back: RdmaPoolConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(cfg, back);
    }
}
