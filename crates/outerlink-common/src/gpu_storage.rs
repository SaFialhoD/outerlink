//! R21: GPU Direct Storage Over Network
//!
//! Provides remote NVMe storage access for GPU VRAM, including host-staged
//! and NVMe-oF data paths, topology probing, pipeline state management,
//! batch read with scatter-gather support, and R10 tier driver integration.
//!
//! Hardware-dependent functionality (actual NVMe I/O, RDMA transfers, GPU DMA,
//! P2PDMA topology probing) is represented as traits with placeholder
//! implementations.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases (matching CUDA types used in the codebase)
// ---------------------------------------------------------------------------

/// Opaque CUDA device pointer (u64 address).
pub type CudaDevicePtr = u64;

/// Opaque CUDA stream handle.
pub type CudaStream = u64;

/// Opaque CUDA event handle.
pub type CudaEvent = u64;

/// PCI bus address (e.g., "0000:02:00.0").
pub type PciAddress = String;

// ---------------------------------------------------------------------------
// 1. Core Storage Types
// ---------------------------------------------------------------------------

/// Handle to a remote storage file/device.
#[derive(Debug)]
pub struct StorageHandle {
    /// Remote node hosting the NVMe.
    pub storage_node: NodeId,
    /// NVMe namespace identifier.
    pub namespace: StorageNamespace,
    /// Selected data path for this handle.
    pub path: StoragePath,
    /// Alignment requirements for direct DMA.
    pub alignment: AlignmentRequirements,
    /// Unique handle ID for tracking.
    #[allow(dead_code)]
    pub(crate) handle_id: u64,
}

/// NVMe namespace description.
#[derive(Debug, Clone)]
pub struct StorageNamespace {
    /// NVMe-oF NQN (NVMe Qualified Name).
    pub nqn: String,
    /// Namespace ID within the subsystem.
    pub nsid: u32,
    /// Total size in bytes.
    pub size: u64,
    /// Logical block size (typically 512 or 4096).
    pub block_size: u32,
}

/// Data path selection for storage operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoragePath {
    /// NVMe -> host RAM -> RDMA -> OpenDMA -> GPU
    /// Host RAM touched on sender only.
    HostStaged {
        /// Size of each pinned buffer in the double-buffer pipeline.
        chunk_size: usize,
    },
    /// NVMe -> ConnectX-5 P2P -> RDMA -> OpenDMA -> GPU
    /// Zero host RAM on either side.
    NvmeOfOffload,
    /// NVMe -> host RAM -> RDMA -> host RAM -> cudaMemcpy -> GPU
    /// Fallback: host RAM touched on both sides.
    FullHostStaged { chunk_size: usize },
}

/// Alignment requirements for direct I/O.
#[derive(Debug, Clone, Copy)]
pub struct AlignmentRequirements {
    /// File offset must be a multiple of this (bytes).
    pub offset_alignment: usize,
    /// Buffer address must be a multiple of this (bytes).
    pub buffer_alignment: usize,
    /// Transfer size should be a multiple of this (bytes).
    pub size_alignment: usize,
}

impl Default for AlignmentRequirements {
    fn default() -> Self {
        Self {
            offset_alignment: 4096,
            buffer_alignment: 4096,
            size_alignment: 512,
        }
    }
}

impl AlignmentRequirements {
    /// Check if a given offset, buffer, and size satisfy alignment requirements.
    pub fn check(
        &self,
        offset: u64,
        buffer: CudaDevicePtr,
        size: usize,
    ) -> Result<(), GpuStorageError> {
        if offset as usize % self.offset_alignment != 0 {
            return Err(GpuStorageError::AlignmentError(format!(
                "offset {} not aligned to {}",
                offset, self.offset_alignment
            )));
        }
        if buffer as usize % self.buffer_alignment != 0 {
            return Err(GpuStorageError::AlignmentError(format!(
                "buffer {:#x} not aligned to {}",
                buffer, self.buffer_alignment
            )));
        }
        if size % self.size_alignment != 0 {
            return Err(GpuStorageError::AlignmentError(format!(
                "size {} not aligned to {}",
                size, self.size_alignment
            )));
        }
        Ok(())
    }

    /// Align a size up to the next multiple of size_alignment.
    pub fn align_size_up(&self, size: usize) -> usize {
        let align = self.size_alignment;
        (size + align - 1) / align * align
    }
}

/// A single read request in a batch operation.
#[derive(Debug, Clone)]
pub struct StorageReadRequest {
    /// Destination GPU buffer (CUDA device pointer).
    pub gpu_buffer: CudaDevicePtr,
    /// Number of bytes to read.
    pub size: usize,
    /// Offset within the file/device to read from.
    pub file_offset: u64,
    /// Offset within the GPU buffer to write to.
    pub buf_offset: usize,
}

/// Asynchronous operation handle.
#[derive(Debug)]
pub struct StorageFuture {
    /// Completion state: 0 = pending, 1 = success, 2 = error.
    state: Arc<AtomicU8>,
    /// Bytes transferred (set on completion).
    bytes_transferred: Arc<AtomicUsize>,
    /// Error (if any).
    error: Arc<Mutex<Option<GpuStorageError>>>,
}

impl StorageFuture {
    /// Create a new pending future.
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(0)),
            bytes_transferred: Arc::new(AtomicUsize::new(0)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Block until the operation completes.
    pub fn wait(&self) -> Result<usize, GpuStorageError> {
        while self.state.load(Ordering::Acquire) == 0 {
            std::hint::spin_loop();
        }
        if let Ok(mut guard) = self.error.lock() {
            if let Some(err) = guard.take() {
                return Err(err);
            }
        }
        Ok(self.bytes_transferred.load(Ordering::Acquire))
    }

    /// Check if completed without blocking.
    pub fn is_complete(&self) -> bool {
        self.state.load(Ordering::Acquire) != 0
    }

    /// Complete the future successfully.
    pub fn complete(&self, bytes: usize) {
        self.bytes_transferred.store(bytes, Ordering::Release);
        self.state.store(1, Ordering::Release);
    }

    /// Complete the future with an error.
    pub fn fail(&self, error: GpuStorageError) {
        if let Ok(mut guard) = self.error.lock() {
            *guard = Some(error);
        }
        self.state.store(2, Ordering::Release);
    }
}

impl Default for StorageFuture {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 2. Storage Connection Types
// ---------------------------------------------------------------------------

/// Connection type to a storage node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageConnectionType {
    /// Standard RDMA connection (for host-staged transfers).
    Rdma,
    /// NVMe-oF initiator connection.
    NvmeOf {
        /// Whether target offload is active on the remote side.
        target_offload: bool,
    },
}

/// Double-buffer pipeline state for overlapping NVMe read and RDMA send.
#[derive(Debug)]
pub struct PipelineState {
    /// Which buffer is currently being filled by NVMe read (0 or 1).
    pub read_idx: AtomicUsize,
    /// Which buffer is currently being sent via RDMA (0 or 1).
    pub send_idx: AtomicUsize,
    /// Chunk size for pipeline operations.
    pub chunk_size: usize,
    /// Total bytes processed through the pipeline.
    pub bytes_processed: AtomicU64,
    /// Pipeline stage.
    pub stage: AtomicU8,
}

/// Pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PipelineStage {
    /// Pipeline not started.
    Idle = 0,
    /// First chunk: NVMe read into buffer A.
    FirstRead = 1,
    /// Steady state: NVMe read and RDMA send overlap.
    Overlapped = 2,
    /// Last chunk: final RDMA send.
    LastSend = 3,
    /// Pipeline complete.
    Complete = 4,
    /// Pipeline errored.
    Error = 5,
}

impl From<u8> for PipelineStage {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Idle,
            1 => Self::FirstRead,
            2 => Self::Overlapped,
            3 => Self::LastSend,
            4 => Self::Complete,
            5 => Self::Error,
            _ => Self::Error,
        }
    }
}

impl PipelineState {
    /// Create a new pipeline state.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            read_idx: AtomicUsize::new(0),
            send_idx: AtomicUsize::new(0),
            chunk_size,
            bytes_processed: AtomicU64::new(0),
            stage: AtomicU8::new(PipelineStage::Idle as u8),
        }
    }

    /// Get the current stage.
    pub fn stage(&self) -> PipelineStage {
        PipelineStage::from(self.stage.load(Ordering::Acquire))
    }

    /// Advance to the next stage.
    pub fn advance(&self, to: PipelineStage) {
        self.stage.store(to as u8, Ordering::Release);
    }

    /// Swap read/send buffer indices.
    pub fn swap_buffers(&self) {
        let current_read = self.read_idx.load(Ordering::Relaxed);
        self.send_idx.store(current_read, Ordering::Relaxed);
        self.read_idx
            .store(1 - current_read, Ordering::Relaxed);
    }

    /// Record bytes processed.
    pub fn add_bytes(&self, n: u64) {
        self.bytes_processed.fetch_add(n, Ordering::Relaxed);
    }

    /// Get total bytes processed.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_processed.load(Ordering::Relaxed)
    }

    /// Compute the number of chunks needed for a total transfer.
    pub fn num_chunks(&self, total_size: usize) -> usize {
        if self.chunk_size == 0 {
            return 0;
        }
        (total_size + self.chunk_size - 1) / self.chunk_size
    }

    /// Get the size of a specific chunk (last chunk may be smaller).
    pub fn chunk_actual_size(&self, chunk_idx: usize, total_size: usize) -> usize {
        let remaining = total_size.saturating_sub(chunk_idx * self.chunk_size);
        remaining.min(self.chunk_size)
    }
}

// ---------------------------------------------------------------------------
// 3. NVMe-oF Configuration
// ---------------------------------------------------------------------------

/// Configuration for the NVMe-oF target (sender side).
#[derive(Debug, Clone)]
pub struct NvmetConfig {
    /// Subsystem NQN.
    pub nqn: String,
    /// NVMe device path (e.g., /dev/nvme0n1).
    pub nvme_device: PathBuf,
    /// Namespace ID to expose.
    pub namespace_id: u32,
    /// RDMA port for the NVMe-oF target.
    pub rdma_port: u16,
    /// Whether to enable ConnectX-5 target offload.
    pub enable_offload: bool,
    /// Maximum I/O queue depth.
    pub max_queue_depth: u32,
}

impl Default for NvmetConfig {
    fn default() -> Self {
        Self {
            nqn: "nqn.2024-01.com.outerlink:nvme0".to_string(),
            nvme_device: PathBuf::from("/dev/nvme0n1"),
            namespace_id: 1,
            rdma_port: 4420,
            enable_offload: false,
            max_queue_depth: 128,
        }
    }
}

impl NvmetConfig {
    /// Generate the configfs commands to set up the NVMe-oF target.
    ///
    /// These commands configure the Linux kernel's nvmet subsystem via
    /// configfs. In production, these would be executed with root privileges.
    pub fn setup_commands(&self) -> Vec<String> {
        let mut cmds = vec![
            format!(
                "mkdir -p /sys/kernel/config/nvmet/subsystems/{}",
                self.nqn
            ),
            format!(
                "echo 1 > /sys/kernel/config/nvmet/subsystems/{}/attr_allow_any_host",
                self.nqn
            ),
            format!(
                "mkdir -p /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}",
                self.nqn, self.namespace_id
            ),
            format!(
                "echo {} > /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}/device_path",
                self.nvme_device.display(),
                self.nqn,
                self.namespace_id
            ),
            format!(
                "echo 1 > /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}/enable",
                self.nqn, self.namespace_id
            ),
            "mkdir -p /sys/kernel/config/nvmet/ports/1".to_string(),
            "echo rdma > /sys/kernel/config/nvmet/ports/1/addr_trtype".to_string(),
            "echo ipv4 > /sys/kernel/config/nvmet/ports/1/addr_adrfam".to_string(),
            "echo 0.0.0.0 > /sys/kernel/config/nvmet/ports/1/addr_traddr".to_string(),
            format!(
                "echo {} > /sys/kernel/config/nvmet/ports/1/addr_trsvcid",
                self.rdma_port
            ),
            format!(
                "ln -s /sys/kernel/config/nvmet/subsystems/{} /sys/kernel/config/nvmet/ports/1/subsystems/{}",
                self.nqn, self.nqn
            ),
        ];

        if self.enable_offload {
            cmds.push(
                "echo 1 > /sys/kernel/config/nvmet/ports/1/param_offload".to_string(),
            );
        }

        cmds
    }
}

// ---------------------------------------------------------------------------
// 4. Topology Probing Types
// ---------------------------------------------------------------------------

/// Topology probe result for P2PDMA compatibility.
#[derive(Debug, Clone)]
pub struct StorageTopology {
    /// NVMe devices and their PCIe locations.
    pub nvme_devices: Vec<NvmeDeviceInfo>,
    /// ConnectX NIC PCIe locations.
    pub nic_devices: Vec<NicDeviceInfo>,
    /// GPU devices and their BAR1 sizes.
    pub gpu_devices: Vec<GpuDeviceInfo>,
    /// P2P compatibility matrix.
    pub p2p_matrix: HashMap<(PciAddress, PciAddress), P2pCompatibility>,
}

impl StorageTopology {
    /// Create an empty topology.
    pub fn empty() -> Self {
        Self {
            nvme_devices: Vec::new(),
            nic_devices: Vec::new(),
            gpu_devices: Vec::new(),
            p2p_matrix: HashMap::new(),
        }
    }

    /// Check P2P compatibility between two devices.
    pub fn check_p2p(
        &self,
        device_a: &str,
        device_b: &str,
    ) -> P2pCompatibility {
        self.p2p_matrix
            .get(&(device_a.to_string(), device_b.to_string()))
            .or_else(|| {
                self.p2p_matrix
                    .get(&(device_b.to_string(), device_a.to_string()))
            })
            .copied()
            .unwrap_or(P2pCompatibility::NotSupported)
    }

    /// Find GPUs with Resizable BAR enabled.
    pub fn rebar_gpus(&self) -> Vec<&GpuDeviceInfo> {
        self.gpu_devices
            .iter()
            .filter(|g| g.rebar_enabled)
            .collect()
    }
}

/// NVMe device information.
#[derive(Debug, Clone)]
pub struct NvmeDeviceInfo {
    /// Device path (e.g., /dev/nvme0n1).
    pub device_path: PathBuf,
    /// PCI bus address.
    pub pci_address: PciAddress,
    /// Device model string.
    pub model: String,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Controller Memory Buffer present.
    pub has_cmb: bool,
    /// Maximum Data Transfer Size (MDTS).
    pub max_transfer_size: usize,
}

/// NIC device information.
#[derive(Debug, Clone)]
pub struct NicDeviceInfo {
    /// Interface name (e.g., mlx5_0).
    pub interface: String,
    /// PCI bus address.
    pub pci_address: PciAddress,
    /// Whether this NIC supports NVMe-oF target offload.
    pub supports_offload: bool,
    /// NIC firmware version.
    pub firmware_version: String,
}

/// GPU device information.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// GPU index (CUDA device ID).
    pub gpu_index: u32,
    /// PCI bus address.
    pub pci_address: PciAddress,
    /// BAR1 size in bytes (256MB or 24GB with rebar).
    pub bar1_size: u64,
    /// Whether Resizable BAR is enabled.
    pub rebar_enabled: bool,
}

/// P2P compatibility between two PCIe devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pCompatibility {
    /// Direct P2P possible (same switch or supported root complex).
    Supported,
    /// P2P may work but unverified.
    MayWork,
    /// P2P not possible (different root complexes or blocked).
    NotSupported,
}

impl fmt::Display for P2pCompatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Supported => write!(f, "Supported"),
            Self::MayWork => write!(f, "MayWork"),
            Self::NotSupported => write!(f, "NotSupported"),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Tier Driver Integration (R10)
// ---------------------------------------------------------------------------

/// Address within a storage tier.
#[derive(Debug, Clone, Copy)]
pub struct TierAddress {
    pub node: NodeId,
    pub tier: StorageTierType,
    pub offset: u64,
}

/// Storage tier type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTierType {
    LocalNvme,
    RemoteNvme,
}

/// Bandwidth descriptor.
#[derive(Debug, Clone, Copy)]
pub struct Bandwidth {
    /// Bytes per second.
    pub bytes_per_sec: u64,
}

impl Bandwidth {
    /// Create from GB/s.
    pub fn from_gbps(gbps: f64) -> Self {
        Self {
            bytes_per_sec: (gbps * 1_000_000_000.0) as u64,
        }
    }

    /// Get as GB/s.
    pub fn as_gbps(&self) -> f64 {
        self.bytes_per_sec as f64 / 1_000_000_000.0
    }
}

/// Trait for NVMe tier drivers (R10 integration).
///
/// R10's memory tiering system calls this trait when accessing pages
/// stored on NVMe tiers (Tier 4 = local, Tier 5 = remote).
pub trait TierDriver: Send + Sync {
    /// Read a page from this tier.
    fn read_page(
        &self,
        page_addr: TierAddress,
        dest_gpu_ptr: CudaDevicePtr,
        page_size: usize,
    ) -> Result<(), GpuStorageError>;

    /// Write a page to this tier.
    fn write_page(
        &self,
        source_gpu_ptr: CudaDevicePtr,
        page_addr: TierAddress,
        page_size: usize,
    ) -> Result<(), GpuStorageError>;

    /// Allocate space for a page on this tier.
    fn alloc_page(&self) -> Result<TierAddress, GpuStorageError>;

    /// Free a page on this tier.
    fn free_page(&self, page_addr: TierAddress) -> Result<(), GpuStorageError>;

    /// Get tier bandwidth (for scheduling decisions).
    fn bandwidth(&self) -> Bandwidth;

    /// Get tier latency (for scheduling decisions).
    fn latency(&self) -> Duration;
}

// ---------------------------------------------------------------------------
// 6. Storage Service Trait
// ---------------------------------------------------------------------------

/// Main storage service API for OuterLink.
pub trait StorageService: Send + Sync {
    /// Open a handle to a remote storage device.
    fn open(
        &self,
        node_id: NodeId,
        device_path: &str,
    ) -> Result<StorageHandle, GpuStorageError>;

    /// Read from remote storage directly to GPU VRAM.
    fn read_to_gpu(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        size: usize,
        file_offset: u64,
    ) -> Result<usize, GpuStorageError>;

    /// Batch read: multiple regions in one call.
    fn batch_read(
        &self,
        handle: &StorageHandle,
        requests: &[StorageReadRequest],
    ) -> Result<Vec<usize>, GpuStorageError>;

    /// Write GPU VRAM to remote storage (for checkpointing).
    fn write_from_gpu(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        size: usize,
        file_offset: u64,
    ) -> Result<usize, GpuStorageError>;

    /// Probe storage topology for P2PDMA capability.
    fn probe_topology(&self) -> Result<StorageTopology, GpuStorageError>;
}

// ---------------------------------------------------------------------------
// 7. Storage Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GPU storage subsystem.
#[derive(Debug, Clone)]
pub struct GpuStorageConfig {
    /// Default chunk size for double-buffered pipeline (bytes).
    pub chunk_size: usize,
    /// Maximum number of concurrent storage connections.
    pub max_connections: usize,
    /// Whether to enable compression for transfers.
    pub enable_compression: bool,
    /// Merge gap for batch reads: reads within this many bytes are merged.
    pub batch_merge_gap: usize,
    /// Default page size for tier driver (from R10).
    pub page_size: usize,
    /// NVMe-oF configuration (optional).
    pub nvmet_config: Option<NvmetConfig>,
}

impl Default for GpuStorageConfig {
    fn default() -> Self {
        Self {
            chunk_size: 4 * 1024 * 1024, // 4MB
            max_connections: 16,
            enable_compression: false,
            batch_merge_gap: 64 * 1024, // 64KB
            page_size: 65536,           // 64KB (R10 page size)
            nvmet_config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Statistics
// ---------------------------------------------------------------------------

/// Statistics for the GPU storage subsystem.
#[derive(Debug, Default)]
pub struct GpuStorageStats {
    /// Total bytes read from remote storage.
    pub bytes_read: AtomicU64,
    /// Total bytes written to remote storage.
    pub bytes_written: AtomicU64,
    /// Total read operations.
    pub read_ops: AtomicU64,
    /// Total write operations.
    pub write_ops: AtomicU64,
    /// Total batch read operations.
    pub batch_read_ops: AtomicU64,
    /// Total reads merged in batch operations.
    pub reads_merged: AtomicU64,
    /// Active connections.
    pub active_connections: AtomicU64,
    /// Pages allocated on NVMe tier.
    pub tier_pages_allocated: AtomicU64,
    /// Pages freed on NVMe tier.
    pub tier_pages_freed: AtomicU64,
}

impl GpuStorageStats {
    pub fn snapshot(&self) -> GpuStorageStatsSnapshot {
        GpuStorageStatsSnapshot {
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            read_ops: self.read_ops.load(Ordering::Relaxed),
            write_ops: self.write_ops.load(Ordering::Relaxed),
            batch_read_ops: self.batch_read_ops.load(Ordering::Relaxed),
            reads_merged: self.reads_merged.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            tier_pages_allocated: self.tier_pages_allocated.load(Ordering::Relaxed),
            tier_pages_freed: self.tier_pages_freed.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of storage statistics.
#[derive(Debug, Clone)]
pub struct GpuStorageStatsSnapshot {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_ops: u64,
    pub write_ops: u64,
    pub batch_read_ops: u64,
    pub reads_merged: u64,
    pub active_connections: u64,
    pub tier_pages_allocated: u64,
    pub tier_pages_freed: u64,
}

// ---------------------------------------------------------------------------
// 9. Batch Read Merge Logic
// ---------------------------------------------------------------------------

/// A merged read group for batch operations.
#[derive(Debug)]
pub struct MergedReadGroup {
    /// Base file offset for the merged read.
    pub base_offset: u64,
    /// Total size of the merged region.
    pub total_size: usize,
    /// Individual entries within this merged group.
    pub entries: Vec<MergedEntry>,
}

/// An individual entry within a merged read group.
#[derive(Debug)]
pub struct MergedEntry {
    /// Original request index (for result mapping).
    pub original_idx: usize,
    /// Offset within the merged read buffer.
    pub local_offset: usize,
    /// Size of this entry.
    pub size: usize,
    /// Destination GPU buffer.
    pub gpu_buffer: CudaDevicePtr,
    /// Offset within the GPU buffer.
    pub buf_offset: usize,
}

/// Merge nearby reads into larger sequential NVMe I/Os.
///
/// Reads within `merge_gap` bytes of each other are merged into a single
/// larger read. This reduces NVMe queue submissions and improves throughput.
pub fn merge_nearby_reads(
    requests: &[StorageReadRequest],
    merge_gap: usize,
) -> Vec<MergedReadGroup> {
    if requests.is_empty() {
        return Vec::new();
    }

    // Sort by file_offset.
    let mut indexed: Vec<(usize, &StorageReadRequest)> =
        requests.iter().enumerate().collect();
    indexed.sort_by_key(|(_, r)| r.file_offset);

    let mut groups: Vec<MergedReadGroup> = Vec::new();
    let mut current_group = MergedReadGroup {
        base_offset: indexed[0].1.file_offset,
        total_size: indexed[0].1.size,
        entries: vec![MergedEntry {
            original_idx: indexed[0].0,
            local_offset: 0,
            size: indexed[0].1.size,
            gpu_buffer: indexed[0].1.gpu_buffer,
            buf_offset: indexed[0].1.buf_offset,
        }],
    };

    for &(orig_idx, req) in &indexed[1..] {
        let current_end = current_group.base_offset + current_group.total_size as u64;
        let gap = req.file_offset.saturating_sub(current_end);

        if gap <= merge_gap as u64 {
            // Merge into current group.
            let local_offset =
                (req.file_offset - current_group.base_offset) as usize;
            let new_end = local_offset + req.size;
            if new_end > current_group.total_size {
                current_group.total_size = new_end;
            }
            current_group.entries.push(MergedEntry {
                original_idx: orig_idx,
                local_offset,
                size: req.size,
                gpu_buffer: req.gpu_buffer,
                buf_offset: req.buf_offset,
            });
        } else {
            // Start new group.
            groups.push(current_group);
            current_group = MergedReadGroup {
                base_offset: req.file_offset,
                total_size: req.size,
                entries: vec![MergedEntry {
                    original_idx: orig_idx,
                    local_offset: 0,
                    size: req.size,
                    gpu_buffer: req.gpu_buffer,
                    buf_offset: req.buf_offset,
                }],
            };
        }
    }
    groups.push(current_group);

    groups
}

// ---------------------------------------------------------------------------
// 10. Path Selection
// ---------------------------------------------------------------------------

/// Select the optimal storage path based on topology.
pub fn select_storage_path(
    topology: &StorageTopology,
    storage_node_nvme: Option<&NvmeDeviceInfo>,
    storage_node_nic: Option<&NicDeviceInfo>,
    receiver_gpu: Option<&GpuDeviceInfo>,
    default_chunk_size: usize,
) -> StoragePath {
    // Check NVMe-oF offload: need NIC with offload support and P2P between NVMe and NIC.
    let offload_available = match (storage_node_nvme, storage_node_nic) {
        (Some(nvme), Some(nic)) if nic.supports_offload => {
            topology.check_p2p(&nvme.pci_address, &nic.pci_address)
                == P2pCompatibility::Supported
        }
        _ => false,
    };

    // Check receiver has OpenDMA (represented by rebar_enabled for simplicity).
    let receiver_has_opendma = receiver_gpu
        .map(|g| g.rebar_enabled)
        .unwrap_or(false);

    match (offload_available, receiver_has_opendma) {
        (true, true) => StoragePath::NvmeOfOffload,
        (false, true) => StoragePath::HostStaged {
            chunk_size: default_chunk_size,
        },
        (_, false) => StoragePath::FullHostStaged {
            chunk_size: default_chunk_size,
        },
    }
}

// ---------------------------------------------------------------------------
// 11. NVMe Tier Manager (page allocation on NVMe)
// ---------------------------------------------------------------------------

/// Manages page allocation on an NVMe tier.
///
/// Tracks which pages are free using a simple bitmap-like structure.
/// In production, this would use a roaring bitmap for memory efficiency.
pub struct NvmeTierManager {
    /// Node that owns this NVMe.
    node_id: NodeId,
    /// Tier type (local or remote).
    tier_type: StorageTierType,
    /// Page size in bytes.
    page_size: usize,
    /// Total capacity in pages.
    total_pages: u64,
    /// Free page set.
    free_pages: RwLock<Vec<u64>>,
    /// Statistics.
    stats: GpuStorageStats,
}

impl NvmeTierManager {
    /// Create a new NVMe tier manager.
    pub fn new(
        node_id: NodeId,
        tier_type: StorageTierType,
        page_size: usize,
        total_capacity_bytes: u64,
    ) -> Self {
        let total_pages = total_capacity_bytes / page_size as u64;
        let free_pages: Vec<u64> = (0..total_pages).collect();
        Self {
            node_id,
            tier_type,
            page_size,
            total_pages,
            free_pages: RwLock::new(free_pages),
            stats: GpuStorageStats::default(),
        }
    }

    /// Get the number of free pages.
    pub fn free_page_count(&self) -> u64 {
        self.free_pages
            .read()
            .map(|fp| fp.len() as u64)
            .unwrap_or(0)
    }

    /// Get the total number of pages.
    pub fn total_pages(&self) -> u64 {
        self.total_pages
    }

    /// Get the page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get statistics.
    pub fn stats(&self) -> &GpuStorageStats {
        &self.stats
    }

    /// Allocate a page.
    pub fn alloc(&self) -> Result<TierAddress, GpuStorageError> {
        let mut free = self.free_pages.write().map_err(|_| {
            GpuStorageError::Internal("lock poisoned".to_string())
        })?;
        let page_idx = free.pop().ok_or(GpuStorageError::TierFull)?;
        self.stats.tier_pages_allocated.fetch_add(1, Ordering::Relaxed);
        Ok(TierAddress {
            node: self.node_id,
            tier: self.tier_type,
            offset: page_idx * self.page_size as u64,
        })
    }

    /// Free a page.
    pub fn free(&self, addr: TierAddress) -> Result<(), GpuStorageError> {
        let page_idx = addr.offset / self.page_size as u64;
        if page_idx >= self.total_pages {
            return Err(GpuStorageError::InvalidAddress(format!(
                "page index {} out of range (total: {})",
                page_idx, self.total_pages
            )));
        }
        let mut free = self.free_pages.write().map_err(|_| {
            GpuStorageError::Internal("lock poisoned".to_string())
        })?;
        free.push(page_idx);
        self.stats.tier_pages_freed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 12. Error Type
// ---------------------------------------------------------------------------

/// Errors for the GPU storage subsystem.
#[derive(Debug, thiserror::Error)]
pub enum GpuStorageError {
    #[error("alignment error: {0}")]
    AlignmentError(String),

    #[error("NVMe tier is full")]
    TierFull,

    #[error("invalid address: {0}")]
    InvalidAddress(String),

    #[error("connection error: {0}")]
    ConnectionError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("P2P not supported between devices")]
    P2pNotSupported,

    #[error("hardware required: {0}")]
    HardwareRequired(String),

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AlignmentRequirements Tests --

    #[test]
    fn test_alignment_check_valid() {
        let align = AlignmentRequirements::default();
        assert!(align.check(4096, 4096, 512).is_ok());
        assert!(align.check(0, 0, 0).is_ok()); // 0 is aligned to anything
    }

    #[test]
    fn test_alignment_check_invalid_offset() {
        let align = AlignmentRequirements::default();
        let result = align.check(100, 4096, 512);
        assert!(matches!(result, Err(GpuStorageError::AlignmentError(_))));
    }

    #[test]
    fn test_alignment_check_invalid_size() {
        let align = AlignmentRequirements::default();
        let result = align.check(4096, 4096, 100);
        assert!(matches!(result, Err(GpuStorageError::AlignmentError(_))));
    }

    #[test]
    fn test_align_size_up() {
        let align = AlignmentRequirements::default();
        assert_eq!(align.align_size_up(512), 512);
        assert_eq!(align.align_size_up(100), 512);
        assert_eq!(align.align_size_up(513), 1024);
        assert_eq!(align.align_size_up(0), 0);
    }

    // -- StoragePath Tests --

    #[test]
    fn test_storage_path_variants() {
        let host = StoragePath::HostStaged {
            chunk_size: 4 * 1024 * 1024,
        };
        let offload = StoragePath::NvmeOfOffload;
        let fallback = StoragePath::FullHostStaged {
            chunk_size: 4 * 1024 * 1024,
        };
        assert_ne!(host, offload);
        assert_ne!(host, fallback);
    }

    // -- PipelineState Tests --

    #[test]
    fn test_pipeline_state_creation() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        assert_eq!(pipeline.stage(), PipelineStage::Idle);
        assert_eq!(pipeline.total_bytes(), 0);
        assert_eq!(pipeline.chunk_size, 4 * 1024 * 1024);
    }

    #[test]
    fn test_pipeline_state_advance() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        pipeline.advance(PipelineStage::FirstRead);
        assert_eq!(pipeline.stage(), PipelineStage::FirstRead);

        pipeline.advance(PipelineStage::Overlapped);
        assert_eq!(pipeline.stage(), PipelineStage::Overlapped);

        pipeline.advance(PipelineStage::Complete);
        assert_eq!(pipeline.stage(), PipelineStage::Complete);
    }

    #[test]
    fn test_pipeline_swap_buffers() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        assert_eq!(pipeline.read_idx.load(Ordering::Relaxed), 0);
        assert_eq!(pipeline.send_idx.load(Ordering::Relaxed), 0);

        pipeline.swap_buffers();
        assert_eq!(pipeline.send_idx.load(Ordering::Relaxed), 0);
        assert_eq!(pipeline.read_idx.load(Ordering::Relaxed), 1);

        pipeline.swap_buffers();
        assert_eq!(pipeline.send_idx.load(Ordering::Relaxed), 1);
        assert_eq!(pipeline.read_idx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_pipeline_num_chunks() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        assert_eq!(pipeline.num_chunks(4 * 1024 * 1024), 1);
        assert_eq!(pipeline.num_chunks(8 * 1024 * 1024), 2);
        assert_eq!(pipeline.num_chunks(5 * 1024 * 1024), 2); // Rounds up
        assert_eq!(pipeline.num_chunks(0), 0);
    }

    #[test]
    fn test_pipeline_chunk_actual_size() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        let total = 5 * 1024 * 1024;
        assert_eq!(pipeline.chunk_actual_size(0, total), 4 * 1024 * 1024);
        assert_eq!(pipeline.chunk_actual_size(1, total), 1 * 1024 * 1024); // Last chunk
    }

    #[test]
    fn test_pipeline_bytes_tracking() {
        let pipeline = PipelineState::new(4 * 1024 * 1024);
        pipeline.add_bytes(1000);
        pipeline.add_bytes(2000);
        assert_eq!(pipeline.total_bytes(), 3000);
    }

    // -- NvmetConfig Tests --

    #[test]
    fn test_nvmet_config_default() {
        let config = NvmetConfig::default();
        assert_eq!(config.rdma_port, 4420);
        assert!(!config.enable_offload);
        assert_eq!(config.max_queue_depth, 128);
    }

    #[test]
    fn test_nvmet_setup_commands() {
        let config = NvmetConfig::default();
        let cmds = config.setup_commands();
        assert!(cmds.len() >= 10);
        assert!(cmds[0].contains("mkdir -p"));
        assert!(cmds.iter().any(|c| c.contains("rdma")));
        // Without offload, no offload command.
        assert!(!cmds.iter().any(|c| c.contains("param_offload")));
    }

    #[test]
    fn test_nvmet_setup_commands_with_offload() {
        let mut config = NvmetConfig::default();
        config.enable_offload = true;
        let cmds = config.setup_commands();
        assert!(cmds.iter().any(|c| c.contains("param_offload")));
    }

    // -- StorageTopology Tests --

    #[test]
    fn test_topology_empty() {
        let topo = StorageTopology::empty();
        assert!(topo.nvme_devices.is_empty());
        assert!(topo.gpu_devices.is_empty());
    }

    #[test]
    fn test_topology_p2p_check() {
        let mut topo = StorageTopology::empty();
        topo.p2p_matrix.insert(
            ("0000:01:00.0".to_string(), "0000:02:00.0".to_string()),
            P2pCompatibility::Supported,
        );

        assert_eq!(
            topo.check_p2p("0000:01:00.0", "0000:02:00.0"),
            P2pCompatibility::Supported
        );
        // Reverse lookup works too.
        assert_eq!(
            topo.check_p2p("0000:02:00.0", "0000:01:00.0"),
            P2pCompatibility::Supported
        );
        // Unknown pair.
        assert_eq!(
            topo.check_p2p("0000:03:00.0", "0000:04:00.0"),
            P2pCompatibility::NotSupported
        );
    }

    #[test]
    fn test_topology_rebar_gpus() {
        let mut topo = StorageTopology::empty();
        topo.gpu_devices.push(GpuDeviceInfo {
            gpu_index: 0,
            pci_address: "0000:01:00.0".to_string(),
            bar1_size: 256 * 1024 * 1024,
            rebar_enabled: false,
        });
        topo.gpu_devices.push(GpuDeviceInfo {
            gpu_index: 1,
            pci_address: "0000:02:00.0".to_string(),
            bar1_size: 24 * 1024 * 1024 * 1024,
            rebar_enabled: true,
        });

        let rebar = topo.rebar_gpus();
        assert_eq!(rebar.len(), 1);
        assert_eq!(rebar[0].gpu_index, 1);
    }

    // -- Batch Read Merge Tests --

    #[test]
    fn test_merge_nearby_reads_no_merge() {
        let requests = vec![
            StorageReadRequest {
                gpu_buffer: 0x1000,
                size: 1024,
                file_offset: 0,
                buf_offset: 0,
            },
            StorageReadRequest {
                gpu_buffer: 0x2000,
                size: 1024,
                file_offset: 1_000_000, // Far away
                buf_offset: 0,
            },
        ];

        let groups = merge_nearby_reads(&requests, 64 * 1024);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_merge_nearby_reads_adjacent() {
        let requests = vec![
            StorageReadRequest {
                gpu_buffer: 0x1000,
                size: 4096,
                file_offset: 0,
                buf_offset: 0,
            },
            StorageReadRequest {
                gpu_buffer: 0x2000,
                size: 4096,
                file_offset: 4096, // Immediately adjacent
                buf_offset: 0,
            },
        ];

        let groups = merge_nearby_reads(&requests, 64 * 1024);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].total_size, 8192);
        assert_eq!(groups[0].entries.len(), 2);
    }

    #[test]
    fn test_merge_nearby_reads_within_gap() {
        let requests = vec![
            StorageReadRequest {
                gpu_buffer: 0x1000,
                size: 4096,
                file_offset: 0,
                buf_offset: 0,
            },
            StorageReadRequest {
                gpu_buffer: 0x2000,
                size: 4096,
                file_offset: 50_000, // Within 64KB gap
                buf_offset: 0,
            },
        ];

        let groups = merge_nearby_reads(&requests, 64 * 1024);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].total_size, 50_000 + 4096);
    }

    #[test]
    fn test_merge_nearby_reads_preserves_original_idx() {
        let requests = vec![
            StorageReadRequest {
                gpu_buffer: 0x2000,
                size: 4096,
                file_offset: 8192, // Higher offset but index 0
                buf_offset: 0,
            },
            StorageReadRequest {
                gpu_buffer: 0x1000,
                size: 4096,
                file_offset: 0, // Lower offset but index 1
                buf_offset: 0,
            },
        ];

        let groups = merge_nearby_reads(&requests, 64 * 1024);
        assert_eq!(groups.len(), 1);
        // After sorting by offset, entry at offset 0 has original_idx=1.
        let entry_at_0 = groups[0]
            .entries
            .iter()
            .find(|e| e.local_offset == 0)
            .unwrap();
        assert_eq!(entry_at_0.original_idx, 1);
    }

    #[test]
    fn test_merge_nearby_reads_empty() {
        let groups = merge_nearby_reads(&[], 64 * 1024);
        assert!(groups.is_empty());
    }

    // -- Path Selection Tests --

    #[test]
    fn test_select_path_nvmeof_offload() {
        let mut topo = StorageTopology::empty();
        let nvme = NvmeDeviceInfo {
            device_path: PathBuf::from("/dev/nvme0n1"),
            pci_address: "0000:01:00.0".to_string(),
            model: "Samsung 980 Pro".to_string(),
            size_bytes: 1_000_000_000_000,
            has_cmb: false,
            max_transfer_size: 1024 * 1024,
        };
        let nic = NicDeviceInfo {
            interface: "mlx5_0".to_string(),
            pci_address: "0000:02:00.0".to_string(),
            supports_offload: true,
            firmware_version: "16.28.2006".to_string(),
        };
        let gpu = GpuDeviceInfo {
            gpu_index: 0,
            pci_address: "0000:03:00.0".to_string(),
            bar1_size: 24 * 1024 * 1024 * 1024,
            rebar_enabled: true,
        };
        topo.p2p_matrix.insert(
            ("0000:01:00.0".to_string(), "0000:02:00.0".to_string()),
            P2pCompatibility::Supported,
        );

        let path = select_storage_path(
            &topo,
            Some(&nvme),
            Some(&nic),
            Some(&gpu),
            4 * 1024 * 1024,
        );
        assert_eq!(path, StoragePath::NvmeOfOffload);
    }

    #[test]
    fn test_select_path_host_staged() {
        let topo = StorageTopology::empty();
        let nvme = NvmeDeviceInfo {
            device_path: PathBuf::from("/dev/nvme0n1"),
            pci_address: "0000:01:00.0".to_string(),
            model: "Samsung 980 Pro".to_string(),
            size_bytes: 1_000_000_000_000,
            has_cmb: false,
            max_transfer_size: 1024 * 1024,
        };
        let nic = NicDeviceInfo {
            interface: "mlx5_0".to_string(),
            pci_address: "0000:02:00.0".to_string(),
            supports_offload: false, // No offload
            firmware_version: "16.28.2006".to_string(),
        };
        let gpu = GpuDeviceInfo {
            gpu_index: 0,
            pci_address: "0000:03:00.0".to_string(),
            bar1_size: 24 * 1024 * 1024 * 1024,
            rebar_enabled: true,
        };

        let path = select_storage_path(
            &topo,
            Some(&nvme),
            Some(&nic),
            Some(&gpu),
            4 * 1024 * 1024,
        );
        assert_eq!(
            path,
            StoragePath::HostStaged {
                chunk_size: 4 * 1024 * 1024
            }
        );
    }

    #[test]
    fn test_select_path_full_host_staged_no_rebar() {
        let topo = StorageTopology::empty();
        let gpu = GpuDeviceInfo {
            gpu_index: 0,
            pci_address: "0000:03:00.0".to_string(),
            bar1_size: 256 * 1024 * 1024,
            rebar_enabled: false, // No OpenDMA
        };

        let path = select_storage_path(
            &topo,
            None,
            None,
            Some(&gpu),
            4 * 1024 * 1024,
        );
        assert_eq!(
            path,
            StoragePath::FullHostStaged {
                chunk_size: 4 * 1024 * 1024
            }
        );
    }

    // -- NvmeTierManager Tests --

    #[test]
    fn test_tier_manager_creation() {
        let mgr = NvmeTierManager::new(0, StorageTierType::RemoteNvme, 65536, 65536 * 100);
        assert_eq!(mgr.total_pages(), 100);
        assert_eq!(mgr.free_page_count(), 100);
        assert_eq!(mgr.page_size(), 65536);
    }

    #[test]
    fn test_tier_manager_alloc_and_free() {
        let mgr = NvmeTierManager::new(0, StorageTierType::LocalNvme, 65536, 65536 * 10);
        assert_eq!(mgr.free_page_count(), 10);

        let addr = mgr.alloc().unwrap();
        assert_eq!(mgr.free_page_count(), 9);
        assert_eq!(addr.tier, StorageTierType::LocalNvme);

        mgr.free(addr).unwrap();
        assert_eq!(mgr.free_page_count(), 10);
    }

    #[test]
    fn test_tier_manager_alloc_exhaustion() {
        let mgr = NvmeTierManager::new(0, StorageTierType::RemoteNvme, 65536, 65536 * 2);
        mgr.alloc().unwrap();
        mgr.alloc().unwrap();
        let result = mgr.alloc();
        assert!(matches!(result, Err(GpuStorageError::TierFull)));
    }

    #[test]
    fn test_tier_manager_free_invalid_address() {
        let mgr = NvmeTierManager::new(0, StorageTierType::LocalNvme, 65536, 65536 * 2);
        let bad_addr = TierAddress {
            node: 0,
            tier: StorageTierType::LocalNvme,
            offset: 65536 * 100, // Way out of range
        };
        let result = mgr.free(bad_addr);
        assert!(matches!(result, Err(GpuStorageError::InvalidAddress(_))));
    }

    #[test]
    fn test_tier_manager_stats_tracking() {
        let mgr = NvmeTierManager::new(0, StorageTierType::RemoteNvme, 65536, 65536 * 10);
        let addr = mgr.alloc().unwrap();
        mgr.free(addr).unwrap();

        let snap = mgr.stats().snapshot();
        assert_eq!(snap.tier_pages_allocated, 1);
        assert_eq!(snap.tier_pages_freed, 1);
    }

    // -- StorageFuture Tests --

    #[test]
    fn test_storage_future_complete() {
        let future = StorageFuture::new();
        assert!(!future.is_complete());

        future.complete(1024);
        assert!(future.is_complete());
        assert_eq!(future.wait().unwrap(), 1024);
    }

    #[test]
    fn test_storage_future_fail() {
        let future = StorageFuture::new();
        future.fail(GpuStorageError::IoError("disk error".to_string()));
        assert!(future.is_complete());
        assert!(future.wait().is_err());
    }

    // -- Bandwidth Tests --

    #[test]
    fn test_bandwidth_conversion() {
        let bw = Bandwidth::from_gbps(7.0);
        assert_eq!(bw.bytes_per_sec, 7_000_000_000);
        assert!((bw.as_gbps() - 7.0).abs() < 0.001);
    }

    // -- GpuStorageConfig Tests --

    #[test]
    fn test_config_defaults() {
        let config = GpuStorageConfig::default();
        assert_eq!(config.chunk_size, 4 * 1024 * 1024);
        assert_eq!(config.page_size, 65536);
        assert!(!config.enable_compression);
        assert!(config.nvmet_config.is_none());
    }

    // -- StorageNamespace Tests --

    #[test]
    fn test_storage_namespace() {
        let ns = StorageNamespace {
            nqn: "nqn.2024-01.com.outerlink:nvme0".to_string(),
            nsid: 1,
            size: 1_000_000_000_000,
            block_size: 4096,
        };
        assert_eq!(ns.nsid, 1);
        assert_eq!(ns.block_size, 4096);
    }

    // -- P2pCompatibility Display Tests --

    #[test]
    fn test_p2p_compatibility_display() {
        assert_eq!(format!("{}", P2pCompatibility::Supported), "Supported");
        assert_eq!(format!("{}", P2pCompatibility::MayWork), "MayWork");
        assert_eq!(
            format!("{}", P2pCompatibility::NotSupported),
            "NotSupported"
        );
    }
}
