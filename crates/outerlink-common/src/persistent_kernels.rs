//! R30: Persistent Kernels with Network Feed
//!
//! Types and algorithms for GPU persistent kernels that poll VRAM ring buffers
//! for incoming data, process it, and write results to output ring buffers --
//! all without kernel launch overhead or CPU synchronization.
//!
//! # Architecture
//!
//! ```text
//! Remote NIC --RDMA/BAR1--> [Input Data Pool] + [Input Ring Buffer (doorbell)]
//!                                    |
//!                           GPU Persistent Kernel (polls ring, processes data)
//!                                    |
//!                           [Output Ring Buffer] + [Output Data Pool]
//!                                    |
//!                           Host/NIC reads output --RDMA-->
//! ```
//!
//! The persistent kernel runs indefinitely (one per GPU), using `__nanosleep(100)`
//! polling to minimize idle power. Warp 0 watches the doorbell; all warps process
//! data in parallel when entries arrive.
//!
//! # Ring Buffer Protocol
//!
//! The producer (NIC via RDMA or CPU) writes:
//! 1. Data payload to a data pool slot
//! 2. Descriptor to the ring entry
//! 3. Head pointer update (the "doorbell")
//!
//! PCIe write ordering guarantees the GPU sees data before the doorbell.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Ring Buffer Types
// ---------------------------------------------------------------------------

/// VRAM-resident ring buffer descriptor.
///
/// The actual memory is allocated in GPU VRAM and optionally registered with
/// BAR1 for NIC RDMA access. This struct holds the host-side metadata.
///
/// Memory layout in VRAM (contiguous allocation):
///   offset 0:                   head (u64, volatile, written by producer)
///   offset 8:                   _pad (56 bytes, cache line isolation)
///   offset 64:                  tail (u64, volatile, written by consumer/GPU)
///   offset 72:                  _pad (56 bytes)
///   offset 128:                 entries[0..capacity] (DoorbellDescriptor array)
///   offset 128 + capacity*48:   padding to page boundary
#[derive(Debug, Clone)]
pub struct RingBuffer {
    /// GPU device pointer to the ring structure in VRAM.
    pub device_ptr: u64,

    /// Ring capacity (must be power of 2 for fast modulo via bitmask).
    pub capacity: u32,

    /// BAR1 physical address (for RDMA registration). 0 if not registered.
    pub bar1_addr: u64,

    /// RDMA memory region key (for remote writes). 0 if not registered.
    pub rkey: u32,

    /// Total allocation size in bytes.
    pub alloc_size: usize,
}

impl RingBuffer {
    /// Create a new ring buffer descriptor.
    ///
    /// `capacity` must be a power of 2 (enforced). If not, it is rounded up.
    /// `device_ptr` is the GPU VRAM address of the allocation.
    pub fn new(device_ptr: u64, capacity: u32) -> Self {
        let capacity = capacity.next_power_of_two();
        let alloc_size = Self::compute_alloc_size(capacity);
        Self {
            device_ptr,
            capacity,
            bar1_addr: 0,
            rkey: 0,
            alloc_size,
        }
    }

    /// Compute total allocation size for a given capacity.
    ///
    /// Layout: 128 bytes header + capacity * 48 bytes descriptors, rounded up to 4 KB.
    pub fn compute_alloc_size(capacity: u32) -> usize {
        let raw = RING_HEADER_SIZE + (capacity as usize * DESCRIPTOR_SIZE);
        // Round up to page boundary (4 KB)
        (raw + 4095) & !4095
    }

    /// Compute the VRAM offset of a descriptor entry by index.
    pub fn descriptor_offset(&self, index: u32) -> usize {
        RING_HEADER_SIZE + (index as usize * DESCRIPTOR_SIZE)
    }

    /// Compute the VRAM address of a descriptor entry.
    pub fn descriptor_addr(&self, index: u32) -> u64 {
        self.device_ptr + self.descriptor_offset(index) as u64
    }

    /// Compute the BAR1 address of a descriptor entry (for RDMA writes).
    /// Returns None if BAR1 is not registered.
    pub fn descriptor_bar1_addr(&self, index: u32) -> Option<u64> {
        if self.bar1_addr == 0 {
            return None;
        }
        Some(self.bar1_addr + self.descriptor_offset(index) as u64)
    }

    /// Check if this ring buffer has BAR1 registration for RDMA.
    pub fn has_bar1(&self) -> bool {
        self.bar1_addr != 0
    }

    /// Map a sequence number to a ring index (fast modulo via bitmask).
    pub fn wrap_index(&self, sequence: u64) -> u32 {
        (sequence & (self.capacity as u64 - 1)) as u32
    }

    /// Check if the ring is full given current head and tail.
    pub fn is_full(&self, head: u64, tail: u64) -> bool {
        (head - tail) >= self.capacity as u64
    }

    /// Number of entries available for consumption.
    pub fn entries_available(&self, head: u64, tail: u64) -> u64 {
        head.saturating_sub(tail)
    }

    /// Number of free slots available for production.
    pub fn free_slots(&self, head: u64, tail: u64) -> u64 {
        self.capacity as u64 - self.entries_available(head, tail)
    }
}

/// Ring header size in bytes (2 cache lines: head + pad + tail + pad).
pub const RING_HEADER_SIZE: usize = 128;

/// Size of a single DoorbellDescriptor in bytes.
pub const DESCRIPTOR_SIZE: usize = 48;

/// Ring header layout -- mirrors the GPU-side `RingHeader` struct.
///
/// Two cache lines (128 bytes total) with head and tail on separate
/// cache lines to avoid false sharing between producer and consumer.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct RingHeader {
    /// Written by producer (NIC/CPU), read by GPU kernel.
    pub head: u64,
    _pad0: [u8; 56],
    /// Written by GPU kernel, read by producer for backpressure.
    pub tail: u64,
    _pad1: [u8; 56],
}

impl RingHeader {
    /// Create a zeroed ring header.
    pub fn new() -> Self {
        Self {
            head: 0,
            _pad0: [0; 56],
            tail: 0,
            _pad1: [0; 56],
        }
    }
}

impl Default for RingHeader {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Doorbell Descriptor
// ---------------------------------------------------------------------------

/// Descriptor for one ring buffer entry.
///
/// Written by producer (NIC via RDMA or CPU), read by GPU kernel.
/// 48 bytes, aligned for efficient GPU memory access.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoorbellDescriptor {
    /// Offset into the data pool where payload lives.
    pub data_offset: u64,
    /// Payload size in bytes.
    pub data_size: u32,
    /// Monotonically increasing sequence number.
    pub sequence: u32,
    /// Source connection identifier (for multiplexing).
    pub connection_id: u32,
    /// Descriptor flags (see `descriptor_flags`).
    pub flags: u32,
    /// PTP timestamp of when data was sent (from R26).
    pub ptp_timestamp: u64,
    /// Reserved for future use (checksum, compression info).
    pub metadata: u64,
}

impl DoorbellDescriptor {
    /// Create a new descriptor with the given data offset, size, and sequence.
    pub fn new(data_offset: u64, data_size: u32, sequence: u32) -> Self {
        Self {
            data_offset,
            data_size,
            sequence,
            connection_id: 0,
            flags: 0,
            ptp_timestamp: 0,
            metadata: 0,
        }
    }

    /// Check if a specific flag is set.
    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Whether this descriptor's data is compressed.
    pub fn is_compressed(&self) -> bool {
        self.has_flag(descriptor_flags::COMPRESSED)
    }

    /// Whether this descriptor uses scatter-gather.
    pub fn is_scatter_gather(&self) -> bool {
        self.has_flag(descriptor_flags::SCATTER_GATHER)
    }

    /// Whether this is the last descriptor in a batch.
    pub fn is_last_in_batch(&self) -> bool {
        self.has_flag(descriptor_flags::LAST_IN_BATCH)
    }

    /// Whether this is an urgent (high-priority) descriptor.
    pub fn is_urgent(&self) -> bool {
        self.has_flag(descriptor_flags::URGENT)
    }

    /// Whether the metadata field contains a CRC32 checksum.
    pub fn has_checksum(&self) -> bool {
        self.has_flag(descriptor_flags::CHECKSUM)
    }
}

/// Descriptor flag constants.
pub mod descriptor_flags {
    /// Data is compressed (R14 integration).
    pub const COMPRESSED: u32 = 1 << 0;
    /// data_offset points to a scatter-gather list (R28 integration).
    pub const SCATTER_GATHER: u32 = 1 << 1;
    /// Last descriptor in a multi-descriptor batch.
    pub const LAST_IN_BATCH: u32 = 1 << 2;
    /// High-priority -- skip nanosleep, process immediately.
    pub const URGENT: u32 = 1 << 3;
    /// metadata field contains CRC32 checksum.
    pub const CHECKSUM: u32 = 1 << 4;
}

// ---------------------------------------------------------------------------
// Data Pool
// ---------------------------------------------------------------------------

/// Pre-allocated VRAM data buffer pool.
///
/// Fixed-size slots for predictable RDMA offsets. Registered with BAR1
/// for NIC direct write access.
#[derive(Debug, Clone)]
pub struct DataPool {
    /// GPU device pointer to pool start.
    pub device_ptr: u64,
    /// BAR1 physical address of pool start. 0 if not registered.
    pub bar1_addr: u64,
    /// RDMA memory region key. 0 if not registered.
    pub rkey: u32,
    /// Bytes per slot.
    pub slot_size: usize,
    /// Number of slots.
    pub slot_count: u32,
    /// Total pool size (slot_size * slot_count).
    pub total_size: usize,
}

impl DataPool {
    /// Create a new data pool descriptor.
    pub fn new(device_ptr: u64, slot_size: usize, slot_count: u32) -> Self {
        Self {
            device_ptr,
            bar1_addr: 0,
            rkey: 0,
            slot_size,
            slot_count,
            total_size: slot_size * slot_count as usize,
        }
    }

    /// Calculate the VRAM address for a given slot index.
    pub fn slot_addr(&self, slot_index: u32) -> u64 {
        self.device_ptr + (slot_index as u64 * self.slot_size as u64)
    }

    /// Calculate the BAR1 address for a given slot index (for RDMA writes).
    /// Returns None if BAR1 is not registered.
    pub fn slot_bar1_addr(&self, slot_index: u32) -> Option<u64> {
        if self.bar1_addr == 0 {
            return None;
        }
        Some(self.bar1_addr + (slot_index as u64 * self.slot_size as u64))
    }

    /// Calculate the data offset (relative to pool start) for a slot.
    pub fn slot_offset(&self, slot_index: u32) -> u64 {
        slot_index as u64 * self.slot_size as u64
    }

    /// Check if a slot index is valid.
    pub fn is_valid_slot(&self, slot_index: u32) -> bool {
        slot_index < self.slot_count
    }
}

// ---------------------------------------------------------------------------
// Control Block
// ---------------------------------------------------------------------------

/// VRAM-resident control structure for kernel monitoring and shutdown.
///
/// Host writes `shutdown_requested`, kernel writes `heartbeat`.
/// Mapped to host via `cudaHostGetDevicePointer` for low-latency monitoring.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ControlBlock {
    /// GPU clock64() timestamp, updated every N iterations by kernel.
    pub heartbeat: u64,
    _pad0: [u8; 56],

    /// Set to 1 by host to request graceful shutdown.
    pub shutdown_requested: u32,
    /// Set to non-zero by kernel on fatal error.
    pub error_code: u32,
    /// Total batches processed (for stats).
    pub batches_processed: u64,
    /// Total bytes processed.
    pub bytes_processed: u64,
    // Explicit padding to fill second cache line to 64 bytes:
    // shutdown_requested(4) + error_code(4) + batches_processed(8) + bytes_processed(8) + pad(40) = 64
    _pad1: [u8; 40],
}

impl ControlBlock {
    /// Create a zeroed control block.
    pub fn new() -> Self {
        Self {
            heartbeat: 0,
            _pad0: [0; 56],
            shutdown_requested: 0,
            error_code: 0,
            batches_processed: 0,
            bytes_processed: 0,
            _pad1: [0; 40],
        }
    }

    /// Check if the kernel has been requested to shut down.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested != 0
    }

    /// Check if the kernel reported an error.
    pub fn has_error(&self) -> bool {
        self.error_code != 0
    }
}

impl Default for ControlBlock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Kernel State
// ---------------------------------------------------------------------------

/// State of the persistent kernel lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelState {
    /// Kernel has not been launched yet.
    NotLaunched,
    /// Kernel is running. `launched_at_epoch_ms` is milliseconds since some reference.
    Running { launched_at_epoch_ms: u64 },
    /// Graceful shutdown has been requested.
    ShuttingDown { requested_at_epoch_ms: u64 },
    /// Kernel encountered a fatal error.
    Failed {
        error_code: u32,
        failed_at_epoch_ms: u64,
    },
    /// Kernel is being recovered after a failure.
    Recovering,
}

// ---------------------------------------------------------------------------
// Persistent Kernel Configuration
// ---------------------------------------------------------------------------

/// Configuration for a persistent kernel instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentKernelConfig {
    /// Input ring buffer capacity (entries). Must be power of 2.
    pub input_ring_size: u32,
    /// Output ring buffer capacity (entries). Must be power of 2.
    pub output_ring_size: u32,
    /// Size of each data pool slot in bytes.
    pub data_slot_size: usize,
    /// Number of input data pool slots.
    pub input_slot_count: u32,
    /// Number of output data pool slots.
    pub output_slot_count: u32,

    /// CUDA blocks per SM (non-cooperative launch).
    pub blocks_per_sm: u32,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
    /// Nanosleep interval for idle polling.
    pub poll_interval_ns: u32,

    /// How often to check the heartbeat (ms).
    pub heartbeat_check_ms: u32,
    /// Heartbeat timeout -- if no update after this many ms, kernel is hung.
    pub heartbeat_timeout_ms: u32,
    /// Maximum recovery attempts before giving up.
    pub max_recovery_attempts: u8,

    /// If Some(N), kernel exits after N iterations (for Windows dev/testing).
    /// None means infinite (production Linux mode).
    pub max_iterations: Option<u64>,
}

impl Default for PersistentKernelConfig {
    fn default() -> Self {
        Self {
            input_ring_size: 256,
            output_ring_size: 256,
            data_slot_size: 65536, // 64 KB
            input_slot_count: 256,
            output_slot_count: 256,

            blocks_per_sm: 1,
            threads_per_block: 256,
            poll_interval_ns: 100,

            heartbeat_check_ms: 100,
            heartbeat_timeout_ms: 500,
            max_recovery_attempts: 3,

            max_iterations: None,
        }
    }
}

impl PersistentKernelConfig {
    /// Compute the total VRAM required for this configuration.
    ///
    /// Includes input ring, output ring, input data pool, output data pool,
    /// and control block.
    pub fn total_vram_bytes(&self) -> usize {
        let input_ring = RingBuffer::compute_alloc_size(self.input_ring_size);
        let output_ring = RingBuffer::compute_alloc_size(self.output_ring_size);
        let input_pool = self.data_slot_size * self.input_slot_count as usize;
        let output_pool = self.data_slot_size * self.output_slot_count as usize;
        let control = std::mem::size_of::<ControlBlock>();
        input_ring + output_ring + input_pool + output_pool + control
    }

    /// Configuration for development/testing with smaller buffers.
    pub fn dev() -> Self {
        Self {
            input_ring_size: 16,
            output_ring_size: 16,
            data_slot_size: 1024,
            input_slot_count: 16,
            output_slot_count: 16,
            max_iterations: Some(1000),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent Kernel Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for a persistent kernel instance.
#[derive(Debug)]
pub struct PersistentKernelStats {
    pub batches_processed: AtomicU64,
    pub bytes_ingested: AtomicU64,
    pub bytes_emitted: AtomicU64,
    pub doorbell_latency_ns_avg: AtomicU64,
    pub processing_time_ns_avg: AtomicU64,
    pub ring_full_stalls: AtomicU64,
    pub errors_recovered: AtomicU32,
    pub uptime_secs: AtomicU64,
}

impl PersistentKernelStats {
    /// Create zeroed statistics.
    pub fn new() -> Self {
        Self {
            batches_processed: AtomicU64::new(0),
            bytes_ingested: AtomicU64::new(0),
            bytes_emitted: AtomicU64::new(0),
            doorbell_latency_ns_avg: AtomicU64::new(0),
            processing_time_ns_avg: AtomicU64::new(0),
            ring_full_stalls: AtomicU64::new(0),
            errors_recovered: AtomicU32::new(0),
            uptime_secs: AtomicU64::new(0),
        }
    }

    /// Take a snapshot for serialization.
    pub fn snapshot(&self) -> PersistentKernelStatsSnapshot {
        PersistentKernelStatsSnapshot {
            batches_processed: self.batches_processed.load(Ordering::Relaxed),
            bytes_ingested: self.bytes_ingested.load(Ordering::Relaxed),
            bytes_emitted: self.bytes_emitted.load(Ordering::Relaxed),
            doorbell_latency_ns_avg: self.doorbell_latency_ns_avg.load(Ordering::Relaxed),
            processing_time_ns_avg: self.processing_time_ns_avg.load(Ordering::Relaxed),
            ring_full_stalls: self.ring_full_stalls.load(Ordering::Relaxed),
            errors_recovered: self.errors_recovered.load(Ordering::Relaxed),
            uptime_secs: self.uptime_secs.load(Ordering::Relaxed),
        }
    }
}

impl Default for PersistentKernelStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of persistent kernel statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentKernelStatsSnapshot {
    pub batches_processed: u64,
    pub bytes_ingested: u64,
    pub bytes_emitted: u64,
    pub doorbell_latency_ns_avg: u64,
    pub processing_time_ns_avg: u64,
    pub ring_full_stalls: u64,
    pub errors_recovered: u32,
    pub uptime_secs: u64,
}

// ---------------------------------------------------------------------------
// Persistent Region Pinning (R10 integration)
// ---------------------------------------------------------------------------

/// Purpose of a pinned VRAM region (for R10 page table integration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinPurpose {
    InputRing,
    OutputRing,
    InputDataPool,
    OutputDataPool,
    ControlBlock,
}

/// A pinned VRAM region that R10's migration engine must not evict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedRegion {
    /// Base VRAM address.
    pub base_addr: u64,
    /// Region size in bytes.
    pub size: usize,
    /// Why this region is pinned.
    pub purpose: PinPurpose,
}

/// Collection of pinned regions for a persistent kernel on one GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentRegionPin {
    /// GPU this kernel runs on.
    pub gpu_id: u8,
    /// All pinned regions.
    pub regions: Vec<PinnedRegion>,
}

impl PersistentRegionPin {
    /// Create pinned region descriptors from a persistent kernel's allocations.
    pub fn from_allocations(
        gpu_id: u8,
        input_ring: &RingBuffer,
        output_ring: &RingBuffer,
        input_pool: &DataPool,
        output_pool: &DataPool,
        control_block_addr: u64,
    ) -> Self {
        Self {
            gpu_id,
            regions: vec![
                PinnedRegion {
                    base_addr: input_ring.device_ptr,
                    size: input_ring.alloc_size,
                    purpose: PinPurpose::InputRing,
                },
                PinnedRegion {
                    base_addr: output_ring.device_ptr,
                    size: output_ring.alloc_size,
                    purpose: PinPurpose::OutputRing,
                },
                PinnedRegion {
                    base_addr: input_pool.device_ptr,
                    size: input_pool.total_size,
                    purpose: PinPurpose::InputDataPool,
                },
                PinnedRegion {
                    base_addr: output_pool.device_ptr,
                    size: output_pool.total_size,
                    purpose: PinPurpose::OutputDataPool,
                },
                PinnedRegion {
                    base_addr: control_block_addr,
                    size: std::mem::size_of::<ControlBlock>(),
                    purpose: PinPurpose::ControlBlock,
                },
            ],
        }
    }

    /// Total pinned VRAM in bytes.
    pub fn total_pinned_bytes(&self) -> usize {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// Check if a given address range overlaps with any pinned region.
    pub fn overlaps(&self, addr: u64, size: usize) -> bool {
        let end = addr + size as u64;
        self.regions.iter().any(|r| {
            let r_end = r.base_addr + r.size as u64;
            addr < r_end && end > r.base_addr
        })
    }
}

// ---------------------------------------------------------------------------
// Ring Buffer Producer Protocol (host-side)
// ---------------------------------------------------------------------------

/// Host-side ring buffer producer state.
///
/// Tracks the producer's view of the ring for writing descriptors and data.
/// The producer maintains its own head counter and reads the consumer's tail
/// for backpressure.
#[derive(Debug)]
pub struct RingProducer {
    /// Current head position (next entry to write).
    head: u64,
    /// Last known tail position (cached from consumer/GPU).
    cached_tail: u64,
    /// Ring capacity.
    capacity: u32,
    /// Next sequence number.
    next_sequence: u32,
}

impl RingProducer {
    /// Create a new producer starting at position 0.
    pub fn new(capacity: u32) -> Self {
        Self {
            head: 0,
            cached_tail: 0,
            capacity: capacity.next_power_of_two(),
            next_sequence: 0,
        }
    }

    /// Check if the ring has space for one more entry.
    pub fn can_produce(&self) -> bool {
        (self.head - self.cached_tail) < self.capacity as u64
    }

    /// Number of entries available for production.
    pub fn available_slots(&self) -> u64 {
        self.capacity as u64 - (self.head - self.cached_tail)
    }

    /// Update the cached tail from the consumer (e.g., after RDMA read of tail).
    pub fn update_tail(&mut self, new_tail: u64) {
        self.cached_tail = new_tail;
    }

    /// Produce an entry: reserve a slot and return its ring index + sequence number.
    ///
    /// Returns None if the ring is full (call `update_tail` and retry).
    pub fn produce(&mut self) -> Option<ProduceResult> {
        if !self.can_produce() {
            return None;
        }
        let index = (self.head & (self.capacity as u64 - 1)) as u32;
        let seq = self.next_sequence;
        self.head += 1;
        self.next_sequence += 1;
        Some(ProduceResult {
            ring_index: index,
            sequence: seq,
            new_head: self.head,
        })
    }

    /// Current head value (to write as doorbell).
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Current cached tail.
    pub fn cached_tail(&self) -> u64 {
        self.cached_tail
    }
}

/// Result of a successful produce operation.
#[derive(Debug, Clone, Copy)]
pub struct ProduceResult {
    /// Index in the ring entry array.
    pub ring_index: u32,
    /// Sequence number for this entry.
    pub sequence: u32,
    /// Updated head value to write as doorbell.
    pub new_head: u64,
}

// ---------------------------------------------------------------------------
// Backpressure State Machine
// ---------------------------------------------------------------------------

/// Backpressure state for a ring buffer producer.
///
/// Implements the exponential backoff protocol: 1us, 2us, 4us, ... up to 100us.
/// After 10ms total wait, reports congestion.
#[derive(Debug, Clone)]
pub struct BackpressureState {
    /// Current backoff delay in microseconds.
    pub current_backoff_us: u64,
    /// Maximum backoff delay.
    pub max_backoff_us: u64,
    /// Total time spent in backpressure (microseconds).
    pub total_wait_us: u64,
    /// Congestion timeout (microseconds). After this, report congestion.
    pub congestion_timeout_us: u64,
    /// Whether congestion has been reported.
    pub congestion_reported: bool,
    /// Number of backoff iterations.
    pub iterations: u32,
}

impl BackpressureState {
    /// Create a new backpressure state with default parameters.
    pub fn new() -> Self {
        Self {
            current_backoff_us: 1,
            max_backoff_us: 100,
            total_wait_us: 0,
            congestion_timeout_us: 10_000, // 10ms
            congestion_reported: false,
            iterations: 0,
        }
    }

    /// Step the backpressure: double the backoff and accumulate wait time.
    ///
    /// Returns `true` if the congestion timeout has been reached.
    pub fn step(&mut self) -> bool {
        self.total_wait_us += self.current_backoff_us;
        self.iterations += 1;
        self.current_backoff_us = (self.current_backoff_us * 2).min(self.max_backoff_us);

        if self.total_wait_us >= self.congestion_timeout_us && !self.congestion_reported {
            self.congestion_reported = true;
            return true;
        }
        false
    }

    /// Reset backpressure state (called when ring becomes available).
    pub fn reset(&mut self) {
        self.current_backoff_us = 1;
        self.total_wait_us = 0;
        self.congestion_reported = false;
        self.iterations = 0;
    }

    /// Whether we are currently in a congested state.
    pub fn is_congested(&self) -> bool {
        self.congestion_reported
    }
}

impl Default for BackpressureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Persistent Kernel Manager (host-side lifecycle)
// ---------------------------------------------------------------------------

/// Host-side manager for a persistent GPU kernel.
///
/// Handles the lifecycle (launch, monitor, shutdown, recovery) and holds
/// references to all VRAM-allocated structures.
///
/// # TODO: requires hardware
/// The actual CUDA operations (cuLaunchKernel, cudaMalloc, etc.) require
/// a real GPU. This struct manages the metadata and state machine.
pub struct PersistentKernelManager {
    /// GPU this kernel runs on.
    pub gpu_id: u8,

    /// VRAM-allocated ring buffers.
    pub input_ring: RingBuffer,
    pub output_ring: RingBuffer,

    /// Data buffer pools.
    pub input_pool: DataPool,
    pub output_pool: DataPool,

    /// Kernel lifecycle state.
    pub state: KernelState,

    /// Configuration.
    pub config: PersistentKernelConfig,

    /// Statistics.
    pub stats: Arc<PersistentKernelStats>,

    /// Recovery attempt counter.
    recovery_attempts: u8,
}

impl PersistentKernelManager {
    /// Create a new persistent kernel manager.
    ///
    /// This does NOT allocate VRAM or launch the kernel. Call `allocate()` then
    /// `launch()` separately.
    ///
    /// `base_device_ptr` is the starting VRAM address for all allocations.
    /// In production, this would come from `cudaMalloc`.
    pub fn new(gpu_id: u8, config: PersistentKernelConfig, base_device_ptr: u64) -> Self {
        let input_ring_size = RingBuffer::compute_alloc_size(config.input_ring_size);
        let output_ring_offset = base_device_ptr + input_ring_size as u64;
        let output_ring_size = RingBuffer::compute_alloc_size(config.output_ring_size);

        let input_pool_offset = output_ring_offset + output_ring_size as u64;
        let input_pool_size = config.data_slot_size * config.input_slot_count as usize;

        let output_pool_offset = input_pool_offset + input_pool_size as u64;

        Self {
            gpu_id,
            input_ring: RingBuffer::new(base_device_ptr, config.input_ring_size),
            output_ring: RingBuffer::new(output_ring_offset, config.output_ring_size),
            input_pool: DataPool::new(
                input_pool_offset,
                config.data_slot_size,
                config.input_slot_count,
            ),
            output_pool: DataPool::new(
                output_pool_offset,
                config.data_slot_size,
                config.output_slot_count,
            ),
            state: KernelState::NotLaunched,
            config,
            stats: Arc::new(PersistentKernelStats::new()),
            recovery_attempts: 0,
        }
    }

    /// Get the pinned region descriptors for R10 integration.
    pub fn pinned_regions(&self) -> PersistentRegionPin {
        // Control block is after the output data pool
        let control_addr = self.output_pool.device_ptr + self.output_pool.total_size as u64;
        PersistentRegionPin::from_allocations(
            self.gpu_id,
            &self.input_ring,
            &self.output_ring,
            &self.input_pool,
            &self.output_pool,
            control_addr,
        )
    }

    /// Request the kernel to shut down gracefully.
    ///
    /// In production, this writes `shutdown_requested = 1` to the VRAM control block.
    /// The kernel checks this flag each iteration and exits its main loop.
    pub fn request_shutdown(&mut self) {
        if matches!(self.state, KernelState::Running { .. }) {
            self.state = KernelState::ShuttingDown {
                requested_at_epoch_ms: 0, // TODO: real timestamp
            };
        }
    }

    /// Mark the kernel as failed.
    pub fn mark_failed(&mut self, error_code: u32) {
        self.state = KernelState::Failed {
            error_code,
            failed_at_epoch_ms: 0,
        };
    }

    /// Attempt recovery after a failure.
    ///
    /// Returns true if recovery should be attempted, false if max attempts exceeded.
    pub fn attempt_recovery(&mut self) -> bool {
        if self.recovery_attempts >= self.config.max_recovery_attempts {
            return false;
        }
        self.recovery_attempts += 1;
        self.state = KernelState::Recovering;
        true
    }

    /// Mark the kernel as successfully launched.
    ///
    /// Note: does NOT reset recovery_attempts. The counter tracks total recovery
    /// attempts across the kernel's lifetime. Only creating a new manager resets it.
    pub fn mark_launched(&mut self) {
        self.state = KernelState::Running {
            launched_at_epoch_ms: 0, // TODO: real timestamp
        };
    }

    /// Whether the kernel is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self.state, KernelState::Running { .. })
    }

    /// Total VRAM allocated by this kernel.
    pub fn total_vram_bytes(&self) -> usize {
        self.config.total_vram_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RingBuffer tests --

    #[test]
    fn test_ring_buffer_new_rounds_capacity() {
        let ring = RingBuffer::new(0x1000, 200);
        assert_eq!(ring.capacity, 256); // Rounded up to next power of 2
    }

    #[test]
    fn test_ring_buffer_new_power_of_two_unchanged() {
        let ring = RingBuffer::new(0x1000, 256);
        assert_eq!(ring.capacity, 256);
    }

    #[test]
    fn test_ring_buffer_alloc_size_page_aligned() {
        let size = RingBuffer::compute_alloc_size(256);
        assert_eq!(size % 4096, 0); // Page aligned
        // 128 header + 256*48 = 12416 bytes, rounded to 16384 (4 pages)
        assert_eq!(size, 16384);
    }

    #[test]
    fn test_ring_buffer_descriptor_offset() {
        let ring = RingBuffer::new(0x1000, 256);
        assert_eq!(ring.descriptor_offset(0), 128); // After header
        assert_eq!(ring.descriptor_offset(1), 128 + 48);
        assert_eq!(ring.descriptor_offset(2), 128 + 96);
    }

    #[test]
    fn test_ring_buffer_descriptor_addr() {
        let ring = RingBuffer::new(0x1000, 256);
        assert_eq!(ring.descriptor_addr(0), 0x1000 + 128);
    }

    #[test]
    fn test_ring_buffer_bar1_addr_none_when_unregistered() {
        let ring = RingBuffer::new(0x1000, 256);
        assert!(ring.descriptor_bar1_addr(0).is_none());
        assert!(!ring.has_bar1());
    }

    #[test]
    fn test_ring_buffer_bar1_addr_when_registered() {
        let mut ring = RingBuffer::new(0x1000, 256);
        ring.bar1_addr = 0xF000_0000;
        assert!(ring.has_bar1());
        assert_eq!(
            ring.descriptor_bar1_addr(0),
            Some(0xF000_0000 + 128)
        );
    }

    #[test]
    fn test_ring_buffer_wrap_index() {
        let ring = RingBuffer::new(0x1000, 256);
        assert_eq!(ring.wrap_index(0), 0);
        assert_eq!(ring.wrap_index(255), 255);
        assert_eq!(ring.wrap_index(256), 0); // Wraps
        assert_eq!(ring.wrap_index(257), 1);
        assert_eq!(ring.wrap_index(512), 0);
    }

    #[test]
    fn test_ring_buffer_full_and_available() {
        let ring = RingBuffer::new(0x1000, 256);
        assert!(!ring.is_full(0, 0));
        assert!(!ring.is_full(100, 0));
        assert!(!ring.is_full(255, 0));
        assert!(ring.is_full(256, 0));
        assert!(ring.is_full(300, 44));

        assert_eq!(ring.entries_available(100, 50), 50);
        assert_eq!(ring.free_slots(100, 50), 206);
    }

    // -- RingHeader tests --

    #[test]
    fn test_ring_header_size() {
        assert_eq!(std::mem::size_of::<RingHeader>(), 128);
    }

    #[test]
    fn test_ring_header_new_zeroed() {
        let header = RingHeader::new();
        assert_eq!(header.head, 0);
        assert_eq!(header.tail, 0);
    }

    // -- DoorbellDescriptor tests --

    #[test]
    fn test_descriptor_size() {
        // Actual size may include alignment padding, but check it's reasonable
        assert!(std::mem::size_of::<DoorbellDescriptor>() >= 48);
    }

    #[test]
    fn test_descriptor_new() {
        let desc = DoorbellDescriptor::new(0x2000, 4096, 42);
        assert_eq!(desc.data_offset, 0x2000);
        assert_eq!(desc.data_size, 4096);
        assert_eq!(desc.sequence, 42);
        assert_eq!(desc.flags, 0);
    }

    #[test]
    fn test_descriptor_flags() {
        let mut desc = DoorbellDescriptor::new(0, 0, 0);
        desc.flags = descriptor_flags::COMPRESSED | descriptor_flags::URGENT;
        assert!(desc.is_compressed());
        assert!(desc.is_urgent());
        assert!(!desc.is_scatter_gather());
        assert!(!desc.is_last_in_batch());
        assert!(!desc.has_checksum());
    }

    #[test]
    fn test_descriptor_all_flags() {
        let mut desc = DoorbellDescriptor::new(0, 0, 0);
        desc.flags = descriptor_flags::COMPRESSED
            | descriptor_flags::SCATTER_GATHER
            | descriptor_flags::LAST_IN_BATCH
            | descriptor_flags::URGENT
            | descriptor_flags::CHECKSUM;
        assert!(desc.is_compressed());
        assert!(desc.is_scatter_gather());
        assert!(desc.is_last_in_batch());
        assert!(desc.is_urgent());
        assert!(desc.has_checksum());
    }

    #[test]
    fn test_descriptor_serialization() {
        let desc = DoorbellDescriptor::new(0x1000, 512, 99);
        let encoded = bincode::serialize(&desc).expect("serialize");
        let decoded: DoorbellDescriptor = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(desc, decoded);
    }

    // -- DataPool tests --

    #[test]
    fn test_data_pool_new() {
        let pool = DataPool::new(0x10000, 65536, 256);
        assert_eq!(pool.slot_size, 65536);
        assert_eq!(pool.slot_count, 256);
        assert_eq!(pool.total_size, 65536 * 256);
    }

    #[test]
    fn test_data_pool_slot_addr() {
        let pool = DataPool::new(0x10000, 65536, 256);
        assert_eq!(pool.slot_addr(0), 0x10000);
        assert_eq!(pool.slot_addr(1), 0x10000 + 65536);
        assert_eq!(pool.slot_addr(10), 0x10000 + 10 * 65536);
    }

    #[test]
    fn test_data_pool_slot_bar1_addr() {
        let mut pool = DataPool::new(0x10000, 65536, 256);
        assert!(pool.slot_bar1_addr(0).is_none());

        pool.bar1_addr = 0xA000_0000;
        assert_eq!(pool.slot_bar1_addr(0), Some(0xA000_0000));
        assert_eq!(pool.slot_bar1_addr(1), Some(0xA000_0000 + 65536));
    }

    #[test]
    fn test_data_pool_slot_offset() {
        let pool = DataPool::new(0x10000, 1024, 10);
        assert_eq!(pool.slot_offset(0), 0);
        assert_eq!(pool.slot_offset(5), 5 * 1024);
    }

    #[test]
    fn test_data_pool_is_valid_slot() {
        let pool = DataPool::new(0x10000, 1024, 10);
        assert!(pool.is_valid_slot(0));
        assert!(pool.is_valid_slot(9));
        assert!(!pool.is_valid_slot(10));
        assert!(!pool.is_valid_slot(100));
    }

    // -- ControlBlock tests --

    #[test]
    fn test_control_block_size() {
        // 2 cache lines: first has heartbeat+pad, second has shutdown+error+counters+pad
        // align(64) rounds to multiple of 64
        let size = std::mem::size_of::<ControlBlock>();
        assert_eq!(size % 64, 0); // Cache-line aligned
        assert_eq!(size, 128); // 2 cache lines
    }

    #[test]
    fn test_control_block_new() {
        let cb = ControlBlock::new();
        assert_eq!(cb.heartbeat, 0);
        assert!(!cb.is_shutdown_requested());
        assert!(!cb.has_error());
    }

    #[test]
    fn test_control_block_shutdown() {
        let mut cb = ControlBlock::new();
        cb.shutdown_requested = 1;
        assert!(cb.is_shutdown_requested());
    }

    #[test]
    fn test_control_block_error() {
        let mut cb = ControlBlock::new();
        cb.error_code = 42;
        assert!(cb.has_error());
    }

    // -- PersistentKernelConfig tests --

    #[test]
    fn test_config_default() {
        let config = PersistentKernelConfig::default();
        assert_eq!(config.input_ring_size, 256);
        assert_eq!(config.output_ring_size, 256);
        assert_eq!(config.data_slot_size, 65536);
        assert_eq!(config.threads_per_block, 256);
        assert!(config.max_iterations.is_none());
    }

    #[test]
    fn test_config_dev() {
        let config = PersistentKernelConfig::dev();
        assert_eq!(config.input_ring_size, 16);
        assert_eq!(config.max_iterations, Some(1000));
    }

    #[test]
    fn test_config_total_vram_default() {
        let config = PersistentKernelConfig::default();
        let total = config.total_vram_bytes();
        // Input ring: ~16 KB, Output ring: ~16 KB
        // Input pool: 256 * 64 KB = 16 MB, Output pool: 256 * 64 KB = 16 MB
        // Control: ~192 bytes
        // Total: ~32 MB + change
        assert!(total > 32_000_000);
        assert!(total < 34_000_000);
    }

    // -- PersistentKernelStats tests --

    #[test]
    fn test_stats_new_zeroed() {
        let stats = PersistentKernelStats::new();
        assert_eq!(stats.batches_processed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.ring_full_stalls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_snapshot() {
        let stats = PersistentKernelStats::new();
        stats.batches_processed.store(100, Ordering::Relaxed);
        stats.bytes_ingested.store(1_000_000, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.batches_processed, 100);
        assert_eq!(snap.bytes_ingested, 1_000_000);
    }

    // -- PinnedRegion tests --

    #[test]
    fn test_pinned_regions_from_allocations() {
        let input_ring = RingBuffer::new(0x1000, 256);
        let output_ring = RingBuffer::new(0x5000, 256);
        let input_pool = DataPool::new(0x10000, 1024, 16);
        let output_pool = DataPool::new(0x20000, 1024, 16);

        let pins =
            PersistentRegionPin::from_allocations(0, &input_ring, &output_ring, &input_pool, &output_pool, 0x30000);

        assert_eq!(pins.gpu_id, 0);
        assert_eq!(pins.regions.len(), 5);
        assert!(pins.total_pinned_bytes() > 0);
    }

    #[test]
    fn test_pinned_regions_overlap_detection() {
        let input_ring = RingBuffer::new(0x1000, 16);
        let output_ring = RingBuffer::new(0x5000, 16);
        let input_pool = DataPool::new(0x10000, 1024, 4);
        let output_pool = DataPool::new(0x20000, 1024, 4);
        let pins =
            PersistentRegionPin::from_allocations(0, &input_ring, &output_ring, &input_pool, &output_pool, 0x30000);

        // Should overlap with input ring region
        assert!(pins.overlaps(0x1000, 100));
        // Should not overlap with address before all regions
        assert!(!pins.overlaps(0x0000, 0x100));
        // Should overlap with control block
        assert!(pins.overlaps(0x30000, 1));
    }

    // -- RingProducer tests --

    #[test]
    fn test_ring_producer_new() {
        let prod = RingProducer::new(256);
        assert_eq!(prod.head(), 0);
        assert_eq!(prod.cached_tail(), 0);
        assert!(prod.can_produce());
        assert_eq!(prod.available_slots(), 256);
    }

    #[test]
    fn test_ring_producer_produce() {
        let mut prod = RingProducer::new(256);
        let result = prod.produce().expect("should produce");
        assert_eq!(result.ring_index, 0);
        assert_eq!(result.sequence, 0);
        assert_eq!(result.new_head, 1);
        assert_eq!(prod.head(), 1);
    }

    #[test]
    fn test_ring_producer_produce_wraps() {
        let mut prod = RingProducer::new(4);
        // Fill 3 slots
        for i in 0..3 {
            let r = prod.produce().expect("should produce");
            assert_eq!(r.ring_index, i);
        }
        // Advance tail to free slots
        prod.update_tail(2);
        // Produce 2 more (indices 3, 0)
        let r = prod.produce().expect("should produce");
        assert_eq!(r.ring_index, 3);
        let r = prod.produce().expect("should produce");
        assert_eq!(r.ring_index, 0); // Wrapped!
    }

    #[test]
    fn test_ring_producer_full() {
        let mut prod = RingProducer::new(4);
        // Fill all 4 slots
        for _ in 0..4 {
            assert!(prod.produce().is_some());
        }
        // Ring is now full
        assert!(!prod.can_produce());
        assert!(prod.produce().is_none());

        // Free one slot
        prod.update_tail(1);
        assert!(prod.can_produce());
        assert!(prod.produce().is_some());
    }

    #[test]
    fn test_ring_producer_rounds_capacity() {
        let prod = RingProducer::new(5); // Not power of 2
        assert_eq!(prod.available_slots(), 8); // Rounded to 8
    }

    // -- BackpressureState tests --

    #[test]
    fn test_backpressure_new() {
        let bp = BackpressureState::new();
        assert_eq!(bp.current_backoff_us, 1);
        assert_eq!(bp.total_wait_us, 0);
        assert!(!bp.is_congested());
    }

    #[test]
    fn test_backpressure_exponential_backoff() {
        let mut bp = BackpressureState::new();
        bp.step();
        assert_eq!(bp.current_backoff_us, 2); // Doubled from 1
        bp.step();
        assert_eq!(bp.current_backoff_us, 4);
        bp.step();
        assert_eq!(bp.current_backoff_us, 8);
    }

    #[test]
    fn test_backpressure_max_backoff() {
        let mut bp = BackpressureState::new();
        // Step many times to exceed max
        for _ in 0..20 {
            bp.step();
        }
        assert!(bp.current_backoff_us <= bp.max_backoff_us);
    }

    #[test]
    fn test_backpressure_congestion_detection() {
        let mut bp = BackpressureState::new();
        bp.congestion_timeout_us = 10; // Low timeout for testing
        let mut congestion_hit = false;
        for _ in 0..100 {
            if bp.step() {
                congestion_hit = true;
                break;
            }
        }
        assert!(congestion_hit);
        assert!(bp.is_congested());
    }

    #[test]
    fn test_backpressure_reset() {
        let mut bp = BackpressureState::new();
        bp.congestion_timeout_us = 5;
        while !bp.step() {}
        assert!(bp.is_congested());

        bp.reset();
        assert!(!bp.is_congested());
        assert_eq!(bp.current_backoff_us, 1);
        assert_eq!(bp.total_wait_us, 0);
    }

    // -- PersistentKernelManager tests --

    #[test]
    fn test_manager_new() {
        let config = PersistentKernelConfig::dev();
        let mgr = PersistentKernelManager::new(0, config, 0x10000);
        assert_eq!(mgr.gpu_id, 0);
        assert_eq!(mgr.state, KernelState::NotLaunched);
        assert!(!mgr.is_running());
    }

    #[test]
    fn test_manager_lifecycle() {
        let config = PersistentKernelConfig::dev();
        let mut mgr = PersistentKernelManager::new(0, config, 0x10000);

        // Launch
        mgr.mark_launched();
        assert!(mgr.is_running());

        // Shutdown
        mgr.request_shutdown();
        assert!(!mgr.is_running());
        assert!(matches!(mgr.state, KernelState::ShuttingDown { .. }));
    }

    #[test]
    fn test_manager_failure_and_recovery() {
        let mut config = PersistentKernelConfig::dev();
        config.max_recovery_attempts = 2;
        let mut mgr = PersistentKernelManager::new(0, config, 0x10000);

        mgr.mark_launched();
        mgr.mark_failed(1);
        assert!(matches!(mgr.state, KernelState::Failed { error_code: 1, .. }));

        // First recovery
        assert!(mgr.attempt_recovery());
        assert_eq!(mgr.state, KernelState::Recovering);
        mgr.mark_launched();

        // Second failure + recovery
        mgr.mark_failed(2);
        assert!(mgr.attempt_recovery());
        mgr.mark_launched();

        // Third failure -- max attempts exceeded
        mgr.mark_failed(3);
        assert!(!mgr.attempt_recovery());
    }

    #[test]
    fn test_manager_pinned_regions() {
        let config = PersistentKernelConfig::dev();
        let mgr = PersistentKernelManager::new(0, config, 0x10000);
        let pins = mgr.pinned_regions();
        assert_eq!(pins.gpu_id, 0);
        assert_eq!(pins.regions.len(), 5);
        assert!(pins.total_pinned_bytes() > 0);
    }

    #[test]
    fn test_manager_vram_usage() {
        let config = PersistentKernelConfig::dev();
        let mgr = PersistentKernelManager::new(0, config.clone(), 0x10000);
        assert_eq!(mgr.total_vram_bytes(), config.total_vram_bytes());
    }

    #[test]
    fn test_manager_shutdown_only_when_running() {
        let config = PersistentKernelConfig::dev();
        let mut mgr = PersistentKernelManager::new(0, config, 0x10000);

        // Shutdown request on NotLaunched should be a no-op
        mgr.request_shutdown();
        assert_eq!(mgr.state, KernelState::NotLaunched);
    }

    #[test]
    fn test_manager_allocations_are_contiguous() {
        let config = PersistentKernelConfig::dev();
        let base = 0x10000u64;
        let mgr = PersistentKernelManager::new(0, config, base);

        // Input ring starts at base
        assert_eq!(mgr.input_ring.device_ptr, base);
        // Output ring starts after input ring
        assert!(mgr.output_ring.device_ptr > mgr.input_ring.device_ptr);
        // Input pool starts after output ring
        assert!(mgr.input_pool.device_ptr > mgr.output_ring.device_ptr);
        // Output pool starts after input pool
        assert!(mgr.output_pool.device_ptr > mgr.input_pool.device_ptr);
    }
}
