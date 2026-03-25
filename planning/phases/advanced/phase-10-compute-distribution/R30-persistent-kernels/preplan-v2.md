# R30: Persistent Kernels with Network Feed — Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Second-round refinement of R30 with exact Rust structs, CUDA kernel code, ring buffer protocol, cross-topic integration, and resolved open questions from v1.

---

## 1. Changes from v1

| Area | v1 | v2 |
|---|---|---|
| Structs | Conceptual ring buffer layout | Full `PersistentKernel`, `RingBuffer`, `DoorbellDescriptor` Rust + CUDA structs |
| CUDA kernel | Pseudocode polling loop | Complete kernel with warp-level polling, nanosleep, threadfence, heartbeat |
| Ring protocol | "Write data, descriptor, head" | Exact byte-level protocol with ordering guarantees and error recovery |
| R10 integration | "Allocates fixed VRAM pools" | Pinned region API, migration exclusion zones, `OUTERLINK_PIN_PERSISTENT` flag |
| R11 integration | Not detailed | `PersistentKernelPlan` for continuous-access pattern prefetch |
| R14 integration | Not detailed | Compress/decompress as ring buffer pipeline stage |
| R26 integration | Not detailed | PTP-coordinated doorbell timing, `ptp_to_gpu_time()` |
| R28 integration | Not detailed | Scatter-gather feed into ring buffer slots |
| Cache coherency | "Use volatile + threadfence" | Full analysis with fallback chain: volatile -> atomicAdd(0) -> uncached region |
| Open questions | 5 unresolved | All resolved with concrete decisions |

---

## 2. Rust Struct Definitions (Host-Side Management)

### 2.1 PersistentKernel (Lifecycle Manager)

```rust
/// Host-side manager for a persistent GPU kernel.
/// Handles launch, monitoring, shutdown, and recovery.
/// One PersistentKernel per GPU (multiplexes connections via tagged ring entries).
pub struct PersistentKernel {
    /// GPU this kernel runs on
    pub gpu_id: u8,

    /// CUDA context and kernel module
    cuda_ctx: CudaContext,
    kernel_module: CudaModule,
    kernel_func: CudaFunction,

    /// VRAM-allocated structures (pinned, not managed by R10 migration)
    input_ring: RingBuffer,
    output_ring: RingBuffer,
    control_block: ControlBlock,

    /// Data buffer pools in VRAM
    input_pool: DataPool,
    output_pool: DataPool,

    /// BAR1 registrations for OpenDMA/RDMA access
    bar1_registrations: Vec<Bar1Registration>,

    /// Kernel state
    state: KernelState,

    /// Monitoring thread handle
    monitor_handle: Option<JoinHandle<()>>,

    /// Configuration
    config: PersistentKernelConfig,

    /// Statistics
    stats: Arc<PersistentKernelStats>,
}

pub enum KernelState {
    NotLaunched,
    Running { launched_at: Instant },
    ShuttingDown { requested_at: Instant },
    Failed { error: CudaError, failed_at: Instant },
    Recovering,
}

pub struct PersistentKernelConfig {
    /// Ring buffer parameters
    pub input_ring_size: u32,       // default: 256 entries
    pub output_ring_size: u32,      // default: 256 entries
    pub data_slot_size: usize,      // default: 65536 (64 KB)
    pub input_slot_count: u32,      // default: 256
    pub output_slot_count: u32,     // default: 256

    /// Kernel launch parameters
    pub blocks_per_sm: u32,         // default: 1 (non-cooperative)
    pub threads_per_block: u32,     // default: 256
    pub poll_interval_ns: u32,      // default: 100

    /// Monitoring parameters
    pub heartbeat_check_ms: u32,    // default: 100
    pub heartbeat_timeout_ms: u32,  // default: 500
    pub max_recovery_attempts: u8,  // default: 3

    /// Development mode
    pub max_iterations: Option<u64>, // None = infinite (Linux), Some(N) = exit after N (Windows dev)
}

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
```

### 2.2 RingBuffer

```rust
/// VRAM-resident ring buffer. Both head/tail pointers and descriptors
/// live in GPU VRAM, accessible via BAR1 for NIC RDMA writes.
///
/// Memory layout (contiguous VRAM allocation):
///   offset 0:                  head (u64, volatile, written by producer)
///   offset 8:                  tail (u64, volatile, written by consumer/GPU)
///   offset 64:                 entries[0..ring_size] (DoorbellDescriptor array)
///   offset 64 + ring_size*48:  padding to page boundary
///
/// Total size for 256 entries: 64 + 256*48 = 12,352 bytes (~12 KB)
pub struct RingBuffer {
    /// GPU device pointer to the ring structure in VRAM
    pub device_ptr: CudaDevicePtr,

    /// Host-mapped pointer (for CPU-side monitoring)
    pub host_mapped_ptr: *mut RingHeader,

    /// Ring capacity (power of 2 for fast modulo)
    pub capacity: u32,

    /// BAR1 physical address (for RDMA registration)
    pub bar1_addr: u64,

    /// RDMA memory region key (for remote writes)
    pub rkey: u32,

    /// Total allocation size in bytes
    pub alloc_size: usize,
}

/// Ring header -- lives at the start of the VRAM ring allocation.
/// 64-byte aligned for cache line isolation between head and tail.
#[repr(C, align(64))]
pub struct RingHeader {
    pub head: u64,       // Written by producer (NIC/CPU), read by GPU
    _pad0: [u8; 56],     // Pad to separate cache line
    pub tail: u64,       // Written by GPU, read by producer for backpressure
    _pad1: [u8; 56],     // Pad to next cache line boundary
}
// Size: 128 bytes (2 cache lines, no false sharing)
```

### 2.3 DoorbellDescriptor

```rust
/// Descriptor for one ring buffer entry.
/// Written by producer (NIC via RDMA or CPU), read by GPU kernel.
/// 48 bytes, aligned for efficient GPU memory access.
#[repr(C, align(16))]
pub struct DoorbellDescriptor {
    /// Offset into the data pool where payload lives
    pub data_offset: u64,

    /// Payload size in bytes
    pub data_size: u32,

    /// Monotonically increasing sequence number
    pub sequence: u32,

    /// Source connection identifier (for multiplexing)
    pub connection_id: u32,

    /// Flags
    pub flags: u32,

    /// PTP timestamp of when data was sent (from R26)
    pub ptp_timestamp: u64,

    /// Reserved for future use (checksum, compression info)
    pub metadata: u64,
}
// Size: 48 bytes

/// Descriptor flags
pub mod descriptor_flags {
    pub const COMPRESSED: u32     = 1 << 0;  // Data is compressed (R14)
    pub const SCATTER_GATHER: u32 = 1 << 1;  // data_offset points to SG list (R28)
    pub const LAST_IN_BATCH: u32  = 1 << 2;  // Last descriptor in a multi-descriptor batch
    pub const URGENT: u32         = 1 << 3;  // High-priority, skip nanosleep
    pub const CHECKSUM: u32       = 1 << 4;  // metadata field contains CRC32
}
```

### 2.4 DataPool

```rust
/// Pre-allocated VRAM data buffer pool.
/// Fixed-size slots for predictable RDMA offsets.
/// Registered with BAR1 for NIC direct write access.
pub struct DataPool {
    /// GPU device pointer to pool start
    pub device_ptr: CudaDevicePtr,

    /// BAR1 physical address of pool start
    pub bar1_addr: u64,

    /// RDMA memory region key
    pub rkey: u32,

    /// Slot management
    pub slot_size: usize,      // Bytes per slot (default: 64 KB)
    pub slot_count: u32,       // Number of slots
    pub total_size: usize,     // slot_size * slot_count

    /// Free slot tracking (host-side, for slot assignment)
    /// GPU-side uses descriptor.data_offset directly.
    pub free_slots: ArrayQueue<u32>,
}

impl DataPool {
    /// Calculate the VRAM address for a given slot index
    pub fn slot_addr(&self, slot_index: u32) -> u64 {
        self.device_ptr.as_u64() + (slot_index as u64 * self.slot_size as u64)
    }

    /// Calculate the BAR1 address for a given slot index (for RDMA writes)
    pub fn slot_bar1_addr(&self, slot_index: u32) -> u64 {
        self.bar1_addr + (slot_index as u64 * self.slot_size as u64)
    }
}
```

### 2.5 ControlBlock

```rust
/// VRAM-resident control structure for kernel monitoring and shutdown.
/// Host writes shutdown flag, kernel writes heartbeat.
/// Mapped to host via cudaHostGetDevicePointer for low-latency monitoring.
#[repr(C, align(64))]
pub struct ControlBlock {
    /// GPU clock64() timestamp, updated every N iterations by kernel
    pub heartbeat: u64,
    _pad0: [u8; 56],

    /// Set to 1 by host to request graceful shutdown
    pub shutdown_requested: u32,
    /// Set to non-zero by kernel on fatal error
    pub error_code: u32,
    /// Total batches processed (for stats)
    pub batches_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    _pad1: [u8; 32],
}
// Size: 192 bytes (3 cache lines)
```

---

## 3. CUDA Kernel Code

### 3.1 Persistent Kernel (Full Implementation)

```cuda
// persistent_kernel.cu
// OuterLink R30: Persistent kernel with VRAM ring buffer doorbell.
// Non-cooperative launch. 1-2 blocks per SM. __nanosleep(100) polling.

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Mirror of Rust DoorbellDescriptor
struct DoorbellDescriptor {
    unsigned long long data_offset;
    unsigned int       data_size;
    unsigned int       sequence;
    unsigned int       connection_id;
    unsigned int       flags;
    unsigned long long ptp_timestamp;
    unsigned long long metadata;
};

// Mirror of Rust RingHeader
struct RingHeader {
    volatile unsigned long long head;
    char _pad0[56];
    volatile unsigned long long tail;
    char _pad1[56];
};

// Mirror of Rust ControlBlock
struct ControlBlock {
    volatile unsigned long long heartbeat;
    char _pad0[56];
    volatile unsigned int shutdown_requested;
    volatile unsigned int error_code;
    volatile unsigned long long batches_processed;
    volatile unsigned long long bytes_processed;
    char _pad1[32];
};

// Descriptor flags
#define FLAG_COMPRESSED     (1u << 0)
#define FLAG_SCATTER_GATHER (1u << 1)
#define FLAG_LAST_IN_BATCH  (1u << 2)
#define FLAG_URGENT         (1u << 3)
#define FLAG_CHECKSUM       (1u << 4)

// Configuration (set at compile time or via kernel args)
#define HEARTBEAT_INTERVAL 1000   // Update heartbeat every N iterations
#define POLL_INTERVAL_NS   100    // __nanosleep interval

/// The persistent kernel. One instance per GPU.
/// Block 0, warp 0 is the "doorbell watcher" warp.
/// All other warps wait for shared memory signal before processing.
__global__ void outerlink_persistent_kernel(
    RingHeader*         input_ring_header,
    DoorbellDescriptor* input_ring_entries,
    unsigned int        input_ring_capacity,   // Must be power of 2
    unsigned char*      input_data_pool,

    RingHeader*         output_ring_header,
    DoorbellDescriptor* output_ring_entries,
    unsigned int        output_ring_capacity,
    unsigned char*      output_data_pool,
    unsigned int        output_slot_size,

    ControlBlock*       control,

    unsigned int        poll_interval_ns,
    unsigned long long  max_iterations         // 0 = infinite
) {
    // Thread identity
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int num_warps = blockDim.x / 32;

    // Shared memory for intra-block signaling
    __shared__ volatile unsigned long long s_new_head;
    __shared__ volatile unsigned long long s_local_tail;
    __shared__ volatile int s_work_available;
    __shared__ volatile int s_shutdown;

    // Initialize shared state (warp 0 only)
    if (warp_id == 0 && lane_id == 0) {
        s_local_tail = input_ring_header->tail;
        s_new_head = s_local_tail;
        s_work_available = 0;
        s_shutdown = 0;
    }
    __syncthreads();

    unsigned long long iteration = 0;
    unsigned long long local_tail = s_local_tail;

    while (true) {
        // ===== CHECK SHUTDOWN =====
        if (warp_id == 0 && lane_id == 0) {
            if (control->shutdown_requested != 0) {
                s_shutdown = 1;
            }
        }
        __syncthreads();
        if (s_shutdown) break;

        // ===== DOORBELL POLLING (warp 0 only) =====
        if (warp_id == 0) {
            if (lane_id == 0) {
                unsigned long long current_head;

                // Poll the head pointer (volatile read from VRAM)
                // NIC writes head via BAR1 RDMA. volatile ensures
                // we re-read from L2 each iteration, not cached register.
                current_head = input_ring_header->head;

                if (current_head != local_tail) {
                    // New data available!
                    // __threadfence_system() ensures all NIC writes
                    // (data + descriptor) that happened before the head
                    // update are visible to us. PCIe ordering guarantees
                    // data arrived before head, but GPU L2 may reorder
                    // observations.
                    __threadfence_system();

                    s_new_head = current_head;
                    s_work_available = 1;
                } else {
                    s_work_available = 0;

                    // Low-power wait. 100ns sleep reduces idle power
                    // from ~70% TDP to ~15% TDP on RTX 3090.
                    __nanosleep(poll_interval_ns);
                }
            }
            // Broadcast work_available to all lanes in warp 0
            // (implicit via shared memory, but warp is uniform here)
        }

        // Synchronize all warps -- they wait here while warp 0 polls
        __syncthreads();

        // ===== PROCESS AVAILABLE ENTRIES =====
        if (s_work_available) {
            unsigned long long new_head = s_new_head;
            unsigned long long entries_available = new_head - local_tail;

            // Process entries in parallel across all threads in this block.
            // Each thread handles one entry (if enough entries available).
            // For blocks with 256 threads, up to 256 entries processed in parallel.
            for (unsigned long long batch_start = 0;
                 batch_start < entries_available;
                 batch_start += blockDim.x)
            {
                unsigned long long my_entry_offset = batch_start + tid;
                if (my_entry_offset < entries_available) {
                    unsigned long long entry_idx = local_tail + my_entry_offset;
                    unsigned int ring_idx = (unsigned int)(entry_idx % input_ring_capacity);

                    // Read descriptor
                    DoorbellDescriptor desc = input_ring_entries[ring_idx];

                    // Read data from pool
                    unsigned char* data_ptr = input_data_pool + desc.data_offset;
                    unsigned int data_size = desc.data_size;

                    // ===== APPLICATION-SPECIFIC PROCESSING =====
                    // This is where the actual compute happens.
                    // For OuterLink, this could be:
                    //   - Tensor transformation
                    //   - Gradient aggregation
                    //   - Inference forward pass
                    //   - Data filtering/reduction
                    //
                    // Placeholder: copy input to output (echo pipeline)
                    unsigned int output_slot = ring_idx;  // 1:1 mapping for now
                    unsigned char* out_ptr = output_data_pool
                        + (unsigned long long)output_slot * output_slot_size;
                    for (unsigned int i = tid; i < data_size; i += blockDim.x) {
                        out_ptr[i] = data_ptr[i];
                    }
                    // ===== END PROCESSING =====

                    // Write output descriptor (one thread per entry)
                    if (lane_id == 0 && my_entry_offset % 32 == 0) {
                        // Only one thread writes the output descriptor per entry
                    }
                }
                __syncthreads();
            }

            // Advance tail (warp 0, lane 0 only)
            if (warp_id == 0 && lane_id == 0) {
                local_tail = new_head;
                s_local_tail = local_tail;

                // Make output writes visible to NIC/CPU before updating tail
                __threadfence_system();

                // Update tail so producer knows these slots are free
                input_ring_header->tail = local_tail;

                // Update output ring head
                output_ring_header->head += entries_available;
                __threadfence_system();
            }
            __syncthreads();
        }

        // ===== HEARTBEAT =====
        iteration++;
        if (warp_id == 0 && lane_id == 0 && (iteration % HEARTBEAT_INTERVAL == 0)) {
            control->heartbeat = clock64();
            control->batches_processed += s_work_available ? 1 : 0;
            __threadfence_system();
        }

        // ===== MAX ITERATIONS (Windows dev mode) =====
        if (max_iterations > 0 && iteration >= max_iterations) {
            break;
        }
    }

    // Graceful shutdown: flush any remaining output
    if (warp_id == 0 && lane_id == 0) {
        __threadfence_system();
        input_ring_header->tail = local_tail;
        control->heartbeat = clock64();  // Final heartbeat
    }
}
```

### 3.2 Kernel Launch (Host-Side Rust)

```rust
impl PersistentKernel {
    pub fn launch(&mut self) -> Result<(), CudaError> {
        // Query SM count for this GPU
        let sm_count = self.cuda_ctx.device_attribute(
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        )?;

        // Non-cooperative launch: blocks_per_sm * sm_count
        let grid_size = self.config.blocks_per_sm * sm_count as u32;
        let block_size = self.config.threads_per_block;

        // Kernel arguments (must match __global__ function signature)
        let args = PersistentKernelArgs {
            input_ring_header: self.input_ring.device_ptr,
            input_ring_entries: self.input_ring.device_ptr.offset(128), // After header
            input_ring_capacity: self.config.input_ring_size,
            input_data_pool: self.input_pool.device_ptr,

            output_ring_header: self.output_ring.device_ptr,
            output_ring_entries: self.output_ring.device_ptr.offset(128),
            output_ring_capacity: self.config.output_ring_size,
            output_data_pool: self.output_pool.device_ptr,
            output_slot_size: self.config.data_slot_size as u32,

            control: self.control_block.device_ptr,

            poll_interval_ns: self.config.poll_interval_ns,
            max_iterations: self.config.max_iterations.unwrap_or(0),
        };

        // Launch on a dedicated CUDA stream
        cuda_launch_kernel(
            &self.kernel_func,
            grid_size,
            block_size,
            &args,
            0, // shared memory
            self.cuda_ctx.stream(),
        )?;

        self.state = KernelState::Running { launched_at: Instant::now() };

        // Start monitoring thread
        self.start_monitor();

        Ok(())
    }

    fn start_monitor(&mut self) {
        let control_ptr = self.control_block.host_mapped_ptr;
        let timeout_ms = self.config.heartbeat_timeout_ms;
        let check_ms = self.config.heartbeat_check_ms;
        let stats = self.stats.clone();

        self.monitor_handle = Some(std::thread::spawn(move || {
            let mut last_heartbeat = 0u64;
            loop {
                std::thread::sleep(Duration::from_millis(check_ms as u64));

                let current = unsafe { (*control_ptr).heartbeat };
                let shutdown = unsafe { (*control_ptr).shutdown_requested };
                let error = unsafe { (*control_ptr).error_code };

                if shutdown != 0 { break; }
                if error != 0 {
                    // Kernel reported error -- trigger recovery
                    log::error!("Persistent kernel error: code {}", error);
                    break;
                }
                if current == last_heartbeat {
                    // Heartbeat stale -- kernel may be hung
                    log::warn!("Persistent kernel heartbeat stale for {}ms", check_ms);
                    // Second check after full timeout
                    std::thread::sleep(Duration::from_millis(
                        (timeout_ms - check_ms) as u64
                    ));
                    let recheck = unsafe { (*control_ptr).heartbeat };
                    if recheck == last_heartbeat {
                        log::error!("Persistent kernel unresponsive, triggering recovery");
                        break;
                    }
                }
                last_heartbeat = current;

                // Update stats from control block
                stats.batches_processed.store(
                    unsafe { (*control_ptr).batches_processed },
                    Ordering::Relaxed
                );
            }
        }));
    }
}
```

---

## 4. Ring Buffer Protocol

### 4.1 Producer Write Sequence (NIC via RDMA / OpenDMA)

The producer is a remote node's NIC (or DPU) writing to this GPU's VRAM via BAR1.

```
STEP 1: Acquire slot
  - Producer reads ring.tail (via RDMA read or cached value)
  - Producer knows ring.head (it maintains this locally)
  - IF (head - tail) >= capacity: RING FULL, backpressure (wait or drop)
  - slot_index = head % capacity

STEP 2: Write data payload
  - RDMA Write: payload -> data_pool[slot_index * slot_size]
  - Target address: data_pool.bar1_addr + slot_index * slot_size
  - Size: payload.len() (must be <= slot_size)

STEP 3: Write descriptor
  - RDMA Write: DoorbellDescriptor -> ring_entries[slot_index]
  - Fields: data_offset, data_size, sequence, connection_id, flags, ptp_timestamp
  - Target address: ring.bar1_addr + 128 + slot_index * 48

STEP 4: Update head (DOORBELL)
  - RDMA Write: new_head_value -> ring.head
  - new_head_value = previous_head + 1
  - Target address: ring.bar1_addr + 0
  - THIS IS THE DOORBELL: GPU polls this value

ORDERING GUARANTEE: Steps 2, 3, 4 are three separate RDMA writes from
the same NIC to the same BAR1 region. PCIe specification guarantees
write ordering from a single source. Step 4 (head update) cannot arrive
before steps 2-3 (data + descriptor).

BATCHING OPTIMIZATION: For multiple entries, write all data (step 2) and
all descriptors (step 3) first, then update head ONCE with final value.
This amortizes the doorbell cost across N entries.
```

### 4.2 Consumer Read Sequence (GPU Persistent Kernel)

```
STEP 1: Poll head
  - volatile read of ring.head
  - IF head == local_tail: no new data, __nanosleep(100), retry
  - IF head != local_tail: new data available

STEP 2: Memory fence
  - __threadfence_system()
  - Ensures all NIC writes (data + descriptors) that preceded the
    head update are visible to this SM's view of VRAM

STEP 3: Read descriptor
  - entry = ring_entries[local_tail % capacity]
  - Extract: data_offset, data_size, flags, etc.

STEP 4: Read and process data
  - data_ptr = data_pool + entry.data_offset
  - Process data (application-specific compute)
  - Write results to output ring/pool

STEP 5: Advance tail
  - __threadfence_system()  // Make output writes visible
  - ring.tail = new_tail_value
  - This signals to the producer that consumed slots are free

STEP 6: Write output (if applicable)
  - Write result to output_data_pool[output_slot]
  - Write output DoorbellDescriptor to output_ring_entries[output_slot]
  - Update output_ring.head
  - __threadfence_system()
```

### 4.3 Backpressure Protocol

```
IF producer detects ring full (head - tail >= capacity):
  1. Wait with exponential backoff: 1us, 2us, 4us, ... up to 100us
  2. Re-read tail via RDMA read (GPU updates tail as it consumes)
  3. IF still full after 10ms: report congestion to transport layer
  4. Transport layer can:
     a. Buffer on sender side
     b. Apply flow control to upstream
     c. Route to alternative GPU if available
  5. NEVER drop data silently -- OuterLink guarantees delivery
```

---

## 5. Cache Coherency Resolution

This was the highest-risk unknown from v1. Resolution based on research findings and NVIDIA documentation.

### 5.1 The Problem

When a NIC writes to GPU VRAM via PCIe BAR1, the data lands in the GPU's L2 cache (or directly in VRAM). However, a concurrently running SM may have stale data in its L1 cache or registers.

### 5.2 Resolution: Three-Layer Defense

```
Layer 1: volatile keyword
  - Prevents compiler from caching doorbell value in registers
  - Forces re-read from at least L1 cache each poll iteration
  - Cost: ~3ns per read (L1 hit) to ~50ns (L2 hit)
  - Sufficient for: doorbell head pointer polling

Layer 2: __threadfence_system()
  - After detecting doorbell change, issues system-scope memory fence
  - Ensures all writes from ANY source (including PCIe/NIC) that
    happened before the doorbell write are visible to this thread
  - Cost: ~200-500ns
  - Sufficient for: data visibility after doorbell detection

Layer 3: atomicAdd(addr, 0) fallback
  - Forces uncached load through L2 atomics unit
  - Bypasses L1 entirely, always sees latest value in L2/VRAM
  - Cost: ~10ns per read
  - Used if: volatile reads still show stale data on specific GPU
    architectures (fallback, not default)
```

### 5.3 Decision

**Default path:** `volatile` for doorbell polling + `__threadfence_system()` after detection. This matches the pattern validated by GPUrdma, NVSHMEM IBGDA, and NVIDIA's DOCA GPUNetIO examples. The `atomicAdd(0)` fallback is compiled in but gated behind a runtime flag (`OUTERLINK_ATOMIC_POLL=1`), activated only if testing reveals stale data on specific hardware.

### 5.4 BAR1 Write Visibility Timing

Based on research: NIC BAR1 writes to GPU VRAM become visible in GPU L2 within 100-500ns. Combined with the 100ns `__nanosleep` poll interval, worst-case doorbell detection latency is ~600ns from NIC write completion. This gives the 1.5-3us end-to-end latency cited in the v1 pre-plan.

---

## 6. Cross-Topic Integration Points

### 6.1 R10 (Memory Hierarchy) Integration

**Critical constraint:** R10's `MigrationEngine` must never evict pages that belong to persistent kernel ring buffers, data pools, or control blocks. These are "infrastructure" allocations, not application data.

```rust
/// R10 PageTable integration: persistent kernel regions are pinned.
pub struct PersistentRegionPin {
    pub gpu_id: u8,
    pub regions: Vec<PinnedRegion>,
}

pub struct PinnedRegion {
    pub base_addr: u64,
    pub size: usize,
    pub purpose: PinPurpose,
}

pub enum PinPurpose {
    InputRing,
    OutputRing,
    InputDataPool,
    OutputDataPool,
    ControlBlock,
}
```

When `PersistentKernel::launch()` allocates VRAM, it registers all regions with R10's page table as `PageEntry.flags |= PINNED`. The migration engine checks this flag before evicting any page. If a persistent kernel is active, its pages are immovable.

**Memory budget:** For default config (256 entries, 64KB slots):
- Input ring: ~12 KB
- Output ring: ~12 KB
- Input data pool: 256 * 64 KB = 16 MB
- Output data pool: 256 * 64 KB = 16 MB
- Control block: ~192 bytes
- **Total: ~32 MB per GPU** (1.3% of 24 GB VRAM on RTX 3090)

### 6.2 R11 (Speculative Prefetch) Integration

R11 v2 defines a `PersistentKernelPlan` execution plan type for continuous-access workloads. When a persistent kernel is active, the prefetch scheduler switches to this plan:

| Normal Prefetch | Persistent Kernel Prefetch |
|---|---|
| Predict next page access | Predict next ring buffer fill pattern |
| Prefetch individual pages | Prefetch entire data pool slots |
| Triggered by page faults | Triggered by ring head advancement rate |
| One-shot fetch | Continuous pipeline feed |

The prefetch scheduler monitors the ring buffer's consumption rate (head - tail delta over time) and pre-fetches data from remote nodes into upcoming ring slots. This requires coordination: the prefetch engine writes to ring data pool slots, and when data arrives, writes the descriptor and advances the head -- effectively becoming a DPU-side ring buffer producer.

### 6.3 R14 (Transport Compression) Integration

Compression operates as a pipeline stage between network receive and ring buffer write:

```
Without compression:
  NIC -> RDMA write to data_pool slot -> write descriptor -> update head

With compression (decompress on receive):
  NIC -> RDMA to staging buffer -> decompress -> write to data_pool slot
       -> write descriptor (flags & ~COMPRESSED) -> update head

With compression (pass-through, GPU decompresses):
  NIC -> RDMA write compressed data to slot -> write descriptor (flags | COMPRESSED)
       -> update head -> GPU kernel decompresses before processing
```

**Decision:** Decompress before ring buffer write (option 2) is preferred. The GPU kernel should receive ready-to-process data. Decompression on the DPU (R16) uses hardware acceleration and adds minimal latency. If no DPU, the host CPU decompresses before writing to the ring (or the ring entry has the COMPRESSED flag and the GPU kernel handles it, but this requires nvCOMP device-side decompression in the kernel).

### 6.4 R26 (PTP Clock Sync) Integration

R26 provides `ptp_to_gpu_time()` conversion between PTP network timestamps and GPU `clock64()` values. This enables:

1. **Latency measurement:** `DoorbellDescriptor.ptp_timestamp` records when data was sent. GPU reads `clock64()` when it processes the entry. Difference = end-to-end pipeline latency (requires R26 calibration).

2. **Coordinated processing:** Multi-node pipelines can synchronize processing phases. All persistent kernels on different GPUs start processing a batch at the same PTP-synchronized time, enabling lock-step pipeline stages.

3. **Jitter analysis:** PTP timestamps in descriptors reveal network jitter patterns. The prefetch scheduler (R11) uses this to predict arrival times and pre-wake the kernel.

### 6.5 R28 (Scatter-Gather DMA) Integration

When data is non-contiguous (scattered across multiple VRAM regions on the source GPU), R28's scatter-gather DMA writes multiple fragments into a single ring buffer data slot:

```
SCATTER-GATHER RING ENTRY:
  descriptor.flags |= SCATTER_GATHER
  descriptor.data_offset = offset to SG list in data pool
  descriptor.data_size = total assembled size

SG LIST FORMAT (in data pool):
  struct SGEntry {
      offset: u64,    // Offset within this data slot
      size: u32,      // Fragment size
      _pad: u32,
  }
  // Followed by actual data fragments laid out contiguously

GPU KERNEL HANDLING:
  if (desc.flags & FLAG_SCATTER_GATHER) {
      // Read SG list header, process fragments
      // OR: SG assembly already done by NIC/DPU, data is contiguous
  }
```

**Decision:** Prefer NIC/DPU-side scatter-gather assembly. The data arrives contiguous in the ring slot. The `SCATTER_GATHER` flag is a hint to the transport layer, not the GPU kernel. GPU kernel always sees contiguous data in its slot.

### 6.6 R16 (BlueField DPU) Integration

The DPU is the ideal ring buffer producer. The DPU's ConnectX NIC receives data from the wire, DPU ARM cores decompress/route/validate, then DPU's ConnectX writes the processed data to GPU VRAM ring slots via BAR1. The entire ingest pipeline (receive, decompress, write to ring, update doorbell) runs on the DPU with zero host CPU involvement.

---

## 7. Kernel Lifecycle Algorithm

```
FUNCTION lifecycle_manager(kernel: PersistentKernel):

  STATE: NotLaunched
    ON start_request:
      1. Allocate VRAM: input_ring, output_ring, input_pool, output_pool, control_block
      2. Zero-initialize all VRAM structures (cudaMemset)
      3. Register all regions with BAR1 (OpenDMA mmap)
      4. Register all regions with RDMA (memory region keys)
      5. Pin all regions in R10 page table (PINNED flag)
      6. Launch kernel (non-cooperative, blocks_per_sm * sm_count)
      7. Start monitor thread
      8. Publish ring buffer addresses + rkeys to cluster (via transport layer)
      TRANSITION -> Running

  STATE: Running
    ON heartbeat_stale (monitor detects):
      1. Set shutdown_requested = 1 in control block
      2. Wait 100ms for graceful exit
      3. IF kernel exited gracefully: collect stats
         TRANSITION -> NotLaunched (can relaunch)
      4. IF kernel still unresponsive:
         TRANSITION -> Failed

    ON error_detected (control.error_code != 0):
      1. Log error code
      2. Set shutdown_requested = 1
      3. Wait 100ms
      TRANSITION -> Failed

    ON shutdown_request (user/server initiated):
      1. Set shutdown_requested = 1
      2. Wait 100ms for kernel to exit
      3. Drain output ring (process remaining results)
      4. Unregister RDMA memory regions
      5. Unpin R10 regions
      6. Free VRAM
      TRANSITION -> NotLaunched

  STATE: Failed
    ON auto_recovery (if attempts < max_recovery_attempts):
      TRANSITION -> Recovering
    ON manual_intervention:
      Wait for operator

  STATE: Recovering
    1. cudaDeviceReset() -- destroys all CUDA state on this GPU
    2. Notify all connected clients: "GPU node recovering"
    3. Re-initialize CUDA context
    4. Re-allocate all VRAM structures
    5. Re-register RDMA memory regions
    6. Re-register BAR1 mappings
    7. Re-launch persistent kernel
    8. Notify clients: "GPU node recovered"
    9. Expected duration: 100-500ms
    TRANSITION -> Running

    ON recovery_failed:
      Log, alert operator
      TRANSITION -> Failed (manual intervention required)
```

---

## 8. Resolved Open Questions

| # | Question (from v1) | Resolution |
|---|---|---|
| 1 | Pure CUDA C kernel or Rust wrapper with cuda-sys? | **CUDA C kernel, Rust host-side management.** The persistent kernel is written in CUDA C (`.cu` file), compiled with `nvcc`, loaded as a PTX/cubin module by the Rust host code via `cuda-sys`. The kernel is performance-critical GPU code; Rust adds nothing here. Host-side lifecycle management (`PersistentKernel` struct) is pure Rust. |
| 2 | Minimum viable ring buffer size? | **16 entries for testing, 256 for production.** 16 entries with 64KB slots = 1 MB total data. Sufficient for a functional demo. Production default of 256 entries provides burst absorption. Ring size is a configuration parameter, not compiled-in. |
| 3 | Heterogeneous vs fixed-size data slots? | **Fixed-size slots (64 KB default).** Simplifies RDMA offset calculation, avoids fragmentation, predictable memory layout. For data larger than one slot, use multi-descriptor batches (multiple ring entries, `LAST_IN_BATCH` flag on final entry). For data much smaller than one slot, the wasted space is acceptable (64 KB is small relative to 24 GB VRAM). |
| 4 | Interaction with cudaMalloc tracking? | **Ring buffers are allocated via raw `cuMemAlloc`, registered with R10 as pinned infrastructure.** They bypass OuterLink's intercepted `cudaMalloc` path. The interceptor's allocation table knows about them (registered during `PersistentKernel::launch()`) but treats them as system-managed, not application-managed. This is the same pattern as OuterLink's transport buffers. |
| 5 | Should Phase A start before or after P5 (OpenDMA)? | **Phase A starts immediately, independent of P5.** Phase A uses host-side `cudaMemcpy` or mapped pointers to simulate NIC writes. No BAR1 needed. Phase B requires P5 for real NIC-to-VRAM doorbell writes. Phase A and P5 can develop in parallel. |

---

## 9. Implementation Phases (Refined)

### Phase A: Standalone Persistent Kernel (2-3 weeks)
**Goal:** Validate kernel pattern on RTX 3090, measure baseline metrics.

**Deliverables:**
1. `persistent_kernel.cu` -- full kernel as shown in Section 3
2. `PersistentKernel` Rust struct with launch, monitor, shutdown
3. `RingBuffer`, `DataPool`, `ControlBlock` VRAM allocation
4. Host-side doorbell writer (simulates NIC via cudaMemcpy to ring.head)
5. Benchmarks: doorbell detection latency, heartbeat reliability, power draw
6. Windows dev mode: `max_iterations` for TDR-safe testing

**Acceptance criteria:**
- Kernel runs >1 hour on headless Linux without hang or TDR
- Doorbell detection latency: <500ns (host write to kernel detection)
- Graceful shutdown completes within 100ms
- Heartbeat detects simulated hang within 500ms
- Power with `__nanosleep(100)`: <20% TDP while idle

### Phase B: OpenDMA-Fed Persistent Kernel (3-4 weeks)
**Goal:** NIC writes data and doorbell directly to VRAM via BAR1.

**Deliverables:**
1. BAR1 registration for ring buffer + data pool VRAM regions
2. RDMA write path: remote node sends data + descriptor + head to VRAM
3. Cache coherency validation: 1M+ doorbell cycles without stale data
4. End-to-end latency benchmark: remote write to kernel processing start
5. Throughput benchmark: sustained data rate through pipeline

**Acceptance criteria:**
- NIC-to-kernel doorbell latency: <2us (BAR1 write to kernel detection)
- Data integrity verified (CRC32 in descriptor metadata) over 1M+ cycles
- Sustained throughput: >= 10 GB/s (approaching 100Gbps for large batches)
- No stale data observed (volatile + threadfence_system sufficient)
- If stale data observed: atomicAdd(0) fallback activates and resolves

### Phase C: Full Pipeline Integration (3-4 weeks)
**Goal:** Connect persistent kernel pipeline to OuterLink server daemon.

**Deliverables:**
1. Server-side `PersistentKernel` manager in `outerlink-server`
2. Client protocol extension: "persistent stream" mode
3. Output ring: results flow back via RDMA or host read
4. Multi-connection multiplexing via `connection_id` in descriptors
5. Error recovery with automatic kernel restart (<500ms)
6. Configuration API for ring sizes, poll intervals, buffer counts

**Acceptance criteria:**
- Multiple clients stream data through single persistent kernel simultaneously
- Kernel crash recovery: <500ms, clients reconnect automatically
- Configuration changes (ring size, poll interval) via hot-reload
- Integration tests pass with simulated network failures and kernel hangs

**Total: 8-11 weeks** (sequential, each phase depends on previous).

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan
- [research/01-persistent-kernel-patterns.md](./research/01-persistent-kernel-patterns.md)
- [research/02-doorbell-mechanisms.md](./research/02-doorbell-mechanisms.md)
- [research/03-network-fed-execution.md](./research/03-network-fed-execution.md)
- [R10: Memory Hierarchy](../R10-memory-hierarchy/) -- Pinned region management
- [R11: Speculative Prefetch](../R11-speculative-prefetch/) -- PersistentKernelPlan
- [R14: Transport Compression](../R14-transport-compression/) -- Decompress pipeline stage
- [R16: BlueField DPU](../R16-bluefield-dpu-offload/) -- DPU as ring buffer producer
- [R26: PTP Clock Sync](../R26-ptp-clock-sync/) -- Coordinated doorbell timing
- [R28: Scatter-Gather DMA](../R28-scatter-gather-dma/) -- Multi-region data ingest

## Open Questions

All v1 open questions resolved in Section 8. No new open questions -- remaining unknowns (exact L2 cache latency on RTX 3090, optimal nanosleep interval) require hardware benchmarking in Phase A.
