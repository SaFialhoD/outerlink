# R21: GPU Direct Storage Over Network -- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Phase:** 9 -- Hardening
**Priority:** MEDIUM
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of R21's remote GPU Direct Storage design. This document resolves open questions from v1, provides exact Rust structs and trait implementations, defines concrete data path protocols, and specifies precise integration points with R10 (memory tiering, NvmeTierDriver), R14 (compression), R28 (scatter-gather), and the OpenDMA subsystem.

---

## 1. Resolved Open Questions (from v1)

### Q1: PCIe topology on our hardware
**Resolved:** Must verify at runtime. The `StorageTopologyProbe` (defined below) runs `lspci -tv` and checks P2PDMA compatibility at startup. AMD Zen root complexes support P2P forwarding between most PCIe devices behind the same root complex. The M.2 slots on AM4/AM5 boards are typically behind the CPU root complex (not the chipset), which is the same hierarchy as the ConnectX-5 PCIe slot. Runtime verification is mandatory; we do NOT assume topology.

### Q2: ConnectX-5 NVMe-oF target offload firmware
**Resolved:** Check at startup via `mlxfwmanager --query`. NVMe-oF target offload requires firmware version >= 16.28.x for ConnectX-5. If firmware is too old, fall back to host-staged sender (Option A). The setup guide includes firmware update instructions.

### Q3: BAR1 size on RTX 3090
**Resolved:** With Resizable BAR enabled in BIOS, RTX 3090 exposes the full 24GB BAR1. Without rebar, BAR1 is 256MB. The `StorageTopologyProbe` reads BAR1 size from sysfs. With 256MB BAR1, we use a rotation/pipelining strategy: map 256MB window, transfer data in 256MB chunks, remap for next chunk. With full rebar, we can map the entire VRAM and do single-shot DMA for transfers up to 24GB.

### Q4: NVMe-oF vs custom RDMA protocol overhead
**Resolved:** For large sequential reads (>1MB), NVMe-oF protocol overhead is <1% of transfer time. The NVMe-oF command/response headers are ~64 bytes per I/O, negligible compared to megabyte payloads. For small random I/O (4KB), NVMe-oF adds ~5us per operation. This is acceptable since small random I/O is not our primary use case (dataset loading and checkpointing are sequential).

### Q5: Optimal I/O size for pipeline
**Resolved:** 4MB chunks for the double-buffered pipeline. At 7 GB/s NVMe throughput, a 4MB read takes ~570us. At 12.5 GB/s RDMA, a 4MB send takes ~320us. With double buffering, the pipeline achieves the NVMe-limited throughput of ~7 GB/s. Smaller chunks (1MB) add more scheduling overhead; larger chunks (16MB) increase latency to first byte.

### Q6: IOMMU configuration
**Resolved:** P2PDMA works with IOMMU in passthrough mode (`iommu=pt` kernel parameter). Full IOMMU (strict mode) may block P2P between devices in different IOMMU groups. Passthrough mode is the recommended configuration for OuterLink and is documented in the setup guide. It provides DMA protection while allowing P2P.

### Q7: Multi-tenant storage access
**Resolved:** NVMe hardware supports multiple I/O queues (up to 64K). When multiple GPU nodes access the same storage node, each gets its own NVMe-oF connection with a dedicated queue pair. Total bandwidth is shared (7 GB/s per NVMe drive). For N concurrent readers of the same drive: each gets ~7/N GB/s. Scaling strategy: stripe data across multiple NVMe drives on the storage node.

### Q8: Error handling for P2P failures
**Resolved:** P2PDMA failures manifest as PCIe completion errors. The kernel NVMe driver translates these to I/O errors (`-EIO`). Our storage API propagates the error to the caller. Recovery strategy: retry once on the P2P path; if it fails again, fall back to host-staged transfer for that operation and mark the P2P path as degraded. R15's failure detection is notified if errors persist.

---

## 2. Rust Structs and Types

### 2.1 Core Storage Types

```rust
/// Handle to a remote storage file/device
#[derive(Debug)]
pub struct StorageHandle {
    /// Remote node hosting the NVMe
    pub storage_node: NodeId,
    /// NVMe namespace identifier (NQN for NVMe-oF)
    pub namespace: StorageNamespace,
    /// Selected data path for this handle
    pub path: StoragePath,
    /// Alignment requirements for direct DMA
    pub alignment: AlignmentRequirements,
    /// Reference to the transport connection
    pub(crate) connection: Arc<StorageConnection>,
}

#[derive(Debug, Clone)]
pub struct StorageNamespace {
    /// NVMe-oF NQN (NVMe Qualified Name)
    pub nqn: String,
    /// Namespace ID within the subsystem
    pub nsid: u32,
    /// Total size in bytes
    pub size: u64,
    /// Logical block size (typically 512 or 4096)
    pub block_size: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum StoragePath {
    /// NVMe -> host RAM -> RDMA -> OpenDMA -> GPU
    /// Host RAM touched on sender only
    HostStaged {
        /// Size of each pinned buffer in the double-buffer pipeline
        chunk_size: usize,
    },
    /// NVMe -> ConnectX-5 P2P -> RDMA -> OpenDMA -> GPU
    /// Zero host RAM on either side
    NvmeOfOffload,
    /// NVMe -> host RAM -> RDMA -> host RAM -> cudaMemcpy -> GPU
    /// Fallback: host RAM touched on both sides
    FullHostStaged {
        chunk_size: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct AlignmentRequirements {
    /// File offset must be a multiple of this (bytes)
    pub offset_alignment: usize,   // 4096 for O_DIRECT
    /// Buffer address must be a multiple of this (bytes)
    pub buffer_alignment: usize,   // 4096 for DMA
    /// Transfer size should be a multiple of this (bytes)
    pub size_alignment: usize,     // 512 minimum, 4096 optimal
}

/// A single read request in a batch operation
#[derive(Debug, Clone)]
pub struct StorageReadRequest {
    /// Destination GPU buffer (CUDA device pointer)
    pub gpu_buffer: CudaDevicePtr,
    /// Number of bytes to read
    pub size: usize,
    /// Offset within the file/device to read from
    pub file_offset: u64,
    /// Offset within the GPU buffer to write to
    pub buf_offset: usize,
}

/// Asynchronous operation handle
#[derive(Debug)]
pub struct StorageFuture {
    /// Completion state
    state: Arc<AtomicU8>,
    /// Bytes transferred (set on completion)
    bytes_transferred: Arc<AtomicUsize>,
    /// Error (if any)
    error: Arc<Mutex<Option<StorageError>>>,
    /// CUDA event signaled on completion (for stream integration)
    cuda_event: Option<CudaEvent>,
}

impl StorageFuture {
    /// Block until the operation completes
    pub fn wait(&self) -> Result<usize> {
        while self.state.load(Ordering::Acquire) == 0 {
            std::hint::spin_loop();
        }
        if let Some(err) = self.error.lock().take() {
            return Err(err);
        }
        Ok(self.bytes_transferred.load(Ordering::Acquire))
    }

    /// Check if completed without blocking
    pub fn is_complete(&self) -> bool {
        self.state.load(Ordering::Acquire) != 0
    }
}
```

### 2.2 Storage Connection Types

```rust
/// Connection to a remote storage node
#[derive(Debug)]
pub struct StorageConnection {
    /// Remote node
    pub node_id: NodeId,
    /// Connection type determines the data path
    pub conn_type: StorageConnectionType,
    /// RDMA queue pair for data transfer
    pub data_qp: Arc<QueuePair>,
    /// RDMA memory regions for pinned buffers
    pub pinned_buffers: Vec<PinnedBuffer>,
    /// Double-buffer state for pipelining
    pub pipeline: PipelineState,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageConnectionType {
    /// Standard RDMA connection (for host-staged transfers)
    Rdma,
    /// NVMe-oF initiator connection (for NVMe-oF path)
    NvmeOf {
        /// Whether target offload is active on the remote side
        target_offload: bool,
    },
}

/// Double-buffer pipeline for overlapping NVMe read and RDMA send
#[derive(Debug)]
pub struct PipelineState {
    /// Two pinned buffers for double-buffering
    pub buffers: [PinnedBuffer; 2],
    /// Which buffer is currently being filled by NVMe read
    pub read_idx: AtomicUsize,
    /// Which buffer is currently being sent via RDMA
    pub send_idx: AtomicUsize,
    /// Chunk size for pipeline operations
    pub chunk_size: usize,
}

/// A pinned (DMA-capable) host memory buffer
#[derive(Debug)]
pub struct PinnedBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    /// RDMA memory region key for this buffer
    pub mr_lkey: u32,
    pub mr_rkey: u32,
}
```

### 2.3 NVMe-oF Configuration Types

```rust
/// Configuration for the NVMe-oF target (sender side)
#[derive(Debug, Clone)]
pub struct NvmetConfig {
    /// Subsystem NQN
    pub nqn: String,
    /// NVMe device path (e.g., /dev/nvme0n1)
    pub nvme_device: PathBuf,
    /// Namespace ID to expose
    pub namespace_id: u32,
    /// RDMA port for the NVMe-oF target
    pub rdma_port: u16,
    /// Whether to enable ConnectX-5 target offload
    pub enable_offload: bool,
    /// Maximum I/O queue depth
    pub max_queue_depth: u32,
}

impl NvmetConfig {
    /// Generate the configfs commands to set up the NVMe-oF target
    pub fn setup_commands(&self) -> Vec<String> {
        vec![
            // Create subsystem
            format!("mkdir -p /sys/kernel/config/nvmet/subsystems/{}", self.nqn),
            format!("echo 1 > /sys/kernel/config/nvmet/subsystems/{}/attr_allow_any_host",
                self.nqn),

            // Add namespace
            format!("mkdir -p /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}",
                self.nqn, self.namespace_id),
            format!("echo {} > /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}/device_path",
                self.nvme_device.display(), self.nqn, self.namespace_id),
            format!("echo 1 > /sys/kernel/config/nvmet/subsystems/{}/namespaces/{}/enable",
                self.nqn, self.namespace_id),

            // Create RDMA port
            format!("mkdir -p /sys/kernel/config/nvmet/ports/1"),
            format!("echo rdma > /sys/kernel/config/nvmet/ports/1/addr_trtype"),
            format!("echo ipv4 > /sys/kernel/config/nvmet/ports/1/addr_adrfam"),
            format!("echo 0.0.0.0 > /sys/kernel/config/nvmet/ports/1/addr_traddr"),
            format!("echo {} > /sys/kernel/config/nvmet/ports/1/addr_trsvcid", self.rdma_port),

            // Link subsystem to port
            format!("ln -s /sys/kernel/config/nvmet/subsystems/{} /sys/kernel/config/nvmet/ports/1/subsystems/{}",
                self.nqn, self.nqn),

            // Enable ConnectX-5 target offload (if supported)
            if self.enable_offload {
                format!("echo 1 > /sys/kernel/config/nvmet/ports/1/param_offload")
            } else {
                String::new()
            },
        ].into_iter().filter(|s| !s.is_empty()).collect()
    }
}

/// Topology probe result for P2PDMA compatibility
#[derive(Debug, Clone)]
pub struct StorageTopology {
    /// NVMe devices and their PCIe locations
    pub nvme_devices: Vec<NvmeDeviceInfo>,
    /// ConnectX-5 NIC PCIe locations
    pub nic_devices: Vec<NicDeviceInfo>,
    /// GPU devices and their BAR1 sizes
    pub gpu_devices: Vec<GpuDeviceInfo>,
    /// P2P compatibility matrix: (device_a, device_b) -> compatible
    pub p2p_matrix: HashMap<(PciAddress, PciAddress), P2pCompatibility>,
}

#[derive(Debug, Clone)]
pub struct NvmeDeviceInfo {
    pub device_path: PathBuf,       // /dev/nvme0n1
    pub pci_address: PciAddress,    // 0000:02:00.0
    pub model: String,
    pub size_bytes: u64,
    pub has_cmb: bool,              // Controller Memory Buffer present
    pub max_transfer_size: usize,   // MDTS (Maximum Data Transfer Size)
}

#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub gpu_index: u32,
    pub pci_address: PciAddress,
    pub bar1_size: u64,             // 256MB or 24GB with rebar
    pub rebar_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum P2pCompatibility {
    /// Direct P2P possible (same switch or supported root complex)
    Supported,
    /// P2P may work but unverified (same root complex, untested chipset)
    MayWork,
    /// P2P not possible (different root complexes or blocked)
    NotSupported,
}
```

---

## 3. Trait Implementations

### 3.1 StorageService Trait

```rust
/// Main storage service API for OuterLink
pub trait StorageService: Send + Sync {
    /// Open a handle to a remote storage file/device
    fn open(
        &self,
        node_id: NodeId,
        device_path: &str,
        flags: OpenFlags,
    ) -> Result<StorageHandle>;

    /// Read from remote storage directly to GPU VRAM
    fn read_to_gpu(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        size: usize,
        file_offset: u64,
        buf_offset: usize,
    ) -> Result<usize>;

    /// Batch read: multiple regions in one call
    fn batch_read(
        &self,
        handle: &StorageHandle,
        requests: &[StorageReadRequest],
    ) -> Result<Vec<usize>>;

    /// Async read integrated with CUDA stream
    fn read_async(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        size: usize,
        file_offset: u64,
        stream: CudaStream,
    ) -> Result<StorageFuture>;

    /// Write GPU VRAM to remote storage (for checkpointing)
    fn write_from_gpu(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        size: usize,
        file_offset: u64,
    ) -> Result<usize>;

    /// Probe storage topology for P2PDMA capability
    fn probe_topology(&self) -> Result<StorageTopology>;
}
```

**Implementation: `OlStorageService`**

```rust
pub struct OlStorageService {
    /// Transport layer for RDMA operations
    transport: Arc<dyn Transport>,
    /// OpenDMA subsystem for GPU BAR1 DMA
    opendma: Arc<dyn OpenDma>,
    /// Topology probe results (cached at startup)
    topology: RwLock<StorageTopology>,
    /// Active storage connections per node
    connections: DashMap<NodeId, Arc<StorageConnection>>,
    /// R14 compression engine (optional)
    compressor: Option<Arc<dyn Compressor>>,
    /// Pipeline thread pool for double-buffered transfers
    pipeline_pool: ThreadPool,
}

impl OlStorageService {
    /// Select the best storage path based on topology probe
    fn select_path(
        &self,
        storage_node: NodeId,
        gpu_id: u32,
    ) -> StoragePath {
        let topo = self.topology.read();

        // Check if NVMe-oF offload is available on the storage node
        let offload_available = self.check_nvmeof_offload(storage_node);

        // Check if P2PDMA is possible between storage node's NVMe and NIC
        let sender_p2p = self.check_sender_p2p(storage_node, &topo);

        // Check if OpenDMA is available on the receiver side
        let receiver_opendma = self.opendma.is_available(gpu_id);

        match (offload_available && sender_p2p, receiver_opendma) {
            (true, true) => StoragePath::NvmeOfOffload,
            (false, true) => StoragePath::HostStaged {
                chunk_size: 4 * 1024 * 1024, // 4MB
            },
            (_, false) => StoragePath::FullHostStaged {
                chunk_size: 4 * 1024 * 1024,
            },
        }
    }
}
```

### 3.2 NvmeTierDriver (R10 Integration)

This is the driver that R10's memory tiering system calls when accessing Tier 4/5 (NVMe).

```rust
/// R10 defines a tier driver trait for each memory tier.
/// R21 provides the implementation for NVMe tiers (Tier 4 = local, Tier 5 = remote).
pub trait TierDriver: Send + Sync {
    /// Read a page from this tier into a destination buffer
    fn read_page(&self, page_addr: TierAddress, dest: PageBuffer) -> Result<()>;

    /// Write a page from source buffer to this tier
    fn write_page(&self, source: PageBuffer, page_addr: TierAddress) -> Result<()>;

    /// Allocate space for a page on this tier
    fn alloc_page(&self) -> Result<TierAddress>;

    /// Free a page on this tier
    fn free_page(&self, page_addr: TierAddress) -> Result<()>;

    /// Get tier bandwidth (for scheduling decisions)
    fn bandwidth(&self) -> Bandwidth;

    /// Get tier latency (for scheduling decisions)
    fn latency(&self) -> Duration;
}

/// R21's implementation for remote NVMe (Tier 5)
pub struct RemoteNvmeTierDriver {
    /// Underlying storage service
    storage: Arc<OlStorageService>,
    /// Storage handle to the remote NVMe
    handle: StorageHandle,
    /// Page size (64KB, from R10)
    page_size: usize,
    /// Free page bitmap on the remote NVMe
    free_map: RwLock<RoaringBitmap>,
    /// Total capacity in pages
    total_pages: u64,
}

impl TierDriver for RemoteNvmeTierDriver {
    fn read_page(&self, page_addr: TierAddress, dest: PageBuffer) -> Result<()> {
        let file_offset = page_addr.offset;
        let gpu_ptr = dest.as_cuda_ptr();
        let size = self.page_size;

        // This call uses the full pipeline: NVMe -> [host RAM] -> RDMA -> OpenDMA -> GPU
        self.storage.read_to_gpu(&self.handle, gpu_ptr, size, file_offset, 0)?;
        Ok(())
    }

    fn write_page(&self, source: PageBuffer, page_addr: TierAddress) -> Result<()> {
        let file_offset = page_addr.offset;
        let gpu_ptr = source.as_cuda_ptr();
        let size = self.page_size;

        // Reverse path: GPU -> OpenDMA read from BAR1 -> RDMA -> host RAM -> NVMe write
        self.storage.write_from_gpu(&self.handle, gpu_ptr, size, file_offset)?;
        Ok(())
    }

    fn alloc_page(&self) -> Result<TierAddress> {
        let mut free_map = self.free_map.write();
        let page_idx = free_map.min()
            .ok_or(StorageError::TierFull)?;
        free_map.remove(page_idx);
        Ok(TierAddress {
            node: self.handle.storage_node,
            tier: Tier::RemoteNvme,
            offset: page_idx as u64 * self.page_size as u64,
        })
    }

    fn free_page(&self, page_addr: TierAddress) -> Result<()> {
        let page_idx = (page_addr.offset / self.page_size as u64) as u32;
        self.free_map.write().insert(page_idx);
        Ok(())
    }

    fn bandwidth(&self) -> Bandwidth {
        // Limited by single NVMe (~7 GB/s) or network (~12.5 GB/s), whichever is less
        Bandwidth::from_gbps(7.0)
    }

    fn latency(&self) -> Duration {
        // NVMe read + network transfer + DMA
        Duration::from_micros(100) // ~100us for first byte (4KB)
    }
}
```

### 3.3 StorageTopologyProbe

```rust
impl OlStorageService {
    /// Probe the system to determine P2PDMA capabilities
    pub fn probe_topology_impl(&self) -> Result<StorageTopology> {
        let mut topo = StorageTopology {
            nvme_devices: Vec::new(),
            nic_devices: Vec::new(),
            gpu_devices: Vec::new(),
            p2p_matrix: HashMap::new(),
        };

        // Step 1: Enumerate NVMe devices
        // Parse /sys/class/nvme/nvme*/address for PCI addresses
        // Parse /sys/class/nvme/nvme*/model for device info
        // Check CMB: nvme id-ctrl /dev/nvmeN | grep cmbs

        // Step 2: Enumerate NICs (ConnectX-5)
        // Parse /sys/class/infiniband/*/device/resource for PCI addresses

        // Step 3: Enumerate GPUs
        // Parse /proc/driver/nvidia/gpus/*/information
        // Read BAR1 size from sysfs: /sys/bus/pci/devices/<addr>/resource1

        // Step 4: Check P2PDMA compatibility between all pairs
        // Use kernel P2PDMA distance check:
        //   cat /sys/bus/pci/devices/<addr1>/p2pmem/<addr2>/distance
        //   or check if both devices are behind same PCIe root complex
        //   via lspci -tv parsing

        // Step 5: Check NVMe-oF target offload capability
        // Query ConnectX-5: mlxreg -d <dev> --reg_name NVMEOF_TARGET_OFFLOAD_CAP

        Ok(topo)
    }
}
```

---

## 4. Algorithms and Protocols

### 4.1 Host-Staged Remote Read Pipeline (Phase 1)

The simplest path that achieves the performance goal: NVMe -> host RAM -> RDMA -> OpenDMA -> GPU.

**Double-buffered pipeline flow:**

```
Time -->
Buffer A:  [NVMe Read Chunk 0] [RDMA Send Chunk 0] [NVMe Read Chunk 2] [RDMA Send Chunk 2] ...
Buffer B:                       [NVMe Read Chunk 1] [RDMA Send Chunk 1] [NVMe Read Chunk 3] ...
```

**Step-by-step for `read_to_gpu()` with host-staged path:**

1. **Setup:** Allocate two pinned buffers (Buffer A, B) of `chunk_size` (4MB) each. Register both with RDMA (ibv_reg_mr).
2. **First chunk:** Issue NVMe read of chunk 0 into Buffer A (`pread(nvme_fd, buf_a, chunk_size, file_offset)`). Wait for completion.
3. **Pipeline loop:** For chunk N (N >= 1):
   a. Issue NVMe read of chunk N into Buffer[N % 2] (async, via io_uring).
   b. Simultaneously, RDMA send Buffer[(N-1) % 2] to receiver node.
   c. Receiver: ConnectX-5 DMA engine writes received data to GPU BAR1 (OpenDMA).
   d. Wait for both NVMe read and RDMA send to complete.
   e. Advance to next chunk.
4. **Last chunk:** RDMA send the final buffer, wait for completion.
5. **Completion:** Signal the CUDA event (if async) or return bytes transferred.

**R14 compression integration:** If compression is enabled and the data is compressible:
- After NVMe read into Buffer A, compress in-place or to a separate compressed buffer.
- RDMA send the compressed data (smaller transfer, faster).
- Receiver decompresses after DMA to GPU (or on host before OpenDMA write).
- NVMe is the bottleneck at ~7 GB/s; compression can reduce network bytes by 2-4x for model weights/embeddings, but adds CPU overhead. Net benefit depends on data compressibility.

```rust
impl OlStorageService {
    fn read_to_gpu_host_staged(
        &self,
        handle: &StorageHandle,
        gpu_buffer: CudaDevicePtr,
        total_size: usize,
        file_offset: u64,
        chunk_size: usize,
    ) -> Result<usize> {
        let conn = &handle.connection;
        let num_chunks = (total_size + chunk_size - 1) / chunk_size;
        let mut bytes_transferred = 0usize;

        // Phase 1: Read first chunk synchronously
        let first_read_size = std::cmp::min(chunk_size, total_size);
        pread_all(
            &conn.nvme_fd,
            &conn.pipeline.buffers[0],
            first_read_size,
            file_offset,
        )?;

        for chunk_idx in 1..=num_chunks {
            let read_buf_idx = chunk_idx % 2;
            let send_buf_idx = (chunk_idx - 1) % 2;
            let send_size = if chunk_idx - 1 == num_chunks - 1 {
                total_size - (num_chunks - 1) * chunk_size
            } else {
                chunk_size
            };

            // Async: start NVMe read for next chunk (if not last)
            let read_future = if chunk_idx < num_chunks {
                let read_size = std::cmp::min(
                    chunk_size,
                    total_size - chunk_idx * chunk_size,
                );
                let read_offset = file_offset + (chunk_idx * chunk_size) as u64;
                Some(self.async_pread(
                    &conn.nvme_fd,
                    &conn.pipeline.buffers[read_buf_idx],
                    read_size,
                    read_offset,
                )?)
            } else {
                None
            };

            // Optional: compress before sending (R14 integration)
            let (send_data, send_len) = if let Some(ref compressor) = self.compressor {
                compressor.compress(
                    &conn.pipeline.buffers[send_buf_idx],
                    send_size,
                )?
            } else {
                (conn.pipeline.buffers[send_buf_idx].ptr, send_size)
            };

            // RDMA send to receiver -> OpenDMA writes to GPU BAR1
            let gpu_offset = bytes_transferred;
            self.rdma_send_to_gpu(
                conn,
                send_data,
                send_len,
                gpu_buffer,
                gpu_offset,
            )?;

            bytes_transferred += send_size;

            // Wait for NVMe read to complete (if started)
            if let Some(future) = read_future {
                future.wait()?;
            }
        }

        Ok(bytes_transferred)
    }
}
```

### 4.2 NVMe-oF Target Offload Pipeline (Phase 2/3)

Zero-copy on both sides: NVMe -> ConnectX-5 P2P -> wire -> ConnectX-5 -> OpenDMA -> GPU.

**Setup flow (performed once per storage node):**

1. **Storage node (sender):**
   a. Load nvmet kernel module: `modprobe nvmet`.
   b. Load nvmet-rdma transport: `modprobe nvmet-rdma`.
   c. Configure nvmet subsystem via configfs (see `NvmetConfig::setup_commands()`).
   d. Enable ConnectX-5 target offload: `echo 1 > /sys/kernel/config/nvmet/ports/1/param_offload`.
   e. Verify offload active: check dmesg for "nvmet_rdma: enabling offload".

2. **GPU node (receiver):**
   a. Load nvme-rdma initiator: `modprobe nvme-rdma`.
   b. Connect: `nvme connect -t rdma -n <nqn> -a <storage_ip> -s <port>`.
   c. Verify: `/dev/nvmeXnY` appears for the remote namespace.

**Read flow (for each read operation):**

1. GPU node's `read_to_gpu()` is called with an NVMe-oF handle.
2. Issue `pread()` on the NVMe-oF block device `/dev/nvmeXnY` into a pinned buffer.
3. On the storage node: ConnectX-5 hardware intercepts the NVMe-oF read command.
4. ConnectX-5 DMA engine reads from local NVMe via embedded PCIe switch.
5. ConnectX-5 sends data over RDMA to GPU node.
6. GPU node: RDMA receive delivers data to pinned buffer (or directly to GPU BAR1 if we bridge NVMe-oF with OpenDMA).
7. If data landed in pinned buffer: cudaMemcpy to GPU. If OpenDMA: zero-copy, data is in VRAM.

**Bridging NVMe-oF with OpenDMA (the hard part):**

The challenge is that NVMe-oF initiator (on the GPU node) expects to receive data into host memory (standard NVMe-oF flow). To deliver directly to GPU BAR1, we need to register the GPU BAR1 region as the RDMA receive buffer for the NVMe-oF connection.

**Approach:** Configure the NVMe-oF initiator's RDMA receive buffers to point to GPU BAR1 addresses. This requires modifying the nvme-rdma kernel module to accept BAR1 addresses instead of host RAM addresses. This is the same technique OpenDMA uses for regular RDMA receives.

If kernel modification is too invasive: fall back to receiving in host pinned RAM and doing a final `cudaMemcpyAsync` to VRAM. This adds one copy on the receiver but the sender is still zero-copy.

### 4.3 Batch Read Protocol (R28 Scatter-Gather Integration)

`batch_read()` handles multiple non-contiguous regions in a single API call, critical for loading datasets with non-sequential access patterns.

**Step-by-step:**

1. Receive array of `StorageReadRequest` entries.
2. Sort by `file_offset` to maximize sequential NVMe access.
3. Identify contiguous or nearby regions that can be merged into larger reads:
   - If two regions are within 64KB of each other, merge into one read and discard the gap.
   - Reduces NVMe I/O count (fewer queue submissions).
4. For each merged region:
   a. Read from NVMe into pinned buffer.
   b. Split the buffer into original request chunks.
   c. RDMA send each chunk to the appropriate GPU buffer offset.
5. Use io_uring for async NVMe reads to overlap with RDMA sends.

**R28 scatter-gather integration:** R28 provides DMA scatter-gather lists (SGLs) that can describe non-contiguous GPU memory regions. When batch_read requests target non-contiguous GPU addresses:
- Build an SGL mapping each request's destination in GPU BAR1.
- A single RDMA transfer can use the SGL to scatter data to multiple GPU locations.
- This avoids issuing separate RDMA sends for each request.

```rust
impl OlStorageService {
    fn batch_read_impl(
        &self,
        handle: &StorageHandle,
        requests: &[StorageReadRequest],
    ) -> Result<Vec<usize>> {
        // Sort by file_offset for sequential NVMe access
        let mut sorted: Vec<(usize, &StorageReadRequest)> =
            requests.iter().enumerate().collect();
        sorted.sort_by_key(|(_, r)| r.file_offset);

        let mut results = vec![0usize; requests.len()];

        // Merge nearby reads
        let merged = self.merge_nearby_reads(&sorted, 64 * 1024);

        for merge_group in &merged {
            // Read merged region from NVMe
            let total_size: usize = merge_group.total_size;
            let base_offset = merge_group.base_offset;

            // Use pipeline for the merged read
            let data = self.read_nvme_region(handle, base_offset, total_size)?;

            // Scatter to individual GPU destinations via R28 SGL
            if self.opendma.supports_sgl() {
                let sgl = merge_group.build_sgl(&data)?;
                self.opendma.scatter_write(sgl)?;
            } else {
                // Fallback: individual RDMA sends
                for entry in &merge_group.entries {
                    let slice = &data[entry.local_offset..entry.local_offset + entry.size];
                    self.rdma_send_to_gpu(
                        &handle.connection,
                        slice.as_ptr(),
                        slice.len(),
                        entry.gpu_buffer,
                        entry.buf_offset,
                    )?;
                    results[entry.original_idx] = entry.size;
                }
            }
        }

        Ok(results)
    }
}
```

### 4.4 Write Path (GPU -> Remote NVMe, for Checkpointing)

Used by R15's checkpoint system to persist GPU state to remote NVMe.

**Step-by-step:**

1. `write_from_gpu()` called with GPU source buffer and remote NVMe destination.
2. Read GPU data via OpenDMA (ConnectX-5 reads from GPU BAR1 via RDMA read).
3. Data arrives in host pinned buffer on the GPU node.
4. RDMA send to storage node.
5. Storage node: receives into pinned buffer, writes to NVMe via `pwrite()`.
6. With NVMe-oF target offload: storage node's ConnectX-5 can chain RDMA receive -> NVMe write in hardware (reverse of the read path).

---

## 5. Integration Points (Exact Function Calls)

### 5.1 R10 (Memory Tiering) Integration

| R21 provides to R10 | Purpose |
|----------------------|---------|
| `RemoteNvmeTierDriver` implementing `TierDriver` | Tier 5 (remote NVMe) driver for R10's tiering system |
| `LocalNvmeTierDriver` implementing `TierDriver` | Tier 4 (local NVMe) driver (simpler, no network) |
| `TierDriver::bandwidth()` returns 7 GB/s | R10 uses this for tier promotion/demotion decisions |
| `TierDriver::latency()` returns ~100us | R10 uses this for prefetch timing |

| R10 calls R21 | Purpose |
|----------------|---------|
| `tier_driver.read_page(addr, buf)` | Page fault on Tier 5 page -> read from remote NVMe to GPU |
| `tier_driver.write_page(buf, addr)` | Evict dirty page to NVMe tier |
| `tier_driver.alloc_page()` | Allocate backing storage for a new NVMe-tier page |
| `tier_driver.free_page(addr)` | Release NVMe backing when page is promoted and old copy freed |

### 5.2 R14 (Compression) Integration

| R14 provides to R21 | Purpose |
|----------------------|---------|
| `Compressor::compress(data, len) -> (ptr, compressed_len)` | Compress data before RDMA transfer. NVMe is the bottleneck at ~7 GB/s; if data compresses 2x, network transfer takes half the time, improving pipeline overlap. |
| `Compressor::decompress(data, len) -> (ptr, decompressed_len)` | Decompress after RDMA receive, before writing to GPU. |

| When compression helps | When it doesn't |
|------------------------|-----------------|
| Model weights (FP16, patterns) -- ~2x compression | Random data (encrypted, already compressed) |
| Embedding tables (sparse, repetitive) -- ~3-4x | Raw activations (high entropy) |
| Optimizer states (Adam momentum) -- ~1.5-2x | Small transfers (<64KB, overhead not amortized) |
| Checkpoint data -- ~2x average | |

**Decision:** Compression is optional and adaptive. Measure compression ratio on first chunk; if ratio < 1.2x (less than 20% reduction), disable compression for the remainder of this transfer.

### 5.3 R28 (Scatter-Gather DMA) Integration

| R28 provides to R21 | Purpose |
|----------------------|---------|
| `ScatterGatherList` type | Describes non-contiguous GPU memory regions for a single DMA operation |
| `opendma.scatter_write(sgl)` | Single DMA operation that writes to multiple GPU BAR1 regions |
| `opendma.supports_sgl()` | Check if SGL DMA is available on this hardware |

| R21 uses R28 for | Purpose |
|-------------------|---------|
| `batch_read()` scatter to multiple GPU buffers | Avoid multiple RDMA sends for batch operations |
| Dataset loading with non-contiguous samples | Load training batch (multiple images/tokens) scattered across NVMe, deliver to scattered GPU addresses |

### 5.4 OpenDMA Integration

| OpenDMA provides to R21 | Purpose |
|--------------------------|---------|
| `opendma.get_bar1_address(gpu_ptr)` | Translate CUDA device pointer to BAR1 physical address for RDMA destination |
| `opendma.register_region(gpu_ptr, size)` | Register GPU memory region for RDMA receive |
| `opendma.is_available(gpu_id)` | Check if OpenDMA is working on this GPU |

| R21 uses OpenDMA for | Purpose |
|-----------------------|---------|
| Receiver-side zero-copy | RDMA receive writes directly to GPU BAR1 instead of host RAM |
| Write path: GPU -> network | RDMA read from GPU BAR1 to send GPU data over network |

### 5.5 R15 (Fault Tolerance) Integration

| R21 provides to R15 | Purpose |
|----------------------|---------|
| `write_from_gpu()` | R15 checkpoint system writes GPU state to remote NVMe for persistence |
| `read_to_gpu()` | R15 recovery reads checkpoint from NVMe back to GPU |

| R15 uses R21 for | Purpose |
|-------------------|---------|
| Cold checkpoint persistence | Every M iterations, write checkpoint to remote NVMe via `write_from_gpu()` |
| Recovery from NVMe checkpoint | On total cluster restart, load last NVMe checkpoint via `read_to_gpu()` |

---

## 6. Refined Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|-------------|--------------|
| **R21-1: Host-Staged Remote Storage** | 2.5 weeks | OpenDMA working, RDMA transport | `OlStorageService`, `read_to_gpu()` with double-buffered pipeline, `StorageTopologyProbe`, basic write path, throughput benchmarks |
| **R21-2: NVMe-oF Integration** | 2.5 weeks | R21-1 complete | `NvmetConfig`, nvmet setup automation, ConnectX-5 target offload config, P2PDMA topology validation, NVMe-oF initiator integration |
| **R21-3: Full P2P Pipeline + API** | 2.5 weeks | R21-2 + R28 scatter-gather | NVMe-oF + OpenDMA bridge, `batch_read()` with SGL support, `read_async()` with CUDA stream, alignment handling, full P2P benchmarks |
| **R21-4: R10 Tiering Integration** | 1.5 weeks | R21-1 + R10 tier driver API | `RemoteNvmeTierDriver`, `LocalNvmeTierDriver`, prefetch for sequential access, eviction to NVMe, integration tests with large datasets |

**Total: 9 weeks** (previously 7-11; tightened by resolving unknowns)

**Parallelism:** R21-1 is the prerequisite. R21-2 builds on R21-1 (adds NVMe-oF to the existing host-staged path). R21-3 needs R21-2 and R28. R21-4 can start once R21-1 is done (uses host-staged path initially, upgraded to P2P later).

---

## 7. Success Criteria (Updated)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Single NVMe -> GPU throughput (host-staged) | **>6 GB/s** | Custom benchmark, 1GB sequential read |
| Single NVMe -> GPU throughput (NVMe-oF offload) | **>6.5 GB/s** | Custom benchmark with P2P path |
| Sender CPU usage (host-staged) | Baseline | perf stat during 10s transfer |
| Sender CPU usage (NVMe-oF offload) | **<5% of baseline** | perf stat during 10s transfer |
| Receiver CPU usage (OpenDMA) | **<2%** | perf stat during 10s transfer |
| Multi-NVMe striped throughput (2x drives) | **>12 GB/s** | Striped read benchmark |
| End-to-end latency (first byte, 4KB) | **<200 us** | Custom latency benchmark |
| R10 Tier 5 page fault latency | **<500 us** | Page fault trace timing (64KB page) |
| Batch read (100 x 1MB scattered) | **>5 GB/s** sustained | Custom batch benchmark |
| Dataset streaming (10+ minutes) | **>5 GB/s** sustained | Training workload with NVMe-backed dataset |
| Write path: GPU -> remote NVMe | **>5 GB/s** | Checkpoint write benchmark |
| **NEW:** P2PDMA topology probe time | **<2 seconds** | Time to complete `probe_topology()` |
| **NEW:** Compressed transfer speedup | **>1.3x** for compressible data | Compare with/without R14 compression |

---

## Related Documents

- [preplan.md](preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-nvidia-gds-architecture.md](research/01-nvidia-gds-architecture.md)
- [research/02-p2pdma-and-nvme.md](research/02-p2pdma-and-nvme.md)
- [research/03-remote-gds-pipeline.md](research/03-remote-gds-pipeline.md)
- R10: Memory Tiering (NvmeTierDriver interface, Tier 4/5 definitions, 64KB pages)
- R14: Compression (NVMe transfer compression, adaptive compression ratio)
- R28: Scatter-Gather DMA (SGL for batch reads, multi-region dataset loading)
- R15: Fault Tolerance (checkpoint write/read via NVMe, cold checkpoint persistence)
- P9: OpenDMA (BAR1 RDMA, receiver-side zero-copy)
