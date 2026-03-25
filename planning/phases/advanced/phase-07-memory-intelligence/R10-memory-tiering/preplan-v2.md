# R10 Pre-Plan v2: Memory Tiering & NVMe as Tier 3

**Date Created:** 2026-03-25
**Date Updated:** 2026-03-25
**Status:** DRAFT (Second Round Refinement)
**Author:** Research Agent (Round 2)
**Supersedes:** `preplan.md` (v1, preserved for reference)

## Purpose

Second-round refinement of the R10 pre-plan. Cross-references findings from R11, R12, R14, R15, R17, R19, R20, R21, and R30 to resolve open questions, specify exact data structures, define Rust trait interfaces, and establish integration points. This document is the bridge between pre-plan and detailed implementation plan.

---

## 1. Resolved Open Questions

### Question 1: How do we handle cudaMallocManaged (UVM) allocations?

**RESOLVED.** Option (c): Convert UVM allocations to OuterLink-managed allocations transparently.

**Rationale from cross-topic analysis:**
- R19 (Network Page Faults) requires SWMR I/S/E coherency on all managed pages. UVM's own coherency would conflict with OuterLink's coherency protocol.
- R19 uses `cuMemMap` for pre-launch paging, which is incompatible with UVM's internal page management.
- The interception layer already controls `cudaMallocManaged`. Replace it with `cuMemCreate` + `cuMemMap` backed by OuterLink's page table. The application sees the same pointer; OuterLink manages placement.

**Implementation:** In the CUDA interception layer, `cudaMallocManaged` calls are redirected to `cuMemAddressReserve` + registration in R10's page table. Physical backing is deferred until first access or explicit memcpy.

### Question 2: Should the tier manager be a separate process or embedded in the server?

**RESOLVED.** Embedded in the server process.

**Rationale:**
- R20 (NCCL Backend) requires `regMr` to coordinate with the VRAM manager synchronously. A separate process would add IPC latency on every memory registration call, which NCCL issues frequently.
- R17 (Topology) uses `hwloc` discovery in the server process. Tier decisions need topology data without IPC.
- R30 (Persistent Kernels) needs pinned memory regions for ring buffer doorbells. The tier manager must respond to pin requests without IPC round-trips.
- **Failure isolation tradeoff:** Accepted. The server process already manages GPU state; a tier manager crash would require server restart regardless.

### Question 3: What happens when all tiers are full?

**RESOLVED.** Cascading strategy:

1. First: Compress pages in-place (R14). nvCOMP at 90+ GB/s on GPU can compress VRAM-resident pages with negligible latency. If compression ratio > 1.5x, the page now fits in less space.
2. Second: Evict to remote NVMe (Tier 5). In a properly sized cluster, remote NVMe should always have space (8 TB per node).
3. Third: If even remote NVMe is full (cluster genuinely out of capacity), return `cudaErrorMemoryAllocation`. This is the correct CUDA error for out-of-memory.

**Integration with R14:** The adaptive compression from R14 is applied during tier migration. When migrating a page from Tier 0 to Tier 2, R14's `Compressor` trait checks compressibility. If beneficial, the page is stored compressed, effectively increasing tier capacity.

### Question 4: What is the real PCIe 5.0 x16 bandwidth under concurrent traffic?

**PARTIALLY RESOLVED.** Based on R17 (Topology) findings:

- `hwloc` discovery will map actual PCIe topology at runtime, including NUMA distances.
- R17 multi-path striping means GPU, NVMe, and ConnectX can share PCIe bandwidth with known contention ratios.
- **Empirical measurement is still needed** on the MS-02 Ultra. However, the architecture accounts for this: the migration engine uses a bandwidth budget that is dynamically adjusted based on observed throughput, not theoretical maximums.
- R17 finding: migration break-even is ~2 RDMA accesses. If a page will be accessed more than twice remotely, migrating it is cheaper than remote access.

**Status:** Architecture handles uncertainty. Exact numbers TBD during Phase R10-E benchmarking.

### Question 5: Can we overlap migration DMA with GPU kernel execution?

**RESOLVED.** Yes, with constraints.

- NVIDIA GPUs have separate copy engines (CE) and compute engines (SM). DMA transfers on the copy engine proceed concurrently with kernel execution.
- R11 (Prefetching) relies on this: prefetch distance is 1-3 kernels ahead, meaning migrations must overlap with current kernel execution.
- R11 bandwidth split: 70% compute, 20% prefetch/migration, 10% system overhead. This is enforced by the migration engine's bandwidth budget.
- **Constraint:** PCIe bandwidth is shared. Under heavy compute with many PCIe-bound operations, migration bandwidth degrades. The migration engine monitors actual throughput and backs off when contention is detected.

**Implementation:** Migrations use a dedicated CUDA stream with low priority. The `MigrationEngine` trait includes a `bandwidth_budget()` method that returns the current available migration bandwidth.

### Question 6: What is practical NVMe RAID-0 bandwidth with page-sized I/O?

**PARTIALLY RESOLVED.** Based on R21 (GPU Direct Storage) findings:

- R21 identifies NVMe-oF offload and P2PDMA paths as the key to high-bandwidth NVMe access.
- 4x Gen4 NVMe RAID-0 theoretical: 28 GB/s sequential. With 64KB random I/O (page-sized): expect 15-20 GB/s due to command overhead and reduced parallelism.
- **Batching is critical:** The NVMe tier driver must batch multiple page operations into larger I/O requests. 16 pages (1 MB) per batch amortizes command overhead and restores near-sequential throughput.
- GDS (if available) adds 2-8x improvement for GPU-to-NVMe direct paths.

**Status:** Architecture uses conservative 15 GB/s estimate. Exact numbers TBD during Phase R10-C benchmarking.

### Question 7: Should pages have a "preferred tier" based on allocation context?

**RESOLVED.** Yes. The PTE includes a `preferred_tier` field.

**Rationale from cross-topic analysis:**
- R11 (Prefetching) classifies access patterns as streaming, working-set, or phased. Each maps to a tier preference.
- R20 (NCCL Backend) registers memory regions (`regMr`) that should be pinned in the tier closest to the transport device. Model weights registered for NCCL should prefer VRAM.
- R30 (Persistent Kernels) ring buffer doorbells must be pinned in VRAM (Tier 0) permanently.

**Implementation:** `preferred_tier` is a 3-bit field in the PTE (0-5 + "any"). The eviction policy treats preferred-tier pages as more expensive to evict. When promoting, preferred tier is the first choice for destination.

### Question 8: How does the tier manager interact with CUDA streams?

**RESOLVED.** Dedicated migration stream per GPU, plus coordination with R19 pre-launch paging.

- R19 (Network Page Faults) uses `cuMemMap` before kernel launch to page in required data. This happens on the kernel's CUDA stream via stream-ordered memory operations.
- Migrations triggered by the tier manager (eviction, promotion) use a separate low-priority CUDA stream to avoid blocking application work.
- R11 (Prefetching) prefetch operations also use the migration stream, but with higher priority within the migration budget.

**Implementation:** `MigrationEngine` owns a `CudaStream` per GPU. All tier-initiated transfers go through this stream. R19 fault-servicing transfers use the application's stream for ordering correctness.

### Question 9: Can we use Linux cgroups or memory.tier sysfs for DRAM/NVMe tiers?

**RESOLVED.** No. Not applicable to OuterLink's use case.

**Rationale:**
- Linux `memory.tier` is for CXL/NUMA tiering of host memory visible to the kernel. OuterLink manages memory at the CUDA level, not the OS level.
- Our DRAM tier uses `cudaHostAlloc` (pinned memory), which is already locked and not subject to kernel page migration.
- Our NVMe tier uses `io_uring` or GDS for direct I/O, bypassing the kernel page cache entirely.
- Using kernel tiering would interfere with OuterLink's explicit placement decisions and break the pinned memory guarantees needed by RDMA (R20 `regMr`).

---

## 2. PTE Layout (64 bytes, exact specification)

Incorporating fields required by R11, R12, R14, R15, R19, R20, and R30.

### Byte-Level Layout

```
Offset  Size    Field                       Description
------  ----    -----                       -----------
0       8       vpn                         Virtual Page Number (u64)
2       1       tier_id                     Current tier (0-5) (u8)
3       1       node_id                     Node hosting this page (u8, 0-255)
4       6       phys_pfn                    Physical Page Frame Number (u48, covers 16 TB at 64KB)
10      4       flags                       Bitfield flags (u32) — see below
14      2       coherency_state             R19: I/S/E state + sharer count (u16)
16      4       access_count                Saturating access counter (u32)
20      4       last_access_ts              Last access timestamp, seconds since epoch (u32)
24      4       alloc_id                    CUDA allocation ID, links to allocation metadata (u32)
28      2       migration_count             Total migrations for this page (u16)
30      2       last_migration_ts_delta     Seconds since last migration, saturating at 65535 (u16)
32      16      dedup_hash                  R12: xxHash128 of page content (u128)
48      4       parity_group_id             R15: Which RS parity group this page belongs to (u32)
52      1       preferred_tier              Preferred tier for this page (u8, 0-5, 0xFF = any)
53      1       access_pattern_type         R11: Detected pattern (u8, enum: streaming/working_set/phased/random/strided)
54      2       prefetch_next_vpn_delta     R11: Signed offset to predicted next page (i16, relative to current VPN)
56      4       ref_count                   Shared page reference count (u32)
60      4       reserved                    Future use (alignment padding to 64 bytes)
```

Total: **64 bytes exactly**, cache-line aligned.

### Flags Field (32 bits)

```
Bit     Name                    Description
---     ----                    -----------
0       VALID                   Page has valid physical backing
1       DIRTY                   Modified since last migration/checkpoint
2       PINNED                  Cannot be evicted (R30 doorbells, R20 regMr regions)
3       MIGRATING               In-flight transfer, block concurrent access
4       SHARED                  Referenced by multiple CUDA contexts
5       SUPERPAGE_MEMBER        Part of a 2MB super-page
6       READ_ONLY               Page is read-only (model weights, constants)
7       ACCESSED                Set on access, cleared by eviction scanner
8       EVICTION_CANDIDATE      Marked for eviction by policy scan
9       PREFETCH_TARGET         R11: Marked for proactive migration
10      COMPRESSED              R14: Page content is stored compressed in current tier
11      DEDUP_CANONICAL         R12: This is the canonical (master) copy for dedup
12      DEDUP_REFERENCE         R12: This page points to a canonical copy (dedup_hash is the key)
13      PARITY_VALID            R15: Parity data for this page's group is current
14      NCCL_REGISTERED         R20: Page is registered with NCCL regMr (avoid eviction during NCCL ops)
15      FAULT_PENDING           R19: A page fault is being serviced for this page
16-31   (reserved)              16 bits for future features
```

### Coherency State Field (16 bits) — R19

```
Bits    Name                Description
----    ----                -----------
0-1     state               2-bit enum: 0=Invalid, 1=Shared, 2=Exclusive (I/S/E per R19 SWMR)
2-9     sharer_bitmap       8-bit bitmap of nodes sharing this page (supports up to 8 nodes)
10-15   (reserved)          Future: transition counters, lock bits
```

### Design Rationale for Key Fields

**dedup_hash (16 bytes, offset 32):** R12 specifies xxHash128 for content-addressable dedup at 64KB pages. Full 128-bit hash avoids collisions across multi-TB address spaces. Truncating to 32 or 64 bits would require a secondary verification step on every dedup candidate, adding latency. 16 bytes is worth it.

**parity_group_id (4 bytes, offset 48):** R15 uses ISA-L Reed-Solomon encoding at 5-20 GB/s. Each parity group consists of K data pages + M parity pages (e.g., 4+2). The group ID links a page to its parity set. 4 bytes supports up to 4 billion parity groups (more than enough for 16 TB / 64KB = 268M pages).

**prefetch_next_vpn_delta (2 bytes, offset 54):** R11 predicts the next page access 1-3 kernels ahead. Storing a relative offset (i16) rather than absolute VPN saves 6 bytes. Range of +/- 32767 pages covers +/- 2 GB of address space, sufficient for stride-based and sequential patterns.

**access_pattern_type (1 byte, offset 53):** R11 classifies pages into pattern types. The eviction policy uses this: streaming pages are evicted first (one-pass data), working-set pages are retained longer, random pages get no special treatment.

---

## 3. Rust Trait Interfaces

### PageTable Trait

```rust
/// Cluster-wide virtual-to-physical page mapping.
/// Backed by a Robin Hood hash table with O(1) amortized lookup.
pub trait PageTable: Send + Sync {
    /// Look up a PTE by virtual page number. Returns None if not mapped.
    fn lookup(&self, vpn: u64) -> Option<PageTableEntry>;

    /// Insert or update a PTE. Returns the previous entry if updating.
    fn upsert(&self, entry: PageTableEntry) -> Option<PageTableEntry>;

    /// Remove a PTE. Returns the removed entry if it existed.
    fn remove(&self, vpn: u64) -> Option<PageTableEntry>;

    /// Atomically update flags on a PTE. Used for state transitions
    /// (e.g., setting MIGRATING, clearing ACCESSED).
    /// Returns false if the VPN is not mapped.
    fn update_flags(&self, vpn: u64, set: PteFlags, clear: PteFlags) -> bool;

    /// Atomically swap the tier/node/pfn fields (migration commit).
    /// Fails if the page is not in MIGRATING state.
    fn commit_migration(
        &self,
        vpn: u64,
        new_tier: TierId,
        new_node: NodeId,
        new_pfn: u64,
    ) -> Result<(), MigrationError>;

    /// Update coherency state (R19 integration).
    fn set_coherency(
        &self,
        vpn: u64,
        state: CoherencyState,
        sharer_bitmap: u8,
    ) -> Result<(), CoherencyError>;

    /// Update dedup hash after content scan (R12 integration).
    fn set_dedup_hash(&self, vpn: u64, hash: u128) -> bool;

    /// Set parity group assignment (R15 integration).
    fn set_parity_group(&self, vpn: u64, group_id: u32) -> bool;

    /// Update access pattern classification (R11 integration).
    fn set_access_pattern(
        &self,
        vpn: u64,
        pattern: AccessPatternType,
        next_vpn_delta: i16,
    ) -> bool;

    /// Scan pages matching a predicate. Used by eviction scanners,
    /// dedup scanners (R12), and parity rebuilders (R15).
    /// Returns an iterator to avoid allocating large result sets.
    fn scan<F>(&self, predicate: F) -> Box<dyn Iterator<Item = PageTableEntry> + '_>
    where
        F: Fn(&PageTableEntry) -> bool + 'static;

    /// Number of mapped pages.
    fn len(&self) -> usize;

    /// Bulk lookup for a contiguous VPN range (super-page support).
    fn lookup_range(&self, start_vpn: u64, count: usize) -> Vec<Option<PageTableEntry>>;
}
```

### TierManager Trait

```rust
/// Per-node service that decides page placement across all tiers.
pub trait TierManager: Send + Sync {
    /// Called when a new CUDA allocation is created. Returns the initial
    /// tier for physical backing (may be deferred/lazy).
    fn allocate(
        &self,
        alloc_id: u32,
        size_bytes: usize,
        hints: AllocationHints,
    ) -> Result<TierAllocation, AllocationError>;

    /// Called when a CUDA allocation is freed. Releases all pages
    /// across all tiers.
    fn deallocate(&self, alloc_id: u32) -> Result<(), DeallocationError>;

    /// Request promotion of a page to a faster tier.
    /// Returns the target tier, or None if the page is already optimal.
    fn request_promotion(&self, vpn: u64) -> Option<TierId>;

    /// Request demotion of a page to a slower tier (eviction).
    /// Returns the target tier and the page to evict.
    fn request_demotion(&self, tier: TierId) -> Option<(u64, TierId)>;

    /// Pin a page in its current tier (R20 regMr, R30 doorbells).
    /// Returns error if tier is critically full.
    fn pin_page(&self, vpn: u64) -> Result<(), PinError>;

    /// Unpin a previously pinned page.
    fn unpin_page(&self, vpn: u64) -> Result<(), PinError>;

    /// Query current capacity and usage for a tier.
    fn tier_status(&self, tier: TierId) -> TierStatus;

    /// Get the migration bandwidth budget (bytes/sec) for a tier pair.
    /// R11 uses this to schedule prefetches within budget.
    fn migration_budget(&self, from: TierId, to: TierId) -> u64;

    /// Notify that a page was accessed (from AccessMonitor).
    /// Updates internal eviction policy state.
    fn notify_access(&self, vpn: u64, access_type: AccessType);

    /// Register a compression callback (R14 integration).
    /// Called during migration to optionally compress page data.
    fn set_compression_hook(&mut self, hook: Box<dyn CompressionHook>);

    /// Register a dedup check callback (R12 integration).
    /// Called after page content is written to check for duplicates.
    fn set_dedup_hook(&mut self, hook: Box<dyn DedupHook>);
}
```

### EvictionPolicy Trait

```rust
/// Determines which page to evict when a tier is full.
/// Implementations: ARC (VRAM), CAR (DRAM), CLOCK (NVMe).
pub trait EvictionPolicy: Send + Sync {
    /// Record a page access. Updates internal recency/frequency state.
    fn record_access(&mut self, vpn: u64);

    /// Record a page insertion into this tier.
    fn record_insert(&mut self, vpn: u64);

    /// Record a page removal from this tier (evicted or freed).
    fn record_remove(&mut self, vpn: u64);

    /// Select the next page to evict. Returns None if tier is empty
    /// or all pages are pinned.
    /// The returned EvictionCandidate includes the destination tier
    /// recommendation from the destination scorer.
    fn select_victim(&mut self) -> Option<EvictionCandidate>;

    /// Check if a VPN is in the ghost list (ARC/CAR only).
    /// R11 uses ghost hits as a signal for prefetch priority.
    fn is_ghost_hit(&self, vpn: u64) -> bool;

    /// Number of pages currently tracked by this policy.
    fn tracked_count(&self) -> usize;

    /// Memory overhead of this policy instance (bytes).
    fn memory_overhead(&self) -> usize;

    /// Reset internal state (e.g., after major workload phase change).
    fn reset(&mut self);
}
```

### MigrationEngine Trait

```rust
/// Executes page transfers between tiers.
pub trait MigrationEngine: Send + Sync {
    /// Submit a migration request. Returns immediately; migration
    /// completes asynchronously.
    fn submit(
        &self,
        request: MigrationRequest,
    ) -> Result<MigrationHandle, MigrationError>;

    /// Submit a batch of migrations. More efficient than individual submits
    /// (amortizes DMA setup, NVMe command overhead).
    fn submit_batch(
        &self,
        requests: Vec<MigrationRequest>,
    ) -> Result<Vec<MigrationHandle>, MigrationError>;

    /// Poll a migration for completion.
    fn poll(&self, handle: &MigrationHandle) -> MigrationStatus;

    /// Wait for a migration to complete (blocking).
    fn wait(&self, handle: &MigrationHandle) -> Result<(), MigrationError>;

    /// Cancel a pending migration if possible.
    fn cancel(&self, handle: &MigrationHandle) -> Result<(), MigrationError>;

    /// Current in-flight migration count.
    fn in_flight_count(&self) -> usize;

    /// Current bandwidth utilization (bytes/sec) per tier pair.
    fn bandwidth_utilization(&self, from: TierId, to: TierId) -> u64;

    /// Set the maximum bandwidth budget for a tier pair (bytes/sec).
    /// R11's 70:20:10 split is enforced through this.
    fn set_bandwidth_limit(&self, from: TierId, to: TierId, limit: u64);
}
```

### AccessMonitor Trait

```rust
/// Tracks memory access patterns from CUDA interception.
/// Feeds data to eviction policies and prefetch engine (R11).
pub trait AccessMonitor: Send + Sync {
    /// Record an explicit memory access (cudaMemcpy, cudaMemset).
    fn record_memcpy(
        &self,
        dst_vpn_start: u64,
        src_vpn_start: u64,
        page_count: usize,
        direction: MemcpyDirection,
    );

    /// Record kernel launch with buffer arguments.
    /// Extracts VPNs from kernel pointer arguments.
    fn record_kernel_launch(
        &self,
        kernel_id: u64,
        buffer_vpns: &[u64],
        launch_config: &KernelLaunchConfig,
    );

    /// Record kernel completion. Used to measure actual access duration.
    fn record_kernel_complete(&self, kernel_id: u64);

    /// Get the detected access pattern for a page.
    fn get_pattern(&self, vpn: u64) -> AccessPatternType;

    /// Get prefetch predictions: which pages will be needed in the
    /// next N kernel launches. R11 calls this to schedule prefetches.
    fn predict_next_accesses(
        &self,
        lookahead_kernels: usize,
    ) -> Vec<PrefetchPrediction>;

    /// Get pages sorted by hotness (descending). Used by eviction
    /// policies and tier promotion decisions.
    fn hot_pages(&self, tier: TierId, limit: usize) -> Vec<(u64, f64)>;

    /// Get pages sorted by coldness (ascending). Used by eviction
    /// to identify candidates.
    fn cold_pages(&self, tier: TierId, limit: usize) -> Vec<(u64, f64)>;
}
```

---

## 4. Concrete Data Structures

### Core Types

```rust
/// Numeric tier identifier.
pub type TierId = u8;  // 0-5
pub type NodeId = u8;   // 0-255

/// Tier constants.
pub mod tiers {
    pub const LOCAL_VRAM: u8 = 0;
    pub const REMOTE_VRAM: u8 = 1;
    pub const LOCAL_DRAM: u8 = 2;
    pub const REMOTE_DRAM: u8 = 3;
    pub const LOCAL_NVME: u8 = 4;
    pub const REMOTE_NVME: u8 = 5;
    pub const TIER_COUNT: usize = 6;
}

/// Page Table Entry — 64 bytes, cache-line aligned.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct PageTableEntry {
    pub vpn: u64,                           // 8 bytes
    pub tier_id: TierId,                    // 1 byte
    pub node_id: NodeId,                    // 1 byte
    pub phys_pfn: [u8; 6],                  // 6 bytes (u48, packed)
    pub flags: PteFlags,                    // 4 bytes (bitflags)
    pub coherency: CoherencyField,          // 2 bytes
    pub access_count: u32,                  // 4 bytes
    pub last_access_ts: u32,                // 4 bytes
    pub alloc_id: u32,                      // 4 bytes
    pub migration_count: u16,               // 2 bytes
    pub last_migration_ts_delta: u16,       // 2 bytes
    pub dedup_hash: u128,                   // 16 bytes
    pub parity_group_id: u32,              // 4 bytes
    pub preferred_tier: u8,                 // 1 byte
    pub access_pattern_type: AccessPatternType, // 1 byte
    pub prefetch_next_vpn_delta: i16,       // 2 bytes
    pub ref_count: u32,                     // 4 bytes
    pub _reserved: u32,                     // 4 bytes padding
}
// Total: 8+1+1+6+4+2+4+4+4+2+2+16+4+1+1+2+4+4 = 64 bytes

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PteFlags: u32 {
        const VALID              = 1 << 0;
        const DIRTY              = 1 << 1;
        const PINNED             = 1 << 2;
        const MIGRATING          = 1 << 3;
        const SHARED             = 1 << 4;
        const SUPERPAGE_MEMBER   = 1 << 5;
        const READ_ONLY          = 1 << 6;
        const ACCESSED           = 1 << 7;
        const EVICTION_CANDIDATE = 1 << 8;
        const PREFETCH_TARGET    = 1 << 9;
        const COMPRESSED         = 1 << 10;
        const DEDUP_CANONICAL    = 1 << 11;
        const DEDUP_REFERENCE    = 1 << 12;
        const PARITY_VALID       = 1 << 13;
        const NCCL_REGISTERED    = 1 << 14;
        const FAULT_PENDING      = 1 << 15;
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AccessPatternType {
    Unknown    = 0,
    Streaming  = 1,  // Sequential one-pass (data loading)
    WorkingSet = 2,  // Repeated access to stable subset (model weights)
    Phased     = 3,  // Different sets in different phases (training)
    Random     = 4,  // Uniform random (hash tables, graphs)
    Strided    = 5,  // Regular non-sequential (convolution, matmul)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CoherencyField {
    raw: u16,
}

impl CoherencyField {
    pub fn state(&self) -> CoherencyState {
        match self.raw & 0x3 {
            0 => CoherencyState::Invalid,
            1 => CoherencyState::Shared,
            2 => CoherencyState::Exclusive,
            _ => CoherencyState::Invalid,
        }
    }

    pub fn sharer_bitmap(&self) -> u8 {
        ((self.raw >> 2) & 0xFF) as u8
    }

    pub fn new(state: CoherencyState, sharers: u8) -> Self {
        Self {
            raw: (state as u16) | ((sharers as u16) << 2),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoherencyState {
    Invalid   = 0,
    Shared    = 1,
    Exclusive = 2,
}
```

### Tier Configuration

```rust
/// Static configuration for a single tier.
pub struct TierConfig {
    pub tier_id: TierId,
    pub name: &'static str,
    pub capacity_bytes: u64,
    pub bandwidth_bytes_per_sec: u64,     // Theoretical max
    pub latency_ns: u64,                  // Typical access latency
    pub eviction_policy_type: EvictionPolicyType,
    pub is_local: bool,
    pub is_persistent: bool,              // NVMe tiers survive restart
    pub migration_batch_size: usize,      // Pages per batch for this tier
    pub min_residency_ms: u32,            // Anti-thrashing minimum
}

#[derive(Clone, Copy)]
pub enum EvictionPolicyType {
    Arc,    // VRAM tiers (0, 1)
    Car,    // DRAM tiers (2, 3)
    Clock,  // NVMe tiers (4, 5)
}

/// Default tier configurations for a 12GB VRAM + 256GB DRAM + 8TB NVMe node.
pub fn default_tier_configs() -> [TierConfig; 6] {
    [
        TierConfig {
            tier_id: 0, name: "Local VRAM",
            capacity_bytes: 12 * GB, bandwidth_bytes_per_sec: 900 * GB,
            latency_ns: 500, eviction_policy_type: EvictionPolicyType::Arc,
            is_local: true, is_persistent: false,
            migration_batch_size: 1, min_residency_ms: 100,
        },
        TierConfig {
            tier_id: 1, name: "Remote VRAM",
            capacity_bytes: 12 * GB, bandwidth_bytes_per_sec: 22 * GB,
            latency_ns: 5_000, eviction_policy_type: EvictionPolicyType::Arc,
            is_local: false, is_persistent: false,
            migration_batch_size: 4, min_residency_ms: 200,
        },
        TierConfig {
            tier_id: 2, name: "Local DRAM",
            capacity_bytes: 256 * GB, bandwidth_bytes_per_sec: 76 * GB,
            latency_ns: 100, eviction_policy_type: EvictionPolicyType::Car,
            is_local: true, is_persistent: false,
            migration_batch_size: 16, min_residency_ms: 500,
        },
        TierConfig {
            tier_id: 3, name: "Remote DRAM",
            capacity_bytes: 256 * GB, bandwidth_bytes_per_sec: 22 * GB,
            latency_ns: 5_000, eviction_policy_type: EvictionPolicyType::Car,
            is_local: false, is_persistent: false,
            migration_batch_size: 8, min_residency_ms: 500,
        },
        TierConfig {
            tier_id: 4, name: "Local NVMe",
            capacity_bytes: 8 * TB, bandwidth_bytes_per_sec: 15 * GB,
            latency_ns: 10_000, eviction_policy_type: EvictionPolicyType::Clock,
            is_local: true, is_persistent: true,
            migration_batch_size: 64, min_residency_ms: 5_000,
        },
        TierConfig {
            tier_id: 5, name: "Remote NVMe",
            capacity_bytes: 8 * TB, bandwidth_bytes_per_sec: 10 * GB,
            latency_ns: 50_000, eviction_policy_type: EvictionPolicyType::Clock,
            is_local: false, is_persistent: true,
            migration_batch_size: 64, min_residency_ms: 10_000,
        },
    ]
}
```

### Migration Types

```rust
/// A request to move a page between tiers.
pub struct MigrationRequest {
    pub vpn: u64,
    pub source_tier: TierId,
    pub source_node: NodeId,
    pub dest_tier: TierId,
    pub dest_node: NodeId,
    pub priority: MigrationPriority,
    pub compress: bool,               // R14: compress during transfer
    pub update_parity: bool,          // R15: recalculate parity after migration
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationPriority {
    Background = 0,    // Cold page demotion
    Normal     = 1,    // Eviction-driven migration
    Prefetch   = 2,    // R11 prefetch migration
    Fault      = 3,    // R19 page fault servicing (highest)
}

/// Opaque handle to track a submitted migration.
pub struct MigrationHandle {
    pub id: u64,
    pub vpn: u64,
    pub submitted_at: Instant,
}

pub enum MigrationStatus {
    Pending,
    InFlight { progress_bytes: usize, total_bytes: usize },
    Completed,
    Failed(MigrationError),
    Cancelled,
}

/// Current status of a tier on this node.
pub struct TierStatus {
    pub tier_id: TierId,
    pub total_pages: u64,
    pub used_pages: u64,
    pub pinned_pages: u64,
    pub migrating_pages: u64,
    pub compressed_pages: u64,          // R14: pages stored compressed
    pub dedup_saved_pages: u64,         // R12: pages saved by dedup
    pub utilization_percent: f32,
    pub pressure_level: PressureLevel,
}

#[derive(Clone, Copy)]
pub enum PressureLevel {
    Low,       // < 70% utilization
    Medium,    // 70-85% utilization
    High,      // 85-95% utilization
    Critical,  // > 95% utilization
}

/// Eviction candidate with destination recommendation.
pub struct EvictionCandidate {
    pub vpn: u64,
    pub source_tier: TierId,
    pub recommended_dest: TierId,
    pub reaccess_probability: f32,     // From eviction policy
}

/// Prefetch prediction from AccessMonitor (R11 integration point).
pub struct PrefetchPrediction {
    pub vpn: u64,
    pub current_tier: TierId,
    pub target_tier: TierId,           // Where to prefetch to
    pub confidence: f32,               // 0.0-1.0
    pub kernels_until_access: u32,     // Estimated kernels before needed
}

/// Hints for initial allocation placement.
pub struct AllocationHints {
    pub preferred_tier: Option<TierId>,
    pub read_only: bool,               // Model weights, constants
    pub temporary: bool,               // Activations, scratch buffers
    pub nccl_registered: bool,         // R20: will be used in NCCL ops
    pub pinned: bool,                  // R30: doorbell buffers
}

/// Result of a tier allocation.
pub struct TierAllocation {
    pub vpn_start: u64,
    pub page_count: usize,
    pub initial_tier: TierId,
}
```

### Hook Traits for Cross-Topic Integration

```rust
/// R14 compression hook, called during migration.
pub trait CompressionHook: Send + Sync {
    /// Attempt to compress page data. Returns compressed data and ratio,
    /// or None if compression is not beneficial (ratio < 1.5x).
    fn try_compress(&self, data: &[u8]) -> Option<(Vec<u8>, f32)>;

    /// Decompress page data.
    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8>;
}

/// R12 dedup hook, called after page content is written.
pub trait DedupHook: Send + Sync {
    /// Hash page content and check for duplicates.
    /// Returns Some(canonical_vpn) if a duplicate was found.
    fn check_dedup(&self, vpn: u64, data: &[u8]) -> Option<u64>;

    /// Notify that a page is being freed — update dedup reference counts.
    fn notify_free(&self, vpn: u64);
}

/// R15 parity hook, called after migration to update fault tolerance state.
pub trait ParityHook: Send + Sync {
    /// Notify that a page has migrated. Parity group may need recalculation.
    fn notify_migration(&self, vpn: u64, old_tier: TierId, new_tier: TierId);

    /// Rebuild a page from parity data (page loss recovery).
    fn rebuild_page(&self, vpn: u64) -> Result<Vec<u8>, ParityError>;
}
```

---

## 5. Integration Points

### R11 (Speculative Prefetching) Integration

**How R11 plugs into R10:**

1. **Data feed:** R11 reads from `AccessMonitor::predict_next_accesses()` to get prefetch candidates. The monitor's kernel-argument analysis gives R11 the buffer VPNs for upcoming kernels.

2. **Prefetch scheduling:** R11 calls `MigrationEngine::submit()` with `priority: MigrationPriority::Prefetch` and the predicted pages. The migration engine respects the 70:20:10 bandwidth split (70% compute, 20% prefetch, 10% system).

3. **PTE fields used:** `access_pattern_type` and `prefetch_next_vpn_delta` are written by the AccessMonitor and read by R11 to determine prefetch strategy (stride-based for `Strided`, sequential for `Streaming`, history-based for `WorkingSet`).

4. **Ghost list signal:** R11 calls `EvictionPolicy::is_ghost_hit()` to detect pages that were recently evicted and re-accessed. Ghost hits with high re-access probability trigger aggressive prefetching of surrounding pages.

5. **Budget enforcement:** R11 queries `TierManager::migration_budget(from, to)` before scheduling prefetches. If the prefetch budget is exhausted, R11 queues predictions for the next window.

### R12 (Memory Deduplication) Integration

**How R12 plugs into R10:**

1. **Hash computation:** After a page is written (detected via `AccessMonitor::record_memcpy` with a write direction), R12's `DedupHook::check_dedup()` is called. It computes xxHash128 of the page content and looks up the hash in a dedup index.

2. **PTE fields used:**
   - `dedup_hash` (16 bytes): Stores the xxHash128 of page content. Updated by R12 after each write.
   - `DEDUP_CANONICAL` flag: Marks the master copy of a deduplicated page.
   - `DEDUP_REFERENCE` flag: Marks a page that shares content with a canonical page. Its `phys_pfn` points to the canonical copy's physical location.

3. **Eviction interaction:** The eviction policy checks `DEDUP_CANONICAL` before evicting. A canonical page can only be evicted if all references are also evicted or remapped.

4. **Scanning:** R12 uses `PageTable::scan()` to find pages with matching `dedup_hash` values during background dedup passes.

### R14 (Transport Compression) Integration

**How R14 plugs into R10:**

1. **Migration-time compression:** When the `MigrationEngine` transfers a page between tiers, it calls `CompressionHook::try_compress()` if the migration request has `compress: true`. If compression is beneficial (ratio > 1.5x), the page is stored compressed in the destination tier.

2. **PTE flag:** `COMPRESSED` flag indicates the page is stored in compressed form. Any read of this page must decompress first. The migration engine handles this transparently.

3. **Capacity boost:** Compressed pages effectively increase tier capacity. `TierStatus::compressed_pages` tracks how many pages are compressed, and the tier manager accounts for this when calculating available space.

4. **Adaptive decision:** R14's adaptive compression (sample first, decide, cache decision per stream type) means compression is only applied when beneficial. The `CompressionHook` returns `None` for incompressible data, avoiding wasted cycles.

### R15 (Fault Tolerance) Integration

**How R15 plugs into R10:**

1. **Parity groups:** R15 assigns pages to RS parity groups (e.g., 4 data + 2 parity). The `parity_group_id` field in the PTE links each page to its group.

2. **Migration notification:** After any page migration, the `ParityHook::notify_migration()` is called. R15 may need to recalculate parity if the page moved to a different failure domain (different node).

3. **Recovery:** On node failure, R15 uses `ParityHook::rebuild_page()` to reconstruct lost pages from parity data. The rebuilt pages are inserted into the page table on surviving nodes.

4. **PTE flag:** `PARITY_VALID` indicates that the parity data for this page's group is current. A migration that invalidates parity clears this flag. R15's background parity repair sets it again.

5. **Bandwidth budget:** R15 parity computation (ISA-L RS at 5-20 GB/s) uses the system's 10% overhead budget. Parity updates are queued behind prefetch and eviction migrations.

### R19 (Network Page Faults) Integration

**How R19 plugs into R10:**

1. **Pre-launch paging:** Before each kernel launch, R19 checks that all kernel-argument buffer pages are in Tier 0 (local VRAM). Pages in remote tiers are migrated using `MigrationEngine::submit()` with `priority: MigrationPriority::Fault`.

2. **Coherency protocol:** R19 manages the I/S/E state in the `coherency` field of the PTE:
   - **Invalid (I):** Page not locally cached. Access triggers migration.
   - **Shared (S):** Multiple nodes have read-only copies. Write triggers invalidation.
   - **Exclusive (E):** Single node has read-write access. Remote read triggers downgrade to Shared.

3. **PTE fields used:** `coherency` (state + sharer_bitmap), `FAULT_PENDING` flag (set while servicing a fault to prevent duplicate handling).

4. **cuMemMap integration:** R19 uses CUDA VMM APIs (`cuMemCreate`, `cuMemMap`, `cuMemUnmap`) to dynamically map/unmap physical backing for virtual pages. The page table's `commit_migration()` method atomically updates the PTE after the `cuMemMap` completes.

### R20 (NCCL Backend) Integration

**How R20 plugs into R10:**

1. **regMr coordination:** When NCCL calls `regMr` on a GPU buffer, the NCCL plugin notifies the tier manager via `TierManager::pin_page()` for all pages in the registered region. This prevents the tier manager from evicting pages that NCCL expects to be in VRAM.

2. **PTE flag:** `NCCL_REGISTERED` is set on pinned NCCL regions. The eviction policy treats these as unpinnable until `deregMr` is called.

3. **One device per transport:** R20 reports each OuterLink transport as a separate NCCL device. The tier manager knows which transport device maps to which tier (ConnectX -> remote VRAM/DRAM, NVMe-oF -> remote NVMe) and can optimize page placement for NCCL traffic patterns.

### R30 (Persistent Kernels) Integration

**How R30 plugs into R10:**

1. **Pinned doorbell regions:** R30 ring buffer doorbells must reside in VRAM permanently. These are allocated with `AllocationHints { pinned: true, preferred_tier: Some(LOCAL_VRAM) }`.

2. **PTE flag:** `PINNED` is set on doorbell pages. The tier manager never evicts pinned pages. If VRAM is critically full, the tier manager evicts non-pinned pages first.

3. **Memory accounting:** Pinned pages for R30 reduce the effective VRAM capacity available for regular pages. `TierStatus::pinned_pages` tracks this.

---

## 6. Refined Phase Estimates

### Phase R10-A: Core Infrastructure (3 weeks)

**Scope:** Page table, 2-tier manager (VRAM + DRAM), basic migration engine, CUDA interception extensions.

**Deliverables:**
1. `PageTableEntry` struct (64 bytes, all fields initialized but downstream fields zeroed)
2. Robin Hood hash table with `PageTable` trait implementation
3. `TierManager` with 2-tier support (Tier 0 + Tier 2 only)
4. `MigrationEngine` for local VRAM <-> DRAM transfers via `cudaMemcpyAsync`
5. CUDA interception: `cudaMalloc` -> allocate in page table, `cudaFree` -> release
6. Basic `AccessMonitor`: intercept-time tracking only (no kernel argument analysis)
7. LRU eviction (placeholder, replaced by ARC in Phase B)

**Acceptance criteria:** CUDA app allocates 2x VRAM. Computation completes correctly with automatic overflow to DRAM.

**Cannot overlap with:** Nothing. This is the foundation.

### Phase R10-B: Eviction & Access Monitoring (3 weeks)

**Scope:** ARC/CAR/CLOCK eviction policies, kernel argument analysis, anti-thrashing.

**Deliverables:**
1. ARC eviction policy (VRAM) with ghost lists
2. CAR eviction policy (DRAM) with ghost lists
3. CLOCK eviction policy (NVMe, stub — no NVMe tier yet)
4. Kernel argument analysis in `AccessMonitor` (extract buffer pointers from `cuLaunchKernel`)
5. `AccessPatternType` classification (streaming/working_set/phased/random/strided)
6. Anti-thrashing: minimum residency, migration rate limiting, hysteresis bands
7. Destination scoring for eviction routing
8. Benchmarking framework

**Acceptance criteria:** Workload with phase changes (training then inference) shows ARC adapting. No thrashing under controlled tests. Ghost hit rate > 0 (policy is learning).

**Can overlap with:** R14-A (compression trait). R14 does not depend on eviction policies.

### Phase R10-C: NVMe Tier (3 weeks)

**Scope:** Tiers 4 and 5 (local and remote NVMe).

**Deliverables:**
1. NVMe tier driver with `io_uring` backend
2. CLOCK eviction policy wired to NVMe tiers
3. Batched I/O (64 pages = 4 MB per batch)
4. NVMe RAID-0 configuration management
5. Write endurance monitoring and alerting
6. GDS evaluation (if hardware supports it)

**Acceptance criteria:** CUDA app with 300 GB working set runs on 12 GB GPU + 256 GB DRAM + NVMe overflow. Results correct.

**Can overlap with:** R14-B (GPU compression). Compression can be tested independently.

### Phase R10-D: Remote Tiers & Cross-Topic Hooks (3 weeks)

**Scope:** Tiers 1, 3, 5 (remote tiers) + integration hooks for R11/R12/R14/R15/R19/R20.

**Deliverables:**
1. Remote VRAM tier (Tier 1) via existing transport
2. Remote DRAM tier (Tier 3) via existing transport
3. Remote NVMe tier (Tier 5) via network + NVMe
4. Distributed PTE management (allocation-based ownership)
5. Cross-node migration coordination (PTE cache for remote lookups)
6. `CompressionHook` integration point (R14)
7. `DedupHook` integration point (R12)
8. `ParityHook` integration point (R15)
9. `AccessMonitor::predict_next_accesses()` for R11
10. `TierManager::pin_page()` / `unpin_page()` for R20/R30
11. `PageTable::set_coherency()` for R19

**Acceptance criteria:** 2-node cluster with combined memory pool. Workload larger than any single node's resources completes correctly. All hook traits are callable (but downstream implementations are stubs).

**Can overlap with:** R20-A/B (NCCL skeleton + TCP backend). R20 needs `pin_page()` which ships in this phase.

### Phase R10-E: Optimization & Hardening (2 weeks)

**Scope:** Super-pages, concurrency, stress testing, performance tuning.

**Deliverables:**
1. 2MB super-page support for contiguous hot regions
2. Migration batching optimization (adaptive batch sizes)
3. Concurrent migration support (multiple pages in-flight per tier pair)
4. `cuMemMap`/`cuMemUnmap` integration for zero-sync page remapping
5. Stress testing under adversarial workloads
6. Performance regression test suite
7. Benchmark: actual bandwidth per tier pair on real hardware

**Acceptance criteria:** Performance within 2x of native (no tiering) for working-set-fits-in-VRAM. Graceful degradation for larger workloads.

**Can overlap with:** R11 (prefetching) can start once R10-D hooks are in place.

### Timeline Summary

```
Week:  1   2   3   4   5   6   7   8   9  10  11  12  13  14
R10-A: ===|===|===|
R10-B:             ===|===|===|
R14-A:             ===|===|===|                          (parallel)
R10-C:                         ===|===|===|
R14-B:                         ===|===|===|===|          (parallel)
R10-D:                                     ===|===|===|
R20-A:                                     ==|==|        (parallel)
R10-E:                                                ===|===|
R11:                                                  ===|===|... (continues)
```

**Total R10 estimate:** 14 weeks (3+3+3+3+2), with R14 and R20 overlapping where possible.

---

## 7. Updated Risk Assessment

### Risks Unchanged from v1

- Risk 1 (CUDA VMM on GeForce): HIGH/MEDIUM — Still needs hardware verification.
- Risk 3 (Thrashing): HIGH — Addressed by ARC + anti-thrashing. Architecture is sound.
- Risk 4 (Metadata overhead): MEDIUM/LOW — 64-byte PTE at 0.1% confirmed acceptable.
- Risk 5 (NVMe endurance): MEDIUM — Addressed by cold-page preference and rate limiting.
- Risk 6 (Concurrent access during migration): MEDIUM — Addressed by MIGRATING flag + copy-on-write.

### Risks Refined by Cross-Topic Analysis

**Risk 2 (Migration overhead) — DOWNGRADED to MEDIUM/LOW:**

R11 prefetching (1-3 kernels ahead, 70:20:10 bandwidth split) means most migrations happen proactively, not reactively. R14 compression reduces transfer size for compressible data by 2-4x. R17 topology awareness ensures migrations take the fastest path. The combination of prefetching + compression + topology-aware routing substantially reduces the effective migration overhead.

### New Risks from Cross-Topic Analysis

**Risk 7: Hook complexity blowout (MEDIUM/MEDIUM)**

R10 now has integration hooks for R11, R12, R14, R15, R19, R20, and R30. If hook interfaces change during downstream development, R10 becomes a bottleneck requiring frequent updates.

**Mitigation:** Hooks are defined as Rust traits with `Box<dyn Trait>` indirection. Downstream topics implement the trait; R10 calls through the trait. Changes to hook behavior don't require R10 changes as long as the trait signature is stable. Trait signatures are frozen after R10-D ships.

**Risk 8: PTE size pressure (LOW/MEDIUM)**

At 64 bytes, the PTE is exactly one cache line. If any downstream topic needs additional per-page metadata, we either exceed the cache line (performance hit) or must compress existing fields. The 4-byte `_reserved` field is the only remaining space.

**Mitigation:** The 4-byte reserved field can accommodate one more u32 field. Beyond that, per-page metadata overflow goes into a separate "extended metadata" table keyed by VPN, looked up only when needed (not on the hot path). The hot path only touches the 64-byte PTE.

---

## 8. Updated Success Criteria

All v1 criteria remain. Added cross-topic criteria:

| Criterion | Target |
|-----------|--------|
| All v1 criteria | (see preplan.md Section 7) |
| Hook trait stability | All hook traits compile with stub implementations |
| R11 data feed | `predict_next_accesses()` returns predictions within 10us |
| R12 hash computation | `set_dedup_hash()` does not block migration path |
| R14 compression integration | Compressed pages correctly stored/retrieved |
| R15 parity group tracking | `parity_group_id` correctly assigned for all pages |
| R19 coherency transitions | I->S->E->I state machine correct under concurrent access |
| R20 pin/unpin | `pin_page()` prevents eviction; `unpin_page()` releases |
| PTE cache-line alignment | `size_of::<PageTableEntry>() == 64`, `align_of == 64` |

---

## 9. Decisions Carried Forward (Still Need Confirmation)

All 7 decisions from v1 still need explicit confirmation before implementation begins. The cross-topic analysis strengthens the recommendations but does not change them:

1. **Base Page Size: 64KB** — Confirmed by R12 (dedup at 64KB), R19 (page fault unit), R17 (migration break-even analysis).
2. **Page Table Structure: Flat Hash Table** — Confirmed. R12 adds scan() requirement, achievable with iteration.
3. **Eviction Policy Architecture: Per-Tier** — Confirmed. ARC/CAR/CLOCK assignments validated.
4. **Access Monitoring: Intercept + Kernel Argument Analysis** — Confirmed by R11 (prefetch data source).
5. **PTE Ownership: Allocation-Based** — Confirmed for Phase 1. R19 coherency works with this model.
6. **NVMe Access: io_uring + GDS evaluation** — Confirmed. R21 adds NVMe-oF consideration for Tier 5.
7. **Virtual Address Space: Cluster-Wide Unified** — Confirmed by R19 (globally meaningful addresses).

---

## Related Documents

- `preplan.md` — Original v1 pre-plan (preserved for reference)
- `research/01-existing-tiering-systems.md` — Survey of tiering approaches
- `research/02-page-management-strategies.md` — Page size, table, PTE design
- `research/03-eviction-policies.md` — Eviction policy analysis
- `../../R14-transport-compression/preplan.md` — R14 compression architecture
- `../../R20-nccl-backend/preplan.md` — R20 NCCL plugin architecture
- `../../../phase-08-smart-memory/R11-speculative-prefetching/` — Prefetch integration
- `../../../phase-08-smart-memory/R12-memory-deduplication/` — Dedup integration
- `../../../phase-08-smart-memory/R19-network-page-faults/` — Page fault integration
- `../../../phase-09-hardening/R15-fault-tolerance/` — Parity integration
- `../../../phase-10-compute-distribution/R30-persistent-kernels/` — Pinned memory integration
- `../../../../docs/architecture/00-project-vision.md` — Project vision
- `../../../../planning/pre-planning/02-FINAL-PREPLAN.md` — Overall project pre-plan

## Open Questions (Remaining)

These are genuinely unresolved and require either hardware testing or downstream topic completion:

1. **CUDA VMM API availability on GeForce:** Must be tested on RTX 3060/4070 before implementation. If restricted, the shadow address space fallback (described in v1 Risk 1) is the contingency.

2. **Optimal hash table load factor under GPU access patterns:** Need benchmarking with real CUDA workload address patterns. Robin Hood hashing theory says 0.9 is fine, but clustering from GPU allocation patterns (large contiguous ranges) may require 0.75.

3. **Extended metadata overflow strategy:** If a future topic (post-R30) needs per-page metadata beyond the 4-byte reserved field, should we use a sidecar hash table or expand PTE to 128 bytes (two cache lines)? Deferred until the need arises.
