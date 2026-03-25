# R19: Network Page Faults / Unified Memory --- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Phase:** 8 --- Smart Memory
**Depends On:** R10 (Memory Tiering), R11 (Speculative Prefetching), R12 (Memory Deduplication)
**Supersedes:** [preplan.md](./preplan.md) (v1)

## Purpose

Second-round refinement of R19's pre-plan. This version adds exact Rust struct definitions, step-by-step pre-launch paging and fault handling flows, the complete I/S/E coherency state machine with all message types, a precise thrashing detection algorithm, and the coordination protocol between R19 (fault handler) and R11 (prefetcher) to prevent double-fetching. All designs integrate cross-topic findings from R10 v2, R11, R12, R15, R17, R22, and R25.

---

## 1. Exact Rust Structs

### 1.1 FaultHandler

The central fault handling component. Lives in the interception layer, intercepts cuLaunchKernel, and ensures all required pages are mapped before launch.

```rust
/// The fault handler. One per CUDA context (one per process per GPU).
/// Coordinates pre-launch mapping and crash recovery.
struct FaultHandler {
    /// The GPU this handler manages.
    gpu_id: GpuId,
    /// Node this GPU is on.
    node_id: NodeId,
    /// Reserved GPU virtual address range for the entire cluster memory pool.
    /// Created at init via cuMemAddressReserve.
    va_range_base: CUdeviceptr,
    va_range_size: usize,
    /// Reference to the page table (R10 v2).
    page_table: Arc<RwLock<PageTable>>,
    /// Reference to the coherency directory (Section 4).
    directory: Arc<CoherencyDirectory>,
    /// Reference to R11 prefetcher for coordination.
    prefetcher: Arc<Prefetcher>,
    /// Reference to R17 topology for source selection.
    topology: Arc<TopologyGraph>,
    /// Pending fault requests (in-flight fetches).
    pending_fetches: DashMap<PageId, FetchState>,
    /// Statistics for observability.
    stats: FaultStats,
    /// Configuration.
    config: FaultConfig,
}

struct FaultConfig {
    /// Maximum parallel RDMA fetch operations.
    /// ConnectX-5 supports 100+ outstanding ops.
    max_parallel_fetches: usize,       // default: 64
    /// Maximum pages to pre-map per kernel launch.
    max_premap_pages: usize,           // default: 4096
    /// Whether crash recovery (kernel re-launch) is enabled.
    crash_recovery_enabled: bool,      // default: true
    /// Maximum re-launch attempts before giving up.
    max_relaunch_attempts: u32,        // default: 3
    /// Timeout for a single page fetch (microseconds).
    fetch_timeout_us: u64,             // default: 10_000 (10ms)
}

struct FaultStats {
    /// Kernels launched total.
    kernels_launched: AtomicU64,
    /// Kernels that required pre-launch mapping (had missing pages).
    kernels_with_faults: AtomicU64,
    /// Total pages fetched via demand fault (not prefetch).
    pages_demand_fetched: AtomicU64,
    /// Kernel crashes caught and recovered.
    kernel_crashes_recovered: AtomicU64,
    /// Kernel crashes that could not be recovered.
    kernel_crashes_fatal: AtomicU64,
    /// Total pre-launch mapping time (microseconds).
    total_premap_time_us: AtomicU64,
}
```

### 1.2 FaultRequest

Represents a single page fault that needs resolution.

```rust
/// A request to fetch and map a remote page.
struct FaultRequest {
    /// Which page is needed.
    page_id: PageId,
    /// Virtual address where the page should be mapped.
    target_va: CUdeviceptr,
    /// Access type needed (read or write).
    access_type: AccessType,
    /// Priority of this fetch.
    priority: TransferPriority,
    /// Where to fetch from (resolved by coherency protocol + topology).
    source: FetchSource,
    /// When this request was created.
    created_at: Instant,
    /// Which kernel triggered this fault (for crash recovery tracking).
    kernel_id: Option<KernelId>,
}

enum AccessType {
    /// Kernel will read this page. Coherency state -> Shared.
    Read,
    /// Kernel will write this page. Coherency state -> Exclusive.
    Write,
    /// Unknown access type. Conservatively treat as Write.
    Unknown,
}

enum FetchSource {
    /// Fetch from the home node (always valid).
    Home(NodeId),
    /// Fetch from the nearest sharer (for Shared pages).
    /// R17 topology selects the closest node that has the page.
    NearestSharer(NodeId),
    /// Page is in local DRAM (R10 tier promotion, no network needed).
    LocalDram,
    /// Page is being fetched by R11 prefetcher. Wait for it instead of double-fetching.
    PrefetchInFlight,
}

/// State of an in-flight fetch operation.
enum FetchState {
    /// Coherency request sent to home node, waiting for reply.
    AwaitingCoherency,
    /// Data transfer in progress (RDMA READ or TCP).
    Transferring { source: NodeId, started_at: Instant },
    /// Data arrived, cuMemMap in progress.
    Mapping,
    /// Complete. Page is mapped and accessible.
    Complete,
    /// Failed. Will retry or escalate.
    Failed { error: FetchError, attempts: u32 },
}
```

### 1.3 CoherencyDirectory

Distributed directory. Each node maintains directory entries for pages it "homes."

```rust
/// The coherency directory. Each node runs one instance.
/// Responsible for pages whose home_node == this node.
struct CoherencyDirectory {
    /// This node's ID.
    node_id: NodeId,
    /// Directory entries for pages homed here.
    /// Key: PageId. Value: directory entry.
    /// Using DashMap for concurrent access from multiple handler threads.
    entries: DashMap<PageId, DirectoryEntry>,
    /// Thrashing detector (shared state, one per directory).
    thrashing: ThrashingDetector,
    /// Message handler for incoming coherency requests.
    message_tx: mpsc::Sender<CoherencyMessage>,
    /// Stats.
    stats: DirectoryStats,
}

/// Per-page directory entry at the home node.
/// Designed to fit within R10 v2's PTE coherency field where possible.
struct DirectoryEntry {
    /// Current coherency state.
    state: PageState,
    /// If state == Exclusive: the owning node.
    owner: Option<NodeId>,
    /// If state == Shared: bitmap of nodes holding read copies.
    /// Uses R10 v2's 13-bit sharer bitmap (supports up to 13 nodes).
    sharers: SharerBitmap,
    /// Monotonically increasing version. Incremented on every write.
    version: u64,
    /// Which node last wrote this page.
    last_writer: NodeId,
    /// Thrashing detection: state transitions in current epoch.
    bounce_count: u32,
    /// When the last state transition occurred.
    last_transition: Instant,
    /// Whether this page is deduped (R12).
    /// Deduped pages are permanently Shared --- never transition to Exclusive.
    is_deduped: bool,
    /// Whether this page is being reconstructed by R15 (fault tolerance).
    /// If true, faults must wait for reconstruction to complete.
    is_recovering: bool,
    /// Whether a fault is currently pending for this page.
    /// Maps to R10 v2's FAULT_PENDING PTE flag.
    fault_pending: bool,
}

/// 13-bit sharer bitmap (from R10 v2 PTE coherency field).
/// Supports up to 13 nodes. For larger clusters, overflow to a heap-allocated set.
#[derive(Clone, Copy)]
struct SharerBitmap {
    /// Bitmap where bit i = 1 means node i has a Shared copy.
    bits: u16,  // Uses 13 of 16 bits. Upper 3 bits reserved.
}

impl SharerBitmap {
    fn add(&mut self, node: NodeId)    { self.bits |= 1 << node.0; }
    fn remove(&mut self, node: NodeId) { self.bits &= !(1 << node.0); }
    fn contains(&self, node: NodeId) -> bool { (self.bits >> node.0) & 1 == 1 }
    fn count(&self) -> u32 { self.bits.count_ones() }
    fn is_empty(&self) -> bool { self.bits == 0 }
    fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..13).filter(move |&i| (self.bits >> i) & 1 == 1).map(NodeId)
    }
    fn clear(&mut self) { self.bits = 0; }
}
```

### 1.4 PageState

The I/S/E state used in both PTE coherency fields and directory entries.

```rust
/// Page coherency state. Matches R10 v2 PTE coherency field encoding.
/// Stored in 2 bytes: 3-bit state + 13-bit sharer bitmap.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    /// Invalid: no local copy. Must fetch before access.
    Invalid,
    /// Shared: read-only local copy. Home tracks us in sharer bitmap.
    Shared,
    /// Exclusive: read-write local copy. We are the sole owner.
    Exclusive,
}
```

### 1.5 Coherency Messages

Complete message types for the directory-based protocol.

```rust
/// All coherency protocol messages.
enum CoherencyMessage {
    // === Requests (Node -> Home) ===

    /// Request a Shared (read-only) copy of a page.
    ReadRequest {
        page_id: PageId,
        requester: NodeId,
        /// True if this is a speculative prefetch (R11).
        /// Tentative reads don't add requester to sharer set.
        tentative: bool,
    },

    /// Request Exclusive (read-write) ownership of a page.
    WriteRequest {
        page_id: PageId,
        requester: NodeId,
    },

    /// Upgrade from Shared to Exclusive (requester already has data).
    UpgradeRequest {
        page_id: PageId,
        requester: NodeId,
    },

    /// Notify home that we're evicting our copy.
    EvictNotify {
        page_id: PageId,
        node: NodeId,
        /// If we held Exclusive, include dirty data for writeback.
        dirty_data: Option<PageData>,
    },

    // === Replies (Home -> Node) ===

    /// Grant Shared access. Includes page data and current version.
    ReadReply {
        page_id: PageId,
        data: PageData,
        version: u64,
    },

    /// Grant Exclusive access. Includes page data and new version.
    WriteReply {
        page_id: PageId,
        data: PageData,
        version: u64,
    },

    /// Grant upgrade (no data transfer, requester already has it).
    UpgradeReply {
        page_id: PageId,
        version: u64,
    },

    /// Reject request due to thrashing mitigation (try again later).
    Nack {
        page_id: PageId,
        reason: NackReason,
        retry_after_us: u64,
    },

    // === Invalidation (Home -> Sharers) ===

    /// Demand invalidation of a Shared copy.
    Invalidate {
        page_id: PageId,
        version: u64,
    },

    /// Acknowledgment of invalidation.
    InvalidateAck {
        page_id: PageId,
        node: NodeId,
    },

    // === Fetch (Home -> Owner) ===

    /// Request data from Exclusive owner (for downgrade E->S).
    FetchRequest {
        page_id: PageId,
    },

    /// Owner returns data and downgrades to Shared.
    FetchReply {
        page_id: PageId,
        data: PageData,
    },

    // === Thrashing Mitigation ===

    /// Home instructs nodes to change behavior for a thrashing page.
    ThrashMitigation {
        page_id: PageId,
        action: ThrashAction,
    },
}

enum NackReason {
    /// Page is pinned due to thrashing. Retry after the pin expires.
    ThrashingPin,
    /// Page is being recovered by R15. Wait for reconstruction.
    Recovery,
    /// Page fault already pending from another requester.
    FaultPending,
}

enum ThrashAction {
    /// Switch to write-broadcast for this page. Writer sends updates to all sharers.
    WriteBroadcast,
    /// Pin the page to current owner for the specified duration.
    Pin { duration_us: u64 },
    /// Promote to shared-read: all nodes get read copies, writer broadcasts diffs.
    SharedReadPromotion,
}

/// 64KB page data. Boxed to avoid stack overflow.
type PageData = Box<[u8; 65536]>;
```

---

## 2. Pre-Launch Paging Flow

Step-by-step: what happens when cuLaunchKernel is intercepted.

```
cuLaunchKernel INTERCEPTION FLOW
=================================

Step 1: INTERCEPT
  The LD_PRELOAD interception layer catches cuLaunchKernel(kernel, grid, block, args, ...).
  Extract kernel function pointer and argument pointers from `args`.

Step 2: IDENTIFY REQUIRED PAGES (via R8 Kernel Param Introspection)
  For each kernel argument that is a device pointer:
    a. Determine the allocation it belongs to (from R10's VA range tracker).
    b. Compute the page range: [base_page_id .. base_page_id + num_pages).
       Each page is 64KB (R10 v2 page size).
    c. Determine access type:
       - If kernel is known read-only for this arg (from R8 analysis): AccessType::Read
       - If kernel writes to this arg: AccessType::Write
       - If unknown: AccessType::Unknown (treated as Write, conservative)
  Collect into: required_pages: Vec<(PageId, AccessType)>

Step 3: CHECK LOCAL MAPPING STATUS
  For each (page_id, access_type) in required_pages:
    pte = page_table.lookup(page_id)
    match (pte.coherency_state, access_type):
      (Exclusive, _)     -> already mapped read-write. Skip.
      (Shared, Read)     -> already mapped read-only. Skip.
      (Shared, Write)    -> need upgrade S->E. Add to upgrade_list.
      (Invalid, _)       -> not mapped. Add to fetch_list.
      (_, Unknown)       -> if state is Invalid: fetch. If Shared: upgrade. If Exclusive: skip.

  Result: fetch_list (pages to retrieve) + upgrade_list (pages to upgrade S->E)

Step 4: CHECK R11 PREFETCH IN-FLIGHT
  For each page in fetch_list:
    if prefetcher.is_in_flight(page_id):
      // R11 is already fetching this page. Don't double-fetch.
      // Move to wait_list instead.
      wait_list.push(page_id)
      fetch_list.remove(page_id)

Step 5: RESOLVE FETCH SOURCES (via coherency directory + R17 topology)
  For each page in fetch_list:
    a. Send coherency request to home node:
       - If access_type == Read:  send ReadRequest { page_id, tentative: false }
       - If access_type == Write: send WriteRequest { page_id }
    b. Home node processes request (Section 4 state machine).
    c. Home replies with data source:
       - ReadReply with data from home's canonical copy, OR
       - ReadReply with redirect to nearest sharer (R17 picks closest), OR
       - WriteReply after invalidating all sharers.
    d. Fetch data from source:
       - Use RDMA READ for RDMA-connected sources (single round-trip, ~2+5.1=7.1 us).
       - Use TCP for TCP-only sources (~73 us).
       - R17 selects the best link via select_route(src, dst, 64KB, DemandFetch).

  For each page in upgrade_list:
    a. Send UpgradeRequest { page_id } to home.
    b. Home invalidates other sharers.
    c. Home replies with UpgradeReply { version }.
    d. No data transfer needed (we already have the data).

Step 6: PARALLEL FETCH EXECUTION
  Issue up to config.max_parallel_fetches (default 64) concurrent RDMA READs.
  ConnectX-5 handles 100+ outstanding operations without degradation.
  As each fetch completes:
    a. cuMemMap(target_va, 64KB, physical_handle, 0)     [~10-30 us]
    b. cuMemSetAccess(target_va, 64KB, access_descriptor) [~5-15 us]
    c. Update PTE: state = Shared|Exclusive, version = reply.version
    d. Set PTE.FAULT_PENDING = false

Step 7: WAIT FOR R11 IN-FLIGHT PAGES
  For each page in wait_list:
    prefetcher.wait_for(page_id, timeout=config.fetch_timeout_us)
    If timeout: escalate to demand fetch (go back to Step 5 for this page).

Step 8: LAUNCH KERNEL
  All required pages are now mapped. Call the real cuLaunchKernel.
  Record kernel_id for crash recovery tracking.

TIMING ANALYSIS:
  Best case (all pages prefetched by R11): ~0 us overhead (nothing to fetch).
  Typical (5% miss rate, 5 pages to fetch): 5 * 7.1 us fetch + 5 * 25 us map = ~161 us
  Worst case (100 pages, all remote): parallel fetch ~200 us + serial map ~2500 us = ~2.7 ms
  (Serial cuMemMap is the bottleneck. Open question: can cuMemMap be parallelized?)
```

---

## 3. Fault Handling Flow (Kernel Crash Recovery)

When pre-launch mapping misses a page and the kernel crashes.

```
KERNEL CRASH RECOVERY FLOW
============================

Step 1: DETECT CRASH
  cuLaunchKernel returns CUresult.
  After kernel completes (or cuStreamSynchronize), check for:
    CUDA_ERROR_ILLEGAL_ADDRESS   -- unmapped page accessed
    CUDA_ERROR_LAUNCH_FAILED     -- other kernel failure (may be related)

Step 2: IDENTIFY MISSING PAGES
  Option A (if CUDA provides faulting address):
    Parse error info to get the faulting virtual address.
    Compute page_id = (faulting_va - va_range_base) / PAGE_SIZE.
    missing_pages = vec![page_id]

  Option B (CUDA does NOT provide faulting address -- likely case):
    Re-analyze kernel arguments (same as Step 2 of pre-launch flow).
    For each argument page: check if it's currently mapped.
    missing_pages = all pages that are currently Invalid in the PTE.
    This over-fetches (maps everything the kernel MIGHT need), but guarantees
    correctness. R11 prefetching should make this list small.

Step 3: FETCH AND MAP MISSING PAGES
  Same as Steps 5-6 of pre-launch flow.
  Priority: DemandFetch (highest) -- these pages stalled a kernel.

Step 4: RESET CUDA CONTEXT
  The CUDA context may be in an error state after ILLEGAL_ADDRESS.
  Call cuCtxSynchronize() to clear pending errors.
  If the error is unrecoverable (context corrupted):
    cuCtxDestroy() + cuCtxCreate() -- recreate the context.
    Re-map ALL locally-resident pages into the new context's VA space.
    This is expensive (~50ms for 10K pages) but should be extremely rare.

Step 5: RE-LAUNCH KERNEL
  Call cuLaunchKernel again with the same arguments.
  Increment relaunch_attempt counter.
  If relaunch_attempt > config.max_relaunch_attempts:
    Log fatal error. Surface to application.
    stats.kernel_crashes_fatal += 1.

Step 6: UPDATE STATS
  stats.kernel_crashes_recovered += 1
  Feed the missed page_ids back to R11 prefetcher:
    prefetcher.report_miss(kernel_id, missing_pages)
  R11 updates its prediction model to prefetch these pages for future kernels.

TIMING ANALYSIS:
  Context sync + error detection: ~100 us
  Page identification (Option B): ~10 us
  Fetch + map (single page, RDMA): ~32 us
  Fetch + map (10 pages, parallel): ~200 us
  Context reset (if needed): ~50 ms (rare)
  Total typical: ~350 us for single page miss (excluding context reset)
  This is 50-100x more expensive than pre-launch mapping, validating that
  R11 prefetching (preventing faults) is the primary mechanism.
```

---

## 4. Coherency Protocol: I/S/E State Machine

### 4.1 State Transition Diagram

```
                      ReadRequest
              +----------------------+
              |                      |
              v                      |
    +=========+    WriteRequest    +===+
    | INVALID | -----------------> | E |  (Exclusive)
    +=========+                    +===+
        ^  ^                        | |
        |  |   INVALIDATE /         | | FetchRequest
        |  |   EVICT                | | (from home, another node wants to read)
        |  |                        | v
        |  |   INVALIDATE         +===+
        |  +<-------------------- | S |  (Shared)
        |                         +===+
        |                           |
        +------ EVICT (voluntary) --+

    UpgradeRequest: S -> E (no data transfer, invalidate other sharers)
    WriteRequest from I: I -> E (fetch data + get exclusive)
    ReadRequest from I: I -> S (fetch data, join sharer set)
    FetchRequest on E owner: E -> S (return data to home, join sharer set)
```

### 4.2 Home Node State Machine (Processing Incoming Requests)

```rust
impl CoherencyDirectory {
    fn handle_read_request(&self, page_id: PageId, requester: NodeId, tentative: bool)
        -> CoherencyMessage
    {
        let mut entry = self.entries.get_mut(&page_id).unwrap();

        // R15 integration: if page is being recovered, NACK.
        if entry.is_recovering {
            return CoherencyMessage::Nack {
                page_id, reason: NackReason::Recovery, retry_after_us: 1000
            };
        }

        // Thrashing check: if page is pinned, only the pin-holder can access.
        if let Some(action) = self.thrashing.check_and_record(&mut entry) {
            return self.apply_thrash_action(page_id, requester, action);
        }

        match entry.state {
            PageState::Invalid => {
                // No one has it. Send data from home's canonical copy.
                if !tentative { entry.sharers.add(requester); }
                entry.state = PageState::Shared;
                CoherencyMessage::ReadReply {
                    page_id,
                    data: self.load_canonical_data(page_id),
                    version: entry.version,
                }
            }
            PageState::Shared => {
                // Others already reading. Add requester to sharer set.
                if !tentative { entry.sharers.add(requester); }
                // Optimization: if requester is close to an existing sharer,
                // redirect to that sharer instead of sending from home.
                // R17 topology picks the nearest sharer.
                let nearest = self.find_nearest_sharer(page_id, requester);
                CoherencyMessage::ReadReply {
                    page_id,
                    data: self.load_data_from(nearest),
                    version: entry.version,
                }
            }
            PageState::Exclusive => {
                // Owner has the latest data. Must fetch from owner first.
                let owner = entry.owner.unwrap();
                // Send FetchRequest to owner. Owner returns data and downgrades to S.
                let data = self.fetch_from_owner(page_id, owner); // blocks until reply
                if !tentative { entry.sharers.add(requester); }
                entry.sharers.add(owner); // Owner also becomes a sharer.
                entry.owner = None;
                entry.state = PageState::Shared;
                CoherencyMessage::ReadReply {
                    page_id,
                    data,
                    version: entry.version,
                }
            }
        }
    }

    fn handle_write_request(&self, page_id: PageId, requester: NodeId)
        -> CoherencyMessage
    {
        let mut entry = self.entries.get_mut(&page_id).unwrap();

        // R12 integration: deduped pages cannot be written exclusively.
        // Writing to a deduped page triggers COW at R12 layer, which creates
        // a new non-deduped page. The write request here is for the NEW page.
        if entry.is_deduped {
            panic!("BUG: WriteRequest for deduped page. R12 COW should have created a new page.");
        }

        if entry.is_recovering {
            return CoherencyMessage::Nack {
                page_id, reason: NackReason::Recovery, retry_after_us: 1000
            };
        }

        if let Some(action) = self.thrashing.check_and_record(&mut entry) {
            return self.apply_thrash_action(page_id, requester, action);
        }

        match entry.state {
            PageState::Invalid => {
                // No one has it. Grant exclusive.
                entry.state = PageState::Exclusive;
                entry.owner = Some(requester);
                entry.sharers.clear();
                entry.version += 1;
                entry.last_writer = requester;
                CoherencyMessage::WriteReply {
                    page_id,
                    data: self.load_canonical_data(page_id),
                    version: entry.version,
                }
            }
            PageState::Shared => {
                // Must invalidate all sharers before granting exclusive.
                let sharers_to_invalidate: Vec<NodeId> = entry.sharers.iter()
                    .filter(|&n| n != requester)
                    .collect();

                // Send INVALIDATE to all sharers in parallel.
                // Wait for all INVALIDATE_ACKs before proceeding.
                self.invalidate_sharers(page_id, &sharers_to_invalidate, entry.version);

                entry.state = PageState::Exclusive;
                entry.owner = Some(requester);
                entry.sharers.clear();
                entry.version += 1;
                entry.last_writer = requester;

                // If requester already had a Shared copy, no data transfer needed.
                // If not, send data.
                let data = self.load_canonical_data(page_id);
                CoherencyMessage::WriteReply {
                    page_id, data, version: entry.version,
                }
            }
            PageState::Exclusive => {
                // Current owner must give up ownership.
                let old_owner = entry.owner.unwrap();
                let data = self.fetch_from_owner_and_invalidate(page_id, old_owner);

                entry.state = PageState::Exclusive;
                entry.owner = Some(requester);
                entry.sharers.clear();
                entry.version += 1;
                entry.last_writer = requester;

                CoherencyMessage::WriteReply {
                    page_id, data, version: entry.version,
                }
            }
        }
    }

    fn handle_upgrade_request(&self, page_id: PageId, requester: NodeId)
        -> CoherencyMessage
    {
        let mut entry = self.entries.get_mut(&page_id).unwrap();

        // Requester must currently be in Shared state.
        assert!(entry.state == PageState::Shared);
        assert!(entry.sharers.contains(requester));

        if entry.is_deduped {
            panic!("BUG: UpgradeRequest for deduped page.");
        }

        // Invalidate all OTHER sharers.
        let others: Vec<NodeId> = entry.sharers.iter()
            .filter(|&n| n != requester)
            .collect();
        self.invalidate_sharers(page_id, &others, entry.version);

        entry.state = PageState::Exclusive;
        entry.owner = Some(requester);
        entry.sharers.clear();
        entry.version += 1;
        entry.last_writer = requester;

        // No data transfer: requester already has the data.
        CoherencyMessage::UpgradeReply {
            page_id, version: entry.version,
        }
    }

    fn handle_evict_notify(&self, page_id: PageId, node: NodeId,
                           dirty_data: Option<PageData>)
    {
        let mut entry = self.entries.get_mut(&page_id).unwrap();

        match entry.state {
            PageState::Shared => {
                entry.sharers.remove(node);
                if entry.sharers.is_empty() {
                    entry.state = PageState::Invalid;
                }
                // Shared pages are clean: no writeback needed.
            }
            PageState::Exclusive => {
                assert_eq!(entry.owner, Some(node));
                if let Some(data) = dirty_data {
                    // Writeback dirty data to home's canonical store.
                    self.store_canonical_data(page_id, data);
                }
                entry.owner = None;
                entry.state = PageState::Invalid;
            }
            PageState::Invalid => {
                // Already evicted or never fetched. No-op.
            }
        }
    }
}
```

### 4.3 Invalidation Protocol

```
Parallel invalidation for S->E transition:

Home sends INVALIDATE(page_id, version) to each sharer simultaneously.
Each sharer:
  1. cuMemUnmap the page's VA range (remove mapping).
  2. Update local PTE: state = Invalid.
  3. Send INVALIDATE_ACK(page_id, node_id) to home.

Home waits for ALL acks before granting Exclusive.
Timeout: 2 * max_rtt to any sharer (from R17 topology).
If a sharer doesn't respond:
  - It may have crashed or silently evicted.
  - After timeout: assume the sharer no longer has the page.
  - Mark the sharer as suspect (feed into heartbeat/phi detector).
  - Proceed with granting Exclusive.

Bulk invalidation optimization (R29 integration):
  If sharer_count > 3 and RDMA multicast is available:
    Use RDMA multicast INVALIDATE instead of point-to-point.
    Wait for individual ACKs (multicast is unreliable, need per-node confirmation).
```

---

## 5. Thrashing Detection and Escalation

### 5.1 Detection Algorithm

```rust
struct ThrashingDetector {
    /// Epoch duration for counting bounces.
    epoch_duration: Duration,  // default: 10 ms
    /// Bounces per epoch before Level 1 mitigation.
    l1_threshold: u32,         // default: 5
    /// Bounces per epoch before Level 2 mitigation.
    l2_threshold: u32,         // default: 10
    /// Bounces per epoch before Level 3 mitigation (pin).
    l3_threshold: u32,         // default: 20
    /// Duration to pin a page at Level 3.
    pin_duration: Duration,    // default: 20 ms (2x epoch)
}

impl ThrashingDetector {
    /// Called by the directory on every state transition (I->S, S->E, E->S, etc.).
    /// Returns None if no thrashing, or a mitigation action.
    fn check_and_record(&self, entry: &mut DirectoryEntry) -> Option<ThrashAction> {
        let now = Instant::now();

        // Reset epoch if expired.
        if now.duration_since(entry.last_transition) > self.epoch_duration {
            entry.bounce_count = 0;
        }

        entry.bounce_count += 1;
        entry.last_transition = now;

        if entry.bounce_count >= self.l3_threshold {
            // Level 3: Pin the page to current owner.
            Some(ThrashAction::Pin { duration_us: self.pin_duration.as_micros() as u64 })
        } else if entry.bounce_count >= self.l2_threshold {
            // Level 2: Write-broadcast (writer sends updates to all sharers).
            Some(ThrashAction::WriteBroadcast)
        } else if entry.bounce_count >= self.l1_threshold {
            // Level 1: Shared-read promotion (all readers get copies, writer broadcasts).
            Some(ThrashAction::SharedReadPromotion)
        } else {
            None
        }
    }
}
```

### 5.2 Escalation Levels

```
Level 0: Normal
  Bounce count < 5 per epoch.
  Standard I/S/E transitions.

Level 1: Shared-Read Promotion (bounce_count >= 5)
  Detected pattern: many readers, one writer alternating S<->E.
  Action:
    - All requesting nodes get Shared copies (no invalidation for reads).
    - Writer broadcasts page updates (diffs) to all sharers after writing.
    - Sharers apply diffs locally without re-fetching.
  Cost: O(sharers) bandwidth per write.
  Benefit: eliminates all read faults.
  Particularly effective for: model weights during training (optimizer writes,
  all GPUs read).

Level 2: Write-Broadcast (bounce_count >= 10)
  Detected pattern: persistent read-write contention from multiple writers.
  Action:
    - When any node writes, it sends the entire updated page to all nodes.
    - All nodes always have the latest copy.
    - No invalidation needed (updates push instead of pull).
  Cost: O(N) * 64KB bandwidth per write.
  Benefit: zero fault latency for all subsequent accesses.
  Only viable for infrequently-written pages (e.g., weights updated once per iteration).

Level 3: Pin (bounce_count >= 20)
  Detected pattern: unresolvable contention. Multiple nodes writing frequently.
  Action:
    - Pin the page to its current owner for pin_duration (20 ms).
    - All other requests receive NACK with retry_after = pin_duration.
    - Stalled requestors:
      a. If they have a Shared copy: use stale data (acceptable for some workloads).
      b. If they have no copy: block until pin expires.
  Cost: one party stalls.
  Benefit: stops the thrashing completely for the pin duration.
  Risk: priority inversion. A high-priority kernel may stall.
  Mitigation: if a DemandFetch request arrives during a pin and the requester
  has no copy at all, override the pin and grant access (correctness over performance).

De-escalation:
  After one epoch with bounce_count < l1_threshold:
    Drop back to Level 0 (normal protocol).
  After two epochs at Level 1 with declining bounce count:
    Drop to Level 0.
  Pins auto-expire after pin_duration.
```

---

## 6. Integration with R11 Prefetcher (Preventing Double-Fetch)

### 6.1 Coordination Protocol

R11 (prefetcher) and R19 (fault handler) share access to the same page table and must coordinate to avoid double-fetching the same page.

```rust
/// Shared state between R11 prefetcher and R19 fault handler.
/// Prevents the scenario where R11 starts prefetching page P, and then
/// R19's pre-launch check also tries to fetch P.
struct PrefetchFaultCoordinator {
    /// Pages currently being fetched by R11 (prefetcher).
    /// Key: PageId. Value: expected completion time.
    prefetch_in_flight: DashMap<PageId, PrefetchInFlightState>,
    /// Pages currently being fetched by R19 (demand fault).
    demand_in_flight: DashMap<PageId, Instant>,
}

struct PrefetchInFlightState {
    /// When the prefetch was initiated.
    started_at: Instant,
    /// Expected completion (based on link latency from R17).
    expected_complete: Instant,
    /// Whether this is a tentative prefetch (low confidence).
    tentative: bool,
    /// Notification channel: R19 can wait on this for completion.
    notify: Arc<Notify>,
}

impl PrefetchFaultCoordinator {
    /// Called by R11 before starting a prefetch.
    /// Returns false if R19 is already demand-fetching this page.
    fn prefetch_start(&self, page_id: PageId, expected_latency: Duration, tentative: bool)
        -> bool
    {
        if self.demand_in_flight.contains_key(&page_id) {
            return false; // R19 already fetching. Don't double-fetch.
        }
        self.prefetch_in_flight.insert(page_id, PrefetchInFlightState {
            started_at: Instant::now(),
            expected_complete: Instant::now() + expected_latency,
            tentative,
            notify: Arc::new(Notify::new()),
        });
        true
    }

    /// Called by R11 when prefetch completes (success or failure).
    fn prefetch_complete(&self, page_id: PageId) {
        if let Some((_, state)) = self.prefetch_in_flight.remove(&page_id) {
            state.notify.notify_waiters(); // Wake up any R19 waiters.
        }
    }

    /// Called by R19 during pre-launch check.
    /// Returns true if R11 is already fetching this page.
    fn is_prefetch_in_flight(&self, page_id: PageId) -> bool {
        self.prefetch_in_flight.contains_key(&page_id)
    }

    /// Called by R19 to wait for R11's in-flight prefetch to complete.
    /// Returns Ok(()) when the prefetch completes, or Err if timeout.
    async fn wait_for_prefetch(&self, page_id: PageId, timeout: Duration)
        -> Result<(), FetchTimeout>
    {
        if let Some(state) = self.prefetch_in_flight.get(&page_id) {
            let notify = state.notify.clone();
            drop(state); // Release the DashMap ref before awaiting.
            tokio::time::timeout(timeout, notify.notified()).await
                .map_err(|_| FetchTimeout)?;
            Ok(())
        } else {
            // Prefetch already completed between our check and this call.
            Ok(())
        }
    }

    /// Called by R19 before starting a demand fetch.
    /// If R11 has a prefetch in-flight, R19 waits instead of double-fetching.
    /// If no prefetch in-flight, R19 registers itself and proceeds.
    fn demand_fetch_start(&self, page_id: PageId) -> DemandFetchAction {
        if let Some(state) = self.prefetch_in_flight.get(&page_id) {
            let remaining = state.expected_complete.saturating_duration_since(Instant::now());
            if remaining < Duration::from_micros(100) {
                // Prefetch is about to complete. Wait for it.
                DemandFetchAction::WaitForPrefetch(state.notify.clone())
            } else {
                // Prefetch has a while to go. Could wait, but for DemandFetch
                // (highest priority), we might want to fetch in parallel on a
                // faster link. Decision: wait if remaining < 500us, else parallel fetch.
                if remaining < Duration::from_micros(500) {
                    DemandFetchAction::WaitForPrefetch(state.notify.clone())
                } else {
                    // Cancel the prefetch and do a demand fetch instead.
                    DemandFetchAction::CancelPrefetchAndFetch
                }
            }
        } else {
            self.demand_in_flight.insert(page_id, Instant::now());
            DemandFetchAction::Fetch
        }
    }
}

enum DemandFetchAction {
    /// R11 prefetch is in-flight and close to completing. Wait for it.
    WaitForPrefetch(Arc<Notify>),
    /// R11 prefetch is in-flight but far from completing. Cancel it, fetch ourselves.
    CancelPrefetchAndFetch,
    /// No prefetch in-flight. Proceed with demand fetch.
    Fetch,
}
```

### 6.2 Flow Integration

```
COMBINED R11 + R19 FLOW FOR A KERNEL LAUNCH
=============================================

Timeline:
  T-5ms: R11 predicts kernel K will launch, needing pages {P1..P100}.
         R11 starts prefetching P1..P95 (95% confidence) via ReadRequest.
         coordinator.prefetch_start(P1), ..., coordinator.prefetch_start(P95)

  T-2ms: Most prefetches complete. P1..P90 mapped as Shared.
         P91..P95 still in-flight (slower links).

  T=0:   cuLaunchKernel intercepted for kernel K.
         R19 pre-launch check:
           Required pages: {P1..P100, P101..P105} (R8 found more args than R11 predicted)

         P1..P90:  PTE shows Shared. Skip (already mapped).
         P91..P95: coordinator.is_prefetch_in_flight(P91) = true.
                   coordinator.demand_fetch_start(P91) -> WaitForPrefetch.
                   R19 waits for R11 to complete (expected <500us).
         P96..P100: Not prefetched (R11 misprediction). R19 issues demand fetch.
                    coordinator.demand_fetch_start(P96) -> Fetch.
                    Sent as ReadRequest/WriteRequest with DemandFetch priority.
         P101..P105: R11 didn't predict these at all. R19 demand fetches.

  T+200us: R11 finishes P91..P95. coordinator.prefetch_complete(P91..P95).
           R19 wakeup: cuMemMap for P91..P95.

  T+400us: R19 demand fetches for P96..P105 complete.
           All pages mapped. Launch kernel K.

  After K completes: R19 reports misses {P96..P105} to R11.
         R11 updates its model: next time kernel K is predicted, include P96..P105.

Net result:
  - R11 handled 90/105 pages (86% hit rate for this kernel).
  - R19 handled 15/105 pages (14% demand fault rate).
  - No double-fetches occurred.
  - Total pre-launch overhead: ~400 us (dominated by demand fetches).
  - Next iteration: R11 will prefetch P96..P105 too, improving to ~100% hit rate.
```

### 6.3 R11 Tentative Reads and Coherency

```
When R11 issues a low-confidence prefetch (tentative=true):
  1. coordinator.prefetch_start(page_id, ..., tentative=true)
  2. Send ReadRequest { page_id, requester, tentative: true } to home.
  3. Home processes the request BUT does NOT add requester to sharer set.
  4. Home sends ReadReply with data.
  5. R11 stores data in a local staging buffer (NOT in the page table).

If the prediction is correct (kernel actually needs the page):
  6a. R11 calls cuMemMap to install the page from staging buffer.
  7a. R11 sends a CONFIRM_TENTATIVE to home. Home adds to sharer set.

If the prediction is wrong (kernel doesn't need the page):
  6b. R11 discards the staging buffer. No coherency cleanup needed.
  7b. Home never knew about this node's copy. Clean state.

Benefit: mispredicted prefetches have zero coherency overhead.
Cost: one extra message (CONFIRM_TENTATIVE) for correct predictions.
Net: worth it when R11's confidence is < 70% for a particular page.
```

---

## 7. Cross-Topic Integration Points

### 7.1 R10 v2 (Virtual Memory Manager)

- **PTE coherency field:** R19 uses the 2-byte coherency field in R10's PTE. Encoding: 3-bit PageState (I/S/E) + 13-bit SharerBitmap. Also uses FAULT_PENDING flag.
- **PageTable trait:** R19 calls `lookup(page_id)`, `update_flags(page_id, new_state)`, and `bulk_lookup(page_ids)` for pre-launch checks.
- **MigrationEngine:** R19's page fetch operations go through R10's MigrationEngine for physical memory allocation and cuMemMap operations.

### 7.2 R11 (Speculative Prefetching)

- **PrefetchFaultCoordinator** (Section 6): shared state prevents double-fetching.
- **Miss reporting:** R19 reports demand-fetched pages to R11 so it learns from mispredictions.
- **Tentative reads:** Low-confidence prefetches use tentative coherency to avoid cleanup cost.

### 7.3 R12 (Memory Deduplication)

- **Deduped pages are permanently Shared:** DirectoryEntry.is_deduped = true means the page never transitions to Exclusive. Any write triggers R12's COW mechanism first.
- **COW fault flow:** Application writes to deduped page -> R12 intercepts -> R12 creates new non-deduped page -> R12 requests Exclusive on the NEW page from R19 -> original deduped page stays Shared.
- **Nearest-copy fetch:** For Shared deduped pages, R17 selects the nearest node that has a copy (not necessarily home). This is especially valuable for model weights shared across all nodes.

### 7.4 R15 (Fault Tolerance)

- **Recovery interaction:** If a page is being reconstructed by R15 (after a node crash), DirectoryEntry.is_recovering = true. Fault requests for recovering pages receive NACK with retry_after.
- **Directory replication:** R15 replicates directory entries for fault tolerance. Without R15, directory state is lost if a home node crashes. Interim: persist directory to local disk periodically; reconstruct from surviving nodes on failure.

### 7.5 R17 (Topology-Aware Scheduling)

- **Source selection:** When fetching a Shared page, R17's `nearest_node_with_page(page_id)` returns the closest node in the topology graph that has a copy. R19 fetches from there instead of always going to home.
- **Route selection:** R19 uses R17's `select_route(src, dst, 64KB, DemandFetch)` for every page fetch, getting the lowest-latency link.

### 7.6 R22 (Post-Copy Migration)

- **R22 uses R19's fault mechanism:** Post-copy VM migration sends the process to the destination node first, then faults in pages on demand as the process accesses them. This is exactly R19's pre-launch paging flow, applied to a migrating workload instead of a stationary one.
- **Integration:** R22 sets up the VA range at the destination, marks all pages as Invalid, and lets R19's fault handler fetch pages from the source node as the migrated process needs them.

### 7.7 R25 (Kernel Splitting)

- **Split kernel faults:** If R25 splits a kernel across multiple GPUs, each GPU's portion may access pages that are local to other GPUs. R19 handles these faults transparently: each GPU's fault handler independently resolves missing pages.
- **Coordination:** R25 provides the page access sets for each kernel split to R19, enabling accurate pre-launch mapping per GPU.

---

## 8. Open Questions Carried Forward

### Must Answer Before Implementation

1. **Does CUDA_ERROR_ILLEGAL_ADDRESS provide the faulting address?** This is the single most critical unknown. If not, crash recovery must over-fetch all possible pages (Option B in Section 3). Must test on RTX 3090 with CUDA 12.x.

2. **Can cuMemMap be called concurrently from multiple threads?** If cuMemMap is serialized internally, pre-launch mapping of N pages takes N * 15us = 1.5ms for 100 pages. If parallelizable, potentially 100us. This determines whether we need to batch cuMemMap calls.

3. **Maximum cuMemAddressReserve range?** Determines cluster memory pool size. If limited to physical VRAM (24 GB on RTX 3090), we need VA recycling. If 48-bit VA (256 TB), we have plenty.

4. **cuMemMap overhead at scale?** With 375K pages per GPU and potentially 1M+ across the cluster, verify CUDA handles this mapping count without degradation.

### Resolved from v1

5. ~~R10 ARC eviction + coherency interaction~~ -- On eviction: Shared pages silently evict (no notification needed, home learns when it sends INVALIDATE and gets no response). Exclusive pages do dirty writeback via EvictNotify.

6. ~~Consistency model~~ -- SWMR with I/S/E (Section 4). Confirmed as best fit.

7. ~~Directory architecture~~ -- Distributed home-node model. Each page's home tracks state. Confirmed.

---

## Related Documents

- [R19 preplan v1](./preplan.md) -- superseded by this document
- [research/01-gpu-page-fault-mechanisms.md](./research/01-gpu-page-fault-mechanisms.md) -- GPU fault survey
- [research/02-distributed-shared-memory.md](./research/02-distributed-shared-memory.md) -- DSM prior art
- [research/03-coherency-and-thrashing.md](./research/03-coherency-and-thrashing.md) -- coherency design
- R10 v2 -- PageTable trait, PTE coherency field, MigrationEngine, FAULT_PENDING flag
- R11 -- Prefetch hit rate >90%, tentative reads, miss reporting
- R12 -- Deduped pages permanently Shared, COW coherency transitions
- R15 -- Recovery interaction, directory replication
- R17 -- Topology-aware source selection, route selection for fetches
- R22 -- Post-copy migration uses R19's fault mechanism
- R25 -- Kernel splitting triggers faults on non-local data
