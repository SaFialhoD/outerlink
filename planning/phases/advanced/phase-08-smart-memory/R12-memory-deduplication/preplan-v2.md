# R12: Memory Deduplication -- Pre-Plan v2 (Cross-Topic Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of R12's pre-plan, incorporating exact Rust structs matching R10 v2's DedupHook trait, cross-topic integration with R14 (Compression), R15 (Fault Tolerance / Parity), R19 (Coherency), R20 (NCCL Broadcast), and R29 (RDMA Multicast). Every struct, pipeline, and interaction is specified precisely enough to code from.

---

## 1. Exact Rust Structs

### 1.1 Core Types (Matching R10 v2)

R10 v2 defines these dedup-related PTE fields and traits that R12 implements:

```rust
/// From R10 v2 PTE -- fields R12 reads and writes
/// dedup_hash: u128        (xxHash128 of page content)
/// flags: DEDUP_CANONICAL  (this page holds the canonical data)
///        DEDUP_REFERENCE   (this page is a reference to a canonical page)
///        COMPRESSED        (page content is compressed, relevant for R14)

/// R10 v2 DedupHook trait -- R12 MUST implement this exactly.
pub trait DedupHook: Send + Sync {
    /// Called when a page is loaded into any tier.
    /// Returns whether the page was deduplicated.
    fn on_page_load(&self, vpn: u64, content: &[u8]) -> DedupResult;

    /// Called when a page is evicted from the last tier it occupies.
    /// R12 must clean up dedup references.
    fn on_page_evict(&self, vpn: u64);

    /// Look up the canonical VPN for a given content hash.
    /// Returns None if no canonical page exists with this hash.
    fn lookup_canonical(&self, hash: u128) -> Option<u64>;
}

#[derive(Debug, Clone)]
pub enum DedupResult {
    /// Page is unique -- no duplicate found. Inserted as new canonical.
    Unique,

    /// Page is a duplicate. The VPN now references the canonical page.
    Deduplicated {
        canonical_vpn: u64,
        canonical_node: NodeId,
    },

    /// Dedup check skipped (disabled, quota reached, etc.)
    Skipped,
}
```

### 1.2 DedupManager

```rust
/// Central deduplication manager for this node.
/// Implements DedupHook and manages the local dedup table.
pub struct DedupManager {
    /// The dedup table: hash -> canonical page info
    dedup_table: DedupTable,

    /// Reference to R10's page table for PTE flag updates
    page_table: Arc<dyn PageTable>,

    /// Configuration
    config: DedupConfig,

    /// Statistics
    stats: DedupStats,

    /// CoW handler for write-faulted dedup pages
    cow_handler: CowHandler,

    /// Async notifier for cross-node dedup events
    /// (CoW breaks, new canonical pages, evictions)
    network_notifier: Arc<dyn DedupNetworkNotifier>,

    /// Whether dedup is currently active (can be disabled at runtime)
    enabled: AtomicBool,
}

pub struct DedupConfig {
    /// Enable/disable dedup
    enabled: bool,

    /// Maximum DDT entries (quota). 0 = unlimited.
    max_entries: usize,

    /// Minimum reference count to keep in DDT.
    /// Entries with refcount=1 for longer than prune_timeout are removed.
    prune_min_refcount: u32,

    /// How long a unique entry (refcount=1) stays before pruning
    prune_timeout_secs: u64,

    /// Whether to dedup cross-node (or intra-node only)
    cross_node_enabled: bool,

    /// Whether to perform full memcmp verification on hash match
    /// (always true in production; can disable for benchmarking)
    verify_on_match: bool,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 0,         // unlimited
            prune_min_refcount: 1,
            prune_timeout_secs: 300, // 5 minutes
            cross_node_enabled: true,
            verify_on_match: true,
        }
    }
}
```

### 1.3 DedupTable (Hash -> Canonical VPN)

```rust
/// The Dedup Descriptor Table (DDT).
/// Maps content hash (xxHash128) to canonical page information.
/// Stored in host DRAM (not VRAM).
pub struct DedupTable {
    /// Hash map: xxHash128 -> DedupEntry
    /// Using hashbrown for Robin Hood hashing (cache-friendly)
    entries: HashMap<u128, DedupEntry, BuildHasherDefault<IdentityHasher>>,

    /// Current number of entries
    len: usize,

    /// Read-write lock for concurrent access
    /// Reads (lookups) are frequent and parallel; writes (inserts/removes) are rare
    lock: RwLock<()>,
}

/// Identity hasher for u128 keys (xxHash128 output is already well-distributed).
/// Avoids double-hashing overhead.
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write_u128(&mut self, i: u128) {
        // Use lower 64 bits as hash bucket index
        self.0 = i as u64;
    }
    fn finish(&self) -> u64 { self.0 }
    // other write methods: unreachable for u128 keys
}

/// A single entry in the dedup table.
/// 40 bytes per entry.
#[repr(C)]
pub struct DedupEntry {
    /// VPN of the canonical page (the one holding the actual data)
    canonical_vpn: u64,           // 8 bytes

    /// Node that owns the canonical page
    canonical_node: NodeId,       // 4 bytes (u32)

    /// Reference count: how many VPNs point to this canonical page
    /// Atomic for lock-free increment/decrement
    refcount: AtomicU32,          // 4 bytes

    /// Tier where the canonical page currently resides
    canonical_tier: Tier,         // 1 byte

    /// Flags
    flags: DedupEntryFlags,       // 1 byte

    /// Timestamp of last refcount change (for pruning)
    last_activity_ns: u64,        // 8 bytes

    /// List of reference VPNs (for invalidation on canonical eviction)
    /// Only stored for entries with refcount > 1
    /// Stored separately in a side table to keep DedupEntry small
    references_key: u64,          // 8 bytes (index into reference side table)

    /// Padding for alignment
    _padding: [u8; 6],            // 6 bytes
    // Total: 40 bytes
}

bitflags::bitflags! {
    pub struct DedupEntryFlags: u8 {
        /// Canonical page is read-only (model weights) -- COW never triggers
        const READ_ONLY = 0x01;
        /// Canonical page is compressed (R14)
        const COMPRESSED = 0x02;
        /// Entry is pending pruning (refcount=1, timer started)
        const PRUNE_PENDING = 0x04;
        /// Cross-node dedup (canonical on different node than some references)
        const CROSS_NODE = 0x08;
    }
}

/// Side table for reference tracking.
/// Only entries with refcount > 1 have entries here.
/// Maps reference_key -> list of (vpn, node_id) referencing this canonical.
pub struct ReferenceTable {
    entries: HashMap<u64, Vec<(u64, NodeId)>>,
    next_key: AtomicU64,
}
```

### 1.4 DedupStats

```rust
/// Deduplication effectiveness metrics.
pub struct DedupStats {
    /// Total pages processed by on_page_load
    pages_checked: AtomicU64,

    /// Pages that were deduplicated (DedupResult::Deduplicated)
    pages_deduped: AtomicU64,

    /// Pages that were unique (DedupResult::Unique)
    pages_unique: AtomicU64,

    /// Pages skipped (quota, disabled, etc.)
    pages_skipped: AtomicU64,

    /// CoW breaks (deduped page was written)
    cow_breaks: AtomicU64,

    /// Bytes saved by dedup (pages_deduped * PAGE_SIZE)
    bytes_saved: AtomicU64,

    /// Current DDT size (entries)
    ddt_size: AtomicUsize,

    /// Current DDT memory usage (bytes)
    ddt_memory_bytes: AtomicUsize,

    /// Hash computation time (cumulative nanoseconds)
    hash_time_ns: AtomicU64,

    /// Memcmp verification time (cumulative nanoseconds)
    verify_time_ns: AtomicU64,

    /// Cross-node dedup count
    cross_node_dedup_count: AtomicU64,

    /// NCCL broadcast pages skipped due to dedup (R20 integration)
    nccl_pages_skipped: AtomicU64,
}

impl DedupStats {
    /// Compute dedup ratio: bytes_saved / (bytes_saved + unique_pages * PAGE_SIZE)
    pub fn dedup_ratio(&self) -> f64 {
        let saved = self.bytes_saved.load(Ordering::Relaxed) as f64;
        let unique = self.pages_unique.load(Ordering::Relaxed) as f64 * 65536.0;
        if saved + unique == 0.0 { return 0.0; }
        saved / (saved + unique)
    }

    /// DDT overhead ratio: ddt_memory / total_managed_pool
    pub fn ddt_overhead_ratio(&self, total_pool_bytes: u64) -> f64 {
        self.ddt_memory_bytes.load(Ordering::Relaxed) as f64 / total_pool_bytes as f64
    }
}
```

---

## 2. DedupHook Implementation

### 2.1 on_page_load -- Full Implementation

```rust
impl DedupHook for DedupManager {
    fn on_page_load(&self, vpn: u64, content: &[u8]) -> DedupResult {
        if !self.enabled.load(Ordering::Relaxed) {
            return DedupResult::Skipped;
        }

        // Check DDT quota
        if self.config.max_entries > 0
            && self.dedup_table.len() >= self.config.max_entries
        {
            self.stats.pages_skipped.fetch_add(1, Ordering::Relaxed);
            return DedupResult::Skipped;
        }

        // Step 1: Hash the page content
        let hash_start = Instant::now();
        let hash: u128 = xxhash_rust::xxh3::xxh3_128(content);
        let hash_elapsed = hash_start.elapsed().as_nanos() as u64;
        self.stats.hash_time_ns.fetch_add(hash_elapsed, Ordering::Relaxed);
        self.stats.pages_checked.fetch_add(1, Ordering::Relaxed);

        // Step 2: Write hash to PTE regardless of match
        self.page_table.update_flags(vpn, |pte| {
            pte.dedup_hash = hash;
        });

        // Step 3: Lookup in DDT
        let _read_lock = self.dedup_table.lock.read();
        if let Some(entry) = self.dedup_table.entries.get(&hash) {
            // Step 4: Hash match found -- verify with full memcmp
            if self.config.verify_on_match {
                let verify_start = Instant::now();
                let canonical_content = self.read_page_content(
                    entry.canonical_vpn,
                    entry.canonical_node,
                    entry.canonical_tier,
                );
                let verified = content == canonical_content.as_slice();
                let verify_elapsed = verify_start.elapsed().as_nanos() as u64;
                self.stats.verify_time_ns.fetch_add(verify_elapsed, Ordering::Relaxed);

                if !verified {
                    // Hash collision (astronomically rare) -- treat as unique
                    drop(_read_lock);
                    return self.insert_as_unique(vpn, hash);
                }
            }

            // Step 5: Confirmed duplicate -- link as reference
            entry.refcount.fetch_add(1, Ordering::Relaxed);
            entry.last_activity_ns = ptp_now();

            // Add to reference tracking
            self.dedup_table.reference_table.add(
                entry.references_key,
                vpn,
                self.local_node_id,
            );

            // Step 6: Update PTE for the new page (reference)
            self.page_table.update_flags(vpn, |pte| {
                pte.flags.insert(PageFlags::DEDUP_REFERENCE);
                pte.flags.remove(PageFlags::DEDUP_CANONICAL);
                // Point to canonical -- the actual data mapping
                // R10 handles the virtual->physical redirection
            });

            // Step 7: Update PTE for canonical (if not already marked)
            self.page_table.update_flags(entry.canonical_vpn, |pte| {
                pte.flags.insert(PageFlags::DEDUP_CANONICAL);
            });

            self.stats.pages_deduped.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes_saved.fetch_add(65536, Ordering::Relaxed);

            let canonical_node = entry.canonical_node;
            let canonical_vpn = entry.canonical_vpn;

            if canonical_node != self.local_node_id {
                self.stats.cross_node_dedup_count.fetch_add(1, Ordering::Relaxed);
            }

            DedupResult::Deduplicated {
                canonical_vpn,
                canonical_node,
            }
        } else {
            // No match -- insert as new canonical
            drop(_read_lock);
            self.insert_as_unique(vpn, hash)
        }
    }

    fn on_page_evict(&self, vpn: u64) {
        // Check if this VPN is a canonical page
        let pte = self.page_table.lookup(vpn);
        let hash = pte.dedup_hash;

        if pte.flags.contains(PageFlags::DEDUP_CANONICAL) {
            let _write_lock = self.dedup_table.lock.write();
            if let Some(entry) = self.dedup_table.entries.get_mut(&hash) {
                if entry.canonical_vpn == vpn {
                    let refcount = entry.refcount.load(Ordering::Relaxed);
                    if refcount <= 1 {
                        // Last reference -- remove from DDT entirely
                        self.dedup_table.entries.remove(&hash);
                        self.stats.ddt_size.fetch_sub(1, Ordering::Relaxed);
                    } else {
                        // Promote one reference to become the new canonical
                        if let Some((new_canonical_vpn, new_node)) =
                            self.dedup_table.reference_table.pop_first(entry.references_key)
                        {
                            entry.canonical_vpn = new_canonical_vpn;
                            entry.canonical_node = new_node;
                            entry.refcount.fetch_sub(1, Ordering::Relaxed);

                            // Update PTEs
                            self.page_table.update_flags(new_canonical_vpn, |pte| {
                                pte.flags.insert(PageFlags::DEDUP_CANONICAL);
                                pte.flags.remove(PageFlags::DEDUP_REFERENCE);
                            });
                        }
                    }
                }
            }
        } else if pte.flags.contains(PageFlags::DEDUP_REFERENCE) {
            // This VPN is a reference -- decrement refcount
            let _write_lock = self.dedup_table.lock.write();
            if let Some(entry) = self.dedup_table.entries.get_mut(&hash) {
                entry.refcount.fetch_sub(1, Ordering::Relaxed);
                self.dedup_table.reference_table.remove(
                    entry.references_key,
                    vpn,
                    self.local_node_id,
                );
                entry.last_activity_ns = ptp_now();

                // Reclaim bytes_saved
                self.stats.bytes_saved.fetch_sub(65536, Ordering::Relaxed);
            }
        }
    }

    fn lookup_canonical(&self, hash: u128) -> Option<u64> {
        let _read_lock = self.dedup_table.lock.read();
        self.dedup_table.entries.get(&hash).map(|e| e.canonical_vpn)
    }
}

impl DedupManager {
    fn insert_as_unique(&self, vpn: u64, hash: u128) -> DedupResult {
        let _write_lock = self.dedup_table.lock.write();

        let ref_key = self.dedup_table.reference_table.next_key
            .fetch_add(1, Ordering::Relaxed);

        self.dedup_table.entries.insert(hash, DedupEntry {
            canonical_vpn: vpn,
            canonical_node: self.local_node_id,
            refcount: AtomicU32::new(1),
            canonical_tier: self.page_table.lookup(vpn).tier,
            flags: DedupEntryFlags::empty(),
            last_activity_ns: ptp_now(),
            references_key: ref_key,
            _padding: [0; 6],
        });

        self.page_table.update_flags(vpn, |pte| {
            pte.flags.insert(PageFlags::DEDUP_CANONICAL);
        });

        self.stats.pages_unique.fetch_add(1, Ordering::Relaxed);
        self.stats.ddt_size.fetch_add(1, Ordering::Relaxed);
        self.stats.ddt_memory_bytes.fetch_add(40, Ordering::Relaxed);

        DedupResult::Unique
    }
}
```

---

## 3. Inline Dedup Pipeline

### 3.1 Step-by-Step Flow

```
CUDA application calls cuMemcpyHtoD(dst_device_ptr, src_host_ptr, size):

  OuterLink interception layer intercepts the call:

  1. DETERMINE AFFECTED PAGES
     vpn_start = dst_device_ptr / PAGE_SIZE
     vpn_end   = (dst_device_ptr + size - 1) / PAGE_SIZE
     num_pages = vpn_end - vpn_start + 1

  2. FOR EACH PAGE IN [vpn_start..=vpn_end]:
     page_offset = (vpn - vpn_start) * PAGE_SIZE
     content = &src_host_ptr[page_offset..page_offset + PAGE_SIZE]

     result = dedup_manager.on_page_load(vpn, content)

     MATCH result:
       Unique:
         // Normal path: R10 allocates page, performs actual memcpy
         r10.migrate_page(vpn, content, Tier::LocalVram)

       Deduplicated { canonical_vpn, canonical_node }:
         IF canonical_node == local_node:
           // Intra-node dedup: map VPN to canonical's physical memory
           // Using CUDA VMM API: cuMemMap(vpn_addr, PAGE_SIZE, canonical_phys)
           // Set access to READ_ONLY via cuMemSetAccess
           r10.link_page_to_canonical(vpn, canonical_vpn)
         ELSE:
           // Cross-node dedup: VPN becomes a remote reference
           // R10 records that this VPN's data lives on canonical_node
           // Any read triggers R10 remote fetch (transparent)
           r10.link_page_to_remote_canonical(vpn, canonical_vpn, canonical_node)

       Skipped:
         // Normal path, no dedup
         r10.migrate_page(vpn, content, Tier::LocalVram)

  3. RETURN to CUDA application (cuMemcpyHtoD completes)

Total overhead per page:
  - Hash:   ~2 us (xxHash128 at 31 GB/s for 64KB)
  - Lookup: ~50 ns (hashbrown HashMap)
  - Verify: ~1 us (memcmp 64KB, only on match)
  - PTE update: ~100 ns
  Total: ~2.2 us (no match) or ~3.2 us (match + verify)

For a 14 GB model (7B params, FP16):
  - 14 GB / 64 KB = 218,750 pages
  - Hash all pages: ~480 ms at 31 GB/s
  - This is < 5% of typical model load time over 100Gbps (~11.2 seconds)
```

### 3.2 Model Weight Loading Optimization

For model loading (the primary dedup target), the flow is batched:

```
PyTorch calls model.to('cuda'):
  -> triggers cuMemcpyHtoD for each parameter tensor
  -> OuterLink batches these into bulk dedup checks:

  1. Collect all pages from the cuMemcpyHtoD call
  2. Batch hash: compute xxHash128 for all pages (can parallelize across CPU cores)
  3. Batch lookup: check all hashes against DDT
  4. Batch link: for all matches, link as references in one batch PTE update
  5. Only transfer unique pages over the wire

  For 4 GPUs loading identical weights (14 GB each):
    GPU 0: loads all 218,750 pages (creates canonical entries)
    GPU 1: hashes all pages, 218,750 matches -> 0 pages transferred, 14 GB saved
    GPU 2: same -> 14 GB saved
    GPU 3: same -> 14 GB saved
    Total VRAM used: 14 GB (instead of 56 GB)
    Savings: 75% = (N-1)/N where N=4
```

---

## 4. CoW Trigger Flow

### 4.1 Write Detection at Interception Layer

```
CUDA application calls cuMemcpy*toD(dst, ...) or cuLaunchKernel(f, ..., params):

  OuterLink interception layer:

  1. IDENTIFY TARGET PAGES
     For cuMemcpyHtoD/DtoD: target pages = [dst..dst+size] / PAGE_SIZE
     For cuLaunchKernel: target pages = kernel output pointer VPNs
       (from R8 kernel param introspection, or conservatively: all arg pointers)

  2. FOR EACH TARGET PAGE vpn:
     pte = page_table.lookup(vpn)

     IF pte.flags.contains(DEDUP_REFERENCE):
       // This page is a reference to a canonical copy.
       // Writing to it would corrupt the shared data.
       // Trigger proactive CoW BEFORE the CUDA call proceeds.

       cow_handler.break_dedup(vpn, pte)
       // After this, vpn points to a private writable copy

     ELSE IF pte.flags.contains(DEDUP_CANONICAL) AND refcount > 1:
       // This page IS the canonical copy, but others reference it.
       // Must copy-on-write to preserve the canonical for other references.

       cow_handler.break_canonical(vpn, pte)
       // After this, vpn points to a private writable copy
       // A different VPN becomes the new canonical

  3. PROCEED with original CUDA call
     (now writing to private, writable pages)
```

### 4.2 CowHandler Implementation

```rust
pub struct CowHandler {
    page_table: Arc<dyn PageTable>,
    dedup_table: Arc<DedupTable>,
    network_notifier: Arc<dyn DedupNetworkNotifier>,
    stats: Arc<DedupStats>,
}

impl CowHandler {
    /// Break dedup for a REFERENCE page: copy canonical data to a new private page.
    pub fn break_dedup(&self, vpn: u64, pte: &Pte) -> Result<(), CowError> {
        let hash = pte.dedup_hash;

        // Step 1: Allocate a new private page in the same tier
        let new_phys = r10_allocate_page(pte.tier)?;

        // Step 2: Copy canonical content to the new page
        let canonical_vpn = self.dedup_table.lookup_canonical(hash)
            .ok_or(CowError::CanonicalNotFound)?;
        r10_copy_page(canonical_vpn, new_phys)?;

        // Step 3: Update PTE: vpn now points to new_phys, writable
        self.page_table.update_flags(vpn, |pte| {
            pte.flags.remove(PageFlags::DEDUP_REFERENCE);
            // dedup_hash remains (for potential future re-dedup after write)
        });

        // Step 4: Decrement refcount in DDT
        {
            let _lock = self.dedup_table.lock.write();
            if let Some(entry) = self.dedup_table.entries.get_mut(&hash) {
                let old_refcount = entry.refcount.fetch_sub(1, Ordering::Relaxed);
                self.dedup_table.reference_table.remove(
                    entry.references_key, vpn, local_node_id(),
                );

                // If refcount dropped to 1, canonical is sole owner
                // Start prune timer (it's now unique)
                if old_refcount == 2 {
                    entry.flags.insert(DedupEntryFlags::PRUNE_PENDING);
                    entry.last_activity_ns = ptp_now();
                }
            }
        }

        // Step 5: Async notify network (fire-and-forget)
        self.network_notifier.send(DedupNetworkMessage::CowBreak {
            vpn,
            hash,
            node: local_node_id(),
        });

        self.stats.cow_breaks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Break dedup for a CANONICAL page: transfer canonical status to another
    /// reference, then make this page private.
    pub fn break_canonical(&self, vpn: u64, pte: &Pte) -> Result<(), CowError> {
        let hash = pte.dedup_hash;

        let _lock = self.dedup_table.lock.write();
        if let Some(entry) = self.dedup_table.entries.get_mut(&hash) {
            // Promote another reference to canonical
            if let Some((new_canonical_vpn, new_node)) =
                self.dedup_table.reference_table.pop_first(entry.references_key)
            {
                // Copy data to the new canonical before we modify the current one
                r10_copy_page(vpn, new_canonical_vpn)?;

                entry.canonical_vpn = new_canonical_vpn;
                entry.canonical_node = new_node;
                entry.refcount.fetch_sub(1, Ordering::Relaxed);

                // Update PTEs
                self.page_table.update_flags(new_canonical_vpn, |pte| {
                    pte.flags.insert(PageFlags::DEDUP_CANONICAL);
                    pte.flags.remove(PageFlags::DEDUP_REFERENCE);
                });
                self.page_table.update_flags(vpn, |pte| {
                    pte.flags.remove(PageFlags::DEDUP_CANONICAL);
                });

                // Notify new canonical's node
                self.network_notifier.send(DedupNetworkMessage::CanonicalTransfer {
                    hash,
                    new_canonical_vpn,
                    new_node,
                });
            } else {
                // No other references -- just remove from DDT
                self.dedup_table.entries.remove(&hash);
                self.page_table.update_flags(vpn, |pte| {
                    pte.flags.remove(PageFlags::DEDUP_CANONICAL);
                });
            }
        }

        self.stats.cow_breaks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
```

### 4.3 R19 Coherency Integration

R19 establishes that deduped pages are permanently in Shared state under the I/S/E (Invalid/Shared/Exclusive) coherency protocol:

```
Deduped page coherency lifecycle:

  1. Page loaded, dedup detected -> state = Shared (S)
     - All references read the same canonical data
     - No coherency traffic needed for reads

  2. Write to deduped page -> CoW trigger:
     - CoW creates private copy -> private copy state = Exclusive (E)
     - Original dedup reference removed
     - Canonical remains Shared for remaining references

  3. Canonical page evicted -> new canonical promoted
     - State remains Shared (S) for all references
     - Only the canonical location changes

  KEY RULE: Deduped pages are NEVER write-faulted in R19's sense.
  The CoW in R12 happens BEFORE the write reaches the coherency protocol.
  By the time R19 sees the write, it's already on a private page in Exclusive state.
```

---

## 5. NCCL Integration (R20 Broadcast)

### 5.1 The Opportunity

NCCL broadcast (allreduce, broadcast) sends identical data to all GPUs. If the receiving nodes already have the data (because dedup detected it), the broadcast can skip those pages.

### 5.2 Dedup-Aware Broadcast Protocol

```
R20 NCCL broadcast of tensor T across N nodes:

  SENDER (rank 0):
    1. Hash all pages of tensor T (may already be in DDT from initial load)
    2. Send BROADCAST_MANIFEST message to all receivers:
       { tensor_id, page_hashes: Vec<u128>, page_vpns: Vec<u64> }

  RECEIVER (rank 1..N-1):
    For each (hash, vpn) in manifest:
      1. Check local DDT: dedup_manager.lookup_canonical(hash)
      2. IF Some(local_canonical_vpn):
         // We already have this exact page content locally!
         // Link the broadcast destination VPN to our local canonical
         r10.link_page_to_canonical(dest_vpn, local_canonical_vpn)
         REPLY to sender: SKIP(vpn)
      3. ELSE:
         // We don't have this content -- need the actual transfer
         REPLY to sender: NEED(vpn)

  SENDER:
    Only transfer pages where ALL receivers replied NEED.
    For pages where ANY receiver has it: send only to receivers that NEED it.

  RESULT:
    For LLM weight broadcast across N identical nodes:
      - Node 0 sends manifest (tiny: N_pages * 16 bytes for hashes)
      - Nodes 1..N-1 all reply SKIP for every page
      - Zero data transferred!
      - Time: 1 RTT for manifest + replies (~10 us RDMA) instead of
        14 GB broadcast (~1.27 seconds at 100Gbps)
```

### 5.3 Dedup Table Query Interface for R20

```rust
/// Interface exposed by DedupManager for R20 to query.
impl DedupManager {
    /// Given a list of content hashes, return which ones are
    /// already available locally as canonical or reference pages.
    /// Used by R20 NCCL broadcast to skip redundant transfers.
    pub fn batch_check_local(&self, hashes: &[u128]) -> Vec<DedupCheckResult> {
        let _read_lock = self.dedup_table.lock.read();
        hashes.iter().map(|hash| {
            if let Some(entry) = self.dedup_table.entries.get(hash) {
                if entry.canonical_node == self.local_node_id {
                    DedupCheckResult::LocalCanonical { vpn: entry.canonical_vpn }
                } else {
                    // We have a reference, but canonical is remote
                    // Still counts as "have it" since our R10 can serve it
                    DedupCheckResult::LocalReference { canonical_vpn: entry.canonical_vpn }
                }
            } else {
                DedupCheckResult::NotFound
            }
        }).collect()
    }
}

pub enum DedupCheckResult {
    /// Page content exists locally as canonical
    LocalCanonical { vpn: u64 },
    /// Page content exists locally as reference (data available via R10)
    LocalReference { canonical_vpn: u64 },
    /// Page content not found locally -- need transfer
    NotFound,
}
```

### 5.4 R29 RDMA Multicast Integration

R29 provides RDMA multicast for initial weight distribution. Combined with dedup:

```
First-time cluster model loading:
  1. Master node loads model weights from disk
  2. Master hashes all pages, inserts into DDT as canonical
  3. R29 multicasts all pages to all nodes simultaneously
  4. Each receiving node:
     - Receives pages via RDMA multicast
     - Inserts each page into local DDT via on_page_load
     - First receiver after master: all pages match master's hashes -> dedup as references
     - BUT: since multicast delivers to all simultaneously, each node gets its OWN copy
     - Dedup applies to SUBSEQUENT loads (e.g., loading same model on a second GPU)

Subsequent model loading (same model):
  1. Node already has weights in DDT from first load
  2. New GPU on same node: on_page_load finds all hashes -> 100% dedup
  3. New GPU on different node: cross-node lookup finds hashes -> link as remote references
```

---

## 6. Parity Interaction (R15 Fault Tolerance)

### 6.1 The Problem

R15 uses erasure coding (parity) to protect against node failures. For deduped pages:
- **Canonical pages** hold the actual data -- they MUST be protected by parity
- **Reference pages** hold only a pointer to the canonical -- they need NO parity

### 6.2 Parity Rules for Dedup

```
R15 parity computation:

  FOR EACH page P:
    IF P.flags.contains(DEDUP_CANONICAL):
      // This page holds actual data that cannot be reconstructed
      // from any other source. It MUST be included in parity.
      include_in_parity_group(P)

    ELSE IF P.flags.contains(DEDUP_REFERENCE):
      // This page is just a pointer to a canonical page.
      // If this node fails, the reference is lost, but the canonical
      // still exists (on another node or the same node).
      // No parity needed for reference pages.
      skip_parity(P)

    ELSE:
      // Normal page -- standard parity rules apply
      include_in_parity_group(P)
```

### 6.3 R15 Query Interface

```rust
/// Interface exposed by DedupManager for R15 parity decisions.
impl DedupManager {
    /// Returns true if this VPN is a dedup reference (not canonical).
    /// R15 uses this to skip parity for reference pages.
    pub fn is_reference(&self, vpn: u64) -> bool {
        let pte = self.page_table.lookup(vpn);
        pte.flags.contains(PageFlags::DEDUP_REFERENCE)
    }

    /// Returns the canonical VPN and node for a given reference.
    /// R15 uses this during failure recovery: if a reference page is lost,
    /// it can be reconstructed by re-linking to the canonical.
    pub fn get_canonical_for_reference(&self, vpn: u64) -> Option<(u64, NodeId)> {
        let pte = self.page_table.lookup(vpn);
        if !pte.flags.contains(PageFlags::DEDUP_REFERENCE) {
            return None;
        }
        let hash = pte.dedup_hash;
        let _lock = self.dedup_table.lock.read();
        self.dedup_table.entries.get(&hash)
            .map(|e| (e.canonical_vpn, e.canonical_node))
    }

    /// Called by R15 when a node fails. For all dedup entries where the
    /// failed node was the canonical, promote a surviving reference.
    pub fn handle_node_failure(&self, failed_node: NodeId) {
        let _lock = self.dedup_table.lock.write();
        let affected: Vec<u128> = self.dedup_table.entries.iter()
            .filter(|(_, e)| e.canonical_node == failed_node)
            .map(|(hash, _)| *hash)
            .collect();

        for hash in affected {
            if let Some(entry) = self.dedup_table.entries.get_mut(&hash) {
                // Try to promote a surviving reference
                let survivors: Vec<_> = self.dedup_table.reference_table
                    .get_entries(entry.references_key)
                    .into_iter()
                    .filter(|(_, node)| *node != failed_node)
                    .collect();

                if let Some((new_vpn, new_node)) = survivors.first() {
                    entry.canonical_vpn = *new_vpn;
                    entry.canonical_node = *new_node;
                    entry.refcount.fetch_sub(1, Ordering::Relaxed);
                    // The new canonical must reconstruct data from R15 parity
                    // if the failed node's copy was the only one with actual data
                } else {
                    // All copies were on the failed node -- data is lost
                    // R15 parity reconstruction is the only recovery path
                    self.dedup_table.entries.remove(&hash);
                }
            }
        }
    }
}
```

### 6.4 Compression Interaction (R14)

Canonical pages are high-value compression targets (one-time cost, many beneficiaries):

```
Canonical page compression strategy:

  1. When a page becomes canonical (on_page_load returns Unique):
     - If page is in a compressible tier (DRAM, NVMe):
       Compress with LZ4 and set COMPRESSED flag
     - If page is in VRAM: leave uncompressed (GPU can't read compressed data)

  2. When a reference page is prefetched (R11) from a compressed canonical:
     - R11 requests compressed data from source
     - Decompress on arrival at destination
     - Store uncompressed in VRAM
     - This is the flow defined in R11 preplan-v2 Section 5.4

  3. Canonical pages that are dedup targets SHOULD be compressed more
     aggressively (higher compression level) because:
     - The compression cost is paid once
     - Every reference benefits from reduced storage/transfer
     - Model weights (primary dedup target) compress well (~0.6-0.8x with LZ4)

  R14 integration:
    DedupEntryFlags::COMPRESSED tracks whether the canonical copy is compressed.
    When serving a reference page, R14 decompresses transparently.
```

---

## 7. Open Questions (Updated from v1)

### Resolved by Cross-Topic Findings

1. **DDT architecture (centralized vs distributed)?** -- RESOLVED. Centralized DDT on coordinator matches R10 v2's coordinator model. Reference table stored alongside DDT. Network cost of lookup (~5us RDMA) is acceptable at load time.

2. **Dedup at load time vs continuous scanning?** -- RESOLVED. Inline dedup at load time (on_page_load hook). No continuous scanning. R14 handles post-load compression of canonical pages.

3. **How does dedup interact with tier migration (R10)?** -- RESOLVED. on_page_evict handles canonical eviction by promoting a reference. Reference pages can be evicted freely (just decrement refcount).

4. **Interaction with parity (R15)?** -- RESOLVED. Section 6 defines exact rules: canonical pages get parity, reference pages skip parity.

5. **NCCL broadcast optimization (R20)?** -- RESOLVED. Section 5 defines dedup-aware broadcast protocol with manifest exchange.

6. **Coherency model (R19)?** -- RESOLVED. Deduped pages are permanently Shared in I/S/E. CoW happens before coherency protocol sees the write.

### Still Open

7. **Tensor-level hints from application?** Worth the API surface? Could skip hashing entirely for known read-only weight tensors. Deferred to optimization phase.

8. **Dedup across different data types (FP16 vs FP32)?** Out of scope for v1. Byte-level dedup only. Same logical weight in different formats will not dedup.

9. **DDT persistence across restarts?** ZFS does this. For OuterLink: model weights are reloaded anyway on restart, so DDT is rebuilt from on_page_load calls. Persistence adds complexity for marginal benefit. Decision: NO persistence in v1.

10. **Atomic refcount operations under high contention?** AtomicU32 with Relaxed ordering is sufficient. Refcount operations are rare (only on load/evict, not on every access). CAS contention is effectively zero.

---

## 8. Success Criteria (Updated)

### Memory Savings

| Metric | Target | How Measured |
|---|---|---|
| Dedup ratio (N identical models) | >= (N-1)/N | bytes_saved / (bytes_saved + unique_bytes) |
| Dedup ratio (mixed workloads) | >= 30% | Same formula across diverse serving |
| DDT memory overhead | < 0.1% of pool | ddt_memory_bytes / total_pool_bytes |

### Performance

| Metric | Target | How Measured |
|---|---|---|
| Hash overhead at load time | < 5% of model load time | Wall clock: load with dedup / load without |
| on_page_load latency (no match) | < 3 us | Microbenchmark |
| on_page_load latency (match + verify) | < 5 us | Microbenchmark |
| CoW latency (local) | < 15 us | Microbenchmark: detect + alloc + copy |
| CoW latency (cross-node RDMA) | < 25 us | Microbenchmark: detect + fetch + copy |
| Inference throughput impact | < 1% regression | tokens/sec with vs without dedup |
| NCCL broadcast skip rate (identical models) | 100% pages skipped | nccl_pages_skipped / total_broadcast_pages |

### Correctness

| Metric | Target | How Measured |
|---|---|---|
| Data corruption from hash collision | 0 incidents | Stress test with memcmp verification |
| CoW correctness (concurrent writes) | 0 data races | Multi-threaded stress test |
| Reference count accuracy | Exact (no leaks) | Long-running alloc/free test |
| Node failure recovery | All canonicals recovered | Kill nodes during dedup, verify data integrity |

---

## 9. Testing Strategy Additions (v2)

### Cross-Topic Integration Tests

| Test | What It Validates |
|---|---|
| Dedup + R14 compression | Canonical pages compress, references decompress on access |
| Dedup + R15 parity | Reference pages excluded from parity; canonical failure triggers promotion |
| Dedup + R19 coherency | Deduped pages stay Shared; CoW produces Exclusive private copy |
| Dedup + R20 NCCL broadcast | Manifest exchange skips already-deduped pages |
| Dedup + R29 RDMA multicast | Initial weight multicast seeds DDT; subsequent loads dedup 100% |
| Dedup + R10 eviction | Canonical eviction promotes reference; refcount stays accurate |
| Dedup + R11 prefetch | Deduped page prefetched once for canonical; references served locally |

---

## Related Documents

- [R12 preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-existing-dedup-systems.md](./research/01-existing-dedup-systems.md) -- KSM, ZFS, VMware TPS survey
- [research/02-hashing-and-detection.md](./research/02-hashing-and-detection.md) -- xxHash128 selection, GPU hashing
- [research/03-copy-on-write-network.md](./research/03-copy-on-write-network.md) -- Network CoW, CUDA memory protection
- R10 v2 preplan -- DedupHook trait, PageTable trait, PTE flags
- R14 Transport Compression -- canonical page compression strategy
- R15 Fault Tolerance -- parity rules for canonical vs reference pages
- R19 Network Page Faults -- I/S/E coherency for deduped pages
- R20 NCCL Integration -- dedup-aware broadcast protocol
- R29 RDMA Multicast -- initial weight distribution with dedup seeding
