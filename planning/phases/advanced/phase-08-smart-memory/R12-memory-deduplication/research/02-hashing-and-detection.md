# R12 Research: Hashing and Duplicate Detection

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Evaluate hash algorithms, GPU-accelerated hashing, content sampling strategies, and hash table designs for efficiently detecting duplicate 64KB memory pages across OuterLink's GPU cluster.

---

## 1. Hash Algorithm Comparison

### Throughput Benchmarks (CPU, Single-Threaded)

| Algorithm | Category | Throughput (Optimized) | Output Size | Notes |
|-----------|----------|----------------------|-------------|-------|
| CRC32C | Checksum | ~32 GB/s (HW SSE4.2) | 32-bit | Intel/AMD hardware instruction |
| xxHash3 | Non-crypto | ~31 GB/s (AVX2) | 64/128-bit | Fastest general-purpose hash |
| xxHash128 | Non-crypto | ~31 GB/s (AVX2) | 128-bit | Same speed as xxHash3, wider output |
| BLAKE3 | Crypto | ~8.4 GB/s (AVX2, 1T) | 256-bit | Parallelizable across cores/SIMD |
| SHA-256 | Crypto | ~0.5-1 GB/s | 256-bit | Standard but slow |

**Without SIMD (pure C, AMD Ryzen 5950X):**

| Algorithm | Throughput |
|-----------|-----------|
| xxHash3 | 7.03 GB/s |
| CRC32C | 7.25 GB/s |
| BLAKE3 | 280 MB/s |

### Algorithm Assessment for OuterLink

**CRC32C:**
- Pros: Hardware-accelerated on all modern x86, 32 GB/s
- Cons: Only 32-bit output. At 1M pages, birthday collision probability is ~0.012% (unacceptable for dedup). Would require full byte comparison on every match
- Verdict: **Useful only as a fast pre-filter, not as primary hash**

**xxHash3/xxHash128:**
- Pros: ~31 GB/s with AVX2, 128-bit output available, battle-tested, BSD license, Rust crate available (`xxhash-rust`)
- Cons: Not cryptographic (irrelevant for our use case -- we're not defending against adversarial inputs)
- At 128-bit: collision probability for 1M pages = ~1.47 x 10^-27 (effectively zero)
- Verdict: **Plan A. Best speed-to-collision-resistance ratio for our use case**

**BLAKE3:**
- Pros: Cryptographic strength, tree-structured parallelism, 256-bit output, ~8.4 GB/s single-threaded
- Cons: 3-4x slower than xxHash3 on CPU. GPU implementations exist but are not faster than CPU for typical page sizes
- Verdict: **Plan B. Use if cryptographic guarantees are needed (multi-tenant future)**

**SHA-256:**
- Pros: Universal standard, hardware acceleration on some CPUs (SHA-NI)
- Cons: 5-10x slower than xxHash3 even with hardware support
- Verdict: **Not recommended. No advantage over BLAKE3**

### Recommendation

**Primary hash: xxHash128** (128-bit, ~31 GB/s on CPU)
- 128-bit output provides collision probability of ~10^-27 at 1M pages
- Even at 10M pages (640GB pool), collision probability is ~10^-25
- Still verify with full memcmp on match (defense in depth, standard practice)

---

## 2. GPU-Accelerated Hashing

### Why Hash on GPU?

For VRAM-resident pages, reading 64KB to host RAM just to hash it wastes PCIe bandwidth. If we can hash on the GPU, the data never leaves the device.

### Catalyst Approach (VEE 2017)

Catalyst demonstrated GPU-accelerated dedup hashing in virtualization:
- Phase 1: Transfer host RAM pages to GPU, hash in parallel with CUDA kernels
- Phase 2: Return hashes to host for DDT lookup
- Result: Higher dedup ratio in less time than KSM

For OuterLink, the approach is inverted: pages are ALREADY on the GPU. We hash them in place.

### CUDA Hashing Performance

**SHA-256 on GPU:**
- ETH Zurich research achieved ~98.4% of peak performance on Tesla V100
- Main loop: 1385 arithmetic instructions, only 22 non-arithmetic (highly efficient)
- On V100: ~200 MH/s for SHA-256 (varies with message size)
- For 64KB pages: this translates to roughly 12-13 GB/s on V100

**BLAKE3 on GPU:**
- SYCL/CUDA implementations exist (itzmeanjan/blake3, Blaze-3/BLAKE3-gpu)
- Tree-structured parallelism maps well to GPU architecture
- Best performance with large inputs (>= 1MB) due to PCIe transfer amortization
- For VRAM-resident data, PCIe overhead is eliminated
- Estimated: 5-15 GB/s depending on GPU and input size

**xxHash on GPU:**
- Less published GPU research (xxHash is already so fast on CPU it rarely needs GPU)
- xxHash3 kernel would be simple: ~100 lines of CUDA
- Expected throughput: 20-50 GB/s on RTX 3090 (memory bandwidth limited at 936 GB/s, compute is trivial)
- The bottleneck is global memory read bandwidth, not ALU

### Recommended GPU Hashing Strategy

For a 64KB page resident in VRAM:

1. **Launch a CUDA kernel** that reads the page and computes xxHash128
2. **One warp (32 threads)** per page, each thread processes 2KB
3. **Warp-level reduction** combines partial hashes
4. **Store hash** in a device-side hash buffer
5. **Async copy** hash results (16 bytes per page) back to host for DDT lookup

**Estimated throughput on RTX 3090:**
- Memory bandwidth: 936 GB/s
- Hashing 1M pages (64GB): ~68ms (memory-bound)
- Hash result transfer: 1M x 16 bytes = 16MB over PCIe 4.0 (< 1ms)

This is effectively free compared to the time spent loading the model weights over the network.

---

## 3. Content Sampling (Fast Pre-Filter)

### Concept

Instead of hashing the full 64KB page, hash a small sample to quickly eliminate non-duplicates. Only pages that pass the pre-filter get a full hash.

### imohash Approach

The `imohash` library samples three 16KB chunks from the beginning, middle, and end of a file:
- Total data read: 48KB out of potentially GB-sized files
- Combined with file size for fast rejection
- Full hash only computed for files where the sample hash AND size match

### Adapted for 64KB Pages

For a 64KB page, sampling is less beneficial:
- Page size is already small (64KB)
- Reading 4KB vs 64KB on modern hardware is negligible (same cache line fill on CPU, same memory transaction on GPU)
- The overhead of two hash passes (sample then full) may exceed a single full-page hash

### When Sampling Makes Sense

| Scenario | Sample Worth It? | Reasoning |
|----------|-----------------|-----------|
| 64KB pages, GPU hashing | No | GPU hashes full 64KB in ~2us, sampling adds complexity |
| 64KB pages, CPU hashing | No | xxHash128 hashes 64KB in ~2us on modern CPU |
| 2MB huge pages, CPU | Maybe | 2MB takes ~60us, sampling 4KB takes ~0.1us |
| Cross-network verification | Yes | Sending 4KB sample over 100Gbps takes ~0.3us vs 5us for 64KB |

### Recommendation

**Skip content sampling for local hashing.** xxHash128 at 31 GB/s processes a 64KB page in ~2 microseconds. The complexity of sampling is not justified.

**Use sampling for cross-network pre-verification:** Before requesting a full 64KB page transfer to verify a hash match, send a 4KB sample hash as a cheap pre-check. This avoids transferring 64KB over the network for false positives (which are already rare at 128-bit but adds defense in depth).

---

## 4. False Positive Rates and Verification

### Birthday Paradox Collision Probabilities

| Hash Size | 1M Pages | 10M Pages | 100M Pages | 50% Collision Threshold |
|-----------|----------|-----------|------------|------------------------|
| 32-bit (CRC32) | 12% | ~100% | ~100% | ~77,000 pages |
| 64-bit | 2.7 x 10^-8 | 2.7 x 10^-6 | 2.7 x 10^-4 | ~5 billion pages |
| 128-bit | 1.5 x 10^-27 | 1.5 x 10^-25 | 1.5 x 10^-23 | ~1.8 x 10^19 pages |
| 256-bit | ~0 | ~0 | ~0 | ~3.4 x 10^38 pages |

### Verification Strategy

Even with 128-bit hashes and effectively zero collision probability, all production dedup systems perform full byte comparison on hash match. This is standard practice because:

1. **Hash algorithms can have implementation bugs**
2. **Bit flips in RAM** (cosmic rays, ECC failures) can corrupt hashes
3. **The cost is negligible:** Comparing 64KB (memcmp) takes ~1us on modern hardware. This only happens on matches, which are the cases we WANT to be right about

### Recommended Verification Pipeline

```
Page written/loaded
    |
    v
Compute xxHash128 (64KB -> 16 bytes)     [~2us on CPU, ~2us on GPU]
    |
    v
Lookup in hash table                       [O(1) hash map, ~50ns]
    |
    +--> No match: insert hash, done       [common case for unique pages]
    |
    +--> Match found:
            |
            v
         Full 64KB memcmp                  [~1us, only on matches]
            |
            +--> Different: hash collision, insert as unique  [~never]
            |
            +--> Identical: deduplicate!                      [success]
```

**Total overhead per page (no match):** ~2us hash + ~50ns lookup = ~2.05us
**Total overhead per page (match):** ~2us hash + ~50ns lookup + ~1us memcmp = ~3.05us

---

## 5. Incremental Hashing (Dirty Page Tracking)

### Problem

If a page is modified, its hash is stale. We need to re-hash only dirty pages, not the entire pool.

### Dirty Page Detection Methods

**Method 1: R10 Page Table Dirty Bit**
- R10's PTE has an access tracking field
- Add a "dirty" flag to the PTE (1 bit)
- On write, mark page dirty
- Dedup daemon re-hashes only dirty pages
- Cost: 1 bit per PTE, already in the 64-byte PTE budget

**Method 2: CUDA Memory Access Tracking**
- For VRAM pages, CUDA UVM can track writes via page faults
- Mark deduped VRAM pages as read-only in GPU page table
- Write triggers fault, handler marks page dirty and makes private COW copy
- This is exactly how we want COW to work anyway

**Method 3: Write Barrier in Interception Layer**
- OuterLink intercepts all CUDA memory operations (cuMemcpy*, cuLaunchKernel)
- On cuMemcpyHtoD or kernel launch with output pointers, mark target pages dirty
- No need for hardware dirty bits; software tracking at the API level

### Recommendation

**Use Method 3 (write barrier in interception layer) as primary.**
OuterLink already intercepts all CUDA calls. We know exactly when data is written to a page because we control cuMemcpyHtoD, cuMemcpyDtoD, and cuLaunchKernel. This is more precise than page-fault-based tracking and has zero runtime overhead for reads (which dominate LLM inference).

**Use Method 2 (CUDA page protection) as fallback** for cases where we can't statically determine which kernel output pointers will be written.

---

## 6. Hash Table Design at Scale

### Memory Cost

For the dedup hash table (DDT), each entry needs:

| Field | Size | Purpose |
|-------|------|---------|
| xxHash128 | 16 bytes | Page content hash |
| Page ID (owner) | 8 bytes | Which page holds the canonical copy |
| Reference count | 4 bytes | How many pages share this content |
| Tier location | 1 byte | Which tier the canonical copy resides in |
| Flags | 1 byte | Read-only, COW-pending, etc. |
| Padding | 2 bytes | Alignment |
| **Total** | **32 bytes** | Per unique content hash |

### Scale Projections

| Pool Size | Pages (64KB) | DDT Size (32B/entry) | DDT % of Pool |
|-----------|-------------|---------------------|--------------|
| 24 GB (1 GPU) | 375,000 | 12 MB | 0.05% |
| 96 GB (4 GPUs) | 1,500,000 | 48 MB | 0.05% |
| 384 GB (16 GPUs) | 6,000,000 | 192 MB | 0.05% |
| 1 TB | 16,777,216 | 512 MB | 0.05% |
| 16 TB (full cluster) | 268,435,456 | 8 GB | 0.05% |

The DDT consistently uses ~0.05% of the managed pool. Even at 16TB scale, 8GB of host DRAM for the DDT is reasonable.

### Hash Table Implementation

**Recommended: Robin Hood hashing (open addressing)**
- Cache-friendly (linear probing with bounded displacement)
- Low overhead per entry (no pointers for chaining)
- Rust crate: `hashbrown` (used by std::collections::HashMap)
- Load factor 0.8 is safe, giving ~20% overhead on top of the raw entry size

**Alternative: Cuckoo hashing**
- O(1) worst-case lookup (important for inline dedup on the write path)
- Higher memory overhead (~50% empty slots)
- Better for read-heavy workloads (model weight dedup is read-heavy)

### DDT Placement

The DDT should reside in **host DRAM on the coordinator node** (or replicated across nodes for fault tolerance). Reasons:
1. VRAM is precious -- don't waste it on metadata
2. DDT lookups happen on the write/load path, which is host-mediated anyway
3. Host DRAM is cheaper and larger than VRAM
4. Network round-trip to coordinator for DDT lookup: ~5us over RDMA (acceptable for load-time dedup)

---

## 7. Dedup Granularity Analysis

### Page-Level (64KB) -- Recommended

| Aspect | Assessment |
|--------|-----------|
| Alignment with R10 | Perfect -- same page size as memory tiering |
| Hash overhead | 0.025% (16-byte hash per 64KB page) |
| False sharing | Low -- 64KB is large enough that unrelated data rarely shares a page |
| Dedup ratio for model weights | Near-perfect -- model layers are aligned to much larger boundaries |
| Implementation complexity | Low -- one hash per page, one DDT entry per unique content |

### Chunk-Level (Variable, 4KB-256KB)

| Aspect | Assessment |
|--------|-----------|
| Alignment with R10 | Poor -- chunks don't align with page boundaries |
| Hash overhead | Higher (more chunks per page) |
| Dedup ratio | Potentially better for partially-similar pages |
| Implementation complexity | High -- content-defined chunking needed, chunk metadata overhead |

### Tensor-Level (Variable, KB to GB)

| Aspect | Assessment |
|--------|-----------|
| Alignment with R10 | Poor -- tensors span multiple pages |
| Hash overhead | Lower (fewer, larger units) |
| Dedup ratio | Potentially perfect for model weights (entire layers match) |
| Implementation complexity | High -- requires understanding of model structure, not general-purpose |

### Verdict

**Page-level (64KB) dedup is the right choice.** It aligns with R10's page table, has minimal overhead, and works generically without understanding application data structures. For model weights, which are loaded as contiguous buffers, page-aligned dedup will naturally detect that every 64KB chunk of the weight tensor is identical across GPUs.

Tensor-level awareness could be added as an optimization hint (e.g., "these N pages are a single tensor, dedup them as a group") but is not required for correctness.

---

## Open Questions

1. Should the DDT be distributed (each node tracks its own pages) or centralized (coordinator tracks all)?
2. How does hash computation interact with encryption in transit? (Hash plaintext before encryption, or hash ciphertext?)
3. For GPU-side hashing, should we use a persistent CUDA kernel or launch on-demand?
4. What is the minimum dedup ratio (pages saved / pages tracked) that justifies keeping an entry in the DDT?

## Related Documents

- R12 Research 01 -- Existing Dedup Systems
- R12 Research 03 -- Copy-on-Write Network
- R10 Memory Tiering (page table design, PTE format)
- R12 Preplan
