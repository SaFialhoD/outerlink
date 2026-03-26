# R15 Research: Erasure Coding Algorithms

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Evaluate erasure coding schemes for protecting GPU memory across OuterLink cluster nodes. Select the optimal algorithm balancing storage overhead, encoding/decoding throughput, failure tolerance, and computational cost for 2-8 node consumer GPU clusters.

---

## 1. Reed-Solomon (RS) Erasure Coding

### Theory

Reed-Solomon codes are Maximum Distance Separable (MDS) codes: an RS(k, m) code splits data into k data fragments and generates m parity fragments. Any k of the (k+m) total fragments suffice to reconstruct the original data. This is provably storage-optimal — no erasure code can achieve lower overhead for the same fault tolerance.

RS operates over Galois Field GF(2^8), meaning the maximum total fragments is 255 (k+m <= 255). For OuterLink's 2-8 node clusters, this is not a constraint.

### Storage Overhead

| Configuration | Data Fragments (k) | Parity Fragments (m) | Storage Overhead | Fault Tolerance |
|---------------|--------------------|-----------------------|------------------|-----------------|
| RS(2,1)       | 2                  | 1                     | 50%              | 1 node          |
| RS(4,2)       | 4                  | 2                     | 50%              | 2 nodes         |
| RS(6,2)       | 6                  | 2                     | 33%              | 2 nodes         |
| RS(4,1)       | 4                  | 1                     | 25%              | 1 node          |
| RS(6,3)       | 6                  | 3                     | 50%              | 3 nodes         |

### Computational Cost

RS encoding/decoding requires Galois Field matrix multiplication. The complexity is O(k * m * block_size) for encoding, O(k^2 * block_size) for decoding (matrix inversion + multiply). This is quadratic in the number of fragments, which is significant for large k but manageable for OuterLink's small cluster sizes (k <= 8).

### CPU-Based RS Performance: Intel ISA-L

Intel's Intelligent Storage Acceleration Library (ISA-L) is the industry standard for CPU-based erasure coding, used by Ceph, HDFS, MinIO, and Carbink.

**ISA-L Performance Numbers:**
- **Single-core encoding:** 5-20+ GB/s depending on configuration and CPU generation (AVX2/AVX-512)
- **ISA-L vs Jerasure:** 16x faster single-threaded, 8x faster with 40 threads
- **ISA-L vs legacy Java coders (HDFS):** 70x faster single-threaded
- **Multi-core scaling:** On AMD EPYC 7601 (32 cores), comparable RS libraries achieved 88 GB/s encoding, 111 GB/s decoding
- **Network bottleneck:** At 100 Gbps (~12.5 GB/s), a single core of ISA-L encoding can saturate the wire

**OuterLink relevance:** Our ConnectX-5 delivers ~12.5 GB/s. A single CPU core running ISA-L at ~10 GB/s nearly saturates this. Encoding will NOT be a bottleneck — network bandwidth is the limiting factor.

### GPU-Accelerated RS Performance

Several research libraries demonstrate RS encoding on GPUs:

**G-CRS (2018, IEEE TPDS):**
- GPU-based Cauchy Reed-Solomon implementation
- 10x faster than CPU libraries (Jerasure, ISA-L)
- 3x faster than PErasure on same GPU architecture
- Optimized for Maxwell/Pascal GPUs; modern Ampere/Ada GPUs would be significantly faster

**PErasure (2015):**
- Parallel Cauchy RS library for GPUs
- Up to 10 GB/s encoding on GTX 780 (2013 GPU)
- Tolerates up to 8 disk failures
- 10x faster than multi-threaded Jerasure on quad-core CPU

**ECS2 (2020):**
- Integrates GPUDirect for direct NVMe-to-GPU data path
- Addresses the data transfer bottleneck between storage and GPU
- Most relevant to OuterLink's OpenDMA architecture

**Projected performance on RTX 3090:**
- The RTX 3090 has ~35 TFLOPS FP32 and 936 GB/s memory bandwidth
- Compared to GTX 780 (3.97 TFLOPS, 288 GB/s), this is ~9x compute, ~3x bandwidth
- Projected RS encoding: 30-90 GB/s on RTX 3090
- This vastly exceeds network bandwidth; GPU-accelerated EC is overkill for encoding speed but valuable for minimizing GPU stalls during recovery

### RS Verdict for OuterLink

**Strong candidate.** Storage-optimal, well-understood, battle-tested in production systems. ISA-L on CPU is fast enough (saturates 100Gbps wire). GPU-accelerated RS available if needed for recovery speed. The quadratic complexity is irrelevant at our cluster sizes (k <= 8).

---

## 2. XOR Parity (RAID-5 Style)

### Theory

The simplest erasure code: XOR all data blocks together to produce a single parity block. To recover any single lost block, XOR all surviving blocks (including parity). This is equivalent to RS(k, 1).

### Properties

| Property | Value |
|----------|-------|
| Storage overhead | 1/k (e.g., 25% for 4 nodes) |
| Fault tolerance | Exactly 1 failure |
| Encoding speed | Fastest possible (single XOR pass) |
| Decoding speed | Fastest possible (single XOR pass) |
| CPU cost | Negligible — memory bandwidth limited |
| Implementation complexity | Trivial (~50 lines of code) |

### Performance

XOR parity is memory-bandwidth-limited, not compute-limited:
- On modern CPUs: limited by DRAM bandwidth (~50 GB/s DDR4, ~80 GB/s DDR5)
- On GPUs: limited by VRAM bandwidth (936 GB/s on RTX 3090)
- With AVX-512: a single core can XOR at ~30 GB/s

### XOR Verdict for OuterLink

**Excellent as a fast tier.** Perfect for hot data that needs minimal overhead and maximum encoding speed. The single-failure limitation is acceptable if combined with RS for cold/important data. HDFS includes XOR-2-1-1024k as a built-in policy alongside RS codes.

---

## 3. Fountain Codes (LT, Raptor)

### Theory

Fountain codes (rateless erasure codes) can generate an unlimited number of encoded symbols from k source symbols. Any (1+epsilon)*k encoded symbols suffice for decoding, where epsilon is a small overhead (typically 1-5%).

**LT Codes:** First practical fountain code. Encoding is simple (XOR random subsets), but decoding requires belief propagation with O(k * ln(k)) complexity.

**Raptor Codes:** Concatenation of a pre-code (LDPC or similar) with an LT code. Achieves O(k) linear-time encoding and decoding. Raptor codes are the first fountain codes with theoretical linear complexity.

### Comparison with Reed-Solomon

| Property | Reed-Solomon | Raptor Codes |
|----------|-------------|--------------|
| Optimality | MDS (storage-optimal) | Near-optimal (1-5% overhead) |
| Encoding complexity | O(k * m * B) | O(k * B) linear |
| Decoding complexity | O(k^2 * B) | O(k * B) linear |
| Max block length | 255 (GF(2^8)) | Unlimited |
| Rateless | No (fixed m) | Yes (generate as many as needed) |
| Repair cost | High (need k blocks) | Lower (local group repair) |
| Maturity in storage | Very high (Ceph, HDFS, MinIO) | Low (mainly streaming/broadcast) |

### Fountain Codes Verdict for OuterLink

**Not recommended.** Fountain codes excel at large-scale broadcast/multicast and very large k values. For OuterLink's 2-8 node clusters, k is tiny (2-7), which eliminates fountain codes' complexity advantage. The near-optimal overhead (extra 1-5% storage) buys us nothing when RS is already storage-optimal at these sizes. No mature storage-focused implementations exist. Raptor codes are patented by Qualcomm (though some patents have expired).

---

## 4. LDPC (Low-Density Parity-Check) Codes

### Brief Assessment

LDPC codes are used in 5G, Wi-Fi 6, and some storage systems. They offer near-optimal performance at very large block sizes but have higher overhead than RS at small sizes. Similar to fountain codes, their advantages only manifest at scale far beyond OuterLink's cluster sizes.

**Verdict: Not recommended.** Same reasoning as fountain codes — RS dominates at small k.

---

## 5. Existing Systems and Their Choices

### Carbink (OSDI '22) — Most Relevant Reference

Carbink implements fault-tolerant far memory using erasure coding, which is the closest prior work to OuterLink's needs.

**Key design decisions:**
- **EC library:** Intel ISA-L v2.30.0 (Reed-Solomon)
- **EC strategy:** EC-Batch (encode across a "spanset" — a group of equal-sized memory spans)
- **Configuration:** 4 data chunks + 2 parity chunks (RS(4,2))
- **Granularity:** Spanset-level encoding, span-level swap-in
- **Recovery:** Read 4x span/parity data to reconstruct lost data
- **Performance vs Hydra:** 29% lower tail latency, 48% higher application performance, at most 35% higher memory usage
- **Recovery characteristics:** Degraded reads concentrated in first second of recovery; hot objects pulled into local memory quickly stop generating degraded reads

**Lessons for OuterLink:**
1. EC-Batch (parity across whole spans, not split fragments) is superior for swap-in performance
2. ISA-L on CPU is sufficient — no need for GPU-accelerated EC
3. Parity update complexity matters when pages migrate frequently
4. Recovery time scales linearly with remote data size

### Ceph (RS via ISA-L)

- Default profiles: RS(4,2), RS(6,3), RS(8,4)
- Uses ISA-L for computation
- Stripe size typically 4KB-4MB
- Recovery I/O is the bottleneck, not computation

### HDFS

- Built-in policies: RS-3-2-1024k, RS-6-3-1024k, RS-10-4-1024k, XOR-2-1-1024k
- ISA-L for native encoding, Java fallback
- 1MB stripe size default

### MinIO

- Uses ISA-L with AVX-512
- Writes erasure-coded objects at near wire speed on 100 Gbps networks

---

## 6. Comparison Table: All Schemes

| Scheme | Storage Overhead | Max Failures | Encode Speed (CPU) | Encode Speed (GPU) | Implementation | Maturity |
|--------|-----------------|--------------|--------------------|--------------------|----------------|----------|
| XOR (k,1) | 1/k (25% at k=4) | 1 | ~50 GB/s (mem BW) | ~900 GB/s (VRAM BW) | Trivial | Very High |
| RS(4,2) | 50% | 2 | ~10 GB/s/core (ISA-L) | ~30-90 GB/s (est.) | ISA-L available | Very High |
| RS(6,2) | 33% | 2 | ~8 GB/s/core (ISA-L) | ~20-60 GB/s (est.) | ISA-L available | Very High |
| RS(6,3) | 50% | 3 | ~6 GB/s/core (ISA-L) | ~15-50 GB/s (est.) | ISA-L available | High |
| Raptor | ~1/k + 3-5% | Configurable | O(k) linear | No mature impl | Complex | Low (storage) |
| LDPC | Near-optimal | Configurable | Fast at large k | Some GPU impls | Complex | Medium |

---

## 7. Recommendation for OuterLink

### Primary Scheme: Reed-Solomon via ISA-L (CPU)

**Configuration: Adaptive based on cluster size**

| Cluster Size | Recommended EC | Overhead | Tolerance | Rationale |
|-------------|---------------|----------|-----------|-----------|
| 2 nodes | RS(2,1) + local DRAM mirror | 50% | 1 node | Minimum viable; parity in partner's DRAM |
| 3 nodes | RS(2,1) distributed | 50% | 1 node | Parity spread across non-owner nodes |
| 4 nodes | RS(3,1) or RS(2,2) | 33% or 100% | 1 or 2 nodes | Trade overhead for tolerance |
| 5-6 nodes | RS(4,2) | 50% | 2 nodes | Sweet spot: Carbink-proven config |
| 7-8 nodes | RS(6,2) | 33% | 2 nodes | Lower overhead, same tolerance |

### Secondary Scheme: XOR Parity for Hot Data

For frequently accessed pages (hot tier in R10's memory hierarchy), use simple XOR parity:
- Single parity block stored on a designated partner node
- Recovery from 1 failure in microseconds (single XOR pass)
- Minimal encoding overhead on the critical path
- Fall back to RS for multi-node failures (rare)

### Why NOT GPU-Accelerated EC

1. **ISA-L on CPU saturates our 100 Gbps network** — encoding is not the bottleneck
2. **GPU VRAM is precious** — using VRAM for EC computation wastes the resource we're protecting
3. **GPU compute should serve applications** — EC encoding in background on CPU cores is free
4. **Exception:** During recovery, GPU-accelerated decoding could speed reconstruction. Worth prototyping but not required for v1.

### Parity Storage Placement

| Data Location | Parity Location | Rationale |
|--------------|----------------|-----------|
| Remote VRAM | Partner node DRAM | Cheaper than VRAM; DRAM is faster than NVMe for recovery |
| Remote DRAM | Another node's DRAM | Symmetric placement |
| NVMe tier | NVMe on another node | Same tier, different node |
| Hot VRAM page | Partner VRAM (XOR only) | Ultra-fast recovery for critical data |

---

## Open Questions

1. **Parity update cost:** When a page is modified, its parity must be updated. For RS, this requires re-encoding the entire stripe. For XOR, only the changed block's XOR contribution needs updating (old XOR new XOR parity = new parity). How frequent are page modifications in inference vs training workloads?

2. **Stripe size:** Carbink uses spansets (groups of spans). What should OuterLink's stripe size be? Must align with R10's 64KB page size. Options: single page per fragment, or multi-page stripes.

3. **Encoding on write path:** Should we encode synchronously (every write waits for parity) or asynchronously (write returns immediately, parity computed in background)? Async is faster but creates a vulnerability window.

4. **Degraded reads during recovery:** How to handle reads to data being reconstructed? Carbink showed degraded reads concentrated in the first second. Can we prioritize reconstruction of hot pages?

---

## Related Documents

- R10: Memory Tiering (page size, tier locations, eviction policies)
- R12: Memory Deduplication (read-only shared pages have implicit redundancy)
- R17: Topology-Aware Scheduling (failure detection, node health monitoring)
- R19: SWMR Consistency (coherency state affects when parity updates are needed)
- Carbink paper: https://www.usenix.org/conference/osdi22/presentation/zhou-yang

## References

- G-CRS: GPU Accelerated Cauchy Reed-Solomon Coding (IEEE TPDS 2018)
- PErasure: A Parallel Cauchy Reed-Solomon Coding Library for GPUs (IEEE 2015)
- ECS2: Fast Erasure Coding Library for GPU-Accelerated Storage (IEEE 2020)
- Intel ISA-L: https://github.com/intel/isa-l
- Carbink: Fault-Tolerant Far Memory (OSDI '22)
- Fast Erasure Coding for Data Storage (USENIX FAST '19)
- klauspost/reedsolomon (Go): https://github.com/klauspost/reedsolomon
- Qualcomm Raptor Codes whitepaper: https://www.qualcomm.com/media/documents/files/why-raptor-codes-are-better-than-reed-solomon-codes-for-streaming-applications.pdf
