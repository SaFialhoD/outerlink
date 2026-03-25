# R28 Research: Scatter-Gather Pipeline Design

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Design the end-to-end pipeline for scatter-gather DMA in OuterLink: how non-contiguous VRAM regions are packed for network transfer, how the receiver unpacks them, how this integrates with OpenDMA BAR1, and when to use hardware scatter-gather vs software pre-packing. Also covers integration with R14 compression.

---

## 1. Sender-Side: Gathering Non-Contiguous VRAM

### Flow Overview

```
App requests transfer of logical tensor T
    |
    v
OuterLink page table lookup (R10)
    -> Returns list of physical page addresses: [P3, P7, P1, P15, P22, ...]
    |
    v
Fragment analysis
    -> Group into contiguous runs: [{P1}, {P3}, {P7}, {P15}, {P22}]
    -> Count fragments: 5
    |
    v
Decision: scatter-gather or pre-pack?
    |
    +--> fragments <= 30: Build SGE list, one RDMA WRITE
    |
    +--> fragments > 30: Software pre-pack to staging buffer, then single-SGE RDMA WRITE
```

### Building the SGE List

When fragment count <= 30 (ConnectX-5 limit):

```
Input: Page list from R10 page table
       [page_addr_0, page_addr_7, page_addr_3, page_addr_15]

Step 1: Merge adjacent pages into contiguous runs
        Run 0: pages 3-3  -> addr=P3,  len=64KB
        Run 1: pages 7-7  -> addr=P7,  len=64KB
        Run 2: pages 15-15 -> addr=P15, len=64KB
        (If pages 3,4,5 were allocated: Run 0: addr=P3, len=192KB)

Step 2: Build ibv_sge array
        sge[0] = { addr=P3,  length=65536, lkey=vram_mr_lkey }
        sge[1] = { addr=P7,  length=65536, lkey=vram_mr_lkey }
        sge[2] = { addr=P15, length=65536, lkey=vram_mr_lkey }

Step 3: Post RDMA WRITE with sge array
        Destination: contiguous buffer on remote node
```

### Contiguous Run Merging

Critical optimization: adjacent pages in the page table should be merged into single SGEs to minimize SGE count.

Example: Tensor allocated across pages [10, 11, 12, 20, 21, 30]:
- Without merging: 6 SGEs
- With merging: 3 SGEs (pages 10-12, 20-21, 30)

This is a simple linear scan of the sorted page list — O(N) where N = number of pages.

### Software Pre-Pack Path

When fragments > 30:

```
Step 1: Allocate contiguous staging buffer (pinned host memory or contiguous VRAM)
Step 2: GPU kernel or cudaMemcpy gathers fragments into staging buffer
Step 3: Single-SGE RDMA WRITE from staging buffer
Step 4: Free staging buffer (or return to pool)
```

Cost: One extra copy (fragment -> staging). For GPU VRAM sources, a simple gather kernel runs at ~1.5 TB/s on RTX 3090 internal bandwidth, so the copy cost for 100MB is ~67us. This is small compared to network transfer time at 100Gbps (~8ms for 100MB).

---

## 2. Receiver-Side: Scattering into Non-Contiguous Remote VRAM

### Challenge

The receiver may also have non-contiguous VRAM for the destination. Two scenarios:

**Scenario A: Remote side has contiguous destination**
Simple RDMA WRITE with gather on sender side. Remote side receives into one contiguous buffer.

**Scenario B: Remote side also has non-contiguous destination**
This is harder. RDMA WRITE can only target ONE contiguous remote address. Options:

### Option 1: Two-Phase Transfer

```
Phase 1: Sender gathers + RDMA WRITE to contiguous remote staging buffer
Phase 2: Remote GPU scatter kernel copies from staging buffer to final locations
```

Requires contiguous staging buffer on receiver. Adds one extra GPU copy on receiver side.

### Option 2: Multiple RDMA WRITEs (One Per Destination Fragment)

```
For each destination fragment:
    RDMA WRITE {
        local: gather from source fragment(s)
        remote: one destination fragment address
    }
```

N destination fragments = N RDMA WRITEs. But each can still use sender-side gather (multi-SGE). Chained in one ibv_post_send() call for batching.

### Option 3: RDMA SEND + Receiver Scatter

```
Sender: Gather fragments, RDMA SEND (one message)
Receiver: ibv_post_recv with multi-SGE scatter list
```

This uses the receiver's scatter capability natively. The incoming message is automatically scattered across the receiver's SGE list. Requires SEND/RECV (not RDMA WRITE), which means receiver must pre-post receive buffers.

**Verdict:** Option 3 is the most elegant for true scatter-scatter (non-contiguous on both sides), but requires switching from RDMA WRITE to SEND/RECV for these transfers. Option 1 is simpler and works with existing RDMA WRITE infrastructure.

### Recommended Approach

| Scenario | Method |
|----------|--------|
| Contiguous source -> Contiguous dest | Single-SGE RDMA WRITE |
| Non-contiguous source -> Contiguous dest | Multi-SGE RDMA WRITE (gather) |
| Contiguous source -> Non-contiguous dest | RDMA WRITE to staging + scatter kernel |
| Non-contiguous source -> Non-contiguous dest | Gather RDMA WRITE to staging + scatter kernel (Phase 1) |
| Non-contiguous source -> Non-contiguous dest | SEND/RECV with dual scatter-gather (Phase 2, optimization) |

---

## 3. Integration with OpenDMA BAR1

### Can BAR1 Address Non-Contiguous VRAM Regions?

**Yes, but with constraints.**

BAR1 is a PCIe aperture that maps to GPU VRAM through the GPU's internal page table (not the OS page table). On G80+ NVIDIA GPUs, BAR1 references go through the GPU's VM subsystem, which means:

- BAR1 virtual addresses are contiguous from the NIC's perspective
- The GPU's page table maps these to physical VRAM pages that may be non-contiguous
- The NIC sees a contiguous address range (BAR1) but the GPU internally scatters/gathers

**Implication:** The NIC performing an RDMA WRITE to a BAR1 address range doesn't know or care that the underlying VRAM is non-contiguous. The GPU's MMU handles the translation.

### BAR1 Size Limitation

The BAR1 aperture size limits how much VRAM can be directly addressed:

| GPU | VRAM | Default BAR1 | Resizable BAR1 |
|-----|------|-------------|----------------|
| RTX 3090 | 24 GB | 256 MB | 24 GB (with rebar) |
| A100 | 80 GB | 64 GB | 64 GB |
| H100 | 80 GB | 64 GB | 128 GB |

With resizable BAR enabled on Pedro's 3090s, the entire 24GB VRAM is BAR1-addressable. Without rebar, only 256MB is directly addressable — transfers larger than this require chunking through a bounce window.

### OpenDMA + Scatter-Gather Interaction

For OpenDMA (direct NIC-to-GPU DMA via BAR1):

**Sender-side gather with OpenDMA:**
- NIC reads from multiple BAR1 addresses (gather SGEs point to BAR1 offsets)
- Each SGE read hits the GPU's BAR1 page table
- GPU's MMU translates to physical VRAM locations
- Data flows: VRAM -> GPU MMU -> BAR1 -> PCIe -> NIC -> wire

**Receiver-side with OpenDMA:**
- NIC writes to BAR1 address on remote GPU
- If destination is contiguous in BAR1 space: single RDMA WRITE works
- If destination is non-contiguous in BAR1 space: need multiple RDMA WRITEs (one per contiguous BAR1 range)

**Key insight:** BAR1 remapping means "non-contiguous in VRAM" doesn't necessarily mean "non-contiguous in BAR1 space." OuterLink's page table (R10) can map logically contiguous BAR1 ranges to physically scattered VRAM. This is the GPU MMU doing scatter for us at zero software cost.

### Optimal OpenDMA Pipeline

```
Sender (non-contiguous VRAM -> wire):
    If BAR1 mapping is contiguous for these pages:
        Single-SGE RDMA WRITE from BAR1 base address (GPU MMU scatters internally)
    Else:
        Multi-SGE RDMA WRITE with SGEs pointing to BAR1 offsets of each page

Receiver (wire -> non-contiguous VRAM):
    If BAR1 mapping is contiguous:
        Single RDMA WRITE to BAR1 base address (GPU MMU scatters internally)
    Else:
        Multiple RDMA WRITEs to each BAR1 page offset
```

---

## 4. Software Gather-Then-Send vs Hardware Scatter-Gather Tradeoff

### Cost Comparison

For transferring 30 non-contiguous 64KB pages (1.875 MB total):

**Hardware Scatter-Gather (30-SGE RDMA WRITE):**
- SGE list build: ~1us (CPU, trivial)
- NIC processes 30 SGE descriptors: ~5us overhead (30 x ~150ns per descriptor read)
- DMA reads from 30 addresses: pipelined with network transmission
- Total overhead: ~6us
- No extra memory bandwidth consumed

**Software Pre-Pack (GPU gather kernel + single-SGE RDMA WRITE):**
- Launch gather kernel: ~5us (kernel launch overhead)
- GPU reads 30 x 64KB from scattered VRAM: ~1.3us at 1.5 TB/s
- Staging buffer write: ~1.3us
- RDMA WRITE from staging: standard single-SGE latency
- Total overhead: ~8us
- Consumes 1.875 MB of staging buffer + GPU memory bandwidth

**Network transfer time for 1.875 MB at 100Gbps: ~150us**

Both overheads are negligible vs network time. Hardware scatter-gather wins by avoiding staging buffer allocation and GPU bandwidth consumption.

### When Software Pre-Pack Wins

1. **> 30 fragments:** Hardware limit exceeded, must pre-pack
2. **Compression integration (R14):** Compression operates on contiguous buffers. If compressing before send, must pre-pack first anyway.
3. **Very small fragments (< 1KB):** NIC descriptor overhead per SGE starts to dominate. Packing many tiny fragments is cheaper than many SGE reads.
4. **Receiver-side scatter needed:** If receiver also has non-contiguous destination with no BAR1 remapping, sender pre-packs and sends layout metadata so receiver can unpack.

### Decision Matrix

| Fragments | Fragment Size | Compression? | Recommendation |
|-----------|--------------|-------------|----------------|
| 1-5 | >= 4KB | No | Hardware scatter-gather |
| 6-30 | >= 4KB | No | Hardware scatter-gather |
| 1-30 | >= 4KB | Yes | Software pre-pack -> compress -> send |
| > 30 | >= 4KB | No | Software pre-pack -> send |
| Any | < 1KB | Any | Software pre-pack -> send |

---

## 5. Integration with R14 Compression

### Challenge

Compression algorithms (LZ4, Zstandard, etc.) operate on contiguous input buffers. Scatter-gather provides non-contiguous data. Three options:

### Option A: Compress Then Scatter-Gather (Not Possible)

Compressed output is one contiguous buffer — no scatter-gather needed on sender side. This doesn't solve the problem.

### Option B: Gather Then Compress Then Send

```
1. Software gather into staging buffer (GPU kernel or CPU copies)
2. Compress staging buffer (GPU-accelerated, R14)
3. Single-SGE RDMA WRITE of compressed data
4. Receiver decompresses into contiguous or non-contiguous destination
```

This is the natural pipeline when compression provides significant bandwidth savings (>2x compression ratio). The gather overhead is amortized by reduced network transfer time.

### Option C: Per-Fragment Compression

```
For each contiguous fragment:
    1. Compress fragment in-place (or to small staging buffer)
    2. Add to SGE list (compressed fragment)
3. Multi-SGE RDMA WRITE of all compressed fragments
4. Receiver decompresses each fragment independently
```

Allows combining scatter-gather with compression, but:
- Per-fragment compression has worse ratios (less context for dictionary)
- More complex receiver logic (must know each fragment's compressed/uncompressed size)
- Works well for large fragments (> 64KB) where per-fragment compression is effective

### Recommended Pipeline

```
Small transfers (< 2MB, few fragments, no compression benefit):
    -> Hardware scatter-gather RDMA WRITE

Medium transfers (2-100MB, compressible):
    -> Software gather -> R14 compress -> single-SGE RDMA WRITE

Large transfers (> 100MB, compressible):
    -> Software gather -> R14 compress -> pipelined RDMA WRITEs
    -> Overlap gather/compress/send using triple buffering
```

---

## 6. Transfer Descriptor Protocol

OuterLink needs a control message format to coordinate scatter-gather transfers between sender and receiver:

### Proposed Descriptor Format

```rust
struct ScatterGatherDescriptor {
    /// Unique transfer ID
    transfer_id: u64,
    /// Total logical size of the data (uncompressed)
    total_size: u64,
    /// Number of fragments
    fragment_count: u32,
    /// Transfer method used
    method: TransferMethod,
    /// Per-fragment metadata (only for software pre-pack with receiver scatter)
    fragments: Vec<FragmentInfo>,
}

enum TransferMethod {
    /// NIC scatter-gather, receiver gets contiguous data
    HardwareGather,
    /// Software pre-packed, contiguous on wire
    SoftwarePrePack,
    /// Software pre-packed + compressed
    CompressedPrePack { algorithm: CompressionAlgo, compressed_size: u64 },
}

struct FragmentInfo {
    /// Offset within the logical tensor
    logical_offset: u64,
    /// Size of this fragment
    size: u64,
    /// Destination page address on receiver (if receiver scatter needed)
    dest_addr: Option<u64>,
}
```

The descriptor is sent as a small control message (RDMA SEND) before the bulk data transfer. The receiver uses it to prepare receive buffers or scatter destinations.

---

## 7. Verdict for OuterLink

### Pipeline Architecture

```
Layer 1: Page Table (R10)
    Provides: physical page addresses for logical tensors

Layer 2: Fragment Analyzer (new, R28)
    Input: page address list
    Output: contiguous runs + fragment count
    Decision: hardware SG vs software pre-pack

Layer 3a: Hardware Path
    Build SGE list -> ibv_post_send with multi-SGE WR

Layer 3b: Software Path
    GPU gather kernel -> [optional R14 compress] -> single-SGE RDMA WRITE

Layer 4: Transfer Protocol
    Send descriptor -> bulk transfer -> completion notification
```

### Implementation Order

1. **Software pre-pack first** — simpler, works for all cases, establishes the protocol
2. **Hardware scatter-gather second** — optimization for fragments <= 30
3. **OpenDMA scatter-gather third** — BAR1-aware SGE building
4. **Compression integration last** — combines with software pre-pack path

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| SGE list build time | < 5us | CPU-side, simple array fill |
| Software gather (1.875 MB) | < 10us | GPU kernel at 1.5 TB/s |
| Decision overhead | < 1us | Simple fragment count check |
| Protocol descriptor | < 500 bytes | Fits in single RDMA SEND |

---

## Related Documents

- [01-rdma-scatter-gather.md](./01-rdma-scatter-gather.md) — RDMA SGE mechanics
- [02-gpu-sparse-data.md](./02-gpu-sparse-data.md) — When data is actually sparse
- [R10 Memory Tiering](../../../R10-memory-tiering/) — Page table providing addresses
- [R14 Transport Compression](../../../R14-transport-compression/) — Compression integration
- [R17 Topology-Aware Scheduling](../../../R17-topology-scheduling/) — Multi-path for large transfers

## Open Questions

- [ ] Can we use BAR1 page table manipulation to make logically contiguous ranges from scattered VRAM? (GPU MMU remapping)
- [ ] What's the actual overhead of 30-SGE ibv_post_send on ConnectX-5? Need benchmark.
- [ ] Should the transfer descriptor be embedded in the RDMA WRITE immediate data (32 bits) or sent as a separate control message?
- [ ] Triple buffering for overlap: how many staging buffers to pre-allocate?
