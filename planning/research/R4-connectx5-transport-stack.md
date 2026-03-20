# R4: ConnectX-5, GPUDirect RDMA, and Transport Stack

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Evaluate ConnectX-5 capabilities, GPUDirect RDMA viability, and choose the transport stack for OutterLink.

---

## CRITICAL FINDING: GPUDirect RDMA Won't Work With Our GPUs

**GPUDirect RDMA is only supported on NVIDIA Data Center (A100, H100) and RTX Professional (A6000) GPUs. GeForce consumer GPUs (RTX 3090 Ti, RTX 5090) are NOT supported.** This is a driver/firmware restriction, not hardware.

This means: The ConnectX-5 is still a **fast Ethernet NIC** (up to 100 GbE) but we can't do zero-copy GPU memory -> network transfers. All GPU data must stage through host pinned memory first.

**Impact:** Our transport path is confirmed as:
```
GPU VRAM -> cudaMemcpy -> pinned host memory -> network send -> network recv -> pinned host memory -> cudaMemcpy -> GPU VRAM
```
This is the same path whether we use TCP or hardware RDMA. The network layer choice affects host-to-host transfer speed, not GPU-to-network.

---

## ConnectX-5 Capabilities (Still Valuable!)

Even without GPUDirect, the ConnectX-5 provides:

| Feature | Benefit for OutterLink |
|---------|----------------------|
| Up to 100 GbE (depends on SFP/cable) | ~11 GB/s TCP throughput |
| 750ns port-to-port latency | Low-latency control messages |
| Hardware RDMA (host memory) | Zero-copy host-to-host transfers |
| Dual port | Bonding for 2x bandwidth or redundancy |
| DPDK support | Kernel bypass option (overkill for us) |

### Hardware RDMA Without GPUDirect

Even without GPUDirect, ConnectX-5 RDMA can still accelerate the host-to-host portion:

```
WITH RDMA (both sides have ConnectX):
GPU -> cudaMemcpy -> pinned host -> RDMA WRITE (zero-copy host-to-host) -> pinned host -> cudaMemcpy -> GPU
                                    ^^^^ NIC DMA directly from pinned memory, no CPU involvement

WITHOUT RDMA (TCP):
GPU -> cudaMemcpy -> pinned host -> TCP send (kernel copies to socket buffer) -> TCP recv -> pinned host -> cudaMemcpy -> GPU
                                    ^^^^ CPU copies data through kernel stack
```

RDMA eliminates the CPU copy in the network layer. For large transfers, this matters.

### Requirement: Both Sides Need ConnectX

RDMA requires RDMA hardware on **both endpoints**. Asymmetric (one ConnectX, one regular NIC) falls back to TCP. SoftRoCE on the non-ConnectX side is slower than TCP (confirmed in R2).

**Recommendation:** Get a ConnectX card for the Threadripper PC too. Used ConnectX-5 25GbE cards are ~$30-50 on eBay. This unlocks RDMA for host-memory transfers.

---

## Transport Alternatives Evaluation

### Tier 1: Recommended

| Transport | Performance | Complexity | GPU-Aware | Rust Support |
|-----------|-------------|-----------|-----------|-------------|
| **TCP + io_uring** | Good (up to ~11 GB/s on 100GbE) | Low | Via pinned memory | `tokio`, `io-uring` crate (mature) |
| **UCX** | Best (auto RDMA/TCP) | Medium | Native CUDA support | `ucx-sys`, `ucx` crate (usable) |

### Tier 2: Situational

| Transport | Performance | Complexity | Why Not Default |
|-----------|-------------|-----------|----------------|
| **libibverbs (raw RDMA)** | Maximum | Very High | UCX wraps this better |
| **GDRCopy** | Best for <4KB | Medium | Only for professional GPUs |

### Tier 3: Not Recommended

| Transport | Why Not |
|-----------|---------|
| **DPDK** | Overkill - dedicates CPU cores, removes NIC from kernel, complex |
| **SoftRoCE** | Slower than TCP (confirmed R2) |
| **NCCL** | Wrong abstraction (static ML topologies, not dynamic sharing) |
| **libfabric** | No GPU support, no Rust bindings |

---

## Recommended Transport Architecture

### Pluggable Design (Rust Trait)

```rust
trait Transport: Send + Sync {
    async fn connect(addr: &str) -> Result<Self>;
    async fn send(buf: &[u8]) -> Result<usize>;
    async fn recv(buf: &mut [u8]) -> Result<usize>;
    async fn send_gpu_buf(gpu_ptr: CUdeviceptr, size: usize) -> Result<()>;
    async fn recv_gpu_buf(gpu_ptr: CUdeviceptr, size: usize) -> Result<()>;
}
```

### Phase 1: Optimized TCP (Start Here)

**Implementation:**
1. `tokio` for async networking
2. CUDA pinned memory (`cudaHostAlloc`) for GPU data staging
3. `TCP_NODELAY`, large socket buffers (8-16 MB)
4. Optional: io_uring zero-copy send (`io-uring` crate) on Linux 6.15+

**Data flow:**
```
GPU VRAM
  -> cudaMemcpyAsync to pinned host buffer
  -> io_uring zero-copy send (or regular send)
  -> wire
  -> recv to pinned host buffer
  -> cudaMemcpyAsync to GPU VRAM
```

**Expected performance:**
| Link | Throughput |
|------|-----------|
| 10 GbE | ~1.1 GB/s |
| 25 GbE | ~2.8 GB/s |
| 100 GbE | ~10 GB/s |

**Rust crates:** `tokio`, `io-uring`, `cudarc` (CUDA FFI)

### Phase 2: UCX Backend (When RDMA Available)

**Why UCX:**
- Auto-negotiates best transport (RDMA if both sides have ConnectX, TCP if not)
- Native CUDA-aware (handles pinned memory staging automatically)
- Battle-tested in production (OpenMPI, Spark RAPIDS)
- Integrates GDRCopy for small messages automatically (if professional GPUs)

**Rust crates:** `ucx-sys` (FFI), write safe wrapper matching our Transport trait

### Phase 3 (Optional): Direct libibverbs

Only if UCX overhead is measurable. Use `sideway` crate (2025, production-focused).

---

## io_uring Deep Dive (Phase 1 Optimization)

io_uring zero-copy networking is a major performance win:

| Scenario | epoll (traditional) | io_uring ZC | Improvement |
|----------|-------------------|-------------|-------------|
| Single TCP flow (MTU 1500) | 68.8 Gbps | 90.4 Gbps | +31% |
| Single TCP flow (4096 payload) | 82.2 Gbps | 116.2 Gbps | +41% |

**Key combo: CUDA pinned memory + io_uring**
1. `cudaHostAlloc` -> pinned host buffer
2. `io_uring_register_buffers` -> register with io_uring
3. `IORING_OP_SENDZC` -> NIC DMA directly from pinned memory
4. Result: near-zero-copy NIC <-> pinned memory <-> GPU

**Requirement:** Linux 6.15+ for zero-copy recv, 6.20+ for latest features.

---

## Updated Bandwidth Reality Table

Now with ConnectX-5 and our actual hardware:

| Connection | Bandwidth | Status for Us |
|-----------|-----------|--------------|
| NVLink 3090 Ti | ~600 GB/s | Available if open-air riser works |
| PCIe 5.0 x16 (TRX50 slot) | ~64 GB/s | Threadripper native |
| PCIe 4.0 x16 (3090 Ti direct) | ~32 GB/s | GPU ceiling |
| Dual-channel RAM (MS-01) | ~90 GB/s | Not the bottleneck |
| **ConnectX-5 100GbE** | **~12.5 GB/s** | **If right SFP modules** |
| ConnectX-5 25GbE | ~3.1 GB/s | More common SFP |
| PCIe 4.0 x4 riser | ~8 GB/s | Open-air setup |
| PCIe 4.0 x1 riser | ~4 GB/s | Budget risers |
| 10 GbE | ~1.2 GB/s | Onboard NIC fallback |

**Key insight confirmed:** With x4 risers (~8 GB/s) and ConnectX-5 at 25GbE (~3.1 GB/s), the network is ~2.5x slower than the riser. At 100GbE (~12.5 GB/s), the network actually EXCEEDS x4 riser bandwidth. The bandwidth gap is narrower than most people think.

---

## Hardware Actions Needed

| Action | Priority | Cost | Impact |
|--------|----------|------|--------|
| Get ConnectX-5 25GbE for Threadripper | HIGH | ~$30-50 (used eBay) | Unlocks RDMA host-to-host |
| Direct cable between ConnectX cards | HIGH | ~$10-30 (DAC/SFP+) | Avoids switch bottleneck |
| Check ConnectX-5 model/speed on MS-01 | HIGH | Free | Know actual bandwidth ceiling |
| Check PCIe topology: `lspci -tv` | HIGH | Free | Verify GPU/NIC share root complex |
| Determine riser lane count | MEDIUM | Free | Know local GPU bandwidth |

## Rust Crate Summary

| Purpose | Crate | Maturity |
|---------|-------|---------|
| Async networking | `tokio` | Production |
| io_uring | `io-uring` | Mature |
| CUDA FFI | `cudarc` | Active development |
| UCX bindings | `ucx-sys`, `ucx` | Usable, x86-64 Linux |
| RDMA (libibverbs) | `sideway` (2025) | Newest, production-focused |
| RDMA raw | `rdma-core-sys` | Low-level, works |

## Related Documents

- [R1: Existing Projects](R1-existing-projects.md)
- [R2: SoftRoCE](R2-softroce-rdma.md)
- [R3: CUDA Interception](R3-cuda-interception.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)

## Open Questions

- [ ] What exact ConnectX-5 model is in the MS-01? (25GbE? 100GbE? InfiniBand?)
- [ ] What SFP modules/cables are available?
- [ ] PCIe topology on MS-01: does GPU slot share root complex with ConnectX-5?
- [ ] Linux kernel version on both PCs? (io_uring ZC needs 6.15+)
- [ ] Budget for a second ConnectX-5 card?
- [ ] Is there a way to enable GPUDirect RDMA on GeForce? (community patches, custom drivers?)
