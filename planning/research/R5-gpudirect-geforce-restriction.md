# R5: GPUDirect RDMA on GeForce - The Restriction and Workarounds

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Determine if GPUDirect RDMA can be enabled on GeForce consumer GPUs (RTX 3090 Ti, RTX 5090).

## TL;DR - NOT CURRENTLY POSSIBLE, BUT THE PROJECT STILL WORKS WITHOUT IT

Nobody has achieved GPUDirect RDMA on GeForce. The block is multi-layered:
1. Proprietary nvidia.ko driver checks GPU product class
2. GPU System Processor (GSP) firmware may independently enforce it
3. Even the open-source kernel modules can't bypass it because RM code runs on signed GSP firmware

**But here's why this doesn't kill OutterLink:** With host-staged transfers + 4x 100GbE bonded ConnectX-5 (~50 GB/s), we still exceed PCIe 4.0 x16 bandwidth. GPUDirect would eliminate one memcpy hop, saving ~10-50us latency per transfer - nice but not essential.

---

## How NVIDIA Blocks It

```
[Your RDMA app / nvidia-peermem / GDRCopy]
        |
        | calls nvidia_p2p_get_pages()
        v
[nvidia.ko kernel module]
        |
        | checks GPU PCI device ID / product class
        | GeForce? -> return -EINVAL (error 22)
        | Datacenter/Professional? -> proceed
        v
[GPU System Processor (GSP) firmware]  <-- signed, not modifiable
        |
        v
[GPU Hardware - BAR1 / VRAM mapping]
```

The check is in the **proprietary Resource Manager (RM)** code inside nvidia.ko. This code runs on the GSP firmware (signed, closed source) on Turing+ GPUs. Even the open-source nvidia kernel modules (`nvidia-open`) are just a thin shim that talks to the GSP - the actual logic is in firmware.

**nvidia-peermem is NOT the blocker.** It's fully open source and contains no GPU type checks. It just calls `nvidia_p2p_get_pages()` which fails on GeForce.

---

## What HAS Been Achieved (P2P, Not RDMA)

| Feature | RTX 3090/3090 Ti | RTX 4090 | RTX 5090 |
|---------|-----------------|----------|----------|
| NVLink (GPU-to-GPU, same PC) | YES (with bridge) | NO (removed) | TBD |
| CUDA P2P (GPU-to-GPU, same PC) | Partial | YES (tinygrad hack) | NOT YET |
| GPUDirect RDMA (GPU-to-NIC) | NO | NO | NO |
| GDRCopy (CPU mapping of GPU mem) | NO | NO | NO |

**tinygrad's P2P hack (RTX 4090 only):**
- Modified nvidia-open kernel modules
- Rewrites `GMMU_APERTURE_PEER` to `GMMU_APERTURE_SYS_NONCOH` in page tables
- Achieves ~24.5 GB/s P2P between GPUs
- Published on GitHub (geohot/tinygrad team)
- Does NOT enable RDMA - only GPU-to-GPU within same machine

**RTX 3090 Ti advantage:** Last GeForce generation with hardware NVLink support. P2P between 3090 Ti cards via NVLink works natively - no hack needed.

---

## Why It's Market Segmentation, Not Technical

The silicon is the same. NVIDIA artificially restricts features to push datacenter customers toward $10,000+ A100/H100 cards instead of $1,500 GeForce cards:

| Feature | GeForce | Datacenter (A100/H100) |
|---------|---------|----------------------|
| CUDA compute | YES | YES |
| NVLink | Removed after 3090 | YES |
| GPUDirect RDMA | BLOCKED | YES |
| GDRCopy | BLOCKED | YES |
| ECC memory | Was blocked, now allowed | YES |
| MIG | NO | YES |
| vGPU | NO | YES |

The EULA reinforces this: *"GeForce SOFTWARE is not licensed for datacenter deployment."*

---

## Legal Risk Assessment

| Scenario | Risk Level | Notes |
|----------|-----------|-------|
| Personal research/tinkering | LOW | No known NVIDIA legal action against individuals |
| Publishing patches on GitHub | LOW-MEDIUM | tinygrad P2P hack is public, no legal action |
| Distributing patched drivers | MEDIUM | More visibility, EULA violation |
| Commercial deployment | HIGH | EULA explicitly prohibits datacenter use |
| Including bypass in OutterLink | NOT RECOMMENDED | Would expose project to legal risk |

**Precedent:** geohot (tinygrad) publicly posted the P2P hack on GitHub and discussed it on Hacker News. No legal action from NVIDIA. But P2P ≠ RDMA, and a formal project shipping a bypass is different from a community hack.

---

## What Would Need to Happen for GeForce RDMA

1. Achieve P2P first (done for 4090, not for 3090 Ti or 5090 via hack)
2. Determine if `nvidia_p2p_get_pages()` rejection is kernel-side only or also GSP firmware
3. If kernel-side: patch nvidia-open modules to bypass product class check
4. If GSP firmware: reverse-engineer signed firmware (extremely hard)

The tinygrad community has an open issue (#46) specifically requesting this. It's an active area of research but unsolved.

---

## Impact on OutterLink Architecture

### Without GPUDirect RDMA (Current Reality)

```
GPU VRAM
  -> cudaMemcpy (~5us, PCIe bandwidth)
  -> Pinned Host Memory
  -> RDMA WRITE via ConnectX-5 (~2us, 100GbE bandwidth)
  -> Pinned Host Memory (remote)
  -> cudaMemcpy (~5us, PCIe bandwidth)
  -> GPU VRAM (remote)

Total: ~12us latency overhead, bandwidth = min(PCIe, network)
```

### With GPUDirect RDMA (Hypothetical)

```
GPU VRAM
  -> RDMA WRITE directly from VRAM via ConnectX-5 (~2us)
  -> GPU VRAM (remote)

Total: ~2us latency, bandwidth = min(PCIe, network)
```

**What we lose without GPUDirect:** ~10us extra latency per transfer (two cudaMemcpy hops). For large bulk transfers this is negligible (amortized over transfer time). For small messages (<4KB) it's significant.

**What we keep:** Full network bandwidth. With 4x 100GbE bonded (~50 GB/s), the network is NOT the bottleneck - PCIe is.

### OutterLink Design Implication

Our transport abstraction should have a `TransportCapability` enum:

```
enum TransportCapability {
    TcpBasic,           // Regular TCP
    TcpZeroCopy,        // TCP + io_uring zero-copy
    RdmaHostStaged,     // RDMA with cudaMemcpy staging (our current path)
    RdmaGpuDirect,      // GPUDirect RDMA (if available - datacenter GPUs)
}
```

OutterLink auto-detects and uses the best available. If someone runs it with A100s + ConnectX, they get GPUDirect. With GeForce + ConnectX, they get RDMA host-staged. With no ConnectX, they get TCP.

---

## Potential Future: DMA-BUF Path

NVIDIA is moving toward Linux kernel DMA-BUF for GPU memory sharing (replacing nvidia-peermem). DMA-BUF is a kernel-standard mechanism, not NVIDIA-proprietary. Open question: does the DMA-BUF path also check GPU product class, or could it bypass the restriction?

This is worth monitoring but not actionable today.

## Related Documents

- [R2: SoftRoCE](R2-softroce-rdma.md)
- [R4: ConnectX-5 + Transport](R4-connectx5-transport-stack.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)

## Key Sources

1. tinygrad/open-gpu-kernel-modules - P2P hack for RTX 4090
2. tinygrad Issue #46 - GPUDirect RDMA request (unsolved)
3. NVIDIA/open-gpu-kernel-modules Discussion #699 - RDMA on consumer GPUs
4. nvidia-peermem.c source - confirms no GPU check in peermem
5. GDRCopy Issue #211 - confirms GeForce not supported
6. NVIDIA GeForce Software License - EULA restrictions
7. forkProj/open-gpu-kernel-modules-P2P - BAR1-based P2P approach
