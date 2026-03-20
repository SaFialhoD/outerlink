# R7: Non-Proprietary GPU DMA - Bypassing NVIDIA's GPUDirect Restriction

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** CRITICAL - POTENTIAL KILLER FEATURE

## Purpose

Determine if we can build a non-proprietary alternative to GPUDirect RDMA that enables direct NIC-to-GPU VRAM access on ALL NVIDIA GPUs, including GeForce.

## TL;DR - YES, THERE ARE VIABLE PATHS

The fundamental mechanism is simple: a ConnectX-5 NIC issues a standard PCIe Memory Write TLP to the GPU's BAR1 physical address. **This is standard PCIe - no proprietary technology needed.** The only challenge is getting the GPU's internal page tables configured so BAR1 accesses reach VRAM.

Three viable paths exist:

| Path | Feasibility | Timeline | Legal Risk |
|------|------------|----------|------------|
| **tinygrad P2P patch + custom RDMA module** | Medium-High | We could attempt this | Low-Medium |
| **nouveau upstream RDMA patches** | High (once merged) | Waiting on kernel merge | None |
| **P2PDMA kernel framework** | Medium | Needs custom provider module | None |

---

## The Key Insight: It's Just PCIe

GPU VRAM is exposed through **PCIe BAR1**. This is a standard PCIe Base Address Register. ANY PCIe device can read/write to any other PCIe device's BAR region - that's how PCIe works. NVIDIA's restriction is in their proprietary `nvidia_p2p_get_pages()` API, but the underlying hardware mechanism (PCIe BAR access) is open and standard.

```
What NVIDIA blocks:        What PCIe allows:
App -> nvidia_p2p_get_pages()    NIC -> PCIe Memory Write TLP -> GPU BAR1 -> VRAM
       "GeForce? DENIED"                (standard PCIe, no permission needed)
```

## The One Problem: GPU MMU Page Tables

On modern GPUs (G80+, which is everything since 2006), BAR1 accesses go through the GPU's internal MMU. The GPU's page tables must be configured to map BAR1 virtual addresses to physical VRAM locations. Without page table setup, writes to BAR1 don't reach VRAM.

**Who can set up GPU page tables:**
1. NVIDIA's proprietary driver (does this normally)
2. The tinygrad patched driver (maps ALL VRAM into BAR1)
3. nouveau (open source NVIDIA driver)

## ReBAR: Full VRAM Through BAR1

With **Resizable BAR (ReBAR)** enabled, the RTX 3090 Ti exposes the **full 24GB of VRAM** through BAR1. Without ReBAR, BAR1 is only 256MB.

ReBAR requirements:
- Motherboard BIOS: UEFI mode, "Above 4G Decoding" enabled, ReBAR enabled
- Compatible CPU (AMD Ryzen 3000+, Intel 10th gen+)
- Driver 465.89+

---

## Path 1: tinygrad P2P Patch + Custom RDMA Module (MOST PROMISING)

### How It Works

1. **tinygrad's patched nvidia-open kernel modules** already solve the page table problem:
   - Calls `kbusEnableStaticBar1Mapping_GH100` to map entire VRAM into BAR1
   - Rewrites GPU MMU aperture types to enable peer-to-peer access
   - Proven on RTX 4090, achieving ~24.5 GB/s P2P between GPUs

2. **We write a custom kernel module** that:
   - Takes the BAR1 physical addresses (now mapped to all VRAM)
   - Registers them with the RDMA subsystem as a peer memory client
   - ConnectX-5's mlx5 driver can then RDMA directly to GPU VRAM

### What Exists vs What We Build

| Component | Status |
|-----------|--------|
| tinygrad BAR1 mapping patches | EXISTS (GitHub) |
| RTX 3090 Ti support | EXISTS (aikitoria fork extends to 5090) |
| Peer memory client API | EXISTS (MLNX_OFED) |
| Custom RDMA registration module | **WE BUILD THIS** |
| Integration testing | **WE DO THIS** |

### The Open Question

**Does nvidia-peermem work once tinygrad patches map all VRAM into BAR1?** If the patched driver has already set up BAR1 page tables for all VRAM, does `nvidia_p2p_get_pages()` then succeed on GeForce? This is the most direct test - if it works, we don't even need a custom module.

If `nvidia_p2p_get_pages()` still blocks (checking GPU class independently of BAR1 state), then we write our own peer memory client that reads BAR1 physical addresses directly from PCI config space.

### Feasibility: Medium-High

The tinygrad patches prove BAR1 mapping works. The peer memory client API is documented. The gap is connecting the two.

---

## Path 2: nouveau Upstream RDMA Patches (CLEANEST LONG-TERM)

### What's Happening

NVIDIA engineer Yonatan Maman posted **upstream kernel patches** implementing GPU Direct RDMA P2P DMA for the nouveau (open source) driver:
- RFC patches: December 2024
- v2 patches: July 2025
- Claims: **10x higher bandwidth and 10x lower latency** vs host-staged transfers

### How It Works

```
nouveau driver
  -> exports GPU VRAM pages as device private pages
  -> HMM (Heterogeneous Memory Management) P2P DMA support
  -> RDMA/mlx5 uses ATS (Address Translation Service) for ODP
  -> ConnectX-5 DMA directly to GPU VRAM
```

### The Catch

- Patches are **not yet merged** into mainline kernel
- nouveau has **very limited compute support** (no CUDA)
- Would need to be used alongside the proprietary driver somehow, or nouveau would need CUDA-equivalent compute (unlikely)

### However: The Code Is Tiny

The entire patch series is **less than 200 lines of code**. It touches:
- `mm/hmm` - P2P page operations for device private pages
- `nouveau/dmem` - HMM P2P DMA support
- `IB/core` - P2P DMA infrastructure
- `RDMA/mlx5` - ATS for ODP memory

### Feasibility: High (once merged, with caveats)

This is the "right" way - fully upstream, fully open source, no legal risk. But nouveau's lack of CUDA is a blocker unless we find a way to use nouveau for memory management alongside NVIDIA's proprietary driver for compute.

---

## Path 3: Linux P2PDMA Framework

### How It Works

Linux kernel P2PDMA (since kernel 4.20) is designed exactly for this: any two PCIe devices DMAing to each other's memory.

A **provider** module registers a BAR as a P2P resource:
```c
pci_p2pdma_add_resource(gpu_pdev, 1 /* BAR1 */, bar1_size, 0);
pci_p2pmem_publish(gpu_pdev);
```

The **mlx5 driver** (ConnectX-5) as a P2PDMA client can then DMA to those pages.

### What's Needed

A kernel module that acts as the P2PDMA provider for GPU BAR1. This module needs to:
1. Know the GPU's BAR1 physical address (from PCI config space)
2. Ensure GPU page tables map BAR1 to VRAM (via tinygrad patch or nouveau)
3. Register BAR1 with P2PDMA
4. Handle topology requirements (GPU and NIC should share PCIe root complex)

### Feasibility: Medium

Uses only upstream kernel infrastructure. No out-of-tree dependencies. But needs GPU MMU setup from another source.

---

## Path 4: DMA-BUF (Vendor-Neutral Framework)

### The Framework

DMA-BUF is Linux's standard buffer sharing framework. A GPU driver exports a buffer as a dma-buf object, an RDMA NIC driver imports it. The framework itself is vendor-neutral.

### The Restriction

NVIDIA's proprietary driver restricts DMA-BUF RDMA export to datacenter GPUs (same as nvidia_p2p_get_pages). But if **nouveau** exports GPU VRAM as dma-buf, there's no restriction.

### Linux 6.19: DMA-BUF for VFIO PCI

As of Linux 6.19, VFIO PCI supports DMA-BUF export of MMIO regions. This enables "low-level interactions such as SPDK drivers interacting directly with dma-buf capable RDMA devices to enable peer-to-peer operations."

### Feasibility: Low-Medium (needs nouveau dma-buf VRAM export)

---

## AMD's Approach: How It Should Work

For reference, **AMD does NOT restrict consumer GPUs**:
- ROCnRDMA is fully open source
- Registers as `peer_memory_client` with IB core
- Works with ConnectX NICs
- NO product class checks

Source: github.com/rocmarchive/ROCnRDMA

This proves the concept works and is legally sound. NVIDIA's restriction is purely a business decision.

---

## Hardware Requirements for Any Path

| Requirement | Why | Status |
|------------|-----|--------|
| ReBAR enabled | Full 24GB VRAM through BAR1 | BIOS setting |
| IOMMU disabled | P2P DMA fails with IOMMU | Kernel boot param: `iommu=off` |
| ACS disabled | ACS forces P2P through root complex | BIOS or kernel patch |
| GPU + NIC same PCIe root complex | Optimal P2P path | Check with `lspci -tv` |
| Linux kernel 5.12+ | DMA-BUF RDMA support | Standard |

---

## Recommended Strategy for OutterLink

### Phase 1: Ship without GPU DMA (works today)
Host-staged transfers: GPU -> cudaMemcpy -> pinned host -> RDMA/TCP -> pinned host -> cudaMemcpy -> GPU

### Phase 2: Experiment with tinygrad + custom RDMA module
1. Apply tinygrad patches to nvidia-open modules
2. Test if nvidia-peermem works on GeForce with patched driver
3. If not: write custom peer memory client that registers BAR1 with RDMA subsystem
4. Benchmark: compare host-staged vs direct BAR1 RDMA

### Phase 3: Monitor nouveau upstream patches
When Yonatan Maman's patches merge, evaluate if nouveau can handle VRAM management while NVIDIA proprietary handles CUDA compute.

### Phase 4: Publish findings
If we achieve non-proprietary GPU RDMA on GeForce, publish the method. This benefits the entire community and establishes OutterLink as the project that solved this.

---

## What This Means for OutterLink

If Path 1 works (tinygrad patch + custom RDMA module):

```
BEFORE (host-staged):
GPU -> cudaMemcpy (5us) -> host -> network (2us) -> host -> cudaMemcpy (5us) -> GPU
Total: ~12us per transfer

AFTER (direct BAR1 RDMA):
GPU VRAM -> ConnectX-5 RDMA direct to BAR1 -> GPU VRAM
Total: ~2us per transfer
```

- **6x lower latency**
- **Zero CPU involvement** in data transfer
- **Works on ALL NVIDIA GPUs** (GeForce, Professional, Datacenter)
- **No proprietary NVIDIA P2P API used**
- **OutterLink's killer differentiator** - no other project has this

---

## Related Documents

- [R5: GPUDirect on GeForce](R5-gpudirect-geforce-restriction.md)
- [R4: ConnectX-5 + Transport](R4-connectx5-transport-stack.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)

## Key Sources

1. envytools: PCI BARs documentation (BAR1 = VRAM mapping)
2. Linux Kernel P2PDMA documentation
3. Linux Kernel DMA-BUF documentation
4. tinygrad/open-gpu-kernel-modules (P2P patches)
5. tinygrad Issue #46 (GPUDirect RDMA request)
6. aikitoria/open-gpu-kernel-modules (RTX 5090 support)
7. Phoronix: NVIDIA P2P DMA RDMA patches for nouveau
8. nouveau mailing list: v2 RDMA patches (July 2025)
9. AMD ROCnRDMA (open source GPU RDMA reference)
10. Intel DMA-BUF RDMA proposal (OFA 2021)
11. NVIDIA ReBAR documentation
12. Phoronix: Linux 6.19 DMA-BUF VFIO PCI support
13. Alibaba DxPU paper (PCIe TLP-level GPU disaggregation)

## Open Questions

- [ ] Does nvidia-peermem work on GeForce with tinygrad BAR1 patches applied?
- [ ] What's the status of Yonatan Maman's nouveau RDMA patches? (last: v2, July 2025)
- [ ] Can nouveau handle VRAM management alongside proprietary CUDA driver?
- [ ] What is the actual BAR1 page table format on Ampere (RTX 3090 Ti)?
- [ ] PCIe topology on both PCs: do GPU and ConnectX-5 share root complex?
- [ ] Performance of BAR1 RDMA vs host-staged on our specific hardware?
