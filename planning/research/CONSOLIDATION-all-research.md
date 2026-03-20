# Research Consolidation: Everything We Know

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete

## Purpose

Single document consolidating all 7 research documents into actionable knowledge for pre-planning.

---

## What We're Building

**OutterLink** is a Rust software layer that makes GPUs across separate PCs work as a unified pool. Any CUDA application runs unmodified - OutterLink transparently intercepts GPU operations and routes them to local or remote GPUs.

### The Two-Layer Architecture

```
WITHIN each PC:  NVLink (112.5 GB/s between GPU pairs)
ACROSS PCs:      ConnectX-5 100GbE (12.5-50 GB/s via RDMA)

PC1:                                    PC2:
GPU_A <==NVLink==> GPU_B    <--network-->    GPU_C <==NVLink==> GPU_D
 24GB   112.5GB/s   24GB    12.5-50GB/s       24GB   112.5GB/s   24GB

                    Unified pool: 96-112 GB VRAM + 512 GB RAM
```

---

## Decisions Made (From Research)

| # | Decision | Answer | Source |
|---|----------|--------|--------|
| D1 | Language | **Rust** | Pedro's preference, matches other projects |
| D2 | CUDA interception | **Driver API + LD_PRELOAD + cuGetProcAddress** | R3 - industry consensus |
| D3 | Transport (Phase 1) | **TCP + io_uring + CUDA pinned memory** | R2 (SoftRoCE killed), R4 |
| D3 | Transport (Phase 2) | **UCX (auto-negotiates RDMA vs TCP)** | R4 |
| D4 | GPU DMA (Phase 1) | **Host-staged** (cudaMemcpy through pinned host mem) | R5 |
| D4 | GPU DMA (Phase 2) | **Direct BAR1 RDMA** (non-proprietary, our killer feature) | R7 |
| D5 | Build approach | **Clean-room in Rust** (study HAMi-core, Cricket, SCUDA patterns) | R1 |
| D7 | License | **Apache 2.0** (recommended, pending Pedro confirmation) | Side-doc |

## Decisions Still Needed

| # | Decision | Options | When |
|---|----------|---------|------|
| D5 | Node discovery | mDNS / config file / central registry | During Phase 3 planning |
| D6 | Target CUDA version | 12.x minimum | When hardware is set up |
| D7 | License | Apache 2.0 confirmed? | Before GitHub repo creation |
| - | Feature name for non-proprietary GPU DMA | TBD | Now |

---

## What We Learned (Research Summary)

### R1: The Competitive Landscape
- **30+ projects surveyed.** No open-source solution fills our gap.
- Closest: SCUDA (early), gVirtuS (GPL, CUDA 12.6), Cricket (Driver API, GPLv3)
- Commercial: Bitfusion (uncertain), OrionX (China), TensorFusion (partial open source)
- **Market gap confirmed.** OutterLink has no direct open-source competitor.

### R2: SoftRoCE Is Dead
- ~30% of TCP throughput (WORSE, not better)
- Can't touch GPU memory (no DMA engine)
- Unstable (crashes at ~1000 connections)
- **Verdict: Use TCP instead. Simpler, faster, battle-tested.**

### R3: CUDA Interception Is Solved
- **Driver API** interception (not Runtime API) catches everything
- LD_PRELOAD + custom `dlsym()` + `cuGetProcAddress` hook
- HAMi-core: 222 functions mapped, production-proven, version-indexed
- Also intercept NVML to fake GPU properties
- **Implementation pattern is clear. No research risk.**

### R4: ConnectX-5 and Transport Stack
- ConnectX-5 = fast NIC with DMA engine (essential for both networking AND direct GPU path)
- GPUDirect RDMA blocked on GeForce (proprietary driver restriction)
- TCP + io_uring: up to ~10 GB/s on 100GbE (Phase 1)
- UCX: auto-negotiates best transport, CUDA-aware (Phase 2)
- 4x 100GbE bonded: ~50 GB/s (exceeds PCIe 4.0 x16)
- Rust crates: `tokio`, `io-uring`, `cudarc`, `ucx-sys`, `sideway`

### R5: NVIDIA's GPUDirect Restriction
- Blocked in proprietary `nvidia_p2p_get_pages()` (checks GPU product class)
- Multi-layer: driver check + possibly GSP firmware
- tinygrad achieved P2P (GPU-to-GPU) on RTX 4090, NOT RDMA (GPU-to-NIC)
- Low legal risk for personal use, higher for distribution
- **Not a dead end - see R7 for bypass.**

### R6: NVLink Cross-PC
- Not physically feasible with existing technology (signaling distance, no NVLink cables)
- NVSwitch not sold separately
- RTX 3090/3090 Ti NVLink: 112.5 GB/s aggregate (4 x4 links)
- **Strategy: NVLink for local pairs, ConnectX-5 for cross-PC. Complementary, not competing.**

### R7: Non-Proprietary GPU DMA (THE BREAKTHROUGH)
- PCIe BAR1 is standard - any device can read/write to it
- GPU MMU page tables must map BAR1 -> VRAM (tinygrad patches solve this)
- ReBAR exposes full 24GB VRAM through BAR1
- **Path: tinygrad BAR1 patches + custom kernel module = direct NIC-to-GPU RDMA**
- nouveau upstream RDMA patches in review (NVIDIA engineer, claims 10x improvement)
- If achieved: ~2us latency (vs ~12us staged), zero CPU, works on ALL GPUs
- **This is OutterLink's killer differentiator.**

---

## The Full Data Path (Both Modes)

### Mode 1: Host-Staged (Phase 1 - Works Today)

```
Local GPU VRAM
  |
  | cudaMemcpyAsync (~5us, limited by PCIe)
  v
Pinned Host Memory (cudaHostAlloc)
  |
  | ConnectX-5 RDMA WRITE or TCP send (~2us network)
  v
Remote Pinned Host Memory
  |
  | cudaMemcpyAsync (~5us, limited by PCIe)
  v
Remote GPU VRAM

Total latency: ~12us
Bandwidth ceiling: min(PCIe, network) = ~12.5 GB/s per 100GbE link
CPU involvement: YES (cudaMemcpy scheduling)
```

### Mode 2: Direct BAR1 RDMA (Phase 2 - OutterLink's Killer Feature)

```
Local GPU VRAM (mapped through BAR1 via tinygrad patches + ReBAR)
  |
  | ConnectX-5 DMA engine reads BAR1 directly (~2us)
  v
Wire (100GbE RDMA)
  |
  | ConnectX-5 DMA engine writes remote GPU BAR1 directly (~2us)
  v
Remote GPU VRAM

Total latency: ~2us
Bandwidth ceiling: min(PCIe, network) = ~12.5 GB/s per 100GbE link
CPU involvement: NONE (pure DMA engine to DMA engine)
```

---

## Hardware Summary

### What We Have

| Component | PC1 (MS-01 Ultra) | PC2 (Threadripper 9960X) |
|-----------|-------------------|--------------------------|
| CPU | TBD | Threadripper 9960X (24c Zen 5) |
| RAM | 256 GB (dual channel) | 256 GB ECC DDR5 4800 |
| PCIe | 5.0 | 5.0 (TRX50) |
| NICs | 2x ConnectX-5 100GbE dual-port | 2x ConnectX-5 100GbE dual-port |
| GPUs | From pool | From pool |
| OS | TBD (Linux) | TBD (Linux) |

### GPU Pool

| GPU | Count | VRAM | NVLink |
|-----|-------|------|--------|
| RTX 3090 Ti | 2 (+ buying 2 more) | 24 GB each | Yes, 112.5 GB/s |
| RTX 5090 | 2 | 32 GB each | TBD |

### Network

- 4x ConnectX-5 100GbE cards (2 per PC), each dual-port = 8 total 100GbE ports
- Bonding options: 1x (12.5 GB/s) to 4x (50 GB/s) to 8x (100 GB/s theoretical)

---

## Bandwidth Reality Table (Final, Corrected)

| Connection | Bandwidth | Role in OutterLink |
|-----------|-----------|-------------------|
| No connection (today) | **0 GB/s** | **The problem we solve** |
| 1 GbE | ~0.125 GB/s | Minimum viable (models that didn't fit now fit) |
| 10 GbE | ~1.2 GB/s | Good for inference |
| 25 GbE | ~3.1 GB/s | Good for training |
| **1x 100 GbE** | **~12.5 GB/s** | **Our single-link speed** |
| **4x 100 GbE bonded** | **~50 GB/s** | **Our max bonded speed** |
| PCIe 4.0 x4 riser | ~8 GB/s | Local GPU on riser (network can beat this!) |
| PCIe 4.0 x16 | ~32 GB/s | Local GPU direct slot |
| PCIe 5.0 x16 | ~64 GB/s | Our motherboards' native speed |
| NVLink (3090 Ti pair) | ~112.5 GB/s | Local GPU pair within each PC |

---

## Technology Stack (Decided)

| Layer | Technology | Crate / Tool |
|-------|-----------|-------------|
| Language | Rust | - |
| CUDA interception | LD_PRELOAD .so (C/Rust FFI) | `cudarc`, raw CUDA FFI |
| NVML interception | LD_PRELOAD .so | Raw FFI |
| Async networking | tokio + io_uring | `tokio`, `io-uring` |
| RDMA (Phase 2) | UCX or libibverbs | `ucx-sys`, `sideway` |
| GPU DMA (Phase 2) | Custom kernel module + tinygrad patches | C kernel module |
| Serialization | TBD (protobuf? custom binary?) | TBD |
| Node discovery | TBD | TBD |

---

## Project Phases (High Level)

| Phase | What | Milestone |
|-------|------|-----------|
| **0: Setup** | GitHub repo, dev environment, CI | Repo exists, builds, has tests |
| **1: PoC** | Device query + memory alloc over TCP | Remote GPU appears to CUDA app |
| **2: Core** | Memory transfers (H2D, D2H) + kernel launch | CUDA kernel runs on remote GPU |
| **3: Completeness** | Streams, events, NVML, multi-GPU | Real apps (PyTorch) work transparently |
| **4: Performance** | io_uring, call batching, UCX/RDMA | Benchmark-validated performance |
| **5: Direct GPU** | BAR1 RDMA via tinygrad patches + custom module | Zero-copy NIC-to-GPU on GeForce |
| **6: Scale** | Multi-node, GPU pooling, smart scheduling | 3+ PCs, NVLink-aware scheduling |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CUDA version changes break interception | HIGH | Version-indexed function table (HAMi-core pattern) |
| BAR1 RDMA doesn't work on our hardware | MEDIUM | Phase 1 (host-staged) works regardless |
| PCIe topology blocks P2P on our motherboards | MEDIUM | Check with `lspci -tv` before building |
| Performance too slow for training workloads | LOW | Memory pooling for inference is valuable even at lower bandwidth |
| NVIDIA patches their driver to block tinygrad approach | LOW | Multiple paths (P2PDMA, nouveau patches, DMA-BUF) |
| Rust FFI complexity with CUDA/kernel modules | MEDIUM | C for interception .so, Rust for server/client logic |

## Related Documents

- [Project Vision](../../docs/architecture/00-project-vision.md)
- [R1](R1-existing-projects.md) | [R2](R2-softroce-rdma.md) | [R3](R3-cuda-interception.md) | [R4](R4-connectx5-transport-stack.md) | [R5](R5-gpudirect-geforce-restriction.md) | [R6](R6-nvlink-cross-pc.md) | [R7](R7-non-proprietary-gpu-dma.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)
- [License Comparison](../../side-docs/notes/01-license-comparison.md)
