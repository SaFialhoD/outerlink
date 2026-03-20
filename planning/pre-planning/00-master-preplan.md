# Pre-Planning Master: What We Need to Plan

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Updated with research results

## Purpose

Map out everything that requires a detailed plan before we write code.

---

## 1. Research: COMPLETE

All high-priority research is done. See [Consolidation](../research/CONSOLIDATION-all-research.md).

| # | Topic | Status | Key Finding |
|---|-------|--------|-------------|
| R1 | Existing projects | DONE | No competitor fills our gap |
| R2 | SoftRoCE | DONE | Dead end - TCP is faster |
| R3 | CUDA interception | DONE | Driver API + LD_PRELOAD, solved |
| R4 | ConnectX-5 + transport | DONE | TCP+io_uring Phase 1, UCX Phase 2 |
| R5 | GPUDirect on GeForce | DONE | Blocked by NVIDIA in software only |
| R6 | NVLink cross-PC | DONE | Not feasible, ConnectX-5 covers gap |
| R7 | Non-proprietary GPU DMA | DONE | VIABLE via PCIe BAR1 + tinygrad patches |

### Remaining Research (Lower Priority, Can Be Done During Planning)

| # | Topic | Priority | When |
|---|-------|----------|------|
| R8 | Serialization formats (protobuf vs custom binary) | MEDIUM | During Phase 2 planning |
| R9 | Benchmarking methodology and tools | MEDIUM | During Phase 1 planning |
| R10 | CUDA call patterns in real ML workloads | LOW | During Phase 4 planning |

---

## 2. Decisions: STATUS

### Made

| # | Decision | Answer | Confidence |
|---|----------|--------|-----------|
| D1 | Language | Rust (C for kernel module + interception .so) | FINAL |
| D2 | CUDA interception | Driver API + LD_PRELOAD + cuGetProcAddress | HIGH |
| D3 | Transport Phase 1 | TCP + io_uring + CUDA pinned memory | HIGH |
| D3 | Transport Phase 2 | UCX (auto RDMA/TCP) | HIGH |
| D4 | GPU DMA Phase 1 | Host-staged (cudaMemcpy through pinned host) | HIGH |
| D4 | GPU DMA Phase 2 | Direct BAR1 RDMA (non-proprietary) | HIGH (concept), MEDIUM (execution) |
| D5 | Build approach | Clean-room Rust, not fork | FINAL |

### Pending (Need ADRs During Planning)

| # | Decision | Options | Depends On |
|---|----------|---------|-----------|
| D6 | Target CUDA version | 12.x minimum | Hardware setup |
| D7 | License | Apache 2.0 (recommended) | Pedro confirmation |
| D8 | Node discovery mechanism | mDNS / config file / central registry | Simplicity preference |
| D9 | Serialization format | protobuf / flatbuffers / custom binary | Performance requirements |
| D10 | Name for non-proprietary GPU DMA feature | TBD | Branding discussion |
| D11 | GPU placement strategy | Which GPUs in which PC | Physical setup |

---

## 3. Plans Needed

Each becomes a detailed document in `/planning/phases/`.

### Infrastructure Plans

| # | Plan | Depends On | Priority |
|---|------|-----------|----------|
| P1 | GitHub Repository Setup | D7 (license) | HIGH - Do first |
| P2 | Development Environment | D1, D6 | HIGH - Do second |
| P3 | CI/CD Pipeline | P1 | MEDIUM |

### Implementation Plans

| # | Plan | Depends On | Description |
|---|------|-----------|-------------|
| P4 | Phase 0: Project Skeleton | P1, P2 | Rust workspace, crate structure, build system |
| P5 | Phase 1: PoC | P4 | Device query + memory alloc/free over TCP |
| P6 | Phase 2: Core Transport | P5 | Memory transfers + kernel launch |
| P7 | Phase 3: CUDA Completeness | P6 | Streams, events, NVML, real apps work |
| P8 | Phase 4: Performance | P7 | io_uring, batching, UCX/RDMA |
| P9 | Phase 5: Direct GPU (BAR1 RDMA) | P8 | Non-proprietary GPU DMA - killer feature |
| P10 | Phase 6: Multi-Node + Scaling | P7 | 3+ PCs, GPU pooling, NVLink-aware scheduling |

### Quality Plans

| # | Plan | Depends On |
|---|------|-----------|
| P11 | Testing Strategy | P4 |
| P12 | Benchmarking Plan | P5 |
| P13 | Documentation Standards | P1 |

---

## 4. Hardware Setup Checklist

Things Pedro needs to do physically before we can code:

| Task | Status | Notes |
|------|--------|-------|
| Install Linux on both PCs | TODO | Distro TBD |
| Install NVIDIA drivers + CUDA | TODO | Version TBD |
| Install ConnectX-5 cards | TODO | 2 per PC |
| Connect PCs with DAC/fiber cables | TODO | Need SFP modules |
| Enable ReBAR in BIOS (both PCs) | TODO | Required for BAR1 full VRAM mapping |
| Enable Above 4G Decoding | TODO | Required for ReBAR |
| Configure IOMMU (disable or passthrough) | TODO | Required for P2P DMA |
| Verify PCIe topology: `lspci -tv` | TODO | GPU + NIC same root complex? |
| Set up NVLink bridges on 3090 Ti pairs | TODO | Open air riser setup |
| Test NVLink: `nvidia-smi topo -m` | TODO | Verify NVLink active |
| Test network: `iperf3` between PCs | TODO | Measure actual bandwidth |
| Install Rust toolchain | TODO | `rustup` |
| Install MLNX_OFED | TODO | For ConnectX-5 RDMA |

---

## 5. Order of Operations

```
CURRENT STEP: Consolidation + Pre-Planning (this document)
                |
                v
STEP 1: Pre-plan complete, decisions locked
                |
                v
STEP 2: Plan P1 - GitHub repo setup (detailed plan)
                |
                v
STEP 3: Plan P2 - Dev environment (detailed plan)
                |
                v
STEP 4: Plan P4 - Project skeleton (crate structure, workspace)
                |
                v
STEP 5: Plan P5 - Phase 1 PoC (detailed implementation plan)
                |
          [HARDWARE SETUP HAPPENS IN PARALLEL]
                |
                v
STEP 6: Execute Phase 1 PoC (FIRST CODE!)
                |
                v
STEP 7: Plan and execute subsequent phases based on PoC results
```

---

## 6. What OutterLink's Unique Value Is

No other project has all of these:

| Feature | OutterLink | SCUDA | gVirtuS | Cricket | TensorFusion |
|---------|-----------|-------|---------|---------|-------------|
| Open source | Apache 2.0 | Yes | GPL | GPLv3 | Partial |
| Modern CUDA (12.x) | Yes | Yes | Yes | Yes | Yes |
| Driver API interception | Yes | No | Partial | Yes | ? |
| Pluggable transport | Yes | No | Yes | ? | ? |
| RDMA support | Yes (UCX) | No | Yes | ? | Yes |
| **Non-proprietary GPU DMA** | **YES** | No | No | No | No |
| GPU + RAM pooling | Yes | No | No | No | Partial |
| Written in Rust | Yes | No | No | No | Yes |
| NVLink-aware scheduling | Yes | No | No | No | ? |

**The killer differentiator:** Non-proprietary GPU DMA via PCIe BAR1. No other project even attempts this.

## Related Documents

- [Consolidation](../research/CONSOLIDATION-all-research.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Hardware Inventory](01-hardware-inventory.md)
- [License Comparison](../../side-docs/notes/01-license-comparison.md)
