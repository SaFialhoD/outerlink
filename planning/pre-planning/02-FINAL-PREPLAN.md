# FINAL PRE-PLAN: Complete Map of Everything

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** COMPLETE - Ready for Planning Phase

## Purpose

This is THE exhaustive map. Every research topic, every decision, every plan, every dependency, every risk, every hardware task. Nothing should be missing. If it's not here, it doesn't exist.

---

## SECTION A: RESEARCH (What We Needed to Know)

### Completed Research

| # | Topic | Document | Key Finding |
|---|-------|----------|-------------|
| R1 | Existing GPU sharing projects | [R1](../research/R1-existing-projects.md) | 30+ projects surveyed. No open-source competitor fills our gap. |
| R2 | SoftRoCE / rdma_rxe | [R2](../research/R2-softroce-rdma.md) | Dead end. Slower than TCP, can't touch GPU memory, unstable. |
| R3 | CUDA interception strategies | [R3](../research/R3-cuda-interception.md) | Solved. Driver API + LD_PRELOAD + cuGetProcAddress. HAMi-core pattern (222 funcs). |
| R4 | ConnectX-5 + transport stack | [R4](../research/R4-connectx5-transport-stack.md) | TCP+io_uring Phase 1, UCX Phase 2. ConnectX-5 DMA engine enables OpenDMA. |
| R5 | GPUDirect on GeForce | [R5](../research/R5-gpudirect-geforce-restriction.md) | Blocked by NVIDIA in proprietary driver. Software restriction, not hardware. |
| R6 | NVLink cross-PC | [R6](../research/R6-nvlink-cross-pc.md) | Not feasible. NVLink for local pairs, ConnectX-5 for cross-PC. |
| R7 | Non-proprietary GPU DMA | [R7](../research/R7-non-proprietary-gpu-dma.md) | VIABLE. PCIe BAR1 + tinygrad patches + custom RDMA module = OpenDMA. |
| - | Consolidation | [Consolidation](../research/CONSOLIDATION-all-research.md) | All findings unified. |

### Remaining Research (To Be Done During Planning)

| # | Topic | Priority | When | Why |
|---|-------|----------|------|-----|
| R8 | Serialization format comparison | MEDIUM | During P6 planning | Need to choose protobuf vs flatbuffers vs custom for RPC |
| R9 | Benchmarking tools and methodology | MEDIUM | During P12 planning | Need to measure performance correctly |
| R10 | CUDA call patterns in ML workloads | LOW | During P8 planning | Optimize batching strategy |
| R11 | Rust CUDA FFI options (`cudarc` vs raw) | MEDIUM | During P4 planning | Choose FFI approach for interception layer |
| R12 | tinygrad patches compatibility with our CUDA version | HIGH | During P9 planning | Verify OpenDMA path works on our hardware |
| R13 | Linux distro comparison for GPU/RDMA workloads | LOW | During P2 planning | Choose Ubuntu vs Fedora vs Arch |

---

## SECTION B: DECISIONS (What We Chose)

### Locked Decisions (ADRs Written)

| # | Decision | Answer | ADR |
|---|----------|--------|-----|
| D1 | Programming language | Rust (C for kernel module + interception .so) | - |
| D2 | CUDA interception strategy | Driver API + LD_PRELOAD + cuGetProcAddress | - |
| D3a | Transport Phase 1 | TCP + io_uring + CUDA pinned memory | - |
| D3b | Transport Phase 2 | UCX (auto-negotiates RDMA vs TCP) | - |
| D4a | GPU DMA Phase 1 | Host-staged (cudaMemcpy through pinned host) | - |
| D4b | GPU DMA Phase 2 | OpenDMA (direct BAR1 RDMA, non-proprietary) | [ADR-002](../../docs/decisions/ADR-002-opendma-naming.md) |
| D5 | Build approach | Clean-room Rust, not fork | - |
| D7 | License | Apache 2.0 | [ADR-001](../../docs/decisions/ADR-001-license.md) |
| D10 | GPU DMA feature name | OpenDMA | [ADR-002](../../docs/decisions/ADR-002-opendma-naming.md) |

### Decisions to Make During Planning

| # | Decision | Options | When | Depends On |
|---|----------|---------|------|-----------|
| D6 | Target CUDA version | 12.x minimum (which exact?) | P2 planning | Hardware setup |
| D8 | Node discovery mechanism | mDNS / config file / central registry | P7 planning | Simplicity vs scalability |
| D9 | Serialization format | protobuf / flatbuffers / custom binary | P6 planning | R8 research |
| D11 | GPU placement strategy | Which GPUs in which PC | Hardware setup | Physical constraints |
| D12 | Crate structure | Monorepo workspace layout | P4 planning | R11 research |
| D13 | Error handling strategy | How to report network errors as CUDA errors | P5 planning | CUDA error code mapping |
| D14 | Handle translation approach | HashMap vs array vs arena allocator | P5 planning | Performance requirements |
| D15 | Async vs sync interception | Block CUDA calls or return immediately | P5 planning | Latency requirements |
| D16 | OpenDMA kernel module license | GPL (required for kernel) vs dual MIT/GPL | P9 planning | Kernel module licensing rules |
| D17 | Multi-GPU device numbering | How remote GPUs are numbered locally | P5 planning | User experience |

---

## SECTION C: PLANS (What We Need to Write)

### Planning Order (Critical Path)

```
PHASE 0: INFRASTRUCTURE (must be done first)
├── P1:  GitHub Repository Setup           <- Do FIRST (needs D7 license ✓)
├── P2:  Development Environment           <- Do SECOND
├── P3:  CI/CD Pipeline                    <- Can parallel with P4
└── P13: Documentation Standards           <- Can parallel with P1

PHASE 0.5: PROJECT SKELETON
└── P4:  Rust Workspace + Crate Structure  <- Needs P1, P2

PHASE 1: PROOF OF CONCEPT
├── P5:  PoC Plan                          <- Needs P4
├── P11: Testing Strategy                  <- Needs P4
└── P12: Benchmarking Plan                 <- Needs P5

PHASE 2-6: IMPLEMENTATION (planned sequentially, each depends on prior)
├── P6:  Core Transport
├── P7:  CUDA Completeness
├── P8:  Performance Optimization
├── P9:  OpenDMA (BAR1 RDMA)
└── P10: Multi-Node + Scaling
```

### Detailed Plan Requirements

Each plan document must contain:

| Section | Description |
|---------|-------------|
| **Goal** | One sentence: what does this phase achieve? |
| **Milestone** | Specific, testable acceptance criteria |
| **Prerequisites** | What must be done/decided before this phase starts |
| **Components** | What gets built, file by file |
| **Interface contracts** | How this phase's output connects to next phase |
| **Test plan** | How we verify this phase works |
| **Risks** | What could go wrong and how we handle it |
| **Estimated scope** | Number of files, crates, functions (not time) |

### Plan Summaries

#### P1: GitHub Repository Setup
- Create repo, configure settings, branch strategy
- README, LICENSE (Apache 2.0), CONTRIBUTING guide
- Issue templates, PR templates, labels
- GitHub Actions skeleton
- .gitignore for Rust + CUDA + kernel modules

#### P2: Development Environment
- Linux distro recommendation (R13)
- NVIDIA driver + CUDA toolkit installation guide
- ConnectX-5 + MLNX_OFED installation guide
- Rust toolchain setup
- BIOS settings (ReBAR, Above 4G, IOMMU)
- Verification checklist (nvidia-smi, ibv_devices, iperf3, etc.)

#### P3: CI/CD Pipeline
- GitHub Actions for Rust build + clippy + tests
- CUDA mock/stub for CI (no GPU in CI)
- Linting, formatting (rustfmt, clippy)
- Doc generation

#### P4: Project Skeleton
- Rust workspace layout:
  - `outterlink-client` (the LD_PRELOAD .so)
  - `outterlink-server` (daemon on GPU nodes)
  - `outterlink-common` (shared types, protocol, transport trait)
  - `outterlink-cli` (management tool)
  - `opendma-module` (kernel module, C)
- Build system (cargo + make for C/kernel parts)
- FFI layer design (Rust <-> CUDA C API)
- Transport trait definition

#### P5: Phase 1 PoC
- Intercept: `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem`
- Server: receive queries, execute on real GPU, return results
- Client: serialize query, send over TCP, deserialize response
- Transport: basic TCP (tokio)
- Milestone: `nvidia-smi`-like query shows remote GPU properties

#### P6: Phase 2 Core Transport
- Intercept: `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`
- Handle translation tables (CUdeviceptr, CUmodule, CUfunction, CUcontext)
- CUDA pinned memory for network transfers
- Milestone: CUDA kernel runs on remote GPU, results come back correct

#### P7: Phase 3 CUDA Completeness
- Streams, events, synchronization
- NVML interception (fake GPU properties)
- Multi-GPU device enumeration
- Milestone: PyTorch `torch.cuda.is_available()` sees remote GPUs, simple inference works

#### P8: Phase 4 Performance
- io_uring zero-copy send/recv
- Call batching (lazy updates pattern from vCUDA)
- UCX transport backend
- RDMA support (ConnectX-5 host-staged)
- Milestone: benchmark shows X GB/s transfer, Y us latency (targets TBD)

#### P9: Phase 5 OpenDMA
- Apply tinygrad BAR1 patches to nvidia-open modules
- Verify full VRAM accessible through BAR1 (with ReBAR)
- Write custom kernel module: register BAR1 with RDMA subsystem
- Test: ConnectX-5 RDMA directly to GPU VRAM
- Benchmark: compare OpenDMA vs host-staged
- Milestone: zero-copy NIC-to-GPU VRAM transfer on GeForce

#### P10: Phase 6 Multi-Node + Scaling
- Node discovery (D8)
- 3+ PC support
- GPU pooling (unified view of all GPUs across all nodes)
- NVLink-aware scheduling (prefer local GPU pairs for bandwidth-heavy ops)
- System RAM pooling
- Milestone: 3 PCs pooling all GPUs, ML workload runs across them

#### P11: Testing Strategy
- Unit tests for handle translation, serialization, transport
- Integration tests with CUDA mock/stub
- End-to-end tests on real hardware
- Regression tests for CUDA version compatibility

#### P12: Benchmarking Plan
- Metrics: latency (per-call), throughput (GB/s), overhead (%)
- Tools: custom benchmark suite, perftest-rdma, NCCL tests
- Baselines: local GPU vs remote GPU vs host-staged vs OpenDMA
- Workloads: microbenchmark (memcpy), GEMM, LLM inference

#### P13: Documentation Standards
- Already defined in `.claude/rules/documentation-framework.md`
- Add: API documentation strategy, user guide outline

---

## SECTION D: HARDWARE SETUP (Pedro's Physical Tasks)

### Critical Path (Must Be Done Before Coding)

| # | Task | Status | Priority | Notes |
|---|------|--------|----------|-------|
| H1 | Choose and install Linux distro on both PCs | TODO | HIGH | Recommend Ubuntu 22.04/24.04 LTS |
| H2 | Install NVIDIA drivers (550+) | TODO | HIGH | After Linux install |
| H3 | Install CUDA toolkit (12.x) | TODO | HIGH | After driver install |
| H4 | Install ConnectX-5 cards (2 per PC) | TODO | HIGH | Physical installation |
| H5 | Install MLNX_OFED | TODO | HIGH | BEFORE NVIDIA driver if possible |
| H6 | Connect PCs with DAC cables | TODO | HIGH | Need QSFP28 DAC for 100GbE |
| H7 | Install Rust toolchain (`rustup`) | TODO | HIGH | After Linux install |
| H8 | Buy 2 more 3090 Ti | TODO | MEDIUM | Pedro's plan |

### Important But Not Blocking

| # | Task | Status | Priority | Notes |
|---|------|--------|----------|-------|
| H9 | Enable ReBAR in BIOS (both PCs) | TODO | HIGH | Required for OpenDMA |
| H10 | Enable Above 4G Decoding | TODO | HIGH | Required for ReBAR |
| H11 | Configure IOMMU (disable or passthrough) | TODO | HIGH | Required for P2P DMA |
| H12 | Set up open air riser mount for 3090 Ti | TODO | MEDIUM | For NVLink bridge fit |
| H13 | Install NVLink bridges | TODO | MEDIUM | After riser setup |

### Verification (After Physical Setup)

| # | Test | Command | Expected |
|---|------|---------|----------|
| V1 | GPUs visible | `nvidia-smi` | All GPUs listed |
| V2 | CUDA works | `cuda-samples deviceQuery` | PASS |
| V3 | NVLink active | `nvidia-smi topo -m` | NV# links shown |
| V4 | ConnectX-5 visible | `ibv_devices` | mlx5_0, mlx5_1, etc. |
| V5 | RDMA works | `rping -s` / `rping -c` | Connection success |
| V6 | Network bandwidth | `iperf3 -c <IP>` | ~12 GB/s per link |
| V7 | PCIe topology | `lspci -tv` | GPU + NIC same root complex |
| V8 | ReBAR active | `nvidia-smi -q | grep BAR1` | BAR1 = 24576 MB |
| V9 | Rust builds | `cargo --version` | Version shown |

---

## SECTION E: RISKS AND MITIGATIONS

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|-----------|
| K1 | CUDA version changes break interception | HIGH | MEDIUM | Version-indexed function table, CI tests |
| K2 | OpenDMA BAR1 path doesn't work on our hardware | HIGH | MEDIUM | Host-staged (Phase 1) works regardless |
| K3 | PCIe topology blocks P2P | MEDIUM | LOW | Check `lspci -tv` early, rearrange cards if needed |
| K4 | Rust FFI with CUDA is painful | MEDIUM | MEDIUM | Use C for interception .so, Rust for server |
| K5 | NVIDIA patches tinygrad approach | LOW | LOW | Multiple paths (P2PDMA, nouveau, DMA-BUF) |
| K6 | Performance too slow for training | LOW | MEDIUM | Memory pooling for inference is valuable even slow |
| K7 | ConnectX-5 bonding is complex | LOW | LOW | Start with single link, add bonding later |
| K8 | cuGetProcAddress changes in future CUDA | MEDIUM | LOW | Monitor CUDA release notes, version-gate |
| K9 | GSP firmware blocks OpenDMA | HIGH | UNKNOWN | Only knowable by testing on real hardware |
| K10 | Kernel module maintenance burden | MEDIUM | HIGH | Minimize kernel code, maximize userspace |

---

## SECTION F: PROJECT STRUCTURE OVERVIEW

```
outterlink/
├── CLAUDE.md                              # Project overview
├── LICENSE                                # Apache 2.0
├── README.md                              # Project README (create during P1)
├── Cargo.toml                             # Workspace root
├── .github/                               # GitHub Actions, templates
├── .claude/rules/                         # Claude rules for documentation
│
├── planning/                              # ALL planning docs
│   ├── pre-planning/
│   │   ├── 00-master-preplan.md          # Original pre-plan
│   │   ├── 01-hardware-inventory.md      # Hardware details
│   │   └── 02-FINAL-PREPLAN.md           # THIS DOCUMENT
│   ├── phases/                            # Phase plans (P1-P13)
│   │   └── (to be created during planning)
│   └── research/                          # All research docs (R1-R7)
│
├── docs/                                  # Project documentation
│   ├── architecture/
│   │   └── 00-project-vision.md
│   ├── decisions/                         # ADRs
│   │   ├── ADR-001-license.md
│   │   └── ADR-002-opendma-naming.md
│   ├── guides/                            # Setup and usage guides
│   └── specs/                             # Technical specifications
│
├── side-docs/                             # Supporting docs
│   ├── references/
│   ├── notes/
│   └── diagrams/
│
├── crates/                                # Rust source (future)
│   ├── outterlink-client/                 # LD_PRELOAD interception library
│   ├── outterlink-server/                 # GPU node daemon
│   ├── outterlink-common/                 # Shared protocol, types, transport
│   └── outterlink-cli/                    # Management CLI
│
├── opendma/                               # OpenDMA kernel module (future, C)
│   ├── module/                            # Kernel module source
│   └── patches/                           # tinygrad-based nvidia-open patches
│
└── benchmarks/                            # Benchmark suite (future)
```

---

## SECTION G: NEXT STEPS (Transition to Planning)

This pre-plan is complete. The transition to planning follows this sequence:

```
PRE-PLAN: COMPLETE (this document)
    |
    v
PLAN P1: GitHub Repository Setup
    - Repo name, description, topics
    - File structure for initial commit
    - Branch strategy (main + dev)
    - Issue labels and templates
    - GitHub Actions skeleton
    - README content
    |
    v
PLAN P2: Development Environment
    - Linux distro + installation
    - NVIDIA driver + CUDA setup
    - ConnectX-5 + MLNX_OFED setup
    - Rust toolchain
    - BIOS configuration guide
    - Verification checklist
    |
    v
PLAN P4: Project Skeleton
    - Cargo workspace layout
    - Crate structure and dependencies
    - FFI layer design
    - Transport trait interface
    - Build system
    |
    v
PLAN P5: Phase 1 PoC
    - Detailed implementation plan
    - Function-by-function interception list
    - Protocol specification
    - Test plan
    |
    v
EXECUTE: First code
```

**When Pedro says "go", we start with Plan P1.**

## Related Documents

- [Research Consolidation](../research/CONSOLIDATION-all-research.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Hardware Inventory](01-hardware-inventory.md)
- [ADR-001: License](../../docs/decisions/ADR-001-license.md)
- [ADR-002: OpenDMA](../../docs/decisions/ADR-002-opendma-naming.md)
