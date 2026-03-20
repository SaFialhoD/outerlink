# Research Synthesis: Initial Findings

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete

## Purpose

Combine findings from R1 (existing projects), R2 (SoftRoCE), and R3 (CUDA interception) into actionable conclusions for OutterLink's architecture.

---

## The Big Picture

After surveying 30+ projects, reading academic papers, and analyzing transport and interception options, here's what we know:

### 1. The Gap Is Real and Clear

No open-source project delivers: **modern CUDA support + Driver API interception + pluggable transport + GPU memory pooling + standard Ethernet**. The pieces exist separately but nobody has assembled them correctly.

### 2. SoftRoCE Is a Dead End (R2)

Our original assumption was wrong. SoftRoCE is:
- **Slower than TCP** (~30% of TCP throughput)
- **Can't touch GPU memory** (no DMA engine)
- **Unstable** (crashes at ~1000 connections)
- **Being deprecated** (removed in RHEL 10)

**But this is GOOD NEWS.** We don't need special RDMA software. TCP is faster, simpler, and battle-tested. The GPU memory path (VRAM -> pinned host -> network -> pinned host -> VRAM) is the same regardless of transport.

### 3. The Interception Strategy Is Solved (R3)

The industry has converged on a clear approach:
1. **LD_PRELOAD** to inject our library
2. **Hook `dlsym()` + `cuGetProcAddress`** for CUDA 11.3+ compatibility
3. **Intercept at Driver API** (`libcuda.so`) - catches everything
4. **Also hook NVML** (`libnvidia-ml.so`) - fake GPU properties
5. **HAMi-core** has a production-proven implementation of this with 222 functions

### 4. Existing Projects to Learn From (R1)

| Project | What to Take |
|---------|-------------|
| **HAMi-core** | Interception architecture (dlsym + cuGetProcAddress + function table) |
| **SCUDA** | Code-generated RPC stubs from CUDA headers |
| **Cricket** | Driver API handle translation, state management |
| **gVirtuS** | Pluggable transport layer (communicator abstraction) |
| **vCUDA** | Call batching ("lazy updates") for performance |

---

## Architecture Decision: What OutterLink Should Be

### Core Architecture

```
┌─────────────────────────────────────────────┐
│                APPLICATION                   │
│           (unmodified CUDA app)              │
└──────────────────┬──────────────────────────┘
                   │ LD_PRELOAD
┌──────────────────▼──────────────────────────┐
│          OUTTERLINK CLIENT (.so)             │
│                                              │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ CUDA Driver  │  │ NVML Hooks           │  │
│  │ API Hooks    │  │ (fake GPU properties)│  │
│  │ (222+ funcs) │  └──────────────────────┘  │
│  └──────┬──────┘                             │
│         │                                    │
│  ┌──────▼──────┐  ┌──────────────────────┐  │
│  │ Handle      │  │ Call Batcher          │  │
│  │ Translation │  │ (lazy updates)        │  │
│  │ Tables      │  └──────────────────────┘  │
│  └──────┬──────┘                             │
│         │                                    │
│  ┌──────▼──────────────────────────────────┐ │
│  │ Transport Abstraction Layer             │ │
│  │ ┌─────────┐ ┌────────┐ ┌─────────────┐ │ │
│  │ │  TCP    │ │io_uring│ │ RDMA (later)│ │ │
│  │ └─────────┘ └────────┘ └─────────────┘ │ │
│  └─────────────────────────────────────────┘ │
└──────────────────┬───────────────────────────┘
                   │ Network
┌──────────────────▼──────────────────────────┐
│          OUTTERLINK SERVER (daemon)          │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ Request Handler                       │   │
│  │ (deserialize, dispatch, respond)      │   │
│  └──────────────┬───────────────────────┘   │
│                 │                             │
│  ┌──────────────▼───────────────────────┐   │
│  │ Real CUDA Driver API                  │   │
│  │ (actual GPU operations)               │   │
│  └──────────────┬───────────────────────┘   │
│                 │                             │
│  ┌──────────────▼───────────────────────┐   │
│  │ GPU Memory Manager                    │   │
│  │ (per-client contexts, VRAM tracking)  │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

### Transport Strategy

**Phase 1:** TCP with CUDA pinned memory (`cudaMallocHost`)
- Simpler, faster than SoftRoCE, works everywhere
- Pipeline: `cudaMemcpy GPU->pinned` -> `send()` -> `recv()` -> `cudaMemcpy pinned->GPU`

**Phase 2:** io_uring for lower syscall overhead
- Batched I/O submission, reduces context switches
- Drop-in improvement, same TCP underneath

**Phase 3 (optional):** Hardware RDMA support
- For users who have ConnectX cards
- GPUDirect RDMA for zero-copy (GPU memory -> network -> GPU memory)
- Transport abstraction layer makes this a plugin swap

### Language Decision Input

| Language | Pros | Cons |
|----------|------|------|
| **C** | Closest to CUDA API, easy LD_PRELOAD .so, matches HAMi-core/Cricket | Manual memory management, slower dev |
| **C++** | Same as C + abstractions, matches gVirtuS | More complexity |
| **Rust** | Memory safety, matches TensorFusion | FFI overhead for CUDA, steeper learning curve |

**Recommendation:** C for the interception layer (.so), with option for C++ or Rust for the server daemon.

### Implementation Priority

| Phase | What | Milestone |
|-------|------|-----------|
| **PoC** | Device query + memory alloc/free over TCP | "Remote GPU appears in `nvidia-smi`-like query" |
| **Phase 1** | + Memory transfers (H2D, D2H) | "Can copy data to/from remote GPU" |
| **Phase 2** | + Module loading + kernel launch | "Can run a CUDA kernel on remote GPU" |
| **Phase 3** | + Streams, events, synchronization | "Real CUDA apps work transparently" |
| **Phase 4** | + Multi-node, GPU pooling, RAM sharing | "Full OutterLink vision" |
| **Phase 5** | + io_uring, batching, performance optimization | "Production-grade performance" |

---

## Decisions Ready to Make

Based on research, these decisions have clear answers:

| # | Decision | Answer | Confidence |
|---|----------|--------|-----------|
| D2 | CUDA interception strategy | Driver API + LD_PRELOAD + cuGetProcAddress | HIGH |
| D3 | Network transport | TCP first, abstraction layer for upgrades | HIGH |

## Decisions Needing Discussion

| # | Decision | Options | Needs |
|---|----------|---------|-------|
| D1 | Programming language | C / C++ / Rust | User preference, team skills |
| D4 | Memory management model | Explicit mapping vs unified virtual addressing | More research |
| D5 | Node discovery | mDNS / config file / central registry | Simplicity vs scalability tradeoff |
| D6 | Target CUDA version | CUDA 12.x minimum | User's current setup |
| D7 | License | Apache 2.0 / GPL / MIT | Open source strategy |

## Remaining Research

| # | Topic | Why |
|---|-------|-----|
| R4 | GPUDirect architecture deep-dive | Design our transport abstraction to be compatible |
| R6 | Network topology | How nodes discover and talk |
| R7 | Distributed memory systems | Existing patterns for memory pooling |

## Related Documents

- [R1: Existing Projects](R1-existing-projects.md)
- [R2: SoftRoCE](R2-softroce-rdma.md)
- [R3: CUDA Interception](R3-cuda-interception.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Pre-Planning Master](../pre-planning/00-master-preplan.md)
