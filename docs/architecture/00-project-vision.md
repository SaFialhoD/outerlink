# OutterLink - Project Vision

**Created:** 2026-03-19
**Last Updated:** 2026-03-23
**Status:** Active Development

## Purpose

Define the core vision and technical thesis behind OutterLink.

## Vision

OutterLink makes multiple NVIDIA GPUs across multiple networked PCs work together as a unified resource pool - shared VRAM, shared compute, shared system RAM. Written in Rust. Built on CUDA Driver API interception, pluggable transport (TCP -> RDMA), and potentially a non-proprietary GPU DMA path that bypasses NVIDIA's artificial restrictions.

## The Problem

Right now, GPUs across separate PCs **cannot share anything**. Effective cross-machine GPU bandwidth: **zero**.

- A 70B parameter model needs ~140 GB VRAM. A single 3090 Ti has 24 GB. Game over.
- You own 4 GPUs across 2 PCs with 112 GB total VRAM. But no single process can use more than what's in one machine.
- NVIDIA's solution (DGX, NVLink, NVSwitch) costs $200K+. Their ConnectX + GPUDirect RDMA path is artificially locked to datacenter GPUs.
- There is NO open-source, production-grade solution for cross-PC GPU resource pooling.

## The Solution

**Any connection is infinitely better than no connection.** OutterLink creates the connection that doesn't exist today.

Even a 1 GB/s link between PCs transforms what's possible - models that didn't fit now fit, workloads that couldn't parallelize now parallelize. And with the right hardware, we're not talking about 1 GB/s:

## Bandwidth Reality Table

| Connection | Bandwidth | What It Means |
|-----------|-----------|---------------|
| No connection (today) | **0 GB/s** | **GPUs are isolated. Wasted potential.** |
| 1 GbE | ~0.125 GB/s | Slow, but models that didn't fit now fit |
| 10 GbE | ~1.2 GB/s | Viable for inference, batch processing |
| 25 GbE | ~3.1 GB/s | Matches PCIe x1 riser. Practical for training |
| **4x 100 GbE bonded** | **~50 GB/s** | **Exceeds PCIe 4.0 x16 (32 GB/s). Remote GPU faster than local riser!** |
| PCIe 4.0 x4 (riser) | ~8 GB/s | Common in multi-GPU rigs |
| PCIe 4.0 x16 (direct slot) | ~32 GB/s | Full speed, limited slots per motherboard |
| PCIe 5.0 x16 | ~64 GB/s | Next gen, available on TRX50 |
| NVLink (3090 Ti pair) | ~112.5 GB/s | Physical bridge, same machine only |

**Key takeaway:** The bandwidth gap between "no connection" and OutterLink is infinite. And with 100GbE ConnectX-5 cards, the network can actually exceed local PCIe bandwidth. A remote GPU over 4x100GbE is faster than a local GPU on a x4 riser.

## The Real Win: Memory Pooling

Bandwidth enables the connection. But **memory pooling** is the killer feature:

```
Without OutterLink:              With OutterLink:
PC1: 2x 24GB + 1x 32GB VRAM    Combined: 112 GB VRAM (2x3090Ti + 2x5090)
PC2: 2x 24GB + 1x 32GB VRAM    + 512 GB system RAM
Can't share. Period.            Any process sees ALL of it.

Max model: ~13B params          Max model: ~70B+ params
```

For memory-bound workloads (large model inference, batch processing, model parallelism), the total pool size matters more than the bandwidth between nodes.

## Target Setup

- **PC1:** Minisforum MS-01 Ultra, 256GB RAM, PCIe 5.0, 2x ConnectX-5 100GbE (dual port)
- **PC2:** Threadripper 9960X, TRX50, 256GB ECC DDR5, PCIe 5.0, 2x ConnectX-5 100GbE (dual port)
- **GPUs:** 2x RTX 3090 Ti (24 GB each) + 2x RTX 5090 (32 GB each)
- **Network:** Up to 4x 100GbE direct cables (~50 GB/s bonded)
- **Software:** OutterLink (Rust)
- **Result:** Unified pool of 112 GB VRAM + 512 GB system RAM

## What Success Looks Like

1. A process on PC1 can allocate and use GPU memory on PC2's GPUs
2. A process on PC2 can allocate and use GPU memory on PC1's GPUs
3. System RAM is pooled and accessible across nodes
4. Scaling is horizontal - add more PCs, get more resources
5. Works on consumer GPUs (GeForce) - no datacenter cards required
6. Transparent to CUDA applications - no code changes needed

## Technical Approach

1. **CUDA Driver API interception** via LD_PRELOAD + dlsym + cuGetProcAddress hooking (~250 hook table entries, following HAMi-core patterns)
2. **Pluggable transport layer** - TCP + Tokio (Phase 1), UCX/RDMA (Phase 2)
3. **Host-staged GPU memory transfers** - cudaMemcpy to/from pinned host memory for network transport
4. **Distributed memory manager** to track and coordinate VRAM + RAM across nodes
5. **Connection manager** for node discovery, health monitoring, and scaling
6. **NVML interception** to present remote GPUs as local devices
7. **(Research)** Non-proprietary GPU DMA via PCIe BAR1 direct access - bypassing NVIDIA's artificial GPUDirect restriction
8. **(Research)** NVLink as cross-PC bridge for 600 GB/s inter-node bandwidth

## Architecture

```
App -> LD_PRELOAD -> OutterLink Client (.so) -> Transport -> OutterLink Server -> Real GPU
```

---

## Current Status

**As of 2026-03-23 — Active Implementation (Phase 1)**

The core protocol layer and interception system are built and passing a full test suite. The project has moved well past planning into working code.

### What Is Built

| Component | Status | Notes |
|-----------|--------|-------|
| Wire protocol | Done | 22-byte big-endian header (magic `OLNK`, version, flags, request_id, msg_type, payload_len). Payload little-endian. |
| CUDA function coverage | Done | ~150 CUDA Driver API functions across all categories (see `docs/specs/cuda-function-coverage.md`) |
| LD_PRELOAD interposition | Done | `interpose.c` — ~250 hook table entries, hooks both `dlsym()` and `cuGetProcAddress()` |
| Client library | Done | Rust, `libouterlink_client.so`. HandleStore with 12 handle maps, retry + reconnect logic |
| Handle translation | Done | Synthetic local handles (type-tagged prefixes) bidirectionally mapped to remote handles |
| Server daemon | Done | Rust, Tokio-based, graceful shutdown, session registry |
| Session management | Done | Per-connection resource tracking, full cleanup on disconnect |
| GpuBackend trait | Done | `StubGpuBackend` (testing, no GPU required) + `CudaGpuBackend` (real GPU via libloading) |
| Bidirectional callbacks | Done | `cuStreamAddCallback` + `cuLaunchHostFunc` — dedicated callback channel, `CallbackRegistry` |
| Stream-ordered memory | Done | 9 pool functions: `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemPoolCreate`, etc. |
| CUDA Graphs | Done | 9 functions: `cuStreamBeginCapture`, `cuGraphInstantiate`, `cuGraphLaunch`, etc. |
| JIT Linker | Done | 5 functions: `cuLinkCreate`, `cuLinkAddData`, `cuLinkComplete`, etc. |
| Library API (CUDA 12+) | Done | 5 functions: `cuLibraryLoadData`, `cuLibraryGetKernel`, `cuKernelGetFunction`, etc. |
| Occupancy API | Done | 4 functions including `cuOccupancyMaxPotentialBlockSize` |
| Cooperative kernels | Done | `cuLaunchCooperativeKernel`, `cuLaunchKernelEx` |
| PCI Bus ID functions | Done | `cuDeviceGetPCIBusId`, `cuDeviceGetByPCIBusId` |
| Managed/unified memory | Done | `cuMemAllocManaged`, `cuMemPrefetchAsync`, `cuMemAdvise`, `cuMemRangeGetAttribute(s)` |
| Test suite | Done | 50+ integration tests against StubGpuBackend; real GPU tests behind `real-gpu-test` feature flag |

### What Is Not Yet Built

| Component | Phase | Notes |
|-----------|-------|-------|
| NVML interception | Future | Present remote GPUs as local NVML devices |
| UCX/RDMA transport | Phase 2 | Auto-negotiate RDMA vs TCP |
| OpenDMA kernel module | Phase 5 | PCIe BAR1 direct VRAM access, bypasses GPUDirect |
| Multi-node routing | Future | Transparent routing across N nodes |
| outerlink-cli features | Future | Currently skeleton only |

### Key Architecture Numbers

- Protocol message types: ~80 (including handshake, response, error)
- Hook table entries in `interpose.c`: ~250 (covers versioned aliases like `_v2`)
- Handle map types: 12 (contexts, device_ptrs, modules, functions, streams, events, mem_pools, link_states, libraries, kernels, graphs, graph_execs)
- Default server port: **14833**
- Default retry policy: 3 retries (100ms / 500ms / 1000ms delays), 5 reconnect attempts with exponential backoff capped at 30s

---

## Related Documents

- [Pre-Planning Master](../../planning/pre-planning/00-master-preplan.md)
- [R1: Existing Projects](../../planning/research/R1-existing-projects.md)
- [R2: SoftRoCE](../../planning/research/R2-softroce-rdma.md)
- [R3: CUDA Interception](../../planning/research/R3-cuda-interception.md)
- [R4: ConnectX-5 + Transport](../../planning/research/R4-connectx5-transport-stack.md)
- [R5: GPUDirect on GeForce](../../planning/research/R5-gpudirect-geforce-restriction.md)
- [Hardware Inventory](../../planning/pre-planning/01-hardware-inventory.md)
- [System Architecture](01-system-architecture.md)
- [CUDA Function Coverage](../specs/cuda-function-coverage.md)

## Open Questions

- [x] What network hardware? -> 4x ConnectX-5 100GbE (dual port) total
- [x] Can PCIe BAR1 direct access bypass GPUDirect restriction? -> YES, viable via tinygrad patches + custom RDMA module (R7)
- [x] Can NVLink physically bridge across PCs? -> NO, not feasible. ConnectX-5 covers the gap (R6)
- [ ] What Linux distributions on target machines?
- [ ] What CUDA version and driver versions on each machine?
- [ ] First real end-to-end test workload? (LLM inference most likely)
- [ ] Maximum nodes to support in v1?
- [ ] Does `StubGpuBackend` need persistent PTX/ELF introspection cache for correctness with complex kernels?
