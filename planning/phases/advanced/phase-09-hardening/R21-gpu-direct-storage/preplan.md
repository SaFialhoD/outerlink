# R21: GPU Direct Storage Over Network — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Phase:** 9 — Hardening
**Priority:** MEDIUM

## Purpose

Define the scope, dependencies, risks, and implementation phases for enabling remote NVMe-to-GPU VRAM transfers without bouncing through host RAM. This combines OpenDMA (BAR1 RDMA) on the receiver with storage-side DMA on the sender to create a direct NVMe --> network --> GPU pipeline.

---

## 1. Scope Definition

### In Scope

| Component | Description |
|-----------|-------------|
| Host-staged remote storage read | NVMe → host RAM → RDMA → OpenDMA → GPU VRAM (sender uses RAM, receiver doesn't) |
| NVMe-oF integration | Expose remote NVMe drives as block devices via NVMe-oF with RDMA transport |
| ConnectX-5 NVMe-oF target offload | Zero-CPU sender: hardware chains NVMe read + RDMA send |
| Storage API (ol_storage) | Rust API for reading/writing remote storage to/from GPU VRAM |
| Batch and async I/O | Multiple concurrent storage reads, CUDA stream integration |
| R10 tiering integration | Remote NVMe as Tier 5 in memory hierarchy, transparent access via page manager |
| Throughput benchmarking | Measure actual end-to-end bandwidth for each path variant |

### Out of Scope (for now)

| Excluded | Reason | Where It Goes |
|----------|--------|---------------|
| NVIDIA cuFile API interception | Only useful for apps already using cuFile; rare in practice | Future consideration |
| SPDK-based storage target | Too complex for initial implementation; kernel nvmet + offload is sufficient | Plan B if kernel path underperforms |
| NVMe Gen5 optimization | We don't have Gen5 hardware yet | Revisit when hardware available |
| Distributed filesystem (CephFS, etc.) | Block-level access via NVMe-oF is sufficient initially | R21 Phase 2+ |
| Write-back caching (GPU → remote NVMe) | Read path is priority; write path for checkpointing comes later | R21 Phase 3 or separate research |
| GPU-side bounce buffers for unaligned I/O | Optimization detail, implement during Phase 3 | Part of API hardening |

### Deliverables

1. Host-staged remote NVMe reader with OpenDMA delivery to GPU
2. NVMe-oF subsystem configuration tooling (nvmet setup automation)
3. ConnectX-5 NVMe-oF target offload configuration and verification
4. `ol_storage` Rust API with `read_to_gpu()`, `batch_read()`, `read_async()` functions
5. R10 integration: Tier 5 page fault handler using `ol_storage` API
6. Benchmark suite measuring throughput for each pipeline variant
7. Documentation: setup guide, PCIe topology requirements, troubleshooting

---

## 2. Dependencies

### Upstream (what R21 needs)

| Dependency | Component | Status | Why |
|------------|-----------|--------|-----|
| P9 OpenDMA (BAR1 RDMA) | outerlink-server | Required | Receiver side: DMA to GPU BAR1 from ConnectX-5 |
| R10 Memory Tiering | outerlink-server | Required | Tier 4/5 integration for automatic NVMe access |
| ConnectX-5 with RDMA | Hardware | Available | Network transport and NVMe-oF target offload |
| NVMe SSD (Gen4+) | Hardware | Available | Storage medium |
| Linux kernel 6.2+ | OS | Available on Ubuntu 24.04 | P2PDMA userspace support, GDS P2PDMA mode |
| AMD Zen chipset | Hardware | Available | P2PDMA support for NVMe-to-NIC peer transfers |
| nvme-cli / nvmetcli | System package | Available | NVMe-oF target/initiator configuration |
| Working RDMA transport | outerlink-common | Required | For host-staged sender path |

### Downstream (what depends on R21)

| Dependent | Component | How It Uses R21 |
|-----------|-----------|-----------------|
| R10 | Memory Tiering | Transparent Tier 5 access (remote NVMe backing store) |
| R28 | Scatter-Gather DMA | Multi-region NVMe reads for GPU |
| Training data loading | Application layer | Direct dataset streaming from remote storage to GPU |
| Checkpointing | Application layer | GPU state → remote NVMe (write path) |

---

## 3. Key Decisions

### Decision 1: Sender Architecture

| Option | Pros | Cons |
|--------|------|------|
| **A: Host-staged (NVMe → RAM → RDMA)** | Simple, works everywhere, uses existing transport | Host RAM touched, CPU orchestrates |
| **B: NVMe-oF target offload** | Zero CPU, zero host RAM, hardware-accelerated | PCIe topology requirements, nvmet setup |
| **C: SPDK userspace NVMe-oF** | Lowest latency, full control | Complex (VFIO, dedicated cores, NVMe unavailable to kernel) |

**Recommendation:** Start with Option A, migrate to Option B once P2PDMA topology is validated. Option C is Plan B only.

### Decision 2: Storage Protocol

| Option | Pros | Cons |
|--------|------|------|
| **A: NVMe-oF (standard)** | Hardware offloads, standard protocol, block device semantics | Protocol overhead, block-level only |
| **B: Custom RDMA protocol** | Full control, can optimize for GPU delivery | No hardware offloads, must build protocol |
| **C: Hybrid (NVMe-oF sender, custom receiver)** | Best of both: HW offload on sender, OpenDMA on receiver | Two protocol stacks |

**Recommendation:** Option C (hybrid). Use NVMe-oF target offload on sender for zero-CPU storage access. Use our custom RDMA + OpenDMA on receiver for direct GPU delivery. The bridge point is the ConnectX-5 on the sender.

### Decision 3: Kernel NVMe-oF vs SPDK

| Option | Pros | Cons |
|--------|------|------|
| **A: Kernel nvmet + ConnectX-5 offload** | Simple, NVMe stays in kernel, HW offload | Slightly higher latency than SPDK |
| **B: SPDK NVMe-oF target** | Lowest latency (polled) | Complex, NVMe unavailable to kernel |

**Recommendation:** Option A. ConnectX-5 hardware offload gives us near-zero CPU usage (same benefit as SPDK polling) with much less complexity. NVMe remains accessible via kernel for filesystem use.

### Decision 4: API Level

| Option | Pros | Cons |
|--------|------|------|
| **A: Block-level API** | Matches NVMe-oF semantics, simple | Apps must manage offsets, no file abstraction |
| **B: File-level API (like cuFile)** | Familiar to CUDA developers, handles alignment | More complex, needs file→block translation |
| **C: Page-level (integrated with R10)** | Transparent to applications, automatic tiering | Most complex, depends on R10 page manager |

**Recommendation:** Build all three layers incrementally. Start with A (block), add B (file) for direct API users, integrate C (page) with R10 for transparent access.

---

## 4. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PCIe topology prevents NVMe-to-NIC P2P | Medium | High | Test on actual hardware early. AMD Zen root complex supports P2P. Fallback to host-staged sender. |
| Consumer NVMe lacks CMB | High | Low | We don't need CMB — ConnectX-5 embedded switch or host-staged path work without it. |
| NVMe-oF target offload requires specific firmware | Medium | Medium | Verify ConnectX-5 firmware supports nvmet offload. Update firmware if needed. |
| NVMe-oF protocol overhead reduces throughput | Low | Medium | For large sequential reads, protocol overhead is negligible. Benchmark to verify. |
| IOMMU blocks P2PDMA | Medium | High | Disable IOMMU or configure passthrough for specific devices. Document in setup guide. |
| BAR1 size limits concurrent DMA regions | Medium | Medium | RTX 3090 has 256MB BAR1 (or larger with rebar). Use rotation/pipelining for large datasets. |
| NVMe-oF + OpenDMA protocol bridging is complex | Medium | Medium | Start with host-staged receiver path. Add OpenDMA receiver incrementally. |
| Alignment requirements cause frequent fallbacks | Low | Low | Most training data I/O is large and aligned. Add GPU bounce buffers for edge cases. |

---

## 5. Implementation Phases

### Phase 1: Host-Staged Remote Storage (2-3 weeks)

**Goal:** Remote NVMe data into GPU VRAM, host RAM touched on sender only.

| Task | Estimate | Details |
|------|----------|---------|
| Storage read abstraction | 2 days | Trait for reading from local/remote NVMe with async support |
| Host-staged sender | 3 days | NVMe read → pinned buffer → RDMA send, with double-buffering |
| OpenDMA receiver integration | 2 days | RDMA receive → BAR1 DMA → GPU VRAM (leverage existing OpenDMA) |
| Basic `ol_storage_read_to_gpu()` | 2 days | Synchronous API wrapping the pipeline |
| Throughput benchmark | 2 days | Measure single-NVMe and multi-NVMe throughput end-to-end |
| Documentation | 1 day | Setup, configuration, benchmark results |

### Phase 2: NVMe-oF Integration (2-3 weeks)

**Goal:** Remote NVMe exposed as block device, sender uses kernel nvmet.

| Task | Estimate | Details |
|------|----------|---------|
| nvmet subsystem automation | 2 days | Scripts/Rust code to configure nvmet, create subsystems, add namespaces |
| NVMe-oF RDMA transport setup | 2 days | Configure RDMA transport for nvmet, verify connectivity |
| ConnectX-5 target offload config | 2 days | Enable and verify hardware offload, measure CPU reduction |
| P2PDMA topology validation | 2 days | Test NVMe-to-NIC P2P on our AMD Zen hardware |
| NVMe-oF initiator integration | 2 days | nvme-cli connect automation, block device management |
| Benchmark: offload vs non-offload | 1 day | CPU usage and throughput comparison |

### Phase 3: Full P2P Pipeline + API (2-3 weeks)

**Goal:** Zero-copy on both sides, complete storage API.

| Task | Estimate | Details |
|------|----------|---------|
| Protocol bridge: NVMe-oF sender → OpenDMA receiver | 3 days | Bridge block-level NVMe-oF reads with BAR1 DMA delivery |
| `ol_storage_batch_read()` | 2 days | Batch I/O for multiple regions in one call |
| `ol_storage_read_async()` | 2 days | CUDA stream integration for compute-after-IO |
| Alignment handling + fallback | 2 days | Detect unaligned I/O, fall back to host-staged |
| Write path (GPU → remote NVMe) | 2 days | For checkpointing: BAR1 read → RDMA → NVMe write |
| End-to-end benchmark suite | 2 days | All variants: host-staged, NVMe-oF, full P2P |

### Phase 4: R10 Tiering Integration (1-2 weeks)

**Goal:** Remote NVMe is transparent Tier 5 in memory hierarchy.

| Task | Estimate | Details |
|------|----------|---------|
| Tier 5 page fault handler | 3 days | R10 page manager calls `ol_storage_read_to_gpu()` on Tier 5 fault |
| Prefetching for sequential access | 2 days | Detect sequential NVMe access patterns, prefetch ahead |
| Eviction to NVMe (write-back) | 2 days | Dirty pages evicted from RAM tiers written to NVMe |
| Integration test: large dataset training | 2 days | Train model with dataset larger than total cluster RAM |

**Total estimate: 7-11 weeks**

---

## 6. Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Single NVMe → GPU throughput (host-staged) | **>6 GB/s** | iozone + custom benchmark |
| Single NVMe → GPU throughput (full P2P) | **>6.5 GB/s** | Custom benchmark with P2P path |
| Sender CPU usage (host-staged) | Baseline | Top/perf during transfer |
| Sender CPU usage (NVMe-oF offload) | **<5% of baseline** | Top/perf during transfer |
| Receiver CPU usage (OpenDMA) | **<2%** | Top/perf during transfer |
| Multi-NVMe striped throughput | **>12 GB/s** (2x NVMe) | Custom benchmark |
| End-to-end latency (first byte, 4KB) | **<200 us** | Custom latency benchmark |
| R10 Tier 5 page fault latency | **<500 us** | Page fault trace timing |
| Dataset streaming sustained | **>5 GB/s** for 10+ minutes | Training workload benchmark |

---

## 7. Open Questions

### Must Answer Before Implementation

- [ ] **PCIe topology on our hardware**: Can our M.2 NVMe slots P2P with the ConnectX-5 PCIe slot? Need to run `lspci -tv` and check if they share a root complex or switch.
- [ ] **ConnectX-5 NVMe-oF target offload firmware**: Does our ConnectX-5 firmware version support nvmet target offload? Need to check with `mlxfwmanager`.
- [ ] **BAR1 size on RTX 3090**: What's the actual BAR1 size? With resizable BAR enabled, it could be up to 24GB (full VRAM). Without rebar, typically 256MB.

### Can Answer During Implementation

- [ ] **NVMe-oF vs custom RDMA protocol overhead**: For large sequential reads (>1MB), is NVMe-oF protocol overhead measurable vs raw RDMA?
- [ ] **Optimal I/O size for pipeline**: What's the sweet spot for chunk size when pipelining NVMe read → RDMA send? (Probably 1-16MB)
- [ ] **IOMMU configuration**: Can we run P2PDMA with IOMMU in passthrough mode (instead of fully disabled)?

### Research Needed

- [ ] **Multi-tenant storage access**: When multiple GPU nodes access the same storage node simultaneously, how does throughput scale? NVMe can handle multiple queues but total bandwidth is shared.
- [ ] **Error handling for P2P failures**: If P2PDMA fails mid-transfer (e.g., PCIe error), how do we detect and recover gracefully?

---

## Related Documents

- [research/01-nvidia-gds-architecture.md](research/01-nvidia-gds-architecture.md) — NVIDIA GDS: what it does and why we can't use it on GeForce
- [research/02-p2pdma-and-nvme.md](research/02-p2pdma-and-nvme.md) — Linux P2PDMA, NVMe CMB, NVMe-oF, SPDK
- [research/03-remote-gds-pipeline.md](research/03-remote-gds-pipeline.md) — Full pipeline design, throughput analysis, API design
- [R10: Memory Tiering](../../phase-08-smart-memory/R10-memory-tiering/README.md) — NVMe as Tier 4/5
- [P9: OpenDMA](../../phase-05-opendma/README.md) — BAR1 RDMA (receiver side)
- [R28: Scatter-Gather DMA](../R28-scatter-gather-dma/README.md) — Multi-region DMA for efficient storage loads
