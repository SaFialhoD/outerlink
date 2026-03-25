# R16: BlueField DPU Offload — Pre-Plan

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Define what needs to be planned for implementing BlueField DPU offload in OuterLink.

---

## 1. Scope Definition

### In Scope
- DOCA SDK Rust FFI bindings (doca-sys, doca-rs crates)
- `outerlink-dpu` binary for BlueField ARM cores
- Host↔DPU control plane (DOCA Comm Channel or shared memory)
- Transport offload (connection management, routing, RDMA operations on DPU)
- Compression offload (DOCA Compress: deflate on BF-2, + LZ4 on BF-3)
- GPU BAR1 access from DPU (OpenDMA via DPU's ConnectX)
- Prefetch scheduling on DPU ARM cores
- Page table management on DPU
- Fallback path when no DPU is present
- Cross-compilation pipeline (x86_64 host → aarch64 DPU)

### Out of Scope
- BlueField-X (GPU+DPU hybrid cards) — different hardware model
- DOCA Platform Framework / Kubernetes integration — enterprise deployment, not core
- DPU-side ML inference — ARM cores have no GPU, not practical
- Multi-DPU per host — design for one DPU per NIC slot
- NIC mode programming — when in NIC mode, treat as plain ConnectX

### Dependencies
- **P8 (Performance phase)**: DPU offload is a performance optimization, requires working baseline
- **R14 (Transport Compression)**: Compression logic that gets offloaded to DPU hardware
- **R11 (Speculative Prefetch)**: Prefetch algorithms that run on DPU ARM cores
- **R17 (Topology-Aware Scheduling)**: Routing decisions that move to DPU
- **R10 (Memory Hierarchy)**: Page table design that the DPU manages
- **OpenDMA (Phase 5)**: BAR1 access pattern that DPU's ConnectX uses natively

---

## 2. Key Decisions Required

### D1: Minimum BlueField Generation
- **Options**: BF-2 only, BF-3 only, or both
- **Recommendation**: Support both. BF-2 for dev/test (cheap), BF-3 for production features
- **Impact**: BF-2 lacks LZ4 HW compression and dedicated packet processing cores

### D2: DOCA SDK Version
- **Options**: Pin to 2.9 LTS, or track latest (currently 2.10)
- **Recommendation**: Start with latest, evaluate LTS for stability
- **Impact**: API availability, BF-3 feature support

### D3: FFI Binding Approach
- **Options**: Manual bindings, bindgen from headers, or hybrid
- **Recommendation**: bindgen for doca-sys, manual safe wrappers in doca-rs
- **Impact**: Build complexity, maintenance burden

### D4: Host↔DPU Communication
- **Options**: DOCA Comm Channel, PCIe shared memory (NTB), or both
- **Recommendation**: Start with Comm Channel (simpler), optimize to shared memory if latency matters
- **Impact**: Control plane latency (Comm Channel ~5-10 us vs shared memory ~1 us)

### D5: DPU Binary Architecture
- **Options**: Single monolithic binary, or separate services per function
- **Recommendation**: Single binary with internal modules (simpler deployment)
- **Impact**: Deployment complexity, resource isolation

### D6: Page Table Location
- **Options**: Full copy on DPU, or DPU queries host for cold pages
- **Recommendation**: Full copy on DPU (fits in 16 GB for up to ~80 GPUs)
- **Impact**: DPU DRAM usage, lookup latency for cold pages

---

## 3. Unknowns to Resolve

### Hardware Unknowns
- [ ] PCIe topology in Pedro's PCs — can DPU's ConnectX reach GPU's BAR1?
- [ ] BF-2 ARM core throughput for OuterLink's routing logic under load
- [ ] Actual DOCA DMA latency to host pinned memory (need to measure)
- [ ] Actual DOCA Compress throughput for our data patterns (tensor data compressibility)

### Software Unknowns
- [ ] DOCA header compatibility with bindgen (complex macros, inline functions)
- [ ] DOCA SDK installation on dev machine (aarch64 sysroot for cross-compilation)
- [ ] DOCA Comm Channel reliability under high message rates
- [ ] BlueField BSP update process and DOCA SDK compatibility matrix

### Integration Unknowns
- [ ] How outerlink-server detects and handshakes with outerlink-dpu
- [ ] Graceful degradation when DPU crashes mid-transfer
- [ ] Testing strategy without physical BlueField hardware

---

## 4. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ARM cores insufficient for line-rate transport | Medium | High | DOCA Flow for HW fast-path, ARM only handles exceptions |
| PCIe topology prevents GPU BAR1 access | Low | Medium | Fall back to host-staged (still get compression + routing) |
| DOCA SDK too complex / unstable | Medium | Medium | Start with minimal API surface, expand gradually |
| BF-2 unavailable at reasonable price | Low | Low | Widely available used, $150-500 range |
| Cross-compilation pipeline fragile | Medium | Medium | Docker-based build with pinned DOCA SDK version |
| DPU offload adds latency (overhead > savings) | Low | High | Measure at each phase, abandon if net negative |

---

## 5. Implementation Phases

### Phase A: DOCA Foundation (4-6 weeks)
**Goal:** Prove Rust↔DOCA FFI works, host↔DPU data path functional

Deliverables:
- [ ] doca-sys crate: bindgen FFI bindings for DOCA Core, DMA, Comm Channel
- [ ] doca-rs crate: Safe Rust wrappers for core primitives
- [ ] Cross-compilation pipeline: Docker image with DOCA SDK, CI builds aarch64 binary
- [ ] outerlink-dpu skeleton: boots on BlueField, connects to host via Comm Channel
- [ ] Proof of concept: DPU copies 1 GB from host pinned memory to DPU DRAM and back
- [ ] Benchmarks: DMA latency and throughput measurements

### Phase B: Transport Offload (6-8 weeks)
**Goal:** DPU handles all OuterLink network I/O, host CPU freed from data path

Deliverables:
- [ ] TransportBackend trait with host and DPU implementations
- [ ] Connection management on DPU (RDMA QP create/destroy/manage)
- [ ] Routing logic on DPU (from R17)
- [ ] outerlink-server delegates transfers to outerlink-dpu
- [ ] End-to-end test: CUDA app runs unmodified, DPU handles transfers
- [ ] Benchmarks: throughput and latency vs host-only baseline

### Phase C: Compression Offload (3-4 weeks)
**Goal:** Wire traffic compressed/decompressed entirely on DPU hardware

Deliverables:
- [ ] DOCA Compress FFI bindings and safe wrappers
- [ ] Adaptive compression logic on DPU (from R14)
- [ ] Compress-before-send and decompress-after-receive pipeline
- [ ] BF-2 deflate and BF-3 LZ4 code paths
- [ ] Benchmarks: compression throughput, ratio, and impact on transfer latency

### Phase D: GPU BAR1 Integration (4-6 weeks)
**Goal:** DPU writes directly to local GPU VRAM — true zero-CPU data movement

Deliverables:
- [ ] GPUDirect RDMA from DPU's ConnectX to GPU BAR1
- [ ] PCIe topology detection and validation
- [ ] Resizable BAR configuration guide
- [ ] End-to-end: remote VRAM → wire → DPU → local GPU VRAM
- [ ] Benchmarks: latency vs host-staged, throughput at PCIe bandwidth

### Phase E: Prefetch on DPU (3-4 weeks)
**Goal:** DPU proactively fetches pages before host requests them

Deliverables:
- [ ] R11 prefetch algorithms ported to DPU ARM cores
- [ ] Traffic pattern monitoring on DPU
- [ ] Prefetch cache management in DPU DRAM
- [ ] Benchmarks: prefetch hit rate, effective latency reduction

**Total estimated timeline: 20-28 weeks** (phases are sequential, each depends on the previous)

---

## 6. Hardware Needed

| Item | Purpose | Estimated Cost | Priority |
|---|---|---|---|
| BlueField-2 DPU (25 GbE, dual-port) | Development and testing | $150-300 (eBay) | HIGH |
| BlueField-2 DPU (100 GbE) | Performance testing at speed | $300-500 (eBay) | MEDIUM |
| BlueField-3 DPU | Production features (LZ4 HW, packet cores) | $2000+ (new) | LOW (future) |
| aarch64 dev board (optional) | DOCA-free ARM development | $50-100 | LOW |

**Minimum to start:** One BF-2 DPU (any speed variant). The 25 GbE model is cheapest and sufficient for validating the architecture. Pedro's existing ConnectX-5 cards remain as the non-DPU baseline.

---

## 7. What Needs to Be Planned in Detail

Before writing the full implementation plan, these items need detailed design:

1. **doca-sys / doca-rs crate API design** — which DOCA APIs to expose, error handling strategy
2. **outerlink-dpu binary architecture** — module structure, thread model, memory allocation
3. **Host↔DPU protocol** — message format, serialization, flow control
4. **TransportBackend trait design** — interface that abstracts host vs DPU transport
5. **Cross-compilation CI pipeline** — Docker image spec, build matrix (BF-2 / BF-3)
6. **Testing strategy** — unit tests (mock DOCA), integration tests (real hardware), CI without hardware
7. **Deployment automation** — how outerlink-dpu gets installed on BlueField filesystem

---

## Related Documents
- [research/01-bluefield-architecture.md](./research/01-bluefield-architecture.md) — Hardware specs
- [research/02-programming-models.md](./research/02-programming-models.md) — DOCA SDK, Rust FFI
- [research/03-outerlink-offload-design.md](./research/03-outerlink-offload-design.md) — Offload design
- [R14: Transport Compression](../R14-transport-compression/) — Compression to offload
- [R17: Topology-Aware Scheduling](../R17-topology-aware-scheduling/) — Routing to offload
- [R11: Speculative Prefetching](../R11-speculative-prefetch/) — Prefetch to offload

## Open Questions
- [ ] Should we acquire a BF-2 before starting Phase A, or can we begin with DOCA SDK exploration on x86_64?
- [ ] Is there a DOCA software emulator that lets us develop without hardware?
- [ ] Should doca-sys be contributed to the open-source Rust ecosystem or kept internal?
- [ ] What is the upgrade path from BF-2 development to BF-3 production?
