# R28: Scatter-Gather DMA — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)

## Purpose

Define the scope, dependencies, decisions, and implementation roadmap for adding scatter-gather DMA to OuterLink. This eliminates the need for staging copies when transferring non-contiguous VRAM regions, reducing latency and memory bandwidth consumption for sparse tensors, fragmented allocations, and multi-tensor batch transfers.

---

## Scope Definition

### In Scope

1. **Fragment analyzer** — Scan R10 page table to identify contiguous runs and fragment count for any logical tensor
2. **Hardware scatter-gather path** — Build ibv_sge lists from page table, post multi-SGE RDMA WRITEs (up to 30 SGEs on ConnectX-5)
3. **Software pre-pack path** — GPU gather kernel for transfers exceeding 30 fragments or requiring compression
4. **Transfer descriptor protocol** — Control message format for coordinating scatter-gather transfers between nodes
5. **Receiver-side scatter** — Handle non-contiguous destinations via staging + scatter kernel, or SEND/RECV with multi-SGE receive
6. **OpenDMA BAR1 integration** — BAR1-aware SGE building, leverage GPU MMU for implicit scatter
7. **Decision logic** — Automatic selection between hardware SG, software pre-pack, and compression based on fragment count, size, and compressibility

### Out of Scope (Handled Elsewhere)

- Page table management — R10 provides this
- Compression algorithms — R14 provides compress/decompress
- Topology-aware routing — R17 handles path selection
- Sparse tensor format awareness — cuSPARSE/framework-level concern
- NCCL collective scatter — R20 handles collective operations

### Boundary: Optimization, Not Replacement

Scatter-gather is an optimization layer. Every transfer that works today (contiguous, single-SGE) continues to work unchanged. Scatter-gather activates only when the fragment analyzer detects non-contiguous page layouts.

---

## Dependencies

### Upstream (Must Exist Before R28)

| Dependency | Component | Why Needed |
|-----------|-----------|------------|
| **R10** | Page Table | Provides physical page addresses for logical tensors |
| **R10** | 64KB Pages | Page size determines SGE granularity |
| **P5/OpenDMA** | BAR1 Access | BAR1 mapping for direct NIC-to-GPU scatter-gather |
| **Transport Layer** | RDMA WRITE | Existing single-SGE path must work before adding multi-SGE |

### Downstream (Uses R28)

| Consumer | How It Uses R28 |
|----------|----------------|
| **R14** | Compression pipeline receives pre-packed data from software gather path |
| **R20** | NCCL backend uses scatter-gather for non-contiguous collective buffers |
| **R21** | GPU Direct Storage can scatter-gather for dataset loading |
| **R12** | Deduplication may produce non-contiguous page layouts after dedup |

### Parallel (No Dependency)

- R15 Fault Tolerance — Independent, but parity fragments may benefit from scatter-gather
- R26 PTP Clock Sync — No interaction

---

## Key Decisions Required

### Decision 1: RDMA WRITE Gather vs SEND/RECV Scatter-Gather

**Context:** RDMA WRITE supports sender-side gather (multi-SGE on local side, contiguous remote target). SEND/RECV supports both sender gather AND receiver scatter. But SEND/RECV requires pre-posted receive buffers and has different flow control.

**Options:**
- **A:** RDMA WRITE only — gather on sender, staging + scatter kernel on receiver
- **B:** SEND/RECV for scatter-scatter cases — native hardware scatter on both sides
- **C:** Hybrid — RDMA WRITE by default, SEND/RECV when both sides are non-contiguous

**Leaning:** Option A for Phase 1 (simpler, works with existing infrastructure), Option C as Phase 2 optimization.

### Decision 2: Fragment Count Threshold for Hardware vs Software

**Context:** Hardware scatter-gather has 30 SGE limit. Software pre-pack has a fixed overhead (~10us GPU kernel launch). Where's the crossover?

**Options:**
- **A:** Always use hardware SG when fragments <= 30
- **B:** Use hardware SG only when fragments <= 10 (conservative)
- **C:** Dynamic threshold based on fragment size (small fragments -> prefer pre-pack)

**Leaning:** Option A with sub-rule: if all fragments < 1KB, use software pre-pack regardless.

### Decision 3: Staging Buffer Management

**Context:** Software pre-pack needs staging buffers. How to manage them?

**Options:**
- **A:** Per-transfer cudaMalloc/cudaFree (simple, fragmentation risk)
- **B:** Pre-allocated pool of staging buffers (fast, fixed memory cost)
- **C:** Use R10's page allocator to carve staging from managed VRAM

**Leaning:** Option B — pool of 4-8 staging buffers, each 4MB, allocated at startup.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 30 SGE limit too low for real workloads | LOW | MEDIUM | Software pre-pack fallback handles unlimited fragments |
| BAR1 page table manipulation not possible from userspace | MEDIUM | HIGH | Fall back to software gather — no BAR1 implicit scatter |
| SGE overhead exceeds benefit for small transfers | LOW | LOW | Decision logic skips SG for tiny transfers |
| Staging buffer pool exhaustion under load | MEDIUM | MEDIUM | Backpressure — queue transfer until staging available |
| UCX doesn't expose SGE control | MEDIUM | MEDIUM | Use raw verbs for SG path, UCX for simple transfers |

---

## Implementation Phases

### Phase 1: Software Pre-Pack (Foundation)

**Deliverables:**
- Fragment analyzer (input: page list, output: contiguous runs)
- GPU gather kernel (non-contiguous VRAM -> contiguous staging)
- GPU scatter kernel (contiguous staging -> non-contiguous VRAM)
- Transfer descriptor protocol (control message format)
- Staging buffer pool

**Acceptance criteria:**
- Transfer non-contiguous VRAM between two nodes via gather -> send -> scatter
- Correctness verified for 1-100 fragment transfers
- Latency overhead < 20us vs contiguous transfer of same total size

### Phase 2: Hardware Scatter-Gather

**Deliverables:**
- SGE list builder (page list -> ibv_sge array with contiguous run merging)
- Multi-SGE ibv_post_send integration
- Decision logic (hardware SG vs software pre-pack)

**Acceptance criteria:**
- Transfers with <= 30 fragments use hardware SG (no staging copy)
- Latency improvement vs software pre-pack for 5-30 fragment transfers
- Correctness verified with ibv_devinfo max_sge confirmation

### Phase 3: OpenDMA Integration

**Deliverables:**
- BAR1-aware SGE builder (SGEs point to BAR1 offsets instead of host memory)
- GPU MMU remapping investigation (can we make scattered VRAM look contiguous in BAR1?)
- OpenDMA scatter-gather path (NIC reads directly from scattered BAR1 regions)

**Acceptance criteria:**
- Zero-CPU scatter-gather: NIC gathers from BAR1, sends over wire
- Latency matches or beats host-staged scatter-gather

### Phase 4: Compression Integration

**Deliverables:**
- Pipeline: gather -> compress -> send -> decompress -> scatter
- Triple buffering for overlap of gather/compress/send stages
- Per-fragment compression option for large fragments

**Acceptance criteria:**
- Compressed scatter-gather transfers achieve higher effective bandwidth than uncompressed
- Pipeline overhead < 10% of transfer time

---

## Estimated Effort

| Phase | Complexity | Estimated Time | Risk |
|-------|-----------|---------------|------|
| Phase 1: Software Pre-Pack | Medium | 2-3 weeks | Low |
| Phase 2: Hardware SG | Medium | 1-2 weeks | Medium |
| Phase 3: OpenDMA SG | High | 2-3 weeks | High |
| Phase 4: Compression | Medium | 1-2 weeks | Low |
| **Total** | | **6-10 weeks** | |

---

## Success Metrics

| Metric | Baseline (No SG) | Target (With SG) |
|--------|------------------|-------------------|
| Non-contiguous 1.875MB transfer latency | ~170us (copy + send) | ~155us (hardware SG, no copy) |
| Staging buffer memory overhead | 0 (always copy to new buffer) | Fixed 32MB pool |
| MoE expert gather (4 x 500MB) | 4 separate RDMA WRITEs | 1 RDMA WRITE with 4 SGEs |
| Maximum fragments per transfer | 1 (must be contiguous) | Unlimited (30 HW + software fallback) |
| CPU involvement for gather | GPU kernel launch per transfer | Zero (hardware SG path) |

---

## Related Documents

- [research/01-rdma-scatter-gather.md](./research/01-rdma-scatter-gather.md) — RDMA SGE details
- [research/02-gpu-sparse-data.md](./research/02-gpu-sparse-data.md) — Sparse data patterns
- [research/03-scatter-gather-pipeline.md](./research/03-scatter-gather-pipeline.md) — Pipeline design
- [R10 Memory Tiering](../../R10-memory-tiering/) — Page table and memory management
- [R14 Transport Compression](../../R14-transport-compression/) — Compression pipeline

## Open Questions

- [ ] Should we expose scatter-gather as a user-facing API, or keep it purely internal to the transport layer?
- [ ] How does scatter-gather interact with QoS and flow control? (large multi-SGE WRs may dominate bandwidth)
- [ ] Can we overlap SGE list building with previous transfer completion? (pipelining the control path)
