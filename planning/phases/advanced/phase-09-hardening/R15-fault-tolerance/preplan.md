# R15: Fault Tolerance & Erasure Coding — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)

## Purpose

Define the scope, dependencies, decisions, and implementation roadmap for adding fault tolerance to OuterLink. This includes erasure-coded data protection, distributed checkpointing, failure detection, and automated recovery — making consumer GPU clusters production-reliable.

---

## Scope Definition

### In Scope

1. **Erasure coding for VRAM protection** — RS and XOR parity across nodes, protecting GPU memory pages from node loss
2. **Failure detection pipeline** — ibv_async_event monitoring + phi accrual failure detector + fallback TCP keepalive
3. **Recovery orchestration** — detect, fence, assess, reconstruct, resume workflow
4. **Distributed checkpointing** — in-memory checkpoint hierarchy (VRAM -> DRAM -> remote DRAM -> NVMe)
5. **Incremental checkpointing** — delta-based updates to minimize network overhead
6. **Hot spare support** — pre-staged nodes ready for immediate failover
7. **Partial failure handling** — GPU crash (node alive), NIC failure, process crash
8. **Cluster membership management** — generation-based protocol with fencing

### Out of Scope (handled by other R-topics)

- Live migration (R22) — graceful, planned data movement (vs crash recovery here)
- GPU Direct Storage integration (R21) — NVMe as storage tier
- Network topology discovery (R17) — already provides heartbeat infrastructure
- Memory page table management (R10) — we consume the page table, don't build it
- NCCL-level collective fault tolerance — we operate below NCCL

### Boundary: Inference vs Training

- **Inference workloads:** Protect allocated VRAM (model weights, KV caches). Recovery = reconstruct VRAM from parity. No checkpoint needed (weights are immutable).
- **Training workloads:** Protect VRAM + optimizer state + gradient accumulation. Recovery = reconstruct from parity + roll back to checkpoint. Some work lost (steps since last checkpoint).

---

## Dependencies

### Upstream (must exist before R15)

| Dependency | Component | Why Needed |
|-----------|-----------|------------|
| **R10** | Memory Tiering | Page table, 64KB page abstraction, tier locations — we protect these pages |
| **R10** | ARC Eviction | Parity placement interacts with eviction — don't evict a parity page if it's protecting active data |
| **R17** | Phi Accrual Detector | Failure detection infrastructure — we extend it, don't rebuild it |
| **R17** | Topology Discovery | Node inventory, link capabilities — needed for parity placement decisions |
| **R12** | Memory Deduplication | Shared read-only pages need different protection than mutable pages |
| **R19** | SWMR Consistency | Coherency state determines parity update timing (dirty pages need fresh parity) |
| **P5+** | Basic Transport | RDMA or TCP transport must work — we use it for parity distribution and recovery |

### Downstream (depends on R15)

| Dependent | Component | Why |
|-----------|-----------|-----|
| **R22** | Live Migration | Uses fault tolerance infrastructure for graceful failover |
| **R24** | Time-Sliced Sharing | Needs checkpoint/restore for context switching between users |

### External Dependencies

| Library | Purpose | License | Risk |
|---------|---------|---------|------|
| Intel ISA-L | RS erasure coding on CPU | BSD-3 | Low — mature, widely used |
| (Optional) G-CRS/PErasure | RS on GPU | Research | Medium — may need adaptation |

---

## Key Decisions

### Decision 1: Erasure Coding Scheme

**Options:**
- A) Reed-Solomon (ISA-L on CPU) — proven, storage-optimal, fast enough
- B) XOR parity only — simplest, fastest, but only 1-failure tolerance
- C) Hybrid RS + XOR — XOR for hot data, RS for cold/important data
- D) Fountain codes — near-optimal at large scale, overkill for small clusters

**Recommended:** Option C (Hybrid). XOR for hot pages (fast path), RS for cold/checkpoint data (robust path). This matches the memory tiering hierarchy from R10.

**Rationale:** XOR parity is zero-overhead on the critical path (encoding is a single pass at memory bandwidth speed). RS adds multi-failure tolerance for data that can't afford loss. At 2-8 nodes, RS computation is trivial.

### Decision 2: Parity Storage Location

**Options:**
- A) Parity in VRAM — fastest recovery, but consumes precious GPU memory
- B) Parity in partner node DRAM — good recovery speed, doesn't waste VRAM
- C) Parity on NVMe — persistent but slow recovery
- D) Tiered parity — hot parity in DRAM, cold parity on NVMe

**Recommended:** Option D (Tiered). Hot data parity in partner DRAM. Cold data parity on NVMe. Critical data (model weights) gets parity in both.

**Rationale:** VRAM is too precious (24 GB on RTX 3090, every GB counts). DRAM is plentiful (~64-128 GB per node) and fast enough. NVMe provides persistence.

### Decision 3: Checkpoint Strategy

**Options:**
- A) Periodic full checkpoints to NVMe
- B) In-memory checkpoints (Gemini-style) with NVMe backup
- C) Incremental checkpoints with periodic full snapshots
- D) Full Gemini + incremental deltas

**Recommended:** Option D. Combines best of Gemini (in-memory, fast) with LowDiff (incremental, low overhead).

**Rationale:** Gemini showed 13x recovery speedup with in-memory checkpoints. Incremental deltas reduce network traffic by 70-90%. OuterLink already has the distributed memory infrastructure.

### Decision 4: Detection Mechanism

**Options:**
- A) Fixed timeout heartbeat (simple, prone to false positives)
- B) Phi accrual failure detector (adaptive, R17 already designed it)
- C) RDMA events only (fast but doesn't catch process crashes)
- D) Multi-layer: RDMA events + phi accrual + TCP fallback

**Recommended:** Option D (Multi-layer). Each layer covers different failure modes.

### Decision 5: Cluster Membership

**Options:**
- A) Static membership (admin configures, no auto-detection)
- B) Raft-based consensus (strong consistency, complex)
- C) Generation-based with coordinator (simpler, good enough for 2-8 nodes)
- D) Gossip-based (eventually consistent, good for large clusters)

**Recommended:** Option C. Generation-based with elected coordinator. For 2-8 nodes, Raft is overkill. Gossip is for hundreds of nodes.

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Parity encoding adds latency to write path** | Medium | Medium | Async parity updates — write returns before parity is computed |
| **False positives in failure detection** | Medium | High | Phi accrual with conservative threshold (6+); require RDMA event confirmation |
| **Split-brain in 2-node clusters** | High (for 2-node) | Critical | External witness or shared-NVMe tiebreaker |
| **Cascading failures during recovery** | Low | Critical | RS(k,2) tolerates 2 simultaneous failures; pause recovery if cluster drops below quorum |
| **VRAM overhead for parity metadata** | Low | Medium | Parity stored in DRAM, not VRAM; only metadata (~bytes per page) in VRAM |
| **ISA-L performance on AMD CPUs** | Low | Low | ISA-L supports SSE/AVX2 which AMD has; AVX-512 is Intel-optimized but not required |
| **Recovery time exceeds target** | Medium | Medium | Prioritize hot pages; pre-compute recovery plan before failure; hot spares |
| **Checkpoint interference with training** | Medium | Medium | Async checkpoint pipeline; RDMA traffic scheduling (Gemini approach) |

---

## Implementation Phases

### Phase R15-A: Foundation (3-4 weeks)

**Deliverables:**
1. Erasure coding library integration (ISA-L Rust bindings or FFI)
2. XOR parity encoder/decoder for single-page and multi-page stripes
3. RS(k,m) encoder/decoder with configurable parameters
4. Parity storage manager (allocate/track parity locations in DRAM)
5. Unit tests: encode, decode, recover from 1 and 2 simulated failures

**Acceptance criteria:**
- XOR encoding at >= 10 GB/s on single core
- RS(4,2) encoding at >= 5 GB/s on single core
- Correct reconstruction from any valid subset of fragments
- Parity metadata overhead < 0.1% of protected data size

### Phase R15-B: Failure Detection (2-3 weeks)

**Deliverables:**
1. ibv_async_event monitoring thread (integrated with transport layer)
2. Phi accrual failure detector (extend R17's heartbeat design)
3. Failure declaration and notification protocol
4. Fencing mechanism (generation-based, invalidate stale connections)
5. Cluster membership manager with coordinator election

**Acceptance criteria:**
- Detection latency < 1 second for node crash
- False positive rate < 0.001% under normal network conditions
- Correct fencing: stale node cannot write after being fenced
- Membership correctly tracks join/leave/fail events

### Phase R15-C: Recovery Pipeline (3-4 weeks)

**Deliverables:**
1. Recovery orchestrator (detect -> fence -> assess -> reconstruct -> resume)
2. Page reconstruction from XOR parity
3. Page reconstruction from RS parity
4. Reconstruction prioritization (hot pages first, based on access frequency)
5. Hot spare activation workflow
6. Integration with R10 page table (update page locations after recovery)

**Acceptance criteria:**
- Single node failure recovery < 30 seconds (with RS parity available)
- Hot page recovery < 5 seconds (with XOR parity)
- No data corruption during recovery (verified by checksums)
- Surviving nodes continue serving requests during recovery

### Phase R15-D: Checkpointing (3-4 weeks)

**Deliverables:**
1. Snapshot engine: GPU->CPU async DMA for checkpoint capture
2. In-memory checkpoint store (local DRAM + remote DRAM via RDMA)
3. Incremental delta computation and application
4. Checkpoint placement optimizer (Gemini-style redundancy)
5. NVMe cold checkpoint writer (background, periodic)
6. Checkpoint recovery: load, apply deltas, restore GPU state

**Acceptance criteria:**
- Checkpoint snapshot overhead < 3% of training throughput
- Full checkpoint of 10 GB model state in < 5 seconds to DRAM
- Incremental delta size < 30% of full checkpoint (for typical training)
- Recovery from in-memory checkpoint < 10 seconds
- Recovery from NVMe checkpoint < 60 seconds

### Phase R15-E: Integration and Testing (2-3 weeks)

**Deliverables:**
1. Integration with R10 (page table), R17 (topology), R19 (coherency)
2. Fault injection framework (kill processes, drop packets, corrupt pages)
3. End-to-end test: training run survives node failure and resumes
4. End-to-end test: inference serving survives node failure
5. Performance benchmarks: overhead of fault tolerance on normal operations
6. Documentation and configuration guide

**Acceptance criteria:**
- Training survives single node failure with < 60 seconds total disruption
- Inference survives single node failure with < 10 seconds disruption
- Fault tolerance overhead < 5% on normal (no-failure) operations
- All failure modes from Section 1 of research/03 are tested

---

## Estimated Total: 13-18 weeks

This is a large feature. The phases can overlap (e.g., start Phase B while finishing Phase A), but the dependency chain is: A -> C (need EC before recovery), B -> C (need detection before recovery), C + D are partially parallel, E requires all others.

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Single node failure recovery time** | < 30 seconds | Time from crash to all operations resumed |
| **Data durability (with RS parity)** | 99.999% | No data loss for any single-node failure |
| **Data durability (with checkpoint)** | > 99.9% | At most N seconds of training work lost |
| **Detection latency** | < 1 second | Time from failure to detection |
| **Normal operation overhead** | < 5% | Throughput reduction vs no fault tolerance |
| **Checkpoint overhead** | < 3% | Training throughput impact of checkpointing |
| **Parity storage overhead** | 25-50% of protected data | Depends on RS configuration |
| **VRAM overhead** | < 1% | Only metadata in VRAM, parity in DRAM |

---

## Open Questions

1. **Parity update frequency:** Should parity be updated on every page write (strong protection, high overhead) or periodically (lower overhead, vulnerability window)? Likely configurable per data class.

2. **Quorum for 2-node clusters:** Two nodes cannot form a majority. Options: external witness, shared NVMe lease, or accept that 2-node clusters can only tolerate 1 failure with manual intervention.

3. **Interaction with CUDA Unified Memory:** If applications use CUDA UVM, page faults may trigger during recovery. Need to ensure the fault handler doesn't deadlock with recovery.

4. **Checkpoint format compatibility:** Should OuterLink checkpoints be DCP-compatible (interoperable with PyTorch ecosystem) or use a custom format optimized for in-memory storage?

5. **Cost of protecting everything:** Should ALL pages be erasure-coded, or only "important" ones? Full protection is simpler but costs 25-50% more memory. Selective protection is more efficient but adds complexity.

6. **Recovery under memory pressure:** If surviving nodes are already near capacity, where do reconstructed pages go? May need to evict cold data to NVMe to make room.

---

## Related Documents

- R15 Research 01: Erasure Coding Algorithms
- R15 Research 02: Distributed Checkpointing
- R15 Research 03: Failure Detection and Recovery
- R10: Memory Tiering & NVMe as Tier 3
- R12: Memory Deduplication
- R17: Topology-Aware Scheduling
- R19: Network Page Faults / SWMR Consistency
- R22: Live Migration (downstream)
