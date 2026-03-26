# R22: Live Migration — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Depends On:** R10 (Memory Tiering), R15 (Fault Tolerance), R19 (Network Page Faults)

## Purpose

Define the scope, unknowns, dependencies, and decision points for implementing live GPU workload migration in OuterLink. This pre-plan establishes WHAT needs to be planned before writing the detailed implementation plan.

---

## 1. Scope Definition

### What Live Migration IS

Move a running GPU workload from one node to another with minimal downtime (<500ms), no data loss, and no application awareness. The CUDA application continues running as if nothing happened.

### What Live Migration IS NOT

- **Checkpoint/restart:** We don't save to disk and reload. We transfer live state over the network.
- **Process migration:** We don't move the host process. The client process stays where it is; only the GPU execution backend moves.
- **Multi-GPU rebalancing:** Initially, single-GPU-to-single-GPU migration. Multi-GPU workload migration is a future extension.

### Boundaries

| In Scope | Out of Scope (initially) |
|----------|-------------------------|
| Single GPU workload migration | Multi-GPU workload migration |
| VRAM contents transfer (pre+post copy) | Graphics interop (OpenGL/Vulkan) workloads |
| CUDA context/stream/event reconstruction | CUDA IPC shared memory migration |
| Integration with R10 page table for dirty tracking | Migration across different GPU architectures (e.g., Ampere→Hopper) |
| Integration with R19 for post-copy page faults | Zero-downtime migration (overlapped execution) |
| Manual and automated migration triggers | ML-predicted migration planning |
| CLI command for manual migration | GUI/dashboard for migration management |

---

## 2. Dependencies

### Hard Dependencies (must exist before R22 work begins)

| Dependency | What We Need From It | Status |
|-----------|---------------------|--------|
| **R10: Memory Tiering** | Page table with 64KB pages, PTE flags (dirty bit), access pattern tracking | NOT STARTED |
| **Core interception layer (P5-P7)** | Full CUDA Driver API interception with handle tracking, module caching, stream/event management | NOT STARTED |
| **Transport layer (P6)** | Reliable bulk data transfer between nodes at wire speed | NOT STARTED |

### Soft Dependencies (enhance but don't block R22)

| Dependency | What It Adds | Impact If Missing |
|-----------|-------------|-------------------|
| **R15: Fault Tolerance** | Erasure coding protects against source failure during post-copy | Post-copy becomes riskier without redundancy; can still do pure pre-copy |
| **R19: Network Page Faults** | Enables hybrid pre+post copy with demand paging | Must do pure pre-copy; higher downtime for large/dirty workloads |
| **R12: Memory Dedup** | Reduces migration data (shared weights already on destination) | Full VRAM transfer required; works but slower for multi-tenant workloads |
| **R17: Topology-Aware Scheduling** | Intelligent destination node selection | Manual destination selection; works but less optimal |

### Implementation Order Within R22

```
Phase 1: Basic stop-and-copy (no dependencies beyond core interception)
    |
Phase 2: Pre-copy with dirty tracking (needs R10 page table)
    |
Phase 3: Hybrid pre+post copy (needs R19 page faults)
    |
Phase 4: Intelligent migration (needs R15, R17, R12)
```

---

## 3. Key Decisions to Make

| # | Decision | Options | Considerations | When |
|---|----------|---------|---------------|------|
| D1 | Dirty tracking granularity | 64KB pages (R10) vs per-buffer (interception-level) | Per-buffer is zero-overhead but conservative; per-page is precise but needs R10 | Before Phase 2 |
| D2 | Pre-copy convergence threshold | Fixed page count vs percentage vs bandwidth-relative | Determines when to switch from pre-copy to quiesce | Before Phase 2 |
| D3 | Quiesce strategy | Wait for kernel completion vs force-kill after timeout | Force-kill causes data corruption; long wait increases downtime | Before Phase 2 |
| D4 | Virtual address preservation | Require same VA on destination vs support pointer rewriting | Same VA is strongly preferred (cuMemAddressReserve); rewriting is complex fallback | Before Phase 1 |
| D5 | Migration rate limiting | Fixed fraction vs adaptive vs workload-aware | Affects both migration time and workload impact during pre-copy | Before Phase 2 |
| D6 | Metadata serialization format | Protobuf vs Flatbuffers vs custom binary | Needs to handle handle maps, page tables, module binaries | Before Phase 1 |
| D7 | Same-arch requirement | Require identical GPU arch or allow cross-arch | Cross-arch needs PTX recompilation; same-arch can use cubin directly | Before Phase 1 |

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Virtual address conflict on destination | Low | High — requires pointer rewriting | Proactively reserve VA ranges during Phase 0; keep VA space layout consistent across nodes |
| Kernel execution takes too long to complete at quiesce | Medium | Medium — increases downtime | Configurable timeout; for training, detect iteration boundaries and time quiesce to boundary |
| Pre-copy never converges (dirty rate > bandwidth) | Medium | Low — switch to hybrid/post-copy | Automatic fallback to post-copy when convergence stalls after N rounds |
| Source node fails during migration | Low | High — data loss if post-copy | R15 erasure coding; for pre-copy phase, no data loss (source still has everything) |
| CUDA internal state not fully captured | Medium | High — destination context is broken | Comprehensive testing with real workloads; Cricket's approach as reference |
| cuMemMap API behavior differences across driver versions | Low | Medium — mapping fails on destination | Pin to specific driver version; test migration across same-version nodes |
| Migration storm (multiple workloads migrating simultaneously) | Low | Medium — bandwidth exhaustion | Global migration scheduler; queue migrations; limit concurrent migrations per node |

---

## 5. Research Completed

| Document | Key Findings |
|----------|-------------|
| [01-vm-migration-techniques.md](research/01-vm-migration-techniques.md) | Hybrid pre+post copy is optimal for OuterLink. Pre-copy converges for inference; post-copy needed for training. R10/R19 integration provides all building blocks. |
| [02-gpu-state-capture.md](research/02-gpu-state-capture.md) | OuterLink's interception layer tracks ~95% of needed state. VRAM is 99%+ of data volume. In-flight kernels CANNOT be migrated — must wait for completion. Metadata is ~20MB total. |
| [03-migration-protocol.md](research/03-migration-protocol.md) | 6-phase protocol designed. Downtime target: 51ms (inference) to 790ms (training worst case). Wire protocol defined. 4-phase implementation roadmap. |

---

## 6. Unknowns Requiring Investigation

### Must Resolve Before Planning

1. **cuMemAddressReserve with specific base address** — Can we reliably reserve the same virtual address range on a different node? If not, the entire migration approach needs a pointer rewriting fallback.
   - **How to resolve:** Write a test program that reserves specific VA ranges on two different GPUs. Test with varying fragmentation levels.

2. **cuModuleLoadData performance** — How long does it take to load a typical ML model's modules? This is on the critical path during activate phase.
   - **How to resolve:** Benchmark cuModuleLoadData with real model cubins (Llama, Stable Diffusion).

3. **Stream ordering guarantee after reconstruction** — After recreating streams on destination, do we need to re-establish any inter-stream ordering that existed on source?
   - **How to resolve:** At quiesce point, all streams are synchronized (empty). No pending ordering exists. Verify this assumption.

### Can Resolve During Implementation

4. **Exact dirty rate for target workloads** — Needed to size pre-copy rounds and predict convergence.
5. **PCIe bandwidth contention during pre-copy** — How much does cuMemcpyDtoH for page reads affect kernel performance?
6. **CUDA IPC handle invalidation** — How to cleanly handle migration of workloads using IPC memory.

---

## 7. Milestones for the Plan

The detailed plan (plan.md) should cover these milestones:

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| M1 | State serialization/deserialization | Can capture full CUDA state from interception layer and reconstruct on a different GPU |
| M2 | Stop-and-copy migration | Pause workload, transfer all VRAM, reconstruct, resume on destination. High downtime but correct. |
| M3 | Pre-copy with dirty tracking | Iterative VRAM transfer while workload runs. Reduced downtime. |
| M4 | Hybrid pre+post copy | Integrate R19 page faults for post-copy. Minimal downtime for all workload types. |
| M5 | Automated migration triggers | Integration with R15 (failure-triggered) and R17 (load-balance-triggered) |
| M6 | Migration CLI and monitoring | User-facing commands, progress reporting, migration history |

---

## 8. Testing Strategy Considerations

| Test Type | What to Verify |
|-----------|---------------|
| **Correctness** | Workload produces identical output after migration (bit-exact for deterministic workloads) |
| **Downtime measurement** | Instrument the pause duration visible to the application |
| **Stress test** | Migrate during peak GPU utilization; verify no corruption |
| **Failure injection** | Kill source/destination at each phase; verify recovery |
| **Concurrent migration** | Multiple workloads migrating simultaneously |
| **Round-trip** | Migrate A→B→A; verify identical state |
| **Large VRAM** | Full 24GB migration; measure total time and downtime |

---

## Related Documents

- [README.md](README.md) — R22 overview
- [research/01-vm-migration-techniques.md](research/01-vm-migration-techniques.md) — VM and GPU migration prior art
- [research/02-gpu-state-capture.md](research/02-gpu-state-capture.md) — GPU state analysis
- [research/03-migration-protocol.md](research/03-migration-protocol.md) — Protocol design
- [R10: Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Page table foundation
- [R15: Fault Tolerance](../../phase-09-hardening/R15-fault-tolerance/README.md) — Failure protection during migration
- [R19: Network Page Faults](../../phase-08-smart-memory/R19-network-page-faults/README.md) — Post-copy mechanism

## Open Questions

- [ ] Should migration support "cancel" mid-flight? (Pre-copy can be cancelled easily; post-copy cannot — destination already running.)
- [ ] What's the policy for migrating away from a node that's also a destination for another migration? (Avoid migration chains.)
- [ ] How does migration interact with R24 time-sliced sharing? (If multiple tenants share a GPU, migrate one tenant without disturbing others.)
- [ ] Should we support "warm standby" — keep a shadow context on a backup node, continuously synced, for instant failover?
