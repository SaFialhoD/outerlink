# R22 Research: Live Migration Protocol Design

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Design the end-to-end protocol for migrating a running GPU workload from source node to destination node, integrating with OuterLink's existing subsystems (R10 page table, R15 fault tolerance, R19 network page faults).

---

## 1. Protocol Overview

### 1.1 Migration Phases

```
                           Source Node                     Destination Node
                           ───────────                     ────────────────
Phase 0: PREPARE           Announce migration intent  ───>  Reserve resources
                           (virtual addr ranges,            (create context shell,
                            VRAM capacity needed)            reserve address space)

Phase 1: PRE-COPY          Stream VRAM pages        ───>   Receive into staging
         (iterative)        Track dirty pages                Mark received pages
                           Continue running workload         Build page table mirror

Phase 2: QUIESCE           Wait for kernel boundary
                           Hold new kernel launches
                           cuStreamSynchronize all
                           Capture metadata snapshot

Phase 3: FINAL DELTA       Send remaining dirty     ───>   Apply final pages
                           pages + metadata                 Reconstruct full state

Phase 4: ACTIVATE                                          Re-create CUDA context
                                                           Load modules, functions
                                                           Create streams, events
                                                           Set up cuMemMap mappings
                                                   <───   Signal "ready"

Phase 5: SWITCHOVER        Stop intercepting calls
                           Redirect client to dest  ───>   Begin intercepting calls
                                                           Release held kernel launches

Phase 6: CLEANUP           Release VRAM allocations        (Running workload)
                           Tear down context
                           Report migration complete
```

### 1.2 Timeline for 24GB Workload

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 0: Prepare | ~10ms | Control plane messages, resource reservation |
| Phase 1: Pre-copy | 2-10s | 24GB at 12.5 GB/s = 1.9s raw; 3-5 rounds with convergence |
| Phase 2: Quiesce | 1-100ms | Wait for current kernel; inference = fast, training = wait for iteration |
| Phase 3: Final delta | 10-200ms | Depends on dirty rate; typically <1GB remaining |
| Phase 4: Activate | 5-50ms | Context creation, module reload, handle setup |
| Phase 5: Switchover | <1ms | Atomic pointer redirect in interception layer |
| Phase 6: Cleanup | ~10ms | Async, doesn't affect workload |
| **Total downtime** | **~20-350ms** | Phases 2-5 only (workload paused) |
| **Total migration time** | **~3-12s** | All phases |

---

## 2. Detailed Phase Design

### 2.1 Phase 0: Prepare

**Trigger:** Migration initiated by one of:
- User command: `outerlink migrate <workload-id> --to <node>`
- Automatic: load balancer detects imbalance (R17 topology-aware scheduling)
- Proactive: node health degradation detected (R15 fault tolerance)
- Maintenance: node drain before scheduled downtime

**Source node actions:**
1. Announce migration intent to cluster coordinator
2. Send workload manifest to destination:
   - Total VRAM allocated (sum of all allocations)
   - Number of allocations and their sizes
   - Virtual address ranges needed (for cuMemMap reservations)
   - Module count and total binary size
   - Estimated dirty rate (from R10 access tracking)

**Destination node actions:**
1. Verify sufficient free VRAM capacity
2. Reserve virtual address ranges via cuMemAddressReserve (matching source addresses)
3. Pre-allocate physical backing via cuMemCreate (empty, to be filled during pre-copy)
4. Create destination CUDA context (but don't activate yet)
5. ACK readiness to source

**Failure handling:**
- If destination lacks capacity: abort, try next-best node
- If virtual address conflict: attempt relocation (requires pointer rewriting — fallback to slower path)

### 2.2 Phase 1: Pre-Copy (Iterative VRAM Transfer)

This is the longest phase, running while the workload continues executing.

**Algorithm:**

```
round = 0
dirty_set = ALL_PAGES  // First round: everything is "dirty"

while round < MAX_ROUNDS and |dirty_set| > CONVERGENCE_THRESHOLD:
    // Transfer dirty pages to destination
    for page in dirty_set (ordered by access frequency, cold first):
        read page from source VRAM (cuMemcpyDtoH to pinned buffer)
        send over network to destination
        destination writes to pre-allocated VRAM (cuMemcpyHtoD)

    // Clear dirty tracking, start next round
    clear_dirty_flags()
    round += 1

    // Pages dirtied during this round form next round's dirty set
    dirty_set = get_dirty_pages()
```

**Page ordering strategy (using R10 data):**
- **Cold pages first:** Pages with lowest access frequency are transferred first — least likely to be dirtied again
- **Hot pages last:** Frequently-written pages are deferred to later rounds — each round catches their latest state
- **Read-only pages single-send:** Pages identified as read-only (model weights) transfer once and never dirty

**Rate limiting:**
- Migration should not starve the running workload of network bandwidth
- Configurable rate limit (default: 50% of available bandwidth)
- Adaptive: increase rate when workload network usage is low, decrease when high

**Dirty tracking integration with R10:**
- R10's page table tracks VRAM in 64KB pages
- During pre-copy, enable dirty tracking via PTE dirty flag
- Every write through the interception layer (cuMemcpy, kernel output) sets the dirty flag
- Between rounds, scan PTE dirty flags to build next round's dirty set

**Convergence analysis:**

| Round | Dirty Set Size (inference) | Dirty Set Size (training) |
|-------|---------------------------|--------------------------|
| 0 | 24 GB (everything) | 24 GB (everything) |
| 1 | ~500 MB (KV cache updates) | ~8 GB (gradients + optimizer) |
| 2 | ~500 MB (same pages) | ~8 GB (same iteration pattern) |
| 3 | ~500 MB (converged) | ~8 GB (not converging) |

For training workloads that don't converge well: switch to hybrid mode — proceed to quiesce and use post-copy for remaining pages.

### 2.3 Phase 2: Quiesce

The critical moment. Workload must be paused at a safe boundary.

**Quiesce procedure:**

```
1. Wait for natural kernel boundary:
   - Monitor cuLaunchKernel calls through interception layer
   - After current kernel completes (detected via cuStreamQuery/cuEventQuery),
     HOLD the next cuLaunchKernel call (don't forward to real driver)

2. Synchronize all streams:
   - cuStreamSynchronize on every active stream
   - This ensures all pending cuMemcpyAsync operations complete

3. Verify GPU is idle:
   - cuCtxSynchronize for belt-and-suspenders

4. Freeze interception layer:
   - All CUDA calls from the application are queued in our interception layer
   - No calls forwarded to real GPU driver
   - Application threads block on their next CUDA call (transparent to them)
```

**Downtime starts here.** The goal is to minimize the time between quiesce and switchover.

**Smart quiesce timing:**
- For inference: quiesce between forward passes (~10-100ms gap)
- For training: quiesce between iterations (after optimizer.step(), before next forward pass)
- The interception layer can detect iteration boundaries by watching for patterns: `cuLaunchKernel(backward)` → `cuLaunchKernel(optimizer)` → pause here

### 2.4 Phase 3: Final Delta

**Actions:**
1. One last dirty page scan — find pages dirtied since last pre-copy round
2. Transfer these pages to destination (they should be small — the convergence target)
3. Send metadata snapshot:
   - All context, module, function, stream, event, cuMemMap state (serialized, ~20MB)
   - R10 page table entries for migrated pages
   - Handle translation table (old handle → page mapping)

**Expected size:** For well-converged pre-copy:
- Inference: ~500MB dirty pages + 20MB metadata = ~520MB → ~42ms at 12.5 GB/s
- Training (non-converged): ~8GB dirty pages + 20MB metadata = ~8GB → ~640ms at 12.5 GB/s

### 2.5 Phase 4: Activate (Reconstruct State on Destination)

**Reconstruction sequence:**

```
1. Create CUDA context (cuCtxCreate with same flags)

2. Load all modules:
   for (module_binary, module_id) in metadata.modules:
       new_handle = cuModuleLoadData(module_binary)
       handle_map[module_id] = new_handle

3. Resolve function handles:
   for (func_name, func_id, module_id) in metadata.functions:
       new_handle = cuModuleGetFunction(handle_map[module_id], func_name)
       handle_map[func_id] = new_handle

4. Create streams:
   for (stream_id, priority) in metadata.streams:
       new_handle = cuStreamCreateWithPriority(priority)
       handle_map[stream_id] = new_handle

5. Create events:
   for (event_id, flags) in metadata.events:
       new_handle = cuEventCreate(flags)
       handle_map[event_id] = new_handle

6. Set up virtual memory mappings:
   // Address ranges already reserved in Phase 0
   for (va_range, phys_alloc, access_flags) in metadata.mem_maps:
       cuMemMap(va_range, phys_alloc)
       cuMemSetAccess(va_range, access_flags)

7. Install R10 page table entries (local tier)

8. Signal READY to source node
```

**Expected duration:** 5-50ms. Module loading is the bottleneck (cuModuleLoadData compiles PTX if not pre-compiled). Mitigation: use cubin (pre-compiled) when available, which loads in <1ms.

### 2.6 Phase 5: Switchover

The atomic moment where the workload moves.

**Mechanism:**
1. Source interception layer updates its routing table: `device_N → destination_node`
2. All queued CUDA calls (held since quiesce) are now forwarded to destination node
3. Application threads unblock — their CUDA calls proceed to the new GPU
4. From the application's perspective, nothing happened (same CUDA handles, same virtual addresses)

**This is atomic from the application's perspective.** The interception layer handles the routing change internally. No application code changes.

**Handle translation:**
- The interception layer already maintains a mapping: `app_handle → real_handle`
- During switchover, `real_handle` is updated to point to destination node's handles
- Application still uses the same `app_handle` it always did

### 2.7 Phase 6: Cleanup

**Source node:**
1. Release all VRAM allocations (cuMemFree)
2. Destroy CUDA context (cuCtxDestroy)
3. Report migration complete to cluster coordinator
4. Update R15's fault tolerance metadata (parity locations may need recomputation)

**Asynchronous** — doesn't affect the workload running on destination.

---

## 3. Integration with R19: Post-Copy via Network Page Faults

The hybrid approach uses R19's page fault mechanism as a safety net:

**Scenario:** Pre-copy hasn't finished transferring all pages when we decide to switchover (e.g., training workload with high dirty rate that won't converge).

**Flow:**
1. Complete Phase 2 (quiesce) and Phase 3 (final delta for hot pages only)
2. Phase 4: Activate on destination, but mark un-transferred pages as REMOTE in R10 page table
3. Phase 5: Switchover — workload resumes on destination
4. When a kernel accesses a REMOTE page: R19's page fault handler fetches it from source
5. Background thread continues pushing remaining pages from source (like post-copy background push)

**Advantages:**
- Reduces downtime dramatically for large/dirty workloads
- Workload resumes immediately after hot pages are transferred
- Cold pages trickle in without affecting latency (they're cold — rarely accessed)

**Risk mitigation:**
- If source fails during post-copy: R15's erasure coding can reconstruct lost pages from parity on other nodes
- Page fault latency (~2-10us for RDMA) is acceptable for cold pages that are infrequently accessed

---

## 4. Performance Model

### 4.1 Total Migration Time

```
T_total = T_precopy + T_quiesce + T_delta + T_activate + T_switchover

where:
  T_precopy = VRAM_size / (bandwidth * rate_limit_fraction)
            = 24GB / (12.5 GB/s * 0.5)
            = ~3.8s

  T_quiesce = kernel_completion_time (workload-dependent)
            = 1-100ms

  T_delta   = dirty_pages_remaining / bandwidth
            = 500MB / 12.5 GB/s
            = ~40ms (inference)
            = 8GB / 12.5 GB/s
            = ~640ms (training, worst case)

  T_activate = context_reconstruction_time
             = ~5-50ms

  T_switchover = ~0.1ms (pointer redirect)
```

### 4.2 Downtime (Application-Visible Pause)

```
T_downtime = T_quiesce + T_delta + T_activate + T_switchover

Inference:  1ms + 40ms + 10ms + 0.1ms  = ~51ms
Training:   100ms + 640ms + 50ms + 0.1ms = ~790ms
Hybrid:     1ms + 40ms (hot only) + 10ms + 0.1ms = ~51ms (remaining pages via post-copy)
```

### 4.3 Impact on Running Workload During Pre-Copy

Pre-copy competes for:
1. **Network bandwidth:** Configurable rate limit. At 50%, workload keeps 50% for its own transfers.
2. **PCIe bandwidth:** cuMemcpyDtoH for page reads shares PCIe with kernel memory accesses. Impact: ~5-15% throughput reduction depending on workload memory intensity.
3. **Host CPU:** Minimal — pinned memory DMA is CPU-light.

---

## 5. Failure Handling

| Failure | Phase | Recovery |
|---------|-------|----------|
| Destination node dies during pre-copy | Phase 1 | Abort migration, workload continues on source, no impact |
| Source node dies during pre-copy | Phase 1 | R15 erasure coding reconstructs VRAM on another node, restart migration to new destination |
| Network partition during pre-copy | Phase 1 | Pause migration, resume when connectivity returns |
| Destination rejects (out of memory) | Phase 0 | Try next-best node from R17's topology |
| Source dies after switchover but before cleanup | Phase 6 | No impact — workload already running on destination, source resources leak but will be cleaned up by R15's failure detector |
| Destination dies after switchover | Phase 6+ | Standard R15 failure recovery — not migration-specific |
| Quiesce timeout (kernel won't finish) | Phase 2 | Configurable timeout; option to force-kill kernel (data loss) or abort migration |

---

## 6. Migration Triggers and Decision Logic

### 6.1 Trigger Types

| Trigger | Source | Urgency | Approach |
|---------|--------|---------|----------|
| Manual drain | User/admin CLI | Low | Full pre-copy, minimize downtime |
| Load balancing | R17 topology scheduler | Low | Full pre-copy, rate-limited to avoid disruption |
| Proactive failover | R15 health monitor | Medium | Accelerated pre-copy (higher rate limit) |
| Emergency failover | R15 failure detection | HIGH | Minimal pre-copy or pure post-copy — prioritize speed |

### 6.2 Destination Selection

Criteria (ordered by priority):
1. Sufficient free VRAM
2. Network proximity (lowest latency hop from source)
3. Current load (least-loaded node)
4. Hardware compatibility (same or higher compute capability)
5. Existing data locality (does destination already have some pages from dedup/cache?)

---

## 7. Wire Protocol Messages

| Message | Direction | Payload |
|---------|-----------|---------|
| `MigrateRequest` | Source → Coordinator | workload_id, reason, preferred_dest |
| `MigrateAssign` | Coordinator → Dest | workload_manifest (VRAM size, address ranges) |
| `MigrateReady` | Dest → Source | reserved_addresses, context_id |
| `PageTransfer` | Source → Dest | page_id, tier, data (64KB) |
| `PageBatch` | Source → Dest | [page_id, data]* (batch for efficiency) |
| `DirtyBitmap` | Source → Dest | bitmap of dirty pages (for round tracking) |
| `QuiesceComplete` | Source → Dest | metadata snapshot (all state) |
| `FinalDelta` | Source → Dest | remaining dirty pages + metadata |
| `ActivateAck` | Dest → Source | handle translation table |
| `SwitchoverExecute` | Source → Dest (+ Client) | atomic redirect command |
| `CleanupComplete` | Source → Coordinator | migration stats |

---

## 8. Implementation Phases

### Phase 1: Basic Migration (stop-and-copy)
- Stop workload, copy all VRAM, reconstruct on destination, resume
- No pre-copy, no dirty tracking, maximum downtime
- Proves the state capture/reconstruct pipeline works
- **Estimated effort:** Medium

### Phase 2: Pre-Copy Migration
- Add iterative pre-copy with interception-level dirty tracking (conservative, per-buffer)
- Reduces downtime significantly
- **Estimated effort:** Medium (dirty tracking is the core addition)

### Phase 3: Hybrid Pre+Post Copy
- Integrate R19 page fault mechanism for post-copy
- PTE-level dirty tracking (exact, per-page)
- Hot/cold page classification from R10 access data
- **Estimated effort:** Low (if R19 is already implemented)

### Phase 4: Intelligent Migration
- ML-predicted dirty rates to optimize pre-copy round count
- Proactive destination preparation (pre-warm caches, pre-fetch model weights from dedup)
- Zero-downtime for inference workloads (pipeline switch: old and new overlap briefly)
- **Estimated effort:** High (optimization, not correctness)

---

## Related Documents

- [01-vm-migration-techniques.md](01-vm-migration-techniques.md) — Prior art, dirty tracking mechanisms
- [02-gpu-state-capture.md](02-gpu-state-capture.md) — What state needs to move
- [R10: Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Page table, dirty flags, access patterns
- [R15: Fault Tolerance](../../../phase-09-hardening/R15-fault-tolerance/README.md) — Erasure coding for failure during migration
- [R17: Topology-Aware Scheduling](../../../phase-08-smart-memory/README.md) — Destination node selection
- [R19: Network Page Faults](../../../phase-08-smart-memory/R19-network-page-faults/README.md) — Post-copy page fault mechanism

## Open Questions

- [ ] Should we support migrating a workload that spans multiple GPUs (multi-GPU migration)? Or only single-GPU-to-single-GPU initially?
- [ ] How does migration interact with NCCL collective operations (R20)? If the workload is in the middle of an all-reduce, migration must wait.
- [ ] Can we do zero-downtime migration for stateless inference by briefly running on both source and destination (request-level cutover instead of mid-execution)?
- [ ] What's the minimum RDMA bandwidth where live migration is practical? Below some threshold, stop-and-copy might be preferable.
- [ ] How do we handle migration of workloads that use CUDA IPC (shared memory between processes)?
