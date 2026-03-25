# R15 Research: Failure Detection and Recovery Workflows

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Design the failure detection and recovery pipeline for OuterLink. This covers: how we detect node/GPU/NIC failures, how we isolate the failure, how we reconstruct lost data from erasure-coded parity, and how we resume operations. Builds on R17's topology-aware scheduling and phi accrual failure detector design.

---

## 1. Failure Modes in a GPU Cluster

### Taxonomy of Failures

| Failure Type | Symptoms | Detection Method | Frequency | Severity |
|-------------|----------|-----------------|-----------|----------|
| **Full node crash** | All connections lost, heartbeat stops | Heartbeat timeout, RDMA async events | Medium | HIGH — all GPU data on node lost |
| **GPU crash (node alive)** | CUDA errors, ECC failures, driver reset | CUDA error codes, nvidia-smi monitoring | Medium | MEDIUM — other GPUs on node survive |
| **NIC failure** | RDMA connections drop, TCP RST | ibv_async_event, TCP keepalive | Low | MEDIUM — data intact but unreachable |
| **NIC port failure** | Single port down | IBV_EVENT_PORT_ERR | Low | LOW if dual-port (failover) |
| **Link degradation** | Increased latency, packet loss | Latency monitoring, CRC errors | Medium | LOW — performance impact, not data loss |
| **Process crash** | OuterLink daemon exits | PID monitoring, socket close | Medium | HIGH — equivalent to node loss for that GPU |
| **Memory corruption** | Silent data corruption | CRC/checksum on pages | Rare | CRITICAL — data integrity compromised |
| **Power failure** | All nodes down simultaneously | Heartbeat timeout (all) | Rare | CRITICAL — only NVMe checkpoints survive |

### Failure Rates in Consumer Hardware

Consumer GPUs (GeForce) lack the enterprise reliability features of datacenter GPUs:
- No ECC memory by default on most GeForce (RTX 3090 has it, but not all models)
- No hardware watchdog timers
- No BMC (Baseboard Management Controller) for out-of-band monitoring
- Driver crashes more common without enterprise-grade testing

**Expected MTBF for a 4-node consumer cluster:** ~24-72 hours under sustained load (extrapolated from datacenter failure rates of ~0.5-2% per GPU per month, adjusted for consumer hardware).

---

## 2. RDMA-Level Failure Detection

### ibv_async_event API

The InfiniBand verbs API provides hardware-level failure notification through asynchronous events. This is the fastest failure signal available — delivered by the NIC hardware itself.

**Key events for OuterLink:**

| Event | Meaning | Action |
|-------|---------|--------|
| IBV_EVENT_QP_FATAL | Queue Pair fatal error — cannot generate completions | Transition QP to Reset, reconnect |
| IBV_EVENT_QP_REQ_ERR | Transport error on request side | Log, attempt QP recovery |
| IBV_EVENT_QP_ACCESS_ERR | Access violation (bad remote key, address) | Bug — investigate, don't auto-recover |
| IBV_EVENT_DEVICE_FATAL | RDMA device unrecoverable error | Declare node as failed |
| IBV_EVENT_PORT_ERR | Physical port went down | Failover to alternate port/path |
| IBV_EVENT_PORT_ACTIVE | Physical port came back up | Re-establish connections |
| IBV_EVENT_PATH_MIG | Connection migrated to alternate path | Log, update routing tables |
| IBV_EVENT_COMM_EST | Communication established on QP | Connection recovery successful |

**Detection latency:** ibv_async_events are delivered within microseconds of the hardware detecting the failure. This is orders of magnitude faster than any heartbeat protocol.

### QP Error Recovery Flow

```
Normal Operation:  QP in RTS (Ready To Send) state
          |
     [Error occurs]
          |
          v
QP transitions to IBV_QPS_ERR state automatically
          |
     [ibv_async_event delivered]
          |
          v
Application calls ibv_modify_qp(QP, IBV_QPS_RESET)
          |
          v
Re-initialize QP: RESET -> INIT -> RTR -> RTS
          |
          v
Resume operations (if remote end is still alive)
```

**Important:** QP recovery only works if the remote node is still alive. If the remote node crashed, QP recovery will fail and we must declare the node as dead.

### ConnectX-5 Dual-Port Failover

ConnectX-5 has two ports. If one port fails:
1. IBV_EVENT_PORT_ERR fires for the failed port
2. If alternate path was configured (APM — Automatic Path Migration), the RC connection migrates
3. IBV_EVENT_PATH_MIG fires on the surviving port
4. Operations continue with zero application-level intervention

**OuterLink should:** Configure APM for all RC connections when both ports are available. This provides transparent single-port failover at the hardware level.

---

## 3. Application-Level Failure Detection

### Phi Accrual Failure Detector (from R17)

R17 already designed a phi accrual failure detector for topology-aware scheduling. This same mechanism serves fault tolerance.

**How it works:**
1. Each node sends periodic heartbeats (e.g., every 100ms)
2. The detector tracks inter-arrival times in a sliding window
3. Instead of binary alive/dead, it computes phi — a continuous suspicion level
4. phi = -log10(probability that the node is still alive given the elapsed time since last heartbeat)
5. As time passes without a heartbeat, phi increases exponentially

**Threshold mapping:**

| phi Value | Probability Node is Alive | Interpretation |
|-----------|--------------------------|----------------|
| 1 | 90% | Normal jitter |
| 2 | 99% | Probably fine |
| 3 | 99.9% | Slight concern |
| 5 | 99.999% | Very likely dead |
| 8 | 99.999999% | Almost certainly dead (Cassandra default) |

**For OuterLink:** Use phi threshold = 6 as default (~99.9999% confidence). On a local 100 Gbps network with ~2us RTT, this translates to:
- Normal heartbeat interval: 100ms
- Detection time at phi=6: ~300-600ms after last heartbeat (depends on jitter history)
- False positive rate: approximately 1 in 1,000,000

### Heartbeat Design

```
[Node A] ---heartbeat (100ms)--> [Node B]
[Node A] <--heartbeat (100ms)--- [Node B]

Heartbeat payload (minimal, fits in single packet):
- Node ID (8 bytes)
- Sequence number (8 bytes)
- Timestamp (8 bytes)
- GPU health flags (8 bytes per GPU)
- Memory pressure indicator (8 bytes)
- Total: ~48-64 bytes
```

**Transport:** Use RDMA UD (Unreliable Datagram) for heartbeats — lowest overhead, no connection state, broadcast-capable. If RDMA is unavailable, fall back to UDP.

### Combined Detection: RDMA Events + Heartbeat

```
Fastest signal: ibv_async_event (microseconds)
    |
    +--> If QP_FATAL or DEVICE_FATAL: immediately suspect node
    |
    +--> Attempt QP recovery (< 100ms)
    |
    +--> If recovery fails AND heartbeat phi > 6: declare node DEAD
    |
    +--> If recovery succeeds: node is alive, just had transient error

Slower signal: Heartbeat timeout (hundreds of ms)
    |
    +--> phi crosses threshold: declare node DEAD
    |
    +--> Used for cases where RDMA events don't fire (process crash, etc.)
```

**Detection latency budget:** Target < 1 second from failure to detection. RDMA events give us microseconds for hardware failures; phi accrual gives us ~300-600ms for software failures.

---

## 4. Recovery Workflow

### Phase 1: Detection and Isolation (~0-1 second)

```
1. Failure signal received (ibv_async_event or phi threshold)
2. Mark node as SUSPECTED in cluster membership
3. Fence the failed node:
   a. Cancel all in-flight RDMA operations to/from failed node
   b. Invalidate all QPs connected to failed node
   c. Mark all memory regions on failed node as INVALID in page table
4. Notify all surviving nodes of the failure
5. Pause operations that depend on failed node's data
```

**Fencing is critical:** We must ensure the failed node cannot corrupt data even if it comes back (split-brain). Use a generation counter — if the node rejoins with an old generation, reject it.

### Phase 2: Assessment (~1-2 seconds)

```
1. Inventory what was on the failed node:
   a. Which pages were exclusively on that node?
   b. Which pages had replicas/parity elsewhere?
   c. Which checkpoints were stored there?
2. Classify data by recoverability:
   a. RECOVERABLE: has parity/replica on surviving nodes
   b. CHECKPOINT-RECOVERABLE: lost, but last checkpoint exists
   c. LOST: no parity, no replica, no checkpoint (bug if this happens)
3. Prioritize recovery order:
   a. Hot pages (actively accessed) first
   b. Checkpoint data second
   c. Cold pages last
```

### Phase 3: Data Reconstruction (~2-30 seconds)

```
For each lost page/region, based on protection scheme:

XOR parity (single failure):
  1. Read page data from all surviving nodes in the parity group
  2. XOR them together to reconstruct the lost page
  3. Place reconstructed page on a surviving node (or hot spare)
  Time: ~microseconds per page (memory bandwidth limited)

Reed-Solomon parity (multi-failure):
  1. Identify which k fragments are available from (k+m) total
  2. Read k fragments from surviving nodes via RDMA
  3. Run RS decoding (ISA-L) to reconstruct lost fragments
  4. Place reconstructed data on surviving nodes
  Time: ~milliseconds per stripe (network + compute)

Checkpoint recovery (no parity available):
  1. Load last checkpoint from nearest surviving copy
  2. Apply any incremental deltas since checkpoint
  3. Some work since last checkpoint is lost (acceptable)
  Time: seconds to minutes depending on checkpoint size/age
```

### Phase 4: Resumption (~0-5 seconds)

```
1. Update page table with new locations for reconstructed data
2. Resume paused CUDA operations:
   a. For intercepted CUDA calls: return to the application
   b. For in-flight kernel launches: re-issue on alternate GPU
3. If training: resume from reconstructed checkpoint
4. Update cluster topology (R17) to reflect new node count
5. Rebalance data if needed (move some pages to reduce load)
```

### Total Recovery Timeline

| Scenario | Detection | Assessment | Reconstruction | Resume | Total |
|----------|-----------|------------|----------------|--------|-------|
| Single node, XOR parity | < 1s | ~1s | ~2-5s | ~1s | **< 10s** |
| Single node, RS parity | < 1s | ~1s | ~5-15s | ~1s | **< 20s** |
| Single node, checkpoint only | < 1s | ~1s | ~10-60s | ~5s | **< 90s** |
| Multi-node, RS parity | < 1s | ~2s | ~10-30s | ~2s | **< 40s** |
| Total cluster restart | N/A | ~5s | ~30-300s | ~10s | **< 5 min** |

---

## 5. Partial Failures

### GPU Crash, Node Alive

The GPU itself may crash (driver reset, ECC error, thermal throttle) while the host node remains operational.

**Detection:**
- CUDA API returns error codes (e.g., CUDA_ERROR_ECC_UNCORRECTABLE)
- nvidia-smi shows GPU in "fallen off the bus" state
- OuterLink's CUDA interception layer catches these errors

**Recovery:**
- VRAM contents are LOST (GPU reset clears VRAM)
- Host DRAM contents are preserved
- NIC and network connections remain active
- Reconstruct VRAM data from: local DRAM checkpoint > remote parity > remote checkpoint

**Advantage over full node crash:** Network connectivity survives, so reconstruction data can be pulled directly via RDMA. Much faster than discovering and routing around a dead node.

### NIC Failure, Node Alive

The RDMA NIC fails but the node (and its GPUs) are still operational.

**Detection:**
- IBV_EVENT_DEVICE_FATAL from verbs layer
- Heartbeats stop (since they use the NIC)
- But GPU monitoring (nvidia-smi) still shows healthy GPUs

**Recovery options:**
1. **Dual-port failover:** If ConnectX-5 has second port on separate network, switch to it
2. **Fallback to TCP:** If a secondary NIC exists (even a 1 Gbps NIC), use TCP for control traffic
3. **Graceful drain:** Copy data off the node via any available path before marking it failed

### Process Crash, Node and GPU Alive

OuterLink daemon crashes but the OS, GPU, and NIC are fine.

**Detection:**
- Socket close events on peer nodes
- Heartbeats stop
- But node responds to ICMP ping

**Recovery:**
- Restart OuterLink daemon on the same node
- VRAM may or may not be intact depending on whether the GPU context was destroyed
- If CUDA context survived: reconnect to existing GPU memory (cuCtxAttach or equivalent)
- If CUDA context died: treat as GPU crash, reconstruct from parity/checkpoint

---

## 6. Hot Spare Nodes

### Concept

Pre-staged nodes that are ready to accept reconstructed data immediately, avoiding the scramble to redistribute data across already-loaded surviving nodes.

### Design

```
Cluster: [Node 0] [Node 1] [Node 2] [Node 3] [Hot Spare]
                                                    |
                                              Pre-registered in cluster
                                              RDMA connections established
                                              No data assigned yet
                                              GPU memory allocated but empty

On Node 2 failure:
  [Node 0] [Node 1] [Node 3] [Hot Spare -> Node 2 replacement]
                                    |
                              Receives reconstructed data
                              Takes over Node 2's role
                              Becomes active member
```

### Hot Spare Overhead

- **VRAM:** 100% of spare GPU VRAM is reserved (24 GB for RTX 3090)
- **Network:** QPs established to all nodes (minimal memory, ~few KB per QP)
- **CPU:** Idle until activation
- **Power:** GPU at idle (~30W for RTX 3090 vs ~350W under load)

### When to Use Hot Spares

Hot spares make sense when:
- Cluster has N+1 or more nodes (spare capacity exists)
- Recovery time is critical (production inference serving)
- Node replacement takes too long (no one around to physically swap hardware)

For small consumer clusters (Pedro's setup), a hot spare might be one of the PCs that's not running at full capacity — its GPUs participate in the pool but with lower allocation, reserving capacity for failover.

---

## 7. Impact on In-Flight Operations

### What Happens to Active CUDA Kernels

When a node fails, operations in three states need handling:

**1. Kernels executing on the FAILED node's GPU:**
- Lost. Cannot recover mid-kernel execution.
- For training: roll back to last checkpoint, re-execute from there.
- For inference: return error to the requesting client, retry on another GPU.

**2. Kernels executing on SURVIVING nodes that READ from failed node:**
- If data was already transferred: kernel may complete normally.
- If data transfer was in-flight: RDMA operations return error (IBV_WC_RETRY_EXC_ERR).
- OuterLink intercept layer catches the error, reconstructs data from parity, retries the transfer.

**3. Kernels executing on SURVIVING nodes that WRITE to failed node:**
- Writes to failed node's memory will fail.
- Buffer writes locally, reconstruct the target page table entry to point to a new location (surviving node or hot spare), then replay the write.

### Consistency Considerations (from R19)

R19's SWMR (Single Writer, Multiple Reader) consistency model interacts with failures:

**If the failed node was the WRITER of a page:**
- Ownership transfers to a surviving node.
- Any dirty writes in the failed node's write buffer are lost.
- Use the last committed version (from parity or checkpoint).

**If the failed node was a READER of a shared page:**
- No data loss — the writer's copy is authoritative.
- Remove the failed node from the reader set in the directory.
- If the page was in Shared state, it remains Shared (minus one sharer).

**Directory recovery:**
- If the directory node fails, the directory must be reconstructed.
- Option A: Replicate directory on 2-3 nodes (simple, some overhead).
- Option B: Store directory in erasure-coded format.
- Option C: Reconstruct from page table entries on surviving nodes (slow but works).

---

## 8. Recommendation for OuterLink

### Detection Stack

```
Layer 1 (fastest):  ibv_async_event monitoring thread
                    - Fires on QP/device/port errors
                    - Detection latency: microseconds
                    - Triggers: immediate QP recovery attempt

Layer 2 (reliable): Phi accrual failure detector
                    - Heartbeat every 100ms via RDMA UD
                    - Phi threshold: 6 (default), configurable
                    - Detection latency: 300-600ms
                    - Triggers: node failure declaration

Layer 3 (backup):   TCP keepalive / ICMP
                    - Fallback when RDMA is unavailable
                    - Detection latency: 1-5 seconds
                    - Triggers: last-resort failure declaration
```

### Recovery Pipeline

```
DETECT (< 1s) --> FENCE (< 100ms) --> ASSESS (< 1s) --> RECONSTRUCT (< 30s) --> RESUME (< 5s)
   |                   |                    |                    |                     |
   |                   |                    |                    |                     |
ibv_event          Cancel RDMA         Inventory lost      RS/XOR decode          Update page
   +               Invalidate QPs      pages. Classify     from parity.           tables.
phi accrual        Update membership    by protection       Pull checkpoint.        Resume CUDA.
                   Generation bump      level               Place on survivor       Rebalance.
                                                            or hot spare
```

### Configuration Defaults

| Parameter | Default | Range | Rationale |
|-----------|---------|-------|-----------|
| Heartbeat interval | 100ms | 10ms - 1s | Balance detection speed vs overhead |
| Phi threshold | 6 | 3 - 10 | Higher = fewer false positives, slower detection |
| Max detection time | 1s | 100ms - 5s | Time from failure to declaration |
| Fencing timeout | 100ms | 10ms - 1s | Time to cancel all ops to failed node |
| Reconstruction priority | Hot pages first | Configurable | Match application access patterns |
| Hot spare activation | Automatic | Auto/Manual | Auto for production, manual for dev |

### Cluster Membership Protocol

Use a lightweight protocol for tracking who's in the cluster:

1. **Membership list** maintained by a coordinator (initially the first node, can migrate)
2. **Generation counter** increments on every membership change
3. **Quorum requirement:** Majority of nodes must agree on membership (prevents split-brain)
4. **Rejoin protocol:** A previously-failed node must present the current generation to rejoin. If it has a stale generation, it must re-sync all data.

For 2-3 node clusters, quorum is tricky (2-node cluster has no majority if one fails). Solution: designate an external witness (could be a lightweight process on a different machine, or use the NVMe-backed membership file as a tiebreaker).

---

## Open Questions

1. **Split-brain in 2-node clusters:** With only 2 nodes, if network partitions, both think the other is dead. Need a tiebreaker — external witness, shared storage, or "last writer wins" with reconciliation.

2. **Recovery vs. rebalancing:** After recovering data, should we redistribute to restore the pre-failure data layout? Or leave it wherever we put it and rebalance lazily? Eager rebalancing is disruptive; lazy rebalancing may leave hot spots.

3. **Cascading failure protection:** If node A fails and we're reconstructing, then node B fails during reconstruction — can we still recover? RS(k,2) handles this for data; but reconstruction operations themselves need to be idempotent and resumable.

4. **Testing failure recovery:** How to systematically test all failure modes without actually breaking hardware? Need a fault injection framework (kill processes, drop packets, corrupt pages).

5. **GPU health monitoring granularity:** CUDA error codes are coarse. Can we detect "GPU degradation" (thermal throttling, memory errors) before a full crash? nvidia-smi metrics could feed into the phi accrual detector.

---

## Related Documents

- R10: Memory Tiering (page table, tier locations)
- R12: Memory Deduplication (implicit redundancy for shared pages)
- R17: Topology-Aware Scheduling (phi accrual detector design, heartbeat protocol)
- R19: SWMR Consistency (coherency state during failure, directory recovery)
- R15 Research 01: Erasure Coding Algorithms (RS/XOR for reconstruction)
- R15 Research 02: Distributed Checkpointing (checkpoint-based recovery)
- R22: Live Migration (graceful migration vs crash recovery)

## References

- NVIDIA RDMA Aware Programming User Manual: https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/events
- NCCL Fault Tolerance: https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/
- Phi Accrual Failure Detector (Hayashibara et al., 2004): https://ieeexplore.ieee.org/document/1353004/
- Carbink: Fault-Tolerant Far Memory (OSDI '22): https://www.usenix.org/conference/osdi22/presentation/zhou-yang
- Gemini: Fast Failure Recovery (SOSP '23): https://dl.acm.org/doi/10.1145/3600006.3613145
- TorchPass Fault Tolerance: https://clockwork.io/blog/torchpass-workload-fault-tolerance/
- Training LLMs with Fault Tolerant HSDP on 100K GPUs: https://arxiv.org/html/2602.00277v1
- Cassandra Phi Accrual Implementation: https://manuel.bernhardt.io/2017/07/26/a-new-adaptive-accrual-failure-detector-for-akka
