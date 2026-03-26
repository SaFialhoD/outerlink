# R29 Research: Multicast for ML Workloads

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Analyze specific ML workloads where multicast provides measurable benefit over unicast or tree-based algorithms. Covers model weight broadcast, AllReduce decomposition, NCCL integration, OpenDMA compatibility, bandwidth savings quantification, and when multicast loses to ring/tree algorithms.

---

## 1. Model Weight Broadcast (Inference)

### The Use Case

In multi-GPU inference, every GPU needs the same model weights:
- LLM inference: All GPUs get the full model (tensor parallelism splits layers, but each GPU needs its assigned layers)
- Pipeline parallelism: Each stage gets its assigned layers
- Data parallelism: Every GPU gets the ENTIRE model

### Current Approach (Without Multicast)

```
Master node (has model loaded):
    For each worker node:
        RDMA WRITE model weights to worker (sequential or parallel)

Time for 8 nodes, 70B model (140GB in FP16):
    Sequential: 8 x 140GB / 12.5 GB/s = 89.6 seconds
    Parallel (link saturated): 140GB / 12.5 GB/s x 8 = 89.6 seconds (same — link is the bottleneck)
    Tree (3 levels): 140GB / 12.5 GB/s x 3 = 33.6 seconds
```

### With Multicast

```
Master node:
    Multicast model weights to ALL nodes simultaneously

Time for 8 nodes, 140GB model:
    Multicast: 140GB / 12.5 GB/s = 11.2 seconds
    + retransmission overhead (if any loss): ~0.5-2 seconds
    Total: ~12-14 seconds
```

### Speedup

| Method | Time (140GB, 8 nodes) | Speedup vs Sequential |
|--------|----------------------|----------------------|
| Sequential unicast | 89.6s | 1x |
| Tree broadcast (3 levels) | 33.6s | 2.7x |
| **Multicast** | **~12s** | **~7x** |

### Practical Considerations

- Model weights are read-only after loading — perfect for multicast (no consistency issues)
- Loading happens once at startup or model swap — not latency-critical
- But 89.6s vs 12s is the difference between "annoying" and "instant" for model serving

### Weight Distribution Pattern

For tensor-parallel inference across 8 GPUs, each GPU needs different weight shards:
- GPU 0: layers 0-7 attention weights
- GPU 1: layers 0-7 MLP weights
- etc.

This means 8 different multicast groups (one per shard), each sending different data to specific receivers. Multicast still wins: each group has 1 sender and 1 receiver (degenerates to unicast), except for replicated data (embeddings, layer norms shared across all GPUs).

**Best multicast target:** Replicated weights (embeddings, layer norms, output projection). These go to ALL GPUs identically — perfect multicast.

---

## 2. AllReduce Decomposition

### AllReduce = ReduceScatter + AllGather

The AllReduce collective can be decomposed:
1. **ReduceScatter:** Each node contributes data, result is scattered (each node gets 1/N of the reduced result)
2. **AllGather:** Each node broadcasts its 1/N chunk to all others (all nodes end up with the full result)

### Multicast for AllGather Phase

The AllGather phase is a natural multicast candidate:
- Each node multicasts its 1/N chunk to all other nodes
- N simultaneous multicast sends (one per node)
- Total data on wire: N x (data_size/N) = data_size (same as ring AllGather)

**Comparison:**

| Algorithm | Steps | Total Data on Wire | Latency |
|-----------|-------|-------------------|---------|
| Ring AllGather | N-1 | data_size x (N-1)/N | O(N) |
| Tree AllGather | log(N) | data_size x log(N) | O(log N) |
| Multicast AllGather | 1 | data_size | O(1) + reliability |

Multicast AllGather has O(1) latency (one shot) and optimal data volume. But reliability overhead may negate the latency advantage.

### ReduceScatter: No Multicast Benefit

ReduceScatter involves each node sending different data to different destinations — this is inherently unicast (point-to-point). Multicast doesn't help here.

### Net AllReduce Improvement

If multicast halves the AllGather phase time:
- AllReduce = ReduceScatter + AllGather
- ReduceScatter takes ~50% of AllReduce time
- AllGather takes ~50% of AllReduce time
- Multicast optimizes AllGather only
- **Net improvement: ~25% faster AllReduce** (if AllGather is halved)

This is modest but significant for training workloads where AllReduce is the communication bottleneck.

---

## 3. NCCL Integration (R20 Backend)

### NCCL Broadcast Collective

NCCL's Broadcast operation distributes data from one root GPU to all others. Current implementation:

```
NCCL Ring Broadcast:
    Root -> GPU1 -> GPU2 -> ... -> GPUN (chain, not ring)
    Latency: O(N) steps
    Each step: RDMA WRITE (reliable, fast)
```

### Multicast-Accelerated Broadcast

OuterLink's NCCL backend (R20) could use multicast for the Broadcast collective:

```
OuterLink Multicast Broadcast:
    Root multicasts to ALL GPUs simultaneously
    NACK-based reliability ensures complete delivery
    Latency: O(1) + reliability overhead
```

### NCCL Algorithm Selection

NCCL dynamically selects algorithms based on message size and node count:

| Message Size | Nodes | NCCL Default | Multicast Advantage |
|-------------|-------|-------------|---------------------|
| < 64KB | Any | Tree (LL protocol) | Minimal — LL latency is already ~5us |
| 64KB - 256KB | 2-8 | Tree (LL128) | Small — tree is already fast |
| 256KB - 4MB | 2-8 | Ring (Simple) | Moderate — multicast saves ring steps |
| > 4MB | 2-8 | Ring (Simple) | Large — multicast saves significant time |

Multicast's biggest NCCL benefit is for **large Broadcast operations** (> 4MB) on clusters with more nodes.

### Integration Architecture

```
NCCL Backend Plugin (R20):
    ncclBroadcast():
        if message_size > 4MB AND num_nodes > 2:
            Use OuterLink multicast path
        else:
            Use standard tree/ring path
```

### NCCL CollNet for Multicast

NCCL supports a "CollNet" plugin interface for network-offloaded collectives. This is designed for SHARP but could be adapted for OuterLink's multicast:
- CollNet provides `allreduce`, `reducescatter`, `allgather` hooks
- OuterLink implements the CollNet interface using multicast for broadcast/allgather

---

## 4. Multicast + OpenDMA (BAR1 Targeting)

### Can Multicast Target BAR1 Addresses?

**Not directly with UD multicast.**

UD multicast uses SEND/RECV, not RDMA WRITE. This means:
- Sender: `ibv_post_send` with `IBV_WR_SEND` on UD QP to multicast group
- Receiver: `ibv_post_recv` on UD QP with pre-posted receive buffers

The receive buffers are in host memory (pinned via `ibv_reg_mr`), not GPU VRAM/BAR1. Data path:

```
Current (UD multicast, no OpenDMA):
    Sender GPU VRAM -> cudaMemcpy -> Host pinned -> UD SEND -> wire
    -> UD RECV -> Host pinned -> cudaMemcpy -> Receiver GPU VRAM
```

This is the host-staged path — not OpenDMA.

### Can We Do Better?

**Option A: UD SEND to BAR1-registered buffers**

If we register GPU VRAM (via BAR1) as the receive buffer for UD QP:
```
ibv_reg_mr(pd, bar1_gpu_vram_addr, size, IBV_ACCESS_LOCAL_WRITE);
ibv_post_recv(ud_qp, &recv_wr_pointing_to_bar1, ...);
```

The NIC would DMA received UD packets directly to GPU VRAM via BAR1. This is theoretically possible but:
- UD messages include a 40-byte GRH (Global Route Header) prepended to the payload
- The GRH would corrupt the first 40 bytes of GPU data in the target buffer
- Receiver must account for GRH offset (first 40 bytes are header, data starts at offset 40)
- GPU VRAM writes via BAR1 have specific alignment requirements

**Feasibility: Possible but requires careful buffer management for GRH handling.**

**Option B: Multicast to host, GPU copy on receive**

Standard approach — use host pinned memory for UD receive, then cudaMemcpy to GPU:
```
UD RECV -> host pinned buffer -> cudaMemcpyAsync -> GPU VRAM
```

This is simpler and proven. The cudaMemcpy adds ~5us latency for small messages but is pipelined with network arrival for large transfers.

**Option C: Unicast RDMA WRITE for repair (OpenDMA)**

The hybrid model: multicast delivers bulk data to host buffers, then:
- Happy path: host -> GPU copy (fast, pipelined)
- Repair: unicast RC RDMA WRITE directly to BAR1 (OpenDMA, zero-copy)

This combines multicast speed for initial delivery with OpenDMA efficiency for retransmission.

### Verdict

For Phase 1: Option B (multicast to host, GPU copy). Simple, works.
For Phase 2: Option C (hybrid — multicast bulk + OpenDMA repair). Best of both worlds.
Option A is interesting but the GRH handling complexity isn't worth the small latency savings.

---

## 5. Bandwidth Savings Quantification

### Model Weight Broadcast

| Cluster Size | Unicast Bandwidth (per broadcast) | Multicast Bandwidth | Savings |
|-------------|----------------------------------|--------------------|---------|
| 2 nodes | 1x | 1x | 0% (degenerates to unicast) |
| 4 nodes | 3x | 1x | 67% |
| 8 nodes | 7x | 1x | 86% |
| 16 nodes | 15x | 1x | 93% |
| 32 nodes | 31x | 1x | 97% |

At 8 nodes (OuterLink's target), multicast saves 86% of sender bandwidth. This matters because the sender's 100Gbps link is the bottleneck — sending to 7 nodes sequentially takes 7x longer.

### AllGather in AllReduce

For AllGather of gradient chunks (each node broadcasts 1/N of gradients):

| Cluster Size | Ring AllGather Steps | Multicast AllGather Steps | Speedup |
|-------------|---------------------|--------------------------|---------|
| 2 | 1 | 1 | 1x |
| 4 | 3 | 1 | 3x |
| 8 | 7 | 1 | 7x |
| 16 | 15 | 1 | 15x |

But each multicast step has reliability overhead. Net speedup depends on loss rate:
- 0% loss: Full Nx speedup
- 1% loss: ~0.9Nx speedup (NACK repair is cheap)
- 10% loss: ~0.5Nx speedup (significant retransmission)

### KV Cache Distribution (Inference)

For distributed KV cache in long-context inference:
- Each token's KV cache is computed on one node, needed by all
- Multicast distributes each token's KV to all nodes simultaneously
- Savings: Same as model weight broadcast (proportional to cluster size)

### When Multicast Doesn't Save Bandwidth

- **Point-to-point transfers:** Only one receiver — multicast = unicast
- **Scatter operations:** Different data to different receivers — must use unicast
- **ReduceScatter:** Each node receives different partial results — unicast only

---

## 6. When Multicast Loses to Ring/Tree

### Small Messages (< 64KB)

- **UD overhead:** 40-byte GRH + sequence numbers + reliability = significant per-message cost
- **Tree/Ring with LL protocol:** NCCL's LL (Low Latency) protocol achieves ~5us for small messages
- **Multicast:** UD SEND latency + reliability tracking adds ~10-20us
- **Verdict:** Tree wins for small messages

### Very Low Node Count (2 nodes)

- Multicast degenerates to unicast (only one receiver)
- Tree/ring also degenerates to direct transfer
- No advantage either way
- **Verdict:** Tie — use whatever's simpler (unicast RC RDMA WRITE)

### High Loss Networks

If RoCE multicast loss rate > 5%:
- Retransmission overhead dominates
- FEC overhead is significant
- Tree broadcast with RC RDMA WRITE has 0% loss (reliable transport)
- **Verdict:** Tree wins on unreliable networks

### Training AllReduce (Latency-Critical)

- Training step time is bounded by AllReduce latency
- NCCL's ring/tree AllReduce is highly optimized with overlapping computation
- Multicast AllGather adds reliability overhead that may negate the speedup
- NCCL already uses CollNet/SHARP when available
- **Verdict:** Tree/ring wins unless multicast reliability overhead < 10% of transfer time

### Summary Decision Matrix

| Scenario | Nodes | Message Size | Multicast? | Why |
|----------|-------|-------------|-----------|-----|
| Model weight load | 4-8 | GBs | **YES** | 4-7x speedup, latency-tolerant |
| NCCL Broadcast | 4-8 | > 4MB | **YES** | Significant speedup, one-to-all |
| NCCL AllGather | 4-8 | > 1MB | **MAYBE** | Depends on loss rate |
| NCCL AllReduce | 4-8 | Any | **NO** | Tree/ring already optimized |
| Small collectives | Any | < 64KB | **NO** | UD overhead too high |
| 2-node cluster | 2 | Any | **NO** | No advantage |
| KV cache distribute | 4-8 | 64KB-1MB | **MAYBE** | Moderate benefit, frequent |

---

## 7. Verdict for OuterLink

### Primary Multicast Use Cases

1. **Model weight broadcast at startup** — Largest benefit. 7x speedup for 8-node cluster loading 140GB model. Latency-tolerant (seconds are OK). Worth implementing first.

2. **NCCL Broadcast collective** — Moderate benefit. 3-7x speedup depending on cluster size. Worth implementing in R20 backend.

3. **Replicated data distribution** — Embeddings, layer norms, config data shared across all nodes. Small data but frequent — multicast amortizes per-node cost.

### Secondary Use Cases (Evaluate Later)

4. **NCCL AllGather** — Depends on loss rate. If < 1% loss, worth it. Benchmark on Pedro's network first.

5. **KV cache distribution** — Moderate benefit for long-context inference. Implement after primary use cases are proven.

### Not Worth Multicast

- AllReduce (tree/ring is better)
- Small messages (< 64KB)
- Point-to-point transfers
- 2-node clusters

### Implementation Priority

```
Phase 1: Model weight broadcast via multicast + NACK
    -> Biggest bang for the buck
    -> Validates multicast + reliability on Pedro's network
    -> If loss rate is unacceptable, fall back to tree (still useful)

Phase 2: NCCL Broadcast collective via multicast
    -> Integrate with R20 backend
    -> Use Phase 1's reliability layer

Phase 3: NCCL AllGather via multicast (if Phase 1/2 show acceptable loss rates)
    -> More complex (N simultaneous multicasts)
    -> Only if benchmarks justify it
```

---

## Related Documents

- [01-rdma-multicast-fundamentals.md](./01-rdma-multicast-fundamentals.md) — Multicast basics and ConnectX-5 limits
- [02-reliable-multicast.md](./02-reliable-multicast.md) — Reliability approaches
- [R20 NCCL Backend](../../../R20-nccl-backend/) — Broadcast and AllGather implementation
- [R12 Memory Deduplication](../../../R12-memory-dedup/) — Shared read-only pages (multicast targets)
- [R17 Topology-Aware Scheduling](../../../R17-topology-scheduling/) — Network topology for multicast routing
- [R28 Scatter-Gather](../../R28-scatter-gather-dma/) — Can combine scatter-gather with multicast receive

## Open Questions

- [ ] Benchmark UD multicast loss rate on Pedro's network (idle and under load)
- [ ] What's the actual GRH overhead for UD SEND to BAR1? Is 40-byte offset manageable?
- [ ] Can NCCL's CollNet interface support our multicast backend, or do we need a custom plugin?
- [ ] Should multicast groups be persistent (pre-created at cluster join) or ephemeral (per-operation)?
- [ ] Memory cost of sender-side retransmission buffer for 140GB model broadcast
