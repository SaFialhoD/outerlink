# R15 Research: Distributed Checkpointing for Training State Recovery

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Evaluate checkpointing strategies for preserving training state (model weights, optimizer states, gradients, data loader position) across OuterLink's distributed GPU cluster. The goal: when a node fails mid-training, recover and resume with minimal lost work.

---

## 1. The Checkpointing Problem in Distributed Training

### Why Checkpointing Matters

Large model training runs are long (days to months) and use many GPUs. Failures are frequent:
- OPT-175B training (992 A100 GPUs): ~110 failures over 2 months
- At 1,024 GPUs, mean time between failures (MTBF) can be hours, not days
- Without checkpointing, any failure restarts training from scratch

### What Must Be Checkpointed

| Component | Size (7B model) | Size (70B model) | Update Frequency |
|-----------|-----------------|-------------------|------------------|
| Model weights (FP16) | ~14 GB | ~140 GB | Every step |
| Optimizer states (Adam FP32) | ~56 GB | ~560 GB | Every step |
| Gradients | ~14 GB | ~140 GB | Transient (computed each step) |
| Data loader state | ~KB | ~KB | Every step |
| LR scheduler state | ~bytes | ~bytes | Every step |
| RNG states | ~KB | ~KB | Every step |

**Total checkpoint size:** ~84 GB for 7B, ~840 GB for 70B model (with Adam optimizer).

With ZeRO sharding across N GPUs, each GPU holds 1/N of this data.

---

## 2. PyTorch Distributed Checkpoint (DCP)

### Overview

PyTorch DCP (torch.distributed.checkpoint) is the official distributed checkpointing solution. It enables saving and loading model state from multiple ranks in parallel, with load-time resharding across different cluster topologies.

### Key Features

- **Parallel I/O:** Each rank saves its own shard simultaneously
- **Load-time resharding:** Save with N GPUs, load with M GPUs (different topology)
- **Sharded state dict:** Each rank only materializes its portion
- **Async checkpointing:** Decouple snapshot (GPU->CPU copy) from persist (CPU->storage)
- **Single file per rank writer:** Reduces metadata overhead

### Performance Numbers

| Metric | Before DCP (PyTorch 1.13) | With DCP | Speedup |
|--------|--------------------------|----------|---------|
| 11B model checkpoint save | ~30 minutes | ~25 seconds | 72x |
| 30B model + optimizer + dataloader | N/A (timeout) | ~3 minutes | Enabled |
| Async checkpoint overhead | N/A | 19x reduction vs sync | — |

### Async Checkpointing Pipeline

```
Step N:   [Forward] [Backward] [Optimizer] [Snapshot GPU->CPU] [Continue Step N+1]
                                                    |
Background:                                   [Persist CPU->Storage]
```

The snapshot (GPU->CPU memcpy) is the synchronous part — typically ~1-3 seconds for multi-GB models. The persist (write to disk/S3) happens in background threads and can take 10-60+ seconds but doesn't block training.

### DCP Limitations

- Requires all ranks to participate in save/load (collective operation)
- Load-time resharding supports data parallelism changes but NOT model parallelism degree changes
- Object storage backends (S3) have eventual consistency — DCP handles this but adds complexity

### Relevance to OuterLink

DCP is the checkpoint FORMAT we should support — it's the standard. But OuterLink's innovation is WHERE checkpoints go: instead of remote storage, we checkpoint to the cluster's own distributed memory (DRAM across nodes), achieving much faster save/restore.

---

## 3. DeepSpeed ZeRO Checkpointing

### ZeRO Sharding Recap

ZeRO partitions training state across GPUs:
- **Stage 1:** Optimizer states partitioned
- **Stage 2:** + Gradients partitioned
- **Stage 3:** + Model parameters partitioned (+ optionally activations)

Each GPU only checkpoints its shard, creating distributed checkpoints.

### Universal Checkpointing (UCP)

DeepSpeed's UCP solves the "checkpoint portability" problem:
- Save with ZeRO-3 on 64 GPUs, resume with ZeRO-2 on 32 GPUs
- Uses "atomic checkpoints" and pattern-based reconfiguration
- Reconfiguration overhead is negligible
- Used to train BLOOM 176B and Microsoft Phi-3

### DeepSpeed Checkpoint Characteristics

| Feature | Standard ZeRO Checkpoint | Universal Checkpoint |
|---------|-------------------------|---------------------|
| Cross-topology resume | No | Yes |
| FP16 weight consolidation | Requires gather (slow, memory-heavy) | Native |
| Save overhead | Low (each GPU saves its shard) | Same as standard |
| Load flexibility | Same parallelism only | Any parallelism strategy |

### Activation Checkpointing (Separate Concern)

DeepSpeed also supports activation checkpointing — recomputing activations during backward pass instead of storing them. This trades compute for memory:
- CPU offload: checkpoint activations to CPU RAM
- Partitioned: split activation checkpoints across model-parallel GPUs
- Contiguous memory buffer: optimize memory layout

This is orthogonal to fault-tolerance checkpointing but relevant because it affects how much VRAM is available for parity storage.

---

## 4. In-Memory Checkpointing Systems

### Gemini (SOSP '23) — Key Reference

Gemini is the most relevant system for OuterLink's checkpointing design.

**Core idea:** Checkpoint to CPU memory of host machines instead of remote storage. CPU memory has orders of magnitude more bandwidth than storage I/O.

**Architecture:**
```
GPU Memory (hot) --> CPU Memory (local) --> CPU Memory (remote) --> Persistent Storage (cold)
     ^                    ^                       ^                        ^
     |                    |                       |                        |
  Training data     Primary checkpoint    Redundant checkpoint      Durable backup
```

**Key innovations:**
1. **Near-optimal checkpoint placement:** Maximize probability of recovering from in-memory checkpoints after failure. Uses optimization algorithm to decide which nodes store redundant copies.
2. **Checkpoint traffic scheduling:** Minimize interference between checkpoint network traffic and training communication. Schedules checkpoint transfers during communication gaps.

**Performance:**
- **Recovery speedup:** 13x faster than existing solutions
- **Checkpoint frequency:** Every iteration (optimal frequency) with no training throughput overhead
- **Checkpoint time reduction:** From 40 minutes to 3 seconds using in-memory hierarchy

**Why this matters for OuterLink:** OuterLink already has a distributed memory pool across nodes. In-memory checkpointing is a natural extension — we can store checkpoints in partner nodes' DRAM without any new infrastructure.

### CheckFreq

**Core idea:** Decouple checkpointing into snapshot (GPU->CPU) and persist (CPU->storage), pipeline them with computation.

**Performance:**
- Checkpoint every 14-19 iterations
- Overhead under 3.5%
- Recovery time: hours to seconds

### LowDiff (2025)

**Core idea:** Incremental/differential checkpointing — only save what changed since last checkpoint.

**Performance:**
- Reduces training time by 68.2% vs CheckFreq, 46.1% vs Gemini
- Per-iteration overhead: 2.4-3.1%
- Key insight: most parameters change very little between iterations

**Why this matters:** For OuterLink, incremental checkpointing means we only need to transmit deltas over the network. With 100 Gbps RDMA, even full checkpoints are fast, but deltas make it essentially free.

---

## 5. NCCL Fault Tolerance and Elastic Training

### NCCL Native Features (NCCL 2.27+)

- **ncclCommShrink:** Resize communicators to exclude faulted ranks without full restart
- **Non-blocking communicators:** Configurable timeouts for async error handling
- **ncclCommAbort:** Clean abort of failed communicators

**Recovery workflow:**
1. Detect failure (NCCL collective timeout or explicit error)
2. Call ncclCommAbort on affected communicators
3. Call ncclCommShrink to create new communicator without failed ranks
4. Resume training from last checkpoint with reduced world size

**NCCL restart overhead at scale:**
- 16K GPUs: ~17 seconds to re-establish connections
- 98K GPUs: ~200 seconds
- For OuterLink (2-8 nodes, 2-8 GPUs): < 1 second

### TorchFT (Late 2024)

Per-step fault tolerance for DDP training. Each step is a distributed transaction — on failure, the replica group is removed and training continues.

### TorchPass (September 2025)

NCCL net plugin for transparent fault tolerance:
- Intercepts network failures, reroutes traffic to alternative paths
- Reconstructs failed worker state from healthy workers' just-in-time checkpoints
- No model-aware checkpointing needed

### PyTorch torchrun (Elastic Training)

- Auto-restarts all processes from last snapshot on failure
- Supports elastic scaling (add/remove nodes)
- Works with DDP and FSDP

---

## 6. Checkpoint Placement Across Memory Tiers

### OuterLink's Memory Hierarchy (from R10)

```
Tier 0: Local VRAM (fastest, smallest, most precious)
Tier 1: Pinned RAM (host CPU memory, pinned for DMA)
Tier 2: Local DRAM (regular host memory)
Tier 3: Remote DRAM/VRAM (other nodes via network)
Tier 4: NVMe (persistent, slowest)
```

### Checkpoint Placement Strategy

| Checkpoint Type | Primary Location | Redundant Copy | Recovery Speed | Durability |
|----------------|-----------------|----------------|----------------|------------|
| Hot (every iteration) | Local DRAM (Tier 2) | Remote DRAM (Tier 3) | ~1-3 sec | Volatile |
| Warm (every N iterations) | Remote DRAM (Tier 3) | Second remote node DRAM | ~5-10 sec | Volatile |
| Cold (every M iterations) | NVMe (Tier 4) | Remote NVMe | ~30-60 sec | Persistent |

**Hot checkpoints** are the Gemini approach — fast, frequent, volatile. If the specific node holding the checkpoint also dies, fall back to warm or cold.

**Warm checkpoints** add cross-node redundancy. With erasure coding (from research/01), we can protect warm checkpoints with RS parity.

**Cold checkpoints** are durable — they survive total cluster failure (power outage). These are the traditional filesystem checkpoints.

### Incremental Checkpoint Design

```
Iteration 0:    [Full Checkpoint] -----> Local DRAM + Remote DRAM
Iteration 1:    [Delta only]      -----> Local DRAM (apply delta)
Iteration 2:    [Delta only]      -----> Local DRAM (apply delta)
...
Iteration N:    [Full Checkpoint] -----> Local DRAM + Remote DRAM + NVMe
```

Delta computation: XOR current state with previous state. Non-zero bytes are the delta. With Adam optimizer, weight updates are typically small, so deltas compress well.

**Estimated delta sizes:**
- Model weights delta per step: ~1-5% of total weight size (sparse updates)
- Optimizer state delta: larger (momentum and variance change more)
- Practical delta: ~10-30% of full checkpoint size per step

### Recovery Time Objectives

| Scenario | Target Recovery Time | Method |
|----------|---------------------|--------|
| Single GPU crash, node alive | < 5 seconds | Reload from local DRAM checkpoint |
| Single node crash | < 30 seconds | Reconstruct from remote DRAM + parity |
| Multiple node crash | < 2 minutes | Reconstruct from surviving nodes + NVMe |
| Total cluster restart | < 5 minutes | Load from NVMe cold checkpoints |

---

## 7. Recommendation for OuterLink

### Architecture: Gemini-Inspired In-Memory Checkpointing

1. **Every-iteration hot checkpoint** to local DRAM (Tier 2):
   - Async snapshot: GPU->CPU DMA during next forward pass
   - Cost: ~1-3 GB/s of memory bandwidth (negligible vs training compute)
   - No network traffic for hot checkpoints

2. **Periodic warm checkpoint** to remote DRAM (Tier 3):
   - Every N iterations (configurable, default N=10-50)
   - Use RDMA one-sided write to partner node's DRAM
   - Apply RS(k,1) or RS(k,2) parity across checkpoint shards
   - Network cost: checkpoint_size / N iterations — amortized

3. **Infrequent cold checkpoint** to NVMe (Tier 4):
   - Every M iterations (configurable, default M=100-1000)
   - Background write to local NVMe + RDMA to remote NVMe
   - Full checkpoint with metadata for resharding

4. **Incremental deltas** between warm checkpoints:
   - Compute XOR delta between consecutive snapshots
   - Transmit only delta to remote nodes
   - Reconstruct full state by applying delta chain

### Checkpoint Format

Support DCP-compatible format for interoperability with PyTorch ecosystem. Store metadata allowing:
- Resharding across different GPU counts
- Mixed precision recovery (FP16 weights + FP32 optimizer states)
- Selective restoration (load only model weights, skip optimizer)

### Integration with Erasure Coding

Warm checkpoints are natural candidates for erasure coding:
- Each node's checkpoint shard is one RS data fragment
- Parity fragments stored on designated nodes
- On node failure: reconstruct checkpoint from surviving shards + parity
- Combined with Gemini's placement optimization for maximum recovery probability

---

## Open Questions

1. **Checkpoint-training interference:** How much does checkpoint RDMA traffic contend with training communication? Gemini showed this can be scheduled; need to quantify for RDMA vs TCP.

2. **Delta compression ratio:** What's the actual compression ratio for training state deltas on common workloads (LLM fine-tuning, vision training)? Need benchmarks.

3. **Checkpoint size with ZeRO-3:** When optimizer states are already sharded across GPUs, each node's checkpoint is 1/N of total. For 8 nodes with a 7B model: ~10.5 GB per node. Fits comfortably in DRAM.

4. **Interaction with R12 deduplication:** Read-only model weights that are deduplicated across nodes — do they need to be checkpointed per-node, or can we checkpoint once and share?

5. **Non-training workloads:** For inference serving, there's no "training state" — but we still need to protect allocated VRAM data. How does checkpointing differ for inference vs training?

---

## Related Documents

- R10: Memory Tiering (tier locations, migration policies)
- R12: Memory Deduplication (shared read-only data)
- R15 Research 01: Erasure Coding Algorithms (parity for checkpoint protection)
- R15 Research 03: Failure Detection and Recovery (when to trigger checkpoint restore)

## References

- PyTorch DCP: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- Gemini: Fast Failure Recovery (SOSP '23): https://dl.acm.org/doi/10.1145/3600006.3613145
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Universal Checkpointing: https://arxiv.org/html/2406.18820v3
- CheckFreq: Frequent, Fine-Grained DNN Checkpointing (2021)
- LowDiff: Efficient Frequent Checkpointing via Low-Cost Differential (2025): https://arxiv.org/html/2509.04084
- TorchFT: https://pytorch.org/blog
- NCCL Fault Tolerance: https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/
- Training LLMs with Fault Tolerant HSDP on 100K GPUs: https://arxiv.org/html/2602.00277v1
- IBM + PyTorch DCP: https://pytorch.org/blog/performant-distributed-checkpointing/
