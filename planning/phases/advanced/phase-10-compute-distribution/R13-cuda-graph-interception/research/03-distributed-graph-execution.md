# R13 Research: Distributed Graph Execution

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Define the execution model for running partitioned CUDA graphs across multiple GPUs on multiple PCs. This covers coordination, synchronization, compute-transfer overlap, dynamic shape handling, performance characteristics, and integration with other OuterLink subsystems (R11 prefetching, R20 NCCL).

---

## TL;DR

The execution model is a coordinator-executor architecture. One process (the OuterLink client, running on the application's PC) coordinates graph launch timing and receives results. Each GPU runs its subgraph independently via `cuGraphLaunch`. Cross-GPU synchronization uses a combination of RDMA completion signals (fast path) and host-side event coordination (fallback). The graph provides perfect advance knowledge of all memory accesses, enabling R11 prefetching to pre-stage data before execution begins. For NCCL operations within graphs, we delegate to R20's backend rather than splitting them. Dynamic shapes require graph re-capture and re-analysis, but the partition plan can often be reused if topology is unchanged.

---

## Execution Architecture

### Coordinator-Executor Model

```
Application PC (Client)
+---------------------------------------+
|  Application                           |
|    |                                   |
|    v                                   |
|  OuterLink Interception Layer          |
|    |                                   |
|    v                                   |
|  Graph Analyzer + Partition Planner    |
|    |                                   |
|    v                                   |
|  Distributed Graph Coordinator         |
|    |         |         |               |
+----+---------+---------+---------------+
     |         |         |
     v         v         v
   GPU_0     GPU_1     GPU_2
  (local)   (PC_B)    (PC_C)
  Executor  Executor  Executor
```

### Coordinator Responsibilities

1. **Graph analysis:** Run DAG analysis and partitioning (02-graph-analysis-and-splitting.md)
2. **Subgraph distribution:** Send subgraph definitions to remote executors
3. **Launch orchestration:** Signal all executors to begin execution
4. **Completion collection:** Wait for all executors to finish, gather results
5. **Result assembly:** Ensure output tensors are in the correct locations for the application

### Executor Responsibilities (Per-GPU)

1. **Subgraph instantiation:** `cuGraphInstantiate` for its assigned subgraph
2. **Execution:** `cuGraphLaunch` on a local stream
3. **Communication:** Send/receive data for cross-partition edges
4. **Completion signaling:** Notify coordinator when done

### Launch Sequence

```
Time -->

Coordinator: [Analyze] [Distribute] [Signal GO]----[Wait]----[Collect results]
                                          |              ^
GPU_0:                          [Instantiate][Launch]...[Done]
GPU_1:                          [Instantiate][Launch]...[Done]
GPU_2:                          [Instantiate][Launch]...[Done]
```

For repeated execution (training loops), instantiation happens once. Subsequent iterations skip directly to launch:

```
Iteration N:   [Signal GO]---[Launch on all GPUs]---[Sync]---[Signal GO]---...
```

---

## Cross-GPU Synchronization

### The Synchronization Problem

When node A on GPU_0 produces data consumed by node B on GPU_1:
1. A must complete
2. A's output data must transfer to GPU_1
3. B must not start until transfer is complete

### Mechanism 1: RDMA Completion Signals (Fast Path — Phase 5+)

With OpenDMA (Phase 5), data transfers happen via direct NIC-to-VRAM RDMA. Completion is signaled by:

1. **RDMA Write with Immediate:** The sender issues an RDMA write carrying the data plus an immediate value. The receiver's NIC delivers a completion queue entry (CQE) when the write lands.
2. **Polling from host callback:** A host callback node in the receiver's subgraph polls the CQE (or a flag in host-pinned memory set by the CQE handler).
3. **GPU-side polling (advanced):** A lightweight GPU kernel polls a flag in mapped host memory, avoiding the host callback entirely.

**Latency:** RDMA write completion + polling overhead. For 100Gbps ConnectX-5: ~2-5us base latency + data_size / 12.5 GB/s.

### Mechanism 2: Host-Staged Transfer with Events (Phase 1)

Without RDMA, transfers go through host memory:

```
GPU_0 VRAM -> cuMemcpyDtoH -> Host RAM -> TCP/network -> Host RAM -> cuMemcpyHtoD -> GPU_1 VRAM
```

Synchronization uses host callback nodes:
1. After node A on GPU_0: host callback initiates async DtoH copy and network send
2. Before node B on GPU_1: host callback waits for network receive and initiates HtoD copy
3. HtoD completion allows node B to proceed

**Latency:** ~100us for small transfers (dominated by CPU overhead), scales with data size.

### Mechanism 3: CUDA IPC (Same-PC Multi-GPU)

When GPUs are on the same PC, use CUDA IPC for zero-copy access:

1. GPU_0 exports memory handle via `cuIpcGetMemHandle`
2. GPU_1 imports and maps it via `cuIpcOpenMemHandle`
3. GPU_1 reads directly from GPU_0's memory (goes through PCIe switch)

No explicit transfer needed — just a synchronization barrier.

**Latency:** PCIe round-trip, ~1-5us for cache-line sized access, bandwidth limited by PCIe topology.

### Choosing the Right Mechanism

| Scenario | Mechanism | Why |
|----------|-----------|-----|
| Same PC, same PCIe switch | CUDA IPC | Zero-copy, lowest latency |
| Same PC, different PCIe switches | CUDA IPC or host-staged | IPC works but crosses CPU, measure both |
| Different PCs, Phase 1 | Host-staged TCP | Only option |
| Different PCs, Phase 2 | UCX (auto-negotiates RDMA or TCP) | Best of available transport |
| Different PCs, Phase 5 | OpenDMA direct RDMA | Lowest latency, highest bandwidth |

R17 (Topology-Aware Scheduling) provides the topology information to make this decision.

---

## Overlapping Compute and Transfer

### Within a Single GPU's Subgraph

CUDA graphs natively overlap independent operations. If GPU_0's subgraph has:
```
[Kernel_1] -> [Send_data_to_GPU1]
[Kernel_2] (independent of Kernel_1)
```

The CUDA scheduler will execute Kernel_2 while the send operation is in progress, because they have no dependency edge. This is free — we get it by correctly structuring the subgraph's dependency edges.

### Across GPUs (Pipeline Overlap)

For a chain of dependent operations split across GPUs:
```
GPU_0: [K1] -> [Send A]
GPU_1:           [Recv A] -> [K2] -> [Send B]
GPU_2:                                 [Recv B] -> [K3]
```

The total time is: cost(K1) + transfer(A) + cost(K2) + transfer(B) + cost(K3).

To overlap, we need pipelining — break the data into chunks:
```
GPU_0: [K1_chunk1] [K1_chunk2] [K1_chunk3]
            \           \           \
GPU_1:    [Recv][K2_c1] [Recv][K2_c2] [Recv][K2_c3]
                  \           \           \
GPU_2:          [Recv][K3_c1] [Recv][K3_c2] [Recv][K3_c3]
```

This requires the kernel to support chunked execution, which ties into R25 (Cooperative Kernel Splitting). For graph-level distribution (R13's scope), we operate at whole-node granularity — pipelining within a node is R25's responsibility.

### Prefetch Integration (R11)

The graph provides complete advance knowledge of memory access patterns. Before launching the distributed graph, we can:

1. Analyze the full DAG to determine what data each GPU will need
2. Issue prefetch operations for all required data
3. By the time `cuGraphLaunch` fires, data is already in place

**This eliminates transfer latency from the critical path for graph replay.**

For the first execution, data must still be transferred. But for iterative workloads (training), the graph is the same each iteration with different data pointers. R11 can prefetch the next iteration's data while the current iteration executes.

```
Iteration N:     [Execute graph]
Iteration N+1:   [Prefetch data] [Execute graph]  <-- overlap
```

---

## Handling Dynamic Shapes

### The Problem

CUDA graphs require static shapes. When shapes change, the application must re-capture the graph. This triggers re-analysis and re-partitioning in OuterLink.

### Frequency of Re-Capture

| Workload | Re-capture Frequency | Why |
|----------|---------------------|-----|
| Fixed-batch training | Never (after warmup) | Same shapes every iteration |
| Variable-batch training | Per unique batch size | PyTorch caches one graph per shape |
| NLP with variable sequence length | Frequent | Different sequences have different lengths |
| Inference with batching | Per unique batch config | Dynamic batching changes shapes |

### Optimization: Partition Plan Caching

When a graph is re-captured with different tensor sizes but the same topology (same kernels in the same order), we can reuse the partition assignment and only update:
- Transfer sizes at partition boundaries
- Tensor pointers in kernel arguments

**Detection:** Compare the new graph's topology signature (hash of node types + edge structure) against cached plans. If match, fast-path update via `cuGraphExecUpdate` on each per-GPU subgraph.

### Padding Strategy

For workloads with many distinct shapes (variable-length NLP), pad inputs to a small set of power-of-2 sizes. This limits the number of unique graphs to cache and amortizes analysis cost.

```
Actual lengths: [47, 123, 89, 256, 15, 201, ...]
Padded to:      [64, 128, 128, 256, 16, 256, ...]
Unique graphs:  4 (for sizes 16, 64, 128, 256)
```

This is standard practice in PyTorch and TensorRT. OuterLink benefits from it because fewer unique topologies means fewer partition plans to compute.

---

## Performance: Graph Mode vs Eager Mode for Distributed Workloads

### Eager Mode (Current OuterLink Without R13)

Each CUDA call intercepted individually:
```
cuLaunchKernel(K1) -> analyze, decide GPU, transfer data, launch
cuLaunchKernel(K2) -> analyze, decide GPU, transfer data, launch
cuLaunchKernel(K3) -> analyze, decide GPU, transfer data, launch
...
```

**Per-call overhead:** ~10-50us (interception + decision + network message)
**For 100 kernels:** ~1-5ms of pure overhead

### Graph Mode (With R13)

Entire graph analyzed once, then executed:
```
cuGraphInstantiate -> analyze full DAG, plan distribution, build subgraphs (one-time: ~5ms)
cuGraphLaunch -> launch all subgraphs simultaneously (~10us)
cuGraphLaunch -> launch all subgraphs simultaneously (~10us)
cuGraphLaunch -> launch all subgraphs simultaneously (~10us)
...
```

**Per-launch overhead:** ~10us (one network message to trigger all executors)
**For 100 kernels over 1000 iterations:** Graph mode saves ~1-5ms * 1000 = 1-5 seconds

### Where Graph Mode Wins Big

| Metric | Eager Mode | Graph Mode | Improvement |
|--------|-----------|------------|-------------|
| Decision overhead per kernel | 10-50us | 0 (pre-planned) | Eliminated |
| Transfer scheduling | Reactive | Proactive (prefetch) | Hides latency |
| Parallelism detection | Per-call heuristic | Full DAG analysis | Optimal |
| GPU utilization | Gaps between calls | Continuous execution | Higher |
| Coordinator load | Per-kernel messages | Per-graph messages | 100x fewer |

### Where Graph Mode Does Not Help

- **Single large kernel:** No parallelism to exploit. One kernel fills one GPU.
- **Dynamic control flow:** Cannot be captured in a graph. Falls back to eager mode.
- **CPU-bound regions:** Host callback nodes serialize on the CPU.
- **Memory-bound transfers:** Graph mode cannot make the network faster, only schedule transfers better.

---

## Integration with R20 NCCL Backend

### NCCL Operations in CUDA Graphs

When PyTorch's DDP captures a training step, the graph includes NCCL collective operations (AllReduce for gradient synchronization). These appear as kernel launches in the captured graph.

**The problem:** NCCL collectives are inherently multi-GPU operations. They cannot be assigned to a single GPU in our partition. They must execute on ALL participating GPUs simultaneously.

### Handling Strategy

1. **Identify NCCL nodes:** Match kernel function pointers or names against known NCCL kernel signatures (e.g., `ncclKernel_AllReduce_*`)
2. **Mark as collective:** Tag these nodes in the shadow graph as collective operations
3. **Partition constraint:** NCCL nodes must be present in ALL per-GPU subgraphs, not assigned to just one
4. **Delegate to R20:** At execution time, R20's NCCL backend handles the actual collective across all GPUs
5. **Synchronization:** NCCL nodes act as global barriers — all GPUs must reach the NCCL node before any can proceed past it

### Example: DDP Training Graph

```
Original graph (single stream capture):
[Forward_K1] -> [Forward_K2] -> ... -> [Loss] -> [Backward_K1] -> ... -> [AllReduce] -> [Optimizer]

Partitioned (2 GPUs, data parallel):
GPU_0: [Forward_K1..K2 on batch_0] -> [Loss_0] -> [Backward_0] -> [AllReduce*] -> [Optimizer_0]
GPU_1: [Forward_K1..K2 on batch_1] -> [Loss_1] -> [Backward_1] -> [AllReduce*] -> [Optimizer_1]

* = NCCL collective, executes on both GPUs via R20
```

This is data parallelism, which is simpler than graph splitting. R13's graph analysis helps here by:
- Automatically detecting the NCCL collective and using it as the partition boundary
- Ensuring forward/backward pass is replicated correctly
- Scheduling data distribution (splitting batches) before graph launch

---

## Integration with R11 Speculative Prefetching

### Graph as Perfect Predictor

R11 uses profiling and heuristics to predict which memory regions will be needed next. A captured CUDA graph eliminates all guesswork:

| R11 Without Graph | R11 With Graph |
|-------------------|----------------|
| Predict next kernel from call history | Know exact sequence from DAG |
| Estimate memory access from kernel signatures | Know exact regions from memcpy nodes + R8 analysis |
| Speculate on data flow | Deterministic data flow from DAG edges |
| May mis-predict and waste bandwidth | Zero mis-predictions |

### Prefetch Scheduling Algorithm

Given the partitioned graph and partition assignments:

```
for each GPU g:
    for each node n in g's subgraph (topological order):
        for each input region r needed by n:
            if r is produced by a node on a different GPU:
                schedule prefetch of r to arrive before n's earliest start time
            if r is produced by a node on the same GPU:
                no transfer needed (already local)
```

The key insight: with the full DAG, we know the earliest time each node can execute (from critical path analysis). We can start prefetching as early as the producer completes, overlapping with other compute work.

### Double-Buffering for Iterative Graphs

For training loops where the same graph repeats:

```
Iteration N:   [Execute graph with data_N]
Iteration N+1: [Prefetch data_N+1 to buffers] [Execute graph with data_N+1]
```

The graph tells us exactly what memory regions each GPU needs at the start of execution. R11 begins transferring the next iteration's data as soon as the current iteration starts, using separate memory buffers (double-buffering).

---

## Error Handling and Fallback

### Graph Execution Failures

| Failure Mode | Detection | Recovery |
|-------------|-----------|----------|
| GPU error during subgraph execution | `cuGraphLaunch` returns error / `cuStreamSynchronize` error | Abort all subgraphs, report to application |
| Network failure mid-transfer | Transfer timeout or RDMA error | Retry transfer or fall back to host-staged path |
| Executor crash (remote PC) | Heartbeat timeout | Re-assign partition to surviving GPUs, re-analyze |
| Out of memory on a GPU | `cuGraphInstantiate` fails | Re-partition with memory constraints, try fewer nodes on that GPU |

### Fallback to Eager Mode

If graph analysis determines that distribution provides no benefit (graph too small, no parallelism, transfer costs dominate), fall back to single-GPU execution:

1. Forward the original `cuGraphInstantiate` to the real driver
2. Execute on the "best" single GPU (most free memory, fastest compute)
3. No distribution overhead at all

**Decision threshold:** If estimated distributed time > 0.9 * estimated single-GPU time, stay on one GPU.

---

## Implementation Phases

### Phase 1: Transparent Capture and Analysis (No Distribution)

- Intercept all graph APIs
- Build shadow graph representation
- Run DAG analysis (topological sort, critical path, parallelism profile)
- Log analysis results for debugging/tuning
- Forward all calls to real driver unchanged

**Goal:** Validate that our analysis is correct by comparing shadow graph against `cuGraphDebugDotPrint` output.

### Phase 2: Single-PC Multi-GPU Distribution

- Partition graphs across GPUs on the same PC
- Use CUDA IPC for cross-GPU data sharing
- Build and instantiate per-GPU subgraphs
- Coordinate launch via shared memory signals
- No network transfers yet

**Goal:** Prove that graph splitting works with the simplest communication mechanism.

### Phase 3: Multi-PC Distribution

- Extend to GPUs across different PCs
- Host-staged transfers for Phase 1 transport
- Network-level synchronization
- Prefetch integration with R11

**Goal:** Full distributed graph execution across the OuterLink cluster.

### Phase 4: OpenDMA Integration

- Replace host-staged transfers with direct RDMA
- GPU-side polling for completion
- Minimize CPU involvement in the data path

**Goal:** Achieve minimum-latency distributed graph execution.

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Graph analysis time (1000 nodes) | < 1ms | Must not bottleneck re-capture |
| Partition planning (1000 nodes, 4 GPUs) | < 0.5ms | Same |
| Subgraph construction | < 2ms | Acceptable one-time cost |
| Launch coordination overhead | < 50us | Must be negligible vs kernel time |
| Cross-GPU synchronization (same PC) | < 10us | CUDA IPC baseline |
| Cross-GPU synchronization (cross PC, RDMA) | < 20us | RDMA write + completion signal |
| Speedup for 4-GPU split (ideal graph) | > 3x | 75% efficiency accounting for communication |

---

## Open Questions

1. **Subgraph serialization format:** What format do we use to send subgraph definitions to remote executors? Serialize the CUgraph API calls? Send a custom protobuf? The remote executor needs enough information to reconstruct and instantiate the subgraph.

2. **Memory lifetime across iterations:** When the same graph replays, tensor pointers may change (PyTorch's memory allocator). Do we need to re-analyze memory regions each iteration, or can we track at the allocation level?

3. **Multi-graph coordination:** An application may have multiple CUDA graphs active simultaneously (e.g., data loading graph + training graph). How do we coordinate resources across multiple distributed graphs?

4. **Profiling-guided re-optimization:** After the first distributed execution, we have real timing data. Should we automatically re-partition to optimize based on actual rather than estimated costs? How often?

5. **Graph capture ordering guarantee:** CUDA guarantees that stream capture respects submission order within a stream and event-based ordering across streams. Does our subgraph reconstruction preserve these guarantees?

---

## Related Documents

- 01-cuda-graph-api.md (API surface for graph interception)
- 02-graph-analysis-and-splitting.md (analysis and partitioning algorithms)
- R10: Memory Hierarchy (memory tracking and page management)
- R11: Speculative Prefetching (integration for perfect prediction)
- R17: Topology-Aware Scheduling (GPU placement and topology info)
- R20: NCCL Backend (handling collective operations)
- R25: Cooperative Kernel Splitting (splitting within individual kernels)

## References

- Turimbetov et al., "A Device-Side Execution Model for Multi-GPU Task Graphs" (Mustard), ICS 2025: https://dl.acm.org/doi/10.1145/3721145.3730426
- NVIDIA, "How to Overlap Data Transfers in CUDA": https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- NVIDIA GPUDirect RDMA Documentation: https://docs.nvidia.com/cuda/gpudirect-rdma/
- Hwang et al., "GPU-driven Code Execution for Distributed Deep Learning", NSDI 2023: https://www.usenix.org/system/files/nsdi23-hwang.pdf
- PyTorch CUDA Graph Documentation: https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
- NVIDIA, "Handling Dynamic Patterns with CUDA Graphs": https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html
- PyGraph, "Robust Compiler Support for CUDA Graphs in PyTorch": https://arxiv.org/html/2503.19779v1
