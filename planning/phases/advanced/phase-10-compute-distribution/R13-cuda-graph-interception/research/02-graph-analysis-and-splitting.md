# R13 Research: Graph Analysis and Splitting for Distributed Execution

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Determine how to analyze a captured CUDA graph's DAG structure and partition it across multiple GPUs for distributed execution. This covers the algorithms for dependency analysis, parallelism detection, partitioning, and communication insertion — the "compiler" that turns a single-GPU graph into a multi-GPU execution plan.

---

## TL;DR

The captured CUDA graph gives us a complete DAG. We perform topological sort, critical path analysis, and parallelism detection to understand the structure. For partitioning, HEFT (Heterogeneous Earliest Finish Time) with communication-cost awareness is the best fit for OuterLink's heterogeneous multi-GPU setup. We must maintain acyclicity in the partition graph, insert explicit transfer nodes at partition boundaries, and track memory regions through the DAG to know what data moves where. Frameworks like PyTorch and TensorFlow already use CUDA graphs heavily via stream capture, so most real-world graphs will have well-structured DAGs with clear parallel branches.

---

## Step 1: DAG Extraction from CUDA Graph

Using the inspection APIs from 01-cuda-graph-api.md, we extract:

### Node Information
For each node, we record:

| Field | Source API | Purpose |
|-------|-----------|---------|
| Node handle | `cuGraphGetNodes` | Identity |
| Node type | `cuGraphNodeGetType` | Classification |
| Node ID | `cuGraphNodeGetLocalId` | Stable identifier |
| Parameters | `cuGraphKernelNodeGetParams`, etc. | Execution details |
| Predecessors | `cuGraphNodeGetDependencies` | Incoming edges |
| Successors | `cuGraphNodeGetDependentNodes` | Outgoing edges |

### Edge Information
`cuGraphGetEdges` returns the full edge list. CUDA 12.3+ edges can carry `CUgraphEdgeData` indicating programmatic dependent launch (affects scheduling).

### Shadow Graph Representation

```
struct ShadowNode {
    id: u64,
    node_type: NodeType,
    predecessors: Vec<u64>,
    successors: Vec<u64>,
    // Type-specific data
    kernel_info: Option<KernelInfo>,    // function, grid, block, args
    memcpy_info: Option<MemcpyInfo>,    // src, dst, size, direction
    memset_info: Option<MemsetInfo>,
    child_graph: Option<ShadowGraph>,   // recursive for child graph nodes
    // Analysis annotations
    estimated_cost: Duration,
    memory_reads: Vec<MemoryRegion>,
    memory_writes: Vec<MemoryRegion>,
    assigned_gpu: Option<GpuId>,
}

struct ShadowGraph {
    nodes: HashMap<u64, ShadowNode>,
    roots: Vec<u64>,
    // Computed properties
    topological_order: Vec<u64>,
    critical_path: Vec<u64>,
    critical_path_length: Duration,
    parallelism_profile: Vec<(usize, Duration)>,  // (width, time) pairs
}
```

---

## Step 2: DAG Analysis Algorithms

### Topological Sort

Foundation for all subsequent analysis. Standard Kahn's algorithm (BFS from roots, decrement in-degree).

**Complexity:** O(V + E) where V = nodes, E = edges. For typical ML graphs (hundreds to low thousands of nodes), this is sub-millisecond.

### Critical Path Analysis

The critical path determines the minimum possible execution time regardless of parallelism. Every scheduling decision is measured against this lower bound.

**Algorithm:**
1. Assign each node an estimated execution cost (see Cost Estimation below)
2. Forward pass: compute earliest start time (EST) for each node = max(EST + cost of all predecessors)
3. Backward pass: compute latest start time (LST) for each node = min(LST of all successors) - own cost
4. Critical path = all nodes where EST == LST (zero slack)
5. Critical path length = EST of terminal node + terminal node cost

**Cost estimation sources:**
- Kernel nodes: estimate from grid dimensions, block size, known kernel execution profiles
- Memcpy nodes: estimated from transfer size and memory bandwidth
- Memset nodes: estimated from size and memory bandwidth
- Host nodes: conservative estimate (these force CPU synchronization)

In Phase 1, we use heuristic estimates. In Phase 2, we profile actual execution times and feed them back.

### Parallelism Detection

Identify independent branches that can execute simultaneously on different GPUs.

**Algorithm:**
1. Compute the antichain decomposition: at each level of the topological sort, count how many nodes have no mutual dependencies
2. The maximum antichain width = maximum parallelism available
3. Build a parallelism profile: for each time step in the critical path schedule, how many independent nodes are ready

**Parallelism profile example for a typical ResNet block:**
```
Time  | Width | Nodes active
------+-------+------------------------------------------
  0   |   1   | Conv2d kernel
  1   |   1   | BatchNorm kernel
  2   |   2   | ReLU kernel + skip-connection memcpy
  3   |   1   | Add kernel (merge point)
```

This profile tells us: this block has limited parallelism (max width 2). Distributing it across GPUs likely costs more in transfer overhead than it saves. The scheduler should keep this as one unit on a single GPU.

A transformer attention block, in contrast:
```
Time  | Width | Nodes active
------+-------+------------------------------------------
  0   |   3   | Q, K, V projection kernels (independent)
  1   |   1   | QK^T matmul
  2   |   1   | Softmax
  3   |   1   | Attention * V matmul
  4   |   1   | Output projection
```

Width 3 at the start means the Q, K, V projections can each go to a different GPU if the input data is replicated.

---

## Step 3: Data Dependency Tracking

### Memory Region Analysis

To partition the graph, we need to know which memory regions each node reads and writes. Without this, we cannot insert the correct transfers at partition boundaries.

**Sources of memory region information:**

| Node Type | Memory Access | How We Know |
|-----------|--------------|-------------|
| Memcpy | Explicit src, dst, size | `cuGraphMemcpyNodeGetParams` directly gives addresses and sizes |
| Memset | Explicit dst, size | `cuGraphMemsetNodeGetParams` directly gives address and size |
| Kernel | Implicit via arguments | **Hard problem** — requires R8 (Kernel Param Introspection) |
| MemAlloc | Produces a new region | `cuGraphMemAllocNodeGetParams` gives the allocation |
| MemFree | Destroys a region | `cuGraphMemFreeNodeGetParams` gives the freed address |

**The kernel argument problem:**
Kernel arguments are an opaque byte buffer. We know the buffer contents but not which bytes are pointers vs scalars vs structs. R8 (Kernel Param Introspection) researches how to resolve this via:
- PTX/SASS analysis of the kernel binary
- `.nv_params` section in cubin
- Runtime tracking of `cuMemAlloc` return values matched against argument values

**Conservative approach (no kernel introspection):**
Without R8, we assume every kernel node depends on ALL memory regions written by its predecessors and produces output consumed by ALL successors. This is safe but over-estimates dependencies, reducing parallelism.

**Producer-consumer chains:**
Once we know memory regions, we build producer-consumer chains:
```
MemAlloc(A) -> Kernel1(writes A) -> Kernel2(reads A, writes B) -> Memcpy(B -> host)
```

Cutting this chain between Kernel1 and Kernel2 requires transferring region A across GPUs.

---

## Step 4: Graph Partitioning Algorithms

### Problem Statement

Given:
- DAG G = (V, E) with node costs and edge weights (data transfer sizes)
- K available GPUs with known compute capabilities and interconnect bandwidth
- Memory capacity constraints per GPU

Find:
- Assignment f: V -> {GPU_1, ..., GPU_K} that minimizes total execution time (makespan)
- Subject to: acyclicity of the quotient graph (no circular dependencies between partitions)

This is NP-complete in general. We use heuristics.

### Algorithm Options

| Algorithm | Approach | Pros | Cons | OuterLink Fit |
|-----------|----------|------|------|---------------|
| **HEFT** | List scheduling with priority queue (upward rank) | Handles heterogeneous GPUs, good makespan, fast O(V^2 * K) | Greedy, no backtracking | **Best fit** for our case |
| **CPOP** | Schedule critical path on fastest processor, rest by EFT | Optimizes critical path | Other paths may suffer | Good complement to HEFT |
| **dagP** | Multilevel acyclic partitioning (coarsen, partition, refine) | Guarantees acyclicity, handles large graphs | Optimizes cut weight not makespan | Good for very large graphs |
| **METIS** | General graph partitioning | Fast, well-tested | Ignores DAG structure, may create cycles | Requires acyclicity post-processing |
| **Min-cut** | Minimize data transfer across partition boundary | Reduces communication | Ignores load balance | Too simplistic alone |
| **Round-robin level** | Assign levels alternately to GPUs | Simple, maintains acyclicity | Poor load balance | Baseline only |

### HEFT Algorithm (Recommended Primary)

**Heterogeneous Earliest Finish Time** works well because:
- OuterLink's GPUs may be heterogeneous (different models, different PCIe bandwidth)
- It naturally accounts for communication cost between GPUs
- It preserves topological order (acyclicity guaranteed)
- It is fast enough for online use (< 1ms for 1000-node graphs)

**Steps:**
1. Compute upward rank for each node: `rank_u(n) = cost(n) + max(c(n,s) + rank_u(s))` for all successors s, where c(n,s) is communication cost if n and s are on different GPUs
2. Sort nodes by decreasing upward rank
3. For each node in priority order, assign it to the GPU that gives the earliest finish time (EFT), considering: computation cost on that GPU, data transfer cost from predecessors on other GPUs, and current load on that GPU

**Communication cost model for OuterLink:**
```
transfer_cost(data_size, src_gpu, dst_gpu) =
    if src_gpu == dst_gpu: 0
    else if same_pc(src_gpu, dst_gpu): data_size / pcie_bandwidth  // ~12 GB/s PCIe 3.0
    else: latency_base + data_size / network_bandwidth              // ~12.5 GB/s for 100Gbps RDMA
```

### Acyclicity Constraint

Critical constraint: the quotient graph (where each partition is a super-node) must be a DAG. If GPU_A depends on GPU_B and GPU_B depends on GPU_A, we have a deadlock risk.

HEFT maintains acyclicity by construction (nodes processed in topological order). For other algorithms, we must verify and fix:

1. Build quotient graph from partition assignment
2. Check for cycles (DFS-based cycle detection, O(V + E))
3. If cycles found: move nodes that create back-edges to the target partition
4. Repeat until acyclic

---

## Step 5: Splitting Strategies

Once partitioning assigns nodes to GPUs, we must build actual executable subgraphs. Three approaches:

### Strategy A: Per-GPU Subgraphs (Recommended)

For each GPU, build a new CUgraph containing only the nodes assigned to that GPU, plus communication nodes at boundaries.

**Process:**
1. Clone the original graph: `cuGraphClone`
2. For each GPU partition:
   a. Clone again to create per-GPU copy
   b. Remove all nodes NOT assigned to this GPU: `cuGraphDestroyNode`
   c. For each incoming cross-GPU edge: insert a receive node (host callback or memcpy from staging buffer)
   d. For each outgoing cross-GPU edge: insert a send node (host callback triggering RDMA or memcpy to staging buffer)
   e. Re-wire dependencies to maintain correctness
3. Instantiate each per-GPU subgraph on its respective GPU context

**Advantages:** Clean separation, each GPU executes a standard CUDA graph. The CUDA runtime handles internal scheduling within each subgraph.

**Disadvantages:** Requires rebuilding graphs (cannot use `cuGraphExecUpdate` for topology changes). Communication nodes are host callbacks, adding CPU overhead.

### Strategy B: Single Graph with Streams per GPU

Keep one graph but assign each node to a different CUDA stream mapped to a different GPU context. Use event nodes for cross-GPU synchronization.

**Advantages:** No graph splitting needed, simpler implementation.

**Disadvantages:** A single `cuGraphLaunch` can only target one GPU context. Multi-context graphs are not supported by CUDA. This approach does not work.

**Verdict: Not viable.**

### Strategy C: Rebuilt Graphs (No Clone)

Instead of cloning and pruning, build entirely new graphs from scratch using `cuGraphCreate` + `cuGraphAdd*Node` for each partition.

**Advantages:** Full control over graph structure. No residual state from original graph.

**Disadvantages:** Must re-extract all node parameters and reconstruct. More code, more room for errors.

**Verdict:** Fallback if cloning proves problematic. Strategy A preferred.

---

## Step 6: Communication Insertion

At every partition boundary (edge where src_gpu != dst_gpu), we must insert data transfer operations.

### What Gets Transferred

The memory regions written by the source node and read by the destination node. From Step 3's producer-consumer chains, we know exactly what data and how much.

### Transfer Mechanisms (by OuterLink phase)

| OuterLink Phase | Mechanism | How Inserted in Graph |
|----------------|-----------|----------------------|
| Phase 1 (host-staged) | cudaMemcpy -> network -> cudaMemcpy | Host callback node triggers async transfer chain |
| Phase 5 (OpenDMA) | Direct NIC-to-VRAM RDMA | Host callback node programs DMA engine, waits on completion |
| With UCX (Phase 2 transport) | UCX put/get | Host callback wraps UCX operation |

### Communication Node Pattern

For each cross-GPU edge (A on GPU_0) -> (B on GPU_1):

**On GPU_0's subgraph (after node A):**
```
[Node A] -> [Send: copy A's output to staging buffer / trigger RDMA write]
```

**On GPU_1's subgraph (before node B):**
```
[Receive: wait for transfer complete, data in local buffer] -> [Node B]
```

**Synchronization:** The send and receive must be coordinated. Options:
1. Polling on a flag in shared memory (if same PC, shared CUDA IPC)
2. Network-level completion signal (RDMA CQE)
3. Host-side event signaling (most portable, but adds CPU hop)

### Overlapping Communication with Compute

The key optimization: while GPU_0 is computing node A's successors (which don't depend on A's output going to GPU_1), GPU_1 can be receiving data. Similarly, while GPU_1 waits for data, GPU_0 continues its own independent work.

This is automatic when using per-GPU subgraphs (Strategy A) because each GPU's CUDA graph scheduler handles its own overlapping. We just need to ensure the communication nodes are correctly positioned in the dependency chain — not blocking independent work.

---

## How Frameworks Build CUDA Graphs

Understanding what graphs we will actually see in practice:

### PyTorch

**Stream capture approach:** PyTorch's `torch.cuda.CUDAGraph` and `torch.compile(mode="reduce-overhead")` use stream capture. The captured graph contains:
- Forward pass: sequence of kernel launches (matmul, activation, normalization)
- Optimizer step: parameter update kernels
- Occasional NCCL kernels if using DDP (these are already distributed)

**CUDAGraph Trees (torch.compile):** PyTorch traces a separate graph for each unique input shape. Dynamic shapes cause repeated re-capture. The graphs are typically linear or have low branching factor (width 2-4 for skip connections).

**Key observation:** PyTorch graphs captured with DDP already contain NCCL collective kernels. OuterLink must identify these and handle them via R20 rather than treating them as regular kernels.

### TensorFlow / XLA

**XLA fusion approach:** XLA compiles TensorFlow operations into fused kernels, then uses stream capture for the fused execution. XLA graphs tend to have:
- Fewer, larger kernel nodes (due to fusion)
- Explicit memory management nodes
- SPMD partitioning already applied for multi-GPU

**Key observation:** XLA may already partition the graph for multi-GPU execution. OuterLink would see per-GPU graphs rather than one monolithic graph. Our interception should detect this and avoid re-partitioning.

### TensorRT

**Explicit construction:** TensorRT builds inference graphs using the explicit API. These are highly optimized with:
- Fused kernels (conv+bn+relu as one kernel)
- Tensor cores utilization
- Memory-efficient scheduling

**Key observation:** TensorRT graphs are already optimized for a single GPU. Splitting them may undo optimizations. For inference, data-parallel replication (each GPU runs the full graph on different inputs) may be better than graph splitting.

---

## Cost-Benefit Analysis: When to Split vs Replicate

Not every graph benefits from splitting. Decision criteria:

| Factor | Split Graph | Replicate Graph |
|--------|------------|-----------------|
| Graph has wide parallelism (width >> GPUs) | Yes | No |
| Graph is mostly linear (width 1-2) | No | Yes |
| Individual kernels are very large (> 100ms) | Yes (split work) | No |
| Total graph time < 10ms | No (overhead dominates) | Yes |
| Data fits on one GPU | Replicate for throughput | - |
| Data doesn't fit on one GPU | Must split | - |
| Training (compute-bound) | Split for speed | Replicate for simplicity |
| Inference (latency-sensitive) | Replicate for throughput | Yes (batch parallel) |

**Hybrid approach:** Replicate the graph across GPUs for data parallelism (each GPU processes different batch elements) but split individual large operations (like matmuls) across GPUs using R25 (Cooperative Kernel Splitting).

---

## Overhead Analysis

### Graph Analysis Cost

| Operation | Complexity | Time (1000 nodes) |
|-----------|-----------|-------------------|
| Topological sort | O(V + E) | ~0.01ms |
| Critical path | O(V + E) | ~0.01ms |
| Parallelism profile | O(V + E) | ~0.01ms |
| HEFT partitioning | O(V^2 * K) | ~0.1ms (4 GPUs) |
| Subgraph construction | O(V + E) | ~0.05ms |
| Total analysis | | ~0.2ms |

### Graph Instantiation Cost

NVIDIA reports cuGraphInstantiate takes 0.1-10ms depending on graph complexity. We need to instantiate K subgraphs (one per GPU), so total: K * instantiation_time.

### Break-Even Analysis

If a graph executes in 50ms and analysis + instantiation costs 5ms, we break even after 1 iteration (50ms eager vs 5ms + 25ms distributed = 30ms). For training loops that run the same graph thousands of times, the one-time analysis cost is negligible.

---

## Open Questions

1. **Memory region tracking without R8:** How accurate can we be about kernel memory access patterns using only the DAG structure and memcpy nodes? Conservative approach works but over-constrains partitioning.

2. **Graph caching:** When PyTorch re-captures a graph with the same topology but different tensor pointers, can we reuse the previous partition plan and just update pointers via `cuGraphExecUpdate`?

3. **NCCL node identification:** NCCL kernels in captured graphs look like regular kernel launches. We need a reliable way to identify them (kernel name matching? known function pointers?).

4. **Conditional node partitioning:** If a conditional node's IF and ELSE branches have different resource requirements, do we partition for the worst case? Or maintain two partition plans?

5. **Profiling feedback loop:** After executing the distributed graph once, we have actual execution times. Should we re-partition using real costs? How often?

---

## Related Documents

- 01-cuda-graph-api.md (API details for the functions used here)
- 03-distributed-graph-execution.md (executing the split graphs)
- R8: Kernel Parameter Introspection (resolving kernel memory access)
- R10: Memory Hierarchy (memory region tracking)
- R11: Speculative Prefetching (graph provides perfect prediction for prefetch)
- R17: Topology-Aware Scheduling (GPU placement decisions)
- R20: NCCL Backend (handling collective operations in graphs)
- R25: Cooperative Kernel Splitting (splitting individual kernels, complementary to graph splitting)

## References

- Turimbetov et al., "A Device-Side Execution Model for Multi-GPU Task Graphs" (Mustard), ICS 2025: https://dl.acm.org/doi/10.1145/3721145.3730426
- Topcuoglu et al., "Performance-effective and low-complexity task scheduling for heterogeneous computing" (HEFT), IEEE TPDS 2002
- Herrmann et al., "Acyclic Partitioning of Large Directed Acyclic Graphs": https://inria.hal.science/hal-01672010/document
- "The TensorFlow Partitioning and Scheduling Problem: It's the Critical Path!": https://ar5iv.labs.arxiv.org/html/1711.01912
- PyTorch CUDA Graph Best Practices: https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html
- NVIDIA Blog — Getting Started with CUDA Graphs: https://developer.nvidia.com/blog/cuda-graphs/
