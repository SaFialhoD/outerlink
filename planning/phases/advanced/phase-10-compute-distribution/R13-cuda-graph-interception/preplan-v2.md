# R13: CUDA Graph Interception — Pre-Plan v2

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of the R13 pre-plan. Resolves open questions from v1, defines exact Rust structs, specifies the graph analysis pipeline and HEFT scheduling algorithm, and locks down integration points with R10 v2, R11 v2, R17 v2, R20 v2, R23, R25, and R26.

---

## 1. Resolved Open Questions

### Q1 (from v1): Should R13 be split into sub-plans?

**Resolution: Yes.** R13 splits into three sub-plans aligned with the milestone groups:
- **R13a:** Interception + Shadow Graph Construction (Milestones 1-2)
- **R13b:** DAG Analysis + Partitioning (Milestones 3-5)
- **R13c:** Distributed Execution + Optimization (Milestones 6-7)

Each sub-plan gets its own plan.md. Dependencies flow R13a -> R13b -> R13c.

### Q2 (from v1): Do we need a prototype before the full plan?

**Resolution: Yes, but scoped.** A standalone prototype that captures a PyTorch `torch.cuda.CUDAGraph` via stream capture, builds the shadow graph, and validates it against `cuGraphDebugDotPrint` output. This prototype is Milestone 1-2 of R13a and serves as the validation gate before proceeding to R13b.

### Q3 (from v1): What is the minimum viable version?

**Resolution:** Milestones 1-4 (transparent analysis without distribution) ship as R13-MVP. This already enables R11 prefetch prediction (the highest-value integration). Distribution (Milestones 5-7) ships as R13-Full.

### Q4 (from v1): Should the partition planner be pluggable?

**Resolution: Yes.** The `GraphPartitioner` trait allows swapping algorithms. HEFT is the default. dagP is the fallback for graphs exceeding 5,000 nodes where HEFT's O(V^2*K) cost exceeds the 1ms budget.

### Q5 (from v1 research): Kernel memory access without R8?

**Resolution: Two-tier approach confirmed.** Tier 1 (conservative): all predecessor outputs assumed needed. Tier 2 (precise): R8 provides per-argument pointer/scalar classification. The `ShadowNode` struct carries both `conservative_deps` and `precise_deps` fields. Analysis uses precise when available, conservative otherwise.

### Q6 (from v1 research): NCCL kernel identification?

**Resolution:** Dual detection. (1) Kernel name prefix matching against `ncclKernel_*` and `ncclDevKernel_*` patterns. (2) Function pointer matching against addresses captured during NCCL initialization intercepts (`ncclCommInitRank` etc.). Nodes tagged `NodeClass::NcclCollective` are delegated to R20's `NcclGraphPlugin`.

### Q7 (from v1 research): Conditional node partitioning?

**Resolution: Atomic in Phase 1.** Conditional nodes (IF/WHILE/SWITCH) and their entire body subgraphs are treated as indivisible units assigned to a single GPU. The cost estimate uses the worst-case (longest) body. Phase 2 may investigate body splitting if profiling shows conditional bodies dominate execution time.

### Q8 (from v1 research): Graph caching and re-capture?

**Resolution: Topology hash caching.** A `TopologyHash` computed from (node_types, edge_structure, kernel_function_pointers) identifies structurally identical graphs. When topology matches a cached plan, reuse the partition assignment and update only tensor pointers via `cuGraphExecUpdate` on each subgraph. Cache eviction: LRU with 64-entry limit.

---

## 2. Core Data Structures

### 2.1 Shadow Graph

```rust
/// Unique identifier for a node within a shadow graph.
type ShadowNodeId = u64;

/// Unique identifier for a GPU in the OuterLink pool.
type GpuId = u32;

/// Classification of a graph node for scheduling purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeClass {
    /// GPU kernel launch (the primary unit of work).
    Kernel,
    /// Memory copy with known src/dst/size.
    Memcpy,
    /// Memory set operation.
    Memset,
    /// Host callback — forces CPU synchronization, acts as barrier.
    HostCallback,
    /// Embedded child graph (analyzed recursively).
    ChildGraph,
    /// No-op dependency node.
    Empty,
    /// Event record for cross-stream sync.
    EventRecord,
    /// Event wait for cross-stream sync.
    EventWait,
    /// Memory allocation within graph.
    MemAlloc,
    /// Memory free within graph.
    MemFree,
    /// NCCL collective operation (detected via name/pointer matching).
    /// Delegated to R20's NcclGraphPlugin. Must be present on ALL GPUs.
    NcclCollective,
    /// Conditional node (IF/WHILE/SWITCH). Treated as atomic with its body.
    Conditional,
}

/// Memory region descriptor for data dependency tracking.
#[derive(Debug, Clone)]
struct MemoryRegion {
    /// Virtual address base (from R10's page table).
    base_addr: u64,
    /// Size in bytes.
    size: u64,
    /// Which GPU currently owns this region (from R10 PageTable).
    owning_gpu: Option<GpuId>,
    /// R10 page indices covering this region (64KB pages).
    page_indices: Vec<u64>,
}

/// Kernel-specific metadata extracted from cuGraphKernelNodeGetParams.
#[derive(Debug, Clone)]
struct KernelNodeInfo {
    /// CUDA function handle.
    func_handle: u64,
    /// Demangled kernel name (for NCCL detection and debugging).
    kernel_name: String,
    /// Grid dimensions (blocks).
    grid_dim: [u32; 3],
    /// Block dimensions (threads).
    block_dim: [u32; 3],
    /// Shared memory bytes.
    shared_mem_bytes: u32,
    /// Raw argument buffer (opaque without R8).
    args_buffer: Vec<u8>,
    /// Minimum compute capability required (from fatbin inspection).
    min_compute_capability: (u32, u32),
    /// Workload classification hint for R23 integration.
    workload_class: WorkloadClass,
}

/// Workload classification (consumed by R23's CapabilityScorer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadClass {
    /// ALU throughput dominant (large grid, small data per thread).
    ComputeBound,
    /// Memory bandwidth dominant (small grid, large data per thread).
    MemoryBound,
    /// Tensor Core dominant (matched known matmul/conv patterns).
    TensorBound,
    /// Unknown — use conservative scoring.
    Unknown,
}

/// Memcpy-specific metadata.
#[derive(Debug, Clone)]
struct MemcpyNodeInfo {
    src: MemoryRegion,
    dst: MemoryRegion,
    size_bytes: u64,
    direction: MemcpyDirection,
}

#[derive(Debug, Clone, Copy)]
enum MemcpyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
}

/// A single node in the shadow graph.
#[derive(Debug, Clone)]
struct ShadowNode {
    /// Stable identifier within this graph.
    id: ShadowNodeId,
    /// Original CUDA CUgraphNode handle (for clone-based subgraph building).
    cuda_handle: u64,
    /// Classification.
    node_class: NodeClass,
    /// Predecessor node IDs (incoming edges).
    predecessors: Vec<ShadowNodeId>,
    /// Successor node IDs (outgoing edges).
    successors: Vec<ShadowNodeId>,
    /// Whether this edge uses programmatic dependent launch (CUDA 12.3+).
    /// If true, co-locate with predecessor to preserve overlap benefit.
    programmatic_deps: Vec<ShadowNodeId>,

    // --- Type-specific data ---
    kernel_info: Option<KernelNodeInfo>,
    memcpy_info: Option<MemcpyNodeInfo>,
    child_graph: Option<Box<ShadowGraph>>,

    // --- Dependency tracking ---
    /// Conservative: all memory regions from predecessors assumed needed.
    conservative_reads: Vec<MemoryRegion>,
    /// Conservative: all outputs assumed consumed by successors.
    conservative_writes: Vec<MemoryRegion>,
    /// Precise: actual memory regions read (populated when R8 is available).
    precise_reads: Option<Vec<MemoryRegion>>,
    /// Precise: actual memory regions written (populated when R8 is available).
    precise_writes: Option<Vec<MemoryRegion>>,

    // --- Analysis annotations (populated by analysis pipeline) ---
    /// Estimated execution cost on a reference GPU.
    estimated_cost_ns: u64,
    /// Upward rank for HEFT scheduling (computed in analysis).
    upward_rank: f64,
    /// Assigned GPU after partitioning.
    assigned_gpu: Option<GpuId>,
    /// Earliest start time on assigned GPU (from HEFT).
    earliest_start_ns: u64,
    /// Earliest finish time on assigned GPU (from HEFT).
    earliest_finish_ns: u64,
}

/// The complete shadow representation of a CUDA graph.
#[derive(Debug, Clone)]
struct ShadowGraph {
    /// All nodes indexed by ID.
    nodes: HashMap<ShadowNodeId, ShadowNode>,
    /// Root nodes (no predecessors).
    roots: Vec<ShadowNodeId>,
    /// Total node count.
    node_count: usize,
    /// Total edge count.
    edge_count: usize,

    // --- Computed by analysis pipeline ---
    /// Nodes in topological order (Kahn's algorithm).
    topological_order: Vec<ShadowNodeId>,
    /// Critical path node IDs (zero-slack nodes).
    critical_path: Vec<ShadowNodeId>,
    /// Critical path length in nanoseconds.
    critical_path_length_ns: u64,
    /// Parallelism profile: (time_offset_ns, concurrent_width) pairs.
    parallelism_profile: Vec<(u64, usize)>,
    /// Maximum parallelism width detected.
    max_parallelism: usize,
    /// Topology hash for partition plan caching.
    topology_hash: u64,

    // --- NCCL detection ---
    /// Indices of nodes classified as NcclCollective.
    nccl_node_ids: Vec<ShadowNodeId>,

    // --- Conditional node tracking ---
    /// Conditional node IDs and their body subgraph bounds.
    conditional_groups: Vec<ConditionalGroup>,
}

/// A conditional node and its indivisible body (treated as atomic unit).
#[derive(Debug, Clone)]
struct ConditionalGroup {
    /// The conditional node itself.
    condition_node: ShadowNodeId,
    /// All nodes in the conditional body (must stay on same GPU).
    body_nodes: Vec<ShadowNodeId>,
    /// Worst-case (longest body) cost estimate in nanoseconds.
    worst_case_cost_ns: u64,
}
```

### 2.2 Graph Partition

```rust
/// The output of the graph partitioner: assignment of nodes to GPUs
/// plus the communication plan for cross-GPU edges.
#[derive(Debug, Clone)]
struct GraphPartition {
    /// Per-GPU assignment: GPU ID -> ordered list of node IDs.
    gpu_assignments: HashMap<GpuId, Vec<ShadowNodeId>>,
    /// Cross-GPU edges that require data transfer.
    cross_edges: Vec<CrossEdge>,
    /// Estimated makespan in nanoseconds.
    estimated_makespan_ns: u64,
    /// Estimated total transfer volume in bytes.
    total_transfer_bytes: u64,
    /// Number of GPUs used (may be less than available if graph is small).
    gpu_count: usize,
    /// Whether this partition was auto-selected or forced.
    strategy: PartitionStrategy,
    /// Topology hash of the source graph (for caching).
    source_topology_hash: u64,
}

/// A cross-GPU edge requiring data transfer.
#[derive(Debug, Clone)]
struct CrossEdge {
    /// Source node (producer).
    src_node: ShadowNodeId,
    /// Destination node (consumer).
    dst_node: ShadowNodeId,
    /// Source GPU.
    src_gpu: GpuId,
    /// Destination GPU.
    dst_gpu: GpuId,
    /// Memory regions to transfer.
    regions: Vec<MemoryRegion>,
    /// Total transfer size in bytes.
    transfer_bytes: u64,
    /// Estimated transfer time in nanoseconds (from R17 topology).
    estimated_transfer_ns: u64,
}

/// Strategy selection for how the graph should be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartitionStrategy {
    /// Graph too small or too linear to benefit from splitting.
    /// Execute on single best GPU.
    SingleGpu,
    /// Graph split across multiple GPUs using HEFT.
    HeftPartition,
    /// Graph split using dagP (for very large graphs > 5000 nodes).
    DagPPartition,
    /// Data-parallel replication (each GPU runs full graph on different data).
    DataParallelReplicate,
}
```

### 2.3 Prefetch Schedule (consumed by R11 v2)

```rust
/// Output of graph analysis consumed by R11's GraphPrefetchSchedule.
/// Provides perfect memory access prediction with confidence=1.0.
#[derive(Debug, Clone)]
struct GraphMemorySchedule {
    /// Per-GPU ordered list of memory accesses with timing.
    per_gpu_accesses: HashMap<GpuId, Vec<ScheduledAccess>>,
    /// Total distinct memory regions accessed across all GPUs.
    total_regions: usize,
    /// Whether this schedule is from a cached topology (pointer update only).
    is_cached_topology: bool,
}

/// A scheduled memory access for prefetch planning.
#[derive(Debug, Clone)]
struct ScheduledAccess {
    /// Memory region needed.
    region: MemoryRegion,
    /// Node that needs this region.
    consuming_node: ShadowNodeId,
    /// Earliest time this region is needed (from HEFT EST).
    needed_at_ns: u64,
    /// Earliest time this region is available (producer's EFT).
    available_at_ns: u64,
    /// Whether the region is already local (no transfer needed).
    is_local: bool,
    /// Access type.
    access_type: AccessType,
}

#[derive(Debug, Clone, Copy)]
enum AccessType {
    Read,
    Write,
    ReadWrite,
}
```

---

## 3. Analysis Pipeline

The graph analysis pipeline runs between `cuStreamEndCapture` (or final `cuGraphAdd*Node`) and `cuGraphInstantiate`. It is synchronous and must complete within the latency budget.

### 3.1 Pipeline Stages

```
Input: CUgraph handle from CUDA driver
  |
  v
[Stage 1: Shadow Graph Construction]  -- O(V + E), ~0.01ms for 1000 nodes
  |  Extract nodes via cuGraphGetNodes
  |  Extract edges via cuGraphGetEdges
  |  Query node types and parameters
  |  Detect NCCL kernels (name + pointer matching)
  |  Identify conditional groups
  |
  v
[Stage 2: Topology Hash + Cache Lookup]  -- O(V + E), ~0.01ms
  |  Compute hash of (node_types, edge_structure, kernel_func_ptrs)
  |  Check partition plan cache
  |  If HIT: fast-path to Stage 6 (pointer update only)
  |
  v
[Stage 3: DAG Analysis]  -- O(V + E), ~0.02ms
  |  Kahn's topological sort
  |  Forward pass: compute EST (earliest start time) per node
  |  Backward pass: compute LST (latest start time) per node
  |  Critical path: nodes where EST == LST
  |  Parallelism profile: concurrent width at each time step
  |  Workload classification per kernel node
  |
  v
[Stage 4: Partition Decision]  -- O(1), ~0.001ms
  |  If max_parallelism < 2: SingleGpu strategy
  |  If critical_path_length < 10ms: SingleGpu strategy
  |  If node_count > 5000: DagPPartition strategy
  |  If NCCL nodes detected and graph is training loop: DataParallelReplicate
  |  Otherwise: HeftPartition
  |
  v
[Stage 5: HEFT Partitioning]  -- O(V^2 * K), ~0.1ms for 1000 nodes, 4 GPUs
  |  Compute upward rank for each node
  |  Sort by decreasing upward rank
  |  For each node: assign to GPU with earliest finish time
  |  Enforce constraints: NCCL on all GPUs, conditionals atomic, programmatic deps co-located
  |  Query R23 GpuProfile for per-GPU cost estimation
  |  Query R17 topology for transfer cost estimation
  |
  v
[Stage 6: Output Generation]  -- O(V + E), ~0.01ms
  |  Build GraphPartition struct
  |  Build GraphMemorySchedule for R11
  |  Emit to R11's prefetch engine (async, non-blocking)
  |  Cache partition plan keyed by topology_hash
  |
  v
Output: GraphPartition + per-GPU ShadowGraph subsets
```

### 3.2 Latency Budget

| Stage | Budget (1000 nodes) | Budget (5000 nodes) |
|-------|-------------------|-------------------|
| Shadow Graph Construction | 0.05 ms | 0.25 ms |
| Topology Hash + Cache | 0.01 ms | 0.05 ms |
| DAG Analysis | 0.02 ms | 0.10 ms |
| Partition Decision | 0.001 ms | 0.001 ms |
| HEFT Partitioning | 0.10 ms | 2.50 ms (switch to dagP) |
| Output Generation | 0.01 ms | 0.05 ms |
| **Total** | **< 0.2 ms** | **< 3.0 ms** |

For graphs exceeding 5,000 nodes, HEFT's O(V^2*K) exceeds the budget with K=4. The pipeline automatically switches to dagP (O(V log V) after coarsening).

### 3.3 Cost Estimation Model

Node execution cost on a specific GPU is estimated as:

```rust
fn estimate_node_cost(node: &ShadowNode, gpu: &GpuProfile) -> u64 {
    match node.node_class {
        NodeClass::Kernel => {
            let total_threads = node.kernel_info.as_ref().map(|k| {
                (k.grid_dim[0] * k.grid_dim[1] * k.grid_dim[2]) as u64
                    * (k.block_dim[0] * k.block_dim[1] * k.block_dim[2]) as u64
            }).unwrap_or(1);

            // Base cost: threads / GPU throughput (threads/ns)
            // Adjusted by workload class and GPU capability
            let throughput = match node.kernel_info.as_ref().map(|k| k.workload_class) {
                Some(WorkloadClass::ComputeBound) => gpu.fp32_tflops * 1e3, // GFLOPS
                Some(WorkloadClass::MemoryBound) => gpu.memory_bandwidth_gbps * 1e9 / 8.0, // bytes/ns
                Some(WorkloadClass::TensorBound) => gpu.tensor_tflops() * 1e3,
                _ => gpu.fp32_tflops * 1e3, // default to compute
            };

            // Heuristic: 10 ops per thread for compute, 128 bytes per thread for memory
            let work_units = match node.kernel_info.as_ref().map(|k| k.workload_class) {
                Some(WorkloadClass::MemoryBound) => total_threads * 128,
                _ => total_threads * 10,
            };

            (work_units as f64 / throughput) as u64 // nanoseconds
        }
        NodeClass::Memcpy => {
            let size = node.memcpy_info.as_ref().map(|m| m.size_bytes).unwrap_or(0);
            // Transfer time based on direction and GPU bandwidth
            (size as f64 / (gpu.memory_bandwidth_gbps * 1e9 / 1e9)) as u64
        }
        NodeClass::HostCallback => 50_000, // 50us conservative estimate
        NodeClass::Conditional => {
            // Use worst-case body cost (set during construction)
            50_000 // placeholder, overridden by ConditionalGroup.worst_case_cost_ns
        }
        _ => 100, // Empty, event nodes: negligible
    }
}
```

---

## 4. HEFT Scheduling Algorithm

### 4.1 Upward Rank Computation

```rust
/// Compute upward rank for all nodes. Must be called after topological sort.
/// upward_rank(n) = avg_cost(n) + max over successors s of (comm_cost(n,s) + upward_rank(s))
///
/// avg_cost(n) is the mean execution cost across all candidate GPUs.
/// comm_cost(n,s) is the average transfer cost if n and s are on different GPUs (0 if same).
fn compute_upward_ranks(
    graph: &ShadowGraph,
    gpu_profiles: &[GpuProfile],    // from R23
    topology: &TopologyGraph,        // from R17
) -> HashMap<ShadowNodeId, f64> {
    let mut ranks: HashMap<ShadowNodeId, f64> = HashMap::new();

    // Process in reverse topological order (leaves first)
    for &node_id in graph.topological_order.iter().rev() {
        let node = &graph.nodes[&node_id];

        // Average cost across all GPUs
        let avg_cost: f64 = gpu_profiles.iter()
            .map(|gpu| estimate_node_cost(node, gpu) as f64)
            .sum::<f64>() / gpu_profiles.len() as f64;

        // Max over successors of (comm_cost + successor_rank)
        let max_successor: f64 = node.successors.iter()
            .map(|&succ_id| {
                let succ_rank = ranks.get(&succ_id).copied().unwrap_or(0.0);
                // Average communication cost (assumes different GPUs)
                let data_size = estimate_transfer_size(node, &graph.nodes[&succ_id]);
                let avg_comm = topology.average_transfer_cost_ns(data_size);
                avg_comm as f64 + succ_rank
            })
            .fold(0.0_f64, f64::max);

        ranks.insert(node_id, avg_cost + max_successor);
    }

    ranks
}
```

### 4.2 GPU Assignment Loop

```rust
/// Assign each node to the GPU that yields the earliest finish time.
fn heft_assign(
    graph: &mut ShadowGraph,
    gpu_profiles: &[GpuProfile],          // from R23
    topology: &TopologyGraph,              // from R17
    gpu_capability_scores: &HashMap<GpuId, f64>,  // from R23 CapabilityScorer
) -> GraphPartition {
    // Sort nodes by decreasing upward rank
    let mut priority_order: Vec<ShadowNodeId> = graph.nodes.keys().copied().collect();
    priority_order.sort_by(|a, b| {
        graph.nodes[b].upward_rank
            .partial_cmp(&graph.nodes[a].upward_rank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Per-GPU availability time (when the GPU becomes free)
    let mut gpu_available: HashMap<GpuId, u64> = HashMap::new();
    for profile in gpu_profiles {
        gpu_available.insert(profile.gpu_id, 0);
    }

    let mut assignments: HashMap<GpuId, Vec<ShadowNodeId>> = HashMap::new();
    let mut cross_edges: Vec<CrossEdge> = Vec::new();

    for node_id in priority_order {
        let node = &graph.nodes[&node_id];

        // --- Constraint checks ---

        // NCCL nodes: must be present on ALL GPUs (handled separately)
        if node.node_class == NodeClass::NcclCollective {
            for profile in gpu_profiles {
                assignments.entry(profile.gpu_id).or_default().push(node_id);
            }
            continue;
        }

        // Conditional groups: entire group assigned together
        // (handled by assigning the condition node, body follows)

        // Programmatic dependent launch: must co-locate with predecessor
        let forced_gpu = node.programmatic_deps.first()
            .and_then(|dep_id| graph.nodes[dep_id].assigned_gpu);

        // --- Find best GPU ---
        let mut best_gpu: Option<GpuId> = forced_gpu;
        let mut best_eft: u64 = u64::MAX;

        if best_gpu.is_none() {
            for profile in gpu_profiles {
                let gpu_id = profile.gpu_id;

                // Hard constraint: compute capability
                if let Some(ref ki) = node.kernel_info {
                    if (profile.compute_capability.0, profile.compute_capability.1)
                        < ki.min_compute_capability
                    {
                        continue; // GPU cannot run this kernel
                    }
                }

                // Compute data arrival time (max over all predecessors on other GPUs)
                let data_ready = node.predecessors.iter()
                    .map(|&pred_id| {
                        let pred = &graph.nodes[&pred_id];
                        if pred.assigned_gpu == Some(gpu_id) {
                            pred.earliest_finish_ns // same GPU, no transfer
                        } else {
                            let transfer_size = estimate_transfer_size(pred, node);
                            let transfer_time = topology.transfer_cost_ns(
                                pred.assigned_gpu.unwrap_or(0),
                                gpu_id,
                                transfer_size,
                            );
                            pred.earliest_finish_ns + transfer_time
                        }
                    })
                    .max()
                    .unwrap_or(0);

                // Earliest start = max(gpu_available, data_ready)
                let est = data_ready.max(*gpu_available.get(&gpu_id).unwrap_or(&0));
                let cost = estimate_node_cost(node, profile);
                let eft = est + cost;

                if eft < best_eft {
                    best_eft = eft;
                    best_gpu = Some(gpu_id);
                }
            }
        }

        let assigned = best_gpu.expect("at least one GPU must be compatible");

        // Update node
        let node_mut = graph.nodes.get_mut(&node_id).unwrap();
        node_mut.assigned_gpu = Some(assigned);
        node_mut.earliest_finish_ns = best_eft;
        node_mut.earliest_start_ns = best_eft
            - estimate_node_cost(node_mut, &gpu_profiles.iter()
                .find(|p| p.gpu_id == assigned).unwrap()) ;

        // Update GPU availability
        gpu_available.insert(assigned, best_eft);
        assignments.entry(assigned).or_default().push(node_id);

        // Record cross-edges
        for &pred_id in &node.predecessors {
            let pred = &graph.nodes[&pred_id];
            if pred.assigned_gpu != Some(assigned) {
                cross_edges.push(CrossEdge {
                    src_node: pred_id,
                    dst_node: node_id,
                    src_gpu: pred.assigned_gpu.unwrap_or(0),
                    dst_gpu: assigned,
                    regions: get_transfer_regions(pred, node),
                    transfer_bytes: estimate_transfer_size(pred, node),
                    estimated_transfer_ns: topology.transfer_cost_ns(
                        pred.assigned_gpu.unwrap_or(0),
                        assigned,
                        estimate_transfer_size(pred, node),
                    ),
                });
            }
        }
    }

    GraphPartition {
        gpu_assignments: assignments,
        cross_edges,
        estimated_makespan_ns: best_eft_overall(&graph),
        total_transfer_bytes: cross_edges.iter().map(|e| e.transfer_bytes).sum(),
        gpu_count: gpu_profiles.len(),
        strategy: PartitionStrategy::HeftPartition,
        source_topology_hash: graph.topology_hash,
    }
}
```

### 4.3 HEFT Integration with R23 GPU Capabilities

HEFT's cost estimation uses R23's `GpuProfile` to compute per-GPU execution cost. The key interaction:

```rust
/// R23 provides this for each GPU in the pool.
/// HEFT calls estimate_node_cost(node, profile) which uses:
///   - profile.fp32_tflops for compute-bound nodes
///   - profile.memory_bandwidth_gbps for memory-bound nodes
///   - profile.tensor_tflops() for tensor-bound nodes
///   - profile.compute_capability for hard compatibility filtering
```

Without R23, HEFT assumes all GPUs are identical (uses the same cost for every GPU). With R23, HEFT naturally assigns more work to faster GPUs because their EFT is lower.

---

## 5. Subgraph Construction

### 5.1 Clone-and-Prune Strategy

```rust
/// Build per-GPU CUDA subgraphs from the partition assignment.
/// Uses Strategy A from research: clone original, remove non-local nodes,
/// insert communication nodes at boundaries.
fn build_subgraphs(
    original_graph: CUgraph,
    partition: &GraphPartition,
    shadow: &ShadowGraph,
) -> HashMap<GpuId, SubgraphBundle> {
    let mut bundles = HashMap::new();

    for (&gpu_id, node_ids) in &partition.gpu_assignments {
        // 1. Clone the original CUDA graph
        let cloned = cuda::cuGraphClone(original_graph);

        // 2. Find all nodes NOT assigned to this GPU
        let all_ids: HashSet<_> = shadow.nodes.keys().collect();
        let local_ids: HashSet<_> = node_ids.iter().collect();
        let remote_ids: Vec<_> = all_ids.difference(&local_ids).collect();

        // 3. Remove remote nodes (in reverse topological order to avoid dangling edges)
        for &remote_id in remote_ids.iter().rev() {
            let cloned_node = cuda::cuGraphNodeFindInClone(
                shadow.nodes[remote_id].cuda_handle,
                cloned,
            );
            cuda::cuGraphDestroyNode(cloned_node);
        }

        // 4. Insert communication nodes for incoming cross-edges
        let incoming: Vec<_> = partition.cross_edges.iter()
            .filter(|e| e.dst_gpu == gpu_id)
            .collect();
        for edge in &incoming {
            insert_receive_node(cloned, edge, shadow);
        }

        // 5. Insert communication nodes for outgoing cross-edges
        let outgoing: Vec<_> = partition.cross_edges.iter()
            .filter(|e| e.src_gpu == gpu_id)
            .collect();
        for edge in &outgoing {
            insert_send_node(cloned, edge, shadow);
        }

        // 6. Instantiate on this GPU's context
        let exec = cuda::cuGraphInstantiate(cloned);

        bundles.insert(gpu_id, SubgraphBundle {
            graph: cloned,
            exec,
            local_node_ids: node_ids.clone(),
            incoming_edges: incoming.into_iter().cloned().collect(),
            outgoing_edges: outgoing.into_iter().cloned().collect(),
        });
    }

    bundles
}

/// Per-GPU subgraph ready for execution.
struct SubgraphBundle {
    /// The CUDA graph template.
    graph: CUgraph,
    /// The instantiated executable.
    exec: CUgraphExec,
    /// Node IDs assigned to this GPU.
    local_node_ids: Vec<ShadowNodeId>,
    /// Cross-edges where this GPU receives data.
    incoming_edges: Vec<CrossEdge>,
    /// Cross-edges where this GPU sends data.
    outgoing_edges: Vec<CrossEdge>,
}
```

### 5.2 Communication Node Insertion

Communication nodes are inserted as host callback nodes (`cuGraphAddHostNode`) that trigger the active transport mechanism:

```rust
/// Insert a receive node before the consumer node.
/// The host callback blocks until the transfer from src_gpu completes.
fn insert_receive_node(graph: CUgraph, edge: &CrossEdge, shadow: &ShadowGraph) {
    let consumer_clone = cuda::cuGraphNodeFindInClone(
        shadow.nodes[&edge.dst_node].cuda_handle,
        graph,
    );

    // Create host callback that waits for transfer completion
    let callback = TransferCallback::new_receive(
        edge.src_gpu,
        edge.dst_gpu,
        edge.regions.clone(),
        edge.transfer_bytes,
    );

    let recv_node = cuda::cuGraphAddHostNode(
        graph,
        &[], // no deps (will be wired below)
        callback.as_cuda_host_fn(),
    );

    // Wire: recv_node -> consumer_clone
    // Remove consumer's dependency on the (now-removed) producer
    // Add dependency: recv_node -> consumer_clone
    cuda::cuGraphAddDependencies(graph, &[recv_node], &[consumer_clone], 1);
}
```

---

## 6. Integration Points

### 6.1 R10 v2: Memory Hierarchy

**R13 queries R10:**
```rust
/// Trait from R10 v2. R13 uses this to determine memory region ownership.
trait PageTable {
    /// Given a virtual address, return which GPU currently owns the page.
    fn owning_gpu(&self, vaddr: u64) -> Option<GpuId>;
    /// Given a virtual address range, return all page indices (64KB pages).
    fn pages_for_range(&self, base: u64, size: u64) -> Vec<u64>;
    /// Query PTE access pattern type for prefetch hints.
    fn access_pattern(&self, page_index: u64) -> AccessPatternType;
}

/// R13 calls PageTable to populate MemoryRegion.owning_gpu and page_indices
/// during shadow graph construction (Stage 1).
```

**R13 informs R10:**
```rust
/// R13 provides graph-derived access pattern hints to R10's PTE.
/// These are written back to PTE.access_pattern_type and PTE.prefetch_hints.
trait AccessMonitor {
    /// Set access pattern for a page based on graph analysis.
    /// Graph analysis yields `AccessPatternType::GraphDeterministic`.
    fn set_access_pattern(&mut self, page_index: u64, pattern: AccessPatternType);
    /// Set prefetch hint: this page will be needed by gpu_id at time_ns.
    fn set_prefetch_hint(&mut self, page_index: u64, gpu_id: GpuId, needed_at_ns: u64);
}
```

### 6.2 R11 v2: Speculative Prefetching

**R13 feeds R11:**
```rust
/// R11 v2 defines this struct to consume R13's graph analysis output.
/// R13 populates it after Stage 6 of the analysis pipeline.
struct GraphPrefetchSchedule {
    /// Ordered list of (region, target_gpu, deadline_ns) tuples.
    /// R11 issues prefetch operations to meet each deadline.
    transfers: Vec<PrefetchEntry>,
    /// Confidence level: 1.0 for graph-derived schedules (deterministic).
    confidence: f64,
}

struct PrefetchEntry {
    region: MemoryRegion,
    target_gpu: GpuId,
    deadline_ns: u64,
    source_gpu: GpuId,
    priority: PrefetchPriority,
}

/// R13 calls this on R11's prefetch engine after partition planning.
trait PrefetchEngine {
    /// Submit a graph-derived prefetch schedule. R11 merges this with
    /// its speculative predictions (graph schedule takes priority due to confidence=1.0).
    fn submit_graph_schedule(&self, schedule: GraphPrefetchSchedule);
}
```

### 6.3 R17 v2: Topology-Aware Scheduling

**R13 queries R17:**
```rust
/// R13 uses R17's topology graph to compute HEFT communication costs.
trait TopologyGraph {
    /// Transfer cost in nanoseconds between two GPUs for a given data size.
    /// Accounts for: PCIe bandwidth, network bandwidth, RDMA vs host-staged,
    /// and current link utilization.
    fn transfer_cost_ns(&self, src_gpu: GpuId, dst_gpu: GpuId, bytes: u64) -> u64;

    /// Average transfer cost across all GPU pairs (for HEFT upward rank).
    fn average_transfer_cost_ns(&self, bytes: u64) -> u64;

    /// Whether two GPUs are on the same physical node (affects sync mechanism).
    fn same_node(&self, gpu_a: GpuId, gpu_b: GpuId) -> bool;
}

/// R17 v2's PlacementDecision includes gpu_capability weight (0.15 factor).
/// R13's HEFT partitioner incorporates this by:
/// 1. Using R17.transfer_cost_ns() for communication cost in HEFT
/// 2. Using R23.capability_score() for computation cost in HEFT
/// The HEFT algorithm naturally balances compute and communication costs.
```

### 6.4 R20 v2: NCCL Backend

**R13 delegates NCCL nodes to R20:**
```rust
/// When R13 detects NcclCollective nodes in the graph, it delegates
/// their handling to R20's graph plugin.
trait NcclGraphPlugin {
    /// Register NCCL collective nodes found in a graph.
    /// R20 will handle these during distributed execution:
    /// - Insert the collective into each per-GPU subgraph
    /// - Coordinate execution across all participating GPUs
    /// - Handle asymmetric bandwidth reporting per GPU (from R23 profiles)
    fn register_graph_collectives(
        &self,
        graph_id: u64,
        nccl_nodes: Vec<NcclNodeInfo>,
        participating_gpus: Vec<GpuId>,
    );
}

struct NcclNodeInfo {
    node_id: ShadowNodeId,
    /// Detected collective type (AllReduce, AllGather, etc.)
    collective_type: NcclCollectiveType,
    /// Data size involved.
    data_bytes: u64,
    /// Kernel name for debugging.
    kernel_name: String,
}

enum NcclCollectiveType {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    /// Unknown collective — treat conservatively as barrier.
    Unknown,
}
```

### 6.5 R23: Heterogeneous GPU Mixing

**R13 queries R23 for HEFT cost model:**
```rust
/// R23 provides GPU capability profiles to R13's HEFT partitioner.
/// See R23 preplan-v2.md for the full GpuProfile struct.
trait GpuCapabilityProvider {
    /// Get profiles for all GPUs in the pool.
    fn all_profiles(&self) -> Vec<GpuProfile>;

    /// Get capability score for a specific (workload_class, gpu) pair.
    /// Used by HEFT to estimate per-GPU execution cost.
    fn capability_score(&self, gpu_id: GpuId, workload_class: WorkloadClass) -> f64;

    /// Check if a kernel's binary is compatible with a GPU.
    fn is_binary_compatible(&self, kernel: &KernelNodeInfo, gpu_id: GpuId) -> bool;
}
```

### 6.6 R25: Cooperative Kernel Splitting

**R13 identifies candidates for R25:**
```rust
/// R13's graph analysis identifies kernels that are:
/// 1. On the critical path (splitting would reduce makespan)
/// 2. Large enough to benefit from splitting (estimated cost > 1ms)
/// 3. On a GPU with CC >= 7.5 (R25's minimum for cooperative launch)
///
/// R25 is SECONDARY to R13. R13 splits at the graph level first.
/// R25 splits individual kernel nodes only when:
/// - The kernel is a bottleneck on the critical path
/// - Graph-level splitting cannot reduce it further
/// - The kernel supports cooperative launch
trait KernelSplitCandidateNotifier {
    fn notify_split_candidates(&self, candidates: Vec<KernelSplitCandidate>);
}

struct KernelSplitCandidate {
    node_id: ShadowNodeId,
    kernel_info: KernelNodeInfo,
    assigned_gpu: GpuId,
    estimated_cost_ns: u64,
    critical_path_slack_ns: u64, // 0 = on critical path
}
```

### 6.7 R26: PTP Timestamps

**R13 uses R26 for coordinated subgraph launches across nodes:**
```rust
/// R26 provides PTP-synchronized timestamps for coordinated launch.
/// When launching subgraphs across multiple nodes, the coordinator
/// specifies a synchronized launch time.
trait PtpClock {
    /// Get current PTP-synchronized time in nanoseconds.
    fn now_ns(&self) -> u64;
    /// Schedule an action at a specific PTP time.
    fn schedule_at(&self, time_ns: u64, callback: Box<dyn FnOnce()>);
}

/// The coordinator uses PTP to issue a synchronized GO signal:
/// 1. Compute launch_time = ptp.now_ns() + margin_ns (margin accounts for message delivery)
/// 2. Send (launch_time, subgraph_exec_handle) to each executor
/// 3. Each executor calls ptp.schedule_at(launch_time, || cuGraphLaunch(exec, stream))
/// This ensures all subgraphs start within PTP precision (~1us) of each other.
```

---

## 7. Interception Function Table

All ~78 CUDA Graph Driver API functions intercepted by R13:

| Category | Functions | R13 Action |
|----------|-----------|------------|
| Stream capture (5) | `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, `cuStreamUpdateCaptureDependencies` | Track capture state. On EndCapture: trigger analysis pipeline. |
| Graph lifecycle (4) | `cuGraphCreate`, `cuGraphDestroy`, `cuGraphClone`, `cuGraphDebugDotPrint` | Track graph handles. Clone used for subgraph building. |
| Node creation (14) | `cuGraphAdd{Kernel,Memcpy,Memset,Host,ChildGraph,Empty,EventRecord,EventWait,ExtSemaphoresSignal,ExtSemaphoresWait,MemAlloc,MemFree,BatchMemOp,Node}Node` | Build shadow node for each. Detect NCCL by kernel name/pointer. |
| Node param get (12) | `cuGraph{Kernel,Memcpy,Memset,Host}NodeGetParams`, `cuGraphChildGraphNodeGetGraph`, `cuGraphEvent{Record,Wait}NodeGetEvent`, etc. | Read-through: forward to driver, cache result in shadow. |
| Node param set (12) | `cuGraph{Kernel,Memcpy,Memset,Host}NodeSetParams`, `cuGraphExec{Kernel,Memcpy,Memset,Host}NodeSetParams`, etc. | Update shadow, forward to driver. If topology unchanged, fast-path. |
| Graph inspection (10) | `cuGraphGetNodes`, `cuGraphGetRootNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphNodeGet{Dependencies,DependentNodes}`, `cuGraphNodeGetLocalId`, `cuGraphNodeGetContainingGraph`, `cuGraphNodeSetEnabled`, `cuGraphNodeGetEnabled` | Forward to driver. R13 uses these during shadow construction. |
| Graph manipulation (4) | `cuGraphAddDependencies`, `cuGraphRemoveDependencies`, `cuGraphDestroyNode`, `cuGraphNodeFindInClone` | Track in shadow. Used during subgraph construction. |
| Instantiation (3) | `cuGraphInstantiate`, `cuGraphInstantiateWithParams`, `cuGraphInstantiateWithFlags` | **Primary interception point.** Run analysis pipeline, build subgraphs, instantiate per-GPU. Return handle to coordinator's distributed executor. |
| Execution (2) | `cuGraphLaunch`, `cuGraphExecDestroy` | Launch: trigger coordinated multi-GPU launch. Destroy: cleanup all subgraphs. |
| Update (2) | `cuGraphExecUpdate`, `cuGraphExecNodeSetParams` | Fast-path: if topology unchanged, propagate param updates to subgraphs. If topology changed, re-analyze. |
| Conditional (2) | `cuGraphConditionalHandleCreate`, conditional via `cuGraphAddNode` | Detect and group. Treat as atomic. |

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Test | Validates |
|------|-----------|
| Kahn's topological sort on synthetic DAGs | Correctness, cycle detection |
| Critical path computation on known DAGs | EST/LST calculation, path identification |
| HEFT partitioning on known DAGs with known costs | Assignment optimality against hand-computed solutions |
| Topology hash uniqueness | Different topologies produce different hashes; same topology with different pointers produces same hash |
| NCCL kernel name matching | Detection regex covers known NCCL kernel name patterns |
| Conditional group identification | Bodies correctly identified and grouped |
| Cost estimation model | Produces reasonable estimates for known kernel configurations |

### 8.2 Integration Tests

| Test | Validates |
|------|-----------|
| PyTorch `torch.cuda.CUDAGraph` capture + shadow graph | Shadow graph matches `cuGraphDebugDotPrint` output |
| PyTorch DDP training graph capture | NCCL nodes correctly identified |
| Partition plan cache hit/miss | Topology hash correctly identifies structural matches |
| Subgraph construction via clone-and-prune | Each subgraph instantiates without error |
| Parameter update fast-path | `cuGraphExecUpdate` correctly propagates to subgraphs |

### 8.3 Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| Analysis pipeline (1000-node synthetic graph) | < 0.2 ms |
| Analysis pipeline (5000-node synthetic graph) | < 3.0 ms |
| Shadow graph memory footprint (1000 nodes) | < 1 MB |
| Partition plan cache lookup | < 0.01 ms |
| Subgraph construction (4 GPUs, 1000 nodes) | < 2 ms |

---

## 9. Acceptance Criteria

1. All ~78 graph API functions intercepted; zero CUDA applications broken by transparent pass-through.
2. Shadow graph topology matches `cuGraphDebugDotPrint` for all tested frameworks (PyTorch, TensorRT).
3. Analysis pipeline completes in < 1ms for graphs with <= 1000 nodes.
4. HEFT partitioning produces makespan within 1.3x of lower bound (critical path length) on synthetic graphs with parallelism width >= 4.
5. GraphMemorySchedule fed to R11 with confidence=1.0 for all graph-mode execution.
6. NCCL kernels detected with zero false negatives on PyTorch DDP training graphs.
7. Distributed execution produces bit-identical results vs single-GPU execution.
8. Adding a second GPU achieves >= 1.5x speedup on graphs with parallelism width >= 4.

---

## 10. Implementation Phases (Updated)

### R13a: Interception + Shadow Graph (4-6 weeks)
- Intercept all 78 functions, transparent pass-through
- Shadow graph construction from both stream capture and explicit API
- Topology hash computation
- NCCL kernel detection
- Conditional group identification
- **Gate:** Shadow graph validates against cuGraphDebugDotPrint

### R13b: DAG Analysis + Partitioning (4-6 weeks)
- Full analysis pipeline (Stages 1-5)
- HEFT partitioner with R23 integration
- dagP fallback for large graphs
- Partition plan caching
- GraphMemorySchedule generation for R11
- **Gate:** Partition quality within 1.3x of optimal on synthetic benchmarks

### R13c: Distributed Execution (6-8 weeks)
- Subgraph construction (clone-and-prune)
- Communication node insertion (host callbacks)
- Coordinator-executor protocol
- PTP-synchronized launch (R26)
- Parameter update fast-path
- **Gate:** Bit-identical results, speedup targets met

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-cuda-graph-api.md](./research/01-cuda-graph-api.md) -- Full CUDA Graph API surface
- [research/02-graph-analysis-and-splitting.md](./research/02-graph-analysis-and-splitting.md) -- Partitioning algorithms
- [research/03-distributed-graph-execution.md](./research/03-distributed-graph-execution.md) -- Execution model
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/) -- PageTable, AccessMonitor traits
- [R11 Speculative Prefetching](../../phase-07-memory-intelligence/R11-speculative-prefetch/) -- GraphPrefetchSchedule
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/) -- TopologyGraph trait
- [R20 NCCL Backend](../../phase-09-collective-communication/R20-nccl-backend/) -- NcclGraphPlugin
- [R23 Heterogeneous GPU Mixing](../R23-heterogeneous-gpu-mixing/) -- GpuProfile, CapabilityScorer
- [R25 Cooperative Kernel Splitting](../../phase-10-compute-distribution/R25-cooperative-kernel-splitting/) -- KernelSplitCandidate
- [R26 PTP Timestamps](../../phase-08-network-optimization/R26-ptp-timestamps/) -- PtpClock
