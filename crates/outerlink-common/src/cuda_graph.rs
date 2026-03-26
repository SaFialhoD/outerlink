//! CUDA Graph Interception (R13) -- shadow graph construction, DAG analysis,
//! and HEFT scheduling for distributed graph execution.
//!
//! When an application captures a CUDA graph (via stream capture or explicit API),
//! OuterLink intercepts the graph and builds a "shadow graph" that mirrors the
//! structure with additional metadata for scheduling. The shadow graph is then
//! analyzed (topological sort, critical path, parallelism profile) and partitioned
//! across GPUs using the HEFT (Heterogeneous Earliest Finish Time) algorithm.
//!
//! # Architecture
//!
//! ```text
//! cuStreamEndCapture / cuGraphAdd*Node
//!    |
//!    v
//! [Shadow Graph Construction] -- O(V + E)
//!    |
//!    v
//! [Topology Hash + Cache Lookup] -- O(V + E)
//!    |
//!    v
//! [DAG Analysis: topo sort, critical path, parallelism] -- O(V + E)
//!    |
//!    v
//! [Partition Decision] -- O(1)
//!    |
//!    v
//! [HEFT Partitioning] -- O(V^2 * K) or dagP for large graphs
//!    |
//!    v
//! [Output: GraphPartition + GraphMemorySchedule]
//! ```
//!
//! # Integration Points
//!
//! - R23 (GPU Mixing): provides `GpuProfile` for per-GPU cost estimation in HEFT.
//! - R11 (Prefetching): consumes `GraphMemorySchedule` for perfect memory prediction.
//! - R17 (Topology): provides transfer cost estimation for cross-GPU edges.
//! - R20 (NCCL): handles nodes classified as `NcclCollective`.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::gpu_mixing::{GpuId, GpuProfile, WorkloadClass};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a node within a shadow graph.
pub type ShadowNodeId = u64;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the CUDA graph analysis pipeline.
#[derive(Debug, Clone)]
pub struct GraphAnalysisConfig {
    /// Maximum graph size (node count) for HEFT scheduling.
    /// Larger graphs fall back to dagP partitioning.
    pub heft_max_nodes: usize,
    /// Minimum parallelism width to consider multi-GPU execution.
    pub min_parallelism_for_split: usize,
    /// Minimum critical path length (ns) to consider multi-GPU execution.
    pub min_critical_path_ns: u64,
    /// LRU cache size for topology-based partition plan caching.
    pub partition_cache_size: usize,
    /// NCCL kernel name prefixes for detection.
    pub nccl_prefixes: Vec<String>,
}

impl Default for GraphAnalysisConfig {
    fn default() -> Self {
        Self {
            heft_max_nodes: 5000,
            min_parallelism_for_split: 2,
            min_critical_path_ns: 10_000_000, // 10ms
            partition_cache_size: 64,
            nccl_prefixes: vec![
                "ncclKernel_".to_string(),
                "ncclDevKernel_".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Node types and metadata
// ---------------------------------------------------------------------------

/// Classification of a graph node for scheduling purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeClass {
    /// GPU kernel launch (the primary unit of work).
    Kernel,
    /// Memory copy with known src/dst/size.
    Memcpy,
    /// Memory set operation.
    Memset,
    /// Host callback -- forces CPU synchronization, acts as barrier.
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
    /// Must be present on ALL GPUs participating in the collective.
    NcclCollective,
    /// Conditional node (IF/WHILE/SWITCH). Treated as atomic with its body.
    Conditional,
}

/// Memory region descriptor for data dependency tracking.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Virtual address base (from R10's page table).
    pub base_addr: u64,
    /// Size in bytes.
    pub size: u64,
    /// Which GPU currently owns this region.
    pub owning_gpu: Option<GpuId>,
    /// R10 page indices covering this region (64KB pages).
    pub page_indices: Vec<u64>,
}

/// Kernel-specific metadata extracted from cuGraphKernelNodeGetParams.
#[derive(Debug, Clone)]
pub struct KernelNodeInfo {
    /// CUDA function handle.
    pub func_handle: u64,
    /// Demangled kernel name (for NCCL detection and debugging).
    pub kernel_name: String,
    /// Grid dimensions (blocks).
    pub grid_dim: [u32; 3],
    /// Block dimensions (threads).
    pub block_dim: [u32; 3],
    /// Shared memory bytes.
    pub shared_mem_bytes: u32,
    /// Raw argument buffer (opaque without R8).
    pub args_buffer: Vec<u8>,
    /// Minimum compute capability required (from fatbin inspection).
    pub min_compute_capability: (u32, u32),
    /// Workload classification hint for R23 integration.
    pub workload_class: WorkloadClass,
}

/// Direction for memory copy operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
}

/// Memcpy-specific metadata.
#[derive(Debug, Clone)]
pub struct MemcpyNodeInfo {
    /// Source memory region.
    pub src: MemoryRegion,
    /// Destination memory region.
    pub dst: MemoryRegion,
    /// Transfer size in bytes.
    pub size_bytes: u64,
    /// Transfer direction.
    pub direction: MemcpyDirection,
}

/// Access type for memory scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

// ---------------------------------------------------------------------------
// Shadow Node
// ---------------------------------------------------------------------------

/// A single node in the shadow graph.
#[derive(Debug, Clone)]
pub struct ShadowNode {
    /// Stable identifier within this graph.
    pub id: ShadowNodeId,
    /// Original CUDA CUgraphNode handle (for clone-based subgraph building).
    pub cuda_handle: u64,
    /// Classification.
    pub node_class: NodeClass,
    /// Predecessor node IDs (incoming edges).
    pub predecessors: Vec<ShadowNodeId>,
    /// Successor node IDs (outgoing edges).
    pub successors: Vec<ShadowNodeId>,
    /// Nodes with programmatic dependent launch (CUDA 12.3+).
    /// If non-empty, co-locate with these predecessors.
    pub programmatic_deps: Vec<ShadowNodeId>,

    // --- Type-specific data ---
    /// Kernel-specific metadata (only for NodeClass::Kernel).
    pub kernel_info: Option<KernelNodeInfo>,
    /// Memcpy-specific metadata (only for NodeClass::Memcpy).
    pub memcpy_info: Option<MemcpyNodeInfo>,
    /// Embedded child graph (only for NodeClass::ChildGraph).
    pub child_graph: Option<Box<ShadowGraph>>,

    // --- Dependency tracking ---
    /// Conservative: all memory regions from predecessors assumed needed.
    pub conservative_reads: Vec<MemoryRegion>,
    /// Conservative: all outputs assumed consumed by successors.
    pub conservative_writes: Vec<MemoryRegion>,
    /// Precise: actual memory regions read (populated when R8 is available).
    pub precise_reads: Option<Vec<MemoryRegion>>,
    /// Precise: actual memory regions written (populated when R8 is available).
    pub precise_writes: Option<Vec<MemoryRegion>>,

    // --- Analysis annotations (populated by analysis pipeline) ---
    /// Estimated execution cost on a reference GPU (nanoseconds).
    pub estimated_cost_ns: u64,
    /// Upward rank for HEFT scheduling.
    pub upward_rank: f64,
    /// Assigned GPU after partitioning.
    pub assigned_gpu: Option<GpuId>,
    /// Earliest start time on assigned GPU (from HEFT).
    pub earliest_start_ns: u64,
    /// Earliest finish time on assigned GPU (from HEFT).
    pub earliest_finish_ns: u64,
}

impl ShadowNode {
    /// Create a new shadow node with minimal required fields.
    pub fn new(id: ShadowNodeId, cuda_handle: u64, node_class: NodeClass) -> Self {
        Self {
            id,
            cuda_handle,
            node_class,
            predecessors: Vec::new(),
            successors: Vec::new(),
            programmatic_deps: Vec::new(),
            kernel_info: None,
            memcpy_info: None,
            child_graph: None,
            conservative_reads: Vec::new(),
            conservative_writes: Vec::new(),
            precise_reads: None,
            precise_writes: None,
            estimated_cost_ns: 0,
            upward_rank: 0.0,
            assigned_gpu: None,
            earliest_start_ns: 0,
            earliest_finish_ns: 0,
        }
    }

    /// Get total thread count for kernel nodes.
    pub fn total_threads(&self) -> u64 {
        self.kernel_info
            .as_ref()
            .map(|k| {
                (k.grid_dim[0] as u64 * k.grid_dim[1] as u64 * k.grid_dim[2] as u64)
                    * (k.block_dim[0] as u64 * k.block_dim[1] as u64 * k.block_dim[2] as u64)
            })
            .unwrap_or(1)
    }
}

// ---------------------------------------------------------------------------
// Conditional Group
// ---------------------------------------------------------------------------

/// A conditional node and its indivisible body (treated as atomic unit).
#[derive(Debug, Clone)]
pub struct ConditionalGroup {
    /// The conditional node itself.
    pub condition_node: ShadowNodeId,
    /// All nodes in the conditional body (must stay on same GPU).
    pub body_nodes: Vec<ShadowNodeId>,
    /// Worst-case (longest body) cost estimate in nanoseconds.
    pub worst_case_cost_ns: u64,
}

// ---------------------------------------------------------------------------
// Shadow Graph
// ---------------------------------------------------------------------------

/// The complete shadow representation of a CUDA graph.
#[derive(Debug, Clone)]
pub struct ShadowGraph {
    /// All nodes indexed by ID.
    pub nodes: HashMap<ShadowNodeId, ShadowNode>,
    /// Root nodes (no predecessors).
    pub roots: Vec<ShadowNodeId>,
    /// Total node count.
    pub node_count: usize,
    /// Total edge count.
    pub edge_count: usize,

    // --- Computed by analysis pipeline ---
    /// Nodes in topological order (Kahn's algorithm).
    pub topological_order: Vec<ShadowNodeId>,
    /// Critical path node IDs (zero-slack nodes).
    pub critical_path: Vec<ShadowNodeId>,
    /// Critical path length in nanoseconds.
    pub critical_path_length_ns: u64,
    /// Parallelism profile: (time_offset_ns, concurrent_width) pairs.
    pub parallelism_profile: Vec<(u64, usize)>,
    /// Maximum parallelism width detected.
    pub max_parallelism: usize,
    /// Topology hash for partition plan caching.
    pub topology_hash: u64,

    // --- NCCL detection ---
    /// Indices of nodes classified as NcclCollective.
    pub nccl_node_ids: Vec<ShadowNodeId>,

    // --- Conditional node tracking ---
    /// Conditional node IDs and their body subgraph bounds.
    pub conditional_groups: Vec<ConditionalGroup>,
}

impl ShadowGraph {
    /// Create a new empty shadow graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            node_count: 0,
            edge_count: 0,
            topological_order: Vec::new(),
            critical_path: Vec::new(),
            critical_path_length_ns: 0,
            parallelism_profile: Vec::new(),
            max_parallelism: 0,
            topology_hash: 0,
            nccl_node_ids: Vec::new(),
            conditional_groups: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: ShadowNode) {
        let id = node.id;
        self.nodes.insert(id, node);
        self.node_count = self.nodes.len();
    }

    /// Add a directed edge from `from` to `to`.
    /// Updates both predecessor and successor lists.
    pub fn add_edge(&mut self, from: ShadowNodeId, to: ShadowNodeId) {
        if let Some(src) = self.nodes.get_mut(&from) {
            if !src.successors.contains(&to) {
                src.successors.push(to);
            }
        }
        if let Some(dst) = self.nodes.get_mut(&to) {
            if !dst.predecessors.contains(&from) {
                dst.predecessors.push(from);
            }
        }
        self.recount_edges();
    }

    /// Recount edges from successor lists.
    fn recount_edges(&mut self) {
        self.edge_count = self.nodes.values().map(|n| n.successors.len()).sum();
    }

    /// Find root nodes (nodes with no predecessors).
    pub fn find_roots(&mut self) {
        self.roots = self
            .nodes
            .values()
            .filter(|n| n.predecessors.is_empty())
            .map(|n| n.id)
            .collect();
        self.roots.sort();
    }

    /// Detect NCCL collective nodes based on kernel name prefixes.
    pub fn detect_nccl_nodes(&mut self, prefixes: &[String]) {
        self.nccl_node_ids.clear();
        for node in self.nodes.values_mut() {
            if node.node_class == NodeClass::Kernel {
                if let Some(ref ki) = node.kernel_info {
                    for prefix in prefixes {
                        if ki.kernel_name.starts_with(prefix) {
                            node.node_class = NodeClass::NcclCollective;
                            self.nccl_node_ids.push(node.id);
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Perform topological sort using Kahn's algorithm.
    /// Returns true if the graph is a DAG (no cycles), false otherwise.
    pub fn topological_sort(&mut self) -> bool {
        let mut in_degree: HashMap<ShadowNodeId, usize> = HashMap::new();
        for (&id, node) in &self.nodes {
            in_degree.entry(id).or_insert(0);
            for &succ in &node.successors {
                *in_degree.entry(succ).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<ShadowNodeId> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        // Sort the initial queue for deterministic ordering
        let mut sorted_queue: Vec<ShadowNodeId> = queue.drain(..).collect();
        sorted_queue.sort();
        queue.extend(sorted_queue);

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            order.push(id);
            let successors: Vec<ShadowNodeId> = self
                .nodes
                .get(&id)
                .map(|n| n.successors.clone())
                .unwrap_or_default();
            let mut ready = Vec::new();
            for succ in successors {
                if let Some(deg) = in_degree.get_mut(&succ) {
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(succ);
                    }
                }
            }
            ready.sort();
            queue.extend(ready);
        }

        let is_dag = order.len() == self.nodes.len();
        self.topological_order = order;
        is_dag
    }

    /// Compute the topology hash for partition plan caching.
    /// Hash is based on (node_types, edge_structure, kernel_func_handles).
    pub fn compute_topology_hash(&mut self) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Sort nodes by ID for deterministic hashing
        let mut node_ids: Vec<ShadowNodeId> = self.nodes.keys().copied().collect();
        node_ids.sort();

        for id in &node_ids {
            let node = &self.nodes[id];
            id.hash(&mut hasher);
            std::mem::discriminant(&node.node_class).hash(&mut hasher);
            if let Some(ref ki) = node.kernel_info {
                ki.func_handle.hash(&mut hasher);
            }
            // Hash sorted successor list
            let mut succs = node.successors.clone();
            succs.sort();
            for s in &succs {
                s.hash(&mut hasher);
            }
        }

        self.topology_hash = hasher.finish();
    }
}

impl Default for ShadowGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Graph Partition
// ---------------------------------------------------------------------------

/// Strategy selection for how the graph should be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
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

/// A cross-GPU edge requiring data transfer.
#[derive(Debug, Clone)]
pub struct CrossEdge {
    /// Source node (producer).
    pub src_node: ShadowNodeId,
    /// Destination node (consumer).
    pub dst_node: ShadowNodeId,
    /// Source GPU.
    pub src_gpu: GpuId,
    /// Destination GPU.
    pub dst_gpu: GpuId,
    /// Memory regions to transfer.
    pub regions: Vec<MemoryRegion>,
    /// Total transfer size in bytes.
    pub transfer_bytes: u64,
    /// Estimated transfer time in nanoseconds.
    pub estimated_transfer_ns: u64,
}

/// The output of the graph partitioner: assignment of nodes to GPUs
/// plus the communication plan for cross-GPU edges.
#[derive(Debug, Clone)]
pub struct GraphPartition {
    /// Per-GPU assignment: GPU ID -> ordered list of node IDs.
    pub gpu_assignments: HashMap<GpuId, Vec<ShadowNodeId>>,
    /// Cross-GPU edges that require data transfer.
    pub cross_edges: Vec<CrossEdge>,
    /// Estimated makespan in nanoseconds.
    pub estimated_makespan_ns: u64,
    /// Estimated total transfer volume in bytes.
    pub total_transfer_bytes: u64,
    /// Number of GPUs used (may be less than available if graph is small).
    pub gpu_count: usize,
    /// Whether this partition was auto-selected or forced.
    pub strategy: PartitionStrategy,
    /// Topology hash of the source graph (for caching).
    pub source_topology_hash: u64,
}

// ---------------------------------------------------------------------------
// Graph Memory Schedule (consumed by R11 prefetching)
// ---------------------------------------------------------------------------

/// A scheduled memory access for prefetch planning.
#[derive(Debug, Clone)]
pub struct ScheduledAccess {
    /// Memory region needed.
    pub region: MemoryRegion,
    /// Node that needs this region.
    pub consuming_node: ShadowNodeId,
    /// Earliest time this region is needed (from HEFT EST).
    pub needed_at_ns: u64,
    /// Earliest time this region is available (producer's EFT).
    pub available_at_ns: u64,
    /// Whether the region is already local (no transfer needed).
    pub is_local: bool,
    /// Access type.
    pub access_type: AccessType,
}

/// Output of graph analysis consumed by R11's GraphPrefetchSchedule.
/// Provides perfect memory access prediction with confidence=1.0.
#[derive(Debug, Clone)]
pub struct GraphMemorySchedule {
    /// Per-GPU ordered list of memory accesses with timing.
    pub per_gpu_accesses: HashMap<GpuId, Vec<ScheduledAccess>>,
    /// Total distinct memory regions accessed across all GPUs.
    pub total_regions: usize,
    /// Whether this schedule is from a cached topology (pointer update only).
    pub is_cached_topology: bool,
}

// ---------------------------------------------------------------------------
// DAG Analysis
// ---------------------------------------------------------------------------

/// Analyze the shadow graph: compute EST/LST, critical path, parallelism profile.
/// Must be called after `topological_sort()`.
pub fn analyze_dag(graph: &mut ShadowGraph) {
    if graph.topological_order.is_empty() {
        return;
    }

    // Forward pass: compute EST (earliest start time)
    let mut est: HashMap<ShadowNodeId, u64> = HashMap::new();
    let mut eft: HashMap<ShadowNodeId, u64> = HashMap::new();

    for &id in &graph.topological_order {
        let node = &graph.nodes[&id];
        let max_pred_eft = node
            .predecessors
            .iter()
            .map(|&p| eft.get(&p).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let node_est = max_pred_eft;
        let node_eft = node_est + node.estimated_cost_ns;
        est.insert(id, node_est);
        eft.insert(id, node_eft);
    }

    // Backward pass: compute LST (latest start time)
    let makespan = eft.values().copied().max().unwrap_or(0);
    let mut lst: HashMap<ShadowNodeId, u64> = HashMap::new();

    for &id in graph.topological_order.iter().rev() {
        let node = &graph.nodes[&id];
        // LST(n) = min over successors s of (LST(s)) - cost(n)
        // For leaf nodes (no successors), LST = makespan - cost
        let min_succ_lst = node
            .successors
            .iter()
            .map(|&s| lst.get(&s).copied().unwrap_or(makespan))
            .min()
            .unwrap_or(makespan);
        let node_lst = min_succ_lst.saturating_sub(node.estimated_cost_ns);
        lst.insert(id, node_lst);
    }

    // Critical path: nodes where EST == LST (zero slack)
    graph.critical_path.clear();
    for &id in &graph.topological_order {
        let node_est = est.get(&id).copied().unwrap_or(0);
        let node_lst = lst.get(&id).copied().unwrap_or(0);
        if node_est == node_lst {
            graph.critical_path.push(id);
        }
    }
    graph.critical_path_length_ns = makespan;

    // Update node EST/EFT fields
    for (&id, node) in graph.nodes.iter_mut() {
        node.earliest_start_ns = est.get(&id).copied().unwrap_or(0);
        node.earliest_finish_ns = eft.get(&id).copied().unwrap_or(0);
    }

    // Parallelism profile: count concurrent nodes at each point in time.
    // We use a sweep-line approach over node [EST, EFT) intervals.
    let mut events: Vec<(u64, i64)> = Vec::new();
    for &id in &graph.topological_order {
        let node_est = est.get(&id).copied().unwrap_or(0);
        let node_eft = eft.get(&id).copied().unwrap_or(0);
        if node_eft > node_est {
            events.push((node_est, 1));
            events.push((node_eft, -1));
        }
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    graph.parallelism_profile.clear();
    let mut current_width: usize = 0;
    let mut max_width: usize = 0;
    for (time, delta) in &events {
        current_width = (current_width as i64 + delta) as usize;
        max_width = max_width.max(current_width);
        graph.parallelism_profile.push((*time, current_width));
    }
    graph.max_parallelism = max_width;
}

// ---------------------------------------------------------------------------
// Cost Estimation
// ---------------------------------------------------------------------------

/// Estimate execution cost of a node on a specific GPU (nanoseconds).
/// Uses kernel metadata and GPU profile for heterogeneous cost modeling.
pub fn estimate_node_cost(node: &ShadowNode, gpu: &GpuProfile) -> u64 {
    match node.node_class {
        NodeClass::Kernel => {
            let total_threads = node.total_threads();
            let workload_class = node
                .kernel_info
                .as_ref()
                .map(|k| k.workload_class)
                .unwrap_or(WorkloadClass::Unknown);

            let throughput = match workload_class {
                WorkloadClass::ComputeBound | WorkloadClass::Unknown => {
                    gpu.fp32_tflops * 1e3 // GFLOPS -> ops/ns
                }
                WorkloadClass::MemoryBound => {
                    gpu.memory_bandwidth_gbps * 1e9 / 8.0 / 1e9 // bytes/ns
                }
                WorkloadClass::TensorBound => {
                    let tensor_tflops = gpu
                        .measured_tensor_tflops
                        .unwrap_or(gpu.fp16_tflops);
                    tensor_tflops * 1e3 // GFLOPS -> ops/ns
                }
            };

            if throughput <= 0.0 {
                return 1000; // 1us minimum for safety
            }

            // Heuristic: 10 ops/thread for compute, 128 bytes/thread for memory
            let work_units = match workload_class {
                WorkloadClass::MemoryBound => total_threads * 128,
                _ => total_threads * 10,
            };

            let cost = (work_units as f64 / throughput) as u64;
            cost.max(100) // minimum 100ns
        }
        NodeClass::Memcpy => {
            let size = node.memcpy_info.as_ref().map(|m| m.size_bytes).unwrap_or(0);
            let bw = gpu.memory_bandwidth_gbps;
            if bw <= 0.0 || size == 0 {
                return 100;
            }
            let cost = (size as f64 / (bw * 1e9 / 1e9)) as u64;
            cost.max(100)
        }
        NodeClass::HostCallback => 50_000, // 50us conservative
        NodeClass::Conditional => 50_000,   // Overridden by ConditionalGroup
        NodeClass::NcclCollective => 100_000, // 100us for NCCL (varies wildly)
        _ => 100, // Empty, event nodes: negligible
    }
}

/// Estimate data transfer size between producer and consumer nodes (bytes).
pub fn estimate_transfer_size(producer: &ShadowNode, _consumer: &ShadowNode) -> u64 {
    // Use conservative writes from producer as estimate
    let from_writes: u64 = producer.conservative_writes.iter().map(|r| r.size).sum();
    if from_writes > 0 {
        return from_writes;
    }
    // Fallback: use kernel argument buffer size as rough proxy
    producer
        .kernel_info
        .as_ref()
        .map(|k| k.args_buffer.len() as u64 * 1024) // Heuristic: args * 1KB
        .unwrap_or(64 * 1024) // 64KB default
}

// ---------------------------------------------------------------------------
// Partition Decision
// ---------------------------------------------------------------------------

/// Decide the partition strategy based on graph characteristics.
pub fn decide_partition_strategy(
    graph: &ShadowGraph,
    config: &GraphAnalysisConfig,
    gpu_count: usize,
) -> PartitionStrategy {
    // Single GPU if pool has only one GPU
    if gpu_count <= 1 {
        return PartitionStrategy::SingleGpu;
    }

    // Single GPU if parallelism is too low
    if graph.max_parallelism < config.min_parallelism_for_split {
        return PartitionStrategy::SingleGpu;
    }

    // Single GPU if critical path is too short (overhead > benefit)
    if graph.critical_path_length_ns < config.min_critical_path_ns {
        return PartitionStrategy::SingleGpu;
    }

    // Data-parallel replication if NCCL nodes present (training loop pattern)
    if !graph.nccl_node_ids.is_empty() {
        return PartitionStrategy::DataParallelReplicate;
    }

    // dagP for very large graphs
    if graph.node_count > config.heft_max_nodes {
        return PartitionStrategy::DagPPartition;
    }

    PartitionStrategy::HeftPartition
}

// ---------------------------------------------------------------------------
// HEFT Scheduling
// ---------------------------------------------------------------------------

/// Compute upward rank for all nodes. Must be called after topological sort.
///
/// upward_rank(n) = avg_cost(n) + max over successors s of (comm_cost(n,s) + upward_rank(s))
pub fn compute_upward_ranks(
    graph: &mut ShadowGraph,
    gpu_profiles: &[GpuProfile],
    avg_transfer_cost_per_byte_ns: f64,
) {
    let mut ranks: HashMap<ShadowNodeId, f64> = HashMap::new();

    // Process in reverse topological order (leaves first)
    for &node_id in graph.topological_order.iter().rev() {
        let node = &graph.nodes[&node_id];

        // Average cost across all GPUs
        let avg_cost: f64 = if gpu_profiles.is_empty() {
            node.estimated_cost_ns as f64
        } else {
            gpu_profiles
                .iter()
                .map(|gpu| estimate_node_cost(node, gpu) as f64)
                .sum::<f64>()
                / gpu_profiles.len() as f64
        };

        // Max over successors of (comm_cost + successor_rank)
        let max_successor: f64 = node
            .successors
            .iter()
            .map(|&succ_id| {
                let succ_rank = ranks.get(&succ_id).copied().unwrap_or(0.0);
                let data_size = estimate_transfer_size(node, &graph.nodes[&succ_id]);
                let avg_comm = data_size as f64 * avg_transfer_cost_per_byte_ns;
                avg_comm + succ_rank
            })
            .fold(0.0_f64, f64::max);

        ranks.insert(node_id, avg_cost + max_successor);
    }

    // Write ranks back to nodes
    for (&id, rank) in &ranks {
        if let Some(node) = graph.nodes.get_mut(&id) {
            node.upward_rank = *rank;
        }
    }
}

/// Run HEFT scheduling: assign each node to the GPU with earliest finish time.
/// Returns a `GraphPartition` describing the assignment.
///
/// `transfer_cost_fn` estimates the transfer time in nanoseconds for moving
/// `bytes` between two GPUs. If None, a default cost model is used.
pub fn heft_assign(
    graph: &mut ShadowGraph,
    gpu_profiles: &[GpuProfile],
    transfer_cost_fn: Option<&dyn Fn(GpuId, GpuId, u64) -> u64>,
) -> GraphPartition {
    let default_transfer = |_src: GpuId, _dst: GpuId, bytes: u64| -> u64 {
        // Default: ~10 GB/s effective transfer rate
        (bytes as f64 / 10e9 * 1e9) as u64
    };
    let transfer_cost = transfer_cost_fn.unwrap_or(&default_transfer);

    // Sort nodes by decreasing upward rank
    let mut priority_order: Vec<ShadowNodeId> = graph.nodes.keys().copied().collect();
    priority_order.sort_by(|a, b| {
        let rank_a = graph.nodes[a].upward_rank;
        let rank_b = graph.nodes[b].upward_rank;
        rank_b
            .partial_cmp(&rank_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Per-GPU availability time
    let mut gpu_available: HashMap<GpuId, u64> = HashMap::new();
    for profile in gpu_profiles {
        gpu_available.insert(profile.gpu_id, 0);
    }

    let mut assignments: HashMap<GpuId, Vec<ShadowNodeId>> = HashMap::new();
    let mut cross_edges: Vec<CrossEdge> = Vec::new();
    let mut overall_makespan: u64 = 0;

    for node_id in priority_order {
        let node = graph.nodes[&node_id].clone();

        // NCCL nodes: must be present on ALL GPUs
        if node.node_class == NodeClass::NcclCollective {
            for profile in gpu_profiles {
                assignments.entry(profile.gpu_id).or_default().push(node_id);
            }
            continue;
        }

        // Check for forced GPU due to programmatic dependent launch
        let forced_gpu = node
            .programmatic_deps
            .first()
            .and_then(|dep_id| graph.nodes.get(dep_id))
            .and_then(|dep| dep.assigned_gpu);

        let mut best_gpu: Option<GpuId> = forced_gpu;
        let mut best_eft: u64 = u64::MAX;

        if best_gpu.is_none() {
            for profile in gpu_profiles {
                let gpu_id = profile.gpu_id;

                // Hard constraint: compute capability
                if let Some(ref ki) = node.kernel_info {
                    if profile.compute_capability < ki.min_compute_capability {
                        continue;
                    }
                }

                // Compute data arrival time
                let data_ready = node
                    .predecessors
                    .iter()
                    .map(|&pred_id| {
                        let pred = &graph.nodes[&pred_id];
                        if pred.assigned_gpu == Some(gpu_id) {
                            pred.earliest_finish_ns
                        } else {
                            let transfer_size = estimate_transfer_size(pred, &node);
                            let transfer_time = transfer_cost(
                                pred.assigned_gpu.unwrap_or(0),
                                gpu_id,
                                transfer_size,
                            );
                            pred.earliest_finish_ns + transfer_time
                        }
                    })
                    .max()
                    .unwrap_or(0);

                let est = data_ready.max(*gpu_available.get(&gpu_id).unwrap_or(&0));
                let cost = estimate_node_cost(&node, profile);
                let eft = est + cost;

                if eft < best_eft {
                    best_eft = eft;
                    best_gpu = Some(gpu_id);
                }
            }
        }

        // If we found a forced GPU, compute its EFT
        if let Some(forced) = forced_gpu {
            if best_gpu == Some(forced) && best_eft == u64::MAX {
                let profile = gpu_profiles.iter().find(|p| p.gpu_id == forced);
                if let Some(profile) = profile {
                    let cost = estimate_node_cost(&node, profile);
                    best_eft = gpu_available.get(&forced).copied().unwrap_or(0) + cost;
                }
            }
        }

        let assigned = match best_gpu {
            Some(gpu) => gpu,
            None => {
                // Fallback: assign to first GPU
                gpu_profiles.first().map(|p| p.gpu_id).unwrap_or(0)
            }
        };

        if best_eft == u64::MAX {
            // Compute EFT for fallback
            if let Some(profile) = gpu_profiles.iter().find(|p| p.gpu_id == assigned) {
                let cost = estimate_node_cost(&node, profile);
                best_eft = gpu_available.get(&assigned).copied().unwrap_or(0) + cost;
            } else {
                best_eft = 0;
            }
        }

        // Update node
        if let Some(node_mut) = graph.nodes.get_mut(&node_id) {
            node_mut.assigned_gpu = Some(assigned);
            node_mut.earliest_finish_ns = best_eft;
            let cost = gpu_profiles
                .iter()
                .find(|p| p.gpu_id == assigned)
                .map(|p| estimate_node_cost(node_mut, p))
                .unwrap_or(0);
            node_mut.earliest_start_ns = best_eft.saturating_sub(cost);
        }

        overall_makespan = overall_makespan.max(best_eft);
        gpu_available.insert(assigned, best_eft);
        assignments.entry(assigned).or_default().push(node_id);

        // Record cross-edges
        for &pred_id in &node.predecessors {
            let pred = &graph.nodes[&pred_id];
            if pred.assigned_gpu != Some(assigned) {
                let transfer_size = estimate_transfer_size(pred, &node);
                cross_edges.push(CrossEdge {
                    src_node: pred_id,
                    dst_node: node_id,
                    src_gpu: pred.assigned_gpu.unwrap_or(0),
                    dst_gpu: assigned,
                    regions: Vec::new(), // Populated by subgraph builder
                    transfer_bytes: transfer_size,
                    estimated_transfer_ns: transfer_cost(
                        pred.assigned_gpu.unwrap_or(0),
                        assigned,
                        transfer_size,
                    ),
                });
            }
        }
    }

    let total_transfer_bytes: u64 = cross_edges.iter().map(|e| e.transfer_bytes).sum();

    GraphPartition {
        gpu_assignments: assignments,
        cross_edges,
        estimated_makespan_ns: overall_makespan,
        total_transfer_bytes,
        gpu_count: gpu_profiles.len(),
        strategy: PartitionStrategy::HeftPartition,
        source_topology_hash: graph.topology_hash,
    }
}

// ---------------------------------------------------------------------------
// Partition Plan Cache
// ---------------------------------------------------------------------------

/// LRU cache for partition plans keyed by topology hash.
pub struct PartitionPlanCache {
    /// Entries ordered by access time (most recent last).
    entries: Vec<(u64, GraphPartition)>,
    /// Maximum cache size.
    max_size: usize,
    /// Cache hit count.
    hits: AtomicU64,
    /// Cache miss count.
    misses: AtomicU64,
}

impl PartitionPlanCache {
    /// Create a new cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up a cached partition plan by topology hash.
    pub fn get(&mut self, topology_hash: u64) -> Option<&GraphPartition> {
        if let Some(pos) = self.entries.iter().position(|(h, _)| *h == topology_hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            // Move to end (most recently used)
            let entry = self.entries.remove(pos);
            self.entries.push(entry);
            self.entries.last().map(|(_, p)| p)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a partition plan into the cache.
    pub fn insert(&mut self, topology_hash: u64, plan: GraphPartition) {
        // Remove existing entry with same hash
        self.entries.retain(|(h, _)| *h != topology_hash);
        // Evict LRU if at capacity
        if self.entries.len() >= self.max_size {
            self.entries.remove(0);
        }
        self.entries.push((topology_hash, plan));
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit count.
    pub fn cache_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cache miss count.
    pub fn cache_misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for the CUDA graph analysis pipeline.
#[derive(Debug, Default)]
pub struct GraphAnalysisStats {
    /// Total graphs analyzed.
    pub graphs_analyzed: AtomicU64,
    /// Total nodes processed across all graphs.
    pub total_nodes_processed: AtomicU64,
    /// Graphs that resulted in single-GPU execution.
    pub single_gpu_decisions: AtomicU64,
    /// Graphs that were partitioned via HEFT.
    pub heft_partitions: AtomicU64,
    /// Partition plan cache hits.
    pub cache_hits: AtomicU64,
    /// NCCL nodes detected.
    pub nccl_nodes_detected: AtomicU64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_mixing::make_test_profile;

    /// Build a simple linear graph: A -> B -> C
    fn build_linear_graph() -> ShadowGraph {
        let mut graph = ShadowGraph::new();
        graph.add_node(ShadowNode::new(1, 0x10, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(2, 0x20, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(3, 0x30, NodeClass::Kernel));
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.find_roots();
        graph
    }

    /// Build a diamond graph: A -> {B, C} -> D
    fn build_diamond_graph() -> ShadowGraph {
        let mut graph = ShadowGraph::new();
        graph.add_node(ShadowNode::new(1, 0x10, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(2, 0x20, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(3, 0x30, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(4, 0x40, NodeClass::Kernel));
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 4);
        graph.find_roots();
        graph
    }

    /// Build a wider parallel graph for testing multi-GPU scheduling.
    fn build_wide_parallel_graph() -> ShadowGraph {
        let mut graph = ShadowGraph::new();
        // Source node
        let mut source = ShadowNode::new(0, 0x00, NodeClass::Kernel);
        source.estimated_cost_ns = 1000;
        source.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xA,
            kernel_name: "source_kernel".to_string(),
            grid_dim: [256, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });
        graph.add_node(source);

        // 4 parallel nodes
        for i in 1..=4 {
            let mut node = ShadowNode::new(i, i as u64 * 0x10, NodeClass::Kernel);
            node.estimated_cost_ns = 5000;
            node.kernel_info = Some(KernelNodeInfo {
                func_handle: 0xB + i as u64,
                kernel_name: format!("parallel_kernel_{}", i),
                grid_dim: [1024, 1, 1],
                block_dim: [256, 1, 1],
                shared_mem_bytes: 0,
                args_buffer: vec![],
                min_compute_capability: (7, 5),
                workload_class: WorkloadClass::ComputeBound,
            });
            graph.add_node(node);
        }

        // Sink node
        let mut sink = ShadowNode::new(5, 0x50, NodeClass::Kernel);
        sink.estimated_cost_ns = 1000;
        sink.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xF,
            kernel_name: "sink_kernel".to_string(),
            grid_dim: [256, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });
        graph.add_node(sink);

        // Wire: source -> [1,2,3,4] -> sink
        for i in 1..=4 {
            graph.add_edge(0, i);
            graph.add_edge(i, 5);
        }
        graph.find_roots();
        graph
    }

    // --- Shadow Graph construction tests ---

    #[test]
    fn test_empty_graph() {
        let graph = ShadowGraph::new();
        assert_eq!(graph.node_count, 0);
        assert_eq!(graph.edge_count, 0);
        assert!(graph.roots.is_empty());
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let graph = build_linear_graph();
        assert_eq!(graph.node_count, 3);
        assert_eq!(graph.edge_count, 2);
        assert_eq!(graph.roots, vec![1]);
    }

    #[test]
    fn test_diamond_graph_structure() {
        let graph = build_diamond_graph();
        assert_eq!(graph.node_count, 4);
        assert_eq!(graph.edge_count, 4);
        assert_eq!(graph.roots, vec![1]);

        // Node 4 should have two predecessors
        let node4 = &graph.nodes[&4];
        assert_eq!(node4.predecessors.len(), 2);
        assert!(node4.predecessors.contains(&2));
        assert!(node4.predecessors.contains(&3));
    }

    // --- Topological sort tests ---

    #[test]
    fn test_topological_sort_linear() {
        let mut graph = build_linear_graph();
        assert!(graph.topological_sort());
        assert_eq!(graph.topological_order, vec![1, 2, 3]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut graph = build_diamond_graph();
        assert!(graph.topological_sort());
        // Node 1 must be first, node 4 must be last
        assert_eq!(graph.topological_order[0], 1);
        assert_eq!(graph.topological_order[3], 4);
        // Nodes 2 and 3 must be between 1 and 4
        let pos2 = graph.topological_order.iter().position(|&x| x == 2).unwrap();
        let pos3 = graph.topological_order.iter().position(|&x| x == 3).unwrap();
        assert!(pos2 > 0 && pos2 < 3);
        assert!(pos3 > 0 && pos3 < 3);
    }

    #[test]
    fn test_topological_sort_detects_cycle() {
        let mut graph = ShadowGraph::new();
        graph.add_node(ShadowNode::new(1, 0x10, NodeClass::Kernel));
        graph.add_node(ShadowNode::new(2, 0x20, NodeClass::Kernel));
        graph.add_edge(1, 2);
        graph.add_edge(2, 1); // cycle
        graph.find_roots();

        assert!(!graph.topological_sort(), "should detect cycle");
        // Order should be incomplete
        assert!(graph.topological_order.len() < 2);
    }

    // --- DAG Analysis tests ---

    #[test]
    fn test_analyze_linear_dag() {
        let mut graph = build_linear_graph();
        // Set costs
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 1000;
        }
        graph.topological_sort();
        analyze_dag(&mut graph);

        // Critical path should be all nodes (linear = all critical)
        assert_eq!(graph.critical_path.len(), 3);
        assert_eq!(graph.critical_path_length_ns, 3000);
        // Max parallelism of a linear graph is 1
        assert_eq!(graph.max_parallelism, 1);
    }

    #[test]
    fn test_analyze_diamond_dag() {
        let mut graph = build_diamond_graph();
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 1000;
        }
        graph.topological_sort();
        analyze_dag(&mut graph);

        // Critical path: A -> B/C -> D = 3000ns
        assert_eq!(graph.critical_path_length_ns, 3000);
        // Parallelism should be >= 2 (B and C run in parallel)
        assert!(
            graph.max_parallelism >= 2,
            "max_parallelism: {}",
            graph.max_parallelism
        );
    }

    #[test]
    fn test_analyze_wide_parallel() {
        let mut graph = build_wide_parallel_graph();
        graph.topological_sort();
        analyze_dag(&mut graph);

        // 4 parallel branches should give max_parallelism >= 4
        assert!(
            graph.max_parallelism >= 4,
            "max_parallelism: {}",
            graph.max_parallelism
        );
    }

    // --- Topology hash tests ---

    #[test]
    fn test_topology_hash_deterministic() {
        let mut g1 = build_diamond_graph();
        let mut g2 = build_diamond_graph();
        g1.compute_topology_hash();
        g2.compute_topology_hash();
        assert_eq!(g1.topology_hash, g2.topology_hash);
    }

    #[test]
    fn test_topology_hash_differs_for_different_graphs() {
        let mut g1 = build_linear_graph();
        let mut g2 = build_diamond_graph();
        g1.compute_topology_hash();
        g2.compute_topology_hash();
        assert_ne!(g1.topology_hash, g2.topology_hash);
    }

    // --- NCCL detection tests ---

    #[test]
    fn test_nccl_detection() {
        let mut graph = ShadowGraph::new();
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Kernel);
        node.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xAABB,
            kernel_name: "ncclKernel_AllReduce_Ring_Sum_fp32".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });
        graph.add_node(node);

        let mut regular = ShadowNode::new(2, 0x20, NodeClass::Kernel);
        regular.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xCCDD,
            kernel_name: "my_custom_kernel".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });
        graph.add_node(regular);

        let prefixes = vec!["ncclKernel_".to_string(), "ncclDevKernel_".to_string()];
        graph.detect_nccl_nodes(&prefixes);

        assert_eq!(graph.nccl_node_ids.len(), 1);
        assert_eq!(graph.nccl_node_ids[0], 1);
        assert_eq!(graph.nodes[&1].node_class, NodeClass::NcclCollective);
        assert_eq!(graph.nodes[&2].node_class, NodeClass::Kernel);
    }

    // --- Partition decision tests ---

    #[test]
    fn test_single_gpu_decision_for_low_parallelism() {
        let mut graph = build_linear_graph();
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 1000;
        }
        graph.topological_sort();
        analyze_dag(&mut graph);

        let config = GraphAnalysisConfig::default();
        let strategy = decide_partition_strategy(&graph, &config, 4);
        assert_eq!(strategy, PartitionStrategy::SingleGpu);
    }

    #[test]
    fn test_single_gpu_decision_for_single_gpu_pool() {
        let mut graph = build_wide_parallel_graph();
        graph.topological_sort();
        analyze_dag(&mut graph);

        let config = GraphAnalysisConfig::default();
        let strategy = decide_partition_strategy(&graph, &config, 1);
        assert_eq!(strategy, PartitionStrategy::SingleGpu);
    }

    #[test]
    fn test_heft_decision_for_parallel_graph() {
        let mut graph = build_wide_parallel_graph();
        graph.topological_sort();
        analyze_dag(&mut graph);
        // Bump critical path to meet threshold
        graph.critical_path_length_ns = 20_000_000;

        let config = GraphAnalysisConfig::default();
        let strategy = decide_partition_strategy(&graph, &config, 4);
        assert_eq!(strategy, PartitionStrategy::HeftPartition);
    }

    #[test]
    fn test_data_parallel_decision_for_nccl_graph() {
        let mut graph = build_wide_parallel_graph();
        graph.topological_sort();
        analyze_dag(&mut graph);
        graph.critical_path_length_ns = 20_000_000;
        graph.nccl_node_ids = vec![1]; // Simulate NCCL detection

        let config = GraphAnalysisConfig::default();
        let strategy = decide_partition_strategy(&graph, &config, 4);
        assert_eq!(strategy, PartitionStrategy::DataParallelReplicate);
    }

    // --- Cost estimation tests ---

    #[test]
    fn test_cost_estimation_kernel() {
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Kernel);
        node.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xAA,
            kernel_name: "compute_kernel".to_string(),
            grid_dim: [1024, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });

        let gpu = make_test_profile(1, "RTX 3060", (8, 6), 12.0, 12.7, 360.0);
        let cost = estimate_node_cost(&node, &gpu);
        assert!(cost > 0, "cost should be positive");
        assert!(cost >= 100, "cost should be at least 100ns minimum");
    }

    #[test]
    fn test_cost_estimation_memcpy() {
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Memcpy);
        node.memcpy_info = Some(MemcpyNodeInfo {
            src: MemoryRegion {
                base_addr: 0,
                size: 1024 * 1024,
                owning_gpu: Some(0),
                page_indices: vec![],
            },
            dst: MemoryRegion {
                base_addr: 0x1000000,
                size: 1024 * 1024,
                owning_gpu: Some(1),
                page_indices: vec![],
            },
            size_bytes: 1024 * 1024,
            direction: MemcpyDirection::DeviceToDevice,
        });

        let gpu = make_test_profile(1, "RTX 3060", (8, 6), 12.0, 12.7, 360.0);
        let cost = estimate_node_cost(&node, &gpu);
        assert!(cost > 0);
    }

    #[test]
    fn test_cost_faster_gpu_is_cheaper() {
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Kernel);
        node.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xAA,
            kernel_name: "compute_kernel".to_string(),
            grid_dim: [1024, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });

        let slow = make_test_profile(1, "RTX 3060", (8, 6), 12.0, 12.7, 360.0);
        let fast = make_test_profile(2, "RTX 4090", (8, 9), 24.0, 82.6, 1008.0);

        let slow_cost = estimate_node_cost(&node, &slow);
        let fast_cost = estimate_node_cost(&node, &fast);
        assert!(
            fast_cost < slow_cost,
            "4090 cost ({}) should be less than 3060 cost ({})",
            fast_cost,
            slow_cost
        );
    }

    #[test]
    fn test_host_callback_fixed_cost() {
        let node = ShadowNode::new(1, 0x10, NodeClass::HostCallback);
        let gpu = make_test_profile(1, "RTX 3060", (8, 6), 12.0, 12.7, 360.0);
        assert_eq!(estimate_node_cost(&node, &gpu), 50_000);
    }

    // --- Upward rank tests ---

    #[test]
    fn test_upward_ranks_linear() {
        let mut graph = build_linear_graph();
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 1000;
        }
        graph.topological_sort();

        let gpu = make_test_profile(1, "Test GPU", (8, 6), 12.0, 12.7, 360.0);
        compute_upward_ranks(&mut graph, &[gpu], 0.0);

        // Node 3 (leaf): rank = cost(3)
        // Node 2: rank = cost(2) + rank(3)
        // Node 1: rank = cost(1) + rank(2)
        // Node 1 should have highest rank
        assert!(graph.nodes[&1].upward_rank >= graph.nodes[&2].upward_rank);
        assert!(graph.nodes[&2].upward_rank >= graph.nodes[&3].upward_rank);
        assert!(graph.nodes[&3].upward_rank > 0.0);
    }

    // --- HEFT scheduling tests ---

    #[test]
    fn test_heft_single_gpu() {
        let mut graph = build_linear_graph();
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 1000;
        }
        graph.topological_sort();
        let gpu = make_test_profile(1, "Only GPU", (8, 6), 12.0, 12.7, 360.0);
        compute_upward_ranks(&mut graph, &[gpu.clone()], 0.0);

        let partition = heft_assign(&mut graph, &[gpu], None);
        // All nodes should be on GPU 1
        assert_eq!(partition.gpu_assignments.len(), 1);
        assert!(partition.gpu_assignments.contains_key(&1));
        assert_eq!(partition.cross_edges.len(), 0);
    }

    #[test]
    fn test_heft_two_gpus_diamond() {
        let mut graph = build_diamond_graph();
        for node in graph.nodes.values_mut() {
            node.estimated_cost_ns = 5000;
            node.kernel_info = Some(KernelNodeInfo {
                func_handle: 0xAA,
                kernel_name: "test_kernel".to_string(),
                grid_dim: [256, 1, 1],
                block_dim: [256, 1, 1],
                shared_mem_bytes: 0,
                args_buffer: vec![],
                min_compute_capability: (7, 5),
                workload_class: WorkloadClass::ComputeBound,
            });
        }
        graph.topological_sort();
        let gpu1 = make_test_profile(1, "GPU 1", (8, 6), 12.0, 12.7, 360.0);
        let gpu2 = make_test_profile(2, "GPU 2", (8, 6), 12.0, 12.7, 360.0);
        compute_upward_ranks(&mut graph, &[gpu1.clone(), gpu2.clone()], 0.001);

        let partition = heft_assign(&mut graph, &[gpu1, gpu2], None);
        // Should use at least 1 GPU
        assert!(!partition.gpu_assignments.is_empty());
        assert!(partition.estimated_makespan_ns > 0);
    }

    #[test]
    fn test_heft_respects_cc_constraints() {
        let mut graph = ShadowGraph::new();
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Kernel);
        node.estimated_cost_ns = 1000;
        node.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xAA,
            kernel_name: "cc90_kernel".to_string(),
            grid_dim: [256, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (9, 0), // Requires CC 9.0!
            workload_class: WorkloadClass::ComputeBound,
        });
        graph.add_node(node);
        graph.find_roots();
        graph.topological_sort();

        let gpu86 = make_test_profile(1, "CC 8.6 GPU", (8, 6), 12.0, 12.7, 360.0);
        let gpu90 = make_test_profile(2, "CC 9.0 GPU", (9, 0), 24.0, 82.6, 1008.0);
        compute_upward_ranks(&mut graph, &[gpu86.clone(), gpu90.clone()], 0.0);

        let partition = heft_assign(&mut graph, &[gpu86, gpu90], None);
        // Node 1 should be assigned to GPU 2 (CC 9.0)
        let assigned_to = graph.nodes[&1].assigned_gpu;
        assert_eq!(assigned_to, Some(2), "should be on CC 9.0 GPU");
    }

    #[test]
    fn test_heft_nccl_on_all_gpus() {
        let mut graph = ShadowGraph::new();
        let mut nccl = ShadowNode::new(1, 0x10, NodeClass::NcclCollective);
        nccl.estimated_cost_ns = 10000;
        graph.add_node(nccl);
        graph.find_roots();
        graph.topological_sort();

        let gpu1 = make_test_profile(1, "GPU 1", (8, 6), 12.0, 12.7, 360.0);
        let gpu2 = make_test_profile(2, "GPU 2", (8, 6), 12.0, 12.7, 360.0);
        compute_upward_ranks(&mut graph, &[gpu1.clone(), gpu2.clone()], 0.0);

        let partition = heft_assign(&mut graph, &[gpu1, gpu2], None);
        // NCCL node should appear in BOTH GPU assignments
        let gpu1_nodes = partition.gpu_assignments.get(&1).unwrap();
        let gpu2_nodes = partition.gpu_assignments.get(&2).unwrap();
        assert!(gpu1_nodes.contains(&1), "NCCL node should be on GPU 1");
        assert!(gpu2_nodes.contains(&1), "NCCL node should be on GPU 2");
    }

    // --- Partition plan cache tests ---

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = PartitionPlanCache::new(4);
        assert!(cache.get(12345).is_none());
        assert_eq!(cache.cache_misses(), 1);

        let plan = GraphPartition {
            gpu_assignments: HashMap::new(),
            cross_edges: Vec::new(),
            estimated_makespan_ns: 1000,
            total_transfer_bytes: 0,
            gpu_count: 1,
            strategy: PartitionStrategy::SingleGpu,
            source_topology_hash: 12345,
        };
        cache.insert(12345, plan);

        assert!(cache.get(12345).is_some());
        assert_eq!(cache.cache_hits(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = PartitionPlanCache::new(2);
        let make_plan = |hash| GraphPartition {
            gpu_assignments: HashMap::new(),
            cross_edges: Vec::new(),
            estimated_makespan_ns: 1000,
            total_transfer_bytes: 0,
            gpu_count: 1,
            strategy: PartitionStrategy::SingleGpu,
            source_topology_hash: hash,
        };

        cache.insert(1, make_plan(1));
        cache.insert(2, make_plan(2));
        assert_eq!(cache.len(), 2);

        // Insert a third -- should evict #1 (LRU)
        cache.insert(3, make_plan(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(1).is_none(), "1 should be evicted");
        assert!(cache.get(2).is_some(), "2 should still exist");
        assert!(cache.get(3).is_some(), "3 should exist");
    }

    // --- ShadowNode helper tests ---

    #[test]
    fn test_shadow_node_total_threads() {
        let mut node = ShadowNode::new(1, 0x10, NodeClass::Kernel);
        node.kernel_info = Some(KernelNodeInfo {
            func_handle: 0xAA,
            kernel_name: "test".to_string(),
            grid_dim: [128, 2, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            args_buffer: vec![],
            min_compute_capability: (7, 5),
            workload_class: WorkloadClass::ComputeBound,
        });
        assert_eq!(node.total_threads(), 128 * 2 * 256);
    }

    #[test]
    fn test_shadow_node_without_kernel_info() {
        let node = ShadowNode::new(1, 0x10, NodeClass::Empty);
        assert_eq!(node.total_threads(), 1);
    }

    // --- Integration: full pipeline test ---

    #[test]
    fn test_full_analysis_pipeline() {
        let mut graph = build_wide_parallel_graph();

        // Step 1: Find roots
        graph.find_roots();
        assert_eq!(graph.roots, vec![0]);

        // Step 2: Topological sort
        assert!(graph.topological_sort());
        assert_eq!(graph.topological_order.len(), 6);
        assert_eq!(graph.topological_order[0], 0); // source first
        assert_eq!(*graph.topological_order.last().unwrap(), 5); // sink last

        // Step 3: Compute topology hash
        graph.compute_topology_hash();
        assert_ne!(graph.topology_hash, 0);

        // Step 4: DAG analysis
        analyze_dag(&mut graph);
        assert!(graph.max_parallelism >= 4);
        assert!(graph.critical_path_length_ns > 0);

        // Step 5: Decide partition strategy
        graph.critical_path_length_ns = 20_000_000; // Bump to meet threshold
        let config = GraphAnalysisConfig::default();
        let strategy = decide_partition_strategy(&graph, &config, 2);
        assert_eq!(strategy, PartitionStrategy::HeftPartition);

        // Step 6: HEFT scheduling
        let gpu1 = make_test_profile(1, "GPU 1", (8, 6), 12.0, 12.7, 360.0);
        let gpu2 = make_test_profile(2, "GPU 2", (8, 6), 12.0, 12.7, 360.0);
        compute_upward_ranks(&mut graph, &[gpu1.clone(), gpu2.clone()], 0.001);
        let partition = heft_assign(&mut graph, &[gpu1, gpu2], None);

        assert!(!partition.gpu_assignments.is_empty());
        assert!(partition.estimated_makespan_ns > 0);
        assert_eq!(partition.strategy, PartitionStrategy::HeftPartition);
    }
}
