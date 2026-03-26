# R25: Cooperative Kernel Splitting --- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Second-round refinement incorporating cross-topic findings from R13, R17, R18, R23, R26 v2 designs. Defines exact Rust structs, kernel classification algorithm, PTX blockIdx injection, split launch orchestration, reduction combining, and go/no-go criteria.

---

## 1. Cross-Topic Integration Summary

| Source | What It Provides to R25 |
|--------|------------------------|
| R13 v2 | ShadowGraph with HEFT partitioning -- graph-level splitting is primary; R25 is the per-kernel supplement |
| R23 v2 | GpuProfile with calibrated TFLOPS scores, CC 7.5+ minimum for all features |
| R26 v2 | Coordinated launch with <5us jitter via hybrid PTP + GPU spin-wait loop |
| R17 v2 | Data placement determines which GPU has which data -- influences block assignment |
| R18 | Coherency for shared global memory across split kernel (Tier 2+ only) |

**Key findings from cross-topic analysis:**
- PTX blockIdx injection is the preferred strategy, proven in academic literature at 1.3-1.8x on 2 GPUs
- ~40-60% of compute-heavy kernels in typical ML workloads are expected to be GREEN/YELLOW
- Build the classifier FIRST as a go/no-go gate before investing in the splitter
- R13's graph context eliminates the data movement problem (data is already where it needs to be)

---

## 2. Rust Struct Definitions

### 2.1 KernelClassifier

Analyzes PTX at module load time to classify every kernel entry point.

```rust
/// Classifies CUDA kernels by analyzing PTX to determine splittability.
/// Runs at cuModuleLoadData / cuModuleLoadFatBinary interception time.
pub struct KernelClassifier {
    /// Cache of classification results: CUfunction -> classification
    cache: DashMap<CUfunction, KernelClassification>,

    /// Statistics for reporting classification distribution
    stats: ClassifierStats,

    /// PTX parser instance (reusable, not per-kernel)
    parser: PtxParser,

    /// Minimum grid size (total blocks) below which splitting is never worth it
    min_grid_size_for_split: u32, // default: 64

    /// Minimum estimated kernel duration (us) below which splitting overhead dominates
    min_kernel_duration_us: f64, // default: 20.0
}

/// Complete classification result for one kernel.
pub struct KernelClassification {
    /// Kernel function handle
    function: CUfunction,

    /// Kernel entry name from PTX (e.g., "_Z9addKernelPfS_S_i")
    entry_name: String,

    /// Traffic light classification
    color: KernelColor,

    /// Detailed reason for the classification
    reason: ClassificationReason,

    /// PTX analysis results
    analysis: PtxAnalysis,

    /// Whether the kernel has been successfully transformed (PTX rewritten)
    transformed: AtomicBool,

    /// Transformed PTX module (lazily created on first split attempt)
    transformed_module: OnceLock<TransformedModule>,
}

/// Traffic light classification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum KernelColor {
    /// Safe to split. No cross-block dependencies, blockIdx-linear access.
    /// Split freely with blockIdx offset injection.
    Green,

    /// Splittable with extra work. Has reduction atomics that can be
    /// redirected to per-GPU local copies with a merge step.
    Yellow,

    /// Do NOT split. Has cooperative group sync, CAS-based data structures,
    /// data-dependent access, or unanalyzable patterns.
    Red,
}

/// Why a kernel got its color.
pub enum ClassificationReason {
    /// GREEN: no atomics, no grid sync, blockIdx-linear access
    FullyIndependent,

    /// GREEN: has atomics but they are all on shared memory (block-local, no concern)
    SharedMemoryAtomicsOnly,

    /// YELLOW: has global atomicAdd/atomicSub (reduction pattern)
    GlobalReductionAtomic { atomic_count: u32, target_count: u32 },

    /// YELLOW: has global atomicMin/atomicMax (min/max reduction)
    GlobalMinMaxAtomic { atomic_count: u32, target_count: u32 },

    /// YELLOW: has histogram-like scatter atomics (atomicAdd to data-dependent bins)
    HistogramPattern { estimated_bin_count: u32 },

    /// RED: uses cooperative_groups grid sync
    CooperativeGridSync,

    /// RED: uses atomicCAS on global memory (lock-free data structure, cannot redirect)
    GlobalCasAtomic { cas_count: u32 },

    /// RED: data-dependent global memory access (cannot predict block-to-data mapping)
    DataDependentAccess,

    /// RED: indirect memory access through pointer loaded from global memory
    IndirectMemoryAccess,

    /// RED: could not parse PTX or SASS-only module with no PTX available
    UnanalyzablePtx { reason: String },

    /// RED: kernel uses dynamic parallelism (child kernel launches)
    DynamicParallelism,

    /// RED: kernel grid is too small to benefit from splitting
    GridTooSmall { block_count: u32 },
}

/// Results of PTX static analysis for one kernel.
pub struct PtxAnalysis {
    /// Number of parameters (total)
    param_count: u32,

    /// Number of pointer parameters (detected by .u64 + ld.global/st.global usage)
    pointer_param_count: u32,

    /// Pointer role classification (read-only, write-only, read-write)
    pointer_roles: Vec<PointerRole>,

    /// Global atomic operations found
    global_atomics: Vec<PtxAtomicInfo>,

    /// Shared memory atomic operations found (safe, block-local)
    shared_atomics: Vec<PtxAtomicInfo>,

    /// Whether cooperative group sync is used (bar.sync with grid-wide scope)
    has_grid_sync: bool,

    /// Whether dynamic parallelism is used (device-side kernel launch)
    has_dynamic_parallelism: bool,

    /// How %ctaid.x/y/z is used: linear (addr = f(ctaid)), or complex/indirect
    blockidx_usage: BlockIdxUsage,

    /// Number of %ctaid reads (all dimensions)
    ctaid_read_count: u32,

    /// Dimensions used: which of x, y, z are read
    dimensions_used: DimensionsUsed,

    /// Estimated compute intensity (instruction count, rough)
    estimated_instruction_count: u32,

    /// Shared memory usage (static + dynamic)
    shared_mem_bytes: u32,

    /// Register usage per thread (if parseable from .maxnreg or annotations)
    registers_per_thread: Option<u32>,
}

/// How a pointer parameter is used in the kernel.
#[derive(Clone, Copy)]
pub enum PointerRole {
    /// Only appears in ld.global instructions -- safe to replicate
    ReadOnly,

    /// Only appears in st.global instructions -- each GPU writes its own region
    WriteOnly,

    /// Appears in both ld.global and st.global -- needs coherency or careful partitioning
    ReadWrite,

    /// Appears as target of atom.global -- needs atomic redirection
    AtomicTarget,

    /// Cannot determine (indirect access, cast to different type, etc.)
    Unknown,
}

/// How blockIdx (ctaid) maps to memory addresses.
pub enum BlockIdxUsage {
    /// addr = ctaid * stride + base -- perfect for splitting
    Linear { stride_expressions: Vec<String> },

    /// addr = f(ctaid) where f is more complex but still deterministic
    Deterministic,

    /// addr depends on data loaded from memory -- cannot predict
    DataDependent,

    /// ctaid is not used for addressing (e.g., batch index only)
    NonAddressing,

    /// Could not determine
    Unknown,
}

#[derive(Clone, Copy, Default)]
pub struct DimensionsUsed {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

/// Info about one atomic instruction found in PTX.
pub struct PtxAtomicInfo {
    /// PTX instruction text (e.g., "atom.global.add.f32")
    instruction: String,

    /// Memory space: global or shared
    memory_space: MemorySpace,

    /// Atomic operation type
    op: AtomicOpType,

    /// Data type and size
    data_type: PtxDataType,

    /// Line number in PTX
    line: u32,

    /// Whether the target address is derived from a parameter (vs computed)
    target_is_param_derived: bool,
}

/// Statistics across all classified kernels.
pub struct ClassifierStats {
    pub total_classified: AtomicU64,
    pub green_count: AtomicU64,
    pub yellow_count: AtomicU64,
    pub red_count: AtomicU64,
    pub unanalyzable_count: AtomicU64,

    /// Green kernels' estimated compute time as fraction of total
    pub green_compute_fraction: AtomicU64, // stored as fixed-point * 1000

    /// Yellow kernels' estimated compute time as fraction of total
    pub yellow_compute_fraction: AtomicU64,
}
```

### 2.2 SplitDecision and BlockPartition

Runtime decision for a specific kernel launch.

```rust
/// Decision made at cuLaunchKernel interception time.
pub struct SplitDecision {
    /// Original kernel launch parameters
    original: LaunchParams,

    /// Classification of this kernel
    classification: Arc<KernelClassification>,

    /// Whether to actually split this launch
    should_split: bool,

    /// Reason for decision (for logging/diagnostics)
    decision_reason: SplitDecisionReason,

    /// If splitting: the partition plan
    partition: Option<BlockPartition>,

    /// If splitting: which GPUs participate
    target_gpus: Vec<GpuTarget>,

    /// If YELLOW: merge plan for combining partial results
    merge_plan: Option<MergePlan>,
}

pub enum SplitDecisionReason {
    /// Splitting: kernel is GREEN/YELLOW and meets size/duration thresholds
    Split { expected_speedup: f32 },

    /// Not splitting: kernel is RED
    RedClassification,

    /// Not splitting: grid too small (< min_grid_size_for_split)
    GridTooSmall { blocks: u32 },

    /// Not splitting: estimated duration below threshold
    DurationTooShort { estimated_us: f64 },

    /// Not splitting: only one GPU available
    SingleGpu,

    /// Not splitting: data not distributed (would need full replication)
    DataNotDistributed { replication_cost_us: f64 },

    /// Not splitting: R13 graph context says this kernel is not the bottleneck
    NotBottleneck { compute_fraction: f32 },
}

/// How blocks are divided across GPUs.
pub struct BlockPartition {
    /// Per-GPU assignment
    assignments: Vec<GpuBlockAssignment>,

    /// Original grid dimensions
    original_grid: Dim3,

    /// Split dimension (which axis we partition along)
    split_dimension: SplitDimension,
}

/// One GPU's block assignment.
pub struct GpuBlockAssignment {
    /// Target GPU
    gpu: GpuTarget,

    /// Grid dimensions for this GPU's launch
    grid_dim: Dim3,

    /// Block offset for each dimension (injected into PTX as __blkoff_x/y/z)
    block_offset: Dim3,

    /// Number of blocks assigned
    block_count: u32,

    /// Fraction of total blocks (for weighted heterogeneous splits)
    fraction: f32,

    /// Remapped kernel arguments (pointer args adjusted for this GPU)
    remapped_args: Vec<KernelArg>,
}

/// Target GPU with performance info from R23.
pub struct GpuTarget {
    /// Device ID in OuterLink's virtual device space
    device_id: DeviceId,

    /// Node ID
    node_id: NodeId,

    /// Calibrated TFLOPS score from R23 GpuProfile
    tflops_score: f32,

    /// SM count
    sm_count: u32,

    /// Compute capability (e.g., 8.6 for RTX 3090)
    compute_capability: (u32, u32),
}

/// Which dimension to split the grid along.
pub enum SplitDimension {
    /// Split gridDim.x (most common -- 1D grids, batch dimension)
    X,
    /// Split gridDim.y (2D grids -- row dimension)
    Y,
    /// Split gridDim.z (3D grids -- depth/batch dimension)
    Z,
}

pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
```

### 2.3 PtxTransformer

Modifies PTX to inject blockIdx offset parameters.

```rust
/// Transforms PTX source to inject blockIdx offset parameters.
/// Applied once at module load time, results cached per kernel.
pub struct PtxTransformer {
    /// Regex patterns for PTX parsing (compiled once, reused)
    patterns: PtxPatterns,

    /// Validation: compare transformed kernel output against original (debug mode)
    validation_enabled: bool,
}

/// Pre-compiled regex patterns for PTX transformation.
pub struct PtxPatterns {
    /// Matches kernel entry point: `.entry <name>(<params>)`
    entry_pattern: Regex,

    /// Matches parameter declarations: `.param .u64 <name>`
    param_pattern: Regex,

    /// Matches %ctaid.x/y/z reads: `mov.u32 %rN, %ctaid.x`
    ctaid_read_pattern: Regex,

    /// Matches register declarations: `.reg .u32 %rN`
    reg_decl_pattern: Regex,

    /// Matches atom.global instructions
    atom_global_pattern: Regex,

    /// Matches atom.shared instructions
    atom_shared_pattern: Regex,

    /// Matches bar.sync (grid-level sync indicator)
    bar_sync_pattern: Regex,

    /// Matches device-side kernel launch (dynamic parallelism)
    device_launch_pattern: Regex,
}

/// Result of transforming a PTX module.
pub struct TransformedModule {
    /// Modified PTX source text
    ptx_source: String,

    /// Names of injected offset parameters
    offset_params: OffsetParams,

    /// For YELLOW kernels: redirected atomic targets
    redirected_atomics: Vec<RedirectedAtomic>,

    /// CUmodule handle after loading transformed PTX
    loaded_module: OnceLock<CUmodule>,

    /// CUfunction handle for the transformed kernel
    loaded_function: OnceLock<CUfunction>,
}

/// Injected offset parameter names.
pub struct OffsetParams {
    /// Parameter name for X offset (e.g., "__outerlink_blkoff_x")
    x: Option<String>,
    /// Parameter name for Y offset
    y: Option<String>,
    /// Parameter name for Z offset
    z: Option<String>,
}

/// An atomic instruction that was redirected to a per-GPU local copy.
pub struct RedirectedAtomic {
    /// Original PTX line number
    original_line: u32,

    /// Original instruction text
    original_instruction: String,

    /// New parameter name for the local atomic target
    local_target_param: String,

    /// Atomic operation type (needed for merge step)
    op_type: AtomicOpType,

    /// Data type (needed for merge step)
    data_type: PtxDataType,
}

/// Atomic operation types for merge step planning.
#[derive(Clone, Copy)]
pub enum AtomicOpType {
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
    Inc,
    Dec,
}

#[derive(Clone, Copy)]
pub enum PtxDataType {
    S32,
    U32,
    S64,
    U64,
    F32,
    F64,
}
```

### 2.4 Split Launch Orchestrator

```rust
/// Coordinates split kernel launches across multiple GPUs.
/// Integrates with R26 PTP for synchronized launch timing.
pub struct SplitLaunchOrchestrator {
    /// Kernel classifier (shared, module-load-time classification)
    classifier: Arc<KernelClassifier>,

    /// PTX transformer (shared)
    transformer: Arc<PtxTransformer>,

    /// R26 PTP clock for coordinated launches
    clock: Arc<PtpClock>,

    /// R17 topology for GPU selection
    topology: Arc<TopologySnapshot>,

    /// R23 GPU profiles for weighted splitting
    gpu_profiles: Vec<GpuProfile>,

    /// R13 graph context (optional -- provides data placement info)
    graph_context: Option<Arc<ShadowGraph>>,

    /// Merge executor for YELLOW kernel post-processing
    merge_executor: MergeExecutor,

    /// Statistics
    stats: OrchestratorStats,
}

/// Merge plan for combining partial results from YELLOW kernels.
pub struct MergePlan {
    /// Per-atomic-target merge operations
    merge_ops: Vec<MergeOp>,

    /// Total estimated merge time (us)
    estimated_merge_time_us: f64,
}

/// One merge operation for one redirected atomic target.
pub struct MergeOp {
    /// Original target address in the application's memory
    original_target: VirtualAddr,

    /// Per-GPU local copy addresses
    local_copies: Vec<(DeviceId, VirtualAddr)>,

    /// How to combine: depends on atomic type
    combine: CombineStrategy,

    /// Data type for the merge
    data_type: PtxDataType,

    /// Number of elements (1 for scalar, N for array/histogram)
    element_count: u32,
}

/// How partial results are combined.
#[derive(Clone, Copy)]
pub enum CombineStrategy {
    /// Sum all partial results (atomicAdd)
    Sum,
    /// Take minimum across all partials (atomicMin)
    Min,
    /// Take maximum across all partials (atomicMax)
    Max,
    /// Bitwise AND across all partials
    BitwiseAnd,
    /// Bitwise OR across all partials
    BitwiseOr,
    /// Bitwise XOR across all partials
    BitwiseXor,
}

/// Executes merge operations after split kernel completion.
pub struct MergeExecutor {
    /// Pre-allocated merge buffers per GPU
    merge_buffers: DashMap<DeviceId, MergeBuffer>,

    /// Whether to use a GPU kernel for large merges (histogram)
    /// or host-side merge for small ones (scalar counter)
    large_merge_threshold: u32, // elements; default: 1024
}
```

---

## 3. Kernel Classification Algorithm: Decision Tree

### 3.1 The Exact Decision Tree

```
Input: PTX source for one kernel entry point
Output: KernelColor (Green, Yellow, Red)

Step 1: CAN WE ANALYZE IT?
  |
  Is PTX available? (not SASS-only)
    |-- NO -> RED (UnanalyzablePtx)
    |-- YES:
        |
  Can we parse the PTX? (valid syntax, entry point found)
    |-- NO -> RED (UnanalyzablePtx)
    |-- YES -> continue

Step 2: HARD BLOCKERS (any one = RED)
  |
  Does the kernel use cooperative group grid sync?
  (Detect: bar.sync with .cta scope, or calls to cooperative_groups::grid_group)
    |-- YES -> RED (CooperativeGridSync)
    |-- NO -> continue
  |
  Does the kernel use dynamic parallelism?
  (Detect: device-side cudaLaunchKernel, or PTX .entry calls from device code)
    |-- YES -> RED (DynamicParallelism)
    |-- NO -> continue

Step 3: ATOMIC ANALYSIS
  |
  Does the kernel have atom.global.cas instructions?
    |-- YES:
        Is the CAS part of a lock-free data structure pattern?
        (Detect: CAS in a retry loop with branch-back, OR CAS result
         used as a branch condition for retry)
          |-- YES -> RED (GlobalCasAtomic)
          |-- NO:
              Is it a simple exchange pattern? (CAS used for atomicExch)
                |-- YES -> continue (treat as YELLOW with exchange merge)
                |-- NO -> RED (GlobalCasAtomic) // conservative

  Does the kernel have atom.global.add / atom.global.sub?
    |-- YES:
        How many unique target addresses? (count distinct base registers)
          |-- 1-16 targets -> YELLOW (GlobalReductionAtomic)
          |-- >16 targets, derived from data -> YELLOW (HistogramPattern)
          |-- >16 targets, from %ctaid -> GREEN (each block writes its own target)

  Does the kernel have atom.global.min / atom.global.max?
    |-- YES:
        Same logic as add/sub:
          |-- 1-16 targets -> YELLOW (GlobalMinMaxAtomic)
          |-- >16 -> YELLOW (HistogramPattern) or RED depending on pattern

  Does the kernel have atom.global.and / atom.global.or / atom.global.xor?
    |-- YES -> YELLOW (same reduction logic)

  Does the kernel have ONLY atom.shared instructions (no atom.global)?
    |-- YES -> Continue (shared atomics are block-local, no splitting concern)

Step 4: MEMORY ACCESS PATTERN
  |
  How does the kernel derive global memory addresses?
    |
  Trace data flow from %ctaid.x/y/z to ld.global/st.global addresses:
    |
    Pattern A: addr = %ctaid * constant + parameter_base
      -> LINEAR. Perfect splitting. Mark as Green candidate.
    |
    Pattern B: addr = f(%ctaid, %tid) where f is arithmetic only
      -> DETERMINISTIC. Still splittable. Mark as Green candidate.
    |
    Pattern C: addr = parameter[loaded_value] (pointer chasing)
      -> DATA_DEPENDENT. Cannot predict. Mark as Red candidate.
    |
    Pattern D: addr = parameter + loaded_index (indirect indexing)
      -> DATA_DEPENDENT. Mark as Red candidate.
    |
    Pattern E: %ctaid is not used in address computation at all
      (kernel uses only %tid, or addresses from parameters directly)
      -> Check if all blocks access the same data (broadcast pattern):
         |-- YES: Green (replicate data to each GPU)
         |-- UNCLEAR: Red (conservative)

Step 5: FINAL CLASSIFICATION
  |
  Any RED from steps 2-4? -> RED (use worst reason)
  |
  Any YELLOW from step 3? -> YELLOW
  |
  All GREEN from steps 3-4? -> GREEN
```

### 3.2 Classification Examples

| Kernel | Atomics | Access Pattern | Color | Reason |
|--------|---------|---------------|-------|--------|
| Elementwise add: `C[i] = A[i] + B[i]` | None | Linear (%ctaid * blockDim + %tid) | GREEN | FullyIndependent |
| Tiled GEMM | None | Deterministic (tile coords from %ctaid) | GREEN | FullyIndependent |
| Sum reduction: `atomicAdd(&total, partial)` | 1 global add target | Linear | YELLOW | GlobalReductionAtomic |
| Histogram: `atomicAdd(&hist[data[i]], 1)` | N global add targets | Data-dependent bin | YELLOW | HistogramPattern |
| Loss + gradient: `atomicAdd(&loss, local_loss)` | 1 global add | Linear for gradient, 1 target for loss | YELLOW | GlobalReductionAtomic |
| Flash Attention (logsumexp) | atomicMax for logsumexp | Complex | YELLOW | GlobalMinMaxAtomic |
| Jacobi iteration with grid.sync() | None | Linear | RED | CooperativeGridSync |
| Lock-free queue: `atomicCAS(&head, old, new)` | CAS in retry loop | Data-dependent | RED | GlobalCasAtomic |
| GNN neighbor aggregation | None | Indirect (graph edges) | RED | DataDependentAccess |
| cuBLAS GEMM (SASS-only) | N/A | N/A | RED | UnanalyzablePtx |

---

## 4. PTX blockIdx Injection: Exact Transformation

### 4.1 Transformation Steps

Given original PTX for a kernel entry:

```
// ORIGINAL PTX
.visible .entry _Z9addKernelPfS_S_i(
    .param .u64 _Z9addKernelPfS_S_i_param_0,  // float* a
    .param .u64 _Z9addKernelPfS_S_i_param_1,  // float* b
    .param .u64 _Z9addKernelPfS_S_i_param_2,  // float* c
    .param .s32 _Z9addKernelPfS_S_i_param_3   // int n
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;

    mov.u32     %r1, %ctaid.x;
    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %tid.x;
    mad.lo.s32  %r4, %r1, %r2, %r3;        // idx = blockIdx.x * blockDim.x + threadIdx.x
    ld.param.s32 %r5, [_Z9addKernelPfS_S_i_param_3];
    setp.ge.s32 %p1, %r4, %r5;
    @%p1 bra    $L__BB0_2;
    // ... load a[idx], b[idx], compute, store c[idx] ...
$L__BB0_2:
    ret;
}
```

**Transformation produces:**

```
// TRANSFORMED PTX
.visible .entry _Z9addKernelPfS_S_i(
    .param .u64 _Z9addKernelPfS_S_i_param_0,  // float* a
    .param .u64 _Z9addKernelPfS_S_i_param_1,  // float* b
    .param .u64 _Z9addKernelPfS_S_i_param_2,  // float* c
    .param .s32 _Z9addKernelPfS_S_i_param_3,  // int n
    .param .u32 __outerlink_blkoff_x           // INJECTED: blockIdx.x offset
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<8>;                         // EXPANDED: need extra registers
    .reg .b64   %rd<8>;

    // INJECTED: load offset and adjust ctaid
    ld.param.u32 %r6, [__outerlink_blkoff_x];
    mov.u32     %r1, %ctaid.x;
    add.u32     %r1, %r1, %r6;                // %r1 = ctaid.x + offset

    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %tid.x;
    mad.lo.s32  %r4, %r1, %r2, %r3;           // idx = (blockIdx.x + offset) * blockDim.x + threadIdx.x
    ld.param.s32 %r5, [_Z9addKernelPfS_S_i_param_3];
    setp.ge.s32 %p1, %r4, %r5;
    @%p1 bra    $L__BB0_2;
    // ... unchanged kernel body ...
$L__BB0_2:
    ret;
}
```

### 4.2 Transformation Algorithm (Pseudocode)

```rust
fn transform_ptx(ptx: &str, analysis: &PtxAnalysis) -> Result<TransformedModule> {
    let mut output = String::with_capacity(ptx.len() + 1024);
    let dims = &analysis.dimensions_used;
    let mut offset_params = OffsetParams::default();

    // Pass 1: Find entry point and parameter list
    let entry_match = patterns.entry_pattern.find(ptx)?;
    let param_end = find_closing_paren(ptx, entry_match.end());

    // Pass 2: Inject offset parameters into parameter list
    let mut param_injection = String::new();
    if dims.x {
        param_injection += ",\n    .param .u32 __outerlink_blkoff_x";
        offset_params.x = Some("__outerlink_blkoff_x".into());
    }
    if dims.y {
        param_injection += ",\n    .param .u32 __outerlink_blkoff_y";
        offset_params.y = Some("__outerlink_blkoff_y".into());
    }
    if dims.z {
        param_injection += ",\n    .param .u32 __outerlink_blkoff_z";
        offset_params.z = Some("__outerlink_blkoff_z".into());
    }

    // Insert params before closing paren
    output.push_str(&ptx[..param_end]);
    output.push_str(&param_injection);
    output.push_str(&ptx[param_end..]);

    // Pass 3: Find register declaration block, expand register count
    // Need extra registers for offset loads and adjusted ctaid values
    let extra_regs_needed = dims.x as u32 + dims.y as u32 + dims.z as u32;
    expand_register_declarations(&mut output, extra_regs_needed);

    // Pass 4: Inject offset load + add after first instruction in the body
    // For each dimension used, insert:
    //   ld.param.u32 %rN, [__outerlink_blkoff_<dim>];
    // Then find every `mov.u32 %rM, %ctaid.<dim>` and append:
    //   add.u32 %rM, %rM, %rN;
    for dim in ["x", "y", "z"] {
        if !dim_is_used(dims, dim) { continue; }

        let offset_reg = allocate_register(&mut output);
        let param_name = format!("__outerlink_blkoff_{}", dim);

        // Insert load right after the opening brace of the function body
        let load_instr = format!(
            "    ld.param.u32 {}, [{}];\n",
            offset_reg, param_name
        );
        insert_after_body_open(&mut output, &load_instr);

        // Find all `mov.u32 %rN, %ctaid.<dim>` and insert add after each
        let ctaid_pattern = format!("mov.u32\\s+(%.+),\\s+%ctaid.{}", dim);
        for m in Regex::new(&ctaid_pattern)?.find_iter(&output.clone()) {
            let target_reg = extract_register(m.as_str());
            let add_instr = format!(
                "    add.u32 {}, {}, {};\n",
                target_reg, target_reg, offset_reg
            );
            insert_after_line(&mut output, m.end(), &add_instr);
        }
    }

    // Pass 5 (YELLOW only): Redirect atomic targets to local copies
    let redirected = if analysis.global_atomics.iter().any(|a| a.memory_space == MemorySpace::Global) {
        redirect_atomics(&mut output, &analysis.global_atomics)?
    } else {
        vec![]
    };

    Ok(TransformedModule {
        ptx_source: output,
        offset_params,
        redirected_atomics: redirected,
        loaded_module: OnceLock::new(),
        loaded_function: OnceLock::new(),
    })
}
```

### 4.3 Multi-Dimensional Grid Handling

For 2D/3D grids, we split along the largest dimension first:

```
Original grid: (256, 128, 1) -- 256 blocks in X, 128 in Y

Split strategy:
  - Split X dimension (larger): GPU A gets X=[0,127], GPU B gets X=[128,255]
  - Y dimension unchanged on both GPUs
  - GPU A: grid=(128, 128, 1), __blkoff_x=0
  - GPU B: grid=(128, 128, 1), __blkoff_x=128
```

For heterogeneous splits (R23 weighted):

```
GPU A (3090, score=1.0): gets 60% of blocks -> grid=(154, 128, 1), __blkoff_x=0
GPU B (3060, score=0.6): gets 40% of blocks -> grid=(102, 128, 1), __blkoff_x=154
```

---

## 5. Split Launch Orchestration: Coordinated Multi-GPU Launch

### 5.1 Launch Sequence

```
cuLaunchKernel intercepted
    |
    v
Step 1: CLASSIFY
  classification = classifier.get_or_classify(function, ptx)
  If RED -> forward to single GPU, return

Step 2: DECIDE
  decision = make_split_decision(classification, launch_params, graph_context)
  If !should_split -> forward to single GPU, return

Step 3: TRANSFORM (cached, only first time)
  transformed = get_or_transform(function, ptx, classification)

Step 4: PREPARE PER-GPU LAUNCHES
  For each GPU in target_gpus:
    a. Load transformed module on this GPU (cuModuleLoadDataEx, cached)
    b. Get transformed function handle (cuModuleGetFunction)
    c. Compute block offset for this GPU's partition
    d. Remap pointer arguments:
       - ReadOnly pointers: ensure data is replicated (or use R18 coherency)
       - WriteOnly pointers: point to this GPU's output region
       - AtomicTarget pointers: point to per-GPU local copy (YELLOW)
    e. Build argument array: original args + __blkoff_x/y/z
    f. Create CUDA stream on this GPU (or reuse from pool)

Step 5: SYNCHRONIZED LAUNCH (R26 PTP integration)
  target_time = ptp_clock.now() + LAUNCH_LEAD_TIME_US  // e.g., now + 50us
  For each GPU (in parallel, one thread per GPU):
    a. Set CUDA context to this GPU
    b. Spin-wait until ptp_clock.now() >= target_time - GPU_SPIN_THRESHOLD
       (hybrid PTP + GPU spin achieves <5us jitter per R26 v2)
    c. cuLaunchKernel(transformed_function, gpu_grid, blockDim, sharedMem,
                      gpu_stream, gpu_args, NULL)

Step 6: WAIT FOR COMPLETION
  For each GPU (in parallel):
    cuStreamSynchronize(gpu_stream)

Step 7: MERGE (YELLOW only)
  If merge_plan is Some:
    For each MergeOp in merge_plan.merge_ops:
      Execute combine strategy:
        - Scalar (1 element): host-side merge
            Read local_copy from each GPU -> combine -> write to original_target
        - Array (>1024 elements): launch merge kernel
            merge_kernel<<<ceil(elements/256), 256>>>(
                original_target, local_copy_A, local_copy_B, elements)
        - Medium (2-1024 elements): host-side loop

Step 8: CLEANUP
  Return partial results to per-GPU buffer pool
  Update orchestrator stats (split count, merge time, total speedup)
  Return CUDA_SUCCESS to application
```

### 5.2 Launch Timing with R26 PTP

R26 v2 specifies a hybrid approach for <5us launch jitter:

1. **PTP coarse alignment:** All nodes agree on a future launch timestamp via PTP-synchronized clocks. ConnectX-5 hardware PTP provides sub-microsecond accuracy.

2. **GPU spin-wait fine alignment:** Each GPU launches a tiny spin kernel that polls a flag in pinned memory. The host sets the flag at the PTP-agreed time. GPU polls at ~100ns granularity.

```rust
/// Coordinate kernel launch across N GPUs with <5us jitter.
fn synchronized_launch(
    launches: &[PreparedLaunch],
    clock: &PtpClock,
) -> Result<()> {
    // Agree on a launch time: now + lead time (enough for setup on all GPUs)
    let launch_time = clock.now_ns() + LAUNCH_LEAD_TIME_NS; // e.g., 50_000ns = 50us

    // Phase 1: Set up all GPUs (in parallel)
    let handles: Vec<_> = launches.iter().map(|launch| {
        std::thread::spawn(move || {
            cuda_set_device(launch.device_id);
            // Pre-stage: copy args, set up stream, load module if needed
            launch.prepare();
        })
    }).collect();
    for h in handles { h.join()?; }

    // Phase 2: Spin-wait and launch (in parallel, one thread per GPU)
    let barrier = Arc::new(std::sync::Barrier::new(launches.len()));
    let handles: Vec<_> = launches.iter().map(|launch| {
        let clock = clock.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            cuda_set_device(launch.device_id);

            // CPU spin-wait until close to launch time
            while clock.now_ns() < launch_time - 1_000 {
                std::hint::spin_loop();
            }

            // Final barrier: all threads are within ~1us of launch time
            barrier.wait();

            // Launch immediately
            unsafe { cuLaunchKernel(/* ... */) };
        })
    }).collect();
    for h in handles { h.join()?; }

    Ok(())
}
```

### 5.3 Data Placement Awareness (R17 Integration)

Before deciding to split, check where data lives:

```rust
fn assess_data_placement(
    args: &[KernelArg],
    pointer_roles: &[PointerRole],
    target_gpus: &[GpuTarget],
    page_table: &PageTable,
) -> DataPlacementAssessment {
    let mut replication_needed_bytes = 0u64;
    let mut already_distributed = true;

    for (arg, role) in args.iter().zip(pointer_roles) {
        if let KernelArg::Pointer { addr, size } = arg {
            match role {
                PointerRole::ReadOnly => {
                    // Check if data is already present on all target GPUs
                    // (R12 dedup pages are always present everywhere)
                    let coverage = check_page_coverage(*addr, *size, target_gpus, page_table);
                    if coverage < 1.0 {
                        replication_needed_bytes += ((1.0 - coverage) * *size as f64) as u64;
                    }
                }
                PointerRole::WriteOnly | PointerRole::ReadWrite => {
                    // Check if data is partitioned across GPUs (ideal for split)
                    let distribution = check_page_distribution(*addr, *size, target_gpus, page_table);
                    if !distribution.is_partitioned {
                        already_distributed = false;
                    }
                }
                _ => {}
            }
        }
    }

    DataPlacementAssessment {
        replication_needed_bytes,
        already_distributed,
        estimated_replication_time_us: replication_needed_bytes as f64 / TRANSPORT_BANDWIDTH_BYTES_PER_US,
    }
}
```

---

## 6. Reduction Combining: Merge Partial Results

### 6.1 Scalar Merge (1-16 elements)

For simple reductions like `atomicAdd(&total_loss, local_loss)`:

```rust
fn merge_scalar(
    original_target: VirtualAddr,
    local_copies: &[(DeviceId, VirtualAddr)],
    combine: CombineStrategy,
    data_type: PtxDataType,
) -> Result<()> {
    // Read partial results from each GPU to host
    let mut partials = Vec::with_capacity(local_copies.len());
    for (device, addr) in local_copies {
        let value = cuda_memcpy_dtoh_scalar(*device, *addr, data_type)?;
        partials.push(value);
    }

    // Combine on host
    let result = match combine {
        CombineStrategy::Sum => partials.iter().fold(0u64, |acc, &v| acc.wrapping_add(v)),
        CombineStrategy::Min => *partials.iter().min().unwrap(),
        CombineStrategy::Max => *partials.iter().max().unwrap(),
        CombineStrategy::BitwiseAnd => partials.iter().fold(!0u64, |acc, &v| acc & v),
        CombineStrategy::BitwiseOr => partials.iter().fold(0u64, |acc, &v| acc | v),
        CombineStrategy::BitwiseXor => partials.iter().fold(0u64, |acc, &v| acc ^ v),
    };

    // Write combined result to original target
    cuda_memcpy_htod_scalar(get_device(original_target), original_target, result, data_type)?;

    Ok(())
}
```

**Cost:** 2 * N small cudaMemcpy (DtoH + HtoD) + host arithmetic. Total: ~5-20us for 2 GPUs.

### 6.2 Array/Histogram Merge (>1024 elements)

For histograms or large reduction arrays, launch a merge kernel:

```rust
fn merge_array(
    original_target: VirtualAddr,
    local_copies: &[(DeviceId, VirtualAddr)],
    combine: CombineStrategy,
    data_type: PtxDataType,
    element_count: u32,
) -> Result<()> {
    // Copy all partial arrays to the target GPU
    let target_device = get_device(original_target);
    let mut staging_addrs = Vec::new();

    for (device, addr) in local_copies {
        if *device == target_device {
            staging_addrs.push(*addr); // Already on target GPU
        } else {
            // P2P copy to target GPU
            let staging = cuda_malloc(target_device, element_count * data_type.size())?;
            cuda_memcpy_peer(staging, target_device, *addr, *device,
                             element_count as usize * data_type.size())?;
            staging_addrs.push(staging);
        }
    }

    // Launch merge kernel on target GPU
    // merge_kernel<<<ceil(element_count/256), 256>>>(
    //     output, partial_A, partial_B, element_count, combine_op
    // )
    launch_merge_kernel(
        target_device,
        original_target,
        &staging_addrs,
        element_count,
        combine,
        data_type,
    )?;

    // Free staging buffers
    for (i, (device, _)) in local_copies.iter().enumerate() {
        if *device != target_device {
            cuda_free(target_device, staging_addrs[i])?;
        }
    }

    Ok(())
}
```

**Cost for 1M-element histogram:** ~80us (P2P copy of 4MB at 12.5 GB/s) + ~10us (merge kernel) = ~90us total. Acceptable if the kernel itself ran for >200us.

### 6.3 Float Reduction Accuracy

Floating-point addition is not associative. Splitting a float reduction across GPUs changes the summation order, producing slightly different results.

```
Single GPU:  total = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
Split (2 GPU): total = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7)
```

The numerical difference is typically within 1-2 ULPs (units in the last place) and is acceptable for ML training (which already uses stochastic gradient descent with inherent noise). For scientific computing requiring exact reproducibility, kernel splitting should be disabled.

---

## 7. Go/No-Go Analysis: Concrete Criteria

### 7.1 Phase A Gate: Classifier Results

Build the classifier, run it on real workloads, measure the distribution.

**GO criteria (proceed to Phase B):**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| GREEN kernels by count | >20% | Enough kernels to split |
| GREEN kernels by compute time | >30% | The splittable kernels consume meaningful compute |
| GREEN + YELLOW by compute time | >40% | Combined coverage justifies the investment |
| PTX availability | >70% of kernels | If most are SASS-only, approach is fundamentally limited |

**NO-GO criteria (stop, R25 is not worth it):**

| Metric | Threshold | Implication |
|--------|-----------|-------------|
| GREEN + YELLOW by compute time | <20% | Too few kernels benefit. Graph splitting (R13) alone is sufficient. |
| PTX availability | <50% | Most kernels are SASS-only. Cannot analyze or transform. |
| SASS-only for top 5 hottest kernels | YES | The kernels that matter most cannot be split. |

**Test workloads for classifier evaluation:**
1. PyTorch ResNet-50 inference (vision)
2. PyTorch GPT-2 inference (language)
3. PyTorch ResNet-50 training forward+backward pass
4. Custom CUDA elementwise benchmark
5. CUTLASS GEMM (PTX-compilable)

### 7.2 Phase B Gate: Splitting Efficiency

Build the splitter for GREEN kernels, measure actual speedup.

**GO criteria (proceed to Phase C):**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Elementwise kernel speedup (2 GPUs) | >1.7x | Must be close to theoretical 2x for trivial case |
| GEMM kernel speedup (2 GPUs, data pre-distributed) | >1.3x | Academic literature achieves 1.3-1.8x |
| Split overhead (launch coordination) | <10us | Must be negligible for kernels >100us |
| Correctness | 100% bit-identical for GREEN | Zero tolerance for incorrect results |

**NO-GO criteria (stop at Phase B):**

| Metric | Threshold | Implication |
|--------|-----------|-------------|
| Best-case speedup | <1.2x | Overhead too high, splitting not viable over network |
| PTX transformation failures | >10% of GREEN kernels | Transformer is not robust enough |
| Correctness failures | ANY | Fundamental problem with approach |

### 7.3 Phase C Gate: YELLOW Kernel Value

**GO criteria (proceed to Phase D):**

| Metric | Threshold |
|--------|-----------|
| YELLOW kernel speedup with merge | >1.2x |
| Merge overhead as fraction of kernel time | <20% |
| Correct merge results for all atomic types | 100% |

### 7.4 Investment Decision Matrix

| Classifier Result | Phase B Result | Decision |
|-------------------|---------------|----------|
| GREEN >30% compute | Speedup >1.5x | FULL INVEST: Build through Phase D |
| GREEN >30% compute | Speedup 1.2-1.5x | MODERATE: Build Phase B+C, skip D |
| GREEN 20-30% compute | Any | MINIMAL: Build Phase B only, evaluate |
| GREEN <20% compute | N/A | STOP: R13 graph splitting is sufficient |

---

## 8. Revised Effort Estimates

| Phase | Weeks | Depends On | Output |
|-------|-------|-----------|--------|
| A: Classifier | 3-4 | Phase 3 (cuModuleLoadData interception) | Classification report for real workloads |
| B: GREEN Splitter | 5-7 | Phase A + Phase 4 (transport) + R26 (PTP) | Working split for independent kernels |
| C: YELLOW Merge | 3-5 | Phase B | Atomic redirection + merge for reductions |
| D: Graph Integration | 5-7 | Phase C + R13 (ShadowGraph) | Full integration with graph engine |

**Total if all phases: 16-23 weeks.** But each phase has a go/no-go gate. Realistic expectation: invest in A+B (8-11 weeks), evaluate, then decide on C+D.

---

## 9. What Is Realistic vs Aspirational

### Realistic (HIGH confidence)

| Feature | Confidence | Why |
|---------|-----------|-----|
| PTX interception at cuModuleLoadData | 95% | Standard LD_PRELOAD interception |
| Kernel classification (GREEN/YELLOW/RED) | 90% | Static PTX analysis is well-understood |
| blockIdx offset injection for 1D grids | 85% | PTX is text-based, regex transformation is feasible |
| GREEN kernel splitting (elementwise, GEMM) | 80% | Academic precedent, 1.3-1.8x demonstrated |
| Launch coordination with <10us overhead | 85% | R26 PTP + RDMA signaling |

### Aspirational (MEDIUM confidence)

| Feature | Confidence | Caveat |
|---------|-----------|--------|
| 2D/3D grid splitting | 65% | More complex PTX transformation, less tested |
| YELLOW kernel merge for reductions | 70% | Correct but float non-associativity may surprise users |
| Histogram merge (large bin count) | 60% | Merge kernel overhead may negate splitting benefit |
| SASS-to-PTX forcing (strip SASS from fat binary) | 50% | NVIDIA may not JIT all PTX correctly; driver version dependent |
| >2 GPU splitting | 45% | Communication overhead scales; diminishing returns after 2 |

### NOT Realistic (do not promise)

| Feature | Why Not |
|---------|---------|
| Splitting cuBLAS/cuDNN library kernels | SASS-only, no PTX available. Use their built-in multi-GPU APIs instead. |
| Splitting cooperative group kernels | Network sync latency (2-5us per point) destroys benefit |
| Linear scaling (Nx on N GPUs) | Communication overhead always takes a cut. Expect 1.3-1.8x on 2 GPUs. |
| Transparent correctness for all kernels | Static analysis cannot prove safety for arbitrary code. Conservative RED default. |
| Splitting kernels with data-dependent access | Cannot predict memory access without running the kernel. Cannot split safely. |

---

## Related Documents

- [preplan.md](preplan.md) -- v1 pre-plan
- [research/01-cuda-execution-model.md](research/01-cuda-execution-model.md) -- CUDA execution model
- [research/02-kernel-splitting-strategies.md](research/02-kernel-splitting-strategies.md) -- Splitting strategies
- [research/03-practical-feasibility.md](research/03-practical-feasibility.md) -- Feasibility assessment
- R13: CUDA Graph Interception -- ShadowGraph, HEFT partitioning (build FIRST)
- R17: Topology-Aware Scheduling -- Data placement awareness
- R18: Virtual NVLink -- Coherency for shared memory across split kernels
- R23: Heterogeneous GPU Mixing -- GpuProfile, weighted block assignment
- R26: PTP Clock Sync -- <5us jitter coordinated launches

## Open Questions (v2)

| # | Question | Status |
|---|----------|--------|
| Q1 | Can we force PTX JIT by stripping SASS from fat binaries? | OPEN -- needs testing with cuModuleLoadFatBinary |
| Q2 | Does CUTLASS provide PTX-analyzable kernels for GEMM? | OPEN -- CUTLASS templates compile to PTX, should be analyzable |
| Q3 | Float reduction non-associativity: acceptable for training? | RESOLVED -- YES. SGD noise >> floating point rounding differences. Document for scientific users. |
| Q4 | Can the classifier run asynchronously (background thread)? | RESOLVED -- YES. Classify at module load time (background), result ready before first launch. |
| Q5 | Should we support >2 GPU splits? | DEFERRED -- Start with 2 GPUs. Add N-way after Phase B proves the concept. Communication overhead likely makes >4 impractical. |
| Q6 | Kernel argument size limit: does adding 12 bytes (3x u32 offsets) break any kernels? | OPEN -- CUDA limit is 4KB. Need to check if any real kernels are near this limit. Very unlikely. |
