//! R25: Cooperative Kernel Splitting
//!
//! Splits a single CUDA kernel across multiple GPUs by analyzing PTX at module
//! load time to classify kernel splittability, partitioning thread blocks across
//! GPUs based on capability scores, and coordinating synchronized launches with
//! optional merge steps for reduction kernels.
//!
//! # Architecture
//!
//! ```text
//! cuModuleLoadData interception
//!    |
//!    v
//! [KernelClassifier] -- PTX static analysis -> KernelColor (Green/Yellow/Red)
//!    |
//!    v
//! cuLaunchKernel interception
//!    |
//!    v
//! [SplitDecision] -- cost model -> split or forward to single GPU
//!    |
//!    v
//! [PtxTransformer] -- blockIdx offset injection (cached)
//!    |
//!    v
//! [SplitLaunchOrchestrator] -- coordinated multi-GPU launch + merge
//! ```
//!
//! # Integration Points
//!
//! - R13 (CUDA Graph): ShadowGraph provides graph-level context for split decisions.
//! - R23 (GPU Mixing): GpuProfile provides calibrated TFLOPS scores for weighted splitting.
//! - R26 (PTP Clock): Coordinated launch timing with <5us jitter.
//! - R17 (Topology): Data placement awareness for block assignment.
//! - R18 (Coherency): Shared global memory coherency for split kernel data.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::cuda_types::CuFunction;
use crate::gpu_mixing::GpuId;
use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Virtual address in CUDA device memory.
pub type VirtualAddr = u64;

/// Device identifier in OuterLink's virtual device space.
pub type DeviceId = u32;

// ---------------------------------------------------------------------------
// Kernel classification types
// ---------------------------------------------------------------------------

/// Traffic light classification for kernel splittability.
///
/// Determines whether a CUDA kernel can be safely split across multiple GPUs.
/// Classification is based on static PTX analysis at module load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Why a kernel received its classification color.
///
/// Provides detailed reasoning for diagnostic and debugging purposes.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationReason {
    /// GREEN: no atomics, no grid sync, blockIdx-linear access.
    FullyIndependent,

    /// GREEN: has atomics but they are all on shared memory (block-local).
    SharedMemoryAtomicsOnly,

    /// YELLOW: has global atomicAdd/atomicSub (reduction pattern).
    GlobalReductionAtomic {
        /// Number of atomic instructions found.
        atomic_count: u32,
        /// Number of distinct target addresses.
        target_count: u32,
    },

    /// YELLOW: has global atomicMin/atomicMax (min/max reduction).
    GlobalMinMaxAtomic {
        /// Number of atomic instructions found.
        atomic_count: u32,
        /// Number of distinct target addresses.
        target_count: u32,
    },

    /// YELLOW: has histogram-like scatter atomics (atomicAdd to data-dependent bins).
    HistogramPattern {
        /// Estimated number of histogram bins.
        estimated_bin_count: u32,
    },

    /// RED: uses cooperative_groups grid sync.
    CooperativeGridSync,

    /// RED: uses atomicCAS on global memory (lock-free data structure).
    GlobalCasAtomic {
        /// Number of CAS instructions found.
        cas_count: u32,
    },

    /// RED: data-dependent global memory access (cannot predict block-to-data mapping).
    DataDependentAccess,

    /// RED: indirect memory access through pointer loaded from global memory.
    IndirectMemoryAccess,

    /// RED: could not parse PTX or SASS-only module with no PTX available.
    UnanalyzablePtx {
        /// Reason parsing failed.
        reason: String,
    },

    /// RED: kernel uses dynamic parallelism (child kernel launches).
    DynamicParallelism,

    /// RED: kernel grid is too small to benefit from splitting.
    GridTooSmall {
        /// Total block count.
        block_count: u32,
    },
}

impl ClassificationReason {
    /// Returns the color implied by this reason.
    pub fn implied_color(&self) -> KernelColor {
        match self {
            Self::FullyIndependent | Self::SharedMemoryAtomicsOnly => KernelColor::Green,
            Self::GlobalReductionAtomic { .. }
            | Self::GlobalMinMaxAtomic { .. }
            | Self::HistogramPattern { .. } => KernelColor::Yellow,
            Self::CooperativeGridSync
            | Self::GlobalCasAtomic { .. }
            | Self::DataDependentAccess
            | Self::IndirectMemoryAccess
            | Self::UnanalyzablePtx { .. }
            | Self::DynamicParallelism
            | Self::GridTooSmall { .. } => KernelColor::Red,
        }
    }
}

/// How a pointer parameter is used in the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerRole {
    /// Only appears in ld.global instructions -- safe to replicate.
    ReadOnly,
    /// Only appears in st.global instructions -- each GPU writes its own region.
    WriteOnly,
    /// Appears in both ld.global and st.global -- needs coherency or careful partitioning.
    ReadWrite,
    /// Appears as target of atom.global -- needs atomic redirection.
    AtomicTarget,
    /// Cannot determine (indirect access, cast to different type, etc.).
    Unknown,
}

/// How blockIdx (ctaid) maps to memory addresses.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockIdxUsage {
    /// addr = ctaid * stride + base -- perfect for splitting.
    Linear {
        /// Stride expressions found in PTX.
        stride_expressions: Vec<String>,
    },
    /// addr = f(ctaid) where f is more complex but still deterministic.
    Deterministic,
    /// addr depends on data loaded from memory -- cannot predict.
    DataDependent,
    /// ctaid is not used for addressing (e.g., batch index only).
    NonAddressing,
    /// Could not determine usage pattern.
    Unknown,
}

/// Which dimensions of blockIdx (ctaid) are read by the kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DimensionsUsed {
    /// Whether %ctaid.x is read.
    pub x: bool,
    /// Whether %ctaid.y is read.
    pub y: bool,
    /// Whether %ctaid.z is read.
    pub z: bool,
}

impl DimensionsUsed {
    /// Returns the number of dimensions actively used.
    pub fn count(&self) -> u32 {
        self.x as u32 + self.y as u32 + self.z as u32
    }

    /// Returns true if any dimension is used.
    pub fn any(&self) -> bool {
        self.x || self.y || self.z
    }
}

/// Memory space for an atomic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    /// Global device memory.
    Global,
    /// Shared (block-local) memory.
    Shared,
}

/// Atomic operation types for merge step planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOpType {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Increment.
    Inc,
    /// Decrement.
    Dec,
}

/// PTX data types for atomic operations and merge steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDataType {
    /// Signed 32-bit integer.
    S32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    S64,
    /// Unsigned 64-bit integer.
    U64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

impl PtxDataType {
    /// Returns the size in bytes of this data type.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::S32 | Self::U32 | Self::F32 => 4,
            Self::S64 | Self::U64 | Self::F64 => 8,
        }
    }

    /// Parse a PTX data type suffix string (e.g., "f32", "u64").
    pub fn from_ptx_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "s32" => Some(Self::S32),
            "u32" => Some(Self::U32),
            "s64" => Some(Self::S64),
            "u64" => Some(Self::U64),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

/// Info about one atomic instruction found in PTX.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxAtomicInfo {
    /// PTX instruction text (e.g., "atom.global.add.f32").
    pub instruction: String,
    /// Memory space: global or shared.
    pub memory_space: MemorySpace,
    /// Atomic operation type.
    pub op: AtomicOpType,
    /// Data type and size.
    pub data_type: PtxDataType,
    /// Line number in PTX.
    pub line: u32,
    /// Whether the target address is derived from a parameter (vs computed).
    pub target_is_param_derived: bool,
}

/// Results of PTX static analysis for one kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxAnalysis {
    /// Number of parameters (total).
    pub param_count: u32,
    /// Number of pointer parameters (detected by .u64 + ld.global/st.global usage).
    pub pointer_param_count: u32,
    /// Pointer role classification (read-only, write-only, read-write).
    pub pointer_roles: Vec<PointerRole>,
    /// Global atomic operations found.
    pub global_atomics: Vec<PtxAtomicInfo>,
    /// Shared memory atomic operations found (safe, block-local).
    pub shared_atomics: Vec<PtxAtomicInfo>,
    /// Whether cooperative group sync is used (bar.sync with grid-wide scope).
    pub has_grid_sync: bool,
    /// Whether dynamic parallelism is used (device-side kernel launch).
    pub has_dynamic_parallelism: bool,
    /// How %ctaid.x/y/z is used: linear (addr = f(ctaid)), or complex/indirect.
    pub blockidx_usage: BlockIdxUsage,
    /// Number of %ctaid reads (all dimensions).
    pub ctaid_read_count: u32,
    /// Dimensions used: which of x, y, z are read.
    pub dimensions_used: DimensionsUsed,
    /// Estimated compute intensity (instruction count, rough).
    pub estimated_instruction_count: u32,
    /// Shared memory usage (static + dynamic) in bytes.
    pub shared_mem_bytes: u32,
    /// Register usage per thread (if parseable from .maxnreg or annotations).
    pub registers_per_thread: Option<u32>,
}

/// Complete classification result for one kernel.
#[derive(Debug)]
pub struct KernelClassification {
    /// Kernel function handle.
    pub function: CuFunction,
    /// Kernel entry name from PTX (e.g., "_Z9addKernelPfS_S_i").
    pub entry_name: String,
    /// Traffic light classification.
    pub color: KernelColor,
    /// Detailed reason for the classification.
    pub reason: ClassificationReason,
    /// PTX analysis results.
    pub analysis: PtxAnalysis,
    /// Whether the kernel has been successfully transformed (PTX rewritten).
    pub transformed: AtomicBool,
    /// Transformed PTX module (lazily created on first split attempt).
    pub transformed_module: OnceLock<TransformedModule>,
}

impl KernelClassification {
    /// Creates a new classification result.
    pub fn new(
        function: CuFunction,
        entry_name: String,
        color: KernelColor,
        reason: ClassificationReason,
        analysis: PtxAnalysis,
    ) -> Self {
        Self {
            function,
            entry_name,
            color,
            reason,
            analysis,
            transformed: AtomicBool::new(false),
            transformed_module: OnceLock::new(),
        }
    }

    /// Returns true if this kernel can be split (GREEN or YELLOW).
    pub fn is_splittable(&self) -> bool {
        matches!(self.color, KernelColor::Green | KernelColor::Yellow)
    }

    /// Returns true if this kernel requires a merge step after splitting.
    pub fn needs_merge(&self) -> bool {
        self.color == KernelColor::Yellow
    }

    /// Returns true if PTX transformation has been completed.
    pub fn is_transformed(&self) -> bool {
        self.transformed.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Classifier statistics
// ---------------------------------------------------------------------------

/// Statistics across all classified kernels.
#[derive(Debug)]
pub struct ClassifierStats {
    /// Total number of kernels classified.
    pub total_classified: AtomicU64,
    /// Number of GREEN kernels.
    pub green_count: AtomicU64,
    /// Number of YELLOW kernels.
    pub yellow_count: AtomicU64,
    /// Number of RED kernels.
    pub red_count: AtomicU64,
    /// Number of unanalyzable kernels (subset of RED).
    pub unanalyzable_count: AtomicU64,
    /// Green kernels' estimated compute time as fraction of total (fixed-point * 1000).
    pub green_compute_fraction: AtomicU64,
    /// Yellow kernels' estimated compute time as fraction of total (fixed-point * 1000).
    pub yellow_compute_fraction: AtomicU64,
}

impl ClassifierStats {
    /// Creates a new zeroed stats instance.
    pub fn new() -> Self {
        Self {
            total_classified: AtomicU64::new(0),
            green_count: AtomicU64::new(0),
            yellow_count: AtomicU64::new(0),
            red_count: AtomicU64::new(0),
            unanalyzable_count: AtomicU64::new(0),
            green_compute_fraction: AtomicU64::new(0),
            yellow_compute_fraction: AtomicU64::new(0),
        }
    }

    /// Records a classification result.
    pub fn record(&self, color: KernelColor, is_unanalyzable: bool) {
        self.total_classified.fetch_add(1, Ordering::Relaxed);
        match color {
            KernelColor::Green => {
                self.green_count.fetch_add(1, Ordering::Relaxed);
            }
            KernelColor::Yellow => {
                self.yellow_count.fetch_add(1, Ordering::Relaxed);
            }
            KernelColor::Red => {
                self.red_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        if is_unanalyzable {
            self.unanalyzable_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the total number of classified kernels.
    pub fn total(&self) -> u64 {
        self.total_classified.load(Ordering::Relaxed)
    }

    /// Returns the fraction of GREEN kernels (0.0 to 1.0).
    pub fn green_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        self.green_count.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Returns the fraction of splittable (GREEN + YELLOW) kernels.
    pub fn splittable_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let splittable = self.green_count.load(Ordering::Relaxed)
            + self.yellow_count.load(Ordering::Relaxed);
        splittable as f64 / total as f64
    }

    /// Returns the green compute fraction as a float (0.0 to 1.0).
    pub fn green_compute_frac(&self) -> f64 {
        self.green_compute_fraction.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Returns the yellow compute fraction as a float (0.0 to 1.0).
    pub fn yellow_compute_frac(&self) -> f64 {
        self.yellow_compute_fraction.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Sets the green compute fraction (value from 0.0 to 1.0, stored as fixed-point * 1000).
    pub fn set_green_compute_fraction(&self, frac: f64) {
        self.green_compute_fraction
            .store((frac * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Sets the yellow compute fraction (value from 0.0 to 1.0, stored as fixed-point * 1000).
    pub fn set_yellow_compute_fraction(&self, frac: f64) {
        self.yellow_compute_fraction
            .store((frac * 1000.0) as u64, Ordering::Relaxed);
    }
}

impl Default for ClassifierStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PtxParser -- stub for PTX analysis
// ---------------------------------------------------------------------------

/// PTX parser for static analysis of kernel entry points.
///
/// Analyzes PTX text to extract kernel metadata: parameter counts, atomic usage,
/// blockIdx access patterns, register usage, and shared memory requirements.
#[derive(Debug, Clone)]
pub struct PtxParser {
    /// Minimum grid size below which splitting is never worth it.
    pub min_grid_size_for_split: u32,
}

impl PtxParser {
    /// Creates a new PTX parser with default configuration.
    pub fn new() -> Self {
        Self {
            min_grid_size_for_split: 64,
        }
    }

    /// Analyzes PTX source text and returns classification for all kernel entries.
    ///
    /// This parses the PTX to detect:
    /// - Grid sync (cooperative groups)
    /// - Dynamic parallelism
    /// - Atomic operations (global vs shared, CAS vs reduction)
    /// - blockIdx usage patterns (linear, deterministic, data-dependent)
    /// - Pointer parameter roles (read-only, write-only, read-write, atomic target)
    ///
    /// TODO: Implement full PTX regex-based parser. Currently returns a stub analysis
    /// for testing the classification decision tree.
    pub fn analyze_ptx(&self, ptx_source: &str) -> Result<PtxAnalysis, String> {
        if ptx_source.is_empty() {
            return Err("Empty PTX source".to_string());
        }

        // Count parameters by looking for .param declarations (not ld.param / st.param)
        let param_count = ptx_source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with(".param") || trimmed.starts_with(",") && trimmed.contains(".param")
            })
            .count() as u32;
        let pointer_param_count = ptx_source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                (trimmed.starts_with(".param .u64") || trimmed.contains(".param .u64"))
                    && !trimmed.contains("ld.param")
                    && !trimmed.contains("st.param")
            })
            .count() as u32;

        // Detect grid sync
        let has_grid_sync = ptx_source.contains("bar.sync")
            && (ptx_source.contains("cooperative_groups") || ptx_source.contains("grid.sync"));

        // Detect dynamic parallelism
        let has_dynamic_parallelism =
            ptx_source.contains("cudaLaunchDevice") || ptx_source.contains(".launch");

        // Detect atomics
        let global_atomics = self.detect_atomics(ptx_source, MemorySpace::Global);
        let shared_atomics = self.detect_atomics(ptx_source, MemorySpace::Shared);

        // Detect blockIdx usage
        let has_ctaid_x = ptx_source.contains("%ctaid.x");
        let has_ctaid_y = ptx_source.contains("%ctaid.y");
        let has_ctaid_z = ptx_source.contains("%ctaid.z");
        let ctaid_read_count = ptx_source.matches("%ctaid").count() as u32;

        let dimensions_used = DimensionsUsed {
            x: has_ctaid_x,
            y: has_ctaid_y,
            z: has_ctaid_z,
        };

        // Determine blockIdx usage pattern
        let blockidx_usage = self.analyze_blockidx_usage(ptx_source, &dimensions_used);

        // Estimate instruction count (rough: count semicolons in non-comment lines)
        let estimated_instruction_count = ptx_source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with('.')
            })
            .count() as u32;

        // Detect shared memory
        let shared_mem_bytes = self.detect_shared_memory_size(ptx_source);

        // Detect register usage
        let registers_per_thread = self.detect_register_count(ptx_source);

        // Build pointer roles
        let pointer_roles = self.analyze_pointer_roles(ptx_source, pointer_param_count, &global_atomics);

        Ok(PtxAnalysis {
            param_count,
            pointer_param_count,
            pointer_roles,
            global_atomics,
            shared_atomics,
            has_grid_sync,
            has_dynamic_parallelism,
            blockidx_usage,
            ctaid_read_count,
            dimensions_used,
            estimated_instruction_count,
            shared_mem_bytes,
            registers_per_thread,
        })
    }

    /// Detect atomic instructions in PTX for a given memory space.
    fn detect_atomics(&self, ptx_source: &str, space: MemorySpace) -> Vec<PtxAtomicInfo> {
        let prefix = match space {
            MemorySpace::Global => "atom.global",
            MemorySpace::Shared => "atom.shared",
        };

        let mut atomics = Vec::new();

        for (line_num, line) in ptx_source.lines().enumerate() {
            let trimmed = line.trim();
            if !trimmed.contains(prefix) {
                continue;
            }

            // Parse atom.{space}.{op}.{type}
            if let Some(info) = self.parse_atomic_instruction(trimmed, space, line_num as u32 + 1) {
                atomics.push(info);
            }
        }

        atomics
    }

    /// Parse a single atomic instruction from PTX text.
    fn parse_atomic_instruction(
        &self,
        instruction: &str,
        space: MemorySpace,
        line: u32,
    ) -> Option<PtxAtomicInfo> {
        let space_str = match space {
            MemorySpace::Global => "global",
            MemorySpace::Shared => "shared",
        };

        // Extract the atom instruction: atom.{space}.{op}.{type}
        let atom_prefix = format!("atom.{}.", space_str);
        let start = instruction.find(&atom_prefix)?;
        let rest = &instruction[start + atom_prefix.len()..];

        // Split on dots and whitespace to get op and type
        let parts: Vec<&str> = rest.split(|c: char| c == '.' || c.is_whitespace()).collect();
        if parts.len() < 2 {
            return None;
        }

        let op = match parts[0] {
            "add" => AtomicOpType::Add,
            "sub" => AtomicOpType::Sub,
            "min" => AtomicOpType::Min,
            "max" => AtomicOpType::Max,
            "and" => AtomicOpType::And,
            "or" => AtomicOpType::Or,
            "xor" => AtomicOpType::Xor,
            "inc" => AtomicOpType::Inc,
            "dec" => AtomicOpType::Dec,
            "cas" => return None, // CAS handled separately
            _ => return None,
        };

        let data_type = PtxDataType::from_ptx_suffix(parts[1])?;

        Some(PtxAtomicInfo {
            instruction: instruction.to_string(),
            memory_space: space,
            op,
            data_type,
            line,
            target_is_param_derived: instruction.contains("param"),
        })
    }

    /// Analyze how blockIdx (ctaid) is used for memory addressing.
    fn analyze_blockidx_usage(&self, ptx_source: &str, dims: &DimensionsUsed) -> BlockIdxUsage {
        if !dims.any() {
            return BlockIdxUsage::NonAddressing;
        }

        // Check for data-dependent patterns: loading from memory then using as index
        let has_indirect = ptx_source.contains("ld.global")
            && (ptx_source.contains("cvta.to.global") || ptx_source.contains("ld.global.nc"));

        // Simple heuristic: if ctaid feeds into mad/mul with constants, it's linear
        let has_linear_pattern = ptx_source.contains("mad.lo") || ptx_source.contains("mul.lo");

        // Data-dependent access ALWAYS takes priority: a kernel that loads
        // from global memory and uses the result as an index is unsafe to split,
        // even if it also contains mad.lo for other index arithmetic.
        if has_indirect {
            BlockIdxUsage::DataDependent
        } else if has_linear_pattern {
            BlockIdxUsage::Linear {
                stride_expressions: Vec::new(),
            }
        } else if dims.any() {
            BlockIdxUsage::Deterministic
        } else {
            BlockIdxUsage::Unknown
        }
    }

    /// Detect shared memory size from PTX declarations.
    fn detect_shared_memory_size(&self, ptx_source: &str) -> u32 {
        let mut total = 0u32;
        for line in ptx_source.lines() {
            let trimmed = line.trim();
            if trimmed.contains(".shared") && trimmed.contains(".align") {
                // Try to extract array size: .shared .align N .bN name[SIZE]
                if let Some(bracket_start) = trimmed.find('[') {
                    if let Some(bracket_end) = trimmed.find(']') {
                        if let Ok(size) = trimmed[bracket_start + 1..bracket_end].parse::<u32>() {
                            total += size;
                        }
                    }
                }
            }
        }
        total
    }

    /// Detect register count from PTX .maxnreg directive.
    fn detect_register_count(&self, ptx_source: &str) -> Option<u32> {
        for line in ptx_source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(".maxnreg") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].parse().ok();
                }
            }
        }
        None
    }

    /// Analyze pointer parameter roles based on their usage in load/store/atomic operations.
    fn analyze_pointer_roles(
        &self,
        ptx_source: &str,
        pointer_count: u32,
        global_atomics: &[PtxAtomicInfo],
    ) -> Vec<PointerRole> {
        let has_stores = ptx_source.contains("st.global");
        let has_loads = ptx_source.contains("ld.global");
        let has_atomics = !global_atomics.is_empty();

        // Simple heuristic: assign roles based on what operations are present
        // TODO: Implement per-parameter data flow tracking for accurate role assignment
        let mut roles = Vec::with_capacity(pointer_count as usize);
        for i in 0..pointer_count {
            let role = if has_atomics && i == pointer_count - 1 {
                // Last pointer param is often the atomic target (output/accumulator)
                PointerRole::AtomicTarget
            } else if has_stores && has_loads {
                if i == 0 {
                    PointerRole::ReadOnly
                } else {
                    PointerRole::ReadWrite
                }
            } else if has_loads && !has_stores {
                PointerRole::ReadOnly
            } else if has_stores && !has_loads {
                PointerRole::WriteOnly
            } else {
                PointerRole::Unknown
            };
            roles.push(role);
        }
        roles
    }

    /// Detect CAS (compare-and-swap) atomic instructions in PTX.
    pub fn detect_cas_atomics(&self, ptx_source: &str) -> Vec<(u32, bool)> {
        let mut cas_info = Vec::new();
        for (line_num, line) in ptx_source.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains("atom.global.cas") {
                // Check if this CAS is in a retry loop (branch back pattern)
                let is_retry_loop = self.is_cas_in_retry_loop(ptx_source, line_num);
                cas_info.push((line_num as u32 + 1, is_retry_loop));
            }
        }
        cas_info
    }

    /// Check if a CAS instruction at the given line is part of a retry loop.
    ///
    /// A retry loop is identified by a conditional branch AFTER the CAS whose
    /// target label appears BEFORE the CAS line (backward branch). Forward
    /// branches (e.g., `@%p bra $done`) are exit conditions, not retry loops.
    fn is_cas_in_retry_loop(&self, ptx_source: &str, cas_line: usize) -> bool {
        let lines: Vec<&str> = ptx_source.lines().collect();
        let search_end = (cas_line + 10).min(lines.len());
        for i in cas_line + 1..search_end {
            let trimmed = lines[i].trim();
            if trimmed.starts_with("@") && trimmed.contains("bra") {
                // Extract the branch target label (last token before ';')
                if let Some(target) = trimmed
                    .split_whitespace()
                    .last()
                    .map(|s| s.trim_end_matches(';'))
                {
                    // Check if target label appears BEFORE the CAS (backward branch)
                    let is_backward = lines[..cas_line].iter().any(|line| {
                        let lt = line.trim();
                        lt.starts_with(target) && lt.ends_with(':')
                    });
                    if is_backward {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for PtxParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// KernelClassifier
// ---------------------------------------------------------------------------

/// Classifies CUDA kernels by analyzing PTX to determine splittability.
///
/// Runs at cuModuleLoadData / cuModuleLoadFatBinary interception time.
/// Results are cached per CUfunction handle for fast lookup at launch time.
pub struct KernelClassifier {
    /// Cache of classification results: CUfunction -> classification.
    cache: dashmap::DashMap<CuFunction, Arc<KernelClassification>>,

    /// Statistics for reporting classification distribution.
    pub stats: ClassifierStats,

    /// PTX parser instance (reusable, not per-kernel).
    parser: PtxParser,

    /// Minimum grid size (total blocks) below which splitting is never worth it.
    pub min_grid_size_for_split: u32,

    /// Minimum estimated kernel duration (us) below which splitting overhead dominates.
    pub min_kernel_duration_us: f64,
}

impl KernelClassifier {
    /// Creates a new kernel classifier with default thresholds.
    pub fn new() -> Self {
        Self {
            cache: dashmap::DashMap::new(),
            stats: ClassifierStats::new(),
            parser: PtxParser::new(),
            min_grid_size_for_split: 64,
            min_kernel_duration_us: 20.0,
        }
    }

    /// Creates a classifier with custom thresholds.
    pub fn with_thresholds(min_grid_size: u32, min_duration_us: f64) -> Self {
        Self {
            cache: dashmap::DashMap::new(),
            stats: ClassifierStats::new(),
            parser: PtxParser::new(),
            min_grid_size_for_split: min_grid_size,
            min_kernel_duration_us: min_duration_us,
        }
    }

    /// Classifies a kernel from its PTX source, or returns the cached result.
    pub fn get_or_classify(
        &self,
        function: CuFunction,
        entry_name: &str,
        ptx_source: &str,
    ) -> Arc<KernelClassification> {
        if let Some(cached) = self.cache.get(&function) {
            return cached.clone();
        }

        let classification = self.classify_kernel(function, entry_name, ptx_source);
        let arc = Arc::new(classification);
        self.cache.insert(function, arc.clone());
        arc
    }

    /// Performs classification on a kernel's PTX source.
    pub fn classify_kernel(
        &self,
        function: CuFunction,
        entry_name: &str,
        ptx_source: &str,
    ) -> KernelClassification {
        // Step 1: Can we analyze it?
        let analysis = match self.parser.analyze_ptx(ptx_source) {
            Ok(a) => a,
            Err(reason) => {
                let reason = ClassificationReason::UnanalyzablePtx { reason };
                self.stats.record(KernelColor::Red, true);
                return KernelClassification::new(
                    function,
                    entry_name.to_string(),
                    KernelColor::Red,
                    reason,
                    PtxAnalysis {
                        param_count: 0,
                        pointer_param_count: 0,
                        pointer_roles: Vec::new(),
                        global_atomics: Vec::new(),
                        shared_atomics: Vec::new(),
                        has_grid_sync: false,
                        has_dynamic_parallelism: false,
                        blockidx_usage: BlockIdxUsage::Unknown,
                        ctaid_read_count: 0,
                        dimensions_used: DimensionsUsed::default(),
                        estimated_instruction_count: 0,
                        shared_mem_bytes: 0,
                        registers_per_thread: None,
                    },
                );
            }
        };

        // Step 2: Hard blockers
        if analysis.has_grid_sync {
            self.stats.record(KernelColor::Red, false);
            return KernelClassification::new(
                function,
                entry_name.to_string(),
                KernelColor::Red,
                ClassificationReason::CooperativeGridSync,
                analysis,
            );
        }

        if analysis.has_dynamic_parallelism {
            self.stats.record(KernelColor::Red, false);
            return KernelClassification::new(
                function,
                entry_name.to_string(),
                KernelColor::Red,
                ClassificationReason::DynamicParallelism,
                analysis,
            );
        }

        // Step 3: Atomic analysis
        let cas_info = self.parser.detect_cas_atomics(ptx_source);
        if !cas_info.is_empty() {
            // Any CAS in a retry loop is a hard RED
            let has_retry_cas = cas_info.iter().any(|(_, is_retry)| *is_retry);
            if has_retry_cas {
                self.stats.record(KernelColor::Red, false);
                return KernelClassification::new(
                    function,
                    entry_name.to_string(),
                    KernelColor::Red,
                    ClassificationReason::GlobalCasAtomic {
                        cas_count: cas_info.len() as u32,
                    },
                    analysis,
                );
            }
            // Non-retry CAS is still RED (conservative)
            self.stats.record(KernelColor::Red, false);
            return KernelClassification::new(
                function,
                entry_name.to_string(),
                KernelColor::Red,
                ClassificationReason::GlobalCasAtomic {
                    cas_count: cas_info.len() as u32,
                },
                analysis,
            );
        }

        // Check for global reduction atomics
        if !analysis.global_atomics.is_empty() {
            let add_sub_count = analysis
                .global_atomics
                .iter()
                .filter(|a| matches!(a.op, AtomicOpType::Add | AtomicOpType::Sub))
                .count() as u32;

            let min_max_count = analysis
                .global_atomics
                .iter()
                .filter(|a| matches!(a.op, AtomicOpType::Min | AtomicOpType::Max))
                .count() as u32;

            let bitwise_count = analysis
                .global_atomics
                .iter()
                .filter(|a| {
                    matches!(
                        a.op,
                        AtomicOpType::And | AtomicOpType::Or | AtomicOpType::Xor
                    )
                })
                .count() as u32;

            let total_atomics = analysis.global_atomics.len() as u32;
            // Estimate distinct targets from param-derived count
            let param_derived = analysis
                .global_atomics
                .iter()
                .filter(|a| a.target_is_param_derived)
                .count() as u32;
            let target_count = param_derived.max(1);

            let (color, reason) = if add_sub_count > 0 {
                if target_count <= 16 {
                    (
                        KernelColor::Yellow,
                        ClassificationReason::GlobalReductionAtomic {
                            atomic_count: add_sub_count,
                            target_count,
                        },
                    )
                } else {
                    (
                        KernelColor::Yellow,
                        ClassificationReason::HistogramPattern {
                            estimated_bin_count: target_count,
                        },
                    )
                }
            } else if min_max_count > 0 {
                (
                    KernelColor::Yellow,
                    ClassificationReason::GlobalMinMaxAtomic {
                        atomic_count: min_max_count,
                        target_count,
                    },
                )
            } else if bitwise_count > 0 {
                (
                    KernelColor::Yellow,
                    ClassificationReason::GlobalReductionAtomic {
                        atomic_count: bitwise_count,
                        target_count,
                    },
                )
            } else {
                // Unknown atomic types -- conservative RED
                (
                    KernelColor::Red,
                    ClassificationReason::UnanalyzablePtx {
                        reason: format!("Unknown global atomic type ({} atomics)", total_atomics),
                    },
                )
            };

            self.stats
                .record(color, matches!(reason, ClassificationReason::UnanalyzablePtx { .. }));
            return KernelClassification::new(
                function,
                entry_name.to_string(),
                color,
                reason,
                analysis,
            );
        }

        // Step 4: Memory access pattern
        match &analysis.blockidx_usage {
            BlockIdxUsage::DataDependent => {
                self.stats.record(KernelColor::Red, false);
                return KernelClassification::new(
                    function,
                    entry_name.to_string(),
                    KernelColor::Red,
                    ClassificationReason::DataDependentAccess,
                    analysis,
                );
            }
            BlockIdxUsage::Unknown if analysis.dimensions_used.any() => {
                self.stats.record(KernelColor::Red, false);
                return KernelClassification::new(
                    function,
                    entry_name.to_string(),
                    KernelColor::Red,
                    ClassificationReason::IndirectMemoryAccess,
                    analysis,
                );
            }
            _ => {}
        }

        // Step 5: If only shared atomics, still GREEN
        let reason = if !analysis.shared_atomics.is_empty() {
            ClassificationReason::SharedMemoryAtomicsOnly
        } else {
            ClassificationReason::FullyIndependent
        };

        self.stats.record(KernelColor::Green, false);
        KernelClassification::new(
            function,
            entry_name.to_string(),
            KernelColor::Green,
            reason,
            analysis,
        )
    }

    /// Returns the number of cached classifications.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clears the classification cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Retrieves a cached classification if available.
    pub fn get_cached(&self, function: CuFunction) -> Option<Arc<KernelClassification>> {
        self.cache.get(&function).map(|v| v.clone())
    }
}

impl Default for KernelClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PTX Transformer types
// ---------------------------------------------------------------------------

/// Injected offset parameter names for blockIdx adjustment.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OffsetParams {
    /// Parameter name for X offset (e.g., "__outerlink_blkoff_x").
    pub x: Option<String>,
    /// Parameter name for Y offset.
    pub y: Option<String>,
    /// Parameter name for Z offset.
    pub z: Option<String>,
}

impl OffsetParams {
    /// Returns the number of offset parameters.
    pub fn count(&self) -> u32 {
        self.x.is_some() as u32 + self.y.is_some() as u32 + self.z.is_some() as u32
    }
}

/// An atomic instruction that was redirected to a per-GPU local copy.
#[derive(Debug, Clone, PartialEq)]
pub struct RedirectedAtomic {
    /// Original PTX line number.
    pub original_line: u32,
    /// Original instruction text.
    pub original_instruction: String,
    /// New parameter name for the local atomic target.
    pub local_target_param: String,
    /// Atomic operation type (needed for merge step).
    pub op_type: AtomicOpType,
    /// Data type (needed for merge step).
    pub data_type: PtxDataType,
}

/// Result of transforming a PTX module for kernel splitting.
#[derive(Debug)]
pub struct TransformedModule {
    /// Modified PTX source text.
    pub ptx_source: String,
    /// Names of injected offset parameters.
    pub offset_params: OffsetParams,
    /// For YELLOW kernels: redirected atomic targets.
    pub redirected_atomics: Vec<RedirectedAtomic>,
    /// CUmodule handle after loading transformed PTX.
    /// TODO: Set by CUDA driver after cuModuleLoadDataEx.
    pub loaded_module: OnceLock<u64>,
    /// CUfunction handle for the transformed kernel.
    /// TODO: Set by CUDA driver after cuModuleGetFunction.
    pub loaded_function: OnceLock<CuFunction>,
}

/// Transforms PTX source to inject blockIdx offset parameters.
///
/// Applied once at module load time, results cached per kernel.
/// The transformation adds extra parameters for block offset values
/// and modifies ctaid reads to add those offsets.
#[derive(Debug)]
pub struct PtxTransformer {
    /// Validation: compare transformed kernel output against original (debug mode).
    pub validation_enabled: bool,
}

impl PtxTransformer {
    /// Creates a new PTX transformer.
    pub fn new(validation_enabled: bool) -> Self {
        Self { validation_enabled }
    }

    /// Transforms PTX source by injecting blockIdx offset parameters.
    ///
    /// For each dimension used by the kernel:
    /// 1. Adds a `.param .u32 __outerlink_blkoff_{x,y,z}` parameter
    /// 2. Inserts a load instruction for the offset
    /// 3. Adds the offset to every `mov.u32 %rN, %ctaid.{x,y,z}` instruction
    ///
    /// For YELLOW kernels, also redirects global atomic targets to local copies.
    pub fn transform(
        &self,
        ptx_source: &str,
        analysis: &PtxAnalysis,
    ) -> Result<TransformedModule, String> {
        if ptx_source.is_empty() {
            return Err("Empty PTX source".to_string());
        }

        let dims = &analysis.dimensions_used;
        if !dims.any() {
            // No blockIdx usage -- no transformation needed, but still valid
            return Ok(TransformedModule {
                ptx_source: ptx_source.to_string(),
                offset_params: OffsetParams::default(),
                redirected_atomics: Vec::new(),
                loaded_module: OnceLock::new(),
                loaded_function: OnceLock::new(),
            });
        }

        let mut output = String::with_capacity(ptx_source.len() + 1024);
        let mut offset_params = OffsetParams::default();

        // Find the closing parenthesis of the parameter list
        let param_end = match ptx_source.find(')') {
            Some(pos) => pos,
            None => return Err("Could not find parameter list end in PTX".to_string()),
        };

        // Inject offset parameters before closing paren
        let mut param_injection = String::new();
        if dims.x {
            param_injection.push_str(",\n    .param .u32 __outerlink_blkoff_x");
            offset_params.x = Some("__outerlink_blkoff_x".to_string());
        }
        if dims.y {
            param_injection.push_str(",\n    .param .u32 __outerlink_blkoff_y");
            offset_params.y = Some("__outerlink_blkoff_y".to_string());
        }
        if dims.z {
            param_injection.push_str(",\n    .param .u32 __outerlink_blkoff_z");
            offset_params.z = Some("__outerlink_blkoff_z".to_string());
        }

        output.push_str(&ptx_source[..param_end]);
        output.push_str(&param_injection);
        output.push_str(&ptx_source[param_end..]);

        // Inject offset load and add instructions after the opening brace
        let body_start = match output.find('{') {
            Some(pos) => pos + 1,
            None => return Err("Could not find function body in PTX".to_string()),
        };

        let mut injections = String::new();
        let mut reg_counter = 100; // Use high register numbers to avoid conflicts

        for (dim, used) in [("x", dims.x), ("y", dims.y), ("z", dims.z)] {
            if !used {
                continue;
            }
            let offset_reg = format!("%__ol_off_{}", dim);
            let param_name = format!("__outerlink_blkoff_{}", dim);

            // Declare the register, load the offset parameter
            injections.push_str(&format!(
                "\n    .reg .u32 {};\n    ld.param.u32 {}, [{}];",
                offset_reg, offset_reg, param_name
            ));

            reg_counter += 1;
        }

        // Insert after body opening brace
        output.insert_str(body_start, &injections);

        // Now process ctaid reads: find "mov.u32 %rN, %ctaid.{dim}" and add offset
        // We need to work on the final string to insert add instructions
        let mut final_output = String::with_capacity(output.len() + 512);
        for line in output.lines() {
            final_output.push_str(line);
            final_output.push('\n');

            let trimmed = line.trim();
            for dim in ["x", "y", "z"] {
                let used = match dim {
                    "x" => dims.x,
                    "y" => dims.y,
                    "z" => dims.z,
                    _ => false,
                };
                if !used {
                    continue;
                }

                let ctaid_pattern = format!("%ctaid.{}", dim);
                if trimmed.contains("mov.u32") && trimmed.contains(&ctaid_pattern) {
                    // Extract the destination register
                    if let Some(dest_reg) = self.extract_mov_dest(trimmed) {
                        let offset_reg = format!("%__ol_off_{}", dim);
                        final_output.push_str(&format!(
                            "    add.u32 {}, {}, {};\n",
                            dest_reg, dest_reg, offset_reg
                        ));
                    }
                }
            }
        }

        // Handle YELLOW kernel atomic redirection
        let redirected_atomics = if !analysis.global_atomics.is_empty() {
            self.redirect_atomics(&mut final_output, &analysis.global_atomics)
        } else {
            Vec::new()
        };

        Ok(TransformedModule {
            ptx_source: final_output,
            offset_params,
            redirected_atomics,
            loaded_module: OnceLock::new(),
            loaded_function: OnceLock::new(),
        })
    }

    /// Extract the destination register from a mov instruction.
    fn extract_mov_dest(&self, instruction: &str) -> Option<String> {
        // Format: "mov.u32 %rN, %ctaid.x"
        let parts: Vec<&str> = instruction.split_whitespace().collect();
        if parts.len() >= 2 {
            let reg = parts[1].trim_end_matches(',');
            if reg.starts_with('%') {
                return Some(reg.to_string());
            }
        }
        None
    }

    /// Redirect global atomic targets to per-GPU local copies.
    ///
    /// For each global atomic instruction, adds a new parameter for the local copy
    /// address and rewrites the instruction to use it.
    fn redirect_atomics(
        &self,
        _ptx: &mut String,
        atomics: &[PtxAtomicInfo],
    ) -> Vec<RedirectedAtomic> {
        // TODO: Implement full atomic redirection in PTX text.
        // For now, record what needs to be redirected for the merge plan.
        atomics
            .iter()
            .enumerate()
            .map(|(i, atomic)| RedirectedAtomic {
                original_line: atomic.line,
                original_instruction: atomic.instruction.clone(),
                local_target_param: format!("__outerlink_local_atomic_{}", i),
                op_type: atomic.op,
                data_type: atomic.data_type,
            })
            .collect()
    }
}

impl Default for PtxTransformer {
    fn default() -> Self {
        Self::new(false)
    }
}

// ---------------------------------------------------------------------------
// Grid dimensions and split planning
// ---------------------------------------------------------------------------

/// 3D grid or block dimensions (mirrors CUDA dim3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim3 {
    /// X dimension.
    pub x: u32,
    /// Y dimension.
    pub y: u32,
    /// Z dimension.
    pub z: u32,
}

impl Dim3 {
    /// Creates a new Dim3.
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Creates a 1D dimension.
    pub fn one_d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Returns the total number of elements (x * y * z).
    pub fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }

    /// Returns the largest dimension value.
    pub fn largest_dim(&self) -> u32 {
        self.x.max(self.y).max(self.z)
    }

    /// Returns which dimension is largest.
    pub fn largest_dimension(&self) -> SplitDimension {
        if self.x >= self.y && self.x >= self.z {
            SplitDimension::X
        } else if self.y >= self.z {
            SplitDimension::Y
        } else {
            SplitDimension::Z
        }
    }

    /// Returns the value for a specific dimension.
    pub fn get(&self, dim: SplitDimension) -> u32 {
        match dim {
            SplitDimension::X => self.x,
            SplitDimension::Y => self.y,
            SplitDimension::Z => self.z,
        }
    }

    /// Returns a new Dim3 with one dimension replaced.
    pub fn with_dim(&self, dim: SplitDimension, value: u32) -> Self {
        let mut result = *self;
        match dim {
            SplitDimension::X => result.x = value,
            SplitDimension::Y => result.y = value,
            SplitDimension::Z => result.z = value,
        }
        result
    }
}

/// Which dimension to split the grid along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplitDimension {
    /// Split gridDim.x (most common -- 1D grids, batch dimension).
    X,
    /// Split gridDim.y (2D grids -- row dimension).
    Y,
    /// Split gridDim.z (3D grids -- depth/batch dimension).
    Z,
}

/// A kernel argument value for launch parameter preparation.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelArg {
    /// Pointer argument (device memory address + estimated size).
    Pointer {
        /// Device virtual address.
        addr: VirtualAddr,
        /// Estimated size of the pointed-to allocation in bytes.
        size: u64,
    },
    /// Scalar integer argument.
    Int(i64),
    /// Scalar float argument.
    Float(f64),
    /// Raw bytes (unknown type).
    Raw(Vec<u8>),
    /// Block offset parameter injected by the transformer.
    BlockOffset(u32),
}

/// Original CUDA kernel launch parameters.
#[derive(Debug, Clone)]
pub struct LaunchParams {
    /// CUDA function handle.
    pub function: CuFunction,
    /// Grid dimensions (number of blocks).
    pub grid_dim: Dim3,
    /// Block dimensions (threads per block).
    pub block_dim: Dim3,
    /// Dynamic shared memory size in bytes.
    pub shared_mem_bytes: u32,
    /// CUDA stream handle (0 = default stream).
    pub stream: u64,
    /// Kernel arguments.
    pub args: Vec<KernelArg>,
}

impl LaunchParams {
    /// Returns the total number of thread blocks.
    pub fn total_blocks(&self) -> u64 {
        self.grid_dim.total()
    }

    /// Returns the total number of threads.
    pub fn total_threads(&self) -> u64 {
        self.grid_dim.total() * self.block_dim.total()
    }
}

// ---------------------------------------------------------------------------
// GPU target and block assignment
// ---------------------------------------------------------------------------

/// Target GPU with performance info from R23 GpuProfile.
#[derive(Debug, Clone)]
pub struct GpuTarget {
    /// Device ID in OuterLink's virtual device space.
    pub device_id: DeviceId,
    /// Node ID (which PC).
    pub node_id: NodeId,
    /// Calibrated TFLOPS score from R23 GpuProfile.
    pub tflops_score: f32,
    /// Number of Streaming Multiprocessors.
    pub sm_count: u32,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
}

impl GpuTarget {
    /// Creates a new GPU target.
    pub fn new(
        device_id: DeviceId,
        node_id: NodeId,
        tflops_score: f32,
        sm_count: u32,
        compute_capability: (u32, u32),
    ) -> Self {
        Self {
            device_id,
            node_id,
            tflops_score,
            sm_count,
            compute_capability,
        }
    }
}

/// One GPU's block assignment within a split kernel launch.
#[derive(Debug, Clone)]
pub struct GpuBlockAssignment {
    /// Target GPU.
    pub gpu: GpuTarget,
    /// Grid dimensions for this GPU's launch.
    pub grid_dim: Dim3,
    /// Block offset for each dimension (injected into PTX as __blkoff_x/y/z).
    pub block_offset: Dim3,
    /// Number of blocks assigned.
    pub block_count: u32,
    /// Fraction of total blocks (for weighted heterogeneous splits).
    pub fraction: f32,
    /// Remapped kernel arguments (pointer args adjusted for this GPU).
    pub remapped_args: Vec<KernelArg>,
}

/// How blocks are divided across GPUs.
#[derive(Debug, Clone)]
pub struct BlockPartition {
    /// Per-GPU assignment.
    pub assignments: Vec<GpuBlockAssignment>,
    /// Original grid dimensions.
    pub original_grid: Dim3,
    /// Split dimension (which axis we partition along).
    pub split_dimension: SplitDimension,
}

impl BlockPartition {
    /// Returns the total number of blocks assigned across all GPUs.
    pub fn total_assigned_blocks(&self) -> u64 {
        self.assignments.iter().map(|a| a.block_count as u64).sum()
    }

    /// Validates that the partition covers all blocks in the original grid.
    pub fn is_valid(&self) -> bool {
        let original_dim = self.original_grid.get(self.split_dimension);
        let assigned: u32 = self.assignments.iter().map(|a| a.grid_dim.get(self.split_dimension)).sum();
        assigned == original_dim
    }
}

// ---------------------------------------------------------------------------
// Split decision
// ---------------------------------------------------------------------------

/// Reason for a split/no-split decision.
#[derive(Debug, Clone, PartialEq)]
pub enum SplitDecisionReason {
    /// Splitting: kernel is GREEN/YELLOW and meets size/duration thresholds.
    Split {
        /// Expected speedup factor (e.g., 1.7 = 70% faster).
        expected_speedup: f32,
    },
    /// Not splitting: kernel is RED.
    RedClassification,
    /// Not splitting: grid too small (< min_grid_size_for_split).
    GridTooSmall {
        /// Total block count.
        blocks: u32,
    },
    /// Not splitting: estimated duration below threshold.
    DurationTooShort {
        /// Estimated duration in microseconds.
        estimated_us: f64,
    },
    /// Not splitting: only one GPU available.
    SingleGpu,
    /// Not splitting: data not distributed (would need full replication).
    DataNotDistributed {
        /// Estimated replication cost in microseconds.
        replication_cost_us: f64,
    },
    /// Not splitting: R13 graph context says this kernel is not the bottleneck.
    NotBottleneck {
        /// This kernel's fraction of total compute time.
        compute_fraction: f32,
    },
}

/// Decision made at cuLaunchKernel interception time.
#[derive(Debug, Clone)]
pub struct SplitDecision {
    /// Original kernel launch parameters.
    pub original: LaunchParams,
    /// Classification of this kernel.
    pub classification: Arc<KernelClassification>,
    /// Whether to actually split this launch.
    pub should_split: bool,
    /// Reason for decision (for logging/diagnostics).
    pub decision_reason: SplitDecisionReason,
    /// If splitting: the partition plan.
    pub partition: Option<BlockPartition>,
    /// If splitting: which GPUs participate.
    pub target_gpus: Vec<GpuTarget>,
    /// If YELLOW: merge plan for combining partial results.
    pub merge_plan: Option<MergePlan>,
}

// ---------------------------------------------------------------------------
// Merge plan for YELLOW kernels
// ---------------------------------------------------------------------------

/// How partial results are combined after a split YELLOW kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombineStrategy {
    /// Sum all partial results (atomicAdd).
    Sum,
    /// Take minimum across all partials (atomicMin).
    Min,
    /// Take maximum across all partials (atomicMax).
    Max,
    /// Bitwise AND across all partials.
    BitwiseAnd,
    /// Bitwise OR across all partials.
    BitwiseOr,
    /// Bitwise XOR across all partials.
    BitwiseXor,
}

impl CombineStrategy {
    /// Derives the combine strategy from an atomic operation type.
    pub fn from_atomic_op(op: AtomicOpType) -> Self {
        match op {
            AtomicOpType::Add | AtomicOpType::Sub | AtomicOpType::Inc | AtomicOpType::Dec => {
                Self::Sum
            }
            AtomicOpType::Min => Self::Min,
            AtomicOpType::Max => Self::Max,
            AtomicOpType::And => Self::BitwiseAnd,
            AtomicOpType::Or => Self::BitwiseOr,
            AtomicOpType::Xor => Self::BitwiseXor,
        }
    }

    /// Combines two u64 values according to this strategy.
    pub fn combine_u64(&self, a: u64, b: u64) -> u64 {
        match self {
            Self::Sum => a.wrapping_add(b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::BitwiseAnd => a & b,
            Self::BitwiseOr => a | b,
            Self::BitwiseXor => a ^ b,
        }
    }

    /// Combines two f64 values according to this strategy.
    pub fn combine_f64(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Sum => a + b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            // Bitwise ops don't apply to floats; treat as sum
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => a + b,
        }
    }

    /// Returns the identity element for this combine strategy.
    pub fn identity_u64(&self) -> u64 {
        match self {
            Self::Sum => 0,
            Self::Min => u64::MAX,
            Self::Max => 0,
            Self::BitwiseAnd => u64::MAX,
            Self::BitwiseOr => 0,
            Self::BitwiseXor => 0,
        }
    }
}

/// One merge operation for one redirected atomic target.
#[derive(Debug, Clone)]
pub struct MergeOp {
    /// Original target address in the application's memory.
    pub original_target: VirtualAddr,
    /// Per-GPU local copy addresses.
    pub local_copies: Vec<(DeviceId, VirtualAddr)>,
    /// How to combine: depends on atomic type.
    pub combine: CombineStrategy,
    /// Data type for the merge.
    pub data_type: PtxDataType,
    /// Number of elements (1 for scalar, N for array/histogram).
    pub element_count: u32,
}

/// Merge plan for combining partial results from YELLOW kernels.
#[derive(Debug, Clone)]
pub struct MergePlan {
    /// Per-atomic-target merge operations.
    pub merge_ops: Vec<MergeOp>,
    /// Total estimated merge time in microseconds.
    pub estimated_merge_time_us: f64,
}

impl MergePlan {
    /// Returns true if all merge operations are scalar (1 element).
    pub fn is_all_scalar(&self) -> bool {
        self.merge_ops.iter().all(|op| op.element_count <= 1)
    }

    /// Returns the total number of elements to merge across all operations.
    pub fn total_elements(&self) -> u64 {
        self.merge_ops.iter().map(|op| op.element_count as u64).sum()
    }
}

// ---------------------------------------------------------------------------
// Merge executor
// ---------------------------------------------------------------------------

/// Threshold for switching from host-side to GPU kernel merge.
pub const LARGE_MERGE_THRESHOLD: u32 = 1024;

/// Estimated transport bandwidth in bytes per microsecond (for merge cost estimation).
/// Conservative estimate for PCIe Gen3 x16.
pub const TRANSPORT_BANDWIDTH_BYTES_PER_US: f64 = 12_500.0;

/// Executes merge operations after split kernel completion.
///
/// For scalar merges (1-16 elements), reads partial results from each GPU to the
/// host, combines them, and writes back. For large array merges (>1024 elements),
/// copies partials to the target GPU and launches a merge kernel.
#[derive(Debug)]
pub struct MergeExecutor {
    /// Pre-allocated merge buffers per GPU.
    merge_buffers: HashMap<DeviceId, MergeBuffer>,
    /// Threshold for switching to GPU kernel merge.
    pub large_merge_threshold: u32,
}

/// Pre-allocated buffer for merge operations on a specific GPU.
#[derive(Debug, Clone)]
pub struct MergeBuffer {
    /// Device this buffer is on.
    pub device_id: DeviceId,
    /// Base address of the buffer in device memory.
    pub base_addr: VirtualAddr,
    /// Total size in bytes.
    pub size: usize,
    /// Current offset (bytes used so far).
    pub offset: usize,
}

impl MergeBuffer {
    /// Creates a new merge buffer descriptor.
    pub fn new(device_id: DeviceId, base_addr: VirtualAddr, size: usize) -> Self {
        Self {
            device_id,
            base_addr,
            size,
            offset: 0,
        }
    }

    /// Allocates space from this buffer. Returns the address and advances the offset.
    pub fn allocate(&mut self, bytes: usize) -> Option<VirtualAddr> {
        if self.offset + bytes > self.size {
            return None;
        }
        let addr = self.base_addr + self.offset as u64;
        self.offset += bytes;
        Some(addr)
    }

    /// Resets the buffer offset to 0 for reuse.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Returns the remaining capacity in bytes.
    pub fn remaining(&self) -> usize {
        self.size - self.offset
    }
}

impl MergeExecutor {
    /// Creates a new merge executor.
    pub fn new() -> Self {
        Self {
            merge_buffers: HashMap::new(),
            large_merge_threshold: LARGE_MERGE_THRESHOLD,
        }
    }

    /// Creates a merge executor with a custom large merge threshold.
    pub fn with_threshold(threshold: u32) -> Self {
        Self {
            merge_buffers: HashMap::new(),
            large_merge_threshold: threshold,
        }
    }

    /// Registers a pre-allocated merge buffer for a GPU.
    pub fn register_buffer(&mut self, device_id: DeviceId, base_addr: VirtualAddr, size: usize) {
        self.merge_buffers
            .insert(device_id, MergeBuffer::new(device_id, base_addr, size));
    }

    /// Returns whether a buffer is registered for the given device.
    pub fn has_buffer(&self, device_id: DeviceId) -> bool {
        self.merge_buffers.contains_key(&device_id)
    }

    /// Resets all merge buffers for reuse.
    pub fn reset_all_buffers(&mut self) {
        for buffer in self.merge_buffers.values_mut() {
            buffer.reset();
        }
    }

    /// Determines merge strategy for a given operation.
    pub fn merge_strategy(&self, element_count: u32) -> MergeStrategy {
        if element_count <= 1 {
            MergeStrategy::Scalar
        } else if element_count <= self.large_merge_threshold {
            MergeStrategy::HostLoop
        } else {
            MergeStrategy::GpuKernel
        }
    }

    /// Estimates the merge time for a merge plan in microseconds.
    pub fn estimate_merge_time(&self, plan: &MergePlan, gpu_count: usize) -> f64 {
        let mut total_us = 0.0;
        for op in &plan.merge_ops {
            let strategy = self.merge_strategy(op.element_count);
            total_us += match strategy {
                MergeStrategy::Scalar => {
                    // 2 * N small memcpy (DtoH + HtoD) + host arithmetic
                    5.0 + gpu_count as f64 * 2.0
                }
                MergeStrategy::HostLoop => {
                    // Read all partials + host loop + write back
                    let bytes_per_partial =
                        op.element_count as f64 * op.data_type.size_bytes() as f64;
                    let transfer_us =
                        bytes_per_partial * gpu_count as f64 / TRANSPORT_BANDWIDTH_BYTES_PER_US;
                    transfer_us * 2.0 + op.element_count as f64 * 0.001 // ~1ns per element
                }
                MergeStrategy::GpuKernel => {
                    // P2P copy of partials + merge kernel
                    let bytes_per_partial =
                        op.element_count as f64 * op.data_type.size_bytes() as f64;
                    let copy_us = bytes_per_partial * (gpu_count as f64 - 1.0)
                        / TRANSPORT_BANDWIDTH_BYTES_PER_US;
                    let kernel_us = 10.0; // Merge kernel is fast
                    copy_us + kernel_us
                }
            };
        }
        total_us
    }
}

impl Default for MergeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge strategy selected based on element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Single element: host-side read + combine + write (fastest for 1-element).
    Scalar,
    /// Small array (2-1024 elements): host-side loop.
    HostLoop,
    /// Large array (>1024 elements): GPU merge kernel.
    GpuKernel,
}

// ---------------------------------------------------------------------------
// Data placement assessment (R17 integration)
// ---------------------------------------------------------------------------

/// Assessment of data placement for a potential kernel split.
#[derive(Debug, Clone)]
pub struct DataPlacementAssessment {
    /// Bytes of read-only data that need replication to target GPUs.
    pub replication_needed_bytes: u64,
    /// Whether writable data is already partitioned across target GPUs.
    pub already_distributed: bool,
    /// Estimated time to replicate needed data in microseconds.
    pub estimated_replication_time_us: f64,
}

impl DataPlacementAssessment {
    /// Returns true if no data replication is needed (ideal for splitting).
    pub fn is_ideal(&self) -> bool {
        self.replication_needed_bytes == 0 && self.already_distributed
    }
}

// ---------------------------------------------------------------------------
// Split Launch Orchestrator
// ---------------------------------------------------------------------------

/// Statistics for the split launch orchestrator.
#[derive(Debug)]
pub struct OrchestratorStats {
    /// Total kernel launches intercepted.
    pub total_launches: AtomicU64,
    /// Launches that were split across GPUs.
    pub split_launches: AtomicU64,
    /// Launches forwarded to single GPU (not split).
    pub single_launches: AtomicU64,
    /// Total merge operations performed.
    pub merge_operations: AtomicU64,
    /// Cumulative merge time in microseconds (fixed-point * 1000).
    pub cumulative_merge_time_us: AtomicU64,
    /// Number of split launches that achieved speedup > 1.0.
    pub beneficial_splits: AtomicU64,
}

impl OrchestratorStats {
    /// Creates a new zeroed stats instance.
    pub fn new() -> Self {
        Self {
            total_launches: AtomicU64::new(0),
            split_launches: AtomicU64::new(0),
            single_launches: AtomicU64::new(0),
            merge_operations: AtomicU64::new(0),
            cumulative_merge_time_us: AtomicU64::new(0),
            beneficial_splits: AtomicU64::new(0),
        }
    }

    /// Records a kernel launch.
    pub fn record_launch(&self, was_split: bool) {
        self.total_launches.fetch_add(1, Ordering::Relaxed);
        if was_split {
            self.split_launches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.single_launches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a merge operation with its duration.
    pub fn record_merge(&self, duration_us: f64) {
        self.merge_operations.fetch_add(1, Ordering::Relaxed);
        self.cumulative_merge_time_us
            .fetch_add((duration_us * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Records a beneficial split (speedup > 1.0).
    pub fn record_beneficial_split(&self) {
        self.beneficial_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the split ratio (fraction of launches that were split).
    pub fn split_ratio(&self) -> f64 {
        let total = self.total_launches.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.split_launches.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Returns the average merge time in microseconds.
    pub fn avg_merge_time_us(&self) -> f64 {
        let count = self.merge_operations.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total = self.cumulative_merge_time_us.load(Ordering::Relaxed) as f64 / 1000.0;
        total / count as f64
    }
}

impl Default for OrchestratorStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinates split kernel launches across multiple GPUs.
///
/// Integrates with R26 PTP for synchronized launch timing, R23 for GPU
/// capability scoring, and R13 for graph-level context.
pub struct SplitLaunchOrchestrator {
    /// Kernel classifier (shared, module-load-time classification).
    pub classifier: Arc<KernelClassifier>,

    /// PTX transformer (shared).
    pub transformer: Arc<PtxTransformer>,

    /// Merge executor for YELLOW kernel post-processing.
    pub merge_executor: MergeExecutor,

    /// Available GPUs for splitting.
    pub gpu_targets: Vec<GpuTarget>,

    /// Statistics.
    pub stats: OrchestratorStats,
}

impl SplitLaunchOrchestrator {
    /// Creates a new split launch orchestrator.
    pub fn new(
        classifier: Arc<KernelClassifier>,
        transformer: Arc<PtxTransformer>,
        gpu_targets: Vec<GpuTarget>,
    ) -> Self {
        Self {
            classifier,
            transformer,
            merge_executor: MergeExecutor::new(),
            gpu_targets,
            stats: OrchestratorStats::new(),
        }
    }

    /// Makes a split decision for a kernel launch.
    ///
    /// Implements the decision tree:
    /// 1. Check classification (RED -> no split)
    /// 2. Check grid size (too small -> no split)
    /// 3. Check GPU count (single GPU -> no split)
    /// 4. Compute block partition based on GPU capability scores
    /// 5. Build merge plan if YELLOW
    pub fn make_split_decision(
        &self,
        launch: &LaunchParams,
        classification: &Arc<KernelClassification>,
    ) -> SplitDecision {
        // Check: RED classification
        if classification.color == KernelColor::Red {
            return SplitDecision {
                original: launch.clone(),
                classification: classification.clone(),
                should_split: false,
                decision_reason: SplitDecisionReason::RedClassification,
                partition: None,
                target_gpus: Vec::new(),
                merge_plan: None,
            };
        }

        // Check: only one GPU available
        if self.gpu_targets.len() < 2 {
            return SplitDecision {
                original: launch.clone(),
                classification: classification.clone(),
                should_split: false,
                decision_reason: SplitDecisionReason::SingleGpu,
                partition: None,
                target_gpus: Vec::new(),
                merge_plan: None,
            };
        }

        // Check: grid too small
        let total_blocks = launch.total_blocks();
        if total_blocks < self.classifier.min_grid_size_for_split as u64 {
            return SplitDecision {
                original: launch.clone(),
                classification: classification.clone(),
                should_split: false,
                decision_reason: SplitDecisionReason::GridTooSmall {
                    blocks: total_blocks as u32,
                },
                partition: None,
                target_gpus: Vec::new(),
                merge_plan: None,
            };
        }

        // Gate: YELLOW kernels cannot be split until atomic redirection is
        // implemented in PtxTransformer::redirect_atomics. Without real PTX
        // rewriting, both sub-kernels would write to the same global address,
        // double-counting every reduction.
        if classification.color == KernelColor::Yellow {
            return SplitDecision {
                original: launch.clone(),
                classification: classification.clone(),
                should_split: false,
                decision_reason: SplitDecisionReason::RedClassification,
                partition: None,
                target_gpus: Vec::new(),
                merge_plan: None,
            };
        }

        // Compute block partition
        let targets = self.gpu_targets.clone();
        let partition = self.compute_partition(&launch.grid_dim, &targets, &launch.args);

        // Build merge plan if YELLOW (currently unreachable due to gate above,
        // but kept for when atomic redirection is implemented)
        let merge_plan = if classification.color == KernelColor::Yellow {
            Some(self.build_merge_plan(classification, &targets))
        } else {
            None
        };

        // Cost model gate: only apply when we have real profiling data
        // (estimated_instruction_count from PTX analysis is too coarse for
        // duration estimation). The cost model is designed for Tier 3 runtime
        // adaptation where actual kernel durations are measured.
        // For static analysis, we rely on the classification color + grid size checks above.
        let merge_overhead_us = merge_plan
            .as_ref()
            .map(|p| p.estimated_merge_time_us)
            .unwrap_or(0.0);

        // Estimate speedup
        let gpu_count = targets.len() as f32;
        let expected_speedup = if merge_overhead_us > 0.0 {
            (gpu_count * 0.85) / (1.0 + merge_overhead_us as f32 / 100.0)
        } else {
            gpu_count * 0.85
        };

        SplitDecision {
            original: launch.clone(),
            classification: classification.clone(),
            should_split: true,
            decision_reason: SplitDecisionReason::Split { expected_speedup },
            partition: Some(partition),
            target_gpus: targets,
            merge_plan,
        }
    }

    /// Computes a block partition across GPUs based on capability scores.
    ///
    /// Splits along the largest grid dimension, distributing blocks proportionally
    /// to each GPU's TFLOPS score for heterogeneous balance.
    pub fn compute_partition(
        &self,
        grid: &Dim3,
        targets: &[GpuTarget],
        args: &[KernelArg],
    ) -> BlockPartition {
        let split_dim = grid.largest_dimension();
        let total_blocks = grid.get(split_dim);

        // Calculate weighted distribution based on TFLOPS scores
        let total_score: f32 = targets.iter().map(|g| g.tflops_score).sum();
        let mut assignments = Vec::with_capacity(targets.len());
        let mut offset = 0u32;
        let mut remaining_blocks = total_blocks;

        for (i, target) in targets.iter().enumerate() {
            let fraction = if total_score > 0.0 {
                target.tflops_score / total_score
            } else {
                1.0 / targets.len() as f32
            };

            let block_count = if i == targets.len() - 1 {
                // Last GPU gets the remainder to avoid rounding issues
                remaining_blocks
            } else {
                let count = (total_blocks as f32 * fraction).round() as u32;
                count.min(remaining_blocks)
            };

            let gpu_grid = grid.with_dim(split_dim, block_count);
            let block_offset = Dim3::new(
                if split_dim == SplitDimension::X { offset } else { 0 },
                if split_dim == SplitDimension::Y { offset } else { 0 },
                if split_dim == SplitDimension::Z { offset } else { 0 },
            );

            // Clone args and append block offsets for each dimension.
            // The transformer injects up to 3 params (__outerlink_blkoff_x/y/z),
            // so we push one BlockOffset per dimension with 0 for non-split dims.
            let mut remapped_args = args.to_vec();
            remapped_args.push(KernelArg::BlockOffset(
                if split_dim == SplitDimension::X { offset } else { 0 },
            ));
            remapped_args.push(KernelArg::BlockOffset(
                if split_dim == SplitDimension::Y { offset } else { 0 },
            ));
            remapped_args.push(KernelArg::BlockOffset(
                if split_dim == SplitDimension::Z { offset } else { 0 },
            ));

            assignments.push(GpuBlockAssignment {
                gpu: target.clone(),
                grid_dim: gpu_grid,
                block_offset,
                block_count,
                fraction,
                remapped_args,
            });

            offset += block_count;
            remaining_blocks -= block_count;
        }

        BlockPartition {
            assignments,
            original_grid: *grid,
            split_dimension: split_dim,
        }
    }

    /// Builds a merge plan for a YELLOW kernel based on its redirected atomics.
    fn build_merge_plan(
        &self,
        classification: &KernelClassification,
        targets: &[GpuTarget],
    ) -> MergePlan {
        let mut merge_ops = Vec::new();
        let mut total_time_us = 0.0;

        // Build merge ops from the classification's atomic analysis
        for atomic in &classification.analysis.global_atomics {
            let combine = CombineStrategy::from_atomic_op(atomic.op);
            let element_count = 1u32; // Assume scalar; TODO: detect array atomics

            let local_copies: Vec<(DeviceId, VirtualAddr)> = targets
                .iter()
                .map(|t| (t.device_id, 0u64)) // Addresses set at launch time
                .collect();

            let op = MergeOp {
                original_target: 0, // Set at launch time
                local_copies,
                combine,
                data_type: atomic.data_type,
                element_count,
            };

            // Estimate merge time
            let strategy = self.merge_executor.merge_strategy(element_count);
            let op_time = match strategy {
                MergeStrategy::Scalar => 5.0 + targets.len() as f64 * 2.0,
                MergeStrategy::HostLoop => element_count as f64 * 0.01,
                MergeStrategy::GpuKernel => 90.0,
            };
            total_time_us += op_time;

            merge_ops.push(op);
        }

        MergePlan {
            merge_ops,
            estimated_merge_time_us: total_time_us,
        }
    }

    /// Handles a kernel launch: classify, decide, and execute.
    ///
    /// This is the main entry point called from cuLaunchKernel interception.
    /// Returns the split decision for diagnostics (actual GPU launches are
    /// performed by the transport layer).
    ///
    /// TODO: Implement actual CUDA driver calls for split launches.
    /// Currently returns the decision without executing.
    pub fn handle_launch(
        &self,
        launch: &LaunchParams,
        entry_name: &str,
        ptx_source: &str,
    ) -> SplitDecision {
        let classification = self.classifier.get_or_classify(
            launch.function,
            entry_name,
            ptx_source,
        );

        let decision = self.make_split_decision(launch, &classification);
        self.stats.record_launch(decision.should_split);

        decision
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Cost model configuration for split vs non-split decisions.
#[derive(Debug, Clone)]
pub struct SplitCostModel {
    /// Fixed overhead of coordinating a split launch (microseconds).
    /// Includes PTP synchronization, argument preparation, stream setup.
    pub split_overhead_us: f64,

    /// Per-GPU overhead for each participant (microseconds).
    /// Includes context switch, module load check, stream creation.
    pub per_gpu_overhead_us: f64,

    /// Efficiency factor (0.0 to 1.0). Accounts for load imbalance,
    /// memory access overhead, etc. Typical: 0.80-0.90 for GREEN kernels.
    pub efficiency_factor: f64,

    /// Minimum kernel duration (us) for splitting to be considered.
    pub min_kernel_duration_us: f64,

    /// Minimum expected speedup to justify splitting.
    pub min_speedup_threshold: f64,
}

impl SplitCostModel {
    /// Creates a default cost model with conservative estimates.
    pub fn new() -> Self {
        Self {
            split_overhead_us: 10.0,
            per_gpu_overhead_us: 5.0,
            efficiency_factor: 0.85,
            min_kernel_duration_us: 20.0,
            min_speedup_threshold: 1.2,
        }
    }

    /// Estimates the speedup of splitting a kernel across N GPUs.
    ///
    /// Returns the expected speedup factor (>1.0 means faster with splitting).
    pub fn estimate_speedup(
        &self,
        kernel_duration_us: f64,
        gpu_count: usize,
        merge_time_us: f64,
    ) -> f64 {
        if kernel_duration_us <= 0.0 || gpu_count <= 1 {
            return 1.0;
        }

        let parallel_time = kernel_duration_us / (gpu_count as f64 * self.efficiency_factor);
        let overhead = self.split_overhead_us + self.per_gpu_overhead_us * gpu_count as f64;
        let split_total = parallel_time + overhead + merge_time_us;

        kernel_duration_us / split_total
    }

    /// Returns true if splitting is expected to be beneficial.
    pub fn should_split(
        &self,
        kernel_duration_us: f64,
        gpu_count: usize,
        merge_time_us: f64,
    ) -> bool {
        if kernel_duration_us < self.min_kernel_duration_us {
            return false;
        }
        let speedup = self.estimate_speedup(kernel_duration_us, gpu_count, merge_time_us);
        speedup >= self.min_speedup_threshold
    }

    /// Finds the optimal number of GPUs for a given kernel duration.
    /// Returns (gpu_count, expected_speedup).
    pub fn optimal_gpu_count(
        &self,
        kernel_duration_us: f64,
        max_gpus: usize,
        merge_time_per_gpu_us: f64,
    ) -> (usize, f64) {
        let mut best_count = 1;
        let mut best_speedup = 1.0;

        for n in 2..=max_gpus {
            let merge_time = merge_time_per_gpu_us * (n as f64 - 1.0);
            let speedup = self.estimate_speedup(kernel_duration_us, n, merge_time);
            if speedup > best_speedup {
                best_speedup = speedup;
                best_count = n;
            }
        }

        (best_count, best_speedup)
    }
}

impl Default for SplitCostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Go/No-Go criteria
// ---------------------------------------------------------------------------

/// Go/No-Go gate criteria for Phase A (classifier results).
#[derive(Debug, Clone)]
pub struct PhaseAGate {
    /// Minimum fraction of GREEN kernels by count.
    pub min_green_fraction: f64,
    /// Minimum fraction of GREEN kernels by compute time.
    pub min_green_compute_fraction: f64,
    /// Minimum fraction of splittable (GREEN + YELLOW) kernels by compute time.
    pub min_splittable_compute_fraction: f64,
    /// Minimum PTX availability fraction.
    pub min_ptx_availability: f64,
}

impl PhaseAGate {
    /// Creates gate criteria with the thresholds from the preplan.
    pub fn new() -> Self {
        Self {
            min_green_fraction: 0.20,
            min_green_compute_fraction: 0.30,
            min_splittable_compute_fraction: 0.40,
            min_ptx_availability: 0.70,
        }
    }

    /// Evaluates whether to proceed to Phase B.
    pub fn evaluate(
        &self,
        green_fraction: f64,
        green_compute_fraction: f64,
        splittable_compute_fraction: f64,
        ptx_availability: f64,
    ) -> GateResult {
        let mut failures = Vec::new();

        if green_fraction < self.min_green_fraction {
            failures.push(format!(
                "GREEN fraction {:.1}% < {:.1}%",
                green_fraction * 100.0,
                self.min_green_fraction * 100.0
            ));
        }
        if green_compute_fraction < self.min_green_compute_fraction {
            failures.push(format!(
                "GREEN compute fraction {:.1}% < {:.1}%",
                green_compute_fraction * 100.0,
                self.min_green_compute_fraction * 100.0
            ));
        }
        if splittable_compute_fraction < self.min_splittable_compute_fraction {
            failures.push(format!(
                "Splittable compute fraction {:.1}% < {:.1}%",
                splittable_compute_fraction * 100.0,
                self.min_splittable_compute_fraction * 100.0
            ));
        }
        if ptx_availability < self.min_ptx_availability {
            failures.push(format!(
                "PTX availability {:.1}% < {:.1}%",
                ptx_availability * 100.0,
                self.min_ptx_availability * 100.0
            ));
        }

        if failures.is_empty() {
            GateResult::Go
        } else {
            GateResult::NoGo { reasons: failures }
        }
    }
}

impl Default for PhaseAGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Go/No-Go gate criteria for Phase B (splitting efficiency).
#[derive(Debug, Clone)]
pub struct PhaseBGate {
    /// Minimum speedup for elementwise kernels on 2 GPUs.
    pub min_elementwise_speedup: f64,
    /// Minimum speedup for GEMM kernels on 2 GPUs.
    pub min_gemm_speedup: f64,
    /// Maximum acceptable split overhead in microseconds.
    pub max_split_overhead_us: f64,
    /// Maximum acceptable PTX transformation failure rate.
    pub max_transform_failure_rate: f64,
}

impl PhaseBGate {
    /// Creates gate criteria with the thresholds from the preplan.
    pub fn new() -> Self {
        Self {
            min_elementwise_speedup: 1.7,
            min_gemm_speedup: 1.3,
            max_split_overhead_us: 10.0,
            max_transform_failure_rate: 0.10,
        }
    }

    /// Evaluates whether to proceed to Phase C.
    pub fn evaluate(
        &self,
        elementwise_speedup: f64,
        gemm_speedup: f64,
        split_overhead_us: f64,
        transform_failure_rate: f64,
        has_correctness_failures: bool,
    ) -> GateResult {
        if has_correctness_failures {
            return GateResult::NoGo {
                reasons: vec!["Correctness failures detected (zero tolerance)".to_string()],
            };
        }

        let mut failures = Vec::new();

        if elementwise_speedup < self.min_elementwise_speedup {
            failures.push(format!(
                "Elementwise speedup {:.2}x < {:.2}x",
                elementwise_speedup, self.min_elementwise_speedup
            ));
        }
        if gemm_speedup < self.min_gemm_speedup {
            failures.push(format!(
                "GEMM speedup {:.2}x < {:.2}x",
                gemm_speedup, self.min_gemm_speedup
            ));
        }
        if split_overhead_us > self.max_split_overhead_us {
            failures.push(format!(
                "Split overhead {:.1}us > {:.1}us",
                split_overhead_us, self.max_split_overhead_us
            ));
        }
        if transform_failure_rate > self.max_transform_failure_rate {
            failures.push(format!(
                "Transform failure rate {:.1}% > {:.1}%",
                transform_failure_rate * 100.0,
                self.max_transform_failure_rate * 100.0
            ));
        }

        if failures.is_empty() {
            GateResult::Go
        } else {
            GateResult::NoGo { reasons: failures }
        }
    }
}

impl Default for PhaseBGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of evaluating a go/no-go gate.
#[derive(Debug, Clone, PartialEq)]
pub enum GateResult {
    /// Proceed to the next phase.
    Go,
    /// Stop: requirements not met.
    NoGo {
        /// Reasons why the gate failed.
        reasons: Vec<String>,
    },
}

impl GateResult {
    /// Returns true if the result is Go.
    pub fn is_go(&self) -> bool {
        matches!(self, Self::Go)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper functions ---

    fn sample_green_ptx() -> &'static str {
        r#".visible .entry _Z9addKernelPfS_S_i(
    .param .u64 _Z9addKernelPfS_S_i_param_0,
    .param .u64 _Z9addKernelPfS_S_i_param_1,
    .param .u64 _Z9addKernelPfS_S_i_param_2,
    .param .s32 _Z9addKernelPfS_S_i_param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;

    mov.u32     %r1, %ctaid.x;
    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %tid.x;
    mad.lo.s32  %r4, %r1, %r2, %r3;
    ld.param.s32 %r5, [_Z9addKernelPfS_S_i_param_3];
    setp.ge.s32 %p1, %r4, %r5;
    @%p1 bra    $L__BB0_2;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32     %f3, %f1, %f2;
    st.global.f32 [%rd3], %f3;
$L__BB0_2:
    ret;
}"#
    }

    fn sample_yellow_ptx() -> &'static str {
        r#".visible .entry _Z13reduceKernelPfS_i(
    .param .u64 _Z13reduceKernelPfS_i_param_0,
    .param .u64 _Z13reduceKernelPfS_i_param_1,
    .param .s32 _Z13reduceKernelPfS_i_param_2
)
{
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;

    mov.u32     %r1, %ctaid.x;
    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %tid.x;
    mad.lo.s32  %r4, %r1, %r2, %r3;
    ld.global.f32 %f1, [%rd1];
    atom.global.add.f32 %f2, [%rd2], %f1;
    ret;
}"#
    }

    fn sample_red_grid_sync_ptx() -> &'static str {
        r#".visible .entry _Z11jacobiStepPfS_i(
    .param .u64 param_0,
    .param .u64 param_1,
    .param .s32 param_2
)
{
    .reg .b32 %r<4>;
    mov.u32 %r1, %ctaid.x;
    bar.sync 0;
    cooperative_groups;
    grid.sync;
    ld.global.f32 %f1, [%rd1];
    st.global.f32 [%rd2], %f1;
    ret;
}"#
    }

    fn sample_red_cas_ptx() -> &'static str {
        r#".visible .entry _Z9lockFreeQPiS_(
    .param .u64 param_0,
    .param .u64 param_1
)
{
    .reg .b32 %r<4>;
    mov.u32 %r1, %ctaid.x;
    mad.lo.s32 %r2, %r1, %r3, %r4;
    ld.global.u32 %r3, [%rd1];
    atom.global.cas.b32 %r4, [%rd1], %r3, %r2;
    @%p1 bra $retry;
    ret;
}"#
    }

    fn sample_red_dynamic_parallelism_ptx() -> &'static str {
        r#".visible .entry _Z10dynKernelPf(
    .param .u64 param_0
)
{
    .reg .b32 %r<2>;
    mov.u32 %r1, %ctaid.x;
    cudaLaunchDevice;
    ret;
}"#
    }

    fn make_gpu_target(id: u32, tflops: f32, sm_count: u32) -> GpuTarget {
        GpuTarget::new(id, 0, tflops, sm_count, (8, 6))
    }

    // =====================================================================
    // KernelColor tests
    // =====================================================================

    #[test]
    fn test_kernel_color_variants() {
        assert_ne!(KernelColor::Green, KernelColor::Yellow);
        assert_ne!(KernelColor::Yellow, KernelColor::Red);
        assert_ne!(KernelColor::Green, KernelColor::Red);
    }

    // =====================================================================
    // ClassificationReason tests
    // =====================================================================

    #[test]
    fn test_classification_reason_implied_color_green() {
        assert_eq!(
            ClassificationReason::FullyIndependent.implied_color(),
            KernelColor::Green
        );
        assert_eq!(
            ClassificationReason::SharedMemoryAtomicsOnly.implied_color(),
            KernelColor::Green
        );
    }

    #[test]
    fn test_classification_reason_implied_color_yellow() {
        assert_eq!(
            ClassificationReason::GlobalReductionAtomic {
                atomic_count: 1,
                target_count: 1,
            }
            .implied_color(),
            KernelColor::Yellow
        );
        assert_eq!(
            ClassificationReason::GlobalMinMaxAtomic {
                atomic_count: 2,
                target_count: 1,
            }
            .implied_color(),
            KernelColor::Yellow
        );
        assert_eq!(
            ClassificationReason::HistogramPattern {
                estimated_bin_count: 256,
            }
            .implied_color(),
            KernelColor::Yellow
        );
    }

    #[test]
    fn test_classification_reason_implied_color_red() {
        assert_eq!(
            ClassificationReason::CooperativeGridSync.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::GlobalCasAtomic { cas_count: 1 }.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::DataDependentAccess.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::IndirectMemoryAccess.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::DynamicParallelism.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::GridTooSmall { block_count: 4 }.implied_color(),
            KernelColor::Red
        );
        assert_eq!(
            ClassificationReason::UnanalyzablePtx {
                reason: "test".into()
            }
            .implied_color(),
            KernelColor::Red
        );
    }

    // =====================================================================
    // PtxDataType tests
    // =====================================================================

    #[test]
    fn test_ptx_data_type_size_bytes() {
        assert_eq!(PtxDataType::S32.size_bytes(), 4);
        assert_eq!(PtxDataType::U32.size_bytes(), 4);
        assert_eq!(PtxDataType::F32.size_bytes(), 4);
        assert_eq!(PtxDataType::S64.size_bytes(), 8);
        assert_eq!(PtxDataType::U64.size_bytes(), 8);
        assert_eq!(PtxDataType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_ptx_data_type_from_suffix() {
        assert_eq!(PtxDataType::from_ptx_suffix("s32"), Some(PtxDataType::S32));
        assert_eq!(PtxDataType::from_ptx_suffix("u32"), Some(PtxDataType::U32));
        assert_eq!(PtxDataType::from_ptx_suffix("f32"), Some(PtxDataType::F32));
        assert_eq!(PtxDataType::from_ptx_suffix("s64"), Some(PtxDataType::S64));
        assert_eq!(PtxDataType::from_ptx_suffix("u64"), Some(PtxDataType::U64));
        assert_eq!(PtxDataType::from_ptx_suffix("f64"), Some(PtxDataType::F64));
        assert_eq!(PtxDataType::from_ptx_suffix("bfloat16"), None);
    }

    // =====================================================================
    // DimensionsUsed tests
    // =====================================================================

    #[test]
    fn test_dimensions_used_default() {
        let d = DimensionsUsed::default();
        assert!(!d.x && !d.y && !d.z);
        assert_eq!(d.count(), 0);
        assert!(!d.any());
    }

    #[test]
    fn test_dimensions_used_single() {
        let d = DimensionsUsed {
            x: true,
            y: false,
            z: false,
        };
        assert_eq!(d.count(), 1);
        assert!(d.any());
    }

    #[test]
    fn test_dimensions_used_all() {
        let d = DimensionsUsed {
            x: true,
            y: true,
            z: true,
        };
        assert_eq!(d.count(), 3);
        assert!(d.any());
    }

    // =====================================================================
    // Dim3 tests
    // =====================================================================

    #[test]
    fn test_dim3_new_and_total() {
        let d = Dim3::new(256, 128, 1);
        assert_eq!(d.total(), 256 * 128);
    }

    #[test]
    fn test_dim3_one_d() {
        let d = Dim3::one_d(1024);
        assert_eq!(d.x, 1024);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
        assert_eq!(d.total(), 1024);
    }

    #[test]
    fn test_dim3_largest_dim() {
        assert_eq!(Dim3::new(256, 128, 1).largest_dim(), 256);
        assert_eq!(Dim3::new(1, 512, 1).largest_dim(), 512);
        assert_eq!(Dim3::new(1, 1, 64).largest_dim(), 64);
    }

    #[test]
    fn test_dim3_largest_dimension() {
        assert_eq!(
            Dim3::new(256, 128, 1).largest_dimension(),
            SplitDimension::X
        );
        assert_eq!(
            Dim3::new(1, 512, 1).largest_dimension(),
            SplitDimension::Y
        );
        assert_eq!(
            Dim3::new(1, 1, 64).largest_dimension(),
            SplitDimension::Z
        );
    }

    #[test]
    fn test_dim3_get_and_with_dim() {
        let d = Dim3::new(10, 20, 30);
        assert_eq!(d.get(SplitDimension::X), 10);
        assert_eq!(d.get(SplitDimension::Y), 20);
        assert_eq!(d.get(SplitDimension::Z), 30);

        let d2 = d.with_dim(SplitDimension::Y, 100);
        assert_eq!(d2.y, 100);
        assert_eq!(d2.x, 10);
        assert_eq!(d2.z, 30);
    }

    // =====================================================================
    // PointerRole tests
    // =====================================================================

    #[test]
    fn test_pointer_role_variants() {
        let roles = [
            PointerRole::ReadOnly,
            PointerRole::WriteOnly,
            PointerRole::ReadWrite,
            PointerRole::AtomicTarget,
            PointerRole::Unknown,
        ];
        // All variants are distinct
        for i in 0..roles.len() {
            for j in i + 1..roles.len() {
                assert_ne!(roles[i], roles[j]);
            }
        }
    }

    // =====================================================================
    // MemorySpace tests
    // =====================================================================

    #[test]
    fn test_memory_space_variants() {
        assert_ne!(MemorySpace::Global, MemorySpace::Shared);
    }

    // =====================================================================
    // AtomicOpType tests
    // =====================================================================

    #[test]
    fn test_atomic_op_type_all_variants() {
        let ops = [
            AtomicOpType::Add,
            AtomicOpType::Sub,
            AtomicOpType::Min,
            AtomicOpType::Max,
            AtomicOpType::And,
            AtomicOpType::Or,
            AtomicOpType::Xor,
            AtomicOpType::Inc,
            AtomicOpType::Dec,
        ];
        assert_eq!(ops.len(), 9);
    }

    // =====================================================================
    // CombineStrategy tests
    // =====================================================================

    #[test]
    fn test_combine_strategy_from_atomic_op() {
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Add),
            CombineStrategy::Sum
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Sub),
            CombineStrategy::Sum
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Min),
            CombineStrategy::Min
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Max),
            CombineStrategy::Max
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::And),
            CombineStrategy::BitwiseAnd
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Or),
            CombineStrategy::BitwiseOr
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Xor),
            CombineStrategy::BitwiseXor
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Inc),
            CombineStrategy::Sum
        );
        assert_eq!(
            CombineStrategy::from_atomic_op(AtomicOpType::Dec),
            CombineStrategy::Sum
        );
    }

    #[test]
    fn test_combine_u64_sum() {
        assert_eq!(CombineStrategy::Sum.combine_u64(10, 20), 30);
    }

    #[test]
    fn test_combine_u64_min() {
        assert_eq!(CombineStrategy::Min.combine_u64(10, 20), 10);
    }

    #[test]
    fn test_combine_u64_max() {
        assert_eq!(CombineStrategy::Max.combine_u64(10, 20), 20);
    }

    #[test]
    fn test_combine_u64_bitwise() {
        assert_eq!(CombineStrategy::BitwiseAnd.combine_u64(0xFF, 0x0F), 0x0F);
        assert_eq!(CombineStrategy::BitwiseOr.combine_u64(0xF0, 0x0F), 0xFF);
        assert_eq!(CombineStrategy::BitwiseXor.combine_u64(0xFF, 0x0F), 0xF0);
    }

    #[test]
    fn test_combine_f64() {
        assert!((CombineStrategy::Sum.combine_f64(1.5, 2.5) - 4.0).abs() < f64::EPSILON);
        assert!((CombineStrategy::Min.combine_f64(1.5, 2.5) - 1.5).abs() < f64::EPSILON);
        assert!((CombineStrategy::Max.combine_f64(1.5, 2.5) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_identity() {
        assert_eq!(CombineStrategy::Sum.identity_u64(), 0);
        assert_eq!(CombineStrategy::Min.identity_u64(), u64::MAX);
        assert_eq!(CombineStrategy::Max.identity_u64(), 0);
        assert_eq!(CombineStrategy::BitwiseAnd.identity_u64(), u64::MAX);
        assert_eq!(CombineStrategy::BitwiseOr.identity_u64(), 0);
        assert_eq!(CombineStrategy::BitwiseXor.identity_u64(), 0);
    }

    // =====================================================================
    // ClassifierStats tests
    // =====================================================================

    #[test]
    fn test_classifier_stats_new() {
        let stats = ClassifierStats::new();
        assert_eq!(stats.total(), 0);
        assert_eq!(stats.green_fraction(), 0.0);
        assert_eq!(stats.splittable_fraction(), 0.0);
    }

    #[test]
    fn test_classifier_stats_record() {
        let stats = ClassifierStats::new();
        stats.record(KernelColor::Green, false);
        stats.record(KernelColor::Green, false);
        stats.record(KernelColor::Yellow, false);
        stats.record(KernelColor::Red, false);
        stats.record(KernelColor::Red, true);

        assert_eq!(stats.total(), 5);
        assert!((stats.green_fraction() - 0.4).abs() < f64::EPSILON);
        assert!((stats.splittable_fraction() - 0.6).abs() < f64::EPSILON);
        assert_eq!(stats.unanalyzable_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_classifier_stats_compute_fractions() {
        let stats = ClassifierStats::new();
        stats.set_green_compute_fraction(0.35);
        stats.set_yellow_compute_fraction(0.15);
        assert!((stats.green_compute_frac() - 0.35).abs() < 0.001);
        assert!((stats.yellow_compute_frac() - 0.15).abs() < 0.001);
    }

    // =====================================================================
    // PtxParser tests
    // =====================================================================

    #[test]
    fn test_ptx_parser_empty_source() {
        let parser = PtxParser::new();
        assert!(parser.analyze_ptx("").is_err());
    }

    #[test]
    fn test_ptx_parser_green_kernel() {
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_green_ptx()).unwrap();
        assert_eq!(analysis.param_count, 4);
        assert!(analysis.dimensions_used.x);
        assert!(!analysis.has_grid_sync);
        assert!(!analysis.has_dynamic_parallelism);
        assert!(analysis.global_atomics.is_empty());
    }

    #[test]
    fn test_ptx_parser_yellow_kernel() {
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_yellow_ptx()).unwrap();
        assert!(!analysis.global_atomics.is_empty());
        assert_eq!(analysis.global_atomics[0].op, AtomicOpType::Add);
        assert_eq!(analysis.global_atomics[0].memory_space, MemorySpace::Global);
    }

    #[test]
    fn test_ptx_parser_grid_sync_detection() {
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_red_grid_sync_ptx()).unwrap();
        assert!(analysis.has_grid_sync);
    }

    #[test]
    fn test_ptx_parser_dynamic_parallelism_detection() {
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_red_dynamic_parallelism_ptx()).unwrap();
        assert!(analysis.has_dynamic_parallelism);
    }

    #[test]
    fn test_ptx_parser_cas_detection() {
        let parser = PtxParser::new();
        let cas_info = parser.detect_cas_atomics(sample_red_cas_ptx());
        assert!(!cas_info.is_empty());
    }

    #[test]
    fn test_ptx_parser_shared_memory_detection() {
        let parser = PtxParser::new();
        let ptx = ".shared .align 4 .b8 smem[1024];\nmov.u32 %r1, %ctaid.x;";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert_eq!(analysis.shared_mem_bytes, 1024);
    }

    #[test]
    fn test_ptx_parser_register_count_detection() {
        let parser = PtxParser::new();
        let ptx = ".maxnreg 32\nmov.u32 %r1, %ctaid.x;";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert_eq!(analysis.registers_per_thread, Some(32));
    }

    #[test]
    fn test_ptx_parser_dimensions_detection() {
        let parser = PtxParser::new();
        let ptx = "mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %ctaid.y;\nmad.lo.s32 %r3, %r1, %r2, %r4;";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert!(analysis.dimensions_used.x);
        assert!(analysis.dimensions_used.y);
        assert!(!analysis.dimensions_used.z);
    }

    #[test]
    fn test_ptx_parser_blockidx_linear() {
        let parser = PtxParser::new();
        let ptx = "mov.u32 %r1, %ctaid.x;\nmad.lo.s32 %r2, %r1, 256, %r3;\nld.global.f32 %f1, [%rd1];";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert!(matches!(analysis.blockidx_usage, BlockIdxUsage::Linear { .. }));
    }

    #[test]
    fn test_ptx_parser_pointer_roles_read_only() {
        let parser = PtxParser::new();
        let ptx = ".param .u64 p0\n.param .u64 p1\nld.global.f32 %f1, [%rd1];\nmov.u32 %r1, %ctaid.x;\nmad.lo.s32 %r2, %r1, 1, 0;";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert_eq!(analysis.pointer_roles.len(), 2);
        assert_eq!(analysis.pointer_roles[0], PointerRole::ReadOnly);
    }

    #[test]
    fn test_ptx_parser_shared_atomics_only() {
        let parser = PtxParser::new();
        let ptx = ".param .u64 p0\nmov.u32 %r1, %ctaid.x;\natom.shared.add.f32 %f1, [%rd1], %f2;\nmad.lo.s32 %r2, %r1, 1, 0;\nld.global.f32 %f3, [%rd2];";
        let analysis = parser.analyze_ptx(ptx).unwrap();
        assert!(analysis.global_atomics.is_empty());
        assert!(!analysis.shared_atomics.is_empty());
    }

    // =====================================================================
    // KernelClassifier tests
    // =====================================================================

    #[test]
    fn test_classifier_green_kernel() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(1, "addKernel", sample_green_ptx());
        assert_eq!(result.color, KernelColor::Green);
        assert!(result.is_splittable());
        assert!(!result.needs_merge());
    }

    #[test]
    fn test_classifier_yellow_kernel() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(2, "reduceKernel", sample_yellow_ptx());
        assert_eq!(result.color, KernelColor::Yellow);
        assert!(result.is_splittable());
        assert!(result.needs_merge());
    }

    #[test]
    fn test_classifier_red_grid_sync() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(3, "jacobiStep", sample_red_grid_sync_ptx());
        assert_eq!(result.color, KernelColor::Red);
        assert!(!result.is_splittable());
        assert_eq!(result.reason, ClassificationReason::CooperativeGridSync);
    }

    #[test]
    fn test_classifier_red_cas() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(4, "lockFreeQ", sample_red_cas_ptx());
        assert_eq!(result.color, KernelColor::Red);
        assert!(matches!(
            result.reason,
            ClassificationReason::GlobalCasAtomic { .. }
        ));
    }

    #[test]
    fn test_classifier_red_dynamic_parallelism() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(5, "dynKernel", sample_red_dynamic_parallelism_ptx());
        assert_eq!(result.color, KernelColor::Red);
        assert_eq!(result.reason, ClassificationReason::DynamicParallelism);
    }

    #[test]
    fn test_classifier_red_empty_ptx() {
        let classifier = KernelClassifier::new();
        let result = classifier.classify_kernel(6, "emptyKernel", "");
        assert_eq!(result.color, KernelColor::Red);
        assert!(matches!(
            result.reason,
            ClassificationReason::UnanalyzablePtx { .. }
        ));
    }

    #[test]
    fn test_classifier_caching() {
        let classifier = KernelClassifier::new();
        let _ = classifier.get_or_classify(10, "addKernel", sample_green_ptx());
        assert_eq!(classifier.cache_size(), 1);

        // Second call should use cache
        let cached = classifier.get_or_classify(10, "addKernel", sample_green_ptx());
        assert_eq!(cached.color, KernelColor::Green);
        assert_eq!(classifier.cache_size(), 1);
    }

    #[test]
    fn test_classifier_stats_tracking() {
        let classifier = KernelClassifier::new();
        classifier.classify_kernel(1, "green", sample_green_ptx());
        classifier.classify_kernel(2, "yellow", sample_yellow_ptx());
        classifier.classify_kernel(3, "red", sample_red_grid_sync_ptx());

        assert_eq!(classifier.stats.total(), 3);
        assert_eq!(classifier.stats.green_count.load(Ordering::Relaxed), 1);
        assert_eq!(classifier.stats.yellow_count.load(Ordering::Relaxed), 1);
        assert_eq!(classifier.stats.red_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_classifier_clear_cache() {
        let classifier = KernelClassifier::new();
        classifier.get_or_classify(1, "k", sample_green_ptx());
        assert_eq!(classifier.cache_size(), 1);
        classifier.clear_cache();
        assert_eq!(classifier.cache_size(), 0);
    }

    #[test]
    fn test_classifier_get_cached() {
        let classifier = KernelClassifier::new();
        assert!(classifier.get_cached(99).is_none());
        classifier.get_or_classify(99, "k", sample_green_ptx());
        assert!(classifier.get_cached(99).is_some());
    }

    #[test]
    fn test_classifier_custom_thresholds() {
        let classifier = KernelClassifier::with_thresholds(128, 50.0);
        assert_eq!(classifier.min_grid_size_for_split, 128);
        assert!((classifier.min_kernel_duration_us - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_classifier_shared_atomics_only_is_green() {
        let classifier = KernelClassifier::new();
        let ptx = r#".visible .entry _Z4testPf(
    .param .u64 p0
)
{
    .reg .b32 %r<4>;
    mov.u32 %r1, %ctaid.x;
    mad.lo.s32 %r2, %r1, 256, %r3;
    atom.shared.add.f32 %f1, [%rd1], %f2;
    ld.global.f32 %f3, [%rd2];
    ret;
}"#;
        let result = classifier.classify_kernel(20, "test", ptx);
        assert_eq!(result.color, KernelColor::Green);
        assert_eq!(result.reason, ClassificationReason::SharedMemoryAtomicsOnly);
    }

    // =====================================================================
    // PtxTransformer tests
    // =====================================================================

    #[test]
    fn test_transformer_empty_ptx() {
        let t = PtxTransformer::new(false);
        let analysis = PtxAnalysis {
            param_count: 0,
            pointer_param_count: 0,
            pointer_roles: Vec::new(),
            global_atomics: Vec::new(),
            shared_atomics: Vec::new(),
            has_grid_sync: false,
            has_dynamic_parallelism: false,
            blockidx_usage: BlockIdxUsage::Unknown,
            ctaid_read_count: 0,
            dimensions_used: DimensionsUsed::default(),
            estimated_instruction_count: 0,
            shared_mem_bytes: 0,
            registers_per_thread: None,
        };
        assert!(t.transform("", &analysis).is_err());
    }

    #[test]
    fn test_transformer_no_dimensions_used() {
        let t = PtxTransformer::new(false);
        let analysis = PtxAnalysis {
            param_count: 1,
            pointer_param_count: 1,
            pointer_roles: vec![PointerRole::ReadOnly],
            global_atomics: Vec::new(),
            shared_atomics: Vec::new(),
            has_grid_sync: false,
            has_dynamic_parallelism: false,
            blockidx_usage: BlockIdxUsage::NonAddressing,
            ctaid_read_count: 0,
            dimensions_used: DimensionsUsed::default(),
            estimated_instruction_count: 10,
            shared_mem_bytes: 0,
            registers_per_thread: None,
        };
        let result = t.transform(sample_green_ptx(), &analysis).unwrap();
        assert_eq!(result.offset_params, OffsetParams::default());
        assert!(result.redirected_atomics.is_empty());
    }

    #[test]
    fn test_transformer_injects_x_offset() {
        let t = PtxTransformer::new(false);
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_green_ptx()).unwrap();

        let result = t.transform(sample_green_ptx(), &analysis).unwrap();
        assert!(result.offset_params.x.is_some());
        assert_eq!(
            result.offset_params.x.as_deref(),
            Some("__outerlink_blkoff_x")
        );
        assert!(result.ptx_source.contains("__outerlink_blkoff_x"));
        assert!(result.ptx_source.contains("ld.param.u32 %__ol_off_x"));
        assert!(result.ptx_source.contains("add.u32"));
    }

    #[test]
    fn test_transformer_injects_y_offset_for_2d() {
        let t = PtxTransformer::new(false);
        let ptx = r#".visible .entry kernel(
    .param .u64 p0
)
{
    .reg .b32 %r<4>;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ctaid.y;
    ret;
}"#;
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(ptx).unwrap();

        let result = t.transform(ptx, &analysis).unwrap();
        assert!(result.offset_params.x.is_some());
        assert!(result.offset_params.y.is_some());
        assert!(result.offset_params.z.is_none());
        assert!(result.ptx_source.contains("__outerlink_blkoff_x"));
        assert!(result.ptx_source.contains("__outerlink_blkoff_y"));
    }

    #[test]
    fn test_transformer_redirects_atomics_for_yellow() {
        let t = PtxTransformer::new(false);
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_yellow_ptx()).unwrap();

        let result = t.transform(sample_yellow_ptx(), &analysis).unwrap();
        assert!(!result.redirected_atomics.is_empty());
        assert_eq!(result.redirected_atomics[0].op_type, AtomicOpType::Add);
    }

    #[test]
    fn test_transformer_validation_mode() {
        let t = PtxTransformer::new(true);
        assert!(t.validation_enabled);
    }

    // =====================================================================
    // OffsetParams tests
    // =====================================================================

    #[test]
    fn test_offset_params_count() {
        assert_eq!(OffsetParams::default().count(), 0);
        assert_eq!(
            OffsetParams {
                x: Some("a".into()),
                y: None,
                z: None
            }
            .count(),
            1
        );
        assert_eq!(
            OffsetParams {
                x: Some("a".into()),
                y: Some("b".into()),
                z: Some("c".into())
            }
            .count(),
            3
        );
    }

    // =====================================================================
    // LaunchParams tests
    // =====================================================================

    #[test]
    fn test_launch_params_total_blocks() {
        let lp = LaunchParams {
            function: 1,
            grid_dim: Dim3::new(256, 1, 1),
            block_dim: Dim3::new(256, 1, 1),
            shared_mem_bytes: 0,
            stream: 0,
            args: Vec::new(),
        };
        assert_eq!(lp.total_blocks(), 256);
    }

    #[test]
    fn test_launch_params_total_threads() {
        let lp = LaunchParams {
            function: 1,
            grid_dim: Dim3::new(100, 1, 1),
            block_dim: Dim3::new(256, 1, 1),
            shared_mem_bytes: 0,
            stream: 0,
            args: Vec::new(),
        };
        assert_eq!(lp.total_threads(), 100 * 256);
    }

    // =====================================================================
    // GpuTarget tests
    // =====================================================================

    #[test]
    fn test_gpu_target_new() {
        let t = GpuTarget::new(0, 1, 35.0, 82, (8, 6));
        assert_eq!(t.device_id, 0);
        assert_eq!(t.node_id, 1);
        assert!((t.tflops_score - 35.0).abs() < f32::EPSILON);
        assert_eq!(t.sm_count, 82);
        assert_eq!(t.compute_capability, (8, 6));
    }

    // =====================================================================
    // BlockPartition tests
    // =====================================================================

    #[test]
    fn test_block_partition_total_assigned() {
        let partition = BlockPartition {
            assignments: vec![
                GpuBlockAssignment {
                    gpu: make_gpu_target(0, 1.0, 82),
                    grid_dim: Dim3::one_d(128),
                    block_offset: Dim3::new(0, 0, 0),
                    block_count: 128,
                    fraction: 0.5,
                    remapped_args: Vec::new(),
                },
                GpuBlockAssignment {
                    gpu: make_gpu_target(1, 1.0, 82),
                    grid_dim: Dim3::one_d(128),
                    block_offset: Dim3::new(128, 0, 0),
                    block_count: 128,
                    fraction: 0.5,
                    remapped_args: Vec::new(),
                },
            ],
            original_grid: Dim3::one_d(256),
            split_dimension: SplitDimension::X,
        };
        assert_eq!(partition.total_assigned_blocks(), 256);
        assert!(partition.is_valid());
    }

    #[test]
    fn test_block_partition_invalid() {
        let partition = BlockPartition {
            assignments: vec![GpuBlockAssignment {
                gpu: make_gpu_target(0, 1.0, 82),
                grid_dim: Dim3::one_d(100),
                block_offset: Dim3::new(0, 0, 0),
                block_count: 100,
                fraction: 1.0,
                remapped_args: Vec::new(),
            }],
            original_grid: Dim3::one_d(256),
            split_dimension: SplitDimension::X,
        };
        assert!(!partition.is_valid());
    }

    // =====================================================================
    // SplitDecisionReason tests
    // =====================================================================

    #[test]
    fn test_split_decision_reason_variants() {
        let split = SplitDecisionReason::Split {
            expected_speedup: 1.7,
        };
        let red = SplitDecisionReason::RedClassification;
        let small = SplitDecisionReason::GridTooSmall { blocks: 10 };
        assert_ne!(split, red);
        assert_ne!(red, small);
    }

    // =====================================================================
    // MergePlan tests
    // =====================================================================

    #[test]
    fn test_merge_plan_all_scalar() {
        let plan = MergePlan {
            merge_ops: vec![MergeOp {
                original_target: 0x1000,
                local_copies: vec![(0, 0x2000), (1, 0x3000)],
                combine: CombineStrategy::Sum,
                data_type: PtxDataType::F32,
                element_count: 1,
            }],
            estimated_merge_time_us: 5.0,
        };
        assert!(plan.is_all_scalar());
        assert_eq!(plan.total_elements(), 1);
    }

    #[test]
    fn test_merge_plan_not_scalar() {
        let plan = MergePlan {
            merge_ops: vec![MergeOp {
                original_target: 0x1000,
                local_copies: vec![(0, 0x2000)],
                combine: CombineStrategy::Sum,
                data_type: PtxDataType::U32,
                element_count: 65536,
            }],
            estimated_merge_time_us: 90.0,
        };
        assert!(!plan.is_all_scalar());
        assert_eq!(plan.total_elements(), 65536);
    }

    // =====================================================================
    // MergeBuffer tests
    // =====================================================================

    #[test]
    fn test_merge_buffer_allocate() {
        let mut buf = MergeBuffer::new(0, 0x10000, 4096);
        assert_eq!(buf.remaining(), 4096);

        let addr = buf.allocate(256).unwrap();
        assert_eq!(addr, 0x10000);
        assert_eq!(buf.remaining(), 3840);

        let addr2 = buf.allocate(256).unwrap();
        assert_eq!(addr2, 0x10100);
    }

    #[test]
    fn test_merge_buffer_overflow() {
        let mut buf = MergeBuffer::new(0, 0x10000, 100);
        assert!(buf.allocate(200).is_none());
    }

    #[test]
    fn test_merge_buffer_reset() {
        let mut buf = MergeBuffer::new(0, 0x10000, 4096);
        buf.allocate(2048).unwrap();
        assert_eq!(buf.remaining(), 2048);
        buf.reset();
        assert_eq!(buf.remaining(), 4096);
    }

    // =====================================================================
    // MergeExecutor tests
    // =====================================================================

    #[test]
    fn test_merge_executor_strategy_scalar() {
        let exec = MergeExecutor::new();
        assert_eq!(exec.merge_strategy(1), MergeStrategy::Scalar);
    }

    #[test]
    fn test_merge_executor_strategy_host_loop() {
        let exec = MergeExecutor::new();
        assert_eq!(exec.merge_strategy(100), MergeStrategy::HostLoop);
        assert_eq!(exec.merge_strategy(1024), MergeStrategy::HostLoop);
    }

    #[test]
    fn test_merge_executor_strategy_gpu_kernel() {
        let exec = MergeExecutor::new();
        assert_eq!(exec.merge_strategy(1025), MergeStrategy::GpuKernel);
        assert_eq!(exec.merge_strategy(65536), MergeStrategy::GpuKernel);
    }

    #[test]
    fn test_merge_executor_custom_threshold() {
        let exec = MergeExecutor::with_threshold(512);
        assert_eq!(exec.merge_strategy(512), MergeStrategy::HostLoop);
        assert_eq!(exec.merge_strategy(513), MergeStrategy::GpuKernel);
    }

    #[test]
    fn test_merge_executor_buffer_registration() {
        let mut exec = MergeExecutor::new();
        assert!(!exec.has_buffer(0));
        exec.register_buffer(0, 0x10000, 4096);
        assert!(exec.has_buffer(0));
    }

    #[test]
    fn test_merge_executor_reset_buffers() {
        let mut exec = MergeExecutor::new();
        exec.register_buffer(0, 0x10000, 4096);
        exec.register_buffer(1, 0x20000, 4096);
        // Allocate from buffer 0
        exec.merge_buffers.get_mut(&0).unwrap().allocate(1024);
        assert_eq!(exec.merge_buffers[&0].remaining(), 3072);
        exec.reset_all_buffers();
        assert_eq!(exec.merge_buffers[&0].remaining(), 4096);
    }

    #[test]
    fn test_merge_executor_estimate_time() {
        let exec = MergeExecutor::new();
        let plan = MergePlan {
            merge_ops: vec![MergeOp {
                original_target: 0,
                local_copies: vec![(0, 0), (1, 0)],
                combine: CombineStrategy::Sum,
                data_type: PtxDataType::F32,
                element_count: 1,
            }],
            estimated_merge_time_us: 0.0, // Will be recalculated
        };
        let time = exec.estimate_merge_time(&plan, 2);
        assert!(time > 0.0);
    }

    // =====================================================================
    // DataPlacementAssessment tests
    // =====================================================================

    #[test]
    fn test_data_placement_ideal() {
        let assess = DataPlacementAssessment {
            replication_needed_bytes: 0,
            already_distributed: true,
            estimated_replication_time_us: 0.0,
        };
        assert!(assess.is_ideal());
    }

    #[test]
    fn test_data_placement_not_ideal() {
        let assess = DataPlacementAssessment {
            replication_needed_bytes: 1024 * 1024,
            already_distributed: false,
            estimated_replication_time_us: 82.0,
        };
        assert!(!assess.is_ideal());
    }

    // =====================================================================
    // OrchestratorStats tests
    // =====================================================================

    #[test]
    fn test_orchestrator_stats_new() {
        let stats = OrchestratorStats::new();
        assert_eq!(stats.total_launches.load(Ordering::Relaxed), 0);
        assert_eq!(stats.split_ratio(), 0.0);
        assert_eq!(stats.avg_merge_time_us(), 0.0);
    }

    #[test]
    fn test_orchestrator_stats_record_launch() {
        let stats = OrchestratorStats::new();
        stats.record_launch(true);
        stats.record_launch(false);
        stats.record_launch(true);

        assert_eq!(stats.total_launches.load(Ordering::Relaxed), 3);
        assert_eq!(stats.split_launches.load(Ordering::Relaxed), 2);
        assert_eq!(stats.single_launches.load(Ordering::Relaxed), 1);
        assert!((stats.split_ratio() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_orchestrator_stats_record_merge() {
        let stats = OrchestratorStats::new();
        stats.record_merge(10.0);
        stats.record_merge(20.0);
        assert_eq!(stats.merge_operations.load(Ordering::Relaxed), 2);
        assert!((stats.avg_merge_time_us() - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_orchestrator_stats_beneficial_split() {
        let stats = OrchestratorStats::new();
        stats.record_beneficial_split();
        stats.record_beneficial_split();
        assert_eq!(stats.beneficial_splits.load(Ordering::Relaxed), 2);
    }

    // =====================================================================
    // SplitLaunchOrchestrator tests
    // =====================================================================

    fn make_orchestrator(gpu_count: usize) -> SplitLaunchOrchestrator {
        let classifier = Arc::new(KernelClassifier::new());
        let transformer = Arc::new(PtxTransformer::new(false));
        let targets: Vec<GpuTarget> = (0..gpu_count)
            .map(|i| make_gpu_target(i as u32, 35.0, 82))
            .collect();
        SplitLaunchOrchestrator::new(classifier, transformer, targets)
    }

    fn make_launch(function: CuFunction, grid_x: u32) -> LaunchParams {
        LaunchParams {
            function,
            grid_dim: Dim3::one_d(grid_x),
            block_dim: Dim3::new(256, 1, 1),
            shared_mem_bytes: 0,
            stream: 0,
            args: vec![
                KernelArg::Pointer {
                    addr: 0x1000,
                    size: 4096,
                },
                KernelArg::Pointer {
                    addr: 0x2000,
                    size: 4096,
                },
                KernelArg::Int(1024),
            ],
        }
    }

    #[test]
    fn test_orchestrator_split_green_kernel() {
        let orch = make_orchestrator(2);
        let launch = make_launch(1, 256);
        let classification = orch.classifier.get_or_classify(1, "addKernel", sample_green_ptx());
        let decision = orch.make_split_decision(&launch, &classification);

        assert!(decision.should_split);
        assert!(matches!(
            decision.decision_reason,
            SplitDecisionReason::Split { .. }
        ));
        assert!(decision.partition.is_some());
        assert!(decision.merge_plan.is_none()); // GREEN = no merge
    }

    #[test]
    fn test_orchestrator_split_yellow_kernel() {
        let orch = make_orchestrator(2);
        let launch = make_launch(2, 256);
        let classification =
            orch.classifier
                .get_or_classify(2, "reduceKernel", sample_yellow_ptx());
        let decision = orch.make_split_decision(&launch, &classification);

        // YELLOW kernels are gated until atomic redirection is implemented.
        // Once PtxTransformer::redirect_atomics rewrites PTX, this should
        // change to assert!(decision.should_split) + merge_plan.is_some().
        assert!(!decision.should_split);
    }

    #[test]
    fn test_orchestrator_no_split_red_kernel() {
        let orch = make_orchestrator(2);
        let launch = make_launch(3, 256);
        let classification =
            orch.classifier
                .get_or_classify(3, "jacobiStep", sample_red_grid_sync_ptx());
        let decision = orch.make_split_decision(&launch, &classification);

        assert!(!decision.should_split);
        assert_eq!(decision.decision_reason, SplitDecisionReason::RedClassification);
    }

    #[test]
    fn test_orchestrator_no_split_single_gpu() {
        let orch = make_orchestrator(1); // Only 1 GPU
        let launch = make_launch(1, 256);
        let classification = orch.classifier.get_or_classify(1, "addKernel", sample_green_ptx());
        let decision = orch.make_split_decision(&launch, &classification);

        assert!(!decision.should_split);
        assert_eq!(decision.decision_reason, SplitDecisionReason::SingleGpu);
    }

    #[test]
    fn test_orchestrator_no_split_small_grid() {
        let orch = make_orchestrator(2);
        let launch = make_launch(1, 4); // Only 4 blocks, below threshold of 64
        let classification = orch.classifier.get_or_classify(1, "addKernel", sample_green_ptx());
        let decision = orch.make_split_decision(&launch, &classification);

        assert!(!decision.should_split);
        assert!(matches!(
            decision.decision_reason,
            SplitDecisionReason::GridTooSmall { .. }
        ));
    }

    #[test]
    fn test_orchestrator_handle_launch() {
        let orch = make_orchestrator(2);
        let launch = make_launch(1, 256);
        let decision = orch.handle_launch(&launch, "addKernel", sample_green_ptx());

        assert!(decision.should_split);
        assert_eq!(orch.stats.total_launches.load(Ordering::Relaxed), 1);
        assert_eq!(orch.stats.split_launches.load(Ordering::Relaxed), 1);
    }

    // =====================================================================
    // Compute partition tests
    // =====================================================================

    #[test]
    fn test_compute_partition_equal_gpus() {
        let orch = make_orchestrator(2);
        let grid = Dim3::one_d(256);
        let partition = orch.compute_partition(&grid, &orch.gpu_targets, &[]);

        assert_eq!(partition.assignments.len(), 2);
        assert_eq!(partition.split_dimension, SplitDimension::X);

        let total: u32 = partition
            .assignments
            .iter()
            .map(|a| a.grid_dim.x)
            .sum();
        assert_eq!(total, 256);

        // Equal GPUs should get equal shares
        assert_eq!(partition.assignments[0].grid_dim.x, 128);
        assert_eq!(partition.assignments[1].grid_dim.x, 128);
        assert_eq!(partition.assignments[0].block_offset.x, 0);
        assert_eq!(partition.assignments[1].block_offset.x, 128);
    }

    #[test]
    fn test_compute_partition_heterogeneous_gpus() {
        let classifier = Arc::new(KernelClassifier::new());
        let transformer = Arc::new(PtxTransformer::new(false));
        let targets = vec![
            make_gpu_target(0, 35.0, 82),  // ~60% of total score
            make_gpu_target(1, 23.0, 48),  // ~40% of total score
        ];
        let orch = SplitLaunchOrchestrator::new(classifier, transformer, targets.clone());

        let grid = Dim3::one_d(100);
        let partition = orch.compute_partition(&grid, &targets, &[]);

        // Faster GPU should get more blocks
        assert!(partition.assignments[0].block_count > partition.assignments[1].block_count);
        let total: u32 = partition.assignments.iter().map(|a| a.block_count).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_compute_partition_2d_grid() {
        let orch = make_orchestrator(2);
        let grid = Dim3::new(256, 128, 1);
        let partition = orch.compute_partition(&grid, &orch.gpu_targets, &[]);

        assert_eq!(partition.split_dimension, SplitDimension::X);
        // Y dimension should be unchanged on both GPUs
        assert_eq!(partition.assignments[0].grid_dim.y, 128);
        assert_eq!(partition.assignments[1].grid_dim.y, 128);
    }

    #[test]
    fn test_compute_partition_3d_grid_splits_largest() {
        let orch = make_orchestrator(2);
        let grid = Dim3::new(8, 8, 256);
        let partition = orch.compute_partition(&grid, &orch.gpu_targets, &[]);

        assert_eq!(partition.split_dimension, SplitDimension::Z);
        assert_eq!(partition.assignments[0].grid_dim.x, 8);
        assert_eq!(partition.assignments[0].grid_dim.y, 8);
    }

    #[test]
    fn test_compute_partition_odd_block_count() {
        let orch = make_orchestrator(2);
        let grid = Dim3::one_d(101); // Odd number
        let partition = orch.compute_partition(&grid, &orch.gpu_targets, &[]);

        let total: u32 = partition.assignments.iter().map(|a| a.grid_dim.x).sum();
        assert_eq!(total, 101); // Must cover all blocks
    }

    #[test]
    fn test_compute_partition_three_gpus() {
        let classifier = Arc::new(KernelClassifier::new());
        let transformer = Arc::new(PtxTransformer::new(false));
        let targets = vec![
            make_gpu_target(0, 10.0, 82),
            make_gpu_target(1, 10.0, 82),
            make_gpu_target(2, 10.0, 82),
        ];
        let orch = SplitLaunchOrchestrator::new(classifier, transformer, targets.clone());

        let grid = Dim3::one_d(300);
        let partition = orch.compute_partition(&grid, &targets, &[]);

        assert_eq!(partition.assignments.len(), 3);
        let total: u32 = partition.assignments.iter().map(|a| a.block_count).sum();
        assert_eq!(total, 300);
    }

    // =====================================================================
    // SplitCostModel tests
    // =====================================================================

    #[test]
    fn test_cost_model_default() {
        let model = SplitCostModel::new();
        assert!((model.split_overhead_us - 10.0).abs() < f64::EPSILON);
        assert!((model.efficiency_factor - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_model_no_speedup_single_gpu() {
        let model = SplitCostModel::new();
        assert!((model.estimate_speedup(100.0, 1, 0.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_model_speedup_two_gpus() {
        let model = SplitCostModel::new();
        // 1000us kernel, 2 GPUs, no merge
        let speedup = model.estimate_speedup(1000.0, 2, 0.0);
        assert!(speedup > 1.0, "Speedup should be > 1.0 for large kernel: {}", speedup);
        assert!(speedup < 2.0, "Speedup should be < 2.0: {}", speedup);
    }

    #[test]
    fn test_cost_model_speedup_decreases_with_merge() {
        let model = SplitCostModel::new();
        let speedup_no_merge = model.estimate_speedup(1000.0, 2, 0.0);
        let speedup_with_merge = model.estimate_speedup(1000.0, 2, 100.0);
        assert!(speedup_with_merge < speedup_no_merge);
    }

    #[test]
    fn test_cost_model_should_split_short_kernel() {
        let model = SplitCostModel::new();
        assert!(!model.should_split(5.0, 2, 0.0)); // 5us kernel too short
    }

    #[test]
    fn test_cost_model_should_split_large_kernel() {
        let model = SplitCostModel::new();
        assert!(model.should_split(1000.0, 2, 0.0)); // 1ms kernel should split
    }

    #[test]
    fn test_cost_model_optimal_gpu_count() {
        let model = SplitCostModel::new();
        let (count, speedup) = model.optimal_gpu_count(1000.0, 4, 0.0);
        assert!(count >= 2, "Should use at least 2 GPUs for 1ms kernel");
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_cost_model_zero_duration() {
        let model = SplitCostModel::new();
        assert!((model.estimate_speedup(0.0, 2, 0.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_model_diminishing_returns() {
        let model = SplitCostModel::new();
        let s2 = model.estimate_speedup(500.0, 2, 0.0);
        let s4 = model.estimate_speedup(500.0, 4, 0.0);
        // Speedup should increase but with diminishing returns
        assert!(s4 > s2);
        assert!(s4 < s2 * 2.0); // Not double the speedup
    }

    // =====================================================================
    // PhaseAGate tests
    // =====================================================================

    #[test]
    fn test_phase_a_gate_go() {
        let gate = PhaseAGate::new();
        let result = gate.evaluate(0.25, 0.35, 0.45, 0.80);
        assert!(result.is_go());
    }

    #[test]
    fn test_phase_a_gate_no_go_low_green() {
        let gate = PhaseAGate::new();
        let result = gate.evaluate(0.10, 0.35, 0.45, 0.80);
        assert!(!result.is_go());
        if let GateResult::NoGo { reasons } = &result {
            assert!(reasons[0].contains("GREEN fraction"));
        }
    }

    #[test]
    fn test_phase_a_gate_no_go_low_ptx() {
        let gate = PhaseAGate::new();
        let result = gate.evaluate(0.25, 0.35, 0.45, 0.50);
        assert!(!result.is_go());
    }

    #[test]
    fn test_phase_a_gate_no_go_multiple_failures() {
        let gate = PhaseAGate::new();
        let result = gate.evaluate(0.05, 0.10, 0.15, 0.30);
        if let GateResult::NoGo { reasons } = &result {
            assert_eq!(reasons.len(), 4);
        } else {
            panic!("Expected NoGo");
        }
    }

    // =====================================================================
    // PhaseBGate tests
    // =====================================================================

    #[test]
    fn test_phase_b_gate_go() {
        let gate = PhaseBGate::new();
        let result = gate.evaluate(1.8, 1.5, 8.0, 0.05, false);
        assert!(result.is_go());
    }

    #[test]
    fn test_phase_b_gate_no_go_correctness() {
        let gate = PhaseBGate::new();
        let result = gate.evaluate(1.8, 1.5, 8.0, 0.05, true);
        assert!(!result.is_go());
        if let GateResult::NoGo { reasons } = &result {
            assert!(reasons[0].contains("Correctness"));
        }
    }

    #[test]
    fn test_phase_b_gate_no_go_low_speedup() {
        let gate = PhaseBGate::new();
        let result = gate.evaluate(1.1, 1.0, 8.0, 0.05, false);
        assert!(!result.is_go());
    }

    #[test]
    fn test_phase_b_gate_no_go_high_overhead() {
        let gate = PhaseBGate::new();
        let result = gate.evaluate(1.8, 1.5, 50.0, 0.05, false);
        assert!(!result.is_go());
    }

    // =====================================================================
    // KernelArg tests
    // =====================================================================

    #[test]
    fn test_kernel_arg_variants() {
        let ptr = KernelArg::Pointer {
            addr: 0x1000,
            size: 4096,
        };
        let int = KernelArg::Int(42);
        let float = KernelArg::Float(3.14);
        let raw = KernelArg::Raw(vec![0u8; 8]);
        let offset = KernelArg::BlockOffset(128);

        assert_ne!(ptr, int);
        assert_ne!(int, float);
        assert_ne!(float, raw);
        assert_ne!(raw, offset);
    }

    // =====================================================================
    // KernelClassification tests
    // =====================================================================

    #[test]
    fn test_kernel_classification_is_transformed() {
        let parser = PtxParser::new();
        let analysis = parser.analyze_ptx(sample_green_ptx()).unwrap();
        let kc = KernelClassification::new(
            1,
            "test".to_string(),
            KernelColor::Green,
            ClassificationReason::FullyIndependent,
            analysis,
        );
        assert!(!kc.is_transformed());
        kc.transformed.store(true, Ordering::Release);
        assert!(kc.is_transformed());
    }

    // =====================================================================
    // Integration-style tests
    // =====================================================================

    #[test]
    fn test_full_pipeline_green_kernel() {
        // Simulate full pipeline: classify -> decide -> transform
        let classifier = KernelClassifier::new();
        let transformer = PtxTransformer::new(false);

        // Step 1: Classify
        let classification = classifier.classify_kernel(1, "addKernel", sample_green_ptx());
        assert_eq!(classification.color, KernelColor::Green);

        // Step 2: Transform
        let result = transformer
            .transform(sample_green_ptx(), &classification.analysis)
            .unwrap();
        assert!(result.offset_params.x.is_some());
        assert!(result.redirected_atomics.is_empty());

        // Step 3: Verify transformed PTX has the offset
        assert!(result.ptx_source.contains("__outerlink_blkoff_x"));
    }

    #[test]
    fn test_full_pipeline_yellow_kernel() {
        let classifier = KernelClassifier::new();
        let transformer = PtxTransformer::new(false);

        let classification = classifier.classify_kernel(2, "reduceKernel", sample_yellow_ptx());
        assert_eq!(classification.color, KernelColor::Yellow);

        let result = transformer
            .transform(sample_yellow_ptx(), &classification.analysis)
            .unwrap();
        assert!(!result.redirected_atomics.is_empty());

        // Verify merge plan can be built from redirected atomics
        let combine = CombineStrategy::from_atomic_op(result.redirected_atomics[0].op_type);
        assert_eq!(combine, CombineStrategy::Sum);
    }

    #[test]
    fn test_orchestrator_end_to_end() {
        let orch = make_orchestrator(2);

        // Launch 1: GREEN kernel with big grid -> should split
        let launch1 = make_launch(1, 512);
        let d1 = orch.handle_launch(&launch1, "addKernel", sample_green_ptx());
        assert!(d1.should_split);

        // Launch 2: RED kernel -> should not split
        let launch2 = make_launch(2, 512);
        let d2 = orch.handle_launch(&launch2, "jacobiStep", sample_red_grid_sync_ptx());
        assert!(!d2.should_split);

        // Launch 3: GREEN kernel with tiny grid -> should not split
        let launch3 = make_launch(3, 8);
        let d3 = orch.handle_launch(&launch3, "addKernel2", sample_green_ptx());
        assert!(!d3.should_split);

        // Verify stats
        assert_eq!(orch.stats.total_launches.load(Ordering::Relaxed), 3);
        assert_eq!(orch.stats.split_launches.load(Ordering::Relaxed), 1);
        assert_eq!(orch.stats.single_launches.load(Ordering::Relaxed), 2);
    }
}
