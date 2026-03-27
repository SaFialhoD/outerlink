//! Heterogeneous GPU Mixing (R23) — capability profiling, scoring, and scheduling.
//!
//! Enables OuterLink pools with mixed GPU models (e.g., RTX 3090 + RTX 4090)
//! to schedule work based on per-GPU capability scores. Implements the Gavel-style
//! three-tier hybrid normalization: static specs, calibration benchmarks, and
//! runtime adaptation via exponential moving averages.
//!
//! # Architecture
//!
//! 1. At registration, each GPU is profiled (hardware attributes + calibration benchmarks).
//! 2. Scores are normalized against a reference GPU (RTX 3060 = 1.0).
//! 3. A single GEU (GPU Equivalent Unit) number summarizes each GPU for fairness/quota.
//! 4. Per-workload-class scores let the scheduler pick the best GPU for each kernel.
//! 5. Runtime adaptation adjusts scores as thermal throttling, contention, etc. occur.
//!
//! # Integration Points
//!
//! - R13 (CUDA Graph Interception): HEFT partitioner uses `capability_score()` for cost estimation.
//! - R17 (Topology Scheduling): `GpuCapabilityProvider` trait feeds placement decisions.
//! - R20 (NCCL Backend): `AsymmetricBandwidthProvider` informs ring/tree topology construction.
//! - R10 (Memory Hierarchy): `GpuBandwidthProvider` guides tier placement.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a GPU in the OuterLink pool.
pub type GpuId = u32;

/// Device identifier within the OuterLink virtual device space.
/// Maps to CUDA device ordinals as seen by the application.
pub type DeviceId = u32;

/// Virtual address in the unified GPU address space.
pub type VirtualAddr = u64;

// ---------------------------------------------------------------------------
// Workload classification (shared with R13)
// ---------------------------------------------------------------------------

/// Classification of a kernel's bottleneck for scheduling purposes.
/// Used by both R23 (scoring) and R13 (HEFT cost estimation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    /// ALU throughput dominant (large grid, small data per thread).
    ComputeBound,
    /// Memory bandwidth dominant (small grid, large data per thread).
    MemoryBound,
    /// Tensor Core dominant (matched known matmul/conv patterns).
    TensorBound,
    /// Unknown -- use conservative scoring.
    Unknown,
}

// ---------------------------------------------------------------------------
// GPU Profile
// ---------------------------------------------------------------------------

/// Complete hardware profile for a GPU in the OuterLink pool.
/// Built at GPU registration time by querying CUDA Driver API and NVML.
#[derive(Debug, Clone)]
pub struct GpuProfile {
    /// OuterLink-assigned GPU identifier (unique across the pool).
    pub gpu_id: GpuId,
    /// Node (PC) this GPU belongs to.
    pub node_id: NodeId,

    // --- Static hardware attributes (set once at registration) ---

    /// GPU model name (e.g., "NVIDIA GeForce RTX 3090").
    pub name: String,
    /// CUDA compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Number of Streaming Multiprocessors.
    pub sm_count: u32,
    /// Total CUDA cores (sm_count * cores_per_sm, arch-dependent).
    pub cuda_cores: u32,
    /// Tensor Core count (0 for pre-Volta).
    pub tensor_core_count: u32,
    /// Tensor Core generation (None for pre-Volta, Some(2) for Turing, etc.).
    pub tensor_core_gen: Option<u32>,
    /// Total VRAM in bytes.
    pub vram_total_bytes: u64,
    /// Theoretical memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Theoretical FP32 peak TFLOPS.
    pub fp32_tflops: f64,
    /// Theoretical FP16 peak TFLOPS (with Tensor Cores if available).
    pub fp16_tflops: f64,
    /// BAR1 aperture size in bytes (256MB default, VRAM size with ReBAR).
    pub bar1_size_bytes: u64,
    /// PCIe generation (3, 4, 5).
    pub pcie_gen: u32,
    /// PCIe link width (x8, x16).
    pub pcie_width: u32,
    /// Measured PCIe bandwidth in GB/s (from calibration benchmark).
    pub pcie_bandwidth_gbps: f64,
    /// Number of async copy engines.
    pub async_engine_count: u32,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: u32,
    /// GPU boost clock in MHz.
    pub boost_clock_mhz: u32,
    /// TDP (thermal design power) in watts.
    pub tdp_watts: u32,

    // --- Precision support flags ---

    pub supports_fp16: bool,
    /// BF16 support (CC 8.0+).
    pub supports_bf16: bool,
    /// TF32 support (CC 8.0+).
    pub supports_tf32: bool,
    /// FP8 support (CC 8.9+).
    pub supports_fp8: bool,
    /// FP4 support (CC 10.0+).
    pub supports_fp4: bool,
    /// INT8 support (CC 7.5+).
    pub supports_int8: bool,

    // --- Driver info ---

    /// NVIDIA driver version string.
    pub driver_version: String,
    /// CUDA driver API version (e.g., 12080 for CUDA 12.8).
    pub cuda_driver_version: u32,
    /// Maximum CUDA toolkit version this driver supports.
    pub max_cuda_toolkit: (u32, u32),

    // --- Calibration benchmarks (set during registration) ---

    /// Measured FP32 GFLOPS (from SGEMM benchmark).
    pub measured_fp32_gflops: f64,
    /// Measured memory bandwidth in GB/s (from stream copy benchmark).
    pub measured_memory_bw_gbps: f64,
    /// Measured host-to-device bandwidth in GB/s.
    pub measured_h2d_bw_gbps: f64,
    /// Measured device-to-host bandwidth in GB/s.
    pub measured_d2h_bw_gbps: f64,
    /// Measured Tensor Core throughput in TFLOPS (FP16 matmul).
    pub measured_tensor_tflops: Option<f64>,

    // --- Dynamic state (updated periodically via NVML) ---

    /// Current free VRAM in bytes.
    pub vram_free_bytes: u64,
    /// Current GPU utilization (0.0 - 1.0).
    pub utilization: f64,
    /// Current GPU temperature in Celsius.
    pub temperature_c: u32,
    /// Current clock speed in MHz (may be lower than boost due to throttling).
    pub current_clock_mhz: u32,
    /// Current power draw in watts.
    pub power_draw_watts: u32,
    /// Whether the GPU is thermally throttling.
    pub is_throttling: bool,

    // --- Computed scores (updated when dynamic state changes) ---

    /// Normalized capability scores per workload class.
    pub capability_scores: WorkloadScores,
    /// GPU Equivalent Units (single number for fairness/quota).
    pub geu: f64,
    /// ReBAR status: true if BAR1 >= VRAM (full mapping available).
    pub has_rebar: bool,
}

/// Per-workload-class normalized scores.
/// Each score is relative to the reference GPU (RTX 3060 = 1.0).
#[derive(Debug, Clone, Default)]
pub struct WorkloadScores {
    /// FP32 throughput normalized.
    pub compute: f64,
    /// Memory bandwidth normalized.
    pub memory: f64,
    /// Tensor Core throughput normalized.
    pub tensor: f64,
    /// VRAM capacity normalized.
    pub capacity: f64,
    /// PCIe/host transfer normalized.
    pub transfer: f64,
    /// Overall score (weighted combination, used by default).
    pub overall: f64,
}

/// Minimum compute capability for OuterLink pool membership.
pub const MIN_COMPUTE_CAPABILITY: (u32, u32) = (7, 5);

/// Default drain timeout in seconds when removing a GPU.
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 30;

/// Maximum calibration cache age before re-running benchmarks.
pub const CALIBRATION_CACHE_MAX_AGE_DAYS: u64 = 90;

/// Number of tensor cores in the reference GPU (RTX 3060).
pub const REFERENCE_TENSOR_CORE_COUNT: u32 = 112;

// ---------------------------------------------------------------------------
// Capability Scorer
// ---------------------------------------------------------------------------

/// Reference values for normalization (RTX 3060 = 1.0).
#[derive(Debug, Clone)]
pub struct ReferenceValues {
    /// RTX 3060 FP32 peak TFLOPS.
    pub fp32_tflops: f64,
    /// RTX 3060 memory bandwidth in GB/s.
    pub memory_bw_gbps: f64,
    /// RTX 3060 Tensor Core FP16 TFLOPS.
    pub tensor_tflops_fp16: f64,
    /// RTX 3060 VRAM in GB.
    pub vram_gb: f64,
    /// RTX 3060 effective PCIe bandwidth in GB/s.
    pub pcie_bw_gbps: f64,
}

impl Default for ReferenceValues {
    fn default() -> Self {
        Self {
            fp32_tflops: 12.7,
            memory_bw_gbps: 360.0,
            tensor_tflops_fp16: 50.6,
            vram_gb: 12.0,
            pcie_bw_gbps: 16.0,
        }
    }
}

/// Weights for GEU computation.
#[derive(Debug, Clone)]
pub struct GeuWeights {
    /// Compute weight (default 0.4).
    pub compute: f64,
    /// Bandwidth weight (default 0.3).
    pub bandwidth: f64,
    /// Capacity weight (default 0.3).
    pub capacity: f64,
}

impl Default for GeuWeights {
    fn default() -> Self {
        Self {
            compute: 0.4,
            bandwidth: 0.3,
            capacity: 0.3,
        }
    }
}

/// Weights used to combine individual scores into overall score.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub compute: f64,
    pub memory: f64,
    pub tensor: f64,
    pub capacity: f64,
    pub transfer: f64,
}

/// Per-workload-class weight overrides for the overall score.
#[derive(Debug, Clone)]
pub struct WorkloadWeightConfig {
    /// Weights when workload is compute-bound.
    pub compute_bound: ScoringWeights,
    /// Weights when workload is memory-bound.
    pub memory_bound: ScoringWeights,
    /// Weights when workload is tensor-bound.
    pub tensor_bound: ScoringWeights,
    /// Weights when workload class is unknown.
    pub default: ScoringWeights,
}

impl Default for WorkloadWeightConfig {
    fn default() -> Self {
        Self {
            compute_bound: ScoringWeights {
                compute: 0.50,
                memory: 0.15,
                tensor: 0.15,
                capacity: 0.10,
                transfer: 0.10,
            },
            memory_bound: ScoringWeights {
                compute: 0.10,
                memory: 0.50,
                tensor: 0.10,
                capacity: 0.20,
                transfer: 0.10,
            },
            tensor_bound: ScoringWeights {
                compute: 0.10,
                memory: 0.15,
                tensor: 0.55,
                capacity: 0.10,
                transfer: 0.10,
            },
            default: ScoringWeights {
                compute: 0.30,
                memory: 0.25,
                tensor: 0.20,
                capacity: 0.15,
                transfer: 0.10,
            },
        }
    }
}

/// Cached kernel-to-GPU affinity entry from runtime observations.
#[derive(Debug, Clone)]
pub struct AffinityEntry {
    /// GPU that ran this kernel fastest.
    pub best_gpu: GpuId,
    /// Exponential moving average of execution time.
    pub avg_execution_ns: u64,
    /// Number of observations.
    pub sample_count: u32,
    /// When this entry was last updated.
    pub last_updated: Instant,
}

/// Computes normalized capability scores for GPUs.
/// Implements the three-tier hybrid normalization approach.
pub struct CapabilityScorer {
    /// Reference GPU values (RTX 3060 as baseline = 1.0).
    pub reference: ReferenceValues,
    /// GEU weight configuration.
    pub geu_weights: GeuWeights,
    /// Workload-class-specific weights for the overall score.
    pub workload_weights: WorkloadWeightConfig,
    /// Kernel-to-GPU affinity cache (key: kernel function pointer hash).
    affinity_cache: HashMap<u64, AffinityEntry>,
}

impl CapabilityScorer {
    /// Create a new scorer with default reference values and weights.
    pub fn new() -> Self {
        Self {
            reference: ReferenceValues::default(),
            geu_weights: GeuWeights::default(),
            workload_weights: WorkloadWeightConfig::default(),
            affinity_cache: HashMap::new(),
        }
    }

    /// Create a scorer with custom configuration.
    pub fn with_config(
        reference: ReferenceValues,
        geu_weights: GeuWeights,
        workload_weights: WorkloadWeightConfig,
    ) -> Self {
        Self {
            reference,
            geu_weights,
            workload_weights,
            affinity_cache: HashMap::new(),
        }
    }

    /// Compute all capability scores for a GPU profile.
    /// Called after calibration benchmarks complete.
    pub fn compute_scores(&self, profile: &mut GpuProfile) {
        let ref_vals = &self.reference;

        // Tier 1 or Tier 2 scores depending on calibration availability
        let use_measured = profile.measured_fp32_gflops > 0.0;

        profile.capability_scores.compute = if use_measured {
            (profile.measured_fp32_gflops / 1000.0) / ref_vals.fp32_tflops
        } else {
            profile.fp32_tflops / ref_vals.fp32_tflops
        };

        profile.capability_scores.memory = if use_measured {
            profile.measured_memory_bw_gbps / ref_vals.memory_bw_gbps
        } else {
            profile.memory_bandwidth_gbps / ref_vals.memory_bw_gbps
        };

        profile.capability_scores.tensor = match profile.measured_tensor_tflops {
            Some(measured) => measured / ref_vals.tensor_tflops_fp16,
            None => {
                // Estimate from Tensor Core gen and count
                match profile.tensor_core_gen {
                    Some(gen) => {
                        let gen_multiplier = match gen {
                            2 => 1.0,  // Turing baseline
                            3 => 2.0,  // Ampere (TF32, higher throughput)
                            4 => 3.5,  // Ada (FP8, higher throughput)
                            5 => 5.0,  // Blackwell (FP4/FP6)
                            _ => 1.0,
                        };
                        let core_ratio =
                            profile.tensor_core_count as f64 / REFERENCE_TENSOR_CORE_COUNT as f64;
                        core_ratio * gen_multiplier
                    }
                    None => 0.0, // No Tensor Cores
                }
            }
        };

        profile.capability_scores.capacity =
            (profile.vram_total_bytes as f64 / 1e9) / ref_vals.vram_gb;

        profile.capability_scores.transfer = if use_measured {
            profile.measured_h2d_bw_gbps / ref_vals.pcie_bw_gbps
        } else {
            profile.pcie_bandwidth_gbps / ref_vals.pcie_bw_gbps
        };

        // Overall score uses default weights
        let w = &self.workload_weights.default;
        profile.capability_scores.overall = w.compute * profile.capability_scores.compute
            + w.memory * profile.capability_scores.memory
            + w.tensor * profile.capability_scores.tensor
            + w.capacity * profile.capability_scores.capacity
            + w.transfer * profile.capability_scores.transfer;

        // GEU
        let gw = &self.geu_weights;
        profile.geu = gw.compute * profile.capability_scores.compute
            + gw.bandwidth * profile.capability_scores.memory
            + gw.capacity * profile.capability_scores.capacity;

        // ReBAR detection
        profile.has_rebar = profile.bar1_size_bytes >= profile.vram_total_bytes;
    }

    /// Get workload-specific capability score for a GPU.
    /// Used by R13's HEFT partitioner and R17's placement decision.
    pub fn capability_score(&self, profile: &GpuProfile, workload_class: WorkloadClass) -> f64 {
        let w = match workload_class {
            WorkloadClass::ComputeBound => &self.workload_weights.compute_bound,
            WorkloadClass::MemoryBound => &self.workload_weights.memory_bound,
            WorkloadClass::TensorBound => &self.workload_weights.tensor_bound,
            WorkloadClass::Unknown => &self.workload_weights.default,
        };

        w.compute * profile.capability_scores.compute
            + w.memory * profile.capability_scores.memory
            + w.tensor * profile.capability_scores.tensor
            + w.capacity * profile.capability_scores.capacity
            + w.transfer * profile.capability_scores.transfer
    }

    /// Apply Tier 3 runtime adaptation based on observed kernel execution.
    /// Uses exponential moving average (alpha = 0.2) to track per-kernel affinity.
    pub fn adapt_from_observation(
        &mut self,
        gpu_id: GpuId,
        kernel_hash: u64,
        observed_ns: u64,
        workload_class: WorkloadClass,
    ) {
        // Key the affinity cache by (kernel_hash, workload_class) so that
        // per-class score adjustments work correctly. Different workload
        // classes on the same kernel can have different optimal GPUs.
        let cache_key = kernel_hash ^ (workload_class as u64).wrapping_mul(0x9E3779B97F4A7C15);

        let entry = self
            .affinity_cache
            .entry(cache_key)
            .or_insert(AffinityEntry {
                best_gpu: gpu_id,
                avg_execution_ns: observed_ns,
                sample_count: 0,
                last_updated: Instant::now(),
            });

        // Compare against the prior average BEFORE incorporating the new sample,
        // so that the best_gpu decision isn't biased by the sample itself.
        let prior_avg = entry.avg_execution_ns;
        if observed_ns < prior_avg {
            entry.best_gpu = gpu_id;
        }

        // Exponential moving average (alpha = 0.2)
        let alpha = 0.2;
        entry.avg_execution_ns =
            ((1.0 - alpha) * prior_avg as f64 + alpha * observed_ns as f64) as u64;
        entry.sample_count += 1;
        entry.last_updated = Instant::now();
    }

    /// Get affinity entry for a kernel hash + workload class, if any observations exist.
    ///
    /// The key must match the (hash, class) pair used in `adapt_from_observation`.
    pub fn get_affinity(&self, kernel_hash: u64, workload_class: WorkloadClass) -> Option<&AffinityEntry> {
        let cache_key = kernel_hash ^ (workload_class as u64).wrapping_mul(0x9E3779B97F4A7C15);
        self.affinity_cache.get(&cache_key)
    }

    /// Number of entries in the affinity cache.
    pub fn affinity_cache_len(&self) -> usize {
        self.affinity_cache.len()
    }
}

impl Default for CapabilityScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Binary Compatibility Checker
// ---------------------------------------------------------------------------

/// Result of checking binary compatibility for a kernel on a target GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatResult {
    /// Exact native binary available -- best performance, no JIT overhead.
    NativeCubin,
    /// Compatible cubin from same major version -- good performance.
    CompatibleCubin,
    /// PTX available for JIT compilation -- may have first-launch latency.
    PtxJit,
    /// No compatible binary -- GPU cannot run this kernel.
    Incompatible,
    /// Module not yet analyzed.
    Unknown,
}

/// JIT compilation status for a (module, compute_capability) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitStatus {
    /// JIT compilation not attempted.
    NotAttempted,
    /// JIT compilation in progress (async).
    InProgress,
    /// JIT compilation succeeded and is cached by the CUDA driver.
    Cached,
    /// JIT compilation failed (incompatible).
    Failed,
}

/// Opaque handle for a CUDA module.
pub type CUmodule = u64;

/// Extracted binary information from a CUDA module's fatbin.
#[derive(Debug, Clone)]
pub struct ModuleBinaryInfo {
    /// Available cubin targets (CC major.minor -> cubin size in bytes).
    pub cubins: HashMap<(u32, u32), usize>,
    /// PTX targets available (CC major.minor -> PTX present).
    pub ptx_targets: HashMap<(u32, u32), bool>,
    /// Highest PTX target version available.
    pub max_ptx_target: Option<(u32, u32)>,
    /// Whether any PTX uses architecture-conditional features (*a suffix).
    pub has_arch_conditional_ptx: bool,
}

/// Checks whether a kernel binary can execute on a target GPU.
/// Inspects fatbin contents extracted during cuModuleLoad interception.
pub struct BinaryCompatibilityChecker {
    /// Per-module extracted binaries.
    module_binaries: HashMap<CUmodule, ModuleBinaryInfo>,
    /// JIT compilation cache status per (module, cc).
    jit_cache: HashMap<(CUmodule, (u32, u32)), JitStatus>,
}

impl BinaryCompatibilityChecker {
    /// Create a new empty checker.
    pub fn new() -> Self {
        Self {
            module_binaries: HashMap::new(),
            jit_cache: HashMap::new(),
        }
    }

    /// Register a module's binary information (called during cuModuleLoad interception).
    pub fn register_module(&mut self, module: CUmodule, info: ModuleBinaryInfo) {
        self.module_binaries.insert(module, info);
    }

    /// Unregister a module (called during cuModuleUnload).
    pub fn unregister_module(&mut self, module: CUmodule) {
        self.module_binaries.remove(&module);
        self.jit_cache.retain(|&(m, _), _| m != module);
    }

    /// Check if a kernel from this module can run on a GPU.
    pub fn is_compatible(&self, module: CUmodule, gpu: &GpuProfile) -> CompatResult {
        let info = match self.module_binaries.get(&module) {
            Some(info) => info,
            None => return CompatResult::Unknown,
        };

        let target_cc = gpu.compute_capability;

        // 1. Exact cubin match
        if info.cubins.contains_key(&target_cc) {
            return CompatResult::NativeCubin;
        }

        // 2. Compatible cubin (same major, lower minor)
        let compatible_cubin = info
            .cubins
            .keys()
            .filter(|cc| cc.0 == target_cc.0 && cc.1 <= target_cc.1)
            .max_by_key(|cc| cc.1);
        if compatible_cubin.is_some() {
            return CompatResult::CompatibleCubin;
        }

        // 3. PTX JIT (forward compatible if not arch-conditional)
        if let Some(max_ptx) = info.max_ptx_target {
            if max_ptx.0 <= target_cc.0 && !info.has_arch_conditional_ptx {
                return CompatResult::PtxJit;
            }
            // Arch-conditional PTX: only compatible within same major
            if info.has_arch_conditional_ptx && max_ptx.0 == target_cc.0 {
                return CompatResult::PtxJit;
            }
        }

        CompatResult::Incompatible
    }

    /// Get JIT status for a module on a specific compute capability.
    pub fn jit_status(&self, module: CUmodule, cc: (u32, u32)) -> JitStatus {
        self.jit_cache
            .get(&(module, cc))
            .copied()
            .unwrap_or(JitStatus::NotAttempted)
    }

    /// Update JIT status (called when async JIT completes).
    pub fn set_jit_status(&mut self, module: CUmodule, cc: (u32, u32), status: JitStatus) {
        self.jit_cache.insert((module, cc), status);
    }

    /// Number of registered modules.
    pub fn module_count(&self) -> usize {
        self.module_binaries.len()
    }
}

impl Default for BinaryCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Workload Classification
// ---------------------------------------------------------------------------

/// Kernel-specific metadata used for workload classification.
/// Mirrors R13's `KernelNodeInfo` fields needed for classification.
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Demangled kernel name (for pattern matching).
    pub kernel_name: String,
    /// Grid dimensions (blocks).
    pub grid_dim: [u32; 3],
    /// Block dimensions (threads).
    pub block_dim: [u32; 3],
    /// Shared memory bytes.
    pub shared_mem_bytes: u32,
    /// Minimum compute capability required.
    pub min_compute_capability: (u32, u32),
}

/// Classify a kernel's workload type from metadata available at dispatch time.
/// Uses heuristics: kernel name pattern matching, then grid geometry analysis.
pub fn classify_workload(kernel: &KernelInfo) -> WorkloadClass {
    let name = kernel.kernel_name.to_lowercase();

    // Tensor-bound patterns (matmul, conv, attention with Tensor Cores)
    if name.contains("gemm")
        || name.contains("cutlass")
        || name.contains("wmma")
        || name.contains("mma_")
        || name.contains("tensorop")
    {
        return WorkloadClass::TensorBound;
    }

    // Memory-bound patterns (copy, reduce, norm, softmax, elementwise)
    if name.contains("reduce")
        || name.contains("softmax")
        || name.contains("layernorm")
        || name.contains("batchnorm")
        || name.contains("elementwise")
        || name.contains("copy")
        || name.contains("transpose")
    {
        return WorkloadClass::MemoryBound;
    }

    // Compute-bound patterns (crypto, FFT, physics)
    if name.contains("fft")
        || name.contains("crypto")
        || name.contains("sha")
        || name.contains("physics")
        || name.contains("raytrace")
    {
        return WorkloadClass::ComputeBound;
    }

    // Heuristic 2: Grid geometry
    let grid_size = (kernel.grid_dim[0] as u64)
        * (kernel.grid_dim[1] as u64)
        * (kernel.grid_dim[2] as u64);
    let shared_mem = kernel.shared_mem_bytes;
    let threads_per_block = (kernel.block_dim[0] as u64)
        * (kernel.block_dim[1] as u64)
        * (kernel.block_dim[2] as u64);

    // Large grid + small shared memory -> likely memory-bound
    if grid_size > 10_000 && shared_mem < 1024 {
        return WorkloadClass::MemoryBound;
    }
    // Large shared memory + decent thread count -> likely compute-bound
    if shared_mem > 16_384 && threads_per_block >= 256 {
        return WorkloadClass::ComputeBound;
    }

    WorkloadClass::Unknown
}

/// Check if PTX source uses architecture-conditional features.
/// Architecture-conditional targets end with 'a' suffix (e.g., `.target sm_90a`).
pub fn detect_arch_conditional(ptx: &str) -> bool {
    ptx.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with(".target") && trimmed.ends_with('a')
    })
}

// ---------------------------------------------------------------------------
// Calibration Cache
// ---------------------------------------------------------------------------

/// Key for calibration result caching.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CalibrationKey {
    /// GPU model name (e.g., "NVIDIA GeForce RTX 3090").
    pub gpu_model: String,
    /// Driver version string (e.g., "570.86.15").
    pub driver_version: String,
}

/// Cached calibration benchmark results.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Measured FP32 GFLOPS.
    pub fp32_gflops: f64,
    /// Measured memory bandwidth in GB/s.
    pub memory_bw_gbps: f64,
    /// Measured host-to-device bandwidth in GB/s.
    pub h2d_bw_gbps: f64,
    /// Measured device-to-host bandwidth in GB/s.
    pub d2h_bw_gbps: f64,
    /// Measured Tensor Core throughput in TFLOPS (None if no Tensor Cores).
    pub tensor_tflops: Option<f64>,
    /// When these benchmarks were run.
    pub measured_at: SystemTime,
}

/// Calibration result cache. Keyed by (gpu_model, driver_version).
/// Cache entries older than 90 days are considered stale.
pub struct CalibrationCache {
    entries: HashMap<CalibrationKey, CalibrationResult>,
}

impl CalibrationCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a cached calibration result.
    /// Returns None if not found or if the entry is older than the max age.
    pub fn get(&self, key: &CalibrationKey) -> Option<&CalibrationResult> {
        let entry = self.entries.get(key)?;
        let age = entry
            .measured_at
            .elapsed()
            .unwrap_or(Duration::from_secs(u64::MAX));
        if age > Duration::from_secs(CALIBRATION_CACHE_MAX_AGE_DAYS * 24 * 3600) {
            return None;
        }
        Some(entry)
    }

    /// Store a calibration result.
    pub fn insert(&mut self, key: CalibrationKey, result: CalibrationResult) {
        self.entries.insert(key, result);
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CalibrationCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU Pool Manager
// ---------------------------------------------------------------------------

/// Statistics for the GPU pool.
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Total GPUs registered.
    pub total_gpus: AtomicU64,
    /// Total scoring operations performed.
    pub scoring_operations: AtomicU64,
    /// Total affinity cache hits.
    pub affinity_hits: AtomicU64,
    /// Total compatibility checks performed.
    pub compatibility_checks: AtomicU64,
    /// Total GPUs rejected at registration.
    pub registration_rejections: AtomicU64,
}

/// Configuration for the GPU pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// GEU weights override.
    pub geu_weights: GeuWeights,
    /// Minimum compute capability for pool membership.
    pub min_compute_capability: (u32, u32),
    /// Drain timeout when removing a GPU.
    pub drain_timeout: Duration,
    /// Whether to require same driver major version across pool.
    pub enforce_driver_version: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            geu_weights: GeuWeights::default(),
            min_compute_capability: MIN_COMPUTE_CAPABILITY,
            drain_timeout: Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS),
            enforce_driver_version: true,
        }
    }
}

/// Manages the heterogeneous GPU pool. Holds all GPU profiles, computes scores,
/// and provides the `GpuCapabilityProvider` interface for other modules.
pub struct GpuPoolManager {
    /// All registered GPU profiles.
    profiles: HashMap<GpuId, GpuProfile>,
    /// Capability scorer.
    scorer: CapabilityScorer,
    /// Binary compatibility checker.
    compat_checker: BinaryCompatibilityChecker,
    /// Pool configuration.
    config: PoolConfig,
    /// Pool statistics.
    stats: PoolStats,
    /// Baseline driver major version (set from first registered GPU).
    baseline_driver_major: Option<u32>,
}

/// Errors from GPU pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPoolError {
    /// GPU compute capability is below minimum.
    InsufficientComputeCapability {
        gpu_id: GpuId,
        cc: (u32, u32),
        min_cc: (u32, u32),
    },
    /// GPU driver version incompatible with pool baseline.
    DriverVersionMismatch {
        gpu_id: GpuId,
        gpu_driver_major: u32,
        pool_driver_major: u32,
    },
    /// GPU ID already registered.
    AlreadyRegistered(GpuId),
    /// GPU ID not found.
    NotFound(GpuId),
}

impl std::fmt::Display for GpuPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientComputeCapability { gpu_id, cc, min_cc } => write!(
                f,
                "GPU {} CC {}.{} is below minimum {}.{}",
                gpu_id, cc.0, cc.1, min_cc.0, min_cc.1
            ),
            Self::DriverVersionMismatch {
                gpu_id,
                gpu_driver_major,
                pool_driver_major,
            } => write!(
                f,
                "GPU {} driver major {} differs from pool baseline {}",
                gpu_id, gpu_driver_major, pool_driver_major
            ),
            Self::AlreadyRegistered(id) => write!(f, "GPU {} already registered", id),
            Self::NotFound(id) => write!(f, "GPU {} not found", id),
        }
    }
}

impl std::error::Error for GpuPoolError {}

impl GpuPoolManager {
    /// Create a new pool manager with default configuration.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            scorer: CapabilityScorer::new(),
            compat_checker: BinaryCompatibilityChecker::new(),
            config: PoolConfig::default(),
            stats: PoolStats::default(),
            baseline_driver_major: None,
        }
    }

    /// Create a pool manager with custom configuration.
    pub fn with_config(config: PoolConfig) -> Self {
        let scorer = CapabilityScorer::with_config(
            ReferenceValues::default(),
            config.geu_weights.clone(),
            WorkloadWeightConfig::default(),
        );
        Self {
            profiles: HashMap::new(),
            scorer,
            compat_checker: BinaryCompatibilityChecker::new(),
            config,
            stats: PoolStats::default(),
            baseline_driver_major: None,
        }
    }

    /// Register a GPU with the pool. Computes scores and validates constraints.
    pub fn register_gpu(&mut self, mut profile: GpuProfile) -> Result<(), GpuPoolError> {
        // Check minimum compute capability
        if profile.compute_capability < self.config.min_compute_capability {
            self.stats
                .registration_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(GpuPoolError::InsufficientComputeCapability {
                gpu_id: profile.gpu_id,
                cc: profile.compute_capability,
                min_cc: self.config.min_compute_capability,
            });
        }

        // Check driver version compatibility
        if self.config.enforce_driver_version {
            let gpu_driver_major = parse_driver_major(&profile.driver_version);
            match self.baseline_driver_major {
                Some(baseline) => {
                    if gpu_driver_major != baseline {
                        self.stats
                            .registration_rejections
                            .fetch_add(1, Ordering::Relaxed);
                        return Err(GpuPoolError::DriverVersionMismatch {
                            gpu_id: profile.gpu_id,
                            gpu_driver_major,
                            pool_driver_major: baseline,
                        });
                    }
                }
                None => {
                    self.baseline_driver_major = Some(gpu_driver_major);
                }
            }
        }

        // Check for duplicate
        if self.profiles.contains_key(&profile.gpu_id) {
            return Err(GpuPoolError::AlreadyRegistered(profile.gpu_id));
        }

        // Compute scores
        self.scorer.compute_scores(&mut profile);

        self.profiles.insert(profile.gpu_id, profile);
        self.stats.total_gpus.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Remove a GPU from the pool.
    pub fn remove_gpu(&mut self, gpu_id: GpuId) -> Result<GpuProfile, GpuPoolError> {
        let profile = self
            .profiles
            .remove(&gpu_id)
            .ok_or(GpuPoolError::NotFound(gpu_id))?;
        // Note: actual drain logic (waiting for in-flight kernels) is handled
        // by the server daemon before calling this.
        Ok(profile)
    }

    /// Get a GPU profile by ID.
    pub fn get_profile(&self, gpu_id: GpuId) -> Option<&GpuProfile> {
        self.profiles.get(&gpu_id)
    }

    /// Get all GPU profiles.
    pub fn all_profiles(&self) -> Vec<&GpuProfile> {
        self.profiles.values().collect()
    }

    /// Get workload-specific capability score for a GPU.
    pub fn capability_score(&self, gpu_id: GpuId, workload_class: WorkloadClass) -> Option<f64> {
        let profile = self.profiles.get(&gpu_id)?;
        self.stats
            .scoring_operations
            .fetch_add(1, Ordering::Relaxed);
        Some(self.scorer.capability_score(profile, workload_class))
    }

    /// Filter GPUs by hard constraints for a kernel.
    /// Returns only GPUs that can actually execute this kernel (CC check + binary compat).
    pub fn filter_compatible(
        &self,
        min_cc: (u32, u32),
        module: Option<CUmodule>,
        candidates: &[GpuId],
    ) -> Vec<GpuId> {
        self.stats
            .compatibility_checks
            .fetch_add(1, Ordering::Relaxed);
        candidates
            .iter()
            .filter(|&&gpu_id| {
                let profile = match self.profiles.get(&gpu_id) {
                    Some(p) => p,
                    None => return false,
                };
                // CC check
                if profile.compute_capability < min_cc {
                    return false;
                }
                // Binary compatibility check
                if let Some(m) = module {
                    let compat = self.compat_checker.is_compatible(m, profile);
                    if compat == CompatResult::Incompatible {
                        return false;
                    }
                }
                true
            })
            .copied()
            .collect()
    }

    /// Get GEU rating for a GPU.
    pub fn geu(&self, gpu_id: GpuId) -> Option<f64> {
        self.profiles.get(&gpu_id).map(|p| p.geu)
    }

    /// Get total GEU across all GPUs in the pool.
    pub fn total_geu(&self) -> f64 {
        self.profiles.values().map(|p| p.geu).sum()
    }

    /// Get GPUs ordered by overall capability score (descending).
    pub fn gpus_by_capability(&self) -> Vec<GpuId> {
        let mut gpus: Vec<_> = self.profiles.values().collect();
        gpus.sort_by(|a, b| {
            b.capability_scores
                .overall
                .partial_cmp(&a.capability_scores.overall)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gpus.iter().map(|p| p.gpu_id).collect()
    }

    /// Get memory bandwidth for a specific GPU in GB/s.
    pub fn memory_bandwidth_gbps(&self, gpu_id: GpuId) -> Option<f64> {
        self.profiles.get(&gpu_id).map(|p| {
            if p.measured_memory_bw_gbps > 0.0 {
                p.measured_memory_bw_gbps
            } else {
                p.memory_bandwidth_gbps
            }
        })
    }

    /// Check if a GPU has ReBAR enabled.
    pub fn has_rebar(&self, gpu_id: GpuId) -> Option<bool> {
        self.profiles.get(&gpu_id).map(|p| p.has_rebar)
    }

    /// Number of GPUs in the pool.
    pub fn gpu_count(&self) -> usize {
        self.profiles.len()
    }

    /// Access the binary compatibility checker.
    pub fn compat_checker(&self) -> &BinaryCompatibilityChecker {
        &self.compat_checker
    }

    /// Access the binary compatibility checker mutably.
    pub fn compat_checker_mut(&mut self) -> &mut BinaryCompatibilityChecker {
        &mut self.compat_checker
    }

    /// Access the capability scorer.
    pub fn scorer(&self) -> &CapabilityScorer {
        &self.scorer
    }

    /// Access the capability scorer mutably.
    pub fn scorer_mut(&mut self) -> &mut CapabilityScorer {
        &mut self.scorer
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
}

impl Default for GpuPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse the major version number from a driver version string like "570.86.15".
fn parse_driver_major(version: &str) -> u32 {
    version
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Create a test GPU profile with the given parameters.
/// Useful for tests and prototyping.
pub fn make_test_profile(
    gpu_id: GpuId,
    name: &str,
    cc: (u32, u32),
    vram_gb: f64,
    fp32_tflops: f64,
    mem_bw_gbps: f64,
) -> GpuProfile {
    GpuProfile {
        gpu_id,
        node_id: 0,
        name: name.to_string(),
        compute_capability: cc,
        sm_count: 0,
        cuda_cores: 0,
        tensor_core_count: 0,
        tensor_core_gen: if cc.0 >= 7 { Some(cc.0 - 5) } else { None },
        vram_total_bytes: (vram_gb * 1e9) as u64,
        memory_bandwidth_gbps: mem_bw_gbps,
        fp32_tflops,
        fp16_tflops: fp32_tflops * 2.0,
        bar1_size_bytes: 256 * 1024 * 1024, // 256MB default (no ReBAR)
        pcie_gen: 4,
        pcie_width: 16,
        pcie_bandwidth_gbps: 16.0,
        async_engine_count: 2,
        l2_cache_bytes: 4 * 1024 * 1024,
        boost_clock_mhz: 1800,
        tdp_watts: 350,
        supports_fp16: true,
        supports_bf16: cc.0 >= 8,
        supports_tf32: cc.0 >= 8,
        supports_fp8: cc >= (8, 9),
        supports_fp4: cc.0 >= 10,
        supports_int8: true,
        driver_version: "570.86.15".to_string(),
        cuda_driver_version: 12080,
        max_cuda_toolkit: (12, 8),
        measured_fp32_gflops: 0.0,
        measured_memory_bw_gbps: 0.0,
        measured_h2d_bw_gbps: 0.0,
        measured_d2h_bw_gbps: 0.0,
        measured_tensor_tflops: None,
        vram_free_bytes: (vram_gb * 1e9) as u64,
        utilization: 0.0,
        temperature_c: 40,
        current_clock_mhz: 1800,
        power_draw_watts: 100,
        is_throttling: false,
        capability_scores: WorkloadScores::default(),
        geu: 0.0,
        has_rebar: false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rtx_3060_profile() -> GpuProfile {
        let mut p = make_test_profile(1, "NVIDIA GeForce RTX 3060", (8, 6), 12.0, 12.7, 360.0);
        p.tensor_core_count = 112;
        p.tensor_core_gen = Some(3); // Ampere
        p
    }

    fn rtx_3090_profile() -> GpuProfile {
        let mut p = make_test_profile(2, "NVIDIA GeForce RTX 3090", (8, 6), 24.0, 35.6, 936.0);
        p.tensor_core_count = 328;
        p.tensor_core_gen = Some(3); // Ampere
        p.bar1_size_bytes = 24 * 1024 * 1024 * 1024; // 24GB ReBAR
        p.vram_total_bytes = 24 * 1024 * 1024 * 1024;
        p
    }

    fn rtx_4090_profile() -> GpuProfile {
        let mut p = make_test_profile(3, "NVIDIA GeForce RTX 4090", (8, 9), 24.0, 82.6, 1008.0);
        p.tensor_core_count = 512;
        p.tensor_core_gen = Some(4); // Ada
        p
    }

    fn rtx_2060_profile() -> GpuProfile {
        let mut p = make_test_profile(4, "NVIDIA GeForce RTX 2060", (7, 5), 6.0, 6.5, 336.0);
        p.tensor_core_count = 34;
        p.tensor_core_gen = Some(2); // Turing
        p
    }

    // --- CapabilityScorer tests ---

    #[test]
    fn test_reference_gpu_scores_near_unity() {
        let scorer = CapabilityScorer::new();
        let mut profile = rtx_3060_profile();
        scorer.compute_scores(&mut profile);

        // Reference GPU should score ~1.0 in compute and memory
        assert!(
            (profile.capability_scores.compute - 1.0).abs() < 0.01,
            "compute: {}",
            profile.capability_scores.compute
        );
        assert!(
            (profile.capability_scores.memory - 1.0).abs() < 0.01,
            "memory: {}",
            profile.capability_scores.memory
        );
        assert!(
            (profile.capability_scores.capacity - 1.0).abs() < 0.01,
            "capacity: {}",
            profile.capability_scores.capacity
        );
    }

    #[test]
    fn test_rtx_3090_scores_higher_than_3060() {
        let scorer = CapabilityScorer::new();
        let mut p3060 = rtx_3060_profile();
        let mut p3090 = rtx_3090_profile();
        scorer.compute_scores(&mut p3060);
        scorer.compute_scores(&mut p3090);

        assert!(p3090.capability_scores.compute > p3060.capability_scores.compute);
        assert!(p3090.capability_scores.memory > p3060.capability_scores.memory);
        assert!(p3090.capability_scores.capacity > p3060.capability_scores.capacity);
        assert!(p3090.geu > p3060.geu);
    }

    #[test]
    fn test_rtx_4090_highest_scores() {
        let scorer = CapabilityScorer::new();
        let mut p3090 = rtx_3090_profile();
        let mut p4090 = rtx_4090_profile();
        scorer.compute_scores(&mut p3090);
        scorer.compute_scores(&mut p4090);

        assert!(p4090.capability_scores.compute > p3090.capability_scores.compute);
        assert!(p4090.capability_scores.overall > p3090.capability_scores.overall);
        assert!(p4090.geu > p3090.geu);
    }

    #[test]
    fn test_geu_weights_applied() {
        let scorer = CapabilityScorer::new();
        let mut profile = rtx_3090_profile();
        scorer.compute_scores(&mut profile);

        // GEU = 0.4 * compute + 0.3 * memory + 0.3 * capacity
        let expected = 0.4 * profile.capability_scores.compute
            + 0.3 * profile.capability_scores.memory
            + 0.3 * profile.capability_scores.capacity;
        assert!(
            (profile.geu - expected).abs() < 0.001,
            "geu: {}, expected: {}",
            profile.geu,
            expected
        );
    }

    #[test]
    fn test_rebar_detection() {
        let scorer = CapabilityScorer::new();
        let mut p3060 = rtx_3060_profile();
        let mut p3090 = rtx_3090_profile();
        scorer.compute_scores(&mut p3060);
        scorer.compute_scores(&mut p3090);

        assert!(!p3060.has_rebar, "3060 should not have ReBAR (256MB BAR1)");
        assert!(p3090.has_rebar, "3090 should have ReBAR (24GB BAR1)");
    }

    #[test]
    fn test_measured_values_override_theoretical() {
        let scorer = CapabilityScorer::new();
        let mut profile = rtx_3060_profile();
        // Set measured values (Tier 2)
        profile.measured_fp32_gflops = 10_000.0; // 10 TFLOPS measured
        profile.measured_memory_bw_gbps = 300.0;
        profile.measured_h2d_bw_gbps = 12.0;
        scorer.compute_scores(&mut profile);

        // Should use measured, not theoretical
        let expected_compute = (10_000.0 / 1000.0) / scorer.reference.fp32_tflops;
        assert!(
            (profile.capability_scores.compute - expected_compute).abs() < 0.001,
            "compute: {}, expected: {}",
            profile.capability_scores.compute,
            expected_compute
        );
        let expected_memory = 300.0 / scorer.reference.memory_bw_gbps;
        assert!(
            (profile.capability_scores.memory - expected_memory).abs() < 0.001,
            "memory: {}, expected: {}",
            profile.capability_scores.memory,
            expected_memory
        );
    }

    #[test]
    fn test_workload_specific_scoring() {
        let scorer = CapabilityScorer::new();
        let mut profile = rtx_4090_profile();
        scorer.compute_scores(&mut profile);

        let compute_score = scorer.capability_score(&profile, WorkloadClass::ComputeBound);
        let memory_score = scorer.capability_score(&profile, WorkloadClass::MemoryBound);
        let tensor_score = scorer.capability_score(&profile, WorkloadClass::TensorBound);

        // 4090 is particularly strong at tensor ops, so tensor score should emphasize that
        assert!(tensor_score > 0.0);
        assert!(compute_score > 0.0);
        assert!(memory_score > 0.0);

        // All scores should be > 1.0 (better than reference GPU)
        assert!(compute_score > 1.0, "compute: {}", compute_score);
        assert!(memory_score > 1.0, "memory: {}", memory_score);
    }

    #[test]
    fn test_affinity_tracking() {
        let mut scorer = CapabilityScorer::new();

        scorer.adapt_from_observation(1, 0xDEAD, 1000, WorkloadClass::ComputeBound);
        assert_eq!(scorer.affinity_cache_len(), 1);

        let entry = scorer.get_affinity(0xDEAD, WorkloadClass::ComputeBound).expect("should exist");
        assert_eq!(entry.best_gpu, 1);
        assert_eq!(entry.sample_count, 1);

        // Second observation with faster time
        scorer.adapt_from_observation(2, 0xDEAD, 500, WorkloadClass::ComputeBound);
        let entry = scorer.get_affinity(0xDEAD, WorkloadClass::ComputeBound).expect("should exist");
        assert_eq!(entry.sample_count, 2);
        // EMA should decrease
        assert!(entry.avg_execution_ns < 1000);
    }

    #[test]
    fn test_no_tensor_cores_zero_score() {
        let scorer = CapabilityScorer::new();
        let mut profile = make_test_profile(10, "Old GPU", (7, 0), 8.0, 6.0, 200.0);
        profile.tensor_core_count = 0;
        profile.tensor_core_gen = None;
        scorer.compute_scores(&mut profile);

        assert_eq!(
            profile.capability_scores.tensor, 0.0,
            "No tensor cores should give 0.0 tensor score"
        );
    }

    // --- BinaryCompatibilityChecker tests ---

    #[test]
    fn test_exact_cubin_match() {
        let mut checker = BinaryCompatibilityChecker::new();
        let mut cubins = HashMap::new();
        cubins.insert((8, 6), 1024);
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins,
                ptx_targets: HashMap::new(),
                max_ptx_target: None,
                has_arch_conditional_ptx: false,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6
        assert_eq!(checker.is_compatible(1, &gpu), CompatResult::NativeCubin);
    }

    #[test]
    fn test_compatible_cubin_same_major() {
        let mut checker = BinaryCompatibilityChecker::new();
        let mut cubins = HashMap::new();
        cubins.insert((8, 0), 1024); // CC 8.0 cubin
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins,
                ptx_targets: HashMap::new(),
                max_ptx_target: None,
                has_arch_conditional_ptx: false,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6 > 8.0
        assert_eq!(
            checker.is_compatible(1, &gpu),
            CompatResult::CompatibleCubin
        );
    }

    #[test]
    fn test_ptx_jit_fallback() {
        let mut checker = BinaryCompatibilityChecker::new();
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins: HashMap::new(), // no cubins
                ptx_targets: HashMap::new(),
                max_ptx_target: Some((7, 5)),
                has_arch_conditional_ptx: false,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6, PTX target 7.5 <= 8 (major)
        assert_eq!(checker.is_compatible(1, &gpu), CompatResult::PtxJit);
    }

    #[test]
    fn test_incompatible_no_matching_binary() {
        let mut checker = BinaryCompatibilityChecker::new();
        let mut cubins = HashMap::new();
        cubins.insert((9, 0), 1024); // Only CC 9.0 cubin
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins,
                ptx_targets: HashMap::new(),
                max_ptx_target: None,
                has_arch_conditional_ptx: false,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6
        assert_eq!(checker.is_compatible(1, &gpu), CompatResult::Incompatible);
    }

    #[test]
    fn test_unknown_module() {
        let checker = BinaryCompatibilityChecker::new();
        let gpu = rtx_3060_profile();
        assert_eq!(checker.is_compatible(999, &gpu), CompatResult::Unknown);
    }

    #[test]
    fn test_arch_conditional_ptx_blocks_cross_major() {
        let mut checker = BinaryCompatibilityChecker::new();
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins: HashMap::new(),
                ptx_targets: HashMap::new(),
                max_ptx_target: Some((9, 0)), // PTX for CC 9.x
                has_arch_conditional_ptx: true,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6 -- different major from PTX 9.0
        assert_eq!(checker.is_compatible(1, &gpu), CompatResult::Incompatible);
    }

    #[test]
    fn test_arch_conditional_ptx_allows_same_major() {
        let mut checker = BinaryCompatibilityChecker::new();
        checker.register_module(
            1,
            ModuleBinaryInfo {
                cubins: HashMap::new(),
                ptx_targets: HashMap::new(),
                max_ptx_target: Some((8, 0)), // PTX for CC 8.x
                has_arch_conditional_ptx: true,
            },
        );

        let gpu = rtx_3060_profile(); // CC 8.6 -- same major as PTX 8.0
        assert_eq!(checker.is_compatible(1, &gpu), CompatResult::PtxJit);
    }

    // --- Workload classification tests ---

    #[test]
    fn test_classify_gemm_as_tensor_bound() {
        let kernel = KernelInfo {
            kernel_name: "cutlass_simt_sgemm_128x128_nn".to_string(),
            grid_dim: [128, 128, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 32768,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::TensorBound);
    }

    #[test]
    fn test_classify_reduce_as_memory_bound() {
        let kernel = KernelInfo {
            kernel_name: "vectorized_reduce_kernel".to_string(),
            grid_dim: [1024, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 256,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::MemoryBound);
    }

    #[test]
    fn test_classify_fft_as_compute_bound() {
        let kernel = KernelInfo {
            kernel_name: "radix2_fft_forward".to_string(),
            grid_dim: [64, 1, 1],
            block_dim: [512, 1, 1],
            shared_mem_bytes: 16384,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::ComputeBound);
    }

    #[test]
    fn test_classify_unknown_large_grid_small_smem() {
        let kernel = KernelInfo {
            kernel_name: "mystery_kernel_v2".to_string(),
            grid_dim: [20000, 1, 1],
            block_dim: [128, 1, 1],
            shared_mem_bytes: 0,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::MemoryBound);
    }

    #[test]
    fn test_classify_unknown_large_smem() {
        let kernel = KernelInfo {
            kernel_name: "custom_op_xyz".to_string(),
            grid_dim: [100, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 32768,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::ComputeBound);
    }

    #[test]
    fn test_classify_truly_unknown() {
        let kernel = KernelInfo {
            kernel_name: "custom_op_abc".to_string(),
            grid_dim: [100, 1, 1],
            block_dim: [64, 1, 1],
            shared_mem_bytes: 4096,
            min_compute_capability: (7, 5),
        };
        assert_eq!(classify_workload(&kernel), WorkloadClass::Unknown);
    }

    #[test]
    fn test_detect_arch_conditional_ptx() {
        let ptx_normal = ".version 8.0\n.target sm_86\n.address_size 64";
        let ptx_conditional = ".version 8.0\n.target sm_90a\n.address_size 64";
        assert!(!detect_arch_conditional(ptx_normal));
        assert!(detect_arch_conditional(ptx_conditional));
    }

    // --- GpuPoolManager tests ---

    #[test]
    fn test_register_and_retrieve() {
        let mut pool = GpuPoolManager::new();
        let profile = rtx_3060_profile();
        pool.register_gpu(profile).expect("should register");
        assert_eq!(pool.gpu_count(), 1);

        let retrieved = pool.get_profile(1).expect("should exist");
        assert_eq!(retrieved.name, "NVIDIA GeForce RTX 3060");
        // Scores should be computed
        assert!(retrieved.capability_scores.overall > 0.0);
    }

    #[test]
    fn test_reject_low_compute_capability() {
        let mut pool = GpuPoolManager::new();
        let mut profile = make_test_profile(99, "Old GPU", (6, 1), 4.0, 5.0, 200.0);
        profile.tensor_core_gen = None;

        let result = pool.register_gpu(profile);
        assert!(result.is_err());
        match result.unwrap_err() {
            GpuPoolError::InsufficientComputeCapability { cc, .. } => {
                assert_eq!(cc, (6, 1));
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_reject_driver_mismatch() {
        let mut pool = GpuPoolManager::new();
        let p1 = rtx_3060_profile(); // driver "570.86.15"
        pool.register_gpu(p1).expect("first GPU");

        let mut p2 = rtx_3090_profile();
        p2.gpu_id = 5;
        p2.driver_version = "535.100.01".to_string(); // Different major

        let result = pool.register_gpu(p2);
        assert!(result.is_err());
        match result.unwrap_err() {
            GpuPoolError::DriverVersionMismatch { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_reject_duplicate_gpu_id() {
        let mut pool = GpuPoolManager::new();
        let p1 = rtx_3060_profile();
        pool.register_gpu(p1).expect("first");

        let p2 = rtx_3060_profile(); // Same ID = 1
        let result = pool.register_gpu(p2);
        assert!(matches!(result, Err(GpuPoolError::AlreadyRegistered(1))));
    }

    #[test]
    fn test_remove_gpu() {
        let mut pool = GpuPoolManager::new();
        pool.register_gpu(rtx_3060_profile()).expect("register");
        assert_eq!(pool.gpu_count(), 1);

        let removed = pool.remove_gpu(1).expect("remove");
        assert_eq!(removed.gpu_id, 1);
        assert_eq!(pool.gpu_count(), 0);
        assert!(pool.remove_gpu(1).is_err());
    }

    #[test]
    fn test_gpus_by_capability_ordering() {
        let mut pool = GpuPoolManager::new();
        pool.register_gpu(rtx_3060_profile()).expect("3060");
        pool.register_gpu(rtx_3090_profile()).expect("3090");
        pool.register_gpu(rtx_4090_profile()).expect("4090");

        let ordered = pool.gpus_by_capability();
        assert_eq!(ordered.len(), 3);
        // 4090 should be first (highest overall score)
        assert_eq!(ordered[0], 3, "4090 should be first");
        // 3090 should be second
        assert_eq!(ordered[1], 2, "3090 should be second");
        // 3060 should be last
        assert_eq!(ordered[2], 1, "3060 should be last");
    }

    #[test]
    fn test_total_geu() {
        let mut pool = GpuPoolManager::new();
        pool.register_gpu(rtx_3060_profile()).expect("3060");
        pool.register_gpu(rtx_3090_profile()).expect("3090");

        let total = pool.total_geu();
        assert!(total > 0.0);
        // 3090 + 3060: 3090 should contribute more
        let geu_3060 = pool.geu(1).expect("3060 geu");
        let geu_3090 = pool.geu(2).expect("3090 geu");
        assert!((total - (geu_3060 + geu_3090)).abs() < 0.001);
        assert!(geu_3090 > geu_3060);
    }

    #[test]
    fn test_filter_compatible_by_cc() {
        let mut pool = GpuPoolManager::new();
        pool.register_gpu(rtx_2060_profile()).expect("2060"); // CC 7.5
        pool.register_gpu(rtx_3060_profile()).expect("3060"); // CC 8.6
        pool.register_gpu(rtx_4090_profile()).expect("4090"); // CC 8.9

        // Filter for CC >= 8.6
        let compatible = pool.filter_compatible((8, 6), None, &[4, 1, 3]);
        assert!(!compatible.contains(&4), "2060 CC 7.5 should be excluded");
        assert!(compatible.contains(&1), "3060 CC 8.6 should be included");
        assert!(compatible.contains(&3), "4090 CC 8.9 should be included");
    }

    // --- Calibration cache tests ---

    #[test]
    fn test_calibration_cache_insert_and_get() {
        let mut cache = CalibrationCache::new();
        let key = CalibrationKey {
            gpu_model: "RTX 3090".to_string(),
            driver_version: "570.86.15".to_string(),
        };
        let result = CalibrationResult {
            fp32_gflops: 30000.0,
            memory_bw_gbps: 800.0,
            h2d_bw_gbps: 14.0,
            d2h_bw_gbps: 13.0,
            tensor_tflops: Some(70.0),
            measured_at: SystemTime::now(),
        };
        cache.insert(key.clone(), result);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn test_calibration_cache_miss() {
        let cache = CalibrationCache::new();
        let key = CalibrationKey {
            gpu_model: "RTX 5090".to_string(),
            driver_version: "600.00.00".to_string(),
        };
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_stats_tracking() {
        let mut pool = GpuPoolManager::new();
        pool.register_gpu(rtx_3060_profile()).expect("register");

        assert_eq!(pool.stats().total_gpus.load(Ordering::Relaxed), 1);

        pool.capability_score(1, WorkloadClass::ComputeBound);
        assert_eq!(pool.stats().scoring_operations.load(Ordering::Relaxed), 1);

        pool.filter_compatible((7, 5), None, &[1]);
        assert_eq!(
            pool.stats().compatibility_checks.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_pool_config_custom_min_cc() {
        let config = PoolConfig {
            min_compute_capability: (8, 0),
            ..Default::default()
        };
        let mut pool = GpuPoolManager::with_config(config);

        // 2060 has CC 7.5, should be rejected with min CC 8.0
        let result = pool.register_gpu(rtx_2060_profile());
        assert!(result.is_err());

        // 3060 has CC 8.6, should be accepted
        let result = pool.register_gpu(rtx_3060_profile());
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_bandwidth_returns_measured_if_available() {
        let mut pool = GpuPoolManager::new();
        let mut profile = rtx_3060_profile();
        profile.measured_memory_bw_gbps = 330.0;
        pool.register_gpu(profile).expect("register");

        let bw = pool.memory_bandwidth_gbps(1).expect("should exist");
        assert!((bw - 330.0).abs() < 0.001, "should use measured value");
    }
}
