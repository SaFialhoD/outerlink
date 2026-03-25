# R23: Heterogeneous GPU Mixing — Pre-Plan v2

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of the R23 pre-plan. Defines exact Rust structs for GPU profiling and capability scoring, specifies the Gavel-style normalization algorithm, resolves open questions from v1, and locks down integration points with R10 v2, R13, R17 v2, R20 v2, and R25.

---

## 1. Resolved Open Questions

### Q1 (from v1): Should GEU weights be globally fixed or tunable per-pool?

**Resolution: Tunable per-pool with sensible defaults.** GEU weights ship with defaults (alpha=0.4 compute, beta=0.3 bandwidth, gamma=0.3 capacity) but can be overridden via pool configuration. Different workload profiles (inference-heavy pools vs training-heavy pools) benefit from different weight distributions. The `PoolConfig` includes a `geu_weights` field.

### Q2 (from v1): How to handle live GPU addition/removal?

**Resolution: Hot-add supported, hot-remove with drain.** Adding a GPU runs the calibration pipeline asynchronously and registers the new profile with the coordinator. No running workloads are disrupted. Removing a GPU triggers a drain phase: the scheduler stops assigning new work to the GPU, waits for in-flight kernels to complete (with a configurable timeout, default 30s), then deregisters. If the GPU crashes, the coordinator marks it as failed and reassigns its pending work.

### Q3 (from v1): What telemetry should R23 expose?

**Resolution:** Three telemetry channels:
1. **Per-GPU metrics** (sampled every 1s via NVML): utilization, temperature, clock speed, power draw, VRAM usage.
2. **Scheduling decisions** (per kernel dispatch): selected GPU, capability score, alternative scores, workload classification.
3. **Aggregate pool metrics** (computed every 10s): total GEU, utilized GEU, per-GPU efficiency ratio (actual throughput / theoretical).

### Q4 (from v1): GPU compatibility simulator?

**Resolution: Deferred to post-v1.** Valuable for user experience but not needed for core scheduling. The registration flow already reports warnings (no ReBAR, old driver, low VRAM). A simulator would be a CLI tool that takes a list of GPU models and reports expected pool characteristics.

### D1 (from v1): Minimum supported CC?

**Resolution: CC 7.5 (Turing) confirmed.** This covers all RTX 2000/3000/4000/5000 series. GPUs below CC 7.5 are rejected at registration with a clear error message. The performance gap between CC 7.5 and older architectures (missing Tensor Cores, lower IPC) makes supporting them a net negative for pool efficiency.

### D2 (from v1): Performance normalization method?

**Resolution: Three-tier hybrid.** See Section 3 for the full algorithm.

### D3 (from v1): Kernel dispatch compatibility strategy?

**Resolution: Pre-warm with async JIT.** See Section 5.

### D4 (from v1): Scheduler integration architecture?

**Resolution: Filter + Score.** R23 first filters by hard constraints, then provides a `CapabilityScore` that R17's scheduler incorporates. See Section 6.

### D5 (from v1): Driver version policy?

**Resolution: Moderate (same major version).** The pool coordinator stores the minimum driver version across all nodes. At registration, if a new node's driver major version differs from the pool baseline, registration is rejected with an upgrade recommendation. The coordinator reports the maximum supported CUDA toolkit version for the pool.

---

## 2. Core Data Structures

### 2.1 GpuProfile

```rust
/// Complete hardware profile for a GPU in the OuterLink pool.
/// Built at GPU registration time by querying CUDA Driver API and NVML.
#[derive(Debug, Clone)]
struct GpuProfile {
    /// OuterLink-assigned GPU identifier (unique across the pool).
    gpu_id: GpuId,
    /// Node (PC) this GPU belongs to.
    node_id: NodeId,

    // --- Static hardware attributes (set once at registration) ---

    /// GPU model name (e.g., "NVIDIA GeForce RTX 3090").
    name: String,
    /// CUDA compute capability (major, minor).
    compute_capability: (u32, u32),
    /// Number of Streaming Multiprocessors.
    sm_count: u32,
    /// Total CUDA cores (sm_count * cores_per_sm, arch-dependent).
    cuda_cores: u32,
    /// Tensor Core count (0 for pre-Volta).
    tensor_core_count: u32,
    /// Tensor Core generation (None for pre-Volta, Some(2) for Turing, etc.).
    tensor_core_gen: Option<u32>,
    /// Total VRAM in bytes.
    vram_total_bytes: u64,
    /// Theoretical memory bandwidth in GB/s.
    memory_bandwidth_gbps: f64,
    /// Theoretical FP32 peak TFLOPS.
    fp32_tflops: f64,
    /// Theoretical FP16 peak TFLOPS (with Tensor Cores if available).
    fp16_tflops: f64,
    /// BAR1 aperture size in bytes (256MB default, VRAM size with ReBAR).
    bar1_size_bytes: u64,
    /// PCIe generation (3, 4, 5).
    pcie_gen: u32,
    /// PCIe link width (x8, x16).
    pcie_width: u32,
    /// Measured PCIe bandwidth in GB/s (from calibration benchmark).
    pcie_bandwidth_gbps: f64,
    /// Number of async copy engines.
    async_engine_count: u32,
    /// L2 cache size in bytes.
    l2_cache_bytes: u32,
    /// GPU boost clock in MHz.
    boost_clock_mhz: u32,
    /// TDP (thermal design power) in watts.
    tdp_watts: u32,

    // --- Precision support flags ---

    supports_fp16: bool,
    supports_bf16: bool,     // CC 8.0+
    supports_tf32: bool,     // CC 8.0+
    supports_fp8: bool,      // CC 8.9+
    supports_fp4: bool,      // CC 10.0+
    supports_int8: bool,     // CC 7.5+

    // --- Driver info ---

    /// NVIDIA driver version string.
    driver_version: String,
    /// CUDA driver API version (e.g., 12080 for CUDA 12.8).
    cuda_driver_version: u32,
    /// Maximum CUDA toolkit version this driver supports.
    max_cuda_toolkit: (u32, u32),

    // --- Calibration benchmarks (set during registration) ---

    /// Measured FP32 GFLOPS (from SGEMM benchmark).
    measured_fp32_gflops: f64,
    /// Measured memory bandwidth in GB/s (from stream copy benchmark).
    measured_memory_bw_gbps: f64,
    /// Measured host-to-device bandwidth in GB/s.
    measured_h2d_bw_gbps: f64,
    /// Measured device-to-host bandwidth in GB/s.
    measured_d2h_bw_gbps: f64,
    /// Measured Tensor Core throughput in TFLOPS (FP16 matmul).
    measured_tensor_tflops: Option<f64>,

    // --- Dynamic state (updated periodically via NVML) ---

    /// Current free VRAM in bytes.
    vram_free_bytes: u64,
    /// Current GPU utilization (0.0 - 1.0).
    utilization: f64,
    /// Current GPU temperature in Celsius.
    temperature_c: u32,
    /// Current clock speed in MHz (may be lower than boost due to throttling).
    current_clock_mhz: u32,
    /// Current power draw in watts.
    power_draw_watts: u32,
    /// Whether the GPU is thermally throttling.
    is_throttling: bool,

    // --- Computed scores (updated when dynamic state changes) ---

    /// Normalized capability scores per workload class (see CapabilityScorer).
    capability_scores: WorkloadScores,
    /// GPU Equivalent Units (single number for fairness/quota).
    geu: f64,
    /// ReBAR status: true if BAR1 >= VRAM (full mapping available).
    has_rebar: bool,
}

/// Per-workload-class normalized scores.
/// Each score is relative to the reference GPU (RTX 3060 = 1.0).
#[derive(Debug, Clone, Default)]
struct WorkloadScores {
    compute: f64,    // FP32 throughput normalized
    memory: f64,     // Memory bandwidth normalized
    tensor: f64,     // Tensor Core throughput normalized
    capacity: f64,   // VRAM capacity normalized
    transfer: f64,   // PCIe/host transfer normalized
    /// Overall score (weighted combination, used by default).
    overall: f64,
}
```

### 2.2 Capability Scorer

```rust
/// Computes normalized capability scores for GPUs.
/// Implements the three-tier hybrid normalization approach.
struct CapabilityScorer {
    /// Reference GPU values (RTX 3060 as baseline = 1.0).
    reference: ReferenceValues,
    /// GEU weight configuration.
    geu_weights: GeuWeights,
    /// Workload-class-specific weights for the overall score.
    workload_weights: WorkloadWeightConfig,
    /// Kernel-to-GPU affinity cache (remembers which GPU ran a kernel well).
    affinity_cache: HashMap<u64, AffinityEntry>, // key: kernel function pointer hash
}

/// Reference values for normalization (RTX 3060 = 1.0).
struct ReferenceValues {
    fp32_tflops: f64,           // 12.7
    memory_bw_gbps: f64,        // 360.0
    tensor_tflops_fp16: f64,    // 50.6 (RTX 3060 FP16 Tensor)
    vram_gb: f64,                // 12.0
    pcie_bw_gbps: f64,          // 16.0 (PCIe 4.0 x16 effective)
}

/// Weights for GEU computation.
#[derive(Debug, Clone)]
struct GeuWeights {
    compute: f64,   // default 0.4
    bandwidth: f64, // default 0.3
    capacity: f64,  // default 0.3
}

/// Per-workload-class weight overrides for the overall score.
struct WorkloadWeightConfig {
    /// Weights when workload is compute-bound.
    compute_bound: ScoringWeights,
    /// Weights when workload is memory-bound.
    memory_bound: ScoringWeights,
    /// Weights when workload is tensor-bound.
    tensor_bound: ScoringWeights,
    /// Weights when workload class is unknown.
    default: ScoringWeights,
}

/// Weights used to combine individual scores into overall score.
#[derive(Debug, Clone)]
struct ScoringWeights {
    compute: f64,
    memory: f64,
    tensor: f64,
    capacity: f64,
    transfer: f64,
}

impl Default for WorkloadWeightConfig {
    fn default() -> Self {
        Self {
            compute_bound: ScoringWeights {
                compute: 0.50, memory: 0.15, tensor: 0.15, capacity: 0.10, transfer: 0.10,
            },
            memory_bound: ScoringWeights {
                compute: 0.10, memory: 0.50, tensor: 0.10, capacity: 0.20, transfer: 0.10,
            },
            tensor_bound: ScoringWeights {
                compute: 0.10, memory: 0.15, tensor: 0.55, capacity: 0.10, transfer: 0.10,
            },
            default: ScoringWeights {
                compute: 0.30, memory: 0.25, tensor: 0.20, capacity: 0.15, transfer: 0.10,
            },
        }
    }
}

/// Cached affinity: which GPU ran this kernel fastest.
struct AffinityEntry {
    best_gpu: GpuId,
    avg_execution_ns: u64,
    sample_count: u32,
    last_updated: Instant,
}
```

### 2.3 Binary Compatibility Checker

```rust
/// Checks whether a kernel binary can execute on a target GPU.
/// Inspects fatbin contents extracted during cuModuleLoad interception.
struct BinaryCompatibilityChecker {
    /// Per-module extracted binaries.
    module_binaries: HashMap<CUmodule, ModuleBinaryInfo>,
    /// JIT compilation cache status per (module, cc).
    jit_cache: HashMap<(CUmodule, (u32, u32)), JitStatus>,
}

/// Extracted binary information from a CUDA module's fatbin.
#[derive(Debug, Clone)]
struct ModuleBinaryInfo {
    /// Available cubin targets (CC major.minor -> cubin bytes).
    cubins: HashMap<(u32, u32), Vec<u8>>,
    /// PTX source if available (target CC -> PTX string).
    ptx: HashMap<(u32, u32), String>,
    /// Highest PTX target version available.
    max_ptx_target: Option<(u32, u32)>,
    /// Whether any PTX uses architecture-conditional features (*a suffix).
    has_arch_conditional_ptx: bool,
}

#[derive(Debug, Clone, Copy)]
enum JitStatus {
    /// JIT compilation not attempted.
    NotAttempted,
    /// JIT compilation in progress (async).
    InProgress,
    /// JIT compilation succeeded and is cached by the CUDA driver.
    Cached,
    /// JIT compilation failed (incompatible).
    Failed,
}

impl BinaryCompatibilityChecker {
    /// Check if a kernel from this module can run on a GPU.
    fn is_compatible(&self, module: CUmodule, gpu: &GpuProfile) -> CompatResult {
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
        let compatible_cubin = info.cubins.keys()
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompatResult {
    /// Exact native binary available — best performance, no JIT overhead.
    NativeCubin,
    /// Compatible cubin from same major version — good performance.
    CompatibleCubin,
    /// PTX available for JIT compilation — may have first-launch latency.
    PtxJit,
    /// No compatible binary — GPU cannot run this kernel.
    Incompatible,
    /// Module not yet analyzed.
    Unknown,
}
```

---

## 3. Capability Normalization Algorithm

### 3.1 Three-Tier Hybrid Approach

```
Tier 1: Static Normalization (available immediately at registration)
  |  Uses theoretical specs from cuDeviceGetAttribute queries.
  |  Scores = specs / reference_gpu_specs.
  |  Accuracy: ~70% (theoretical != achieved, but good enough to start).
  |
  v
Tier 2: Calibration Benchmarks (run once at registration, ~30 seconds)
  |  Runs four micro-benchmarks:
  |  (a) SGEMM 4096x4096 — measures achieved FP32 GFLOPS
  |  (b) Stream copy 256MB — measures achieved memory bandwidth
  |  (c) H2D + D2H 64MB — measures PCIe transfer throughput
  |  (d) FP16 matmul 4096x4096 (if Tensor Cores) — measures tensor throughput
  |  Results cached per (gpu_model, driver_version) — only run once per model.
  |  Scores = measured / reference_gpu_measured.
  |  Accuracy: ~90%.
  |
  v
Tier 3: Runtime Adaptation (continuous, after first few kernels)
  |  Observe actual kernel execution times per GPU.
  |  Maintain exponential moving average of throughput per (kernel_hash, gpu_id).
  |  If observed performance deviates > 20% from calibrated score,
  |  adjust that GPU's score for that workload class.
  |  Handles: thermal throttling, power limits, contention, aging.
  |  Accuracy: ~95%+.
```

### 3.2 Score Computation

```rust
impl CapabilityScorer {
    /// Compute all capability scores for a GPU profile.
    /// Called after calibration benchmarks complete.
    fn compute_scores(&self, profile: &mut GpuProfile) {
        let ref_vals = &self.reference;

        // Tier 1 or Tier 2 scores depending on availability
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
                            2 => 1.0,   // Turing baseline
                            3 => 2.0,   // Ampere (TF32, higher throughput)
                            4 => 3.5,   // Ada (FP8, higher throughput)
                            5 => 5.0,   // Blackwell (FP4/FP6)
                            _ => 1.0,
                        };
                        let core_ratio = profile.tensor_core_count as f64 / 112.0; // RTX 3060 has 112
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
        profile.capability_scores.overall =
            w.compute * profile.capability_scores.compute
            + w.memory * profile.capability_scores.memory
            + w.tensor * profile.capability_scores.tensor
            + w.capacity * profile.capability_scores.capacity
            + w.transfer * profile.capability_scores.transfer;

        // GEU
        let gw = &self.geu_weights;
        profile.geu =
            gw.compute * profile.capability_scores.compute
            + gw.bandwidth * profile.capability_scores.memory
            + gw.capacity * profile.capability_scores.capacity;

        // ReBAR detection
        profile.has_rebar = profile.bar1_size_bytes >= profile.vram_total_bytes;
    }

    /// Get workload-specific capability score for HEFT integration (called by R13).
    fn capability_score(&self, profile: &GpuProfile, workload_class: WorkloadClass) -> f64 {
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
    fn adapt_from_observation(
        &mut self,
        gpu_id: GpuId,
        kernel_hash: u64,
        observed_ns: u64,
        workload_class: WorkloadClass,
    ) {
        let entry = self.affinity_cache.entry(kernel_hash).or_insert(AffinityEntry {
            best_gpu: gpu_id,
            avg_execution_ns: observed_ns,
            sample_count: 0,
            last_updated: Instant::now(),
        });

        // Exponential moving average (alpha = 0.2)
        let alpha = 0.2;
        entry.avg_execution_ns = ((1.0 - alpha) * entry.avg_execution_ns as f64
            + alpha * observed_ns as f64) as u64;
        entry.sample_count += 1;
        entry.last_updated = Instant::now();

        if observed_ns < entry.avg_execution_ns {
            entry.best_gpu = gpu_id;
        }
    }
}
```

### 3.3 Reference GPU Values (RTX 3060)

| Metric | Value | Source |
|--------|-------|--------|
| FP32 TFLOPS | 12.7 | Theoretical: 3584 cores * 2 * 1.78 GHz |
| Memory BW | 360 GB/s | Theoretical: 15 Gbps * 192-bit / 8 |
| Tensor FP16 TFLOPS | 50.6 | Theoretical: 112 Tensor Cores * spec throughput |
| VRAM | 12 GB | Spec |
| PCIe BW | 16 GB/s | PCIe 4.0 x16 effective (~80% of theoretical 32 GB/s) |

### 3.4 Expected Score Table

| GPU | Compute | Memory | Tensor | Capacity | Overall | GEU |
|-----|---------|--------|--------|----------|---------|-----|
| RTX 2060 | 0.51 | 0.93 | 0.43 | 0.50 | 0.59 | 0.62 |
| RTX 2080 Ti | 1.06 | 1.71 | 0.87 | 0.92 | 1.12 | 1.22 |
| RTX 3060 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| RTX 3090 | 2.80 | 2.60 | 2.93 | 2.00 | 2.65 | 2.50 |
| RTX 4070 | 2.29 | 1.40 | 3.28 | 1.00 | 2.09 | 1.64 |
| RTX 4090 | 6.50 | 2.80 | 9.14 | 2.00 | 5.13 | 4.04 |
| RTX 5090 | 8.25 | 4.98 | 12.14 | 2.67 | 7.03 | 5.59 |

Range: RTX 2060 (0.59) to RTX 5090 (7.03) = 11.9x overall gap. This validates the need for capability-aware scheduling.

---

## 4. Calibration Benchmark Suite

### 4.1 Benchmarks

| Benchmark | Purpose | Duration | Output |
|-----------|---------|----------|--------|
| SGEMM 4096x4096 | FP32 compute throughput | ~5s (warmup + 100 iterations) | GFLOPS |
| Stream copy 256MB | Memory bandwidth | ~3s (warmup + 50 iterations) | GB/s |
| H2D/D2H 64MB | PCIe transfer throughput | ~5s (warmup + 50 each direction) | GB/s each direction |
| HGEMM 4096x4096 (FP16) | Tensor Core throughput | ~5s (warmup + 100 iterations) | TFLOPS |

**Total calibration time:** ~18 seconds per GPU model.

### 4.2 Caching Strategy

```rust
/// Calibration results are cached per (gpu_model_name, driver_version).
/// Cache is stored on the coordinator node.
struct CalibrationCache {
    entries: HashMap<CalibrationKey, CalibrationResult>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CalibrationKey {
    gpu_model: String,        // e.g., "NVIDIA GeForce RTX 3090"
    driver_version: String,   // e.g., "570.86.15"
}

struct CalibrationResult {
    fp32_gflops: f64,
    memory_bw_gbps: f64,
    h2d_bw_gbps: f64,
    d2h_bw_gbps: f64,
    tensor_tflops: Option<f64>,
    measured_at: SystemTime,
}
```

When a GPU registers, the coordinator checks the cache. If a matching entry exists and is less than 90 days old, calibration is skipped and cached values are used. Otherwise, calibration benchmarks are dispatched to the GPU's node.

### 4.3 Benchmark Kernels

Calibration kernels are bundled with the OuterLink server binary as embedded PTX (compiled for CC 7.5 minimum). They are loaded via `cuModuleLoadData` and launched with `cuLaunchKernel`. This avoids requiring any CUDA toolkit on the target node beyond the driver.

---

## 5. PTX JIT Pre-Warming

### 5.1 Strategy

When OuterLink intercepts `cuModuleLoad` or `cuModuleLoadData`:

```
1. Extract fatbin from the loaded module
2. Parse fatbin to find cubins and PTX payloads
3. For each GPU in the pool:
   a. Check if native cubin exists for that GPU's CC → skip (already optimal)
   b. Check if compatible cubin exists → skip (good enough)
   c. If only PTX available → trigger async JIT compilation on that GPU
4. JIT compilation runs in the background on each target GPU
5. Result is cached by the CUDA driver on that node
6. When a kernel is later dispatched to that GPU, JIT cache is warm
```

### 5.2 Async JIT Flow

```rust
/// Trigger async JIT compilation for a module on a target GPU.
/// Called by the coordinator when a new module is loaded.
async fn prewarm_jit(
    module_ptx: &[u8],
    target_gpu: &GpuProfile,
    jit_cache: &mut HashMap<(CUmodule, (u32, u32)), JitStatus>,
) {
    let key = (module_handle, target_gpu.compute_capability);
    jit_cache.insert(key, JitStatus::InProgress);

    // Send PTX to the target node's OuterLink server
    // The server calls cuModuleLoadData with JIT options targeting its GPU
    // This triggers CUDA driver JIT compilation and caching
    match remote_jit_compile(target_gpu.node_id, module_ptx).await {
        Ok(_) => jit_cache.insert(key, JitStatus::Cached),
        Err(_) => jit_cache.insert(key, JitStatus::Failed),
    };
}
```

### 5.3 Architecture-Conditional PTX Handling

PTX compiled with `compute_90a` or `compute_100a` (architecture-specific features) is NOT forward-compatible across major versions. Detection:

```rust
/// Check if PTX uses architecture-conditional features.
fn detect_arch_conditional(ptx: &str) -> bool {
    // Architecture-conditional targets end with 'a' suffix
    // e.g., ".target sm_90a" or ".target compute_100a"
    ptx.lines().any(|line| {
        line.trim().starts_with(".target") && line.trim().ends_with('a')
    })
}
```

If arch-conditional PTX is detected, the scheduler only dispatches to GPUs within the same CC major version.

---

## 6. Integration Points

### 6.1 R17 v2: Topology-Aware Scheduling

**R23 provides capability scores to R17's PlacementDecision:**

```rust
/// R17 v2's scoring function includes a gpu_capability weight.
/// R23 provides the capability score for each (gpu, workload_class) pair.
///
/// R17's combined score (from R17 v2 preplan):
///   total_score = 0.35 * locality_score      // R17: data proximity
///               + 0.25 * network_score       // R17: network path quality
///               + 0.25 * load_score          // inverse utilization
///               + 0.15 * capability_score    // R23: GPU fitness
///
/// R23 provides the capability_score via:
trait GpuCapabilityProvider {
    /// Get all GPU profiles in the pool.
    fn all_profiles(&self) -> &[GpuProfile];

    /// Get workload-specific capability score for a GPU.
    /// Score is normalized (1.0 = reference GPU).
    /// Used by R17's PlacementDecision and R13's HEFT partitioner.
    fn capability_score(&self, gpu_id: GpuId, workload_class: WorkloadClass) -> f64;

    /// Filter GPUs by hard constraints for a kernel.
    /// Returns only GPUs that can actually execute this kernel.
    fn filter_compatible(
        &self,
        kernel_info: &KernelNodeInfo,
        candidates: &[GpuId],
    ) -> Vec<GpuId>;

    /// Check binary compatibility (cubin/PTX) for a specific module+GPU.
    fn binary_compat(
        &self,
        module: CUmodule,
        gpu_id: GpuId,
    ) -> CompatResult;

    /// Get GEU rating for a GPU (for quota/fairness calculations).
    fn geu(&self, gpu_id: GpuId) -> f64;
}
```

### 6.2 R13: CUDA Graph Interception

**R23 provides GPU profiles for HEFT scheduling:**

R13's HEFT algorithm calls `capability_score(gpu_id, workload_class)` for each kernel node to estimate per-GPU execution cost. R23's `filter_compatible()` eliminates GPUs that cannot run a kernel (wrong CC, no binary). This is the primary integration: R13 cannot partition a graph across heterogeneous GPUs without R23's cost model.

```rust
/// R13's HEFT partitioner uses R23 as follows:
///
/// 1. For each kernel node, call filter_compatible() to get candidate GPUs.
/// 2. For each candidate, call capability_score(gpu, node.workload_class)
///    to estimate execution cost: cost_ns = base_cost / capability_score.
/// 3. HEFT's EFT calculation uses these per-GPU costs.
/// 4. The result: faster GPUs get more work (lower EFT -> more nodes assigned).
```

### 6.3 R20 v2: NCCL Backend

**R23 provides asymmetric bandwidth information to NCCL:**

```rust
/// R20 v2 needs to report accurate per-GPU bandwidth to NCCL's
/// topology engine for heterogeneous ring/tree construction.
///
/// R23 provides this via:
trait AsymmetricBandwidthProvider {
    /// Get GPU-to-GPU bandwidth in GB/s (considers PCIe gen, link width, NIC speed).
    /// Used by R20 to build NCCL topology XML with accurate link speeds.
    fn gpu_pair_bandwidth(&self, src: GpuId, dst: GpuId) -> f64;

    /// Get per-GPU memory bandwidth (for NCCL kernel scheduling).
    fn memory_bandwidth(&self, gpu_id: GpuId) -> f64;

    /// Get ordered list of GPUs by capability (for NCCL ring ordering).
    /// NCCL benefits from homogeneous-ish ring segments.
    fn gpus_by_capability(&self) -> Vec<GpuId>;
}
```

### 6.4 R10 v2: Memory Hierarchy

**R23 informs R10's tier placement with bandwidth data:**

```rust
/// R10 v2's MigrationEngine uses GPU bandwidth profiles from R23
/// to make tier placement decisions.
///
/// For memory-bound workloads:
///   - Prefer placing pages on high-bandwidth GPUs
///   - R23's memory bandwidth score influences page migration priority
///
/// R23 provides:
trait GpuBandwidthProvider {
    /// Memory bandwidth for a specific GPU in GB/s.
    fn memory_bandwidth_gbps(&self, gpu_id: GpuId) -> f64;
    /// Whether this GPU has ReBAR enabled (affects OpenDMA viability).
    fn has_rebar(&self, gpu_id: GpuId) -> bool;
    /// BAR1 window size for windowed DMA when ReBAR is absent.
    fn bar1_size(&self, gpu_id: GpuId) -> u64;
}
```

### 6.5 R25: Cooperative Kernel Splitting

**R23 gates R25's kernel splitting by GPU capability:**

```rust
/// R25 (Cooperative Kernel Splitting) requires CC >= 7.5 for cooperative launch.
/// R23 provides the CC check and also informs R25 which GPUs are suitable
/// for receiving split kernel shards.
///
/// Only GPUs that satisfy ALL of:
///   1. CC >= 7.5
///   2. Binary compatible with the kernel
///   3. Sufficient VRAM for the shard
///   4. Not currently throttling
/// are eligible for cooperative kernel execution.
///
/// R23 provides this filter:
fn filter_for_cooperative_launch(
    &self,
    kernel: &KernelNodeInfo,
    candidates: &[GpuId],
) -> Vec<GpuId> {
    candidates.iter()
        .filter(|&&gpu_id| {
            let profile = self.get_profile(gpu_id);
            profile.compute_capability >= (7, 5)
                && self.binary_compat(kernel.module, gpu_id) != CompatResult::Incompatible
                && !profile.is_throttling
        })
        .copied()
        .collect()
}
```

---

## 7. Workload Classification

### 7.1 Heuristic Classification at Interception Time

Classifying a kernel's bottleneck without running it uses heuristics from kernel metadata:

```rust
/// Classify a kernel's workload type from metadata available at dispatch time.
fn classify_workload(kernel: &KernelNodeInfo) -> WorkloadClass {
    let total_threads =
        (kernel.grid_dim[0] * kernel.grid_dim[1] * kernel.grid_dim[2]) as u64
        * (kernel.block_dim[0] * kernel.block_dim[1] * kernel.block_dim[2]) as u64;
    let shared_mem = kernel.shared_mem_bytes;

    // Heuristic 1: Kernel name pattern matching
    let name = kernel.kernel_name.to_lowercase();

    // Tensor-bound patterns (matmul, conv, attention with Tensor Cores)
    if name.contains("gemm") || name.contains("cutlass") || name.contains("wmma")
        || name.contains("mma_") || name.contains("tensorop")
    {
        return WorkloadClass::TensorBound;
    }

    // Memory-bound patterns (copy, reduce, norm, softmax, elementwise)
    if name.contains("reduce") || name.contains("softmax") || name.contains("layernorm")
        || name.contains("batchnorm") || name.contains("elementwise")
        || name.contains("copy") || name.contains("transpose")
    {
        return WorkloadClass::MemoryBound;
    }

    // Compute-bound patterns (crypto, FFT, physics)
    if name.contains("fft") || name.contains("crypto") || name.contains("sha")
        || name.contains("physics") || name.contains("raytrace")
    {
        return WorkloadClass::ComputeBound;
    }

    // Heuristic 2: Grid geometry
    // Large grid + small shared memory → likely memory-bound (many threads doing simple ops)
    // Moderate grid + large shared memory → likely compute-bound (complex ops with data reuse)
    let threads_per_block = (kernel.block_dim[0] * kernel.block_dim[1] * kernel.block_dim[2]) as u64;
    let grid_size = (kernel.grid_dim[0] * kernel.grid_dim[1] * kernel.grid_dim[2]) as u64;

    if grid_size > 10_000 && shared_mem < 1024 {
        return WorkloadClass::MemoryBound;
    }
    if shared_mem > 16_384 && threads_per_block >= 256 {
        return WorkloadClass::ComputeBound;
    }

    WorkloadClass::Unknown
}
```

### 7.2 Runtime Refinement

After observing actual kernel execution times across different GPU types, the classifier refines:

```rust
/// If a kernel runs disproportionately faster on a high-bandwidth GPU
/// vs a high-compute GPU, reclassify it as memory-bound (and vice versa).
fn refine_classification(
    kernel_hash: u64,
    observations: &[(GpuId, u64)], // (gpu, execution_ns) pairs
    profiles: &[GpuProfile],
) -> WorkloadClass {
    // Compare execution time ratios against capability ratios
    // If time_ratio ≈ memory_bw_ratio → memory-bound
    // If time_ratio ≈ compute_ratio → compute-bound
    // If time_ratio ≈ tensor_ratio → tensor-bound

    // Requires at least 2 observations on different GPU types
    if observations.len() < 2 { return WorkloadClass::Unknown; }

    // ... ratio comparison logic ...
    // This is the Gavel-style effective throughput insight applied per-kernel.
    todo!()
}
```

---

## 8. GPU Registration Flow

```
Node startup:
  |
  v
[1. Query hardware attributes via CUDA Driver API]
  |  cuDeviceGetAttribute for all ~20 attributes
  |  cuDeviceTotalMem, cuDeviceGetName
  |  cuDriverGetVersion
  |  NVML: nvmlDeviceGetBAR1MemoryInfo, temperature, power
  |
  v
[2. Build static GpuProfile]
  |  Compute derived metrics (FP32 TFLOPS, bandwidth, etc.)
  |  Set precision support flags based on CC
  |  Detect ReBAR from BAR1 size
  |
  v
[3. Check calibration cache on coordinator]
  |  Key: (gpu_model_name, driver_version)
  |  If HIT and fresh: use cached benchmarks, skip to step 5
  |
  v
[4. Run calibration benchmarks (async, ~18 seconds)]
  |  SGEMM, stream copy, H2D/D2H, HGEMM
  |  Store results in profile and in coordinator's cache
  |
  v
[5. Compute capability scores]
  |  CapabilityScorer::compute_scores(&mut profile)
  |  Compute GEU
  |
  v
[6. Validate pool compatibility]
  |  Check driver major version matches pool baseline
  |  Check CC >= 7.5 minimum
  |  Warn if no ReBAR
  |  Warn if low VRAM (< 6 GB)
  |
  v
[7. Register with coordinator]
  |  Transmit full GpuProfile
  |  Coordinator updates its pool state
  |  R17's topology graph updated with new GPU node
  |  R20 notified for NCCL topology update
  |
  v
[8. Start periodic telemetry (every 1s)]
     NVML queries: utilization, temperature, clock, power, VRAM free
     Update dynamic fields in GpuProfile
     Detect throttling events
```

---

## 9. Dynamic Monitoring and Throttle Detection

```rust
/// Periodic GPU state monitor (runs every 1 second per GPU).
struct GpuMonitor {
    /// Per-GPU monitoring state.
    gpu_states: HashMap<GpuId, MonitorState>,
}

struct MonitorState {
    /// Exponential moving average of clock speed (detect gradual throttling).
    avg_clock_mhz: f64,
    /// Baseline clock speed (from first N non-throttled samples).
    baseline_clock_mhz: f64,
    /// Number of consecutive throttling samples.
    throttle_streak: u32,
    /// Whether we've notified the scheduler about throttling.
    throttle_notified: bool,
}

impl GpuMonitor {
    /// Called every 1 second with fresh NVML data.
    fn update(&mut self, gpu_id: GpuId, profile: &mut GpuProfile) {
        let state = self.gpu_states.entry(gpu_id).or_insert(MonitorState {
            avg_clock_mhz: profile.boost_clock_mhz as f64,
            baseline_clock_mhz: profile.boost_clock_mhz as f64,
            throttle_streak: 0,
            throttle_notified: false,
        });

        // EMA of clock speed (alpha = 0.3)
        state.avg_clock_mhz = 0.7 * state.avg_clock_mhz
            + 0.3 * profile.current_clock_mhz as f64;

        // Throttle detection: clock dropped > 15% below baseline
        let throttle_threshold = state.baseline_clock_mhz * 0.85;
        if state.avg_clock_mhz < throttle_threshold {
            state.throttle_streak += 1;
            if state.throttle_streak >= 5 && !state.throttle_notified {
                // Sustained throttling detected (5+ seconds)
                profile.is_throttling = true;
                // Reduce capability scores proportionally
                let throttle_factor = state.avg_clock_mhz / state.baseline_clock_mhz;
                profile.capability_scores.compute *= throttle_factor;
                profile.capability_scores.tensor *= throttle_factor;
                // Memory bandwidth is less affected by clock throttling
                profile.capability_scores.memory *= (1.0 + throttle_factor) / 2.0;
                state.throttle_notified = true;
            }
        } else {
            state.throttle_streak = 0;
            if state.throttle_notified {
                // Recovered from throttling — restore scores
                profile.is_throttling = false;
                state.throttle_notified = false;
                // Scores will be recomputed on next full score update
            }
        }
    }
}
```

---

## 10. User-Facing GPU Policies

### 10.1 Environment Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `OUTERLINK_GPU_PIN` | `node:gpu` | None | Pin all work to a specific GPU (debugging) |
| `OUTERLINK_MIN_CC` | `major.minor` | `7.5` | Minimum compute capability for pool membership |
| `OUTERLINK_MIN_TFLOPS` | `float` | `0.0` | Exclude GPUs below this FP32 TFLOPS threshold |
| `OUTERLINK_MIN_VRAM_GB` | `float` | `0.0` | Exclude GPUs below this VRAM threshold |
| `OUTERLINK_PREFER` | `compute\|memory\|tensor\|capacity` | None | Override workload classification for all kernels |
| `OUTERLINK_EXCLUDE_NODES` | `node1,node2,...` | None | Exclude specific nodes from the pool |
| `OUTERLINK_EXCLUDE_GPUS` | `node:gpu,...` | None | Exclude specific GPUs |
| `OUTERLINK_REQUIRE_REBAR` | `bool` | `false` | Only use GPUs with ReBAR enabled |

### 10.2 Configuration File

```toml
# outerlink-pool.toml
[pool]
min_compute_capability = "7.5"
driver_version_policy = "same_major"  # strict | same_major | permissive

[geu_weights]
compute = 0.4
bandwidth = 0.3
capacity = 0.3

[calibration]
cache_ttl_days = 90
benchmark_timeout_seconds = 60

[monitoring]
telemetry_interval_ms = 1000
throttle_threshold_percent = 15
throttle_streak_threshold = 5
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

| Test | Validates |
|------|-----------|
| Score computation for known GPU specs | Scores match expected table in Section 3.4 |
| Workload classification heuristics | Known kernel names classified correctly |
| Binary compatibility checker | Cubin/PTX/arch-conditional detection correct |
| GEU computation | GEU values match expected for reference GPUs |
| Throttle detection | Sustained clock drop triggers throttle flag |
| Calibration cache hit/miss | Cache key matching works correctly |

### 11.2 Integration Tests

| Test | Validates |
|------|-----------|
| GPU registration flow (end-to-end) | Profile built, calibrated, scored, registered |
| R17 integration: PlacementDecision with capability scores | Faster GPUs preferred for compute-bound kernels |
| R13 integration: HEFT uses capability scores | Heterogeneous HEFT assigns more work to faster GPUs |
| Driver version mismatch rejection | Incompatible driver detected at registration |
| PTX JIT pre-warming | Async JIT completes and cache is warm |
| Hot-add GPU to running pool | New GPU available for scheduling within 30s |

### 11.3 Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| Calibration suite total runtime | < 30 seconds per GPU model |
| Score computation | < 0.01 ms per GPU |
| Workload classification | < 0.001 ms per kernel |
| Binary compatibility check | < 0.01 ms per (module, GPU) pair |
| Monitoring telemetry overhead | < 1% CPU on NVML polling thread |

---

## 12. Acceptance Criteria

1. Any GeForce GPU with CC >= 7.5 can join the pool and be characterized automatically within 30 seconds.
2. Capability scores are within 10% of hand-computed values for all reference GPUs in the expected score table.
3. The scheduler routes kernels only to compatible GPUs (CC, binary, driver) with zero incompatible dispatches.
4. Work distribution across heterogeneous GPUs is proportional to their workload-specific capability scores.
5. Adding a slower GPU to the pool never decreases total throughput for async workloads.
6. Adding a slower GPU to the pool for sync workloads either improves throughput or the scheduler automatically excludes it (threshold rule: > 10x gap).
7. Thermal throttling is detected within 5 seconds and scheduling adjusted within 1 additional second.
8. PTX JIT pre-warming completes for all pool GPUs before first kernel dispatch to each GPU.
9. Driver version mismatches detected and reported at registration time with clear user guidance.
10. GEU values enable fair quota allocation across users with mixed GPU hardware.

---

## 13. Estimated Effort (Updated)

| Component | Complexity | Estimate |
|-----------|-----------|----------|
| GpuProfile struct + attribute queries | Low | 2-3 days |
| CapabilityScorer (Tier 1 + 2) | Medium | 3-4 days |
| Calibration benchmark suite | Medium | 3-4 days |
| BinaryCompatibilityChecker | High | 4-5 days |
| PTX JIT pre-warming | Medium | 2-3 days |
| WorkloadClassifier (heuristic) | Medium | 2-3 days |
| R17 integration (GpuCapabilityProvider) | Medium | 2-3 days |
| R13 integration (HEFT cost model) | Medium | 2-3 days |
| R20 integration (asymmetric bandwidth) | Low | 1-2 days |
| GPU monitoring + throttle detection | Medium | 2-3 days |
| User policy system | Low | 1-2 days |
| Runtime adaptation (Tier 3) | High | 3-4 days |
| Testing (unit + integration + benchmarks) | High | 4-5 days |
| **Total** | | **30-44 days** |

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-gpu-capability-landscape.md](./research/01-gpu-capability-landscape.md) -- GPU hardware specs
- [research/02-heterogeneous-scheduling.md](./research/02-heterogeneous-scheduling.md) -- Scheduling approaches
- [research/03-practical-mixing-scenarios.md](./research/03-practical-mixing-scenarios.md) -- Real-world scenarios
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/) -- PageTable, MigrationEngine
- [R13 CUDA Graph Interception](../R13-cuda-graph-interception/) -- HEFT consumer of capability scores
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/) -- PlacementDecision integration
- [R20 NCCL Backend](../../phase-09-collective-communication/R20-nccl-backend/) -- Asymmetric bandwidth reporting
- [R25 Cooperative Kernel Splitting](../R25-cooperative-kernel-splitting/) -- CC gating for cooperative launch
