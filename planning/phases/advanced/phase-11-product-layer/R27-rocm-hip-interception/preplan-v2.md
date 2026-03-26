# R27: ROCm/HIP Interception — Pre-Plan v2

**Created:** 2026-03-26
**Last Updated:** 2026-03-26
**Status:** Draft
**Priority:** LOW (Phase 11 — Product Layer)
**Depends On:** P7 (CUDA Completeness — proven interception pattern), R23 (Heterogeneous GPU Mixing)

## Purpose

Comprehensive pre-plan for extending OuterLink's GPU interception layer to AMD's HIP/ROCm stack. This enables AMD GPUs (RDNA, CDNA, GCN) to join the OuterLink pool alongside NVIDIA GPUs, achieving true vendor-agnostic GPU pooling. Defines the interception mechanism, API translation layer, core data structures, cross-vendor execution model, and integration points with R23 (heterogeneous GPU mixing) and R10 (memory tiering).

---

## 1. Summary and Dependencies

### 1.1 What This Enables

- AMD GPUs (RX 7900 XTX, MI250, MI300X, etc.) participate in OuterLink pools
- Mixed AMD + NVIDIA clusters: a single application can use GPUs from both vendors
- HIP-native applications run transparently across remote AMD GPUs
- CUDA applications can run on AMD GPUs via HIP's built-in CUDA compatibility layer
- Cheaper AMD consumer GPUs (RDNA) used for inference alongside NVIDIA training GPUs

### 1.2 Dependency Chain

| Dependency | What R27 Needs From It | Status |
|------------|----------------------|--------|
| P7 CUDA Completeness | Proven interception pattern (LD_PRELOAD + dlsym + cuGetProcAddress). R27 replicates this for HIP. | Foundation |
| R23 Heterogeneous GPU Mixing | `GpuProfile` struct extended for AMD GPUs. Capability scoring for cross-vendor scheduling. | Required |
| R10 Memory Tiering | Vendor-neutral page table. Memory regions must be addressable regardless of GPU vendor. | Required |
| R17 Topology Scheduling | Network cost model must handle AMD-to-NVIDIA cross-node transfers. | Required |
| R13 CUDA Graph Interception | HIP Graph API equivalent (hipGraph*) needs parallel interception. | Optional (Phase 2) |
| P1 Transport Layer | TCP transport is vendor-agnostic. No changes needed. | Ready |

### 1.3 Non-Goals (Phase 1)

- OpenCL interception (separate future work if needed)
- HSA thunk layer interception (too low-level, HIP layer is sufficient)
- Kernel binary translation (AMDGPU ISA to PTX or vice versa) — we route workloads to compatible GPUs
- HIP Graph distributed execution (deferred to R27 Phase 2, after R13 is stable)

---

## 2. ROCm/HIP Software Stack Architecture

Understanding the stack is critical for choosing the right interception layer.

```
┌─────────────────────────────────────────────────────────┐
│  Application (HIP C++ code)                             │
├─────────────────────────────────────────────────────────┤
│  HIP Runtime API  (libamdhip64.so)                      │  <── WE INTERCEPT HERE
│  hipMalloc, hipMemcpy, hipLaunchKernel, hipModuleLaunch │
├─────────────────────────────────────────────────────────┤
│  CLR — Compute Language Runtime (hipamd + rocclr)       │
│  Virtual device interface, backend abstraction          │
├─────────────────────────────────────────────────────────┤
│  ROCR-Runtime / HSA Runtime (libhsa-runtime64.so)       │
│  Agent enumeration, queue management, memory regions    │
├─────────────────────────────────────────────────────────┤
│  HSA Thunk (libhsakmt.so / ROCT)                        │
│  User-space ↔ kernel driver interface                   │
├─────────────────────────────────────────────────────────┤
│  KFD Kernel Driver (/dev/kfd)                           │
│  Kernel Fusion Driver for GPU compute access            │
└─────────────────────────────────────────────────────────┘
```

### 2.1 Why Intercept at the HIP Runtime Layer

| Layer | Pros | Cons |
|-------|------|------|
| **HIP Runtime (libamdhip64.so)** | Highest-level, closest to CUDA mapping, `hipGetProcAddress` parallel to `cuGetProcAddress`, well-documented API, portable across RDNA/CDNA/GCN | Some runtime internals bypass HIP (rare) |
| CLR (rocclr) | Could catch OpenCL too | Internal API, unstable, undocumented |
| HSA Runtime | Catches everything | Too low-level, hundreds of HSA calls per HIP call, massive surface area |
| HSA Thunk (libhsakmt) | Catches kernel-level ops | ioctl-level, would need to reverse-engineer KFD protocol |

**Decision: Intercept at the HIP Runtime layer (libamdhip64.so).** This mirrors our CUDA strategy of intercepting the Driver API (libcuda.so). The HIP API is the AMD equivalent of the CUDA Driver API — it is the highest-level API that all frameworks (PyTorch, TensorFlow, JAX) call through.

### 2.2 HIP vs CUDA: Key Architectural Differences

| Aspect | CUDA | HIP (AMD) |
|--------|------|-----------|
| Shared library | `libcuda.so` (driver), `libcudart.so` (runtime) | `libamdhip64.so` (unified) |
| Dynamic dispatch | `cuGetProcAddress` (CUDA 11.3+) | `hipGetProcAddress` (HIP 6.2+) |
| Symbol resolution | `dlsym` on `libcuda.so` | `dlsym` on `libamdhip64.so` |
| Context model | Explicit contexts (`CUcontext`), context stack | Implicit primary context per device, `hipCtx_t` for CUDA compat |
| Module format | PTX (IR) + cubin (native) | AMDGPU ISA (.co code objects) + LLVM bitcode |
| Compute capability | SM versions (7.5, 8.0, 8.6, 9.0) | GCN arch names (gfx906, gfx908, gfx90a, gfx1100, etc.) |
| Warp/wavefront | 32 threads (warp) | 64 threads (wavefront, GCN/CDNA) or 32 (RDNA wave32 mode) |
| Compute units | Streaming Multiprocessors (SMs) | Compute Units (CUs) — 4 SIMDs per CU |
| Management | NVML (`libnvidia-ml.so`) | ROCm SMI (`librocm_smi64.so`) / AMD SMI |
| Graph API | CUDA Graphs (`cuGraphCreate`, `cuStreamBeginCapture`) | HIP Graphs (`hipGraphCreate`, `hipStreamBeginCapture`) — Beta |

---

## 3. HIP-to-CUDA Function Mapping Table

The following table maps the ~120 core HIP functions R27 must intercept to their CUDA equivalents (which OuterLink already intercepts). This mapping is the foundation of the translation layer.

### 3.1 Initialization and Device Management

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipInit` | `cuInit` | Initialize AMD backend, register with OuterLink server |
| `hipDriverGetVersion` | `cuDriverGetVersion` | Return OuterLink-reported HIP version |
| `hipRuntimeGetVersion` | `cudaRuntimeGetVersion` | Return OuterLink-reported runtime version |
| `hipGetDeviceCount` | `cuDeviceGetCount` | Return pool GPU count (AMD subset or all) |
| `hipGetDevice` | `cuDeviceGet` | Return OuterLink virtual device ordinal |
| `hipSetDevice` | `cuCtxSetCurrent` | Set active GPU context in OuterLink |
| `hipGetDeviceProperties` | `cuDeviceGetAttribute` (many) | Return `hipDeviceProp_t` from `AmdGpuProfile` |
| `hipDeviceGetAttribute` | `cuDeviceGetAttribute` | Translate attribute enum, return from profile |
| `hipGetDeviceFlags` | `cuCtxGetFlags` | Forward to server |
| `hipSetDeviceFlags` | `cuCtxSetFlags` | Forward to server |
| `hipDeviceSynchronize` | `cuCtxSynchronize` | Synchronize remote GPU |
| `hipDeviceReset` | `cuDevicePrimaryCtxReset` | Reset remote device state |
| `hipDeviceGetName` | `cuDeviceGetName` | Return name from profile |
| `hipDeviceTotalMem` | `cuDeviceTotalMem` | Return VRAM from profile |
| `hipDeviceGetPCIBusId` | `cuDeviceGetPCIBusId` | Return PCI info from profile |
| `hipDeviceGetByPCIBusId` | `cuDeviceGetByPCIBusId` | Lookup by PCI address |
| `hipDeviceCanAccessPeer` | `cuDeviceCanAccessPeer` | Check OuterLink connectivity |
| `hipDeviceEnablePeerAccess` | `cuCtxEnablePeerAccess` | Enable OuterLink cross-GPU path |
| `hipDeviceDisablePeerAccess` | `cuCtxDisablePeerAccess` | Disable cross-GPU path |

### 3.2 Context Management

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipCtxCreate` | `cuCtxCreate_v2` | Create remote context on AMD GPU |
| `hipCtxDestroy` | `cuCtxDestroy_v2` | Destroy remote context |
| `hipCtxSetCurrent` | `cuCtxSetCurrent` | Switch active context |
| `hipCtxGetCurrent` | `cuCtxGetCurrent` | Return current context handle |
| `hipCtxGetDevice` | `cuCtxGetDevice` | Return device for context |
| `hipCtxSynchronize` | `cuCtxSynchronize` | Synchronize context |
| `hipCtxPushCurrent` | `cuCtxPushCurrent` | Push to context stack |
| `hipCtxPopCurrent` | `cuCtxPopCurrent` | Pop from context stack |
| `hipDevicePrimaryCtxRetain` | `cuDevicePrimaryCtxRetain` | Retain primary context |
| `hipDevicePrimaryCtxRelease` | `cuDevicePrimaryCtxRelease` | Release primary context |
| `hipDevicePrimaryCtxGetState` | `cuDevicePrimaryCtxGetState` | Query primary context state |
| `hipDevicePrimaryCtxSetFlags` | `cuDevicePrimaryCtxSetFlags` | Set primary context flags |
| `hipDevicePrimaryCtxReset` | `cuDevicePrimaryCtxReset` | Reset primary context |

### 3.3 Memory Management

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipMalloc` | `cuMemAlloc_v2` | Allocate on remote AMD GPU via R10 |
| `hipFree` | `cuMemFree_v2` | Free remote allocation |
| `hipMemcpy` | `cuMemcpy` | Copy via OuterLink transport |
| `hipMemcpyAsync` | `cuMemcpyAsync` | Async copy via transport |
| `hipMemcpyHtoD` | `cuMemcpyHtoD_v2` | Host-to-device transfer |
| `hipMemcpyDtoH` | `cuMemcpyDtoH_v2` | Device-to-host transfer |
| `hipMemcpyDtoD` | `cuMemcpyDtoD_v2` | Device-to-device (may cross vendors) |
| `hipMemcpyHtoDAsync` | `cuMemcpyHtoDAsync_v2` | Async host-to-device |
| `hipMemcpyDtoHAsync` | `cuMemcpyDtoHAsync_v2` | Async device-to-host |
| `hipMemset` | `cuMemsetD8` | Memset on remote GPU |
| `hipMemsetAsync` | `cuMemsetD8Async` | Async memset |
| `hipMemsetD32` | `cuMemsetD32` | 32-bit memset |
| `hipMemsetD16` | `cuMemsetD16` | 16-bit memset |
| `hipMemGetInfo` | `cuMemGetInfo_v2` | Query free/total memory |
| `hipHostMalloc` | `cuMemAllocHost` | Allocate pinned host memory |
| `hipHostFree` | `cuMemFreeHost` | Free pinned host memory |
| `hipHostRegister` | `cuMemHostRegister` | Register host memory |
| `hipHostUnregister` | `cuMemHostUnregister` | Unregister host memory |
| `hipHostGetDevicePointer` | `cuMemHostGetDevicePointer` | Get device pointer for host mem |
| `hipHostGetFlags` | `cuMemHostGetFlags` | Query host allocation flags |
| `hipMallocManaged` | `cuMemAllocManaged` | Managed memory allocation |
| `hipMemPrefetchAsync` | `cuMemPrefetchAsync` | Prefetch managed memory |
| `hipMemAdvise` | `cuMemAdvise` | Memory advisory hints |
| `hipMemRangeGetAttribute` | `cuMemRangeGetAttribute` | Query range attributes |
| `hipMallocAsync` | `cuMemAllocAsync` | Pool-based async alloc |
| `hipFreeAsync` | `cuMemFreeAsync` | Pool-based async free |
| `hipMemPoolCreate` | `cuMemPoolCreate` | Create memory pool |
| `hipMemPoolDestroy` | `cuMemPoolDestroy` | Destroy memory pool |
| `hipMemPoolGetAttribute` | `cuMemPoolGetAttribute` | Query pool attribute |
| `hipMemPoolSetAttribute` | `cuMemPoolSetAttribute` | Set pool attribute |
| `hipMemPoolTrimTo` | `cuMemPoolTrimTo` | Trim pool to size |
| `hipMallocFromPoolAsync` | `cuMemAllocFromPoolAsync` | Alloc from specific pool |
| `hipMemGetAddressRange` | `cuMemGetAddressRange` | Query allocation range |

### 3.4 Module and Kernel Management

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipModuleLoad` | `cuModuleLoad` | Load AMDGPU code object on remote GPU |
| `hipModuleLoadData` | `cuModuleLoadData` | Load module from memory |
| `hipModuleLoadDataEx` | `cuModuleLoadDataEx` | Load with options (no-op on AMD) |
| `hipModuleUnload` | `cuModuleUnload` | Unload module |
| `hipModuleGetFunction` | `cuModuleGetFunction` | Get kernel function handle |
| `hipModuleGetGlobal` | `cuModuleGetGlobal` | Get global variable |
| `hipModuleLaunchKernel` | `cuLaunchKernel` | Launch kernel on remote AMD GPU |
| `hipLaunchKernel` | `cudaLaunchKernel` | Runtime kernel launch |
| `hipFuncGetAttribute` | `cuFuncGetAttribute` | Query function attribute |
| `hipFuncSetAttribute` | `cuFuncSetAttribute` | Set function attribute |
| `hipFuncSetCacheConfig` | `cuFuncSetCacheConfig` | Set cache configuration |
| `hipFuncSetSharedMemConfig` | `cuFuncSetSharedMemConfig` | Set shared mem config |
| `hipOccupancyMaxActiveBlocksPerMultiprocessor` | `cuOccupancyMax*` | Occupancy calculation |
| `hipOccupancyMaxPotentialBlockSize` | N/A (runtime only) | Compute optimal block size |
| `hipGetProcAddress` | `cuGetProcAddress` | Dynamic function resolution (intercept required) |

### 3.5 Stream and Event Management

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipStreamCreate` | `cuStreamCreate` | Create remote stream |
| `hipStreamCreateWithFlags` | `cuStreamCreateWithFlags` | Create with flags |
| `hipStreamCreateWithPriority` | `cuStreamCreateWithPriority` | Create with priority |
| `hipStreamDestroy` | `cuStreamDestroy` | Destroy remote stream |
| `hipStreamSynchronize` | `cuStreamSynchronize` | Synchronize remote stream |
| `hipStreamWaitEvent` | `cuStreamWaitEvent` | Wait on event in stream |
| `hipStreamQuery` | `cuStreamQuery` | Query stream completion |
| `hipStreamAddCallback` | `cuStreamAddCallback` | Register callback (client-side) |
| `hipLaunchHostFunc` | `cuLaunchHostFunc` | Host callback in stream |
| `hipEventCreate` | `cuEventCreate` | Create remote event |
| `hipEventCreateWithFlags` | `cuEventCreateWithFlags` | Create with flags |
| `hipEventDestroy` | `cuEventDestroy` | Destroy remote event |
| `hipEventRecord` | `cuEventRecord` | Record event in stream |
| `hipEventSynchronize` | `cuEventSynchronize` | Synchronize event |
| `hipEventElapsedTime` | `cuEventElapsedTime` | Compute elapsed time |
| `hipEventQuery` | `cuEventQuery` | Query event completion |

### 3.6 Error Handling

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipGetLastError` | `cudaGetLastError` | Return last error (client-side) |
| `hipPeekAtLastError` | `cudaPeekAtLastError` | Peek at last error |
| `hipGetErrorName` | `cuGetErrorName` | Translate HIP error code to name |
| `hipGetErrorString` | `cuGetErrorString` | Translate HIP error code to string |

### 3.7 HIP Graph API (Phase 2)

| HIP Function | CUDA Equivalent | R27 Action |
|-------------|----------------|------------|
| `hipGraphCreate` | `cuGraphCreate` | Create graph on remote GPU |
| `hipGraphDestroy` | `cuGraphDestroy` | Destroy graph |
| `hipGraphInstantiate` | `cuGraphInstantiate` | Instantiate executable graph |
| `hipGraphLaunch` | `cuGraphLaunch` | Launch graph |
| `hipGraphExecDestroy` | `cuGraphExecDestroy` | Destroy executable graph |
| `hipStreamBeginCapture` | `cuStreamBeginCapture` | Begin stream capture |
| `hipStreamEndCapture` | `cuStreamEndCapture` | End stream capture |
| `hipStreamIsCapturing` | `cuStreamIsCapturing` | Query capture state |
| `hipGraphAddKernelNode` | `cuGraphAddKernelNode` | Add kernel node |
| `hipGraphAddMemcpyNode` | `cuGraphAddMemcpyNode` | Add memcpy node |
| `hipGraphAddMemsetNode` | `cuGraphAddMemsetNode` | Add memset node |
| `hipGraphAddHostNode` | `cuGraphAddHostNode` | Add host callback node |
| `hipGraphAddChildGraphNode` | `cuGraphAddChildGraphNode` | Add child graph |
| `hipGraphAddEmptyNode` | `cuGraphAddEmptyNode` | Add empty node |
| `hipGraphGetNodes` | `cuGraphGetNodes` | Get all nodes |
| `hipGraphGetEdges` | `cuGraphGetEdges` | Get all edges |
| `hipGraphNodeGetType` | `cuGraphNodeGetType` | Query node type |
| `hipGraphExecUpdate` | `cuGraphExecUpdate` | Update executable graph |

**Total intercepted functions: ~120 Phase 1, ~140 with Phase 2 graphs.**

---

## 4. Core Data Structures

### 4.1 GPU Vendor Abstraction

```rust
/// GPU vendor identifier. Used throughout OuterLink to dispatch
/// vendor-specific operations and validate binary compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA GPU (CUDA/PTX).
    Nvidia,
    /// AMD GPU (HIP/ROCm/AMDGPU ISA).
    Amd,
}

/// AMD GPU architecture family. Determines wavefront size, instruction set,
/// and feature availability. Parallel to NVIDIA's compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdArchFamily {
    /// GCN (Graphics Core Next) — Polaris, Vega.
    /// Wave64 only. 4x SIMD16 per CU.
    Gcn,
    /// CDNA (Compute DNA) — MI100, MI250, MI300.
    /// Wave64. Matrix ALUs (MFMA). Datacenter compute.
    Cdna,
    /// RDNA (Radeon DNA) — RX 5000/6000/7000/9000 series.
    /// Wave32 (native) + Wave64 (compat). Work Group Processors (WGPs).
    Rdna,
}

/// AMD GPU architecture target, identified by gfx ISA version.
/// This is the AMD equivalent of NVIDIA's SM version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AmdGpuArch {
    /// GCN arch name as reported by hipDeviceProp_t.gcnArchName.
    /// Examples: "gfx906" (MI50/Vega 20), "gfx90a" (MI210/MI250),
    /// "gfx940" (MI300A), "gfx942" (MI300X), "gfx1100" (RX 7900 XTX),
    /// "gfx1201" (RX 9070 XT).
    pub gfx_name: String,
    /// Parsed major version (first digit after gfx).
    /// 9xx = GCN/CDNA, 10xx = RDNA1, 11xx = RDNA3, 12xx = RDNA4.
    pub major: u32,
    /// Parsed minor version.
    pub minor: u32,
    /// Parsed stepping.
    pub stepping: u32,
    /// Architecture family (derived from major version).
    pub family: AmdArchFamily,
    /// Native wavefront size (32 for RDNA wave32, 64 for GCN/CDNA).
    pub native_wavefront_size: u32,
    /// Whether wave32 mode is supported (RDNA only).
    pub supports_wave32: bool,
    /// Whether matrix ALUs (MFMA) are available (CDNA only).
    pub has_matrix_alu: bool,
}

impl AmdGpuArch {
    /// Parse a gfxXYZ[S] string into an AmdGpuArch.
    /// Examples: "gfx906" -> (9, 0, 6), "gfx1100" -> (11, 0, 0).
    pub fn from_gfx_name(name: &str) -> Option<Self> {
        // Strip "gfx" prefix, parse remaining digits.
        let digits = name.strip_prefix("gfx")?;
        let num: u32 = digits.parse().ok()?;

        // gfxMMS or gfxMMSS format:
        // For 3-digit: M=major, M=minor, S=stepping (e.g., gfx906)
        // For 4-digit: MM=major, S=minor, S=stepping (e.g., gfx1100)
        let (major, minor, stepping) = if num >= 1000 {
            (num / 100, (num / 10) % 10, num % 10)
        } else {
            (num / 100, (num / 10) % 10, num % 10)
        };

        let family = match major {
            7 | 8 => AmdArchFamily::Gcn,
            9 => {
                // gfx9xx: GCN for gfx900-gfx906, CDNA for gfx908+
                if minor >= 0 && stepping >= 8 || minor > 0 {
                    AmdArchFamily::Cdna
                } else {
                    AmdArchFamily::Gcn
                }
            }
            10 => AmdArchFamily::Rdna, // RDNA 1 (gfx1010, gfx1030, etc.)
            11 => AmdArchFamily::Rdna, // RDNA 3 (gfx1100, gfx1101, etc.)
            12 => AmdArchFamily::Rdna, // RDNA 4 (gfx1200, gfx1201, etc.)
            _ => AmdArchFamily::Gcn,   // Unknown, assume GCN
        };

        let native_wavefront_size = match family {
            AmdArchFamily::Rdna => 32,
            _ => 64,
        };

        Some(Self {
            gfx_name: name.to_string(),
            major,
            minor,
            stepping,
            family,
            native_wavefront_size,
            supports_wave32: matches!(family, AmdArchFamily::Rdna),
            has_matrix_alu: matches!(family, AmdArchFamily::Cdna),
        })
    }

    /// Check if a code object compiled for `target_arch` can run on this GPU.
    /// AMD GPU ISA is NOT forward-compatible like CUDA PTX.
    /// The gfx target must match exactly (with some exceptions for
    /// compatible steppings within the same family).
    pub fn is_compatible_with(&self, target_gfx: &str) -> bool {
        // Exact match always works.
        if self.gfx_name == target_gfx {
            return true;
        }
        // Some architectures are binary-compatible:
        // gfx900 == gfx902 (stepping difference only)
        // gfx1030 == gfx1031 == gfx1032 (RDNA2 variants)
        // gfx1100 == gfx1101 (RDNA3 variants)
        // This list must be maintained as new GPUs are released.
        let compat_groups: &[&[&str]] = &[
            &["gfx900", "gfx902"],
            &["gfx1010", "gfx1011", "gfx1012"],
            &["gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"],
            &["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
            &["gfx1150", "gfx1151"],
            &["gfx1200", "gfx1201"],
        ];
        for group in compat_groups {
            if group.contains(&self.gfx_name.as_str()) && group.contains(&target_gfx) {
                return true;
            }
        }
        false
    }
}
```

### 4.2 AMD GPU Profile (Extension of R23's GpuProfile)

```rust
/// Complete hardware profile for an AMD GPU in the OuterLink pool.
/// Parallel to `GpuProfile` (which is NVIDIA-centric), this captures
/// AMD-specific hardware attributes, calibration data, and runtime state.
///
/// Both `GpuProfile` (NVIDIA) and `AmdGpuProfile` (AMD) are stored in
/// the pool's GPU registry and exposed through R23's `GpuCapabilityProvider`.
#[derive(Debug, Clone)]
pub struct AmdGpuProfile {
    /// OuterLink-assigned GPU identifier (unique across the entire pool,
    /// shared namespace with NVIDIA GPUs).
    pub gpu_id: GpuId,
    /// Node (PC) this GPU belongs to.
    pub node_id: NodeId,
    /// Vendor tag (always Amd for this struct).
    pub vendor: GpuVendor,

    // --- Static hardware attributes (set once at registration) ---

    /// GPU model name (e.g., "AMD Radeon RX 7900 XTX").
    pub name: String,
    /// Architecture target (parsed from hipDeviceProp_t.gcnArchName).
    pub arch: AmdGpuArch,
    /// Number of Compute Units (CUs). Equivalent to NVIDIA's SM count.
    pub compute_unit_count: u32,
    /// Stream processors per CU. Typically 64 for GCN/CDNA, 128 for RDNA WGP.
    pub stream_processors_per_cu: u32,
    /// Total stream processors (compute_unit_count * stream_processors_per_cu).
    pub total_stream_processors: u32,
    /// Matrix ALU count (0 for non-CDNA, >0 for MI-series).
    pub matrix_alu_count: u32,
    /// Matrix ALU generation (None for non-CDNA, Some for MI-series).
    /// Equivalent to NVIDIA's Tensor Core generation.
    pub matrix_alu_gen: Option<u32>,
    /// Native wavefront size (32 or 64).
    pub wavefront_size: u32,
    /// Total VRAM in bytes.
    pub vram_total_bytes: u64,
    /// VRAM type (e.g., GDDR6, HBM2e, HBM3).
    pub vram_type: String,
    /// Theoretical memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Theoretical FP32 peak TFLOPS.
    pub fp32_tflops: f64,
    /// Theoretical FP16 peak TFLOPS.
    pub fp16_tflops: f64,
    /// Theoretical BF16 peak TFLOPS (CDNA2+ and RDNA3+).
    pub bf16_tflops: f64,
    /// Theoretical INT8 peak TOPS (CDNA2+ and RDNA3+).
    pub int8_tops: f64,
    /// PCIe generation (3, 4, 5).
    pub pcie_gen: u32,
    /// PCIe link width (x8, x16).
    pub pcie_width: u32,
    /// Measured PCIe bandwidth in GB/s.
    pub pcie_bandwidth_gbps: f64,
    /// Number of async copy engines (SDMA engines on AMD).
    pub sdma_engine_count: u32,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: u32,
    /// Infinity Cache size in bytes (RDNA2+, 0 otherwise).
    pub infinity_cache_bytes: u64,
    /// GPU boost clock in MHz.
    pub boost_clock_mhz: u32,
    /// TDP (thermal design power) in watts.
    pub tdp_watts: u32,

    // --- Precision support flags ---

    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,  // CDNA3 (MI300)
    pub supports_int8: bool,
    /// Packed math (two FP16 ops per SP clock). RDNA2+ and CDNA.
    pub supports_packed_math: bool,

    // --- Driver info ---

    /// ROCm version string (e.g., "6.3.0").
    pub rocm_version: String,
    /// HIP runtime version (e.g., 60300000 for HIP 6.3).
    pub hip_runtime_version: u32,
    /// HSA runtime version.
    pub hsa_version: String,
    /// AMDGPU kernel driver version.
    pub kernel_driver_version: String,

    // --- Calibration benchmarks (set during registration) ---

    /// Measured FP32 GFLOPS (from SGEMM benchmark).
    pub measured_fp32_gflops: f64,
    /// Measured memory bandwidth in GB/s (from stream copy benchmark).
    pub measured_memory_bw_gbps: f64,
    /// Measured host-to-device bandwidth in GB/s.
    pub measured_h2d_bw_gbps: f64,
    /// Measured device-to-host bandwidth in GB/s.
    pub measured_d2h_bw_gbps: f64,
    /// Measured Matrix ALU throughput in TFLOPS (FP16 matmul).
    /// Equivalent to NVIDIA's Tensor Core benchmark.
    pub measured_matrix_tflops: Option<f64>,

    // --- Dynamic state (updated periodically via ROCm SMI / AMD SMI) ---

    /// Current free VRAM in bytes.
    pub vram_free_bytes: u64,
    /// Current GPU utilization (0.0 - 1.0).
    pub utilization: f64,
    /// Current GPU temperature in Celsius (edge sensor).
    pub temperature_c: u32,
    /// Current clock speed in MHz.
    pub current_clock_mhz: u32,
    /// Current power draw in watts.
    pub power_draw_watts: u32,
    /// Whether the GPU is thermally throttling.
    pub is_throttling: bool,

    // --- Computed scores (updated when dynamic state changes) ---

    /// Normalized capability scores per workload class.
    /// Uses same WorkloadScores struct as NVIDIA GPUs.
    pub capability_scores: WorkloadScores,
    /// GPU Equivalent Units (single number for fairness/quota).
    pub geu: f64,
}
```

### 4.3 Unified GPU Profile Enum

```rust
/// A GPU in the OuterLink pool, regardless of vendor.
/// This is the type stored in the pool registry and used by
/// R23's capability scorer, R17's topology scheduler, and
/// R13's HEFT partitioner.
#[derive(Debug, Clone)]
pub enum UnifiedGpuProfile {
    Nvidia(GpuProfile),
    Amd(AmdGpuProfile),
}

impl UnifiedGpuProfile {
    /// Get the GPU ID (unique across all vendors).
    pub fn gpu_id(&self) -> GpuId {
        match self {
            Self::Nvidia(p) => p.gpu_id,
            Self::Amd(p) => p.gpu_id,
        }
    }

    /// Get the node ID.
    pub fn node_id(&self) -> NodeId {
        match self {
            Self::Nvidia(p) => p.node_id,
            Self::Amd(p) => p.node_id,
        }
    }

    /// Get the vendor.
    pub fn vendor(&self) -> GpuVendor {
        match self {
            Self::Nvidia(_) => GpuVendor::Nvidia,
            Self::Amd(_) => GpuVendor::Amd,
        }
    }

    /// Get FP32 peak TFLOPS (for capability scoring).
    pub fn fp32_tflops(&self) -> f64 {
        match self {
            Self::Nvidia(p) => p.fp32_tflops,
            Self::Amd(p) => p.fp32_tflops,
        }
    }

    /// Get memory bandwidth in GB/s.
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::Nvidia(p) => p.memory_bandwidth_gbps,
            Self::Amd(p) => p.memory_bandwidth_gbps,
        }
    }

    /// Get total VRAM in bytes.
    pub fn vram_total_bytes(&self) -> u64 {
        match self {
            Self::Nvidia(p) => p.vram_total_bytes,
            Self::Amd(p) => p.vram_total_bytes,
        }
    }

    /// Get free VRAM in bytes.
    pub fn vram_free_bytes(&self) -> u64 {
        match self {
            Self::Nvidia(p) => p.vram_free_bytes,
            Self::Amd(p) => p.vram_free_bytes,
        }
    }

    /// Get capability scores.
    pub fn capability_scores(&self) -> &WorkloadScores {
        match self {
            Self::Nvidia(p) => &p.capability_scores,
            Self::Amd(p) => &p.capability_scores,
        }
    }

    /// Get GEU value.
    pub fn geu(&self) -> f64 {
        match self {
            Self::Nvidia(p) => p.geu,
            Self::Amd(p) => p.geu,
        }
    }

    /// Check if a kernel binary is compatible with this GPU.
    /// NVIDIA kernels cannot run on AMD GPUs and vice versa.
    /// Cross-vendor execution requires the translation layer.
    pub fn is_binary_compatible(&self, binary_format: BinaryFormat) -> bool {
        match (self, binary_format) {
            (Self::Nvidia(_), BinaryFormat::Ptx | BinaryFormat::Cubin) => true,
            (Self::Amd(p), BinaryFormat::AmdgpuIsa(ref target)) => {
                p.arch.is_compatible_with(target)
            }
            _ => false, // Cross-vendor binary execution is not possible
        }
    }

    /// Whether this GPU is currently available for scheduling.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Nvidia(p) => !p.is_throttling && p.utilization < 0.95,
            Self::Amd(p) => !p.is_throttling && p.utilization < 0.95,
        }
    }
}

/// Binary format for kernel code objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryFormat {
    /// NVIDIA PTX intermediate representation.
    Ptx,
    /// NVIDIA native binary (cubin).
    Cubin,
    /// AMD GPU ISA code object for a specific gfx target.
    AmdgpuIsa(String),
    /// HIP source (can be compiled for either vendor via hiprtc/nvrtc).
    HipSource,
}
```

### 4.4 HIP Context and Handle Management

```rust
/// Client-side state for HIP interception. Mirrors OuterLinkClient
/// but for the HIP API surface.
///
/// Lives in the LD_PRELOAD library alongside (or instead of) OuterLinkClient.
/// When a process links against both libamdhip64.so and libcuda.so, both
/// clients coexist — they share a single connection to the OuterLink server
/// but maintain separate handle namespaces.
pub struct HipInterceptClient {
    /// Handle translation tables (synthetic local <-> real remote).
    /// Same HandleStore type as CUDA client, separate instance.
    pub handles: HandleStore,
    /// Server address (shared with CUDA client if both are active).
    pub server_addr: String,
    /// Connection state.
    pub connected: AtomicBool,
    /// Tokio runtime for async transport from sync FFI context.
    runtime: tokio::runtime::Runtime,
    /// Active TCP connection to the server.
    connection: std::sync::Mutex<Option<Arc<TcpTransportConnection>>>,
    /// Monotonically increasing request ID.
    next_request_id: AtomicU64,
    /// The remote HIP context handle currently active.
    pub current_remote_ctx: AtomicU64,
    /// Retry and reconnect configuration.
    retry_config: RetryConfig,
    /// Reconnect mutex.
    reconnect_in_progress: std::sync::Mutex<()>,
    /// Session ID assigned by the server.
    session_id: AtomicU64,
    /// Client-side callback registry (hipStreamAddCallback / hipLaunchHostFunc).
    pub callback_registry: Arc<CallbackRegistry>,
    /// Callback connection.
    callback_connection: std::sync::Mutex<Option<Arc<TcpTransportConnection>>>,
    /// Callback listener running flag.
    callback_listener_running: Arc<AtomicBool>,
    /// Vendor tag (always Amd, used in wire protocol to tell server
    /// which driver API to call on the remote GPU).
    pub vendor: GpuVendor,
    /// Cache of device properties fetched from server.
    /// Avoids round-trip for repeated hipGetDeviceProperties calls.
    device_props_cache: std::sync::Mutex<HashMap<i32, HipDeviceProps>>,
}

/// Cached device properties returned by hipGetDeviceProperties.
/// Populated from AmdGpuProfile on first query, then served locally.
#[derive(Debug, Clone)]
pub struct HipDeviceProps {
    /// Raw hipDeviceProp_t fields, stored as key-value pairs.
    /// Translated to hipDeviceProp_t when returned to the application.
    pub name: String,
    pub total_global_mem: u64,
    pub shared_mem_per_block: u64,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub multi_processor_count: i32,
    pub compute_mode: i32,
    pub concurrent_kernels: i32,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
    pub gcn_arch: i32,
    pub gcn_arch_name: String,
    pub max_shared_memory_per_multiprocessor: u64,
    pub arch: AmdGpuArch,
}

/// HIP module handle — represents a loaded code object on a remote AMD GPU.
/// Maps hipModule_t (opaque pointer) to an internal ID.
#[derive(Debug, Clone)]
pub struct HipModuleEntry {
    /// Synthetic local handle (returned to application).
    pub local_handle: u64,
    /// Real remote handle (on the server's AMD GPU).
    pub remote_handle: u64,
    /// gfx target this module was compiled for.
    pub target_arch: String,
    /// Functions extracted from this module.
    pub functions: HashMap<String, HipFunctionEntry>,
}

/// HIP function handle — a kernel within a loaded module.
#[derive(Debug, Clone)]
pub struct HipFunctionEntry {
    /// Synthetic local handle.
    pub local_handle: u64,
    /// Real remote handle.
    pub remote_handle: u64,
    /// Kernel name (demangled).
    pub kernel_name: String,
    /// Module this function belongs to.
    pub module_handle: u64,
    /// Attributes cache (shared mem, regs, etc.).
    pub attributes: HashMap<u32, i32>,
}
```

### 4.5 HIP-to-OuterLink Translation Layer

```rust
/// The translation layer sits between the HIP interception hooks and
/// the OuterLink wire protocol. It converts HIP API semantics into
/// the vendor-neutral protocol messages that the server understands.
///
/// Architecture:
///   Application -> dlsym("hipMalloc")
///               -> hook_hipMalloc (interpose_hip.c)
///               -> ol_hipMalloc (hip_ffi.rs)
///               -> HipTranslator::translate_malloc (this module)
///               -> Wire protocol -> Server -> real hipMalloc on AMD GPU
pub struct HipTranslator {
    /// Reference to the client for sending wire messages.
    client: &'static HipInterceptClient,
}

/// Translation result: maps HIP error codes to hipError_t values.
/// HIP error codes are a superset of CUDA error codes in most cases,
/// but some codes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipError {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,   // hipErrorOutOfMemory (vs CUDA's cudaErrorMemoryAllocation)
    NotInitialized = 3,
    Deinitialized = 4,
    InvalidDevice = 101, // Same as CUDA
    InvalidContext = 201, // Same as CUDA
    InvalidMemcpyDirection = 21,
    LaunchFailure = 719,
    NotSupported = 801,  // Same as CUDA
    Unknown = 999,       // Same as CUDA
}

/// Direction of HIP memcpy, mapping to OuterLink's MemcpyDirection.
/// HIP uses the same enum values as CUDA for the core directions,
/// but adds hipMemcpyDefault which auto-detects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4, // Auto-detect based on pointer type
}

impl HipMemcpyKind {
    /// Convert to OuterLink's MemcpyDirection.
    /// hipMemcpyDefault requires querying the pointer to determine direction.
    pub fn to_memcpy_direction(self, src_is_device: bool, dst_is_device: bool) -> MemcpyDirection {
        match self {
            Self::HostToHost => MemcpyDirection::HostToHost,
            Self::HostToDevice => MemcpyDirection::HostToDevice,
            Self::DeviceToHost => MemcpyDirection::DeviceToHost,
            Self::DeviceToDevice => MemcpyDirection::DeviceToDevice,
            Self::Default => {
                match (src_is_device, dst_is_device) {
                    (false, false) => MemcpyDirection::HostToHost,
                    (false, true) => MemcpyDirection::HostToDevice,
                    (true, false) => MemcpyDirection::DeviceToHost,
                    (true, true) => MemcpyDirection::DeviceToDevice,
                }
            }
        }
    }
}

/// Wire protocol vendor discriminator.
/// Added to MessageHeader to tell the server which driver API set to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ApiVendor {
    /// CUDA Driver API (existing).
    Cuda = 0,
    /// HIP Runtime API (R27).
    Hip = 1,
}

/// Extended message header that includes vendor discriminator.
/// Backward-compatible: existing CUDA messages default to ApiVendor::Cuda.
#[derive(Debug, Clone)]
pub struct VendorMessageHeader {
    /// Base header (message type, request ID, payload size).
    pub base: MessageHeader,
    /// Which vendor's API this message targets.
    pub api_vendor: ApiVendor,
}
```

### 4.6 Server-Side AMD GPU Executor

```rust
/// Server-side component that executes HIP API calls on a local AMD GPU.
/// Parallel to the existing CudaExecutor, but calls HIP functions instead.
///
/// The server determines which executor to use based on the ApiVendor
/// field in the message header.
pub struct HipExecutor {
    /// Device ordinal on this physical machine.
    pub device_ordinal: i32,
    /// hipDevice_t handle.
    pub device: i32,
    /// Primary context for this device.
    pub primary_ctx: u64,
    /// Module registry (loaded code objects).
    pub modules: HashMap<u64, HipModuleState>,
    /// Stream registry.
    pub streams: HashMap<u64, u64>,
    /// Event registry.
    pub events: HashMap<u64, u64>,
    /// Memory allocation tracker.
    pub allocations: HashMap<u64, AllocationInfo>,
    /// Device profile (populated at registration).
    pub profile: AmdGpuProfile,
}

/// Server-side state for a loaded HIP module.
#[derive(Debug)]
pub struct HipModuleState {
    /// Real hipModule_t handle.
    pub handle: u64,
    /// Functions extracted from the module.
    pub functions: HashMap<String, u64>,
    /// Original binary data (for potential re-loading).
    pub binary: Vec<u8>,
    /// Target architecture this was compiled for.
    pub target_arch: String,
}

/// Information about a device memory allocation.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Device pointer (real, on the AMD GPU).
    pub device_ptr: u64,
    /// Allocation size in bytes.
    pub size: u64,
    /// Which memory pool this came from (if pool-based).
    pub pool: Option<u64>,
    /// Whether this is managed memory.
    pub is_managed: bool,
}
```

---

## 5. Interception Mechanism

### 5.1 LD_PRELOAD Architecture for HIP

The HIP interception follows the exact same pattern as the existing CUDA interception, but targets `libamdhip64.so` instead of `libcuda.so`.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Application (links against libamdhip64.so)                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  dlsym(RTLD_DEFAULT, "hipMalloc")                              │ │
│  │  hipGetProcAddress("hipMalloc", ...)                           │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│  ┌──────────────────────▼──────────────────────────────────────────┐ │
│  │  interpose_hip.so (LD_PRELOAD)                                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐    │ │
│  │  │  dlsym() override:                                      │    │ │
│  │  │    if name in hip_hook_table -> return hook_fn           │    │ │
│  │  │    else -> real_dlsym(handle, name)                      │    │ │
│  │  ├─────────────────────────────────────────────────────────┤    │ │
│  │  │  hipGetProcAddress() override:                           │    │ │
│  │  │    if name in hip_hook_table -> return hook_fn           │    │ │
│  │  │    else -> real_hipGetProcAddress(name, ...)             │    │ │
│  │  ├─────────────────────────────────────────────────────────┤    │ │
│  │  │  hook_hipMalloc(ptr, size):                              │    │ │
│  │  │    return ol_hipMalloc(ptr, size)  [Rust FFI call]       │    │ │
│  │  └─────────────────────────────────────────────────────────┘    │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│  ┌──────────────────────▼──────────────────────────────────────────┐ │
│  │  outerlink-hip-client.so (Rust cdylib)                          │ │
│  │  ol_hipMalloc -> HipTranslator -> wire protocol -> TCP          │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
└─────────────────────────┼────────────────────────────────────────────┘
                          │ TCP
┌─────────────────────────▼────────────────────────────────────────────┐
│  OuterLink Server                                                     │
│  VendorMessageHeader.api_vendor == Hip                                │
│  -> HipExecutor.execute(msg)                                          │
│  -> real hipMalloc on AMD GPU                                         │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 C Interposition Layer (interpose_hip.c)

```c
/*
 * OuterLink HIP Runtime API Interposition Library
 *
 * Loaded via LD_PRELOAD to intercept HIP Runtime API calls.
 * Mirrors interpose.c (CUDA) but targets libamdhip64.so.
 *
 * Two entry points intercepted:
 *   1. dlsym()             -- classic dynamic symbol resolution
 *   2. hipGetProcAddress() -- HIP 6.2+ driver entry point API
 *
 * Architecture:
 *   Application -> dlsym("hipMalloc")
 *               -> returns hook_hipMalloc (this file)
 *               -> calls ol_hipMalloc (Rust FFI)
 *               -> serializes + sends to remote OuterLink server
 */

// Hook table structure (same pattern as interpose.c)
static const hook_entry_t hip_hook_table[] = {
    /* Init */
    { "hipInit",                     (void *)hook_hipInit },
    { "hipDriverGetVersion",         (void *)hook_hipDriverGetVersion },
    { "hipRuntimeGetVersion",        (void *)hook_hipRuntimeGetVersion },

    /* Device */
    { "hipGetDeviceCount",           (void *)hook_hipGetDeviceCount },
    { "hipGetDevice",                (void *)hook_hipGetDevice },
    { "hipSetDevice",                (void *)hook_hipSetDevice },
    { "hipGetDeviceProperties",      (void *)hook_hipGetDeviceProperties },
    { "hipDeviceGetAttribute",       (void *)hook_hipDeviceGetAttribute },
    { "hipDeviceSynchronize",        (void *)hook_hipDeviceSynchronize },
    { "hipDeviceReset",              (void *)hook_hipDeviceReset },
    { "hipDeviceGetName",            (void *)hook_hipDeviceGetName },
    { "hipDeviceTotalMem",           (void *)hook_hipDeviceTotalMem },
    { "hipDeviceCanAccessPeer",      (void *)hook_hipDeviceCanAccessPeer },
    { "hipDeviceEnablePeerAccess",   (void *)hook_hipDeviceEnablePeerAccess },
    { "hipDeviceDisablePeerAccess",  (void *)hook_hipDeviceDisablePeerAccess },

    /* Context */
    { "hipCtxCreate",                (void *)hook_hipCtxCreate },
    { "hipCtxDestroy",               (void *)hook_hipCtxDestroy },
    { "hipCtxSetCurrent",            (void *)hook_hipCtxSetCurrent },
    { "hipCtxGetCurrent",            (void *)hook_hipCtxGetCurrent },
    { "hipCtxGetDevice",             (void *)hook_hipCtxGetDevice },
    { "hipCtxSynchronize",           (void *)hook_hipCtxSynchronize },
    { "hipCtxPushCurrent",           (void *)hook_hipCtxPushCurrent },
    { "hipCtxPopCurrent",            (void *)hook_hipCtxPopCurrent },
    { "hipDevicePrimaryCtxRetain",   (void *)hook_hipDevicePrimaryCtxRetain },
    { "hipDevicePrimaryCtxRelease",  (void *)hook_hipDevicePrimaryCtxRelease },
    { "hipDevicePrimaryCtxGetState", (void *)hook_hipDevicePrimaryCtxGetState },
    { "hipDevicePrimaryCtxSetFlags", (void *)hook_hipDevicePrimaryCtxSetFlags },
    { "hipDevicePrimaryCtxReset",    (void *)hook_hipDevicePrimaryCtxReset },

    /* Memory */
    { "hipMalloc",                   (void *)hook_hipMalloc },
    { "hipFree",                     (void *)hook_hipFree },
    { "hipMemcpy",                   (void *)hook_hipMemcpy },
    { "hipMemcpyAsync",              (void *)hook_hipMemcpyAsync },
    { "hipMemcpyHtoD",               (void *)hook_hipMemcpyHtoD },
    { "hipMemcpyDtoH",               (void *)hook_hipMemcpyDtoH },
    { "hipMemcpyDtoD",               (void *)hook_hipMemcpyDtoD },
    { "hipMemcpyHtoDAsync",          (void *)hook_hipMemcpyHtoDAsync },
    { "hipMemcpyDtoHAsync",          (void *)hook_hipMemcpyDtoHAsync },
    { "hipMemset",                   (void *)hook_hipMemset },
    { "hipMemsetAsync",              (void *)hook_hipMemsetAsync },
    { "hipMemsetD32",                (void *)hook_hipMemsetD32 },
    { "hipMemsetD16",                (void *)hook_hipMemsetD16 },
    { "hipMemGetInfo",               (void *)hook_hipMemGetInfo },
    { "hipHostMalloc",               (void *)hook_hipHostMalloc },
    { "hipHostFree",                 (void *)hook_hipHostFree },
    { "hipHostRegister",             (void *)hook_hipHostRegister },
    { "hipHostUnregister",           (void *)hook_hipHostUnregister },
    { "hipHostGetDevicePointer",     (void *)hook_hipHostGetDevicePointer },
    { "hipHostGetFlags",             (void *)hook_hipHostGetFlags },
    { "hipMallocManaged",            (void *)hook_hipMallocManaged },
    { "hipMemPrefetchAsync",         (void *)hook_hipMemPrefetchAsync },
    { "hipMemAdvise",                (void *)hook_hipMemAdvise },
    { "hipMallocAsync",              (void *)hook_hipMallocAsync },
    { "hipFreeAsync",                (void *)hook_hipFreeAsync },
    { "hipMemPoolCreate",            (void *)hook_hipMemPoolCreate },
    { "hipMemPoolDestroy",           (void *)hook_hipMemPoolDestroy },

    /* Module */
    { "hipModuleLoad",               (void *)hook_hipModuleLoad },
    { "hipModuleLoadData",           (void *)hook_hipModuleLoadData },
    { "hipModuleLoadDataEx",         (void *)hook_hipModuleLoadDataEx },
    { "hipModuleUnload",             (void *)hook_hipModuleUnload },
    { "hipModuleGetFunction",        (void *)hook_hipModuleGetFunction },
    { "hipModuleGetGlobal",          (void *)hook_hipModuleGetGlobal },
    { "hipModuleLaunchKernel",       (void *)hook_hipModuleLaunchKernel },
    { "hipLaunchKernel",             (void *)hook_hipLaunchKernel },
    { "hipFuncGetAttribute",         (void *)hook_hipFuncGetAttribute },
    { "hipFuncSetAttribute",         (void *)hook_hipFuncSetAttribute },
    { "hipFuncSetCacheConfig",       (void *)hook_hipFuncSetCacheConfig },
    { "hipFuncSetSharedMemConfig",   (void *)hook_hipFuncSetSharedMemConfig },

    /* Stream */
    { "hipStreamCreate",             (void *)hook_hipStreamCreate },
    { "hipStreamCreateWithFlags",    (void *)hook_hipStreamCreateWithFlags },
    { "hipStreamCreateWithPriority", (void *)hook_hipStreamCreateWithPriority },
    { "hipStreamDestroy",            (void *)hook_hipStreamDestroy },
    { "hipStreamSynchronize",        (void *)hook_hipStreamSynchronize },
    { "hipStreamWaitEvent",          (void *)hook_hipStreamWaitEvent },
    { "hipStreamQuery",              (void *)hook_hipStreamQuery },
    { "hipStreamAddCallback",        (void *)hook_hipStreamAddCallback },
    { "hipLaunchHostFunc",           (void *)hook_hipLaunchHostFunc },

    /* Event */
    { "hipEventCreate",              (void *)hook_hipEventCreate },
    { "hipEventCreateWithFlags",     (void *)hook_hipEventCreateWithFlags },
    { "hipEventDestroy",             (void *)hook_hipEventDestroy },
    { "hipEventRecord",              (void *)hook_hipEventRecord },
    { "hipEventSynchronize",         (void *)hook_hipEventSynchronize },
    { "hipEventElapsedTime",         (void *)hook_hipEventElapsedTime },
    { "hipEventQuery",               (void *)hook_hipEventQuery },

    /* Error */
    { "hipGetLastError",             (void *)hook_hipGetLastError },
    { "hipPeekAtLastError",          (void *)hook_hipPeekAtLastError },
    { "hipGetErrorName",             (void *)hook_hipGetErrorName },
    { "hipGetErrorString",           (void *)hook_hipGetErrorString },

    /* Occupancy */
    { "hipOccupancyMaxActiveBlocksPerMultiprocessor",
                                     (void *)hook_hipOccupancyMaxActiveBlocksPerMultiprocessor },
    { "hipOccupancyMaxPotentialBlockSize",
                                     (void *)hook_hipOccupancyMaxPotentialBlockSize },

    /* Dynamic dispatch (critical: must intercept to prevent bypass) */
    { "hipGetProcAddress",           (void *)hook_hipGetProcAddress },

    /* Sentinel */
    { NULL, NULL },
};
```

### 5.3 hipGetProcAddress Interception

This is the most critical interception point. Like CUDA's `cuGetProcAddress`, HIP's `hipGetProcAddress` (introduced in HIP 6.2) allows applications to dynamically resolve function pointers at runtime, bypassing normal `dlsym` resolution. If we don't intercept this, applications using it will get direct pointers to the real `libamdhip64.so` functions, bypassing our hooks.

```c
/*
 * hipGetProcAddress interception.
 *
 * When an application calls hipGetProcAddress("hipMalloc", &funcPtr, ...),
 * we check our hook table. If the function is one we intercept, we return
 * our hook pointer instead of the real one.
 *
 * This is identical to our cuGetProcAddress interception strategy.
 */
static hipError_t hook_hipGetProcAddress(
    const char *symbol,
    void **pfn,
    int hipVersion,
    uint64_t flags,
    void *symbolStatus  /* hipDriverProcAddressQueryResult* */
) {
    ensure_init();

    /* Check our hook table first */
    void *hook = lookup_hook(symbol);
    if (hook) {
        *pfn = hook;
        if (symbolStatus) {
            *(int *)symbolStatus = 0; /* HIP_GET_PROC_ADDRESS_SUCCESS */
        }
        return 0; /* hipSuccess */
    }

    /* Not intercepted -- forward to real hipGetProcAddress */
    typedef hipError_t (*real_fn_t)(const char *, void **, int, uint64_t, void *);
    static real_fn_t real_fn = NULL;
    if (!real_fn) {
        real_fn = (real_fn_t)real_dlsym(RTLD_NEXT, "hipGetProcAddress");
    }
    if (real_fn) {
        return real_fn(symbol, pfn, hipVersion, flags, symbolStatus);
    }

    /* hipGetProcAddress itself not found -- HIP not installed? */
    return 1; /* hipErrorInvalidValue */
}
```

### 5.4 Dual-Vendor LD_PRELOAD

When both CUDA and HIP applications need to be intercepted, or when a single application uses both (e.g., PyTorch with both backends), both interposition libraries can be loaded simultaneously:

```bash
# HIP-only application
LD_PRELOAD=./outerlink_hip_interpose.so OUTERLINK_SERVER=10.0.0.1:14833 ./my_hip_app

# CUDA-only application (existing)
LD_PRELOAD=./outerlink_cuda_interpose.so OUTERLINK_SERVER=10.0.0.1:14833 ./my_cuda_app

# Mixed application (both CUDA and HIP)
LD_PRELOAD="./outerlink_cuda_interpose.so:./outerlink_hip_interpose.so" \
    OUTERLINK_SERVER=10.0.0.1:14833 ./my_mixed_app
```

The two interposition libraries are independent — they intercept different symbol namespaces (`cu*` vs `hip*`) and share a single `OuterLinkClient` connection to the server.

---

## 6. Key Algorithms

### 6.1 HIP-to-OuterLink Memory Management Translation

```rust
/// Translate hipMalloc into an OuterLink memory allocation.
///
/// Flow:
/// 1. Application calls hipMalloc(&ptr, size)
/// 2. Hook calls ol_hipMalloc(ptr, size)
/// 3. We send AllocRequest to server with vendor=Hip
/// 4. Server's HipExecutor calls real hipMalloc on remote AMD GPU
/// 5. Server returns (real_device_ptr, allocation_id)
/// 6. We create synthetic local handle, store mapping
/// 7. Return synthetic handle to application
///
/// R10 integration: The allocation is registered in R10's page table
/// with the AMD GPU's node_id and tier=LOCAL_VRAM. This makes it
/// visible to the tiering system for cross-node/cross-vendor migration.
fn translate_hip_malloc(
    client: &HipInterceptClient,
    size: u64,
) -> Result<u64, HipError> {
    // Build wire message
    let msg = AllocRequest {
        vendor: ApiVendor::Hip,
        size_bytes: size,
        flags: 0,
        pool: None,
    };

    // Send to server
    let response = client.send_and_recv(MessageType::MemAlloc, &msg)?;

    // Parse response: (result_code, remote_ptr)
    let result_code = response.read_u32();
    if result_code != 0 {
        return Err(HipError::from_u32(result_code));
    }

    let remote_ptr = response.read_u64();

    // Create synthetic local handle
    let local_ptr = client.handles.register_device_ptr(remote_ptr, size);

    // Track allocation size for hipMemGetAddressRange
    client.track_allocation(local_ptr, size);

    Ok(local_ptr)
}
```

### 6.2 Kernel Launch Translation

```rust
/// Translate hipModuleLaunchKernel into an OuterLink kernel launch.
///
/// Key differences from CUDA cuLaunchKernel:
/// - HIP uses hipFunction_t (maps to hipModule_t function handles)
/// - Kernel arguments are passed the same way (void** array)
/// - Grid/block dimensions use hipDim3 (identical to CUDA dim3)
/// - Shared memory and stream parameters are identical
///
/// The server-side HipExecutor calls the real hipModuleLaunchKernel
/// on the target AMD GPU. No kernel binary translation occurs —
/// the module was already loaded with the correct gfx ISA.
fn translate_hip_module_launch_kernel(
    client: &HipInterceptClient,
    func: u64,       // synthetic hipFunction_t handle
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
    block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: u64,      // synthetic hipStream_t handle
    kernel_params: &[u64],  // pointers to kernel arguments
    extra: Option<&[u8]>,   // HIP_LAUNCH_PARAM_* (alternative arg passing)
) -> Result<(), HipError> {
    // Translate synthetic handles to remote handles
    let remote_func = client.handles.translate_function(func)?;
    let remote_stream = if stream == 0 {
        0 // default stream
    } else {
        client.handles.translate_stream(stream)?
    };

    // Translate device pointers in kernel params.
    // Each kernel_params[i] points to an argument value. If that value
    // is a device pointer, we must translate it from the synthetic
    // (client-local) address space to the real (server-side) address space.
    //
    // This uses the same translate_device_ptrs_in_params logic as CUDA,
    // walking the argument buffer and replacing any value that falls in
    // our allocation ranges.
    let translated_params = translate_device_ptrs_in_params(
        &client.handles,
        kernel_params,
    )?;

    // Build wire message
    let msg = KernelLaunchRequest {
        vendor: ApiVendor::Hip,
        function: remote_func,
        grid: [grid_dim_x, grid_dim_y, grid_dim_z],
        block: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem: shared_mem_bytes,
        stream: remote_stream,
        params: translated_params,
        extra: extra.map(|e| e.to_vec()),
    };

    // Send to server (async launch — don't wait for completion)
    client.send(MessageType::KernelLaunch, &msg)?;

    Ok(())
}
```

### 6.3 Device Property Translation

```rust
/// Translate hipGetDeviceProperties into OuterLink's AmdGpuProfile.
///
/// When the application calls hipGetDeviceProperties(&prop, device),
/// we populate the hipDeviceProp_t struct from our cached AmdGpuProfile.
/// This avoids a round-trip on repeated calls (frameworks like PyTorch
/// call this multiple times during initialization).
fn translate_hip_get_device_properties(
    client: &HipInterceptClient,
    device: i32,
) -> Result<HipDeviceProps, HipError> {
    // Check cache first
    if let Some(cached) = client.device_props_cache.lock().unwrap().get(&device) {
        return Ok(cached.clone());
    }

    // Request from server
    let msg = DevicePropsRequest {
        vendor: ApiVendor::Hip,
        device_ordinal: device,
    };
    let response = client.send_and_recv(MessageType::DeviceGetProperties, &msg)?;

    // Parse AmdGpuProfile from response and convert to HipDeviceProps
    let profile: AmdGpuProfile = deserialize_amd_profile(&response)?;

    let props = HipDeviceProps {
        name: profile.name.clone(),
        total_global_mem: profile.vram_total_bytes,
        shared_mem_per_block: 65536, // 64KB default, queried from device
        regs_per_block: 65536,
        warp_size: profile.wavefront_size as i32,
        max_threads_per_block: 1024,
        max_threads_dim: [1024, 1024, 1024],
        max_grid_size: [2147483647, 65535, 65535],
        clock_rate: (profile.boost_clock_mhz * 1000) as i32, // kHz
        memory_clock_rate: 0, // filled from detailed query
        memory_bus_width: 0,  // filled from detailed query
        l2_cache_size: profile.l2_cache_bytes as i32,
        multi_processor_count: profile.compute_unit_count as i32,
        compute_mode: 0, // default
        concurrent_kernels: 1,
        pci_bus_id: 0,    // filled from detailed query
        pci_device_id: 0,
        pci_domain_id: 0,
        gcn_arch: 0, // deprecated
        gcn_arch_name: profile.arch.gfx_name.clone(),
        max_shared_memory_per_multiprocessor: 65536,
        arch: profile.arch.clone(),
    };

    // Cache for future calls
    client.device_props_cache.lock().unwrap().insert(device, props.clone());

    Ok(props)
}
```

### 6.4 Cross-Vendor Memory Transfer Algorithm

```rust
/// Handle hipMemcpy/cuMemcpy that crosses the vendor boundary.
///
/// When data flows between an NVIDIA GPU and an AMD GPU (both in the
/// OuterLink pool), neither GPU can directly access the other's memory.
/// The transfer must be staged through host memory.
///
/// Flow for AMD-to-NVIDIA transfer:
///   1. AMD GPU copies data to pinned host memory (hipMemcpyDtoH)
///   2. Host memory is sent over network to the NVIDIA node
///   3. NVIDIA node copies data to GPU memory (cuMemcpyHtoD)
///
/// This is identical to the existing cross-node transfer path.
/// The key insight: vendor differences are transparent because both
/// paths go through host-staged transfers. Cross-vendor is just
/// another form of cross-node.
///
/// With OpenDMA (Phase 5), cross-vendor transfers could potentially
/// use BAR1 direct access regardless of GPU vendor, since BAR1
/// is a PCIe standard — but this is speculative and not planned.
fn cross_vendor_memcpy(
    src_gpu: &UnifiedGpuProfile,
    dst_gpu: &UnifiedGpuProfile,
    src_ptr: u64,
    dst_ptr: u64,
    size: u64,
) -> Result<(), TransferError> {
    // Stage 1: DtoH on source GPU (vendor-appropriate API)
    let host_buffer = allocate_pinned_host(size)?;
    match src_gpu.vendor() {
        GpuVendor::Nvidia => {
            cuda_memcpy_dtoh(host_buffer, src_ptr, size)?;
        }
        GpuVendor::Amd => {
            hip_memcpy_dtoh(host_buffer, src_ptr, size)?;
        }
    }

    // Stage 2: Network transfer (if different nodes)
    if src_gpu.node_id() != dst_gpu.node_id() {
        network_send(src_gpu.node_id(), dst_gpu.node_id(), host_buffer, size)?;
    }

    // Stage 3: HtoD on destination GPU (vendor-appropriate API)
    match dst_gpu.vendor() {
        GpuVendor::Nvidia => {
            cuda_memcpy_htod(dst_ptr, host_buffer, size)?;
        }
        GpuVendor::Amd => {
            hip_memcpy_htod(dst_ptr, host_buffer, size)?;
        }
    }

    free_pinned_host(host_buffer)?;
    Ok(())
}
```

---

## 7. Integration Points

### 7.1 R23: Heterogeneous GPU Mixing

R23 currently assumes all GPUs are NVIDIA. R27 extends R23 to support AMD GPUs in the capability scoring and scheduling systems.

**Changes to R23:**

```rust
/// R23's GpuCapabilityProvider trait must be extended to handle AMD GPUs.
/// The existing trait uses NVIDIA-specific fields (compute_capability,
/// cuda_cores, tensor_core_count). R27 adds vendor-neutral accessors.
trait GpuCapabilityProvider {
    /// Get all GPU profiles in the pool (both vendors).
    fn all_profiles(&self) -> Vec<UnifiedGpuProfile>;

    /// Get capability score for a specific (workload_class, gpu) pair.
    /// The scorer uses vendor-appropriate metrics:
    /// - NVIDIA: fp32_tflops from CUDA cores, tensor from Tensor Cores
    /// - AMD: fp32_tflops from CU count * SP/CU, tensor from Matrix ALUs
    fn capability_score(&self, gpu_id: GpuId, workload_class: WorkloadClass) -> f64;

    /// Check binary compatibility. Cross-vendor is always false.
    fn is_binary_compatible(&self, gpu_id: GpuId, binary: &BinaryFormat) -> bool;

    /// Get vendor for a GPU.
    fn vendor(&self, gpu_id: GpuId) -> GpuVendor;
}

/// Extended CapabilityScorer that handles both vendors.
/// Uses the same 3-tier normalization (static, calibration, runtime EMA)
/// regardless of vendor. The reference GPU (RTX 3060 = 1.0) remains
/// the same — AMD GPUs are scored relative to RTX 3060 too.
impl CapabilityScorer {
    fn score_amd_gpu(profile: &AmdGpuProfile, class: WorkloadClass) -> f64 {
        let ref_values = ReferenceValues::default();
        match class {
            WorkloadClass::ComputeBound => {
                profile.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0)
            }
            WorkloadClass::MemoryBound => {
                profile.measured_memory_bw_gbps / ref_values.memory_bw_gbps
            }
            WorkloadClass::TensorBound => {
                // Use Matrix ALU throughput if available (CDNA),
                // fall back to packed FP16 throughput for RDNA.
                let tensor_equiv = profile.measured_matrix_tflops
                    .unwrap_or(profile.fp16_tflops * 0.5); // RDNA packed math approximation
                tensor_equiv / ref_values.tensor_tflops_fp16
            }
            WorkloadClass::Unknown => {
                // Conservative: use overall score
                let compute = profile.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0);
                let memory = profile.measured_memory_bw_gbps / ref_values.memory_bw_gbps;
                let capacity = profile.vram_total_bytes as f64
                    / (ref_values.vram_gb * 1024.0 * 1024.0 * 1024.0);
                compute * 0.4 + memory * 0.3 + capacity * 0.3
            }
        }
    }
}
```

**GEU computation for AMD GPUs** uses the same formula as NVIDIA:
```
GEU = (compute_score * 0.4) + (bandwidth_score * 0.3) + (capacity_score * 0.3)
```

This ensures fair resource accounting across vendors.

### 7.2 R10: Memory Tiering

R10's memory tiering system is already vendor-neutral at the page table level. The key changes:

```rust
/// R10's PageTable already uses TierId and NodeId, which are vendor-agnostic.
/// The only R27-specific change is in the page metadata: we need to know
/// which vendor's API to use when migrating a page.
///
/// Addition to PageTableEntry (PTE):
struct PageTableEntry {
    // ... existing fields ...

    /// Vendor of the GPU that owns this page.
    /// Used when issuing migration commands (hipMemcpy vs cuMemcpy).
    pub vendor: GpuVendor,
}

/// Migration path selection considers vendor:
/// - Same vendor: direct DtoD if same node, staged otherwise
/// - Cross vendor: always staged through host memory
fn select_migration_path(
    src_tier: TierId,
    dst_tier: TierId,
    src_vendor: GpuVendor,
    dst_vendor: GpuVendor,
    same_node: bool,
) -> MigrationPath {
    if src_vendor == dst_vendor && same_node {
        // Can use vendor's native DtoD copy
        MigrationPath::DirectDtoD
    } else if src_vendor == dst_vendor && !same_node {
        // Same vendor, different node: host-staged
        MigrationPath::HostStaged
    } else {
        // Cross-vendor: always host-staged (no direct AMD<->NVIDIA DMA)
        MigrationPath::HostStaged
    }
}
```

### 7.3 R17: Topology Scheduling

R17's topology graph needs to account for vendor compatibility constraints.

```rust
/// Extension to R17's PlacementDecision.
/// When the scheduler considers placing a workload on a GPU, it must
/// check that the kernel binary is compatible with the target GPU's
/// vendor and architecture.
///
/// A CUDA kernel (PTX/cubin) can ONLY run on NVIDIA GPUs.
/// A HIP kernel (AMDGPU ISA) can ONLY run on compatible AMD GPUs.
///
/// This is a hard constraint — not a preference, not a cost factor.
/// The scheduler filters out incompatible GPUs before scoring.
fn filter_compatible_gpus(
    workload: &WorkloadDescriptor,
    candidates: &[UnifiedGpuProfile],
) -> Vec<&UnifiedGpuProfile> {
    candidates.iter()
        .filter(|gpu| gpu.is_binary_compatible(workload.binary_format.clone()))
        .collect()
}
```

### 7.4 Wire Protocol Extension

The existing wire protocol (MessageHeader + payload) is extended with a vendor discriminator. This is backward-compatible: existing messages default to `ApiVendor::Cuda`.

```rust
/// Wire protocol changes for R27:
///
/// Option A (chosen): Add vendor field to MessageHeader.
/// The 1-byte vendor field fits in the existing header padding.
///
/// Option B (rejected): Use separate message types for HIP.
/// Would double the message type enum size and complicate routing.

/// Updated MessageHeader layout:
/// [0..4]   message_type: u32
/// [4..8]   request_id: u32
/// [8..12]  payload_size: u32
/// [12]     api_vendor: u8     // NEW: 0=CUDA, 1=HIP
/// [13..16] reserved: [u8; 3]  // padding
///
/// Total: 16 bytes (unchanged from current header size,
/// which already has 3 bytes of padding after payload_size).
```

---

## 8. Cross-Vendor Considerations

### 8.1 Running HIP Code on NVIDIA GPUs

HIP is designed to be portable. When compiled with `hipcc --platform=nvidia`, HIP code compiles to CUDA and runs on NVIDIA GPUs natively. OuterLink can leverage this:

**Scenario:** Application submits HIP kernel, but all available GPUs in the pool are NVIDIA.

**Strategy:**
1. At module load time (`hipModuleLoad`), detect the binary format.
2. If it's AMDGPU ISA (.co), it can only run on AMD GPUs.
3. If the application uses HIP RTC (`hiprtcCompileProgram`), we can potentially re-compile for NVIDIA using NVRTC. This is speculative and deferred.
4. **Phase 1 policy:** Route HIP workloads ONLY to AMD GPUs. If no AMD GPUs are available, return `hipErrorNoDevice`.

### 8.2 Running CUDA Code on AMD GPUs

This is the reverse direction. CUDA kernels (PTX/cubin) cannot run on AMD GPUs.

**Strategy:**
1. CUDA workloads are ONLY routed to NVIDIA GPUs. This is unchanged.
2. AMD GPUs in the pool are invisible to CUDA workloads.
3. The scheduler (R17/R23) maintains separate candidate lists per binary format.

### 8.3 Mixed-Vendor Pool Topology

```
Pool Registry:
┌────────────────────────────────────────────────────┐
│  GPU 0: NVIDIA RTX 3090  (Node A)  [CUDA/PTX]     │
│  GPU 1: NVIDIA RTX 3090  (Node A)  [CUDA/PTX]     │
│  GPU 2: AMD RX 7900 XTX  (Node B)  [HIP/gfx1100]  │
│  GPU 3: AMD MI250        (Node C)  [HIP/gfx90a]    │
└────────────────────────────────────────────────────┘

Scheduling rules:
  CUDA app → GPU 0, GPU 1 only
  HIP app (gfx1100 binary) → GPU 2 only
  HIP app (gfx90a binary) → GPU 3 only
  HIP RTC (source) → GPU 2, GPU 3 (compile per-target)
  Memory: Any GPU can hold data (migration is vendor-neutral)
```

### 8.4 Shared Memory Pool

Even though kernels are vendor-locked, **memory is not.** Data allocated on any GPU can be migrated to any other GPU through host-staged transfers. This means:

- A CUDA application can allocate memory that happens to be physically on an AMD GPU (transparent to the app, via R10 tiering).
- When the CUDA kernel needs that data, R10 migrates it to an NVIDIA GPU before launch.
- This enables efficient use of AMD GPUs as "memory expansion" nodes even for CUDA workloads.

### 8.5 Warp/Wavefront Size Handling

AMD GPUs use wavefronts of 32 or 64 threads (vs NVIDIA's fixed 32-thread warps). This affects:

1. **Occupancy calculations:** `hipOccupancyMaxActiveBlocksPerMultiprocessor` must use the correct wavefront size.
2. **Warp-level primitives:** `__shfl_*`, `__ballot`, `__any`, `__all` — HIP versions use the device's wavefront size.
3. **Block size recommendations:** `hipOccupancyMaxPotentialBlockSize` must account for wavefront granularity.

OuterLink handles this transparently: the server-side HipExecutor calls the real HIP APIs on the AMD GPU, which already use the correct wavefront size. The client doesn't need to know.

---

## 9. Performance Targets

### 9.1 Latency Overhead

| Operation | Target Overhead | Notes |
|-----------|----------------|-------|
| `hipMalloc` interception | < 5 us (local) | Same as CUDA: handle table lookup + wire msg |
| `hipMemcpy` (1MB, same node) | < 10% overhead vs native | Host-staged: 2x PCIe + network |
| `hipModuleLaunchKernel` | < 10 us (local) | Async: fire-and-forget to server |
| `hipStreamSynchronize` | < 50 us + actual GPU time | Round-trip to server |
| `hipGetDeviceProperties` (cached) | < 1 us | Served from client-side cache |
| `hipGetProcAddress` | < 1 us | Hash table lookup in hook table |
| Cross-vendor memcpy (1MB) | < 2x same-vendor | Extra staging step |

### 9.2 Bandwidth

| Transfer Type | Target | Path |
|--------------|--------|------|
| AMD GPU → Host (same node) | PCIe gen4 x16: ~25 GB/s | Direct hipMemcpyDtoH |
| AMD GPU → NVIDIA GPU (same node) | ~12 GB/s | Staged: DtoH + HtoD |
| AMD GPU → NVIDIA GPU (cross-node, 10GbE) | ~1.1 GB/s | Staged: DtoH + network + HtoD |
| AMD GPU → NVIDIA GPU (cross-node, 25GbE) | ~2.8 GB/s | Staged: DtoH + network + HtoD |

### 9.3 Function Coverage

| Phase | Functions Intercepted | Coverage |
|-------|---------------------|----------|
| R27 Phase 1 | ~120 (core HIP API) | Memory, module, stream, event, device, context |
| R27 Phase 2 | ~140 (+ HIP Graph API) | + hipGraph*, hipStreamBeginCapture, etc. |
| R27 Phase 3 | ~160 (+ texture, surface) | + hipTexRefGet*, hipArray*, hipMipmapped* |

Target: **Zero HIP application failures** due to missing interception. Unintercepted functions must either be forwarded transparently or return a meaningful error.

---

## 10. Test Plan

### 10.1 Unit Tests

| Test | Validates |
|------|-----------|
| `AmdGpuArch::from_gfx_name` parsing | Correct parsing of gfx906, gfx90a, gfx1100, gfx1201, etc. |
| `AmdGpuArch::is_compatible_with` | Compatibility groups work correctly |
| `HipError` code mapping | All HIP error codes map to correct values |
| `HipMemcpyKind::to_memcpy_direction` | Correct direction inference for hipMemcpyDefault |
| `HipTranslator` handle translation | Synthetic-to-remote handle roundtrip |
| Hook table completeness | Every function in hip_hook_table has a corresponding FFI function |
| Device props cache | Cache hit/miss behavior, invalidation on device change |
| Wire protocol vendor field | ApiVendor::Hip correctly serialized/deserialized |

### 10.2 Integration Tests

| Test | Validates |
|------|-----------|
| **hipMalloc/hipFree roundtrip** | Allocate on remote AMD GPU, free, verify no leak |
| **hipMemcpy correctness** | Copy 1MB H2D, D2H, verify bit-exact data integrity |
| **hipModuleLoad + hipModuleLaunchKernel** | Load .co file, launch simple vector_add kernel, verify results |
| **hipStreamCreate + async ops** | Create stream, async memcpy, kernel launch, synchronize |
| **hipEventCreate + timing** | Create events, record, elapsed time > 0 |
| **hipGetDeviceProperties** | All fields populated correctly for known AMD GPU models |
| **hipGetProcAddress bypass prevention** | Application using hipGetProcAddress still gets our hooks |
| **dlsym bypass prevention** | Application using dlsym(RTLD_NEXT, "hipMalloc") gets our hooks |
| **Multiple devices** | hipSetDevice(0), hipSetDevice(1), verify correct routing |
| **Peer access** | hipDeviceEnablePeerAccess between two OuterLink AMD GPUs |
| **Error propagation** | Server returns hipErrorOutOfMemory, client sees correct error |
| **Callback execution** | hipStreamAddCallback fires on client side after remote completion |

### 10.3 Cross-Vendor Integration Tests

| Test | Validates |
|------|-----------|
| **Mixed pool registration** | NVIDIA + AMD GPUs both visible in pool, correct vendor tags |
| **CUDA app ignores AMD GPUs** | cuDeviceGetCount returns only NVIDIA count |
| **HIP app ignores NVIDIA GPUs** | hipGetDeviceCount returns only AMD count |
| **Cross-vendor memcpy** | hipMemcpy from AMD GPU, cuMemcpy to NVIDIA GPU, verify data |
| **R23 scoring cross-vendor** | GEU scores are comparable between vendors (same benchmark) |
| **R10 migration cross-vendor** | Page migrates from AMD VRAM to NVIDIA VRAM via host staging |
| **Dual LD_PRELOAD** | Both interpose.so and interpose_hip.so loaded, both work |

### 10.4 Framework Compatibility Tests

| Framework | Test |
|-----------|------|
| **PyTorch (ROCm)** | `torch.cuda.is_available()` returns True on AMD GPUs via HIP |
| **PyTorch (ROCm)** | Simple training loop (ResNet-18, 10 batches) completes without error |
| **TensorFlow (ROCm)** | `tf.config.list_physical_devices('GPU')` lists AMD GPUs |
| **JAX (ROCm)** | `jax.devices()` lists AMD GPUs |
| **hipBLAS** | SGEMM on remote AMD GPU produces correct results |
| **MIOpen** | Convolution forward pass on remote AMD GPU |
| **hipFFT** | 1D FFT on remote AMD GPU matches CPU reference |

### 10.5 Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| `hipMalloc` latency (1000 calls) | < 5 us avg overhead |
| `hipMemcpy` throughput (1MB, DtoH, same node) | > 20 GB/s |
| `hipMemcpy` throughput (1MB, DtoH, cross-node 10GbE) | > 1.0 GB/s |
| `hipModuleLaunchKernel` latency (null kernel) | < 15 us total |
| `hipStreamSynchronize` latency (after null kernel) | < 100 us total |
| Hook table lookup time | < 500 ns |
| Device props cache lookup | < 100 ns |

### 10.6 Stress Tests

| Test | Purpose |
|------|---------|
| 1000 concurrent hipMalloc/hipFree from 8 threads | Thread safety of handle store |
| 100 streams with interleaved operations | Stream handle management under load |
| Rapid hipSetDevice switching | Context switch correctness |
| 10000 kernel launches in tight loop | Pipeline throughput, no handle exhaustion |
| OOM recovery | hipMalloc returns hipErrorOutOfMemory, subsequent calls still work |

---

## 11. Implementation Phases

### R27a: Core Interception + Memory (6-8 weeks)

**Files to create:**
- `crates/outerlink-hip-client/csrc/interpose_hip.c` — C interposition layer
- `crates/outerlink-hip-client/csrc/interpose_hip.h` — FFI declarations
- `crates/outerlink-hip-client/src/lib.rs` — HipInterceptClient
- `crates/outerlink-hip-client/src/ffi.rs` — FFI exports
- `crates/outerlink-hip-client/src/translator.rs` — HipTranslator
- `crates/outerlink-common/src/vendor.rs` — GpuVendor, AmdGpuArch, BinaryFormat

**Files to modify:**
- `crates/outerlink-common/src/gpu_mixing.rs` — Add UnifiedGpuProfile, AMD scoring
- `crates/outerlink-common/src/memory/types.rs` — Add vendor field to PTE
- `crates/outerlink-common/src/protocol.rs` — Add ApiVendor to MessageHeader
- `crates/outerlink-server/src/lib.rs` — Add HipExecutor dispatch

**Deliverables:**
1. LD_PRELOAD interception of ~60 HIP functions (init, device, context, memory)
2. hipGetProcAddress interception preventing bypass
3. HandleStore for HIP handles (parallel to CUDA handles)
4. Wire protocol vendor discriminator
5. Server-side HipExecutor stub (memory operations)

**Gate:** `hipMalloc` + `hipMemcpy` + `hipFree` roundtrip works on remote AMD GPU.

### R27b: Module/Kernel + Stream/Event (4-6 weeks)

**Deliverables:**
1. hipModuleLoad / hipModuleLaunchKernel working
2. Stream and event management
3. Callback support (hipStreamAddCallback, hipLaunchHostFunc)
4. Occupancy query functions
5. Error handling parity

**Gate:** PyTorch ROCm simple inference (ResNet-18 forward pass) works on remote AMD GPU.

### R27c: R23 Integration + Cross-Vendor (4-6 weeks)

**Deliverables:**
1. AmdGpuProfile populated from ROCm SMI / AMD SMI
2. Capability scoring for AMD GPUs (calibration benchmarks)
3. UnifiedGpuProfile in pool registry
4. Cross-vendor memory migration via R10
5. R17 topology scheduler filters by vendor compatibility
6. Dual LD_PRELOAD support

**Gate:** Mixed AMD + NVIDIA pool works. CUDA app uses only NVIDIA GPUs, HIP app uses only AMD GPUs, memory migrates between vendors.

### R27d: HIP Graph API + Polish (4-6 weeks, optional Phase 2)

**Deliverables:**
1. HIP Graph API interception (~20 functions)
2. Stream capture support
3. Framework compatibility testing (PyTorch, TF, JAX)
4. Performance optimization (batch requests, pipeline)
5. ROCm version compatibility matrix

**Gate:** All framework compatibility tests pass. Performance targets met.

---

## 12. Open Questions

### Q1: Should the HIP client be a separate crate or part of outerlink-client?

**Current thinking:** Separate crate (`outerlink-hip-client`). The CUDA and HIP clients share common code (HandleStore, transport, protocol) via `outerlink-common`, but have separate C interposition layers and FFI surfaces. A single combined crate would entangle CUDA and HIP headers/dependencies.

**Risk:** Code duplication between the two client crates. Mitigated by maximizing shared code in `outerlink-common`.

### Q2: How to handle ROCm version fragmentation?

AMD's ROCm releases change the HIP API more frequently than NVIDIA changes CUDA. Functions appear, disappear, and change signatures between ROCm versions.

**Current thinking:** Version-indexed function table (same as our CUDA strategy from HAMi-core). The hook table includes version guards. At init time, query `hipRuntimeGetVersion` to determine which functions to intercept.

### Q3: Can we intercept HIP applications that were compiled with the NVIDIA backend?

When HIP code is compiled with `hipcc --platform=nvidia`, it links against `libcudart.so`, not `libamdhip64.so`. These applications would be caught by our existing CUDA interception, not R27.

**Resolution:** This is fine. Such applications are CUDA applications from the interception perspective. No R27 changes needed.

### Q4: How to handle hipDeviceProp_t differences across ROCm versions?

The `hipDeviceProp_t` struct has grown over ROCm versions (new fields added). Applications compiled with older ROCm may expect a smaller struct.

**Current thinking:** Return the struct matching the ROCm version the application was compiled against. Query via `hipRuntimeGetVersion` at init time, maintain struct layout tables per version.

### Q5: AMD Infinity Cache — should OuterLink account for it?

RDNA2+ GPUs have Infinity Cache (up to 128MB on RX 7900 XTX). This acts as a massive L3 that significantly boosts effective bandwidth for certain access patterns.

**Current thinking:** Track it in AmdGpuProfile but don't model it in R23's scoring. The calibration benchmarks already capture its effect (measured bandwidth includes Infinity Cache benefit). Explicit modeling would add complexity without clear value.

### Q6: ROCm SMI vs AMD SMI — which to use for GPU monitoring?

ROCm SMI (`librocm_smi64.so`) is the established library. AMD SMI (`libamd_smi.so`) is the newer successor. Both provide similar functionality.

**Current thinking:** Support ROCm SMI first (wider deployment), add AMD SMI support when ROCm SMI is deprecated. Use runtime detection to pick whichever is available.

### Q7: Can OuterLink pool AMD dGPUs and APU iGPUs?

AMD APUs (like Ryzen with integrated graphics) expose a GPU through ROCm. Could these be useful in the pool?

**Current thinking:** Technically yes, but practically no for Phase 1. APU iGPUs share system memory (no dedicated VRAM), have very few CUs, and would score extremely low in R23. Exclude them via a minimum CU count threshold (e.g., require >= 16 CUs).

### Q8: HIP's hipMallocManaged vs CUDA's cuMemAllocManaged — semantic differences?

Both provide "managed memory" that migrates between host and device. However, the migration granularity and page fault behavior may differ between vendors.

**Current thinking:** For Phase 1, treat managed memory as device memory. The application expects transparent migration, but OuterLink's R10 tiering provides its own migration policy. Map `hipMallocManaged` to `hipMalloc` on the server side and let R10 handle migration.

### Q9: RCCL (AMD's NCCL equivalent) — does R27 need to intercept it?

RCCL is AMD's implementation of the NCCL collective communication library. If HIP applications use RCCL for multi-GPU communication, R27 would need to intercept it similarly to how R20 intercepts NCCL.

**Current thinking:** Deferred. RCCL interception is a separate topic (R27-RCCL), dependent on R20's NCCL backend being stable first. For Phase 1, RCCL calls on remote AMD GPUs are forwarded transparently.

### Q10: Kernel binary format detection — how to distinguish .co from .cubin in hipModuleLoadData?

When `hipModuleLoadData` is called with a binary blob, we need to determine if it's an AMDGPU code object (.co) or a CUDA binary (cubin/fatbin). This matters for routing to the correct GPU.

**Current thinking:** Parse the ELF header. AMDGPU code objects are ELF files with `e_machine = EM_AMDGPU (0xe0)`. CUDA fat binaries start with a magic number (`0x466243b1`). This detection is straightforward.

---

## Related Documents

- [R27 README](./README.md) — Component overview
- [R23 Heterogeneous GPU Mixing](../../phase-10-compute-distribution/R23-heterogeneous-gpu-mixing/) — GpuProfile, CapabilityScorer
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/) — PageTable, tier system
- [R17 Topology Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/) — Placement decisions
- [R13 CUDA Graph Interception](../../phase-10-compute-distribution/R13-cuda-graph-interception/) — Graph API pattern (HIP Graph parallel)
- [R20 NCCL Backend](../../phase-09-collective-communication/R20-nccl-backend/) — Collective communication (RCCL parallel)
- [CUDA Interpose.c](../../../../crates/outerlink-client/csrc/interpose.c) — Existing CUDA interception pattern
- [Research Consolidation](../../../research/CONSOLIDATION-all-research.md) — Interception architecture decisions
- [HIP Runtime API Reference (ROCm 7.2)](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [HIP Porting CUDA Driver API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_driver_api.html)
- [ROCm SMI Library](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/)
- [hipDeviceProp_t Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html)
- [HIP Graph API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
