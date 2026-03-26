//! ROCm/HIP Interception (R27) -- AMD GPU support for OuterLink pools.
//!
//! Extends OuterLink's GPU interception layer to AMD's HIP/ROCm stack, enabling
//! AMD GPUs (RDNA, CDNA, GCN) to join the OuterLink pool alongside NVIDIA GPUs.
//! Achieves true vendor-agnostic GPU pooling with mixed AMD + NVIDIA clusters.
//!
//! # Architecture
//!
//! ```text
//! Application (HIP C++ code)
//!    |
//!    v
//! LD_PRELOAD interpose_hip.so
//!    |  dlsym / hipGetProcAddress interception
//!    v
//! outerlink-hip-client.so (Rust cdylib)
//!    |  HipTranslator -> wire protocol
//!    v
//! OuterLink Server
//!    |  VendorMessageHeader.api_vendor == Hip
//!    v
//! HipExecutor -> real HIP API on AMD GPU
//! ```
//!
//! # Integration Points
//!
//! - R23 (GPU Mixing): `AmdGpuProfile` + `UnifiedGpuProfile` for cross-vendor scoring.
//! - R10 (Memory Tiering): Vendor-neutral page table with vendor tag for migration.
//! - R17 (Topology Scheduling): Binary compatibility filtering per vendor.
//! - R13 (CUDA Graph Interception): HIP Graph API parallel (Phase 2).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::gpu_mixing::{GpuId, GpuProfile, ReferenceValues, WorkloadClass, WorkloadScores};
use crate::memory::types::{MemcpyDirection, NodeId};

// ---------------------------------------------------------------------------
// GPU Vendor Abstraction
// ---------------------------------------------------------------------------

/// GPU vendor identifier. Used throughout OuterLink to dispatch
/// vendor-specific operations and validate binary compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA GPU (CUDA/PTX).
    Nvidia,
    /// AMD GPU (HIP/ROCm/AMDGPU ISA).
    Amd,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
        }
    }
}

// ---------------------------------------------------------------------------
// AMD Architecture Family + Arch Target
// ---------------------------------------------------------------------------

/// AMD GPU architecture family. Determines wavefront size, instruction set,
/// and feature availability. Parallel to NVIDIA's compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdArchFamily {
    /// GCN (Graphics Core Next) -- Polaris, Vega.
    /// Wave64 only. 4x SIMD16 per CU.
    Gcn,
    /// CDNA (Compute DNA) -- MI100, MI250, MI300.
    /// Wave64. Matrix ALUs (MFMA). Datacenter compute.
    Cdna,
    /// RDNA (Radeon DNA) -- RX 5000/6000/7000/9000 series.
    /// Wave32 (native) + Wave64 (compat). Work Group Processors (WGPs).
    Rdna,
}

impl std::fmt::Display for AmdArchFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gcn => write!(f, "GCN"),
            Self::Cdna => write!(f, "CDNA"),
            Self::Rdna => write!(f, "RDNA"),
        }
    }
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
    /// Parsed major version (first digit(s) after gfx).
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
    ///
    /// The gfx naming scheme uses positional hex digits:
    /// - Last char = stepping (hex: 0-9, a-f)
    /// - Second-to-last char = minor (hex: 0-9, a-f)
    /// - All preceding chars = major (decimal)
    ///
    /// Examples: "gfx906" -> (9, 0, 6), "gfx90a" -> (9, 0, 10),
    /// "gfx1100" -> (11, 0, 0), "gfx1201" -> (12, 0, 1).
    ///
    /// Returns None if the string is not a valid gfx name.
    pub fn from_gfx_name(name: &str) -> Option<Self> {
        let chars: &str = name.strip_prefix("gfx")?;
        if chars.len() < 3 {
            return None;
        }

        // Parse stepping (last char) and minor (second-to-last) as hex digits.
        // Everything before those two is the major version in decimal.
        let char_vec: Vec<char> = chars.chars().collect();
        let len = char_vec.len();

        let stepping = hex_digit_value(char_vec[len - 1])?;
        let minor = hex_digit_value(char_vec[len - 2])?;

        let major_str: String = char_vec[..len - 2].iter().collect();
        let major: u32 = major_str.parse().ok()?;

        let family = match major {
            7 | 8 => AmdArchFamily::Gcn,
            9 => {
                // gfx9xx: GCN for gfx900-gfx906 (minor=0, stepping<=6),
                // CDNA for gfx908+ (stepping>=8, or minor>0 like gfx940/942)
                if minor > 0 || stepping >= 8 {
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

    /// Check if a code object compiled for `target_gfx` can run on this GPU.
    ///
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
        let compat_groups: &[&[&str]] = &[
            &["gfx900", "gfx902"],
            &["gfx1010", "gfx1011", "gfx1012"],
            &[
                "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
            ],
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

    /// Return the minimum Compute Unit count for OuterLink pool membership.
    /// Excludes APU iGPUs which have very few CUs.
    pub fn min_cu_count() -> u32 {
        16
    }
}

/// Parse a single hex digit character to its numeric value.
fn hex_digit_value(ch: char) -> Option<u32> {
    match ch {
        '0'..='9' => Some(ch as u32 - '0' as u32),
        'a'..='f' => Some(10 + (ch as u32 - 'a' as u32)),
        'A'..='F' => Some(10 + (ch as u32 - 'A' as u32)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Binary Format
// ---------------------------------------------------------------------------

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

impl BinaryFormat {
    /// Detect binary format from raw bytes by inspecting magic numbers.
    ///
    /// - AMDGPU code objects are ELF files with e_machine = EM_AMDGPU (0xe0).
    /// - CUDA fat binaries start with magic 0x466243b1.
    /// - PTX is text starting with ".version".
    pub fn detect(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        // Check for CUDA fatbin magic
        if data.len() >= 4 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == 0x466243b1 {
                return Some(Self::Cubin);
            }
        }

        // Check for ELF (AMDGPU code object)
        if data.len() >= 20 && data[0..4] == [0x7f, b'E', b'L', b'F'] {
            // e_machine is at offset 18 (ELF64) as a u16 LE
            let e_machine = u16::from_le_bytes([data[18], data[19]]);
            if e_machine == 0xe0 {
                // EM_AMDGPU
                // TODO: Extract actual gfx target from ELF notes section
                return Some(Self::AmdgpuIsa("unknown".to_string()));
            }
        }

        // Check for PTX text
        if data.len() >= 8 {
            if let Ok(text) = std::str::from_utf8(&data[..std::cmp::min(data.len(), 64)]) {
                if text.contains(".version") {
                    return Some(Self::Ptx);
                }
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// AMD GPU Profile
// ---------------------------------------------------------------------------

/// Complete hardware profile for an AMD GPU in the OuterLink pool.
///
/// Parallel to `GpuProfile` (which is NVIDIA-centric), this captures
/// AMD-specific hardware attributes, calibration data, and runtime state.
/// Both `GpuProfile` (NVIDIA) and `AmdGpuProfile` (AMD) are stored in
/// the pool's GPU registry and exposed through R23's capability scoring.
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
    /// FP16 support.
    pub supports_fp16: bool,
    /// BF16 support (CDNA2+ and RDNA3+).
    pub supports_bf16: bool,
    /// FP8 support (CDNA3 / MI300).
    pub supports_fp8: bool,
    /// INT8 support.
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
    pub capability_scores: WorkloadScores,
    /// GPU Equivalent Units (single number for fairness/quota).
    pub geu: f64,
}

// ---------------------------------------------------------------------------
// Unified GPU Profile
// ---------------------------------------------------------------------------

/// A GPU in the OuterLink pool, regardless of vendor.
///
/// This is the type stored in the pool registry and used by
/// R23's capability scorer, R17's topology scheduler, and
/// R13's HEFT partitioner.
#[derive(Debug, Clone)]
pub enum UnifiedGpuProfile {
    /// An NVIDIA GPU (CUDA).
    Nvidia(GpuProfile),
    /// An AMD GPU (HIP/ROCm).
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

    /// Get GPU name.
    pub fn name(&self) -> &str {
        match self {
            Self::Nvidia(p) => &p.name,
            Self::Amd(p) => &p.name,
        }
    }

    /// Check if a kernel binary is compatible with this GPU.
    ///
    /// NVIDIA kernels cannot run on AMD GPUs and vice versa.
    /// Cross-vendor binary execution is not possible.
    pub fn is_binary_compatible(&self, binary_format: &BinaryFormat) -> bool {
        match (self, binary_format) {
            (Self::Nvidia(_), BinaryFormat::Ptx | BinaryFormat::Cubin) => true,
            (Self::Amd(p), BinaryFormat::AmdgpuIsa(ref target)) => {
                p.arch.is_compatible_with(target)
            }
            // HIP source can potentially be compiled for either vendor
            (_, BinaryFormat::HipSource) => true,
            _ => false,
        }
    }

    /// Whether this GPU is currently available for scheduling.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Nvidia(p) => !p.is_throttling && p.utilization < 0.95,
            Self::Amd(p) => !p.is_throttling && p.utilization < 0.95,
        }
    }

    /// Whether this GPU has a specific vendor.
    pub fn is_vendor(&self, vendor: GpuVendor) -> bool {
        self.vendor() == vendor
    }
}

// ---------------------------------------------------------------------------
// HIP Error Codes
// ---------------------------------------------------------------------------

/// HIP error codes. Maps HIP error semantics to integer values.
///
/// HIP error codes are largely parallel to CUDA error codes but with
/// some differences in numbering and semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HipError {
    /// Operation completed successfully.
    Success = 0,
    /// Invalid value passed as argument.
    InvalidValue = 1,
    /// Out of GPU memory.
    OutOfMemory = 2,
    /// HIP runtime not initialized.
    NotInitialized = 3,
    /// HIP runtime deinitialized.
    Deinitialized = 4,
    /// Invalid device ordinal.
    InvalidDevice = 101,
    /// Invalid context.
    InvalidContext = 201,
    /// Invalid memcpy direction.
    InvalidMemcpyDirection = 21,
    /// Kernel launch failure.
    LaunchFailure = 719,
    /// Operation not supported.
    NotSupported = 801,
    /// Unknown error.
    Unknown = 999,
}

impl HipError {
    /// Convert from a u32 error code to HipError.
    pub fn from_u32(code: u32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            101 => Self::InvalidDevice,
            201 => Self::InvalidContext,
            21 => Self::InvalidMemcpyDirection,
            719 => Self::LaunchFailure,
            801 => Self::NotSupported,
            _ => Self::Unknown,
        }
    }

    /// Convert to u32 for wire protocol.
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Get the human-readable error name.
    pub fn error_name(self) -> &'static str {
        match self {
            Self::Success => "hipSuccess",
            Self::InvalidValue => "hipErrorInvalidValue",
            Self::OutOfMemory => "hipErrorOutOfMemory",
            Self::NotInitialized => "hipErrorNotInitialized",
            Self::Deinitialized => "hipErrorDeinitialized",
            Self::InvalidDevice => "hipErrorInvalidDevice",
            Self::InvalidContext => "hipErrorInvalidContext",
            Self::InvalidMemcpyDirection => "hipErrorInvalidMemcpyDirection",
            Self::LaunchFailure => "hipErrorLaunchFailure",
            Self::NotSupported => "hipErrorNotSupported",
            Self::Unknown => "hipErrorUnknown",
        }
    }

    /// Get the human-readable error string.
    pub fn error_string(self) -> &'static str {
        match self {
            Self::Success => "no error",
            Self::InvalidValue => "invalid argument",
            Self::OutOfMemory => "out of memory",
            Self::NotInitialized => "HIP driver not initialized",
            Self::Deinitialized => "HIP driver deinitialized",
            Self::InvalidDevice => "invalid device ordinal",
            Self::InvalidContext => "invalid device context",
            Self::InvalidMemcpyDirection => "invalid memcpy direction",
            Self::LaunchFailure => "unspecified launch failure",
            Self::NotSupported => "operation not supported",
            Self::Unknown => "unknown error",
        }
    }

    /// Check if this represents a successful operation.
    pub fn is_success(self) -> bool {
        self == Self::Success
    }
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.error_name(), self.error_string())
    }
}

impl std::error::Error for HipError {}

// ---------------------------------------------------------------------------
// HIP Memcpy Kind
// ---------------------------------------------------------------------------

/// Direction of HIP memcpy, mapping to OuterLink's MemcpyDirection.
///
/// HIP uses the same enum values as CUDA for the core directions,
/// but adds hipMemcpyDefault which auto-detects based on pointer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HipMemcpyKind {
    /// Host to host.
    HostToHost = 0,
    /// Host to device.
    HostToDevice = 1,
    /// Device to host.
    DeviceToHost = 2,
    /// Device to device.
    DeviceToDevice = 3,
    /// Auto-detect based on pointer type.
    Default = 4,
}

impl HipMemcpyKind {
    /// Convert from u32 (wire protocol).
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::HostToHost),
            1 => Some(Self::HostToDevice),
            2 => Some(Self::DeviceToHost),
            3 => Some(Self::DeviceToDevice),
            4 => Some(Self::Default),
            _ => None,
        }
    }

    /// Convert to OuterLink's MemcpyDirection.
    ///
    /// `hipMemcpyDefault` requires knowing pointer types to determine
    /// the actual direction. The `src_is_device` and `dst_is_device`
    /// parameters are used only for the Default variant.
    pub fn to_memcpy_direction(
        self,
        src_is_device: bool,
        dst_is_device: bool,
    ) -> MemcpyDirection {
        match self {
            Self::HostToHost => MemcpyDirection::HostToHost,
            Self::HostToDevice => MemcpyDirection::HostToDevice,
            Self::DeviceToHost => MemcpyDirection::DeviceToHost,
            Self::DeviceToDevice => MemcpyDirection::DeviceToDevice,
            Self::Default => match (src_is_device, dst_is_device) {
                (false, false) => MemcpyDirection::HostToHost,
                (false, true) => MemcpyDirection::HostToDevice,
                (true, false) => MemcpyDirection::DeviceToHost,
                (true, true) => MemcpyDirection::DeviceToDevice,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Wire Protocol Vendor Discriminator
// ---------------------------------------------------------------------------

/// Wire protocol vendor discriminator.
///
/// Added to message headers to tell the server which driver API set to use.
/// Backward-compatible: existing CUDA messages default to `ApiVendor::Cuda`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ApiVendor {
    /// CUDA Driver API (existing).
    Cuda = 0,
    /// HIP Runtime API (R27).
    Hip = 1,
}

impl ApiVendor {
    /// Convert from u8 (wire protocol deserialization).
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Cuda),
            1 => Some(Self::Hip),
            _ => None,
        }
    }
}

impl Default for ApiVendor {
    fn default() -> Self {
        Self::Cuda
    }
}

/// Wire protocol header extension that embeds the API vendor discriminator.
///
/// Extends the existing `MessageHeader` (defined in the protocol module) with
/// a vendor byte at offset [12] in the padding region. This allows the server
/// to dispatch incoming messages to the correct executor (CUDA vs HIP) before
/// parsing the payload.
///
/// Layout (16 bytes total):
/// - bytes [0..3]:  message_type (u32)
/// - bytes [4..7]:  payload_size (u32)
/// - bytes [8..11]: request_id (u32)
/// - byte  [12]:    api_vendor (u8) — 0=CUDA, 1=HIP
/// - bytes [13..15]: reserved (zeroed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VendorMessageHeader {
    /// Message type code.
    pub message_type: u32,
    /// Payload size in bytes (following the header).
    pub payload_size: u32,
    /// Request ID for correlating request/response pairs.
    pub request_id: u32,
    /// API vendor discriminator.
    pub api_vendor: ApiVendor,
    /// Reserved for future use.
    pub _reserved: [u8; 3],
}

impl VendorMessageHeader {
    /// Size of the header on the wire.
    pub const SIZE: usize = 16;

    /// Create a new header for a HIP message.
    pub fn new_hip(message_type: u32, payload_size: u32, request_id: u32) -> Self {
        Self {
            message_type,
            payload_size,
            request_id,
            api_vendor: ApiVendor::Hip,
            _reserved: [0; 3],
        }
    }

    /// Create a new header for a CUDA message (backwards compatible).
    pub fn new_cuda(message_type: u32, payload_size: u32, request_id: u32) -> Self {
        Self {
            message_type,
            payload_size,
            request_id,
            api_vendor: ApiVendor::Cuda,
            _reserved: [0; 3],
        }
    }

    /// Serialize to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.message_type.to_le_bytes());
        buf[4..8].copy_from_slice(&self.payload_size.to_le_bytes());
        buf[8..12].copy_from_slice(&self.request_id.to_le_bytes());
        buf[12] = self.api_vendor as u8;
        // bytes 13..15 stay zero (reserved)
        buf
    }

    /// Deserialize from bytes (little-endian).
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Option<Self> {
        let message_type = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let payload_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let request_id = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let api_vendor = ApiVendor::from_u8(buf[12])?;
        Some(Self {
            message_type,
            payload_size,
            request_id,
            api_vendor,
            _reserved: [0; 3],
        })
    }
}

// ---------------------------------------------------------------------------
// HIP Device Properties Cache
// ---------------------------------------------------------------------------

/// Cached device properties returned by hipGetDeviceProperties.
///
/// Populated from `AmdGpuProfile` on first query, then served locally
/// to avoid round-trips for repeated calls (frameworks like PyTorch
/// call this multiple times during initialization).
#[derive(Debug, Clone)]
pub struct HipDeviceProps {
    /// GPU name.
    pub name: String,
    /// Total global memory in bytes.
    pub total_global_mem: u64,
    /// Shared memory per block in bytes.
    pub shared_mem_per_block: u64,
    /// Registers per block.
    pub regs_per_block: i32,
    /// Wavefront size (warp_size in HIP API, 32 or 64).
    pub warp_size: i32,
    /// Maximum threads per block.
    pub max_threads_per_block: i32,
    /// Maximum threads per dimension [x, y, z].
    pub max_threads_dim: [i32; 3],
    /// Maximum grid size [x, y, z].
    pub max_grid_size: [i32; 3],
    /// Clock rate in kHz.
    pub clock_rate: i32,
    /// Memory clock rate in kHz.
    pub memory_clock_rate: i32,
    /// Memory bus width in bits.
    pub memory_bus_width: i32,
    /// L2 cache size in bytes.
    pub l2_cache_size: i32,
    /// Number of compute units (multi-processors).
    pub multi_processor_count: i32,
    /// Compute mode.
    pub compute_mode: i32,
    /// Whether concurrent kernels are supported.
    pub concurrent_kernels: i32,
    /// PCI bus ID.
    pub pci_bus_id: i32,
    /// PCI device ID.
    pub pci_device_id: i32,
    /// PCI domain ID.
    pub pci_domain_id: i32,
    /// GCN architecture number (deprecated).
    pub gcn_arch: i32,
    /// GCN architecture name (e.g., "gfx1100").
    pub gcn_arch_name: String,
    /// Maximum shared memory per multiprocessor in bytes.
    pub max_shared_memory_per_multiprocessor: u64,
    /// Parsed architecture information.
    pub arch: AmdGpuArch,
}

impl HipDeviceProps {
    /// Build device properties from an AmdGpuProfile.
    ///
    /// Translates the OuterLink profile into the format expected by
    /// the HIP runtime's hipDeviceProp_t structure.
    pub fn from_profile(profile: &AmdGpuProfile) -> Self {
        Self {
            name: profile.name.clone(),
            total_global_mem: profile.vram_total_bytes,
            shared_mem_per_block: 65536, // 64KB default
            regs_per_block: 65536,
            warp_size: profile.wavefront_size as i32,
            max_threads_per_block: 1024,
            max_threads_dim: [1024, 1024, 1024],
            max_grid_size: [2_147_483_647, 65535, 65535],
            clock_rate: (profile.boost_clock_mhz * 1000) as i32, // kHz
            memory_clock_rate: 0, // Filled from detailed query
            memory_bus_width: 0,  // Filled from detailed query
            l2_cache_size: profile.l2_cache_bytes as i32,
            multi_processor_count: profile.compute_unit_count as i32,
            compute_mode: 0, // Default
            concurrent_kernels: 1,
            pci_bus_id: 0,
            pci_device_id: 0,
            pci_domain_id: 0,
            gcn_arch: 0, // Deprecated field
            gcn_arch_name: profile.arch.gfx_name.clone(),
            max_shared_memory_per_multiprocessor: 65536,
            arch: profile.arch.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// HIP Module and Function Entries
// ---------------------------------------------------------------------------

/// HIP module handle -- represents a loaded code object on a remote AMD GPU.
///
/// Maps hipModule_t (opaque pointer) to an internal ID with metadata
/// about the target architecture and extracted functions.
#[derive(Debug, Clone)]
pub struct HipModuleEntry {
    /// Synthetic local handle (returned to application).
    pub local_handle: u64,
    /// Real remote handle (on the server's AMD GPU).
    pub remote_handle: u64,
    /// gfx target this module was compiled for.
    pub target_arch: String,
    /// Functions extracted from this module (kernel_name -> entry).
    pub functions: HashMap<String, HipFunctionEntry>,
}

/// HIP function handle -- a kernel within a loaded module.
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
    /// Attributes cache (attribute_id -> value).
    pub attributes: HashMap<u32, i32>,
}

// ---------------------------------------------------------------------------
// Server-Side State
// ---------------------------------------------------------------------------

/// Server-side state for a loaded HIP module.
#[derive(Debug)]
pub struct HipModuleState {
    /// Real hipModule_t handle.
    pub handle: u64,
    /// Functions extracted from the module (name -> handle).
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

/// Server-side component that executes HIP API calls on a local AMD GPU.
///
/// Parallel to the existing CUDA executor, but calls HIP functions instead.
/// The server determines which executor to use based on the `ApiVendor`
/// field in the message header.
pub struct HipExecutor {
    /// Device ordinal on this physical machine.
    pub device_ordinal: i32,
    /// hipDevice_t handle.
    pub device: i32,
    /// Primary context for this device.
    pub primary_ctx: u64,
    /// Module registry (handle -> state).
    pub modules: HashMap<u64, HipModuleState>,
    /// Stream registry (local_handle -> real_handle).
    pub streams: HashMap<u64, u64>,
    /// Event registry (local_handle -> real_handle).
    pub events: HashMap<u64, u64>,
    /// Memory allocation tracker (device_ptr -> info).
    pub allocations: HashMap<u64, AllocationInfo>,
    /// Device profile (populated at registration).
    pub profile: AmdGpuProfile,
}

impl HipExecutor {
    /// Create a new HIP executor for a device.
    ///
    /// In production, this calls the real HIP runtime to initialize.
    /// The profile must be pre-populated from device enumeration.
    pub fn new(device_ordinal: i32, profile: AmdGpuProfile) -> Self {
        Self {
            device_ordinal,
            device: device_ordinal,
            primary_ctx: 0, // TODO: Acquire via real hipDevicePrimaryCtxRetain
            modules: HashMap::new(),
            streams: HashMap::new(),
            events: HashMap::new(),
            allocations: HashMap::new(),
            profile,
        }
    }

    /// Register a memory allocation.
    pub fn track_allocation(&mut self, ptr: u64, size: u64, pool: Option<u64>, managed: bool) {
        self.allocations.insert(
            ptr,
            AllocationInfo {
                device_ptr: ptr,
                size,
                pool,
                is_managed: managed,
            },
        );
    }

    /// Remove an allocation from tracking. Returns the info if found.
    pub fn untrack_allocation(&mut self, ptr: u64) -> Option<AllocationInfo> {
        self.allocations.remove(&ptr)
    }

    /// Get allocation info by pointer.
    pub fn get_allocation(&self, ptr: u64) -> Option<&AllocationInfo> {
        self.allocations.get(&ptr)
    }

    /// Register a loaded module.
    pub fn register_module(&mut self, handle: u64, state: HipModuleState) {
        self.modules.insert(handle, state);
    }

    /// Unload a module. Returns the state if found.
    pub fn unload_module(&mut self, handle: u64) -> Option<HipModuleState> {
        self.modules.remove(&handle)
    }

    /// Register a stream.
    pub fn register_stream(&mut self, local: u64, real: u64) {
        self.streams.insert(local, real);
    }

    /// Destroy a stream. Returns the real handle if found.
    pub fn destroy_stream(&mut self, local: u64) -> Option<u64> {
        self.streams.remove(&local)
    }

    /// Register an event.
    pub fn register_event(&mut self, local: u64, real: u64) {
        self.events.insert(local, real);
    }

    /// Destroy an event. Returns the real handle if found.
    pub fn destroy_event(&mut self, local: u64) -> Option<u64> {
        self.events.remove(&local)
    }

    /// Get total tracked allocation size.
    pub fn total_allocated_bytes(&self) -> u64 {
        self.allocations.values().map(|a| a.size).sum()
    }
}

// ---------------------------------------------------------------------------
// HIP-to-CUDA Function Mapping Table
// ---------------------------------------------------------------------------

/// Category for HIP function mappings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HipFunctionCategory {
    /// Initialization and device management.
    Initialization,
    /// Context management.
    Context,
    /// Memory management.
    Memory,
    /// Module and kernel management.
    Module,
    /// Stream and event management.
    StreamEvent,
    /// Error handling.
    Error,
    /// Graph API (Phase 2).
    Graph,
}

/// A single HIP-to-CUDA function mapping entry.
#[derive(Debug, Clone)]
pub struct HipFunctionMapping {
    /// HIP function name (e.g., "hipMalloc").
    pub hip_name: &'static str,
    /// CUDA equivalent function name (e.g., "cuMemAlloc_v2").
    pub cuda_equivalent: &'static str,
    /// Category of this function.
    pub category: HipFunctionCategory,
    /// Brief description of what R27 does with this call.
    pub action: &'static str,
}

/// Build the complete HIP-to-CUDA function mapping table.
///
/// Returns all ~120 Phase 1 + ~20 Phase 2 (Graph) function mappings
/// across 7 categories. This table is the foundation of the translation
/// layer and also serves as the hook table for the C interposition library.
pub fn build_hip_function_table() -> Vec<HipFunctionMapping> {
    let mut table = Vec::with_capacity(140);

    // Category 1: Initialization and Device Management (19 functions)
    let init_fns = [
        ("hipInit", "cuInit", "Initialize AMD backend"),
        ("hipDriverGetVersion", "cuDriverGetVersion", "Return HIP version"),
        ("hipRuntimeGetVersion", "cudaRuntimeGetVersion", "Return runtime version"),
        ("hipGetDeviceCount", "cuDeviceGetCount", "Return AMD GPU count"),
        ("hipGetDevice", "cuDeviceGet", "Return virtual device ordinal"),
        ("hipSetDevice", "cuCtxSetCurrent", "Set active GPU context"),
        ("hipGetDeviceProperties", "cuDeviceGetAttribute", "Return hipDeviceProp_t"),
        ("hipDeviceGetAttribute", "cuDeviceGetAttribute", "Translate attribute enum"),
        ("hipGetDeviceFlags", "cuCtxGetFlags", "Forward to server"),
        ("hipSetDeviceFlags", "cuCtxSetFlags", "Forward to server"),
        ("hipDeviceSynchronize", "cuCtxSynchronize", "Synchronize remote GPU"),
        ("hipDeviceReset", "cuDevicePrimaryCtxReset", "Reset remote device"),
        ("hipDeviceGetName", "cuDeviceGetName", "Return name from profile"),
        ("hipDeviceTotalMem", "cuDeviceTotalMem", "Return VRAM from profile"),
        ("hipDeviceGetPCIBusId", "cuDeviceGetPCIBusId", "Return PCI info"),
        ("hipDeviceGetByPCIBusId", "cuDeviceGetByPCIBusId", "Lookup by PCI address"),
        ("hipDeviceCanAccessPeer", "cuDeviceCanAccessPeer", "Check connectivity"),
        ("hipDeviceEnablePeerAccess", "cuCtxEnablePeerAccess", "Enable cross-GPU path"),
        ("hipDeviceDisablePeerAccess", "cuCtxDisablePeerAccess", "Disable cross-GPU path"),
    ];
    for (hip, cuda, action) in init_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Initialization,
            action,
        });
    }

    // Category 2: Context Management (13 functions)
    let ctx_fns = [
        ("hipCtxCreate", "cuCtxCreate_v2", "Create remote context"),
        ("hipCtxDestroy", "cuCtxDestroy_v2", "Destroy remote context"),
        ("hipCtxSetCurrent", "cuCtxSetCurrent", "Switch active context"),
        ("hipCtxGetCurrent", "cuCtxGetCurrent", "Return current context"),
        ("hipCtxGetDevice", "cuCtxGetDevice", "Return device for context"),
        ("hipCtxSynchronize", "cuCtxSynchronize", "Synchronize context"),
        ("hipCtxPushCurrent", "cuCtxPushCurrent", "Push to context stack"),
        ("hipCtxPopCurrent", "cuCtxPopCurrent", "Pop from context stack"),
        ("hipDevicePrimaryCtxRetain", "cuDevicePrimaryCtxRetain", "Retain primary context"),
        ("hipDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease", "Release primary context"),
        ("hipDevicePrimaryCtxGetState", "cuDevicePrimaryCtxGetState", "Query primary context"),
        ("hipDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags", "Set primary context flags"),
        ("hipDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset", "Reset primary context"),
    ];
    for (hip, cuda, action) in ctx_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Context,
            action,
        });
    }

    // Category 3: Memory Management (33 functions)
    let mem_fns = [
        ("hipMalloc", "cuMemAlloc_v2", "Allocate on remote AMD GPU"),
        ("hipFree", "cuMemFree_v2", "Free remote allocation"),
        ("hipMemcpy", "cuMemcpy", "Copy via transport"),
        ("hipMemcpyAsync", "cuMemcpyAsync", "Async copy"),
        ("hipMemcpyHtoD", "cuMemcpyHtoD_v2", "Host-to-device"),
        ("hipMemcpyDtoH", "cuMemcpyDtoH_v2", "Device-to-host"),
        ("hipMemcpyDtoD", "cuMemcpyDtoD_v2", "Device-to-device"),
        ("hipMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2", "Async host-to-device"),
        ("hipMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2", "Async device-to-host"),
        ("hipMemset", "cuMemsetD8", "Memset on remote GPU"),
        ("hipMemsetAsync", "cuMemsetD8Async", "Async memset"),
        ("hipMemsetD32", "cuMemsetD32", "32-bit memset"),
        ("hipMemsetD16", "cuMemsetD16", "16-bit memset"),
        ("hipMemGetInfo", "cuMemGetInfo_v2", "Query free/total memory"),
        ("hipHostMalloc", "cuMemAllocHost", "Allocate pinned host memory"),
        ("hipHostFree", "cuMemFreeHost", "Free pinned host memory"),
        ("hipHostRegister", "cuMemHostRegister", "Register host memory"),
        ("hipHostUnregister", "cuMemHostUnregister", "Unregister host memory"),
        ("hipHostGetDevicePointer", "cuMemHostGetDevicePointer", "Get device pointer"),
        ("hipHostGetFlags", "cuMemHostGetFlags", "Query host allocation flags"),
        ("hipMallocManaged", "cuMemAllocManaged", "Managed memory allocation"),
        ("hipMemPrefetchAsync", "cuMemPrefetchAsync", "Prefetch managed memory"),
        ("hipMemAdvise", "cuMemAdvise", "Memory advisory hints"),
        ("hipMemRangeGetAttribute", "cuMemRangeGetAttribute", "Query range attributes"),
        ("hipMallocAsync", "cuMemAllocAsync", "Pool-based async alloc"),
        ("hipFreeAsync", "cuMemFreeAsync", "Pool-based async free"),
        ("hipMemPoolCreate", "cuMemPoolCreate", "Create memory pool"),
        ("hipMemPoolDestroy", "cuMemPoolDestroy", "Destroy memory pool"),
        ("hipMemPoolGetAttribute", "cuMemPoolGetAttribute", "Query pool attribute"),
        ("hipMemPoolSetAttribute", "cuMemPoolSetAttribute", "Set pool attribute"),
        ("hipMemPoolTrimTo", "cuMemPoolTrimTo", "Trim pool to size"),
        ("hipMallocFromPoolAsync", "cuMemAllocFromPoolAsync", "Alloc from pool"),
        ("hipMemGetAddressRange", "cuMemGetAddressRange", "Query allocation range"),
    ];
    for (hip, cuda, action) in mem_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Memory,
            action,
        });
    }

    // Category 4: Module and Kernel Management (15 functions)
    let module_fns = [
        ("hipModuleLoad", "cuModuleLoad", "Load AMDGPU code object"),
        ("hipModuleLoadData", "cuModuleLoadData", "Load module from memory"),
        ("hipModuleLoadDataEx", "cuModuleLoadDataEx", "Load with options"),
        ("hipModuleUnload", "cuModuleUnload", "Unload module"),
        ("hipModuleGetFunction", "cuModuleGetFunction", "Get kernel function"),
        ("hipModuleGetGlobal", "cuModuleGetGlobal", "Get global variable"),
        ("hipModuleLaunchKernel", "cuLaunchKernel", "Launch kernel"),
        ("hipLaunchKernel", "cudaLaunchKernel", "Runtime kernel launch"),
        ("hipFuncGetAttribute", "cuFuncGetAttribute", "Query function attribute"),
        ("hipFuncSetAttribute", "cuFuncSetAttribute", "Set function attribute"),
        ("hipFuncSetCacheConfig", "cuFuncSetCacheConfig", "Set cache config"),
        ("hipFuncSetSharedMemConfig", "cuFuncSetSharedMemConfig", "Set shared mem config"),
        ("hipOccupancyMaxActiveBlocksPerMultiprocessor", "cuOccupancyMaxActiveBlocksPerMultiprocessor", "Occupancy calc"),
        ("hipOccupancyMaxPotentialBlockSize", "N/A", "Compute optimal block size"),
        ("hipGetProcAddress", "cuGetProcAddress", "Dynamic function resolution"),
    ];
    for (hip, cuda, action) in module_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Module,
            action,
        });
    }

    // Category 5: Stream and Event Management (16 functions)
    let stream_event_fns = [
        ("hipStreamCreate", "cuStreamCreate", "Create remote stream"),
        ("hipStreamCreateWithFlags", "cuStreamCreateWithFlags", "Create with flags"),
        ("hipStreamCreateWithPriority", "cuStreamCreateWithPriority", "Create with priority"),
        ("hipStreamDestroy", "cuStreamDestroy", "Destroy remote stream"),
        ("hipStreamSynchronize", "cuStreamSynchronize", "Synchronize remote stream"),
        ("hipStreamWaitEvent", "cuStreamWaitEvent", "Wait on event"),
        ("hipStreamQuery", "cuStreamQuery", "Query completion"),
        ("hipStreamAddCallback", "cuStreamAddCallback", "Register callback"),
        ("hipLaunchHostFunc", "cuLaunchHostFunc", "Host callback in stream"),
        ("hipEventCreate", "cuEventCreate", "Create remote event"),
        ("hipEventCreateWithFlags", "cuEventCreateWithFlags", "Create with flags"),
        ("hipEventDestroy", "cuEventDestroy", "Destroy remote event"),
        ("hipEventRecord", "cuEventRecord", "Record event"),
        ("hipEventSynchronize", "cuEventSynchronize", "Synchronize event"),
        ("hipEventElapsedTime", "cuEventElapsedTime", "Compute elapsed time"),
        ("hipEventQuery", "cuEventQuery", "Query event completion"),
    ];
    for (hip, cuda, action) in stream_event_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::StreamEvent,
            action,
        });
    }

    // Category 6: Error Handling (4 functions)
    let error_fns = [
        ("hipGetLastError", "cudaGetLastError", "Return last error"),
        ("hipPeekAtLastError", "cudaPeekAtLastError", "Peek at last error"),
        ("hipGetErrorName", "cuGetErrorName", "Translate error code to name"),
        ("hipGetErrorString", "cuGetErrorString", "Translate error code to string"),
    ];
    for (hip, cuda, action) in error_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Error,
            action,
        });
    }

    // Category 7: Graph API - Phase 2 (18 functions)
    let graph_fns = [
        ("hipGraphCreate", "cuGraphCreate", "Create graph"),
        ("hipGraphDestroy", "cuGraphDestroy", "Destroy graph"),
        ("hipGraphInstantiate", "cuGraphInstantiate", "Instantiate graph"),
        ("hipGraphLaunch", "cuGraphLaunch", "Launch graph"),
        ("hipGraphExecDestroy", "cuGraphExecDestroy", "Destroy exec graph"),
        ("hipStreamBeginCapture", "cuStreamBeginCapture", "Begin capture"),
        ("hipStreamEndCapture", "cuStreamEndCapture", "End capture"),
        ("hipStreamIsCapturing", "cuStreamIsCapturing", "Query capture state"),
        ("hipGraphAddKernelNode", "cuGraphAddKernelNode", "Add kernel node"),
        ("hipGraphAddMemcpyNode", "cuGraphAddMemcpyNode", "Add memcpy node"),
        ("hipGraphAddMemsetNode", "cuGraphAddMemsetNode", "Add memset node"),
        ("hipGraphAddHostNode", "cuGraphAddHostNode", "Add host node"),
        ("hipGraphAddChildGraphNode", "cuGraphAddChildGraphNode", "Add child graph"),
        ("hipGraphAddEmptyNode", "cuGraphAddEmptyNode", "Add empty node"),
        ("hipGraphGetNodes", "cuGraphGetNodes", "Get all nodes"),
        ("hipGraphGetEdges", "cuGraphGetEdges", "Get all edges"),
        ("hipGraphNodeGetType", "cuGraphNodeGetType", "Query node type"),
        ("hipGraphExecUpdate", "cuGraphExecUpdate", "Update exec graph"),
    ];
    for (hip, cuda, action) in graph_fns {
        table.push(HipFunctionMapping {
            hip_name: hip,
            cuda_equivalent: cuda,
            category: HipFunctionCategory::Graph,
            action,
        });
    }

    table
}

/// Look up a HIP function mapping by name.
///
/// Returns None if the function is not in the interception table,
/// meaning it should be forwarded to the real libamdhip64.so.
/// Lazily-built static lookup table for O(1) function resolution.
/// The table is built once on first access and reused for the lifetime
/// of the process — critical for hot-path interception performance.
static HIP_FUNCTION_TABLE: std::sync::OnceLock<HashMap<&'static str, HipFunctionMapping>> =
    std::sync::OnceLock::new();

pub fn lookup_hip_function(name: &str) -> Option<HipFunctionMapping> {
    let table = HIP_FUNCTION_TABLE.get_or_init(|| {
        build_hip_function_table()
            .into_iter()
            .map(|m| (m.hip_name, m))
            .collect()
    });
    table.get(name).cloned()
}

// ---------------------------------------------------------------------------
// AMD GPU Capability Scoring (extends R23)
// ---------------------------------------------------------------------------

/// Score an AMD GPU for a specific workload class.
///
/// Uses the same 3-tier normalization as NVIDIA GPUs:
/// static specs, calibration benchmarks, and runtime adaptation.
/// The reference GPU (RTX 3060 = 1.0) remains the same --
/// AMD GPUs are scored relative to RTX 3060 for cross-vendor fairness.
pub fn score_amd_gpu(profile: &AmdGpuProfile, class: WorkloadClass) -> f64 {
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
            let tensor_equiv = profile
                .measured_matrix_tflops
                .unwrap_or(profile.fp16_tflops * 0.5); // RDNA packed math approximation
            tensor_equiv / ref_values.tensor_tflops_fp16
        }
        WorkloadClass::Unknown => {
            // Conservative: weighted combination
            let compute = profile.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0);
            let memory = profile.measured_memory_bw_gbps / ref_values.memory_bw_gbps;
            let capacity =
                profile.vram_total_bytes as f64 / (ref_values.vram_gb * 1024.0 * 1024.0 * 1024.0);
            compute * 0.4 + memory * 0.3 + capacity * 0.3
        }
    }
}

/// Compute full WorkloadScores for an AMD GPU profile.
///
/// Populates all score dimensions and the overall weighted score.
pub fn compute_amd_workload_scores(profile: &AmdGpuProfile) -> WorkloadScores {
    let ref_values = ReferenceValues::default();

    let compute = profile.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0);
    let memory = profile.measured_memory_bw_gbps / ref_values.memory_bw_gbps;
    let tensor = {
        let tensor_equiv = profile
            .measured_matrix_tflops
            .unwrap_or(profile.fp16_tflops * 0.5);
        tensor_equiv / ref_values.tensor_tflops_fp16
    };
    let capacity =
        profile.vram_total_bytes as f64 / (ref_values.vram_gb * 1024.0 * 1024.0 * 1024.0);
    let transfer = profile.pcie_bandwidth_gbps / ref_values.pcie_bw_gbps;

    let overall = compute * 0.30 + memory * 0.25 + tensor * 0.20 + capacity * 0.15 + transfer * 0.10;

    WorkloadScores {
        compute,
        memory,
        tensor,
        capacity,
        transfer,
        overall,
    }
}

/// Compute GEU (GPU Equivalent Units) for an AMD GPU.
///
/// Uses the same formula as NVIDIA:
/// GEU = (compute_score * 0.4) + (bandwidth_score * 0.3) + (capacity_score * 0.3)
pub fn compute_amd_geu(profile: &AmdGpuProfile) -> f64 {
    let ref_values = ReferenceValues::default();
    let compute = profile.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0);
    let bandwidth = profile.measured_memory_bw_gbps / ref_values.memory_bw_gbps;
    let capacity =
        profile.vram_total_bytes as f64 / (ref_values.vram_gb * 1024.0 * 1024.0 * 1024.0);
    compute * 0.4 + bandwidth * 0.3 + capacity * 0.3
}

// ---------------------------------------------------------------------------
// Cross-Vendor Migration Path
// ---------------------------------------------------------------------------

/// Migration path selection for cross-vendor or cross-node transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPath {
    /// Direct device-to-device copy (same vendor, same node).
    DirectDtoD,
    /// Host-staged transfer (through pinned host memory).
    HostStaged,
}

/// Select the optimal migration path based on vendor and node topology.
///
/// Rules:
/// - Same vendor + same node: direct DtoD (vendor's native copy)
/// - Same vendor + different node: host-staged (network transfer)
/// - Cross vendor: always host-staged (no direct AMD<->NVIDIA DMA)
pub fn select_migration_path(
    src_vendor: GpuVendor,
    dst_vendor: GpuVendor,
    same_node: bool,
) -> MigrationPath {
    if src_vendor == dst_vendor && same_node {
        MigrationPath::DirectDtoD
    } else {
        // Cross-vendor or cross-node: always host-staged
        MigrationPath::HostStaged
    }
}

// ---------------------------------------------------------------------------
// GPU Pool Filtering
// ---------------------------------------------------------------------------

/// Filter GPUs by binary compatibility with a workload.
///
/// Returns only GPUs that can execute the given binary format.
/// This is a hard constraint -- incompatible GPUs are excluded entirely.
pub fn filter_compatible_gpus<'a>(
    binary_format: &BinaryFormat,
    candidates: &'a [UnifiedGpuProfile],
) -> Vec<&'a UnifiedGpuProfile> {
    candidates
        .iter()
        .filter(|gpu| gpu.is_binary_compatible(binary_format))
        .collect()
}

/// Filter GPUs by vendor.
pub fn filter_by_vendor<'a>(
    vendor: GpuVendor,
    candidates: &'a [UnifiedGpuProfile],
) -> Vec<&'a UnifiedGpuProfile> {
    candidates.iter().filter(|gpu| gpu.vendor() == vendor).collect()
}

/// Count GPUs by vendor in a pool.
pub fn count_by_vendor(vendor: GpuVendor, pool: &[UnifiedGpuProfile]) -> usize {
    pool.iter().filter(|gpu| gpu.vendor() == vendor).count()
}

// ---------------------------------------------------------------------------
// HIP Intercept Client (Client-Side State)
// ---------------------------------------------------------------------------

/// Client-side state for HIP interception.
///
/// Mirrors OuterLinkClient but for the HIP API surface.
/// Lives in the LD_PRELOAD library alongside (or instead of) OuterLinkClient.
/// When a process links against both libamdhip64.so and libcuda.so, both
/// clients coexist -- they share a single connection to the OuterLink server
/// but maintain separate handle namespaces.
pub struct HipInterceptClient {
    /// Server address (shared with CUDA client if both are active).
    pub server_addr: String,
    /// Connection state.
    pub connected: AtomicBool,
    /// Monotonically increasing request ID.
    next_request_id: AtomicU64,
    /// The remote HIP context handle currently active.
    pub current_remote_ctx: AtomicU64,
    /// Session ID assigned by the server.
    session_id: AtomicU64,
    /// Vendor tag (always Amd).
    pub vendor: GpuVendor,
    /// Cache of device properties fetched from server.
    device_props_cache: std::sync::Mutex<HashMap<i32, HipDeviceProps>>,
    /// Last HIP error code (per-thread in production, simplified here).
    last_error: AtomicU64,
    /// Tracked allocations (device_ptr -> size) for hipMemGetAddressRange.
    allocations: std::sync::Mutex<HashMap<u64, u64>>,
    /// Module entries (local_handle -> entry).
    modules: std::sync::Mutex<HashMap<u64, HipModuleEntry>>,
    /// Handle translation table: synthetic local handles <-> real remote handles.
    /// Maps local_handle -> remote_handle for all object types.
    // TODO: Production uses HandleStore with typed sub-tables per object kind.
    handle_map: std::sync::Mutex<HashMap<u64, u64>>,
    /// Whether a reconnect is currently in progress (prevents concurrent reconnects).
    reconnect_in_progress: std::sync::Mutex<bool>,
    /// Number of reconnection attempts made in the current session.
    reconnect_attempts: AtomicU64,
    /// Whether the callback listener thread is running.
    callback_listener_running: AtomicBool,
}

impl HipInterceptClient {
    /// Create a new HIP intercept client.
    ///
    /// Does not connect immediately -- call `connect()` to establish
    /// the server connection.
    pub fn new(server_addr: String) -> Self {
        Self {
            server_addr,
            connected: AtomicBool::new(false),
            next_request_id: AtomicU64::new(1),
            current_remote_ctx: AtomicU64::new(0),
            session_id: AtomicU64::new(0),
            vendor: GpuVendor::Amd,
            device_props_cache: std::sync::Mutex::new(HashMap::new()),
            last_error: AtomicU64::new(0),
            allocations: std::sync::Mutex::new(HashMap::new()),
            modules: std::sync::Mutex::new(HashMap::new()),
            handle_map: std::sync::Mutex::new(HashMap::new()),
            reconnect_in_progress: std::sync::Mutex::new(false),
            reconnect_attempts: AtomicU64::new(0),
            callback_listener_running: AtomicBool::new(false),
        }
    }

    /// Get the next request ID.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Check if connected to the server.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }

    /// Set connected state.
    pub fn set_connected(&self, connected: bool) {
        self.connected.store(connected, Ordering::Release);
    }

    /// Get the session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id.load(Ordering::Acquire)
    }

    /// Set the session ID.
    pub fn set_session_id(&self, id: u64) {
        self.session_id.store(id, Ordering::Release);
    }

    /// Get the last HIP error code.
    pub fn get_last_error(&self) -> HipError {
        HipError::from_u32(self.last_error.load(Ordering::Acquire) as u32)
    }

    /// Set the last error and return it (matches hipGetLastError semantics:
    /// returns the error and resets to Success).
    pub fn get_and_reset_last_error(&self) -> HipError {
        let err = self.last_error.swap(0, Ordering::AcqRel);
        HipError::from_u32(err as u32)
    }

    /// Peek at the last error without resetting it.
    pub fn peek_last_error(&self) -> HipError {
        HipError::from_u32(self.last_error.load(Ordering::Acquire) as u32)
    }

    /// Record an error.
    pub fn set_last_error(&self, error: HipError) {
        self.last_error
            .store(error.as_u32() as u64, Ordering::Release);
    }

    /// Cache device properties for a device ordinal.
    pub fn cache_device_props(&self, device: i32, props: HipDeviceProps) {
        self.device_props_cache
            .lock()
            .unwrap()
            .insert(device, props);
    }

    /// Get cached device properties. Returns None on cache miss.
    pub fn get_cached_device_props(&self, device: i32) -> Option<HipDeviceProps> {
        self.device_props_cache
            .lock()
            .unwrap()
            .get(&device)
            .cloned()
    }

    /// Invalidate cached device properties (e.g., on device reset).
    pub fn invalidate_device_props(&self, device: i32) {
        self.device_props_cache.lock().unwrap().remove(&device);
    }

    /// Invalidate all cached device properties.
    pub fn invalidate_all_device_props(&self) {
        self.device_props_cache.lock().unwrap().clear();
    }

    /// Track a memory allocation.
    pub fn track_allocation(&self, ptr: u64, size: u64) {
        self.allocations.lock().unwrap().insert(ptr, size);
    }

    /// Untrack a memory allocation. Returns size if found.
    pub fn untrack_allocation(&self, ptr: u64) -> Option<u64> {
        self.allocations.lock().unwrap().remove(&ptr)
    }

    /// Get allocation size for a pointer.
    pub fn get_allocation_size(&self, ptr: u64) -> Option<u64> {
        self.allocations.lock().unwrap().get(&ptr).copied()
    }

    /// Get allocation count.
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }

    /// Register a loaded module.
    pub fn register_module(&self, entry: HipModuleEntry) {
        self.modules
            .lock()
            .unwrap()
            .insert(entry.local_handle, entry);
    }

    /// Unregister a module. Returns the entry if found.
    pub fn unregister_module(&self, local_handle: u64) -> Option<HipModuleEntry> {
        self.modules.lock().unwrap().remove(&local_handle)
    }

    /// Get a module entry by local handle.
    pub fn get_module(&self, local_handle: u64) -> Option<HipModuleEntry> {
        self.modules.lock().unwrap().get(&local_handle).cloned()
    }

    /// Get module count.
    pub fn module_count(&self) -> usize {
        self.modules.lock().unwrap().len()
    }

    /// Connect to the OuterLink server.
    ///
    /// TODO: Implement actual TCP connection via transport layer.
    /// This is a stub that sets the connected flag for testing.
    pub fn connect(&self) -> Result<(), HipError> {
        // TODO: Real connection via TcpTransport
        self.set_connected(true);
        Ok(())
    }

    /// Disconnect from the server.
    ///
    /// Resets ALL client state to prevent stale data on reconnect:
    /// atomics (error, context, session), allocation/module maps, handle map,
    /// device props cache, and callback listener flag.
    pub fn disconnect(&self) {
        self.set_connected(false);
        self.last_error.store(0, Ordering::Release);
        self.current_remote_ctx.store(0, Ordering::Release);
        self.session_id.store(0, Ordering::Release);
        self.callback_listener_running.store(false, Ordering::Release);
        self.allocations.lock().unwrap().clear();
        self.modules.lock().unwrap().clear();
        self.device_props_cache.lock().unwrap().clear();
        self.handle_map.lock().unwrap().clear();
    }
}

// ---------------------------------------------------------------------------
// HIP Translator
// ---------------------------------------------------------------------------

/// The translation layer between HIP interception hooks and the OuterLink
/// wire protocol.
///
/// Converts HIP API semantics into vendor-neutral protocol messages.
/// In production, holds a reference to `HipInterceptClient` for sending
/// wire messages. This common-crate version provides the translation
/// logic without the actual network transport.
pub struct HipTranslator {
    /// API vendor tag for all outgoing messages.
    pub api_vendor: ApiVendor,
}

impl HipTranslator {
    /// Create a new HIP translator.
    pub fn new() -> Self {
        Self {
            api_vendor: ApiVendor::Hip,
        }
    }

    /// Translate hipMemcpyKind to OuterLink MemcpyDirection.
    ///
    /// For `hipMemcpyDefault`, uses pointer type detection parameters
    /// to determine the actual direction.
    pub fn translate_memcpy_direction(
        &self,
        kind: HipMemcpyKind,
        src_is_device: bool,
        dst_is_device: bool,
    ) -> MemcpyDirection {
        kind.to_memcpy_direction(src_is_device, dst_is_device)
    }

    /// Translate a HIP error code from the server response.
    pub fn translate_error(&self, code: u32) -> HipError {
        HipError::from_u32(code)
    }

    /// Build device properties from an AMD GPU profile.
    pub fn translate_device_props(&self, profile: &AmdGpuProfile) -> HipDeviceProps {
        HipDeviceProps::from_profile(profile)
    }

    /// Determine if a function name should be intercepted.
    pub fn should_intercept(&self, function_name: &str) -> bool {
        lookup_hip_function(function_name).is_some()
    }
}

impl Default for HipTranslator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: Create test profiles
// ---------------------------------------------------------------------------

/// Create a test AmdGpuProfile for RX 7900 XTX.
///
/// Used in unit tests and benchmarks.
#[cfg(test)]
fn make_test_rx7900xtx(gpu_id: GpuId, node_id: NodeId) -> AmdGpuProfile {
    AmdGpuProfile {
        gpu_id,
        node_id,
        vendor: GpuVendor::Amd,
        name: "AMD Radeon RX 7900 XTX".to_string(),
        arch: AmdGpuArch::from_gfx_name("gfx1100").unwrap(),
        compute_unit_count: 96,
        stream_processors_per_cu: 128,
        total_stream_processors: 12288,
        matrix_alu_count: 0,
        matrix_alu_gen: None,
        wavefront_size: 32,
        vram_total_bytes: 24 * 1024 * 1024 * 1024,
        vram_type: "GDDR6".to_string(),
        memory_bandwidth_gbps: 960.0,
        fp32_tflops: 61.4,
        fp16_tflops: 122.8,
        bf16_tflops: 122.8,
        int8_tops: 245.6,
        pcie_gen: 4,
        pcie_width: 16,
        pcie_bandwidth_gbps: 25.0,
        sdma_engine_count: 2,
        l2_cache_bytes: 6 * 1024 * 1024,
        infinity_cache_bytes: 96 * 1024 * 1024,
        boost_clock_mhz: 2500,
        tdp_watts: 355,
        supports_fp16: true,
        supports_bf16: true,
        supports_fp8: false,
        supports_int8: true,
        supports_packed_math: true,
        rocm_version: "6.3.0".to_string(),
        hip_runtime_version: 60300000,
        hsa_version: "1.2".to_string(),
        kernel_driver_version: "6.7.0".to_string(),
        measured_fp32_gflops: 55000.0,
        measured_memory_bw_gbps: 850.0,
        measured_h2d_bw_gbps: 24.0,
        measured_d2h_bw_gbps: 24.0,
        measured_matrix_tflops: None,
        vram_free_bytes: 20 * 1024 * 1024 * 1024,
        utilization: 0.0,
        temperature_c: 45,
        current_clock_mhz: 2500,
        power_draw_watts: 100,
        is_throttling: false,
        capability_scores: WorkloadScores::default(),
        geu: 0.0,
    }
}

/// Create a test AmdGpuProfile for MI250 (CDNA2).
#[cfg(test)]
fn make_test_mi250(gpu_id: GpuId, node_id: NodeId) -> AmdGpuProfile {
    AmdGpuProfile {
        gpu_id,
        node_id,
        vendor: GpuVendor::Amd,
        name: "AMD Instinct MI250".to_string(),
        arch: AmdGpuArch::from_gfx_name("gfx90a").unwrap(),
        compute_unit_count: 104,
        stream_processors_per_cu: 64,
        total_stream_processors: 6656,
        matrix_alu_count: 104,
        matrix_alu_gen: Some(2),
        wavefront_size: 64,
        vram_total_bytes: 64 * 1024 * 1024 * 1024,
        vram_type: "HBM2e".to_string(),
        memory_bandwidth_gbps: 1638.0,
        fp32_tflops: 45.3,
        fp16_tflops: 362.0,
        bf16_tflops: 362.0,
        int8_tops: 362.0,
        pcie_gen: 4,
        pcie_width: 16,
        pcie_bandwidth_gbps: 25.0,
        sdma_engine_count: 4,
        l2_cache_bytes: 8 * 1024 * 1024,
        infinity_cache_bytes: 0,
        boost_clock_mhz: 1700,
        tdp_watts: 500,
        supports_fp16: true,
        supports_bf16: true,
        supports_fp8: false,
        supports_int8: true,
        supports_packed_math: true,
        rocm_version: "6.3.0".to_string(),
        hip_runtime_version: 60300000,
        hsa_version: "1.2".to_string(),
        kernel_driver_version: "6.7.0".to_string(),
        measured_fp32_gflops: 42000.0,
        measured_memory_bw_gbps: 1500.0,
        measured_h2d_bw_gbps: 24.0,
        measured_d2h_bw_gbps: 24.0,
        measured_matrix_tflops: Some(180.0),
        vram_free_bytes: 60 * 1024 * 1024 * 1024,
        utilization: 0.0,
        temperature_c: 50,
        current_clock_mhz: 1700,
        power_draw_watts: 200,
        is_throttling: false,
        capability_scores: WorkloadScores::default(),
        geu: 0.0,
    }
}

/// Create a test NVIDIA GpuProfile for RTX 3090.
#[cfg(test)]
fn make_test_rtx3090(gpu_id: GpuId, node_id: NodeId) -> GpuProfile {
    GpuProfile {
        gpu_id,
        node_id,
        name: "NVIDIA GeForce RTX 3090".to_string(),
        compute_capability: (8, 6),
        sm_count: 82,
        cuda_cores: 10496,
        tensor_core_count: 328,
        tensor_core_gen: Some(3),
        vram_total_bytes: 24 * 1024 * 1024 * 1024,
        memory_bandwidth_gbps: 936.0,
        fp32_tflops: 35.6,
        fp16_tflops: 71.2,
        bar1_size_bytes: 256 * 1024 * 1024,
        pcie_gen: 4,
        pcie_width: 16,
        pcie_bandwidth_gbps: 25.0,
        async_engine_count: 2,
        l2_cache_bytes: 6 * 1024 * 1024,
        boost_clock_mhz: 1695,
        tdp_watts: 350,
        supports_fp16: true,
        supports_bf16: true,
        supports_tf32: true,
        supports_fp8: false,
        supports_fp4: false,
        supports_int8: true,
        driver_version: "550.54.14".to_string(),
        cuda_driver_version: 12040,
        max_cuda_toolkit: (12, 4),
        measured_fp32_gflops: 33000.0,
        measured_memory_bw_gbps: 850.0,
        measured_h2d_bw_gbps: 24.0,
        measured_d2h_bw_gbps: 24.0,
        measured_tensor_tflops: Some(65.0),
        vram_free_bytes: 20 * 1024 * 1024 * 1024,
        utilization: 0.0,
        temperature_c: 40,
        current_clock_mhz: 1695,
        power_draw_watts: 100,
        is_throttling: false,
        capability_scores: WorkloadScores::default(),
        geu: 0.0,
        has_rebar: false,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AmdGpuArch::from_gfx_name parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_gfx906_gcn() {
        let arch = AmdGpuArch::from_gfx_name("gfx906").unwrap();
        assert_eq!(arch.major, 9);
        assert_eq!(arch.minor, 0);
        assert_eq!(arch.stepping, 6);
        assert_eq!(arch.family, AmdArchFamily::Gcn);
        assert_eq!(arch.native_wavefront_size, 64);
        assert!(!arch.supports_wave32);
        assert!(!arch.has_matrix_alu);
    }

    #[test]
    fn test_parse_gfx908_cdna() {
        let arch = AmdGpuArch::from_gfx_name("gfx908").unwrap();
        assert_eq!(arch.major, 9);
        assert_eq!(arch.minor, 0);
        assert_eq!(arch.stepping, 8);
        assert_eq!(arch.family, AmdArchFamily::Cdna);
        assert!(arch.has_matrix_alu);
        assert!(!arch.supports_wave32);
    }

    #[test]
    fn test_parse_gfx90a_cdna2() {
        let arch = AmdGpuArch::from_gfx_name("gfx90a").unwrap();
        assert_eq!(arch.major, 9);
        assert_eq!(arch.minor, 0);
        assert_eq!(arch.stepping, 10); // 'a' = 10
        assert_eq!(arch.family, AmdArchFamily::Cdna);
        assert!(arch.has_matrix_alu);
    }

    #[test]
    fn test_parse_gfx940_cdna3() {
        let arch = AmdGpuArch::from_gfx_name("gfx940").unwrap();
        assert_eq!(arch.major, 9);
        assert_eq!(arch.minor, 4);
        assert_eq!(arch.stepping, 0);
        assert_eq!(arch.family, AmdArchFamily::Cdna);
        assert!(arch.has_matrix_alu);
    }

    #[test]
    fn test_parse_gfx942_mi300x() {
        let arch = AmdGpuArch::from_gfx_name("gfx942").unwrap();
        assert_eq!(arch.major, 9);
        assert_eq!(arch.minor, 4);
        assert_eq!(arch.stepping, 2);
        assert_eq!(arch.family, AmdArchFamily::Cdna);
    }

    #[test]
    fn test_parse_gfx1030_rdna2() {
        let arch = AmdGpuArch::from_gfx_name("gfx1030").unwrap();
        assert_eq!(arch.major, 10);
        assert_eq!(arch.minor, 3);
        assert_eq!(arch.stepping, 0);
        assert_eq!(arch.family, AmdArchFamily::Rdna);
        assert_eq!(arch.native_wavefront_size, 32);
        assert!(arch.supports_wave32);
        assert!(!arch.has_matrix_alu);
    }

    #[test]
    fn test_parse_gfx1100_rdna3() {
        let arch = AmdGpuArch::from_gfx_name("gfx1100").unwrap();
        assert_eq!(arch.major, 11);
        assert_eq!(arch.minor, 0);
        assert_eq!(arch.stepping, 0);
        assert_eq!(arch.family, AmdArchFamily::Rdna);
        assert!(arch.supports_wave32);
    }

    #[test]
    fn test_parse_gfx1201_rdna4() {
        let arch = AmdGpuArch::from_gfx_name("gfx1201").unwrap();
        assert_eq!(arch.major, 12);
        assert_eq!(arch.minor, 0);
        assert_eq!(arch.stepping, 1);
        assert_eq!(arch.family, AmdArchFamily::Rdna);
    }

    #[test]
    fn test_parse_gfx900_gcn_vega() {
        let arch = AmdGpuArch::from_gfx_name("gfx900").unwrap();
        assert_eq!(arch.family, AmdArchFamily::Gcn);
        assert!(!arch.has_matrix_alu);
    }

    #[test]
    fn test_parse_invalid_no_prefix() {
        assert!(AmdGpuArch::from_gfx_name("1100").is_none());
    }

    #[test]
    fn test_parse_invalid_too_short() {
        assert!(AmdGpuArch::from_gfx_name("gfx").is_none());
        assert!(AmdGpuArch::from_gfx_name("gfx9").is_none());
        assert!(AmdGpuArch::from_gfx_name("gfx90").is_none());
    }

    #[test]
    fn test_parse_invalid_non_hex() {
        assert!(AmdGpuArch::from_gfx_name("gfxZZZ").is_none());
    }

    #[test]
    fn test_parse_gfx_name_roundtrip() {
        for name in &[
            "gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx942", "gfx1030", "gfx1100",
            "gfx1101", "gfx1200", "gfx1201",
        ] {
            let arch = AmdGpuArch::from_gfx_name(name).unwrap();
            assert_eq!(&arch.gfx_name, name);
        }
    }

    // -----------------------------------------------------------------------
    // AmdGpuArch::is_compatible_with tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compat_exact_match() {
        let arch = AmdGpuArch::from_gfx_name("gfx1100").unwrap();
        assert!(arch.is_compatible_with("gfx1100"));
    }

    #[test]
    fn test_compat_rdna3_group() {
        let arch = AmdGpuArch::from_gfx_name("gfx1100").unwrap();
        assert!(arch.is_compatible_with("gfx1101"));
        assert!(arch.is_compatible_with("gfx1102"));
        assert!(arch.is_compatible_with("gfx1103"));
    }

    #[test]
    fn test_compat_rdna2_group() {
        let arch = AmdGpuArch::from_gfx_name("gfx1030").unwrap();
        assert!(arch.is_compatible_with("gfx1031"));
        assert!(arch.is_compatible_with("gfx1032"));
        assert!(arch.is_compatible_with("gfx1036"));
    }

    #[test]
    fn test_compat_rdna4_group() {
        let arch = AmdGpuArch::from_gfx_name("gfx1200").unwrap();
        assert!(arch.is_compatible_with("gfx1201"));
    }

    #[test]
    fn test_compat_gcn_group() {
        let arch = AmdGpuArch::from_gfx_name("gfx900").unwrap();
        assert!(arch.is_compatible_with("gfx902"));
    }

    #[test]
    fn test_incompat_cross_family() {
        let rdna3 = AmdGpuArch::from_gfx_name("gfx1100").unwrap();
        assert!(!rdna3.is_compatible_with("gfx906"));
        assert!(!rdna3.is_compatible_with("gfx90a"));
        assert!(!rdna3.is_compatible_with("gfx1030"));
    }

    #[test]
    fn test_incompat_cdna_vs_gcn() {
        let cdna = AmdGpuArch::from_gfx_name("gfx908").unwrap();
        assert!(!cdna.is_compatible_with("gfx906"));
    }

    #[test]
    fn test_incompat_rdna3_vs_rdna4() {
        let rdna3 = AmdGpuArch::from_gfx_name("gfx1100").unwrap();
        assert!(!rdna3.is_compatible_with("gfx1200"));
    }

    #[test]
    fn test_incompat_rdna2_vs_rdna1() {
        let rdna2 = AmdGpuArch::from_gfx_name("gfx1030").unwrap();
        assert!(!rdna2.is_compatible_with("gfx1010"));
    }

    // -----------------------------------------------------------------------
    // HipError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hip_error_from_u32() {
        assert_eq!(HipError::from_u32(0), HipError::Success);
        assert_eq!(HipError::from_u32(1), HipError::InvalidValue);
        assert_eq!(HipError::from_u32(2), HipError::OutOfMemory);
        assert_eq!(HipError::from_u32(3), HipError::NotInitialized);
        assert_eq!(HipError::from_u32(4), HipError::Deinitialized);
        assert_eq!(HipError::from_u32(101), HipError::InvalidDevice);
        assert_eq!(HipError::from_u32(201), HipError::InvalidContext);
        assert_eq!(HipError::from_u32(21), HipError::InvalidMemcpyDirection);
        assert_eq!(HipError::from_u32(719), HipError::LaunchFailure);
        assert_eq!(HipError::from_u32(801), HipError::NotSupported);
        assert_eq!(HipError::from_u32(999), HipError::Unknown);
    }

    #[test]
    fn test_hip_error_unknown_code() {
        assert_eq!(HipError::from_u32(12345), HipError::Unknown);
        assert_eq!(HipError::from_u32(u32::MAX), HipError::Unknown);
    }

    #[test]
    fn test_hip_error_roundtrip() {
        for error in [
            HipError::Success,
            HipError::InvalidValue,
            HipError::OutOfMemory,
            HipError::NotInitialized,
            HipError::InvalidDevice,
            HipError::LaunchFailure,
            HipError::NotSupported,
            HipError::Unknown,
        ] {
            assert_eq!(HipError::from_u32(error.as_u32()), error);
        }
    }

    #[test]
    fn test_hip_error_names() {
        assert_eq!(HipError::Success.error_name(), "hipSuccess");
        assert_eq!(HipError::OutOfMemory.error_name(), "hipErrorOutOfMemory");
        assert_eq!(HipError::InvalidDevice.error_name(), "hipErrorInvalidDevice");
    }

    #[test]
    fn test_hip_error_strings() {
        assert_eq!(HipError::Success.error_string(), "no error");
        assert_eq!(HipError::OutOfMemory.error_string(), "out of memory");
    }

    #[test]
    fn test_hip_error_is_success() {
        assert!(HipError::Success.is_success());
        assert!(!HipError::OutOfMemory.is_success());
        assert!(!HipError::Unknown.is_success());
    }

    #[test]
    fn test_hip_error_display() {
        let s = format!("{}", HipError::OutOfMemory);
        assert!(s.contains("hipErrorOutOfMemory"));
        assert!(s.contains("out of memory"));
    }

    // -----------------------------------------------------------------------
    // HipMemcpyKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_memcpy_kind_from_u32() {
        assert_eq!(HipMemcpyKind::from_u32(0), Some(HipMemcpyKind::HostToHost));
        assert_eq!(HipMemcpyKind::from_u32(1), Some(HipMemcpyKind::HostToDevice));
        assert_eq!(HipMemcpyKind::from_u32(2), Some(HipMemcpyKind::DeviceToHost));
        assert_eq!(HipMemcpyKind::from_u32(3), Some(HipMemcpyKind::DeviceToDevice));
        assert_eq!(HipMemcpyKind::from_u32(4), Some(HipMemcpyKind::Default));
        assert_eq!(HipMemcpyKind::from_u32(5), None);
    }

    #[test]
    fn test_memcpy_kind_to_direction_explicit() {
        assert_eq!(
            HipMemcpyKind::HostToHost.to_memcpy_direction(false, false),
            MemcpyDirection::HostToHost
        );
        assert_eq!(
            HipMemcpyKind::HostToDevice.to_memcpy_direction(false, true),
            MemcpyDirection::HostToDevice
        );
        assert_eq!(
            HipMemcpyKind::DeviceToHost.to_memcpy_direction(true, false),
            MemcpyDirection::DeviceToHost
        );
        assert_eq!(
            HipMemcpyKind::DeviceToDevice.to_memcpy_direction(true, true),
            MemcpyDirection::DeviceToDevice
        );
    }

    #[test]
    fn test_memcpy_default_h2h() {
        assert_eq!(
            HipMemcpyKind::Default.to_memcpy_direction(false, false),
            MemcpyDirection::HostToHost
        );
    }

    #[test]
    fn test_memcpy_default_h2d() {
        assert_eq!(
            HipMemcpyKind::Default.to_memcpy_direction(false, true),
            MemcpyDirection::HostToDevice
        );
    }

    #[test]
    fn test_memcpy_default_d2h() {
        assert_eq!(
            HipMemcpyKind::Default.to_memcpy_direction(true, false),
            MemcpyDirection::DeviceToHost
        );
    }

    #[test]
    fn test_memcpy_default_d2d() {
        assert_eq!(
            HipMemcpyKind::Default.to_memcpy_direction(true, true),
            MemcpyDirection::DeviceToDevice
        );
    }

    // -----------------------------------------------------------------------
    // ApiVendor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_vendor_from_u8() {
        assert_eq!(ApiVendor::from_u8(0), Some(ApiVendor::Cuda));
        assert_eq!(ApiVendor::from_u8(1), Some(ApiVendor::Hip));
        assert_eq!(ApiVendor::from_u8(2), None);
        assert_eq!(ApiVendor::from_u8(255), None);
    }

    #[test]
    fn test_api_vendor_default() {
        assert_eq!(ApiVendor::default(), ApiVendor::Cuda);
    }

    // -----------------------------------------------------------------------
    // GpuVendor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(format!("{}", GpuVendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", GpuVendor::Amd), "AMD");
    }

    #[test]
    fn test_gpu_vendor_equality() {
        assert_eq!(GpuVendor::Nvidia, GpuVendor::Nvidia);
        assert_eq!(GpuVendor::Amd, GpuVendor::Amd);
        assert_ne!(GpuVendor::Nvidia, GpuVendor::Amd);
    }

    // -----------------------------------------------------------------------
    // BinaryFormat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_binary_format_detect_cubin() {
        let data = [0xb1, 0x43, 0x62, 0x46, 0x00, 0x00, 0x00, 0x00]; // Magic LE
        let fmt = BinaryFormat::detect(&data);
        assert_eq!(fmt, Some(BinaryFormat::Cubin));
    }

    #[test]
    fn test_binary_format_detect_amdgpu_elf() {
        let mut data = vec![0x7f, b'E', b'L', b'F']; // ELF magic
        data.extend_from_slice(&[0; 14]); // pad to offset 18
        data.push(0xe0); // e_machine LE low byte
        data.push(0x00); // e_machine LE high byte
        let fmt = BinaryFormat::detect(&data);
        assert_eq!(fmt, Some(BinaryFormat::AmdgpuIsa("unknown".to_string())));
    }

    #[test]
    fn test_binary_format_detect_ptx() {
        let data = b".version 7.0\n.target sm_80";
        let fmt = BinaryFormat::detect(data);
        assert_eq!(fmt, Some(BinaryFormat::Ptx));
    }

    #[test]
    fn test_binary_format_detect_too_short() {
        assert_eq!(BinaryFormat::detect(&[0, 1, 2]), None);
    }

    #[test]
    fn test_binary_format_detect_unknown() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(BinaryFormat::detect(&data), None);
    }

    // -----------------------------------------------------------------------
    // UnifiedGpuProfile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_profile_nvidia() {
        let profile = make_test_rtx3090(0, 0);
        let unified = UnifiedGpuProfile::Nvidia(profile);
        assert_eq!(unified.gpu_id(), 0);
        assert_eq!(unified.node_id(), 0);
        assert_eq!(unified.vendor(), GpuVendor::Nvidia);
        assert!(unified.is_vendor(GpuVendor::Nvidia));
        assert!(!unified.is_vendor(GpuVendor::Amd));
    }

    #[test]
    fn test_unified_profile_amd() {
        let profile = make_test_rx7900xtx(1, 1);
        let unified = UnifiedGpuProfile::Amd(profile);
        assert_eq!(unified.gpu_id(), 1);
        assert_eq!(unified.node_id(), 1);
        assert_eq!(unified.vendor(), GpuVendor::Amd);
        assert!(unified.is_vendor(GpuVendor::Amd));
    }

    #[test]
    fn test_unified_profile_nvidia_binary_compat() {
        let profile = make_test_rtx3090(0, 0);
        let unified = UnifiedGpuProfile::Nvidia(profile);
        assert!(unified.is_binary_compatible(&BinaryFormat::Ptx));
        assert!(unified.is_binary_compatible(&BinaryFormat::Cubin));
        assert!(!unified.is_binary_compatible(&BinaryFormat::AmdgpuIsa("gfx1100".to_string())));
        assert!(unified.is_binary_compatible(&BinaryFormat::HipSource));
    }

    #[test]
    fn test_unified_profile_amd_binary_compat() {
        let profile = make_test_rx7900xtx(1, 0);
        let unified = UnifiedGpuProfile::Amd(profile);
        assert!(!unified.is_binary_compatible(&BinaryFormat::Ptx));
        assert!(!unified.is_binary_compatible(&BinaryFormat::Cubin));
        assert!(unified.is_binary_compatible(&BinaryFormat::AmdgpuIsa("gfx1100".to_string())));
        assert!(unified.is_binary_compatible(&BinaryFormat::AmdgpuIsa("gfx1101".to_string())));
        assert!(!unified.is_binary_compatible(&BinaryFormat::AmdgpuIsa("gfx906".to_string())));
        assert!(unified.is_binary_compatible(&BinaryFormat::HipSource));
    }

    #[test]
    fn test_unified_profile_availability() {
        let mut profile = make_test_rx7900xtx(1, 0);
        let unified = UnifiedGpuProfile::Amd(profile.clone());
        assert!(unified.is_available());

        profile.is_throttling = true;
        let unified = UnifiedGpuProfile::Amd(profile.clone());
        assert!(!unified.is_available());

        profile.is_throttling = false;
        profile.utilization = 0.96;
        let unified = UnifiedGpuProfile::Amd(profile);
        assert!(!unified.is_available());
    }

    #[test]
    fn test_unified_profile_accessors() {
        let profile = make_test_rx7900xtx(5, 2);
        let unified = UnifiedGpuProfile::Amd(profile.clone());
        assert_eq!(unified.name(), "AMD Radeon RX 7900 XTX");
        assert_eq!(unified.vram_total_bytes(), profile.vram_total_bytes);
        assert_eq!(unified.vram_free_bytes(), profile.vram_free_bytes);
        assert!((unified.fp32_tflops() - profile.fp32_tflops).abs() < f64::EPSILON);
        assert!((unified.memory_bandwidth_gbps() - profile.memory_bandwidth_gbps).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // HipDeviceProps tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_device_props_from_profile() {
        let profile = make_test_rx7900xtx(0, 0);
        let props = HipDeviceProps::from_profile(&profile);
        assert_eq!(props.name, "AMD Radeon RX 7900 XTX");
        assert_eq!(props.total_global_mem, profile.vram_total_bytes);
        assert_eq!(props.warp_size, 32); // RDNA wave32
        assert_eq!(props.multi_processor_count, 96);
        assert_eq!(props.gcn_arch_name, "gfx1100");
        assert_eq!(props.max_threads_per_block, 1024);
        assert_eq!(props.clock_rate, 2500 * 1000); // kHz
    }

    #[test]
    fn test_device_props_cdna_wave64() {
        let profile = make_test_mi250(0, 0);
        let props = HipDeviceProps::from_profile(&profile);
        assert_eq!(props.warp_size, 64); // CDNA wave64
        assert_eq!(props.multi_processor_count, 104);
        assert_eq!(props.gcn_arch_name, "gfx90a");
    }

    // -----------------------------------------------------------------------
    // Capability Scoring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_amd_compute_bound() {
        let profile = make_test_rx7900xtx(0, 0);
        let score = score_amd_gpu(&profile, WorkloadClass::ComputeBound);
        // 55000 / (12.7 * 1000) = ~4.33
        assert!(score > 4.0 && score < 5.0, "compute score: {score}");
    }

    #[test]
    fn test_score_amd_memory_bound() {
        let profile = make_test_rx7900xtx(0, 0);
        let score = score_amd_gpu(&profile, WorkloadClass::MemoryBound);
        // 850 / 360 = ~2.36
        assert!(score > 2.0 && score < 3.0, "memory score: {score}");
    }

    #[test]
    fn test_score_amd_tensor_rdna_fallback() {
        let profile = make_test_rx7900xtx(0, 0);
        let score = score_amd_gpu(&profile, WorkloadClass::TensorBound);
        // No matrix ALUs, uses fp16 * 0.5 = 61.4 / 50.6 = ~1.21
        assert!(score > 1.0 && score < 1.5, "tensor score: {score}");
    }

    #[test]
    fn test_score_amd_tensor_cdna_matrix_alu() {
        let profile = make_test_mi250(0, 0);
        let score = score_amd_gpu(&profile, WorkloadClass::TensorBound);
        // 180.0 / 50.6 = ~3.56
        assert!(score > 3.0 && score < 4.0, "tensor score: {score}");
    }

    #[test]
    fn test_score_amd_unknown_weighted() {
        let profile = make_test_rx7900xtx(0, 0);
        let score = score_amd_gpu(&profile, WorkloadClass::Unknown);
        assert!(score > 0.0, "unknown score should be positive");
    }

    #[test]
    fn test_compute_amd_workload_scores() {
        let profile = make_test_rx7900xtx(0, 0);
        let scores = compute_amd_workload_scores(&profile);
        assert!(scores.compute > 0.0);
        assert!(scores.memory > 0.0);
        assert!(scores.tensor > 0.0);
        assert!(scores.capacity > 0.0);
        assert!(scores.transfer > 0.0);
        assert!(scores.overall > 0.0);
    }

    #[test]
    fn test_compute_amd_geu() {
        let profile = make_test_rx7900xtx(0, 0);
        let geu = compute_amd_geu(&profile);
        assert!(geu > 0.0, "GEU should be positive");
        // GEU = 0.4*compute + 0.3*bandwidth + 0.3*capacity
        // Should be > 1.0 because RX 7900 XTX is much stronger than RTX 3060
        assert!(geu > 1.0, "GEU should exceed reference: {geu}");
    }

    #[test]
    fn test_geu_mi250_higher_capacity() {
        let rx7900 = make_test_rx7900xtx(0, 0);
        let mi250 = make_test_mi250(1, 1);
        let geu_rx = compute_amd_geu(&rx7900);
        let geu_mi = compute_amd_geu(&mi250);
        // MI250 has 64GB vs 24GB, and higher bandwidth
        assert!(geu_mi > geu_rx, "MI250 should have higher GEU due to capacity/bandwidth");
    }

    // -----------------------------------------------------------------------
    // Migration Path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_same_vendor_same_node() {
        assert_eq!(
            select_migration_path(GpuVendor::Amd, GpuVendor::Amd, true),
            MigrationPath::DirectDtoD
        );
        assert_eq!(
            select_migration_path(GpuVendor::Nvidia, GpuVendor::Nvidia, true),
            MigrationPath::DirectDtoD
        );
    }

    #[test]
    fn test_migration_same_vendor_diff_node() {
        assert_eq!(
            select_migration_path(GpuVendor::Amd, GpuVendor::Amd, false),
            MigrationPath::HostStaged
        );
    }

    #[test]
    fn test_migration_cross_vendor_same_node() {
        assert_eq!(
            select_migration_path(GpuVendor::Amd, GpuVendor::Nvidia, true),
            MigrationPath::HostStaged
        );
        assert_eq!(
            select_migration_path(GpuVendor::Nvidia, GpuVendor::Amd, true),
            MigrationPath::HostStaged
        );
    }

    #[test]
    fn test_migration_cross_vendor_diff_node() {
        assert_eq!(
            select_migration_path(GpuVendor::Amd, GpuVendor::Nvidia, false),
            MigrationPath::HostStaged
        );
    }

    // -----------------------------------------------------------------------
    // GPU Pool Filtering tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_compatible_ptx() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let amd = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1));
        let pool = vec![nvidia, amd];

        let compat = filter_compatible_gpus(&BinaryFormat::Ptx, &pool);
        assert_eq!(compat.len(), 1);
        assert_eq!(compat[0].vendor(), GpuVendor::Nvidia);
    }

    #[test]
    fn test_filter_compatible_amdgpu_isa() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let amd = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1));
        let pool = vec![nvidia, amd];

        let compat =
            filter_compatible_gpus(&BinaryFormat::AmdgpuIsa("gfx1100".to_string()), &pool);
        assert_eq!(compat.len(), 1);
        assert_eq!(compat[0].vendor(), GpuVendor::Amd);
    }

    #[test]
    fn test_filter_compatible_hip_source() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let amd = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1));
        let pool = vec![nvidia, amd];

        let compat = filter_compatible_gpus(&BinaryFormat::HipSource, &pool);
        assert_eq!(compat.len(), 2); // HIP source can compile for either
    }

    #[test]
    fn test_filter_by_vendor() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let amd1 = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1));
        let amd2 = UnifiedGpuProfile::Amd(make_test_mi250(2, 2));
        let pool = vec![nvidia, amd1, amd2];

        let amd_gpus = filter_by_vendor(GpuVendor::Amd, &pool);
        assert_eq!(amd_gpus.len(), 2);

        let nvidia_gpus = filter_by_vendor(GpuVendor::Nvidia, &pool);
        assert_eq!(nvidia_gpus.len(), 1);
    }

    #[test]
    fn test_count_by_vendor() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let amd = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1));
        let pool = vec![nvidia, amd];

        assert_eq!(count_by_vendor(GpuVendor::Nvidia, &pool), 1);
        assert_eq!(count_by_vendor(GpuVendor::Amd, &pool), 1);
    }

    #[test]
    fn test_filter_compatible_wrong_amdgpu_arch() {
        let amd = UnifiedGpuProfile::Amd(make_test_rx7900xtx(0, 0)); // gfx1100
        let pool = vec![amd];
        let compat =
            filter_compatible_gpus(&BinaryFormat::AmdgpuIsa("gfx906".to_string()), &pool);
        assert_eq!(compat.len(), 0);
    }

    // -----------------------------------------------------------------------
    // HipInterceptClient tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_client_creation() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        assert_eq!(client.vendor, GpuVendor::Amd);
        assert!(!client.is_connected());
        assert_eq!(client.session_id(), 0);
    }

    #[test]
    fn test_client_connect_disconnect() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        client.connect().unwrap();
        assert!(client.is_connected());
        client.disconnect();
        assert!(!client.is_connected());
    }

    #[test]
    fn test_client_request_ids() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        let id1 = client.next_request_id();
        let id2 = client.next_request_id();
        let id3 = client.next_request_id();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_client_session_id() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        assert_eq!(client.session_id(), 0);
        client.set_session_id(42);
        assert_eq!(client.session_id(), 42);
    }

    #[test]
    fn test_client_last_error() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        assert_eq!(client.get_last_error(), HipError::Success);

        client.set_last_error(HipError::OutOfMemory);
        assert_eq!(client.peek_last_error(), HipError::OutOfMemory);
        assert_eq!(client.get_and_reset_last_error(), HipError::OutOfMemory);
        assert_eq!(client.get_last_error(), HipError::Success); // Reset
    }

    #[test]
    fn test_client_device_props_cache() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        let profile = make_test_rx7900xtx(0, 0);
        let props = HipDeviceProps::from_profile(&profile);

        assert!(client.get_cached_device_props(0).is_none());

        client.cache_device_props(0, props.clone());
        let cached = client.get_cached_device_props(0).unwrap();
        assert_eq!(cached.name, props.name);

        client.invalidate_device_props(0);
        assert!(client.get_cached_device_props(0).is_none());
    }

    #[test]
    fn test_client_device_props_cache_invalidate_all() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        let p0 = HipDeviceProps::from_profile(&make_test_rx7900xtx(0, 0));
        let p1 = HipDeviceProps::from_profile(&make_test_mi250(1, 1));

        client.cache_device_props(0, p0);
        client.cache_device_props(1, p1);
        assert!(client.get_cached_device_props(0).is_some());
        assert!(client.get_cached_device_props(1).is_some());

        client.invalidate_all_device_props();
        assert!(client.get_cached_device_props(0).is_none());
        assert!(client.get_cached_device_props(1).is_none());
    }

    #[test]
    fn test_client_allocation_tracking() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        assert_eq!(client.allocation_count(), 0);

        client.track_allocation(0x1000, 4096);
        client.track_allocation(0x2000, 8192);
        assert_eq!(client.allocation_count(), 2);
        assert_eq!(client.get_allocation_size(0x1000), Some(4096));
        assert_eq!(client.get_allocation_size(0x2000), Some(8192));
        assert_eq!(client.get_allocation_size(0x3000), None);

        let size = client.untrack_allocation(0x1000);
        assert_eq!(size, Some(4096));
        assert_eq!(client.allocation_count(), 1);
    }

    #[test]
    fn test_client_module_tracking() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        assert_eq!(client.module_count(), 0);

        let entry = HipModuleEntry {
            local_handle: 100,
            remote_handle: 200,
            target_arch: "gfx1100".to_string(),
            functions: HashMap::new(),
        };

        client.register_module(entry.clone());
        assert_eq!(client.module_count(), 1);

        let got = client.get_module(100).unwrap();
        assert_eq!(got.remote_handle, 200);
        assert_eq!(got.target_arch, "gfx1100");

        let removed = client.unregister_module(100).unwrap();
        assert_eq!(removed.remote_handle, 200);
        assert_eq!(client.module_count(), 0);
    }

    #[test]
    fn test_client_disconnect_clears_state() {
        let client = HipInterceptClient::new("localhost:14833".to_string());
        client.connect().unwrap();
        client.track_allocation(0x1000, 4096);
        client.cache_device_props(0, HipDeviceProps::from_profile(&make_test_rx7900xtx(0, 0)));
        let entry = HipModuleEntry {
            local_handle: 100,
            remote_handle: 200,
            target_arch: "gfx1100".to_string(),
            functions: HashMap::new(),
        };
        client.register_module(entry);

        client.disconnect();
        assert!(!client.is_connected());
        assert_eq!(client.allocation_count(), 0);
        assert_eq!(client.module_count(), 0);
        assert!(client.get_cached_device_props(0).is_none());
    }

    // -----------------------------------------------------------------------
    // HipExecutor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_executor_creation() {
        let profile = make_test_rx7900xtx(0, 0);
        let executor = HipExecutor::new(0, profile);
        assert_eq!(executor.device_ordinal, 0);
        assert_eq!(executor.device, 0);
        assert!(executor.modules.is_empty());
        assert!(executor.streams.is_empty());
        assert!(executor.events.is_empty());
        assert!(executor.allocations.is_empty());
    }

    #[test]
    fn test_executor_allocation_tracking() {
        let profile = make_test_rx7900xtx(0, 0);
        let mut executor = HipExecutor::new(0, profile);

        executor.track_allocation(0x1000, 4096, None, false);
        executor.track_allocation(0x2000, 8192, Some(1), true);

        assert_eq!(executor.total_allocated_bytes(), 4096 + 8192);

        let alloc = executor.get_allocation(0x1000).unwrap();
        assert_eq!(alloc.size, 4096);
        assert!(!alloc.is_managed);
        assert!(alloc.pool.is_none());

        let alloc2 = executor.get_allocation(0x2000).unwrap();
        assert!(alloc2.is_managed);
        assert_eq!(alloc2.pool, Some(1));

        let removed = executor.untrack_allocation(0x1000).unwrap();
        assert_eq!(removed.size, 4096);
        assert_eq!(executor.total_allocated_bytes(), 8192);
    }

    #[test]
    fn test_executor_stream_management() {
        let profile = make_test_rx7900xtx(0, 0);
        let mut executor = HipExecutor::new(0, profile);

        executor.register_stream(1, 0xABC);
        executor.register_stream(2, 0xDEF);
        assert_eq!(executor.streams.len(), 2);

        let real = executor.destroy_stream(1).unwrap();
        assert_eq!(real, 0xABC);
        assert_eq!(executor.streams.len(), 1);

        assert!(executor.destroy_stream(99).is_none());
    }

    #[test]
    fn test_executor_event_management() {
        let profile = make_test_rx7900xtx(0, 0);
        let mut executor = HipExecutor::new(0, profile);

        executor.register_event(1, 0x111);
        executor.register_event(2, 0x222);
        assert_eq!(executor.events.len(), 2);

        let real = executor.destroy_event(1).unwrap();
        assert_eq!(real, 0x111);
        assert!(executor.destroy_event(999).is_none());
    }

    #[test]
    fn test_executor_module_management() {
        let profile = make_test_rx7900xtx(0, 0);
        let mut executor = HipExecutor::new(0, profile);

        let state = HipModuleState {
            handle: 0xAAA,
            functions: HashMap::from([("my_kernel".to_string(), 0xBBB)]),
            binary: vec![0x7f, b'E', b'L', b'F'],
            target_arch: "gfx1100".to_string(),
        };

        executor.register_module(0xAAA, state);
        assert_eq!(executor.modules.len(), 1);

        let unloaded = executor.unload_module(0xAAA).unwrap();
        assert_eq!(unloaded.target_arch, "gfx1100");
        assert_eq!(unloaded.functions["my_kernel"], 0xBBB);
        assert!(executor.modules.is_empty());
    }

    // -----------------------------------------------------------------------
    // HIP Function Mapping Table tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_function_table_completeness() {
        let table = build_hip_function_table();
        // Phase 1: ~120, Phase 2 graphs: ~18 = ~138
        assert!(
            table.len() >= 118,
            "Expected at least 118 functions, got {}",
            table.len()
        );
    }

    #[test]
    fn test_function_table_categories() {
        let table = build_hip_function_table();
        let init_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Initialization)
            .count();
        let ctx_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Context)
            .count();
        let mem_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Memory)
            .count();
        let module_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Module)
            .count();
        let stream_event_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::StreamEvent)
            .count();
        let error_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Error)
            .count();
        let graph_count = table
            .iter()
            .filter(|f| f.category == HipFunctionCategory::Graph)
            .count();

        assert_eq!(init_count, 19);
        assert_eq!(ctx_count, 13);
        assert_eq!(mem_count, 33);
        assert_eq!(module_count, 15);
        assert_eq!(stream_event_count, 16);
        assert_eq!(error_count, 4);
        assert_eq!(graph_count, 18);
    }

    #[test]
    fn test_function_table_no_duplicates() {
        let table = build_hip_function_table();
        let mut names = std::collections::HashSet::new();
        for entry in &table {
            assert!(
                names.insert(entry.hip_name),
                "Duplicate function: {}",
                entry.hip_name
            );
        }
    }

    #[test]
    fn test_lookup_hip_function_found() {
        let mapping = lookup_hip_function("hipMalloc").unwrap();
        assert_eq!(mapping.cuda_equivalent, "cuMemAlloc_v2");
        assert_eq!(mapping.category, HipFunctionCategory::Memory);
    }

    #[test]
    fn test_lookup_hip_function_not_found() {
        assert!(lookup_hip_function("hipSomeUnknownFunction").is_none());
    }

    #[test]
    fn test_lookup_critical_functions() {
        // These are the most critical functions that MUST be in the table
        let critical = [
            "hipInit",
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipModuleLaunchKernel",
            "hipStreamCreate",
            "hipStreamSynchronize",
            "hipEventCreate",
            "hipGetProcAddress",
            "hipGetDeviceCount",
            "hipSetDevice",
        ];
        for name in &critical {
            assert!(
                lookup_hip_function(name).is_some(),
                "Missing critical function: {name}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // HipTranslator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_translator_creation() {
        let translator = HipTranslator::new();
        assert_eq!(translator.api_vendor, ApiVendor::Hip);
    }

    #[test]
    fn test_translator_default() {
        let translator = HipTranslator::default();
        assert_eq!(translator.api_vendor, ApiVendor::Hip);
    }

    #[test]
    fn test_translator_memcpy_direction() {
        let translator = HipTranslator::new();
        assert_eq!(
            translator.translate_memcpy_direction(HipMemcpyKind::HostToDevice, false, true),
            MemcpyDirection::HostToDevice
        );
        assert_eq!(
            translator.translate_memcpy_direction(HipMemcpyKind::Default, true, false),
            MemcpyDirection::DeviceToHost
        );
    }

    #[test]
    fn test_translator_error() {
        let translator = HipTranslator::new();
        assert_eq!(translator.translate_error(0), HipError::Success);
        assert_eq!(translator.translate_error(2), HipError::OutOfMemory);
    }

    #[test]
    fn test_translator_should_intercept() {
        let translator = HipTranslator::new();
        assert!(translator.should_intercept("hipMalloc"));
        assert!(translator.should_intercept("hipGetProcAddress"));
        assert!(!translator.should_intercept("hipSomethingThatDoesNotExist"));
    }

    #[test]
    fn test_translator_device_props() {
        let translator = HipTranslator::new();
        let profile = make_test_rx7900xtx(0, 0);
        let props = translator.translate_device_props(&profile);
        assert_eq!(props.name, profile.name);
        assert_eq!(props.total_global_mem, profile.vram_total_bytes);
    }

    // -----------------------------------------------------------------------
    // AmdArchFamily display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_arch_family_display() {
        assert_eq!(format!("{}", AmdArchFamily::Gcn), "GCN");
        assert_eq!(format!("{}", AmdArchFamily::Cdna), "CDNA");
        assert_eq!(format!("{}", AmdArchFamily::Rdna), "RDNA");
    }

    // -----------------------------------------------------------------------
    // AmdGpuArch min CU count
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_cu_count() {
        assert_eq!(AmdGpuArch::min_cu_count(), 16);
    }

    // -----------------------------------------------------------------------
    // hex_digit_value helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hex_digit_value_decimal() {
        assert_eq!(hex_digit_value('0'), Some(0));
        assert_eq!(hex_digit_value('9'), Some(9));
    }

    #[test]
    fn test_hex_digit_value_hex() {
        assert_eq!(hex_digit_value('a'), Some(10));
        assert_eq!(hex_digit_value('f'), Some(15));
        assert_eq!(hex_digit_value('A'), Some(10));
    }

    #[test]
    fn test_hex_digit_value_invalid() {
        assert_eq!(hex_digit_value('z'), None);
        assert_eq!(hex_digit_value('g'), None);
    }

    // -----------------------------------------------------------------------
    // HipModuleEntry + HipFunctionEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_entry_with_functions() {
        let func = HipFunctionEntry {
            local_handle: 10,
            remote_handle: 20,
            kernel_name: "vector_add".to_string(),
            module_handle: 1,
            attributes: HashMap::from([(0, 256), (1, 32)]),
        };
        let mut functions = HashMap::new();
        functions.insert("vector_add".to_string(), func.clone());

        let module = HipModuleEntry {
            local_handle: 1,
            remote_handle: 2,
            target_arch: "gfx1100".to_string(),
            functions,
        };

        assert_eq!(module.functions.len(), 1);
        let f = &module.functions["vector_add"];
        assert_eq!(f.kernel_name, "vector_add");
        assert_eq!(f.attributes[&0], 256);
    }

    // -----------------------------------------------------------------------
    // AllocationInfo tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocation_info_basic() {
        let info = AllocationInfo {
            device_ptr: 0x1000,
            size: 65536,
            pool: None,
            is_managed: false,
        };
        assert_eq!(info.device_ptr, 0x1000);
        assert_eq!(info.size, 65536);
        assert!(!info.is_managed);
    }

    #[test]
    fn test_allocation_info_managed_pool() {
        let info = AllocationInfo {
            device_ptr: 0x2000,
            size: 1024,
            pool: Some(42),
            is_managed: true,
        };
        assert_eq!(info.pool, Some(42));
        assert!(info.is_managed);
    }

    // -----------------------------------------------------------------------
    // HipModuleState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_state() {
        let state = HipModuleState {
            handle: 0xABC,
            functions: HashMap::from([("kern".to_string(), 0xDEF)]),
            binary: vec![1, 2, 3, 4],
            target_arch: "gfx90a".to_string(),
        };
        assert_eq!(state.handle, 0xABC);
        assert_eq!(state.functions["kern"], 0xDEF);
        assert_eq!(state.binary.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Mixed pool scenario tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_pool_cuda_app_sees_only_nvidia() {
        let nvidia1 = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let nvidia2 = UnifiedGpuProfile::Nvidia(make_test_rtx3090(1, 0));
        let amd1 = UnifiedGpuProfile::Amd(make_test_rx7900xtx(2, 1));
        let amd2 = UnifiedGpuProfile::Amd(make_test_mi250(3, 2));
        let pool = vec![nvidia1, nvidia2, amd1, amd2];

        // CUDA app: only NVIDIA GPUs visible
        let cuda_gpus = filter_compatible_gpus(&BinaryFormat::Cubin, &pool);
        assert_eq!(cuda_gpus.len(), 2);
        for gpu in &cuda_gpus {
            assert_eq!(gpu.vendor(), GpuVendor::Nvidia);
        }
    }

    #[test]
    fn test_mixed_pool_hip_app_sees_only_compatible_amd() {
        let nvidia = UnifiedGpuProfile::Nvidia(make_test_rtx3090(0, 0));
        let rx7900 = UnifiedGpuProfile::Amd(make_test_rx7900xtx(1, 1)); // gfx1100
        let mi250 = UnifiedGpuProfile::Amd(make_test_mi250(2, 2)); // gfx90a
        let pool = vec![nvidia, rx7900, mi250];

        // HIP app with gfx1100 binary: only RX 7900 XTX
        let gfx1100_gpus =
            filter_compatible_gpus(&BinaryFormat::AmdgpuIsa("gfx1100".to_string()), &pool);
        assert_eq!(gfx1100_gpus.len(), 1);
        assert_eq!(gfx1100_gpus[0].gpu_id(), 1);

        // HIP app with gfx90a binary: only MI250
        let gfx90a_gpus =
            filter_compatible_gpus(&BinaryFormat::AmdgpuIsa("gfx90a".to_string()), &pool);
        assert_eq!(gfx90a_gpus.len(), 1);
        assert_eq!(gfx90a_gpus[0].gpu_id(), 2);
    }

    #[test]
    fn test_mixed_pool_geu_cross_vendor_comparable() {
        let rx7900 = make_test_rx7900xtx(0, 0);
        let rtx3090 = make_test_rtx3090(1, 1);

        let geu_amd = compute_amd_geu(&rx7900);
        // Compute NVIDIA GEU with the same formula for comparison
        let ref_values = ReferenceValues::default();
        let geu_nvidia = {
            let compute = rtx3090.measured_fp32_gflops / (ref_values.fp32_tflops * 1000.0);
            let bandwidth = rtx3090.measured_memory_bw_gbps / ref_values.memory_bw_gbps;
            let capacity =
                rtx3090.vram_total_bytes as f64 / (ref_values.vram_gb * 1024.0 * 1024.0 * 1024.0);
            compute * 0.4 + bandwidth * 0.3 + capacity * 0.3
        };

        // Both should be > 1.0 (stronger than reference RTX 3060)
        assert!(geu_amd > 1.0, "AMD GEU: {geu_amd}");
        assert!(geu_nvidia > 1.0, "NVIDIA GEU: {geu_nvidia}");
        // RX 7900 XTX should have higher compute score
        assert!(geu_amd > geu_nvidia, "AMD GEU should be higher: {geu_amd} vs {geu_nvidia}");
    }
}
