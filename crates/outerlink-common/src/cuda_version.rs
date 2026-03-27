//! CUDA version compatibility types for cross-node negotiation.
//!
//! When OuterLink connects a client application to a remote GPU server,
//! the two sides may run different CUDA toolkit and driver versions.
//! This module provides types for encoding, comparing, and negotiating
//! CUDA versions so that the interception layer can select the correct
//! function variants and reject incompatible connections early.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// CudaVersion
// ---------------------------------------------------------------------------

/// CUDA toolkit version (major.minor).
///
/// NVIDIA encodes this as `major * 1000 + minor * 10` in `cuDriverGetVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaVersion {
    pub major: u32,
    pub minor: u32,
}

impl CudaVersion {
    /// Create a new CudaVersion.
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Decode from NVIDIA's `cuDriverGetVersion` encoding
    /// where `v = major * 1000 + minor * 10`.
    pub fn from_encoded(v: u32) -> Self {
        Self {
            major: v / 1000,
            minor: (v % 1000) / 10,
        }
    }

    /// Encode to NVIDIA's integer representation.
    pub fn to_encoded(self) -> u32 {
        self.major * 1000 + self.minor * 10
    }

    /// Check forward-compatibility: same major version and self.minor <= other.minor.
    ///
    /// CUDA's compatibility contract guarantees that a binary built against
    /// toolkit X.Y runs on any driver X.Z where Z >= Y (same major).
    pub fn is_compatible_with(&self, other: &CudaVersion) -> bool {
        self.major == other.major && self.minor <= other.minor
    }
}

impl fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl PartialOrd for CudaVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CudaVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_encoded().cmp(&other.to_encoded())
    }
}

// ---------------------------------------------------------------------------
// DriverVersion
// ---------------------------------------------------------------------------

/// NVIDIA GPU driver version (e.g. "535.129.03").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DriverVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// Original patch string preserving zero-padding (e.g. "03").
    patch_str: String,
}

impl DriverVersion {
    /// Create a new DriverVersion.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch, patch_str: format!("{:02}", patch) }
    }

    /// Parse from the string format NVIDIA uses, e.g. "535.129.03".
    /// Preserves the original patch formatting for round-trip fidelity.
    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        let patch = parts[2].parse::<u32>().ok()?;
        Some(Self { major, minor, patch, patch_str: parts[2].to_string() })
    }
}

impl fmt::Display for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch_str)
    }
}

// ---------------------------------------------------------------------------
// VersionNegotiation
// ---------------------------------------------------------------------------

/// Holds both sides' versions and provides negotiation logic.
#[derive(Debug, Clone)]
pub struct VersionNegotiation {
    pub client_cuda: CudaVersion,
    pub server_cuda: CudaVersion,
    pub client_driver: DriverVersion,
    pub server_driver: DriverVersion,
}

impl VersionNegotiation {
    /// The negotiated CUDA version is the minimum of the two sides.
    /// Returns `None` if the major versions differ (incompatible).
    pub fn negotiated_version(&self) -> Option<CudaVersion> {
        if self.client_cuda.major != self.server_cuda.major {
            return None;
        }
        Some(min_cuda_version(&self.client_cuda, &self.server_cuda))
    }

    /// Whether the client and server are compatible (same major CUDA version).
    pub fn is_compatible(&self) -> bool {
        self.negotiated_version().is_some()
    }

    /// Human-readable compatibility report.
    pub fn compatibility_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Client: CUDA {} (driver {})\n",
            self.client_cuda, self.client_driver
        ));
        report.push_str(&format!(
            "Server: CUDA {} (driver {})\n",
            self.server_cuda, self.server_driver
        ));
        match self.negotiated_version() {
            Some(v) => {
                report.push_str(&format!("Negotiated CUDA version: {v}\n"));
                report.push_str("Status: Compatible");
            }
            None => {
                report.push_str(&format!(
                    "Incompatible: client major {} != server major {}\n",
                    self.client_cuda.major, self.server_cuda.major
                ));
                report.push_str("Status: Incompatible");
            }
        }
        report
    }
}

// ---------------------------------------------------------------------------
// FunctionVersion
// ---------------------------------------------------------------------------

/// A versioned CUDA function name (e.g. `cuMemAlloc_v2`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionVersion {
    pub base_name: String,
    pub version_suffix: Option<String>,
}

impl FunctionVersion {
    /// Construct from parts.
    pub fn new(base_name: impl Into<String>, version_suffix: Option<String>) -> Self {
        Self {
            base_name: base_name.into(),
            version_suffix,
        }
    }

    /// The full CUDA symbol name, e.g. `cuMemAlloc_v2`.
    pub fn full_name(&self) -> String {
        match &self.version_suffix {
            Some(suffix) => format!("{}_{}", self.base_name, suffix),
            None => self.base_name.clone(),
        }
    }

    /// Parse a full symbol name, splitting on the last `_vN` suffix.
    ///
    /// - `"cuMemAlloc_v2"` -> base=`"cuMemAlloc"`, suffix=`Some("v2")`
    /// - `"cuInit"` -> base=`"cuInit"`, suffix=`None`
    /// - `"cuMemAllocManaged_v3"` -> base=`"cuMemAllocManaged"`, suffix=`Some("v3")`
    pub fn from_full_name(s: &str) -> Self {
        // Find the last occurrence of "_v" followed by digits
        if let Some(pos) = s.rfind("_v") {
            let suffix_part = &s[pos + 1..]; // skip the '_'
            // Verify it matches "vN" pattern (v followed by digits)
            if suffix_part.len() > 1 && suffix_part[1..].chars().all(|c| c.is_ascii_digit()) {
                return Self {
                    base_name: s[..pos].to_string(),
                    version_suffix: Some(suffix_part.to_string()),
                };
            }
        }
        Self {
            base_name: s.to_string(),
            version_suffix: None,
        }
    }

    /// Extract the numeric version, or 1 for unversioned (the original API).
    pub fn version_number(&self) -> u32 {
        match &self.version_suffix {
            Some(s) => s[1..].parse::<u32>().unwrap_or(1),
            None => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// VersionedFunctionTable
// ---------------------------------------------------------------------------

/// Registry of versioned CUDA functions.
///
/// Maps base function names to their known versioned variants, allowing
/// the interception layer to pick the right variant for a given CUDA version.
#[derive(Debug, Clone)]
pub struct VersionedFunctionTable {
    functions: HashMap<String, Vec<FunctionVersion>>,
}

impl VersionedFunctionTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a function variant.
    pub fn register(&mut self, func: FunctionVersion) {
        self.functions
            .entry(func.base_name.clone())
            .or_default()
            .push(func);
    }

    /// Look up the highest-version variant whose version number does not
    /// exceed the given CUDA version's minor number.
    ///
    /// Heuristic: CUDA function version N was typically introduced in
    /// Look up the best function variant for a given CUDA version.
    ///
    /// CUDA guarantees backward compatibility: newer drivers always support
    /// older function ABI versions. Since function version suffixes (_v2, _v3)
    /// are ABI revisions — NOT CUDA minor version numbers — the correct strategy
    /// is to return the highest available variant. cuGetProcAddress on the server
    /// side will resolve to the actual supported version at runtime.
    ///
    /// The `cuda_version` parameter is reserved for future use when we have a
    /// per-function "introduced at" table. Currently unused.
    pub fn lookup(&self, name: &str, _cuda_version: &CudaVersion) -> Option<&FunctionVersion> {
        let variants = self.functions.get(name)?;
        if variants.is_empty() {
            return None;
        }
        // Return the highest ABI version available.
        variants.iter().max_by_key(|v| v.version_number())
    }

    /// Total number of registered variants (across all base names).
    pub fn count(&self) -> usize {
        self.functions.values().map(|v| v.len()).sum()
    }
}

impl Default for VersionedFunctionTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GpuFeature / CudaCapability
// ---------------------------------------------------------------------------

/// GPU features gated by compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFeature {
    /// Tensor Cores (sm_70+, Volta and later)
    TensorCores,
    /// Hardware ray tracing (sm_75+, Turing and later)
    RayTracing,
    /// NVLink support (sm_70+, various SKUs)
    NvLink,
    /// Multi-Instance GPU (sm_80+, Ampere A100 and later)
    Mig,
    /// Full-rate FP64 (sm_60+, GP100 / data-center parts)
    FP64Full,
    /// BF16 support (sm_80+, Ampere and later)
    BF16,
    /// FP8 support (sm_89+, Ada Lovelace and later)
    FP8,
}

/// GPU compute capability (e.g. 8.6 for RTX 3090).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaCapability {
    pub compute_major: u32,
    pub compute_minor: u32,
}

impl CudaCapability {
    /// Create a new compute capability.
    pub fn new(compute_major: u32, compute_minor: u32) -> Self {
        Self {
            compute_major,
            compute_minor,
        }
    }

    /// Shorthand SM number, e.g. 86 for 8.6.
    pub fn sm(&self) -> u32 {
        self.compute_major * 10 + self.compute_minor
    }

    /// Whether this GPU supports a given feature.
    pub fn supports_feature(&self, feature: GpuFeature) -> bool {
        let sm = self.sm();
        match feature {
            GpuFeature::TensorCores => sm >= 70,
            GpuFeature::RayTracing => sm >= 75,
            GpuFeature::NvLink => sm >= 70,
            GpuFeature::Mig => sm >= 80,
            GpuFeature::FP64Full => sm >= 60,
            GpuFeature::BF16 => sm >= 80,
            GpuFeature::FP8 => sm >= 89,
        }
    }
}

impl fmt::Display for CudaCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.compute_major, self.compute_minor)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Return the lower of two CUDA versions (for negotiation to the common
/// denominator).
pub fn min_cuda_version(client: &CudaVersion, server: &CudaVersion) -> CudaVersion {
    if client <= server {
        *client
    } else {
        *server
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- CudaVersion encoding/decoding ---

    #[test]
    fn cuda_version_encode_12_4() {
        let v = CudaVersion::new(12, 4);
        assert_eq!(v.to_encoded(), 12040);
    }

    #[test]
    fn cuda_version_decode_12040() {
        let v = CudaVersion::from_encoded(12040);
        assert_eq!(v.major, 12);
        assert_eq!(v.minor, 4);
    }

    #[test]
    fn cuda_version_roundtrip() {
        for encoded in [11070, 11080, 12000, 12020, 12040, 12060] {
            let v = CudaVersion::from_encoded(encoded);
            assert_eq!(v.to_encoded(), encoded, "roundtrip failed for {encoded}");
        }
    }

    #[test]
    fn cuda_version_display() {
        assert_eq!(CudaVersion::new(12, 4).to_string(), "12.4");
        assert_eq!(CudaVersion::new(11, 7).to_string(), "11.7");
    }

    // --- CudaVersion compatibility ---

    #[test]
    fn compatible_same_major_lower_minor() {
        let client = CudaVersion::new(12, 2);
        let server = CudaVersion::new(12, 4);
        assert!(client.is_compatible_with(&server));
    }

    #[test]
    fn compatible_same_version() {
        let v = CudaVersion::new(12, 4);
        assert!(v.is_compatible_with(&v));
    }

    #[test]
    fn incompatible_higher_minor() {
        let client = CudaVersion::new(12, 6);
        let server = CudaVersion::new(12, 4);
        assert!(!client.is_compatible_with(&server));
    }

    #[test]
    fn incompatible_different_major() {
        let client = CudaVersion::new(11, 8);
        let server = CudaVersion::new(12, 4);
        assert!(!client.is_compatible_with(&server));
    }

    // --- CudaVersion ordering ---

    #[test]
    fn cuda_version_ordering() {
        let v11_7 = CudaVersion::new(11, 7);
        let v12_0 = CudaVersion::new(12, 0);
        let v12_4 = CudaVersion::new(12, 4);
        assert!(v11_7 < v12_0);
        assert!(v12_0 < v12_4);
    }

    // --- DriverVersion ---

    #[test]
    fn driver_version_parse_valid() {
        let dv = DriverVersion::from_string("535.129.03").unwrap();
        assert_eq!(dv.major, 535);
        assert_eq!(dv.minor, 129);
        assert_eq!(dv.patch, 3);
    }

    #[test]
    fn driver_version_parse_invalid_two_parts() {
        assert!(DriverVersion::from_string("535.129").is_none());
    }

    #[test]
    fn driver_version_parse_invalid_non_numeric() {
        assert!(DriverVersion::from_string("abc.129.03").is_none());
    }

    #[test]
    fn driver_version_display() {
        let dv = DriverVersion::new(535, 129, 3);
        assert_eq!(dv.to_string(), "535.129.03");
    }

    // --- VersionNegotiation ---

    #[test]
    fn negotiation_compatible_picks_min() {
        let neg = VersionNegotiation {
            client_cuda: CudaVersion::new(12, 2),
            server_cuda: CudaVersion::new(12, 4),
            client_driver: DriverVersion::new(535, 129, 3),
            server_driver: DriverVersion::new(550, 54, 14),
        };
        assert!(neg.is_compatible());
        assert_eq!(neg.negotiated_version(), Some(CudaVersion::new(12, 2)));
    }

    #[test]
    fn negotiation_incompatible_different_major() {
        let neg = VersionNegotiation {
            client_cuda: CudaVersion::new(11, 8),
            server_cuda: CudaVersion::new(12, 4),
            client_driver: DriverVersion::new(520, 61, 5),
            server_driver: DriverVersion::new(550, 54, 14),
        };
        assert!(!neg.is_compatible());
        assert_eq!(neg.negotiated_version(), None);
    }

    #[test]
    fn negotiation_report_contains_status() {
        let neg = VersionNegotiation {
            client_cuda: CudaVersion::new(12, 2),
            server_cuda: CudaVersion::new(12, 4),
            client_driver: DriverVersion::new(535, 129, 3),
            server_driver: DriverVersion::new(550, 54, 14),
        };
        let report = neg.compatibility_report();
        assert!(report.contains("Compatible"));
        assert!(report.contains("12.2"));
    }

    #[test]
    fn negotiation_report_incompatible() {
        let neg = VersionNegotiation {
            client_cuda: CudaVersion::new(11, 8),
            server_cuda: CudaVersion::new(12, 4),
            client_driver: DriverVersion::new(520, 61, 5),
            server_driver: DriverVersion::new(550, 54, 14),
        };
        let report = neg.compatibility_report();
        assert!(report.contains("Incompatible"));
    }

    // --- FunctionVersion ---

    #[test]
    fn function_version_full_name_with_suffix() {
        let fv = FunctionVersion::new("cuMemAlloc", Some("v2".into()));
        assert_eq!(fv.full_name(), "cuMemAlloc_v2");
    }

    #[test]
    fn function_version_full_name_no_suffix() {
        let fv = FunctionVersion::new("cuInit", None);
        assert_eq!(fv.full_name(), "cuInit");
    }

    #[test]
    fn function_version_from_full_name_versioned() {
        let fv = FunctionVersion::from_full_name("cuMemAlloc_v2");
        assert_eq!(fv.base_name, "cuMemAlloc");
        assert_eq!(fv.version_suffix, Some("v2".into()));
    }

    #[test]
    fn function_version_from_full_name_unversioned() {
        let fv = FunctionVersion::from_full_name("cuInit");
        assert_eq!(fv.base_name, "cuInit");
        assert_eq!(fv.version_suffix, None);
    }

    #[test]
    fn function_version_from_full_name_v3() {
        let fv = FunctionVersion::from_full_name("cuMemAllocManaged_v3");
        assert_eq!(fv.base_name, "cuMemAllocManaged");
        assert_eq!(fv.version_suffix, Some("v3".into()));
    }

    #[test]
    fn function_version_number() {
        assert_eq!(FunctionVersion::from_full_name("cuMemAlloc_v2").version_number(), 2);
        assert_eq!(FunctionVersion::from_full_name("cuInit").version_number(), 1);
    }

    // --- VersionedFunctionTable ---

    #[test]
    fn function_table_register_and_count() {
        let mut table = VersionedFunctionTable::new();
        table.register(FunctionVersion::from_full_name("cuMemAlloc"));
        table.register(FunctionVersion::from_full_name("cuMemAlloc_v2"));
        table.register(FunctionVersion::from_full_name("cuInit"));
        assert_eq!(table.count(), 3);
    }

    #[test]
    fn function_table_lookup_picks_highest_version() {
        let mut table = VersionedFunctionTable::new();
        table.register(FunctionVersion::from_full_name("cuMemAlloc"));
        table.register(FunctionVersion::from_full_name("cuMemAlloc_v2"));

        // Always picks the highest ABI version (CUDA backward compat guarantee)
        let result = table.lookup("cuMemAlloc", &CudaVersion::new(12, 4)).unwrap();
        assert_eq!(result.version_number(), 2);
    }

    #[test]
    fn function_table_lookup_highest_even_for_cuda_12_0() {
        let mut table = VersionedFunctionTable::new();
        table.register(FunctionVersion::from_full_name("cuMemAlloc"));
        table.register(FunctionVersion::from_full_name("cuMemAlloc_v2"));

        // CUDA 12.0 still gets v2 — ABI versions != CUDA minor versions
        let result = table.lookup("cuMemAlloc", &CudaVersion::new(12, 0)).unwrap();
        assert_eq!(result.version_number(), 2);
    }

    #[test]
    fn function_table_lookup_missing_returns_none() {
        let table = VersionedFunctionTable::new();
        assert!(table.lookup("cuNonExistent", &CudaVersion::new(12, 4)).is_none());
    }

    // --- min_cuda_version ---

    #[test]
    fn min_version_picks_lower() {
        let a = CudaVersion::new(12, 2);
        let b = CudaVersion::new(12, 4);
        assert_eq!(min_cuda_version(&a, &b), a);
        assert_eq!(min_cuda_version(&b, &a), a);
    }

    #[test]
    fn min_version_equal() {
        let v = CudaVersion::new(12, 4);
        assert_eq!(min_cuda_version(&v, &v), v);
    }

    // --- CudaCapability ---

    #[test]
    fn capability_sm_number() {
        let cap = CudaCapability::new(8, 6);
        assert_eq!(cap.sm(), 86);
    }

    #[test]
    fn capability_display() {
        assert_eq!(CudaCapability::new(8, 6).to_string(), "8.6");
    }

    #[test]
    fn rtx3090_supports_tensor_cores() {
        let rtx3090 = CudaCapability::new(8, 6);
        assert!(rtx3090.supports_feature(GpuFeature::TensorCores));
    }

    #[test]
    fn rtx3090_supports_bf16() {
        let rtx3090 = CudaCapability::new(8, 6);
        assert!(rtx3090.supports_feature(GpuFeature::BF16));
    }

    #[test]
    fn rtx3090_no_fp8() {
        let rtx3090 = CudaCapability::new(8, 6);
        assert!(!rtx3090.supports_feature(GpuFeature::FP8));
    }

    #[test]
    fn ada_lovelace_supports_fp8() {
        let rtx4090 = CudaCapability::new(8, 9);
        assert!(rtx4090.supports_feature(GpuFeature::FP8));
    }

    #[test]
    fn pascal_no_tensor_cores() {
        let p100 = CudaCapability::new(6, 0);
        assert!(!p100.supports_feature(GpuFeature::TensorCores));
        assert!(p100.supports_feature(GpuFeature::FP64Full));
    }

    #[test]
    fn turing_supports_ray_tracing() {
        let rtx2080 = CudaCapability::new(7, 5);
        assert!(rtx2080.supports_feature(GpuFeature::RayTracing));
    }

    #[test]
    fn volta_no_ray_tracing() {
        let v100 = CudaCapability::new(7, 0);
        assert!(!v100.supports_feature(GpuFeature::RayTracing));
    }

    // --- GpuFeature enum variants ---

    #[test]
    fn gpu_feature_all_variants_exist() {
        // Ensure all variants are constructible (compile-time check mostly)
        let features = [
            GpuFeature::TensorCores,
            GpuFeature::RayTracing,
            GpuFeature::NvLink,
            GpuFeature::Mig,
            GpuFeature::FP64Full,
            GpuFeature::BF16,
            GpuFeature::FP8,
        ];
        assert_eq!(features.len(), 7);
    }
}
