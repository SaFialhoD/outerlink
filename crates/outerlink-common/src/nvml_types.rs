//! NVML type definitions for GPU monitoring virtualization.
//!
//! These types mirror the NVIDIA Management Library (NVML) C API structures
//! used for GPU health monitoring, temperature, power, clock, and utilization
//! queries. They are used by the NVML interception layer to serve virtual
//! GPU information from remote OuterLink servers.

/// NVML return codes matching `nvmlReturn_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NvmlReturn {
    Success = 0,
    Uninitialized = 1,
    InvalidArgument = 2,
    NotSupported = 3,
    NoPermission = 4,
    AlreadyInitialized = 5,
    NotFound = 6,
    InsufficientSize = 7,
    InsufficientPower = 8,
    DriverNotLoaded = 9,
    Timeout = 10,
    IrqIssue = 11,
    LibraryNotFound = 12,
    FunctionNotFound = 13,
    CorruptedInforom = 14,
    GpuIsLost = 15,
    ResetRequired = 16,
    OperatingSystem = 17,
    LibRmVersionMismatch = 18,
    Unknown = 999,
}

impl NvmlReturn {
    /// Convert from raw u32 value.
    pub fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Success,
            1 => Self::Uninitialized,
            2 => Self::InvalidArgument,
            3 => Self::NotSupported,
            4 => Self::NoPermission,
            5 => Self::AlreadyInitialized,
            6 => Self::NotFound,
            7 => Self::InsufficientSize,
            8 => Self::InsufficientPower,
            9 => Self::DriverNotLoaded,
            10 => Self::Timeout,
            11 => Self::IrqIssue,
            12 => Self::LibraryNotFound,
            13 => Self::FunctionNotFound,
            14 => Self::CorruptedInforom,
            15 => Self::GpuIsLost,
            16 => Self::ResetRequired,
            17 => Self::OperatingSystem,
            18 => Self::LibRmVersionMismatch,
            999 => Self::Unknown,
            _ => Self::Unknown,
        }
    }

    /// Convert to raw u32 value.
    pub fn as_raw(self) -> u32 {
        self as u32
    }

    /// Check if this result indicates success.
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// NVML temperature sensor types matching `nvmlTemperatureSensors_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum NvmlTemperatureSensors {
    Gpu = 0,
}

/// NVML clock types matching `nvmlClockType_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum NvmlClockType {
    Graphics = 0,
    Sm = 1,
    Mem = 2,
    Video = 3,
}

impl NvmlClockType {
    pub fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Graphics),
            1 => Some(Self::Sm),
            2 => Some(Self::Mem),
            3 => Some(Self::Video),
            _ => None,
        }
    }
}

/// Snapshot of a single GPU's NVML-queryable state.
///
/// This is the payload transferred from server to client in an
/// `NvmlSnapshotResponse` message. All string fields use fixed-size
/// arrays to allow simple serialization without length prefixes.
#[derive(Debug, Clone)]
pub struct NvmlGpuSnapshot {
    /// Device name, e.g. "NVIDIA GeForce RTX 3090" (up to 96 bytes, nul-padded).
    pub name: [u8; 96],
    /// Device UUID string, e.g. "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (up to 96 bytes).
    pub uuid: [u8; 96],
    /// PCI bus ID string, e.g. "00000000:01:00.0" (up to 32 bytes).
    pub pci_bus_id: [u8; 32],
    /// PCI device ID.
    pub pci_device_id: u32,
    /// PCI subsystem ID.
    pub pci_subsystem_id: u32,
    /// Total VRAM in bytes.
    pub vram_total: u64,
    /// Used VRAM in bytes.
    pub vram_used: u64,
    /// Free VRAM in bytes.
    pub vram_free: u64,
    /// GPU temperature in degrees Celsius.
    pub temperature_gpu: u32,
    /// Power usage in milliwatts.
    pub power_usage_mw: u32,
    /// Power management limit in milliwatts.
    pub power_limit_mw: u32,
    /// GPU utilization percentage (0-100).
    pub utilization_gpu: u32,
    /// Memory utilization percentage (0-100).
    pub utilization_memory: u32,
    /// Graphics clock in MHz.
    pub clock_graphics_mhz: u32,
    /// SM clock in MHz.
    pub clock_sm_mhz: u32,
    /// Memory clock in MHz.
    pub clock_mem_mhz: u32,
    /// Video clock in MHz.
    pub clock_video_mhz: u32,
    /// Compute capability major version.
    pub compute_cap_major: u32,
    /// Compute capability minor version.
    pub compute_cap_minor: u32,
    /// Driver version string (up to 32 bytes, nul-padded).
    pub driver_version: [u8; 32],
    /// NVML version string (up to 32 bytes, nul-padded).
    pub nvml_version: [u8; 32],
    /// CUDA driver version as integer (e.g. 12040 for 12.4).
    pub cuda_driver_version: i32,
    /// Fan speed percentage (0-100).
    pub fan_speed_pct: u32,
    /// Number of GPU cores (CUDA cores).
    pub num_cores: u32,
    /// GPU architecture enum value (from nvmlDeviceArchitecture_t).
    pub architecture: u32,
}

/// Serialized size of `NvmlGpuSnapshot` in bytes.
///
/// Layout (all little-endian):
///   name:               96 bytes
///   uuid:               96 bytes
///   pci_bus_id:         32 bytes
///   pci_device_id:       4 bytes
///   pci_subsystem_id:    4 bytes
///   vram_total:          8 bytes
///   vram_used:           8 bytes
///   vram_free:           8 bytes
///   temperature_gpu:     4 bytes
///   power_usage_mw:      4 bytes
///   power_limit_mw:      4 bytes
///   utilization_gpu:     4 bytes
///   utilization_memory:  4 bytes
///   clock_graphics_mhz:  4 bytes
///   clock_sm_mhz:        4 bytes
///   clock_mem_mhz:       4 bytes
///   clock_video_mhz:     4 bytes
///   compute_cap_major:   4 bytes
///   compute_cap_minor:   4 bytes
///   driver_version:     32 bytes
///   nvml_version:       32 bytes
///   cuda_driver_version: 4 bytes
///   fan_speed_pct:       4 bytes
///   num_cores:           4 bytes
///   architecture:        4 bytes
///   -----------------------------------------
///   Total:             380 bytes
pub const NVML_SNAPSHOT_SIZE: usize = 380;

impl NvmlGpuSnapshot {
    /// Serialize to a byte buffer (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(NVML_SNAPSHOT_SIZE);
        buf.extend_from_slice(&self.name);
        buf.extend_from_slice(&self.uuid);
        buf.extend_from_slice(&self.pci_bus_id);
        buf.extend_from_slice(&self.pci_device_id.to_le_bytes());
        buf.extend_from_slice(&self.pci_subsystem_id.to_le_bytes());
        buf.extend_from_slice(&self.vram_total.to_le_bytes());
        buf.extend_from_slice(&self.vram_used.to_le_bytes());
        buf.extend_from_slice(&self.vram_free.to_le_bytes());
        buf.extend_from_slice(&self.temperature_gpu.to_le_bytes());
        buf.extend_from_slice(&self.power_usage_mw.to_le_bytes());
        buf.extend_from_slice(&self.power_limit_mw.to_le_bytes());
        buf.extend_from_slice(&self.utilization_gpu.to_le_bytes());
        buf.extend_from_slice(&self.utilization_memory.to_le_bytes());
        buf.extend_from_slice(&self.clock_graphics_mhz.to_le_bytes());
        buf.extend_from_slice(&self.clock_sm_mhz.to_le_bytes());
        buf.extend_from_slice(&self.clock_mem_mhz.to_le_bytes());
        buf.extend_from_slice(&self.clock_video_mhz.to_le_bytes());
        buf.extend_from_slice(&self.compute_cap_major.to_le_bytes());
        buf.extend_from_slice(&self.compute_cap_minor.to_le_bytes());
        buf.extend_from_slice(&self.driver_version);
        buf.extend_from_slice(&self.nvml_version);
        buf.extend_from_slice(&self.cuda_driver_version.to_le_bytes());
        buf.extend_from_slice(&self.fan_speed_pct.to_le_bytes());
        buf.extend_from_slice(&self.num_cores.to_le_bytes());
        buf.extend_from_slice(&self.architecture.to_le_bytes());
        debug_assert_eq!(buf.len(), NVML_SNAPSHOT_SIZE);
        buf
    }

    /// Deserialize from a byte buffer (little-endian).
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < NVML_SNAPSHOT_SIZE {
            return None;
        }
        let mut off = 0usize;

        let mut name = [0u8; 96];
        name.copy_from_slice(&buf[off..off + 96]);
        off += 96;

        let mut uuid = [0u8; 96];
        uuid.copy_from_slice(&buf[off..off + 96]);
        off += 96;

        let mut pci_bus_id = [0u8; 32];
        pci_bus_id.copy_from_slice(&buf[off..off + 32]);
        off += 32;

        let pci_device_id = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let pci_subsystem_id = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let vram_total = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        off += 8;
        let vram_used = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        off += 8;
        let vram_free = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        off += 8;
        let temperature_gpu = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let power_usage_mw = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let power_limit_mw = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let utilization_gpu = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let utilization_memory = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let clock_graphics_mhz = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let clock_sm_mhz = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let clock_mem_mhz = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let clock_video_mhz = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let compute_cap_major = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let compute_cap_minor = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;

        let mut driver_version = [0u8; 32];
        driver_version.copy_from_slice(&buf[off..off + 32]);
        off += 32;

        let mut nvml_version = [0u8; 32];
        nvml_version.copy_from_slice(&buf[off..off + 32]);
        off += 32;

        let cuda_driver_version = i32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let fan_speed_pct = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let num_cores = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let architecture = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        // off += 4; -- last field

        debug_assert_eq!(off + 4, NVML_SNAPSHOT_SIZE);

        Some(Self {
            name,
            uuid,
            pci_bus_id,
            pci_device_id,
            pci_subsystem_id,
            vram_total,
            vram_used,
            vram_free,
            temperature_gpu,
            power_usage_mw,
            power_limit_mw,
            utilization_gpu,
            utilization_memory,
            clock_graphics_mhz,
            clock_sm_mhz,
            clock_mem_mhz,
            clock_video_mhz,
            compute_cap_major,
            compute_cap_minor,
            driver_version,
            nvml_version,
            cuda_driver_version,
            fan_speed_pct,
            num_cores,
            architecture,
        })
    }

    /// Create a default stub snapshot matching an RTX 3090.
    pub fn stub_rtx3090(index: u32) -> Self {
        let mut name = [0u8; 96];
        let name_str = b"NVIDIA GeForce RTX 3090";
        name[..name_str.len()].copy_from_slice(name_str);

        let mut uuid = [0u8; 96];
        let uuid_str = format!("GPU-00000000-0000-0000-0000-{:012x}", index);
        let uuid_bytes = uuid_str.as_bytes();
        uuid[..uuid_bytes.len()].copy_from_slice(uuid_bytes);

        let mut pci_bus_id = [0u8; 32];
        let pci_str = format!("00000000:{:02x}:00.0", index + 1);
        let pci_bytes = pci_str.as_bytes();
        pci_bus_id[..pci_bytes.len()].copy_from_slice(pci_bytes);

        let mut driver_version = [0u8; 32];
        let drv_str = b"535.129.03";
        driver_version[..drv_str.len()].copy_from_slice(drv_str);

        let mut nvml_version = [0u8; 32];
        let nvml_str = b"12.535.129.03";
        nvml_version[..nvml_str.len()].copy_from_slice(nvml_str);

        Self {
            name,
            uuid,
            pci_bus_id,
            pci_device_id: 0x2204,  // RTX 3090
            pci_subsystem_id: 0,
            vram_total: 24_576 * 1024 * 1024,   // 24 GiB
            vram_used: 512 * 1024 * 1024,        // 512 MiB used
            vram_free: 24_064 * 1024 * 1024,     // rest free
            temperature_gpu: 42,
            power_usage_mw: 30_000,              // 30W idle
            power_limit_mw: 350_000,             // 350W limit
            utilization_gpu: 0,
            utilization_memory: 0,
            clock_graphics_mhz: 210,             // idle clock
            clock_sm_mhz: 210,
            clock_mem_mhz: 9501,                 // memory clock
            clock_video_mhz: 1695,
            compute_cap_major: 8,
            compute_cap_minor: 6,
            driver_version,
            nvml_version,
            cuda_driver_version: 12040,
            fan_speed_pct: 30,
            num_cores: 10496,
            architecture: 2,                     // NVML_DEVICE_ARCH_AMPERE
        }
    }

    /// Extract the name as a Rust string (up to first nul byte).
    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..end]).unwrap_or("Unknown")
    }

    /// Extract the UUID as a Rust string (up to first nul byte).
    pub fn uuid_str(&self) -> &str {
        let end = self.uuid.iter().position(|&b| b == 0).unwrap_or(self.uuid.len());
        std::str::from_utf8(&self.uuid[..end]).unwrap_or("")
    }

    /// Extract the PCI bus ID as a Rust string (up to first nul byte).
    pub fn pci_bus_id_str(&self) -> &str {
        let end = self.pci_bus_id.iter().position(|&b| b == 0).unwrap_or(self.pci_bus_id.len());
        std::str::from_utf8(&self.pci_bus_id[..end]).unwrap_or("")
    }

    /// Extract the driver version as a Rust string (up to first nul byte).
    pub fn driver_version_str(&self) -> &str {
        let end = self.driver_version.iter().position(|&b| b == 0).unwrap_or(self.driver_version.len());
        std::str::from_utf8(&self.driver_version[..end]).unwrap_or("")
    }

    /// Extract the NVML version as a Rust string (up to first nul byte).
    pub fn nvml_version_str(&self) -> &str {
        let end = self.nvml_version.iter().position(|&b| b == 0).unwrap_or(self.nvml_version.len());
        std::str::from_utf8(&self.nvml_version[..end]).unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvml_return_roundtrip() {
        for raw in 0..=18 {
            let ret = NvmlReturn::from_raw(raw);
            assert_eq!(ret.as_raw(), raw, "roundtrip failed for {raw}");
        }
        assert_eq!(NvmlReturn::from_raw(999), NvmlReturn::Unknown);
    }

    #[test]
    fn test_nvml_return_unknown_for_unrecognized() {
        assert_eq!(NvmlReturn::from_raw(42), NvmlReturn::Unknown);
        assert_eq!(NvmlReturn::from_raw(500), NvmlReturn::Unknown);
    }

    #[test]
    fn test_nvml_return_success_check() {
        assert!(NvmlReturn::Success.is_success());
        assert!(!NvmlReturn::Uninitialized.is_success());
        assert!(!NvmlReturn::InvalidArgument.is_success());
    }

    #[test]
    fn test_snapshot_serialization_roundtrip() {
        let snap = NvmlGpuSnapshot::stub_rtx3090(0);
        let bytes = snap.to_bytes();
        assert_eq!(bytes.len(), NVML_SNAPSHOT_SIZE);

        let decoded = NvmlGpuSnapshot::from_bytes(&bytes).expect("deserialization failed");
        assert_eq!(decoded.name, snap.name);
        assert_eq!(decoded.uuid, snap.uuid);
        assert_eq!(decoded.pci_bus_id, snap.pci_bus_id);
        assert_eq!(decoded.vram_total, snap.vram_total);
        assert_eq!(decoded.vram_used, snap.vram_used);
        assert_eq!(decoded.vram_free, snap.vram_free);
        assert_eq!(decoded.temperature_gpu, snap.temperature_gpu);
        assert_eq!(decoded.power_usage_mw, snap.power_usage_mw);
        assert_eq!(decoded.power_limit_mw, snap.power_limit_mw);
        assert_eq!(decoded.compute_cap_major, snap.compute_cap_major);
        assert_eq!(decoded.compute_cap_minor, snap.compute_cap_minor);
        assert_eq!(decoded.cuda_driver_version, snap.cuda_driver_version);
        assert_eq!(decoded.fan_speed_pct, snap.fan_speed_pct);
        assert_eq!(decoded.num_cores, snap.num_cores);
        assert_eq!(decoded.architecture, snap.architecture);
    }

    #[test]
    fn test_snapshot_too_short_returns_none() {
        let bytes = vec![0u8; NVML_SNAPSHOT_SIZE - 1];
        assert!(NvmlGpuSnapshot::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_snapshot_string_accessors() {
        let snap = NvmlGpuSnapshot::stub_rtx3090(0);
        assert_eq!(snap.name_str(), "NVIDIA GeForce RTX 3090");
        assert!(snap.uuid_str().starts_with("GPU-"));
        assert!(snap.pci_bus_id_str().contains(":00.0"));
        assert_eq!(snap.driver_version_str(), "535.129.03");
        assert_eq!(snap.nvml_version_str(), "12.535.129.03");
    }

    #[test]
    fn test_clock_type_from_raw() {
        assert_eq!(NvmlClockType::from_raw(0), Some(NvmlClockType::Graphics));
        assert_eq!(NvmlClockType::from_raw(1), Some(NvmlClockType::Sm));
        assert_eq!(NvmlClockType::from_raw(2), Some(NvmlClockType::Mem));
        assert_eq!(NvmlClockType::from_raw(3), Some(NvmlClockType::Video));
        assert_eq!(NvmlClockType::from_raw(4), None);
    }

    #[test]
    fn test_stub_different_indices() {
        let snap0 = NvmlGpuSnapshot::stub_rtx3090(0);
        let snap1 = NvmlGpuSnapshot::stub_rtx3090(1);
        // Different UUIDs and PCI bus IDs
        assert_ne!(snap0.uuid, snap1.uuid);
        assert_ne!(snap0.pci_bus_id, snap1.pci_bus_id);
        // Same hardware specs
        assert_eq!(snap0.vram_total, snap1.vram_total);
        assert_eq!(snap0.num_cores, snap1.num_cores);
    }
}
