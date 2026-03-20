//! CUDA type definitions matching the CUDA Driver API.
//!
//! These are Rust representations of CUDA's C types, used throughout
//! the OuterLink protocol for serialization and handle translation.

/// CUDA device ordinal
pub type CuDevice = i32;

/// CUDA device pointer (GPU memory address)
pub type CuDevicePtr = u64;

/// Opaque handle types - represented as u64 for serialization.
/// On the client side, these are synthetic (locally generated).
/// On the server side, these are real CUDA handles cast to u64.
pub type CuContext = u64;
pub type CuModule = u64;
pub type CuFunction = u64;
pub type CuStream = u64;
pub type CuEvent = u64;

/// CUDA UUID (16 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CuUuid {
    pub bytes: [u8; 16],
}

/// CUDA result/error codes matching CUresult enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuResult {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    NotFound = 500,
    NotReady = 600,
    LaunchFailed = 719,
    /// OuterLink-specific: network/transport error
    TransportError = 900,
    /// OuterLink-specific: remote server error
    RemoteError = 901,
    /// OuterLink-specific: handle not found
    HandleNotFound = 902,
    Unknown = 999,
}

impl CuResult {
    /// Convert from raw u32 value
    pub fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            201 => Self::InvalidContext,
            202 => Self::ContextAlreadyCurrent,
            205 => Self::MapFailed,
            206 => Self::UnmapFailed,
            500 => Self::NotFound,
            600 => Self::NotReady,
            719 => Self::LaunchFailed,
            900 => Self::TransportError,
            901 => Self::RemoteError,
            902 => Self::HandleNotFound,
            _ => Self::Unknown,
        }
    }

    /// Convert to raw u32 value for returning to CUDA applications
    pub fn as_raw(self) -> u32 {
        self as u32
    }

    /// Check if this result indicates success
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// CUDA device attribute codes (subset - extend as needed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CuDeviceAttribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MultiprocessorCount = 16,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaxSharedMemoryPerMultiprocessor = 81,
}

impl CuDeviceAttribute {
    pub fn from_raw(value: i32) -> Option<Self> {
        match value {
            1 => Some(Self::MaxThreadsPerBlock),
            2 => Some(Self::MaxBlockDimX),
            3 => Some(Self::MaxBlockDimY),
            4 => Some(Self::MaxBlockDimZ),
            5 => Some(Self::MaxGridDimX),
            6 => Some(Self::MaxGridDimY),
            7 => Some(Self::MaxGridDimZ),
            16 => Some(Self::MultiprocessorCount),
            75 => Some(Self::ComputeCapabilityMajor),
            76 => Some(Self::ComputeCapabilityMinor),
            81 => Some(Self::MaxSharedMemoryPerMultiprocessor),
            _ => None,
        }
    }
}
