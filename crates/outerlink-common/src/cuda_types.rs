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
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    // NOTE: codes 9-99 reserved by NVIDIA, not commonly encountered
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    ContextAlreadyInUse = 203,
    MapFailed = 205,
    UnmapFailed = 206,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    PeerAccessUnsupported = 216,
    InvalidPtx = 217,
    InvalidGraphicsContext = 218,
    NvlinkUncorrectable = 219,
    JitCompilerNotFound = 220,
    UnsupportedPtxVersion = 221,
    JitCompilationDisabled = 222,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    PrimaryContextActive = 708,
    LaunchFailed = 719,
    MpsConnectionFailed = 805,
    MpsRpcFailure = 806,
    MpsServerNotReady = 807,
    MpsMaxClientsReached = 808,
    MpsMaxConnectionsReached = 809,
    StreamCaptureWrongThread = 908,
    Timeout = 909,
    SystemNotReady = 910,
    Unknown = 999,
    /// OuterLink-specific: network/transport error
    TransportError = 10000,
    /// OuterLink-specific: remote server error
    RemoteError = 10001,
    /// OuterLink-specific: handle not found
    HandleNotFound = 10002,
}

impl CuResult {
    /// Convert from raw u32 value.
    ///
    /// Unrecognized CUDA error codes are mapped to `Unknown` and the raw
    /// value is logged at `trace` level so it remains observable for
    /// debugging.
    pub fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            5 => Self::ProfilerDisabled,
            6 => Self::ProfilerNotInitialized,
            7 => Self::ProfilerAlreadyStarted,
            8 => Self::ProfilerAlreadyStopped,
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            200 => Self::InvalidImage,
            201 => Self::InvalidContext,
            202 => Self::ContextAlreadyCurrent,
            203 => Self::ContextAlreadyInUse,
            205 => Self::MapFailed,
            206 => Self::UnmapFailed,
            208 => Self::AlreadyMapped,
            209 => Self::NoBinaryForGpu,
            216 => Self::PeerAccessUnsupported,
            217 => Self::InvalidPtx,
            218 => Self::InvalidGraphicsContext,
            219 => Self::NvlinkUncorrectable,
            220 => Self::JitCompilerNotFound,
            221 => Self::UnsupportedPtxVersion,
            222 => Self::JitCompilationDisabled,
            500 => Self::NotFound,
            600 => Self::NotReady,
            700 => Self::IllegalAddress,
            708 => Self::PrimaryContextActive,
            719 => Self::LaunchFailed,
            805 => Self::MpsConnectionFailed,
            806 => Self::MpsRpcFailure,
            807 => Self::MpsServerNotReady,
            808 => Self::MpsMaxClientsReached,
            809 => Self::MpsMaxConnectionsReached,
            908 => Self::StreamCaptureWrongThread,
            909 => Self::Timeout,
            910 => Self::SystemNotReady,
            999 => Self::Unknown,
            10000 => Self::TransportError,
            10001 => Self::RemoteError,
            10002 => Self::HandleNotFound,
            other => {
                tracing::trace!("unmapped CUDA error code {other}, returning CuResult::Unknown");
                Self::Unknown
            }
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
