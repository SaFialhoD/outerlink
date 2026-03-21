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
    NoDevice = 100,
    InvalidDevice = 101,
    DeviceNotLicensed = 102,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    EccUncorrectable = 214,
    UnsupportedLimit = 215,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidHandle = 400,
    IllegalState = 401,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    ContextIsDestroyed = 709,
    Assert = 710,
    TooManyPeers = 711,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    HardwareStackError = 714,
    IllegalInstruction = 715,
    MisalignedAddress = 716,
    InvalidAddressSpace = 717,
    InvalidPc = 718,
    LaunchFailed = 719,
    CooperativeLaunchTooLarge = 720,
    NotPermitted = 800,
    NotSupported = 801,
    SystemNotReady = 802,
    SystemDriverMismatch = 803,
    CompatNotSupportedOnDevice = 804,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalidated = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolation = 905,
    StreamCaptureImplicit = 906,
    CapturedEvent = 907,
    Timeout = 909,
    GraphExecUpdateFailure = 910,
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
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            102 => Self::DeviceNotLicensed,
            200 => Self::InvalidImage,
            201 => Self::InvalidContext,
            202 => Self::ContextAlreadyCurrent,
            205 => Self::MapFailed,
            206 => Self::UnmapFailed,
            208 => Self::AlreadyMapped,
            209 => Self::NoBinaryForGpu,
            210 => Self::AlreadyAcquired,
            211 => Self::NotMapped,
            212 => Self::NotMappedAsArray,
            213 => Self::NotMappedAsPointer,
            214 => Self::EccUncorrectable,
            215 => Self::UnsupportedLimit,
            300 => Self::InvalidSource,
            301 => Self::FileNotFound,
            302 => Self::SharedObjectSymbolNotFound,
            303 => Self::SharedObjectInitFailed,
            304 => Self::OperatingSystem,
            400 => Self::InvalidHandle,
            401 => Self::IllegalState,
            500 => Self::NotFound,
            600 => Self::NotReady,
            700 => Self::IllegalAddress,
            701 => Self::LaunchOutOfResources,
            702 => Self::LaunchTimeout,
            703 => Self::LaunchIncompatibleTexturing,
            704 => Self::PeerAccessAlreadyEnabled,
            705 => Self::PeerAccessNotEnabled,
            709 => Self::ContextIsDestroyed,
            710 => Self::Assert,
            711 => Self::TooManyPeers,
            712 => Self::HostMemoryAlreadyRegistered,
            713 => Self::HostMemoryNotRegistered,
            714 => Self::HardwareStackError,
            715 => Self::IllegalInstruction,
            716 => Self::MisalignedAddress,
            717 => Self::InvalidAddressSpace,
            718 => Self::InvalidPc,
            719 => Self::LaunchFailed,
            720 => Self::CooperativeLaunchTooLarge,
            800 => Self::NotPermitted,
            801 => Self::NotSupported,
            802 => Self::SystemNotReady,
            803 => Self::SystemDriverMismatch,
            804 => Self::CompatNotSupportedOnDevice,
            900 => Self::StreamCaptureUnsupported,
            901 => Self::StreamCaptureInvalidated,
            902 => Self::StreamCaptureMerge,
            903 => Self::StreamCaptureUnmatched,
            904 => Self::StreamCaptureUnjoined,
            905 => Self::StreamCaptureIsolation,
            906 => Self::StreamCaptureImplicit,
            907 => Self::CapturedEvent,
            909 => Self::Timeout,
            910 => Self::GraphExecUpdateFailure,
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
