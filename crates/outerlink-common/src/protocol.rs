//! OuterLink wire protocol.
//!
//! Binary protocol for client-server communication.
//! Header: 4 bytes magic "OLNK" + 2 bytes version + 2 bytes flags +
//!         8 bytes request_id + 2 bytes msg_type + 4 bytes payload_len = 22 bytes
//!
//! All header fields are big-endian. Payload is little-endian (x86 native).

use crate::cuda_types::*;

/// Protocol magic bytes
pub const MAGIC: [u8; 4] = *b"OLNK";

/// Protocol version
pub const VERSION: u16 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 22;

/// Maximum payload size (256 MB - should be enough for any single CUDA operation)
pub const MAX_PAYLOAD_SIZE: u32 = 256 * 1024 * 1024;

/// Message type codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MessageType {
    // Handshake
    Handshake = 0x0001,
    HandshakeAck = 0x0002,

    // Device queries
    Init = 0x0010,
    DriverGetVersion = 0x0011,
    DeviceGet = 0x0012,
    DeviceGetCount = 0x0013,
    DeviceGetName = 0x0014,
    DeviceGetAttribute = 0x0015,
    DeviceTotalMem = 0x0016,
    DeviceGetUuid = 0x0017,

    // Context
    CtxCreate = 0x0020,
    CtxDestroy = 0x0021,
    CtxSetCurrent = 0x0022,
    CtxGetCurrent = 0x0023,
    CtxGetDevice = 0x0024,
    CtxSynchronize = 0x0025,
    DevicePrimaryCtxRetain = 0x0026,
    DevicePrimaryCtxRelease = 0x0027,
    DevicePrimaryCtxGetState = 0x0028,
    DevicePrimaryCtxSetFlags = 0x0029,
    DevicePrimaryCtxReset = 0x002A,

    // Memory
    MemAlloc = 0x0030,
    MemFree = 0x0031,
    MemcpyHtoD = 0x0032,
    MemcpyDtoH = 0x0033,
    MemcpyDtoD = 0x0034,
    MemGetInfo = 0x0035,
    MemAllocHost = 0x0036,
    MemFreeHost = 0x0037,
    MemcpyHtoDAsync = 0x0038,
    MemcpyDtoHAsync = 0x0039,
    MemsetD8 = 0x003A,
    MemsetD32 = 0x003B,
    MemsetD8Async = 0x003C,
    MemsetD32Async = 0x003D,

    // Module
    ModuleLoadData = 0x0040,
    ModuleUnload = 0x0041,
    ModuleGetFunction = 0x0042,
    ModuleGetGlobal = 0x0043,
    ModuleLoadDataEx = 0x0044,
    FuncGetAttribute = 0x0045,

    // Execution
    LaunchKernel = 0x0050,

    // Stream
    StreamCreate = 0x0060,
    StreamDestroy = 0x0061,
    StreamSynchronize = 0x0062,
    StreamWaitEvent = 0x0063,
    StreamQuery = 0x0064,
    StreamCreateWithPriority = 0x0065,
    StreamGetPriority = 0x0066,
    StreamGetFlags = 0x0067,
    StreamGetCtx = 0x0068,

    // Event
    EventCreate = 0x0070,
    EventDestroy = 0x0071,
    EventRecord = 0x0072,
    EventSynchronize = 0x0073,
    EventElapsedTime = 0x0074,
    EventQuery = 0x0075,

    // Response (server -> client)
    Response = 0x00F0,
    Error = 0x00FF,
}

impl MessageType {
    pub fn from_raw(value: u16) -> Option<Self> {
        // Safety: all variants are explicitly listed
        match value {
            0x0001 => Some(Self::Handshake),
            0x0002 => Some(Self::HandshakeAck),
            0x0010 => Some(Self::Init),
            0x0011 => Some(Self::DriverGetVersion),
            0x0012 => Some(Self::DeviceGet),
            0x0013 => Some(Self::DeviceGetCount),
            0x0014 => Some(Self::DeviceGetName),
            0x0015 => Some(Self::DeviceGetAttribute),
            0x0016 => Some(Self::DeviceTotalMem),
            0x0017 => Some(Self::DeviceGetUuid),
            0x0020 => Some(Self::CtxCreate),
            0x0021 => Some(Self::CtxDestroy),
            0x0022 => Some(Self::CtxSetCurrent),
            0x0023 => Some(Self::CtxGetCurrent),
            0x0024 => Some(Self::CtxGetDevice),
            0x0025 => Some(Self::CtxSynchronize),
            0x0026 => Some(Self::DevicePrimaryCtxRetain),
            0x0027 => Some(Self::DevicePrimaryCtxRelease),
            0x0028 => Some(Self::DevicePrimaryCtxGetState),
            0x0029 => Some(Self::DevicePrimaryCtxSetFlags),
            0x002A => Some(Self::DevicePrimaryCtxReset),
            0x0030 => Some(Self::MemAlloc),
            0x0031 => Some(Self::MemFree),
            0x0032 => Some(Self::MemcpyHtoD),
            0x0033 => Some(Self::MemcpyDtoH),
            0x0034 => Some(Self::MemcpyDtoD),
            0x0035 => Some(Self::MemGetInfo),
            0x0036 => Some(Self::MemAllocHost),
            0x0037 => Some(Self::MemFreeHost),
            0x0038 => Some(Self::MemcpyHtoDAsync),
            0x0039 => Some(Self::MemcpyDtoHAsync),
            0x003A => Some(Self::MemsetD8),
            0x003B => Some(Self::MemsetD32),
            0x003C => Some(Self::MemsetD8Async),
            0x003D => Some(Self::MemsetD32Async),
            0x0040 => Some(Self::ModuleLoadData),
            0x0041 => Some(Self::ModuleUnload),
            0x0042 => Some(Self::ModuleGetFunction),
            0x0043 => Some(Self::ModuleGetGlobal),
            0x0044 => Some(Self::ModuleLoadDataEx),
            0x0045 => Some(Self::FuncGetAttribute),
            0x0050 => Some(Self::LaunchKernel),
            0x0060 => Some(Self::StreamCreate),
            0x0061 => Some(Self::StreamDestroy),
            0x0062 => Some(Self::StreamSynchronize),
            0x0063 => Some(Self::StreamWaitEvent),
            0x0064 => Some(Self::StreamQuery),
            0x0065 => Some(Self::StreamCreateWithPriority),
            0x0066 => Some(Self::StreamGetPriority),
            0x0067 => Some(Self::StreamGetFlags),
            0x0068 => Some(Self::StreamGetCtx),
            0x0070 => Some(Self::EventCreate),
            0x0071 => Some(Self::EventDestroy),
            0x0072 => Some(Self::EventRecord),
            0x0073 => Some(Self::EventSynchronize),
            0x0074 => Some(Self::EventElapsedTime),
            0x0075 => Some(Self::EventQuery),
            0x00F0 => Some(Self::Response),
            0x00FF => Some(Self::Error),
            _ => None,
        }
    }
}

/// Wire header (22 bytes)
#[derive(Debug, Clone)]
pub struct MessageHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub request_id: u64,
    pub msg_type: MessageType,
    pub payload_len: u32,
}

impl MessageHeader {
    /// Serialize header to bytes (big-endian)
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_be_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_be_bytes());
        buf[8..16].copy_from_slice(&self.request_id.to_be_bytes());
        buf[16..18].copy_from_slice(&(self.msg_type as u16).to_be_bytes());
        buf[18..22].copy_from_slice(&self.payload_len.to_be_bytes());
        buf
    }

    /// Deserialize header from bytes (big-endian)
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Option<Self> {
        let magic = [buf[0], buf[1], buf[2], buf[3]];
        if magic != MAGIC {
            return None;
        }

        let version = u16::from_be_bytes([buf[4], buf[5]]);
        if version != VERSION {
            return None; // Version mismatch
        }

        let flags = u16::from_be_bytes([buf[6], buf[7]]);
        let request_id = u64::from_be_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]);
        let msg_type_raw = u16::from_be_bytes([buf[16], buf[17]]);
        let payload_len = u32::from_be_bytes([buf[18], buf[19], buf[20], buf[21]]);

        if payload_len > MAX_PAYLOAD_SIZE {
            return None; // Payload too large
        }

        let msg_type = MessageType::from_raw(msg_type_raw)?;

        Some(Self {
            magic,
            version,
            flags,
            request_id,
            msg_type,
            payload_len,
        })
    }

    /// Create a new request header
    pub fn new_request(request_id: u64, msg_type: MessageType, payload_len: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            request_id,
            msg_type,
            payload_len,
        }
    }

    /// Create a response header
    pub fn new_response(request_id: u64, payload_len: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            request_id,
            msg_type: MessageType::Response,
            payload_len,
        }
    }
}

/// Response payload - returned from server for every request
#[derive(Debug, Clone)]
pub struct ResponsePayload {
    /// CUDA result code
    pub result: CuResult,
    /// Response-specific data (varies by request type)
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = MessageHeader::new_request(42, MessageType::DeviceGetCount, 0);
        let bytes = header.to_bytes();
        let decoded = MessageHeader::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.magic, MAGIC);
        assert_eq!(decoded.version, VERSION);
        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.msg_type, MessageType::DeviceGetCount);
        assert_eq!(decoded.payload_len, 0);
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"XXXX");
        assert!(MessageHeader::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_message_type_roundtrip() {
        for raw in [0x0001u16, 0x0010, 0x0020, 0x0030, 0x0050, 0x00F0, 0x00FF] {
            let mt = MessageType::from_raw(raw).unwrap();
            assert_eq!(mt as u16, raw);
        }
    }
}
