//! Request handler: dispatches incoming protocol messages to the GPU backend.
//!
//! The handler is the bridge between the wire protocol (raw bytes) and the
//! [`GpuBackend`] trait.  For every incoming `(MessageHeader, payload)` pair
//! it:
//!
//! 1. Matches on `msg_type`.
//! 2. Deserialises the request payload (little-endian fields).
//! 3. Calls the appropriate `GpuBackend` method.
//! 4. Serialises a response payload: 4 bytes `CuResult` + method-specific data.
//! 5. Returns `(MessageHeader, response_bytes)`.

use outerlink_common::cuda_types::CuResult;
use outerlink_common::protocol::{MessageHeader, MessageType};

use crate::gpu_backend::GpuBackend;

/// Encode a `CuResult` as 4 little-endian bytes.
fn encode_result(r: CuResult) -> [u8; 4] {
    (r as u32).to_le_bytes()
}

/// Build a response that carries only a `CuResult` (no extra data).
fn result_only(request_id: u64, r: CuResult) -> (MessageHeader, Vec<u8>) {
    let payload = encode_result(r).to_vec();
    let header = MessageHeader::new_response(request_id, payload.len() as u32);
    (header, payload)
}

/// Build a response with a `CuResult::Success` prefix followed by `data`.
fn success_with(request_id: u64, data: &[u8]) -> (MessageHeader, Vec<u8>) {
    let mut payload = encode_result(CuResult::Success).to_vec();
    payload.extend_from_slice(data);
    let header = MessageHeader::new_response(request_id, payload.len() as u32);
    (header, payload)
}

/// Build an error response from a `Result<T, CuResult>` that failed.
fn error_response(request_id: u64, err: CuResult) -> (MessageHeader, Vec<u8>) {
    result_only(request_id, err)
}

/// Dispatch a single request to the `GpuBackend` and return a response.
///
/// # Payload layouts (all little-endian)
///
/// | MessageType          | Request payload         | Response data (after 4-byte CuResult) |
/// |----------------------|-------------------------|---------------------------------------|
/// | Init                 | (empty)                 | (empty)                               |
/// | DriverGetVersion     | (empty)                 | i32 version                           |
/// | DeviceGet            | i32 ordinal             | i32 device                            |
/// | DeviceGetCount       | (empty)                 | i32 count                             |
/// | DeviceGetName        | i32 device              | u32 len + UTF-8 bytes                 |
/// | DeviceGetAttribute   | i32 attrib, i32 device  | i32 value                             |
/// | DeviceTotalMem       | i32 device              | u64 bytes                             |
/// | DeviceGetUuid        | i32 device              | 16 bytes                              |
/// | MemAlloc             | u64 size                | u64 device_ptr                        |
/// | MemFree              | u64 device_ptr          | (empty)                               |
/// | MemcpyHtoD           | u64 dst + raw bytes     | (empty)                               |
/// | MemcpyDtoH           | u64 src, u64 size       | raw bytes                             |
/// | MemGetInfo           | (empty)                 | u64 free, u64 total                   |
/// | CtxCreate            | u32 flags, i32 device   | u64 ctx_handle                        |
/// | CtxDestroy           | u64 ctx                 | (empty)                               |
/// | CtxSetCurrent        | u64 ctx                 | (empty)                               |
/// | CtxGetCurrent        | (empty)                 | u64 ctx                               |
/// | CtxGetDevice         | u64 ctx                 | i32 device                            |
/// | CtxSynchronize       | (empty)                 | (empty)                               |
pub fn handle_request(
    backend: &dyn GpuBackend,
    header: &MessageHeader,
    payload: &[u8],
) -> (MessageHeader, Vec<u8>) {
    let rid = header.request_id;

    match header.msg_type {
        MessageType::Handshake => {
            tracing::info!("client handshake received");
            result_only(rid, CuResult::Success)
        }

        MessageType::Init => {
            let r = backend.init();
            result_only(rid, r)
        }

        MessageType::DriverGetVersion => match backend.driver_get_version() {
            Ok(ver) => success_with(rid, &ver.to_le_bytes()),
            Err(e) => error_response(rid, e),
        },

        MessageType::DeviceGet => {
            // Request: i32 ordinal
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ordinal = i32::from_le_bytes(payload[..4].try_into().unwrap());
            // In CUDA, cuDeviceGet just validates the ordinal and returns it.
            let count = match backend.device_get_count() {
                Ok(c) => c,
                Err(e) => return error_response(rid, e),
            };
            if ordinal < 0 || ordinal >= count {
                return error_response(rid, CuResult::InvalidDevice);
            }
            success_with(rid, &ordinal.to_le_bytes())
        }

        MessageType::DeviceGetCount => match backend.device_get_count() {
            Ok(count) => success_with(rid, &count.to_le_bytes()),
            Err(e) => error_response(rid, e),
        },

        MessageType::DeviceGetName => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.device_get_name(device) {
                Ok(name) => {
                    let name_bytes = name.as_bytes();
                    let mut data = (name_bytes.len() as u32).to_le_bytes().to_vec();
                    data.extend_from_slice(name_bytes);
                    success_with(rid, &data)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DeviceGetAttribute => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let attrib = i32::from_le_bytes(payload[..4].try_into().unwrap());
            let device = i32::from_le_bytes(payload[4..8].try_into().unwrap());
            match backend.device_get_attribute(attrib, device) {
                Ok(val) => success_with(rid, &val.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DeviceTotalMem => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.device_total_mem(device) {
                Ok(bytes) => success_with(rid, &(bytes as u64).to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DeviceGetUuid => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.device_get_uuid(device) {
                Ok(uuid) => success_with(rid, &uuid),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemAlloc => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let size = u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
            match backend.mem_alloc(size) {
                Ok(ptr) => success_with(rid, &ptr.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemFree => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ptr = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let r = backend.mem_free(ptr);
            result_only(rid, r)
        }

        MessageType::MemcpyHtoD => {
            // First 8 bytes: dst device pointer, rest: raw data
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let data = &payload[8..];
            if data.is_empty() {
                return error_response(rid, CuResult::InvalidValue);
            }
            let r = backend.memcpy_htod(dst, data);
            result_only(rid, r)
        }

        MessageType::MemcpyDtoH => {
            if payload.len() < 16 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let src = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let size = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
            match backend.memcpy_dtoh(src, size) {
                Ok(buf) => success_with(rid, &buf),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemGetInfo => match backend.mem_get_info() {
            Ok((free, total)) => {
                let mut data = (free as u64).to_le_bytes().to_vec();
                data.extend_from_slice(&(total as u64).to_le_bytes());
                success_with(rid, &data)
            }
            Err(e) => error_response(rid, e),
        },

        MessageType::CtxCreate => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let flags = u32::from_le_bytes(payload[..4].try_into().unwrap());
            let device = i32::from_le_bytes(payload[4..8].try_into().unwrap());
            match backend.ctx_create(flags, device) {
                Ok(ctx) => success_with(rid, &ctx.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxDestroy => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ctx = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.ctx_destroy(ctx) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxSetCurrent => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ctx = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.ctx_set_current(ctx) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxGetCurrent => match backend.ctx_get_current() {
            Ok(ctx) => success_with(rid, &ctx.to_le_bytes()),
            Err(e) => error_response(rid, e),
        },

        MessageType::CtxGetDevice => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ctx = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.ctx_get_device(ctx) {
                Ok(device) => success_with(rid, &device.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxSynchronize => match backend.ctx_synchronize() {
            Ok(()) => result_only(rid, CuResult::Success),
            Err(e) => error_response(rid, e),
        },

        // Anything else is unimplemented for the PoC.
        _ => {
            tracing::warn!(msg_type = ?header.msg_type, "unhandled message type");
            error_response(rid, CuResult::InvalidValue)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::StubGpuBackend;

    /// Helper: build a request header.
    fn req(msg_type: MessageType, payload_len: u32) -> MessageHeader {
        MessageHeader::new_request(1, msg_type, payload_len)
    }

    /// Extract the CuResult from the first 4 bytes of a response payload.
    fn response_result(payload: &[u8]) -> CuResult {
        CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()))
    }

    #[test]
    fn test_handshake() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::Handshake, 0);
        let (resp_hdr, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(resp_hdr.msg_type, MessageType::Response);
    }

    #[test]
    fn test_init() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::Init, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_driver_get_version() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::DriverGetVersion, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ver = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(ver, 12040);
    }

    #[test]
    fn test_device_get_count() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::DeviceGetCount, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let count = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_device_get_valid() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGet, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let dev = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(dev, 0);
    }

    #[test]
    fn test_device_get_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 5i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGet, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidDevice);
    }

    #[test]
    fn test_device_get_name() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGetName, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let name_len = u32::from_le_bytes(resp[4..8].try_into().unwrap()) as usize;
        let name = std::str::from_utf8(&resp[8..8 + name_len]).unwrap();
        assert_eq!(name, "OuterLink Virtual GPU");
    }

    #[test]
    fn test_device_get_attribute() {
        let gpu = StubGpuBackend::new();
        let mut payload = 75i32.to_le_bytes().to_vec(); // ComputeCapabilityMajor
        payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::DeviceGetAttribute, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(val, 8);
    }

    #[test]
    fn test_device_total_mem() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceTotalMem, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let mem = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(mem as usize, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_device_get_uuid() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGetUuid, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..8], b"OLNK");
    }

    #[test]
    fn test_mem_alloc_free() {
        let gpu = StubGpuBackend::new();

        // Alloc
        let payload = 1024u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(ptr, 0);

        // Free
        let payload = ptr.to_le_bytes();
        let hdr = req(MessageType::MemFree, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_memcpy_roundtrip_via_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc 64 bytes.
        let alloc_payload = 64u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &alloc_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // HtoD: write 64 bytes of 0xAB.
        let mut htod_payload = ptr.to_le_bytes().to_vec();
        htod_payload.extend_from_slice(&vec![0xAB; 64]);
        let hdr = req(MessageType::MemcpyHtoD, htod_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &htod_payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // DtoH: read them back.
        let mut dtoh_payload = ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..], &vec![0xAB; 64]);
    }

    #[test]
    fn test_mem_get_info() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::MemGetInfo, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let free = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        let total = u64::from_le_bytes(resp[12..20].try_into().unwrap());
        assert_eq!(total as usize, 24 * 1024 * 1024 * 1024);
        assert_eq!(free, total);
    }

    #[test]
    fn test_response_header_type() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::Init, 0);
        let (resp_hdr, _) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(resp_hdr.msg_type, MessageType::Response);
        assert_eq!(resp_hdr.request_id, 1);
    }

    #[test]
    fn test_short_payload_rejected() {
        let gpu = StubGpuBackend::new();
        // DeviceGetAttribute needs 8 bytes but we send 2.
        let hdr = req(MessageType::DeviceGetAttribute, 2);
        let (_, resp) = handle_request(&gpu, &hdr, &[0, 0]);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_unhandled_message_type() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::LaunchKernel, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- Context operation tests -----

    #[test]
    fn test_ctx_create() {
        let gpu = StubGpuBackend::new();
        let mut payload = 0u32.to_le_bytes().to_vec(); // flags
        payload.extend_from_slice(&0i32.to_le_bytes()); // device
        let hdr = req(MessageType::CtxCreate, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(ctx, 0);
    }

    #[test]
    fn test_ctx_create_invalid_device() {
        let gpu = StubGpuBackend::new();
        let mut payload = 0u32.to_le_bytes().to_vec(); // flags
        payload.extend_from_slice(&99i32.to_le_bytes()); // invalid device
        let hdr = req(MessageType::CtxCreate, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidDevice);
    }

    #[test]
    fn test_ctx_destroy() {
        let gpu = StubGpuBackend::new();

        // Create a context first.
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &create_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Destroy it.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxDestroy, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_destroy_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xDEADu64.to_le_bytes();
        let hdr = req(MessageType::CtxDestroy, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidContext);
    }

    #[test]
    fn test_ctx_set_current() {
        let gpu = StubGpuBackend::new();

        // Create a context.
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &create_payload);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Set it as current.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxSetCurrent, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_set_current_null() {
        let gpu = StubGpuBackend::new();
        // Set ctx to 0 (unset current).
        let payload = 0u64.to_le_bytes();
        let hdr = req(MessageType::CtxSetCurrent, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_get_current() {
        let gpu = StubGpuBackend::new();

        // Create a context (auto-sets as current).
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &create_payload);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get current.
        let hdr = req(MessageType::CtxGetCurrent, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let current = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(current, ctx);
    }

    #[test]
    fn test_ctx_get_device() {
        let gpu = StubGpuBackend::new();

        // Create a context on device 0.
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &create_payload);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get device for this context.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxGetDevice, payload.len() as u32);
        let (_, resp) = handle_request(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let device = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(device, 0);
    }

    #[test]
    fn test_ctx_synchronize() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::CtxSynchronize, 0);
        let (_, resp) = handle_request(&gpu, &hdr, &[]);
        assert_eq!(response_result(&resp), CuResult::Success);
    }
}
