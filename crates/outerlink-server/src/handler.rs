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
use crate::session::ConnectionSession;

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
/// | Init                 | u32 flags               | (empty)                               |
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
/// | DevicePrimaryCtxRetain  | i32 device           | u64 ctx_handle                        |
/// | DevicePrimaryCtxRelease | i32 device           | (empty)                               |
/// | DevicePrimaryCtxGetState| i32 device           | u32 flags, i32 active                 |
/// | DevicePrimaryCtxSetFlags| i32 device, u32 flags| (empty)                               |
/// | DevicePrimaryCtxReset   | i32 device           | (empty)                               |
/// | FuncGetAttribute        | i32 attrib, u64 func | i32 value                             |
pub fn handle_request(
    backend: &dyn GpuBackend,
    header: &MessageHeader,
    payload: &[u8],
    session: &mut ConnectionSession,
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
                Ok(ptr) => {
                    session.track_mem_alloc(ptr);
                    success_with(rid, &ptr.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemFree => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ptr = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.mem_free(ptr) {
                Ok(()) => {
                    session.untrack_mem_alloc(ptr);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => result_only(rid, e),
            }
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
                Ok(ctx) => {
                    session.set_current_ctx(ctx);
                    session.track_context(ctx);
                    success_with(rid, &ctx.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxDestroy => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ctx = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.ctx_destroy(ctx) {
                Ok(()) => {
                    session.clear_if_current(ctx);
                    session.untrack_context(ctx);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxSetCurrent => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ctx = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match session.validate_set_current(ctx, backend.ctx_exists(ctx)) {
                Ok(()) => {
                    session.set_current_ctx(ctx);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::CtxGetCurrent => {
            let ctx = session.current_ctx();
            success_with(rid, &ctx.to_le_bytes())
        }

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

        // --- Primary context operations ---

        MessageType::DevicePrimaryCtxRetain => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.primary_ctx_retain(device) {
                Ok(ctx) => {
                    session.track_primary_ctx(device, ctx);
                    success_with(rid, &ctx.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DevicePrimaryCtxRelease => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.primary_ctx_release(device) {
                Ok(()) => {
                    // Untrack: this session no longer claims this device's primary context.
                    // If another session also retained it, the backend refcount handles that.
                    session.untrack_primary_ctx(device);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DevicePrimaryCtxGetState => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.primary_ctx_get_state(device) {
                Ok((flags, active)) => {
                    let mut data = [0u8; 8];
                    data[..4].copy_from_slice(&flags.to_le_bytes());
                    data[4..8].copy_from_slice(&active.to_le_bytes());
                    success_with(rid, &data)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DevicePrimaryCtxSetFlags => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            let flags = u32::from_le_bytes(payload[4..8].try_into().unwrap());
            match backend.primary_ctx_set_flags(device, flags) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::DevicePrimaryCtxReset => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let device = i32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.primary_ctx_reset(device) {
                Ok(_old_ctx) => {
                    // Always untrack regardless of old_ctx — real CudaGpuBackend
                    // returns None since CUDA doesn't expose the old handle.
                    session.untrack_primary_ctx(device);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        // --- Module operations ---

        MessageType::ModuleLoadData => {
            if payload.is_empty() {
                return error_response(rid, CuResult::InvalidValue);
            }
            match backend.module_load_data(payload) {
                Ok(handle) => {
                    session.track_module(handle);
                    success_with(rid, &handle.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::ModuleLoadDataEx => {
            // Wire format:
            //   4B image_len (u32 LE)
            //   4B num_options (u32 LE)
            //   num_options * 12B (4B option i32 LE + 8B value u64 LE)
            //   image_len bytes of image data
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let image_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            let num_options = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
            // CUDA JIT has ~30 defined options; 256 is extremely generous.
            // Reject before allocating to prevent a crafted client from
            // causing a huge Vec::with_capacity allocation.
            if num_options > 256 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let options_size = match num_options.checked_mul(12) {
                Some(v) => v,
                None => return error_response(rid, CuResult::InvalidValue),
            };
            let expected_len = match (8usize).checked_add(options_size).and_then(|v| v.checked_add(image_len)) {
                Some(v) => v,
                None => return error_response(rid, CuResult::InvalidValue),
            };
            if payload.len() < expected_len || image_len == 0 {
                return error_response(rid, CuResult::InvalidValue);
            }
            // Parse options
            let mut options = Vec::with_capacity(num_options);
            for i in 0..num_options {
                let base = 8 + i * 12;
                let opt_id = i32::from_le_bytes(payload[base..base + 4].try_into().unwrap());
                let opt_val = u64::from_le_bytes(payload[base + 4..base + 12].try_into().unwrap());
                options.push((opt_id, opt_val));
            }
            let image = &payload[8 + options_size..8 + options_size + image_len];
            match backend.module_load_data_ex(image, &options) {
                Ok(handle) => {
                    session.track_module(handle);
                    success_with(rid, &handle.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::ModuleUnload => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let module = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.module_unload(module) {
                Ok(()) => {
                    session.untrack_module(module);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::ModuleGetFunction => {
            if payload.len() < 13 {
                // Need at least 8B module + 4B name_len + 1B name
                return error_response(rid, CuResult::InvalidValue);
            }
            let module = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let name_len = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
            if payload.len() < 12 + name_len {
                return error_response(rid, CuResult::InvalidValue);
            }
            let name = match std::str::from_utf8(&payload[12..12 + name_len]) {
                Ok(s) => s,
                Err(_) => return error_response(rid, CuResult::InvalidValue),
            };
            match backend.module_get_function(module, name) {
                Ok(handle) => success_with(rid, &handle.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::ModuleGetGlobal => {
            if payload.len() < 13 {
                // Need at least 8B module + 4B name_len + 1B name
                return error_response(rid, CuResult::InvalidValue);
            }
            let module = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let name_len = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
            if payload.len() < 12 + name_len {
                return error_response(rid, CuResult::InvalidValue);
            }
            let name = match std::str::from_utf8(&payload[12..12 + name_len]) {
                Ok(s) => s,
                Err(_) => return error_response(rid, CuResult::InvalidValue),
            };
            match backend.module_get_global(module, name) {
                Ok((dptr, size)) => {
                    let mut data = dptr.to_le_bytes().to_vec();
                    data.extend_from_slice(&(size as u64).to_le_bytes());
                    success_with(rid, &data)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::FuncGetAttribute => {
            // Payload: 4B attrib (i32 LE) + 8B func handle (u64 LE) = 12 bytes
            if payload.len() < 12 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let attrib = i32::from_le_bytes(payload[..4].try_into().unwrap());
            let func = u64::from_le_bytes(payload[4..12].try_into().unwrap());
            match backend.func_get_attribute(attrib, func) {
                Ok(val) => success_with(rid, &val.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        // --- Stream operations ---

        MessageType::StreamCreate => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let flags = u32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.stream_create(flags) {
                Ok(handle) => {
                    session.track_stream(handle);
                    success_with(rid, &handle.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::StreamDestroy => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let stream = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.stream_destroy(stream) {
                Ok(()) => {
                    session.untrack_stream(stream);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::StreamSynchronize => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let stream = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.stream_synchronize(stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::StreamQuery => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let stream = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.stream_query(stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        // --- Event operations ---

        MessageType::EventCreate => {
            if payload.len() < 4 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let flags = u32::from_le_bytes(payload[..4].try_into().unwrap());
            match backend.event_create(flags) {
                Ok(handle) => {
                    session.track_event(handle);
                    success_with(rid, &handle.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::EventDestroy => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let event = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.event_destroy(event) {
                Ok(()) => {
                    session.untrack_event(event);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::EventRecord => {
            if payload.len() < 16 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let event = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let stream = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            match backend.event_record(event, stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::EventSynchronize => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let event = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.event_synchronize(event) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::EventElapsedTime => {
            if payload.len() < 16 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let start = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let end = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            match backend.event_elapsed_time(start, end) {
                Ok(ms) => success_with(rid, &ms.to_le_bytes()),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::EventQuery => {
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let event = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.event_query(event) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        // --- Kernel launch ---

        MessageType::LaunchKernel => {
            // 8B func + 3x4B grid + 3x4B block + 4B shared_mem + 8B stream = 44 bytes minimum
            // The params slice starts at byte 44 and may be empty (zero-parameter kernel).
            if payload.len() < 44 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let func = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let grid_x = u32::from_le_bytes(payload[8..12].try_into().unwrap());
            let grid_y = u32::from_le_bytes(payload[12..16].try_into().unwrap());
            let grid_z = u32::from_le_bytes(payload[16..20].try_into().unwrap());
            let block_x = u32::from_le_bytes(payload[20..24].try_into().unwrap());
            let block_y = u32::from_le_bytes(payload[24..28].try_into().unwrap());
            let block_z = u32::from_le_bytes(payload[28..32].try_into().unwrap());
            let shared_mem = u32::from_le_bytes(payload[32..36].try_into().unwrap());
            let stream = u64::from_le_bytes(payload[36..44].try_into().unwrap());
            let params = &payload[44..];
            match backend.launch_kernel(
                func,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_mem,
                stream,
                params,
            ) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemcpyHtoDAsync => {
            // [8B dst][8B stream][data...]
            if payload.len() < 16 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let stream = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            let data = &payload[16..];
            if data.is_empty() {
                return error_response(rid, CuResult::InvalidValue);
            }
            match backend.memcpy_htod_async(dst, data, stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemcpyDtoHAsync => {
            // [8B src][8B size][8B stream]
            if payload.len() < 24 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let src = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let size = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
            let stream = u64::from_le_bytes(payload[16..24].try_into().unwrap());
            match backend.memcpy_dtoh_async(src, size, stream) {
                Ok(buf) => success_with(rid, &buf),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemsetD8 => {
            // [8B dst][1B value][8B count]
            if payload.len() < 17 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let value = payload[8];
            let count = u64::from_le_bytes(payload[9..17].try_into().unwrap()) as usize;
            match backend.memset_d8(dst, value, count) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemsetD32 => {
            // [8B dst][4B value][8B count]
            if payload.len() < 20 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let value = u32::from_le_bytes(payload[8..12].try_into().unwrap());
            let count = u64::from_le_bytes(payload[12..20].try_into().unwrap()) as usize;
            match backend.memset_d32(dst, value, count) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemsetD8Async => {
            // [8B dst][1B value][8B count][8B stream]
            if payload.len() < 25 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let value = payload[8];
            let count = u64::from_le_bytes(payload[9..17].try_into().unwrap()) as usize;
            let stream = u64::from_le_bytes(payload[17..25].try_into().unwrap());
            match backend.memset_d8_async(dst, value, count, stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemsetD32Async => {
            // [8B dst][4B value][8B count][8B stream]
            if payload.len() < 28 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let value = u32::from_le_bytes(payload[8..12].try_into().unwrap());
            let count = u64::from_le_bytes(payload[12..20].try_into().unwrap()) as usize;
            let stream = u64::from_le_bytes(payload[20..28].try_into().unwrap());
            match backend.memset_d32_async(dst, value, count, stream) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemcpyDtoD => {
            // [8B dst][8B src][8B size] = 24 bytes minimum
            if payload.len() < 24 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let dst = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let src = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            let size = u64::from_le_bytes(payload[16..24].try_into().unwrap());
            match backend.memcpy_dtod(dst, src, size) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemAllocHost => {
            // [8B size]
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let size = u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
            match backend.mem_alloc_host(size) {
                Ok(ptr) => {
                    session.track_host_alloc(ptr);
                    success_with(rid, &ptr.to_le_bytes())
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::MemFreeHost => {
            // [8B ptr]
            if payload.len() < 8 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let ptr = u64::from_le_bytes(payload[..8].try_into().unwrap());
            match backend.mem_free_host(ptr) {
                Ok(()) => {
                    session.untrack_host_alloc(ptr);
                    result_only(rid, CuResult::Success)
                }
                Err(e) => error_response(rid, e),
            }
        }

        MessageType::StreamWaitEvent => {
            // [8B stream][8B event][4B flags] = 20 bytes minimum
            if payload.len() < 20 {
                return error_response(rid, CuResult::InvalidValue);
            }
            let stream = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let event = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            let flags = u32::from_le_bytes(payload[16..20].try_into().unwrap());
            match backend.stream_wait_event(stream, event, flags) {
                Ok(()) => result_only(rid, CuResult::Success),
                Err(e) => error_response(rid, e),
            }
        }

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
    use crate::session::ConnectionSession;

    /// Helper: build a request header.
    fn req(msg_type: MessageType, payload_len: u32) -> MessageHeader {
        MessageHeader::new_request(1, msg_type, payload_len)
    }

    /// Extract the CuResult from the first 4 bytes of a response payload.
    fn response_result(payload: &[u8]) -> CuResult {
        CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()))
    }

    /// Shorthand: dispatch a request with a fresh session (for tests that don't
    /// need to inspect session state across calls).
    fn dispatch(
        gpu: &StubGpuBackend,
        hdr: &MessageHeader,
        payload: &[u8],
    ) -> (MessageHeader, Vec<u8>) {
        let mut session = ConnectionSession::new();
        handle_request(gpu, hdr, payload, &mut session)
    }

    /// Dispatch using a shared mutable session (for multi-step tests).
    fn dispatch_with(
        gpu: &StubGpuBackend,
        hdr: &MessageHeader,
        payload: &[u8],
        session: &mut ConnectionSession,
    ) -> (MessageHeader, Vec<u8>) {
        handle_request(gpu, hdr, payload, session)
    }

    #[test]
    fn test_handshake() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::Handshake, 0);
        let (resp_hdr, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(resp_hdr.msg_type, MessageType::Response);
    }

    #[test]
    fn test_init() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::Init, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_driver_get_version() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::DriverGetVersion, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ver = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(ver, 12040);
    }

    #[test]
    fn test_device_get_count() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::DeviceGetCount, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::Success);
        let count = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_device_get_valid() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGet, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let dev = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(dev, 0);
    }

    #[test]
    fn test_device_get_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 5i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGet, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidDevice);
    }

    #[test]
    fn test_device_get_name() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGetName, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
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
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(val, 8);
    }

    #[test]
    fn test_device_total_mem() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceTotalMem, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let mem = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(mem as usize, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_device_get_uuid() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DeviceGetUuid, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..8], b"OLNK");
    }

    #[test]
    fn test_mem_alloc_free() {
        let gpu = StubGpuBackend::new();

        // Alloc
        let payload = 1024u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(ptr, 0);

        // Free
        let payload = ptr.to_le_bytes();
        let hdr = req(MessageType::MemFree, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_memcpy_roundtrip_via_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc 64 bytes.
        let alloc_payload = 64u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&alloc_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // HtoD: write 64 bytes of 0xAB.
        let mut htod_payload = ptr.to_le_bytes().to_vec();
        htod_payload.extend_from_slice(&vec![0xAB; 64]);
        let hdr = req(MessageType::MemcpyHtoD, htod_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&htod_payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // DtoH: read them back.
        let mut dtoh_payload = ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..], &vec![0xAB; 64]);
    }

    #[test]
    fn test_mem_get_info() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::MemGetInfo, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
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
        let (resp_hdr, _) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(resp_hdr.msg_type, MessageType::Response);
        assert_eq!(resp_hdr.request_id, 1);
    }

    #[test]
    fn test_short_payload_rejected() {
        let gpu = StubGpuBackend::new();
        // DeviceGetAttribute needs 8 bytes but we send 2.
        let hdr = req(MessageType::DeviceGetAttribute, 2);
        let (_, resp) = dispatch(&gpu, &hdr,&[0, 0]);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_unhandled_message_type() {
        let gpu = StubGpuBackend::new();
        // HandshakeAck is a server->client type, not handled as a request.
        let hdr = req(MessageType::HandshakeAck, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- Context operation tests -----

    #[test]
    fn test_ctx_create() {
        let gpu = StubGpuBackend::new();
        let mut payload = 0u32.to_le_bytes().to_vec(); // flags
        payload.extend_from_slice(&0i32.to_le_bytes()); // device
        let hdr = req(MessageType::CtxCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
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
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidDevice);
    }

    #[test]
    fn test_ctx_destroy() {
        let gpu = StubGpuBackend::new();

        // Create a context first.
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&create_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Destroy it.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_destroy_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xDEADu64.to_le_bytes();
        let hdr = req(MessageType::CtxDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidContext);
    }

    #[test]
    fn test_ctx_set_current() {
        let gpu = StubGpuBackend::new();

        // Create a context.
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&create_payload);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Set it as current.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxSetCurrent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_set_current_null() {
        let gpu = StubGpuBackend::new();
        // Set ctx to 0 (unset current).
        let payload = 0u64.to_le_bytes();
        let hdr = req(MessageType::CtxSetCurrent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_ctx_get_current() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Create a context (auto-sets as current in session).
        let mut create_payload = 0u32.to_le_bytes().to_vec();
        create_payload.extend_from_slice(&0i32.to_le_bytes());
        let hdr = req(MessageType::CtxCreate, create_payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &create_payload, &mut session);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get current (uses the same session).
        let hdr = req(MessageType::CtxGetCurrent, 0);
        let (_, resp) = dispatch_with(&gpu, &hdr, &[], &mut session);
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
        let (_, resp) = dispatch(&gpu, &hdr,&create_payload);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get device for this context.
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxGetDevice, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let device = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(device, 0);
    }

    #[test]
    fn test_ctx_synchronize() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::CtxSynchronize, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    // ----- Module handler tests -----

    #[test]
    fn test_module_load_data() {
        let gpu = StubGpuBackend::new();
        let payload = b"fake ptx data";
        let hdr = req(MessageType::ModuleLoadData, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_module_load_data_empty() {
        let gpu = StubGpuBackend::new();
        let hdr = req(MessageType::ModuleLoadData, 0);
        let (_, resp) = dispatch(&gpu, &hdr,&[]);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_load_data_ex() {
        let gpu = StubGpuBackend::new();
        // Build wire payload: 4B image_len + 4B num_options + options + image
        let image = b"fake ptx data";
        let options: Vec<(i32, u64)> = vec![(0, 32), (7, 4)]; // MAX_REGISTERS=32, OPT_LEVEL=4
        let mut payload = Vec::new();
        payload.extend_from_slice(&(image.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(options.len() as u32).to_le_bytes());
        for &(opt, val) in &options {
            payload.extend_from_slice(&opt.to_le_bytes());
            payload.extend_from_slice(&val.to_le_bytes());
        }
        payload.extend_from_slice(image);
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_module_load_data_ex_no_options() {
        let gpu = StubGpuBackend::new();
        let image = b"ptx code";
        let mut payload = Vec::new();
        payload.extend_from_slice(&(image.len() as u32).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes()); // 0 options
        payload.extend_from_slice(image);
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_module_load_data_ex_empty_image() {
        let gpu = StubGpuBackend::new();
        // image_len=0, num_options=0 => should fail
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_load_data_ex_truncated_header() {
        let gpu = StubGpuBackend::new();
        // Only 4 bytes, needs at least 8
        let payload = [1u8, 0, 0, 0];
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_load_data_ex_truncated_options() {
        let gpu = StubGpuBackend::new();
        // Claims 1 option but payload too short
        let mut payload = Vec::new();
        payload.extend_from_slice(&4u32.to_le_bytes());  // image_len=4
        payload.extend_from_slice(&1u32.to_le_bytes());  // num_options=1
        // Missing 12B of option data and 4B of image
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_load_data_ex_tracked_in_session() {
        let gpu = StubGpuBackend::new();
        let image = b"ptx";
        let mut payload = Vec::new();
        payload.extend_from_slice(&(image.len() as u32).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(image);
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let mut session = ConnectionSession::new();
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        // Module should be tracked — unload via session should work
        let unload_payload = handle.to_le_bytes();
        let hdr2 = req(MessageType::ModuleUnload, unload_payload.len() as u32);
        let (_, resp2) = dispatch_with(&gpu, &hdr2, &unload_payload, &mut session);
        assert_eq!(response_result(&resp2), CuResult::Success);
    }

    #[test]
    fn test_module_unload() {
        let gpu = StubGpuBackend::new();

        // Load first.
        let payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,payload);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Unload.
        let payload = module.to_le_bytes();
        let hdr = req(MessageType::ModuleUnload, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_module_unload_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::ModuleUnload, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_get_function() {
        let gpu = StubGpuBackend::new();

        // Load module.
        let load_payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, load_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,load_payload);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get function.
        let name = b"my_kernel";
        let mut payload = module.to_le_bytes().to_vec();
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name);
        let hdr = req(MessageType::ModuleGetFunction, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let func = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(func, 0);
    }

    #[test]
    fn test_module_get_function_invalid_module() {
        let gpu = StubGpuBackend::new();
        let name = b"kern";
        let mut payload = 0xBADu64.to_le_bytes().to_vec();
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name);
        let hdr = req(MessageType::ModuleGetFunction, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_get_global() {
        let gpu = StubGpuBackend::new();

        // Load module.
        let load_payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, load_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,load_payload);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get global.
        let name = b"my_global";
        let mut payload = module.to_le_bytes().to_vec();
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name);
        let hdr = req(MessageType::ModuleGetGlobal, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let dptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        let size = u64::from_le_bytes(resp[12..20].try_into().unwrap());
        assert_ne!(dptr, 0);
        assert_eq!(size, 256);
    }

    #[test]
    fn test_module_get_global_invalid_module() {
        let gpu = StubGpuBackend::new();
        let name = b"g";
        let mut payload = 0xBADu64.to_le_bytes().to_vec();
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name);
        let hdr = req(MessageType::ModuleGetGlobal, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- Stream handler tests -----

    #[test]
    fn test_stream_create() {
        let gpu = StubGpuBackend::new();
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_stream_destroy() {
        let gpu = StubGpuBackend::new();

        // Create.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Destroy.
        let payload = stream.to_le_bytes();
        let hdr = req(MessageType::StreamDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_stream_destroy_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::StreamDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_stream_synchronize() {
        let gpu = StubGpuBackend::new();

        // Create stream.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Synchronize.
        let payload = stream.to_le_bytes();
        let hdr = req(MessageType::StreamSynchronize, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_stream_query() {
        let gpu = StubGpuBackend::new();

        // Create stream.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Query.
        let payload = stream.to_le_bytes();
        let hdr = req(MessageType::StreamQuery, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_stream_query_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::StreamQuery, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- Event handler tests -----

    #[test]
    fn test_event_create() {
        let gpu = StubGpuBackend::new();
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let handle = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(handle, 0);
    }

    #[test]
    fn test_event_destroy() {
        let gpu = StubGpuBackend::new();

        // Create.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Destroy.
        let payload = event.to_le_bytes();
        let hdr = req(MessageType::EventDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_event_destroy_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::EventDestroy, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_event_record() {
        let gpu = StubGpuBackend::new();

        // Create event.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Record on default stream (0).
        let mut payload = event.to_le_bytes().to_vec();
        payload.extend_from_slice(&0u64.to_le_bytes());
        let hdr = req(MessageType::EventRecord, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_event_record_invalid() {
        let gpu = StubGpuBackend::new();
        let mut payload = 0xBADu64.to_le_bytes().to_vec();
        payload.extend_from_slice(&0u64.to_le_bytes());
        let hdr = req(MessageType::EventRecord, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_event_synchronize_handler() {
        let gpu = StubGpuBackend::new();

        // Create event.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Synchronize.
        let payload = event.to_le_bytes();
        let hdr = req(MessageType::EventSynchronize, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_event_elapsed_time() {
        let gpu = StubGpuBackend::new();

        // Create two events.
        let flags = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, flags.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&flags);
        let e1 = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let hdr = req(MessageType::EventCreate, flags.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&flags);
        let e2 = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Record both on default stream.
        let mut rec = e1.to_le_bytes().to_vec();
        rec.extend_from_slice(&0u64.to_le_bytes());
        let hdr = req(MessageType::EventRecord, rec.len() as u32);
        dispatch(&gpu, &hdr,&rec);

        let mut rec = e2.to_le_bytes().to_vec();
        rec.extend_from_slice(&0u64.to_le_bytes());
        let hdr = req(MessageType::EventRecord, rec.len() as u32);
        dispatch(&gpu, &hdr,&rec);

        // Get elapsed time.
        let mut payload = e1.to_le_bytes().to_vec();
        payload.extend_from_slice(&e2.to_le_bytes());
        let hdr = req(MessageType::EventElapsedTime, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ms = f32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert!(ms > 0.0);
    }

    #[test]
    fn test_event_elapsed_time_not_recorded() {
        let gpu = StubGpuBackend::new();

        // Create two events but don't record them.
        let flags = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, flags.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&flags);
        let e1 = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let hdr = req(MessageType::EventCreate, flags.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&flags);
        let e2 = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let mut payload = e1.to_le_bytes().to_vec();
        payload.extend_from_slice(&e2.to_le_bytes());
        let hdr = req(MessageType::EventElapsedTime, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::NotReady);
    }

    #[test]
    fn test_event_query() {
        let gpu = StubGpuBackend::new();

        // Create event.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Query.
        let payload = event.to_le_bytes();
        let hdr = req(MessageType::EventQuery, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_event_query_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::EventQuery, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- MemcpyDtoD handler tests -----

    #[test]
    fn test_memcpy_dtod_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc src and dst.
        let alloc_payload = 64u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let src = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let dst = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // HtoD: write to src.
        let mut htod_payload = src.to_le_bytes().to_vec();
        htod_payload.extend_from_slice(&vec![0xCD; 64]);
        let hdr = req(MessageType::MemcpyHtoD, htod_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &htod_payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // DtoD: copy src -> dst.
        let mut dtod_payload = dst.to_le_bytes().to_vec();
        dtod_payload.extend_from_slice(&src.to_le_bytes());
        dtod_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoD, dtod_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &dtod_payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // DtoH: read back from dst.
        let mut dtoh_payload = dst.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..], &vec![0xCD; 64]);
    }

    #[test]
    fn test_memcpy_dtod_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 24)
        let hdr = req(MessageType::MemcpyDtoD, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_memcpy_dtod_invalid_src() {
        let gpu = StubGpuBackend::new();
        let dst_ptr = {
            let alloc_payload = 64u64.to_le_bytes();
            let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
            let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
            u64::from_le_bytes(resp[4..12].try_into().unwrap())
        };
        let mut payload = dst_ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xBADu64.to_le_bytes());
        payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoD, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- MemAllocHost / MemFreeHost handler tests -----

    #[test]
    fn test_mem_alloc_host_handler() {
        let gpu = StubGpuBackend::new();
        let payload = 4096u64.to_le_bytes();
        let hdr = req(MessageType::MemAllocHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(ptr, 0);
    }

    #[test]
    fn test_mem_alloc_host_zero() {
        let gpu = StubGpuBackend::new();
        let payload = 0u64.to_le_bytes();
        let hdr = req(MessageType::MemAllocHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_mem_alloc_host_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 4]; // too short (needs 8)
        let hdr = req(MessageType::MemAllocHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_mem_free_host_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc host memory.
        let alloc_payload = 4096u64.to_le_bytes();
        let hdr = req(MessageType::MemAllocHost, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Free it.
        let payload = ptr.to_le_bytes();
        let hdr = req(MessageType::MemFreeHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_mem_free_host_invalid() {
        let gpu = StubGpuBackend::new();
        let payload = 0xBADu64.to_le_bytes();
        let hdr = req(MessageType::MemFreeHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_mem_free_host_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 4]; // too short (needs 8)
        let hdr = req(MessageType::MemFreeHost, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- StreamWaitEvent handler tests -----

    #[test]
    fn test_stream_wait_event_handler() {
        let gpu = StubGpuBackend::new();

        // Create stream.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Create event.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // StreamWaitEvent.
        let mut payload = stream.to_le_bytes().to_vec();
        payload.extend_from_slice(&event.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        let hdr = req(MessageType::StreamWaitEvent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_stream_wait_event_default_stream() {
        let gpu = StubGpuBackend::new();

        // Create event.
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // StreamWaitEvent with default stream (0).
        let mut payload = 0u64.to_le_bytes().to_vec();
        payload.extend_from_slice(&event.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        let hdr = req(MessageType::StreamWaitEvent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_stream_wait_event_invalid_event() {
        let gpu = StubGpuBackend::new();

        // Create stream.
        let s_payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, s_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &s_payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let mut payload = stream.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xBADu64.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        let hdr = req(MessageType::StreamWaitEvent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_stream_wait_event_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 20)
        let hdr = req(MessageType::StreamWaitEvent, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- LaunchKernel handler tests -----

    #[test]
    fn test_launch_kernel_handler() {
        let gpu = StubGpuBackend::new();

        // Load module.
        let mod_payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, mod_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,mod_payload);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Get function.
        let kern_name = b"kern";
        let mut func_payload = module.to_le_bytes().to_vec();
        func_payload.extend_from_slice(&(kern_name.len() as u32).to_le_bytes());
        func_payload.extend_from_slice(kern_name);
        let hdr = req(MessageType::ModuleGetFunction, func_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&func_payload);
        let func = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Launch kernel on default stream.
        let mut payload = func.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridX
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridY
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridZ
        payload.extend_from_slice(&256u32.to_le_bytes()); // blockX
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockY
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockZ
        payload.extend_from_slice(&0u32.to_le_bytes()); // shared_mem
        payload.extend_from_slice(&0u64.to_le_bytes()); // stream (default)
        payload.extend_from_slice(&0u32.to_le_bytes()); // num_params = 0 (matches real client wire format)
        let hdr = req(MessageType::LaunchKernel, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_launch_kernel_invalid_func() {
        let gpu = StubGpuBackend::new();

        let mut payload = 0xBADu64.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridX
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridY
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridZ
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockX
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockY
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockZ
        payload.extend_from_slice(&0u32.to_le_bytes()); // shared_mem
        payload.extend_from_slice(&0u64.to_le_bytes()); // stream
        let hdr = req(MessageType::LaunchKernel, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_launch_kernel_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short
        let hdr = req(MessageType::LaunchKernel, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr,&payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_launch_kernel_with_serialized_params() {
        let gpu = StubGpuBackend::new();

        // Load module + get function (same setup as test_launch_kernel_handler)
        let mod_payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, mod_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, mod_payload);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let kern_name = b"kern";
        let mut func_payload = module.to_le_bytes().to_vec();
        func_payload.extend_from_slice(&(kern_name.len() as u32).to_le_bytes());
        func_payload.extend_from_slice(kern_name);
        let hdr = req(MessageType::ModuleGetFunction, func_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &func_payload);
        let func = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Build launch payload with kernel params in the new serialized format
        let mut payload = func.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridX
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridY
        payload.extend_from_slice(&1u32.to_le_bytes()); // gridZ
        payload.extend_from_slice(&256u32.to_le_bytes()); // blockX
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockY
        payload.extend_from_slice(&1u32.to_le_bytes()); // blockZ
        payload.extend_from_slice(&0u32.to_le_bytes()); // shared_mem
        payload.extend_from_slice(&0u64.to_le_bytes()); // stream (default)
        // Serialized params: 2 params (u64 ptr + u32 count)
        payload.extend_from_slice(&2u32.to_le_bytes()); // num_params = 2
        payload.extend_from_slice(&8u32.to_le_bytes()); // param 0 size
        payload.extend_from_slice(&0x1234u64.to_le_bytes()); // param 0 data
        payload.extend_from_slice(&4u32.to_le_bytes()); // param 1 size
        payload.extend_from_slice(&1024u32.to_le_bytes()); // param 1 data
        let hdr = req(MessageType::LaunchKernel, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    // ----- MemcpyHtoDAsync / MemcpyDtoHAsync handler tests -----

    #[test]
    fn test_memcpy_htod_async_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc 64 bytes.
        let alloc_payload = 64u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // Create a stream.
        let stream_payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, stream_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &stream_payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // HtoDAsync: [8B dst][8B stream][data]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&stream.to_le_bytes());
        payload.extend_from_slice(&vec![0xEE; 64]);
        let hdr = req(MessageType::MemcpyHtoDAsync, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // Read back to verify.
        let mut dtoh_payload = ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..], &vec![0xEE; 64]);
    }

    #[test]
    fn test_memcpy_htod_async_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs >= 17)
        let hdr = req(MessageType::MemcpyHtoDAsync, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_memcpy_dtoh_async_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc and fill.
        let alloc_payload = 32u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let mut htod_payload = ptr.to_le_bytes().to_vec();
        htod_payload.extend_from_slice(&vec![0xBB; 32]);
        let hdr = req(MessageType::MemcpyHtoD, htod_payload.len() as u32);
        dispatch(&gpu, &hdr, &htod_payload);

        // Create a stream.
        let stream_payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, stream_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &stream_payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // DtoHAsync: [8B src][8B size][8B stream]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&32u64.to_le_bytes());
        payload.extend_from_slice(&stream.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoHAsync, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(&resp[4..], &vec![0xBB; 32]);
    }

    #[test]
    fn test_memcpy_dtoh_async_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 24)
        let hdr = req(MessageType::MemcpyDtoHAsync, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- MemsetD8 / MemsetD32 handler tests -----

    #[test]
    fn test_memset_d8_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc 64 bytes.
        let alloc_payload = 64u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // MemsetD8: [8B dst][1B value][8B count]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.push(0xAA); // value
        payload.extend_from_slice(&64u64.to_le_bytes()); // count
        let hdr = req(MessageType::MemsetD8, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // Read back to verify.
        let mut dtoh_payload = ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&64u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert!(resp[4..].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_memset_d8_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 17)
        let hdr = req(MessageType::MemsetD8, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_memset_d32_handler() {
        let gpu = StubGpuBackend::new();

        // Alloc 16 bytes (4 u32 elements).
        let alloc_payload = 16u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // MemsetD32: [8B dst][4B value][8B count]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // value
        payload.extend_from_slice(&4u64.to_le_bytes()); // count (4 elements)
        let hdr = req(MessageType::MemsetD32, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);

        // Read back to verify.
        let mut dtoh_payload = ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&16u64.to_le_bytes());
        let hdr = req(MessageType::MemcpyDtoH, dtoh_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &dtoh_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val_bytes = 0xDEADBEEFu32.to_le_bytes();
        for i in 0..4 {
            assert_eq!(&resp[4+i*4..4+i*4+4], &val_bytes);
        }
    }

    #[test]
    fn test_memset_d32_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 20)
        let hdr = req(MessageType::MemsetD32, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // ----- MemsetD8Async / MemsetD32Async handler tests -----

    #[test]
    fn test_memset_d8_async_handler() {
        let gpu = StubGpuBackend::new();

        let alloc_payload = 32u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let stream_payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, stream_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &stream_payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // MemsetD8Async: [8B dst][1B value][8B count][8B stream]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.push(0x55); // value
        payload.extend_from_slice(&32u64.to_le_bytes()); // count
        payload.extend_from_slice(&stream.to_le_bytes()); // stream
        let hdr = req(MessageType::MemsetD8Async, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_memset_d8_async_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 25)
        let hdr = req(MessageType::MemsetD8Async, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_memset_d32_async_handler() {
        let gpu = StubGpuBackend::new();

        let alloc_payload = 16u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, alloc_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &alloc_payload);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let stream_payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, stream_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &stream_payload);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        // MemsetD32Async: [8B dst][4B value][8B count][8B stream]
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xCAFEu32.to_le_bytes()); // value
        payload.extend_from_slice(&4u64.to_le_bytes()); // count
        payload.extend_from_slice(&stream.to_le_bytes()); // stream
        let hdr = req(MessageType::MemsetD32Async, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_memset_d32_async_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 10]; // too short (needs 28)
        let hdr = req(MessageType::MemsetD32Async, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // --- Session resource tracking tests ---

    #[test]
    fn test_handler_tracks_mem_alloc() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        let payload = 256u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.mem_alloc_count(), 1);
    }

    #[test]
    fn test_handler_untracks_mem_free() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Alloc
        let payload = 256u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.mem_alloc_count(), 1);

        // Free
        let payload = ptr.to_le_bytes();
        let hdr = req(MessageType::MemFree, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.mem_alloc_count(), 0);
    }

    #[test]
    fn test_handler_tracks_ctx_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Create context
        let mut payload = 0u32.to_le_bytes().to_vec(); // flags
        payload.extend_from_slice(&0i32.to_le_bytes()); // device
        let hdr = req(MessageType::CtxCreate, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.context_count(), 1);

        // Destroy context
        let payload = ctx.to_le_bytes();
        let hdr = req(MessageType::CtxDestroy, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.context_count(), 0);
    }

    #[test]
    fn test_handler_tracks_module_load_and_unload() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Load module
        let payload = b"fake_ptx_data";
        let hdr = req(MessageType::ModuleLoadData, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.module_count(), 1);

        // Unload module
        let payload = module.to_le_bytes();
        let hdr = req(MessageType::ModuleUnload, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.module_count(), 0);
    }

    #[test]
    fn test_handler_tracks_stream_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Create stream
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.stream_count(), 1);

        // Destroy stream
        let payload = stream.to_le_bytes();
        let hdr = req(MessageType::StreamDestroy, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.stream_count(), 0);
    }

    #[test]
    fn test_handler_tracks_event_create_and_destroy() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Create event
        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let event = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.event_count(), 1);

        // Destroy event
        let payload = event.to_le_bytes();
        let hdr = req(MessageType::EventDestroy, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.event_count(), 0);
    }

    #[test]
    fn test_handler_tracks_host_alloc_and_free() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Alloc host memory
        let payload = 512u64.to_le_bytes();
        let hdr = req(MessageType::MemAllocHost, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_eq!(session.host_alloc_count(), 1);

        // Free host memory
        let payload = ptr.to_le_bytes();
        let hdr = req(MessageType::MemFreeHost, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert_eq!(session.host_alloc_count(), 0);
    }

    #[test]
    fn test_handler_cleanup_frees_all_via_session() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Allocate various resources through the handler.
        let payload = 128u64.to_le_bytes();
        let hdr = req(MessageType::MemAlloc, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        let _ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::StreamCreate, payload.len() as u32);
        dispatch_with(&gpu, &hdr, &payload, &mut session);

        let payload = 0u32.to_le_bytes();
        let hdr = req(MessageType::EventCreate, payload.len() as u32);
        dispatch_with(&gpu, &hdr, &payload, &mut session);

        assert_eq!(session.total_tracked_resources(), 3);

        // Cleanup via session.
        let report = session.cleanup(&gpu);
        assert_eq!(report.succeeded, 3);
        assert_eq!(report.failed, 0);
        assert_eq!(session.total_tracked_resources(), 0);
    }

    // --- Primary context handler tests ---

    #[test]
    fn test_primary_ctx_retain() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert_ne!(ctx, 0);
        // Second retain returns same handle
        let (_, resp2) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp2), CuResult::Success);
        let ctx2 = u64::from_le_bytes(resp2[4..12].try_into().unwrap());
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn test_primary_ctx_retain_tracks_in_session() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        dispatch_with(&gpu, &hdr, &payload, &mut session);
        // Primary ctx should be tracked as a context
        assert_eq!(session.context_count(), 1);
    }

    #[test]
    fn test_primary_ctx_release() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        let payload = 0i32.to_le_bytes();
        // Retain first
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        dispatch_with(&gpu, &hdr, &payload, &mut session);
        // Release
        let hdr = req(MessageType::DevicePrimaryCtxRelease, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_primary_ctx_get_state() {
        let gpu = StubGpuBackend::new();
        let payload = 0i32.to_le_bytes();
        // Not active
        let hdr = req(MessageType::DevicePrimaryCtxGetState, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let flags = u32::from_le_bytes(resp[4..8].try_into().unwrap());
        let active = i32::from_le_bytes(resp[8..12].try_into().unwrap());
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
    }

    #[test]
    fn test_primary_ctx_set_flags() {
        let gpu = StubGpuBackend::new();
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&0i32.to_le_bytes());
        payload[4..8].copy_from_slice(&0x04u32.to_le_bytes());
        let hdr = req(MessageType::DevicePrimaryCtxSetFlags, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        // Verify flags via GetState
        let state_payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxGetState, state_payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &state_payload);
        let flags = u32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(flags, 0x04);
    }

    #[test]
    fn test_primary_ctx_set_flags_rejected_when_active() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        // Retain
        let retain_payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxRetain, retain_payload.len() as u32);
        dispatch_with(&gpu, &hdr, &retain_payload, &mut session);
        // Try to set flags while active
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&0i32.to_le_bytes());
        payload[4..8].copy_from_slice(&0x04u32.to_le_bytes());
        let hdr = req(MessageType::DevicePrimaryCtxSetFlags, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::PrimaryContextActive);
    }

    #[test]
    fn test_primary_ctx_reset() {
        let gpu = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        // Retain
        let payload = 0i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        let ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
        assert!(gpu.ctx_exists(ctx));
        // Reset
        let hdr = req(MessageType::DevicePrimaryCtxReset, payload.len() as u32);
        let (_, resp) = dispatch_with(&gpu, &hdr, &payload, &mut session);
        assert_eq!(response_result(&resp), CuResult::Success);
        assert!(!gpu.ctx_exists(ctx));
    }

    #[test]
    fn test_primary_ctx_retain_invalid_device() {
        let gpu = StubGpuBackend::new();
        let payload = 99i32.to_le_bytes();
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_ne!(response_result(&resp), CuResult::Success);
    }

    #[test]
    fn test_primary_ctx_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 2]; // too short
        let hdr = req(MessageType::DevicePrimaryCtxRetain, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // --- FuncGetAttribute tests ---

    /// Helper: load a module + get a function handle, returning the func u64.
    fn setup_function(gpu: &StubGpuBackend) -> u64 {
        let load_payload = b"ptx";
        let hdr = req(MessageType::ModuleLoadData, load_payload.len() as u32);
        let (_, resp) = dispatch(gpu, &hdr, load_payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let module = u64::from_le_bytes(resp[4..12].try_into().unwrap());

        let name = b"my_kernel";
        let mut payload = module.to_le_bytes().to_vec();
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name);
        let hdr = req(MessageType::ModuleGetFunction, payload.len() as u32);
        let (_, resp) = dispatch(gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        u64::from_le_bytes(resp[4..12].try_into().unwrap())
    }

    #[test]
    fn test_func_get_attribute_max_threads() {
        let gpu = StubGpuBackend::new();
        let func = setup_function(&gpu);
        // attrib 0 = MAX_THREADS_PER_BLOCK
        let mut payload = 0i32.to_le_bytes().to_vec();
        payload.extend_from_slice(&func.to_le_bytes());
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(val, 1024);
    }

    #[test]
    fn test_func_get_attribute_num_regs() {
        let gpu = StubGpuBackend::new();
        let func = setup_function(&gpu);
        // attrib 4 = NUM_REGS
        let mut payload = 4i32.to_le_bytes().to_vec();
        payload.extend_from_slice(&func.to_le_bytes());
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(val, 32);
    }

    #[test]
    fn test_func_get_attribute_ptx_version() {
        let gpu = StubGpuBackend::new();
        let func = setup_function(&gpu);
        // attrib 5 = PTX_VERSION
        let mut payload = 5i32.to_le_bytes().to_vec();
        payload.extend_from_slice(&func.to_le_bytes());
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::Success);
        let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
        assert_eq!(val, 80);
    }

    #[test]
    fn test_func_get_attribute_invalid_func() {
        let gpu = StubGpuBackend::new();
        // Use a function handle that does not exist
        let mut payload = 0i32.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xDEADu64.to_le_bytes());
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_func_get_attribute_invalid_attrib() {
        let gpu = StubGpuBackend::new();
        let func = setup_function(&gpu);
        // attrib 9999 = unknown
        let mut payload = 9999i32.to_le_bytes().to_vec();
        payload.extend_from_slice(&func.to_le_bytes());
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_func_get_attribute_short_payload() {
        let gpu = StubGpuBackend::new();
        let payload = [0u8; 6]; // need 12, sending 6
        let hdr = req(MessageType::FuncGetAttribute, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // --- ModuleLoadDataEx: num_options cap (fix #5) ---

    #[test]
    fn test_module_load_data_ex_num_options_too_large() {
        let gpu = StubGpuBackend::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&4u32.to_le_bytes());         // image_len=4
        payload.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // num_options=max u32
        // Don't need to provide full payload -- should reject before reading options
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    #[test]
    fn test_module_load_data_ex_num_options_at_cap() {
        let gpu = StubGpuBackend::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&4u32.to_le_bytes());     // image_len=4
        payload.extend_from_slice(&257u32.to_le_bytes());   // num_options=257 (over 256 cap)
        let hdr = req(MessageType::ModuleLoadDataEx, payload.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload);
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }

    // --- ModuleLoadDataEx: overflow in options_size (fix #6) ---

    #[test]
    fn test_module_load_data_ex_options_size_overflow() {
        let gpu = StubGpuBackend::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&4u32.to_le_bytes());     // image_len=4
        // 256 options is within cap but 256*12 = 3072 which is fine.
        // We need to test that checked arithmetic doesn't panic.
        // With the cap at 256, overflow via num_options*12 is impossible,
        // but we test that image_len + options_size doesn't overflow.
        // Use image_len = u32::MAX and num_options = 200.
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // image_len = max
        payload2.extend_from_slice(&200u32.to_le_bytes());          // num_options = 200
        let hdr = req(MessageType::ModuleLoadDataEx, payload2.len() as u32);
        let (_, resp) = dispatch(&gpu, &hdr, &payload2);
        // Should reject (not panic) due to payload length mismatch
        assert_eq!(response_result(&resp), CuResult::InvalidValue);
    }
}
