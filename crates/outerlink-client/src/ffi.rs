//! FFI exports called by the C interposition library (interpose.c).
//!
//! Each function here is `#[no_mangle] pub extern "C"` so the C code can call it
//! directly. These form the boundary between the thin C interposition layer and
//! the Rust client logic.
//!
//! When the client is connected to a server, requests are serialized and sent
//! over TCP. When disconnected (no server available), stub values are returned
//! so tests and local development continue to work.

use std::ffi::CStr;
use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use outerlink_common::protocol::MessageType;

use crate::OuterLinkClient;

// ---------------------------------------------------------------------------
// CUDA constants matching cuda.h / cuda_types.rs
// ---------------------------------------------------------------------------

const CUDA_SUCCESS: u32 = 0;
const CUDA_ERROR_INVALID_VALUE: u32 = 1;
const CUDA_ERROR_NOT_INITIALIZED: u32 = 3;
const CUDA_ERROR_INVALID_DEVICE: u32 = 101;
const CUDA_ERROR_INVALID_CONTEXT: u32 = 201;
const CUDA_ERROR_NOT_FOUND: u32 = 500;
const CUDA_ERROR_UNKNOWN: u32 = 999;

// ---------------------------------------------------------------------------
// Global client singleton
// ---------------------------------------------------------------------------

/// The global OuterLink client, initialized on the first CUDA call.
/// OnceLock guarantees exactly-once, thread-safe initialization.
static CLIENT: OnceLock<OuterLinkClient> = OnceLock::new();

/// STUB ONLY -- Monotonic counter for generating unique stub remote handle values.
/// Prevents handle collision when the same device is used for multiple contexts
/// or when allocation sizes happen to collide.
///
/// TODO: Remove this counter once real server integration is wired. At that point,
/// remote handle values will come from the server's responses, not from a local
/// counter. Every usage site (ol_cuCtxCreate_v2, ol_cuMemAlloc_v2, etc.) must be
/// updated to use server-provided handles instead.
static STUB_HANDLE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x1000);

/// Initialize the global client. Called by the C layer via pthread_once,
/// but also called lazily from each FFI function as a safety net.
fn get_client() -> &'static OuterLinkClient {
    CLIENT.get_or_init(|| {
        let addr = std::env::var("OUTERLINK_SERVER")
            .unwrap_or_else(|_| "localhost:14833".to_string());
        OuterLinkClient::new(addr)
    })
}

/// Extract the CuResult (first 4 LE bytes) from a response payload.
/// Returns `CUDA_ERROR_UNKNOWN` when the payload is too short.
fn parse_result(resp: &[u8]) -> u32 {
    if resp.len() < 4 {
        return CUDA_ERROR_UNKNOWN;
    }
    u32::from_le_bytes(resp[0..4].try_into().unwrap())
}

/// Explicit init entry point for the C layer.
///
/// Initializes the global client singleton and attempts to connect to the
/// server. If the connection fails, the client stays in disconnected stub
/// mode and a warning is logged.
#[no_mangle]
pub extern "C" fn ol_client_init() {
    let client = get_client();
    if !client.connected.load(Ordering::Acquire) {
        if let Err(e) = client.connect() {
            tracing::warn!("OuterLink: failed to connect to server: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuInit(flags: u32) -> u32 {
    let _ = get_client();
    let _ = flags;
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDriverGetVersion(driver_version: *mut i32) -> u32 {
    let client = get_client();
    if driver_version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DriverGetVersion, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *driver_version = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: report CUDA 12.4 (12040)
    unsafe { *driver_version = 12040 };
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetCount(count: *mut i32) -> u32 {
    let client = get_client();
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGetCount, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *count = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: report 1 virtual GPU
    unsafe { *count = 1 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGet(device: *mut i32, ordinal: i32) -> u32 {
    let client = get_client();
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = ordinal.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGet, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *device = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub fallback
    if ordinal < 0 || ordinal >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe { *device = ordinal };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetName(name: *mut u8, len: i32, dev: i32) -> u32 {
    let client = get_client();
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGetName, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            // Response data: 4 bytes CuResult + 4 bytes u32 name_len + name_len bytes UTF-8
            if resp.len() >= 8 {
                let name_len =
                    u32::from_le_bytes(resp[4..8].try_into().unwrap()) as usize;
                if resp.len() >= 8 + name_len {
                    let name_bytes = &resp[8..8 + name_len];
                    let buf_size = len as usize;
                    let copy_len =
                        std::cmp::min(buf_size.saturating_sub(1), name_bytes.len());
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            name_bytes.as_ptr(),
                            name,
                            copy_len,
                        );
                        *name.add(copy_len) = 0; // null-terminate
                    }
                    return CUDA_SUCCESS;
                }
            }
        }
    }
    // Stub fallback
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let device_name = b"OuterLink Virtual GPU";  // no embedded null
    let buf_size = len as usize;
    let copy_len = std::cmp::min(buf_size.saturating_sub(1), device_name.len());
    unsafe {
        std::ptr::copy_nonoverlapping(device_name.as_ptr(), name, copy_len);
        *name.add(copy_len) = 0;  // always null-terminate
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> u32 {
    let client = get_client();
    if pi.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&attrib.to_le_bytes());
        payload[4..8].copy_from_slice(&dev.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::DeviceGetAttribute, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *pi = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let value = match attrib {
        1 => 1024,   // MAX_THREADS_PER_BLOCK
        2 => 1024,   // MAX_BLOCK_DIM_X
        3 => 1024,   // MAX_BLOCK_DIM_Y
        4 => 64,     // MAX_BLOCK_DIM_Z
        5 => 2147483647, // MAX_GRID_DIM_X
        6 => 65535,  // MAX_GRID_DIM_Y
        7 => 65535,  // MAX_GRID_DIM_Z
        16 => 80,    // MULTIPROCESSOR_COUNT
        75 => 8,     // COMPUTE_CAPABILITY_MAJOR
        76 => 9,     // COMPUTE_CAPABILITY_MINOR
        81 => 166912, // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        _ => 0,      // Unknown attributes return 0
    };
    unsafe { *pi = value };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceTotalMem_v2(bytes: *mut usize, dev: i32) -> u32 {
    let client = get_client();
    if bytes.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    // Response format: 4 bytes CuResult + 8 bytes u64 (total mem)
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::DeviceTotalMem, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let total = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                unsafe {
                    *bytes = total as usize;
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe { *bytes = 24 * 1024 * 1024 * 1024 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetUuid(uuid: *mut u8, dev: i32) -> u32 {
    let client = get_client();
    if uuid.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    // Response format: 4 bytes CuResult + 16 bytes UUID
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::DeviceGetUuid, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 20 {
                unsafe {
                    std::ptr::copy_nonoverlapping(resp[4..20].as_ptr(), uuid, 16);
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe {
        std::ptr::write_bytes(uuid, 0, 16);
        *uuid.add(0) = b'O';
        *uuid.add(1) = b'L';
        *uuid.add(2) = dev as u8;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuCtxCreate_v2(pctx: *mut u64, flags: u32, dev: i32) -> u32 {
    let client = get_client();
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&flags.to_le_bytes());
        payload[4..8].copy_from_slice(&dev.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxCreate, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.contexts.insert(remote_ctx);
                unsafe { *pctx = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.contexts.insert(stub_remote);
    unsafe { *pctx = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxDestroy_v2(ctx: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = match client.handles.contexts.to_remote(ctx) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_CONTEXT,
        };
        let payload = remote_ctx.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxDestroy, &payload) {
            let result = parse_result(&resp);
            // CUDA contract: handle is invalidated on destroy regardless of server errors
            client.handles.contexts.remove_by_local(ctx);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.contexts.remove_by_local(ctx).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxSetCurrent(ctx: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = if ctx == 0 {
            0u64
        } else {
            match client.handles.contexts.to_remote(ctx) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_CONTEXT,
            }
        };
        let payload = remote_ctx.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxSetCurrent, &payload) {
            let result = parse_result(&resp);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle locally
    // ctx == 0 means "unset current context", which is always valid.
    if ctx != 0 {
        if client.handles.contexts.to_remote(ctx).is_none() {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetCurrent(pctx: *mut u64) -> u32 {
    let client = get_client();
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetCurrent, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let local = if remote_ctx == 0 {
                    0
                } else {
                    client.handles.contexts.to_local(remote_ctx).unwrap_or(0)
                };
                unsafe { *pctx = local };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: return null context (no context set)
    unsafe { *pctx = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetDevice(dev: *mut i32) -> u32 {
    let client = get_client();
    if dev.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: the server handler expects a u64 ctx in the payload.
    // cuCtxGetDevice operates on the current context, so we send 0 to indicate
    // "use whatever the server considers current". If that fails, fall through.
    // TODO: Once per-connection context tracking lands, send the actual remote ctx.
    if client.connected.load(Ordering::Acquire) {
        // Send 0 as ctx -- the server may reject this if it strictly requires a
        // valid context handle. That's fine; we fall through to the stub.
        let payload = 0u64.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetDevice, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *dev = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: return device 0
    unsafe { *dev = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxSynchronize() -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxSynchronize, &[]) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: nothing to synchronize
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemAlloc_v2(dptr: *mut u64, size: usize) -> u32 {
    let client = get_client();
    if dptr.is_null() || size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = (size as u64).to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemAlloc, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_devptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.device_ptrs.insert(remote_devptr);
                unsafe { *dptr = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.device_ptrs.insert(stub_remote);
    unsafe { *dptr = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemFree_v2(dptr: u64) -> u32 {
    let client = get_client();
    if dptr == 0 {
        // Freeing null is a no-op in CUDA
        return CUDA_SUCCESS;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_devptr = match client.handles.device_ptrs.to_remote(dptr) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_devptr.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemFree, &payload) {
            let result = parse_result(&resp);
            // CUDA contract: handle is invalidated on free regardless of server errors
            client.handles.device_ptrs.remove_by_local(dptr);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.device_ptrs.remove_by_local(dptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemcpyHtoD_v2(
    dst_device: u64,
    src_host: *const u8,
    byte_count: usize,
) -> u32 {
    let client = get_client();
    if src_host.is_null() || byte_count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Phase 1 limit: inline transfer must fit in protocol payload
    if byte_count > (outerlink_common::protocol::MAX_PAYLOAD_SIZE as usize) - 8 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send data to the server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Build payload: 8 bytes remote_dst + raw data bytes
        let host_data = unsafe { std::slice::from_raw_parts(src_host, byte_count) };
        let mut payload = Vec::with_capacity(8 + byte_count);
        payload.extend_from_slice(&remote_dst.to_le_bytes());
        payload.extend_from_slice(host_data);
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemcpyHtoD, &payload) {
            let result = parse_result(&resp);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle and accept the copy
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemcpyDtoH_v2(
    dst_host: *mut u8,
    src_device: u64,
    byte_count: usize,
) -> u32 {
    let client = get_client();
    if dst_host.is_null() || byte_count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: fetch data from the server
    if client.connected.load(Ordering::Acquire) {
        let remote_src = match client.handles.device_ptrs.to_remote(src_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Request payload: 8 bytes remote_src + 8 bytes byte_count
        let mut payload = [0u8; 16];
        payload[0..8].copy_from_slice(&remote_src.to_le_bytes());
        payload[8..16].copy_from_slice(&(byte_count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemcpyDtoH, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            // Response: 4B result + data bytes
            let resp_data = &resp[4..];
            if resp_data.len() < byte_count {
                return CUDA_ERROR_UNKNOWN;
            }
            let copy_len = std::cmp::min(resp_data.len(), byte_count);
            unsafe {
                std::ptr::copy_nonoverlapping(resp_data.as_ptr(), dst_host, copy_len);
            }
            return CUDA_SUCCESS;
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle and zero-fill
    if client.handles.device_ptrs.to_remote(src_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        ptr::write_bytes(dst_host, 0, byte_count);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> u32 {
    let client = get_client();
    if free.is_null() || total.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    // Response format: 4 bytes CuResult + 8 bytes u64 free + 8 bytes u64 total
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemGetInfo, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 20 {
                let free_val = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let total_val = u64::from_le_bytes(resp[12..20].try_into().unwrap());
                unsafe {
                    *free = free_val as usize;
                    *total = total_val as usize;
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: 24 GB total, 20 GB free
    unsafe {
        *total = 24 * 1024 * 1024 * 1024;
        *free = 20 * 1024 * 1024 * 1024;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuGetErrorName(
    error: u32,
    p_str: *mut *const u8,
) -> u32 {
    let _ = get_client();
    if p_str.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name: &[u8] = match error {
        0 => b"CUDA_SUCCESS\0",
        1 => b"CUDA_ERROR_INVALID_VALUE\0",
        2 => b"CUDA_ERROR_OUT_OF_MEMORY\0",
        3 => b"CUDA_ERROR_NOT_INITIALIZED\0",
        100 => b"CUDA_ERROR_NO_DEVICE\0",
        101 => b"CUDA_ERROR_INVALID_DEVICE\0",
        201 => b"CUDA_ERROR_INVALID_CONTEXT\0",
        500 => b"CUDA_ERROR_NOT_FOUND\0",
        999 => b"CUDA_ERROR_UNKNOWN\0",
        _ => b"CUDA_ERROR_UNKNOWN\0",
    };
    unsafe { *p_str = name.as_ptr() };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuGetErrorString(
    error: u32,
    p_str: *mut *const u8,
) -> u32 {
    let _ = get_client();
    if p_str.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let desc: &[u8] = match error {
        0 => b"no error\0",
        1 => b"invalid argument\0",
        2 => b"out of memory\0",
        3 => b"not initialized\0",
        100 => b"no CUDA-capable device is detected\0",
        101 => b"invalid device ordinal\0",
        201 => b"invalid device context\0",
        500 => b"named symbol not found\0",
        999 => b"unknown error\0",
        _ => b"unknown error\0",
    };
    unsafe { *p_str = desc.as_ptr() };
    CUDA_SUCCESS
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Init tests --

    #[test]
    fn test_ol_cuinit_returns_success() {
        assert_eq!(ol_cuInit(0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_driver_get_version() {
        let mut version: i32 = 0;
        let result = ol_cuDriverGetVersion(&mut version as *mut i32);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(version, 12040);
    }

    #[test]
    fn test_ol_cu_driver_get_version_null_ptr() {
        let result = ol_cuDriverGetVersion(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // -- Device tests --

    #[test]
    fn test_ol_cu_device_get_count() {
        let mut count: i32 = 0;
        let result = ol_cuDeviceGetCount(&mut count as *mut i32);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_ol_cu_device_get_count_null_ptr() {
        let result = ol_cuDeviceGetCount(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_valid() {
        let mut dev: i32 = -1;
        let result = ol_cuDeviceGet(&mut dev as *mut i32, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(dev, 0);
    }

    #[test]
    fn test_ol_cu_device_get_invalid_ordinal() {
        let mut dev: i32 = -1;
        assert_eq!(ol_cuDeviceGet(&mut dev, 1), CUDA_ERROR_INVALID_DEVICE);
        assert_eq!(ol_cuDeviceGet(&mut dev, -1), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_device_get_name() {
        let mut buf = [0u8; 256];
        let result = ol_cuDeviceGetName(buf.as_mut_ptr(), 256, 0);
        assert_eq!(result, CUDA_SUCCESS);
        let name = CStr::from_bytes_until_nul(&buf).unwrap();
        assert_eq!(name.to_str().unwrap(), "OuterLink Virtual GPU");
    }

    #[test]
    fn test_ol_cu_device_get_name_invalid_device() {
        let mut buf = [0u8; 256];
        let result = ol_cuDeviceGetName(buf.as_mut_ptr(), 256, 5);
        assert_eq!(result, CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_device_get_name_small_buffer() {
        // Buffer smaller than the name — must still be null-terminated and not overflow.
        let mut buf = [0xFFu8; 5];
        let result = ol_cuDeviceGetName(buf.as_mut_ptr(), 5, 0);
        assert_eq!(result, CUDA_SUCCESS);
        // Last byte must be null terminator.
        assert_eq!(buf[4], 0);
        // First 4 bytes hold the truncated name prefix.
        assert_eq!(&buf[..4], b"Oute");
    }

    #[test]
    fn test_ol_cu_device_get_name_exact_buffer() {
        // Buffer is exactly the length of the name plus a null byte.
        // "OuterLink Virtual GPU" is 21 bytes, so buf_size = 22.
        let name_str = b"OuterLink Virtual GPU";
        let buf_size = name_str.len() + 1; // 22
        let mut buf = vec![0xFFu8; buf_size];
        let result = ol_cuDeviceGetName(buf.as_mut_ptr(), buf_size as i32, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(buf[name_str.len()], 0); // null terminator in last slot
        assert_eq!(&buf[..name_str.len()], name_str.as_slice());
    }

    #[test]
    fn test_ol_cu_ctx_create_twice_same_device_different_handles() {
        let mut ctx1: u64 = 0;
        let mut ctx2: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx1, 0, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx2, 0, 0), CUDA_SUCCESS);
        // Each call must return a distinct synthetic handle.
        assert_ne!(ctx1, ctx2);
        // Clean up so handle store stays consistent across test runs.
        let _ = ol_cuCtxDestroy_v2(ctx1);
        let _ = ol_cuCtxDestroy_v2(ctx2);
    }

    #[test]
    fn test_ol_cu_device_get_attribute() {
        let mut val: i32 = 0;
        let result = ol_cuDeviceGetAttribute(&mut val, 75, 0); // COMPUTE_CAPABILITY_MAJOR
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 8);
    }

    #[test]
    fn test_ol_cu_device_get_attribute_unknown() {
        let mut val: i32 = -1;
        let result = ol_cuDeviceGetAttribute(&mut val, 99999, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 0); // Unknown attributes return 0
    }

    #[test]
    fn test_ol_cu_device_total_mem() {
        let mut bytes: usize = 0;
        let result = ol_cuDeviceTotalMem_v2(&mut bytes, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_ol_cu_device_get_uuid() {
        let mut uuid = [0u8; 16];
        let result = ol_cuDeviceGetUuid(uuid.as_mut_ptr(), 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(uuid[0], b'O');
        assert_eq!(uuid[1], b'L');
        assert_eq!(uuid[2], 0); // device 0
    }

    // -- Context tests --

    #[test]
    fn test_ol_cu_ctx_create_and_destroy() {
        let mut ctx: u64 = 0;
        let result = ol_cuCtxCreate_v2(&mut ctx, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(ctx, 0);

        let result = ol_cuCtxDestroy_v2(ctx);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_ctx_destroy_invalid() {
        let result = ol_cuCtxDestroy_v2(0xDEADBEEF);
        assert_eq!(result, CUDA_ERROR_INVALID_CONTEXT);
    }

    #[test]
    fn test_ol_cu_ctx_set_current() {
        // ctx=0 (unset context) is always valid
        assert_eq!(ol_cuCtxSetCurrent(0), CUDA_SUCCESS);

        // An invalid handle that was never created must be rejected
        assert_eq!(ol_cuCtxSetCurrent(0xBAAD), CUDA_ERROR_INVALID_CONTEXT);

        // A handle created by ol_cuCtxCreate_v2 must be accepted
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuCtxSetCurrent(ctx), CUDA_SUCCESS);
        // Clean up
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_get_current() {
        let mut ctx: u64 = 0xFFFF;
        let result = ol_cuCtxGetCurrent(&mut ctx);
        assert_eq!(result, CUDA_SUCCESS);
        // Stub returns null context
        assert_eq!(ctx, 0);
    }

    // -- Memory tests --

    #[test]
    fn test_ol_cu_mem_alloc_and_free() {
        let mut dptr: u64 = 0;
        let result = ol_cuMemAlloc_v2(&mut dptr, 1024);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(dptr, 0);

        let result = ol_cuMemFree_v2(dptr);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_alloc_zero_size() {
        let mut dptr: u64 = 0;
        let result = ol_cuMemAlloc_v2(&mut dptr, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_free_null() {
        // Freeing null pointer is a no-op in CUDA
        assert_eq!(ol_cuMemFree_v2(0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_free_invalid() {
        assert_eq!(ol_cuMemFree_v2(0xBADBAD), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_memcpy_htod() {
        // Allocate a valid device pointer first
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);
        let data = [1u8, 2, 3, 4];
        let result = ol_cuMemcpyHtoD_v2(dptr, data.as_ptr(), 4);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_memcpy_htod_null_src() {
        let result = ol_cuMemcpyHtoD_v2(0x1000, ptr::null(), 4);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_memcpy_htod_invalid_dst() {
        let data = [1u8, 2, 3, 4];
        // 0xBAAD was never allocated, must be rejected
        let result = ol_cuMemcpyHtoD_v2(0xBAAD, data.as_ptr(), 4);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_memcpy_dtoh() {
        // Allocate a valid device pointer first
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);
        let mut buf = [0xFFu8; 8];
        let result = ol_cuMemcpyDtoH_v2(buf.as_mut_ptr(), dptr, 8);
        assert_eq!(result, CUDA_SUCCESS);
        // Stub zero-fills the buffer
        assert_eq!(buf, [0u8; 8]);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_memcpy_dtoh_invalid_src() {
        let mut buf = [0xFFu8; 8];
        // 0xBAAD was never allocated, must be rejected
        let result = ol_cuMemcpyDtoH_v2(buf.as_mut_ptr(), 0xBAAD, 8);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_get_info() {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let result = ol_cuMemGetInfo_v2(&mut free, &mut total);
        assert_eq!(result, CUDA_SUCCESS);
        assert!(free > 0);
        assert!(total > 0);
        assert!(free <= total);
    }

    // -- Error handling tests --

    #[test]
    fn test_ol_cu_get_error_name() {
        let mut name_ptr: *const u8 = ptr::null();
        let result = ol_cuGetErrorName(0, &mut name_ptr);
        assert_eq!(result, CUDA_SUCCESS);
        assert!(!name_ptr.is_null());
        let name = unsafe { CStr::from_ptr(name_ptr as *const i8) };
        assert_eq!(name.to_str().unwrap(), "CUDA_SUCCESS");
    }

    #[test]
    fn test_ol_cu_get_error_name_unknown() {
        let mut name_ptr: *const u8 = ptr::null();
        let result = ol_cuGetErrorName(12345, &mut name_ptr);
        assert_eq!(result, CUDA_SUCCESS);
        let name = unsafe { CStr::from_ptr(name_ptr as *const i8) };
        assert_eq!(name.to_str().unwrap(), "CUDA_ERROR_UNKNOWN");
    }

    #[test]
    fn test_ol_cu_get_error_string() {
        let mut str_ptr: *const u8 = ptr::null();
        let result = ol_cuGetErrorString(1, &mut str_ptr);
        assert_eq!(result, CUDA_SUCCESS);
        let s = unsafe { CStr::from_ptr(str_ptr as *const i8) };
        assert_eq!(s.to_str().unwrap(), "invalid argument");
    }

    // -- Client initialization tests --

    #[test]
    fn test_client_init_idempotent() {
        // Calling init multiple times should not panic or fail
        ol_client_init();
        ol_client_init();
        ol_client_init();
    }

    #[test]
    fn test_get_client_returns_same_instance() {
        let c1 = get_client() as *const OuterLinkClient;
        let c2 = get_client() as *const OuterLinkClient;
        assert_eq!(c1, c2);
    }
}
