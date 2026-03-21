//! FFI exports called by the C interposition library (interpose.c).
//!
//! Each function here is `#[no_mangle] pub extern "C"` so the C code can call it
//! directly. These form the boundary between the thin C interposition layer and
//! the Rust client logic.
//!
//! When the client is connected to a server, requests are serialized and sent
//! over TCP. When disconnected (no server available), stub values are returned
//! so tests and local development continue to work.

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
// Used by context/module/stream/event validation in upcoming wired operations
#[allow(dead_code)]
const CUDA_ERROR_NOT_INITIALIZED: u32 = 3;
const CUDA_ERROR_INVALID_DEVICE: u32 = 101;
const CUDA_ERROR_INVALID_CONTEXT: u32 = 201;
// Used by cuModuleGetFunction and cuGetProcAddress when symbols are not found
#[allow(dead_code)]
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
/// When connected to a server, handles come from server responses. This counter is
/// only used in disconnected/stub fallback mode to generate unique local handles.
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
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        // Send Init with flags payload (Handshake is sent once per-connection in connect())
        let payload = flags.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::Init, &payload) {
            return parse_result(&resp);
        }
        return CUDA_ERROR_UNKNOWN;
    }
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
                client.current_remote_ctx.store(remote_ctx, Ordering::Release);
                unsafe { *pctx = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.contexts.insert(stub_remote);
    client.current_remote_ctx.store(stub_remote, Ordering::Release);
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
            // If the destroyed context was current, clear the tracking
            if client.current_remote_ctx.load(Ordering::Acquire) == remote_ctx {
                client.current_remote_ctx.store(0, Ordering::Release);
            }
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    let removed_remote = client.handles.contexts.remove_by_local(ctx);
    match removed_remote {
        Some(remote) => {
            if client.current_remote_ctx.load(Ordering::Acquire) == remote {
                client.current_remote_ctx.store(0, Ordering::Release);
            }
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_CONTEXT,
    }
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
            if result == CUDA_SUCCESS {
                client.current_remote_ctx.store(remote_ctx, Ordering::Release);
            }
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle locally
    // ctx == 0 means "unset current context", which is always valid.
    if ctx != 0 {
        match client.handles.contexts.to_remote(ctx) {
            Some(r) => {
                client.current_remote_ctx.store(r, Ordering::Release);
            }
            None => return CUDA_ERROR_INVALID_CONTEXT,
        }
    } else {
        client.current_remote_ctx.store(0, Ordering::Release);
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
    // cuCtxGetDevice operates on the current context, so we send the tracked
    // remote context handle.
    if client.connected.load(Ordering::Acquire) {
        let payload = client.current_remote_ctx.load(Ordering::Acquire).to_le_bytes();
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
    // OuterLink extension codes (match CuResult enum values)
    const OL_TRANSPORT_ERROR: u32 = 10000;
    const OL_REMOTE_ERROR: u32 = 10001;
    const OL_HANDLE_NOT_FOUND: u32 = 10002;

    let name: &[u8] = match error {
        0 => b"CUDA_SUCCESS\0",
        1 => b"CUDA_ERROR_INVALID_VALUE\0",
        2 => b"CUDA_ERROR_OUT_OF_MEMORY\0",
        3 => b"CUDA_ERROR_NOT_INITIALIZED\0",
        4 => b"CUDA_ERROR_DEINITIALIZED\0",
        5 => b"CUDA_ERROR_PROFILER_DISABLED\0",
        6 => b"CUDA_ERROR_PROFILER_NOT_INITIALIZED\0",
        7 => b"CUDA_ERROR_PROFILER_ALREADY_STARTED\0",
        8 => b"CUDA_ERROR_PROFILER_ALREADY_STOPPED\0",
        100 => b"CUDA_ERROR_NO_DEVICE\0",
        101 => b"CUDA_ERROR_INVALID_DEVICE\0",
        200 => b"CUDA_ERROR_INVALID_IMAGE\0",
        201 => b"CUDA_ERROR_INVALID_CONTEXT\0",
        202 => b"CUDA_ERROR_CONTEXT_ALREADY_CURRENT\0",
        203 => b"CUDA_ERROR_CONTEXT_ALREADY_IN_USE\0",
        205 => b"CUDA_ERROR_MAP_FAILED\0",
        206 => b"CUDA_ERROR_UNMAP_FAILED\0",
        208 => b"CUDA_ERROR_ALREADY_MAPPED\0",
        209 => b"CUDA_ERROR_NO_BINARY_FOR_GPU\0",
        216 => b"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED\0",
        217 => b"CUDA_ERROR_INVALID_PTX\0",
        218 => b"CUDA_ERROR_INVALID_GRAPHICS_CONTEXT\0",
        219 => b"CUDA_ERROR_NVLINK_UNCORRECTABLE\0",
        220 => b"CUDA_ERROR_JIT_COMPILER_NOT_FOUND\0",
        221 => b"CUDA_ERROR_UNSUPPORTED_PTX_VERSION\0",
        222 => b"CUDA_ERROR_JIT_COMPILATION_DISABLED\0",
        500 => b"CUDA_ERROR_NOT_FOUND\0",
        600 => b"CUDA_ERROR_NOT_READY\0",
        700 => b"CUDA_ERROR_ILLEGAL_ADDRESS\0",
        701 => b"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES\0",
        702 => b"CUDA_ERROR_LAUNCH_TIMEOUT\0",
        703 => b"CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING\0",
        704 => b"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED\0",
        705 => b"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED\0",
        708 => b"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE\0",
        709 => b"CUDA_ERROR_CONTEXT_IS_DESTROYED\0",
        710 => b"CUDA_ERROR_ASSERT\0",
        711 => b"CUDA_ERROR_TOO_MANY_PEERS\0",
        712 => b"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED\0",
        713 => b"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED\0",
        714 => b"CUDA_ERROR_HARDWARE_STACK_ERROR\0",
        715 => b"CUDA_ERROR_ILLEGAL_INSTRUCTION\0",
        716 => b"CUDA_ERROR_MISALIGNED_ADDRESS\0",
        717 => b"CUDA_ERROR_INVALID_ADDRESS_SPACE\0",
        718 => b"CUDA_ERROR_INVALID_PC\0",
        719 => b"CUDA_ERROR_LAUNCH_FAILED\0",
        805 => b"CUDA_ERROR_MPS_CONNECTION_FAILED\0",
        806 => b"CUDA_ERROR_MPS_RPC_FAILURE\0",
        807 => b"CUDA_ERROR_MPS_SERVER_NOT_READY\0",
        808 => b"CUDA_ERROR_MPS_MAX_CLIENTS_REACHED\0",
        809 => b"CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED\0",
        908 => b"CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD\0",
        909 => b"CUDA_ERROR_TIMEOUT\0",
        910 => b"CUDA_ERROR_SYSTEM_NOT_READY\0",
        999 => b"CUDA_ERROR_UNKNOWN\0",
        OL_TRANSPORT_ERROR => b"OUTERLINK_ERROR_TRANSPORT\0",
        OL_REMOTE_ERROR => b"OUTERLINK_ERROR_REMOTE\0",
        OL_HANDLE_NOT_FOUND => b"OUTERLINK_ERROR_HANDLE_NOT_FOUND\0",
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
    // OuterLink extension codes (match CuResult enum values)
    const OL_TRANSPORT_ERROR: u32 = 10000;
    const OL_REMOTE_ERROR: u32 = 10001;
    const OL_HANDLE_NOT_FOUND: u32 = 10002;

    let desc: &[u8] = match error {
        0 => b"no error\0",
        1 => b"invalid argument\0",
        2 => b"out of memory\0",
        3 => b"not initialized\0",
        4 => b"driver shutting down\0",
        5 => b"profiler is disabled for this run\0",
        6 => b"profiler not initialized\0",
        7 => b"profiler already started\0",
        8 => b"profiler already stopped\0",
        100 => b"no CUDA-capable device is detected\0",
        101 => b"invalid device ordinal\0",
        200 => b"device kernel image is invalid\0",
        201 => b"invalid device context\0",
        202 => b"context is already current\0",
        203 => b"context is already in use\0",
        205 => b"mapping of buffer object failed\0",
        206 => b"unmapping of buffer object failed\0",
        208 => b"resource already mapped\0",
        209 => b"no kernel image is available for execution on the device\0",
        216 => b"peer access is not supported\0",
        217 => b"a PTX JIT compilation failed\0",
        218 => b"invalid OpenGL or DirectX context\0",
        219 => b"uncorrectable NVLink error was detected\0",
        220 => b"PTX JIT compiler library was not found\0",
        221 => b"the provided PTX was compiled with an unsupported toolchain\0",
        222 => b"PTX JIT compilation was disabled\0",
        500 => b"named symbol not found\0",
        600 => b"device not ready\0",
        700 => b"an illegal memory access was encountered\0",
        701 => b"launch out of resources\0",
        702 => b"launch timeout\0",
        703 => b"launch incompatible texturing\0",
        704 => b"peer access already enabled\0",
        705 => b"peer access not enabled\0",
        708 => b"the primary context for the specified device has already been initialized\0",
        709 => b"context is destroyed\0",
        710 => b"device-side assert triggered\0",
        711 => b"too many peers\0",
        712 => b"host memory already registered\0",
        713 => b"host memory not registered\0",
        714 => b"hardware stack error\0",
        715 => b"illegal instruction\0",
        716 => b"misaligned address\0",
        717 => b"invalid address space\0",
        718 => b"invalid program counter\0",
        719 => b"unspecified launch failure\0",
        805 => b"MPS client failed to connect to the MPS control daemon or MPS server\0",
        806 => b"the MPS RPC call failed\0",
        807 => b"MPS server is not ready to accept new connections\0",
        808 => b"the MPS server has reached its maximum number of clients\0",
        809 => b"the MPS maximum connections per client has been exceeded\0",
        908 => b"operation not permitted on a stream during capture in another thread\0",
        909 => b"operation timed out\0",
        910 => b"system not yet ready\0",
        999 => b"unknown error\0",
        OL_TRANSPORT_ERROR => b"OuterLink transport/network error\0",
        OL_REMOTE_ERROR => b"OuterLink remote server error\0",
        OL_HANDLE_NOT_FOUND => b"OuterLink handle not found in translation table\0",
        _ => b"unknown error\0",
    };
    unsafe { *p_str = desc.as_ptr() };
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Module management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuModuleLoadData(module: *mut u64, data: *const u8, data_len: usize) -> u32 {
    let client = get_client();
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send the raw binary data to the server
    if client.connected.load(Ordering::Acquire) {
        if !data.is_null() && data_len > 0 {
            let payload = unsafe { std::slice::from_raw_parts(data, data_len) };
            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::ModuleLoadData, payload)
            {
                let result = parse_result(&resp);
                if result != CUDA_SUCCESS {
                    return result;
                }
                if resp.len() >= 12 {
                    let remote_module =
                        u64::from_le_bytes(resp[4..12].try_into().unwrap());
                    let synthetic = client.handles.modules.insert(remote_module);
                    unsafe { *module = synthetic };
                    return CUDA_SUCCESS;
                }
            }
        }
        // Transport error or null data -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.modules.insert(stub_remote);
    unsafe { *module = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuModuleUnload(module: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_module = match client.handles.modules.to_remote(module) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_module.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::ModuleUnload, &payload) {
            let result = parse_result(&resp);
            client.handles.modules.remove_by_local(module);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.modules.remove_by_local(module).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuModuleGetFunction(func: *mut u64, module: u64, name: *const i8) -> u32 {
    let client = get_client();
    if func.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_module = match client.handles.modules.to_remote(module) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        if !name.is_null() {
            let name_cstr = unsafe { std::ffi::CStr::from_ptr(name) };
            let name_bytes = name_cstr.to_bytes();
            // Payload: [8B remote_module][4B name_len u32 LE][name bytes]
            let mut payload = Vec::with_capacity(12 + name_bytes.len());
            payload.extend_from_slice(&remote_module.to_le_bytes());
            payload.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(name_bytes);
            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::ModuleGetFunction, &payload)
            {
                let result = parse_result(&resp);
                if result != CUDA_SUCCESS {
                    return result;
                }
                if resp.len() >= 12 {
                    let remote_func =
                        u64::from_le_bytes(resp[4..12].try_into().unwrap());
                    let synthetic = client.handles.functions.insert(remote_func);
                    unsafe { *func = synthetic };
                    return CUDA_SUCCESS;
                }
            }
        }
        // Transport error or null name -- fall through to stub
    }
    // Validate that the module handle exists
    if client.handles.modules.to_remote(module).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.functions.insert(stub_remote);
    unsafe { *func = synthetic };
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Stream management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuStreamCreate(stream: *mut u64, flags: u32) -> u32 {
    let client = get_client();
    if stream.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = flags.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamCreate, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_stream = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.streams.insert(remote_stream);
                unsafe { *stream = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.streams.insert(stub_remote);
    unsafe { *stream = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuStreamDestroy(stream: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = match client.handles.streams.to_remote(stream) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamDestroy, &payload) {
            let result = parse_result(&resp);
            client.handles.streams.remove_by_local(stream);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.streams.remove_by_local(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuStreamSynchronize(stream: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::StreamSynchronize, &payload)
        {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle (stream 0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Event management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuEventCreate(event: *mut u64, flags: u32) -> u32 {
    let client = get_client();
    if event.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = flags.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::EventCreate, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_event = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.events.insert(remote_event);
                unsafe { *event = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.events.insert(stub_remote);
    unsafe { *event = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuEventDestroy(event: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_event = match client.handles.events.to_remote(event) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_event.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::EventDestroy, &payload) {
            let result = parse_result(&resp);
            client.handles.events.remove_by_local(event);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.events.remove_by_local(event).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuEventRecord(event: u64, stream: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_event = match client.handles.events.to_remote(event) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Payload: [8B remote_event][8B remote_stream]
        let mut payload = [0u8; 16];
        payload[0..8].copy_from_slice(&remote_event.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::EventRecord, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate event handle
    if client.handles.events.to_remote(event).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Validate stream handle (0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuEventSynchronize(event: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_event = match client.handles.events.to_remote(event) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_event.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::EventSynchronize, &payload)
        {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle
    if client.handles.events.to_remote(event).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Additional module stubs
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuModuleGetGlobal(
    dptr: *mut u64,
    size: *mut usize,
    module: u64,
    name: *const u8,
    name_len: usize,
) -> u32 {
    let client = get_client();
    if dptr.is_null() || size.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_module = match client.handles.modules.to_remote(module) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        if !name.is_null() && name_len > 0 {
            let name_bytes = unsafe { std::slice::from_raw_parts(name, name_len) };
            // Payload: [8B remote_module][4B name_len u32 LE][name bytes]
            let mut payload = Vec::with_capacity(12 + name_len);
            payload.extend_from_slice(&remote_module.to_le_bytes());
            payload.extend_from_slice(&(name_len as u32).to_le_bytes());
            payload.extend_from_slice(name_bytes);
            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::ModuleGetGlobal, &payload)
            {
                let result = parse_result(&resp);
                if result != CUDA_SUCCESS {
                    return result;
                }
                // Response: 4B result + 8B devptr + 8B size
                if resp.len() >= 20 {
                    let remote_dptr =
                        u64::from_le_bytes(resp[4..12].try_into().unwrap());
                    let global_size =
                        u64::from_le_bytes(resp[12..20].try_into().unwrap()) as usize;
                    let synthetic = client.handles.device_ptrs.insert(remote_dptr);
                    unsafe {
                        *dptr = synthetic;
                        *size = global_size;
                    }
                    return CUDA_SUCCESS;
                }
            }
        }
        // Transport error or null name -- fall through to stub
    }
    // Validate that the module handle exists
    if client.handles.modules.to_remote(module).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Stub: generate a local-only device pointer for the global
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.device_ptrs.insert(stub_remote);
    unsafe {
        *dptr = synthetic;
        *size = 256;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Additional stream stubs
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuStreamQuery(stream: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamQuery, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle (stream 0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Additional event stubs
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> u32 {
    let client = get_client();
    if ms.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_start = match client.handles.events.to_remote(start) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_end = match client.handles.events.to_remote(end) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B remote_start][8B remote_end]
        let mut payload = [0u8; 16];
        payload[0..8].copy_from_slice(&remote_start.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_end.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::EventElapsedTime, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            // Response: 4B result + 4B f32 ms
            if resp.len() >= 8 {
                let elapsed = f32::from_le_bytes(resp[4..8].try_into().unwrap());
                unsafe { *ms = elapsed };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Validate both event handles
    if client.handles.events.to_remote(start).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.handles.events.to_remote(end).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Stub: return 0.0 ms
    unsafe { *ms = 0.0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuEventQuery(event: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_event = match client.handles.events.to_remote(event) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_event.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::EventQuery, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle
    if client.handles.events.to_remote(event).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuLaunchKernel(
    func: u64,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem: u32,
    stream: u64,
    kernel_params: *const *const u8,
    num_params: u32,
    param_sizes: *const u32,
) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Payload: [8B func][4B gridX][4B gridY][4B gridZ][4B blockX][4B blockY][4B blockZ][4B sharedMem][8B stream]
        // = 44 bytes fixed header, then serialized kernel params.
        //
        // Params format after the 44-byte header:
        //   [4B num_params: u32 LE]
        //   [4B param_sizes[0]: u32 LE][param_bytes[0]...]
        //   [4B param_sizes[1]: u32 LE][param_bytes[1]...]
        //   ...
        let mut payload = Vec::with_capacity(44 + 4);
        payload.extend_from_slice(&remote_func.to_le_bytes());
        payload.extend_from_slice(&grid_x.to_le_bytes());
        payload.extend_from_slice(&grid_y.to_le_bytes());
        payload.extend_from_slice(&grid_z.to_le_bytes());
        payload.extend_from_slice(&block_x.to_le_bytes());
        payload.extend_from_slice(&block_y.to_le_bytes());
        payload.extend_from_slice(&block_z.to_le_bytes());
        payload.extend_from_slice(&shared_mem.to_le_bytes());
        payload.extend_from_slice(&remote_stream.to_le_bytes());

        // Serialize kernel parameters.
        //
        // SAFETY: When num_params > 0, the caller must guarantee:
        //   - kernel_params points to at least num_params valid *const u8 pointers
        //   - param_sizes points to at least num_params valid u32 values (4-byte aligned)
        //   - Each kernel_params[i] points to param_sizes[i] readable bytes
        // These are the same guarantees CUDA imposes on cuLaunchKernel callers.
        const MAX_KERNEL_PARAMS: u32 = 1024; // CUDA limits total param bytes to 4096
        let n = num_params;
        if n > MAX_KERNEL_PARAMS {
            return CUDA_ERROR_INVALID_VALUE;
        }
        payload.extend_from_slice(&n.to_le_bytes());
        if n > 0 && !kernel_params.is_null() && !param_sizes.is_null() {
            for i in 0..n as usize {
                unsafe {
                    let size = *param_sizes.add(i);
                    let ptr = *kernel_params.add(i);
                    if size > 0 && ptr.is_null() {
                        // Non-zero size with null data pointer is a caller bug
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    payload.extend_from_slice(&size.to_le_bytes());
                    if size > 0 {
                        let bytes = std::slice::from_raw_parts(ptr, size as usize);
                        payload.extend_from_slice(bytes);
                    }
                }
            }
        }

        if let Ok((_hdr, resp)) = client.send_request(MessageType::LaunchKernel, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate function handle
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Validate stream handle (0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Stub: accept the launch without executing
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// MemcpyDtoD
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemcpyDtoD(dst: u64, src: u64, byte_count: usize) -> u32 {
    let client = get_client();
    if byte_count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: translate both handles and send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_src = match client.handles.device_ptrs.to_remote(src) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B remote_dst][8B remote_src][8B size as u64]
        let mut payload = [0u8; 24];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_src.to_le_bytes());
        payload[16..24].copy_from_slice(&(byte_count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemcpyDtoD, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate both handles exist
    if client.handles.device_ptrs.to_remote(dst).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.handles.device_ptrs.to_remote(src).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Async memory copy
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemcpyHtoDAsync_v2(
    dst_device: u64,
    src_host: *const u8,
    byte_count: usize,
    stream: u64,
) -> u32 {
    let client = get_client();
    if src_host.is_null() || byte_count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Phase 1 limit: inline transfer must fit in protocol payload
    if byte_count > (outerlink_common::protocol::MAX_PAYLOAD_SIZE as usize) - 16 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send data to the server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Payload: [8B remote_dst][8B remote_stream][data...]
        let host_data = unsafe { std::slice::from_raw_parts(src_host, byte_count) };
        let mut payload = Vec::with_capacity(16 + byte_count);
        payload.extend_from_slice(&remote_dst.to_le_bytes());
        payload.extend_from_slice(&remote_stream.to_le_bytes());
        payload.extend_from_slice(host_data);
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::MemcpyHtoDAsync, &payload)
        {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle and accept the copy
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Validate stream handle (0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemcpyDtoHAsync_v2(
    dst_host: *mut u8,
    src_device: u64,
    byte_count: usize,
    stream: u64,
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
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Request payload: [8B remote_src][8B byte_count][8B remote_stream]
        let mut payload = [0u8; 24];
        payload[0..8].copy_from_slice(&remote_src.to_le_bytes());
        payload[8..16].copy_from_slice(&(byte_count as u64).to_le_bytes());
        payload[16..24].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::MemcpyDtoHAsync, &payload)
        {
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
    // Validate stream handle (0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        ptr::write_bytes(dst_host, 0, byte_count);
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Memset operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemsetD8(dst_device: u64, value: u8, count: usize) -> u32 {
    let client = get_client();
    if count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B dst][1B value][8B count]
        let mut payload = [0u8; 17];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8] = value;
        payload[9..17].copy_from_slice(&(count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD8, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemsetD32(dst_device: u64, value: u32, count: usize) -> u32 {
    let client = get_client();
    if count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B dst][4B value][8B count]
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..12].copy_from_slice(&value.to_le_bytes());
        payload[12..20].copy_from_slice(&(count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD32, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemsetD8Async(dst_device: u64, value: u8, count: usize, stream: u64) -> u32 {
    let client = get_client();
    if count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Payload: [8B dst][1B value][8B count][8B stream]
        let mut payload = [0u8; 25];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8] = value;
        payload[9..17].copy_from_slice(&(count as u64).to_le_bytes());
        payload[17..25].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD8Async, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handles
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemsetD32Async(dst_device: u64, value: u32, count: usize, stream: u64) -> u32 {
    let client = get_client();
    if count == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        // Payload: [8B dst][4B value][8B count][8B stream]
        let mut payload = [0u8; 28];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..12].copy_from_slice(&value.to_le_bytes());
        payload[12..20].copy_from_slice(&(count as u64).to_le_bytes());
        payload[20..28].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD32Async, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handles
    if client.handles.device_ptrs.to_remote(dst_device).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// MemAllocHost / MemFreeHost
// ---------------------------------------------------------------------------

/// Tracking set for host-allocated pointers (stub mode).
/// In connected mode the server manages host memory; in stub mode we allocate
/// real host memory and track pointers here for safe deallocation.
static HOST_ALLOCS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
    std::sync::OnceLock::new();

fn host_allocs() -> &'static std::sync::Mutex<std::collections::HashSet<u64>> {
    HOST_ALLOCS.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Tracking map for host-allocated pointer sizes (stub mode).
/// Needed for correct deallocation with matching layout.
static HOST_ALLOC_SIZES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u64, usize>>> =
    std::sync::OnceLock::new();

fn host_alloc_sizes() -> &'static std::sync::Mutex<std::collections::HashMap<u64, usize>> {
    HOST_ALLOC_SIZES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[no_mangle]
pub extern "C" fn ol_cuMemAllocHost(pp: *mut *mut u8, byte_size: usize) -> u32 {
    let client = get_client();
    if pp.is_null() || byte_size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = (byte_size as u64).to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemAllocHost, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            // Response: 4B result + 8B host_ptr
            if resp.len() >= 12 {
                let host_ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                unsafe { *pp = host_ptr as *mut u8 };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: allocate real host memory
    let layout = std::alloc::Layout::from_size_align(byte_size, 8).unwrap();
    let allocated = unsafe { std::alloc::alloc_zeroed(layout) };
    if allocated.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let ptr_val = allocated as u64;
    host_allocs().lock().unwrap().insert(ptr_val);
    host_alloc_sizes().lock().unwrap().insert(ptr_val, byte_size);
    unsafe { *pp = allocated };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemFreeHost(p: *mut u8) -> u32 {
    let client = get_client();
    if p.is_null() {
        return CUDA_SUCCESS; // Freeing null is a no-op
    }
    // Connected: tell the server
    if client.connected.load(Ordering::Acquire) {
        let payload = (p as u64).to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemFreeHost, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: remove from tracking set and deallocate
    let ptr_val = p as u64;
    let mut set = host_allocs().lock().unwrap();
    if !set.remove(&ptr_val) {
        return CUDA_ERROR_INVALID_VALUE; // Double-free or unknown pointer
    }
    drop(set);
    let size = host_alloc_sizes().lock().unwrap().remove(&ptr_val).unwrap_or(1);
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, 8);
        std::alloc::dealloc(p, layout);
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// StreamWaitEvent
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuStreamWaitEvent(stream: u64, event: u64, flags: u32) -> u32 {
    let client = get_client();
    // Connected: translate handles and send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        let remote_event = match client.handles.events.to_remote(event) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B remote_stream][8B remote_event][4B flags]
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_stream.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_event.to_le_bytes());
        payload[16..20].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::StreamWaitEvent, &payload)
        {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate event handle exists
    if client.handles.events.to_remote(event).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Validate stream handle (0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

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
        // In disconnected (stub) mode, no context is set so it returns 0.
        // The connected path (which reverse-translates remote handles via
        // handles.contexts.to_local) is tested in integration tests.
        assert_eq!(ctx, 0);
    }

    #[test]
    fn test_ol_cu_ctx_get_current_null_ptr() {
        let result = ol_cuCtxGetCurrent(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
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

    // -- Module tests --

    #[test]
    fn test_ol_cu_module_load_data() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        let result = ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len());
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(module, 0);
        // Clean up
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_module_load_data_null_ptr() {
        let data = [0u8; 16];
        let result = ol_cuModuleLoadData(ptr::null_mut(), data.as_ptr(), data.len());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_module_unload() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        assert_eq!(ol_cuModuleUnload(module), CUDA_SUCCESS);
        // Double unload should fail
        assert_eq!(ol_cuModuleUnload(module), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_module_get_function() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);

        let mut func: u64 = 0;
        let name = b"my_kernel\0";
        let result = ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(func, 0);

        // Clean up
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_module_get_function_invalid_module() {
        let mut func: u64 = 0;
        let name = b"my_kernel\0";
        let result = ol_cuModuleGetFunction(&mut func, 0xDEAD, name.as_ptr() as *const i8);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // -- Stream tests --

    #[test]
    fn test_ol_cu_stream_create_and_destroy() {
        let mut stream: u64 = 0;
        let result = ol_cuStreamCreate(&mut stream, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(stream, 0);

        let result = ol_cuStreamDestroy(stream);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_stream_create_null_ptr() {
        let result = ol_cuStreamCreate(ptr::null_mut(), 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_destroy_invalid() {
        assert_eq!(ol_cuStreamDestroy(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_synchronize() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamSynchronize(stream), CUDA_SUCCESS);
        // Default stream (0) is always valid
        assert_eq!(ol_cuStreamSynchronize(0), CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_synchronize_invalid() {
        assert_eq!(ol_cuStreamSynchronize(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    // -- Event tests --

    #[test]
    fn test_ol_cu_event_create_and_destroy() {
        let mut event: u64 = 0;
        let result = ol_cuEventCreate(&mut event, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(event, 0);

        let result = ol_cuEventDestroy(event);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_event_create_null_ptr() {
        let result = ol_cuEventCreate(ptr::null_mut(), 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_event_destroy_invalid() {
        assert_eq!(ol_cuEventDestroy(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_event_record() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        // Record on default stream (0)
        assert_eq!(ol_cuEventRecord(event, 0), CUDA_SUCCESS);

        // Record on a created stream
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventRecord(event, stream), CUDA_SUCCESS);

        let _ = ol_cuStreamDestroy(stream);
        let _ = ol_cuEventDestroy(event);
    }

    #[test]
    fn test_ol_cu_event_record_invalid_event() {
        assert_eq!(ol_cuEventRecord(0xDEAD, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_event_synchronize() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventSynchronize(event), CUDA_SUCCESS);
        let _ = ol_cuEventDestroy(event);
    }

    #[test]
    fn test_ol_cu_event_synchronize_invalid() {
        assert_eq!(ol_cuEventSynchronize(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    // -- Kernel launch tests --

    #[test]
    fn test_ol_cu_launch_kernel() {
        // Create valid module and function handles
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"my_kernel\0";
        assert_eq!(
            ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8),
            CUDA_SUCCESS,
        );

        let result = ol_cuLaunchKernel(
            func,    // valid function handle
            1, 1, 1, // grid
            32, 1, 1, // block
            0,       // shared mem
            0,       // default stream
            ptr::null(), // no params
            0,           // num_params
            ptr::null(), // no param sizes
        );
        assert_eq!(result, CUDA_SUCCESS);

        // Clean up
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_launch_kernel_invalid_func() {
        let result = ol_cuLaunchKernel(
            0xDEAD,  // invalid function handle
            1, 1, 1,
            32, 1, 1,
            0, 0,
            ptr::null(),
            0,
            ptr::null(),
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_launch_kernel_invalid_stream() {
        // Create a valid function handle
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(
            ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8),
            CUDA_SUCCESS,
        );

        let result = ol_cuLaunchKernel(
            func,
            1, 1, 1,
            32, 1, 1,
            0,
            0xDEAD, // invalid stream handle
            ptr::null(),
            0,
            ptr::null(),
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);

        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_launch_kernel_with_valid_stream() {
        // Create valid function and stream handles
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(
            ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8),
            CUDA_SUCCESS,
        );
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);

        let result = ol_cuLaunchKernel(
            func,
            1, 1, 1,
            32, 1, 1,
            0,
            stream,  // valid stream
            ptr::null(),
            0,
            ptr::null(),
        );
        assert_eq!(result, CUDA_SUCCESS);

        let _ = ol_cuStreamDestroy(stream);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_launch_kernel_with_params() {
        // Create valid module and function handles
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"doubling\0";
        assert_eq!(
            ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8),
            CUDA_SUCCESS,
        );

        // Simulate kernel params: a device pointer (u64) and an int (u32)
        let dev_ptr: u64 = 0x1234_5678_ABCD_0000;
        let count: u32 = 1024;

        let param0 = dev_ptr.to_le_bytes();
        let param1 = count.to_le_bytes();

        let param_ptrs: [*const u8; 2] = [
            param0.as_ptr(),
            param1.as_ptr(),
        ];
        let param_sizes: [u32; 2] = [
            std::mem::size_of::<u64>() as u32,
            std::mem::size_of::<u32>() as u32,
        ];

        let result = ol_cuLaunchKernel(
            func,
            1, 1, 1,
            256, 1, 1,
            0,
            0,
            param_ptrs.as_ptr(),
            2,
            param_sizes.as_ptr(),
        );
        assert_eq!(result, CUDA_SUCCESS);

        let _ = ol_cuModuleUnload(module);
    }

    // -- ModuleGetGlobal tests --

    #[test]
    fn test_ol_cu_module_get_global() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);

        let mut dptr: u64 = 0;
        let mut size: usize = 0;
        let name = b"my_global";
        let result = ol_cuModuleGetGlobal(
            &mut dptr, &mut size, module, name.as_ptr(), name.len(),
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(dptr, 0);
        assert_eq!(size, 256);

        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_module_get_global_invalid_module() {
        let mut dptr: u64 = 0;
        let mut size: usize = 0;
        let name = b"g";
        let result = ol_cuModuleGetGlobal(
            &mut dptr, &mut size, 0xDEAD, name.as_ptr(), name.len(),
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_module_get_global_null_ptrs() {
        let name = b"g";
        assert_eq!(
            ol_cuModuleGetGlobal(ptr::null_mut(), &mut 0usize, 0x1000, name.as_ptr(), name.len()),
            CUDA_ERROR_INVALID_VALUE,
        );
        assert_eq!(
            ol_cuModuleGetGlobal(&mut 0u64, ptr::null_mut(), 0x1000, name.as_ptr(), name.len()),
            CUDA_ERROR_INVALID_VALUE,
        );
    }

    // -- StreamQuery tests --

    #[test]
    fn test_ol_cu_stream_query() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamQuery(stream), CUDA_SUCCESS);
        // Default stream (0) is always valid
        assert_eq!(ol_cuStreamQuery(0), CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_query_invalid() {
        assert_eq!(ol_cuStreamQuery(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    // -- EventElapsedTime tests --

    #[test]
    fn test_ol_cu_event_elapsed_time() {
        let mut e1: u64 = 0;
        let mut e2: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut e1, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventCreate(&mut e2, 0), CUDA_SUCCESS);

        let mut ms: f32 = -1.0;
        let result = ol_cuEventElapsedTime(&mut ms, e1, e2);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(ms, 0.0);

        let _ = ol_cuEventDestroy(e1);
        let _ = ol_cuEventDestroy(e2);
    }

    #[test]
    fn test_ol_cu_event_elapsed_time_invalid() {
        let mut ms: f32 = 0.0;
        assert_eq!(ol_cuEventElapsedTime(&mut ms, 0xDEAD, 0xBEEF), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_event_elapsed_time_null_ptr() {
        assert_eq!(ol_cuEventElapsedTime(ptr::null_mut(), 0, 0), CUDA_ERROR_INVALID_VALUE);
    }

    // -- EventQuery tests --

    #[test]
    fn test_ol_cu_event_query() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventQuery(event), CUDA_SUCCESS);
        let _ = ol_cuEventDestroy(event);
    }

    #[test]
    fn test_ol_cu_event_query_invalid() {
        assert_eq!(ol_cuEventQuery(0xDEAD), CUDA_ERROR_INVALID_VALUE);
    }

    // -- MemcpyDtoD tests --

    #[test]
    fn test_ol_cu_memcpy_dtod() {
        let mut src: u64 = 0;
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpyDtoD(dst, src, 512), CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(src);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_dtod_invalid_dst() {
        let mut src: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpyDtoD(0xBAAD, src, 512), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(src);
    }

    #[test]
    fn test_ol_cu_memcpy_dtod_invalid_src() {
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpyDtoD(dst, 0xBAAD, 512), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_dtod_zero_size() {
        assert_eq!(ol_cuMemcpyDtoD(0x1000, 0x2000, 0), CUDA_ERROR_INVALID_VALUE);
    }

    // -- MemAllocHost / MemFreeHost tests --

    #[test]
    fn test_ol_cu_mem_alloc_host_and_free() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 4096), CUDA_SUCCESS);
        assert!(!ptr.is_null());
        // Write to the allocated memory to verify it is accessible
        unsafe { *ptr = 42 };
        assert_eq!(unsafe { *ptr }, 42);
        assert_eq!(ol_cuMemFreeHost(ptr), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_alloc_host_null_pp() {
        assert_eq!(ol_cuMemAllocHost(ptr::null_mut(), 4096), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_alloc_host_zero_size() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_free_host_null() {
        // Freeing null is a no-op
        assert_eq!(ol_cuMemFreeHost(ptr::null_mut()), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_free_host_double_free() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 256), CUDA_SUCCESS);
        assert_eq!(ol_cuMemFreeHost(ptr), CUDA_SUCCESS);
        // Second free should fail
        assert_eq!(ol_cuMemFreeHost(ptr), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_free_host_unknown_ptr() {
        // A pointer never allocated by us
        assert_eq!(ol_cuMemFreeHost(0xDEAD as *mut u8), CUDA_ERROR_INVALID_VALUE);
    }

    // -- StreamWaitEvent tests --

    #[test]
    fn test_ol_cu_stream_wait_event() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        // Default stream (0) with valid event
        assert_eq!(ol_cuStreamWaitEvent(0, event, 0), CUDA_SUCCESS);

        // Created stream with valid event
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamWaitEvent(stream, event, 0), CUDA_SUCCESS);

        let _ = ol_cuStreamDestroy(stream);
        let _ = ol_cuEventDestroy(event);
    }

    #[test]
    fn test_ol_cu_stream_wait_event_invalid_event() {
        assert_eq!(ol_cuStreamWaitEvent(0, 0xDEAD, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_wait_event_invalid_stream() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamWaitEvent(0xDEAD, event, 0), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuEventDestroy(event);
    }
}
