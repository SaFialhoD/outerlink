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
// Returned by server when cuDevicePrimaryCtxSetFlags is called on an active context
#[allow(dead_code)]
const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: u32 = 708;
// Returned when peer access was already enabled
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: u32 = 704;
// Returned when peer access was not previously enabled
const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: u32 = 705;
const CUDA_ERROR_UNKNOWN: u32 = 999;

// ---------------------------------------------------------------------------
// Occupancy stub constants (Ampere / GA102 defaults)
// ---------------------------------------------------------------------------

/// Number of SMs on the stub GPU (RTX 3090 = 82 SMs).
const STUB_NUM_SMS: i32 = 82;
/// Maximum resident threads per SM on Ampere.
const STUB_MAX_THREADS_PER_SM: i32 = 2048;
/// Maximum resident blocks per SM on Ampere.
const STUB_MAX_BLOCKS_PER_SM: i32 = 16;

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

/// Tracks which peer contexts have been enabled via `cuCtxEnablePeerAccess` in
/// stub mode. Used to return correct error codes on double-enable and
/// disable-without-enable.
// Track device allocation sizes for cuMemGetAddressRange stub
static DEVICE_ALLOC_SIZES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u64, usize>>> =
    std::sync::OnceLock::new();
fn device_alloc_sizes() -> &'static std::sync::Mutex<std::collections::HashMap<u64, usize>> {
    DEVICE_ALLOC_SIZES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

static STUB_PEER_ACCESS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
    std::sync::OnceLock::new();

fn stub_peer_access() -> &'static std::sync::Mutex<std::collections::HashSet<u64>> {
    STUB_PEER_ACCESS.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

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
        16 => 82,    // MULTIPROCESSOR_COUNT
        75 => 8,     // COMPUTE_CAPABILITY_MAJOR
        76 => 6,     // COMPUTE_CAPABILITY_MINOR
        81 => 102400, // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
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
// Context stack & query operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuCtxPushCurrent_v2(ctx: u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = match client.handles.contexts.to_remote(ctx) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_CONTEXT,
        };
        let payload = remote_ctx.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxPushCurrent, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle exists
    match client.handles.contexts.to_remote(ctx) {
        Some(_) => CUDA_SUCCESS,
        None => CUDA_ERROR_INVALID_CONTEXT,
    }
}

#[no_mangle]
pub extern "C" fn ol_cuCtxPopCurrent_v2(pctx: *mut u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxPopCurrent, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 && !pctx.is_null() {
                let remote_ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let local = client.handles.contexts.to_local(remote_ctx).unwrap_or(0);
                unsafe { *pctx = local };
            }
            return CUDA_SUCCESS;
        }
        // Transport error -- fall through to stub
    }
    // Stub: return null context
    if !pctx.is_null() {
        unsafe { *pctx = 0 };
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetApiVersion(ctx: u64, version: *mut u32) -> u32 {
    let client = get_client();
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = match client.handles.contexts.to_remote(ctx) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_CONTEXT,
        };
        let payload = remote_ctx.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetApiVersion, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *version = u32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: return CUDA 12.0
    unsafe { *version = 12000 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetLimit(pvalue: *mut u64, limit: u32) -> u32 {
    let client = get_client();
    if pvalue.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = limit.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetLimit, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                unsafe { *pvalue = u64::from_le_bytes(resp[4..12].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: return reasonable defaults
    let default = match limit {
        0x00 => 1024,       // CU_LIMIT_STACK_SIZE
        0x01 => 1_048_576,  // CU_LIMIT_PRINTF_FIFO_SIZE
        0x02 => 8_388_608,  // CU_LIMIT_MALLOC_HEAP_SIZE
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    unsafe { *pvalue = default };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxSetLimit(limit: u32, value: u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&limit.to_le_bytes());
        payload[4..12].copy_from_slice(&value.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxSetLimit, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: accept silently
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetStreamPriorityRange(least: *mut i32, greatest: *mut i32) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetStreamPriorityRange, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                if !least.is_null() {
                    unsafe { *least = i32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                }
                if !greatest.is_null() {
                    unsafe { *greatest = i32::from_le_bytes(resp[8..12].try_into().unwrap()) };
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: standard NVIDIA range
    if !least.is_null() {
        unsafe { *least = 0 };
    }
    if !greatest.is_null() {
        unsafe { *greatest = -1 };
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetFlags(flags: *mut u32) -> u32 {
    let client = get_client();
    if flags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetFlags, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *flags = u32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: default flags
    unsafe { *flags = 0 };
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Cache and shared memory configuration
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuCtxGetCacheConfig(pconfig: *mut u32) -> u32 {
    let client = get_client();
    if pconfig.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetCacheConfig, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *pconfig = u32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: default is PREFER_NONE (0)
    unsafe { *pconfig = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxSetCacheConfig(config: u32) -> u32 {
    let client = get_client();
    if config > 0x03 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = config.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxSetCacheConfig, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: accept silently
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxGetSharedMemConfig(pconfig: *mut u32) -> u32 {
    let client = get_client();
    if pconfig.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxGetSharedMemConfig, &[]) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *pconfig = u32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: default is DEFAULT_BANK_SIZE (0)
    unsafe { *pconfig = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxSetSharedMemConfig(config: u32) -> u32 {
    let client = get_client();
    if config > 0x02 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = config.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::CtxSetSharedMemConfig, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: accept silently
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuFuncSetCacheConfig(func: u64, config: u32) -> u32 {
    let client = get_client();
    if config > 0x03 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 12];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..12].copy_from_slice(&config.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::FuncSetCacheConfig, &payload) {
            return parse_result(&resp);
        }
    }
    // Stub fallback: validate function handle exists
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuFuncSetSharedMemConfig(func: u64, config: u32) -> u32 {
    let client = get_client();
    if config > 0x02 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 12];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..12].copy_from_slice(&config.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::FuncSetSharedMemConfig, &payload) {
            return parse_result(&resp);
        }
    }
    // Stub fallback: validate function handle exists
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Primary context management
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuDevicePrimaryCtxRetain(pctx: *mut u64, dev: i32) -> u32 {
    let client = get_client();
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DevicePrimaryCtxRetain, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.contexts.insert_or_get(remote_ctx);
                // NOTE: Do NOT update current_remote_ctx -- CUDA semantics say
                // Retain does NOT set current context.
                unsafe { *pctx = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: deterministic remote handle per device for idempotent insert_or_get
    let stub_remote = 0xFFFF_0000_0000_0000u64 + dev as u64;
    let synthetic = client.handles.contexts.insert_or_get(stub_remote);
    unsafe { *pctx = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDevicePrimaryCtxRelease(dev: i32) -> u32 {
    let client = get_client();
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DevicePrimaryCtxRelease, &payload) {
            return parse_result(&resp);
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDevicePrimaryCtxGetState(dev: i32, flags: *mut u32, active: *mut i32) -> u32 {
    let client = get_client();
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if flags.is_null() || active.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DevicePrimaryCtxGetState, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let f = u32::from_le_bytes(resp[4..8].try_into().unwrap());
                let a = i32::from_le_bytes(resp[8..12].try_into().unwrap());
                unsafe {
                    *flags = f;
                    *active = a;
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: inactive, no flags
    unsafe {
        *flags = 0;
        *active = 0;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDevicePrimaryCtxSetFlags(dev: i32, flags: u32) -> u32 {
    let client = get_client();
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&dev.to_le_bytes());
        payload[4..8].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DevicePrimaryCtxSetFlags, &payload) {
            return parse_result(&resp);
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDevicePrimaryCtxReset(dev: i32) -> u32 {
    let client = get_client();
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DevicePrimaryCtxReset, &payload) {
            return parse_result(&resp);
        }
    }
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
                device_alloc_sizes().lock().unwrap().insert(synthetic, size);
                unsafe { *dptr = synthetic };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.device_ptrs.insert(stub_remote);
    device_alloc_sizes().lock().unwrap().insert(synthetic, size);
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
            device_alloc_sizes().lock().unwrap().remove(&dptr);
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: just remove from local handles
    if client.handles.device_ptrs.remove_by_local(dptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    device_alloc_sizes().lock().unwrap().remove(&dptr);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemcpyHtoD_v2(
    dst_device: u64,
    src_host: *const u8,
    byte_count: usize,
) -> u32 {
    let client = get_client();
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    if src_host.is_null() {
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
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    if dst_host.is_null() {
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
pub extern "C" fn ol_cuModuleLoadDataEx(
    module: *mut u64,
    data: *const u8,
    data_len: usize,
    num_options: u32,
    options: *const i32,
    option_values: *const u64,
) -> u32 {
    let client = get_client();
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: serialize image + options into wire format and send
    if client.connected.load(Ordering::Acquire) {
        if !data.is_null() && data_len > 0 {
            let image = unsafe { std::slice::from_raw_parts(data, data_len) };
            // Build wire payload: 4B image_len + 4B num_options + options + image
            let n = num_options as usize;
            let mut payload = Vec::with_capacity(8 + n * 12 + data_len);
            payload.extend_from_slice(&(data_len as u32).to_le_bytes());
            payload.extend_from_slice(&num_options.to_le_bytes());
            if n > 0 && !options.is_null() && !option_values.is_null() {
                let opts = unsafe { std::slice::from_raw_parts(options, n) };
                let vals = unsafe { std::slice::from_raw_parts(option_values, n) };
                for i in 0..n {
                    payload.extend_from_slice(&opts[i].to_le_bytes());
                    payload.extend_from_slice(&vals[i].to_le_bytes());
                }
            }
            payload.extend_from_slice(image);
            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::ModuleLoadDataEx, &payload)
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
    // Stub: generate a local-only handle (ignore options)
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
// Function attribute query
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuFuncGetAttribute(pi: *mut i32, attrib: i32, func: u64) -> u32 {
    let client = get_client();
    if pi.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 12];
        payload[0..4].copy_from_slice(&attrib.to_le_bytes());
        payload[4..12].copy_from_slice(&remote_func.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::FuncGetAttribute, &payload)
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
    // Stub fallback: validate function handle exists
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let value = match attrib {
        0 => 1024,  // MAX_THREADS_PER_BLOCK
        1 => 0,     // SHARED_SIZE_BYTES
        2 => 0,     // CONST_SIZE_BYTES
        3 => 0,     // LOCAL_SIZE_BYTES
        4 => 32,    // NUM_REGS
        5 => 80,    // PTX_VERSION
        6 => 80,    // BINARY_VERSION
        7 => 0,     // CACHE_MODE_CA
        8 => 0,     // MAX_DYNAMIC_SHARED_SIZE_BYTES
        9 => 0,     // PREFERRED_SHARED_MEMORY_CARVEOUT
        10 => 0,    // CLUSTER_SIZE_REQUESTED
        11 => 0,    // REQUIRED_CLUSTER_WIDTH
        12 => 0,    // REQUIRED_CLUSTER_HEIGHT
        13 => 0,    // REQUIRED_CLUSTER_DEPTH
        14 => 0,    // NON_PORTABLE_CLUSTER_SIZE_ALLOWED
        15 => 0,    // CLUSTER_SCHEDULING_POLICY_PREFERENCE
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    unsafe { *pi = value };
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Function attribute set
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuFuncSetAttribute(func: u64, attrib: i32, value: i32) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 16];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..12].copy_from_slice(&attrib.to_le_bytes());
        payload[12..16].copy_from_slice(&value.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::FuncSetAttribute, &payload)
        {
            return parse_result(&resp);
        }
    }
    // Stub fallback: validate function handle exists
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Only settable attributes: 8 (MAX_DYNAMIC_SHARED_SIZE_BYTES), 9 (PREFERRED_SHARED_MEMORY_CARVEOUT)
    match attrib {
        8 | 9 => CUDA_SUCCESS,
        _ => CUDA_ERROR_INVALID_VALUE,
    }
}

// ---------------------------------------------------------------------------
// Memory address range query
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemGetAddressRange_v2(
    pbase: *mut u64,
    psize: *mut usize,
    dptr: u64,
) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_dptr = match client.handles.device_ptrs.to_remote(dptr) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_dptr.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::MemGetAddressRange, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 20 {
                // base is the remote pointer; we map it back to local as `dptr`
                // since for exact-match allocations base == remote_dptr.
                let _remote_base = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let size = u64::from_le_bytes(resp[12..20].try_into().unwrap()) as usize;
                if !pbase.is_null() {
                    unsafe { *pbase = dptr };
                }
                if !psize.is_null() {
                    unsafe { *psize = size };
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback: validate the device pointer exists in our handle table
    if client.handles.device_ptrs.to_remote(dptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let size = device_alloc_sizes().lock().unwrap().get(&dptr).copied().unwrap_or(0);
    if !pbase.is_null() {
        unsafe { *pbase = dptr };
    }
    if !psize.is_null() {
        unsafe { *psize = size };
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(
    num_blocks: *mut i32,
    func: u64,
    block_size: i32,
    dynamic_smem_size: u64,
) -> u32 {
    let client = get_client();
    if num_blocks.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..12].copy_from_slice(&block_size.to_le_bytes());
        payload[12..20].copy_from_slice(&dynamic_smem_size.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(
            MessageType::OccupancyMaxActiveBlocksPerMultiprocessor,
            &payload,
        ) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *num_blocks = i32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if block_size <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let blocks = std::cmp::min(STUB_MAX_THREADS_PER_SM / block_size, STUB_MAX_BLOCKS_PER_SM);
    unsafe { *num_blocks = std::cmp::max(blocks, 1) };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    num_blocks: *mut i32,
    func: u64,
    block_size: i32,
    dynamic_smem_size: u64,
    flags: u32,
) -> u32 {
    let client = get_client();
    if num_blocks.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 24];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..12].copy_from_slice(&block_size.to_le_bytes());
        payload[12..20].copy_from_slice(&dynamic_smem_size.to_le_bytes());
        payload[20..24].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(
            MessageType::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            &payload,
        ) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe { *num_blocks = i32::from_le_bytes(resp[4..8].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if block_size <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let blocks = std::cmp::min(STUB_MAX_THREADS_PER_SM / block_size, STUB_MAX_BLOCKS_PER_SM);
    unsafe { *num_blocks = std::cmp::max(blocks, 1) };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuOccupancyMaxPotentialBlockSize(
    min_grid_size: *mut i32,
    block_size: *mut i32,
    func: u64,
    callback: *const std::ffi::c_void,
    dynamic_smem_size: u64,
    block_size_limit: i32,
) -> u32 {
    let client = get_client();
    if min_grid_size.is_null() || block_size.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // If callback is non-NULL, invoke it to get dynamic shared memory for the
    // chosen block size, then compute occupancy locally.
    if !callback.is_null() {
        let bs = if block_size_limit > 0 && block_size_limit < 256 {
            block_size_limit
        } else {
            256
        };
        // The CUDA callback signature is: size_t (*CUoccupancyB2DSize)(int blockSize)
        let callback_fn: unsafe extern "C" fn(i32) -> usize =
            unsafe { std::mem::transmute(callback) };
        let _dynamic_smem = unsafe { callback_fn(bs) };
        let blocks_per_sm = STUB_MAX_THREADS_PER_SM / bs;
        unsafe {
            *block_size = bs;
            *min_grid_size = blocks_per_sm * STUB_NUM_SMS;
        }
        return CUDA_SUCCESS;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..16].copy_from_slice(&dynamic_smem_size.to_le_bytes());
        payload[16..20].copy_from_slice(&block_size_limit.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(
            MessageType::OccupancyMaxPotentialBlockSize,
            &payload,
        ) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                unsafe {
                    *min_grid_size = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                    *block_size = i32::from_le_bytes(resp[8..12].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let bs = if block_size_limit > 0 && block_size_limit < 256 {
        block_size_limit
    } else {
        256
    };
    let blocks_per_sm = STUB_MAX_THREADS_PER_SM / bs;
    unsafe {
        *block_size = bs;
        *min_grid_size = blocks_per_sm * STUB_NUM_SMS;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuOccupancyMaxPotentialBlockSizeWithFlags(
    min_grid_size: *mut i32,
    block_size: *mut i32,
    func: u64,
    callback: *const std::ffi::c_void,
    dynamic_smem_size: u64,
    block_size_limit: i32,
    flags: u32,
) -> u32 {
    let client = get_client();
    if min_grid_size.is_null() || block_size.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // If callback is non-NULL, invoke it to get dynamic shared memory for the
    // chosen block size, then compute occupancy locally.
    if !callback.is_null() {
        let bs = if block_size_limit > 0 && block_size_limit < 256 {
            block_size_limit
        } else {
            256
        };
        // The CUDA callback signature is: size_t (*CUoccupancyB2DSize)(int blockSize)
        let callback_fn: unsafe extern "C" fn(i32) -> usize =
            unsafe { std::mem::transmute(callback) };
        let _dynamic_smem = unsafe { callback_fn(bs) };
        let blocks_per_sm = STUB_MAX_THREADS_PER_SM / bs;
        unsafe {
            *block_size = bs;
            *min_grid_size = blocks_per_sm * STUB_NUM_SMS;
        }
        return CUDA_SUCCESS;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_func = match client.handles.functions.to_remote(func) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = [0u8; 24];
        payload[0..8].copy_from_slice(&remote_func.to_le_bytes());
        payload[8..16].copy_from_slice(&dynamic_smem_size.to_le_bytes());
        payload[16..20].copy_from_slice(&block_size_limit.to_le_bytes());
        payload[20..24].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(
            MessageType::OccupancyMaxPotentialBlockSizeWithFlags,
            &payload,
        ) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                unsafe {
                    *min_grid_size = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                    *block_size = i32::from_le_bytes(resp[8..12].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    if client.handles.functions.to_remote(func).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let bs = if block_size_limit > 0 && block_size_limit < 256 {
        block_size_limit
    } else {
        256
    };
    let blocks_per_sm = STUB_MAX_THREADS_PER_SM / bs;
    unsafe {
        *block_size = bs;
        *min_grid_size = blocks_per_sm * STUB_NUM_SMS;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Pointer attribute queries
// ---------------------------------------------------------------------------

/// Query a single attribute of a pointer.
///
/// The `data` output is a `void*` that may point to different-sized types
/// depending on the attribute. For simplicity, we always transfer the value
/// as u64 over the wire and write the appropriate size based on attribute:
///   - MEMORY_TYPE (2): unsigned int (4 bytes)
///   - CONTEXT (1): CUcontext (8 bytes, pointer-sized)
///   - DEVICE_POINTER (3): CUdeviceptr (8 bytes)
///   - HOST_POINTER (4): void* (8 bytes)
///   - IS_MANAGED (6): unsigned int (4 bytes)
///   - DEVICE_ORDINAL (8): int (4 bytes)
#[no_mangle]
pub extern "C" fn ol_cuPointerGetAttribute(data: *mut u8, attribute: i32, dev_ptr: u64) -> u32 {
    let client = get_client();
    if data.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ptr = match client.handles.device_ptrs.to_remote(dev_ptr) {
            Some(r) => r,
            None => {
                // Check host allocations -- host pointers are not in the handle map
                if !host_allocs().lock().unwrap().contains(&dev_ptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                dev_ptr // host pointers are passed as-is
            }
        };
        let mut payload = [0u8; 12];
        payload[0..4].copy_from_slice(&attribute.to_le_bytes());
        payload[4..12].copy_from_slice(&remote_ptr.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::PointerGetAttribute, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let val = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                write_pointer_attribute(data, attribute, val);
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback
    let is_device = client.handles.device_ptrs.to_remote(dev_ptr).is_some();
    let is_host = host_allocs().lock().unwrap().contains(&dev_ptr);
    if !is_device && !is_host {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let val: u64 = match attribute {
        1 => 0,                                      // CONTEXT
        2 => if is_device { 2 } else { 1 },          // MEMORY_TYPE
        3 => if is_device { dev_ptr } else { 0 },    // DEVICE_POINTER
        4 => if is_host { dev_ptr } else { 0 },      // HOST_POINTER
        6 => 0,                                       // IS_MANAGED
        8 => 0,                                       // DEVICE_ORDINAL
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    write_pointer_attribute(data, attribute, val);
    CUDA_SUCCESS
}

/// Write a pointer attribute value to the caller's `data` buffer.
///
/// The size written depends on the attribute type:
/// - 4 bytes for MEMORY_TYPE(2), IS_MANAGED(6), DEVICE_ORDINAL(8)
/// - 8 bytes for CONTEXT(1), DEVICE_POINTER(3), HOST_POINTER(4)
fn write_pointer_attribute(data: *mut u8, attribute: i32, val: u64) {
    match attribute {
        // 4-byte attributes
        2 | 6 | 8 => unsafe {
            std::ptr::copy_nonoverlapping(
                (val as u32).to_ne_bytes().as_ptr(),
                data,
                4,
            );
        },
        // 8-byte attributes (pointer-sized)
        1 | 3 | 4 => unsafe {
            std::ptr::copy_nonoverlapping(
                val.to_ne_bytes().as_ptr(),
                data,
                8,
            );
        },
        // Unknown: write as u64
        _ => unsafe {
            std::ptr::copy_nonoverlapping(
                val.to_ne_bytes().as_ptr(),
                data,
                8,
            );
        },
    }
}

/// Query multiple pointer attributes in one call.
#[no_mangle]
pub extern "C" fn ol_cuPointerGetAttributes(
    num_attributes: u32,
    attributes: *const i32,
    data: *mut *mut u8,
    ptr: u64,
) -> u32 {
    let client = get_client();
    if attributes.is_null() || data.is_null() || num_attributes == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let n = num_attributes as usize;
    // Read the attribute array
    let attrs: Vec<i32> = unsafe {
        std::slice::from_raw_parts(attributes, n).to_vec()
    };

    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ptr = match client.handles.device_ptrs.to_remote(ptr) {
            Some(r) => r,
            None => {
                if !host_allocs().lock().unwrap().contains(&ptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                ptr
            }
        };
        // [4B numAttrs][8B ptr][N*4B attributes]
        let mut payload = Vec::with_capacity(12 + n * 4);
        payload.extend_from_slice(&num_attributes.to_le_bytes());
        payload.extend_from_slice(&remote_ptr.to_le_bytes());
        for &a in &attrs {
            payload.extend_from_slice(&a.to_le_bytes());
        }
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::PointerGetAttributes, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 4 + n * 8 {
                for i in 0..n {
                    let off = 4 + i * 8;
                    let val = u64::from_le_bytes(resp[off..off + 8].try_into().unwrap());
                    let data_i = unsafe { *data.add(i) };
                    if !data_i.is_null() {
                        write_pointer_attribute(data_i, attrs[i], val);
                    }
                }
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub fallback: call single-attribute for each
    for i in 0..n {
        let data_i = unsafe { *data.add(i) };
        if !data_i.is_null() {
            let result = ol_cuPointerGetAttribute(data_i, attrs[i], ptr);
            if result != CUDA_SUCCESS {
                return result;
            }
        }
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Stream-ordered memory / pool operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemAllocAsync(dptr: *mut u64, size: usize, stream: u64) -> u32 {
    let client = get_client();
    if dptr.is_null() || size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&(size as u64).to_le_bytes());
        let remote_stream = if stream == 0 { 0u64 } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        payload.extend_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemAllocAsync, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_devptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.device_ptrs.insert(remote_devptr);
                device_alloc_sizes().lock().unwrap().insert(synthetic, size);
                unsafe { *dptr = synthetic };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: generate a local-only handle (same as cuMemAlloc stub)
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.device_ptrs.insert(stub_remote);
    device_alloc_sizes().lock().unwrap().insert(synthetic, size);
    unsafe { *dptr = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemFreeAsync(dptr: u64, stream: u64) -> u32 {
    let client = get_client();
    if dptr == 0 {
        return CUDA_SUCCESS; // Freeing null is a no-op
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_devptr = match client.handles.device_ptrs.to_remote(dptr) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 { 0u64 } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&remote_devptr.to_le_bytes());
        payload.extend_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemFreeAsync, &payload) {
            let result = parse_result(&resp);
            client.handles.device_ptrs.remove_by_local(dptr);
            device_alloc_sizes().lock().unwrap().remove(&dptr);
            return result;
        }
    }
    // Stub: just remove from local handles
    if client.handles.device_ptrs.remove_by_local(dptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    device_alloc_sizes().lock().unwrap().remove(&dptr);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetDefaultMemPool(pool: *mut u64, dev: i32) -> u32 {
    let client = get_client();
    if pool.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let payload = dev.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGetDefaultMemPool, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_pool = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.mem_pools.insert_or_get(remote_pool);
                unsafe { *pool = synthetic };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: deterministic handle per device
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let stub_remote = 0xDEFA_0000_u64 | (dev as u64);
    let synthetic = client.handles.mem_pools.insert_or_get(stub_remote);
    unsafe { *pool = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemPoolCreate(pool: *mut u64, alloc_type: i32, loc_type: i32, loc_id: i32) -> u32 {
    let client = get_client();
    if pool.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&alloc_type.to_le_bytes());
        payload.extend_from_slice(&loc_type.to_le_bytes());
        payload.extend_from_slice(&loc_id.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemPoolCreate, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_pool = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.mem_pools.insert(remote_pool);
                unsafe { *pool = synthetic };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: generate a local-only handle
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.mem_pools.insert(stub_remote);
    unsafe { *pool = synthetic };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemPoolDestroy(pool: u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        let remote_pool = match client.handles.mem_pools.to_remote(pool) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_pool.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemPoolDestroy, &payload) {
            let result = parse_result(&resp);
            client.handles.mem_pools.remove_by_local(pool);
            return result;
        }
    }
    // Stub: just remove from local handles
    if client.handles.mem_pools.remove_by_local(pool).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemPoolGetAttribute(pool: u64, attr: i32, value: *mut u64) -> u32 {
    let client = get_client();
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_pool = match client.handles.mem_pools.to_remote(pool) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = Vec::with_capacity(12);
        payload.extend_from_slice(&remote_pool.to_le_bytes());
        payload.extend_from_slice(&attr.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemPoolGetAttribute, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                unsafe { *value = u64::from_le_bytes(resp[4..12].try_into().unwrap()) };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: return 0 (default attribute value)
    unsafe { *value = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemPoolSetAttribute(pool: u64, attr: i32, value: u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        let remote_pool = match client.handles.mem_pools.to_remote(pool) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&remote_pool.to_le_bytes());
        payload.extend_from_slice(&attr.to_le_bytes());
        payload.extend_from_slice(&value.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemPoolSetAttribute, &payload) {
            return parse_result(&resp);
        }
    }
    // Stub: no-op (pretend it succeeded)
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemPoolTrimTo(pool: u64, min_bytes: u64) -> u32 {
    let client = get_client();
    if client.connected.load(Ordering::Acquire) {
        let remote_pool = match client.handles.mem_pools.to_remote(pool) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&remote_pool.to_le_bytes());
        payload.extend_from_slice(&min_bytes.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemPoolTrimTo, &payload) {
            return parse_result(&resp);
        }
    }
    // Stub: no-op
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemAllocFromPoolAsync(dptr: *mut u64, size: usize, pool: u64, stream: u64) -> u32 {
    let client = get_client();
    if dptr.is_null() || size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.connected.load(Ordering::Acquire) {
        let remote_pool = match client.handles.mem_pools.to_remote(pool) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_stream = if stream == 0 { 0u64 } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&(size as u64).to_le_bytes());
        payload.extend_from_slice(&remote_pool.to_le_bytes());
        payload.extend_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemAllocFromPoolAsync, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_devptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                let synthetic = client.handles.device_ptrs.insert(remote_devptr);
                device_alloc_sizes().lock().unwrap().insert(synthetic, size);
                unsafe { *dptr = synthetic };
                return CUDA_SUCCESS;
            }
        }
    }
    // Stub: generate a local-only handle (same as cuMemAlloc stub)
    let stub_remote = STUB_HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let synthetic = client.handles.device_ptrs.insert(stub_remote);
    device_alloc_sizes().lock().unwrap().insert(synthetic, size);
    unsafe { *dptr = synthetic };
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
            let server_result = parse_result(&resp);
            if server_result != CUDA_SUCCESS {
                return server_result;
            }
            // Phase 2: Wait for all pending callbacks on this stream to complete.
            // The server sync ensures all GPU work is done and CallbackReady
            // notifications have been sent. Now wait for the client-side
            // callback invocations to finish.
            if client.callback_registry.has_pending(stream) {
                client.callback_registry.wait_all_completed(
                    stream,
                    std::time::Duration::from_secs(30),
                );
            }
            return CUDA_SUCCESS;
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle (stream 0 = default stream, always valid)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

/// `cuStreamAddCallback` -- register a host callback to be invoked when all
/// preceding work on the stream completes.
///
/// The callback function pointer and user_data are stored locally in the
/// CallbackRegistry. Only a unique callback_id is sent to the server.
/// When the server signals that the callback should fire (via CallbackReady
/// on the callback channel), the client invokes the function locally.
#[no_mangle]
pub extern "C" fn ol_cuStreamAddCallback(
    stream: u64,
    callback: u64, // CUstreamCallback fn ptr
    user_data: u64,
    flags: u32,
) -> u32 {
    let client = get_client();

    // Connected: register locally and send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };

        // Ensure callback channel is established before sending the request
        if let Err(e) = client.ensure_callback_channel() {
            tracing::warn!(error = %e, "failed to establish callback channel, falling back to immediate execution");
            // Fall through to stub-mode immediate execution
        } else {
            // Register in local registry
            let callback_id = client.callback_registry.register(
                crate::callback::CallbackKind::StreamAddCallback,
                callback,
                user_data,
                stream,
            );

            // Send to server: u64 stream, u64 callback_id, u32 flags
            let mut payload = Vec::with_capacity(20);
            payload.extend_from_slice(&remote_stream.to_le_bytes());
            payload.extend_from_slice(&callback_id.to_le_bytes());
            payload.extend_from_slice(&flags.to_le_bytes());

            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::StreamAddCallback, &payload)
            {
                return parse_result(&resp);
            }

            // Transport error -- fire immediately as fallback
            client.callback_registry.fire(callback_id, CUDA_SUCCESS);
            return CUDA_SUCCESS;
        }
    }

    // Stub: fire callback immediately (synchronous execution)
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if callback != 0 {
        type StreamCb = unsafe extern "C" fn(u64, u32, u64);
        let cb: StreamCb = unsafe { std::mem::transmute(callback) };
        unsafe { cb(stream, CUDA_SUCCESS, user_data) };
    }
    CUDA_SUCCESS
}

/// `cuLaunchHostFunc` -- enqueue a host function to execute on the host when
/// all preceding work on the stream completes.
///
/// Similar to cuStreamAddCallback but with a simpler signature (no stream/status args).
#[no_mangle]
pub extern "C" fn ol_cuLaunchHostFunc(
    stream: u64,
    func: u64, // CUhostFn fn ptr
    user_data: u64,
) -> u32 {
    let client = get_client();

    // Connected: register locally and send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = if stream == 0 {
            0u64
        } else {
            match client.handles.streams.to_remote(stream) {
                Some(r) => r,
                None => return CUDA_ERROR_INVALID_VALUE,
            }
        };

        // Ensure callback channel is established
        if let Err(e) = client.ensure_callback_channel() {
            tracing::warn!(error = %e, "failed to establish callback channel, falling back to immediate execution");
        } else {
            let callback_id = client.callback_registry.register(
                crate::callback::CallbackKind::LaunchHostFunc,
                func,
                user_data,
                stream,
            );

            // Send to server: u64 stream, u64 callback_id
            let mut payload = Vec::with_capacity(16);
            payload.extend_from_slice(&remote_stream.to_le_bytes());
            payload.extend_from_slice(&callback_id.to_le_bytes());

            if let Ok((_hdr, resp)) =
                client.send_request(MessageType::LaunchHostFunc, &payload)
            {
                return parse_result(&resp);
            }

            // Transport error -- fire immediately as fallback
            client.callback_registry.fire(callback_id, CUDA_SUCCESS);
            return CUDA_SUCCESS;
        }
    }

    // Stub: fire function immediately
    if stream != 0 && client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if func != 0 {
        type HostFn = unsafe extern "C" fn(u64);
        let cb: HostFn = unsafe { std::mem::transmute(func) };
        unsafe { cb(user_data) };
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

// ---------------------------------------------------------------------------
// cuEventRecordWithFlags
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuEventRecordWithFlags(event: u64, stream: u64, flags: u32) -> u32 {
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
        // Payload: [8B remote_event][8B remote_stream][4B flags]
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_event.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_stream.to_le_bytes());
        payload[16..20].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::EventRecordWithFlags, &payload) {
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
// Stream priority/flags/ctx
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuStreamCreateWithPriority(
    stream: *mut u64,
    flags: u32,
    priority: i32,
) -> u32 {
    let client = get_client();
    if stream.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&flags.to_le_bytes());
        payload[4..8].copy_from_slice(&priority.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::StreamCreateWithPriority, &payload)
        {
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
pub extern "C" fn ol_cuStreamGetPriority(stream: u64, priority: *mut i32) -> u32 {
    let client = get_client();
    if priority.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = match client.handles.streams.to_remote(stream) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamGetPriority, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                let val = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                unsafe { *priority = val };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle, return 0 as default priority
    if client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *priority = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuStreamGetFlags(stream: u64, flags: *mut u32) -> u32 {
    let client = get_client();
    if flags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = match client.handles.streams.to_remote(stream) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamGetFlags, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                let val = u32::from_le_bytes(resp[4..8].try_into().unwrap());
                unsafe { *flags = val };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle, return 0 as default flags
    if client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *flags = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuStreamGetCtx(stream: u64, pctx: *mut u64) -> u32 {
    let client = get_client();
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_stream = match client.handles.streams.to_remote(stream) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let payload = remote_stream.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::StreamGetCtx, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let remote_ctx = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                // Translate remote context handle back to local handle.
                let local_ctx = if remote_ctx == 0 {
                    0
                } else {
                    client.handles.contexts.to_local(remote_ctx).unwrap_or(0)
                };
                unsafe { *pctx = local_ctx };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate handle, return 0 (no context info in stub mode)
    if client.handles.streams.to_remote(stream).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *pctx = 0 };
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
// cuLaunchCooperativeKernel
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuLaunchCooperativeKernel(
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
        // Same payload format as LaunchKernel
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

        // Serialize kernel parameters (same logic as cuLaunchKernel).
        const MAX_KERNEL_PARAMS: u32 = 1024;
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

        if let Ok((_hdr, resp)) = client.send_request(MessageType::LaunchCooperativeKernel, &payload) {
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
    // Stub: accept the cooperative launch without executing
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// cuDeviceGetPCIBusId
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetPCIBusId(pci_bus_id: *mut u8, len: i32, dev: i32) -> u32 {
    if pci_bus_id.is_null() || len < 13 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        // Payload: [4B len (i32 LE)][4B dev (i32 LE)]
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&len.to_le_bytes());
        payload[4..8].copy_from_slice(&dev.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGetPCIBusId, &payload) {
            let result = parse_result(&resp);
            if result == CUDA_SUCCESS && resp.len() > 4 {
                let data = &resp[4..];
                let copy_len = data.len().min(len as usize);
                unsafe {
                    ptr::copy_nonoverlapping(data.as_ptr(), pci_bus_id, copy_len);
                }
            }
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: generate synthetic PCI bus ID
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let bus_id = format!("0000:{:02x}:00.0\0", dev + 1);
    let bytes = bus_id.as_bytes();
    if (len as usize) < bytes.len() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), pci_bus_id, bytes.len());
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// cuDeviceGetByPCIBusId
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetByPCIBusId(dev: *mut i32, pci_bus_id: *const u8) -> u32 {
    if dev.is_null() || pci_bus_id.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        // Payload: NUL-terminated string
        let c_str = unsafe { std::ffi::CStr::from_ptr(pci_bus_id as *const i8) };
        let mut payload = c_str.to_bytes().to_vec();
        payload.push(0); // NUL terminator
        if let Ok((_hdr, resp)) = client.send_request(MessageType::DeviceGetByPCIBusId, &payload) {
            let result = parse_result(&resp);
            if result == CUDA_SUCCESS && resp.len() >= 8 {
                let device = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                unsafe { *dev = device; }
            }
            return result;
        }
        // Transport error -- fall through to stub
    }
    // Stub: parse the PCI bus ID
    let c_str = unsafe { std::ffi::CStr::from_ptr(pci_bus_id as *const i8) };
    let bus_id_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return CUDA_ERROR_INVALID_VALUE,
    };
    let parts: Vec<&str> = bus_id_str.split(':').collect();
    if parts.len() != 3 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let bus = match u8::from_str_radix(parts[1], 16) {
        Ok(b) => b,
        Err(_) => return CUDA_ERROR_INVALID_VALUE,
    };
    if bus == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let device = (bus as i32) - 1;
    // Stub only has 1 device (device 0)
    if device < 0 || device >= 1 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *dev = device; }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// MemcpyDtoD
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemcpyDtoD(dst: u64, src: u64, byte_count: usize) -> u32 {
    let client = get_client();
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
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
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    if src_host.is_null() {
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
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    if dst_host.is_null() {
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
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
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
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
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
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
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
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
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
// MemsetD16 / MemsetD16Async
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemsetD16(dst_device: u64, value: u16, count: usize) -> u32 {
    let client = get_client();
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
    }
    // Connected: send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst_device) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        // Payload: [8B dst][4B value(u32 zero-extended)][8B count] = 20B
        let mut payload = [0u8; 20];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..12].copy_from_slice(&(value as u32).to_le_bytes());
        payload[12..20].copy_from_slice(&(count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD16, &payload) {
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
pub extern "C" fn ol_cuMemsetD16Async(dst_device: u64, value: u16, count: usize, stream: u64) -> u32 {
    let client = get_client();
    // CUDA: zero-count memset is a successful no-op
    if count == 0 {
        return CUDA_SUCCESS;
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
        // Payload: [8B dst][4B value][8B count][8B stream] = 28B
        let mut payload = [0u8; 28];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..12].copy_from_slice(&(value as u32).to_le_bytes());
        payload[12..20].copy_from_slice(&(count as u64).to_le_bytes());
        payload[20..28].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemsetD16Async, &payload) {
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
// Memcpy (generic, direction-agnostic)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuMemcpy(dst: u64, src: u64, byte_count: usize) -> u32 {
    let client = get_client();
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    // TODO(Phase 2): cuMemcpy is direction-agnostic under UVA. Currently only
    // handles device-to-device because both pointers go through device_ptrs
    // handle translation. Host pointer support needs a separate path.
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
        // Payload: [8B dst][8B src][8B size] = 24B
        let mut payload = [0u8; 24];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_src.to_le_bytes());
        payload[16..24].copy_from_slice(&(byte_count as u64).to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::Memcpy, &payload) {
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

#[no_mangle]
pub extern "C" fn ol_cuMemcpyAsync(dst: u64, src: u64, byte_count: usize, stream: u64) -> u32 {
    let client = get_client();
    // CUDA: zero-size memcpy is a successful no-op
    if byte_count == 0 {
        return CUDA_SUCCESS;
    }
    // Connected: translate all handles and send to server
    if client.connected.load(Ordering::Acquire) {
        let remote_dst = match client.handles.device_ptrs.to_remote(dst) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        let remote_src = match client.handles.device_ptrs.to_remote(src) {
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
        // Payload: [8B dst][8B src][8B size][8B stream] = 32B
        let mut payload = [0u8; 32];
        payload[0..8].copy_from_slice(&remote_dst.to_le_bytes());
        payload[8..16].copy_from_slice(&remote_src.to_le_bytes());
        payload[16..24].copy_from_slice(&(byte_count as u64).to_le_bytes());
        payload[24..32].copy_from_slice(&remote_stream.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemcpyAsync, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate all handles
    if client.handles.device_ptrs.to_remote(dst).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if client.handles.device_ptrs.to_remote(src).is_none() {
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
// Host memory utilities
// ---------------------------------------------------------------------------

/// Tracking map for registered host memory: ptr -> (size, flags) (stub mode).
static REGISTERED_HOST: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u64, (usize, u32)>>> =
    std::sync::OnceLock::new();

fn registered_host() -> &'static std::sync::Mutex<std::collections::HashMap<u64, (usize, u32)>> {
    REGISTERED_HOST.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[no_mangle]
pub extern "C" fn ol_cuMemHostGetDevicePointer(pdptr: *mut u64, p: *mut u8, flags: u32) -> u32 {
    let client = get_client();
    if pdptr.is_null() || p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let host_ptr = p as u64;
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = host_ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemHostGetDevicePointer, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 12 {
                let dev_ptr = u64::from_le_bytes(resp[4..12].try_into().unwrap());
                unsafe { *pdptr = dev_ptr };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: UVA means device pointer == host pointer, if known.
    let known = host_allocs().lock().unwrap().contains(&host_ptr)
        || registered_host().lock().unwrap().contains_key(&host_ptr);
    if !known {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *pdptr = host_ptr };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemHostGetFlags(p_flags: *mut u32, p: *mut u8) -> u32 {
    let client = get_client();
    if p_flags.is_null() || p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let host_ptr = p as u64;
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let payload = host_ptr.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemHostGetFlags, &payload) {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                let flags = u32::from_le_bytes(resp[4..8].try_into().unwrap());
                unsafe { *p_flags = flags };
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: check registered host (has explicit flags), then host_allocs (flags = 0).
    if let Some(&(_size, flags)) = registered_host().lock().unwrap().get(&host_ptr) {
        unsafe { *p_flags = flags };
        return CUDA_SUCCESS;
    }
    if host_allocs().lock().unwrap().contains(&host_ptr) {
        unsafe { *p_flags = 0 };
        return CUDA_SUCCESS;
    }
    CUDA_ERROR_INVALID_VALUE
}

#[no_mangle]
pub extern "C" fn ol_cuMemHostRegister(p: *mut u8, byte_size: usize, flags: u32) -> u32 {
    let client = get_client();
    if p.is_null() || byte_size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let host_ptr = p as u64;
    // Connected: tell the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = host_ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&(byte_size as u64).to_le_bytes());
        payload.extend_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemHostRegister, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: track in local map.
    let mut map = registered_host().lock().unwrap();
    if map.contains_key(&host_ptr) || host_allocs().lock().unwrap().contains(&host_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    map.insert(host_ptr, (byte_size, flags));
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuMemHostUnregister(p: *mut u8) -> u32 {
    let client = get_client();
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let host_ptr = p as u64;
    // Connected: tell the server
    if client.connected.load(Ordering::Acquire) {
        let payload = host_ptr.to_le_bytes();
        if let Ok((_hdr, resp)) = client.send_request(MessageType::MemHostUnregister, &payload) {
            return parse_result(&resp);
        }
        // Transport error -- fall through to stub
    }
    // Stub: remove from local map.
    let mut map = registered_host().lock().unwrap();
    if map.remove(&host_ptr).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
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

// ---------------------------------------------------------------------------
// Peer access
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ol_cuDeviceCanAccessPeer(
    can_access_peer: *mut i32,
    dev: i32,
    peer_dev: i32,
) -> u32 {
    let client = get_client();
    if can_access_peer.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&dev.to_le_bytes());
        payload[4..8].copy_from_slice(&peer_dev.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::DeviceCanAccessPeer, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *can_access_peer = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: self-peer returns 0 per CUDA spec, invalid devices return error
    if dev < 0 || dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if peer_dev < 0 || peer_dev >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // Single-device stub: dev == peer_dev == 0 -> 0 per spec
    unsafe { *can_access_peer = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuDeviceGetP2PAttribute(
    value: *mut i32,
    attrib: i32,
    src_device: i32,
    dst_device: i32,
) -> u32 {
    let client = get_client();
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&attrib.to_le_bytes());
        payload[4..8].copy_from_slice(&src_device.to_le_bytes());
        payload[8..12].copy_from_slice(&dst_device.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::DeviceGetP2PAttribute, &payload)
        {
            let result = parse_result(&resp);
            if result != CUDA_SUCCESS {
                return result;
            }
            if resp.len() >= 8 {
                unsafe {
                    *value = i32::from_le_bytes(resp[4..8].try_into().unwrap());
                }
                return CUDA_SUCCESS;
            }
        }
        // Transport error -- fall through to stub
    }
    // Stub: validate devices, return 0 for all attributes
    if src_device < 0 || src_device >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if dst_device < 0 || dst_device >= 1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe { *value = 0 };
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxEnablePeerAccess(peer_ctx: u64, flags: u32) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = match client.handles.contexts.to_remote(peer_ctx) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_CONTEXT,
        };
        let mut payload = [0u8; 12];
        payload[..8].copy_from_slice(&remote_ctx.to_le_bytes());
        payload[8..12].copy_from_slice(&flags.to_le_bytes());
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::CtxEnablePeerAccess, &payload)
        {
            return parse_result(&resp);
        }
        return CUDA_ERROR_UNKNOWN;
    }
    // Stub: validate context handle and track state
    if client.handles.contexts.to_remote(peer_ctx).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    let mut peers = stub_peer_access().lock().unwrap();
    if !peers.insert(peer_ctx) {
        return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn ol_cuCtxDisablePeerAccess(peer_ctx: u64) -> u32 {
    let client = get_client();
    // Connected: ask the server
    if client.connected.load(Ordering::Acquire) {
        let remote_ctx = match client.handles.contexts.to_remote(peer_ctx) {
            Some(r) => r,
            None => return CUDA_ERROR_INVALID_CONTEXT,
        };
        let payload = remote_ctx.to_le_bytes();
        if let Ok((_hdr, resp)) =
            client.send_request(MessageType::CtxDisablePeerAccess, &payload)
        {
            return parse_result(&resp);
        }
        return CUDA_ERROR_UNKNOWN;
    }
    // Stub: validate context handle and check state
    if client.handles.contexts.to_remote(peer_ctx).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    let mut peers = stub_peer_access().lock().unwrap();
    if !peers.remove(&peer_ctx) {
        return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
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
    fn test_ol_cu_module_load_data_ex_stub() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        let options: [i32; 2] = [0, 7]; // CU_JIT_MAX_REGISTERS, CU_JIT_OPTIMIZATION_LEVEL
        let values: [u64; 2] = [32, 4];
        let result = ol_cuModuleLoadDataEx(
            &mut module,
            data.as_ptr(),
            data.len(),
            2,
            options.as_ptr(),
            values.as_ptr(),
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(module, 0);
        // Clean up
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_module_load_data_ex_null_module() {
        let data = [0u8; 16];
        let result = ol_cuModuleLoadDataEx(
            ptr::null_mut(),
            data.as_ptr(),
            data.len(),
            0,
            ptr::null(),
            ptr::null(),
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_module_load_data_ex_no_options() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        let result = ol_cuModuleLoadDataEx(
            &mut module,
            data.as_ptr(),
            data.len(),
            0,
            ptr::null(),
            ptr::null(),
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(module, 0);
        // Module should be usable
        let mut func: u64 = 0;
        let name = b"test_kern\0";
        let r = ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8);
        assert_eq!(r, CUDA_SUCCESS);
        let _ = ol_cuModuleUnload(module);
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

    // -- Stream priority/flags/ctx tests --

    #[test]
    fn test_ol_cu_stream_create_with_priority() {
        let mut stream: u64 = 0;
        let result = ol_cuStreamCreateWithPriority(&mut stream, 0, -1);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(stream, 0);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_create_with_priority_null_ptr() {
        let result = ol_cuStreamCreateWithPriority(ptr::null_mut(), 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_get_priority() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreateWithPriority(&mut stream, 0, 0), CUDA_SUCCESS);
        let mut priority: i32 = -999;
        assert_eq!(ol_cuStreamGetPriority(stream, &mut priority), CUDA_SUCCESS);
        // In stub mode, priority is 0 (server not connected)
        assert_eq!(priority, 0);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_priority_null_ptr() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let result = ol_cuStreamGetPriority(stream, ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_priority_invalid() {
        let mut priority: i32 = 0;
        assert_eq!(ol_cuStreamGetPriority(0xDEAD, &mut priority), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_get_flags() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let mut flags: u32 = 999;
        assert_eq!(ol_cuStreamGetFlags(stream, &mut flags), CUDA_SUCCESS);
        // In stub mode, flags default to 0
        assert_eq!(flags, 0);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_flags_null_ptr() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let result = ol_cuStreamGetFlags(stream, ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_flags_invalid() {
        let mut flags: u32 = 0;
        assert_eq!(ol_cuStreamGetFlags(0xDEAD, &mut flags), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_get_ctx() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let mut ctx: u64 = 999;
        assert_eq!(ol_cuStreamGetCtx(stream, &mut ctx), CUDA_SUCCESS);
        // In stub mode, ctx is 0
        assert_eq!(ctx, 0);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_ctx_null_ptr() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let result = ol_cuStreamGetCtx(stream, ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_get_ctx_invalid() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuStreamGetCtx(0xDEAD, &mut ctx), CUDA_ERROR_INVALID_VALUE);
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
        // CUDA: zero-size memcpy is a successful no-op
        assert_eq!(ol_cuMemcpyDtoD(0x1000, 0x2000, 0), CUDA_SUCCESS);
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

    // -- Primary context tests --

    #[test]
    fn test_ol_cu_primary_ctx_retain() {
        let mut ctx: u64 = 0;
        let result = ol_cuDevicePrimaryCtxRetain(&mut ctx, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(ctx, 0);
    }

    #[test]
    fn test_ol_cu_primary_ctx_retain_idempotent() {
        let mut ctx1: u64 = 0;
        let mut ctx2: u64 = 0;
        assert_eq!(ol_cuDevicePrimaryCtxRetain(&mut ctx1, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuDevicePrimaryCtxRetain(&mut ctx2, 0), CUDA_SUCCESS);
        assert_eq!(ctx1, ctx2, "repeated retain must return same handle");
    }

    #[test]
    fn test_ol_cu_primary_ctx_retain_null_ptr() {
        assert_eq!(ol_cuDevicePrimaryCtxRetain(ptr::null_mut(), 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_primary_ctx_retain_invalid_device() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuDevicePrimaryCtxRetain(&mut ctx, 99), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_primary_ctx_release() {
        assert_eq!(ol_cuDevicePrimaryCtxRelease(0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_primary_ctx_release_invalid_device() {
        assert_eq!(ol_cuDevicePrimaryCtxRelease(99), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_primary_ctx_get_state() {
        let mut flags: u32 = 0xFF;
        let mut active: i32 = -1;
        let result = ol_cuDevicePrimaryCtxGetState(0, &mut flags, &mut active);
        assert_eq!(result, CUDA_SUCCESS);
        // In stub mode, should be inactive with no flags
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
    }

    #[test]
    fn test_ol_cu_primary_ctx_get_state_null_ptrs() {
        let mut active: i32 = 0;
        assert_eq!(
            ol_cuDevicePrimaryCtxGetState(0, ptr::null_mut(), &mut active),
            CUDA_ERROR_INVALID_VALUE
        );
        let mut flags: u32 = 0;
        assert_eq!(
            ol_cuDevicePrimaryCtxGetState(0, &mut flags, ptr::null_mut()),
            CUDA_ERROR_INVALID_VALUE
        );
    }

    #[test]
    fn test_ol_cu_primary_ctx_get_state_invalid_device() {
        let mut flags: u32 = 0;
        let mut active: i32 = 0;
        assert_eq!(
            ol_cuDevicePrimaryCtxGetState(99, &mut flags, &mut active),
            CUDA_ERROR_INVALID_DEVICE
        );
    }

    #[test]
    fn test_ol_cu_primary_ctx_set_flags() {
        assert_eq!(ol_cuDevicePrimaryCtxSetFlags(0, 0x04), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_primary_ctx_set_flags_invalid_device() {
        assert_eq!(ol_cuDevicePrimaryCtxSetFlags(99, 0), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_primary_ctx_reset() {
        assert_eq!(ol_cuDevicePrimaryCtxReset(0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_primary_ctx_reset_invalid_device() {
        assert_eq!(ol_cuDevicePrimaryCtxReset(99), CUDA_ERROR_INVALID_DEVICE);
    }

    // -- FuncGetAttribute tests --

    #[test]
    fn test_ol_cu_func_get_attribute_max_threads() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let mut val: i32 = -1;
        let result = ol_cuFuncGetAttribute(&mut val, 0, func); // MAX_THREADS_PER_BLOCK
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 1024);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_get_attribute_num_regs() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let mut val: i32 = -1;
        let result = ol_cuFuncGetAttribute(&mut val, 4, func); // NUM_REGS
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 32);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_get_attribute_null_ptr() {
        let result = ol_cuFuncGetAttribute(ptr::null_mut(), 0, 1);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_func_get_attribute_invalid_func() {
        let mut val: i32 = -1;
        let result = ol_cuFuncGetAttribute(&mut val, 0, 0xDEAD);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_func_get_attribute_invalid_attrib() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let mut val: i32 = -1;
        let result = ol_cuFuncGetAttribute(&mut val, 9999, func);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuModuleUnload(module);
    }

    // --- Context push/pop tests ---

    #[test]
    fn test_ol_cu_ctx_push_current_stub() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        assert_ne!(ctx, 0);
        let result = ol_cuCtxPushCurrent_v2(ctx);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_push_invalid() {
        let result = ol_cuCtxPushCurrent_v2(0xDEAD_BEEF);
        assert_eq!(result, CUDA_ERROR_INVALID_CONTEXT);
    }

    #[test]
    fn test_ol_cu_ctx_pop_current_stub() {
        let mut popped: u64 = 0xFFFF;
        let result = ol_cuCtxPopCurrent_v2(&mut popped);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(popped, 0);
    }

    // --- Context API version tests ---

    #[test]
    fn test_ol_cu_ctx_get_api_version_stub() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        let mut version: u32 = 0;
        let result = ol_cuCtxGetApiVersion(ctx, &mut version);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(version, 12000);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_get_api_version_null_ptr() {
        let result = ol_cuCtxGetApiVersion(1, ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // --- Context limit tests ---

    #[test]
    fn test_ol_cu_ctx_get_limit_stack_size() {
        let mut value: u64 = 0;
        let result = ol_cuCtxGetLimit(&mut value, 0x00);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(value, 1024);
    }

    #[test]
    fn test_ol_cu_ctx_get_limit_null_ptr() {
        let result = ol_cuCtxGetLimit(ptr::null_mut(), 0x00);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_ctx_get_limit_unknown() {
        let mut value: u64 = 0;
        let result = ol_cuCtxGetLimit(&mut value, 0xFF);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_ctx_set_limit() {
        let result = ol_cuCtxSetLimit(0x00, 4096);
        assert_eq!(result, CUDA_SUCCESS);
    }

    // --- Stream priority range tests ---

    #[test]
    fn test_ol_cu_ctx_get_stream_priority_range() {
        let mut least: i32 = -99;
        let mut greatest: i32 = 99;
        let result = ol_cuCtxGetStreamPriorityRange(&mut least, &mut greatest);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(least, 0);
        assert_eq!(greatest, -1);
    }

    #[test]
    fn test_ol_cu_ctx_get_stream_priority_range_null_ptrs() {
        // Both null is valid -- CUDA allows querying one at a time
        let result = ol_cuCtxGetStreamPriorityRange(ptr::null_mut(), ptr::null_mut());
        assert_eq!(result, CUDA_SUCCESS);
    }

    // --- Context flags tests ---

    #[test]
    fn test_ol_cu_ctx_get_flags() {
        let mut flags: u32 = 0xFF;
        let result = ol_cuCtxGetFlags(&mut flags);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(flags, 0);
    }

    #[test]
    fn test_ol_cu_ctx_get_flags_null_ptr() {
        let result = ol_cuCtxGetFlags(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // --- Occupancy tests ---

    /// Helper: create module + function for occupancy tests.
    fn setup_func_for_occupancy() -> u64 {
        let mut module: u64 = 0;
        let data = b"ptx";
        assert_eq!(
            ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()),
            CUDA_SUCCESS
        );
        let mut func: u64 = 0;
        assert_eq!(
            ol_cuModuleGetFunction(&mut func, module, b"kern\0".as_ptr() as *const i8),
            CUDA_SUCCESS
        );
        func
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks() {
        let func = setup_func_for_occupancy();
        let mut num_blocks: i32 = 0;
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut num_blocks, func, 256, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(num_blocks, 8); // 2048/256
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks_null_ptr() {
        let func = setup_func_for_occupancy();
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(ptr::null_mut(), func, 256, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks_invalid_func() {
        let mut num_blocks: i32 = 0;
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut num_blocks, 0xDEAD, 256, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks_zero_block_size() {
        let func = setup_func_for_occupancy();
        let mut num_blocks: i32 = 0;
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut num_blocks, func, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks_with_flags() {
        let func = setup_func_for_occupancy();
        let mut num_blocks: i32 = 0;
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&mut num_blocks, func, 512, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(num_blocks, 4); // 2048/512
    }

    #[test]
    fn test_ol_cu_occupancy_max_active_blocks_with_flags_null_ptr() {
        let result = ol_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ptr::null_mut(), 1, 256, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;
        let result = ol_cuOccupancyMaxPotentialBlockSize(
            &mut min_grid,
            &mut block_sz,
            func,
            ptr::null(),
            0,
            0,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 256);
        assert_eq!(min_grid, 656); // (2048/256)*82
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_with_callback() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;
        // Real callback that returns 0 dynamic shared memory
        unsafe extern "C" fn zero_smem(_block_size: i32) -> usize { 0 }
        let cb_ptr = zero_smem as *const std::ffi::c_void;
        let result = ol_cuOccupancyMaxPotentialBlockSize(
            &mut min_grid,
            &mut block_sz,
            func,
            cb_ptr,
            0,
            0,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 256);
        assert_eq!(min_grid, 656);
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_null_ptrs() {
        let result = ol_cuOccupancyMaxPotentialBlockSize(
            ptr::null_mut(),
            ptr::null_mut(),
            1,
            ptr::null(),
            0,
            0,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_with_limit() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;
        let result = ol_cuOccupancyMaxPotentialBlockSize(
            &mut min_grid,
            &mut block_sz,
            func,
            ptr::null(),
            0,
            128,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 128);
        assert_eq!(min_grid, 1312); // (2048/128)*82
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_with_flags() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;
        let result = ol_cuOccupancyMaxPotentialBlockSizeWithFlags(
            &mut min_grid,
            &mut block_sz,
            func,
            ptr::null(),
            0,
            0,
            0,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 256);
        assert_eq!(min_grid, 656);
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_with_flags_null_ptrs() {
        let result = ol_cuOccupancyMaxPotentialBlockSizeWithFlags(
            ptr::null_mut(),
            ptr::null_mut(),
            1,
            ptr::null(),
            0,
            0,
            0,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_occupancy_max_potential_block_size_with_flags_invalid_func() {
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;
        let result = ol_cuOccupancyMaxPotentialBlockSizeWithFlags(
            &mut min_grid,
            &mut block_sz,
            0xDEAD,
            ptr::null(),
            0,
            0,
            0,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // --- Peer access tests ---

    #[test]
    fn test_ol_cu_device_can_access_peer_same_device() {
        let mut can_access: i32 = -1;
        let result = ol_cuDeviceCanAccessPeer(&mut can_access, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(can_access, 0); // self-peer returns 0
    }

    #[test]
    fn test_ol_cu_device_can_access_peer_null_ptr() {
        let result = ol_cuDeviceCanAccessPeer(ptr::null_mut(), 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_can_access_peer_invalid_device() {
        let mut can_access: i32 = -1;
        assert_eq!(ol_cuDeviceCanAccessPeer(&mut can_access, 99, 0), CUDA_ERROR_INVALID_DEVICE);
        assert_eq!(ol_cuDeviceCanAccessPeer(&mut can_access, 0, 99), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_device_get_p2p_attribute_stub() {
        let mut value: i32 = -1;
        let result = ol_cuDeviceGetP2PAttribute(&mut value, 0, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(value, 0); // stub returns 0 for all
    }

    #[test]
    fn test_ol_cu_device_get_p2p_attribute_null_ptr() {
        let result = ol_cuDeviceGetP2PAttribute(ptr::null_mut(), 0, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_p2p_attribute_invalid_device() {
        let mut value: i32 = -1;
        assert_eq!(ol_cuDeviceGetP2PAttribute(&mut value, 0, 99, 0), CUDA_ERROR_INVALID_DEVICE);
        assert_eq!(ol_cuDeviceGetP2PAttribute(&mut value, 0, 0, 99), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_ctx_enable_peer_access_valid() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        let result = ol_cuCtxEnablePeerAccess(ctx, 0);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_enable_peer_access_invalid_context() {
        let result = ol_cuCtxEnablePeerAccess(0xDEAD_BEEF, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_CONTEXT);
    }

    #[test]
    fn test_ol_cu_ctx_disable_peer_access_after_enable() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        // Enable first, then disable should succeed
        assert_eq!(ol_cuCtxEnablePeerAccess(ctx, 0), CUDA_SUCCESS);
        let result = ol_cuCtxDisablePeerAccess(ctx);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_disable_peer_access_without_enable() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        // Disable without prior enable should return PEER_ACCESS_NOT_ENABLED
        let result = ol_cuCtxDisablePeerAccess(ctx);
        assert_eq!(result, CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_disable_peer_access_invalid_context() {
        let result = ol_cuCtxDisablePeerAccess(0xDEAD_BEEF);
        assert_eq!(result, CUDA_ERROR_INVALID_CONTEXT);
    }

    #[test]
    fn test_ol_cu_ctx_enable_peer_access_double_enable() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuCtxEnablePeerAccess(ctx, 0), CUDA_SUCCESS);
        // Second enable should return ALREADY_ENABLED
        let result = ol_cuCtxEnablePeerAccess(ctx, 0);
        assert_eq!(result, CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
        let _ = ol_cuCtxDisablePeerAccess(ctx);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_ctx_peer_access_enable_disable_reenable() {
        let mut ctx: u64 = 0;
        assert_eq!(ol_cuCtxCreate_v2(&mut ctx, 0, 0), CUDA_SUCCESS);
        // Enable -> Disable -> Re-enable should all succeed
        assert_eq!(ol_cuCtxEnablePeerAccess(ctx, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuCtxDisablePeerAccess(ctx), CUDA_SUCCESS);
        assert_eq!(ol_cuCtxEnablePeerAccess(ctx, 0), CUDA_SUCCESS);
        let _ = ol_cuCtxDisablePeerAccess(ctx);
        let _ = ol_cuCtxDestroy_v2(ctx);
    }

    #[test]
    fn test_ol_cu_occupancy_callback_invoked() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;

        // A real callback that returns 1024 bytes of dynamic shared memory
        unsafe extern "C" fn smem_callback(_block_size: i32) -> usize {
            1024
        }

        let cb_ptr = smem_callback as *const std::ffi::c_void;
        let result = ol_cuOccupancyMaxPotentialBlockSize(
            &mut min_grid,
            &mut block_sz,
            func,
            cb_ptr,
            0,
            0,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 256);
        // With callback invoked, grid size should still be computed correctly
        assert!(min_grid > 0);
    }

    #[test]
    fn test_ol_cu_occupancy_callback_with_flags_invoked() {
        let func = setup_func_for_occupancy();
        let mut min_grid: i32 = 0;
        let mut block_sz: i32 = 0;

        unsafe extern "C" fn smem_callback(_block_size: i32) -> usize {
            512
        }

        let cb_ptr = smem_callback as *const std::ffi::c_void;
        let result = ol_cuOccupancyMaxPotentialBlockSizeWithFlags(
            &mut min_grid,
            &mut block_sz,
            func,
            cb_ptr,
            0,
            0,
            0,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(block_sz, 256);
        assert!(min_grid > 0);
    }

    // -- FuncSetAttribute tests --

    #[test]
    fn test_ol_cu_func_set_attribute_max_dynamic_shared() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        // attrib 8 = MAX_DYNAMIC_SHARED_SIZE_BYTES
        let result = ol_cuFuncSetAttribute(func, 8, 65536);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_set_attribute_read_only_rejected() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        // attrib 0 = MAX_THREADS_PER_BLOCK (read-only)
        let result = ol_cuFuncSetAttribute(func, 0, 512);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_set_attribute_invalid_func() {
        let result = ol_cuFuncSetAttribute(0xDEAD, 8, 100);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_func_set_attribute_invalid_attrib() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuFuncSetAttribute(func, 9999, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuModuleUnload(module);
    }

    // -- MemGetAddressRange tests --

    #[test]
    fn test_ol_cu_mem_get_address_range_basic() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 4096), CUDA_SUCCESS);
        assert_ne!(dptr, 0);

        let mut base: u64 = 0;
        let mut size: usize = 0;
        let result = ol_cuMemGetAddressRange_v2(&mut base, &mut size, dptr);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(base, dptr);
        assert_eq!(size, 4096);
        let _ = ol_cuMemFree_v2(dptr);
    }

    // --- Pointer attribute tests ---

    #[test]
    fn test_ol_cu_pointer_get_attribute_memory_type_device() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);
        let mut val: u32 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u32 as *mut u8,
            2, // MEMORY_TYPE
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 2); // CU_MEMORYTYPE_DEVICE
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_device_pointer() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 512), CUDA_SUCCESS);
        let mut val: u64 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u64 as *mut u8,
            3, // DEVICE_POINTER
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, dptr);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_mem_get_address_range_null_pbase() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);

        let mut size: usize = 0;
        let result = ol_cuMemGetAddressRange_v2(ptr::null_mut(), &mut size, dptr);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_host_pointer_for_device() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 256), CUDA_SUCCESS);
        let mut val: u64 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u64 as *mut u8,
            4, // HOST_POINTER
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 0); // device pointers have no host pointer
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_is_managed() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 256), CUDA_SUCCESS);
        let mut val: u32 = 99;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u32 as *mut u8,
            6, // IS_MANAGED
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 0);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_device_ordinal() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 256), CUDA_SUCCESS);
        let mut val: i32 = -1;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut i32 as *mut u8,
            8, // DEVICE_ORDINAL
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 0);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_null_data() {
        let result = ol_cuPointerGetAttribute(ptr::null_mut(), 2, 1);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_unknown_ptr() {
        let mut val: u64 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u64 as *mut u8,
            2,
            0xDEAD_BEEF,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_invalid_attrib() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 256), CUDA_SUCCESS);
        let mut val: u64 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u64 as *mut u8,
            9999,
            dptr,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_pointer_get_attributes_multiple() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);

        let attrs: [i32; 2] = [2, 8]; // MEMORY_TYPE, DEVICE_ORDINAL
        let mut val0: u32 = 0;
        let mut val1: i32 = -1;
        let mut data_ptrs: [*mut u8; 2] = [
            &mut val0 as *mut u32 as *mut u8,
            &mut val1 as *mut i32 as *mut u8,
        ];

        let result = ol_cuPointerGetAttributes(
            2,
            attrs.as_ptr(),
            data_ptrs.as_mut_ptr(),
            dptr,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val0, 2); // DEVICE
        assert_eq!(val1, 0); // ordinal 0
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_mem_get_address_range_null_psize() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);

        let mut base: u64 = 0;
        let result = ol_cuMemGetAddressRange_v2(&mut base, ptr::null_mut(), dptr);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(base, dptr);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_mem_get_address_range_both_null() {
        let mut dptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dptr, 1024), CUDA_SUCCESS);

        let result = ol_cuMemGetAddressRange_v2(ptr::null_mut(), ptr::null_mut(), dptr);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(dptr);
    }

    #[test]
    fn test_ol_cu_mem_get_address_range_invalid_ptr() {
        let result = ol_cuMemGetAddressRange_v2(ptr::null_mut(), ptr::null_mut(), 0xDEAD);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attributes_null_attrs() {
        let mut data_ptr: *mut u8 = ptr::null_mut();
        let result = ol_cuPointerGetAttributes(
            1,
            ptr::null(),
            &mut data_ptr,
            1,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attributes_null_data() {
        let attrs: [i32; 1] = [2];
        let result = ol_cuPointerGetAttributes(
            1,
            attrs.as_ptr(),
            ptr::null_mut(),
            1,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attributes_zero_count() {
        let attrs: [i32; 1] = [2];
        let mut data_ptr: *mut u8 = ptr::null_mut();
        let result = ol_cuPointerGetAttributes(
            0,
            attrs.as_ptr(),
            &mut data_ptr,
            1,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attributes_unknown_ptr() {
        let attrs: [i32; 1] = [2];
        let mut val: u32 = 0;
        let mut data_ptrs: [*mut u8; 1] = [&mut val as *mut u32 as *mut u8];
        let result = ol_cuPointerGetAttributes(
            1,
            attrs.as_ptr(),
            data_ptrs.as_mut_ptr(),
            0xDEAD_BEEF,
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_pointer_get_attribute_memory_type_host() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 512), CUDA_SUCCESS);
        assert!(!ptr.is_null());
        let host_val = ptr as u64;
        let mut val: u32 = 0;
        let result = ol_cuPointerGetAttribute(
            &mut val as *mut u32 as *mut u8,
            2, // MEMORY_TYPE
            host_val,
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(val, 1); // CU_MEMORYTYPE_HOST
        let _ = ol_cuMemFreeHost(ptr);
    }

    // -- MemsetD16 tests --

    #[test]
    fn test_ol_cu_memset_d16() {
        let mut ptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut ptr, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemsetD16(ptr, 0xBEEF, 512), CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(ptr);
    }

    #[test]
    fn test_ol_cu_memset_d16_invalid_ptr() {
        assert_eq!(ol_cuMemsetD16(0xBAAD, 0xFF, 16), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_memset_d16_zero_count() {
        assert_eq!(ol_cuMemsetD16(0x1000, 0xFF, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_memset_d16_async() {
        let mut ptr: u64 = 0;
        let mut stream: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut ptr, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuMemsetD16Async(ptr, 0xCAFE, 512, stream), CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
        let _ = ol_cuMemFree_v2(ptr);
    }

    #[test]
    fn test_ol_cu_memset_d16_async_invalid_stream() {
        let mut ptr: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut ptr, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemsetD16Async(ptr, 0xFF, 512, 0xBAD), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(ptr);
    }

    #[test]
    fn test_ol_cu_memset_d16_async_zero_count() {
        assert_eq!(ol_cuMemsetD16Async(0x1000, 0xFF, 0, 0), CUDA_SUCCESS);
    }

    // -- Memcpy (generic) tests --

    #[test]
    fn test_ol_cu_memcpy() {
        let mut src: u64 = 0;
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpy(dst, src, 512), CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(src);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_invalid_dst() {
        let mut src: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpy(0xBAAD, src, 512), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(src);
    }

    #[test]
    fn test_ol_cu_memcpy_invalid_src() {
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpy(dst, 0xBAAD, 512), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_zero_size() {
        assert_eq!(ol_cuMemcpy(0x1000, 0x2000, 0), CUDA_SUCCESS);
    }

    // -- MemcpyAsync (generic) tests --

    #[test]
    fn test_ol_cu_memcpy_async() {
        let mut src: u64 = 0;
        let mut dst: u64 = 0;
        let mut stream: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpyAsync(dst, src, 512, stream), CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
        let _ = ol_cuMemFree_v2(src);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_async_invalid_stream() {
        let mut src: u64 = 0;
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemcpyAsync(dst, src, 512, 0xBAD), CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuMemFree_v2(src);
        let _ = ol_cuMemFree_v2(dst);
    }

    #[test]
    fn test_ol_cu_memcpy_async_zero_size() {
        assert_eq!(ol_cuMemcpyAsync(0x1000, 0x2000, 0, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_memcpy_async_default_stream() {
        let mut src: u64 = 0;
        let mut dst: u64 = 0;
        assert_eq!(ol_cuMemAlloc_v2(&mut src, 1024), CUDA_SUCCESS);
        assert_eq!(ol_cuMemAlloc_v2(&mut dst, 1024), CUDA_SUCCESS);
        // Stream 0 = default, always valid.
        assert_eq!(ol_cuMemcpyAsync(dst, src, 512, 0), CUDA_SUCCESS);
        let _ = ol_cuMemFree_v2(src);
        let _ = ol_cuMemFree_v2(dst);
    }

    // -- MemHostGetDevicePointer tests --

    #[test]
    fn test_ol_cu_mem_host_get_device_pointer() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 256), CUDA_SUCCESS);
        let mut dev_ptr: u64 = 0;
        assert_eq!(ol_cuMemHostGetDevicePointer(&mut dev_ptr, ptr, 0), CUDA_SUCCESS);
        assert_eq!(dev_ptr, ptr as u64);
        let _ = ol_cuMemFreeHost(ptr);
    }

    #[test]
    fn test_ol_cu_mem_host_get_device_pointer_null_out() {
        let fake = 0x1000 as *mut u8;
        assert_eq!(ol_cuMemHostGetDevicePointer(ptr::null_mut(), fake, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_get_device_pointer_null_input() {
        let mut dev_ptr: u64 = 0;
        assert_eq!(ol_cuMemHostGetDevicePointer(&mut dev_ptr, ptr::null_mut(), 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_get_device_pointer_unknown() {
        let mut dev_ptr: u64 = 0;
        assert_eq!(ol_cuMemHostGetDevicePointer(&mut dev_ptr, 0xDEAD as *mut u8, 0), CUDA_ERROR_INVALID_VALUE);
    }

    // -- MemHostGetFlags tests --

    #[test]
    fn test_ol_cu_mem_host_get_flags_alloc() {
        let mut ptr: *mut u8 = ptr::null_mut();
        assert_eq!(ol_cuMemAllocHost(&mut ptr, 128), CUDA_SUCCESS);
        let mut flags: u32 = 0xFF;
        assert_eq!(ol_cuMemHostGetFlags(&mut flags, ptr), CUDA_SUCCESS);
        assert_eq!(flags, 0); // default for cuMemAllocHost
        let _ = ol_cuMemFreeHost(ptr);
    }

    #[test]
    fn test_ol_cu_mem_host_get_flags_null_out() {
        let fake = 0x1000 as *mut u8;
        assert_eq!(ol_cuMemHostGetFlags(ptr::null_mut(), fake), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_get_flags_unknown() {
        let mut flags: u32 = 0;
        assert_eq!(ol_cuMemHostGetFlags(&mut flags, 0xDEAD as *mut u8), CUDA_ERROR_INVALID_VALUE);
    }

    // -- MemHostRegister / MemHostUnregister tests --

    #[test]
    fn test_ol_cu_mem_host_register_and_unregister() {
        // Allocate real memory so the pointer is valid.
        let layout = std::alloc::Layout::from_size_align(4096, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());

        assert_eq!(ol_cuMemHostRegister(ptr, 4096, 0), CUDA_SUCCESS);

        // Should be visible to GetDevicePointer.
        let mut dev_ptr: u64 = 0;
        assert_eq!(ol_cuMemHostGetDevicePointer(&mut dev_ptr, ptr, 0), CUDA_SUCCESS);
        assert_eq!(dev_ptr, ptr as u64);

        // Should be visible to GetFlags (flags = 0).
        let mut flags: u32 = 0xFF;
        assert_eq!(ol_cuMemHostGetFlags(&mut flags, ptr), CUDA_SUCCESS);
        assert_eq!(flags, 0);

        assert_eq!(ol_cuMemHostUnregister(ptr), CUDA_SUCCESS);

        // After unregister, should fail.
        assert_eq!(ol_cuMemHostGetDevicePointer(&mut dev_ptr, ptr, 0), CUDA_ERROR_INVALID_VALUE);

        unsafe { std::alloc::dealloc(ptr, layout) };
    }

    #[test]
    fn test_ol_cu_mem_host_register_null() {
        assert_eq!(ol_cuMemHostRegister(ptr::null_mut(), 4096, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_register_zero_size() {
        let fake = 0x1000 as *mut u8;
        assert_eq!(ol_cuMemHostRegister(fake, 0, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_unregister_null() {
        assert_eq!(ol_cuMemHostUnregister(ptr::null_mut()), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_unregister_not_registered() {
        assert_eq!(ol_cuMemHostUnregister(0xDEAD as *mut u8), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_host_register_with_flags() {
        let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());

        assert_eq!(ol_cuMemHostRegister(ptr, 1024, 0x03), CUDA_SUCCESS);
        let mut flags: u32 = 0;
        assert_eq!(ol_cuMemHostGetFlags(&mut flags, ptr), CUDA_SUCCESS);
        assert_eq!(flags, 0x03);

        assert_eq!(ol_cuMemHostUnregister(ptr), CUDA_SUCCESS);
        unsafe { std::alloc::dealloc(ptr, layout) };
    }

    // -- CtxGetCacheConfig / CtxSetCacheConfig tests --

    #[test]
    fn test_ol_cu_ctx_get_cache_config_default() {
        let mut config: u32 = 0xFF;
        let result = ol_cuCtxGetCacheConfig(&mut config);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(config, 0); // PREFER_NONE
    }

    #[test]
    fn test_ol_cu_ctx_get_cache_config_null_ptr() {
        let result = ol_cuCtxGetCacheConfig(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_ctx_set_cache_config() {
        let result = ol_cuCtxSetCacheConfig(0x01); // PREFER_SHARED
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_ctx_set_cache_config_invalid() {
        let result = ol_cuCtxSetCacheConfig(0x04);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // -- CtxGetSharedMemConfig / CtxSetSharedMemConfig tests --

    #[test]
    fn test_ol_cu_ctx_get_shared_mem_config_default() {
        let mut config: u32 = 0xFF;
        let result = ol_cuCtxGetSharedMemConfig(&mut config);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(config, 0); // DEFAULT_BANK_SIZE
    }

    #[test]
    fn test_ol_cu_ctx_get_shared_mem_config_null_ptr() {
        let result = ol_cuCtxGetSharedMemConfig(ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_ctx_set_shared_mem_config() {
        let result = ol_cuCtxSetSharedMemConfig(0x01); // FOUR_BYTE_BANK_SIZE
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_ctx_set_shared_mem_config_invalid() {
        let result = ol_cuCtxSetSharedMemConfig(0x03);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // -- FuncSetCacheConfig tests --

    #[test]
    fn test_ol_cu_func_set_cache_config() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuFuncSetCacheConfig(func, 0x02); // PREFER_L1
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_set_cache_config_invalid_func() {
        let result = ol_cuFuncSetCacheConfig(0xDEAD, 0x01);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_func_set_cache_config_invalid_config() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuFuncSetCacheConfig(func, 0x04);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuModuleUnload(module);
    }

    // -- FuncSetSharedMemConfig tests --

    #[test]
    fn test_ol_cu_func_set_shared_mem_config() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuFuncSetSharedMemConfig(func, 0x01); // FOUR_BYTE_BANK_SIZE
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_func_set_shared_mem_config_invalid_func() {
        let result = ol_cuFuncSetSharedMemConfig(0xDEAD, 0x01);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_func_set_shared_mem_config_invalid_config() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuFuncSetSharedMemConfig(func, 0x03);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
        let _ = ol_cuModuleUnload(module);
    }

    // --- cuEventRecordWithFlags tests ---

    #[test]
    fn test_ol_cu_event_record_with_flags() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventRecordWithFlags(event, 0, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_event_record_with_flags_nonzero() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventRecordWithFlags(event, 0, 0x01), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_event_record_with_flags_invalid_event() {
        assert_eq!(ol_cuEventRecordWithFlags(0xDEAD, 0, 0), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_event_record_with_flags_with_stream() {
        let mut event: u64 = 0;
        assert_eq!(ol_cuEventCreate(&mut event, 0), CUDA_SUCCESS);
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuEventRecordWithFlags(event, stream, 0), CUDA_SUCCESS);
    }

    // --- cuLaunchCooperativeKernel tests ---

    #[test]
    fn test_ol_cu_launch_cooperative_kernel() {
        let mut module: u64 = 0;
        let data = [0u8; 16];
        assert_eq!(ol_cuModuleLoadData(&mut module, data.as_ptr(), data.len()), CUDA_SUCCESS);
        let mut func: u64 = 0;
        let name = b"kern\0";
        assert_eq!(ol_cuModuleGetFunction(&mut func, module, name.as_ptr() as *const i8), CUDA_SUCCESS);

        let result = ol_cuLaunchCooperativeKernel(
            func, 1, 1, 1, 32, 1, 1, 0, 0,
            std::ptr::null(), 0, std::ptr::null(),
        );
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuModuleUnload(module);
    }

    #[test]
    fn test_ol_cu_launch_cooperative_kernel_invalid_func() {
        let result = ol_cuLaunchCooperativeKernel(
            0xDEAD, 1, 1, 1, 32, 1, 1, 0, 0,
            std::ptr::null(), 0, std::ptr::null(),
        );
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // --- cuDeviceGetPCIBusId tests ---

    #[test]
    fn test_ol_cu_device_get_pci_bus_id() {
        let mut buf = [0u8; 32];
        let result = ol_cuDeviceGetPCIBusId(buf.as_mut_ptr(), 32, 0);
        assert_eq!(result, CUDA_SUCCESS);
        let nul_pos = buf.iter().position(|&b| b == 0).unwrap();
        let id = std::str::from_utf8(&buf[..nul_pos]).unwrap();
        assert_eq!(id, "0000:01:00.0");
    }

    #[test]
    fn test_ol_cu_device_get_pci_bus_id_short_buffer() {
        let mut buf = [0u8; 10];
        let result = ol_cuDeviceGetPCIBusId(buf.as_mut_ptr(), 10, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_pci_bus_id_null() {
        let result = ol_cuDeviceGetPCIBusId(std::ptr::null_mut(), 32, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_pci_bus_id_invalid_device() {
        let mut buf = [0u8; 32];
        let result = ol_cuDeviceGetPCIBusId(buf.as_mut_ptr(), 32, 5);
        assert_eq!(result, CUDA_ERROR_INVALID_DEVICE);
    }

    // --- cuDeviceGetByPCIBusId tests ---

    #[test]
    fn test_ol_cu_device_get_by_pci_bus_id() {
        let mut dev: i32 = -1;
        let bus_id = b"0000:01:00.0\0";
        let result = ol_cuDeviceGetByPCIBusId(&mut dev, bus_id.as_ptr());
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(dev, 0);
    }

    #[test]
    fn test_ol_cu_device_get_by_pci_bus_id_invalid() {
        let mut dev: i32 = -1;
        let bus_id = b"0000:02:00.0\0"; // bus 02 -> device 1, invalid for stub
        let result = ol_cuDeviceGetByPCIBusId(&mut dev, bus_id.as_ptr());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_by_pci_bus_id_null() {
        let mut dev: i32 = -1;
        let result = ol_cuDeviceGetByPCIBusId(&mut dev, std::ptr::null());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // --- Stream-ordered memory / pool tests ---

    #[test]
    fn test_ol_cu_mem_alloc_async() {
        let mut dptr: u64 = 0;
        let result = ol_cuMemAllocAsync(&mut dptr, 1024, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(dptr, 0);
        assert_eq!(ol_cuMemFreeAsync(dptr, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_alloc_async_null_ptr() {
        let result = ol_cuMemAllocAsync(ptr::null_mut(), 1024, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_alloc_async_zero_size() {
        let mut dptr: u64 = 0;
        let result = ol_cuMemAllocAsync(&mut dptr, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_free_async_null() {
        // Freeing null is a no-op
        assert_eq!(ol_cuMemFreeAsync(0, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_device_get_default_mem_pool() {
        let mut pool: u64 = 0;
        let result = ol_cuDeviceGetDefaultMemPool(&mut pool, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(pool, 0);
        // Second call returns same handle
        let mut pool2: u64 = 0;
        assert_eq!(ol_cuDeviceGetDefaultMemPool(&mut pool2, 0), CUDA_SUCCESS);
        assert_eq!(pool, pool2);
    }

    #[test]
    fn test_ol_cu_device_get_default_mem_pool_null() {
        let result = ol_cuDeviceGetDefaultMemPool(ptr::null_mut(), 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_device_get_default_mem_pool_invalid_device() {
        let mut pool: u64 = 0;
        let result = ol_cuDeviceGetDefaultMemPool(&mut pool, 5);
        assert_eq!(result, CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn test_ol_cu_mem_pool_create_destroy() {
        let mut pool: u64 = 0;
        let result = ol_cuMemPoolCreate(&mut pool, 1, 1, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(pool, 0);
        assert_eq!(ol_cuMemPoolDestroy(pool), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_pool_create_null() {
        let result = ol_cuMemPoolCreate(ptr::null_mut(), 1, 1, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_pool_destroy_invalid() {
        let result = ol_cuMemPoolDestroy(0xDEAD);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_pool_get_attribute() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let mut value: u64 = 0xFF;
        let result = ol_cuMemPoolGetAttribute(pool, 1, &mut value);
        assert_eq!(result, CUDA_SUCCESS);
        // Stub returns 0 for unset attributes
        assert_eq!(value, 0);
    }

    #[test]
    fn test_ol_cu_mem_pool_get_attribute_null() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let result = ol_cuMemPoolGetAttribute(pool, 1, ptr::null_mut());
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_pool_set_attribute() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let result = ol_cuMemPoolSetAttribute(pool, 1, 42);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_pool_trim_to() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let result = ol_cuMemPoolTrimTo(pool, 0);
        assert_eq!(result, CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_alloc_from_pool_async() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let mut dptr: u64 = 0;
        let result = ol_cuMemAllocFromPoolAsync(&mut dptr, 256, pool, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_ne!(dptr, 0);
        assert_eq!(ol_cuMemFreeAsync(dptr, 0), CUDA_SUCCESS);
    }

    #[test]
    fn test_ol_cu_mem_alloc_from_pool_async_null() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let result = ol_cuMemAllocFromPoolAsync(ptr::null_mut(), 256, pool, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_mem_alloc_from_pool_async_zero_size() {
        let mut pool: u64 = 0;
        assert_eq!(ol_cuMemPoolCreate(&mut pool, 1, 1, 0), CUDA_SUCCESS);
        let mut dptr: u64 = 0;
        let result = ol_cuMemAllocFromPoolAsync(&mut dptr, 0, pool, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    // ----- Callback tests (stub mode) -----

    #[test]
    fn test_ol_cu_stream_add_callback_stub_fires_immediately() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CB_CALLED: AtomicU32 = AtomicU32::new(0);
        static CB_STATUS: AtomicU32 = AtomicU32::new(u32::MAX);

        unsafe extern "C" fn my_callback(_stream: u64, status: u32, _user_data: u64) {
            CB_CALLED.store(1, Ordering::SeqCst);
            CB_STATUS.store(status, Ordering::SeqCst);
        }

        CB_CALLED.store(0, Ordering::SeqCst);
        CB_STATUS.store(u32::MAX, Ordering::SeqCst);

        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);

        let result = ol_cuStreamAddCallback(
            stream,
            my_callback as *const () as u64,
            0, // user_data
            0, // flags
        );
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(CB_CALLED.load(Ordering::SeqCst), 1);
        assert_eq!(CB_STATUS.load(Ordering::SeqCst), CUDA_SUCCESS);

        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_add_callback_default_stream() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static CB_CALLED2: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn my_callback2(_stream: u64, _status: u32, _user_data: u64) {
            CB_CALLED2.store(1, Ordering::SeqCst);
        }

        CB_CALLED2.store(0, Ordering::SeqCst);

        let result = ol_cuStreamAddCallback(0, my_callback2 as *const () as u64, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(CB_CALLED2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ol_cu_stream_add_callback_invalid_stream() {
        let result = ol_cuStreamAddCallback(0xDEAD, 0, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_stream_add_callback_null_callback() {
        // Null callback pointer should succeed but not crash.
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let result = ol_cuStreamAddCallback(stream, 0, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_launch_host_func_stub_fires_immediately() {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
        static HF_CALLED: AtomicU32 = AtomicU32::new(0);
        static HF_USERDATA: AtomicU64 = AtomicU64::new(0);

        unsafe extern "C" fn my_host_fn(user_data: u64) {
            HF_CALLED.store(1, Ordering::SeqCst);
            HF_USERDATA.store(user_data, Ordering::SeqCst);
        }

        HF_CALLED.store(0, Ordering::SeqCst);
        HF_USERDATA.store(0, Ordering::SeqCst);

        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);

        let result = ol_cuLaunchHostFunc(stream, my_host_fn as *const () as u64, 0xCAFE);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(HF_CALLED.load(Ordering::SeqCst), 1);
        assert_eq!(HF_USERDATA.load(Ordering::SeqCst), 0xCAFE);

        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_launch_host_func_default_stream() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static HF_CALLED2: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn my_host_fn2(_user_data: u64) {
            HF_CALLED2.store(1, Ordering::SeqCst);
        }

        HF_CALLED2.store(0, Ordering::SeqCst);

        let result = ol_cuLaunchHostFunc(0, my_host_fn2 as *const () as u64, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(HF_CALLED2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ol_cu_launch_host_func_invalid_stream() {
        let result = ol_cuLaunchHostFunc(0xDEAD, 0, 0);
        assert_eq!(result, CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn test_ol_cu_launch_host_func_null_func() {
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        let result = ol_cuLaunchHostFunc(stream, 0, 0);
        assert_eq!(result, CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
    }

    #[test]
    fn test_ol_cu_stream_add_callback_passes_user_data() {
        use std::sync::atomic::{AtomicU64, Ordering as AtomOrdering};
        static UD_VALUE: AtomicU64 = AtomicU64::new(0);

        unsafe extern "C" fn cb_with_data(_stream: u64, _status: u32, user_data: u64) {
            UD_VALUE.store(user_data, AtomOrdering::SeqCst);
        }

        UD_VALUE.store(0, AtomOrdering::SeqCst);

        let result = ol_cuStreamAddCallback(0, cb_with_data as *const () as u64, 0xBEEF_CAFE, 0);
        assert_eq!(result, CUDA_SUCCESS);
        assert_eq!(UD_VALUE.load(AtomOrdering::SeqCst), 0xBEEF_CAFE);
    }

    #[test]
    fn test_ol_cu_stream_synchronize_with_no_pending_callbacks() {
        // StreamSynchronize should succeed even when callback system is unused.
        let mut stream: u64 = 0;
        assert_eq!(ol_cuStreamCreate(&mut stream, 0), CUDA_SUCCESS);
        assert_eq!(ol_cuStreamSynchronize(stream), CUDA_SUCCESS);
        let _ = ol_cuStreamDestroy(stream);
    }
}
