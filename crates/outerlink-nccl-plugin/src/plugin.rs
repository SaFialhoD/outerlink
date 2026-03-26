//! NCCL Net Plugin function implementations.
//!
//! Each function matches the signature expected by the `NcclNet_v8` struct.
//! Phase 1 uses blocking TCP (`std::net`) for transport. The async/RDMA path
//! will be introduced in later phases.

use crate::ffi_types::*;
use crate::handles::HandleTable;

use std::ffi::{c_int, c_void, CStr};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Panic guard macro -- catches panics at the FFI boundary
// ---------------------------------------------------------------------------

macro_rules! ffi_guard {
    ($body:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(_) => NCCL_INTERNAL_ERROR,
        }
    };
}

// ---------------------------------------------------------------------------
// Internal state types
// ---------------------------------------------------------------------------

struct ListenState {
    listener: TcpListener,
}

struct SendComm {
    stream: TcpStream,
}

struct RecvComm {
    stream: TcpStream,
}

/// A registered memory region (for TCP, just pointer + size metadata).
struct MemoryRegion {
    _data: *mut c_void,
    _size: usize,
}

// MemoryRegion holds a raw pointer but we only use it within the same process
// and the pointer lifetime is managed by the caller (NCCL).
unsafe impl Send for MemoryRegion {}
unsafe impl Sync for MemoryRegion {}

/// Tracks an in-flight async request.
pub struct RequestState {
    done: bool,
    size: c_int,
}

// ---------------------------------------------------------------------------
// Global handle tables
// ---------------------------------------------------------------------------

fn listen_handles() -> &'static HandleTable<ListenState> {
    static TABLE: OnceLock<HandleTable<ListenState>> = OnceLock::new();
    TABLE.get_or_init(HandleTable::new)
}

fn send_handles() -> &'static HandleTable<SendComm> {
    static TABLE: OnceLock<HandleTable<SendComm>> = OnceLock::new();
    TABLE.get_or_init(HandleTable::new)
}

fn recv_handles() -> &'static HandleTable<RecvComm> {
    static TABLE: OnceLock<HandleTable<RecvComm>> = OnceLock::new();
    TABLE.get_or_init(HandleTable::new)
}

fn mr_handles() -> &'static HandleTable<MemoryRegion> {
    static TABLE: OnceLock<HandleTable<MemoryRegion>> = OnceLock::new();
    TABLE.get_or_init(HandleTable::new)
}

fn request_handles() -> &'static HandleTable<RequestState> {
    static TABLE: OnceLock<HandleTable<RequestState>> = OnceLock::new();
    TABLE.get_or_init(HandleTable::new)
}

// ---------------------------------------------------------------------------
// Handle encoding helpers
// ---------------------------------------------------------------------------

/// Encode a u64 handle ID as a `*mut c_void` for NCCL.
fn id_to_ptr(id: u64) -> *mut c_void {
    id as usize as *mut c_void
}

/// Decode a `*mut c_void` back to a u64 handle ID.
fn ptr_to_id(ptr: *mut c_void) -> u64 {
    ptr as usize as u64
}

// ---------------------------------------------------------------------------
// Handle buffer helpers (for listen/connect address exchange)
// ---------------------------------------------------------------------------

/// Write a socket address into the NCCL handle buffer (up to HANDLE_SIZE bytes).
fn write_addr_to_handle(handle: *mut c_void, addr: &SocketAddr) {
    let addr_str = addr.to_string();
    let bytes = addr_str.as_bytes();
    let len = bytes.len().min(HANDLE_SIZE - 1);
    let dst = handle as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, len);
        // Null-terminate
        *dst.add(len) = 0;
    }
}

/// Read a socket address from the NCCL handle buffer.
fn read_addr_from_handle(handle: *const c_void) -> Result<SocketAddr, ()> {
    let cstr = unsafe { CStr::from_ptr(handle as *const i8) };
    let s = cstr.to_str().map_err(|_| ())?;
    s.parse().map_err(|_| ())
}

// ---------------------------------------------------------------------------
// Plugin functions
// ---------------------------------------------------------------------------

/// Initialize the plugin. Called once by NCCL on load.
pub unsafe extern "C" fn net_init(_log: *mut c_void) -> NcclResult {
    ffi_guard!({
        // Phase 1: no special initialization needed.
        // Future: wire up tracing subscriber to NCCL's log function.
        NCCL_SUCCESS
    })
}

/// Report number of available network devices.
pub unsafe extern "C" fn net_devices(ndev: *mut c_int) -> NcclResult {
    ffi_guard!({
        if ndev.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        // Phase 1: single TCP device
        *ndev = 1;
        NCCL_SUCCESS
    })
}

/// Fill in device properties for device `dev`.
pub unsafe extern "C" fn net_get_properties(
    dev: c_int,
    props: *mut NcclNetProperties_v8,
) -> NcclResult {
    ffi_guard!({
        if dev != 0 || props.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let p = &mut *props;
        p.name = b"OuterLink-TCP\0".as_ptr() as *const i8;
        p.pciPath = std::ptr::null();
        p.guid = 0x4F55_5445_524C_4E4B; // "OUTERLNK" in hex
        p.ptrSupport = NCCL_PTR_HOST;
        p.regIsGlobal = 0;
        p.forceFlush = 0;
        p.speed = 100_000; // 100 Gbps
        p.port = 0;
        p.maxComms = 65536;
        p.maxRecvs = 8;
        p.latency = 0.0;
        p.netDeviceType = 0; // NCCL_NET_DEVICE_HOST
        p.netDeviceVersion = 0;
        NCCL_SUCCESS
    })
}

/// Create a TCP listener. Writes the listen address into `handle` so the
/// remote side can connect.
pub unsafe extern "C" fn net_listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if dev != 0 || handle.is_null() || listen_comm.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        // Bind to any available port on localhost
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(l) => l,
            Err(_) => return NCCL_SYSTEM_ERROR,
        };
        let addr = match listener.local_addr() {
            Ok(a) => a,
            Err(_) => return NCCL_SYSTEM_ERROR,
        };

        write_addr_to_handle(handle, &addr);

        let id = listen_handles().insert(ListenState { listener });
        *listen_comm = id_to_ptr(id);
        NCCL_SUCCESS
    })
}

/// Connect to the address stored in `handle`. Returns a send communicator.
pub unsafe extern "C" fn net_connect(
    dev: c_int,
    handle: *const c_void,
    send_comm: *mut *mut c_void,
    send_dev_comm: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if dev != 0 || handle.is_null() || send_comm.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let addr = match read_addr_from_handle(handle) {
            Ok(a) => a,
            Err(_) => return NCCL_INVALID_ARGUMENT,
        };
        let stream = match TcpStream::connect(addr) {
            Ok(s) => s,
            Err(_) => return NCCL_SYSTEM_ERROR,
        };
        // Disable Nagle for lower latency
        let _ = stream.set_nodelay(true);

        let id = send_handles().insert(SendComm { stream });
        *send_comm = id_to_ptr(id);
        if !send_dev_comm.is_null() {
            *send_dev_comm = std::ptr::null_mut();
        }
        NCCL_SUCCESS
    })
}

/// Accept a connection on the listener. Returns a recv communicator.
pub unsafe extern "C" fn net_accept(
    listen_comm: *mut c_void,
    recv_comm: *mut *mut c_void,
    recv_dev_comm: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if listen_comm.is_null() || recv_comm.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let lid = ptr_to_id(listen_comm);
        let guard = match listen_handles().get(lid) {
            Some(g) => g,
            None => return NCCL_INVALID_ARGUMENT,
        };
        let (stream, _) = match guard.listener.accept() {
            Ok(s) => s,
            Err(_) => return NCCL_SYSTEM_ERROR,
        };
        let _ = stream.set_nodelay(true);

        let id = recv_handles().insert(RecvComm { stream });
        *recv_comm = id_to_ptr(id);
        if !recv_dev_comm.is_null() {
            *recv_dev_comm = std::ptr::null_mut();
        }
        NCCL_SUCCESS
    })
}

/// Register a memory region. For TCP transport this is a no-op -- we just
/// record the pointer and size for bookkeeping.
pub unsafe extern "C" fn net_reg_mr(
    _comm: *mut c_void,
    data: *mut c_void,
    size: usize,
    _kind: c_int,
    mr_handle: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if mr_handle.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let id = mr_handles().insert(MemoryRegion {
            _data: data,
            _size: size,
        });
        *mr_handle = id_to_ptr(id);
        NCCL_SUCCESS
    })
}

/// Register a memory region with DMA-buf fd. For TCP this behaves the same
/// as `net_reg_mr`.
pub unsafe extern "C" fn net_reg_mr_dma_buf(
    _comm: *mut c_void,
    data: *mut c_void,
    size: usize,
    _kind: c_int,
    _offset: u64,
    _fd: c_int,
    mr_handle: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if mr_handle.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let id = mr_handles().insert(MemoryRegion {
            _data: data,
            _size: size,
        });
        *mr_handle = id_to_ptr(id);
        NCCL_SUCCESS
    })
}

/// Deregister a memory region.
pub unsafe extern "C" fn net_dereg_mr(
    _comm: *mut c_void,
    mr_handle: *mut c_void,
) -> NcclResult {
    ffi_guard!({
        let id = ptr_to_id(mr_handle);
        mr_handles().remove(id);
        NCCL_SUCCESS
    })
}

/// Initiate an asynchronous send over TCP.
///
/// Phase 1: This is actually synchronous (blocking TCP write), but we
/// present it as async to match the NCCL API contract.
pub unsafe extern "C" fn net_isend(
    send_comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    _tag: c_int,
    _mr_handle: *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if send_comm.is_null() || request.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let sid = ptr_to_id(send_comm);
        let mut guard = match send_handles().get_mut(sid) {
            Some(g) => g,
            None => return NCCL_INVALID_ARGUMENT,
        };

        // Send size header (4 bytes, little-endian) then data
        let size_bytes = (size as u32).to_le_bytes();
        if guard.stream.write_all(&size_bytes).is_err() {
            return NCCL_SYSTEM_ERROR;
        }
        if size > 0 {
            let buf = std::slice::from_raw_parts(data as *const u8, size as usize);
            if guard.stream.write_all(buf).is_err() {
                return NCCL_SYSTEM_ERROR;
            }
        }

        // Mark request as immediately done (synchronous TCP)
        let req_id = request_handles().insert(RequestState {
            done: true,
            size,
        });
        *request = id_to_ptr(req_id);
        NCCL_SUCCESS
    })
}

/// Initiate an asynchronous receive over TCP.
///
/// Phase 1: Synchronous blocking TCP read.
pub unsafe extern "C" fn net_irecv(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    _tags: *mut c_int,
    _mr_handles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if recv_comm.is_null() || request.is_null() || n < 1 {
            return NCCL_INVALID_ARGUMENT;
        }
        let rid = ptr_to_id(recv_comm);
        let mut guard = match recv_handles().get_mut(rid) {
            Some(g) => g,
            None => return NCCL_INVALID_ARGUMENT,
        };

        // Read only the first buffer (n=1 for most cases)
        // Read size header
        let mut size_buf = [0u8; 4];
        if guard.stream.read_exact(&mut size_buf).is_err() {
            return NCCL_SYSTEM_ERROR;
        }
        let recv_size = u32::from_le_bytes(size_buf) as c_int;

        if recv_size > 0 {
            let dst_ptr = *data;
            let buf = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, recv_size as usize);
            if guard.stream.read_exact(buf).is_err() {
                return NCCL_SYSTEM_ERROR;
            }
        }

        // Write actual received size back
        if !sizes.is_null() {
            *sizes = recv_size;
        }

        let req_id = request_handles().insert(RequestState {
            done: true,
            size: recv_size,
        });
        *request = id_to_ptr(req_id);
        NCCL_SUCCESS
    })
}

/// Flush (no-op for TCP, data is already in host memory).
pub unsafe extern "C" fn net_iflush(
    _recv_comm: *mut c_void,
    _n: c_int,
    _data: *mut *mut c_void,
    _sizes: *mut c_int,
    _mr_handles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        if request.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        // Flush is a no-op for TCP -- immediately done
        let req_id = request_handles().insert(RequestState {
            done: true,
            size: 0,
        });
        *request = id_to_ptr(req_id);
        NCCL_SUCCESS
    })
}

/// Test if an async request has completed.
pub unsafe extern "C" fn net_test(
    request: *mut c_void,
    done: *mut c_int,
    size: *mut c_int,
) -> NcclResult {
    ffi_guard!({
        if request.is_null() || done.is_null() {
            return NCCL_INVALID_ARGUMENT;
        }
        let req_id = ptr_to_id(request);
        match request_handles().get(req_id) {
            Some(state) => {
                *done = if state.done { 1 } else { 0 };
                if !size.is_null() {
                    *size = state.size;
                }
                // Clean up completed requests
                if state.done {
                    drop(state);
                    request_handles().remove(req_id);
                }
                NCCL_SUCCESS
            }
            None => NCCL_INVALID_ARGUMENT,
        }
    })
}

/// Close a send communicator.
pub unsafe extern "C" fn net_close_send(comm: *mut c_void) -> NcclResult {
    ffi_guard!({
        let id = ptr_to_id(comm);
        send_handles().remove(id);
        NCCL_SUCCESS
    })
}

/// Close a recv communicator.
pub unsafe extern "C" fn net_close_recv(comm: *mut c_void) -> NcclResult {
    ffi_guard!({
        let id = ptr_to_id(comm);
        recv_handles().remove(id);
        NCCL_SUCCESS
    })
}

/// Close a listen communicator.
pub unsafe extern "C" fn net_close_listen(comm: *mut c_void) -> NcclResult {
    ffi_guard!({
        let id = ptr_to_id(comm);
        listen_handles().remove(id);
        NCCL_SUCCESS
    })
}
