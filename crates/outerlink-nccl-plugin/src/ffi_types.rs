//! C-compatible type definitions for the NCCL Net Plugin v8 API.
//!
//! These types mirror the NCCL net plugin header (`nccl_net.h`) and define
//! the struct layout that NCCL expects when it loads a plugin via `ncclNet_v8`.

use std::ffi::{c_char, c_int, c_void};

// --- Result codes ---

pub type NcclResult = c_int;
pub const NCCL_SUCCESS: NcclResult = 0;
pub const NCCL_SYSTEM_ERROR: NcclResult = 2;
pub const NCCL_INTERNAL_ERROR: NcclResult = 3;
pub const NCCL_INVALID_ARGUMENT: NcclResult = 5;
pub const NCCL_IN_PROGRESS: NcclResult = 7;

// --- Pointer support flags ---

pub const NCCL_PTR_HOST: c_int = 1;
#[allow(dead_code)]
pub const NCCL_PTR_CUDA: c_int = 2;

/// Maximum handle size for NCCL (must be <= NCCL_NET_HANDLE_MAXSIZE = 128).
/// We use 64 bytes which is sufficient for a socket address + metadata.
pub const HANDLE_SIZE: usize = 64;

// --- Properties struct ---

#[repr(C)]
pub struct NcclNetProperties_v8 {
    pub name: *const c_char,
    pub pciPath: *const c_char,
    pub guid: u64,
    /// Bitmask: NCCL_PTR_HOST=1, NCCL_PTR_CUDA=2
    pub ptrSupport: c_int,
    pub regIsGlobal: c_int,
    pub forceFlush: c_int,
    /// Link speed in Mbps
    pub speed: c_int,
    pub port: c_int,
    pub maxComms: c_int,
    pub maxRecvs: c_int,
    pub latency: f32,
    /// 0 = NCCL_NET_DEVICE_HOST
    pub netDeviceType: c_int,
    pub netDeviceVersion: c_int,
}

// --- Function pointer types ---

pub type InitFn = unsafe extern "C" fn(log: *mut c_void) -> NcclResult;
pub type DevicesFn = unsafe extern "C" fn(ndev: *mut c_int) -> NcclResult;
pub type GetPropertiesFn =
    unsafe extern "C" fn(dev: c_int, props: *mut NcclNetProperties_v8) -> NcclResult;
pub type ListenFn = unsafe extern "C" fn(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> NcclResult;
pub type ConnectFn = unsafe extern "C" fn(
    dev: c_int,
    handle: *const c_void,
    send_comm: *mut *mut c_void,
    send_dev_comm: *mut *mut c_void,
) -> NcclResult;
pub type AcceptFn = unsafe extern "C" fn(
    listen_comm: *mut c_void,
    recv_comm: *mut *mut c_void,
    recv_dev_comm: *mut *mut c_void,
) -> NcclResult;
pub type RegMrFn = unsafe extern "C" fn(
    comm: *mut c_void,
    data: *mut c_void,
    size: usize,
    kind: c_int,
    mr_handle: *mut *mut c_void,
) -> NcclResult;
pub type RegMrDmaBufFn = unsafe extern "C" fn(
    comm: *mut c_void,
    data: *mut c_void,
    size: usize,
    kind: c_int,
    offset: u64,
    fd: c_int,
    mr_handle: *mut *mut c_void,
) -> NcclResult;
pub type DeregMrFn =
    unsafe extern "C" fn(comm: *mut c_void, mr_handle: *mut c_void) -> NcclResult;
pub type IsendFn = unsafe extern "C" fn(
    send_comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    tag: c_int,
    mr_handle: *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult;
pub type IrecvFn = unsafe extern "C" fn(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    tags: *mut c_int,
    mr_handles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult;
pub type IflushFn = unsafe extern "C" fn(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    mr_handles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult;
pub type TestFn =
    unsafe extern "C" fn(request: *mut c_void, done: *mut c_int, size: *mut c_int) -> NcclResult;
pub type CloseFn = unsafe extern "C" fn(comm: *mut c_void) -> NcclResult;

// --- The main plugin struct ---

#[repr(C)]
pub struct NcclNet_v8 {
    pub name: *const c_char,
    pub init: InitFn,
    pub devices: DevicesFn,
    pub getProperties: GetPropertiesFn,
    pub listen: ListenFn,
    pub connect: ConnectFn,
    pub accept: AcceptFn,
    pub regMr: RegMrFn,
    pub regMrDmaBuf: RegMrDmaBufFn,
    pub deregMr: DeregMrFn,
    pub isend: IsendFn,
    pub irecv: IrecvFn,
    pub iflush: IflushFn,
    pub test: TestFn,
    pub closeSend: CloseFn,
    pub closeRecv: CloseFn,
    pub closeListen: CloseFn,
    pub getDeviceMr: Option<unsafe extern "C" fn() -> NcclResult>,
    pub irecvConsumed: Option<unsafe extern "C" fn() -> NcclResult>,
}

// Safety: The function pointers in NcclNet_v8 are all stateless extern "C" functions
// that use interior mutability (DashMap) for any shared state. The static struct
// itself is immutable after initialization.
unsafe impl Send for NcclNet_v8 {}
unsafe impl Sync for NcclNet_v8 {}
